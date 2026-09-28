"""Frozen-field, matched layer heads: technical preflight or fixed 6000 updates.

Only TRAIN RGB is read. A completed source-layer cache and exact prebuilt
selector are required; no JIT compilation or validation-based selection.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
ARMS = {'median4_mass': ('median4', 'mass'), 'top4_mass': ('top4', 'mass'),
        'top4_normalized': ('top4', 'normalized')}
SPEC = {'protocol': 'ibgs_fixed_layer_heads_v1', 'arms': list(ARMS), 'seed': 42,
        'implementation_revision': 'contiguous_selector_slots_v2',
        'train_steps': 6000, 'preflight_names': ['002.png', '041.png'],
        'chunk_pixels': 65536, 'head_lr': .001, 'adam_eps': 1e-8,
        'adam_betas': [.9, .999], 'weight_decay': 0., 'fused_loss_scale': .5,
        'l1_weight': .8, 'dssim_weight': .2, 'source_slots': 4, 'layers': 4,
        'missing_sources': 'preserve original 1..4 neighbors; pad absent slots with zero mass, no invented view',
        'field_updates': 0, 'VAL_reads': 0, 'teacher_calls': 0, 'semantic_reads': 0,
        'preflight_internal_seconds': 300, 'train_internal_seconds_per_arm': 4800,
        'selection': 'fixed final, all three arms; no early/best selection',
        'empty_support': 'exact base per pixel; fail either phase if an entire image is empty',
        'precision': 'FP32 head/evidence, matmul/cudnn TF32 false, no autocast',
        'scope': 'same fixed geometry/seed/order/head parameters, not same historical training compute'}


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.md', '.cpp', '.cu'}}


def completed(folder):
    receipt, launch = read(folder/'execution_receipt.json'), read(folder/'launch_receipt.json')
    require(receipt['status'] == launch['status'] == 'completed' and launch['exit_code'] == 0
            and launch['natural_completion'] is True and launch['execution_receipt_sha256']
            == sha(folder/'execution_receipt.json'), 'Natural completed parent required')
    require(receipt['plan_sha256'] == launch['plan_sha256'] == sha(folder/'plan.json'),
            'Parent plan identity mismatch')
    return receipt


def matched_order(names, phase):
    require(len(names) == len(set(names)) == 350, 'Exactly 350 unique TRAIN cameras required')
    require(phase in ('preflight', 'train') and set(SPEC['preflight_names']) <= set(names),
            'Known phase and fixed preflight cameras required')
    if phase == 'preflight':
        return SPEC['preflight_names'].copy()
    rng = np.random.default_rng(42); ordered = sorted(names); indices = []
    while len(indices) < SPEC['train_steps']:
        indices.extend(rng.permutation(350).tolist())
    return [ordered[i] for i in indices[:SPEC['train_steps']]]


def evidence_slots(package, mode):
    # The selector returns strided mode slices of [H,W,2,4] buffers.
    return {key: package[mode][key].contiguous() for key in ('ids', 'depth', 'weights')}


def prepare(args):
    cache = args.cache.resolve(); completion = completed(cache)
    parent = read(cache/'plan.json'); manifest = read(cache/'cache_manifest.json')
    require(manifest['status'] == 'completed' and completion['cache_manifest_sha256']
            == sha(cache/'cache_manifest.json'), 'Completed source cache required')
    require(tree(parent['source_snapshot']) == parent['source_hashes'], 'Cache source drift')
    contract = read(parent['data_contract']['path'])
    rows = contract['train_rows']; names = [r['name'] for r in rows]
    require(len(names) == len(set(names)) == 350 and all(r['split'] == 'train' for r in rows),
            'Exactly 350 TRAIN source rows required')
    require(sorted(r['name'] for r in manifest['records']) == sorted(names), 'Incomplete source cache')
    output = args.output.resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    snapshot = output/'source_snapshot'; snapshot.parent.mkdir(parents=True)
    shutil.copytree(parent['source_snapshot'], snapshot)
    for relative in ('src/bridge_rgs/ibgs_layer_fusion.py', 'src/bridge_rgs/ibgs_layer_evidence.py',
                     'src/bridge_rgs/ibgs_layer_adapter.py', 'tests/test_ibgs_layer_adapter.py',
                     'tests/test_ibgs_layer_fusion.py', 'tests/test_ibgs_layer_evidence.py',
                     'tests/test_ibgs_layer_heads.py', 'docs/ibgs_layer_heads_protocol.md'):
        target = snapshot/(relative[4:] if relative.startswith('src/') else Path(relative).name)
        shutil.copyfile(ROOT/relative, target)
    shutil.copyfile(Path(__file__), snapshot/Path(__file__).name)
    inputs = {str(cache/name): sha(cache/name) for name in
              ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'cache_manifest.json')}
    inputs.update({str(ROOT/'uv.lock'): sha(ROOT/'uv.lock')})
    for item in (parent['checkpoint'], parent['data_contract'], manifest['shared_eroded_valid'],
                 parent['selector']['binary'], parent['backend']):
        require(sha(item['path']) == item['sha256'], 'Bound parent input drift')
        inputs[item['path']] = item['sha256']
    for record in manifest['records']:
        for key in ('ids', 'depth', 'weights'):
            item = record[key]; require(sha(item['path']) == item['sha256'], 'Changed cache array')
            inputs[item['path']] = item['sha256']
    if args.phase == 'train':
        require(args.preflight is not None, 'Successful frozen preflight required before full training')
        prior = args.preflight.resolve(); pre = completed(prior); pp = read(prior/'plan.json')
        require(pp['phase'] == 'preflight' and pre['completed_updates'] == 6
                and pp['cache_manifest_sha256'] == sha(cache/'cache_manifest.json')
                and pp['specification'] == SPEC, 'Preflight contract mismatch')
        # Core implementation must be identical; worker/protocol may not silently change either.
        for relative in ('bridge_rgs/ibgs_layer_fusion.py', 'bridge_rgs/ibgs_layer_evidence.py',
                         'bridge_rgs/ibgs_layer_adapter.py', 'train_ibgs_layer_heads.py'):
            require(pp['source_hashes'][relative] == sha(snapshot/relative), 'Implementation changed after preflight')
        for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json'):
            inputs[str(prior/name)] = sha(prior/name)
    order = matched_order(names, args.phase)
    write(output/'camera_order.json', {'names': order})
    inputs[str(output/'camera_order.json')] = sha(output/'camera_order.json')
    plan = {'specification': SPEC, 'phase': args.phase, 'output': str(output),
            'source_snapshot': str(snapshot), 'source_hashes': tree(snapshot), 'input_hashes': inputs,
            'cache_manifest': str(cache/'cache_manifest.json'), 'cache_manifest_sha256': sha(cache/'cache_manifest.json'),
            'checkpoint': parent['checkpoint'], 'data_contract': parent['data_contract'],
            'selector': parent['selector'], 'backend': parent['backend'],
            'runtime_sources': parent['runtime_sources'], 'interpreter': parent['interpreter'],
            'neighbors': parent['neighbors'], 'camera_order': str(output/'camera_order.json'),
            'external_timeout_seconds': 360 if args.phase == 'preflight' else 14580,
            'preparation_RGB_decodes': 0, 'preparation_CUDA_calls': 0}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC and tree(plan['source_snapshot']) == plan['source_hashes'],
            'Frozen training implementation changed')
    require(all(sha(p) == h for p, h in plan['input_hashes'].items()), 'Training input changed')
    require(all(sha(p) == h for p, h in plan['runtime_sources'].items()), 'Backend runtime changed')


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def finite(values, label):
    import torch
    for name, value in values:
        require(value is not None and torch.isfinite(value).all(), label+': '+name)


def execute(plan, report):
    import torch
    snapshot = Path(plan['source_snapshot'])
    sys.path[:0] = [str(snapshot/'official'), str(snapshot)]
    from color_aggregation_network import ColorFusionResidualNet
    from diff_plane_rasterization import _C

    from bridge_rgs.ibgs_adapter import BridgeCamera, _checked_image, _SourceRGBBank
    from bridge_rgs.ibgs_layer_adapter import (
        load_fixed_full,
        load_selector_extension,
        render_layer_capture,
    )
    from bridge_rgs.ibgs_layer_evidence import build_layer_evidence
    from bridge_rgs.ibgs_layer_fusion import MassPreservingLayerFusion
    from bridge_rgs.ibgs_losses import masked_photometric_loss
    from bridge_rgs.ibgs_warm_training import FIELD_KEYS, initialize_fusion
    require(sha(_C.__file__) == plan['backend']['sha256'] and torch.__version__ == '2.8.0+cu128',
            'Fixed original backend/Torch required')
    require(Path(sys.prefix).resolve() == Path(plan['interpreter']).parent.parent.resolve(),
            'Use the original isolated uv environment')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    field, background = load_fixed_full(plan['checkpoint']['path'])
    versions = {k: getattr(field, k)._version for k in FIELD_KEYS}
    extension = load_selector_extension(path=plan['selector']['binary']['path'], expected_sha256=plan['selector']['binary']['sha256'])
    contract = read(plan['data_contract']['path']); rows = contract['train_rows']
    cameras = {row['name']: BridgeCamera(row, i) for i, row in enumerate(rows)}
    indices = {row['name']: i for i, row in enumerate(rows)}
    bank = _SourceRGBBank(rows, contract['pixel_hashes'])
    valid = torch.from_numpy(_checked_image(rows[0]['valid_path'], contract['pixel_hashes'][rows[0]['valid_path']], rgb=False) > 0).cuda()
    manifest = read(plan['cache_manifest'])
    source_valid = torch.from_numpy(np.load(manifest['shared_eroded_valid']['path'], allow_pickle=False)).cuda()
    h, w = source_valid.shape
    zero_layer = torch.zeros((h, w, 4), device='cuda')
    absent_source = {'ids': torch.full((h, w, 4), -1, dtype=torch.int32, device='cuda'),
                     'depth': zero_layer, 'weights': zero_layer,
                     'rgb': torch.zeros((h, w, 3), device='cuda'), 'valid': torch.zeros_like(source_valid)}
    cache = {r['name']: {k: np.load(r[k]['path'], mmap_mode='r', allow_pickle=False)
                         for k in ('ids', 'depth', 'weights')} for r in manifest['records']}
    order = read(plan['camera_order'])['names']; initial_hashes = None
    for arm, (mode, reduction) in ARMS.items():
        if plan['phase'] == 'train':
            signal.alarm(SPEC['train_internal_seconds_per_arm'])
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
        directory = Path(plan['output'])/arm; directory.mkdir()
        raw_head = ColorFusionResidualNet(height=rows[0]['height'], width=rows[0]['width']).cuda()
        initialize_fusion(raw_head)
        hashes = {k: tensor_sha(v) for k, v in raw_head.state_dict().items()}
        if initial_hashes is None:
            initial_hashes = hashes
        require(hashes == initial_hashes, 'Matched head initialization differs')
        net = MassPreservingLayerFusion(raw_head, reduction=reduction)
        optimizer = torch.optim.Adam(net.parameters(), lr=.001, eps=1e-8, betas=(.9, .999), weight_decay=0.)
        record = {'arm': arm, 'status': 'failed', 'updates': 0, 'initial_head_hashes': hashes,
                  'parameter_count': sum(p.numel() for p in net.parameters()), 'empty_images': 0}
        report['arms'].append(record)
        with (directory/'trace.jsonl').open('x') as trace:
            for step, name in enumerate(order, 1):
                tick = time.monotonic(); camera = cameras[name]
                optimizer.zero_grad(set_to_none=True)
                package = render_layer_capture(camera, field, background, extension=extension)
                report['raster_calls'] += 1; report['selector_calls'] += 1
                selected = evidence_slots(package, mode); neighbors = plan['neighbors'][name]
                require(1 <= len(neighbors) <= 4 and len(neighbors) == len(set(neighbors)) and name not in neighbors,
                        'Original distinct TRAIN sources excluding target required')
                sources = []
                for source_name in neighbors:
                    source = {k: torch.from_numpy(np.array(v, copy=True)).cuda() for k, v in cache[source_name].items()}
                    source.update(rgb=bank[indices[source_name]].permute(1, 2, 0).contiguous().cuda(), valid=source_valid)
                    sources.append(source)
                sources.extend([absent_source]*(4-len(sources)))
                w2c = camera.world_view_transform.T.contiguous()
                sw2c = torch.stack([cameras[n].world_view_transform.T for n in neighbors]+[w2c]*(4-len(neighbors)))
                ref_to_src = (sw2c @ w2c.inverse().unsqueeze(0)).contiguous()
                source_centers = torch.inverse(sw2c)[:, :3, 3].contiguous()
                base = package['raw'].permute(1, 2, 0).contiguous()
                evidence = build_layer_evidence(selected['ids'], selected['depth'], selected['weights'], base, sources,
                    ref_to_src=ref_to_src, focal=torch.tensor(package['metadata']['focal'], device='cuda'),
                    principal=torch.tensor(package['metadata']['principal'], device='cuda'),
                    target_world_to_camera=w2c, target_campos=camera.camera_center.contiguous(),
                    source_campos=source_centers, chunk_pixels=SPEC['chunk_pixels'])
                del sources, selected
                result = net(evidence['features'], evidence['target_weights'], evidence['support'],
                             package['ray'].reshape(3, -1).T, base.reshape(-1, 3))
                active = result['active']; require(active.any(), 'Entire image has no supported source evidence')
                prediction = result['image_pred'].T.reshape_as(package['raw'])
                target = bank[indices[name]].cuda()
                objective, terms = masked_photometric_loss(prediction, target, valid)
                loss = .5*objective; require(torch.isfinite(loss), 'Nonfinite head objective')
                loss.backward()
                finite(((n, p.grad) for n, p in net.named_parameters()), 'Head gradient')
                mlp_grad_l1 = float(sum(p.grad.abs().sum() for p in raw_head.per_view_mlp.parameters()))
                cnn_grad_l1 = float(sum(p.grad.abs().sum() for p in raw_head.conv_decoder.parameters()))
                if plan['phase'] == 'preflight' and step == 2:
                    require(mlp_grad_l1 > 0 and cnn_grad_l1 > 0, 'Second-step MLP/CNN gradients must be nonzero')
                optimizer.step(); finite(net.named_parameters(), 'Updated head')
                finite(((str(i)+'/'+key, value) for i, state in enumerate(optimizer.state.values())
                        for key, value in state.items() if isinstance(value, torch.Tensor)), 'Adam state')
                require(all(getattr(field, k).grad is None and getattr(field, k)._version == v
                            for k, v in versions.items()), 'Frozen field changed or received gradients')
                record['updates'] += 1; report['completed_updates'] += 1
                torch.cuda.synchronize()
                row = {'step': step, 'name': name, 'sources': neighbors, 'loss': float(loss),
                       'l1': float(terms['l1']), 'dssim': float(terms['dssim']),
                       'active_fraction': float(active.float().mean()),
                       'mean_supported_mass': float(result['supported_mass'].mean()),
                       'mlp_grad_l1': mlp_grad_l1, 'cnn_grad_l1': cnn_grad_l1,
                       'elapsed_seconds': time.monotonic()-tick,
                       'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
                trace.write(json.dumps(row, allow_nan=False)+'\n'); trace.flush()
                if step <= 2 or step % 200 == 0:
                    print(json.dumps({'arm': arm, **row}), flush=True)
                del package, evidence, result, prediction, target, loss, objective, terms, base, active, source
        torch.save({'protocol': SPEC['protocol'], 'arm': arm, 'step': len(order),
                    'head': {k: v.detach().cpu() for k, v in raw_head.state_dict().items()},
                    'optimizer': optimizer.state_dict(), 'field_checkpoint': plan['checkpoint'],
                    'cache_manifest_sha256': plan['cache_manifest_sha256'], 'specification': SPEC,
                    'plan_sha256': report['plan_sha256']}, directory/'last.pt')
        record.update(status='completed', elapsed_seconds=time.monotonic()-started,
                      checkpoint={'path': str(directory/'last.pt'), 'sha256': sha(directory/'last.pt')},
                      trace_sha256=sha(directory/'trace.jsonl'), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        del net, raw_head, optimizer
        gc.collect(); torch.cuda.empty_cache()
    report.update(field_unchanged=True, source_RGB_decode_count=bank.decode_count,
                  source_RGB_names=sorted(bank.decoded_names), valid_decodes=1,
                  field_updates=0, VAL_reads=0, semantic_reads=0, teacher_calls=0)
    require(report['completed_updates'] == 3*len(order) == report['raster_calls'] == report['selector_calls'],
            'Incomplete matched training')
    report['actual_imports'] = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and name.split('.')[0] in ('bridge_rgs', 'gaussian_renderer', 'scene', 'utils', 'arguments', 'color_aggregation_network'):
            p = Path(path).resolve(); relative = str(p.relative_to(snapshot))
            require(plan['source_hashes'].get(relative) == sha(p), 'Unbound model/runtime import')
            report['actual_imports'][name] = {'path': str(p), 'sha256': sha(p)}


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true'); action.add_argument('--run', type=Path)
    parser.add_argument('--phase', choices=('preflight', 'train'))
    parser.add_argument('--cache', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight', type=Path); parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        require(args.phase and args.cache and args.output, 'Preparation needs phase/cache/output')
        prepare(args); return
    require(args.expected_plan_sha256 and sha(args.run) == args.expected_plan_sha256, 'Explicit plan SHA required')
    plan = read(args.run)
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Run frozen worker')
    report = {'status': 'failed', 'plan_sha256': args.expected_plan_sha256, 'arms': [],
              'completed_updates': 0, 'raster_calls': 0, 'selector_calls': 0}
    started = time.monotonic()
    def deadline(*_):
        raise TimeoutError('Fixed per-arm deadline exceeded')
    signal.signal(signal.SIGALRM, deadline)
    if plan['phase'] == 'preflight':
        signal.alarm(SPEC['preflight_internal_seconds'])
    write(Path(plan['output'])/'execution_started.json', {'pid': os.getpid(), 'plan_sha256': args.expected_plan_sha256})
    try:
        verify(plan); execute(plan, report); verify(plan)
        report.update(status='completed', sources_inputs_unchanged=True)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        signal.alarm(0); report['elapsed_seconds'] = time.monotonic()-started
        write(Path(plan['output'])/'execution_receipt.json', report)
    print(json.dumps({'status': report['status'], 'receipt_sha256': sha(Path(plan['output'])/'execution_receipt.json')}))


if __name__ == '__main__':
    main()
