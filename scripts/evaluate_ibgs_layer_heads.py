"""Three fixed layer heads, then official RGB scoring in a separate process.

No training-worker import, source-depth refresh, semantic target, or model selection.
Prepare is permitted only after the frozen 6000 x 3 training run naturally finishes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
OLD_EVALUATION = RUNS/'ibgs_warm_evaluation_v1'
TRAINING_PLAN_SHA = 'b9a44c95b4e68b955f4fcc27abf27917134b4c5df8f456ed31c04a69b3a41c1a'
OLD_PLAN_SHA = '84425a934197cd32e08b4b7262dad9b371449731c996d9c53159c66d05f43e7b'
LEGACY_SHA = 'b4e611437bfcd098e4c2c0686ec5f5d3cd4980e667c707103b3dbdd65c00ec7a'
HELPER_SHA = '041a40203529db626a42fca92098fde8b8bf3eb70dafae799b436396104bc36b'
ARMS = {'median4_mass': ('median4', 'mass'), 'top4_mass': ('top4', 'mass'),
        'top4_normalized': ('top4', 'normalized')}
PRIMARY = 'top4_mass'
SPEC = {
    'protocol': 'ibgs_fixed_layer_heads_evaluation_v1', 'arms': list(ARMS),
    'primary_candidate': PRIMARY, 'views': 50, 'train_source_cameras': 350,
    'target_calls': 150, 'selector_calls': 150, 'source_depth_calls': 0,
    'native_arrays': 150, 'delivered_pngs': 150, 'VAL_rgb_reads': 50, 'lpips_calls': 150,
    'source_slots': 4, 'layers': 4, 'chunk_pixels': 65536,
    'missing_sources': 'original neighbors only; absent slots -1/0/false, divisor remains four',
    'bootstrap_repeats': 5000, 'bootstrap_seed': 20260926,
    'comparisons': ['top4_mass_minus_median4_mass', 'top4_mass_minus_top4_normalized',
                    'top4_mass_minus_old_full_fused', 'top4_mass_minus_E_rgb'],
    'sole_gate': 'top4_mass minus E: PSNR >= .15 dB, paired lower > 0, SSIM >= 0, LPIPS <= 0',
    'render_seconds': 600, 'render_external_seconds': 660,
    'score_seconds': 300, 'score_external_seconds': 360,
    'annotation_reads': 0, 'mask_outputs': 0, 'teacher_calls': 0,
    'passive_feature_diagnostics': 'pooled feature magnitude and zero fraction on supported pixels; '
                                   'one read-only CNN pre-hook, no additional head forward or intervention',
    'optimizer_steps': 0, 'field_updates': 0, 'backward': 0,
    'selection': 'fixed last for all arms; primary never reselected; no automatic adoption',
    'interpretation': 'selection/mass ablations on reused development views; historical full '
                      'is an engineering reference with different training; no novelty claim',
}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(payload)


def files(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.md', '.cpp', '.cu'}}


def copy_bound_sources(origin, destination, hashes):
    """Preserve every bound source, including legitimate .py inside __pycache__."""
    origin, destination = Path(origin), Path(destination)
    for relative, expected in hashes.items():
        source, target = origin/relative, destination/relative
        require(source.resolve().is_relative_to(origin.resolve())
                and target.resolve().is_relative_to(destination.resolve())
                and sha(source) == expected, 'Invalid or changed bound source')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    require(files(destination) == hashes, 'Bound source copy differs')


def legacy():
    path = Path(__file__).with_name('ibgs_warm_evaluation_base.py')
    if not path.exists():
        path = OLD_EVALUATION/'source_snapshot/evaluate_ibgs_warm.py'
    require(sha(path) == LEGACY_SHA, 'Frozen official evaluation helper changed')
    spec = importlib.util.spec_from_file_location('layer_head_official_base', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def natural(folder, stage=None):
    prefix = '' if stage is None else stage+'_'
    receipt_path = Path(folder)/(prefix+'execution_receipt.json')
    result = read(receipt_path); launch = read(Path(folder)/(prefix+'launch_receipt.json'))
    plan_sha = sha(Path(folder)/'plan.json')
    require(result['status'] == launch['status'] == 'completed'
            and launch['exit_code'] == 0 and launch['natural_completion'] is True
            and result['plan_sha256'] == launch['plan_sha256'] == plan_sha
            and launch['execution_receipt_sha256'] == sha(receipt_path),
            'Natural completed stage bound to this plan required')
    return result


def validate_endpoint(saved, arm, parent, plan_sha, *, parameter_count=66095):
    import torch
    require(saved['protocol'] == 'ibgs_fixed_layer_heads_v1' and saved['arm'] == arm
            and saved['step'] == 6000 and saved['plan_sha256'] == plan_sha
            and saved['specification'] == parent['specification']
            and saved['field_checkpoint'] == parent['checkpoint']
            and saved['cache_manifest_sha256'] == parent['cache_manifest_sha256'],
            'Fixed endpoint identity mismatch')
    head = saved['head']
    require(bool(head) and sum(v.numel() for v in head.values()) == parameter_count
            and all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in head.values()),
            'Finite original FP32 head parameter schema required')


def prediction_barrier(records, names):
    require(len(names) == len(set(names)) == 50, 'Exactly 50 fixed evaluation names required')
    expected = {(arm, name) for arm in ARMS for name in names}
    require(len(records) == 150 and {(r['arm'], r['name']) for r in records} == expected,
            'All 150 unique predictions must precede target RGB access')


def prepare(args):
    import torch
    training = args.training.resolve()
    require(sha(training/'plan.json') == TRAINING_PLAN_SHA, 'Fixed training plan required')
    completed = natural(training)  # Before any endpoint access or output creation.
    parent = read(training/'plan.json')
    require(parent['phase'] == 'train' and completed['completed_updates'] == 18000
            and completed['raster_calls'] == completed['selector_calls'] == 18000
            and completed['field_unchanged'] is True
            and {a['arm'] for a in completed['arms']} == set(ARMS)
            and len(completed['arms']) == 3, 'All three fixed training arms must complete')
    require(files(parent['source_snapshot']) == parent['source_hashes'], 'Training source changed')
    inputs = {str(training/n): sha(training/n) for n in
              ('plan.json', 'execution_receipt.json', 'launch_receipt.json')}
    inputs.update(parent['input_hashes']); inputs.update(parent['runtime_sources'])
    endpoints = {}
    initial = completed['arms'][0]['initial_head_hashes']
    for arm in completed['arms']:
        require(arm['status'] == 'completed' and arm['updates'] == 6000
                and arm['initial_head_hashes'] == initial and arm['parameter_count'] == 66095,
                'Incomplete or unmatched head training')
        bound = arm['checkpoint']; require(sha(bound['path']) == bound['sha256'], 'Endpoint changed')
        saved = torch.load(bound['path'], map_location='cpu', weights_only=False)
        validate_endpoint(saved, arm['arm'], parent, TRAINING_PLAN_SHA)
        endpoints[arm['arm']] = bound; inputs[bound['path']] = bound['sha256']
        del saved
    contract = read(parent['data_contract']['path'])
    rows = contract['train_rows']; names = [r['name'] for r in rows]
    require(len(names) == len(set(names)) == 350 and all(r['split'] == 'train' for r in rows),
            'Exactly 350 distinct TRAIN source rows required')
    inputs.update(contract['pixel_hashes'])
    cache = read(parent['cache_manifest'])
    require(sha(parent['cache_manifest']) == parent['cache_manifest_sha256']
            and cache['status'] == 'completed' and cache['checkpoint'] == parent['checkpoint']
            and sorted(r['name'] for r in cache['records']) == sorted(names), 'Wrong source cache')
    base = legacy()
    require(sha(OLD_EVALUATION/'plan.json') == OLD_PLAN_SHA, 'Old evaluation population changed')
    old = read(OLD_EVALUATION/'plan.json')
    old_score = natural(OLD_EVALUATION, 'score'); natural(OLD_EVALUATION, 'render')
    views = old['views']
    require(len(views) == len({v['camera']['name'] for v in views}) == 50
            and not {v['camera']['name'] for v in views} & set(names), 'VAL must not enter TRAIN bank')
    for view in views:
        require(view['source_indices'] == base.neighbors(view['camera'], rows)
                and view['source_names'] == [rows[i]['name'] for i in view['source_indices']]
                and 1 <= len(view['source_names']) <= 4
                and np.array_equal(view['camera']['K'], contract['common_camera']['K'])
                and (view['width'], view['height']) == (1320, 989), 'Camera/source selection differs')
    for name in ('plan.json', 'render_execution_receipt.json', 'render_launch_receipt.json',
                 'score_execution_receipt.json', 'score_launch_receipt.json'):
        inputs[str(OLD_EVALUATION/name)] = sha(OLD_EVALUATION/name)
    references = {'old_full_fused': old_score['metrics']['full/fused'],
                  'E_rgb': old['reference_metrics']['E_rgb']}
    for bound in references.values():
        require(sha(bound['path']) == bound['sha256'], 'Bound reference changed')
        metrics = read(bound['path'])
        require(metrics['inherited_common_reference_fingerprint'] == old['reference_fingerprint'],
                'Different reference camera/GT fingerprint')
        inputs[bound['path']] = bound['sha256']
    for name in ('plan.json', 'execution_receipt.json'):
        path = RUNS/'rgb_fixed_ensemble_v1'/name; inputs[str(path)] = sha(path)
    e_receipt = read(RUNS/'rgb_fixed_ensemble_v1/execution_receipt.json')
    require(e_receipt['status'] == 'completed' and e_receipt['metrics_sha256'] == references['E_rgb']['sha256'],
            'E must remain bound to its completed execution; no historical launch invented')
    for path in old['perceptual_weights']:
        inputs[path] = old['input_hashes'][path]
    require(not {v['source_image_path'] for v in views} & inputs.keys(), 'No VAL RGB payload in preparation')
    require(all(sha(p) == h for p, h in inputs.items()), 'Input/source bytes changed')
    output = args.output.resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk run required')
    snapshot = output/'source_snapshot'; snapshot.mkdir(parents=True)
    copy_bound_sources(parent['source_snapshot'], snapshot/'render', parent['source_hashes'])
    old_snapshot = Path(old['source_snapshot'])
    expected_scoring = {k[len('scoring/'):]: v for k, v in old['source_hashes'].items() if k.startswith('scoring/')}
    require(files(old_snapshot/'scoring') == expected_scoring, 'Old scoring source changed')
    copy_bound_sources(old_snapshot/'scoring', snapshot/'scoring', expected_scoring)
    require(sha(old_snapshot/'rgb_scoring_helpers.py') == HELPER_SHA, 'RGB helper changed')
    shutil.copy2(old_snapshot/'rgb_scoring_helpers.py', snapshot/'rgb_scoring_helpers.py')
    shutil.copy2(old_snapshot/'evaluate_ibgs_warm.py', snapshot/'ibgs_warm_evaluation_base.py')
    for path in (Path(__file__), ROOT/'tests/test_ibgs_layer_evaluation.py',
                 ROOT/'docs/ibgs_layer_evaluation_protocol.md'):
        shutil.copy2(path, snapshot/path.name)
    require(not torch.cuda.is_initialized(), 'CPU-only preparation required')
    plan = {'status': 'prepared_after_natural_training_completion', 'specification': SPEC,
            'output': str(output), 'training': str(training), 'training_plan_sha256': TRAINING_PLAN_SHA,
            'training_specification': parent['specification'], 'endpoints': endpoints,
            'source_snapshot': str(snapshot), 'source_hashes': files(snapshot), 'input_hashes': inputs,
            'runtime_sources': parent['runtime_sources'], 'interpreter': parent['interpreter'],
            'checkpoint': parent['checkpoint'], 'data_contract': parent['data_contract'],
            'cache_manifest': parent['cache_manifest'], 'cache_manifest_sha256': parent['cache_manifest_sha256'],
            'selector': parent['selector'], 'backend': parent['backend'],
            'views': views, 'reference_metrics': references, 'reference_fingerprint': old['reference_fingerprint'],
            'scoring_protocol': old['scoring_protocol'], 'scoring_numerics': old['scoring_numerics'],
            'main_runtime': old['main_runtime'], 'perceptual_weights': old['perceptual_weights'],
            'prepare_VAL_payload_reads': 0, 'prepare_CUDA_initialized': False}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def observed_head_forward(net, *inputs):
    """Observe actual CNN input; never recompute or replace its layer evidence."""
    import torch
    captured = []
    width = net.backbone.per_view_feat_dim

    def observe(_module, arguments):
        grid = arguments[0]
        require(grid.ndim == 4 and grid.shape[0] == 1 and grid.shape[1] == width+6,
                'Pooled/ray/base CNN grid required')
        with torch.no_grad():
            pooled = grid[:, :width].detach()
            require(torch.isfinite(pooled).all(), 'Nonfinite pooled evidence')
            captured.append(({
                'pooled_feature_mean_abs': float(pooled.abs().mean()),
                'pooled_feature_max_abs': float(pooled.abs().amax()),
                'ray_mean_abs': float(grid[:, width:width+3].detach().abs().mean()),
                'base_rgb_mean_abs': float(grid[:, width+3:].detach().abs().mean()),
            }, pooled.abs().amax(dim=1).reshape(-1) == 0))
        # Returning None preserves the original arguments, including their storage.

    handle = net.backbone.conv_decoder.register_forward_pre_hook(observe)
    try:
        result = net(*inputs)
    finally:
        handle.remove()
    require(len(captured) == 1, 'Exactly one original head forward required')
    stats, zero = captured[0]; active = result['active']
    require(active.dtype == torch.bool and active.shape == zero.shape and active.any(),
            'Nonempty supported pixel grid required')
    active_count = int(active.sum()); zero_active_count = int((zero & active).sum())
    stats.update(pooled_zero_pixels=int(zero.sum()), pooled_pixels=zero.numel(),
                 supported_pixels=active_count, supported_zero_pool_pixels=zero_active_count,
                 supported_zero_pool_fraction=zero_active_count/active_count)
    return result, stats


def infer_layer_head(package, mode, net, camera, cameras, names, source_provider, absent_source,
                     evidence_builder, *, chunk_pixels=65536, observe_features=False):
    """The training inference path, factored locally without importing its worker.

    source_provider returns the same cache slots, TRAIN RGB and shared valid as
    training. Kept as an injectable function for small CPU wiring contracts.
    """
    import torch
    require(mode in ('median4', 'top4') and 1 <= len(names) <= 4
            and len(names) == len(set(names)) and camera.image_name not in names,
            'Original distinct sources excluding target required')
    selected = {k: package[mode][k].contiguous() for k in ('ids', 'depth', 'weights')}
    sources = [source_provider(name) for name in names]
    sources.extend([absent_source]*(4-len(sources)))
    w2c = camera.world_view_transform.T.contiguous()
    sw2c = torch.stack([cameras[n].world_view_transform.T for n in names]+[w2c]*(4-len(names)))
    ref_to_src = (sw2c @ w2c.inverse().unsqueeze(0)).contiguous()
    source_centers = torch.inverse(sw2c)[:, :3, 3].contiguous()
    base = package['raw'].permute(1, 2, 0).contiguous()
    evidence = evidence_builder(selected['ids'], selected['depth'], selected['weights'], base, sources,
        ref_to_src=ref_to_src, focal=torch.tensor(package['metadata']['focal'], device=base.device),
        principal=torch.tensor(package['metadata']['principal'], device=base.device),
        target_world_to_camera=w2c, target_campos=camera.camera_center.contiguous(),
        source_campos=source_centers, chunk_pixels=chunk_pixels)
    inputs = (evidence['features'], evidence['target_weights'], evidence['support'],
              package['ray'].reshape(3, -1).T, base.reshape(-1, 3))
    if observe_features:
        result, feature_stats = observed_head_forward(net, *inputs)
    else:
        result, feature_stats = net(*inputs), {}
    require(result['active'].any(), 'Entire image has no supported source evidence')
    return result['image_pred'].T.reshape_as(package['raw']), {
        'active_fraction': float(result['active'].float().mean()),
        'mean_supported_mass': float(result['supported_mass'].mean()),
        'source_names': list(names), 'absent_source_slots': 4-len(names), **feature_stats}


def set_numerics(torch, flags):
    torch.backends.cudnn.allow_tf32 = flags['cudnn_tf32']
    torch.backends.cudnn.benchmark = flags['benchmark']
    torch.set_float32_matmul_precision(flags['matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32 = flags['matmul_tf32']


def render_stage(plan, report):
    import torch
    base = legacy(); root = Path(plan['source_snapshot'])/'render'
    sys.path[:0] = [str(root/'official'), str(root)]
    from color_aggregation_network import ColorFusionResidualNet
    from diff_plane_rasterization import _C

    from bridge_rgs.ibgs_adapter import BridgeCamera, _SourceRGBBank
    from bridge_rgs.ibgs_layer_adapter import (
        load_fixed_full,
        load_selector_extension,
        render_layer_capture,
    )
    from bridge_rgs.ibgs_layer_evidence import build_layer_evidence
    from bridge_rgs.ibgs_layer_fusion import MassPreservingLayerFusion
    from bridge_rgs.ibgs_warm_training import FIELD_KEYS
    require(sha(_C.__file__) == plan['backend']['sha256'] and torch.__version__ == '2.8.0+cu128'
            and Path(sys.prefix).resolve() == Path(plan['interpreter']).parent.parent.resolve(),
            'Original isolated renderer/backend required')
    before = base.numerical_flags(torch)
    set_numerics(torch, {'cudnn_tf32': False, 'matmul_tf32': False, 'benchmark': False, 'matmul_precision': 'highest'})
    torch.set_num_threads(4); torch.cuda.reset_peak_memory_stats()
    field = None; versions = {}; records = []; original_backend = _C.rasterize_gaussians
    try:
        field, background = load_fixed_full(plan['checkpoint']['path'])
        versions = {k: (id(getattr(field, k)), getattr(field, k)._version) for k in FIELD_KEYS}
        extension = load_selector_extension(plan['selector']['binary']['path'], plan['selector']['binary']['sha256'])
        contract = read(plan['data_contract']['path']); rows = contract['train_rows']
        cameras = {row['name']: BridgeCamera(row, i) for i, row in enumerate(rows)}
        indices = {row['name']: i for i, row in enumerate(rows)}
        bank = _SourceRGBBank(rows, contract['pixel_hashes'])
        manifest = read(plan['cache_manifest'])
        source_valid = torch.from_numpy(np.load(manifest['shared_eroded_valid']['path'], allow_pickle=False)).cuda()
        h, w = source_valid.shape
        zero = torch.zeros((h, w, 4), device='cuda')
        absent = {'ids': torch.full((h, w, 4), -1, dtype=torch.int32, device='cuda'),
                  'depth': zero, 'weights': zero, 'rgb': torch.zeros((h, w, 3), device='cuda'),
                  'valid': torch.zeros_like(source_valid)}
        cache = {r['name']: {k: np.load(r[k]['path'], mmap_mode='r', allow_pickle=False)
                             for k in ('ids', 'depth', 'weights')} for r in manifest['records']}

        def source_provider(name):
            require(name in cameras, 'Non-TRAIN source requested')
            source = {k: torch.from_numpy(np.array(v, copy=True)).cuda() for k, v in cache[name].items()}
            source.update(rgb=bank[indices[name]].permute(1, 2, 0).contiguous().cuda(), valid=source_valid)
            return source

        with torch.no_grad():
            for arm, (mode, reduction) in ARMS.items():
                saved = torch.load(plan['endpoints'][arm]['path'], map_location='cpu', weights_only=False)
                validate_endpoint(saved, arm, {'specification': plan['training_specification'],
                    'checkpoint': plan['checkpoint'], 'cache_manifest_sha256': plan['cache_manifest_sha256']},
                    plan['training_plan_sha256'])
                raw_head = ColorFusionResidualNet(height=h, width=w).cuda()
                raw_head.load_state_dict(saved['head'], strict=True)
                net = MassPreservingLayerFusion(raw_head, reduction=reduction).eval().requires_grad_(False)
                head_versions = {k: (id(v), v._version) for k, v in net.named_parameters()}
                for view in plan['views']:
                    source = view['camera']
                    row = {'name': source['name'], 'image_id': source['image_id'],
                           'K': contract['common_camera']['K'], 'w2c_original': source['w2c'],
                           'width': view['width'], 'height': view['height']}
                    camera = BridgeCamera(row, -1)
                    require(all(getattr(camera, k) == getattr(next(iter(cameras.values())), k)
                                for k in ('Fx', 'Fy', 'Cx', 'Cy', 'FoVx', 'FoVy', 'image_width', 'image_height')),
                            'Target/source camera intrinsics differ')
                    package = render_layer_capture(camera, field, background, extension=extension)
                    report['target_calls'] += 1; report['selector_calls'] += 1
                    prediction, stats = infer_layer_head(package, mode, net, camera, cameras,
                        view['source_names'], source_provider, absent, build_layer_evidence,
                        observe_features=True)
                    value = prediction.permute(1, 2, 0).contiguous().cpu().numpy()
                    require(value.dtype == np.float32 and value.shape == (989, 1320, 3)
                            and np.isfinite(value).all(), 'Invalid native RGB output')
                    path = Path(plan['output'])/'native'/arm/(Path(source['name']).stem+'.npy')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('xb') as stream:
                        np.save(stream, value, allow_pickle=False)
                    records.append({'arm': arm, 'name': source['name'], 'path': str(path), 'sha256': sha(path),
                                    'shape': list(value.shape), 'dtype': 'float32', **stats,
                                    'actual_camera': package['metadata']})
                    del package, prediction, value
                require(all((id(v), v._version) == head_versions[k] for k, v in net.named_parameters()),
                        'Head mutated during evaluation')
                del net, raw_head, saved
                torch.cuda.empty_cache()
                print(json.dumps({'arm': arm, 'native_outputs': len(records)}), flush=True)
        prediction_barrier(records, [v['camera']['name'] for v in plan['views']])
        require(report['target_calls'] == report['selector_calls'] == 150, 'Render count differs')
        require(bank.decoded_names <= set(cameras), 'Non-TRAIN photo decoded')
        path = Path(plan['output'])/'native_predictions.json'
        write(path, {'status': 'completed', 'records': records, 'finished_utc': base.utc(), 'VAL_payload_reads': 0})
        report.update(native_predictions={'path': str(path), 'sha256': sha(path)},
                      source_rgb_decodes=bank.decode_count, source_rgb_names=sorted(bank.decoded_names),
                      actual_numerics=base.numerical_flags(torch),
                      selector_binary=plan['selector']['binary'], backend=plan['backend'])
    finally:
        report['field_unchanged'] = field is not None and all(
            (id(getattr(field, k)), getattr(field, k)._version) == v and getattr(field, k).grad is None
            and not getattr(field, k).requires_grad for k, v in versions.items())
        report['hook_restored'] = _C.rasterize_gaussians is original_backend
        report['peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        set_numerics(torch, before); report['numeric_flags_restored'] = base.numerical_flags(torch) == before
        report['actual_imports'] = base.actual_imports(root, {'bridge_rgs', 'gaussian_renderer', 'scene',
                                                             'utils', 'arguments', 'color_aggregation_network'})
    require(all(report[k] for k in ('field_unchanged', 'hook_restored', 'numeric_flags_restored')),
            'Runtime restoration failed')


def score_stage(plan, report):
    import importlib.metadata

    import cv2
    import torch
    base = legacy(); output = Path(plan['output'])
    rendered = natural(output, 'render')
    bound = rendered['native_predictions']; require(sha(bound['path']) == bound['sha256'], 'Native barrier changed')
    native = read(bound['path'])['records']; names = [v['camera']['name'] for v in plan['views']]
    prediction_barrier(native, names)
    official, helper = base.scoring_modules(plan['source_snapshot'])
    from bridge_rgs.evaluate import distortion_render_grid
    require({k: importlib.metadata.version(k) for k in plan['main_runtime']} == plan['main_runtime'],
            'Scoring runtime differs')
    views = {v['camera']['name']: v for v in plan['views']}; pngs = []
    before = base.numerical_flags(torch)
    try:
        for record in native:
            require(sha(record['path']) == record['sha256'], 'Native output changed')
            rgb = np.load(record['path'], allow_pickle=False); camera = views[record['name']]['camera']
            require(rgb.shape == (989, 1320, 3) and rgb.dtype == np.float32 and np.isfinite(rgb).all(), 'Bad native array')
            _, _, _, back = distortion_render_grid(camera['K'], camera['distortion'], camera['width'], camera['height'], 'colmap_corner_v2')
            rgb = np.clip(rgb, 0., 1.)
            if back is not None:
                rgb = cv2.remap(rgb, back[..., 0], back[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            delivered = np.rint(rgb*255).astype(np.uint8)
            path = output/'predictions'/record['arm']/record['name']; path.parent.mkdir(parents=True, exist_ok=True)
            require(not path.exists() and cv2.imwrite(str(path), delivered[..., ::-1]), 'Fresh PNG save failed')
            pngs.append({'arm': record['arm'], 'name': record['name'], 'path': str(path), 'sha256': sha(path)})
        prediction_barrier(pngs, names)
        barrier = output/'predictions_receipt.json'
        write(barrier, {'status': 'all_150_pngs_before_GT', 'plan_sha256': report['plan_sha256'],
                        'records': pngs, 'finished_utc': base.utc(), 'VAL_payload_reads': 0})
        report.update(predictions_finished_utc=base.utc(),
                      predictions_receipt={'path': str(barrier), 'sha256': sha(barrier)})
        torch.set_num_threads(8); cv2.setNumThreads(8)
        set_numerics(torch, plan['scoring_numerics'])
        require(base.numerical_flags(torch) == plan['scoring_numerics'], 'Scoring flags differ')
        torch.cuda.reset_peak_memory_stats(); perceptual = official._lpips('cuda')
        rows = {a: [] for a in ARMS}; index = {(p['arm'], p['name']): p for p in pngs}
        report['scoring_started_utc'] = base.utc()
        for view in plan['views']:
            camera = view['camera']; payload = Path(view['source_image_path']).read_bytes()
            report['VAL_rgb_reads'] += 1
            require(hashlib.sha256(payload).hexdigest() == view['source_image_sha256'], 'GT RGB changed')
            target = official._decode_rgb(payload, camera['width'], camera['height'], 'target')
            for arm in ARMS:
                item = index[arm, camera['name']]; payload = Path(item['path']).read_bytes()
                require(hashlib.sha256(payload).hexdigest() == item['sha256'], 'PNG changed')
                prediction = official._decode_rgb(payload, camera['width'], camera['height'], 'prediction')
                score = helper.score_rgb_only(official.score_official_arrays, prediction, target, perceptual, 'cuda')
                report['lpips_calls'] += 1
                rows[arm].append({'name': camera['name'], 'width': camera['width'], 'height': camera['height'],
                                  'rgb_sha256': item['sha256'], 'source_rgb_sha256': view['source_image_sha256'], **score})
        metrics, descriptors = {}, {}
        for arm, values in rows.items():
            metrics[arm] = {'modality': 'RGB_only', 'views': values, 'evaluation_views': 50,
                            'inherited_common_reference_fingerprint': plan['reference_fingerprint'],
                            'scoring_protocol': plan['scoring_protocol'],
                            **{k: float(np.mean([r[k] for r in values])) for k in helper.RGB_KEYS}}
            path = output/f'metrics_{arm}.json'; write(path, metrics[arm])
            descriptors[arm] = {'path': str(path), 'sha256': sha(path)}
        references = {'median4_mass': descriptors['median4_mass'], 'top4_normalized': descriptors['top4_normalized'],
                      **plan['reference_metrics']}
        pairs = {}
        for reference, desc in references.items():
            require(sha(desc['path']) == desc['sha256'], 'Reference metrics changed')
            pair = helper.paired_rgb(read(desc['path']), metrics[PRIMARY])
            pair.update(reference_metrics_sha256=desc['sha256'], candidate_metrics_sha256=descriptors[PRIMARY]['sha256'])
            path = output/f'paired_{PRIMARY}_minus_{reference}.json'; write(path, pair)
            pairs[reference] = {'path': str(path), 'sha256': sha(path)}
        clauses = helper.gate_clauses(read(pairs['E_rgb']['path'])['metrics'])
        path = output/'rgb_gate.json'
        write(path, {'primary_candidate': PRIMARY, 'reference': 'E_rgb', 'clauses': clauses,
                     'passed': all(clauses.values()), 'primary_pair_sha256': pairs['E_rgb']['sha256'],
                     'plan_sha256': report['plan_sha256'], 'automatic_adoption': False})
        require(report['VAL_rgb_reads'] == 50 and report['lpips_calls'] == 150, 'Scoring count differs')
        report.update(metrics=descriptors, comparisons=pairs, gate={'path': str(path), 'sha256': sha(path)},
                      gate_passed=all(clauses.values()), actual_numerics=base.numerical_flags(torch),
                      semantic_status='No semantic output or reassessment; current E remains unchanged')
    finally:
        report['peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0
        set_numerics(torch, before); report['numeric_flags_restored'] = base.numerical_flags(torch) == before
        report['actual_imports'] = base.actual_imports(Path(plan['source_snapshot'])/'scoring', {'bridge_rgs'})
    require(report['numeric_flags_restored'], 'Scoring flags were not restored')


def verify(plan):
    require(plan['specification'] == SPEC and files(plan['source_snapshot']) == plan['source_hashes'], 'Frozen source/spec changed')
    require(all(sha(p) == h for p, h in plan['input_hashes'].items()), 'Bound input changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--prepare', action='store_true')
    operation.add_argument('--render', action='store_true')
    operation.add_argument('--score', action='store_true')
    parser.add_argument('--training', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        require(args.training and args.output, 'Training and fresh output required')
        prepare(args); return
    require(args.expected_plan_sha256 and sha(args.plan) == args.expected_plan_sha256, 'Explicit plan SHA required')
    plan = read(args.plan)
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Execute frozen worker')
    output = Path(plan['output']); stage = 'render' if args.render else 'score'; started = time.monotonic()
    base = legacy(); write(output/f'{stage}_execution_started.json', {'plan_sha256': args.expected_plan_sha256, 'utc': base.utc()})
    report = {'status': 'failed', 'stage': stage, 'plan_sha256': args.expected_plan_sha256,
              'target_calls': 0, 'selector_calls': 0, 'source_depth_calls': 0, 'source_rgb_decodes': 0,
              'VAL_rgb_reads': 0, 'lpips_calls': 0, 'annotation_reads': 0, 'mask_outputs': 0,
              'teacher_calls': 0, 'field_updates': 0, 'backward': 0, 'optimizer_steps': 0}
    old_handler = signal.getsignal(signal.SIGALRM)
    def deadline(*_):
        raise TimeoutError(stage+' exceeded fixed budget')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(SPEC[stage+'_seconds'])
    try:
        verify(plan)
        (render_stage if args.render else score_stage)(plan, report)
        verify(plan); report.update(status='completed', sources_and_inputs_unchanged=True)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
        report.update(elapsed_seconds=time.monotonic()-started, finished_utc=base.utc())
        write(output/f'{stage}_execution_receipt.json', report)
    print(json.dumps({'status': report['status'], 'stage': stage,
                      'receipt_sha256': sha(output/f'{stage}_execution_receipt.json')}))


if __name__ == '__main__':
    main()
