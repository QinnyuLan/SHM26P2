"""Fixed 16-TRAIN startup-to-endpoint RGB recovery description, not selection.

Preparation requires the completed start-transfer and both fixed 6000-step
arms. Rendering uses the final-evaluation loader/source refresh, while scoring
uses the start-transfer's existing targets and valid support. No VAL inputs.
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
from types import SimpleNamespace

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
START = Path('/mnt/data/SHM2026/runs/ibgs_start_transfer_v1')
START_PLAN_SHA = '858b5601187e9d4d0841983614de46f816259d26bddc3e9b70cbb6f2da0ea42d'
ARMS = ('full', 'no_source')
READOUTS = ('raw', 'fused')
NAMES = ['002.png', '021.png', '041.png', '059.png', '079.png', '100.png',
         '118.png', '137.png', '156.png', '176.png', '200.png', '220.png',
         '241.png', '259.png', '278.png', '300.png']
SPEC = {'protocol': 'ibgs_warm_train_recovery_v1', 'arms': list(ARMS), 'readouts': list(READOUTS),
        'names': NAMES, 'steps_per_arm': 6000, 'source_depth_calls': 700,
        'target_calls': 32, 'native_arrays': 64, 'cached_target_loads': 16,
        'internal_seconds': 300, 'external_seconds': 360,
        'original_target_decodes': 0, 'VAL_reads': 0, 'semantic_label_reads': 0,
        'selection': 'No gate or endpoint selection; fixed-last TRAIN recovery description only',
        'source_scope': 'All 350 TRAIN views; each target excludes itself, other diagnostic targets may be sources',
        'score': 'Frozen start-transfer FP64 per-view valid RGB MSE and PSNR, raw and clipped separately; equal-view means',
        'camera': 'Raw manifest K/w2c doubles as in training; old AA metadata checked after FP32 cast',
        'barrier': 'All 64 predictions saved before cached scoring targets are loaded; source RGB remains permitted'}


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(payload)


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and p.suffix != '.pyc'}


def natural(folder):
    receipt, launch = read(folder/'execution_receipt.json'), read(folder/'launch_receipt.json')
    require(receipt['status'] == launch['status'] == 'completed' and launch['exit_code'] == 0
            and launch['natural_completion'] is True, 'Natural completion required before endpoint access')
    require(sha(folder/'execution_receipt.json') == launch['execution_receipt_sha256']
            and receipt['plan_sha256'] == launch['plan_sha256'] == sha(folder/'plan.json'), 'Completion lineage mismatch')
    return receipt


def endpoint_records(completed):
    rows = completed['arms']
    require(len(rows) == 2 and {r['arm'] for r in rows} == set(ARMS), 'Both fixed arms required')
    require(all(r['status'] == 'completed' and r['steps'] == 6000 for r in rows), 'Both fixed 6000-step endpoints required')
    return {r['arm']: r for r in rows}


def prediction_barrier(records):
    expected = {(a, r, n) for a in ARMS for r in READOUTS for n in NAMES}
    require(len(records) == 64 and {(v['arm'], v['readout'], v['name']) for v in records} == expected
            and all(v.get('path') and v.get('sha256') for v in records), 'All 64 fixed predictions must precede cached targets')


def start_helpers(snapshot):
    path = Path(snapshot)/'ibgs_start_transfer_helpers.py'
    spec = importlib.util.spec_from_file_location('fixed_start_metric_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    import torch

    training = args.training.resolve()
    require(args.training_plan_sha256 and sha(training/'plan.json') == args.training_plan_sha256, 'Explicit fixed training plan SHA required')
    completed = natural(training)
    endpoints = endpoint_records(completed)  # Gate before any checkpoint deserialization.
    require(sha(START/'plan.json') == START_PLAN_SHA, 'Fixed start-transfer plan required')
    start_receipt = natural(START)
    start = read(START/'plan.json')
    require(start_receipt['render_calls'] == start_receipt['cached_target_loads'] == 16
            and start_receipt['field_unchanged'] and start_receipt['sources_inputs_unchanged'], 'Incomplete start comparison')
    require(sha(START/'analysis.json') == start_receipt['analysis_sha256'], 'Start analysis changed')
    require(tree(start['source_snapshot']) == start['source_hashes'], 'Start source changed')
    parent = read(training/'plan.json')
    require({p: h for p, h in tree(parent['source_snapshot']).items() if Path(p).suffix in {'.py', '.md'}}
            == parent['source_hashes'], 'Training source changed')
    output = args.output.resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    inputs = {str(root/name): sha(root/name) for root in (training, START)
              for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json')}
    inputs.update({str(START/'analysis.json'): sha(START/'analysis.json'), **start['input_hashes']})
    contract_ref = None
    for arm, record in endpoints.items():
        require(sha(record['last_checkpoint']) == record['checkpoint_sha256']
                and sha(record['receipt_path']) == record['receipt_sha256'], 'Endpoint files changed')
        saved = torch.load(record['last_checkpoint'], map_location='cpu', weights_only=False)
        require(saved['status'] == 'completed' and saved['step'] == 6000 and saved['arm'] == arm
                and saved['source_plan_sha256'] == args.training_plan_sha256
                and saved['sh_degree'] == 3 and saved['field']['_xyz'].shape == (996009, 3), 'Wrong fixed endpoint')
        require(contract_ref is None or saved['data_contract'] == contract_ref, 'Different TRAIN contracts')
        contract_ref = saved['data_contract']
        inputs.update({record['last_checkpoint']: record['checkpoint_sha256'], record['receipt_path']: record['receipt_sha256']})
        del saved
    require(sha(contract_ref['path']) == contract_ref['sha256'] == start['input_hashes'][contract_ref['path']], 'Different start/final TRAIN contract')
    contract = read(contract_ref['path'])
    rows = {v['name']: v for v in contract['train_rows']}
    require(len(rows) == 350 and all(v['split'] == 'train' for v in rows.values()), 'TRAIN-only source bank required')
    require([v['name'] for v in start['views']] == NAMES, 'Fixed start population changed')
    for view in start['views']:
        row = rows[view['name']]
        require(row['K'] == view['camera']['K'] and row['w2c_original'] == view['camera']['w2c_original'], 'Different actual camera inputs')
    runtime = parent['runtime_sources']
    require(all(isinstance(v, str) for v in runtime.values()), 'Expected frozen path-to-SHA runtime bindings')
    inputs.update(contract['pixel_hashes'])
    inputs[str(ROOT/'uv.lock')] = sha(ROOT/'uv.lock')
    for path, expected in {**inputs, **runtime}.items():
        require(sha(path) == expected, 'Bound input changed: '+path)
    output.mkdir(); snapshot = output/'source_snapshot'; snapshot.mkdir()
    shutil.copytree(parent['source_snapshot'], snapshot/'render')
    shutil.copy2(Path(__file__), snapshot/Path(__file__).name)
    shutil.copy2(Path(start['source_snapshot'])/'check_ibgs_start_transfer.py', snapshot/'ibgs_start_transfer_helpers.py')
    shutil.copy2(ROOT/'tests/test_ibgs_warm_train_recovery.py', snapshot/'test_ibgs_warm_train_recovery.py')
    shutil.copy2(ROOT/'docs/ibgs_warm_train_recovery_protocol.md', snapshot/'ibgs_warm_train_recovery_protocol.md')
    plan = {'specification': SPEC, 'output': str(output), 'source_snapshot': str(snapshot),
            'source_hashes': tree(snapshot), 'input_hashes': inputs, 'runtime_sources': runtime,
            'training_plan_sha256': args.training_plan_sha256, 'data_contract': contract_ref,
            'endpoints': {a: {'path': r['last_checkpoint'], 'sha256': r['checkpoint_sha256']} for a, r in endpoints.items()},
            'start_plan_sha256': START_PLAN_SHA, 'start_analysis': str(START/'analysis.json'),
            'views': start['views'], 'prepare_cached_target_payload_loads': 0,
            'prepare_cuda_initialized': torch.cuda.is_initialized()}
    require(not plan['prepare_cuda_initialized'], 'CPU-only prepare required')
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC and Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Frozen entry and contract required')
    require(tree(plan['source_snapshot']) == plan['source_hashes'], 'Changed frozen source')
    for path, expected in {**plan['input_hashes'], **plan['runtime_sources']}.items():
        require(sha(path) == expected, 'Changed bound input: '+path)


def score(plan, records):
    prediction_barrier(records)
    helpers = start_helpers(plan['source_snapshot'])
    previous = {v['name']: v for v in read(plan['start_analysis'])['views']}
    index = {(v['arm'], v['readout'], v['name']): v for v in records}
    rows = []
    for view in plan['views']:
        with np.load(view['target']['path'], allow_pickle=False) as cache:
            target, valid = cache['rgb'], cache['valid']
            valid_sha = hashlib.sha256(valid.tobytes()).hexdigest()
            for arm in ARMS:
                for readout in READOUTS:
                    record = index[arm, readout, view['name']]
                    require(sha(record['path']) == record['sha256'] and record['valid_sha256'] == valid_sha, 'Prediction or fixed support changed')
                    value = np.load(record['path'], allow_pickle=False)
                    raw = helpers.metrics(value, target, valid)
                    clipped = helpers.metrics(np.clip(value, 0., 1.), target, valid)
                    comparisons = {}
                    for reference in ('old_AA', 'IBGS_clipped'):
                        old = previous[view['name']][reference]
                        require(old['valid_pixels'] == clipped['valid_pixels'], 'Reference support changed')
                        comparisons[reference] = {'mse_delta': clipped['mse']-old['mse'],
                            'psnr_delta': clipped['psnr']-old['psnr'] if clipped['psnr'] is not None and old['psnr'] is not None else None}
                    rows.append({'arm': arm, 'readout': readout, 'name': view['name'],
                                 'unclipped': raw, 'clipped': clipped, 'clipped_minus_reference': comparisons})
    summary = {}
    for arm in ARMS:
        for readout in READOUTS:
            selected = [v for v in rows if (v['arm'], v['readout']) == (arm, readout)]
            summary[arm+'/'+readout] = {kind: {'equal_view_mse': float(np.mean([v[kind]['mse'] for v in selected])),
                'mean_per_view_psnr': float(np.mean([v[kind]['psnr'] for v in selected])) if all(v[kind]['psnr'] is not None for v in selected) else None}
                for kind in ('unclipped', 'clipped')}
    return {'views': rows, 'summary': summary, 'adoption_gate': None, 'scope': SPEC['selection'],
            'source_scope': SPEC['source_scope'], 'cached_target_loads': 16}


def run(plan, report):
    import torch

    root = Path(plan['source_snapshot'])/'render'
    sys.path[:0] = [str(root/'official'), str(root)]
    from color_aggregation_network import ColorFusionResidualNet, fuse_color
    from diff_plane_rasterization import _C
    from gaussian_renderer import render, render_depth

    from bridge_rgs.ibgs_adapter import TrainScene
    from bridge_rgs.ibgs_warm_training import SourceAblatedNet, fuse_training, load_saved_field

    require(plan['runtime_sources'].get(str(Path(_C.__file__).resolve())) == sha(_C.__file__), 'Wrong renderer binary')
    torch.set_num_threads(4)
    flags = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    contract = read(plan['data_contract']['path'])
    neighbors = {r['name']: [n['name'] for n in r['neighbors_4']] for r in contract['neighbors']['views']}
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    options = SimpleNamespace(shuffle_source_frame=False, enable_exposure_correction=False,
                              nb_visible_src_frames=3, residual_resolution_scale=1.)
    output = Path(plan['output']); records = []
    try:
        with torch.inference_mode():
            for arm in ARMS:
                saved = torch.load(plan['endpoints'][arm]['path'], map_location='cpu', weights_only=False)
                field, background = load_saved_field(saved)
                scene = TrainScene(contract['train_rows'], contract['pixel_hashes'], neighbors, field, contract['manifest_scene_radius'])
                cameras = {c.image_name: c for c in scene.getTrainCameras()}
                valid_sha = hashlib.sha256(scene.valid_cpu.numpy().tobytes()).hexdigest()
                net = ColorFusionResidualNet(height=989, width=1320).cuda()
                net.load_state_dict(saved['head']); net.eval()
                wrapped = SourceAblatedNet(net, arm).eval()
                for index, camera in enumerate(scene.getTrainCameras()):
                    scene.rendered_depth_list[index] = render_depth(camera, field, scene, pipe, options, background, True, 4, 4, .01)
                    report['source_depth_calls'] += 1
                require(scene.rendered_depth_list.updates == 350, 'Incomplete source depth refresh')
                for view in plan['views']:
                    camera = cameras[view['name']]
                    require(view['name'] not in camera.nearest_names, 'Target identity used as source')
                    require(camera.world_view_transform.T.contiguous().cpu().numpy().tobytes() ==
                            np.asarray(view['camera']['w2c_original'], np.float32).tobytes(), 'Actual camera differs from start')
                    package = render(camera, field, scene, pipe, options, background, learnt_normal=True,
                                     nb_src_frames=4, buffer_length=4, depth_error_threshold=.01,
                                     do_find_closest_frame=False, do_render_src_depth=False,
                                     render_geo=True, return_depth_normal=False)
                    report['target_calls'] += 1
                    fusion = fuse_training(package, wrapped, options, 6000, fuse_color)
                    report['empty_fusion_views'] += int(fusion['empty_support'])
                    for readout, tensor in (('raw', package['render']), ('fused', fusion['image_pred'])):
                        value = tensor.permute(1, 2, 0).contiguous().cpu().numpy()
                        require(value.dtype == np.float32 and value.shape == (989, 1320, 3) and np.isfinite(value).all(), 'Invalid native RGB')
                        path = output/'arrays'/arm/readout/(view['name']+'.npy'); path.parent.mkdir(parents=True, exist_ok=True)
                        with path.open('xb') as stream:
                            np.save(stream, value, allow_pickle=False)
                        records.append({'arm': arm, 'readout': readout, 'name': view['name'], 'path': str(path),
                            'sha256': sha(path), 'valid_sha256': valid_sha, 'shape': list(value.shape), 'dtype': 'float32',
                            'source_names': camera.nearest_names})
                    del package, fusion
                report['source_rgb_decodes'] += scene.original_image_list.decode_count
                require(scene.original_image_list.decoded_names <= set(cameras), 'Non-TRAIN source opened')
                del scene, cameras, field, background, net, wrapped, saved
                torch.cuda.empty_cache()
                print(json.dumps({'arm': arm, 'saved_outputs': len(records)}), flush=True)
        prediction_barrier(records)
        require(report['source_depth_calls'] == 700 and report['target_calls'] == 32, 'Fixed render count changed')
        write(output/'prediction_receipt.json', {'records': records, 'cached_target_loads': 0})
        analysis = score(plan, records); write(output/'analysis.json', analysis)
        report.update(predictions_sha256=sha(output/'prediction_receipt.json'), analysis_sha256=sha(output/'analysis.json'),
                      cached_target_loads=16, summary=analysis['summary'], peak_allocated_bytes=torch.cuda.max_memory_allocated())
        report['actual_imports'] = {}
        for name, module in list(sys.modules.items()):
            if name.split('.')[0] in ('bridge_rgs', 'scene', 'utils', 'gaussian_renderer', 'arguments', 'color_aggregation_network') and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve(); relative = str(path.relative_to(plan['source_snapshot']))
                require(sha(path) == plan['source_hashes'][relative], 'Unbound actual import')
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}
    finally:
        torch.set_float32_matmul_precision(flags[0])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = flags[1:]
        report['numeric_flags_restored'] = flags == (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                                                   torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)


def main():
    parser = argparse.ArgumentParser()
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--prepare', action='store_true'); operation.add_argument('--execute', action='store_true')
    parser.add_argument('--training', type=Path); parser.add_argument('--training-plan-sha256')
    parser.add_argument('--output', type=Path); parser.add_argument('--plan', type=Path); parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        prepare(args); return
    require(args.expected_plan_sha256 and sha(args.plan) == args.expected_plan_sha256, 'Explicit fixed plan SHA required')
    plan = read(args.plan); output = Path(plan['output']); started = time.monotonic()
    write(output/'execution_started.json', {'plan_sha256': args.expected_plan_sha256})
    report = {'status': 'failed', 'plan_sha256': args.expected_plan_sha256, 'source_depth_calls': 0,
              'target_calls': 0, 'source_rgb_decodes': 0, 'empty_fusion_views': 0, 'cached_target_loads': 0,
              'backward': 0, 'optimizer_steps': 0, 'original_target_decodes': 0, 'VAL_reads': 0, 'semantic_label_reads': 0}
    def timeout(*_):
        raise TimeoutError('Fixed TRAIN endpoint diagnostic deadline')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan); run(plan, report); verify(plan)
        report.update(status='completed', sources_inputs_unchanged=True)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        signal.alarm(0); report['elapsed_seconds'] = time.monotonic()-started
        write(output/'execution_receipt.json', report)
    print(json.dumps(report['summary'], allow_nan=False))


if __name__ == '__main__':
    main()
