"""Four fixed warm-IBGS RGB endpoints: isolated render, then official scoring.

Prepare only after both training arms naturally complete. No semantic payloads.
The renderer environment never imports LPIPS or opens a VAL image.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.dont_write_bytecode = True

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
ARMS = ('full', 'no_source')
READOUTS = ('raw', 'fused')
SCORING_PARENT = RUNS/'rgb_fixed_ensemble_v1'
SCORER_SHA = '9b3799367a983495677122bf1bf75b06116342fde703466e43e3d8de092433f0'
REFERENCE_PATHS = {
    'aa_1m': RUNS/'rgb_capacity_1m_reference_v1/evaluation_official/official_metrics.json',
    'E_rgb': SCORING_PARENT/'rgb_metrics.json',
}
REFERENCE_HASHES = {
    'aa_1m': '0b4b90fca81ad4f220794b9656ce308cca4dc68a8efe83769efbfde2cd51100e',
    'E_rgb': '40d022bc522315eaa864fa114261f6e66c5ca6991b3de74bf309a3a4e2ad2ef9',
}
SPEC = {
    'protocol': 'ibgs_warm_evaluation_v1', 'arms': list(ARMS), 'readouts': list(READOUTS),
    'primary_candidate': 'full/fused', 'views': 50, 'train_sources': 350,
    'source_depth_calls': 700, 'target_calls': 100, 'native_arrays': 200, 'delivered_pngs': 200,
    'bootstrap_repeats': 5000, 'bootstrap_seed': 20260926,
    'rgb_gate': {'psnr_gain_db': .15, 'psnr_ci_lower_strict': 0., 'ssim_min': 0., 'lpips_max': 0.},
    'render_seconds': 600, 'render_external_seconds': 660,
    'score_seconds': 300, 'score_external_seconds': 360,
    'semantic_outputs': 0, 'annotation_reads': 0, 'teacher_calls': 0,
    'selection': 'fixed last, all four reported; no automatic E replacement',
}


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


def utc():
    return datetime.now(UTC).isoformat()


def files(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*')) if p.is_file() and p.suffix in {'.py', '.md'}}


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verified(path, expected):
    require(sha(path) == expected, 'Changed bound file: '+str(path))


def natural(folder, receipt_name='execution_receipt.json', launch_name='launch_receipt.json'):
    receipt, launch = read(Path(folder)/receipt_name), read(Path(folder)/launch_name)
    require(receipt['status'] == launch['status'] == 'completed' and launch['exit_code'] == 0
            and launch['natural_completion'] is True, 'Natural successful completion required')
    if 'execution_receipt_sha256' in launch:
        verified(Path(folder)/receipt_name, launch['execution_receipt_sha256'])
    return receipt


def runtime_records(value):
    result = {}
    for key, item in value.items():
        if isinstance(item, str):
            result[key] = item
        else:
            result[item['path']] = item['sha256']
    return result


def numerical_flags(torch):
    return {'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'benchmark': torch.backends.cudnn.benchmark,
            'matmul_precision': torch.get_float32_matmul_precision()}


def actual_imports(snapshot, prefixes):
    result = {}
    for name, module in sys.modules.copy().items():
        path = getattr(module, '__file__', None)
        if path and name.split('.')[0] in prefixes:
            path = Path(path).resolve()
            require(path.is_relative_to(snapshot), 'Non-frozen import: '+str(path))
            result[name] = {'path': str(path), 'sha256': sha(path)}
    return result


def neighbors(camera, rows):
    """Same FP64 distance/angle/name order and strict filters as data contract."""
    target = np.asarray(camera['w2c'], np.float64)
    target_center = -np.linalg.solve(target[:3, :3], target[:3, 3])
    target_ray = np.linalg.solve(target[:3, :3], np.asarray([0., 0., 1.]))
    target_ray /= np.linalg.norm(target_ray)
    candidates = []
    for index, row in enumerate(rows):
        if row['name'] == camera['name']:
            continue
        pose = np.asarray(row['w2c_original'], np.float64)
        distance = float(np.linalg.norm(-np.linalg.solve(pose[:3, :3], pose[:3, 3])-target_center))
        direction = np.linalg.solve(pose[:3, :3], np.asarray([0., 0., 1.]))
        direction /= np.linalg.norm(direction)
        angle = float(np.degrees(np.arccos(np.clip(direction@target_ray, -1., 1.))))
        if .01 < distance < 1.5 and angle < 30.:
            candidates.append((distance, angle, row['name'], index))
    candidates.sort()
    return [x[3] for x in candidates[:4]]


def prediction_barrier(records):
    expected = {(a, r) for a in ARMS for r in READOUTS}
    require(len(records) == 200 and {(x['arm'], x['readout']) for x in records} == expected,
            'Exactly four fixed endpoints and 200 predictions required')
    names = None
    for arm, branch in sorted(expected):
        found = [x['name'] for x in records if x['arm'] == arm and x['readout'] == branch]
        require(len(found) == len(set(found)) == 50, 'Missing or duplicate camera prediction')
        names = sorted(found) if names is None else names
        require(sorted(found) == names, 'Endpoint camera populations differ')


def scoring_modules(snapshot):
    sys.path.insert(0, str(Path(snapshot)/'scoring'))
    from bridge_rgs import official_evaluate as official
    helpers = load_file('ibgs_rgb_score_helpers', Path(snapshot)/'rgb_scoring_helpers.py')
    verified(official.__file__, SCORER_SHA)
    return official, helpers


def prepare(args):
    import torch
    training = args.training.resolve()
    require(args.training_plan_sha256 is not None, 'Explicit completed training plan SHA required')
    verified(training/'plan.json', args.training_plan_sha256)
    completed = natural(training)
    require(completed['plan_sha256'] == args.training_plan_sha256, 'Training receipt/plan mismatch')
    parent = read(training/'plan.json')
    require({a['arm'] for a in completed['arms']} == set(ARMS), 'Two fixed training arms required')
    output = args.output.resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk directory required')
    # No active checkpoint is hashed or loaded before the natural-completion gate.
    inputs = {str(training/n): sha(training/n) for n in ('plan.json', 'execution_receipt.json', 'launch_receipt.json')}
    endpoints, contract_ref = {}, None
    for arm in completed['arms']:
        require(arm['status'] == 'completed' and arm['steps'] == 6000, 'Incomplete fixed endpoint')
        verified(arm['last_checkpoint'], arm['checkpoint_sha256'])
        verified(arm['receipt_path'], arm['receipt_sha256'])
        saved = torch.load(arm['last_checkpoint'], map_location='cpu', weights_only=False)
        require(saved['status'] == 'completed' and saved['arm'] == arm['arm'] and saved['step'] == 6000 and saved['sh_degree'] == 3
                and saved['field']['_xyz'].shape == (996009, 3), 'Endpoint identity changed')
        require(saved['source_plan_sha256'] == args.training_plan_sha256, 'Endpoint lineage mismatch')
        ref = saved['data_contract']
        require(contract_ref is None or ref == contract_ref, 'Different TRAIN contracts')
        contract_ref = ref
        endpoints[arm['arm']] = {'path': arm['last_checkpoint'], 'sha256': arm['checkpoint_sha256']}
        inputs.update({arm['last_checkpoint']: arm['checkpoint_sha256'], arm['receipt_path']: arm['receipt_sha256']})
        del saved
    verified(contract_ref['path'], contract_ref['sha256'])
    contract = read(contract_ref['path'])
    require(len(contract['train_rows']) == 350 and all(r['split'] == 'train' for r in contract['train_rows']),
            'TRAIN-only source bank required')
    inputs[contract_ref['path']] = contract_ref['sha256']
    inputs.update(contract['pixel_hashes'])
    runtime = runtime_records(parent['runtime_sources'])
    inputs.update(runtime)
    require(files(parent['source_snapshot']) == parent['source_hashes'], 'Training snapshot changed')
    output.mkdir()
    snapshot = output/'source_snapshot'; snapshot.mkdir()
    shutil.copytree(parent['source_snapshot'], snapshot/'render')
    shutil.copytree(SCORING_PARENT/'source_snapshot/bridge_rgs', snapshot/'scoring/bridge_rgs')
    shutil.copy2(SCORING_PARENT/'source_snapshot/evaluate_fixed_rgb_ensemble.py', snapshot/'rgb_scoring_helpers.py')
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    shutil.copy2(ROOT/'tests/test_ibgs_warm_evaluation.py', snapshot/'test_ibgs_warm_evaluation.py')
    shutil.copy2(ROOT/'docs/ibgs_warm_evaluation_protocol.md', snapshot/'ibgs_warm_evaluation_protocol.md')
    official, helpers = scoring_modules(snapshot)
    reference = official.read_official_reference(workspace_root=ROOT)
    old_receipt = read(RUNS/'rgb_capacity_1m_reference_v1/evaluation_official/execution_receipt.json')
    old_sources = {r['camera']['name']: r for r in old_receipt['source_records']}
    # Expected RGB SHA comes from completed metadata; target bytes remain unopened.
    from bridge_rgs.evaluate import distortion_render_grid
    views = []
    for camera, target in zip(reference['cameras'], reference['targets'], strict=True):
        old = old_sources[camera['name']]
        require(camera == old['camera'], 'Original camera population changed')
        K, w, h, _ = distortion_render_grid(camera['K'], camera['distortion'], camera['width'], camera['height'], 'colmap_corner_v2')
        common = contract['common_camera']
        require((w, h) == (common['width'], common['height']) == (1320, 989)
                and np.array_equal(K, np.asarray(common['K'], np.float32))
                and np.array_equal(np.asarray(camera['K'], np.float64), np.asarray(common['K'], np.float64)),
                'This IBGS port requires the official canvas to equal the shared native source camera')
        require(camera['name'] not in {r['name'] for r in contract['train_rows']}, 'VAL alias in TRAIN bank')
        indices = neighbors(camera, contract['train_rows'])
        views.append({'camera': camera, 'source_image_path': target['source_image_path'],
                      'source_image_sha256': old['source_image_sha256'],
                      'render_K': K.tolist(), 'width': w, 'height': h, 'source_indices': indices,
                      'source_names': [contract['train_rows'][i]['name'] for i in indices]})
    old_plan = read(SCORING_PARENT/'plan.json')
    ensemble_receipt = read(SCORING_PARENT/'execution_receipt.json')
    require(ensemble_receipt['status'] == old_receipt['status'] == 'completed'
            and ensemble_receipt['metrics_sha256'] == REFERENCE_HASHES['E_rgb']
            and old_receipt['official_metrics_sha256'] == REFERENCE_HASHES['aa_1m'],
            'References must be bound completed evaluations')
    inputs[str(SCORING_PARENT/'execution_receipt.json')] = sha(SCORING_PARENT/'execution_receipt.json')
    inputs[str(SCORING_PARENT/'plan.json')] = sha(SCORING_PARENT/'plan.json')
    inputs[str(RUNS/'rgb_capacity_1m_reference_v1/evaluation_official/execution_receipt.json')] = sha(RUNS/'rgb_capacity_1m_reference_v1/evaluation_official/execution_receipt.json')
    for key, path in REFERENCE_PATHS.items():
        verified(path, REFERENCE_HASHES[key]); inputs[str(path)] = REFERENCE_HASHES[key]
        metrics = read(path); helpers.rgb_rows(metrics)
        require(metrics.get('official_evaluation_fingerprint', metrics.get('inherited_common_reference_fingerprint'))
                == old_plan['reference_fingerprint'], 'Reference fingerprints differ')
    for path in old_plan['perceptual_weights']:
        inputs[path] = old_plan['input_hashes'][path]
    inputs[str(ROOT/'uv.lock')] = sha(ROOT/'uv.lock')
    for path, expected in inputs.items():
        verified(path, expected)
    plan = {'status': 'prepared_after_completed_training', 'specification': SPEC, 'output': str(output),
            'training': str(training), 'training_plan_sha256': args.training_plan_sha256,
            'source_snapshot': str(snapshot), 'source_hashes': files(snapshot), 'input_hashes': inputs,
            'runtime_sources': runtime, 'endpoints': endpoints, 'data_contract': contract_ref,
            'views': views, 'reference_fingerprint': old_plan['reference_fingerprint'],
            'reference_metrics': {k: {'path': str(p), 'sha256': REFERENCE_HASHES[k]} for k, p in REFERENCE_PATHS.items()},
            'perceptual_weights': old_plan['perceptual_weights'], 'scoring_protocol': official.SCORING_PROTOCOL,
            'scoring_numerics': numerical_flags(torch),
            'main_runtime': {k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'opencv-python-headless', 'scikit-image', 'lpips')},
            'prepare_VAL_payload_reads': 0, 'prepare_cuda_initialized': torch.cuda.is_initialized()}
    require(not plan['prepare_cuda_initialized'], 'Prepare must remain CPU-only')
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC, 'Changed evaluation specification')
    require(files(plan['source_snapshot']) == plan['source_hashes'], 'Changed frozen source set')
    for path, expected in plan['input_hashes'].items():
        verified(path, expected)


def render_stage(plan, report):
    import torch
    root = Path(plan['source_snapshot'])/'render'
    sys.path[:0] = [str(root/'official'), str(root)]
    from color_aggregation_network import ColorFusionResidualNet, fuse_color
    from diff_plane_rasterization import _C
    from gaussian_renderer import render, render_depth

    from bridge_rgs.ibgs_adapter import BridgeCamera, TrainScene
    from bridge_rgs.ibgs_warm_training import SourceAblatedNet, fuse_training, load_saved_field
    require(plan['runtime_sources'].get(str(Path(_C.__file__).resolve())) == sha(_C.__file__), 'Wrong renderer binary')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    contract = read(plan['data_contract']['path'])
    neighbor_map = {r['name']: [n['name'] for n in r['neighbors_4']] for r in contract['neighbors']['views']}
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    options = SimpleNamespace(shuffle_source_frame=False, enable_exposure_correction=False,
                              nb_visible_src_frames=3, residual_resolution_scale=1.)
    output = Path(plan['output']); records = []
    with torch.inference_mode():
        for arm in ARMS:
            saved = torch.load(plan['endpoints'][arm]['path'], map_location='cpu', weights_only=False)
            field, background = load_saved_field(saved)
            scene = TrainScene(contract['train_rows'], contract['pixel_hashes'], neighbor_map,
                               field, contract['manifest_scene_radius'])
            net = ColorFusionResidualNet(height=989, width=1320).cuda()
            net.load_state_dict(saved['head']); net.eval()
            wrapped = SourceAblatedNet(net, arm).eval()
            for index, camera in enumerate(scene.getTrainCameras()):
                scene.rendered_depth_list[index] = render_depth(camera, field, scene, pipe, options, background,
                                                               True, 4, 4, .01)
                report['source_depth_calls'] += 1
            require(scene.rendered_depth_list.updates == 350, 'Incomplete final-field depth refresh')
            for view in plan['views']:
                source = view['camera']
                row = {'name': source['name'], 'image_id': source['image_id'], 'K': contract['common_camera']['K'],
                           'w2c_original': source['w2c'], 'width': view['width'], 'height': view['height']}
                camera = BridgeCamera(row, -1)
                require(all(getattr(camera, key) == getattr(scene.cameras[0], key)
                            for key in ('Fx', 'Fy', 'Cx', 'Cy', 'FoVx', 'FoVy', 'image_width', 'image_height')),
                        'Target/source actual camera intrinsics differ')
                camera.nearest_id, camera.nearest_names = view['source_indices'], view['source_names']
                package = render(camera, field, scene, pipe, options, background,
                                 learnt_normal=True, nb_src_frames=4, buffer_length=4, depth_error_threshold=.01,
                                 do_find_closest_frame=False, do_render_src_depth=False,
                                 render_geo=True, return_depth_normal=False)
                report['target_calls'] += 1
                fusion = fuse_training(package, wrapped, options, 6000, fuse_color)
                for branch, tensor in (('raw', package['render']), ('fused', fusion['image_pred'])):
                    value = tensor.permute(1, 2, 0).contiguous().cpu().numpy()
                    require(value.dtype == np.float32 and value.shape == (989, 1320, 3)
                            and np.isfinite(value).all(), 'Invalid native RGB endpoint')
                    path = output/'native'/arm/branch/(Path(source['name']).stem+'.npy')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('xb') as stream:
                        np.save(stream, value, allow_pickle=False)
                    records.append({'arm': arm, 'readout': branch, 'name': source['name'],
                                    'path': str(path), 'sha256': sha(path), 'shape': list(value.shape), 'dtype': 'float32'})
                report['empty_fusion_views'] += int(fusion['empty_support'])
                del package, fusion
            report['source_rgb_decodes'] += scene.original_image_list.decode_count
            require(scene.original_image_list.decoded_names <= {r['name'] for r in contract['train_rows']},
                    'Renderer opened a non-TRAIN image')
            del field, background, scene, net, wrapped, saved
            torch.cuda.empty_cache()
            print(json.dumps({'arm': arm, 'completed_native_outputs': len(records)}), flush=True)
    prediction_barrier(records)
    require(report['source_depth_calls'] == 700 and report['target_calls'] == 100, 'Wrong render budget')
    write(output/'native_predictions.json', {'status': 'completed', 'records': records, 'finished_utc': utc(),
                                           'VAL_payload_reads': 0})
    report.update(native_predictions={'path': str(output/'native_predictions.json'), 'sha256': sha(output/'native_predictions.json')},
                  peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                  actual_numerics=numerical_flags(torch),
                  actual_imports=actual_imports(root, {'bridge_rgs', 'scene', 'utils', 'gaussian_renderer',
                                                     'arguments', 'color_aggregation_network'}))


def score_stage(plan, report):
    import cv2
    import torch
    output = Path(plan['output'])
    rendered = natural(output, 'render_execution_receipt.json', 'render_launch_receipt.json')
    require(rendered['plan_sha256'] == report['plan_sha256'], 'Renderer used another plan')
    verified(rendered['native_predictions']['path'], rendered['native_predictions']['sha256'])
    native = read(rendered['native_predictions']['path'])['records']; prediction_barrier(native)
    official, helpers = scoring_modules(plan['source_snapshot'])
    from bridge_rgs.evaluate import distortion_render_grid
    require({k: importlib.metadata.version(k) for k in plan['main_runtime']} == plan['main_runtime'], 'Main scoring runtime changed')
    views = {v['camera']['name']: v for v in plan['views']}
    pngs = []
    for record in native:
        verified(record['path'], record['sha256'])
        rgb = np.load(record['path'], allow_pickle=False)
        require(rgb.dtype == np.float32 and rgb.shape == (989, 1320, 3) and np.isfinite(rgb).all(), 'Bad native payload')
        camera = views[record['name']]['camera']
        _, _, _, back = distortion_render_grid(camera['K'], camera['distortion'], camera['width'], camera['height'], 'colmap_corner_v2')
        rgb = np.clip(rgb, 0., 1.)
        if back is not None:
            rgb = cv2.remap(rgb, back[..., 0], back[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        delivered = np.rint(rgb*255).astype(np.uint8)
        path = output/'predictions'/record['arm']/record['readout']/record['name']
        path.parent.mkdir(parents=True, exist_ok=True)
        require(not path.exists() and cv2.imwrite(str(path), delivered[..., ::-1]), 'Could not save fresh PNG')
        pngs.append({**{k: record[k] for k in ('arm', 'readout', 'name')}, 'path': str(path), 'sha256': sha(path)})
    prediction_barrier(pngs)
    write(output/'predictions_receipt.json', {'status': 'all_200_pngs_before_GT', 'records': pngs,
                                            'finished_utc': utc(), 'VAL_payload_reads': 0})
    report['predictions_finished_utc'] = utc()
    torch.set_num_threads(8); cv2.setNumThreads(8)
    flags = plan['scoring_numerics']
    torch.backends.cudnn.allow_tf32 = flags['cudnn_tf32']
    torch.backends.cudnn.benchmark = flags['benchmark']
    torch.set_float32_matmul_precision(flags['matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32 = flags['matmul_tf32']
    require(numerical_flags(torch) == flags, 'Scoring numerical flags differ')
    torch.cuda.reset_peak_memory_stats()
    perceptual = official._lpips('cuda')
    rows = {(a, r): [] for a in ARMS for r in READOUTS}
    index = {(p['arm'], p['readout'], p['name']): p for p in pngs}
    report['scoring_started_utc'] = utc()
    for view in plan['views']:
        camera = view['camera']; target_bytes = Path(view['source_image_path']).read_bytes()
        report['VAL_rgb_reads'] += 1
        require(hashlib.sha256(target_bytes).hexdigest() == view['source_image_sha256'], 'Original RGB changed')
        target = official._decode_rgb(target_bytes, camera['width'], camera['height'], 'target')
        for arm in ARMS:
            for branch in READOUTS:
                record = index[arm, branch, camera['name']]
                payload = Path(record['path']).read_bytes()
                require(hashlib.sha256(payload).hexdigest() == record['sha256'], 'Prediction changed')
                prediction = official._decode_rgb(payload, camera['width'], camera['height'], 'prediction')
                score = helpers.score_rgb_only(official.score_official_arrays, prediction, target, perceptual, 'cuda')
                rows[arm, branch].append(dict(name=camera['name'], width=camera['width'], height=camera['height'],
                                             source_rgb_sha256=view['source_image_sha256'], rgb_sha256=record['sha256'], **score))
    metrics = {}
    for (arm, branch), values in rows.items():
        metrics[arm, branch] = {'views': values, 'validation_views': 50,
                               'inherited_common_reference_fingerprint': plan['reference_fingerprint'],
                               'scoring_protocol': plan['scoring_protocol'],
                               **{k: float(np.mean([v[k] for v in values])) for k in helpers.RGB_KEYS}}
        write(output/f'metrics_{arm}_{branch}.json', metrics[arm, branch])
    comparisons = {}
    for (arm, branch), candidate in metrics.items():
        for reference, desc in plan['reference_metrics'].items():
            prior = read(desc['path'])
            result = helpers.paired_rgb(prior, candidate)
            result['gate_clauses'] = helpers.gate_clauses(result['metrics'])
            name = f'{arm}_{branch}_minus_{reference}'
            write(output/f'paired_{name}.json', result); comparisons[name] = result
    for branch in READOUTS:
        result = helpers.paired_rgb(metrics['no_source', branch], metrics['full', branch])
        result['gate_clauses'] = helpers.gate_clauses(result['metrics'])
        name = f'full_minus_no_source_{branch}'
        write(output/f'paired_{name}.json', result); comparisons[name] = result
    report.update(comparisons={k: {'path': str(output/f'paired_{k}.json'), 'sha256': sha(output/f'paired_{k}.json'),
                                  'gates': v['gate_clauses']} for k, v in comparisons.items()},
                  metrics={f'{a}/{r}': {'path': str(output/f'metrics_{a}_{r}.json'),
                                        'sha256': sha(output/f'metrics_{a}_{r}.json')} for a, r in metrics},
                  actual_numerics=numerical_flags(torch),
                  actual_imports=actual_imports(Path(plan['source_snapshot'])/'scoring', {'bridge_rgs'}),
                  peak_allocated_bytes=int(torch.cuda.max_memory_allocated()), semantic_status='E unchanged; not re-evaluated')


def main():
    parser = argparse.ArgumentParser()
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--prepare', action='store_true')
    operation.add_argument('--render', action='store_true')
    operation.add_argument('--score', action='store_true')
    parser.add_argument('--training', type=Path)
    parser.add_argument('--training-plan-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        prepare(args); return
    require(args.expected_plan_sha256 is not None, 'Explicit execution plan SHA required')
    verified(args.plan, args.expected_plan_sha256)
    plan = read(args.plan); verify(plan)
    stage = 'render' if args.render else 'score'
    output = Path(plan['output']); started = time.monotonic()
    write(output/f'{stage}_execution_started.json', {'plan_sha256': sha(args.plan), 'started_utc': utc()})
    report = {'status': 'failed', 'plan_sha256': sha(args.plan), 'stage': stage, 'started_utc': utc(),
              'source_depth_calls': 0, 'target_calls': 0, 'source_rgb_decodes': 0, 'empty_fusion_views': 0,
              'VAL_rgb_reads': 0, 'annotation_reads': 0, 'teacher_calls': 0, 'optimizer_steps': 0}

    def timeout(*_):
        raise TimeoutError(stage+' exceeded fixed budget')

    signal.signal(signal.SIGALRM, timeout); signal.alarm(SPEC[stage+'_seconds'])
    try:
        (render_stage if args.render else score_stage)(plan, report)
        verify(plan)
        report.update(status='completed', sources_and_inputs_unchanged=True)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, finished_utc=utc())
        write(output/f'{stage}_execution_receipt.json', report)
        signal.alarm(0)
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
