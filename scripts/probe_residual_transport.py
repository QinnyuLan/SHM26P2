"""Fixed TRAIN source-image leave-out residual transport; CPU preparation first."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
PARENT = Path('/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1')
RUNTIME = Path('/mnt/data/SHM2026/runs/readout_geometry_probe_v1')
GUARD_PARENT = Path('/mnt/data/SHM2026/runs/scene_geometry_routes_evaluation_v1')
PARENT_SHA = '40191687627d54da91b78c8b819e83bc04b04b986ae6568efe042c6dbcd447f3'
BASE_SHA = '35b489fe45ad34ad279018609b6d5eaf6493e4c0d82f9c86ec16925268078092'
NAMES = ['002.png', '021.png', '041.png', '059.png', '079.png', '100.png',
         '118.png', '137.png', '156.png', '176.png', '200.png', '220.png',
         '241.png', '259.png', '278.png', '300.png']
SPEC = {
    'protocol': 'residual_transport_probe_v1', 'targets': NAMES,
    'base_sha256': BASE_SHA, 'pixel_protocol': 'colmap_corner_v2',
    'sources_per_target': 4, 'minimum_alpha': .95, 'depth_relative_tolerance': .01,
    'distance': 'squared center distance / manifest scene_radius^2 + squared SO3 angle / pi^2',
    'source_exclusion': 'all 16 targets, then closest remaining camera per target; no coverage rescue',
    'wrong_control': 'reflect source residual and RGB-valid horizontally, never ED/alpha',
    'common_support': 'true geometry visibility AND normal RGB-valid AND mirrored RGB-valid',
    'folds': 'A even / B odd target indices; each target uses other-fold fitted scalar',
    'fit': 'true/wrong each 2 scalars in [0,1], equal-view unclipped full-valid RGB MSE',
    'internal_seconds': 120, 'external_seconds': 180,
    'teacher_calls': 0, 'backward': 0, 'optimizer_steps': 0, 'semantic_GT': 0, 'VAL': 0,
    'scope': 'source-image leave-out and scalar cross-fit only; base model fitted all targets; no adoption gate',
}


def require(value, message):
    if not value:
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


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def camera_distance(first, second, radius):
    require(np.isfinite(radius) and radius > 0, 'Positive manifest scene radius')
    a, b = (np.asarray(v['w2c_original'], np.float64) for v in (first, second))
    require(a.shape == b.shape == (4, 4) and np.isfinite(a).all() and np.isfinite(b).all(), 'Finite poses')
    ca, cb = -a[:3, :3].T@a[:3, 3], -b[:3, :3].T@b[:3, 3]
    angle = np.arccos(np.clip((np.trace(a[:3, :3]@b[:3, :3].T)-1)/2, -1, 1))
    return float(np.sum((ca-cb)**2)/radius**2 + (angle/np.pi)**2)


def select_sources(train, target_names, radius, count=4):
    require(len({v['name'] for v in train}) == len(train), 'Duplicate camera name')
    require(all(v['split'] == 'train' for v in train), 'TRAIN only')
    lookup = {v['name']: v for v in train}
    require(len(set(target_names)) == len(target_names) and set(target_names) <= lookup.keys(), 'Fixed distinct targets')
    bank = [v for v in train if v['name'] not in target_names]
    require(len(bank) >= count+1, 'Insufficient source cameras; no fallback')
    result = []
    for i, name in enumerate(target_names):
        ranked = sorted((camera_distance(lookup[name], v, radius), v['name']) for v in bank)
        result.append({'name': name, 'fold': 'A' if i % 2 == 0 else 'B',
                       'excluded_nearest': {'name': ranked[0][1], 'distance': ranked[0][0]},
                       'sources': [{'name': n, 'distance': d} for d, n in ranked[1:count+1]]})
    return result


def actual_camera(view):
    """The same FP32 values go to gsplat and saved geometry, never source doubles."""
    K, w2c = (np.asarray(view[key], dtype=np.float32) for key in ('K', 'w2c_original'))
    require(K.shape == (3, 3) and w2c.shape == (4, 4)
            and np.isfinite(K).all() and np.isfinite(w2c).all(), 'Finite actual camera')
    return K, w2c


def validate_pixel_access(path, color, target, plan, records):
    views = {v['name']: v for v in plan['views']}
    if color:
        names = NAMES if target else plan['source_names']
        require(path in {views[name]['image_path'] for name in names}, 'Wrong RGB role/source')
        if target:
            prediction_barrier(records, NAMES)
    else:
        require(not target and path in {v['valid_path'] for v in views.values()}, 'Only bound valid masks')
    require(path in plan['pixel_input_hashes'], 'Unbound pixels')


def cross_fit(rows, fit_function):
    require([row['name'] for row in rows] == NAMES
            and [row['fold'] for row in rows] == ['A' if i % 2 == 0 else 'B' for i in range(16)],
            'Fixed 16 names and alternating folds required')
    return {arm: {fold: fit_function([r['statistics'][arm] for r in rows if r['fold'] == fold])
                  for fold in ('A', 'B')} for arm in ('true', 'wrong')}


def heldout_coefficient(row, arm, fits):
    require(arm in ('zero', 'true', 'wrong') and row['fold'] in ('A', 'B'), 'Fixed arm/fold')
    other = 'B' if row['fold'] == 'A' else 'A'
    return (0., None) if arm == 'zero' else (fits[arm][other]['coefficient'], other)


def prediction_barrier(records, names):
    require([r['name'] for r in records] == list(names) and len(set(names)) == 16,
            'All fixed 16 target transports must finish before target RGB')
    require(all(r.get('sha256') and r.get('path') for r in records), 'Missing saved transport')


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk directory')
    parent = read(PARENT/'plan.json'); audit = read(PARENT/'cpu_endpoint_audit.json')
    launch = read(PARENT/'launch_receipt.json')
    require(sha(PARENT/'plan.json') == PARENT_SHA and audit['status'] == 'passed'
            and audit['plan_sha256'] == PARENT_SHA and audit['checkpoint_sha256'] == BASE_SHA
            and launch['status'] == 'completed' and launch['exit_code'] == 0
            and launch['plan_sha256'] == PARENT_SHA, 'Completed fixed 1M lineage')
    original = Path(parent['source_snapshot'])
    require(tree(original) == parent['source_hashes'], 'Original 1M package changed')
    manifest_path = str((ROOT/parent['config']['manifest']).resolve())
    require(sha(manifest_path) == parent['input_hashes'][manifest_path], 'Manifest changed')
    manifest = read(manifest_path)
    require(manifest['pixel_protocol']['id'] == SPEC['pixel_protocol'], 'Native corner profile required')
    train = sorted((v for v in manifest['views'] if v['split'] == 'train'), key=lambda v: v['name'])
    labeled = [v for v in train if v.get('mask_path')]
    require(len(train) == 350 and len(labeled) == 259
            and [labeled[i*258//15]['name'] for i in range(16)] == NAMES, 'Fixed target sampling')
    selection = select_sources(train, NAMES, manifest['scene_radius'])
    source_names = sorted({s['name'] for row in selection for s in row['sources']})
    used_names = sorted(set(NAMES)|set(source_names))
    views = [{k: v[k] for k in ('name', 'split', 'width', 'height', 'K', 'w2c_original', 'image_path', 'valid_path')}
             for v in train if v['name'] in used_names]
    pixel_inputs = {v[k]: parent['input_hashes'][v[k]] for v in views for k in ('image_path', 'valid_path')}
    checkpoint = str(PARENT/'training/last.pt')
    inputs = {str(p): sha(p) for p in (PARENT/'plan.json', PARENT/'launch_receipt.json',
              PARENT/'cpu_endpoint_audit.json', Path(manifest_path), ROOT/'uv.lock', RUNTIME/'plan.json', GUARD_PARENT/'plan.json')}
    require(sha(checkpoint) == BASE_SHA, 'Checkpoint bytes changed')
    inputs[checkpoint] = BASE_SHA
    runtime = read(RUNTIME/'plan.json')
    guard_parent = read(GUARD_PARENT/'plan.json')
    guard = Path(guard_parent['source_snapshot'])/'train_direct_q_head_matched.py'
    require(sha(guard) == guard_parent['source_hashes'][guard.name], 'Frozen resource helper changed')
    snapshot = output/'source_snapshot'
    shutil.copytree(original/'bridge_rgs', snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for source, destination in ((ROOT/'src/bridge_rgs/residual_transport.py', snapshot/'bridge_rgs/residual_transport.py'),
              (Path(__file__), snapshot/Path(__file__).name), (guard, snapshot/guard.name),
              (ROOT/'tests/test_residual_transport_probe.py', snapshot/'test_residual_transport_probe.py'),
              (ROOT/'tests/test_residual_transport.py', snapshot/'test_residual_transport.py'),
              (ROOT/'docs/residual_transport_probe_protocol.md', snapshot/'residual_transport_probe_protocol.md'),
              (ROOT/'docs/residual_transport_coordinate_contract.md', snapshot/'residual_transport_coordinate_contract.md')):
        shutil.copy2(source, destination)
    plan = {k: runtime[k] for k in ('expected_gsplat_binary', 'installed_sources', 'runtime_versions', 'environment')}
    plan.update(status='prepared', specification=SPEC, output=str(output), source_snapshot=str(snapshot),
                source_hashes=tree(snapshot), inherited_package_hashes={k: h for k, h in parent['source_hashes'].items() if k.startswith('bridge_rgs/')},
                input_hashes=inputs, pixel_input_hashes=pixel_inputs, base_checkpoint=checkpoint,
                manifest=manifest_path, camera_selection=selection, views=views, source_names=source_names,
                expected_scene_renders=len(used_names), train_names=[v['name'] for v in train],
                prepare_pixel_reads=0, prepare_checkpoint_loads=0)
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json'),
                      'targets': 16, 'unique_sources': len(source_names)}))


def verify(plan, *, pixels=False):
    require(plan['specification'] == SPEC and tree(Path(plan['source_snapshot'])) == plan['source_hashes'], 'Frozen source/spec changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Frozen entry only')
    bindings = {**plan['input_hashes'], **plan['installed_sources'],
                plan['expected_gsplat_binary']['path']: plan['expected_gsplat_binary']['sha256']}
    if pixels:
        bindings.update(plan['pixel_input_hashes'])
    require(all(sha(p) == h for p, h in bindings.items()), 'Bound input changed')
    for key, value in plan['environment'].items():
        require(os.environ.get(key) == value, 'Environment changed: '+key)
    for key, value in plan['runtime_versions'].items():
        require(importlib.metadata.version(key) == value, 'Runtime changed: '+key)


def save_arrays(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez(stream, **arrays)
    return {'path': str(path), 'sha256': sha(path)}


def run(plan, report):
    import cv2
    import gsplat
    import torch
    from gsplat import rendering

    from bridge_rgs import residual_transport as helper
    from bridge_rgs.train import load_scene
    guard = importlib.import_module('train_direct_q_head_matched')
    counts = report['counts']; output = Path(plan['output'])
    arrays_dir = output/'arrays'; arrays_dir.mkdir()
    views = {v['name']: v for v in plan['views']}
    decoded = {}; rendered = {}; scene = state = None
    modes = flags = None
    old_threads = torch.get_num_threads()
    previous_flags = guard.numerical_flags(torch, {'cudnn_allow_tf32': True,
        'matmul_allow_tf32': False, 'matmul_precision': 'highest', 'cudnn_benchmark': False})
    report['numerics_actual'] = guard.read_numerical_flags(torch)
    report['numerics_before'] = previous_flags
    old_raster, old_low = gsplat.rasterization, rendering.rasterize_to_pixels
    def raster(*args, **kwargs):
        counts['high_raster'] += 1; return old_raster(*args, **kwargs)
    def low(*args, **kwargs):
        counts['low_raster'] += 1; return old_low(*args, **kwargs)
    gsplat.rasterization, rendering.rasterize_to_pixels = raster, low
    def decode(path, color, *, target=False):
        validate_pixel_access(path, color, target, plan, report['transports'])
        require(path in plan['pixel_input_hashes'] and sha(path) == plan['pixel_input_hashes'][path], 'Wrong/changed pixel source')
        if path not in decoded:
            value = cv2.imread(path, cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED)
            require(value is not None, 'Image decode failed')
            decoded[path] = cv2.cvtColor(value, cv2.COLOR_BGR2RGB).astype(np.float32)/255 if color else value > 0
            counts['target_RGB_decodes' if target else 'source_RGB_decodes' if color else 'valid_decodes'] += 1
        return decoded[path]
    try:
        torch.set_num_threads(4); torch.cuda.reset_peak_memory_stats()
        scene, state = load_scene(plan['base_checkpoint'])
        modes = {n: m.training for n, m in scene.named_modules()}
        flags = {n: p.requires_grad for n, p in scene.named_parameters()}
        scene.eval().requires_grad_(False)
        require(scene.pixel_protocol == SPEC['pixel_protocol'] and scene.sh_degree == 3, 'Wrong field/profile')
        all_train = [v for v in read(plan['manifest'])['views'] if v['split'] == 'train']
        expected_cameras = torch.tensor([v['w2c_original'] for v in all_train], dtype=torch.float32)
        require(torch.equal(state['training_cameras'].cpu(), expected_cameras), 'Original training camera order differs')
        with torch.no_grad():
            for name in sorted(views):
                v = views[name]
                K32, w2c32 = actual_camera(v)
                result = scene.render(torch.tensor(K32, device='cuda'),
                    torch.tensor(w2c32, device='cuda'), v['width'], v['height'],
                    semantics=False, refine=False, absgrad=False)
                counts['scene'] += 1
                data = {k: result[k].detach().cpu().numpy() for k in ('rgb', 'depth', 'alpha')}
                data['rgb'] = np.clip(data['rgb'], 0, 1)
                data['depth'], data['alpha'] = data['depth'][..., 0], data['alpha'][..., 0]
                require(all(np.isfinite(x).all() for x in data.values()), 'Nonfinite scene output')
                data['K'], data['w2c'] = K32, w2c32
                data['valid'] = decode(v['valid_path'], False)
                require(data['rgb'].shape == (v['height'], v['width'], 3) and data['valid'].shape == data['depth'].shape, 'Native grids differ')
                if name in plan['source_names']:
                    data['residual'] = decode(v['image_path'], True)-data['rgb']
                record = save_arrays(arrays_dir/('render_'+name+'.npz'), **data)
                rendered[name] = data; report['renders'].append(dict(name=name, **record))
        for selection in plan['camera_selection']:
            name = selection['name']; target = rendered[name]
            sources = [dict(rendered[s['name']], K=rendered[s['name']]['K'].astype(np.float64),
                        w2c=rendered[s['name']]['w2c'].astype(np.float64)) for s in selection['sources']]
            transported = helper.transport_residual(target['depth'], target['alpha'], target['valid'],
                target['K'].astype(np.float64), target['w2c'].astype(np.float64), sources)
            record = save_arrays(arrays_dir/('transport_'+name+'.npz'),
                **{k: transported[k] for k in ('true_residual', 'wrong_residual', 'valid', 'source_count')})
            report['transports'].append(dict(name=name, source_summaries=transported['source_summaries'], **record))
        prediction_barrier(report['transports'], NAMES)
        report['prediction_barrier_complete'] = True
        scoring = {}; rows = []
        for selection, record in zip(plan['camera_selection'], report['transports'], strict=True):
            name = selection['name']; base = rendered[name]['rgb']; valid = rendered[name]['valid']
            target = decode(views[name]['image_path'], True, target=True)
            truth = save_arrays(arrays_dir/('target_'+name+'.npz'), rgb=target, valid=valid)
            with np.load(record['path'], allow_pickle=False) as cache:
                transported = {k: cache[k].copy() for k in cache.files}
            scoring[name] = (base, target, valid, transported)
            stats = {arm: helper.shrink_statistics(base, target, transported[arm+'_residual'], valid)
                     for arm in ('true', 'wrong')}
            rows.append({'name': name, 'fold': selection['fold'], 'target': truth, 'statistics': stats})
        fits = cross_fit(rows, helper.fit_shrinkage)
        for row in rows:
            base, target, valid, transport = scoring[row['name']]
            row['metrics'] = {}
            for arm in ('zero', 'true', 'wrong'):
                coefficient, other = heldout_coefficient(row, arm, fits)
                residual = np.zeros_like(base) if arm == 'zero' else transport[arm+'_residual']
                predicted = base.astype(np.float64)+coefficient*residual.astype(np.float64)
                error = predicted-target.astype(np.float64); clipped = np.clip(predicted, 0, 1)-target
                row['metrics'][arm] = {'coefficient': coefficient, 'fit_fold': None if arm == 'zero' else other,
                    'unclipped_MSE': float(np.mean(error[valid]**2)), 'clipped_MSE': float(np.mean(clipped[valid]**2)),
                    'support_clipped_MSE': float(np.mean(clipped[transport['valid']]**2)) if np.any(transport['valid']) else None}
            row['transport_pixels'] = int(transport['valid'].sum()); row['valid_pixels'] = int(valid.sum())
        summary = {arm: {metric: float(np.mean([r['metrics'][arm][metric] for r in rows]))
                        for metric in ('unclipped_MSE', 'clipped_MSE')} for arm in ('zero', 'true', 'wrong')}
        analysis = {'scope': SPEC['scope'], 'fits': fits, 'views': rows, 'equal_view_mean': summary,
                    'performance_adoption_gate': None, 'semantic_metrics': None}
        write(output/'analysis.json', analysis); report['analysis_sha256'] = sha(output/'analysis.json')
        require(counts['scene'] == counts['high_raster'] == plan['expected_scene_renders']
                and counts['target_RGB_decodes'] == 16 and counts['source_RGB_decodes'] == len(plan['source_names']), 'Fixed counts differ')
        report['actual_imports'] = {}
        for name, module in list(sys.modules.items()):
            if name.startswith('bridge_rgs') and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve(); relative = str(path.relative_to(plan['source_snapshot']))
                require(sha(path) == plan['source_hashes'][relative], 'Unbound package import')
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}
        report['actual_gsplat_binary'] = guard.loaded_gsplat_binaries()
        require(plan['expected_gsplat_binary'] in report['actual_gsplat_binary'].values(), 'Wrong binary')
    finally:
        gsplat.rasterization, rendering.rasterize_to_pixels = old_raster, old_low
        if scene is not None:
            report['state_tensor_exact'] = all(torch.equal(v.detach().cpu(), state['model'][k].detach().cpu()) for k, v in scene.state_dict().items())
            for name, module in scene.named_modules():
                module.training = modes[name]
            for name, parameter in scene.named_parameters():
                parameter.requires_grad_(flags[name])
            report['scene_flags_restored'] = all(m.training == modes[n] for n, m in scene.named_modules()) and all(p.requires_grad == flags[n] for n, p in scene.named_parameters())
        guard.numerical_flags(torch, previous_flags)
        report['numerics_restored'] = guard.read_numerical_flags(torch) == previous_flags
        report['peak_cuda_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        torch.set_num_threads(old_threads)
        require(scene is None or report['state_tensor_exact'], 'Model changed')


def execute(path, expected):
    require(sha(path) == expected, 'Plan identity changed')
    plan = read(path); output = Path(plan['output'])
    require(not any((output/n).exists() for n in ('execution_started.json', 'execution_receipt.json', 'analysis.json')), 'One attempt only')
    write(output/'execution_started.json', {'plan_sha256': expected})
    started = time.monotonic()
    report = {'status': 'running', 'plan_sha256': expected, 'counts': {k: 0 for k in
              ('scene', 'high_raster', 'low_raster', 'source_RGB_decodes', 'target_RGB_decodes', 'valid_decodes')},
              'renders': [], 'transports': [], 'semantic_GT_decodes': 0, 'VAL_decodes': 0, 'model_updates': 0}
    def expired(*_):
        raise TimeoutError('Fixed residual transport deadline exhausted')
    old_handler = signal.signal(signal.SIGALRM, expired); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan); sys.path.insert(0, plan['source_snapshot'])
        require(not any(k.startswith('bridge_rgs') for k in sys.modules), 'No live bridge package')
        report['gpu_before'] = importlib.import_module('train_direct_q_head_matched').gpu_inventory()
        run(plan, report); verify(plan, pixels=True)
        report.update(status='completed', source_inputs_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
        report['elapsed_seconds'] = time.monotonic()-started
        write(output/'execution_receipt.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path); action.add_argument('--execute', type=Path)
    parser.add_argument('--expected-plan-sha256'); args = parser.parse_args()
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode')
    prepare(args.prepare) if args.prepare else execute(args.execute, args.expected_plan_sha256)
