"""Fixed 48-forward TRAIN renderer decomposition; preparation is CPU only."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import shutil
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
PARENT = Path('/mnt/data/SHM2026/runs/residual_transport_probe_v1')
START = Path('/mnt/data/SHM2026/runs/ibgs_start_transfer_v1')
BASE_SHA = '35b489fe45ad34ad279018609b6d5eaf6493e4c0d82f9c86ec16925268078092'
LINEAGE = {
    str(PARENT/'plan.json'): '061586d6309693a41c66b720159076e9bb52d6c1d7d605ec5d939a51402b297b',
    str(PARENT/'execution_receipt.json'): '2a9f1eda9baddb4ce79a6880bf489a577d83b586e83c82fadd3450705130e5a5',
    str(PARENT/'launch_receipt.json'): '4f7d7572be40dd8d84d501daa925351939974ff758930dc1e5dec9215dcdcda0',
    str(PARENT/'independent_cpu_review.json'): 'c2476de5dac9951b3f4318c1f55711f903fbf9dde5eea1c866d7cec877ce2b33',
    str(START/'plan.json'): '858b5601187e9d4d0841983614de46f816259d26bddc3e9b70cbb6f2da0ea42d',
    str(START/'execution_receipt.json'): 'bfd015f7c642a39e9284031de72c6b5756ffc4fa85f05ebe4800f142dfb24b61',
    str(START/'launch_receipt.json'): '50222d3206b4bfd3ea3b3dba9c29c441b321714d827e995cc05918d3852c89c2',
}
NAMES = ['002.png', '021.png', '041.png', '059.png', '079.png', '100.png',
         '118.png', '137.png', '156.png', '176.png', '200.png', '220.png',
         '241.png', '259.png', '278.png', '300.png']
CONDITIONS = {'A': {'rasterize_mode': 'antialiased', 'near_plane': .01, 'eps2d': .3},
              'C': {'rasterize_mode': 'classic', 'near_plane': .01, 'eps2d': .3},
              'N': {'rasterize_mode': 'classic', 'near_plane': .2, 'eps2d': .3}}
NUMERICS = {'cudnn_allow_tf32': True, 'matmul_allow_tf32': False,
            'matmul_precision': 'highest', 'cudnn_benchmark': False}
SPEC = {'protocol': 'ibgs_renderer_decomposition_v1', 'targets': NAMES,
        'conditions': CONDITIONS, 'condition_order': ['A', 'C', 'N'],
        'I': 'Reuse the original IBGS start-transfer raw FP32 predictions; no IBGS execution',
        'base_sha256': BASE_SHA, 'pixel_protocol': 'colmap_corner_v2', 'SH_degree': 3,
        'eps2d': .3, 'far_plane': 1e6, 'numerics': NUMERICS,
        'scene_calls': 48, 'high_raster_calls': 48, 'low_raster_calls': 48,
        'cached_target_loads': 16, 'backward': 0, 'optimizer_steps': 0,
        'original_image_decodes': 0, 'semantic_label_decodes': 0, 'VAL_decodes': 0,
        'internal_seconds': 90, 'external_seconds': 120,
        'AA_identity': 'clipped A must be array_equal to the original 16 AA caches; no tolerance rescue',
        'target_barrier': 'All 48 raw prediction files saved and hashed before cached target payloads are loaded',
        'score': 'FP64 raw/clipped RGB MSE on fixed cached valid pixels, equal view means and mean per-view PSNR',
        'interpretation': 'Fixed-order MSE telescoping; mode includes support changes; not independent physical attribution',
        'scope': 'TRAIN starting-point renderer diagnostic, no fitting/adoption/novelty claim'}


def require(condition, message):
    if not condition:
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


def actual_camera(camera):
    require(camera['split'] == 'train' and camera['name'] in NAMES, 'Fixed TRAIN camera required')
    K, w2c = (np.asarray(camera[k], np.float32) for k in ('K', 'w2c_original'))
    require(K.shape == (3, 3) and w2c.shape == (4, 4) and np.isfinite(K).all()
            and np.isfinite(w2c).all(), 'Finite actual FP32 camera required')
    return K, w2c


def condition_kwargs(kwargs, condition):
    require(condition in CONDITIONS, 'Unknown renderer condition')
    required = {'rasterize_mode': 'antialiased', 'near_plane': .01,
                'far_plane': 1e6, 'packed': False, 'absgrad': False,
                'render_mode': 'RGB+ED', 'sh_degree': 3}
    require(all(kwargs.get(k) == v for k, v in required.items())
            and kwargs.get('eps2d', .3) == .3, 'Original frozen renderer arguments changed')
    return dict(kwargs, **CONDITIONS[condition])


@contextmanager
def raster_hooks(gsplat, rendering, condition, counts):
    original, low = gsplat.rasterization, rendering.rasterize_to_pixels
    def raster(*args, **kwargs):
        changed = condition_kwargs(kwargs, condition)
        counts['high_raster'] += 1
        return original(*args, **changed)
    def pixels(*args, **kwargs):
        counts['low_raster'] += 1
        return low(*args, **kwargs)
    gsplat.rasterization, rendering.rasterize_to_pixels = raster, pixels
    try:
        yield
    finally:
        gsplat.rasterization, rendering.rasterize_to_pixels = original, low


def prediction_barrier(records):
    require([(r['condition'], r['name']) for r in records] ==
            [(c, n) for c in CONDITIONS for n in NAMES]
            and all(r.get('path') and r.get('sha256') for r in records),
            'All fixed 48 saved predictions must precede cached target loading')


def identity(prediction, previous):
    require(prediction.dtype == previous.dtype == np.float32
            and prediction.shape == previous.shape and np.isfinite(prediction).all()
            and np.isfinite(previous).all(), 'Finite same-shape FP32 AA arrays required')
    clipped = np.clip(prediction, 0, 1)
    return {'clipped_array_exact': bool(np.array_equal(clipped, previous)),
            'clipped_max_abs_difference': float(np.max(np.abs(clipped.astype(np.float64)-previous)))}


def metrics(prediction, target, valid):
    require(prediction.shape == target.shape == (*valid.shape, 3) and valid.dtype == bool
            and valid.any() and np.isfinite(prediction).all() and np.isfinite(target).all(),
            'Finite RGB and nonempty fixed support required')
    p, y = prediction.astype(np.float64)[valid], target.astype(np.float64)[valid]
    result = {}
    for mode, value in (('raw', p), ('clipped', np.clip(p, 0, 1))):
        mse = float(np.mean((value-y)**2))
        result[mode] = {'mse': mse, 'psnr': -10*float(np.log10(mse)) if mse > 0 else None,
                        'perfect_match': mse == 0}
    result['valid_pixels'] = int(valid.sum())
    return result


def telescoping(values):
    require(set(values) == {'A', 'C', 'N', 'I'} and all(np.isfinite(v) for v in values.values()),
            'Four finite MSE values required')
    edges = {f'{b}_minus_{a}': float(values[b]-values[a]) for a, b in [('A', 'C'), ('C', 'N'), ('N', 'I')]}
    total = float(values['I']-values['A'])
    return {**edges, 'I_minus_A': total, 'closure_residual': float(sum(edges.values())-total)}


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk directory required')
    require(all(sha(p) == h for p, h in LINEAGE.items()), 'Fixed predecessor chain changed')
    parent, start = read(PARENT/'plan.json'), read(START/'plan.json')
    for directory in (PARENT, START):
        execution, launch = read(directory/'execution_receipt.json'), read(directory/'launch_receipt.json')
        require(execution['status'] == launch['status'] == 'completed' and launch['natural_completion']
                and launch['exit_code'] == 0 and launch['plan_sha256'] == execution['plan_sha256'] == sha(directory/'plan.json')
                and launch['execution_receipt_sha256'] == sha(directory/'execution_receipt.json')
                and execution['analysis_sha256'] == sha(directory/'analysis.json'), 'Natural completed predecessor required')
    require(read(PARENT/'independent_cpu_review.json')['status'] == 'passed'
            and tree(parent['source_snapshot']) == parent['source_hashes']
            and tree(start['source_snapshot']) == start['source_hashes'], 'Predecessor audit/source mismatch')
    start_execution = read(START/'execution_receipt.json')
    require(start_execution['field_unchanged'] and start_execution['render_calls'] == 16
            and parent['base_checkpoint'] == start['checkpoint']
            and sha(parent['base_checkpoint']) == BASE_SHA, 'Original unchanged 1M field required')
    inputs = dict(LINEAGE)
    for p in (PARENT/'analysis.json', START/'analysis.json', ROOT/'uv.lock', Path(parent['base_checkpoint'])):
        inputs[str(p)] = sha(p)
    predictions = {r['name']: r for r in start_execution['predictions']}
    require([v['name'] for v in start['views']] == NAMES and list(predictions) == NAMES, 'Fixed 16 names changed')
    views = []
    for v in start['views']:
        camera = v['camera']; actual_camera(camera)
        require(set(camera) == {'name', 'split', 'image_id', 'width', 'height', 'K', 'w2c_original'},
                'Camera metadata must not contain image/label paths')
        records = {'old_AA': v['old_render'], 'target': v['target'], 'I': predictions[v['name']]}
        for r in records.values():
            require(sha(r['path']) == r['sha256'], 'Cached predecessor output changed')
            inputs[r['path']] = r['sha256']
        views.append({'name': v['name'], 'camera': camera, **records})
    installed = dict(parent['installed_sources'])
    package = ROOT/'.venv/lib/python3.11/site-packages/gsplat'
    for rel in ('__init__.py', 'cuda/_backend.py', 'cuda/csrc/ProjectionEWA3DGSFused.cu',
                'cuda/csrc/SphericalHarmonicsCUDA.cu', 'cuda/include/Common.h'):
        installed[str(package/rel)] = sha(package/rel)
    binary = parent['expected_gsplat_binary']; installed[binary['path']] = binary['sha256']
    require(all(sha(p) == h for p, h in installed.items()), 'Original gsplat source/binary changed')
    snapshot = output/'source_snapshot'; snapshot.mkdir(parents=True)
    shutil.copytree(Path(parent['source_snapshot'])/'bridge_rgs', snapshot/'bridge_rgs',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(Path(parent['source_snapshot'])/'train_direct_q_head_matched.py', snapshot/'train_direct_q_head_matched.py')
    for p in (Path(__file__), ROOT/'tests/test_ibgs_renderer_decomposition.py',
              ROOT/'docs/ibgs_renderer_decomposition_protocol.md'):
        shutil.copy2(p, snapshot/p.name)
    inherited = {k: v for k, v in parent['source_hashes'].items()
                 if k.startswith('bridge_rgs/') or k == 'train_direct_q_head_matched.py'}
    require(all(tree(snapshot).get(k) == v for k, v in inherited.items()), 'Frozen package differs')
    plan = {'specification': SPEC, 'output': str(output), 'source_snapshot': str(snapshot),
            'source_hashes': tree(snapshot), 'inherited_source_hashes': inherited,
            'input_hashes': inputs, 'installed_sources': installed, 'views': views,
            'base_checkpoint': parent['base_checkpoint'], 'expected_gsplat_binary': binary,
            'runtime_versions': parent['runtime_versions'], 'environment': parent['environment'],
            'prepare_array_payload_loads': 0, 'prepare_checkpoint_loads': 0,
            'I_provenance': {'plan': str(START/'plan.json'), 'execution_receipt': str(START/'execution_receipt.json'),
                'scope': 'Saved original warm-start raw RGB, not either trained 6000-step endpoint'}}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json'), 'sources': len(plan['source_hashes'])}))


def verify(plan):
    require(plan['specification'] == SPEC and Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name,
            'Frozen entry/specification required')
    require(tree(plan['source_snapshot']) == plan['source_hashes'], 'Frozen sources changed')
    require(all(sha(p) == h for p, h in {**plan['input_hashes'], **plan['installed_sources']}.items()), 'Bound source/input changed')
    require({k: importlib.metadata.version(k) for k in plan['runtime_versions']} == plan['runtime_versions'], 'Runtime versions changed')
    require(all(os.environ.get(k) == v for k, v in plan['environment'].items()), 'Fixed runtime environment required')


def run(plan, report):
    import gsplat
    import torch
    from gsplat import rendering

    from bridge_rgs.train import load_scene
    guard = importlib.import_module('train_direct_q_head_matched')
    output = Path(plan['output']); arrays = output/'arrays'; arrays.mkdir()
    scene = state = None
    old_threads = torch.get_num_threads()
    old_flags = guard.numerical_flags(torch, NUMERICS)
    report.update(numerics_before=old_flags, numerics_actual=guard.read_numerical_flags(torch))
    original_raster, original_low = gsplat.rasterization, rendering.rasterize_to_pixels
    report['actual_entry'] = {key: {'path': str(Path(inspect.getfile(fn)).resolve()), 'function': fn.__name__}
                              for key, fn in [('rasterization', original_raster), ('rasterize_to_pixels', original_low)]}
    for r in report['actual_entry'].values():
        r['sha256'] = sha(r['path'])
        require(plan['installed_sources'].get(r['path']) == r['sha256'], 'Unbound actual gsplat entry')
    try:
        torch.set_num_threads(4); torch.cuda.reset_peak_memory_stats()
        scene, state = load_scene(plan['base_checkpoint'])
        modes = {n: m.training for n, m in scene.named_modules()}
        flags = {n: p.requires_grad for n, p in scene.named_parameters()}
        scene.eval().requires_grad_(False)
        require(scene.pixel_protocol == SPEC['pixel_protocol'] and scene.sh_degree == 3
                and len(scene.splats['means']) == 996009, 'Wrong starting field')
        report['field_count'] = len(scene.splats['means'])
        report['camera_depth_counts'] = []
        means = state['model']['splats.means'].numpy().astype(np.float64)
        with torch.no_grad():
            for condition in CONDITIONS:
                counts = report['condition_counts'][condition]
                with raster_hooks(gsplat, rendering, condition, counts):
                    for view in plan['views']:
                        camera = view['camera']; K, w2c = actual_camera(camera)
                        if condition == 'A':
                            z = means@w2c[2, :3].astype(np.float64)+float(w2c[2, 3])
                            report['camera_depth_counts'].append({'name': view['name'],
                                'CPU_FP64_from_stored_FP32': True, 'z_lt_001': int(np.sum(z < .01)),
                                'z_ge_001_lt_02': int(np.sum((z >= .01) & (z < .2))),
                                'z_eq_02': int(np.sum(z == .2)), 'z_gt_far': int(np.sum(z > 1e6))})
                        result = scene.render(torch.tensor(K, device='cuda'), torch.tensor(w2c, device='cuda'),
                            camera['width'], camera['height'], semantics=False, refine=False, absgrad=False)
                        counts['scene'] += 1
                        rgb = result['rgb'].detach().cpu().numpy()
                        require(rgb.dtype == np.float32 and rgb.shape == (camera['height'], camera['width'], 3)
                                and np.isfinite(rgb).all(), 'Finite native FP32 RGB required')
                        radii = result['info']['radii']
                        require(radii.shape == (1, len(means), 2), 'Unexpected gsplat radii schema')
                        path = arrays/f'{condition}_{view["name"]}.raw.npy'
                        with path.open('xb') as stream:
                            np.save(stream, rgb, allow_pickle=False)
                        report['predictions'].append({'condition': condition, 'name': view['name'],
                            'path': str(path), 'sha256': sha(path), 'shape': list(rgb.shape), 'dtype': str(rgb.dtype),
                            'projected_valid_gaussians': int((radii > 0).all(-1).sum().item()),
                            'actual_K_fp32_sha256': hashlib.sha256(K.tobytes()).hexdigest(),
                            'actual_w2c_fp32_sha256': hashlib.sha256(w2c.tobytes()).hexdigest()})
        prediction_barrier(report['predictions']); report['prediction_barrier_complete'] = True
        records = {(r['condition'], r['name']): r for r in report['predictions']}
        report['AA_identity'] = []
        # Fail closed on the complete AA identity check before opening target payloads.
        for view in plan['views']:
            value = np.load(records['A', view['name']]['path'], allow_pickle=False)
            with np.load(view['old_AA']['path'], allow_pickle=False) as old:
                check = identity(value, old['rgb']); K, w2c = actual_camera(view['camera'])
                require(np.array_equal(K, old['K']) and np.array_equal(w2c, old['w2c']), 'Cached AA camera changed')
            report['AA_identity'].append({'name': view['name'], **check})
        write(output/'AA_identity.json', report['AA_identity'])
        report['AA_identity_sha256'] = sha(output/'AA_identity.json')
        require(all(r['clipped_array_exact'] for r in report['AA_identity']), 'AA cache identity failed; no tolerance rescue')
        rows = []
        for view in plan['views']:
            with np.load(view['target']['path'], allow_pickle=False) as target:
                y, valid = target['rgb'], target['valid']
            report['cached_target_loads'] += 1
            with np.load(view['old_AA']['path'], allow_pickle=False) as old:
                require(np.array_equal(valid, old['valid']), 'Fixed valid support changed')
            row = {'name': view['name'], 'metrics': {}}
            for condition in ('A', 'C', 'N', 'I'):
                p = view['I']['path'] if condition == 'I' else records[condition, view['name']]['path']
                value = np.load(p, allow_pickle=False)
                require(value.dtype == np.float32, 'Saved raw RGB must be FP32')
                row['metrics'][condition] = metrics(value, y, valid)
            row['MSE_path'] = {mode: telescoping({c: row['metrics'][c][mode]['mse'] for c in ('A', 'C', 'N', 'I')})
                               for mode in ('raw', 'clipped')}
            rows.append(row)
        summary = {c: {mode: {'mean_view_mse': float(np.mean([r['metrics'][c][mode]['mse'] for r in rows])),
                              'mean_view_psnr': (float(np.mean([r['metrics'][c][mode]['psnr'] for r in rows]))
                                                 if all(r['metrics'][c][mode]['psnr'] is not None for r in rows) else None)}
                       for mode in ('raw', 'clipped')} for c in ('A', 'C', 'N', 'I')}
        analysis = {'scope': SPEC['scope'], 'views': rows, 'equal_view_mean': summary,
                    'MSE_path': {mode: telescoping({c: summary[c][mode]['mean_view_mse'] for c in summary})
                                 for mode in ('raw', 'clipped')},
                    'interpretation': SPEC['interpretation'], 'performance_adoption_gate': None}
        write(output/'analysis.json', analysis); report['analysis_sha256'] = sha(output/'analysis.json')
        require(all(c == {'scene': 16, 'high_raster': 16, 'low_raster': 16}
                    for c in report['condition_counts'].values()) and report['cached_target_loads'] == 16, 'Counts differ')
        report['actual_imports'] = {}
        for name, module in list(sys.modules.items()):
            if name.startswith('bridge_rgs') and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve(); relative = str(path.relative_to(plan['source_snapshot']))
                require(sha(path) == plan['source_hashes'][relative], 'Unbound scene import')
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}
        report['actual_gsplat_binary'] = guard.loaded_gsplat_binaries()
        require(plan['expected_gsplat_binary'] in report['actual_gsplat_binary'].values(), 'Wrong loaded gsplat binary')
    finally:
        report['hooks_restored'] = gsplat.rasterization is original_raster and rendering.rasterize_to_pixels is original_low
        if scene is not None:
            report['state_tensor_exact'] = all(torch.equal(v.detach().cpu(), state['model'][k].detach().cpu()) for k, v in scene.state_dict().items())
            for n, m in scene.named_modules():
                m.training = modes[n]
            for n, p in scene.named_parameters():
                p.requires_grad_(flags[n])
            report['scene_flags_restored'] = all(m.training == modes[n] for n, m in scene.named_modules()) and all(p.requires_grad == flags[n] for n, p in scene.named_parameters())
        guard.numerical_flags(torch, old_flags)
        report['numerics_restored'] = guard.read_numerical_flags(torch) == old_flags
        report['peak_cuda_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        report['peak_cuda_reserved_bytes'] = int(torch.cuda.max_memory_reserved())
        torch.set_num_threads(old_threads)
        require(report['hooks_restored'] and report['numerics_restored']
                and (scene is None or report['state_tensor_exact'] and report['scene_flags_restored']), 'Restoration failed')


def execute(path, expected):
    require(sha(path) == expected, 'Plan identity changed')
    plan = read(path); output = Path(plan['output'])
    require(not any((output/n).exists() for n in ('execution_started.json', 'execution_receipt.json', 'analysis.json')), 'One attempt only')
    write(output/'execution_started.json', {'plan_sha256': expected})
    started = time.monotonic()
    report = {'status': 'running', 'plan_sha256': expected, 'condition_counts':
              {c: {'scene': 0, 'high_raster': 0, 'low_raster': 0} for c in CONDITIONS},
              'predictions': [], 'cached_target_loads': 0, 'backward': 0, 'optimizer_steps': 0,
              'original_image_decodes': 0, 'semantic_label_decodes': 0, 'VAL_decodes': 0}
    def expired(*_):
        raise TimeoutError('Fixed 90-second internal deadline exhausted')
    old_handler = signal.signal(signal.SIGALRM, expired); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan); require(not any(k.startswith('bridge_rgs') for k in sys.modules), 'No live scene package')
        sys.path.insert(0, plan['source_snapshot'])
        report['gpu_before'] = importlib.import_module('train_direct_q_head_matched').gpu_inventory()
        run(plan, report); verify(plan)
        report.update(status='completed', sources_inputs_unchanged=True)
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
