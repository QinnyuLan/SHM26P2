"""Fixed 16-TRAIN forward check of differentiable AA and isolated near=.01 port."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
START = Path('/mnt/data/SHM2026/runs/ibgs_start_transfer_v1')
DECOMPOSITION = Path('/mnt/data/SHM2026/runs/ibgs_renderer_decomposition_v1')
PORT = Path('/mnt/data/SHM2026/third_party/ibgs_aa_port_v1')
SPEC = {
    'protocol': 'ibgs_aa_compatibility_v1', 'render_calls': 16,
    'backward': 0, 'optimizer_steps': 0, 'source_RGB_decodes': 0,
    'original_target_RGB_decodes': 0, 'semantic_label_decodes': 0, 'VAL_decodes': 0,
    'near': .01, 'eps2d': .3, 'internal_seconds': 100, 'external_seconds': 120,
    'scope': 'Original fixed 1M field; TRAIN-only forward compatibility, no fitting or adoption',
    'remaining_differences': 'IBGS alpha cap, support bounds and projection arithmetic retained',
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
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x') as stream:
        stream.write(payload)


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and p.suffix != '.pyc' and '__pycache__' not in p.parts}


def prepare(output):
    output = output.resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk directory required')
    parent = read(START/'plan.json')
    build = read(PORT/'build_receipt.json')
    port = read(PORT/'preparation.json')
    require(build['status'] == 'completed' and build['exit_code'] == 0 and build['natural_completion'], 'Completed isolated build required')
    require(tree(parent['source_snapshot']) == parent['source_hashes'], 'Starting source snapshot changed')
    require(all(sha(PORT/'source'/k) == h for k, h in port['source_hashes'].items()), 'Isolated near sources changed')
    require(sha(build['binary']['path']) == build['binary']['sha256'], 'Isolated binary changed')
    require(sha(port['parent_binary']['path']) == port['parent_binary']['sha256'], 'Original binary changed')
    for directory in (START, DECOMPOSITION):
        execution, launch = read(directory/'execution_receipt.json'), read(directory/'launch_receipt.json')
        require(execution['status'] == launch['status'] == 'completed' and launch['natural_completion']
                and launch['exit_code'] == 0 and launch['execution_receipt_sha256'] == sha(directory/'execution_receipt.json')
                and execution['analysis_sha256'] == sha(directory/'analysis.json'), 'Completed predecessor required')
    require(all(r['clipped_array_exact'] for r in read(DECOMPOSITION/'AA_identity.json')), 'Original AA exact replay required')
    snapshot = output/'source_snapshot'
    shutil.copytree(parent['source_snapshot'], snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT/'src/bridge_rgs/ibgs_antialias.py', snapshot/'bridge_rgs/ibgs_antialias.py')
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    shutil.copy2(ROOT/'docs/ibgs_aa_compatibility_protocol.md', snapshot/'ibgs_aa_compatibility_protocol.md')
    inputs = dict(parent['input_hashes'])
    for directory, names in ((START, ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'analysis.json')),
                             (DECOMPOSITION, ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'analysis.json', 'AA_identity.json')),
                             (PORT, ('preparation.json', 'build_receipt.json', 'near_plane.patch'))):
        for name in names:
            inputs[str(directory/name)] = sha(directory/name)
    backend = {str(PORT/'source'/k): h for k, h in port['source_hashes'].items()}
    backend[build['binary']['path']] = build['binary']['sha256']
    package = PORT/'python/diff_plane_rasterization/__init__.py'
    backend[str(package)] = sha(package)
    plan = {'specification': SPEC, 'output': str(output), 'source_snapshot': str(snapshot),
            'source_hashes': tree(snapshot), 'input_hashes': inputs, 'backend_hashes': backend,
            'binary': build['binary'], 'backend_python': str(PORT/'python'),
            'repository': str(snapshot/'ibgs'), 'checkpoint': parent['checkpoint'], 'views': parent['views'],
            'prepare_array_payloads_loaded': 0, 'prepare_checkpoint_loads': 0}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC and Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Frozen entry required')
    require(tree(plan['source_snapshot']) == plan['source_hashes'], 'Frozen source changed')
    require(all(sha(p) == h for p, h in {**plan['input_hashes'], **plan['backend_hashes']}.items()), 'Bound input/backend changed')


def run(plan, report):
    sys.path[:0] = [plan['source_snapshot'], plan['backend_python'], plan['repository']]
    import torch
    from check_ibgs_start_transfer import metrics, prediction_barrier
    from diff_plane_rasterization import _C, GaussianRasterizer
    from gaussian_renderer import render

    from bridge_rgs.ibgs_adapter import BridgeCamera
    from bridge_rgs.ibgs_antialias import aa_opacity_rasterizer
    from bridge_rgs.ibgs_warm_training import load_warm_field

    require(Path(_C.__file__).resolve() == Path(plan['binary']['path']).resolve()
            and sha(_C.__file__) == plan['binary']['sha256'], 'Wrong loaded near backend')
    torch.set_num_threads(4)
    old_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.benchmark = False
    field, background = load_warm_field(plan['checkpoint'])
    keys = ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest', '_normal', '_offset')
    state_hashes = lambda: {k: hashlib.sha256(getattr(field, k).detach().cpu().numpy().tobytes()).hexdigest() for k in keys}
    before = state_hashes()
    output = Path(plan['output']); (output/'arrays').mkdir()
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    options = SimpleNamespace(shuffle_source_frame=False, enable_exposure_correction=False)
    original_forward = GaussianRasterizer.forward
    try:
        with torch.no_grad(), aa_opacity_rasterizer(GaussianRasterizer) as aa_state:
            for index, view in enumerate(plan['views']):
                camera = BridgeCamera(view['camera'], index)
                require(np.array_equal(camera.world_view_transform.T.cpu().numpy(), np.asarray(view['camera']['w2c_original'], np.float32)), 'Actual camera changed')
                result = render(camera, field, None, pipe, options, background,
                    learnt_normal=True, nb_src_frames=1, buffer_length=4, depth_error_threshold=.01,
                    render_geo=False, return_depth_normal=False, do_find_closest_frame=False)
                report['render_calls'] += 1
                rgb = result['render'].permute(1, 2, 0).contiguous().cpu().numpy()
                require(rgb.dtype == np.float32 and np.isfinite(rgb).all(), 'Finite native RGB required')
                path = output/'arrays'/(view['name']+'.raw.npy')
                with path.open('xb') as stream:
                    np.save(stream, rgb, allow_pickle=False)
                report['predictions'].append({'name': view['name'], 'path': str(path), 'sha256': sha(path)})
                del result, camera, rgb
            report['antialias_calls'] = aa_state.calls
            require(aa_state.calls == 16, 'AA must apply exactly once per renderer call')
        prediction_barrier(report['predictions'])
        report['prediction_barrier_complete'] = True
        write(output/'prediction_receipt.json', {'predictions': report['predictions'], 'cached_target_loads': 0})
        rows = []
        for view, prediction in zip(plan['views'], report['predictions'], strict=True):
            with np.load(view['target']['path'], allow_pickle=False) as target, np.load(view['old_render']['path'], allow_pickle=False) as old:
                report['cached_target_loads'] += 1
                y, valid, aa = target['rgb'], target['valid'], old['rgb']
                require(np.array_equal(valid, old['valid']), 'Valid support changed')
                value = np.load(prediction['path'], allow_pickle=False)
                rows.append({'name': view['name'], 'AA': metrics(aa, y, valid),
                    'corrected_raw': metrics(value, y, valid),
                    'corrected_clipped': metrics(np.clip(value, 0, 1), y, valid),
                    'RGB_delta_to_AA': {'max_abs': float(np.max(np.abs(value.astype(np.float64)-aa))),
                                        'mean_abs_valid': float(np.abs(np.clip(value, 0, 1).astype(np.float64)-aa)[valid].mean())}})
        summary = {key: {'mean_view_mse': float(np.mean([r[key]['mse'] for r in rows])),
                         'mean_view_psnr': float(np.mean([r[key]['psnr'] for r in rows]))}
                   for key in ('AA', 'corrected_raw', 'corrected_clipped')}
        write(output/'analysis.json', {'scope': SPEC['scope'], 'views': rows, 'summary': summary})
        report['analysis_sha256'] = sha(output/'analysis.json')
        report['actual_binary'] = {'path': str(Path(_C.__file__).resolve()), 'sha256': sha(_C.__file__)}
        report['actual_imports'] = {}
        for name, module in list(sys.modules.items()):
            if name.split('.')[0] in ('bridge_rgs', 'scene', 'utils', 'gaussian_renderer', 'color_aggregation_network') and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve(); relative = str(path.relative_to(plan['source_snapshot']))
                require(sha(path) == plan['source_hashes'][relative], 'Unbound imported source')
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}
        require(report['render_calls'] == report['cached_target_loads'] == 16, 'Fixed counts changed')
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = old_flags
        report['field_unchanged'] = state_hashes() == before
        report['hook_restored'] = GaussianRasterizer.forward is original_forward
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        require(report['field_unchanged'] and report['hook_restored'], 'State restoration failed')


def execute(path, expected):
    require(sha(path) == expected, 'Plan identity changed')
    plan = read(path); output = Path(plan['output'])
    write(output/'execution_started.json', {'plan_sha256': expected})
    report = {'status': 'running', 'plan_sha256': expected, 'predictions': [], 'render_calls': 0,
              'cached_target_loads': 0, 'backward': 0, 'optimizer_steps': 0,
              'original_image_decodes': 0, 'semantic_label_decodes': 0, 'VAL_decodes': 0}
    started = time.monotonic()
    def expired(*_):
        raise TimeoutError('100-second compatibility limit')
    old = signal.signal(signal.SIGALRM, expired); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan); run(plan, report); verify(plan)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)
        report['elapsed_seconds'] = time.monotonic()-started
        write(output/'execution_receipt.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', type=Path); actions.add_argument('--execute', type=Path)
    parser.add_argument('--expected-plan-sha256'); args = parser.parse_args()
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode')
    prepare(args.prepare) if args.prepare else execute(args.execute, args.expected_plan_sha256)
