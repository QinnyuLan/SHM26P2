"""Fixed TRAIN-only AA-to-IBGS raw-RGB starting-point comparison.

CPU preparation reads hashes/NPY headers only. Execution is a separately
authorized 16-render check, not training or an IBGS quality claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import signal
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
PARENT = Path('/mnt/data/SHM2026/runs/residual_transport_probe_v1')
PORT = Path('/mnt/data/SHM2026/runs/ibgs_port_preflight_v1')
CHECKPOINT = Path('/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1/training/last.pt')
BASE_SHA = '35b489fe45ad34ad279018609b6d5eaf6493e4c0d82f9c86ec16925268078092'
CONTRACT_SHA = '599f6cbf09a16f56bef6f37d0e0f8ce0a3f24449376213963ee033fc2ea9d84b'
PARENT_SHA = '061586d6309693a41c66b720159076e9bb52d6c1d7d605ec5d939a51402b297b'
PARENT_EXEC_SHA = '2a9f1eda9baddb4ce79a6880bf489a577d83b586e83c82fadd3450705130e5a5'
AUDIT_SHA = 'c2476de5dac9951b3f4318c1f55711f903fbf9dde5eea1c866d7cec877ce2b33'
BINARY_SHA = '436b2b55df2b37cddda59e606ec0855ad125eb211058b502197674bd2b93adaf'
NAMES = ['002.png', '021.png', '041.png', '059.png', '079.png', '100.png',
         '118.png', '137.png', '156.png', '176.png', '200.png', '220.png',
         '241.png', '259.png', '278.png', '300.png']
SPEC = {'protocol': 'ibgs_start_transfer_v1', 'targets': NAMES,
        'internal_seconds': 100, 'external_seconds': 120, 'render_calls': 16,
        'backward': 0, 'optimizer_steps': 0, 'fusion_calls': 0, 'source_RGB_decodes': 0,
        'original_target_RGB_decodes': 0, 'semantic_label_decodes': 0, 'VAL_decodes': 0,
        'camera': 'Same raw manifest FP64 K/w2c inputs as IBGS training; old AA used their FP32 casts; identical input geometry does not imply identical raster arithmetic',
        'score': 'FP64 RGB MSE over all 3 channels and the same cached valid pixels; equal view means',
        'outputs': 'raw unclipped FP32 RGB saved before cached TRAIN targets are loaded',
        'scope': 'Warm starting-field AA migration cost only; field fitted all 16 TRAIN views; no adoption gate'}


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


def array_headers(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            require(name.endswith('.npy') and '/' not in name, 'Simple NPY members required')
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), 'Unsupported NPY version')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(stream)
                require(not dtype.hasobject, 'No object arrays')
                result[name[:-4]] = {'shape': list(shape), 'dtype': str(dtype)}
    return result


def metrics(prediction, target, valid):
    require(prediction.shape == target.shape == (*valid.shape, 3) and valid.dtype == bool
            and valid.any() and np.isfinite(prediction).all() and np.isfinite(target).all(), 'Finite RGB and nonempty fixed support')
    error = prediction.astype(np.float64)[valid]-target.astype(np.float64)[valid]
    mse = float(np.mean(error*error))
    return {'mse': mse, 'psnr': -10*float(np.log10(mse)) if mse > 0 else None,
            'perfect_match': mse == 0, 'valid_pixels': int(valid.sum())}


def prediction_barrier(records):
    require([r['name'] for r in records] == NAMES and all(r.get('path') and r.get('sha256') for r in records),
            'All fixed 16 saved predictions must precede target-cache loading')


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    parent, execution, launch, audit, analysis = (read(PARENT/n) for n in
        ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json', 'analysis.json'))
    require(sha(PARENT/'plan.json') == PARENT_SHA and sha(PARENT/'execution_receipt.json') == PARENT_EXEC_SHA
            and sha(PARENT/'independent_cpu_review.json') == AUDIT_SHA, 'Fixed old probe lineage')
    require(execution['status'] == launch['status'] == 'completed' and launch['natural_completion']
            and launch['exit_code'] == 0 and audit['status'] == 'passed'
            and execution['plan_sha256'] == launch['plan_sha256'] == audit['plan_sha256'] == PARENT_SHA
            and launch['execution_receipt_sha256'] == audit['execution_receipt_sha256'] == PARENT_EXEC_SHA
            and sha(PARENT/'analysis.json') == execution['analysis_sha256'], 'Completed and independently audited parent')
    require(tree(parent['source_snapshot']) == parent['source_hashes'], 'Old probe sources changed')
    contract = read(PORT/'data_contract.json'); port = read(PORT/'plan_v2.json')
    require(sha(PORT/'data_contract.json') == CONTRACT_SHA and sha(CHECKPOINT) == BASE_SHA
            and parent['base_checkpoint'] == str(CHECKPOINT) and parent['specification']['base_sha256'] == BASE_SHA,
            'Same fixed 1M starting field and data contract')
    binary = port['binary']
    require(binary['sha256'] == BINARY_SHA and sha(binary['path']) == BINARY_SHA, 'Fixed corrected IBGS binary')
    rows = {v['name']: v for v in contract['train_rows']}
    old_views = {v['name']: v for v in parent['views']}
    renders = {v['name']: v for v in execution['renders']}
    references = {v['name']: v for v in analysis['views']}
    require([v['name'] for v in analysis['views']] == NAMES and len(rows) == 350, 'Fixed TRAIN population')
    inputs = {str(p): sha(p) for p in (CHECKPOINT, PORT/'data_contract.json', PORT/'plan_v2.json',
              PARENT/'plan.json', PARENT/'execution_receipt.json', PARENT/'launch_receipt.json',
              PARENT/'independent_cpu_review.json', PARENT/'independent_audit_launch_receipt.json',
              PARENT/'analysis.json', ROOT/'uv.lock')}
    views = []
    for name in NAMES:
        row = rows[name]
        require(row['split'] == old_views[name]['split'] == 'train', 'TRAIN cameras only')
        for key, old_key in [('K_fp32', 'K'), ('w2c_fp32', 'w2c_original')]:
            require(np.array_equal(np.asarray(row[key], np.float32), np.asarray(old_views[name][old_key], np.float32)), 'Camera mismatch')
        shape = [row['height'], row['width']]
        for record, required in [(renders[name], {'rgb': ('float32', shape+[3]), 'valid': ('bool', shape),
                                'K': ('float32', [3, 3]), 'w2c': ('float32', [4, 4])}),
                                 (references[name]['target'], {'rgb': ('float32', shape+[3]), 'valid': ('bool', shape)})]:
            require(sha(record['path']) == record['sha256'], 'Cached array changed')
            header = array_headers(record['path'])
            require(all(header[k] == {'dtype': d, 'shape': s} for k, (d, s) in required.items()), 'Cached array schema mismatch')
            inputs[record['path']] = record['sha256']
        # Keep only camera metadata, never propagate raw RGB/mask paths into execution.
        camera = {k: row[k] for k in ('name', 'split', 'image_id', 'width', 'height')}
        camera.update(K=row['K'], w2c_original=row['w2c_original'])
        views.append({'name': name, 'camera': camera, 'old_render': renders[name],
                      'target': references[name]['target'], 'old_reference': references[name]['metrics']['zero']})
    snapshot = output/'source_snapshot'; (snapshot/'bridge_rgs').mkdir(parents=True)
    (snapshot/'bridge_rgs/__init__.py').write_text('')
    for name in ('ibgs_adapter.py', 'ibgs_losses.py', 'ibgs_camera_contract.py', 'ibgs_warm_training.py'):
        shutil.copy2(ROOT/'src/bridge_rgs'/name, snapshot/'bridge_rgs'/name)
    shutil.copy2(Path(__file__), snapshot/Path(__file__).name)
    shutil.copy2(ROOT/'tests/test_ibgs_start_transfer.py', snapshot/'test_ibgs_start_transfer.py')
    repository = Path(port['repository']); frozen_repo = snapshot/'ibgs'
    for directory in ('scene', 'utils', 'gaussian_renderer', 'arguments'):
        shutil.copytree(repository/directory, frozen_repo/directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(repository/'color_aggregation_network.py', frozen_repo/'color_aggregation_network.py')
    bindings = {binary['path']: BINARY_SHA}
    for p in (repository/'submodules/diff-plane-rasterization').rglob('*'):
        if p.is_file() and p.suffix in ('.py', '.cu', '.h', '.cpp') and not any(k in p.parts for k in ('build', '__pycache__')):
            bindings[str(p)] = sha(p)
    package = Path(binary['path']).parent/'__init__.py'
    bindings[str(package)] = sha(package)
    plan = {'specification': SPEC, 'output': str(output), 'source_snapshot': str(snapshot),
            'repository': str(frozen_repo), 'checkpoint': str(CHECKPOINT), 'binary': binary,
            'source_hashes': tree(snapshot), 'input_hashes': inputs, 'installed_sources': bindings,
            'views': views, 'prepare_array_payloads_loaded': 0, 'prepare_checkpoint_loads': 0,
            'old_AA_equal_view_mse': analysis['equal_view_mean']['zero']['clipped_MSE']}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC and Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name,
            'Frozen entry and specification required')
    require(tree(plan['source_snapshot']) == plan['source_hashes'], 'Frozen sources changed')
    require(all(sha(p) == h for p, h in {**plan['input_hashes'], **plan['installed_sources']}.items()), 'Bound input/backend changed')


def run(plan, report):
    import torch
    from diff_plane_rasterization import _C

    sys.path.insert(0, plan['repository']); sys.path.insert(0, plan['source_snapshot'])
    from gaussian_renderer import render

    from bridge_rgs.ibgs_adapter import BridgeCamera
    from bridge_rgs.ibgs_warm_training import load_warm_field

    require(Path(_C.__file__).resolve() == Path(plan['binary']['path']).resolve()
            and sha(_C.__file__) == BINARY_SHA, 'Actually loaded fixed binary required')
    torch.set_num_threads(4); torch.manual_seed(20260927); torch.cuda.manual_seed_all(20260927)
    old_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.benchmark = False
    field = background = None
    try:
        field, background = load_warm_field(plan['checkpoint'])
        field_keys = ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest', '_normal', '_offset')
        before = {k: hashlib.sha256(getattr(field, k).detach().cpu().numpy().tobytes()).hexdigest() for k in field_keys}
        report.update(field_count=len(field._xyz), actual_binary=plan['binary'], field_hashes_before=before)
        pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
        options = SimpleNamespace(shuffle_source_frame=False, enable_exposure_correction=False)
        output = Path(plan['output']); arrays = output/'arrays'; arrays.mkdir()
        records = []
        with torch.no_grad():
            for index, view in enumerate(plan['views']):
                camera = BridgeCamera(view['camera'], index)
                actual_w2c = camera.world_view_transform.T.contiguous().cpu().numpy()
                require(actual_w2c.dtype == np.float32 and actual_w2c.tobytes() ==
                        np.asarray(view['camera']['w2c_original'], np.float32).tobytes(),
                        'Actual IBGS w2c differs from the original-camera FP32 cast')
                actual_w2c_sha = hashlib.sha256(actual_w2c.tobytes()).hexdigest()
                package = render(camera, field, None, pipe, options, background,
                    learnt_normal=True, nb_src_frames=1, buffer_length=4, depth_error_threshold=.01,
                    render_geo=False, return_depth_normal=False, do_find_closest_frame=False)
                report['render_calls'] += 1
                rgb = package['render'].permute(1, 2, 0).contiguous().cpu().numpy()
                require(rgb.dtype == np.float32 and np.isfinite(rgb).all(), 'Finite FP32 raw RGB required')
                path = arrays/(view['name']+'.raw.npy')
                with path.open('xb') as stream:
                    np.save(stream, rgb, allow_pickle=False)
                records.append({'name': view['name'], 'path': str(path), 'sha256': sha(path), 'shape': list(rgb.shape), 'dtype': str(rgb.dtype),
                                'actual_w2c_fp32_sha256': actual_w2c_sha})
                del camera, package, rgb
        prediction_barrier(records); write(output/'prediction_receipt.json', {'records': records, 'target_cache_loads': 0})
        report.update(predictions=records, prediction_barrier_complete=True)
        rows = []
        for view, record in zip(plan['views'], records, strict=True):
            with np.load(view['old_render']['path'], allow_pickle=False) as old, np.load(view['target']['path'], allow_pickle=False) as truth:
                report['cached_target_loads'] += 1
                valid, target, aa = truth['valid'], truth['rgb'], old['rgb']
                require(np.array_equal(valid, old['valid']) and np.array_equal(old['K'], np.asarray(view['camera']['K'], np.float32))
                        and np.array_equal(old['w2c'], np.asarray(view['camera']['w2c_original'], np.float32)), 'Actual cached camera/support mismatch')
                require(hashlib.sha256(old['w2c'].tobytes()).hexdigest() == record['actual_w2c_fp32_sha256'],
                        'Actual IBGS w2c is not byte-exact with the old AA cached w2c')
                raw = np.load(record['path'], allow_pickle=False)
                aa_score = metrics(aa, target, valid)
                require(abs(aa_score['mse']-view['old_reference']['clipped_MSE']) <= 1e-14, 'Old AA baseline metric mismatch')
                raw_score, clipped = metrics(raw, target, valid), metrics(np.clip(raw, 0, 1), target, valid)
                rows.append({'name': view['name'], 'old_AA': aa_score, 'IBGS_raw': raw_score, 'IBGS_clipped': clipped,
                             'clipped_mse_delta': clipped['mse']-aa_score['mse'],
                             'clipped_psnr_delta': clipped['psnr']-aa_score['psnr'] if clipped['psnr'] is not None and aa_score['psnr'] is not None else None,
                             'prediction_max_abs_difference': float(np.max(np.abs(raw.astype(np.float64)-aa.astype(np.float64))))})
        summary = {arm: {'equal_view_mse': float(np.mean([v[arm]['mse'] for v in rows])),
                         'mean_per_view_psnr': float(np.mean([v[arm]['psnr'] for v in rows])) if all(v[arm]['psnr'] is not None for v in rows) else None}
                   for arm in ('old_AA', 'IBGS_raw', 'IBGS_clipped')}
        require(abs(summary['old_AA']['equal_view_mse']-plan['old_AA_equal_view_mse']) <= 1e-14, 'Old AA population mean changed')
        write(output/'analysis.json', {'scope': SPEC['scope'], 'views': rows, 'summary': summary, 'adoption_gate': None})
        report.update(analysis_sha256=sha(output/'analysis.json'), summary=summary)
        after = {k: hashlib.sha256(getattr(field, k).detach().cpu().numpy().tobytes()).hexdigest() for k in field_keys}
        report.update(field_hashes_after=after, field_unchanged=before == after,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        require(before == after and report['render_calls'] == report['cached_target_loads'] == 16, 'Fixed counts/unchanged field')
        report['actual_imports'] = {}
        for name, module in list(sys.modules.items()):
            if name.split('.')[0] in ('bridge_rgs', 'scene', 'utils', 'gaussian_renderer', 'color_aggregation_network') and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve(); relative = str(path.relative_to(plan['source_snapshot']))
                require(sha(path) == plan['source_hashes'][relative], 'Unbound actual Python import')
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = old_flags
        report['numeric_flags_restored'] = old_flags == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)


def execute(path, expected):
    require(sha(path) == expected, 'Plan SHA mismatch')
    plan = read(path); output = Path(plan['output'])
    write(output/'execution_started.json', {'plan_sha256': expected})
    report = {'status': 'failed', 'plan_sha256': expected, 'render_calls': 0, 'cached_target_loads': 0,
              'backward': 0, 'optimizer_steps': 0, 'fusion_calls': 0, 'original_image_decodes': 0,
              'semantic_label_decodes': 0, 'VAL_decodes': 0}
    started = time.monotonic()
    def timeout(*_):
        raise TimeoutError('Fixed IBGS start-transfer deadline')
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--execute', type=Path)
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    require(bool(args.prepare) != bool(args.execute), 'Choose exactly prepare or execute')
    if args.prepare:
        prepare(args.prepare)
    else:
        execute(args.execute, args.expected_plan_sha256)
