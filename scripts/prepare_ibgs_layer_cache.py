"""Fixed old IBGS field: 350 TRAIN source-free top4 layer caches, no RGB bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
TRAINING = RUNS/'ibgs_warm_matched_v1'
SELECT_CHECK = RUNS/'ibgs_layer_select_check_v1'
TRAIN_PLAN_SHA = 'c1af18ac17a1ce934208fa91f2b987f7a46c9920b7e35eb3ba9c95dafc056691'
CHECK_PLAN_SHA = '57cf2807d1df0309b50f1c6289f8e8b306cd9661bd7b039325ba1ebb99b19439'
CHECK_EXEC_SHA = '556bef539851b7a3733575d98fbeb1e6eeaec02dc318d8a94881f20009cc7768'
ENDPOINT_SHA = 'e977dc1c5c676e930c47ed78e56a5f95c25c8127a5c02f23876ddb8b4e1aa3ee'
BACKEND_SHA = '436b2b55df2b37cddda59e606ec0855ad125eb211058b502197674bd2b93adaf'
SELECTOR_SHA = 'fd20717c1e8a64f29d0ea827c03e738a9d165634154cafa4bf2471a9bb79fc2a'
EXTENSION_SHA = '5847a3c2595edc147a40469dc9c3f09e732e056dd72333a9dfb946e440a631d4'
ADAPTER_SHA = '450a95687eeece23066bf0a950caaa84d43d29c6aaff1f925ad14f4c47d198b1'
CONTRACT_SHA = '599f6cbf09a16f56bef6f37d0e0f8ce0a3f24449376213963ee033fc2ea9d84b'
SPEC = {
    'protocol': 'ibgs_fixed_old_full_top4_TRAIN_source_cache_v1',
    'format': 'ibgs_layer_source_cache_v1', 'endpoint': 'old full, step6000, noAA near.2',
    'views': 350, 'width': 1320, 'height': 989, 'point_count': 996009, 'slots': 4,
    'scene_calls': 350, 'selector_calls': 350, 'selection': 'top4 original alpha*T; no renormalization',
    'files': {'ids': 'int32 [H,W,4]', 'depth': 'float32 [H,W,4]', 'weights': 'float32 [H,W,4]'},
    'missing': {'id': -1, 'depth': 0., 'weight': 0.},
    'shared_valid': 'one TRAIN undistortion valid map eroded radius1 with outside invalid; saved bool [H,W]',
    'masking': 'raw full-grid selections are unchanged; eroded valid is separate for consumers',
    'source_RGB_decodes': 0, 'target_decodes': 0, 'semantic_label_decodes': 0,
    'VAL_reads': 0, 'source_depth_renders': 0, 'backward': 0, 'optimizer_steps': 0,
    'internal_seconds': 300, 'external_seconds': 360, 'output_byte_limit': 30*1024**3,
    'mass_sum_roundoff_allowance': float(8*np.finfo(np.float32).eps),
    'scope': 'Frozen-field source evidence cache; no network, target cache, training or quality claim',
}
CAMERA_KEYS = ('name', 'image_id', 'split', 'width', 'height', 'K', 'w2c_original')


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


def tree(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.md', '.cpp', '.cu'}}


def natural(folder, expected_plan):
    receipt, launch = read(folder/'execution_receipt.json'), read(folder/'launch_receipt.json')
    require(sha(folder/'plan.json') == expected_plan
            and receipt['status'] == launch['status'] == 'completed'
            and launch['exit_code'] == 0 and launch['natural_completion'] is True
            and receipt['plan_sha256'] == launch['plan_sha256'] == expected_plan
            and launch['execution_receipt_sha256'] == sha(folder/'execution_receipt.json'),
            'Natural completed bound parent required')
    return receipt


def camera_contract(contract):
    rows = contract['train_rows']
    names = [r['name'] for r in rows]
    require(contract['pixel_protocol'] == 'colmap_corner_v2'
            and len(names) == contract['train_count'] == 350 and names == sorted(set(names))
            and all(r['split'] == 'train' and Path(r['name']).name == r['name'] for r in rows),
            'Exactly the fixed sorted 350 TRAIN cameras required')
    require(all((r['width'], r['height']) == (SPEC['width'], SPEC['height'])
                and r['K'] == contract['common_camera']['K'] for r in rows), 'Common native TRAIN grid required')
    valid_paths = {r['valid_path'] for r in rows}
    require(len(valid_paths) == 1, 'Exactly one shared TRAIN valid map required')
    path = next(iter(valid_paths))
    neighbors = {r['name']: [v['name'] for v in r['neighbors_4']] for r in contract['neighbors']['views']}
    require(sorted(neighbors) == names and all(len(v) <= 4 and len(v) == len(set(v))
            and n not in v and set(v) <= set(names) for n, v in neighbors.items()), 'Original TRAIN-only neighbors required')
    return ([{k: r[k] for k in CAMERA_KEYS} for r in rows], neighbors,
            {'path': path, 'sha256': contract['pixel_hashes'][path]})


def validate_field_metadata(saved):
    require(saved['protocol'] == 'ibgs_warm_matched_v1' and saved['arm'] == 'full'
            and saved['step'] == 6000 and saved['sh_degree'] == 3,
            'Only fixed old full 6000-step field is reusable')
    require(saved['field']['_xyz'].shape == (SPEC['point_count'], 3), 'Point count/order identity changed')


def validate_cached_selection(ids, depth, weights, *, height, width, point_count):
    shape = (height, width, 4)
    require(ids.dtype == np.int32 and depth.dtype == weights.dtype == np.float32
            and ids.shape == depth.shape == weights.shape == shape, 'Cache dtype/shape mismatch')
    require(np.isfinite(depth).all() and np.isfinite(weights).all(), 'Nonfinite cache')
    require(((ids >= -1) & (ids < point_count)).all(), 'Invalid cached point ID')
    empty = ids == -1
    require((depth[empty] == 0).all() and (weights[empty] == 0).all()
            and (depth[~empty] > 0).all() and (weights[~empty] > 0).all(),
            'Missing slots must be -1/0/0; present slots require positive depth/mass')
    require((weights <= 1).all(), 'Individual mass outside [0,1]')
    mass = weights.astype(np.float64).sum(-1)
    require((mass <= 1+SPEC['mass_sum_roundoff_allowance']).all(), 'Selected mass exceeds compositing total')
    # The same Gaussian cannot occupy two selected slots on a ray.
    for a in range(4):
        for b in range(a+1, 4):
            require(not ((ids[..., a] >= 0) & (ids[..., a] == ids[..., b])).any(), 'Duplicate per-ray cached ID')
    return {'present_slots': int((~empty).sum()), 'empty_slots': int(empty.sum()),
            'maximum_selected_mass': float(mass.max()), 'minimum_selected_mass': float(mass.min())}


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    training, checked = natural(TRAINING, TRAIN_PLAN_SHA), natural(SELECT_CHECK, CHECK_PLAN_SHA)
    require(sha(SELECT_CHECK/'execution_receipt.json') == CHECK_EXEC_SHA
            and checked['numerical_status'] == 'passed' and checked['sources_inputs_unchanged'] is True,
            'Completed selector numerical check required')
    old, selected = read(TRAINING/'plan.json'), read(SELECT_CHECK/'plan.json')
    full = next(v for v in training['arms'] if v['arm'] == 'full')
    require(full['steps'] == 6000 and full['status'] == 'completed'
            and full['checkpoint_sha256'] == ENDPOINT_SHA and sha(full['last_checkpoint']) == ENDPOINT_SHA,
            'Fixed old full endpoint changed')
    require(sha(old['data_contract']) == old['data_contract_sha256'] == CONTRACT_SHA, 'Data contract changed')
    contract = read(old['data_contract']); views, neighbors, valid = camera_contract(contract)
    require(sha(valid['path']) == valid['sha256'], 'Shared validity changed')
    require(tree(old['source_snapshot']) == old['source_hashes'], 'Frozen old package changed')
    selector_file = Path(selected['source_snapshot'])/'bridge_rgs/ibgs_layer_select.py'
    require(sha(selector_file) == selected['source_hashes']['bridge_rgs/ibgs_layer_select.py'] == SELECTOR_SHA,
            'Checked selector source required')
    binary = checked['build']['binary']
    require(binary['sha256'] == EXTENSION_SHA and sha(binary['path']) == EXTENSION_SHA,
            'Use already-checked selector binary; no compilation')
    runtime = dict(old['runtime_sources']); runtime[binary['path']] = EXTENSION_SHA
    require(all(sha(p) == h for p, h in runtime.items()), 'Old backend/runtime changed')
    backend_paths = [p for p, h in runtime.items() if h == BACKEND_SHA]
    require(len(backend_paths) == 1, 'Exactly one original non-AA backend required')
    inputs = {str(folder/name): sha(folder/name) for folder in (TRAINING, SELECT_CHECK)
              for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json')}
    for p in [old['data_contract'], contract['manifest'], full['last_checkpoint'], full['receipt_path'],
              valid['path'], ROOT/'uv.lock', SELECT_CHECK/'analysis.json']:
        inputs[str(p)] = sha(p)
    require(inputs[contract['manifest']] == contract['manifest_sha256']
            and inputs[str(SELECT_CHECK/'analysis.json')] == checked['analysis_sha256'], 'Manifest/check analysis changed')
    require(not ({r['image_path'] for r in contract['train_rows']} & inputs.keys()), 'No source RGB byte reads in prepare')
    require(shutil.disk_usage(output.parent).free >= SPEC['output_byte_limit'], 'Require 30 GiB free before cache')
    snapshot = output/'source_snapshot'; snapshot.mkdir(parents=True)
    for name, digest in old['source_hashes'].items():
        path = snapshot/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(old['source_snapshot'])/name, path)
        require(sha(path) == digest, 'Old source copy changed')
    copied = {'bridge_rgs/ibgs_layer_select.py': SELECTOR_SHA, **checked['build']['source_hashes']}
    for name, digest in copied.items():
        path = snapshot/name; path.parent.mkdir(parents=True, exist_ok=True)
        origin = Path(selected['source_snapshot'])/name
        require(sha(origin) == digest, 'Checked selector source changed')
        shutil.copyfile(origin, path)
    adapter = ROOT/'src/bridge_rgs/ibgs_layer_adapter.py'
    require(sha(adapter) == ADAPTER_SHA, 'Reviewed layer adapter changed')
    shutil.copyfile(adapter, snapshot/'bridge_rgs/ibgs_layer_adapter.py')
    for path in [Path(__file__), ROOT/'tests/test_ibgs_layer_cache.py', ROOT/'docs/ibgs_layer_cache_protocol.md',
                 ROOT/'tests/test_ibgs_layer_adapter.py']:
        shutil.copyfile(path, snapshot/path.name)
    plan = {'protocol': SPEC['protocol'], 'specification': SPEC, 'output': str(output),
            'source_snapshot': str(snapshot), 'source_hashes': tree(snapshot),
            'inherited_source_hashes': old['source_hashes'], 'input_hashes': inputs, 'runtime_sources': runtime,
            'checkpoint': {'path': full['last_checkpoint'], 'sha256': ENDPOINT_SHA},
            'data_contract': {'path': old['data_contract'], 'sha256': CONTRACT_SHA},
            'shared_valid_input': valid, 'views': views, 'neighbors': neighbors,
            'selector': {'source_sha256': SELECTOR_SHA, 'adapter_sha256': ADAPTER_SHA, 'binary': binary, 'check_run': str(SELECT_CHECK),
                         'check_plan_sha256': CHECK_PLAN_SHA, 'check_execution_sha256': CHECK_EXEC_SHA},
            'backend': {'path': backend_paths[0], 'sha256': BACKEND_SHA},
            'interpreter': old['interpreter'], 'environment': old['environment'],
            'torch_version': checked['build']['torch'], 'cuda_version': checked['build']['cuda'],
            'numpy_version': np.__version__, 'external_timeout_seconds': 360,
            'prepare_model_deserializations': 0, 'prepare_pixel_decodes': 0, 'prepare_GPU_calls': 0}
    write(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    require(plan['protocol'] == SPEC['protocol'] and plan['specification'] == SPEC, 'Changed cache contract')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name,
            'Run frozen cache entry')
    require(tree(plan['source_snapshot']) == plan['source_hashes'], 'Frozen source changed')
    require(all(plan['source_hashes'].get(k) == h for k, h in plan['inherited_source_hashes'].items()),
            'Old field/render package inheritance changed')
    for group in ('input_hashes', 'runtime_sources'):
        require(all(sha(p) == h for p, h in plan[group].items()), 'Changed '+group)


def save_array(path, value):
    with path.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
    return {'path': str(path), 'sha256': sha(path), 'shape': list(value.shape),
            'dtype': str(value.dtype), 'bytes': path.stat().st_size}


def actual_imports(plan):
    snapshot = Path(plan['source_snapshot']); result = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and name.split('.')[0] in ('bridge_rgs', 'scene', 'utils', 'arguments', 'gaussian_renderer', 'color_aggregation_network'):
            path = Path(path).resolve()
            require(path.is_relative_to(snapshot) and plan['source_hashes'].get(str(path.relative_to(snapshot))) == sha(path),
                    'Unbound runtime Python source: '+str(path))
            result[name] = {'path': str(path), 'sha256': sha(path)}
    return result


def run(plan, report):
    import torch
    snapshot = Path(plan['source_snapshot'])
    sys.path[:0] = [str(snapshot/'official'), str(snapshot)]
    from diff_plane_rasterization import _C

    from bridge_rgs import ibgs_layer_select as selector
    from bridge_rgs.ibgs_adapter import BridgeCamera, _checked_image
    from bridge_rgs.ibgs_layer_adapter import load_selector_extension, render_layer_capture
    from bridge_rgs.ibgs_losses import erode_valid
    from bridge_rgs.ibgs_warm_training import FIELD_KEYS, load_saved_field

    require(str(Path(_C.__file__).resolve()) == plan['backend']['path'] and sha(_C.__file__) == BACKEND_SHA,
            'Original non-AA backend required')
    require(sha(selector.__file__) == SELECTOR_SHA and torch.__version__ == plan['torch_version']
            and torch.version.cuda == plan['cuda_version'] and np.__version__ == plan['numpy_version'],
            'Selector/Torch/CUDA/NumPy differs')
    require(Path(sys.prefix).resolve() == Path(plan['interpreter']).parent.parent.resolve(), 'Use original isolated interpreter')
    binary = Path(plan['selector']['binary']['path'])
    extension = load_selector_extension(binary, EXTENSION_SHA)
    require(sha(extension.__file__) == EXTENSION_SHA, 'Loaded selector binary differs')
    flags = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    original_backend = _C.rasterize_gaussians
    field = None; versions = {}; records = []
    try:
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        saved = torch.load(plan['checkpoint']['path'], map_location='cpu', weights_only=False)
        validate_field_metadata(saved)
        field, background = load_saved_field(saved); del saved
        for key in FIELD_KEYS:
            value = getattr(field, key); value.requires_grad_(False)
            require(torch.isfinite(value).all().item(), 'Nonfinite fixed field '+key)
        require(torch.isfinite(background).all().item(), 'Nonfinite background')
        versions = {k: getattr(field, k)._version for k in FIELD_KEYS}
        tensors = {k: getattr(field, k) for k in FIELD_KEYS}
        report['field_versions_before'] = versions
        valid_input = plan['shared_valid_input']
        valid = _checked_image(valid_input['path'], valid_input['sha256'], rgb=False) > 0
        report['shared_valid_decodes'] += 1
        require(valid.shape == (SPEC['height'], SPEC['width']), 'Shared valid grid changed')
        valid = erode_valid(torch.from_numpy(valid), 1).numpy()
        root = Path(plan['output'])
        shared = save_array(root/'shared_eroded_valid.npy', valid)
        report['saved_bytes'] += shared['bytes']
        directory = root/'cache'; directory.mkdir()
        with torch.inference_mode():
            for index, row in enumerate(plan['views']):
                camera = BridgeCamera(row, index)
                require(not camera.nearest_id and not camera.nearest_names, 'Source-free render required')
                result = render_layer_capture(camera, field, background, extension=extension)
                meta = result['metadata']
                report['scene_calls'] += 1
                report['raster_calls'] += meta['raster_calls']
                report['selector_calls'] += meta['selector_calls']
                require(meta['raster_calls'] == meta['selector_calls'] == 1
                        and meta['field_backward'] == meta['source_rgb_reads'] == meta['source_depth_renders'] == 0,
                        'One source-free render/selector required')
                require((meta['point_count'], meta['height'], meta['width'])
                        == (SPEC['point_count'], SPEC['height'], SPEC['width']), 'Field/grid changed')
                require(not torch.any(result['status'] != 0).item(), 'Nonzero selector status')
                top = {k: result['top4'][k].contiguous().cpu().numpy() for k in ('ids', 'depth', 'weights')}
                quality = validate_cached_selection(**top, height=meta['height'], width=meta['width'],
                                                    point_count=meta['point_count'])
                folder = directory/Path(row['name']).stem; folder.mkdir()
                files = {k: save_array(folder/(k+'.npy'), a) for k, a in top.items()}
                report['saved_bytes'] += sum(item['bytes'] for item in files.values())
                require(report['saved_bytes'] <= SPEC['output_byte_limit'], '30 GiB cache byte limit exceeded')
                actual = {**meta, 'viewmatrix': camera.world_view_transform.detach().cpu().tolist(),
                          'projmatrix': camera.full_proj_transform.detach().cpu().tolist(),
                          'matrix_layout': 'actual FP32 backend matrices; viewmatrix/projmatrix transposed convention'}
                require(np.array_equal(np.asarray(actual['viewmatrix'], np.float32).T,
                                       np.asarray(meta['world_to_camera'], np.float32)), 'Actual camera binding changed')
                records.append({'name': row['name'], 'camera': row, 'actual_camera': actual,
                                **files, 'selector_status_nonzero': 0, 'cache_validation': quality})
                del result, camera, top
                if (index+1) % 50 == 0:
                    print(json.dumps({'completed_TRAIN_caches': index+1, 'saved_bytes': report['saved_bytes']}), flush=True)
        require(report['scene_calls'] == report['raster_calls'] == report['selector_calls'] == 350,
                'Complete 350/350 cache required')
        require([r['name'] for r in records] == [v['name'] for v in plan['views']], 'Cache view order differs')
        manifest = {'status': 'completed', 'format': SPEC['format'], 'plan_sha256': report['plan_sha256'],
                    'source_snapshot': plan['source_snapshot'], 'checkpoint': plan['checkpoint'],
                    'data_contract': plan['data_contract'], 'selector': plan['selector'], 'backend': plan['backend'],
                    'point_count': SPEC['point_count'], 'neighbors': plan['neighbors'],
                    'shared_eroded_valid': shared, 'records': records, 'saved_bytes': report['saved_bytes'],
                    'full_grid_unmasked_selection': True, 'source_RGB_decodes': 0,
                    'scope': 'Only source top4 cache; no target/base/ray/median/RGB cache'}
        write(root/'cache_manifest.json', manifest)
        report['cache_manifest_sha256'] = sha(root/'cache_manifest.json')
    finally:
        if field is not None and versions:
            report['field_versions_after'] = {k: getattr(field, k)._version for k in FIELD_KEYS}
            report['field_versions_unchanged'] = all(getattr(field, k) is tensors[k]
                and getattr(field, k)._version == v and not getattr(field, k).requires_grad for k, v in versions.items())
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        torch.set_float32_matmul_precision(flags[0])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = flags[1:]
        report['numeric_flags_restored'] = flags == (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
        report['hook_restored'] = _C.rasterize_gaussians is original_backend
        report['actual_imports'] = actual_imports(plan)
        report['actual_backend'] = {'path': str(Path(_C.__file__).resolve()), 'sha256': sha(_C.__file__)}
        report['actual_selector_binary'] = {'path': str(Path(extension.__file__).resolve()), 'sha256': sha(extension.__file__)}
    require(report['field_versions_unchanged'] and report['hook_restored'] and report['numeric_flags_restored'],
            'Frozen-field/restoration check failed')


def execute(plan_path, expected_sha):
    require(sha(plan_path) == expected_sha, 'Explicit plan SHA required')
    plan = read(plan_path); output = Path(plan['output']); started = time.monotonic()
    write(output/'execution_started.json', {'plan_sha256': expected_sha})
    report = {'status': 'failed', 'plan_sha256': expected_sha, 'scene_calls': 0, 'raster_calls': 0,
              'selector_calls': 0, 'shared_valid_decodes': 0, 'saved_bytes': 0,
              'source_RGB_decodes': 0, 'target_decodes': 0, 'semantic_label_decodes': 0,
              'source_depth_renders': 0, 'VAL_reads': 0, 'backward': 0, 'optimizer_steps': 0}
    old_handler = signal.getsignal(signal.SIGALRM)
    def deadline(*_):
        raise TimeoutError('Fixed 300 second source-cache deadline')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan); run(plan, report); verify(plan)
        report.update(status='completed', cache_status='safe_complete_350', sources_inputs_unchanged=True)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
        report['elapsed_seconds'] = time.monotonic()-started
        write(output/'execution_receipt.json', report)
    print(json.dumps({'status': report['status'], 'receipt_sha256': sha(output/'execution_receipt.json'),
                      'cache_manifest_sha256': report['cache_manifest_sha256']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--prepare', type=Path)
    operation.add_argument('--execute', type=Path)
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        path = prepare(args.prepare)
        print(json.dumps({'plan': str(path), 'plan_sha256': sha(path), 'worker_sha256': sha(__file__)}))
    else:
        require(args.expected_plan_sha256 is not None, 'Pass explicit frozen plan SHA')
        execute(args.execute, args.expected_plan_sha256)


if __name__ == '__main__':
    main()
