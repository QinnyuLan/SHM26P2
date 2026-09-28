"""Fixed old IBGS endpoint: 16 source-free captures, then a CPU ray replay.

Preparation is metadata/hash-only. Capture runs in the old isolated backend;
replay runs in the main CPU environment. Neither stage opens an RGB target,
source image, source depth bank, annotation, or a future AA endpoint.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import shutil
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
TRAINING = RUNS/'ibgs_warm_matched_v1'
RECOVERY = RUNS/'ibgs_warm_train_recovery_v1'
TRAIN_PLAN_SHA = 'c1af18ac17a1ce934208fa91f2b987f7a46c9920b7e35eb3ba9c95dafc056691'
ENDPOINT_SHA = 'e977dc1c5c676e930c47ed78e56a5f95c25c8127a5c02f23876ddb8b4e1aa3ee'
BINARY_SHA = '436b2b55df2b37cddda59e606ec0855ad125eb211058b502197674bd2b93adaf'
SPEC = {
    'protocol': 'ibgs_thin_rays_capture_v1', 'endpoint': 'old full, step 6000; no AA, near .2',
    'capture_calls': 16, 'source_slots': 1, 'source_content': 'all-zero placeholder',
    'render_geo': True, 'render_depth_only': False, 'buffer_length': 4,
    'source_rgb_decodes': 0, 'source_depth_renders': 0, 'target_decodes': 0,
    'fusion_calls': 0, 'backward': 0, 'optimizer_steps': 0, 'VAL_reads': 0,
    'CPU_valid_members': 16, 'cached_target_RGB_members': 0,
    'centers_per_view': 512, 'neighborhood': '3x3', 'rays': 73728,
    'capture_internal_seconds': 300, 'capture_external_seconds': 360,
    'replay_internal_seconds': 600, 'replay_external_seconds': 660,
    'capture_byte_limit': 24*1024**3, 'replay_byte_limit': 8*1024**3,
    'per_view_prefix_byte_limit': 2*1024**3, 'minimum_free_bytes': 34*1024**3,
    'summary_atol': 2e-5, 'summary_rtol': 2e-4,
    'summary_interpretation': 'Descriptive reconstruction tolerance, not a CUDA error bound or scientific gate',
    'exact_cuda_ledger': False,
    'invalid_referenced_attributes': 'Fail the complete diagnostic; no mechanism population is reported',
    'future_descriptive_strata': {'nearer_than_median_fraction': .01,
        'projected_short_axis_std_max_px': 2., 'projected_axis_ratio_min': 4.,
        'omitted_top4_mass_description_min': .01, 'total_narrow_front_mass_max': .5,
        'same_id_neighborhood_min_rays': 3, 'same_id_min_cameras': 2},
    'selection': 'No training/adoption gate; all discrepancies remain in full denominator',
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


def safe_json(value):
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def tree(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.md'}}


def library(snapshot=None, name='ibgs_ray_replay'):
    path = (Path(snapshot)/'bridge_rgs'/(name+'.py') if snapshot else
            ROOT/'src/bridge_rgs'/(name+'.py'))
    spec = importlib.util.spec_from_file_location('fixed_'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def natural(folder):
    receipt, launch = read(folder/'execution_receipt.json'), read(folder/'launch_receipt.json')
    require(receipt['status'] == launch['status'] == 'completed'
            and launch['natural_completion'] is True and launch['exit_code'] == 0,
            'Completed natural producer required')
    require(receipt['plan_sha256'] == launch['plan_sha256'] == sha(folder/'plan.json')
            and launch['execution_receipt_sha256'] == sha(folder/'execution_receipt.json'),
            'Producer completion binding mismatch')
    return receipt


def prepare(args):
    require(sha(TRAINING/'plan.json') == TRAIN_PLAN_SHA, 'Fixed old training plan required')
    parent, completed = read(TRAINING/'plan.json'), natural(TRAINING)
    recover = natural(RECOVERY)
    audit = read(RECOVERY/'independent_cpu_review_v2.json')
    audit_launch = read(RECOVERY/'independent_audit_v2_launch_receipt.json')
    require(audit['status'] == 'passed' and audit_launch['status'] == 'completed'
            and audit_launch['exit_code'] == 0 and audit_launch['natural_completion'] is True,
            'Completed independent reference review required')
    require(audit_launch['report_sha256'] == sha(RECOVERY/'independent_cpu_review_v2.json'),
            'Reference audit binding mismatch')
    require(audit['execution_receipt_sha256'] == sha(RECOVERY/'execution_receipt.json')
            and audit['plan_sha256'] == audit_launch['plan_sha256'] == sha(RECOVERY/'plan.json'),
            'Reference review belongs to another execution')
    endpoint = next(row for row in completed['arms'] if row['arm'] == 'full')
    require(endpoint['steps'] == 6000 and endpoint['status'] == 'completed'
            and endpoint['checkpoint_sha256'] == ENDPOINT_SHA
            and sha(endpoint['last_checkpoint']) == ENDPOINT_SHA, 'Fixed old full endpoint required')
    require(tree(parent['source_snapshot']) == parent['source_hashes'], 'Old source changed')
    require(sha(RECOVERY/'prediction_receipt.json') == recover['predictions_sha256'], 'Reference list changed')
    replay = library()
    predictions = read(RECOVERY/'prediction_receipt.json')['records']
    refs = {r['name']: r for r in predictions if r['arm'] == 'full' and r['readout'] == 'raw'}
    require(set(refs) == set(replay.TARGETS), 'Exactly fixed sixteen old raw references required')
    contract = read(parent['data_contract'])
    require(contract['train_count'] == 350 and contract['pixel_protocol'] == 'colmap_corner_v2',
            'Fixed TRAIN corner camera contract required')
    cameras = {r['name']: r for r in contract['train_rows']}
    camera_keys = ('name', 'image_id', 'split', 'width', 'height', 'K', 'w2c_original')
    views = [{'camera': {k: cameras[n][k] for k in camera_keys},
              'raw_reference': {'path': refs[n]['path'], 'sha256': refs[n]['sha256']}}
             for n in replay.TARGETS]
    cached_views = {v['name']: v for v in read(RECOVERY/'plan.json')['views']}
    for view in views:
        cached = cached_views[view['camera']['name']]
        require(cached['camera'] == view['camera'], 'Camera differs from old raw prediction')
        view['valid_cache'] = dict(cached['target'], member='valid')
    require(all(v['camera']['split'] == 'train' for v in views), 'Only TRAIN metadata allowed')
    inputs = {str(folder/name): sha(folder/name) for folder in (TRAINING, RECOVERY)
              for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json')}
    for path in [RECOVERY/'prediction_receipt.json', RECOVERY/'independent_cpu_review_v2.json',
                 RECOVERY/'independent_audit_v2_launch_receipt.json', Path(parent['data_contract']),
                 Path(endpoint['last_checkpoint']), Path(endpoint['receipt_path']), ROOT/'uv.lock']:
        inputs[str(path)] = sha(path)
    for view in views:
        ref = view['raw_reference']; require(sha(ref['path']) == ref['sha256'], 'Reference RGB changed')
        inputs[ref['path']] = ref['sha256']
        valid = view['valid_cache']; require(sha(valid['path']) == valid['sha256'], 'Validity cache changed')
        inputs[valid['path']] = valid['sha256']
    runtime = dict(parent['runtime_sources'])
    require(any(h == BINARY_SHA for h in runtime.values()), 'Original backend binding missing')
    # Headers supplement the old binary's existing C++ source bindings; never
    # substitute the separate near/AA backend tree.
    cuda = Path('/mnt/data/SHM2026/third_party/ibgs/submodules/diff-plane-rasterization/cuda_rasterizer')
    for name in ('rasterizer_impl.h', 'config.h'):
        runtime[str(cuda/name)] = sha(cuda/name)
    require(all(sha(p) == h for p, h in runtime.items()), 'Original runtime changed')
    output = args.output.resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    require(shutil.disk_usage(output.parent).free >= SPEC['minimum_free_bytes'], 'Insufficient bounded capture/replay disk budget')
    snapshot = output/'source_snapshot'; snapshot.mkdir(parents=True)
    for name, digest in parent['source_hashes'].items():
        dst = snapshot/name; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(parent['source_snapshot'])/name, dst)
        require(sha(dst) == digest, 'Frozen parent copy changed')
    additions = {
        Path(__file__): snapshot/'capture_ibgs_thin_rays.py',
        ROOT/'src/bridge_rgs/ibgs_ray_replay.py': snapshot/'bridge_rgs/ibgs_ray_replay.py',
        ROOT/'tests/test_ibgs_ray_replay.py': snapshot/'test_ibgs_ray_replay.py',
        ROOT/'tests/test_capture_ibgs_thin_rays.py': snapshot/'test_capture_ibgs_thin_rays.py',
        ROOT/'src/bridge_rgs/ibgs_thin_summary.py': snapshot/'bridge_rgs/ibgs_thin_summary.py',
        ROOT/'tests/test_ibgs_thin_summary.py': snapshot/'test_ibgs_thin_summary.py',
        ROOT/'docs/ibgs_thin_rays_capture_protocol.md': snapshot/'ibgs_thin_rays_capture_protocol.md',
    }
    for src, dst in additions.items():
        shutil.copyfile(src, dst)
    plan = {'protocol': SPEC['protocol'], 'specification': SPEC, 'output': str(output),
        'source_snapshot': str(snapshot), 'source_hashes': tree(snapshot),
        'inherited_source_hashes': parent['source_hashes'], 'runtime_sources': runtime,
        'input_hashes': inputs, 'checkpoint': {'path': endpoint['last_checkpoint'], 'sha256': ENDPOINT_SHA},
        'views': views, 'interpreter': parent['interpreter'], 'environment': parent['environment'],
        'ABI': replay.ABI, 'point_count': 996009, 'cpu_numpy_version': np.__version__,
        'description_policy': library(name='ibgs_thin_summary').POLICY,
        'no_array_decode_or_model_deserialization_during_prepare': True}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json'),
                      'sources': len(plan['source_hashes']), 'inputs': len(inputs)}))


def verify(plan):
    require(plan['protocol'] == SPEC['protocol'] and plan['specification'] == SPEC, 'Changed fixed specification')
    require(tree(plan['source_snapshot']) == plan['source_hashes'], 'Frozen source changed')
    require(all(plan['source_hashes'].get(p) == h for p, h in plan['inherited_source_hashes'].items()),
            'Old source inheritance changed')
    for group in ('input_hashes', 'runtime_sources'):
        require(all(sha(p) == h for p, h in plan[group].items()), f'Changed {group}')


def prefix_lengths(n, instances, width, height, addresses):
    """Exact fromChunk prefix sizes; the absolute GPU address sets padding."""
    require(all(type(v) is int and v > 0 for v in (n, width, height))
            and type(instances) is int and instances >= 0, 'Invalid ABI dimensions')
    fields = {'geometry': [4*n, 3*n, 4*n, 8*n, 24*n, 16*n, 12*n, 4*n],
              'binning': [4*instances, 4*instances, 8*instances, 8*instances],
              'image': [4*width*height, 4*width*height, 8*width*height,
                        4*width*height, 4*width*height, 4*width*height]}
    lengths = {}
    for key, sizes in fields.items():
        offset = 0
        for size in sizes:
            offset += (-(int(addresses[key])+offset)) % 128
            offset += size
        lengths[key] = offset
    return lengths


@contextmanager
def observing_call(backend, callback):
    original = backend.rasterize_gaussians
    def wrapped(*args):
        require(len(args) == 29, 'Unexpected original CUDA input ABI')
        output = original(*args)
        require(isinstance(output, tuple) and len(output) == 13, 'Unexpected original CUDA output ABI')
        callback(args, output)
        return output
    backend.rasterize_gaussians = wrapped
    try:
        yield
    finally:
        backend.rasterize_gaussians = original


def save_array(path, value):
    with path.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'shape': list(value.shape), 'dtype': str(value.dtype)}


def capture(plan, report):
    import torch

    snapshot = Path(plan['source_snapshot'])
    sys.path[:0] = [str(snapshot/'official'), str(snapshot)]
    from diff_plane_rasterization import _C
    from gaussian_renderer import render

    from bridge_rgs.ibgs_adapter import BridgeCamera
    from bridge_rgs.ibgs_warm_training import FIELD_KEYS, load_saved_field

    require(sha(_C.__file__) == BINARY_SHA, 'Capture requires original non-AA binary')
    require(Path(sys.prefix).resolve() == Path(plan['interpreter']).parent.parent.resolve(),
            'Use fixed original isolated interpreter')
    torch.set_num_threads(4)
    original_backend = _C.rasterize_gaussians
    flags = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        replay = library(snapshot)
        saved = torch.load(plan['checkpoint']['path'], map_location='cpu', weights_only=False)
        require(saved['arm'] == 'full' and saved['step'] == 6000, 'Wrong fixed endpoint')
        field, background = load_saved_field(saved)
        require(len(field._xyz) == plan['point_count'], 'Changed Gaussian row order/size')
        for key in FIELD_KEYS:
            getattr(field, key).requires_grad_(False)
        versions = {k: getattr(field, k)._version for k in FIELD_KEYS}
        pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
        options = SimpleNamespace(shuffle_source_frame=False, enable_exposure_correction=False,
                                  nb_visible_src_frames=3, residual_resolution_scale=1.)
        root = Path(plan['output'])/'capture'; root.mkdir()
        records = []
        with torch.inference_mode():
            for view in plan['views']:
                camera = BridgeCamera(view['camera'], -1)
                require(not camera.nearest_id and not camera.nearest_names, 'Empty source selection required')
                directory = root/view['camera']['name']; directory.mkdir()
                captured = []
                def observe(args, result, captured=captured, directory=directory, view=view):
                    report['raster_calls'] += 1
                    require(not captured and args[15] == 1 and args[16] == 4 and args[26] is True
                            and args[27] is False, 'Fixed one-call geometry mode changed')
                    require(all(int(torch.count_nonzero(args[k])) == 0 for k in (11, 12, 13, 14)),
                            'Placeholder source data must be all zero')
                    n, instances, h, w = len(args[1]), int(result[0]), int(args[20]), int(args[21])
                    buffers = dict(zip(('geometry', 'binning', 'image'), result[10:13], strict=True))
                    addresses = {k: int(t.data_ptr()) for k, t in buffers.items()}
                    sizes = prefix_lengths(n, instances, w, h, addresses)
                    require(sum(sizes.values()) <= SPEC['per_view_prefix_byte_limit'], 'Per-view prefix budget exceeded')
                    require(all(t.dtype == torch.uint8 and t.ndim == 1 and t.is_contiguous()
                                and t.numel() >= sizes[k] for k, t in buffers.items()), 'Invalid original buffer ABI')
                    copies = {k: t[:sizes[k]].cpu().numpy().copy() for k, t in buffers.items()}
                    parsed = replay.parse_buffers(**copies, point_count=n, num_rendered=instances,
                                                  width=w, height=h, addresses=addresses)
                    require(parsed['prefix_bytes'] == sizes, 'Independent prefix length mismatch')
                    arrays = {k: save_array(directory/(k+'.npy'), v) for k, v in copies.items()}
                    values = {'all_map': args[8], 'background': args[0], 'viewmatrix': args[9],
                              'projmatrix': args[10], 'campos': args[24],
                              'raw_rgb': result[1].permute(1, 2, 0).contiguous(),
                              'median_depth': result[4].reshape(h, w)}
                    if args[2].numel():
                        values['colors_precomp'] = args[2]
                    for key, tensor in values.items():
                        arrays[key] = save_array(directory/(key+'.npy'), tensor.detach().cpu().numpy())
                    tanx, tany = np.float32(args[18]), np.float32(args[19])
                    settings = {'width': w, 'height': h, 'point_count': n, 'num_rendered': instances,
                        'focal': [float(np.float32(w)/(np.float32(2)*tanx)), float(np.float32(h)/(np.float32(2)*tany))],
                        'principal': [(w-1)*.5, (h-1)*.5], 'tanfov': [float(tanx), float(tany)],
                        'scale_modifier': float(args[6]), 'sh_degree': int(args[23]), 'buffer_length': int(args[16]),
                        'depth_error_threshold': float(args[17]), 'prefiltered': bool(args[25]),
                        'render_geo': bool(args[26]), 'render_depth_only': bool(args[27]), 'debug': bool(args[28]),
                        'source_slots': int(args[15]), 'zero_source_verified': True,
                        'colors': 'precomputed' if 'colors_precomp' in arrays else 'actual geometry RGB'}
                    captured.append({'name': view['camera']['name'], 'settings': settings, 'arrays': arrays,
                        'addresses': addresses, 'original_buffer_bytes': {k: t.numel() for k, t in buffers.items()},
                        'prefix_bytes': sizes, 'address_mod128': {k: v % 128 for k, v in addresses.items()}})
                with observing_call(_C, observe):
                    package = render(camera, field, SimpleNamespace(), pipe, options, background,
                        learnt_normal=True, nb_src_frames=4, buffer_length=4, depth_error_threshold=.01,
                        do_find_closest_frame=False, do_render_src_depth=False, render_geo=True,
                        return_depth_normal=False)
                report['target_calls'] += 1
                require(len(captured) == 1, 'Exactly one backend call per camera required')
                record = captured[0]
                previous = np.load(view['raw_reference']['path'], allow_pickle=False)
                raw = package['render'].permute(1, 2, 0).contiguous().cpu().numpy()
                require(previous.shape == raw.shape and previous.dtype == raw.dtype == np.float32, 'Raw reference shape/dtype changed')
                exact = previous.tobytes() == raw.tobytes()
                finite = bool(np.isfinite(previous).all() and np.isfinite(raw).all())
                record['old_raw_comparison'] = {'exact': exact,
                    'finite': finite,
                    'max_abs_error': float(np.max(np.abs(previous.astype(np.float64)-raw))) if finite else None,
                    'reference': view['raw_reference'], 'old_median_reference_available': False}
                write(directory/'record.json', record)
                records.append({'name': record['name'], 'path': str(directory/'record.json'), 'sha256': sha(directory/'record.json')})
                report['saved_bytes'] += sum(a['bytes'] for a in record['arrays'].values())
                require(report['saved_bytes'] <= SPEC['capture_byte_limit'], 'Capture disk budget exceeded')
                require(exact and finite, 'Source-free raw RGB differs/is nonfinite; preserve evidence and stop')
                del package, captured, record, previous, raw, camera
                gc.collect()
        report['field_versions_unchanged'] = all(getattr(field, k)._version == v for k, v in versions.items())
        require(report['field_versions_unchanged'], 'Field mutated during observational capture')
        require(report['raster_calls'] == report['target_calls'] == 16, 'Capture count changed')
        write(Path(plan['output'])/'captures.json', {'records': records})
        report['captures_sha256'] = sha(Path(plan['output'])/'captures.json')
        report['actual_imports'] = {}
        for name, module in list(sys.modules.items()):
            path = getattr(module, '__file__', None)
            if path and (name.split('.')[0] in ('bridge_rgs', 'scene', 'utils', 'arguments', 'gaussian_renderer', 'color_aggregation_network')
                         or name == 'fixed_ibgs_ray_replay'):
                path = Path(path).resolve(); relative = str(path.relative_to(snapshot))
                require(plan['source_hashes'].get(relative) == sha(path), 'Unbound local runtime import')
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}
        report['actual_binary'] = {'path': str(Path(_C.__file__).resolve()), 'sha256': sha(_C.__file__)}
    finally:
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        torch.set_float32_matmul_precision(flags[0])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = flags[1:]
        report['numeric_flags_restored'] = flags == (torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
        report['hook_restored'] = _C.rasterize_gaussians is original_backend


def pack_traces(rows):
    offsets = np.zeros(len(rows)+1, np.int64)
    offsets[1:] = np.cumsum([len(r['ids']) for r in rows], dtype=np.int64)
    output = {'offsets': offsets, 'xy': np.stack([r['xy'] for r in rows])}
    for key in ('ids', 'order', 'alpha', 'T', 'w', 'z', 'center_depth', 'median_mask', 'top4_mask'):
        output[key] = np.concatenate([r[key] for r in rows])
    output['median_slots'] = np.stack([r['median_slots'] for r in rows])
    return output


def load_validity(path, shape):
    """Read exactly one lazy NPZ member; the neighboring RGB member is forbidden."""
    with np.load(path, allow_pickle=False) as cached:
        valid = cached['valid']
    require(valid.shape == shape and valid.dtype == np.bool_, 'Expected original bool geometry-valid mask')
    return valid


def replay_capture(plan, report):
    root = Path(plan['output']); producer = natural(root)
    require(producer['target_calls'] == producer['raster_calls'] == 16
            and producer['captures_sha256'] == sha(root/'captures.json'), 'Complete capture required')
    replay = library(plan['source_snapshot'])
    summary = library(plan['source_snapshot'], 'ibgs_thin_summary')
    require(summary.POLICY == plan['description_policy'] and np.__version__ == plan['cpu_numpy_version'],
            'Description policy or CPU NumPy version changed')
    views = {v['camera']['name']: v for v in plan['views']}
    captures = read(root/'captures.json')['records']
    require([r['name'] for r in captures] == list(replay.TARGETS), 'Fixed capture order required')
    directory = root/'replay'; directory.mkdir()
    per_view, mechanisms = [], []
    for source in captures:
        require(sha(source['path']) == source['sha256'], 'Capture metadata changed')
        record = read(source['path']); settings = record['settings']
        arrays = {}
        for key, item in record['arrays'].items():
            require(sha(item['path']) == item['sha256'], 'Captured array changed')
            arrays[key] = np.load(item['path'], mmap_mode='r', allow_pickle=False)
        parsed = replay.parse_buffers(arrays['geometry'], arrays['binning'], arrays['image'],
            point_count=settings['point_count'], num_rendered=settings['num_rendered'],
            width=settings['width'], height=settings['height'], addresses=record['addresses'])
        sampling = replay.sample_rays(source['name'], settings['width'], settings['height'])
        rows = replay.replay_rays(parsed, sampling['xy'], arrays['all_map'],
            focal=settings['focal'], principal=settings['principal'], background=arrays['background'],
            raw_rgb=arrays['raw_rgb'], median_depth=arrays['median_depth'],
            colors_precomp=arrays.get('colors_precomp'), atol=SPEC['summary_atol'], rtol=SPEC['summary_rtol'],
            render_geo=settings['render_geo'], render_depth_only=settings['render_depth_only'], buffer_length=settings['buffer_length'])
        # NpzFile is lazy: read only 'valid', never the cached RGB member.
        valid = load_validity(views[source['name']]['valid_cache']['path'], (settings['height'], settings['width']))
        report['valid_members_loaded'] += 1
        validity = valid[sampling['xy'][:, 1], sampling['xy'][:, 0]]
        mechanism = summary.summarize_view(source['name'], rows, sampling, parsed['conic_opacity'], validity)
        mechanism_path = directory/(source['name']+'.mechanism.json')
        write(mechanism_path, mechanism); mechanisms.append(mechanism)
        traces = pack_traces(rows); traces.update(cell_id=sampling['cell_id'], is_center=sampling['is_center'])
        traces['validity'] = validity
        path = directory/(source['name']+'.npz')
        require(sum(a.nbytes for a in traces.values())+report['saved_bytes'] <= SPEC['replay_byte_limit'], 'Replay trace budget exceeded')
        with path.open('xb') as stream:
            np.savez(stream, **traces)
        summaries = [{k: r[k] for k in ('summary', 'production', 'comparison', 'threshold_margins',
                                       'finite_cpu_arithmetic', 'stop_ordinal', 'mass_closure_error', 'tile')} for r in rows]
        # Nonfinite arithmetic remains an explicit status; strict JSON uses NA,
        # while the trace NPZ preserves the original inf/NaN without alteration.
        summary_path = directory/(source['name']+'.json')
        write(summary_path, safe_json({'name': source['name'], 'rays': summaries}))
        consistent = np.asarray([r['comparison']['status'] == 'summary_consistent_reconstruction' for r in rows])
        masses = [float(r['w'][r['top4_mask'] & ~r['median_mask']].sum(dtype=np.float64))
                  for r, ok in zip(rows, consistent, strict=True) if ok]
        per_view.append({'name': source['name'], 'rays': len(rows), 'centers': int(sampling['is_center'].sum()),
            'consistent_rays': int(consistent.sum()), 'consistent_centers': int(consistent[sampling['is_center']].sum()),
            'valid_rays': int(validity.sum()), 'valid_centers': int(validity[sampling['is_center']].sum()),
            'consistent_valid_rays': int((consistent & validity).sum()),
            'consistent_valid_centers': int((consistent & validity & sampling['is_center']).sum()),
            'mismatch_rays': int((~consistent).sum()),
            'consistent_subset_omitted_top4_mass_mean': float(np.mean(masses)) if masses else None,
            'trace': {'path': str(path), 'sha256': sha(path)},
            'mechanism': {'path': str(mechanism_path), 'sha256': sha(mechanism_path)},
            'summaries': {'path': str(summary_path), 'sha256': sha(summary_path)}})
        report['saved_bytes'] += path.stat().st_size+summary_path.stat().st_size+mechanism_path.stat().st_size
        require(report['saved_bytes'] <= SPEC['replay_byte_limit'], 'Replay output budget exceeded')
        del arrays, parsed, rows, traces, summaries, valid
        gc.collect()
    report.update(rays=sum(r['rays'] for r in per_view), centers=sum(r['centers'] for r in per_view),
                  consistent_rays=sum(r['consistent_rays'] for r in per_view),
                  consistent_centers=sum(r['consistent_centers'] for r in per_view))
    require(report['rays'] == 73728 and report['centers'] == 8192 and report['valid_members_loaded'] == 16,
            'Incomplete fixed ray population')
    write(root/'replay_analysis.json', {'views': per_view, 'full_ray_denominator': 73728,
        'full_center_denominator': 8192, 'exact_cuda_ledger': False, 'mechanism_or_adoption_gate': None,
        'mechanism_summary': summary.summarize_views(mechanisms),
        'scope': 'Only summary-consistent finite reconstructions may be described; top4 mass is definitionally selected, not performance/novelty evidence'})
    report['analysis_sha256'] = sha(root/'replay_analysis.json')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--capture', action='store_true')
    mode.add_argument('--replay', action='store_true')
    parser.add_argument('--output', type=Path); parser.add_argument('--plan', type=Path)
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        prepare(args); return
    require(args.expected_plan_sha256 and sha(args.plan) == args.expected_plan_sha256, 'Explicit plan SHA required')
    plan = read(args.plan); output = Path(plan['output']); stage = 'capture' if args.capture else 'replay'
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/'capture_ibgs_thin_rays.py', 'Run frozen worker only')
    started = time.monotonic()
    write(output/(stage+'_started.json'), {'plan_sha256': args.expected_plan_sha256, 'stage': stage})
    report = {'status': 'failed', 'plan_sha256': args.expected_plan_sha256, 'stage': stage,
        'raster_calls': 0, 'target_calls': 0, 'source_rgb_decodes': 0, 'source_depth_renders': 0,
        'target_decodes': 0, 'fusion_calls': 0, 'backward': 0, 'optimizer_steps': 0, 'VAL_reads': 0,
        'saved_bytes': 0, 'valid_members_loaded': 0, 'cached_target_RGB_members': 0}
    def timeout(*_):
        raise TimeoutError('Fixed thin-ray stage deadline')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(SPEC[stage+'_internal_seconds'])
    try:
        verify(plan)
        (capture if args.capture else replay_capture)(plan, report)
        if args.capture:
            require(report['numeric_flags_restored'] and report['hook_restored'], 'Capture restoration failed')
        verify(plan); report.update(status='completed', sources_inputs_unchanged=True)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        signal.alarm(0); report['elapsed_seconds'] = time.monotonic()-started
        filename = 'execution_receipt.json' if args.capture else 'replay_execution_receipt.json'
        write(output/filename, safe_json(report))
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
