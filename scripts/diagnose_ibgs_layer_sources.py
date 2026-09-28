"""Bounded old-IBGS source-layer query: observation only, no RGB/label input.

Capture full-geometry and actual depth-only source renders separately. The CPU
stage reconstructs source predicates with ideal and 8-fraction-bit interpolation;
it never claims those predicates or contribution traces are an exact CUDA ledger.
"""
from __future__ import annotations

import argparse
import gc
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
RUNS = Path('/mnt/data/SHM2026/runs')
PARENT = RUNS/'ibgs_thin_rays_capture_v1'
SPEC = {
    'protocol': 'ibgs_layer_sources_v1', 'target_centers': 8192,
    'eligible_target_centers': 8131, 'target_views': 16, 'sources_per_target': 4,
    'unique_sources': 64, 'full_geometry_renders': 64, 'depth_only_renders': 64,
    'source_rgb_decodes': 0, 'target_rgb_decodes': 0, 'semantic_decodes': 0,
    'backward': 0, 'optimizer_steps': 0, 'source_validity': 'same shared valid, radius-one erosion',
    'candidate_union': 'median4 union top4, finite positive plane depth; all eligible centers retained',
    'gate_relative_error': .01, 'projection_epsilon': 1e-8,
    'interpolation': 'ideal bilinear and nearest-even 8-fraction-bit approximation, both reported',
    'predicate_scope': 'CPU FP32 transform/projection and texture approximation, not production gate attestation',
    'gate_ambiguity': 'ideal and quantized predicate disagree, or fractional tie to quantization midpoint',
    'summary_atol': 2e-5, 'summary_rtol': 2e-4,
    'invalid_referenced_attributes': 'fail complete diagnostic; never silently exclude bad source attributes',
    'capture_internal_seconds': 300, 'capture_external_seconds': 360,
    'analyze_internal_seconds': 600, 'analyze_external_seconds': 660,
    'output_byte_limit': 40*1024**3, 'per_source_prefix_limit': 2*1024**3,
    'minimum_free_bytes': 42*1024**3,
    'scientific_or_adoption_gate': None,
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def old(snapshot=None):
    folder = Path(snapshot) if snapshot else PARENT/'source_snapshot'
    return load_module(folder/'capture_ibgs_thin_rays.py', 'layer_capture_helpers')


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def project_query(xy, depth, ref_to_src, focal, principal):
    """Separate-round FP32 equivalent of original CUDA point projection.

    Actual ref_to_src is produced and saved by Torch CUDA in capture using the
    author's matrix expression. CPU FMA/texture differences remain disclosed.
    """
    xy, depth, matrix, focal, principal = (np.asarray(v, np.float32) for v in
                                         (xy, depth, ref_to_src, focal, principal))
    require(xy.shape == (len(depth), 2) and depth.ndim == 1 and matrix.shape == (4, 4)
            and focal.shape == principal.shape == (2,) and np.isfinite(matrix).all()
            and np.isfinite(focal).all() and (focal > 0).all(), 'Invalid projection inputs')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        inv_focal = np.float32(1)/focal
        points = np.column_stack((((xy[:, 0]-principal[0])*depth)*inv_focal[0],
                                  ((xy[:, 1]-principal[1])*depth)*inv_focal[1], depth))
        transformed = np.empty_like(points)
        for k in range(3):
            transformed[:, k] = ((matrix[k, 0]*points[:, 0]+matrix[k, 1]*points[:, 1])
                                  +matrix[k, 2]*points[:, 2])+matrix[k, 3]
        inv_z = np.float32(1)/(transformed[:, 2]+np.float32(1e-8))
        uv = transformed[:, :2]*focal*inv_z[:, None]+principal
    return uv, transformed[:, 2]


def bilinear_query(image, uv):
    """Unnormalized clamp texture at uv+.5; invalid domain stays explicitly NA."""
    image, uv = np.asarray(image), np.asarray(uv, np.float32)
    require(image.ndim == 2 and image.dtype == np.float32 and np.isfinite(image).all()
            and uv.ndim == 2 and uv.shape[1] == 2, 'Finite FP32 depth and Nx2 UV required')
    h, w = image.shape
    inside = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] <= w-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h-1)
    # CUDA receives tex(p+.5); that addition rounds before its fraction is
    # formed. Bounds above deliberately use the original, un-clamped p.
    texture_coordinate = np.where(inside[:, None], np.float32(uv+np.float32(.5)), np.float32(.5))
    safe = texture_coordinate.astype(np.float64)-.5
    floor = np.floor(safe).astype(np.int64); frac = safe-floor
    taps = floor[:, None, :]+np.array(((0, 0), (1, 0), (0, 1), (1, 1)), np.int64)
    taps = np.minimum(np.maximum(taps, 0), [w-1, h-1]).astype(np.int32)
    def weights(f):
        x, y = f[:, 0], f[:, 1]
        return np.column_stack(((1-x)*(1-y), x*(1-y), (1-x)*y, x*y))
    ideal_weights = weights(frac)
    quantized_weights = weights(np.rint(frac*256)/256).astype(np.float32)
    values = image[taps[:, :, 1], taps[:, :, 0]]
    ideal = np.sum(ideal_weights*values.astype(np.float64), axis=1)
    quantized = np.sum(quantized_weights*values, axis=1, dtype=np.float32)
    ideal[~inside] = np.nan; quantized[~inside] = np.nan
    midpoint = np.any(np.mod(frac*256, 1) == .5, axis=1) & inside
    return {'taps': taps, 'ideal_weights': ideal_weights, 'quantized_weights': quantized_weights,
            'inside': inside, 'ideal': ideal, 'quantized': quantized, 'quantization_midpoint': midpoint}


def depth_predicate(depth_map, uv, source_z):
    """Candidate G_F or representative G_M; no total-source-weight condition."""
    sample = bilinear_query(depth_map, uv)
    source_z = np.asarray(source_z, np.float32)
    require(source_z.shape == sample['inside'].shape, 'Depth/pixel length mismatch')
    geometric = sample['inside'] & np.isfinite(source_z) & (source_z > 0)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        ei = np.abs(sample['ideal']-source_z.astype(np.float64))/(source_z.astype(np.float64)+1e-8)
        eq = np.abs(sample['quantized']-source_z)* (np.float32(1)/(source_z+np.float32(1e-8)))
    gi = geometric & (sample['ideal'] > 0) & (ei < .01)
    gq = geometric & (sample['quantized'] > 0) & (eq < np.float32(.01))
    return dict(sample, source_z=source_z, geometric=geometric, ideal_error=ei, quantized_error=eq,
                ideal_margin=.01-ei, quantized_margin=np.float32(.01)-eq,
                ideal_gate=gi, quantized_gate=gq,
                ambiguous=(gi != gq) | sample['quantization_midpoint'])


def candidate_table(trace, summaries, conic, footprint):
    """Only existing captured center records; neither errors nor labels select IDs."""
    eligible = np.asarray([r['comparison']['status'] == 'summary_consistent_reconstruction'
                           and r['finite_cpu_arithmetic'] for r in summaries], bool)
    eligible &= trace['validity'] & trace['is_center']
    centers = np.flatnonzero(eligible)
    rows, indices = [], []
    for center_index, ray_index in enumerate(centers):
        start, stop = trace['offsets'][ray_index:ray_index+2]
        select = ((trace['median_mask'][start:stop] | trace['top4_mask'][start:stop])
                  & np.isfinite(trace['z'][start:stop]) & (trace['z'][start:stop] > 0))
        selected = np.flatnonzero(select)+start
        rows.extend([center_index]*len(selected)); indices.extend(selected.tolist())
    indices = np.asarray(indices, np.int64); center_indices = np.asarray(rows, np.int64)
    ids = trace['ids'][indices]
    short, ratio, finite_footprint = footprint(conic[ids])
    median_actual = np.asarray([summaries[i]['production']['median_depth'] for i in centers], np.float32)
    median_replay = np.asarray([summaries[i]['summary']['median_depth'] for i in centers], np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        gap = (median_replay[center_indices]-trace['z'][indices])/median_replay[center_indices]
    narrow = finite_footprint & (short <= 2) & (ratio >= 4) & (gap >= .01)
    result = {'center_ray_index': centers, 'center_xy': trace['xy'][centers],
              'center_cell': trace['cell_id'][centers], 'center_median_depth': median_actual,
              'candidate_center_index': center_indices, 'ids': ids,
              'short_std': short, 'axis_ratio': ratio, 'relative_front_gap': gap,
              'finite_footprint': finite_footprint, 'narrow_front': narrow}
    for key in ('order', 'w', 'z', 'median_mask', 'top4_mask'):
        result[key] = trace[key][indices]
    return result


def save_npz(path, arrays, base):
    with Path(path).open('xb') as stream:
        np.savez(stream, **arrays)
    return {'path': str(path), 'sha256': base.sha(path), 'bytes': Path(path).stat().st_size}


def completed_parent(base):
    plan = base.read(PARENT/'plan.json'); base.natural(PARENT)
    receipt = base.read(PARENT/'replay_execution_receipt.json')
    launch = base.read(PARENT/'replay_launch_receipt.json')
    require(receipt['status'] == launch['status'] == 'completed' and launch['exit_code'] == 0
            and launch['natural_completion'] is True
            and receipt['plan_sha256'] == launch['plan_sha256'] == base.sha(PARENT/'plan.json')
            and launch['execution_receipt_sha256'] == base.sha(PARENT/'replay_execution_receipt.json')
            and receipt['analysis_sha256'] == base.sha(PARENT/'replay_analysis.json'), 'Completed bound parent replay required')
    require(base.tree(plan['source_snapshot']) == plan['source_hashes'], 'Captured parent source changed')
    return plan


def prepare(args):
    base = old(); parent = completed_parent(base)
    output = args.output.resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    require(shutil.disk_usage(output.parent).free >= SPEC['minimum_free_bytes'], 'Insufficient fixed disk budget')
    training = base.read(base.TRAINING/'plan.json'); contract = base.read(training['data_contract'])
    camera_keys = ('name', 'image_id', 'split', 'width', 'height', 'K', 'w2c_original')
    camera_rows = {r['name']: r for r in contract['train_rows']}
    neighbors = {r['name']: [n['name'] for n in r['neighbors_4']] for r in contract['neighbors']['views']}
    replay = base.library(parent['source_snapshot'])
    description = base.library(parent['source_snapshot'], 'ibgs_thin_summary')
    capture_records = {r['name']: r for r in base.read(PARENT/'captures.json')['records']}
    replay_records = {r['name']: r for r in base.read(PARENT/'replay_analysis.json')['views']}
    sources = []
    for view in parent['views']:
        name = view['camera']['name']; require(len(neighbors[name]) == 4, 'Four existing neighbors required')
        for slot, source_name in enumerate(neighbors[name]):
            sources.append({'name': source_name, 'target': name, 'slot': slot,
                            'camera': {k: camera_rows[source_name][k] for k in camera_keys}})
    require(len(sources) == len({r['name'] for r in sources}) == 64
            and not set(capture_records).intersection(r['name'] for r in sources), 'Fixed 64 distinct non-target sources required')
    require(len({r['valid_path'] for r in contract['train_rows']}) == 1, 'Shared geometry-valid metadata required')
    snapshot = output/'source_snapshot'; snapshot.mkdir(parents=True)
    for name, digest in parent['source_hashes'].items():
        target = snapshot/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(parent['source_snapshot'])/name, target)
        require(base.sha(target) == digest, 'Parent source copy changed')
    for source, target in ((Path(__file__), snapshot/Path(__file__).name),
                           (ROOT/'tests/test_ibgs_layer_sources.py', snapshot/'test_ibgs_layer_sources.py'),
                           (ROOT/'docs/ibgs_layer_sources_protocol.md', snapshot/'ibgs_layer_sources_protocol.md')):
        shutil.copyfile(source, target)
    inputs = {str(PARENT/name): base.sha(PARENT/name) for name in
              ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'replay_execution_receipt.json',
               'replay_launch_receipt.json', 'replay_analysis.json', 'captures.json')}
    inputs.update({str(training['data_contract']): base.sha(training['data_contract']),
                   parent['checkpoint']['path']: parent['checkpoint']['sha256'], str(ROOT/'uv.lock'): base.sha(ROOT/'uv.lock')})
    views = []; directory = output/'candidates'; directory.mkdir()
    for view in parent['views']:
        name = view['camera']['name']; reference = replay_records[name]
        record_item = capture_records[name]; record = base.read(record_item['path'])
        inputs[record_item['path']] = record_item['sha256']
        for item in record['arrays'].values():
            inputs[item['path']] = item['sha256']
        for key in ('trace', 'summaries'):
            item = reference[key]; inputs[item['path']] = item['sha256']
            require(base.sha(item['path']) == item['sha256'], 'Bound parent trace changed')
        arrays = {k: np.load(v['path'], mmap_mode='r', allow_pickle=False) for k, v in record['arrays'].items()
                  if k in ('geometry', 'binning', 'image')}
        st = record['settings']
        parsed = replay.parse_buffers(**arrays, point_count=st['point_count'], num_rendered=st['num_rendered'],
                                      width=st['width'], height=st['height'], addresses=record['addresses'])
        with np.load(reference['trace']['path'], allow_pickle=False) as trace:
            table = candidate_table(trace, base.read(reference['summaries']['path'])['rays'],
                                    parsed['conic_opacity'], description.footprint)
        item = save_npz(directory/(name+'.npz'), table, base); inputs[item['path']] = item['sha256']
        views.append({'camera': view['camera'], 'settings': st, 'candidates': item,
                      'eligible_centers': len(table['center_xy']), 'candidate_count': len(table['ids']),
                      'captured_viewmatrix': record['arrays']['viewmatrix']})
        del arrays, parsed, table
    require(sum(v['eligible_centers'] for v in views) == 8131, 'Frozen eligible population changed')
    valid = dict(parent['views'][0]['valid_cache'])
    inputs[valid['path']] = valid['sha256']
    for path, digest in inputs.items():
        require(base.sha(path) == digest, 'Input identity changed before freeze')
    plan = {'protocol': SPEC['protocol'], 'specification': SPEC, 'output': str(output),
            'source_snapshot': str(snapshot), 'source_hashes': base.tree(snapshot),
            'inherited_source_hashes': parent['source_hashes'], 'runtime_sources': parent['runtime_sources'],
            'input_hashes': inputs, 'checkpoint': parent['checkpoint'], 'interpreter': parent['interpreter'],
            'point_count': parent['point_count'], 'views': views, 'sources': sources,
            'geometry_valid_cache': valid, 'numpy_version': np.__version__,
            'description_policy': description.POLICY,
            'prepare_scope': 'Existing model-derived traces plus camera metadata/valid-cache hash; no images or model loaded'}
    base.write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': base.sha(output/'plan.json'),
                      'sources': len(plan['source_hashes']), 'inputs': len(inputs),
                      'candidates': sum(v['candidate_count'] for v in views)}))


def verify(plan, base):
    require(plan['specification'] == SPEC and plan['protocol'] == SPEC['protocol'], 'Changed fixed contract')
    require(base.tree(plan['source_snapshot']) == plan['source_hashes'], 'Frozen source changed')
    require(all(plan['source_hashes'].get(k) == v for k, v in plan['inherited_source_hashes'].items()), 'Parent code changed')
    for key in ('input_hashes', 'runtime_sources'):
        require(all(base.sha(p) == h for p, h in plan[key].items()), f'Changed {key}')


def capture(plan, report, base):
    import torch
    snapshot = Path(plan['source_snapshot']); sys.path[:0] = [str(snapshot/'official'), str(snapshot)]
    from diff_plane_rasterization import _C
    from gaussian_renderer import render, render_depth

    from bridge_rgs.ibgs_adapter import BridgeCamera, _SourceDepthBank
    from bridge_rgs.ibgs_losses import erode_valid
    from bridge_rgs.ibgs_warm_training import FIELD_KEYS, load_saved_field
    require(base.sha(_C.__file__) == base.BINARY_SHA and plan['checkpoint']['sha256'] == base.ENDPOINT_SHA,
            'Original old endpoint/backend only')
    require(Path(sys.prefix).resolve() == Path(plan['interpreter']).parent.parent.resolve(), 'Use original isolated environment')
    original = _C.rasterize_gaussians
    flags = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        saved = torch.load(plan['checkpoint']['path'], map_location='cpu', weights_only=False)
        require(saved['arm'] == 'full' and saved['step'] == 6000, 'Wrong endpoint')
        field, background = load_saved_field(saved)
        for key in FIELD_KEYS:
            getattr(field, key).requires_grad_(False)
        versions = {key: getattr(field, key)._version for key in FIELD_KEYS}
        target_views = {v['camera']['name']: v for v in plan['views']}
        target_cameras = {n: BridgeCamera(v['camera'], -1) for n, v in target_views.items()}
        for name, camera in target_cameras.items():
            prior = np.load(target_views[name]['captured_viewmatrix']['path'], allow_pickle=False)
            require(np.array_equal(prior, camera.world_view_transform.cpu().numpy()), 'Target camera FP32 identity changed')
        source_cameras = {s['name']: BridgeCamera(s['camera'], i) for i, s in enumerate(plan['sources'])}
        transforms = {}
        for name, target in target_cameras.items():
            selected = [s for s in plan['sources'] if s['target'] == name]
            require([s['slot'] for s in selected] == list(range(4)), 'Original source order changed')
            world_to_src = torch.stack([source_cameras[s['name']].world_view_transform.T for s in selected])
            batch = world_to_src @ target.world_view_transform.T.inverse().unsqueeze(0)
            for s, transform in zip(selected, batch, strict=True):
                transforms[s['name']] = transform
        h, w = plan['views'][0]['settings']['height'], plan['views'][0]['settings']['width']
        valid = base.load_validity(plan['geometry_valid_cache']['path'], (h, w)); report['valid_members_loaded'] += 1
        eroded = erode_valid(torch.from_numpy(valid.copy()), 1)
        bank = _SourceDepthBank([s['camera'] for s in plan['sources']], eroded)
        root = Path(plan['output'])/'sources'; root.mkdir()
        valid_record = base.save_array(root/'eroded_valid.npy', eroded.numpy())
        original_valid_record = base.save_array(root/'original_valid.npy', valid)
        report['saved_bytes'] = sum(p.stat().st_size for p in Path(plan['output']).rglob('*') if p.is_file())
        pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
        options = SimpleNamespace(shuffle_source_frame=False, enable_exposure_correction=False,
                                  nb_visible_src_frames=3, residual_resolution_scale=1.)
        records = []; replay = base.library(snapshot)
        with torch.inference_mode():
            for index, source in enumerate(plan['sources']):
                camera = source_cameras[source['name']]; directory = root/source['name']; directory.mkdir()
                require(not camera.nearest_id, 'Empty-source rendering required')
                captures = []
                def observe(args, result, captures=captures, directory=directory, source=source):
                    report['raster_calls'] += 1
                    require(all(int(torch.count_nonzero(args[k])) == 0 for k in (11, 12, 13, 14)), 'No source RGB/depth permitted')
                    require(args[15] == 1 and args[16] == 4, 'Changed slot ABI')
                    if args[27]:
                        require(args[26] is False, 'Depth-only call expected')
                        report['depth_only_renders'] += 1
                        return
                    require(args[26] is True and not captures, 'One full-geometry capture per source')
                    report['full_geometry_renders'] += 1
                    n, instances = len(args[1]), int(result[0])
                    buffers = dict(zip(('geometry', 'binning', 'image'), result[10:13], strict=True))
                    addresses = {k: int(v.data_ptr()) for k, v in buffers.items()}
                    sizes = base.prefix_lengths(n, instances, w, h, addresses)
                    require(sum(sizes.values()) <= SPEC['per_source_prefix_limit'], 'Prefix limit exceeded')
                    copies = {k: v[:sizes[k]].cpu().numpy().copy() for k, v in buffers.items()}
                    parsed = replay.parse_buffers(**copies, point_count=n, num_rendered=instances,
                                                  width=w, height=h, addresses=addresses)
                    require(parsed['prefix_bytes'] == sizes, 'ABI prefix mismatch')
                    arrays = {k: base.save_array(directory/(k+'.npy'), v) for k, v in copies.items()}
                    values = {'all_map': args[8], 'background': args[0], 'viewmatrix': args[9],
                              'projmatrix': args[10], 'campos': args[24],
                              'raw_rgb': result[1].permute(1, 2, 0).contiguous(), 'median_depth': result[4].reshape(h, w)}
                    if args[2].numel():
                        values['colors_precomp'] = args[2]
                    for k, tensor in values.items():
                        arrays[k] = base.save_array(directory/(k+'.npy'), tensor.detach().cpu().numpy())
                    fx = np.float32(w)/(np.float32(2)*np.float32(args[18]))
                    fy = np.float32(h)/(np.float32(2)*np.float32(args[19]))
                    captures.append({'name': source['name'], 'target': source['target'], 'slot': source['slot'],
                                     'addresses': addresses, 'arrays': arrays, 'prefix_bytes': sizes,
                                     'original_buffer_bytes': {k: v.numel() for k, v in buffers.items()},
                                     'settings': {'width': w, 'height': h, 'point_count': n, 'num_rendered': instances,
                                          'focal': [float(fx), float(fy)], 'principal': [(w-1)*.5, (h-1)*.5],
                                          'render_geo': True, 'render_depth_only': False, 'buffer_length': 4}})
                with base.observing_call(_C, observe):
                    package = render(camera, field, SimpleNamespace(), pipe, options, background,
                                     learnt_normal=True, nb_src_frames=4, buffer_length=4, depth_error_threshold=.01,
                                     do_find_closest_frame=False, do_render_src_depth=False, render_geo=True,
                                     return_depth_normal=False)
                    depth = render_depth(camera, field, SimpleNamespace(), pipe, options, background,
                                         learnt_normal=True, nb_src_frames=4, buffer_length=4, depth_error_threshold=.01)
                require(len(captures) == 1, 'Exactly one full capture required')
                bank[index] = depth
                row = captures[0]
                row['arrays']['source_gate_depth'] = base.save_array(directory/'source_gate_depth.npy', bank[index].numpy()[0])
                # Exact author's FP32 CUDA expression with TF32 disabled; no new raster.
                row['arrays']['ref_to_src'] = base.save_array(directory/'ref_to_src.npy', transforms[source['name']].cpu().numpy())
                row['source_gate_depth_scope'] = 'Actual render_depth_only output times original radius-one-eroded shared validity'
                base.write(directory/'record.json', row)
                records.append({'name': source['name'], 'path': str(directory/'record.json'), 'sha256': base.sha(directory/'record.json')})
                report['saved_bytes'] += sum(v['bytes'] for v in row['arrays'].values())
                require(report['saved_bytes'] <= SPEC['output_byte_limit'], 'Output limit exceeded')
                bank.cache.clear(); del package, depth, captures, row, camera
                gc.collect()
        require(report['full_geometry_renders'] == report['depth_only_renders'] == 64
                and report['raster_calls'] == 128, 'Fixed render count mismatch')
        report['field_versions_unchanged'] = all(getattr(field, k)._version == v for k, v in versions.items())
        require(report['field_versions_unchanged'], 'Field mutation')
        base.write(Path(plan['output'])/'captures.json', {'records': records, 'eroded_valid': valid_record,
                                                        'original_valid': original_valid_record})
        report['captures_sha256'] = base.sha(Path(plan['output'])/'captures.json')
        report['actual_binary'] = {'path': str(Path(_C.__file__).resolve()), 'sha256': base.sha(_C.__file__)}
        report['actual_imports'] = {}
        for name, module in list(sys.modules.items()):
            path = getattr(module, '__file__', None)
            if path and (name.split('.')[0] in ('bridge_rgs', 'scene', 'utils', 'arguments', 'gaussian_renderer')
                         or name in ('layer_capture_helpers', 'fixed_ibgs_ray_replay')):
                path = Path(path).resolve(); relative = str(path.relative_to(snapshot))
                require(plan['source_hashes'].get(relative) == base.sha(path), 'Unbound actual import')
                report['actual_imports'][name] = {'path': str(path), 'sha256': base.sha(path)}
    finally:
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        torch.set_float32_matmul_precision(flags[0])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = flags[1:]
        report['numeric_flags_restored'] = flags == (torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
        report['hook_restored'] = _C.rasterize_gaussians is original


def median_support(table, prediction):
    n = len(table['center_xy']); center = table['candidate_center_index']
    output = {}
    for label in ('ideal', 'quantized'):
        supported = table['median_mask'] & prediction['geometric'] & (prediction[label] > 0)
        output[label] = np.bincount(center[supported], weights=table['w'][supported].astype(np.float64), minlength=n)
    return output


def source_identity_weights(records, tap_indices, ids):
    """Look for the exact Gaussian ID in full accepted RGB contributions, not top4."""
    tables = [{int(point): j for j, point in enumerate(row['ids'])} for row in records]
    values = {key: np.full(tap_indices.shape, np.nan, np.float64) for key in ('w', 'alpha', 'T', 'z')}
    present = np.zeros(tap_indices.shape, bool)
    for i, point in enumerate(ids):
        for k, tap in enumerate(tap_indices[i]):
            if tap < 0:
                continue
            position = tables[int(tap)].get(int(point))
            if position is None:
                values['w'][i, k] = values['alpha'][i, k] = 0
                continue
            present[i, k] = True
            for key, value in values.items():
                value[i, k] = float(records[int(tap)][key][position])
    return values, present


def describe_query(table, arrays, distribution):
    center = table['candidate_center_index']; output = {}
    for stratum, select in (('all', np.ones(len(center), bool)), ('narrow_front', table['narrow_front']),
                            ('omitted_narrow_front', table['narrow_front'] & ~table['median_mask']),
                            ('other', ~table['narrow_front'])):
        stable = select & ~arrays['GF_ambiguous'] & ~arrays['GM_ambiguous'][center]
        gm, gf = arrays['GM_quantized_gate'][center], arrays['GF_quantized_gate']
        interpretable = select & arrays['W_interpretable']
        w = arrays['W_quantized']
        item = {'candidate_source_queries': int(select.sum()),
                'geometric_valid': int((select & arrays['GF_geometric']).sum()),
                'source_depth_positive': int((select & arrays['GF_geometric'] & (arrays['GF_quantized'] > 0)).sum()),
                'all_positive_texture_taps_valid': int((select & arrays['all_positive_taps_valid']).sum()),
                'gate_ambiguous': int((select & (arrays['GF_ambiguous'] | arrays['GM_ambiguous'][center])).sum()),
                'summary_consistent_W_queries': int(interpretable.sum()),
                'same_id_positive_W': int((interpretable & (w > 0)).sum()),
                'GM_GF_stable_counts': {f'{int(m)}_{int(f)}': int((stable & (gm == m) & (gf == f)).sum())
                                       for m in (False, True) for f in (False, True)},
                'same_id_positive_W_by_GM_GF': {f'{int(m)}_{int(f)}': int((stable & interpretable & (w > 0)
                                                             & (gm == m) & (gf == f)).sum())
                                               for m in (False, True) for f in (False, True)},
                'W_quantized_consistent': distribution(w[interpretable]),
                'W_ideal_consistent': distribution(arrays['W_ideal'][interpretable]),
                'target_weight_all': distribution(table['w'][select].astype(np.float64))}
        output[stratum] = item
    return output


def analyze(plan, report, base):
    root = Path(plan['output']); producer = base.natural(root)
    require(producer['full_geometry_renders'] == producer['depth_only_renders'] == 64
            and producer['raster_calls'] == 128 and producer['captures_sha256'] == base.sha(root/'captures.json'),
            'Complete fixed source capture required')
    require(np.__version__ == plan['numpy_version'], 'Frozen NumPy version required')
    replay = base.library(plan['source_snapshot']); description = base.library(plan['source_snapshot'], 'ibgs_thin_summary')
    require(description.POLICY == plan['description_policy'], 'Thin descriptors changed')
    manifest = base.read(root/'captures.json')
    for key in ('eroded_valid', 'original_valid'):
        require(base.sha(manifest[key]['path']) == manifest[key]['sha256'], 'Source valid mask changed')
    valid = np.load(manifest['original_valid']['path'], allow_pickle=False)
    candidate_tables = {}
    targets = {v['camera']['name']: v for v in plan['views']}
    for name, view in targets.items():
        with np.load(view['candidates']['path'], allow_pickle=False) as file:
            candidate_tables[name] = {k: file[k] for k in file.files}
    directory = root/'queries'; directory.mkdir(); summaries = []
    report['saved_bytes'] = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    for record_item in manifest['records']:
        require(base.sha(record_item['path']) == record_item['sha256'], 'Source capture metadata changed')
        record = base.read(record_item['path']); settings = record['settings']
        arrays = {}
        for key, item in record['arrays'].items():
            require(base.sha(item['path']) == item['sha256'], 'Source captured array changed')
            arrays[key] = np.load(item['path'], mmap_mode='r', allow_pickle=False)
        table = candidate_tables[record['target']]; st = targets[record['target']]['settings']
        center = table['candidate_center_index']
        uv, source_z = project_query(table['center_xy'][center], table['z'], arrays['ref_to_src'],
                                    st['focal'], st['principal'])
        gf = depth_predicate(arrays['source_gate_depth'], uv, source_z)
        median_uv, median_z = project_query(table['center_xy'], table['center_median_depth'], arrays['ref_to_src'],
                                           st['focal'], st['principal'])
        gm = depth_predicate(arrays['source_gate_depth'], median_uv, median_z)
        support = median_support(table, gf)
        query = {'uv': uv, 'median_uv': median_uv,
                 'median_supported_weight_ideal': support['ideal'], 'median_supported_weight_quantized': support['quantized']}
        for prefix, predicate in (('GF', gf), ('GM', gm)):
            for key, value in predicate.items():
                query[prefix+'_'+key] = value
        # Preserve depth-only predicate separately, then add the original source
        # predicate's positive median-slot support condition.
        for label in ('ideal', 'quantized'):
            query['GM_'+label+'_depth_gate'] = gm[label+'_gate'].copy()
            query['GM_'+label+'_gate'] = gm[label+'_gate'] & (support[label] > 0)
        midpoint_support = np.bincount(center[table['median_mask'] & gf['quantization_midpoint']],
                                       minlength=len(table['center_xy'])) > 0
        query['GM_ambiguous'] = ((query['GM_ideal_gate'] != query['GM_quantized_gate'])
                                 | gm['quantization_midpoint'] | midpoint_support)
        # Only positive-weight taps influence interpolation. All four records
        # remain stored including duplicated clamped edge taps and zero weights.
        legal = gf['geometric']; taps = gf['taps']
        tap_index = np.full((len(center), 4), -1, np.int64)
        positions = np.flatnonzero(legal)
        xy, inverse = np.unique(taps[legal].reshape(-1, 2), axis=0, return_inverse=True)
        if len(positions):
            tap_index[positions] = inverse.reshape(-1, 4)
        parsed = replay.parse_buffers(arrays['geometry'], arrays['binning'], arrays['image'],
            point_count=settings['point_count'], num_rendered=settings['num_rendered'], width=settings['width'],
            height=settings['height'], addresses=record['addresses'])
        rows = replay.replay_rays(parsed, xy, arrays['all_map'], focal=settings['focal'], principal=settings['principal'],
            background=arrays['background'], raw_rgb=arrays['raw_rgb'], median_depth=arrays['median_depth'],
            colors_precomp=arrays.get('colors_precomp'), atol=SPEC['summary_atol'], rtol=SPEC['summary_rtol'],
            render_geo=True, render_depth_only=False, buffer_length=4)
        status = np.asarray([0 if row['comparison']['status'] == 'summary_consistent_reconstruction'
                             else 2 if not row['finite_cpu_arithmetic'] else 1 for row in rows], np.int8)
        identity, present = source_identity_weights(rows, tap_index, table['ids'])
        query['source_tap_indices'] = tap_index; query['same_id_present'] = present
        for key, value in identity.items():
            query['same_id_'+key] = value
        needed = (gf['ideal_weights'] > 0) | (gf['quantized_weights'] > 0)
        ok_taps = np.zeros(tap_index.shape, bool)
        nonnegative = tap_index >= 0
        ok_taps[nonnegative] = status[tap_index[nonnegative]] == 0
        query['W_interpretable'] = legal & np.all(ok_taps | ~needed, axis=1)
        query['all_positive_taps_valid'] = legal & np.all(valid[taps[:, :, 1], taps[:, :, 0]] | ~needed, axis=1)
        for label in ('ideal', 'quantized'):
            weights = gf[label+'_weights'].astype(np.float64)
            values = np.sum(np.where(weights > 0, np.nan_to_num(identity['w'], nan=0)*weights, 0), axis=1)
            values[~legal] = np.nan; query['W_'+label] = values
        trace = base.pack_traces(rows) if rows else {'offsets': np.zeros(1, np.int64), 'xy': xy}
        trace['summary_status'] = status
        for key in ('final_T', 'rgb', 'median_weight', 'median_depth', 'n_contrib', 'median_low', 'median_high'):
            trace['production_'+key] = np.asarray([r['production'][key] for r in rows])
            trace['reconstructed_'+key] = np.asarray([r['summary'][key] for r in rows])
        trace_item = save_npz(directory/(record['name']+'.taps.npz'), trace, base)
        query_item = save_npz(directory/(record['name']+'.queries.npz'), query, base)
        summary = {'name': record['name'], 'target': record['target'], 'source_slot': record['slot'],
                   'eligible_centers': len(table['center_xy']), 'candidate_count': len(center),
                   'unique_taps': len(rows), 'summary_consistent_taps': int((status == 0).sum()),
                   'mismatch_taps': int((status == 1).sum()), 'nonfinite_taps': int((status == 2).sum()),
                   'strata': describe_query(table, query, description.distribution),
                   'queries': query_item, 'traces': trace_item,
                   'source_capture': record_item, 'candidate_table': targets[record['target']]['candidates']}
        summaries.append(summary)
        report['sources_processed'] += 1; report['query_rows'] += len(center); report['unique_taps'] += len(rows)
        report['saved_bytes'] += trace_item['bytes']+query_item['bytes']
        require(report['saved_bytes'] <= SPEC['output_byte_limit'], 'Total capture plus analysis output limit exceeded')
        del arrays, parsed, rows, trace, query, gf, gm, identity
        gc.collect()
    totals = {key: sum(row[key] for row in summaries) for key in
              ('candidate_count', 'unique_taps', 'summary_consistent_taps', 'mismatch_taps', 'nonfinite_taps')}
    combined = {}
    for stratum in ('all', 'narrow_front', 'omitted_narrow_front', 'other'):
        items = [s['strata'][stratum] for s in summaries]
        counts = {key: sum(v[key] for v in items) for key in ('candidate_source_queries', 'geometric_valid',
                    'source_depth_positive', 'all_positive_texture_taps_valid', 'gate_ambiguous',
                    'summary_consistent_W_queries', 'same_id_positive_W')}
        for key in ('GM_GF_stable_counts', 'same_id_positive_W_by_GM_GF'):
            counts[key] = {bin_key: sum(v[key][bin_key] for v in items) for bin_key in items[0][key]}
        combined[stratum] = counts
    analysis = {'sources': summaries, 'totals': totals, 'strata': combined,
                'full_target_center_denominator': 8192, 'eligible_target_centers': 8131,
                'all_candidate_source_queries_retained': True, 'summary_tolerance': [SPEC['summary_atol'], SPEC['summary_rtol']],
                'predicate_scope': SPEC['predicate_scope'], 'exact_cuda_contribution_ledger': False,
                'inference_limit': 'G_M is reconstructed original source acceptance including median-slot support; '
                    'G_F is a counterfactual single-layer predicate. Same-ID source RGB weight is model evidence, '
                    'not physical cable, source color accuracy, or proof of an improved readout.',
                'scientific_or_adoption_gate': None}
    base.write(root/'analysis.json', base.safe_json(analysis))
    report['analysis_sha256'] = base.sha(root/'analysis.json')
    require(report['sources_processed'] == 64, 'Incomplete fixed source population')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--capture', action='store_true')
    mode.add_argument('--analyze', action='store_true')
    parser.add_argument('--output', type=Path); parser.add_argument('--plan', type=Path)
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        prepare(args); return
    payload = json.loads(args.plan.read_text()); base = old(payload['source_snapshot'])
    require(args.expected_plan_sha256 and base.sha(args.plan) == args.expected_plan_sha256, 'Explicit frozen plan SHA required')
    require(Path(__file__).resolve() == Path(payload['source_snapshot'])/'diagnose_ibgs_layer_sources.py', 'Frozen entry only')
    stage = 'capture' if args.capture else 'analyze'; root = Path(payload['output'])
    started = time.monotonic()
    base.write(root/(stage+'_started.json'), {'stage': stage, 'plan_sha256': args.expected_plan_sha256})
    report = {'status': 'failed', 'plan_sha256': args.expected_plan_sha256, 'stage': stage,
              'raster_calls': 0, 'full_geometry_renders': 0, 'depth_only_renders': 0,
              'source_rgb_decodes': 0, 'target_rgb_decodes': 0, 'semantic_decodes': 0,
              'backward': 0, 'optimizer_steps': 0, 'VAL_reads': 0, 'valid_members_loaded': 0,
              'sources_processed': 0, 'query_rows': 0, 'unique_taps': 0, 'saved_bytes': 0}
    def timeout(*_):
        raise TimeoutError('Fixed source-layer stage time limit')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(SPEC[stage+'_internal_seconds'])
    try:
        verify(payload, base)
        (capture if args.capture else analyze)(payload, report, base)
        if args.capture:
            require(report['numeric_flags_restored'] and report['hook_restored'], 'Capture restoration failed')
        verify(payload, base); report.update(status='completed', sources_inputs_unchanged=True)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        signal.alarm(0); report['elapsed_seconds'] = time.monotonic()-started
        filename = 'execution_receipt.json' if args.capture else 'analysis_execution_receipt.json'
        base.write(root/filename, base.safe_json(report))
    print(json.dumps(base.safe_json(report), allow_nan=False))


if __name__ == '__main__':
    main()
