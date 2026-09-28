"""One fixed CPU-only TRAIN structural-axis diagnostic; never opens pixel targets."""
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
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
DOCUMENT = 'projective_structure_axes_protocol.md'
LEGACY_SOURCE = ROOT/'runs/h3_moments/02_cross/source_snapshot/bridge_rgs'
NAMES = tuple(f'{i:03d}.png' for i in (2, 21, 41, 59, 79, 100, 118, 137, 156, 176, 200, 220, 241, 259, 278, 300))
SPEC = {
    'protocol': 'projective_structure_axes_train_v1', 'classes': [1, 3],
    'minimum_votes': 3, 'minimum_purity': .9, 'minimum_observations': 3,
    'maximum_reprojection_rms_px': 3., 'neighbors': 16, 'minimum_linearity': .65,
    'blocks_per_dimension': 4, 'block_bounds_quantiles': [.02, .98],
    'blocks_from': 'all class candidates, then retain usable local lines',
    'axis_weighting': 'linearity normalized within each occupied usable block; equal blocks',
    'bootstrap_repeats': 256, 'bootstrap_seed': '20260927 + class_id',
    'camera_groups': 'floor(sorted TRAIN name index * 4 / 350)',
    'delete_group': 'drop whole point if any saved inlier observation touches group; rerun PCA/blocks',
    'stride': 16, 'minimum_projection_sine': .001, 'collinear_degrees': 10.,
    'offsets_native_px': np.linspace(-256, 256, 17).tolist(),
    'K_precision': 'manifest K -> float32 -> float64', 'R_precision': 'w2c_original float64',
    'global_constant': 'equal mean of 350 per-camera valid-grid dyad means',
    'per_camera_constant': 'principal direction of that camera valid-grid dyad mean',
    'wrong_pose': 'evaluation view rotation at (index+8)%16; keep own K',
    'sample_support': 'feature index [0,fw-1] x [0,fh-1]; include direction-valid gate',
    'pairwise_support': 'same axis/query/offset validity intersection for each pair of controls',
    'axis_gate': {'usable_points': 128, 'occupied_blocks': 8, 'relative_eigengap': .25,
                  'bootstrap_p95_angle_degrees': 10., 'leave_group_angle_degrees': 15.},
    'field_gate': {'valid_fraction_per_axis': .99, 'per_camera_median_angle_degrees': 5.,
                   'minimum_views_one_axis': 8, 'maximum_collinear_fraction': .25, 'collinear_denominator': 'both axes direction-valid centers'},
    'views': list(NAMES), 'internal_timeout_seconds': 110, 'external_timeout_seconds': 120,
    'gpu_calls': 0, 'new_pixel_payload_reads': 0, 'axis_retries': 0,
    'limitations': 'Conditional TRAIN GT-derived votes and SfM support; not OOF, physical 3D truth, or a performance test.'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, data):
    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    text = json.dumps(data, indent=2, allow_nan=False, default=default)+'\n'
    with Path(path).open('x') as stream:
        stream.write(text)


def source_files(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def camera_metadata(manifest):
    train = sorted((v for v in manifest['views'] if v['split'] == 'train'), key=lambda v: v['name'])
    require(len(train) == 350 and len({v['name'] for v in train}) == 350, 'Need unique 350 TRAIN views')
    require(len({v['image_id'] for v in train}) == 350, 'Duplicate camera image IDs')
    labeled = [v for v in train if v.get('mask_path')]
    require(len(labeled) == 259, 'Need the fixed labeled TRAIN population')
    require(tuple(labeled[i*258//15]['name'] for i in range(16)) == NAMES, 'Fixed 16 names changed')
    rows = []
    for index, v in enumerate(train):
        require(v['width'] == 1320 and v['height'] == 989, 'Expected original legacy native dimensions')
        K = np.asarray(v['K'], np.float32).astype(np.float64)
        pose = np.asarray(v['w2c_original'], np.float64)
        require(K.shape == (3, 3) and pose.shape == (4, 4)
                and np.isfinite(K).all() and np.isfinite(pose).all(), 'Malformed geometry')
        require(np.array_equal(pose, v['w2c']), 'Legacy poses must be unrefined originals')
        rows.append({'name': v['name'], 'image_id': int(v['image_id']), 'camera_id': int(v['camera_id']),
                     'width': v['width'], 'height': v['height'], 'K': K.tolist(),
                     'w2c_original': pose.tolist(), 'group': index*4//350, 'split': 'train'})
    return rows


def candidate_mask(arrays, class_id):
    counts = arrays['semantic_counts'].astype(np.float64)
    total = counts.sum(1)
    return ((counts.argmax(1) == class_id) & (total >= SPEC['minimum_votes'])
            & (counts.max(1)/np.maximum(total, 1) >= SPEC['minimum_purity'])
            & (arrays['num_observations'] >= SPEC['minimum_observations'])
            & (arrays['reprojection_error'] <= SPEC['maximum_reprojection_rms_px']))


def point_camera_groups(arrays, cameras):
    ids, offsets = arrays['observation_image_ids'], arrays['observation_offsets']
    lookup = {c['image_id']: c['group'] for c in cameras}
    require(set(ids.tolist()).issubset(lookup), 'Non-TRAIN saved observation')
    require(offsets.shape == (len(arrays['points'])+1,) and offsets[0] == 0
            and offsets[-1] == len(ids) and np.array_equal(np.diff(offsets), arrays['num_observations']),
            'Observation CSR is inconsistent')
    by_observation = np.array([lookup[int(i)] for i in ids], np.int8)
    touched = np.zeros((len(arrays['points']), 4), bool)
    rows = np.repeat(np.arange(len(touched)), np.diff(offsets))
    touched[rows, by_observation] = True
    return touched


def estimate_axis(points, indices, helper, bootstrap=False, class_id=1):
    points = np.asarray(points, np.float64)
    result = {'candidate_points': len(points), 'usable_points': 0, 'occupied_blocks': 0,
              'axis': None, 'relative_eigengap': None, 'bootstrap_p95_angle_degrees': None}
    evidence = {'candidate_indices': np.asarray(indices, np.int64)}
    if len(points) < SPEC['neighbors']:
        result['unavailable_reason'] = 'fewer_than_16_candidates'
        return result, evidence
    local = helper.local_line_evidence(points, neighbors=16, minimum_linearity=.65)
    block_ids, metadata = helper.spatial_blocks(points, bins=4)
    use = local['usable']
    result.update(usable_points=int(use.sum()), block_partition=metadata)
    evidence.update(usable_indices=np.asarray(indices)[use], candidate_block_ids=block_ids,
                    usable_directions=local['directions'][use], usable_linearity=local['linearity'][use])
    if not use.any():
        result['unavailable_reason'] = 'no_usable_local_lines'
        return result, evidence
    tensors, keys = helper.orientation_blocks(local['directions'][use], local['linearity'][use], block_ids[use])
    axis = helper.principal_axis(tensors)
    result.update(occupied_blocks=len(keys), axis=axis['axis'], eigenvalues=axis['eigenvalues'],
                  relative_eigengap=axis['relative_eigengap'])
    evidence.update(block_orientation_tensors=tensors, occupied_block_ids=keys)
    if bootstrap:
        boot = helper.bootstrap_axis(tensors, repeats=256, seed=20260927+class_id)
        result['bootstrap_p95_angle_degrees'] = boot['angle_95_percentile_degrees']
        evidence['bootstrap_axes'] = boot['axes']
    return result, evidence


def axis_gates(base, leaves):
    base_ok = (base['usable_points'] >= 128 and base['occupied_blocks'] >= 8
               and base['relative_eigengap'] is not None and base['relative_eigengap'] >= .25
               and base['bootstrap_p95_angle_degrees'] is not None
               and base['bootstrap_p95_angle_degrees'] <= 10.)
    leave_ok = [r['usable_points'] >= 128 and r['occupied_blocks'] >= 8
                and r.get('angle_to_base_degrees') is not None and r['angle_to_base_degrees'] <= 15.
                for r in leaves]
    return {'base': bool(base_ok), 'leave_groups': [bool(v) for v in leave_ok],
            'passed': bool(base_ok and all(leave_ok))}


def summary(values):
    values = np.asarray(values, np.float64).reshape(-1)
    return ({'count': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
             'p95': float(np.quantile(values, .95))} if len(values) else
            {'count': 0, 'mean': None, 'median': None, 'p95': None})


def equal_camera_constant(fields, helper):
    matrices = []
    for direction, valid in fields:
        vectors = direction[valid]
        if not len(vectors):
            return None
        matrices.append(vectors.T @ vectors / len(vectors))
    values, vectors = np.linalg.eigh(np.mean(matrices, axis=0))
    return {'direction': helper.canonical_axis(vectors[:, -1]),
            'dyad_eigenvalue_gap': float(values[-1]-values[0]), 'camera_count': len(matrices)}


def field_diagnostics(axes, cameras, helper):
    fields = {c: [] for c in (1, 3)}
    grids = [helper.feature_centers(v['width'], v['height'], 16) for v in cameras]
    for view, grid in zip(cameras, grids):
        for class_id in (1, 3):
            direction, valid, _ = helper.project_direction(view['K'], np.asarray(view['w2c_original'])[:3, :3],
                axes[class_id], grid, minimum_sine=.001)
            fields[class_id].append((direction, valid))
    constants = {c: equal_camera_constant(fields[c], helper) for c in (1, 3)}
    lookup = {v['name']: i for i, v in enumerate(cameras)}
    selected = [lookup[n] for n in NAMES]
    records, all_valid, all_collinear = [], {1: [], 3: []}, []
    all_angles = {c: {'global': [], 'per_camera': []} for c in (1, 3)}
    support_pool = {}
    for j, index in enumerate(selected):
        view, grid = cameras[index], grids[index]
        row = {'name': view['name'], 'feature_shape': list(grid.shape[:2]), 'axes': {}}
        direction_pair, valid_pair = [], []
        for class_id, fixed in ((1, [1., 0.]), (3, [0., 1.])):
            direction, valid = fields[class_id][index]
            direction_pair.append(direction); valid_pair.append(valid)
            all_valid[class_id].append(valid.reshape(-1))
            per_camera = helper.best_constant_direction(direction, valid)[0] if valid.any() else None
            global_direction = None if constants[class_id] is None else constants[class_id]['direction']
            angles = {}
            for label, constant in [('global', global_direction), ('per_camera', per_camera)]:
                a = helper.unoriented_degrees(direction[valid], constant) if constant is not None else np.array([])
                angles[label] = summary(a)
                all_angles[class_id][label].append(a)
            wrong = cameras[selected[(j+8)%16]]
            wd, wv, _ = helper.project_direction(view['K'], np.asarray(wrong['w2c_original'])[:3, :3],
                axes[class_id], grid, minimum_sine=.001)
            controls = {'horizontal_vertical': (np.broadcast_to(fixed, direction.shape), np.ones_like(valid)),
                        'projective': (direction, valid), 'wrong_pose': (wd, wv)}
            for label, constant in [('global', global_direction), ('per_camera', per_camera)]:
                controls[label] = ((np.broadcast_to(constant, direction.shape), np.ones_like(valid))
                                   if constant is not None else (np.zeros_like(direction), np.zeros_like(valid)))
            sample_masks = {}
            for label, (d, ok) in controls.items():
                _, supported = helper.line_samples(grid, d, SPEC['offsets_native_px'], view['width'], view['height'], 16)
                sample_masks[label] = supported & ok[..., None]
                support_pool.setdefault((class_id, label, label), []).append(int(sample_masks[label].sum()))
            pairs = {}
            for a in controls:
                pairs[a] = {}
                for b in controls:
                    common = sample_masks[a] & sample_masks[b]
                    pairs[a][b] = float(common.mean())
                    if a != b:
                        support_pool.setdefault((class_id, a, b), []).append(int(common.sum()))
            row['axes'][str(class_id)] = {'valid_fraction': float(valid.mean()), 'degenerate_fraction': float((~valid).mean()),
                'per_camera_constant': per_camera, 'angles_to_constant_degrees': angles,
                'sample_valid_fraction': {k: float(v.mean()) for k, v in sample_masks.items()},
                'pairwise_common_sample_valid_fraction': pairs,
                'sample_count_per_control': int(next(iter(sample_masks.values())).size)}
        both = valid_pair[0] & valid_pair[1]
        collinear = helper.unoriented_degrees(direction_pair[0][both], direction_pair[1][both]) < 10.
        # Report both denominators; the frozen gate uses only jointly valid centers.
        row['axes_collinear_fraction_all_centers'] = float(collinear.sum()/both.size)
        row['axes_collinear_fraction_common_valid'] = float(collinear.mean()) if both.any() else None
        row['both_axes_valid_fraction'] = float(both.mean())
        all_collinear.append((int(collinear.sum()), int(both.size), int(both.sum())))
        records.append(row)
    fractions = {str(c): float(np.concatenate(all_valid[c]).mean()) for c in (1, 3)}
    medians_ge5 = {str(c): sum(r['axes'][str(c)]['angles_to_constant_degrees']['per_camera']['median'] is not None
                    and r['axes'][str(c)]['angles_to_constant_degrees']['per_camera']['median'] >= 5. for r in records)
                   for c in (1, 3)}
    total = sum(v[1] for v in all_collinear)
    joint_count = sum(v[2] for v in all_collinear)
    collinear_all = sum(v[0] for v in all_collinear)/total
    collinear = sum(v[0] for v in all_collinear)/joint_count if joint_count else None
    gates = {'both_axis_valid_at_least_99percent': all(v >= .99 for v in fractions.values()),
             'one_axis_at_least_8_views_median_fanout_5degrees': max(medians_ge5.values()) >= 8,
             'collinear_at_most_25percent': collinear is not None and collinear <= .25}
    gates['passed'] = bool(all(gates.values()))
    sample_denominator = sum(r['axes']['1']['sample_count_per_control'] for r in records)
    return {'global_constants': constants, 'views': records, 'valid_fraction_per_axis': fractions,
            'views_median_fanout_at_least_5degrees': medians_ge5,
            'angles_to_constant_degrees_pooled': {str(c): {label: summary(np.concatenate(v))
                for label, v in all_angles[c].items()} for c in (1, 3)},
            'axes_collinear_fraction_all_centers': collinear_all,
            'axes_collinear_fraction_common_valid': collinear,
            'both_axes_valid_fraction': sum(v[2] for v in all_collinear)/total,
            'pooled_pairwise_sample_support': {str(c): {a: {b: sum(support_pool[c, a, b])/sample_denominator
                 for b in ('horizontal_vertical', 'global', 'per_camera', 'projective', 'wrong_pose')}
                 for a in ('horizontal_vertical', 'global', 'per_camera', 'projective', 'wrong_pose')}
                 for c in (1, 3)}, 'gates': gates}


def prepare(output):
    output = Path(output).resolve()
    require(not output.exists(), 'Fresh diagnostic output required')
    manifest_path = ROOT/'artifacts/prepared/manifest.json'
    manifest = read(manifest_path)
    require(manifest.get('pixel_protocol', 'legacy_mixed_v1') == 'legacy_mixed_v1', 'Need original legacy protocol')
    cameras = camera_metadata(manifest)
    forbidden = {str(Path(v[k]).resolve()) for v in manifest['views'] for k in
                 ('image_path', 'mask_path', 'valid_path', 'source_image_path', 'source_annotation_path') if v.get(k)}
    paths = [manifest_path, Path(manifest['init_points_path']), Path(manifest['geometry_audit_path']), ROOT/'uv.lock',
             *(LEGACY_SOURCE/name for name in ('prepare.py', 'data.py', 'geometry.py')),
             ROOT/'src/bridge_rgs/structure_axes.py', ROOT/'src/bridge_rgs/__init__.py', Path(__file__),
             ROOT/'tests/test_projective_structure_diagnostic.py', ROOT/'tests/test_structure_axes.py', ROOT/'docs'/DOCUMENT]
    require(not forbidden.intersection(str(p.resolve()) for p in paths), 'Forbidden pixel payload binding')
    require(all(p.is_file() for p in paths), 'Missing fixed provenance/source file')
    inputs = {str(p.resolve()): sha(p) for p in paths}
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'; (snapshot/'bridge_rgs').mkdir(parents=True)
    for name in ('structure_axes.py', '__init__.py'):
        shutil.copy2(ROOT/'src/bridge_rgs'/name, snapshot/'bridge_rgs'/name)
    for p in (Path(__file__), ROOT/'tests/test_projective_structure_diagnostic.py', ROOT/'tests/test_structure_axes.py', ROOT/'docs'/DOCUMENT):
        shutil.copy2(p, snapshot/p.name)
    plan = {'status': 'prepared_not_executed', 'specification': SPEC, 'output': str(output),
            'source_snapshot': str(snapshot), 'source_hashes': source_files(snapshot), 'input_hashes': inputs,
            'manifest': str(manifest_path), 'init_points': manifest['init_points_path'], 'cameras': cameras,
            'axis_estimation_during_prepare': False, 'prepare_pixel_payload_reads': 0,
            'runtime_versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy')},
            'external_timeout_seconds': 120,
            'env': {'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(snapshot),
                    'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'},
            'semantic_information': 'Existing semantic_counts incorporates TRAIN GT; no new pixel payload will be read.'}
    write_new(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    require(plan['specification'] == SPEC, 'Frozen specification differs')
    require(plan['runtime_versions'] == {name: importlib.metadata.version(name) for name in ('numpy', 'scipy')},
            'Numerical runtime version differs')
    require(source_files(plan['source_snapshot']) == plan['source_hashes'], 'Frozen source tree changed')
    for path, digest in plan['input_hashes'].items():
        require(sha(path) == digest, f'Bound input changed: {path}')
    require(camera_metadata(read(plan['manifest'])) == plan['cameras'], 'Camera metadata changed')


def execute(plan_path):
    plan_path = Path(plan_path).resolve(); plan = read(plan_path); output = Path(plan['output'])
    require(not (output/'execution_receipt.json').exists(), 'Never retry or overwrite execution')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU execution requires CUDA_VISIBLE_DEVICES empty')
    verify(plan)
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Execute frozen entrypoint')
    helper = importlib.import_module('bridge_rgs.structure_axes')
    require(Path(helper.__file__).resolve() == Path(plan['source_snapshot'])/'bridge_rgs/structure_axes.py', 'Wrong actual helper')
    def timed_out(signum, frame):
        raise TimeoutError('Fixed 110-second internal diagnostic budget exceeded')
    previous = signal.signal(signal.SIGALRM, timed_out); signal.alarm(110)
    started = time.perf_counter()
    report = {'status': 'running', 'plan_sha256': sha(plan_path), 'specification': SPEC,
              'started_utc': datetime.now(UTC).isoformat(), 'new_pixel_payload_reads': 0,
              'runtime_versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy')},
              'scene_renders': 0, 'backwards': 0, 'optimizer_steps': 0, 'gpu_calls': 0,
              'actual_imports': {'structure_axes': {'path': helper.__file__, 'sha256': sha(helper.__file__)}},
              'semantics_scope': 'Existing TRAIN GT-derived counts; no new masks/valid/RGB opened',
              'axis_scope': 'Conditional on fixed TRAIN track selection/calibration; support deletion is not OOF'}
    failure = None
    try:
        with np.load(plan['init_points'], allow_pickle=False) as z:
            keys = ('points', 'covariances', 'track_ids', 'reprojection_error', 'num_observations',
                    'observation_image_ids', 'observation_offsets', 'semantic_counts')
            arrays = {k: z[k] for k in keys}
        require(arrays['points'].shape == (60000, 3) and arrays['semantic_counts'].shape == (60000, 5), 'Wrong point source')
        require(all(np.isfinite(v).all() for v in arrays.values()), 'Nonfinite point metadata')
        require(np.all(arrays['semantic_counts'] >= 0), 'Negative cached votes')
        touched = point_camera_groups(arrays, plan['cameras'])
        classes, evidence, axes = {}, {}, {}
        for class_id in (1, 3):
            selected = candidate_mask(arrays, class_id); indices = np.flatnonzero(selected)
            base, saved = estimate_axis(arrays['points'][indices], indices, helper, True, class_id)
            evidence.update({f'class_{class_id}_base_{k}': v for k, v in saved.items()})
            leaves = []
            for group in range(4):
                kept = np.flatnonzero(selected & ~touched[:, group])
                row, saved = estimate_axis(arrays['points'][kept], kept, helper)
                row.update(group=group, removed_base_candidates=int(len(indices)-len(kept)),
                           angle_to_base_degrees=(float(helper.unoriented_degrees(row['axis'], base['axis']))
                               if row['axis'] is not None and base['axis'] is not None else None))
                leaves.append(row)
                evidence.update({f'class_{class_id}_drop_{group}_{k}': v for k, v in saved.items()})
            classes[str(class_id)] = {'base': base, 'leave_groups': leaves, 'gates': axis_gates(base, leaves)}
            axes[class_id] = base['axis']
        report['classes'] = classes
        report['fields'] = field_diagnostics(axes, plan['cameras'], helper) if all(v is not None for v in axes.values()) else None
        report['investment_gate_passed'] = bool(all(c['gates']['passed'] for c in classes.values())
            and report['fields'] is not None and report['fields']['gates']['passed'])
        coverage = all(r['usable_points'] >= 128 and r['occupied_blocks'] >= 8 and r['axis'] is not None
                       for c in classes.values() for r in [c['base'], *c['leave_groups']])
        coverage = coverage and report['fields'] is not None and report['fields']['axes_collinear_fraction_common_valid'] is not None
        report['scientific_status'] = ('inconclusive_coverage' if not coverage else
                                       'fixed_investment_conditions_met' if report['investment_gate_passed'] else 'not_supported')
        report['decision'] = ('eligible_for_separate_future_control_design' if report['investment_gate_passed']
                              else 'stop_current_two_global_axis_implementation')
        evidence_path = output/'axis_evidence.npz'
        require(not evidence_path.exists(), 'Evidence already exists')
        np.savez_compressed(evidence_path, **evidence)
        report['axis_evidence'] = {'path': str(evidence_path), 'sha256': sha(evidence_path),
                                  'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in evidence.items()}}
        report['status'] = 'completed'
    except BaseException as error:  # noqa: BLE001 - persist failure before re-raising
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        failure = error
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        try:
            verify(plan)
            report['bound_sources_inputs_unchanged'] = True
        except BaseException as error:  # noqa: BLE001 - persist failure before re-raising
            report['bound_sources_inputs_unchanged'] = False
            report.update(status='failed', verification_error=str(error))
            failure = failure or error
        report['elapsed_seconds'] = time.perf_counter()-started
        report['finished_utc'] = datetime.now(UTC).isoformat()
        report['torch_imported'] = 'torch' in sys.modules
        write_new(output/'execution_receipt.json', report)
    if failure is not None:
        raise failure
    return output/'execution_receipt.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path)
    mode.add_argument('--execute', type=Path)
    args = parser.parse_args()
    print(prepare(args.prepare) if args.prepare else execute(args.execute))


if __name__ == '__main__':
    main()
