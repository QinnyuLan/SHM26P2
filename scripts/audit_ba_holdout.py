"""Independent geometry-only audit of the fixed TRAIN bounded-BA holdout.

This file never imports the producer's ba_holdout implementation or a renderer.
The real-data entry point requires a completed execution receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def rank_tracks(track_ids):
    """Sort the original track IDs by the protocol's exact SHA256 byte order."""
    ids = np.asarray(track_ids, dtype=np.int64)
    if ids.ndim != 1 or len(set(ids.tolist())) != len(ids):
        raise ValueError('Track IDs must be a unique vector')
    return np.asarray(sorted(ids.tolist(), key=lambda t: (
        hashlib.sha256(f'ba_track_holdout_v1:track:{t}'.encode()).digest(), t)), dtype=np.int64)


def split_observations(track_id, image_ids):
    """Return support/score local row indices; order is the fixed observation hash."""
    ids = np.asarray(image_ids, dtype=np.int64)
    if ids.ndim != 1 or len(set(ids.tolist())) != len(ids):
        raise ValueError('Image IDs must be unique within a track')
    order = sorted(range(len(ids)), key=lambda i: (
        hashlib.sha256(f'ba_track_holdout_v1:obs:{int(track_id)}:{int(ids[i])}'.encode()).digest(), int(ids[i])))
    cut = (2 * len(ids) + 2) // 3
    return np.asarray(order[:cut], np.int64), np.asarray(order[cut:], np.int64)


def independent_dlt(poses, normalized_xy):
    """Homogeneous multiview DLT constructed row by row, in FP64."""
    poses, xy = np.asarray(poses, np.float64), np.asarray(normalized_xy, np.float64)
    if len(poses) < 2 or xy.shape != (len(poses), 2) or poses.shape not in ((len(poses), 3, 4), (len(poses), 4, 4)):
        raise ValueError('DLT needs at least two aligned cameras and observations')
    if not np.isfinite(poses).all() or not np.isfinite(xy).all():
        return np.full(3, np.nan)
    design = np.asarray([row for camera, (x, y) in zip(poses, xy, strict=True)
                         for row in (x * camera[2, :4] - camera[0, :4],
                                     y * camera[2, :4] - camera[1, :4])])
    _, _, right = np.linalg.svd(design, full_matrices=False)
    homogeneous = right[-1]
    if not np.isfinite(homogeneous).all() or abs(homogeneous[3]) < 1e-12:
        return np.full(3, np.nan)
    return homogeneous[:3] / homogeneous[3]


def radial_errors(point, poses, normalized_xy, focals):
    """Native undistorted radial errors, retaining the actual camera depths."""
    point, poses, xy, focal = (np.asarray(v, np.float64) for v in (point, poses, normalized_xy, focals))
    camera_points = np.asarray([camera[:3, :3] @ point + camera[:3, 3] for camera in poses])
    depth = camera_points[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        residual = (camera_points[:, :2] / depth[:, None] - xy) * focal
    return np.linalg.norm(residual, axis=1), depth


def support_parallax(point, support_poses):
    """Maximum angle of camera-center-to-DLT-point rays, in degrees."""
    point, poses = np.asarray(point, np.float64), np.asarray(support_poses, np.float64)
    centers = np.asarray([-camera[:3, :3].T @ camera[:3, 3] for camera in poses])
    direction = point - centers
    lengths = np.linalg.norm(direction, axis=1)
    if not np.isfinite(direction).all() or (lengths <= 0).any() or len(poses) < 2:
        return np.nan
    direction /= lengths[:, None]
    cosines = [np.dot(direction[i], direction[j]) for i in range(len(direction)) for j in range(i + 1, len(direction))]
    return float(np.rad2deg(np.arccos(np.clip(min(cosines), -1., 1.))))


def paired_track_summary(original, candidate, repeats=2000, seed=20260927):
    """Each supplied scalar is one track's mean score-observation radial error."""
    a, b = np.asarray(original, np.float64), np.asarray(candidate, np.float64)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Aligned finite track scores required')
    if (a < 0).any() or (b < 0).any():
        raise ValueError('Radial scores must be nonnegative')
    if not len(a):
        return {'tracks': 0, 'original_mean': None, 'candidate_mean': None,
                'candidate_minus_original': None, 'relative_reduction': None, 'difference_95_interval': None}
    difference = b - a
    draws = np.empty(repeats, np.float64)
    rng = np.random.default_rng(seed)
    for i in range(repeats):
        indices = rng.integers(0, len(a), len(a))
        draws[i] = np.mean(difference[indices])
    return {'tracks': len(a), 'original_mean': float(np.mean(a)), 'candidate_mean': float(np.mean(b)),
            'candidate_minus_original': float(np.mean(difference)),
            'relative_reduction': float(-np.mean(difference) / np.mean(a)) if np.mean(a) > 0 else None,
            'difference_95_interval': np.percentile(draws, [2.5, 97.5]).tolist()}


def independent_layout(arrays, manifest):
    ids, offsets, image_ids = (np.asarray(arrays[k], np.int64) for k in
                               ('point_track_ids', 'observation_offsets', 'image_id'))
    counts = np.diff(offsets)
    if len(offsets) != len(ids)+1 or offsets[0] != 0 or offsets[-1] != len(image_ids) or (counts < 3).any():
        raise ValueError('Invalid saved CSR tracks')
    index = {int(value): i for i, value in enumerate(ids)}
    ranked = np.array([index[int(value)] for value in rank_tracks(ids)], np.int64)
    fit = ranked[:500]
    check = np.array([i for i in ranked[500:] if counts[i] >= 6][:5000], np.int64)
    if len(fit) != 500 or len(check) != 5000:
        raise ValueError('Fixed fit/check population is unavailable')
    train = sorted((v for v in manifest['views'] if v['split'] == 'train'), key=lambda v: v['name'])
    if len(train) != 350 or len({v['image_id'] for v in train}) != 350:
        raise ValueError('Expected 350 unique TRAIN cameras')
    fit_parts, all_parts, support_parts, score_parts = [], [], [], []
    for i in fit:
        fit_parts.append(np.arange(offsets[i], offsets[i+1], dtype=np.int64))
    for i in check:
        rows = np.arange(offsets[i], offsets[i+1], dtype=np.int64)
        support, score = split_observations(ids[i], image_ids[rows])
        all_parts.append(rows); support_parts.append(rows[support]); score_parts.append(rows[score])

    def csr(parts):
        return np.r_[0, np.cumsum([len(p) for p in parts])].astype(np.int64)

    layout = {'track_hash_order_point_indices': ranked, 'fit_point_indices': fit, 'fit_track_ids': ids[fit],
              'fit_offsets': csr(fit_parts), 'fit_rows': np.concatenate(fit_parts),
              'check_point_indices': check, 'check_track_ids': ids[check],
              'check_offsets': csr(all_parts), 'check_rows': np.concatenate(all_parts),
              'support_offsets': csr(support_parts), 'support_rows': np.concatenate(support_parts),
              'score_offsets': csr(score_parts), 'score_rows': np.concatenate(score_parts),
              'score_track_indices': np.repeat(np.arange(5000), [len(x) for x in score_parts]),
              'check_observation_track_indices': np.repeat(np.arange(5000), [len(x) for x in all_parts]),
              'train_image_ids': np.array([v['image_id'] for v in train], np.int64),
              'train_camera_ids': np.array([v['camera_id'] for v in train], np.int64),
              'train_group_ids': np.arange(350, dtype=np.int64)*4//350,
              'baseline_poses': np.array([v['w2c_original'] for v in train], np.float64)}
    layout['score_image_ids'] = image_ids[layout['score_rows']]
    return layout, train


def reconstruct(arrays, layout, train, manifest, poses):
    """Independent support DLT followed by projection of all observations."""
    image_ids = arrays['image_id']
    native_xy = arrays['undistorted_native_xy']
    id_to_camera = {v['image_id']: i for i, v in enumerate(train)}
    intrinsics = np.array([manifest['source_cameras'][str(v['camera_id'])]['K'] for v in train], np.float64)
    camera_indices = np.array([id_to_camera[int(i)] for i in image_ids], np.int64)
    point_count, packed_count = len(layout['check_track_ids']), len(layout['check_rows'])
    result = {'points': np.full((point_count, 3), np.nan), 'check_uv': np.full((packed_count, 2), np.nan),
                  'check_depth': np.full(packed_count, np.nan), 'check_residual_uv': np.full((packed_count, 2), np.nan),
                  'support_parallax_degrees': np.full(point_count, np.nan), 'finite': np.zeros(point_count, bool),
                  'positive': np.zeros(point_count, bool)}
    for i in range(point_count):
        support = layout['support_rows'][slice(*layout['support_offsets'][i:i+2])]
        cameras = camera_indices[support]
        K = intrinsics[cameras]
        normalized = (native_xy[support]-K[:, :2, 2])/K[:, [0, 1], [0, 1]]
        try:
            point = independent_dlt(poses[cameras], normalized)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(point).all():
            continue
        result['points'][i] = point
        sl = slice(*layout['check_offsets'][i:i+2])
        rows = layout['check_rows'][sl]
        ci = camera_indices[rows]
        pc = np.array([poses[j, :3, :3]@point+poses[j, :3, 3] for j in ci])
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            uv = pc[:, :2]/pc[:, 2, None]*intrinsics[ci][:, [0, 1], [0, 1]]+intrinsics[ci, :2, 2]
        result['check_uv'][sl] = uv
        result['check_depth'][sl] = pc[:, 2]
        result['check_residual_uv'][sl] = uv-native_xy[rows]
        result['finite'][i] = np.isfinite(uv).all() and np.isfinite(pc[:, 2]).all()
        result['positive'][i] = bool((pc[:, 2] > 0).all())
        result['support_parallax_degrees'][i] = support_parallax(point, poses[cameras])
    packed_lookup = {int(row): j for j, row in enumerate(layout['check_rows'])}
    score_idx = np.array([packed_lookup[int(r)] for r in layout['score_rows']])
    result['score_radial_error'] = np.linalg.norm(result['check_residual_uv'][score_idx], axis=1)
    result['per_track_mean_radial_error'] = np.array([
        np.mean(result['score_radial_error'][slice(*layout['score_offsets'][i:i+2])])
        for i in range(point_count)])
    return result


def describe(values):
    values = np.asarray(values, np.float64)
    if not len(values) or not np.isfinite(values).all():
        return {'count': len(values), 'mean': None, 'median': None, 'rms': None, 'p95': None}
    return {'count': len(values), 'mean': float(np.mean(values)), 'median': float(np.median(values)),
                'rms': float(np.sqrt(np.dot(values, values)/len(values))), 'p95': float(np.quantile(values, .95))}


def grouped_track_values(values, track_indices, selection):
    """Within-group observations are averaged within track before averaging tracks."""
    selected_tracks = np.unique(track_indices[selection])
    return np.asarray([np.mean(values[selection & (track_indices == track)]) for track in selected_tracks])


def independent_gate(eligible, candidate_valid, score_tracks, score_images, train_ids, groups,
                     original_scores, candidate_scores, accepted):
    use = eligible[score_tracks]
    counts = np.array([np.count_nonzero(use & (score_images == i)) for i in train_ids])
    group_observations, group_differences = [], []
    for group in range(4):
        selected = use & np.isin(score_images, train_ids[groups == group])
        group_observations.append(int(selected.sum()))
        before = grouped_track_values(original_scores, score_tracks, selected)
        effective_candidate = np.where(candidate_valid[score_tracks], candidate_scores, np.nan)
        after = grouped_track_values(effective_candidate, score_tracks, selected)
        group_differences.append(float((after-before).mean()) if len(before) and np.isfinite(after).all() else None)
    valid = bool(eligible.any() and candidate_valid[eligible].all())
    before = grouped_track_values(original_scores, score_tracks, use)
    after = grouped_track_values(candidate_scores, score_tracks, use)
    paired = paired_track_summary(before, after) if valid else None
    tests = {
        'minimum_eligible_tracks': int(eligible.sum()) >= 1000,
        'minimum_supported_score_cameras': int((counts >= 5).sum()) >= 200,
        'minimum_each_group_score_observations': min(group_observations) >= 100,
        'ba_accepted': bool(accepted), 'all_candidate_valid': valid,
        'relative_reduction_at_least_one_percent': bool(paired and paired['relative_reduction'] is not None
                                                       and paired['relative_reduction'] >= .01),
        'paired_upper_below_zero': bool(paired and paired['difference_95_interval'][1] < 0),
        'at_least_three_groups_improve': sum(x is not None and x < 0 for x in group_differences) >= 3,
    }
    return {'checks': tests, 'passed': all(tests.values()), 'paired': paired,
            'score_cameras_with_five_rows': int((counts >= 5).sum()),
            'group_score_observations': group_observations, 'group_track_mean_differences': group_differences}


def evaluation_report(base, candidate, layout):
    eligible = base['finite'] & base['positive'] & (base['support_parallax_degrees'] >= .5)
    valid = candidate['finite'] & candidate['positive']
    bad = eligible & ~valid
    out = {f'{prefix}_{key}': val for prefix, data in [('baseline', base), ('candidate', candidate)]
           for key, val in data.items()}
    out.update(baseline_eligible=eligible, candidate_valid=valid, candidate_bad_on_fixed_population=bad,
               per_track_difference=candidate['per_track_mean_radial_error']-base['per_track_mean_radial_error'])
    report = {'selected_check_tracks': len(eligible), 'baseline_eligible_tracks': int(eligible.sum()),
              'baseline_rejections': {'nonfinite': int((~base['finite']).sum()),
                 'nonpositive_depth': int((base['finite'] & ~base['positive']).sum()),
                 'insufficient_parallax': int((base['finite'] & base['positive'] &
                                              ~(base['support_parallax_degrees'] >= .5)).sum())},
              'candidate_bad_tracks_on_fixed_population': int(bad.sum()),
              'candidate_low_parallax_on_fixed_population_descriptive': int((eligible &
                                                 (candidate['support_parallax_degrees'] < .5)).sum()),
              'fixed_denominator_usable': bool(eligible.any() and not bad.any()),
              'baseline_equal_track': describe(base['per_track_mean_radial_error'][eligible]),
              'candidate_equal_track': describe(np.where(valid[eligible],
                                     candidate['per_track_mean_radial_error'][eligible], np.nan)),
              'by_view': [], 'by_name_group': []}
    track, score_ids = layout['score_track_indices'], layout['score_image_ids']
    keep = eligible[track]
    new_score = np.where(bad[track], np.nan, candidate['score_radial_error'])
    out['group_track_score_counts'] = np.zeros((4, len(eligible)), np.int64)
    out['baseline_group_per_track_error'] = np.full((4, len(eligible)), np.nan)
    out['candidate_group_per_track_error'] = np.full((4, len(eligible)), np.nan)
    for g in range(4):
        mask = keep & np.isin(score_ids, layout['train_image_ids'][layout['train_group_ids'] == g])
        group_tracks = np.unique(track[mask])
        for t in group_tracks:
            rows = mask & (track == t)
            out['group_track_score_counts'][g, t] = rows.sum()
            out['baseline_group_per_track_error'][g, t] = base['score_radial_error'][rows].mean()
            out['candidate_group_per_track_error'][g, t] = new_score[rows].mean()
        before = out['baseline_group_per_track_error'][g, group_tracks]
        after = out['candidate_group_per_track_error'][g, group_tracks]
        report['by_name_group'].append({'group': g, 'score_observations': int(mask.sum()),
                  'tracks': len(group_tracks), 'candidate_bad_tracks': int(bad[group_tracks].sum()),
                  'baseline': describe(before), 'candidate': describe(after), 'difference': describe(after-before)})
    for image in layout['train_image_ids']:
        mask = keep & (score_ids == image)
        before, after = base['score_radial_error'][mask], new_score[mask]
        report['by_view'].append({'image_id': int(image), 'score_observations': int(mask.sum()),
                   'candidate_bad_observations': int(bad[track[mask]].sum()), 'baseline': describe(before),
                   'candidate': describe(after), 'difference': describe(after-before)})
    return out, report


def gate_report(report, bootstrap, accepted):
    coverage = {'eligible_tracks_ge_1000': report['baseline_eligible_tracks'] >= 1000,
                'score_cameras_ge_200_each_ge_5_rows': sum(v['score_observations'] >= 5 for v in report['by_view']) >= 200,
                'all_four_groups_ge_100_score_rows': all(v['score_observations'] >= 100 for v in report['by_name_group'])}
    old, new = report['baseline_equal_track']['mean'], report['candidate_equal_track']['mean']
    relative = (old-new)/old if old is not None and old > 0 and new is not None else None
    count = sum(v['difference']['mean'] is not None and v['difference']['mean'] < 0 for v in report['by_name_group'])
    signal = {'ba_accepted': bool(accepted), 'all_candidates_valid_on_fixed_population': report['fixed_denominator_usable'],
              'relative_mean_reduction_ge_one_percent': relative is not None and relative >= .01,
              'paired_ci_upper_negative': bootstrap is not None and bootstrap['ci95'][1] < 0,
              'at_least_three_groups_improve': count >= 3}
    cov, sig = all(coverage.values()), all(signal.values())
    return {'status': 'supported_heldout_reprojection' if cov and sig else ('inconclusive_coverage' if not cov else 'not_supported'),
                'passed': cov and sig, 'coverage_passed': cov, 'signal_passed': sig, 'coverage': coverage, 'signal': signal,
                'relative_mean_reduction': relative, 'improving_groups': count,
                'decision': 'consider_matched_reference' if cov and sig else 'stop'}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def completed_inputs(folder, expected_plan_sha256):
    """No NPZ arrays may be read before this completed-receipt/source/input guard."""
    plan_path, receipt_path = folder/'plan.json', folder/'execution_receipt.json'
    ensure(file_sha(plan_path) == expected_plan_sha256, 'Unexpected plan SHA')
    plan, receipt = read_json(plan_path), read_json(receipt_path)
    ensure(receipt['status'] == 'completed' and receipt['plan_sha256'] == expected_plan_sha256,
           'A completed bound receipt is required')
    ensure(receipt['inputs_and_sources_unchanged'] is True, 'Producer inputs/source changed')
    for key in ('gpu_calls', 'new_pixel_decodes', 'label_array_loads', 'rgb_array_loads', 'model_renders'):
        ensure(receipt[key] == 0, f'Unexpected activity: {key}')
    snapshot = Path(plan['source_snapshot'])
    ensure(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Use the frozen independent checker')
    bound = {str(snapshot/key): value for key, value in plan['source_hashes'].items()}
    actual_names = {str(p.relative_to(snapshot)) for p in snapshot.rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    ensure(actual_names == set(plan['source_hashes']), 'Source file set changed')
    bound.update(plan['input_hashes'])
    bound.update({str(folder/key): value for key, value in receipt['output_files'].items()})
    bound[str(plan_path)] = expected_plan_sha256
    bound[str(receipt_path)] = file_sha(receipt_path)
    for path, value in bound.items():
        ensure(file_sha(path) == value, f'Changed bound file: {path}')
    return plan, receipt, bound


class Comparison:
    def __init__(self):
        self.array_errors, self.scalar_max_abs, self.scalar_count = {}, 0., 0

    def array(self, actual, expected, name, exact=False):
        a, b = np.asarray(actual), np.asarray(expected)
        ensure(a.shape == b.shape, f'Shape differs: {name}')
        if exact or a.dtype.kind in 'biu':
            ensure(np.array_equal(a, b, equal_nan=True), f'Exact array differs: {name}')
            self.array_errors[name] = 0.
        else:
            ensure(np.array_equal(np.isfinite(a), np.isfinite(b)), f'Finite support differs: {name}')
            ensure(np.allclose(a, b, rtol=2e-8, atol=2e-8, equal_nan=True), f'Numerical array differs: {name}')
            finite = np.isfinite(a) & np.isfinite(b)
            self.array_errors[name] = float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.

    def values(self, actual, expected, path='report'):
        if isinstance(expected, dict):
            for key, value in expected.items():
                ensure(key in actual, f'Missing report key: {path}.{key}')
                self.values(actual[key], value, f'{path}.{key}')
        elif isinstance(expected, (list, tuple)):
            ensure(len(actual) == len(expected), f'List length differs: {path}')
            for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
                self.values(a, b, f'{path}[{index}]')
        elif isinstance(expected, float):
            ensure(actual is not None and np.isclose(actual, expected, rtol=2e-8, atol=2e-8), f'Value differs: {path}: {actual} vs {expected}')
            self.scalar_count += 1
            self.scalar_max_abs = max(self.scalar_max_abs, abs(actual-expected))
        else:
            ensure(actual == expected, f'Value differs: {path}: {actual} vs {expected}')
            self.scalar_count += 1


def audit_run(folder, expected_plan_sha256):
    folder = Path(folder).resolve()
    output = folder/'independent_cpu_review.json'
    ensure(not output.exists(), 'Do not overwrite an independent review')
    start = time.perf_counter()
    plan, _receipt, bindings = completed_inputs(folder, expected_plan_sha256)
    check = Comparison()
    manifest = read_json(plan['manifest'])
    geometry_keys = ('point_positions', 'point_track_ids', 'observation_offsets', 'image_id',
                     'raw_xy', 'undistorted_native_xy', 'saved_track_rms')
    with np.load(folder/'geometry_input.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in geometry_keys}
        layout, train = independent_layout(arrays, manifest)
        ensure(set(saved.files) == set(geometry_keys) | {'layout_'+k for k in layout}, 'Unexpected geometry ledger keys')
        for key, value in layout.items():
            check.array(saved['layout_'+key], value, 'layout.'+key, exact=True)
    with np.load(plan['observations'], allow_pickle=False) as original:
        for key in geometry_keys:
            check.array(arrays[key], original[key], 'input_geometry.'+key, exact=True)
    ensure(len(arrays['point_track_ids']) == 60000 and len(arrays['image_id']) == 457102, 'Unexpected saved population')
    for i in range(60000):
        row = arrays['image_id'][slice(*arrays['observation_offsets'][i:i+2])]
        ensure((np.diff(row) > 0).all(), 'Repeated or unsorted image ID in track')
    with np.load(folder/'candidate_poses.npz', allow_pickle=False) as archive:
        pose_arrays = {key: archive[key] for key in archive.files}
    check.array(pose_arrays['image_id'], layout['train_image_ids'], 'camera.image_id', exact=True)
    check.array(pose_arrays['original'], layout['baseline_poses'], 'camera.original', exact=True)
    poses = pose_arrays['candidate']
    ensure(poses.shape == (350, 4, 4) and np.isfinite(poses).all(), 'Invalid candidate camera array')
    ensure(np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-12, rtol=0), 'Non-homogeneous camera')
    ensure(np.allclose(poses[:, :3, :3] @ poses[:, :3, :3].transpose(0, 2, 1), np.eye(3), atol=1e-8, rtol=0), 'Non-orthogonal camera')
    ensure(np.allclose(np.linalg.det(poses[:, :3, :3]), 1., atol=1e-8, rtol=0), 'Improper camera rotation')
    fit = read_json(folder/'fit_result.json')
    ensure(fit['plan_sha256'] == expected_plan_sha256, 'Fit source differs')
    ensure(fit['fit_track_ids'] == layout['fit_track_ids'].tolist() and fit['check_tracks_used'] == 0,
           'Fit/check isolation differs')
    ensure(fit['selected_points'] == 500 and fit['train_observations'] == len(layout['fit_rows']) and fit['nfev'] <= 20,
           'Actual BA budget differs')
    fit_ids = np.unique(arrays['image_id'][layout['fit_rows']])
    image_index = {int(value): i for i, value in enumerate(layout['train_image_ids'])}
    # The unchanged solver exports initial fit XYZ, so its BEFORE coordinate RMS
    # is independently checkable. Its optimized XYZ/AFTER residual are not.
    fit_residuals = []
    for pi in layout['fit_point_indices']:
        for row in range(*arrays['observation_offsets'][pi:pi+2]):
            v = train[image_index[int(arrays['image_id'][row])]]
            pose = np.asarray(v['w2c_original'], np.float64)
            ensure(np.array_equal(pose, np.asarray(v['w2c'], np.float64)), 'Manifest camera changed from original')
            K = np.asarray(manifest['source_cameras'][str(v['camera_id'])]['K'], np.float64)
            point = np.asarray(arrays['point_positions'][pi], np.float64)
            pc = pose[:3, :3]@point+pose[:3, 3]
            xy = (arrays['undistorted_native_xy'][row]-K[:2, 2])/K.diagonal()[:2]
            fit_residuals.extend(((pc[:2]/max(float(pc[2]), 1e-5)-xy)*K.diagonal()[:2]).tolist())
    check.values(fit['rms_native_px_before'], float(np.sqrt(np.mean(np.square(fit_residuals)))), 'fit.before_rms')
    centers = np.array([-layout['baseline_poses'][image_index[int(i)], :3, :3].T @
                        layout['baseline_poses'][image_index[int(i)], :3, 3] for i in fit_ids])
    med = np.median(centers, axis=0)
    anchor0 = int(np.argmax(np.linalg.norm(centers-med, axis=1)))
    anchor1 = int(np.argmax(np.linalg.norm(centers-centers[anchor0], axis=1)))
    anchors = {int(fit_ids[anchor0]), int(fit_ids[anchor1])}
    ensure(set(fit['fixed_anchor_image_ids']) == anchors, 'Anchor selection differs')
    scale = max(float(np.percentile(np.linalg.norm(centers-med, axis=1), 90)), 1e-3)
    bounds = np.array([.01]*3+[.01*scale]*3)
    check.values(fit['camera_delta_bounds'], bounds.tolist(), 'fit.camera_delta_bounds')
    check.values(fit['camera_prior_std'], (bounds/3).tolist(), 'fit.camera_prior_std')
    optimized = set(map(int, fit_ids))-anchors
    ensure(fit['optimized_cameras'] == len(optimized), 'Optimized camera count differs')
    ensure(set(map(int, pose_arrays['override_image_id'])) == (optimized if fit['accepted'] else set()), 'Returned overrides differ')
    if fit['accepted']:
        ensure(np.isfinite(fit['rms_native_px_after']) and fit['rms_native_px_after'] <= fit['rms_native_px_before'],
               'BA accepted without the original RMS condition')
    changes = []
    for i, v in enumerate(train):
        old, new = layout['baseline_poses'][i], poses[i]
        relative = new @ np.linalg.inv(old)
        rot = Rotation.from_matrix(relative[:3, :3]).as_rotvec()
        displacement = np.linalg.norm(-new[:3, :3].T@new[:3, 3]+old[:3, :3].T@old[:3, 3])
        if v['image_id'] not in optimized or not fit['accepted']:
            ensure(np.array_equal(old, new), 'Anchor/non-fit/unaccepted camera changed')
        ensure(np.all(np.abs(np.r_[rot, relative[:3, 3]]) <= bounds+1e-10), 'Camera box bound exceeded')
        changes.append({'image_id': v['image_id'], 'name': v['name'], 'anchor': v['image_id'] in anchors,
                            'pose_exact_original': bool(np.array_equal(old, new)), 'left_rotation_vector': rot.tolist(),
                            'rotation_radians': float(np.linalg.norm(rot)), 'left_translation': relative[:3, 3].tolist(),
                            'camera_center_displacement': float(displacement)})
    base = reconstruct(arrays, layout, train, manifest, layout['baseline_poses'])
    candidate = reconstruct(arrays, layout, train, manifest, poses)
    expected, report = evaluation_report(base, candidate, layout)
    expected['candidate_poses'] = poses
    baseline_expected, baseline_report = evaluation_report(base, base, layout)
    baseline_expected['candidate_poses'] = layout['baseline_poses']
    for rows in (report['by_view'], baseline_report['by_view']):
        for row, view in zip(rows, train, strict=True):
            row['name'] = view['name']
    with np.load(folder/'holdout_arrays.npz', allow_pickle=False) as actual, np.load(folder/'baseline_screen.npz', allow_pickle=False) as baseline_screen:
        ensure(set(actual.files) == set(expected) and set(baseline_screen.files) == set(baseline_expected), 'Unexpected evaluation array schema')
        for key in expected:
            check.array(actual[key], expected[key], 'evaluation.'+key)
            check.array(baseline_screen[key], baseline_expected[key], 'baseline_screen.'+key)
            if key.startswith('baseline_'):
                check.array(actual[key], baseline_screen[key], 'baseline_frozen.'+key, exact=True)
        for key in base:
            check.array(baseline_screen['baseline_'+key], baseline_screen['candidate_'+key], 'baseline_identity.'+key, exact=True)
    check.values(read_json(folder/'baseline_screen.json'), baseline_report, 'baseline_report')
    analysis = read_json(folder/'analysis.json')
    ensure(analysis['plan_sha256'] == expected_plan_sha256, 'Analysis is from another plan')
    check.values(analysis['specification'], plan['analysis_specification'], 'analysis.specification')
    check.values(analysis['fit'], {k: v for k, v in fit.items() if k not in ('elapsed_seconds', 'plan_sha256')}, 'analysis.fit')
    counts = np.diff(arrays['observation_offsets'])
    selection = {'train_images': [{'image_id': v['image_id'], 'name': v['name']} for v in train],
                 'point_count': 60000, 'observation_count': 457102, 'fit_tracks': 500, 'check_tracks': 5000,
                 'remaining_hash_rank_tracks_with_fewer_than_six_observations': int((counts[layout['track_hash_order_point_indices'][500:]] < 6).sum()),
                 'check_support_observations': len(layout['support_rows']),
                 'check_score_observations': len(layout['score_rows']),
                 'selection_uses_labels_or_reprojection_errors': False}
    check.values(analysis['selection'], selection, 'analysis.selection')
    check.values(analysis['heldout'], report, 'analysis.heldout')
    check.values(analysis['pose_changes'], changes, 'analysis.pose_changes')
    use = expected['baseline_eligible']
    paired = paired_track_summary(base['per_track_mean_radial_error'][use], candidate['per_track_mean_radial_error'][use]) if report['fixed_denominator_usable'] else None
    bootstrap = None if paired is None else {'repeats': 2000, 'seed': 20260927, 'resampling_unit': 'baseline-eligible track',
                            'mean_difference': paired['candidate_minus_original'], 'ci95': paired['difference_95_interval']}
    check.values(analysis['paired_bootstrap'], bootstrap, 'analysis.paired_bootstrap')
    gate = gate_report(report, bootstrap, fit['accepted'])
    check.values(analysis['gate'], gate, 'analysis.gate')
    coverage = {'fit_observed_cameras': len(fit_ids), 'optimized_cameras_solver_report': len(optimized),
                'score_cameras': sum(v['score_observations'] > 0 for v in report['by_view']),
                'score_cameras_with_nonzero_pose_change': sum(v['score_observations'] > 0 and not c['pose_exact_original']
                           for v, c in zip(report['by_view'], changes, strict=True)),
                'pose_changes_are_not_optimized_parameter_count': True}
    check.values(analysis['camera_coverage_descriptive'], coverage, 'analysis.camera_coverage')
    for path, value in bindings.items():
        ensure(file_sha(path) == value, f'Bound input/source/output changed during audit: {path}')
    result = {'status': 'passed', 'plan_sha256': expected_plan_sha256, 'checker_sha256': file_sha(__file__),
              'execution_receipt_sha256': file_sha(folder/'execution_receipt.json'),
              'source_input_output_bindings_before_after': len(bindings), 'source_input_output_hashes_exact': True,
              'independent_implementation': 'NumPy/SciPy own SHA split, DLT SVD, native projection, group/track means and bootstrap; no ba_holdout import',
              'array_max_abs_errors': check.array_errors, 'scalar_checks': check.scalar_count,
              'scalar_max_abs_difference': check.scalar_max_abs, 'gate': gate, 'paired': paired,
              'camera_coverage': coverage, 'support_dlt_point_displacement': describe(np.linalg.norm(candidate['points'][use]-base['points'][use], axis=1)),
              'fit_optimized_points_independently_verified': False,
              'fit_scope': 'Fit XYZ are not exported by the unchanged solver; accepted/RMS are bound solver metadata, not reconstructed fit residuals.',
              'label_or_image_decode_calls': 0, 'gpu_calls': 0, 'optimizer_calls': 0,
              'elapsed_seconds': time.perf_counter()-start}
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    result = audit_run(args.run, args.expected_plan_sha256)
    print(json.dumps({'status': result['status'], 'gate': result['gate'], 'paired': result['paired'],
                      'output': str(Path(args.run)/'independent_cpu_review.json')}, allow_nan=False))


if __name__ == '__main__':
    main()
