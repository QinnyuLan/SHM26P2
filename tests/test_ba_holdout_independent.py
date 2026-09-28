"""Synthetic contracts for the independent auditor; no geometry cache is loaded."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
SCRIPT = HERE.with_name('audit_ba_holdout.py')
if not SCRIPT.exists():
    SCRIPT = HERE.parents[1] / 'scripts/audit_ba_holdout.py'
spec = importlib.util.spec_from_file_location('independent_ba_audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def cameras():
    poses = np.repeat(np.eye(4)[None], 6, axis=0)
    poses[:, 0, 3] = np.linspace(-1.5, 1.5, 6)
    point = np.array([.2, -.1, 6.])
    camera_points = point + poses[:, :3, 3]
    xy = camera_points[:, :2] / camera_points[:, 2, None]
    return poses, point, xy


def test_hash_ranks_are_protocol_strings_not_integer_order():
    ids = np.array([17, 8, 31, 2, 98, 12])
    expected = sorted(ids.tolist(), key=lambda t: hashlib.sha256(f'ba_track_holdout_v1:track:{t}'.encode()).hexdigest())
    assert audit.rank_tracks(ids).tolist() == expected
    support, score = audit.split_observations(987, ids)
    obs = sorted(range(6), key=lambda i: hashlib.sha256(f'ba_track_holdout_v1:obs:987:{ids[i]}'.encode()).hexdigest())
    assert support.tolist() == obs[:4] and score.tolist() == obs[4:]
    assert not set(support) & set(score)
    for n in (6, 7, 8, 9):
        left, right = audit.split_observations(3, np.arange(n))
        assert len(left) == int(np.ceil(2 * n / 3)) and len(left) + len(right) == n
    with pytest.raises(ValueError, match='unique'):
        audit.split_observations(3, [1, 1, 2])


def test_own_dlt_uses_only_support_and_native_focal_units():
    poses, point, xy = cameras()
    estimate = audit.independent_dlt(poses[:4], xy[:4])
    np.testing.assert_allclose(estimate, point, atol=1e-13)
    corrupted_score = xy.copy(); corrupted_score[4, 0] += .002
    # Score corruption does not enter support triangulation.
    np.testing.assert_array_equal(audit.independent_dlt(poses[:4], corrupted_score[:4]), estimate)
    errors, depth = audit.radial_errors(estimate, poses, corrupted_score, np.tile([1000., 700.], (6, 1)))
    assert errors[4] == pytest.approx(2.) and (depth > 0).all()
    assert audit.support_parallax(estimate, poses[:4]) > .5


def test_parallax_and_negative_depth_are_not_hidden_by_denominator_clamp():
    poses, point, xy = cameras()
    assert audit.support_parallax(point, np.repeat(poses[:1], 4, axis=0)) == pytest.approx(0., abs=2e-6)
    _, depth = audit.radial_errors(np.array([0., 0., -2.]), poses, xy, np.ones((6, 2)))
    assert (depth < 0).all()
    assert np.isnan(audit.independent_dlt(poses[:4], np.full((4, 2), np.nan))).all()


def test_paired_bootstrap_is_track_equal_and_common_draw():
    a = np.array([1., 3., 8.]); b = np.array([.9, 3.1, 6.])
    result = audit.paired_track_summary(a, b)
    samples = np.random.default_rng(20260927).integers(3, size=(2000, 3))
    expected = (b - a)[samples].mean(axis=1)
    np.testing.assert_allclose(result['difference_95_interval'], np.quantile(expected, [.025, .975]), rtol=0, atol=1e-15)
    assert result['candidate_minus_original'] == pytest.approx((b-a).mean())
    assert result['relative_reduction'] == pytest.approx(1-b.mean()/a.mean())
    assert audit.paired_track_summary([], [])['difference_95_interval'] is None
    with pytest.raises(ValueError, match='finite'):
        audit.paired_track_summary(a, [0., np.nan, 2.])


def test_fixed_track_population_has_no_parallax_backfill_or_six_view_fit_filter():
    ids = np.arange(10000, 16000, dtype=np.int64)
    counts = np.full(len(ids), 6)
    ranked_ids = audit.rank_tracks(ids)
    counts[ranked_ids[:100]-10000] = 3
    counts[ranked_ids[500:550]-10000] = 4
    offsets = np.r_[0, counts.cumsum()]
    arrays = {'point_track_ids': ids, 'observation_offsets': offsets,
              'image_id': np.concatenate([np.arange(n) for n in counts])}
    train = [{'name': f'{i:03}', 'split': 'train', 'image_id': i, 'camera_id': 0,
              'w2c_original': np.eye(4).tolist()} for i in range(350)]
    layout, _ = audit.independent_layout(arrays, {'views': train})
    np.testing.assert_array_equal(layout['fit_track_ids'], ranked_ids[:500])
    np.testing.assert_array_equal(layout['check_track_ids'], ranked_ids[550:5550])
    assert set(layout['fit_track_ids']).isdisjoint(layout['check_track_ids'])
    assert len(layout['check_track_ids']) == 5000
    assert set(layout['support_rows']).isdisjoint(layout['score_rows'])


def test_group_track_weighting_and_candidate_low_parallax_is_not_rejection():
    # Track 0 has three group-0 rows, track 1 one; group mean must be (1+9)/2, not 3.
    layout = {'score_track_indices': np.array([0, 0, 0, 1]), 'score_image_ids': np.array([1, 2, 3, 4]),
              'train_image_ids': np.arange(1, 5), 'train_group_ids': np.zeros(4, np.int64)}
    base = {'finite': np.ones(2, bool), 'positive': np.ones(2, bool),
            'support_parallax_degrees': np.ones(2), 'score_radial_error': np.array([1., 1., 1., 9.]),
            'per_track_mean_radial_error': np.array([1., 9.])}
    candidate = {k: v.copy() for k, v in base.items()}
    candidate['support_parallax_degrees'][:] = .01
    candidate['score_radial_error'] *= .9
    candidate['per_track_mean_radial_error'] *= .9
    arrays, report = audit.evaluation_report(base, candidate, layout)
    assert arrays['candidate_valid'].all() and report['fixed_denominator_usable']
    assert report['candidate_low_parallax_on_fixed_population_descriptive'] == 2
    assert report['by_name_group'][0]['baseline']['mean'] == 5.
    candidate['positive'][0] = False
    _, invalid = audit.evaluation_report(base, candidate, layout)
    assert invalid['baseline_eligible_tracks'] == 2
    assert invalid['candidate_equal_track']['mean'] is None
    assert invalid['by_name_group'][0]['candidate']['mean'] is None
    assert not invalid['fixed_denominator_usable']
    json.dumps(invalid, allow_nan=False)


def test_gate_separates_coverage_and_invalid_candidate_no_silent_removal():
    report = {'baseline_eligible_tracks': 1000,
              'by_view': [{'score_observations': 5} for _ in range(200)],
              'by_name_group': [{'score_observations': 100, 'difference': {'mean': -.03}} for _ in range(4)],
              'baseline_equal_track': {'mean': 1.}, 'candidate_equal_track': {'mean': .97},
              'fixed_denominator_usable': True}
    bootstrap = {'ci95': [-.04, -.02]}
    assert audit.gate_report(report, bootstrap, True)['passed']
    report['fixed_denominator_usable'] = False
    assert not audit.gate_report(report, bootstrap, True)['passed']
    report['fixed_denominator_usable'] = True
    report['by_name_group'][0]['score_observations'] = 99
    assert audit.gate_report(report, bootstrap, True)['status'] == 'inconclusive_coverage'


def test_incomplete_receipt_rejected_before_any_npz_read(tmp_path, monkeypatch):
    (tmp_path/'plan.json').write_text('{}')
    (tmp_path/'execution_receipt.json').write_text(json.dumps({'status': 'running'}))
    monkeypatch.setattr(audit.np, 'load', lambda *a, **k: pytest.fail('Incomplete run must not read arrays'))
    with pytest.raises(AssertionError, match='completed'):
        audit.completed_inputs(tmp_path, audit.file_sha(tmp_path/'plan.json'))


def test_reconstruction_actual_schema_has_native_projection_and_no_score_fitting():
    poses, point, xy = cameras()
    K = np.array([[1000., 0., 720.], [0., 700., 350.], [0., 0., 1.]])
    native = xy*np.array([1000., 700.])+np.array([720., 350.])
    native[5, 0] += 2.
    arrays = {'image_id': np.arange(6), 'undistorted_native_xy': native}
    layout = {'check_track_ids': np.array([77]), 'check_rows': np.arange(6),
              'check_offsets': np.array([0, 6]), 'support_rows': np.arange(4),
              'support_offsets': np.array([0, 4]), 'score_rows': np.array([4, 5]), 'score_offsets': np.array([0, 2])}
    train = [{'image_id': i, 'camera_id': 1} for i in range(6)]
    result = audit.reconstruct(arrays, layout, train, {'source_cameras': {'1': {'K': K.tolist()}}}, poses)
    np.testing.assert_allclose(result['points'][0], point, atol=1e-12)
    np.testing.assert_allclose(result['score_radial_error'], [0., 2.], atol=1e-10)
    assert result['per_track_mean_radial_error'][0] == pytest.approx(1.)
    assert result['finite'][0] and result['positive'][0]
