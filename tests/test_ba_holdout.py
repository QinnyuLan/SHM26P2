"""Synthetic only: no actual bridge observations, labels, models, or GPU."""
import copy
import hashlib
import json

import numpy as np
import pytest

from bridge_rgs import ba_holdout as ba


def hash_order(ids):
    return sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        f'ba_track_holdout_v1:track:{ids[i]}'.encode()).digest())


@pytest.fixture(scope='module')
def fixture_data():
    n = 5500
    ids = np.arange(10001, 10001 + n, dtype=np.int64)
    counts = 6 + np.arange(n) % 3
    # Fit tracks intentionally include short tracks; they must not be skipped.
    counts[np.array(hash_order(ids)[:10])] = 3
    offsets = np.r_[0, np.cumsum(counts)]
    image_ids = np.concatenate([np.arange(1, k + 1) for k in counts])
    xyz = np.stack((np.linspace(-.3, .4, n), np.linspace(.2, -.1, n),
                    np.linspace(4.5, 6., n)), axis=1).astype(np.float32)
    views = []
    centers = {}
    K = np.array([[100., 0, 80.], [0, 100., 60.], [0, 0, 1.]])
    for i in range(1, 13):
        center = np.array([-.8 + .15 * i, .05 * (i % 3), 0.])
        pose = np.eye(4)
        pose[:3, 3] = -center
        centers[i] = center
        views.append({'image_id': i, 'name': f'{i:03}.png', 'camera_id': 1, 'split': 'train',
                          'w2c': pose.tolist(), 'w2c_original': pose.tolist()})
    points = np.repeat(xyz.astype(float), counts, axis=0)
    pc = points - np.array([centers[int(i)] for i in image_ids])
    xy = pc[:, :2] / pc[:, 2, None] * 100 + K[:2, 2]
    arrays = {'point_positions': xyz, 'point_track_ids': ids, 'observation_offsets': offsets,
                  'image_id': image_ids, 'raw_xy': xy.copy(), 'undistorted_native_xy': xy,
                  'saved_track_rms': np.zeros(n, np.float32)}
    manifest = {'views': views + [{'split': 'val', 'image_id': 999, 'forbidden': 'not accessed'}],
                    'source_cameras': {'1': {'model': 'SIMPLE_PINHOLE', 'width': 160, 'height': 120,
                        'params': [100., 80., 60.], 'K': K.tolist()}}}
    return arrays, manifest


@pytest.fixture(scope='module')
def problem(fixture_data):
    return ba.prepare_problem(*fixture_data)


def test_fixed_hash_layout_and_disjoint_support(problem, fixture_data):
    arrays, _ = fixture_data
    order = hash_order(arrays['point_track_ids'])
    l = problem.layout
    np.testing.assert_array_equal(l['fit_point_indices'], order[:500])
    assert np.any(np.diff(l['fit_offsets']) == 3)
    np.testing.assert_array_equal(l['check_point_indices'], order[500:])
    assert not set(l['fit_rows']).intersection(l['check_rows'])
    for i, p in enumerate(l['check_point_indices'][:20]):
        raw = list(range(*arrays['observation_offsets'][p:p+2]))
        wanted = sorted(raw, key=lambda r: hashlib.sha256(
            f"ba_track_holdout_v1:obs:{arrays['point_track_ids'][p]}:{arrays['image_id'][r]}"
            .encode()).digest())
        s = l['support_rows'][slice(*l['support_offsets'][i:i+2])]
        t = l['score_rows'][slice(*l['score_offsets'][i:i+2])]
        np.testing.assert_array_equal(np.r_[s, t], wanted)
        assert len(s) == (2 * len(raw) + 2) // 3 and len(t) >= 2
    np.testing.assert_array_equal(l['train_group_ids'], np.arange(12) * 4 // 12)


def test_only_geometry_keys_and_no_input_mutation(fixture_data):
    arrays, manifest = fixture_data
    class Guard(dict):
        def __getitem__(self, key):
            assert key in ba.GEOMETRY_KEYS, 'A label/pixel field was requested'
            return super().__getitem__(key)
    source = Guard(arrays, primary_label=object(), image_rgb=object())
    p = ba.prepare_problem(source, manifest)
    p.arrays['raw_xy'][0] += 100
    assert not np.array_equal(p.arrays['raw_xy'], arrays['raw_xy'])
    assert set(ba.GEOMETRY_KEYS) == set(arrays)


@pytest.mark.parametrize('fault', ['nontrain', 'duplicate_image', 'modified_pose', 'few_tracks'])
def test_invalid_geometry_refused(fixture_data, fault):
    arrays, manifest = copy.deepcopy(fixture_data)
    if fault == 'nontrain':
        arrays['image_id'][-1] = 999
    elif fault == 'duplicate_image':
        arrays['image_id'][1] = arrays['image_id'][0]
    elif fault == 'modified_pose':
        manifest['views'][0]['w2c'][0][3] += .001
    else:
        arrays['point_track_ids'][0] = arrays['point_track_ids'][1]
    with pytest.raises(ValueError):
        ba.prepare_problem(arrays, manifest)


def test_wrapper_unchanged_solver_contract_and_fit_only(monkeypatch, problem):
    expected = problem.layout['fit_track_ids']
    def solver(cloud, images, cameras, splits, **kwargs):
        assert kwargs == {'max_points': 500, 'max_nfev': 20, 'seed': 42}
        np.testing.assert_array_equal(cloud.track_ids, expected)
        assert len(cloud.points) == len(cloud.observations) == 500
        assert set(splits.values()) == {'train'}
        np.testing.assert_array_equal(cloud.points,
            problem.arrays['point_positions'][problem.layout['fit_point_indices']])
        return {}, {'accepted': False, 'success': False, 'nfev': 20}
    monkeypatch.setattr(ba, 'bounded_bundle_adjustment', solver)
    overrides, audit = ba.run_fit(problem)
    assert overrides == {} and audit['check_tracks_used'] == 0
    assert not audit['optimized_fit_points_exported']
    assert audit['fit_track_ids'] == expected.tolist()


def test_support_only_dlt_and_zero_change(problem):
    result = ba.evaluate(problem, {})
    a = result['arrays']
    assert a['baseline_eligible'].all() and a['candidate_valid'].all()
    assert np.max(a['baseline_score_radial_error']) < 1e-10
    assert np.max(abs(a['per_track_difference'])) == 0
    p = copy.deepcopy(problem)
    rows = p.layout['score_rows'][:1]
    p.arrays['normalized_xy'][rows] += 10
    p.arrays['undistorted_native_xy'][rows] += 1000
    changed = ba._reconstruct(p, {i: im.w2c for i, im in p.images.items()})
    np.testing.assert_array_equal(changed['points'], a['baseline_points'])
    assert changed['score_radial_error'][0] > 100


def fake_geometry(problem, bad_first=False, low_parallax=False):
    l = problem.layout
    track = l['score_track_indices']
    radial = (track % 7 + 1.).astype(float)
    positive = np.ones(5000, bool)
    positive[0] = not bad_first
    return {'points': np.ones((5000, 3)), 'check_uv': np.zeros((len(l['check_rows']), 2)),
                'check_depth': np.ones(len(l['check_rows'])),
                'check_residual_uv': np.zeros((len(l['check_rows']), 2)),
                'finite': np.ones(5000, bool), 'positive': positive,
                'support_parallax_degrees': np.full(5000, .01 if low_parallax else 10.),
                'score_radial_error': radial, 'per_track_mean_radial_error': np.arange(5000) % 7 + 1.}


def test_candidate_low_parallax_descriptive_and_group_equal_track(monkeypatch, problem):
    values = iter([fake_geometry(problem), fake_geometry(problem, low_parallax=True)])
    monkeypatch.setattr(ba, '_reconstruct', lambda *_: next(values))
    e = ba.evaluate(problem, {})
    assert e['arrays']['candidate_valid'].all()
    assert e['report']['candidate_low_parallax_on_fixed_population_descriptive'] == 5000
    different_from_observation_weighting = False
    for g in e['report']['by_name_group']:
        ids = problem.layout['train_image_ids'][problem.layout['train_group_ids'] == g['group']]
        mask = np.isin(problem.layout['score_image_ids'], ids)
        track = problem.layout['score_track_indices'][mask]
        expected = np.mean(np.unique(track) % 7 + 1.) if len(track) else None
        assert g['baseline']['mean'] == pytest.approx(expected)
        if len(track):
            different_from_observation_weighting |= not np.isclose(expected, np.mean(track % 7 + 1.))
    assert different_from_observation_weighting


def test_invalid_candidate_never_dropped_or_bootstrapped(monkeypatch, problem):
    values = iter([fake_geometry(problem), fake_geometry(problem, bad_first=True)])
    monkeypatch.setattr(ba, '_reconstruct', lambda *_: next(values))
    e = ba.evaluate(problem, {})
    assert e['report']['baseline_eligible_tracks'] == 5000
    assert e['report']['candidate_bad_tracks_on_fixed_population'] == 1
    assert e['report']['candidate_equal_track']['count'] == 5000
    assert e['report']['candidate_equal_track']['mean'] is None
    report = ba.analyze(problem, e, {'accepted': True})
    assert report['paired_bootstrap'] is None
    assert not report['gate']['signal']['all_candidates_valid_on_fixed_population']
    json.dumps(report, allow_nan=False)


def gate_fixture():
    return {'baseline_eligible_tracks': 1000, 'fixed_denominator_usable': True,
                'baseline_equal_track': {'mean': 1.}, 'candidate_equal_track': {'mean': .98},
                'by_view': [{'score_observations': 5}] * 200,
                'by_name_group': [{'score_observations': 100, 'difference': {'mean': -.02}}] * 4}


def test_all_coverage_signal_gates_and_no_convergence_gate():
    r = gate_fixture()
    b = {'ci95': [-.03, -.01]}
    assert ba._gate(r, b, {'accepted': True, 'success': False})['passed']
    for mutate in (
        lambda q: q.update(baseline_eligible_tracks=999),
        lambda q: q.update(by_view=q['by_view'][:199]),
        lambda q: q['by_name_group'][0].update(score_observations=99),
        lambda q: q.update(fixed_denominator_usable=False),
        lambda q: q.update(candidate_equal_track={'mean': .991}),
        lambda q: q.update(by_name_group=[{'score_observations': 100,
            'difference': {'mean': -.01 if i < 2 else 0.}} for i in range(4)]),
    ):
        q = copy.deepcopy(r)
        mutate(q)
        assert not ba._gate(q, b, {'accepted': True})['passed']
    assert not ba._gate(r, b, {'accepted': False})['passed']
    assert not ba._gate(r, {'ci95': [-.03, 0.]}, {'accepted': True})['passed']


def test_paired_track_bootstrap_rng_and_pose_metadata(monkeypatch, problem):
    before, after = fake_geometry(problem), fake_geometry(problem)
    after['score_radial_error'] *= .9
    after['per_track_mean_radial_error'] *= .9
    values = iter([before, after])
    monkeypatch.setattr(ba, '_reconstruct', lambda *_: next(values))
    e = ba.evaluate(problem, {})
    state = np.random.get_state()
    result = ba.analyze(problem, e, {'accepted': True, 'optimized_cameras': 10,
                                   'fixed_anchor_image_ids': [1, 12]})
    assert all(np.array_equal(x, y) for x, y in zip(state, np.random.get_state()))
    delta = e['arrays']['per_track_difference']
    rng = np.random.default_rng(20260927)
    draws = [np.mean(delta[rng.integers(0, len(delta), len(delta))]) for _ in range(2000)]
    np.testing.assert_array_equal(result['paired_bootstrap']['ci95'], np.percentile(draws, [2.5, 97.5]))
    assert all(v['pose_exact_original'] for v in result['pose_changes'])
    assert sum(v['anchor'] for v in result['pose_changes']) == 2
    assert result['camera_coverage_descriptive']['optimized_cameras_solver_report'] == 10
    assert result['camera_coverage_descriptive']['score_cameras_with_nonzero_pose_change'] == 0
    json.dumps(result, allow_nan=False)
