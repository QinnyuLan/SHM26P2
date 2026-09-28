"""CPU scientific contracts for the fixed forward-only visibility diagnostic."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().with_name('audit_shared_visibility_intervention.py')
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parents[1]/'scripts/audit_shared_visibility_intervention.py'
spec = importlib.util.spec_from_file_location('visibility_audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_fixed_sample_and_exact_forward_budget():
    assert audit.NAMES == ['002.png', '043.png', '084.png', '125.png', '167.png',
                           '215.png', '257.png', '300.png']
    assert audit.ORDER == ['base', 'minus', 'plus']*2
    assert len(audit.NAMES)*len(audit.ORDER) == audit.SPEC['scene_calls'] == 48
    assert 2*audit.SPEC['scene_calls']+len(audit.NAMES) == audit.SPEC['raster_calls'] == 104
    assert audit.SPEC['group_points'] == 3004
    assert audit.SPEC['backwards'] == audit.SPEC['optimizer_steps'] == 0


def test_geometry_group_uses_rotated_camera_center_and_strict_two_distances():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    center = np.array([10., 20., 30.])
    pose = np.eye(4); pose[:3, :3] = rotation; pose[:3, 3] = -rotation@center
    points = np.array([center, center+[1, 0, 0], center+[0, .5, 0], center+[0, 0, .5]])
    initial = np.array([center+[0, .5, 0], center+[0, 0, 1.5]])
    # Point 1 is exactly at the camera threshold, 2 is an initial point,
    # and 3 is near an initial point. Only point 0 is retained at distance .25.
    assert audit.group_ids(points, pose[None], initial, .25).tolist() == [0]
    # Exact equality to either threshold is excluded, not rounded inward.
    assert audit.group_ids(np.array([[0., 0., 0.], [1., 0., 0.]]), np.eye(4)[None],
                           np.array([[0., 1., 0.]]), 1.).tolist() == []


def test_logit_interventions_are_independent_odds_shifts_not_alpha_scaling():
    original = torch.tensor([-3., -.5, .75, 3.], dtype=torch.float64)
    before = original.clone(); ids = torch.tensor([1, 3])
    for case, factor in [('minus', .5), ('plus', 2.), ('base', 1.)]:
        changed = audit.shifted_logits(original, ids, case)
        assert torch.equal(changed[[0, 2]], before[[0, 2]])
        assert torch.allclose(changed[ids].exp()/original[ids].exp(),
                              torch.full((2,), factor, dtype=torch.float64), atol=1e-15)
        if case != 'base':
            assert not torch.allclose(changed[ids].sigmoid(), original[ids].sigmoid()*factor)
        changed.add_(100)
        assert torch.equal(original, before)


def test_train_class_weights_reuse_fp32_formula_and_counts():
    counts = [285818023, 17050754, 23347376, 6091203, 2011398]
    actual = audit.weights_from_counts(counts)
    frequencies = np.array(counts, dtype=np.float64)/sum(counts)
    expected = np.power(np.maximum(frequencies, .002), -.25)
    expected = np.minimum(expected/expected.mean(), 3).astype(np.float32)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


def test_contribution_keeps_all_geometry_and_only_replaces_color_background():
    kwargs = {'means': torch.randn(5, 3), 'opacities': torch.rand(5),
              'quats': torch.randn(5, 4), 'scales': torch.rand(5, 3),
              'viewmats': torch.eye(4)[None], 'Ks': torch.eye(3)[None],
              'colors': torch.rand(5, 21), 'backgrounds': torch.rand(1, 21),
              'width': 19, 'height': 13, 'near_plane': .01, 'far_plane': 1e6,
              'rasterize_mode': 'antialiased', 'packed': False, 'absgrad': False}
    changed = audit.contribution_arguments(kwargs, torch.tensor([1, 4]))
    for key in kwargs.keys()-{'colors', 'backgrounds'}:
        assert changed[key] is kwargs[key]
    assert changed['means'].shape[0] == 5
    assert changed['colors'].tolist() == [[0, 1], [1, 1], [0, 1], [0, 1], [1, 1]]
    assert torch.equal(changed['backgrounds'], torch.zeros(1, 2))
    assert kwargs['colors'].shape == (5, 21)


@pytest.mark.parametrize('bad_path', ['total', 'semantic', 'rgb', 'mass'])
def test_mass_contract_rejects_wrong_alpha_in_any_path(bad_path):
    alpha = np.full((2, 3), .8, np.float32)
    rendered = np.stack([alpha*.5, alpha], -1)
    semantic, rgb = alpha.copy(), alpha.copy()
    if bad_path == 'total':
        rendered[0, 0, 1] += .01
    elif bad_path == 'semantic':
        semantic[0, 0] += .01
    elif bad_path == 'rgb':
        rgb[0, 0] += .01
    else:
        rendered[0, 0, 0] = .9
    with pytest.raises(ValueError, match='inconclusive_render_contract'):
        audit.verify_mass(rendered, alpha, semantic, rgb)


def test_mass_contract_returns_unnormalized_full_scene_contribution():
    alpha = np.array([[0., .5, 1.]], np.float32)
    expected = np.array([[0., .2, .1]], np.float32)
    mass, report = audit.verify_mass(np.stack([expected, alpha], -1), alpha, alpha, alpha)
    np.testing.assert_array_equal(mass, expected)
    assert report['full_mass_alpha_max_error'] == report['scene_alpha_max_error'] == 0
    assert report['rgb_alpha_max_error'] == 0


def test_coverage_uses_same_known_valid_pixels_and_both_resource_limits():
    mass = np.full(257, .25)
    mass[-1] = 1000  # An ignored pixel cannot increase either coverage statistic.
    keep = np.ones(257, bool); keep[-1] = False
    result = audit.coverage(mass, keep)
    assert result == {'mass_sum': 64., 'effective_pixels': 256., 'measurable': True}
    assert not audit.coverage(np.full(255, .3), np.ones(255, bool))['measurable']
    assert not audit.coverage(np.full(300, .1), np.ones(300, bool))['measurable']
    assert audit.coverage(np.zeros(3), np.ones(3, bool))['effective_pixels'] == 0


def test_measurement_matches_independent_common_support_affine_ce_and_raw_rgb():
    rgb = np.zeros((2, 3, 3), np.float32)
    rendered = np.array([[[2., -1., .5], [.2, .3, .4], [9., 9., 9.]],
                         [[.6, .7, .8], [.5, .5, .5], [7., 7., 7.]]], np.float32)
    labels = np.array([[0, 2, 255], [4, 1, 255]], np.uint8)
    valid = np.array([[True, True, True], [True, False, False]])
    mass = np.array([[.1, .2, 100], [.7, 100, 100]], np.float32)
    raw = np.full((2, 3, 5), .1, np.float32)  # Deliberately not normalized.
    raw[0, 0, 0] = .2; raw[0, 1, 2] = .4; raw[1, 0, 4] = 0
    ids = np.array([[1, 2, 3], [0, 1, 4]], np.uint8)
    prediction = {'rgb': rendered, 'raw': raw, 'raw_ids': ids,
                  'alpha': np.ones((2, 3), np.float32)}
    weights = np.array([.5, .8, 1.2, 1.3, 1.5], np.float32)
    result = audit.score_prediction(prediction, mass, rgb, labels, valid, weights)
    locations = [(0, 0), (0, 1), (1, 0)]
    errors = [sum(float(c)**2 for c in rendered[p])/3 for p in locations]
    ce = [-np.log((1-5e-7)*float(raw[p][int(labels[p])])+5e-7/5)*float(weights[labels[p]])
          for p in locations]
    weighted = [float(mass[p]) for p in locations]
    assert result['rgb_mass'] == pytest.approx(np.dot(weighted, errors)/sum(weighted), rel=1e-14)
    assert result['ce_mass'] == pytest.approx(np.dot(weighted, ce)/sum(weighted), rel=1e-14)
    assert result['ce_full'] == pytest.approx(np.mean(ce), rel=1e-14)
    full = sum(np.square(rendered[p].astype(np.float64)).mean()
               for p in [(0, 0), (0, 1), (0, 2), (1, 0)])/4
    assert result['rgb_full'] == pytest.approx(full, rel=1e-14)
    matrix = np.zeros((5, 5), np.int64); matrix[0, 1] = matrix[2, 2] = matrix[4, 0] = 1
    np.testing.assert_array_equal(result['cm'], matrix)
    # Changing finite predictions outside the common support cannot change either main loss.
    altered = deepcopy(prediction); altered['rgb'][0, 2] = 100; altered['raw'][0, 2] = 20
    changed = audit.score_prediction(altered, mass, rgb, labels, valid, weights)
    assert changed['rgb_mass'] == result['rgb_mass'] and changed['ce_mass'] == result['ce_mass']
    assert changed['rgb_full'] > result['rgb_full']  # Still part of complete valid RGB.


def test_repeat_tolerance_and_effect_direction_use_independent_scalar_repeats():
    values = [1., .8, 1.1, 1.002, .801, 1.103]
    result = audit.effect(values)
    assert result['base'] == pytest.approx(1.001)
    assert result['minus_gain'] == pytest.approx(1.001-.8005)
    assert result['plus_gain'] == pytest.approx(1.001-1.1015)
    assert result['repeat_tolerance'] == pytest.approx(.03)
    zero = audit.effect([0.]*6)
    assert zero['minus_relative_gain'] is None
    assert zero['repeat_tolerance'] == 10*32*np.finfo(np.float64).eps


def rows_for_gate(covered=8, joint=8):
    rows = []
    for i, name in enumerate(audit.NAMES):
        means = {'rgb_full': [.1, .1, .101], 'ce_full': [1., .99, 1.01],
                 'rgb_mass': [.2, .19 if i < joint else .2, .21],
                 'ce_mass': [1., .98 if i < joint else 1., 1.01]}
        predictions = [dict({k: v[j % 3] for k, v in means.items()},
                            cm=(np.eye(5, dtype=np.int64)*(100 if j >= 3 else 1)).tolist())
                       for j in range(6)]
        rows.append({'name': name, 'coverage': {'measurable': i < covered},
                     'predictions': predictions})
    return rows


def test_gate_requires_joint_view_count_and_does_not_double_count_repeat_cm():
    passed = audit.summarize(rows_for_gate(joint=6))
    assert passed['decision'] == 'necessary_interference_signal_present'
    assert passed['required_joint_views'] == passed['joint_improving_views'] == 6
    np.testing.assert_array_equal(passed['pooled_raw']['base']['confusion_matrix'], np.eye(5)*8)
    failed = audit.summarize(rows_for_gate(joint=5))
    assert failed['decision'] == 'specified_intervention_not_supported'
    assert not failed['clauses']['joint_views']


@pytest.mark.parametrize('covered', [0, 1])
def test_gate_keeps_coverage_inconclusive_and_two_covered_views_sufficient(covered):
    assert audit.summarize(rows_for_gate(covered=covered))['decision'] == 'inconclusive_coverage'
    assert audit.summarize(rows_for_gate(covered=2, joint=2))['decision'] == 'necessary_interference_signal_present'


@pytest.mark.parametrize('violation', ['reverse', 'full_rgb', 'full_ce', 'relative'])
def test_gate_rejects_each_required_control_despite_joint_local_improvement(violation):
    rows = rows_for_gate()
    for row in rows:
        for index, pred in enumerate(row['predictions']):
            if violation == 'reverse' and index % 3 == 2:
                pred['rgb_mass'] = .18  # Opposite intervention improves more.
            if violation == 'full_rgb' and index % 3 == 1:
                pred['rgb_full'] = .1001
            if violation == 'full_ce' and index % 3 == 1:
                pred['ce_full'] = 1.
            if violation == 'relative' and index % 3 == 1:
                pred['rgb_mass'] = .1999  # .05%, below the fixed .1% resource gate.
    assert audit.summarize(rows)['decision'] == 'specified_intervention_not_supported'


def test_aggregate_is_camera_equal_not_mass_or_pixel_pooled():
    rows = rows_for_gate(covered=2)
    for index, pred in enumerate(rows[0]['predictions']):
        pred['rgb_mass'] = [.1, .08, .11][index % 3]
    for index, pred in enumerate(rows[1]['predictions']):
        pred['rgb_mass'] = [.9, .88, .91][index % 3]
    rows[0]['coverage']['mass_sum'] = 10000
    rows[1]['coverage']['mass_sum'] = 64
    result = audit.summarize(rows)
    assert result['effects']['rgb_mass']['base'] == pytest.approx(.5)
    assert result['effects']['rgb_mass']['minus_gain'] == pytest.approx(.02)


def test_state_restore_covers_features_buffers_flags_modes_and_exception():
    scene = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    scene.register_parameter('sem_features', torch.nn.Parameter(torch.randn(5, 16)))
    scene.register_buffer('semantic_prior_counts', torch.arange(6).reshape(2, 3))
    scene[0].eval(); scene[1].bias.requires_grad_(False)
    tensors = {k: v.clone() for k, v in scene.state_dict().items()}
    flags = {k: p.requires_grad for k, p in scene.named_parameters()}
    modes = {k: m.training for k, m in scene.named_modules()}
    identities = {k: id(p) for k, p in scene.named_parameters()}
    with (pytest.raises(RuntimeError, match='synthetic failure'),
          audit.restore_scene(scene) as report, torch.no_grad()):
        assert all(not p.requires_grad for p in scene.parameters())
        assert all(not m.training for m in scene.modules())
        for tensor in scene.state_dict().values():
            tensor.add_(11)
        raise RuntimeError('synthetic failure')
    assert all(report.values())
    assert report['tensor_hashes_before'] == report['tensor_hashes_after']
    assert 'sem_features' in report['tensor_hashes_before']
    assert all(torch.equal(scene.state_dict()[k], v) for k, v in tensors.items())
    assert flags == {k: p.requires_grad for k, p in scene.named_parameters()}
    assert modes == {k: m.training for k, m in scene.named_modules()}
    assert identities == {k: id(p) for k, p in scene.named_parameters()}


def test_raster_capture_restores_callable_on_exception_and_keeps_raw_values():
    raw = torch.rand(1, 2, 3, 21); alpha = torch.rand(1, 2, 3, 1)
    def rasterization(**kwargs):
        return raw, alpha, {}
    module = SimpleNamespace(rasterization=rasterization)
    with (pytest.raises(RuntimeError, match='synthetic failure'),
          audit.capture_scene(module) as capture):
        module.rasterization(render_mode='RGB+ED')
        module.rasterization(colors=torch.rand(4, 21))
        assert capture['calls'] == 2 and capture['semantic_calls'] == 1
        assert torch.equal(capture['raw'], raw[0, ..., :5])
        raise RuntimeError('synthetic failure')
    assert module.rasterization is rasterization


def test_early_target_read_fails_before_any_decoder(monkeypatch):
    import cv2
    def forbidden(*args, **kwargs):
        raise AssertionError('Decoder was reached')
    monkeypatch.setattr(cv2, 'imread', forbidden)
    with pytest.raises(ValueError, match='GT must not be decoded'):
        audit.read_targets({}, complete=False)


def assembled_measurement_audit(module):
    """Exercise actual output types through scoring, aggregation and restoration."""
    truth = np.zeros((16, 16, 3), np.float32)
    labels = np.zeros((16, 16), np.uint8)
    valid = np.ones((16, 16), bool)
    mass = np.full((16, 16), .5, np.float32)
    predictions = []
    for rgb_value, correct_probability in [( .2, .5), (.1, .6), (.3, .4)]*2:
        raw = np.full((16, 16, 5), .05, np.float32)
        raw[..., 0] = correct_probability
        prediction = {'rgb': np.full((16, 16, 3), rgb_value, np.float32), 'raw': raw,
                      'raw_ids': raw.argmax(-1).astype(np.uint8),
                      'alpha': np.ones((16, 16), np.float32)}
        predictions.append(module.score_prediction(prediction, mass, truth, labels, valid,
                                                  np.ones(5, np.float32)))
    rows = [{'name': name, 'coverage': module.coverage(mass, valid),
             'predictions': deepcopy(predictions)} for name in module.NAMES]
    scene = torch.nn.Module()
    scene.register_parameter('sem_features', torch.nn.Parameter(torch.ones(3, 16)))
    scene.register_buffer('semantic_prior_counts', torch.zeros(3, 5))
    with module.restore_scene(scene) as restoration, torch.no_grad():
        scene.sem_features.add_(1)
        summary = module.summarize(rows)
    return {'status': 'completed', 'rows': rows, 'summary': summary, 'restoration': restoration}


def test_complete_measurement_audit_serializes_strict_json_without_numpy_default():
    result = assembled_measurement_audit(audit)
    encoded = json.dumps(result, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded['summary']['decision'] == 'necessary_interference_signal_present'
    assert all(type(v) is bool for v in decoded['summary']['clauses'].values())
    assert decoded['restoration']['all_tensors_restored_exact'] is True
    assert decoded['restoration']['tensor_hashes_before'] == decoded['restoration']['tensor_hashes_after']
    assert decoded['rows'][0]['predictions'][0]['ce_mass'] > decoded['rows'][0]['predictions'][1]['ce_mass']
