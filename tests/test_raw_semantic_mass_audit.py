import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('raw_mass', Path(__file__).parents[1]/'scripts/audit_raw_semantic_mass.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_fixed_train_only_alternating_sample():
    manifest = {'views': [{'name': f'{i:03}.png', 'split': 'train', 'mask_path': '/no/decode'} for i in range(259)]
                + [{'name': 'val.png', 'split': 'val', 'mask_path': '/forbidden'}]}
    indices, views = M.fixed_views(manifest)
    assert indices == [0, 36, 73, 110, 147, 184, 221, 258]
    assert [v['name'] for v in views] == [f'{i:03}.png' for i in indices]
    manifest['views'][0]['split'] = 'val'
    with pytest.raises(ValueError):
        M.fixed_views(manifest)


def test_weighted_ce_analytic_gradient_and_background_gauge():
    rng = np.random.default_rng(5)
    p = rng.dirichlet(np.ones(5), size=30)
    labels = rng.integers(5, size=30)
    weights = M.class_weights([800, 100, 40, 30, 30])
    bias = np.array([.4, -.2, .3, .1])
    value, grad = M.objective(bias, np.log(p), labels, weights)
    assert value > 0
    numeric = []
    for i in range(4):
        d = np.eye(4)[i]*1e-5
        numeric.append((M.objective(bias+d, np.log(p), labels, weights)[0]-M.objective(bias-d, np.log(p), labels, weights)[0])/2e-5)
    np.testing.assert_allclose(grad, numeric, rtol=1e-7, atol=1e-9)


def test_bias_fit_improves_fixed_ce_bounded_no_geometry_input():
    p = np.tile(np.array([.9, .025, .025, .025, .025], np.float32), (10, 10, 1))
    target = np.full((10, 10), 2, np.uint8)
    arrays = [('train.png', p, target, np.ones((10, 10), bool))]
    b, audit = M.fit_bias(arrays, np.ones(5))
    assert b[0] == 0 and max(abs(b)) <= 6
    assert audit['weighted_ce_after'] < audit['weighted_ce_before']
    b2, audit2 = M.fit_bias(arrays, np.ones(5))
    np.testing.assert_array_equal(b, b2)
    assert audit == audit2


def test_alpha_bound_is_distinct_from_current_cable_probability():
    # Pixel 1 has low total alpha. Pixel 2 is opaque but poorly classified.
    p = np.array([[[.8, .01, .17, .01, .01], [.8, .01, .17, .01, .01]]], np.float32)
    alpha = np.array([[.2, 1.]], np.float32)
    target = np.array([[2, 2]], np.uint8)
    result = M.summarize(p, alpha, target, np.ones((1, 2), bool), np.array([0, 0, 2, 0, 0]))
    cable = result['by_gt_class'][2]
    assert cable['semantic_alpha_le_half'] == 1
    assert cable['p_true_le_half'] == 2 and cable['p_background_gt_p_true'] == 2
    assert result['raw']['confusion_matrix'][2][0] == 2
    assert result['calibrated']['confusion_matrix'][2][2] == 2


def test_alpha_capture_is_observation_only_and_finally_restored():
    calls = []
    def render(**kw):
        calls.append(kw)
        return ('color', 'alpha', {})
    module = SimpleNamespace(rasterization=render)
    with pytest.raises(RuntimeError), M.capture_semantic_alpha(module) as records:
        assert module.rasterization(render_mode='RGB+ED') == ('color', 'alpha', {})
        assert records == [{'render_mode': 'RGB+ED', 'alpha': 'alpha'}]
        raise RuntimeError('test finally')
    assert module.rasterization is render and calls == [{'render_mode': 'RGB+ED'}]


def test_scalar_temperature_argmax_and_two_class_mass_boundary():
    # Any positive temperature has the same class order; other foreground
    # competitors invalidate treating .5 as a general necessary threshold.
    p = np.array([[.4, .02, .5, .04, .04]])
    for t in (.1, 1., 10.):
        assert (np.log(p)/t).argmax(1)[0] == 2
    for mass in (.1, .499, .5):
        assert np.array([1-mass, mass]).argmax() == 0
    assert np.array([.4, .3, .3]).argmax() == 0
