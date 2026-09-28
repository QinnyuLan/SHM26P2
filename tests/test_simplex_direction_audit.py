"""Three small independent-checker contracts; synthetic arrays only."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

module_spec = importlib.util.spec_from_file_location(
    'independent_direction', Path(__file__).parents[1]/'scripts/audit_simplex_direction.py')
m = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(m)
SPEC = {'delta': 5e-7, 'bins': [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, .1, None],
        'absolute_tolerance': 2e-7, 'relative_tolerance': 5e-4, 'signal_multiplier': 10.}


def test_g32_mean_lmo_ties_and_wrong_dtype_refusal(tmp_path):
    g0 = np.array([[1, -2, -2, 1, 0], [0, -1, 0, 0, 0]], np.float32)
    g1 = np.array([[1, -2, -2, 1, 0], [-4, 0, 0, 0, 0]], np.float32)
    files = [tmp_path/'a.npy', tmp_path/'b.npy']
    for path, value in zip(files, (g0, g1), strict=True):
        np.save(path, value)
    records = [{'gradient': str(p)} for p in files]
    mean, vertex = m.reduce_gradients(records, (2, 5))
    np.testing.assert_array_equal(mean, (g0.astype(np.float64)+g1)/2)
    np.testing.assert_array_equal(vertex.argmax(1), [1, 0])
    np.save(files[1], g1.astype(np.float64))
    with pytest.raises(ValueError, match='g32'):
        m.reduce_gradients(records, (2, 5))


def test_raw_line_bins_preserve_background_and_reject_invalid_cache():
    a, v = np.array([0, .8, .3], np.float32), np.array([.1, .9, .2], np.float32)
    y = np.array([2, 0, 4], np.uint8)
    weights = np.array([.5, 1., 1.7, .9, 1.2])
    r = m.view_terms(a, v, y, weights, SPEC, gamma=.25)
    p = (1-5e-7)*a.astype(np.float64)+1e-7
    derivative = -weights[y]*(1-5e-7)*(v.astype(np.float64)-a)/p
    assert r['B64'] == pytest.approx(derivative.mean(), rel=1e-15)
    assert sum(r['bins']['pixel_counts']) == 3
    assert sum(r['bins']['positive'])+sum(r['bins']['negative']) == pytest.approx(r['B64'])
    assert r['bins']['pixel_counts'][0] == 1
    assert r['F_candidate'] != r['F']
    with pytest.raises(ValueError, match='domain'):
        m.view_terms(a, v, np.array([255, 0, 4], np.uint8), weights, SPEC)


def test_consistency_gate_rejects_scaling_sign_and_insufficient_signal():
    assert all(m.gate_values(-.1, -.1, -.1, SPEC).values())
    assert not all(m.gate_values(-.101, -.1, -.1, SPEC).values())
    assert not all(m.gate_values(.1, .1, .1, SPEC).values())
    assert not all(m.gate_values(-1e-9, -1e-9, -1e-9, SPEC).values())
    with pytest.raises(ValueError, match='Nonfinite'):
        m.gate_values(np.nan, -.1, -.1, SPEC)
