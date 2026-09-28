"""Independent decomposition identities on tiny arrays, without experiment data."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import audit_residual_transport_decomposition as audit


def test_masked_render_component_and_full_identity():
    base = np.ones((1, 2, 3), np.float32)*.2
    s = np.zeros_like(base, dtype=np.float64); s[:, 0] = .1
    support = np.array([[True, False]])
    parts = audit.components(base, s, np.ones_like(base)*.5, support)
    np.testing.assert_allclose(parts['render_only'][0, 0], .5-base[0, 0].astype(np.float64))
    np.testing.assert_array_equal(parts['full_photo'][0, 1], 0)
    np.testing.assert_array_equal(parts['full_photo'], parts['render_only']+s)


def test_five_terms_and_corrupted_cross_sign_refused():
    rng = np.random.default_rng(2)
    base, truth, s, d = [rng.normal(size=(2, 3, 3)) for _ in range(4)]
    valid = np.ones((2, 3), bool)
    result = audit.expansion(base, truth, s, d, .3, valid)
    assert result['sum'] == pytest.approx(result['direct_MSE_change'], abs=1e-14)
    bad = {**result, 'terms': {**result['terms'], 'cross_source_render': -result['terms']['cross_source_render']}}
    with pytest.raises(ValueError, match='Numerical mismatch'):
        audit.previous.Audit().close(result, bad, 'wrong cross term')
