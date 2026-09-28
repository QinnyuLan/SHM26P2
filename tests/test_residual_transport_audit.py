"""Small independent-audit fixtures; no experiment files or CUDA."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('transport_audit', Path(__file__).parents[1]/'scripts/audit_residual_transport.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_independent_corner_replay_and_reflected_support():
    h, w = 3, 5
    target = {'depth': np.full((h, w), 2., np.float32), 'alpha': np.ones((h, w), np.float32),
              'valid': np.ones((h, w), bool), 'K': np.eye(3, dtype=np.float32), 'w2c': np.eye(4, dtype=np.float32)}
    source = {k: v.copy() for k, v in target.items()}
    source['residual'] = np.broadcast_to(np.arange(w)[None, :, None]/10, (h, w, 3)).copy()
    source['valid'][:, 1] = False
    source['depth'][:, 0] = 3
    result, rows = audit.replay(target, [source])
    np.testing.assert_array_equal(result['valid'][0], [False, False, True, False, True])
    np.testing.assert_array_equal(result['true_residual'][:, -1], .4)
    np.testing.assert_array_equal(result['wrong_residual'][:, -1], 0)
    assert rows[0]['common_pixels'] == 6
    wrong = result['source_count'].copy(); wrong[0, 0] = 1
    with pytest.raises(ValueError, match='Exact count'):
        audit.Audit().close(result['source_count'], wrong, 'corrupted support')


def test_independent_fit_equal_views_and_scalar_corruption():
    rows = []
    for size, coefficient in [(1, .2), (40, .8)]:
        shape = (1, size, 3)
        rows.append(audit.sufficient(np.zeros(shape), np.full(shape, coefficient), np.ones(shape), np.ones(shape[:2], bool)))
    fit = audit.fit(rows)
    assert fit['coefficient'] == pytest.approx(.5)
    with pytest.raises(ValueError, match='Numerical mismatch'):
        audit.Audit().close(fit, {**fit, 'coefficient': .8}, 'wrong fold coefficient')


def test_bound_file_corruption_rejected(tmp_path):
    path = tmp_path/'artifact'; path.write_bytes(b'original')
    expected = audit.sha(path); path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA'):
        audit.Audit().bind(str(path), expected)
