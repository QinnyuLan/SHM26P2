"""Only synthetic/cache-free contracts for the independent decomposition."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
WORKER = HERE/'decompose_residual_transport.py'
if not WORKER.exists():
    WORKER = HERE.parent/'scripts/decompose_residual_transport.py'
spec = importlib.util.spec_from_file_location('decomposition_worker_test', WORKER)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
HELPER = HERE/'residual_transport.py'
if not HELPER.exists():
    HELPER = HERE.parent/'src/bridge_rgs/residual_transport.py'
h = w.load_helper(HELPER)


def test_shared_support_zero_outside_and_exact_algebra():
    base = np.full((2, 2, 3), .4)
    s = np.zeros_like(base); s[0] = .1
    mask = np.array([[True, True], [False, False]])
    count = mask.astype(np.int32)*2
    out = w.decompose_arrays(base, s, np.full_like(base, .7), mask, count)
    np.testing.assert_allclose(out['render_only'][0], .3)
    np.testing.assert_allclose(out['full_photo'][0], .4)
    np.testing.assert_array_equal(out['render_only'][~mask], 0)
    np.testing.assert_array_equal(out['full_photo'][~mask], 0)
    with pytest.raises(ValueError, match='zero outside'):
        w.decompose_arrays(base, s+.1, base, mask, count)
    with pytest.raises(ValueError, match='support/count'):
        w.decompose_arrays(base, s, base, mask, np.ones((2, 2), np.int32))


def test_actual_helper_render_warp_preserves_source_common_mask_and_counts():
    shape = (3, 5)
    source = {'depth': np.full(shape, 2.), 'alpha': np.ones(shape),
              'valid': np.ones(shape, bool), 'K': np.eye(3), 'w2c': np.eye(4),
              'residual': np.full(shape+(3,), .05)}
    source['valid'][:, 1] = False
    base = np.full(shape+(3,), .4)
    args = (np.full(shape, 2.), np.ones(shape), np.ones(shape, bool), np.eye(3), np.eye(4))
    old = h.transport_residual(*args, [source])
    warped = h.transport_residual(*args, [{**source, 'residual': np.full_like(base, .6)}])
    np.testing.assert_array_equal(old['valid'], warped['valid'])
    np.testing.assert_array_equal(old['source_count'], warped['source_count'])
    out = w.decompose_arrays(base, old['true_residual'], warped['true_residual'], old['valid'], old['source_count'])
    np.testing.assert_allclose(out['full_photo'][out['valid']], .25)


def test_five_term_expansion_includes_interaction_and_does_not_clip():
    rng = np.random.default_rng(20260927)
    base, truth = rng.random((3, 4, 3)), rng.random((3, 4, 3))
    s, d = rng.normal(size=(2, 3, 4, 3))
    valid = np.ones((3, 4), bool); valid[0, 0] = False
    out = w.five_term_expansion(base, truth, s, d, .7, valid)
    direct = np.mean((base+.7*(s+d)-truth)[valid]**2)-np.mean((base-truth)[valid]**2)
    assert out['sum'] == pytest.approx(direct, abs=1e-14)
    assert out['terms']['cross_source_render'] != 0
    scored = w.metrics(base, truth, s+d, .7, valid)
    assert scored['clipped_MSE'] != scored['unclipped_MSE']
    json.dumps(out, allow_nan=False)


def test_other_fold_shrinkage_no_evaluation_target_coefficient():
    rows = []
    for i, name in enumerate(w.NAMES):
        fold = 'A' if i % 2 == 0 else 'B'
        numerator = .2 if fold == 'A' else .8
        rows.append({'name': name, 'fold': fold,
                     'statistics': {a: {'numerator': numerator, 'denominator': 1.,
                                        'zero_mse': 1., 'valid_pixels': i+1} for a in w.ARMS[1:]}})
    fits = w.cross_fit(rows, h)
    assert w.coefficient_for(rows[0], 'full_photo', fits) == (.8, 'B')
    assert w.coefficient_for(rows[1], 'source_residual', fits) == (.2, 'A')
    assert w.coefficient_for(rows[0], 'zero', fits) == (0., None)
    rows[0]['fold'] = 'B'
    with pytest.raises(ValueError, match='folds'):
        w.cross_fit(rows, h)


def test_prediction_barrier_rejects_partial_and_unbound_outputs():
    records = [{'name': n, 'path': '/fake/'+n, 'sha256': 'a'*64} for n in w.NAMES]
    w.prediction_barrier(records)
    with pytest.raises(ValueError, match='precede targets'):
        w.prediction_barrier(records[:-1])
    records[0]['sha256'] = None
    with pytest.raises(ValueError, match='precede targets'):
        w.prediction_barrier(records)


def test_equal_view_log_mean_is_not_log_mean_mse_and_zero_is_na():
    rows = []
    for i, name in enumerate(w.NAMES):
        mse = .01 if i % 2 == 0 else .0001
        metrics = {d+'_MSE': mse for d in ('clipped', 'unclipped')}
        metrics.update({d+'_log10_MSE': np.log10(mse).item() for d in ('clipped', 'unclipped')})
        rows.append({'name': name, 'fold': 'A' if i % 2 == 0 else 'B',
                     'metrics': {a: metrics.copy() for a in w.ARMS}})
    summary = w.summarize(rows)
    assert summary['all']['full_photo']['clipped_mean_PSNR_dB'] == 30
    assert summary['all']['full_photo']['clipped_mean_log10_MSE'] != np.log10(summary['all']['full_photo']['clipped_MSE'])
    zero = w.metrics(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)),
                     np.zeros((1, 1, 3)), 0, np.ones((1, 1), bool))
    rows[0]['metrics']['zero'] = zero
    summary = w.summarize(rows)
    assert summary['all']['zero']['clipped_mean_PSNR_dB'] is None
    assert summary['all']['zero']['clipped_zero_MSE_views'] == 1
    json.dumps(summary, allow_nan=False)
