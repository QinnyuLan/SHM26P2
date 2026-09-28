"""Synthetic only: no source data, checkpoint, renderer extension, or GPU."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
WORKER = HERE/'diagnose_continuous_semantic_geometry.py'
if not WORKER.exists():
    WORKER = HERE.parent/'scripts/diagnose_continuous_semantic_geometry.py'
spec = importlib.util.spec_from_file_location('continuous_preflight_test', WORKER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_covariance_direction_has_unit_metric_and_rotates():
    g = np.array([[1., 2., -1.], [.1, -.3, .7]])
    s = np.array([[.1, 1., 2.], [3., .3, 1.]])
    R = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    d, stats = m.covariance_direction(g, R, s)
    np.testing.assert_allclose(np.linalg.norm(d/s, axis=1).max(), 1., atol=1e-15)
    assert stats['nonzero_rows'] == 2
    Q = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotated, _ = m.covariance_direction(g @ Q.T, Q @ R, s)
    np.testing.assert_allclose(rotated, d @ Q.T, atol=1e-15)


@pytest.mark.parametrize('which', ['zero', 'scale', 'rotation', 'nan'])
def test_invalid_direction_rejected(which):
    g = np.ones((2, 3)); s = np.ones_like(g); R = np.tile(np.eye(3), (2, 1, 1))
    if which == 'zero':
        g[:] = 0
    if which == 'scale':
        s[0, 0] = 0
    if which == 'rotation':
        R[0, 0, 0] = 2
    if which == 'nan':
        g[0, 0] = np.nan
    with pytest.raises(ValueError):
        m.covariance_direction(g, R, s)


def test_actual_fp32_displacement_is_used_and_one_sided_reported():
    base = np.array([[1e4, 2., -.7]], np.float32)
    d = np.array([[.2, -.3, .7]], np.float64)
    g = np.array([[.3, .2, .1]], np.float32)
    h = 1/256
    plus, minus = m.displaced(base, d, h)
    loss = lambda x: float(np.sum(x.astype(np.float64)*g.astype(np.float64)))
    row = m.fd_record(g, plus, minus, h, loss(plus), loss(minus), loss(base), loss(base))
    assert row['passed'] and row['relative_error'] < 1e-7
    ideal = float(np.sum(g*d))
    assert abs(row['analytic_actual_fp32_displacement']-ideal) > .001
    side = m.actual_displacement_stats(base, plus, minus, d, np.eye(3)[None], np.ones((1, 3)),
                                      h, g, (loss(base), loss(plus), loss(minus)))
    assert side['plus']['cast_error_rms'] > 0
    np.testing.assert_allclose(side['minus']['analytic_actual_change'], side['minus']['actual_loss_change'], atol=1e-12)


def test_repeat_floor_and_cross_zero_do_not_fake_main_success():
    base = np.zeros((1, 3), np.float32); d = np.ones((1, 3), np.float64)
    plus, minus = m.displaced(base, d, 1/64)
    row = m.fd_record(np.ones_like(base), plus, minus, 1/64, 1.1, .9, 1., 1.1)
    assert not row['measurable'] and not row['passed']
    good = m.fd_record(np.ones_like(base), plus, minus, 1/64, .046875, -.046875, 0., 0.)
    zero = m.fd_record(np.zeros_like(base), plus, minus, 1/64, 0., 0., 0., 0.)
    rows = []
    for view in m.SPEC['names']:
        for own in m.SPEC['losses']:
            for h in m.SPEC['amplitudes']:
                for measured in m.SPEC['losses']:
                    rows.append({**(good if own == measured else zero), 'view': view, 'amplitude': h,
                                 'direction_loss': own, 'measured_loss': measured})
    summary = m.summarize(rows)
    assert summary['numerical_status'] == 'passed' and summary['main_count'] == 12
    json.dumps({'records': rows, 'summary': summary}, allow_nan=False)


def test_standard_pair_keeps_means_graph_and_fixed_q_unchanged():
    means = torch.tensor([[.2, .3, .4]], requires_grad=True)
    q = torch.tensor([[.2, .1, .3, .1, .3000001]])
    original_q = q.clone()
    s = {'quats': torch.tensor([[1., 0., 0., 0.]]), 'log_scales': torch.zeros(1, 3),
         'opacity_logits': torch.zeros(1), 'sh0': torch.zeros(1, 1, 3), 'sh_rest': torch.zeros(1, 15, 3)}
    scene = SimpleNamespace(splats=s, background_logits=torch.zeros(3))
    calls = []
    def fake(**kwargs):
        calls.append(kwargs)
        assert kwargs['means'] is means and not kwargs['absgrad']
        size = 4 if 'sh_degree' in kwargs else 5
        value = means.square().sum().expand(1, 2, 3, size)
        return value, torch.ones(1, 2, 3, 1), {}
    view = {'K': np.eye(3), 'w2c_original': np.eye(4), 'width': 3, 'height': 2}
    counts = {'raster': 0}
    rgb, raw = m.standard_pair(scene, means, q, view, fake, counts)
    assert rgb.shape == (2, 3, 3) and raw.shape == (2, 3, 5) and counts['raster'] == 2
    assert calls[1]['colors'] is q and torch.equal(q, original_q)
    assert calls[0]['render_mode'] == 'RGB+ED'
    assert torch.count_nonzero(torch.autograd.grad(rgb.sum()+raw.sum(), means)[0]) == 3


def test_collapsed_float32_endpoints_fail():
    with pytest.raises(ValueError, match='collapsed'):
        m.displaced(np.full((1, 3), 1e20, np.float32), np.ones((1, 3), np.float64), 1/256)
