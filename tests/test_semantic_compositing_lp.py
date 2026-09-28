import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

S = importlib.util.spec_from_file_location('compositing_lp', Path(__file__).parents[1]/'scripts/audit_semantic_compositing_lp.py')
M = importlib.util.module_from_spec(S); S.loader.exec_module(M)


def test_fixed_median_error_nearest_background_and_tie():
    labels = np.full((3, 5), 2, np.uint8)
    labels[1, 0] = labels[1, 4] = 0
    p = np.zeros((3, 5, 5), np.float32); p[..., 0] = 1
    rays, audit = M.select_rays(p, labels, np.ones((3, 5), bool))
    assert (rays[0]['y'], rays[0]['x']) == (1, 2)
    assert (rays[1]['y'], rays[1]['x']) == (1, 0)
    assert audit['pair_distance_squared'] == 4
    with pytest.raises(ValueError):
        M.select_rays(p, labels, np.zeros_like(labels, bool))


def test_original_compositor_derivative_reconstructs_and_detaches_scene():
    W = torch.tensor([[.1, .4], [.3, .2]])
    geometry = torch.ones(1, requires_grad=True)
    def rasterization(**kw):
        colors = kw['colors']
        a = W.sum(-1).view(1, 1, 2, 1)
        raw = (W @ colors).view(1, 1, 2, 5)+(1-a)*kw['backgrounds']
        assert not kw['means'].requires_grad
        return raw, a, {}
    module = SimpleNamespace(rasterization=rasterization)
    p = torch.tensor([[.1, .1, .6, .1, .1], [.7, .1, .1, .05, .05]])
    with torch.no_grad(), M.differentiable_semantic_colors(module) as capture:
        raw, alpha, _ = module.rasterization(colors=p, means=geometry, backgrounds=torch.tensor([[1., 0, 0, 0, 0]]))
    assert module.rasterization is rasterization
    for i in range(2):
        g = torch.autograd.grad(raw[0, 0, i, 0], capture[0]['colors'], retain_graph=i == 0)[0]
        torch.testing.assert_close(g[:, 0], W[i])
        assert not g[:, 1:].any()
        norm = raw[0, 0, i].detach().numpy(); norm=norm/norm.sum()
        audit = M.verify_ray(g[:, 0].numpy(), p.numpy(), float(alpha[0, 0, i, 0]), raw[0, 0, i].detach().numpy(), norm)
        assert audit['raw_max_absolute_error'] < 1e-7
    assert geometry.grad is None


def test_capture_restores_after_exception():
    original = lambda **kwargs: None
    module = SimpleNamespace(rasterization=original)
    with pytest.raises(RuntimeError), M.differentiable_semantic_colors(module):
        raise RuntimeError('failure')
    assert module.rasterization is original


@pytest.mark.parametrize('W,alpha,labels,expected', [
    ([[1., 0.], [0., 1.]], [1., 1.], [0, 2], 1.),
    ([[1.], [1.]], [1., 1.], [0, 2], 0.),
    ([[.2]], [.2], [2], -.6),
    ([[]], [0.], [2], -1.),
])
def test_known_lp_cases_with_primal_and_dual_bounds(W, alpha, labels, expected):
    report, q, dual = M.solve_lp(np.array(W), alpha, labels)
    c = report['certificate']
    assert report['success']
    assert c['primal_margin_lower_bound'] == pytest.approx(expected, abs=1e-8)
    assert c['dual_margin_upper_bound'] == pytest.approx(expected, abs=1e-8)
    assert abs(c['upper_minus_lower']) < 1e-8
    assert dual.min() >= 0 and dual.sum() == pytest.approx(1)
    np.testing.assert_allclose(q.sum(-1), 1.)
    if expected == 0:
        assert c['conclusion'] == 'near_zero_or_unresolved_bound_only'


def test_dual_bound_uses_tiny_full_weight_even_if_candidate_solver_ignored_it():
    W = np.array([[1., 1e-12], [1., 0.]])
    alpha = np.ones(2)
    q = np.array([[.5, 0, .5, 0, 0], [0, 0, 1, 0, 0]])
    lam = np.zeros(8); lam[0] = lam[5] = .5  # ray0 target2 vs bg; ray1 target0 vs2
    c, _, _ = M.certificate(W, alpha, np.array([2, 0]), q, lam)
    assert c['dual_margin_upper_bound'] == pytest.approx(5e-13, abs=1e-20)
    assert c['primal_margin_lower_bound'] == 0


def test_ray_prerequisite_rejects_missing_tail_and_negative_weights():
    colors = np.array([[0, 0, 1., 0, 0], [1., 0, 0, 0, 0]])
    with pytest.raises(ValueError):
        M.verify_ray([.4, 0], colors, 1., [.6, 0, .4, 0, 0], [.6, 0, .4, 0, 0])
    with pytest.raises(ValueError):
        M.verify_ray([-.1, 1.1], colors, 1., [.6, 0, .4, 0, 0], [.6, 0, .4, 0, 0])


def test_overlap_reports_cross_view_independence_without_discarding_tails():
    W = np.array([[.6, .2, 0], [.4, .4, 0], [0, 0, .9], [0, 0, .3]])
    report = M.row_overlap(W)
    assert report['overlap_matrix'][0][1] == pytest.approx(.6)
    assert report['same_view']['positive_overlap_pairs'] == 2
    assert report['cross_view']['positive_overlap_pairs'] == 0
    W[2, 0] = 1e-14
    report = M.row_overlap(W)
    assert report['cross_view']['positive_overlap_pairs'] == 2
    assert report['overlap_matrix'][0][2] == 1e-14
