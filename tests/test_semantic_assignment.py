import numpy as np
import torch

from bridge_rgs.semantic_assignment import affine_raw_ce, realize_probabilities, waterfill_counts


def test_affine_counts_match_hand_responsibilities_including_fixed_components():
    q = torch.tensor([[.2, .1, .5, .1, .1], [.6, .1, .1, .1, .1]], dtype=torch.float64, requires_grad=True)
    W = torch.tensor([[.2, .5], [0., 0.], [.3, .1]], dtype=torch.float64)
    labels = torch.tensor([[2, 2, 0]])
    cw = torch.tensor([.5, .9, .8, 1.2, 1.6], dtype=torch.float64)
    residual = torch.zeros(3, 5, dtype=torch.float64); residual[:, 0] = 1-W.sum(-1)
    raw = (W @ q+residual).view(1, 3, 5)
    loss = affine_raw_ce(raw, labels, torch.ones_like(labels), cw)
    g = torch.autograd.grad(loss, q, retain_graph=True)[0]
    target = raw.detach()[0, torch.arange(3), labels[0]]
    hand = torch.zeros_like(q)
    for r, c in enumerate(labels[0]):
        hand[:, c] += cw[c]*(1-5e-7)*W[r]/((1-5e-7)*target[r]+1e-7)/3
    torch.testing.assert_close(-g*q.detach(), hand*q.detach(), atol=1e-12, rtol=1e-12)
    assert bool((-g >= 0).all()) and torch.isfinite(loss)
    # A zero-alpha foreground pixel's noise loss is finite and adds no q responsibility.
    isolated = affine_raw_ce(raw[:, 1:2], labels[:, 1:2], torch.ones((1, 1)), cw)
    gi = torch.autograd.grad(isolated, q)[0]
    assert torch.count_nonzero(gi) == 0


def test_waterfill_kkt_bounds_and_zero_row_keep_actual_qold():
    M = np.array([[9, 0, 0, 1e-9, 1], [1, 2, 3, 4, 5], [0, 0, 0, 0, 0]], float)
    qold = np.array([[.2]*5, [.2]*5, [9.99999e-6, .2, .2, .2, .39999000001]])
    q, active = waterfill_counts(M, qold)
    np.testing.assert_array_equal(q[2], qold[2])
    np.testing.assert_array_equal(active, [True, True, False])
    np.testing.assert_allclose(q[:2].sum(-1), 1, atol=1e-12)
    assert (q[:2] >= 1e-5).all()
    for row in range(2):
        free = q[row] > 1e-5
        lagrange = M[row, free]/q[row, free]
        np.testing.assert_allclose(lagrange, lagrange[0], atol=1e-10)
        assert (M[row, ~free]/1e-5 <= lagrange[0]+1e-10).all()


def test_waterfill_single_nonzero_and_allzero():
    q, _ = waterfill_counts([[1, 0, 0, 0, 0]], [[.2]*5])
    np.testing.assert_allclose(q, [[1-4e-5, 1e-5, 1e-5, 1e-5, 1e-5]])
    empty, active = waterfill_counts(np.zeros((2, 5)), np.full((2, 5), .2))
    np.testing.assert_array_equal(empty, np.full((2, 5), .2)); assert not active.any()


def test_feature_realization_preserves_full_rank_decoder():
    torch.manual_seed(4)
    W = torch.randn(5, 16); b = torch.randn(5); f = torch.randn(7, 16)
    q = np.eye(5)[np.arange(7)%5]*(1-5e-5)+1e-5
    new, _ = realize_probabilities(f, W, b, q)
    np.testing.assert_allclose((new @ W.T+b).softmax(-1), q, atol=5e-6, rtol=0)
