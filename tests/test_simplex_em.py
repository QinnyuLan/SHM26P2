import numpy as np
import pytest

from bridge_rgs.simplex_em import em_step
from bridge_rgs.simplex_optimization import frank_wolfe_gap, validate_simplex


def mixture_objective(q):
    # Independent rays with fixed residual background, affine noise, unequal
    # view weights and unequal numbers of pixels per view.
    views = [
        (np.array([[.4, .4, 0.], [.7, .1, 0.], [.2, .3, 0.], [0., 0., 0.]]),
         np.array([0, 1, 2, 2]), .3),
        (np.array([[.1, .8, 0.], [.5, .2, 0.]]), np.array([1, 2]), .7),
    ]
    class_weights = np.array([.6, 1.3, 1.8])
    delta = 5e-7
    gradient = np.zeros_like(q)
    responsibilities = np.zeros_like(q)
    loss = 0.
    for W, labels, view_weight in views:
        background = np.zeros((len(labels), 3))
        background[:, 0] = 1-W.sum(axis=1)
        raw = W@q + background
        p = (1-delta)*raw + delta/3
        for ray, cls in enumerate(labels):
            weight = view_weight*class_weights[cls]/len(labels)
            loss -= weight*np.log(p[ray, cls])
            for point in range(len(q)):
                gradient[point, cls] -= weight*(1-delta)*W[ray, point]/p[ray, cls]
                # Direct latent responsibility, separate from helper's q*(-g).
                responsibilities[point, cls] += weight*((1-delta)*W[ray, point]*q[point, cls]/p[ray, cls])
    return float(loss), gradient, responsibilities


def test_fixed_mixture_em_monotone_with_background_noise_and_view_weights():
    q = np.array([[.2, .3, .5], [.5, .2, .3], [.3, .3, .4]])
    initial = q.copy()
    for _ in range(8):
        before, gradient, independent_counts = mixture_objective(q)
        saved_q, saved_g = q.copy(), gradient.copy()
        step = em_step(q, gradient)
        np.testing.assert_allclose(step.responsibilities, independent_counts, atol=1e-15)
        after = mixture_objective(step.proposal)[0]
        # The exact Jensen lower bound is stronger than nonincrease alone.
        kl = np.sum(step.proposal*np.log(step.proposal/q), axis=1)
        bound = np.dot(step.row_responsibility, kl)
        assert before-after >= bound-2e-14
        assert after < before
        assert step.proposal[-1].tobytes() == initial[-1].tobytes()
        np.testing.assert_array_equal(q, saved_q)
        np.testing.assert_array_equal(gradient, saved_g)
        validate_simplex(step.proposal)
        q = step.proposal


def test_exact_mixture_gradient_matches_finite_difference():
    q = np.array([[.2, .3, .5], [.5, .2, .3], [.3, .3, .4]])
    gradient = mixture_objective(q)[1]
    h = 1e-6
    for i, c in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[i, c] += h
        minus[i, c] -= h
        fd = (mixture_objective(plus)[0]-mixture_objective(minus)[0])/(2*h)
        assert fd == pytest.approx(gradient[i, c], abs=3e-9)


@pytest.mark.parametrize('class0_weight', [0., .1])
def test_zero_lock_can_have_positive_fw_gap_including_zero_responsibility(class0_weight):
    q = np.array([[1., 0.]])
    delta = .02
    frequencies = np.array([[class0_weight, 1-class0_weight]])
    def objective(q):
        p = (1-delta)*q+delta/2
        return float(-np.sum(frequencies*np.log(p))), -(1-delta)*frequencies/p
    loss, gradient = objective(q)
    step = em_step(q, gradient)
    assert step.proposal.tobytes() == q.tobytes()
    assert bool(step.zero_responsibility_rows[0]) == (class0_weight == 0)
    gap = frank_wolfe_gap(q, gradient)
    assert gap.total_gap > 0
    assert objective(.99*q+.01*gap.vertex)[0] < loss


def test_zero_responsibility_rows_preserved_and_no_implicit_simplex_repair():
    q = np.array([[.2, .3, .5], [.2, .3, .5-1e-13], [0., .4, .6]])
    g = np.array([[0., 0., 0.], [0., 0., 0.], [-7., -1., -2.]])
    step = em_step(q, g)
    assert step.proposal[:2].tobytes() == q[:2].tobytes()
    np.testing.assert_array_equal(step.zero_responsibility_rows, [True, True, False])
    np.testing.assert_allclose(step.proposal[2], [0., .25, .75], atol=1e-16)
    assert step.proposal[2, 0] == 0  # no floor despite a nonzero derivative


@pytest.mark.parametrize('gradient', [np.array([[0., 1e-300]]), np.array([[0., np.nan]]),
                                     np.array([[0., -np.inf]]), np.zeros((1, 2), np.float32),
                                     np.zeros((2, 2))])
def test_invalid_gradient_is_rejected_without_clamping(gradient):
    with pytest.raises(ValueError):
        em_step(np.array([[1., 0.]]), gradient)


@pytest.mark.parametrize('q', [np.array([[1.1, -.1]]), np.array([[.3, .3]]),
                              np.array([[.5, .5]], np.float32)])
def test_invalid_q_is_rejected(q):
    with pytest.raises(ValueError):
        em_step(q, np.zeros_like(q))


@pytest.mark.parametrize(('q', 'g'), [
    (np.array([[1e-200, 1.]]), np.array([[-1e-200, -1.]])),  # product rounds to zero
    (np.array([[1e-300, 1.]]), np.array([[-1e-10, -1.]])),   # subnormal responsibility
    (np.array([[.5, .5]]), np.array([[-2e-300, -2e100]])),   # normalized positive count vanishes
    (np.array([[.5+4e-13, .5+4e-13]]), np.full((1, 2), -np.finfo(np.float64).max)),
])
def test_extreme_arithmetic_fails_loud_without_support_repair(q, g):
    with pytest.raises(ValueError):
        em_step(q, g)
