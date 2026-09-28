import numpy as np
import pytest
from scipy.optimize import minimize

from bridge_rgs.simplex_optimization import (
    feasible_direction,
    frank_wolfe_gap,
    linear_minimization_oracle,
    project_simplex,
    projected_gradient_step,
    validate_simplex,
)


def test_projection_matches_independent_scipy_constrained_optimization():
    rng = np.random.default_rng(20260927)
    values = rng.normal(size=(7, 5)) * 2
    expected = []
    for row in values:
        result = minimize(
            lambda q, row=row: .5 * np.square(q - row).sum(), np.full(5, .2),
            jac=lambda q, row=row: q - row, method="SLSQP", bounds=[(0, 1)] * 5,
            constraints={"type": "eq", "fun": lambda q: q.sum() - 1,
                         "jac": lambda q: np.ones(5)},
            options={"ftol": 1e-13, "maxiter": 200},
        )
        assert result.success, result.message
        expected.append(result.x)
    actual = project_simplex(values)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    assert (actual == 0).any()  # There is no interior probability floor.
    for x, p in zip(values, actual, strict=True):
        active = p > 0
        theta = (x - p)[active]
        np.testing.assert_allclose(theta, theta[0], atol=1e-13)
        assert np.all(x[~active] <= theta[0] + 1e-13)


def test_projection_translation_extremes_and_input_unchanged():
    values = np.array([[.3, -.7, .8], [-.1, -.1, -.1]])
    saved = values.copy()
    result = project_simplex(values)
    np.testing.assert_allclose(result, project_simplex(values + 16), atol=2e-15)
    np.testing.assert_array_equal(values, saved)
    large = np.finfo(np.float64).max
    actual = project_simplex(np.array([[large, -large], [large, large]]))
    np.testing.assert_array_equal(actual, [[1, 0], [.5, .5]])
    np.testing.assert_allclose(project_simplex(result), result, atol=2e-16)


def test_pg_preserves_zero_and_constant_gradient_rows_bitwise():
    q = np.array([[.1, .2, .7], [0, .4, .6000000000001], [.2, .3, .5]])
    gradient = np.array([[0., 0., 0.], [9., 9., 9.], [.2, -.4, .1]])
    saved_q, saved_g = q.copy(), gradient.copy()
    step = projected_gradient_step(q, gradient, .25)
    assert step.proposal[:2].tobytes() == q[:2].tobytes()
    np.testing.assert_array_equal(step.changed_rows, [False, False, True])
    np.testing.assert_array_equal(step.direction, step.proposal - q)
    assert step.gradient_dot_direction == np.sum(gradient * step.direction)
    assert step.gradient_dot_direction < 0
    np.testing.assert_array_equal(q, saved_q)
    np.testing.assert_array_equal(gradient, saved_g)
    assert projected_gradient_step(q, gradient, 0).proposal.tobytes() == q.tobytes()


def test_feasible_segment_and_projection_descent_inequality():
    q = np.array([[.1, .4, .5], [.2, .3, .5]])
    g = np.array([[4., -2., .5], [-1., 2., 1.]])
    step = projected_gradient_step(q, g, .13)
    assert step.gradient_dot_direction <= -np.square(step.direction).sum() / .13 + 1e-14
    for amount in [0, .2, .7, 1]:
        validate_simplex(q + amount * step.direction)
    np.testing.assert_array_equal(feasible_direction(q, step.proposal), step.direction)


def test_pg_removes_large_common_gradient_offset_before_forming_trial():
    q = np.array([[.1, .2, .7], [.2, .3, .5]])
    g = np.array([[0., 1., 2.], [2., 0., 1.]])
    huge_common_offset = float(2**50)
    first = projected_gradient_step(q, g, .125)
    shifted = projected_gradient_step(q, g + huge_common_offset, .125)
    np.testing.assert_array_equal(first.proposal, shifted.proposal)
    # This check concerns the proposal, not roundoff in a large-offset dot product.


def test_lmo_ties_gap_scaling_and_actual_direction():
    q = np.array([[.2, .3, .5], [0, 1., 0]])
    g = np.array([[3., -2., -2.], [4., 1., 3.]])
    np.testing.assert_array_equal(linear_minimization_oracle(g), [[0, 1, 0], [0, 1, 0]])
    gap = frank_wolfe_gap(q, g)
    np.testing.assert_allclose(gap.per_row_gap, [1, 0], atol=1e-15)
    assert gap.total_gap == pytest.approx(1)
    assert gap.total_gap == -np.sum(g * gap.direction)
    np.testing.assert_allclose(gap.centered_per_row_gap, gap.per_row_gap, atol=1e-15)
    assert frank_wolfe_gap(q, 7 * g).total_gap == pytest.approx(7 * gap.total_gap)


def test_near_simplex_negative_directional_roundoff_not_hidden():
    q = np.array([[.5, .5 - 1e-13]])
    validate_simplex(q)
    gap = frank_wolfe_gap(q, np.ones_like(q))
    assert gap.total_gap < 0
    assert gap.centered_total_gap == 0
    assert 0 < gap.max_simplex_sum_error < 1e-12
    assert projected_gradient_step(q, np.ones_like(q), 1).proposal.tobytes() == q.tobytes()


def _mixture_problem():
    # Known fixed contributions, residual class-0 background, and affine noise.
    weights = np.array([[.5, .3, .1], [.7, .1, .1], [.1, .6, .2],
                        [.1, .2, .65], [.3, .2, .2], [0., 0., 0.]])
    labels = np.array([0, 1, 1, 2, 2, 2])
    class_weights = np.array([.5, 1.1, 1.7])
    delta = 5e-7

    def objective(q):
        raw = weights @ q
        raw[:, 0] += 1 - weights.sum(axis=1)
        p = (1 - delta) * raw[np.arange(len(labels)), labels] + delta / 3
        loss = -np.mean(class_weights[labels] * np.log(p))
        gradient = np.zeros_like(q)
        for ray, label in enumerate(labels):
            gradient[:, label] -= (class_weights[label] * (1 - delta)
                                   * weights[ray] / p[ray] / len(labels))
        return float(loss), gradient

    return objective


def test_affine_mixture_ce_exact_gradient_and_projected_step():
    objective = _mixture_problem()
    q = np.array([[.2, .3, .5], [.3, .5, .2], [.4, .1, .5]])
    initial, gradient = objective(q)
    epsilon = 1e-6
    for row in range(3):
        for cls in range(3):
            direction = np.zeros_like(q)
            direction[row, cls] = epsilon
            finite_difference = (objective(q + direction)[0]
                                 - objective(q - direction)[0]) / (2 * epsilon)
            assert finite_difference == pytest.approx(gradient[row, cls], abs=2e-9, rel=1e-6)
    step = projected_gradient_step(q, gradient, .05)
    assert objective(step.proposal)[0] <= initial + 1e-4 * step.gradient_dot_direction
    assert step.gradient_dot_direction < 0


def test_mixture_ce_scipy_optimum_and_fw_bound():
    objective = _mixture_problem()
    q = np.full((3, 3), 1 / 3)
    constraints = {"type": "eq", "fun": lambda x: x.reshape(3, 3).sum(axis=1) - 1,
                   "jac": lambda x: np.kron(np.eye(3), np.ones((1, 3)))}
    result = minimize(
        lambda x: objective(x.reshape(3, 3))[0], q.ravel(),
        jac=lambda x: objective(x.reshape(3, 3))[1].ravel(),
        method="SLSQP", bounds=[(0, 1)] * 9, constraints=constraints,
        options={"ftol": 1e-13, "maxiter": 500},
    )
    assert result.success, result.message
    optimum = result.x.reshape(3, 3)
    validate_simplex(optimum)
    f_star, g_star = objective(optimum)
    assert frank_wolfe_gap(optimum, g_star).total_gap < 1e-6
    for step_size in [0, .05, .2]:
        trial = projected_gradient_step(q, objective(q)[1], step_size).proposal
        value, gradient = objective(trial)
        gap = frank_wolfe_gap(trial, gradient).total_gap
        assert value - f_star >= -1e-12
        assert value - f_star <= gap + 1e-12


def test_fw_bound_against_analytic_ce_optimum_including_boundary_q():
    frequencies = np.array([.2, .3, .5])
    delta = .03
    exact_q = ((frequencies - delta / 3) / (1 - delta))[None]
    def objective(q):
        p = (1 - delta) * q + delta / 3
        return float(-np.sum(frequencies * np.log(p))), -(1 - delta) * frequencies / p
    optimum, gradient = objective(exact_q)
    assert abs(frank_wolfe_gap(exact_q, gradient).total_gap) < 1e-14
    for q in [np.array([[1., 0, 0]]), np.array([[.1, .8, .1]])]:
        value, gradient = objective(q)
        assert value - frank_wolfe_gap(q, gradient).total_gap <= optimum + 1e-14


@pytest.mark.parametrize("bad", [np.ones((2, 3), np.float32), np.ones(3),
                                np.ones((0, 3)), np.ones((3, 1)),
                                np.array([[np.nan, 1.]]), np.array([[np.inf, 1.]])])
def test_invalid_matrix_rejected(bad):
    with pytest.raises(ValueError):
        project_simplex(bad)
    with pytest.raises(ValueError):
        linear_minimization_oracle(bad)


@pytest.mark.parametrize("bad", [np.array([[.5, .4]]), np.array([[-1e-16, 1.]]),
                                np.array([[1. + 1e-15, 0.]])])
def test_infeasible_q_not_repaired(bad):
    with pytest.raises(ValueError):
        projected_gradient_step(bad, np.zeros_like(bad), 1)
    with pytest.raises(ValueError):
        frank_wolfe_gap(bad, np.zeros_like(bad))


@pytest.mark.parametrize("step_size", [-1, np.nan, np.inf, [1], True])
def test_invalid_step_rejected(step_size):
    with pytest.raises(ValueError):
        projected_gradient_step(np.array([[.5, .5]]), np.array([[0., 1.]]), step_size)


def test_shape_and_arithmetic_overflow_fail_closed():
    q = np.array([[.5, .5]])
    with pytest.raises(ValueError, match="shapes"):
        frank_wolfe_gap(q, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="shapes"):
        feasible_direction(q, np.full((2, 2), .5))
    large = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="representable"):
        projected_gradient_step(q, np.array([[large, 0.]]), 2)
    with pytest.raises(ValueError, match="representable"):
        frank_wolfe_gap(q, np.array([[-large, large]]))
