"""Synthetic-only contracts for the conditional Gaussian slab reference."""
import itertools

import numpy as np
import pytest
import torch
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
from scipy.special import ndtr

from bridge_rgs.semantic_partition import (
    evaluate_coefficients,
    mixprob,
    normal_interval_probability,
    prepared_coefficients,
    render_reference,
)

DT = torch.float64


def example(dtype=DT):
    return (torch.tensor([[1., .2, -.3], [.1, .8, .4]], dtype=dtype),
            torch.tensor([[.3, .04], [.04, .2]], dtype=dtype),
            torch.tensor([.6, 0., .8], dtype=dtype))


def test_closed_diagonal_conditioning_and_modes():
    P = torch.tensor([[2., 0, 0], [0, 3., 0]], dtype=DT)
    noise = torch.diag(torch.tensor([1., 4.], dtype=DT))
    normal = torch.tensor([1., 0, 0], dtype=DT)
    delta = torch.tensor([[.5, 1.], [-1., .7]], dtype=DT)
    for mode in ("integrated", "point", "marginal"):
        c = prepared_coefficients(P, noise, normal, .2, .6, .3, mode)
        torch.testing.assert_close(c.conditional_covariance,
                                   torch.diag(torch.tensor([.2, 4/13, 1.], dtype=DT)))
        variance = {"integrated": .2, "point": 0., "marginal": 1.}[mode]
        mean = delta[:, 0].numpy() * .4 if mode != "marginal" else np.zeros(2)
        sd = np.sqrt(.09 + variance)
        expected = ndtr((mean - .2 + .6)/sd) - ndtr((mean - .2 - .6)/sd)
        np.testing.assert_allclose(evaluate_coefficients(delta, c).numpy(), expected, atol=1e-15)


def test_multivariate_independent_quadrature_and_monte_carlo():
    P, noise, normal = example()
    delta = np.array([.8, -.35])
    p, s, n = P.numpy(), noise.numpy(), normal.numpy()
    # Independent NumPy Gaussian conditioning, followed by integration of the
    # *unintegrated* soft slab over all three local coordinates.
    C = p @ p.T + s
    mean = np.linalg.solve(C, p).T @ delta
    V = np.eye(3) - p.T @ np.linalg.solve(C, p)
    eigenvalues, vectors = np.linalg.eigh(V)
    root = vectors @ np.diag(np.sqrt(eigenvalues))
    b, w, tau = .25, .7, .6
    xs, ws = hermgauss(24)
    choices = np.array(list(itertools.product(range(len(xs)), repeat=3)))
    samples = np.sqrt(2) * xs[choices] @ root.T + mean
    weights = np.prod(ws[choices] / np.sqrt(np.pi), axis=1)
    z = samples @ n - b
    quadrature = np.dot(weights, ndtr((z+w)/tau)-ndtr((z-w)/tau))
    coeff = prepared_coefficients(P, noise, normal, b, w, tau)
    actual = evaluate_coefficients(torch.tensor(delta), coeff).item()
    assert actual == pytest.approx(quadrature, abs=2e-9)
    rng = np.random.default_rng(42)
    samples = rng.standard_normal((160_000, 3)) @ root.T + mean
    z = samples @ n - b
    monte_carlo = np.mean(ndtr((z+w)/tau)-ndtr((z-w)/tau))
    assert actual == pytest.approx(monte_carlo, abs=.002)


def test_integrating_over_image_observations_recovers_marginal_mass():
    P, noise, normal = example()
    C = (P @ P.T + noise).numpy()
    root = np.linalg.cholesky(C)
    x, w = hermgauss(32)
    ij = np.array(list(itertools.product(range(len(x)), repeat=2)))
    delta = torch.tensor(np.sqrt(2) * x[ij] @ root.T)
    weights = np.prod(w[ij]/np.sqrt(np.pi), axis=1)
    integrated = evaluate_coefficients(delta, prepared_coefficients(P, noise, normal, .4, .8, .7))
    marginal = evaluate_coefficients(delta, prepared_coefficients(P, noise, normal, .4, .8, .7,
                                                                  "marginal"))
    assert np.dot(weights, integrated.numpy()) == pytest.approx(marginal[0].item(), abs=2e-10)
    assert torch.equal(marginal, marginal[0].expand_as(marginal))


@pytest.mark.parametrize("mode", ["integrated", "point", "marginal"])
def test_image_rotation_scale_and_local_rotation_invariance(mode):
    P, noise, normal = example()
    delta = torch.tensor([[.5, -.4], [1.5, .3]], dtype=DT)
    coeff = prepared_coefficients(P, noise, normal, .2, .6, .3, mode)
    expected = evaluate_coefficients(delta, coeff)
    T = torch.tensor([[.6, -.8], [.8, .6]], dtype=DT) * 13.
    changed = prepared_coefficients(T @ P, T @ noise @ T.T, normal, .2, .6, .3, mode)
    torch.testing.assert_close(evaluate_coefficients(delta @ T.T, changed), expected,
                               atol=2e-14, rtol=2e-14)
    Q = torch.tensor([[0., 0, 1], [1., 0, 0], [0, 1., 0]], dtype=DT)
    local = prepared_coefficients(P @ Q.T, noise, Q @ normal, .2, .6, .3, mode)
    torch.testing.assert_close(evaluate_coefficients(delta, local), expected,
                               atol=2e-14, rtol=2e-14)


def test_normal_sign_and_center_flip_are_same_slab():
    P, noise, normal = example()
    d = torch.tensor([[2., -.3], [.4, .8]], dtype=DT)
    a = prepared_coefficients(P, noise, normal, .7, .4, .2)
    b = prepared_coefficients(P, noise, -normal, -.7, .4, .2)
    torch.testing.assert_close(evaluate_coefficients(d, a), evaluate_coefficients(d, b),
                               atol=1e-15, rtol=1e-14)


def test_slab_represents_background_cable_background_without_changing_density():
    P = torch.tensor([[[1., 0, 0], [0, 1., 0]]], dtype=DT)
    coeff = prepared_coefficients(P, torch.zeros(1, 2, 2, dtype=DT),
                                  torch.tensor([[1., 0, 0]], dtype=DT), 0., .5, .05)
    delta = torch.tensor([[[-1., 0]], [[0., 0]], [[1., 0]]], dtype=DT)
    W = torch.ones(3, 1, dtype=DT)
    out = torch.tensor([[1., 0, 0, 0, 0]], dtype=DT)
    inside = torch.tensor([[0., 0, 1., 0, 0]], dtype=DT)
    result = render_reference(W, torch.zeros(3, dtype=DT), delta,
                              coefficients=coeff, q_in=inside, q_out=out)
    assert result["probabilities"].argmax(-1).tolist() == [0, 2, 0]
    assert torch.equal(result["alpha"], torch.ones(3, dtype=DT))
    torch.testing.assert_close(result["probabilities"].sum(-1), result["alpha"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_reference_equal_endpoints_identity_tiny_tails_and_no_mutation(dtype):
    P, noise, normal = example(dtype)
    coeff = prepared_coefficients(P.expand(2, 2, 3), noise, normal, 0., .5, .2)
    W = torch.tensor([[.2, .5], [0., 0.], [1e-12, .7]], dtype=dtype)
    residual = 1 - W.sum(-1)
    delta = torch.arange(12, dtype=dtype).reshape(3, 2, 2) * .1
    q = torch.tensor([[.1, .2, .3, .15, .25], [.4, .1, .2, .1, .2]], dtype=dtype)
    saved = [x.clone() for x in (W, residual, delta, q)]
    result = render_reference(W, residual, delta, coefficients=coeff, q_in=q, q_out=q)
    expected = W @ q
    expected[:, 0] += residual
    torch.testing.assert_close(result["probabilities"], expected, atol=1e-7 if dtype == torch.float32
                               else 1e-15, rtol=1e-6 if dtype == torch.float32 else 1e-14)
    assert result["gate"].shape == (3, 2)
    assert torch.equal(result["alpha"], W.sum(-1))
    assert result["probabilities"][1].tolist() == [1., 0, 0, 0, 0]
    for value, before in zip((W, residual, delta, q), saved, strict=True):
        assert torch.equal(value, before)
    assert torch.equal(mixprob(result["gate"], q, q), q.expand(3, 2, 5))


@pytest.mark.parametrize("lower,upper", [(8., 8.3), (-8.3, -8.), (10., 10.0001), (-.01, .02)])
def test_stable_tail_probability_against_independent_density_quadrature(lower, upper):
    expected = quad(lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi), lower, upper,
                    epsabs=1e-100, epsrel=1e-12)[0]
    actual = normal_interval_probability(torch.tensor(lower, dtype=DT),
                                         torch.tensor(upper, dtype=DT)).item()
    assert actual > 0
    assert actual == pytest.approx(expected, rel=2e-10)
    if lower >= 8:
        # Direct FP64 subtraction can already lose almost all significant digits.
        assert abs(actual-expected) < abs(float(ndtr(upper)-ndtr(lower))-expected)


def test_infinite_interval_and_no_projection_information():
    actual = normal_interval_probability(torch.tensor(-float("inf"), dtype=DT),
                                         torch.tensor(float("inf"), dtype=DT))
    assert actual.item() == 1
    P = torch.zeros(2, 3, dtype=DT)
    n = torch.tensor([1., 0, 0], dtype=DT)
    delta = torch.tensor([[1., 2.], [30., -4.]], dtype=DT)
    integrated = prepared_coefficients(P, torch.eye(2, dtype=DT), n, .3, .7, .2)
    marginal = prepared_coefficients(P, torch.eye(2, dtype=DT), n, .3, .7, .2, "marginal")
    assert torch.equal(evaluate_coefficients(delta, integrated), evaluate_coefficients(delta, marginal))


@pytest.mark.parametrize("mode", ["integrated", "point", "marginal"])
def test_float64_gradcheck_through_projection_covariance_and_semantics(mode):
    P, _, normal = example()
    L = torch.tensor([[.5, .02], [.03, .4]], dtype=DT)
    delta = torch.tensor([[.2, -.3], [.5, .7]], dtype=DT)
    scalar = torch.tensor([.1, .7, .4], dtype=DT)
    logits = torch.tensor([[.2, -.1, .4], [-.3, .2, .1]], dtype=DT)
    arguments = tuple(x.clone().requires_grad_() for x in (P, L, normal, scalar, delta, logits))

    def function(p, l, n, s, d, z):
        c = prepared_coefficients(p, l @ l.T, n / n.norm(), s[0], s[1], s[2], mode)
        g = evaluate_coefficients(d, c)
        return mixprob(g, z[0].softmax(-1), z[1].softmax(-1))

    assert torch.autograd.gradcheck(function, arguments, eps=1e-6, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("case", ["noise_negative", "noise_asymmetric", "singular", "normal",
                                  "width", "tau", "nan", "mode", "dtype"])
def test_invalid_covariance_and_parameters_fail(case):
    P, noise, normal = example()
    width, tau, mode = .5, .2, "integrated"
    if case == "noise_negative":
        noise = torch.diag(torch.tensor([.1, -1e-3], dtype=DT))
    elif case == "noise_asymmetric":
        noise[0, 1] += .01
    elif case == "singular":
        P = torch.zeros_like(P); noise = torch.zeros_like(noise)
    elif case == "normal":
        normal = normal * 2
    elif case == "width":
        width = 0.
    elif case == "tau":
        tau = -1.
    elif case == "nan":
        P[0, 0] = float("nan")
    elif case == "mode":
        mode = "unknown"
    else:
        noise = noise.float()
    with pytest.raises(ValueError):
        prepared_coefficients(P, noise, normal, 0., width, tau, mode)


def test_invalid_endpoints_or_visibility_fail_without_renormalization():
    P, noise, n = example()
    c = prepared_coefficients(P[None], noise, n, 0., .4, .2)
    W = torch.tensor([[.6]], dtype=DT)
    delta = torch.zeros(1, 1, 2, dtype=DT)
    q = torch.tensor([[.3, .7]], dtype=DT)
    for bad in (torch.tensor([[-.1, 1.1]], dtype=DT), q*.9):
        with pytest.raises(ValueError):
            mixprob(torch.tensor(.5, dtype=DT), q, bad)
    with pytest.raises(ValueError):
        render_reference(W, torch.tensor([.3], dtype=DT), delta,
                         coefficients=c, q_in=q, q_out=q)
    with pytest.raises(ValueError):
        render_reference(W.clone().requires_grad_(), torch.tensor([.4], dtype=DT), delta,
                         coefficients=c, q_in=q, q_out=q)
