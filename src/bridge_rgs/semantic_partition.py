"""Bounded reference for a density-preserving Gaussian semantic slab.

This is conditional-Gaussian half-space integration, as used by 3D-HGS
(https://arxiv.org/html/2406.02720v4, section 3.2, equations 8--9) and XClipGS
(https://arxiv.org/html/2608.07760v1, section 3.2). A slab is the difference
of two such half-spaces; neither this operator nor its semantic use is claimed
here as a demonstrated novel method or model improvement.

xi ~ N(0,I) is a *dimensionless*, Gaussian-local coordinate. P maps xi to
image-coordinate displacement; noise is covariance in those image units
squared (e.g. EWA antialiasing, not a calibrated pose covariance). b, w and tau
are in local standard-deviation units. Conditioning uses the linear EWA
observation delta = P xi + noise. It is not an exact perspective ray integral,
a surface intersection, or a change to the renderer's visibility model.

Only covariance/probability excursions within documented roundoff tolerances
are clamped. Invalid covariance, normal, endpoints or visibility mass fail.
The dense compositor is a small-reference calculation, not a production shader.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

MODES = ("integrated", "point", "marginal")


def _finite_tensor(value, name, like=None):
    if not isinstance(value, torch.Tensor):
        if like is None or isinstance(value, bool):
            raise ValueError(f"{name} must be a floating tensor")
        value = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64")
    if like is not None and (value.dtype != like.dtype or value.device != like.device):
        raise ValueError(f"{name} dtype/device must match P")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _roundoff_unit(value, name):
    # Absolute tolerance is appropriate for dimensionless probabilities and
    # conditional variances whose mathematical range is exactly [0,1].
    tol = 128 * torch.finfo(value.dtype).eps
    if not bool(((value >= -tol) & (value <= 1 + tol)).all()):
        raise ValueError(f"{name} outside [0,1] beyond roundoff")
    return value.clamp(0, 1)


def normal_interval_probability(lower, upper):
    """Phi(upper)-Phi(lower), avoiding cancellation of two CDFs near one.

Positive and negative tails use erfc; intervals straddling zero use erf.
Representational underflow in an extremely remote/narrow tail may still be
zero. Infinite endpoints are supported, NaN or reversed endpoints are not.
"""
    if (not isinstance(lower, torch.Tensor) or not isinstance(upper, torch.Tensor)
            or lower.dtype not in (torch.float32, torch.float64)
            or lower.dtype != upper.dtype or lower.device != upper.device):
        raise ValueError("Interval endpoints must share a floating dtype/device")
    lower, upper = torch.broadcast_tensors(lower, upper)
    if bool(torch.isnan(lower).any() | torch.isnan(upper).any() | (lower > upper).any()):
        raise ValueError("Invalid normal interval")
    root_two = math.sqrt(2.)
    positive = .5 * (torch.erfc(lower / root_two) - torch.erfc(upper / root_two))
    negative = .5 * (torch.erfc(-upper / root_two) - torch.erfc(-lower / root_two))
    central = .5 * (torch.erf(upper / root_two) - torch.erf(lower / root_two))
    result = torch.where(lower >= 0, positive, torch.where(upper <= 0, negative, central))
    return _roundoff_unit(result, "Normal interval mass")


@dataclass(frozen=True)
class PartitionCoefficients:
    slope: torch.Tensor  # [...,2]; n^T P^T C^-1, zero in marginal mode
    offset: torch.Tensor  # [...]; slab center b
    half_width: torch.Tensor  # [...]; w > 0
    standard_deviation: torch.Tensor  # [...]; sqrt(tau^2 + variance_used)
    conditional_variance: torch.Tensor  # [...]; n^T V n, including roundoff clamp
    variance_used: torch.Tensor  # conditional / zero / one by mode
    conditional_mean_map: torch.Tensor  # [...,3,2]
    conditional_covariance: torch.Tensor  # [...,3,3], roundoff may be slightly non-PSD
    mode: str


def prepared_coefficients(P, noise, normal, b, w, tau, mode="integrated"):
    """Prepare batched slab coefficients without explicit matrix inverses.

    P[...,2,3], noise[...,2,2], unit normal[...,3], and scalar fields b/w/tau
    broadcast over their leading dimensions. Tensor inputs must share dtype
    and device; Python scalar fields are converted to P's dtype/device.
    All three modes use the same parameters. ``point`` discards V;
    ``marginal`` is the *unconditional* constant E[g], with mean 0/variance 1.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown partition mode {mode!r}")
    P = _finite_tensor(P, "P")
    noise = _finite_tensor(noise, "noise", P)
    normal = _finite_tensor(normal, "normal", P)
    b, w, tau = (_finite_tensor(value, name, P)
                 for value, name in ((b, "b"), (w, "w"), (tau, "tau")))
    if P.shape[-2:] != (2, 3) or noise.shape[-2:] != (2, 2) or normal.shape[-1:] != (3,):
        raise ValueError("Require P[...,2,3], noise[...,2,2], normal[...,3]")
    try:
        batch = torch.broadcast_shapes(P.shape[:-2], noise.shape[:-2], normal.shape[:-1],
                                       b.shape, w.shape, tau.shape)
    except RuntimeError as exc:
        raise ValueError("Incompatible partition batch dimensions") from exc
    P, noise, normal = (P.expand(*batch, 2, 3), noise.expand(*batch, 2, 2),
                        normal.expand(*batch, 3))
    b, w, tau = (value.expand(batch) for value in (b, w, tau))
    eps = torch.finfo(P.dtype).eps
    if not bool((abs(normal.square().sum(-1) - 1) <= 64 * eps).all()):
        raise ValueError("normal must be unit length; normalization is not implicit")
    if not bool(((w > 0) & (tau > 0)).all()):
        raise ValueError("w and tau must be strictly positive")
    # Relative covariance validation, without an absolute +1 that would accept
    # an arbitrarily indefinite covariance merely because its units are tiny.
    magnitude = noise.abs().amax(dim=(-2, -1)).clamp_min(torch.finfo(P.dtype).tiny)
    if not bool(((noise - noise.mT).abs().amax(dim=(-2, -1)) <= 64 * eps * magnitude).all()):
        raise ValueError("noise covariance must be symmetric")
    noise = (noise + noise.mT) * .5
    if not bool((torch.linalg.eigvalsh(noise)[..., 0] >= -64 * eps * magnitude).all()):
        raise ValueError("noise covariance must be positive semidefinite")
    covariance = P @ P.mT + noise
    if not bool(torch.isfinite(covariance).all()):
        raise ValueError("Nonfinite observation covariance")
    chol, info = torch.linalg.cholesky_ex(covariance)
    if bool((info != 0).any()):
        raise ValueError("Observation covariance must be positive definite")
    solved = torch.cholesky_solve(P, chol)
    mean_map = solved.mT
    V = torch.eye(3, dtype=P.dtype, device=P.device) - P.mT @ solved
    V = (V + V.mT) * .5
    eig = torch.linalg.eigvalsh(V)
    if not bool(((eig[..., 0] >= -128 * eps) & (eig[..., -1] <= 1 + 128 * eps)).all()):
        raise ValueError("Invalid conditional covariance beyond roundoff")
    conditional_variance = _roundoff_unit(
        torch.einsum("...i,...ij,...j->...", normal, V, normal), "Conditional variance")
    slope = torch.einsum("...i,...ij->...j", normal, mean_map)
    if mode == "integrated":
        variance_used = conditional_variance
    elif mode == "point":
        variance_used = torch.zeros_like(conditional_variance)
    else:
        slope = torch.zeros_like(slope)
        variance_used = torch.ones_like(conditional_variance)
    std = torch.sqrt(tau.square() + variance_used)
    if not bool((torch.isfinite(std) & (std > 0)).all()):
        raise ValueError("Invalid effective slab standard deviation")
    return PartitionCoefficients(slope, b, w, std, conditional_variance, variance_used,
                                 mean_map, V, mode)


def evaluate_coefficients(delta, coeff):
    """Return integrated membership; e.g. delta[R,N,2] broadcasts coeff[N]."""
    if not isinstance(coeff, PartitionCoefficients):
        raise TypeError("Require prepared PartitionCoefficients")
    delta = _finite_tensor(delta, "delta", coeff.slope)
    if delta.shape[-1:] != (2,):
        raise ValueError("delta must end in two image coordinates")
    center = (delta * coeff.slope).sum(-1) - coeff.offset
    lower = (center - coeff.half_width) / coeff.standard_deviation
    upper = (center + coeff.half_width) / coeff.standard_deviation
    return normal_interval_probability(lower, upper)


def _simplex(value, name, like):
    value = _finite_tensor(value, name, like)
    if value.ndim < 1 or value.shape[-1] < 2:
        raise ValueError("Endpoint probabilities need at least two classes")
    # No endpoint clamp or renormalization: invalid categorical inputs fail.
    if not bool(((value >= 0) & (value <= 1)).all()):
        raise ValueError(f"{name} has invalid probability components")
    if not bool((abs(value.sum(-1) - 1) <= 64 * torch.finfo(value.dtype).eps).all()):
        raise ValueError(f"{name} must sum to one")
    return value


def mixprob(gate, q_in, q_out):
    """Convex endpoint mixture, without a second softmax or normalization."""
    gate = _roundoff_unit(_finite_tensor(gate, "gate"), "Gate")
    q_in, q_out = _simplex(q_in, "q_in", gate), _simplex(q_out, "q_out", gate)
    if q_in.shape[-1] != q_out.shape[-1]:
        raise ValueError("Endpoint class dimensions differ")
    return q_out + gate[..., None] * (q_in - q_out)


def render_reference(weights, residual_background, delta, *, coefficients, q_in, q_out):
    """Small dense reference with fixed full-scene W[R,N] and background mass.

    residual_background[R] is 1-alpha, assigned to class zero. All W entries,
    including small positive tails, are retained. No depth sorting, opacity
    normalization, density changes or independent semantic visibility occurs.
    ``coefficients`` must have one batch entry per Gaussian. Geometry-derived
    coefficients may be differentiable; fixed W/residual may not require grad.
    """
    weights = _finite_tensor(weights, "weights")
    residual_background = _finite_tensor(residual_background, "residual_background", weights)
    delta = _finite_tensor(delta, "delta", weights)
    if (weights.ndim != 2 or min(weights.shape) < 1
            or residual_background.shape != weights.shape[:1]
            or delta.shape != (*weights.shape, 2)):
        raise ValueError("Require weights[R,N], residual[R], delta[R,N,2]")
    if weights.requires_grad or residual_background.requires_grad:
        raise ValueError("Reference visibility weights and residual must be fixed")
    if not bool(((weights >= 0) & (weights <= 1)).all()
                & ((residual_background >= 0) & (residual_background <= 1)).all()):
        raise ValueError("Invalid visibility mass")
    alpha = weights.sum(-1)
    if not bool((abs(alpha + residual_background - 1)
                 <= 128 * torch.finfo(weights.dtype).eps).all()):
        raise ValueError("Visibility plus residual must sum to one; never renormalized")
    if not isinstance(coefficients, PartitionCoefficients):
        raise TypeError("Require prepared coefficients")
    if coefficients.slope.shape != (weights.shape[1], 2):
        raise ValueError("One coefficient batch entry per Gaussian required")
    q_in, q_out = _simplex(q_in, "q_in", weights), _simplex(q_out, "q_out", weights)
    if q_in.ndim != 2 or q_in.shape != q_out.shape or q_in.shape[0] != weights.shape[1]:
        raise ValueError("Endpoint probabilities must be [N,K]")
    gate = evaluate_coefficients(delta, coefficients)
    probabilities = torch.einsum("rn,rnk->rk", weights, mixprob(gate, q_in, q_out))
    background = torch.zeros_like(probabilities)
    background[:, 0] = residual_background
    return {"probabilities": probabilities + background, "alpha": alpha, "gate": gate}
