"""FP64 product-simplex primitives for isolated convex optimization diagnostics.

No renderer, target loading, optimizer state, probability floor, or objective
normalization lives here. A Frank--Wolfe gap is an optimality bound only for the
same complete differentiable convex objective whose exact gradient is supplied.
Floating-point outputs are diagnostics, not interval-certified bounds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIMPLEX_ATOL = 1e-12


@dataclass(frozen=True)
class ProjectedStep:
    proposal: np.ndarray
    direction: np.ndarray
    gradient_dot_direction: float
    changed_rows: np.ndarray


@dataclass(frozen=True)
class FrankWolfeGap:
    vertex: np.ndarray
    direction: np.ndarray
    per_row_gap: np.ndarray
    total_gap: float
    centered_per_row_gap: np.ndarray
    centered_total_gap: float
    max_simplex_sum_error: float


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must have dtype float64; conversion is the caller's choice")
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        raise ValueError(f"{name} must have nonempty shape [N, K], K >= 2")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_simplex(q: np.ndarray) -> np.ndarray:
    """Validate, without copying or repairing, FP64 rows in the unit simplex.

    Zero probabilities are allowed. Row sums tolerate only FP64 roundoff
    (absolute 1e-12); negative entries are never silently clipped.
    """
    q = _matrix(q, "q")
    if (q < 0).any() or (q > 1).any():
        raise ValueError("q must lie in [0, 1] without a probability floor")
    if not np.allclose(q.sum(axis=1), 1.0, rtol=0, atol=SIMPLEX_ATOL):
        raise ValueError("q rows must sum to one; no implicit normalization is performed")
    return q


def _same_shape(q: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = validate_simplex(q)
    gradient = _matrix(gradient, "gradient")
    if gradient.shape != q.shape:
        raise ValueError("q and gradient shapes must match")
    return q, gradient


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean projection onto {q >= 0, sum(q) = 1}, in FP64.

    The sorted-threshold solution uses translation by each row maximum.
    Values more than one below that maximum are necessarily inactive, so
    replacing their shifted values by -1 is exact in real arithmetic. This
    also prevents overflow for opposite-sign, finite FP64 extremes. It is
    not a floor on the output probabilities, which may be exactly zero.
    """
    values = _matrix(values, "values")
    with np.errstate(over="ignore"):
        shifted = values - values.max(axis=1, keepdims=True)
    shifted = np.maximum(shifted, -1.0)
    ordered = np.sort(shifted, axis=1)[:, ::-1]
    count = np.arange(1, values.shape[1] + 1, dtype=np.float64)
    thresholds = (np.cumsum(ordered, axis=1) - 1.0) / count
    support_size = (ordered > thresholds).sum(axis=1)
    theta = thresholds[np.arange(len(values)), support_size - 1]
    projected = np.maximum(shifted - theta[:, None], 0.0)
    validate_simplex(projected)
    return projected


def feasible_direction(q: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return candidate - q; the entire segment is feasible in real arithmetic."""
    q, candidate = validate_simplex(q), validate_simplex(candidate)
    if q.shape != candidate.shape:
        raise ValueError("q and candidate shapes must match")
    return candidate - q


def projected_gradient_step(
    q: np.ndarray, gradient: np.ndarray, step_size: float
) -> ProjectedStep:
    """Propose Euclidean PG; the caller must evaluate its full objective.

    A nonnegative finite scalar step is required. Zero-step and exactly
    constant-gradient rows (including all-zero rows) retain q byte-for-byte.
    The reported dot product uses the actual FP64 proposal displacement and
    the supplied gradient, without normalization or descent clamping.
    """
    q, gradient = _same_shape(q, gradient)
    if isinstance(step_size, (bool, np.bool_)) or np.ndim(step_size) != 0:
        raise ValueError("step_size must be a finite nonnegative scalar")
    try:
        step_size = float(step_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("step_size must be a finite nonnegative scalar") from exc
    if not np.isfinite(step_size) or step_size < 0:
        raise ValueError("step_size must be a finite nonnegative scalar")
    proposal = q.copy()
    active = np.any(gradient != gradient[:, :1], axis=1) & (step_size > 0)
    if active.any():
        with np.errstate(over="ignore", invalid="ignore"):
            tangent = gradient[active] - gradient[active].min(axis=1, keepdims=True)
            trial = q[active] - step_size * tangent
        if not np.isfinite(tangent).all():
            raise ValueError("per-row gradient range is not representable in FP64")
        if not np.isfinite(trial).all():
            raise ValueError("q - step_size * centered gradient is not representable in FP64")
        proposal[active] = project_simplex(trial)
    direction = feasible_direction(q, proposal)
    with np.errstate(over="ignore", invalid="ignore"):
        dot = float(np.sum(gradient * direction, dtype=np.float64))
    if not np.isfinite(dot):
        raise ValueError("gradient dot direction is not representable in FP64")
    return ProjectedStep(proposal, direction, dot, np.any(proposal != q, axis=1))


def linear_minimization_oracle(gradient: np.ndarray) -> np.ndarray:
    """Minimize <gradient, s> over each simplex; ties use the first class."""
    gradient = _matrix(gradient, "gradient")
    vertex = np.zeros_like(gradient)
    vertex[np.arange(len(gradient)), np.argmin(gradient, axis=1)] = 1.0
    return vertex


def frank_wolfe_gap(q: np.ndarray, gradient: np.ndarray) -> FrankWolfeGap:
    """Return sum_i <g_i, q_i - argmin_s <g_i,s>> with no hidden scaling.

    The primary gap is the actual FP64 directional dot product, including
    any negative roundoff (never clamped). The additional centered gap uses
    sum_k q_ik * (g_ik - min_k g_ik) to expose cancellation from common row
    offsets. The two expressions agree for exact simplex rows in exact
    arithmetic. Approximate feasibility and approximate/stochastic renderer
    gradients do not certify the true objective's suboptimality.
    """
    q, gradient = _same_shape(q, gradient)
    vertex = linear_minimization_oracle(gradient)
    direction = feasible_direction(q, vertex)
    with np.errstate(over="ignore", invalid="ignore"):
        per_row = np.sum(-gradient * direction, axis=1, dtype=np.float64)
        total = float(np.sum(per_row, dtype=np.float64))
        shifted = gradient - gradient.min(axis=1, keepdims=True)
        centered = np.sum(q * shifted, axis=1, dtype=np.float64)
        centered_total = float(np.sum(centered, dtype=np.float64))
    if (not np.isfinite(per_row).all() or not np.isfinite(total)
            or not np.isfinite(centered).all() or not np.isfinite(centered_total)):
        raise ValueError("Frank-Wolfe gap is not representable in FP64")
    return FrankWolfeGap(
        vertex, direction, per_row, total, centered, centered_total,
        float(np.max(np.abs(q.sum(axis=1) - 1.0))),
    )
