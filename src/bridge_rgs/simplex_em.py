"""Standard Jensen/EM M-step for fixed nonnegative linear categorical mixtures.

The caller supplies the complete gradient of the fixed weighted negative log
likelihood, including fixed background/noise in its denominator. No probability
normalization, clamp, adaptive weight, or regularizer may be hidden in that
objective if the usual exact-arithmetic monotonicity guarantee is invoked.
This FP64 helper does not certify an approximate renderer gradient or a later
FP32 cast: the actual full objective must still be checked by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simplex_optimization import validate_simplex


@dataclass(frozen=True)
class EMStep:
    proposal: np.ndarray
    responsibilities: np.ndarray
    row_responsibility: np.ndarray
    zero_responsibility_rows: np.ndarray
    changed_rows: np.ndarray


def em_step(q: np.ndarray, gradient: np.ndarray) -> EMStep:
    """Return u/U for u=q*(-gradient), retaining every U=0 row exactly.

    Inputs must be FP64; q is validated with the existing 1e-12 row-sum
    tolerance but never repaired. All gradient entries must be nonpositive,
    including on q=0 entries: small positive values are rejected, not clipped.
    Zero q entries remain zero. A zero-responsibility row need not be unseen;
    zero locking can suppress all useful classes while its FW gap is positive.

    Positive subnormal responsibilities/proposals and underflow to zero are
    deliberately rejected rather than silently changing their support. This
    conservative numeric domain is not a probability floor: exact zeros are
    allowed and no positive values are raised. Overflow is likewise rejected.
    Neither input is modified. No convergence or optimality claim is returned.
    """
    q = validate_simplex(q)
    gradient = np.asarray(gradient)
    if (gradient.dtype != np.dtype(np.float64) or gradient.shape != q.shape
            or not np.isfinite(gradient).all() or (gradient > 0).any()):
        raise ValueError("gradient must be finite FP64, q-shaped, and nonpositive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        counts = q * (-gradient)
        totals = counts.sum(axis=1, dtype=np.float64)
    if not np.isfinite(counts).all() or not np.isfinite(totals).all():
        raise ValueError("responsibilities are not representable in FP64")
    positive_product = (q > 0) & (gradient < 0)
    if np.any(positive_product & (counts < np.finfo(np.float64).tiny)):
        raise ValueError("positive responsibility underflows the supported FP64 normal range")
    zero_rows = totals == 0
    proposal = q.copy()
    with np.errstate(under="ignore", invalid="ignore", divide="ignore"):
        proposal[~zero_rows] = counts[~zero_rows] / totals[~zero_rows, None]
    if np.any((counts > 0) & (proposal < np.finfo(np.float64).tiny)):
        raise ValueError("positive normalized responsibility underflows the FP64 normal range")
    validate_simplex(proposal)
    return EMStep(proposal, counts, totals, zero_rows, np.any(proposal != q, axis=1))
