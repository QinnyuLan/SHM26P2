"""Diagnostic raw-probability assignment math; not part of the production trainer."""
from __future__ import annotations

import numpy as np
import torch


def affine_raw_ce(raw, target, valid, class_weights, delta=5e-7):
    """Per-view mean weighted CE on positive affine-noise raw probabilities.

    No clamp/normalization; residual background and uniform noise stay fixed.
    The FP64 loss retains the original raw tensor's gradient path.
    """
    keep = valid.bool() & (target >= 0) & (target < 5)
    if raw.shape != (*target.shape, 5) or valid.shape != target.shape or not 0 < delta < 1:
        raise ValueError('Invalid raw objective grid/noise')
    if not bool(keep.any()) or not bool(torch.isfinite(raw).all()) or bool((raw < 0).any()):
        raise ValueError('Finite nonnegative raw probabilities and valid pixels required')
    labels = target[keep].long()
    values = raw[keep].double().gather(1, labels[:, None])[:, 0]
    noisy = (1-delta)*values+delta/5
    weights = class_weights.double()[labels]
    return -(noisy.log()*weights).mean()


def waterfill_counts(counts, qold, epsilon=1e-5):
    """Exact bounded-simplex multinomial M-step, with unchanged all-zero rows."""
    counts, qold = np.asarray(counts, np.float64), np.asarray(qold, np.float64)
    if (counts.ndim != 2 or counts.shape[1] != 5 or qold.shape != counts.shape
            or not np.isfinite(counts).all() or not np.isfinite(qold).all()
            or (counts < 0).any() or (qold < 0).any() or not 0 < epsilon < .2):
        raise ValueError('Invalid counts/actual qold or epsilon')
    active_rows = counts.sum(-1) > 0
    output = qold.copy()
    M = counts[active_rows]
    free = np.ones_like(M, bool)
    result = np.full_like(M, epsilon)
    for _ in range(5):
        remaining = 1-epsilon*(5-free.sum(-1))
        mass = (M*free).sum(-1)
        proposal = np.divide(M*remaining[:, None], mass[:, None], out=np.zeros_like(M), where=mass[:, None] > 0)
        violating = free & (proposal < epsilon)
        result[free] = proposal[free]
        result[violating] = epsilon
        if not violating.any():
            break
        free[violating] = False
    if (result < epsilon).any() or not np.allclose(result.sum(-1), 1., atol=1e-12, rtol=0):
        raise ValueError('Water-filling did not produce a valid bounded simplex')
    output[active_rows] = result
    return output, active_rows


def realize_probabilities(current, weight, bias, target):
    """Minimum-norm delta for a fixed full-rank decoder; target is already interior."""
    f = torch.as_tensor(current, dtype=torch.float64, device='cpu')
    W = torch.as_tensor(weight, dtype=torch.float64, device='cpu')
    b = torch.as_tensor(bias, dtype=torch.float64, device='cpu')
    q = torch.as_tensor(target, dtype=torch.float64, device='cpu')
    if (q.shape != (len(f), 5) or W.shape != (5, f.shape[1]) or b.shape != (5,)
            or not all(bool(torch.isfinite(v).all()) for v in (f, W, b, q))
            or bool((q <= 0).any()) or not torch.allclose(q.sum(-1), torch.ones(len(q), dtype=q.dtype), atol=1e-10, rtol=0)):
        raise ValueError('Strictly positive simplex targets and finite features required')
    D = W[1:]-W[0]
    if int(torch.linalg.matrix_rank(D)) != 4:
        raise ValueError('Decoder class difference matrix must have rank4')
    residual = (q[:, 1:].log()-q[:, :1].log())-(f @ D.T+b[1:]-b[0])
    difference = residual @ torch.linalg.pinv(D).T
    candidate = (f+difference).float()
    return candidate, {'delta_rms': float(difference.square().mean().sqrt()) if difference.numel() else 0.,
                       'delta_max_abs': float(difference.abs().max()) if difference.numel() else 0.,
                       'delta_l2_quantiles': np.quantile(difference.norm(dim=-1).numpy(), [0, .5, 1]).tolist() if len(f) else None}
