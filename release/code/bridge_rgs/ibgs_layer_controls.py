"""Optional fixed-input layer controls; never enabled by the production pipeline."""
from __future__ import annotations

import torch


@torch.no_grad()
def permute_supported_layer_features(features, w, q):
    """Each active layer reads the next active layer's entire feature vector.

    features [N,4 layers,4 sources,7], w [N,4], q [N,4,4] share floating
    dtype/device. Active means actual-dtype w*q>0, as in the fusion module.
    For each pixel/source, a0<a1<... are its active layer indices and
    output[a_k] = input[a_(k+1 mod count)]. Empty/singleton groups are no-ops.
    Inactive values, including NaNs and signed zeros, are copied unchanged.
    The returned tensor is detached and owns new storage. No coefficient,
    support, camera, ID, depth, or head parameter is changed or inferred.
    """
    if not isinstance(features, torch.Tensor) or features.ndim != 4 or features.shape[1:] != (4, 4, 7):
        raise ValueError('Expected features [N,4,4,7]')
    n = features.shape[0]
    if not all(isinstance(v, torch.Tensor) for v in (w, q)) or w.shape != (n, 4) or q.shape != (n, 4, 4):
        raise ValueError('Expected w [N,4] and q [N,4,4]')
    if not features.is_floating_point() or not all(v.dtype == features.dtype and v.device == features.device for v in (w, q)):
        raise ValueError('Features and coefficients must share floating dtype/device')
    w, q = w.detach(), q.detach()
    if not bool(torch.isfinite(w).all() and torch.isfinite(q).all()
                and (w >= 0).all() and (w.sum(-1) <= 1+1e-5).all()
                and (q >= 0).all() and (q <= 1).all()):
        raise ValueError('Invalid original mass or source support')
    active = w[..., None]*q > 0
    identity = torch.arange(4, device=features.device).reshape(1, 4, 1).expand(n, 4, 4)
    index = identity.clone()
    found = ~active
    for offset in range(1, 5):
        candidate = (identity+offset) % 4
        take = ~found & torch.gather(active, 1, candidate)
        index = torch.where(take, candidate, index)
        found = found | take
    return torch.gather(features.detach(), 1, index[..., None].expand(n, 4, 4, 7))
