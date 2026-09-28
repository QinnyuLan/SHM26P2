"""Experimental fixed-field layer evidence readout; no demonstrated quality gain.

The caller supplies four layers and four TRAIN sources. Source RGB is a mixed
observation, not an estimate of an individual Gaussian's radiance. Selection
and visibility policy are deliberately outside this module.
"""
from __future__ import annotations

import torch
from torch import nn


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def aggregate_layer_evidence(mlp, features, weights, support, *, reduction='mass'):
    """Sum original target contribution mass; divide by four source slots only.

    features: [pixels, layers=4, sources=4, 7], containing source-minus-base
    RGB, camera displacement, and ray cosine. weights: [pixels,4], original
    alpha*T, never renormalized. support: [pixels,4,4] in [0,1]. Missing slots
    use zero mass/support. Only the MLP parameters receive gradients.
    ``normalized`` is an explicit ablation: divide by total supported mass.
    """
    _require(features.ndim == 4 and features.shape[1:] == (4, 4, 7),
             'Expected four layers, four sources and seven features')
    _require(reduction in ('mass', 'normalized'), 'Unknown evidence reduction')
    n = features.shape[0]
    _require(weights.shape == (n, 4) and support.shape == (n, 4, 4),
             'Layer mass/support shape mismatch')
    _require(features.is_floating_point() and
             all(v.device == features.device and v.dtype == features.dtype for v in (weights, support)),
             'Features, mass and support must share floating dtype/device')
    weights, support, features = weights.detach(), support.detach(), features.detach()
    _require(torch.isfinite(weights).all() and (weights >= 0).all()
             and (weights.sum(-1) <= 1+1e-5).all(), 'Invalid original contribution mass')
    _require(torch.isfinite(support).all() and ((support >= 0) & (support <= 1)).all(),
             'Support must lie in [0,1]')
    mass = weights[..., None]*support
    active = mass > 0
    safe = torch.where(active[..., None], features, torch.zeros_like(features))
    _require(torch.isfinite(safe).all(), 'Nonfinite active evidence')
    encoded = mlp(safe.reshape(-1, 7)).reshape(n, 4, 4, -1)
    # Mask after the MLP as well: phi(0) generally contains learned biases.
    numerator = (encoded*mass[..., None]).sum(dim=(1, 2))
    total = mass.sum(dim=(1, 2))
    if reduction == 'mass':
        pooled = numerator/4
    else:
        # Safe denominator only for exactly empty support; no mass floor.
        denominator = torch.where(total > 0, total, torch.ones_like(total))
        pooled = numerator/denominator[:, None]
    return pooled, active.any(dim=(1, 2)), total/4


class MassPreservingLayerFusion(nn.Module):
    """Reuse an IBGS MLP/CNN with unchanged trainable parameter count.

    All inputs are detached. Pixels without any positive supported contribution
    return their base color exactly, even when the spatial CNN has bias or
    neighboring evidence. At supported pixels, mass weighting does NOT impose
    a residual magnitude or Lipschitz bound on the nonlinear CNN.
    """

    def __init__(self, backbone, *, reduction='mass'):
        super().__init__()
        _require(all(hasattr(backbone, key) for key in
                     ('height', 'width', 'per_view_mlp', 'conv_decoder', 'per_view_feat_dim')),
                 'An IBGS-compatible residual backbone is required')
        _require(reduction in ('mass', 'normalized'), 'Unknown evidence reduction')
        self.backbone = backbone
        self.reduction = reduction

    def forward(self, features, weights, support, ray, base):
        h, w = self.backbone.height, self.backbone.width
        _require(features.shape[0] == h*w and ray.shape == base.shape == (h*w, 3),
                 'Full image grid mismatch')
        _require(all(v.device == features.device and v.dtype == features.dtype for v in (ray, base))
                 and torch.isfinite(ray).all() and torch.isfinite(base).all(),
                 'Finite base/ray on the evidence grid required')
        pooled, active, mass = aggregate_layer_evidence(self.backbone.per_view_mlp,
                                                      features, weights, support,
                                                      reduction=self.reduction)
        _require(pooled.shape[-1] == self.backbone.per_view_feat_dim, 'MLP feature width changed')
        grid = torch.cat((pooled, ray.detach(), base.detach()), -1).T.reshape(1, -1, h, w)
        residual = self.backbone.conv_decoder(grid).squeeze(0).permute(1, 2, 0).reshape(h*w, 3)
        _require(residual.shape == base.shape, 'CNN must return three residual channels')
        residual = torch.where(active[:, None], residual, torch.zeros_like(residual))
        return {'image_pred': base.detach()+residual, 'residual': residual,
                'supported_mass': mass, 'active': active}
