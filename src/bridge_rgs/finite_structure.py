"""Finite, constant-count structure actions for controlled experiments.

This is a standard support-constrained split plus explicit donor deletion.
It does not select actions, fit attributes, preserve rendered RGB/opacity,
or claim a new split formula. It never changes a live scene or optimizer.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .densification import quaternion_to_matrix, support_constrained_split_geometry


@dataclass(frozen=True)
class StructureExchange:
    tensors: dict[str, Tensor]
    source_indices: Tensor
    retained_count: int
    parent_indices: Tensor
    donor_indices: Tensor
    offsets: Tensor
    shrink_axes: Tensor


@torch.no_grad()
def split_exchange(
    parameters: Mapping[str, Tensor],
    parents: Tensor,
    donors: Tensor,
    directions: Tensor | None = None,
    *,
    scale_shrink: float = 1.6,
    minimum_axis_fraction: float = .25,
) -> StructureExchange:
    """Replace K parents by 2K children and delete exactly K named donors.

    Unaffected rows retain their order; negative-offset children precede
    positive-offset children. ``source_indices`` also transports categorical
    probabilities or other per-Gaussian attributes without changing values.
    No opacity threshold, extra pruning, nearest-neighbor lookup or selection
    is implicit. The caller must measure full-image RGB and semantic effects.
    """
    required = {'means', 'quats', 'log_scales', 'opacity_logits'}
    if not required.issubset(parameters):
        raise ValueError(f'Missing geometry parameters: {required - set(parameters)}')
    means = parameters['means']
    if (not isinstance(means, Tensor) or means.ndim != 2 or means.shape[1] != 3
            or means.dtype not in (torch.float32, torch.float64)):
        raise ValueError('Expected floating [N,3] means')
    count = len(means)
    if count == 0 or not math.isfinite(scale_shrink) or scale_shrink <= 1:
        raise ValueError('Require nonempty geometry and finite scale_shrink > 1')
    if not 0 < minimum_axis_fraction <= 1:
        raise ValueError('minimum_axis_fraction must lie in (0,1]')
    for name, value in parameters.items():
        if (not isinstance(value, Tensor) or value.ndim == 0 or len(value) != count
                or value.device != means.device or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Invalid finite, same-device per-Gaussian tensor: {name}')
    if (parameters['quats'].shape != (count, 4)
            or parameters['log_scales'].shape != (count, 3)
            or parameters['opacity_logits'].shape not in ((count,), (count, 1))
            or any(parameters[k].dtype != means.dtype for k in required)):
        raise ValueError('Invalid geometry shapes or dtypes')
    quaternion_norms = parameters['quats'].norm(dim=-1)
    if not bool(torch.isfinite(quaternion_norms).all()) or bool((quaternion_norms <= 1e-12).any()):
        raise ValueError('Quaternion norms must be finite and exceed 1e-12')
    for indices in (parents, donors):
        if (indices.ndim != 1 or indices.dtype != torch.int64 or indices.device != means.device
                or bool(((indices < 0) | (indices >= count)).any())
                or indices.unique().numel() != indices.numel()):
            raise ValueError('Indices must be unique in-range int64 vectors on the scene device')
    k = len(parents)
    if len(donors) != k or torch.isin(parents, donors).any():
        raise ValueError('Require equally many disjoint parents and donors')
    scales = parameters['log_scales'][parents].exp()
    rotations = quaternion_to_matrix(parameters['quats'][parents])
    if directions is None:
        axes = scales.argmax(-1)
        directions = rotations.gather(2, axes[:, None, None].expand(-1, 3, 1)).squeeze(-1)
    if (directions.shape != (k, 3) or directions.dtype != means.dtype
            or directions.device != means.device or not bool(torch.isfinite(directions).all())):
        raise ValueError('Expected finite directions [K,3] matching geometry dtype/device')
    if k:
        offsets, axes = support_constrained_split_geometry(scales, rotations, directions, minimum_axis_fraction)
    else:
        offsets = means.new_empty((0, 3))
        axes = parents.clone()
    keep = torch.ones(count, dtype=torch.bool, device=means.device)
    keep[parents] = False
    keep[donors] = False
    retained = torch.where(keep)[0]
    source = torch.cat((retained, parents, parents))
    if len(source) != count:
        raise RuntimeError('Structure exchange changed the point budget')
    result = {name: value.detach()[source].clone() for name, value in parameters.items()}
    start = len(retained)
    result['means'][start:start+k] = means[parents] - offsets
    result['means'][start+k:] = means[parents] + offsets
    child_scales = parameters['log_scales'][parents].clone()
    child_scales.scatter_add_(1, axes[:, None], torch.full_like(axes[:, None], -math.log(scale_shrink), dtype=means.dtype))
    result['log_scales'][start:] = child_scales.repeat(2, 1)
    # Exactly the existing densification initialization; it is not a theorem
    # of image-space transmittance conservation after displacement/shrinkage.
    alpha = parameters['opacity_logits'][parents].sigmoid().clamp(1e-6, 1-1e-6)
    child_alpha = (1-(1-alpha).sqrt()).clamp(1e-6, 1-1e-6)
    child_logits = torch.logit(child_alpha)
    result['opacity_logits'][start:] = torch.cat((child_logits, child_logits))
    if not all(bool(torch.isfinite(t).all()) for t in result.values()):
        raise ValueError('Nonfinite structure action')
    return StructureExchange(result, source, start, parents.clone(), donors.clone(), offsets, axes)
