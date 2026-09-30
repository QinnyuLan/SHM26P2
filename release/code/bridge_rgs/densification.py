"""Deterministic, fixed-budget splitting from persistent RGB residuals.

This module deliberately accepts evidence from the renderer rather than
equating semantic uncertainty or Gaussian size with geometric error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


def normalized_absgrad(gradient: Tensor, width: int, height: int, num_cameras: int = 1) -> Tensor:
    """Match gsplat DefaultStrategy's [-1,1] screen-space gradient convention.

    Verified against gsplat/strategy/default.py::_update_state: the x/y
    derivatives are multiplied by width/2 and height/2, respectively.
    """
    if gradient.shape[-1] != 2 or min(width, height, num_cameras) <= 0:
        raise ValueError("Expected screen gradients [...,2] and positive image/camera sizes")
    scale = gradient.new_tensor([width / 2, height / 2]) * num_cameras
    return (gradient.detach() * scale).norm(dim=-1)


def adaptive_gradient_threshold(average: Tensor, view_counts: Tensor,
                                base_threshold: float, min_views: int = 2,
                                quantile: float | None = None, floor: float = 0.) -> float:
    """Optionally cap a fixed threshold by supported-scene gradient statistics.

    A positive floor prevents growth on numerically negligible gradients. The
    same rule should be used in a matched AbsGrad ablation.
    """
    if base_threshold <= 0 or floor < 0 or floor > base_threshold:
        raise ValueError("Require 0 <= floor <= positive base_threshold")
    if quantile is None:
        return base_threshold
    if not 0 < quantile < 1:
        raise ValueError("Gradient quantile must lie strictly between zero and one")
    values = average[(view_counts >= min_views) & torch.isfinite(average) & (average > 0)]
    return max(floor, min(base_threshold, float(values.quantile(quantile)))) if len(values) else base_threshold


def hybrid_density_priority(
    gradient_sum: Tensor, gradient_observations: Tensor, gradient_views: Tensor,
    residual_sum: Tensor, residual_observations: Tensor, residual_views: Tensor,
    gradient_threshold: float = 8e-4, residual_threshold: float = .03,
    min_views: int = 2, residual_weight: float = .35,
) -> tuple[Tensor, Tensor]:
    """Combine rank-normalized evidence without equating RGB and gradient units.

    Undiagnosed points retain an independently supported RGB-gradient path.
    Remaining residuals add a second path; they are not an availability gate
    on the much denser full-resolution gradient observations.
    """
    if gradient_threshold <= 0 or residual_threshold <= 0 or min_views < 1 or residual_weight < 0:
        raise ValueError("Thresholds/min_views must be positive and residual_weight nonnegative")
    gradient = gradient_sum / gradient_observations.clamp_min(1)
    residual = residual_sum / residual_observations.clamp_min(1)
    grad_valid = (gradient_views >= min_views) & (gradient >= gradient_threshold) & torch.isfinite(gradient)
    residual_valid = (residual_views >= min_views) & (residual >= residual_threshold) & torch.isfinite(residual)

    def ranks(value, valid):
        result = torch.zeros_like(value)
        selected = torch.where(valid)[0]
        if selected.numel():
            order = selected[torch.argsort(value[selected], stable=True)]
            result[order] = torch.arange(1, len(order) + 1, device=value.device, dtype=value.dtype) / len(order)
        return result

    score = ranks(gradient, grad_valid) + residual_weight * ranks(residual, residual_valid)
    support = torch.maximum(torch.where(grad_valid, gradient_views, 0),
                            torch.where(residual_valid, residual_views, 0))
    return score, support


@torch.no_grad()
def reset_opacity_with_optimizer(opacity_logits: nn.Parameter, optimizer: Optimizer,
                                 max_opacity: float = .03) -> int:
    """Cap opacity and clear changed rows' moments without replacing the parameter."""
    if not 0 < max_opacity < 1:
        raise ValueError("Opacity cap must be between zero and one")
    ceiling = torch.logit(opacity_logits.new_tensor(max_opacity))
    changed = opacity_logits > ceiling
    opacity_logits.clamp_(max=ceiling)
    for value in optimizer.state.get(opacity_logits, {}).values():
        if isinstance(value, Tensor) and value.shape == opacity_logits.shape:
            value[changed] = 0
    return int(changed.sum())


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """Convert gsplat's wxyz quaternions to rotation matrices."""
    quaternion = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), -1).reshape(*quaternions.shape[:-1], 3, 3)


def support_constrained_split_geometry(
    scales: Tensor, rotation: Tensor, direction: Tensor, minimum_axis_fraction: float = .25,
) -> tuple[Tensor, Tensor]:
    """Return a half-Mahalanobis-unit offset and a supported shrink axis.

    PCA directions are projected into the subspace of Gaussian axes with
    scale >= one quarter of the largest axis. A direction orthogonal to that
    subspace falls back to the major axis. The radial ellipsoid radius is
    used here; sqrt(v.T @ covariance @ v) is a projected standard deviation,
    and can place children far outside an anisotropic parent's support.
    """
    if not 0 < minimum_axis_fraction <= 1:
        raise ValueError("minimum_axis_fraction must lie in (0,1]")
    if scales.shape[-1] != 3 or direction.shape != scales.shape or rotation.shape != (*scales.shape[:-1], 3, 3):
        raise ValueError("Expected scales/directions [...,3] and rotations [...,3,3]")
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError("Gaussian scales must be finite and positive")
    maximum, major_axis = scales.max(-1)
    relative_scales = scales / maximum[..., None]
    supported = relative_scales >= minimum_axis_fraction
    local = (rotation.transpose(-1, -2) @ direction[..., None]).squeeze(-1)
    local = torch.where(supported, local, 0)
    length = local.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(local).scatter_(-1, major_axis[..., None], 1)
    local = torch.where(length > 1e-6, local / length.clamp_min(1e-6), fallback)
    # Avoid dividing by excluded extremely thin axes, including float underflow.
    safe_relative_scales = torch.where(supported, relative_scales, 1)
    radius = maximum / (local / safe_relative_scales).square().sum(-1).sqrt()
    offset = .5 * radius[..., None] * (rotation @ local[..., None]).squeeze(-1)
    variance_contribution = (local * relative_scales).square()
    shrink_axis = variance_contribution.argmax(-1)
    return offset, shrink_axis


def density_priority(
    rgb_residual_sum: Tensor,
    observation_count: Tensor,
    residual_view_counts: Tensor,
    min_views: int = 2,
    semantic_boundary: Tensor | None = None,
    geometry_gate: Tensor | None = None,
    boundary_weight: float = 0.15,
    rgb_threshold: float = 0.01,
) -> Tensor:
    """Score only persistent RGB underfit; semantics can modulate its priority.

    ``residual_view_counts`` must count distinct camera IDs with positive
    remaining-RGB evidence. Semantic boundary error alone never activates a
    candidate. Robust median normalization keeps error scales comparable.
    ``geometry_gate`` should encode projection/visibility/multiview support.
    """
    residual = rgb_residual_sum / observation_count.clamp_min(1)
    supported = (residual_view_counts >= min_views) & (observation_count > 0) & (residual > rgb_threshold)
    supported &= torch.isfinite(residual)
    nonzero = residual[supported]
    normalizer = nonzero.median().clamp_min(1e-6) if nonzero.numel() else residual.new_tensor(1)
    score = (torch.nan_to_num(residual, nan=0, posinf=0, neginf=0) / normalizer).clamp(0, 10)
    if semantic_boundary is not None:
        if geometry_gate is None:
            raise ValueError("Semantic boundary priority requires an explicit geometry_gate")
        boundary = torch.nan_to_num(semantic_boundary, nan=0, posinf=0, neginf=0).clamp_min(0)
        scale = boundary[boundary > 0].median().clamp_min(1e-6) if (boundary > 0).any() else boundary.new_tensor(1)
        score += boundary_weight * (boundary / scale).clamp(max=3) * geometry_gate.clamp(0, 1)
    return score * supported.to(score.dtype)


@dataclass
class LocalStructure:
    direction: Tensor
    kind: Tensor  # 0: insufficient/volumetric, 1: line, 2: plane
    eigenvalues: Tensor
    confidence: Tensor


@torch.no_grad()
def estimate_local_structure(
    reference_points: Tensor,
    query_points: Tensor,
    neighbors: int = 16,
    max_reference_points: int = 50_000,
    chunk_size: int = 128,
    linearity_threshold: float = 0.65,
    planarity_threshold: float = 0.45,
) -> LocalStructure:
    """Classify observed neighborhoods by PCA, without an N-by-N allocation.

    Reference points should be reliable SfM/depth points, preferably filtered
    by track support. Splats may be supplied as a weaker geometry-only
    fallback. Eigenvector signs are fixed for reproducible splits. Reference
    subsampling uses evenly spaced indices, never a hidden random generator.
    """
    if neighbors < 3 or chunk_size < 1 or max_reference_points < 3:
        raise ValueError("Need at least three neighbors/reference points and a positive chunk size")
    references = reference_points[torch.isfinite(reference_points).all(-1)]
    if len(references) > max_reference_points:
        selection = torch.linspace(0, len(references) - 1, max_reference_points, device=references.device).long()
        references = references[selection]
    count = len(query_points)
    directions = torch.zeros_like(query_points)
    directions[:, 0] = 1
    kinds = torch.zeros(count, dtype=torch.long, device=query_points.device)
    eigenvalues = query_points.new_zeros((count, 3))
    confidence = query_points.new_zeros(count)
    if len(references) < 3 or count == 0:
        return LocalStructure(directions, kinds, eigenvalues, confidence)
    for start in range(0, count, chunk_size):
        queries = query_points[start:start + chunk_size]
        distances = torch.cdist(queries.float(), references.float())
        nearest = distances.topk(min(neighbors, len(references)), largest=False, sorted=True).indices
        neighborhood = references[nearest]
        centered = neighborhood - neighborhood.mean(1, keepdim=True)
        covariance = centered.transpose(-1, -2) @ centered / max(1, neighborhood.shape[1] - 1)
        values, vectors = torch.linalg.eigh(covariance.float())
        values = values.to(query_points.dtype).clamp_min(0)
        principal = vectors[..., -1].to(query_points.dtype)
        sign = principal.gather(-1, principal.abs().argmax(-1, keepdim=True)).sign()
        principal = principal * torch.where(sign == 0, torch.ones_like(sign), sign)
        largest = values[:, -1].clamp_min(1e-12)
        linearity = (values[:, 2] - values[:, 1]) / largest
        planarity = (values[:, 1] - values[:, 0]) / largest
        local_radius = values.sum(-1).sqrt().clamp_min(1e-6)
        center_distance = (queries - neighborhood.mean(1)).norm(dim=-1)
        # Do not extrapolate a bridge tangent to far-away sky/background splats.
        # A nearest-neighbor query always returns points, even with no local support.
        nondegenerate = (values[:, 2] > 1e-12) & (center_distance <= 2.5 * local_radius)
        line = (linearity >= linearity_threshold) & nondegenerate
        plane = (~line) & (planarity >= planarity_threshold) & nondegenerate
        size = len(queries)
        directions[start:start + size] = principal
        kinds[start:start + size] = line.long() + 2 * plane.long()
        eigenvalues[start:start + size] = values
        confidence[start:start + size] = torch.where(line, linearity, torch.where(plane, planarity, 0))
    return LocalStructure(directions, kinds, eigenvalues, confidence)


@dataclass
class DensificationResult:
    tensors: dict[str, Tensor]
    source_indices: Tensor
    reset_moments: Tensor
    split_count: int
    pruned_count: int
    line_splits: int
    plane_splits: int
    duplicate_count: int = 0


@torch.no_grad()
def structure_guided_densify(
    parameters: Mapping[str, Tensor],
    scores: Tensor,
    residual_view_counts: Tensor,
    max_gaussians: int,
    max_splits: int = 1024,
    min_views: int = 2,
    opacity_threshold: float = 0.005,
    recycle_count: int = 0,
    reference_points: Tensor | None = None,
    contribution: Tensor | None = None,
    scale_shrink: float = 1.6,
    neighbors: int = 16,
    structure_guided: bool = True,
    clone_scale_threshold: float | None = None,
    screen_radii: Tensor | None = None,
    split_screen_radius: float | None = None,
    split_opacity_mode: str = "transmittance",
    shrink_unstructured_all_axes: bool = False,
    allocation_groups: Tensor | None = None,
    foreground_fraction: float = .65,
    structured_split_rule: str = "legacy",
) -> DensificationResult:
    """Prune weak splats and replace persistent-underfit parents by two children.

    Required keys: ``means``, ``quats``, ``log_scales``, ``opacity_logits``.
    Every parameter must have Gaussian count as its first dimension; arbitrary
    appearance and semantic features are copied. The result never exceeds
    ``max_gaussians``. With a full budget, optional recycling removes the
    lowest-contribution unselected splats to make room for splits.

    ``source_indices`` enables other per-Gaussian state to migrate. Child
    moments must be zeroed; use :func:`apply_densification` for Adam/AdamW.
    """
    required = {"means", "quats", "log_scales", "opacity_logits"}
    if not required.issubset(parameters):
        raise ValueError(f"Missing Gaussian keys: {required - set(parameters)}")
    means = parameters["means"]
    count = len(means)
    if max_gaussians < 1 or max_splits < 0 or recycle_count < 0 or scale_shrink <= 1:
        raise ValueError("Budget must be positive; counts nonnegative; scale_shrink > 1")
    if split_opacity_mode not in {"transmittance", "preserve"}:
        raise ValueError("split_opacity_mode must be transmittance or preserve")
    if structured_split_rule not in {"legacy", "support_constrained"}:
        raise ValueError("structured_split_rule must be legacy or support_constrained")
    if not 0 <= foreground_fraction <= 1:
        raise ValueError("foreground_fraction must lie in [0,1]")
    if count == 0:
        raise ValueError("Cannot densify an empty Gaussian field")
    if scores.shape != (count,) or residual_view_counts.shape != (count,):
        raise ValueError("Scores and residual view counts must have shape [N]")
    if any(len(value) != count for value in parameters.values()):
        raise ValueError("Every parameter must have N as its first dimension")
    scores = torch.nan_to_num(scores, nan=0, posinf=0, neginf=0).clamp_min(0)
    opacity = parameters["opacity_logits"].sigmoid().reshape(count, -1).mean(-1)
    eligible = (scores > 0) & (residual_view_counts >= min_views) & (opacity >= opacity_threshold)
    keep = (opacity >= opacity_threshold) & torch.isfinite(means).all(-1)
    if not keep.any():
        # Preserve a nonempty field even after a bad opacity warmup. Hard budget
        # pruning below still applies. A caller should diagnose NaN parameters.
        keep[opacity.argmax()] = True
    retention = opacity if contribution is None else torch.nan_to_num(contribution, nan=0, posinf=0, neginf=0).clamp_min(0)
    if retention.shape != (count,):
        raise ValueError("contribution must have shape [N]")
    # Stable sort resolves ties by source index, making allocation reproducible.
    candidate_order = torch.argsort(scores, descending=True, stable=True)
    candidate_order = candidate_order[eligible[candidate_order] & keep[candidate_order]]
    if allocation_groups is not None:
        if allocation_groups.shape != (count,):
            raise ValueError("allocation_groups must have shape [N]")
        foreground = allocation_groups.bool()[candidate_order]
        foreground_quota = round(max_splits * foreground_fraction)
        reserved_fg = candidate_order[foreground][:foreground_quota]
        reserved_other = candidate_order[~foreground][:max_splits - foreground_quota]
        reserved = torch.cat((reserved_fg, reserved_other))
        selected = torch.zeros_like(keep)
        selected[reserved] = True
        extra = candidate_order[~selected[candidate_order]][:max(0, max_splits - len(reserved))]
        candidate_order = torch.cat((reserved, extra))
    else:
        candidate_order = candidate_order[:max_splits]
    protected = torch.zeros_like(keep)
    protected[candidate_order] = True
    if int(keep.sum()) > max_gaussians:
        # Honor budget first. Prefer preserving split candidates where possible.
        rank = retention + protected.to(retention.dtype) * (retention.max() + 1)
        order = torch.argsort(rank, descending=True, stable=True)
        order = order[keep[order]][:max_gaussians]
        keep.zero_()
        keep[order] = True
        candidate_order = candidate_order[keep[candidate_order]]
    available = max_gaussians - int(keep.sum())
    desired = min(len(candidate_order), max_splits)
    retire = min(recycle_count, max(0, desired - available))
    if retire:
        recyclable = torch.where(keep & ~protected)[0]
        order = torch.argsort(retention[recyclable], stable=True)
        retired = recyclable[order[:retire]]
        keep[retired] = False
        available += len(retired)
    allocation_count = min(desired, available)
    if allocation_groups is not None:
        foreground = allocation_groups.bool()[candidate_order]
        fg_count = round(allocation_count * foreground_fraction)
        reserved = torch.cat((candidate_order[foreground][:fg_count],
                              candidate_order[~foreground][:allocation_count - fg_count]))
        reserved_mask = torch.zeros_like(keep)
        reserved_mask[reserved] = True
        extra = candidate_order[~reserved_mask[candidate_order]][:allocation_count - len(reserved)]
        selected_parents = torch.cat((reserved, extra))
    else:
        selected_parents = candidate_order[:allocation_count]
    clone_mask = torch.zeros(len(selected_parents), dtype=torch.bool, device=means.device)
    if clone_scale_threshold is not None:
        if clone_scale_threshold <= 0:
            raise ValueError("clone_scale_threshold must be positive")
        clone_mask = parameters["log_scales"][selected_parents].exp().max(-1).values <= clone_scale_threshold
        if screen_radii is not None and split_screen_radius is not None:
            clone_mask &= screen_radii[selected_parents] < split_screen_radius
    clone_indices = selected_parents[clone_mask]
    parent_indices = selected_parents[~clone_mask]
    duplicate_count = len(clone_indices)
    split_count = len(parent_indices)
    # A parent is replaced by two children (net +1), not retained as a third copy.
    ordinary = keep.clone()
    ordinary[parent_indices] = False
    retained_indices = torch.where(ordinary)[0]
    # Small primitives retain their existing optimizer history and gain one
    # new copy (standard 3DGS growth); only large parents are replaced.
    source_indices = torch.cat((retained_indices, clone_indices, parent_indices, parent_indices))
    reset_moments = torch.zeros(len(source_indices), dtype=torch.bool, device=means.device)
    reset_moments[len(retained_indices):] = True
    tensors = {name: value.detach()[source_indices].clone() for name, value in parameters.items()}
    line_count = plane_count = 0
    if split_count:
        parent_means = means[parent_indices]
        scales = parameters["log_scales"][parent_indices].exp()
        rotation = quaternion_to_matrix(parameters["quats"][parent_indices])
        major_axis = scales.argmax(-1)
        default_direction = rotation.gather(-1, major_axis[:, None, None].expand(-1, 3, 1)).squeeze(-1)
        if structure_guided:
            structure = estimate_local_structure(
                means.detach() if reference_points is None else reference_points,
                parent_means, neighbors=neighbors,
            )
        else:
            structure = LocalStructure(
                default_direction, torch.zeros(split_count, dtype=torch.long, device=means.device),
                means.new_zeros((split_count, 3)), means.new_zeros(split_count),
            )
        direction = torch.where((structure.kind > 0)[:, None], structure.direction, default_direction)
        local_direction = (rotation.transpose(-1, -2) @ direction[..., None]).squeeze(-1)
        # Spatial extent along the selected observed tangent. The Gaussian's
        # shape sets the child spacing, not its confidence or covariance prior.
        extent = (local_direction.square() * scales.square()).sum(-1).sqrt()
        offset = 0.5 * extent[:, None] * direction
        aligned_axis = local_direction.abs().argmax(-1)
        if structured_split_rule == "support_constrained":
            supported_offset, supported_axis = support_constrained_split_geometry(scales, rotation, direction)
            structural = structure.kind > 0
            offset = torch.where(structural[:, None], supported_offset, offset)
            aligned_axis = torch.where(structural, supported_axis, aligned_axis)
        begin = len(retained_indices) + duplicate_count
        tensors["means"][begin:begin + split_count] = parent_means - offset
        tensors["means"][begin + split_count:] = parent_means + offset
        child_log_scales = parameters["log_scales"][parent_indices].detach().clone()
        child_log_scales.scatter_add_(
            1, aligned_axis[:, None],
            torch.full_like(aligned_axis[:, None], -torch.log(means.new_tensor(scale_shrink)), dtype=means.dtype),
        )
        if shrink_unstructured_all_axes:
            unstructured = structure.kind == 0
            child_log_scales[unstructured] = parameters["log_scales"][parent_indices[unstructured]] - torch.log(means.new_tensor(scale_shrink))
        tensors["log_scales"][begin:] = child_log_scales.repeat(2, 1)
        parent_alpha = parameters["opacity_logits"][parent_indices].sigmoid().clamp(1e-6, 1 - 1e-6)
        child_alpha = 1 - (1 - parent_alpha).sqrt() if split_opacity_mode == "transmittance" else parent_alpha
        child_logits = torch.logit(child_alpha.clamp(1e-6, 1 - 1e-6))
        tensors["opacity_logits"][begin:] = torch.cat((child_logits, child_logits), 0)
        line_count = int((structure.kind == 1).sum())
        plane_count = int((structure.kind == 2).sum())
    return DensificationResult(
        tensors, source_indices, reset_moments, split_count,
        count - int(keep.sum()), line_count, plane_count,
        duplicate_count,
    )


@torch.no_grad()
def apply_densification(
    parameters: nn.ParameterDict,
    result: DensificationResult,
    optimizers: Optimizer | Sequence[Optimizer] | Mapping[str, Optimizer],
) -> None:
    """Replace field parameters and migrate shape-compatible optimizer state.

    Adam/AdamW/SGD per-row moments are retained for unchanged splats and zeroed
    for both children; scalar step counters are preserved. All other model
    parameters and optimizer groups remain unchanged. Optimizers holding the
    field must all be passed; an omitted optimizer would keep stale references.
    """
    if isinstance(optimizers, Optimizer):
        optimizer_list = [optimizers]
    elif isinstance(optimizers, Mapping):
        optimizer_list = list(optimizers.values())
    else:
        optimizer_list = list(optimizers)
    # A user may map several names to one optimizer.
    optimizer_list = list({id(optimizer): optimizer for optimizer in optimizer_list}.values())
    for name, value in result.tensors.items():
        old = parameters[name]
        new = nn.Parameter(value, requires_grad=old.requires_grad)
        parameters[name] = new
        for optimizer in optimizer_list:
            found = False
            for group in optimizer.param_groups:
                replacement = []
                for parameter in group["params"]:
                    if parameter is old:
                        replacement.append(new)
                        found = True
                    else:
                        replacement.append(parameter)
                group["params"] = replacement
            if not found:
                continue
            old_state = optimizer.state.pop(old, {})
            new_state = {}
            for key, state in old_state.items():
                if isinstance(state, Tensor) and state.shape == old.shape:
                    migrated = state[result.source_indices].clone()
                    migrated[result.reset_moments] = 0
                    new_state[key] = migrated
                elif isinstance(state, Tensor):
                    new_state[key] = state.clone()
                else:
                    new_state[key] = state
            if new_state:
                optimizer.state[new] = new_state
