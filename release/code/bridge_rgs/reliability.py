"""Camera residual diagnostics and uncertainty-aware semantic evidence.

Camera twists are translation-first and left-multiply world-to-camera poses.
All covariance arguments concern *estimation uncertainty*, never splat shape.
These local linear diagnostics do not constitute calibrated BA uncertainty.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .coordinates import CORNER, LEGACY, inside_bilinear_centers, sampling_grid
from .coordinates import pixel_protocol as resolve_protocol


def skew(vector: Tensor) -> Tensor:
    """Return the cross-product matrix of vectors with final dimension three."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        *vector.shape[:-1], 3, 3
    )


def se3_exp(twist: Tensor) -> Tensor:
    """Differentiable SE(3) exponential, including finite derivatives at zero.

    Input is ``[..., tx, ty, tz, rx, ry, rz]``. Translation uses the SO(3)
    left Jacobian, rather than treating the first three entries as a direct
    translation after a finite rotation.
    """
    if twist.shape[-1] != 6:
        raise ValueError("A camera twist must have six components")
    translation, rotation = twist[..., :3], twist[..., 3:]
    theta2 = rotation.square().sum(-1)
    small = theta2 < 1e-6
    # Clamp only the unused closed-form branch; sqrt(0) otherwise poisons
    # autograd through torch.where, even when its branch is not selected.
    theta = theta2.clamp_min(1e-12).sqrt()
    a = torch.where(small, 1 - theta2 / 6 + theta2.square() / 120, torch.sin(theta) / theta)
    b = torch.where(
        small, 0.5 - theta2 / 24 + theta2.square() / 720,
        (1 - torch.cos(theta)) / theta.square(),
    )
    c = torch.where(
        small, 1 / 6 - theta2 / 120 + theta2.square() / 5040,
        (theta - torch.sin(theta)) / theta.pow(3),
    )
    omega = skew(rotation)
    omega2 = omega @ omega
    identity = torch.eye(3, dtype=twist.dtype, device=twist.device).expand_as(omega)
    rotation_matrix = identity + a[..., None, None] * omega + b[..., None, None] * omega2
    left_jacobian = identity + b[..., None, None] * omega + c[..., None, None] * omega2
    top = torch.cat((rotation_matrix, (left_jacobian @ translation[..., None])), dim=-1)
    bottom = twist.new_zeros((*twist.shape[:-1], 1, 4))
    bottom[..., 0, 3] = 1
    return torch.cat((top, bottom), dim=-2)


@dataclass
class Projection:
    uv: Tensor
    depth: Tensor
    camera_jacobian: Tensor
    point_jacobian: Tensor


def projection_jacobians(points_world: Tensor, world_to_camera: Tensor, K: Tensor) -> Projection:
    """Pinhole projections and analytic left-pose/world-point Jacobians.

    Shapes: points ``[N,3]``, pose ``[4,4]``, K ``[3,3]``. The usual pinhole
    bottom row ``[0,0,1]`` is required; skew in K is supported. Invalid/behind
    camera points remain identifiable from ``depth`` and must be gated.
    """
    xyz = points_world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    depth = xyz[:, 2]
    denominator = depth.clamp_min(1e-6)
    homogeneous = xyz @ K.T
    uv = homogeneous[:, :2] / denominator[:, None]
    jacobian = K[:2, :].expand(len(xyz), -1, -1) / denominator[:, None, None]
    correction = torch.zeros_like(jacobian)
    correction[:, :, 2] = homogeneous[:, :2] / denominator[:, None].square()
    jacobian = jacobian - correction
    identity = torch.eye(3, dtype=xyz.dtype, device=xyz.device).expand(len(xyz), -1, -1)
    camera_jacobian = jacobian @ torch.cat((identity, -skew(xyz)), dim=-1)
    return Projection(uv, depth, camera_jacobian, jacobian @ world_to_camera[:3, :3])


def propagate_projection_covariance(
    camera_jacobian: Tensor,
    point_jacobian: Tensor,
    camera_covariance: Tensor,
    point_covariance: Tensor | None = None,
    observation_std: float = 0.5,
    camera_point_cross_covariance: Tensor | None = None,
) -> Tensor:
    """Propagate estimation covariance to pixels, preserving supplied cross terms.

    Camera covariance is ``[6,6]`` or ``[N,6,6]``; point covariance is
    ``[3,3]`` or ``[N,3,3]``. Optional camera-point cross covariance is ``[...,6,3]``.
    Independent covariances are only an approximation when cross terms are
    unavailable. The output includes an observation noise floor in pixels.
    """
    if observation_std < 0:
        raise ValueError("observation_std cannot be negative")
    covariance = camera_jacobian @ camera_covariance @ camera_jacobian.transpose(-1, -2)
    if point_covariance is not None:
        covariance = covariance + point_jacobian @ point_covariance @ point_jacobian.transpose(-1, -2)
    if camera_point_cross_covariance is not None:
        cross = camera_jacobian @ camera_point_cross_covariance @ point_jacobian.transpose(-1, -2)
        covariance = covariance + cross + cross.transpose(-1, -2)
    identity = torch.eye(2, device=covariance.device, dtype=covariance.dtype)
    return 0.5 * (covariance + covariance.transpose(-1, -2)) + observation_std**2 * identity


@dataclass
class ResidualAttribution:
    delta: Tensor
    remaining: Tensor
    explained_fraction: Tensor
    before_energy: Tensor
    after_energy: Tensor
    accepted: bool = True


def _broadcast_weights(weights: Tensor | None, residual: Tensor) -> Tensor:
    if weights is None:
        return torch.ones_like(residual)
    if weights.shape == residual.shape[:-1]:
        weights = weights[..., None]
    return torch.broadcast_to(weights, residual.shape).clamp_min(0)


def camera_residual_attribution(
    residual: Tensor,
    jacobian: Tensor,
    weights: Tensor | None = None,
    damping: float = 1e-3,
    prior_precision: Tensor | None = None,
    max_translation: float | None = None,
    max_rotation: float | None = None,
) -> ResidualAttribution:
    """Solve a small damped weighted least-squares residual attribution.

    ``jacobian.shape == (*residual.shape, 6)`` and residual is render minus
    observation. Damping is relative to each normal-matrix diagonal, with a
    unit floor. Thus zero-information columns and fully masked inputs remain
    well-defined. This returns a local diagnostic, not an identified cause.
    """
    if jacobian.shape != (*residual.shape, 6):
        raise ValueError("Jacobian must have residual.shape + (6,)")
    if damping <= 0:
        raise ValueError("Positive damping is required")
    weight = _broadcast_weights(weights, residual).reshape(-1)
    r, J = residual.reshape(-1), jacobian.reshape(-1, 6)
    finite = torch.isfinite(r) & torch.isfinite(J).all(-1) & torch.isfinite(weight)
    weight = torch.where(finite, weight, torch.zeros_like(weight))
    r = torch.where(finite, r, torch.zeros_like(r))
    J = torch.where(finite[:, None], J, torch.zeros_like(J))
    # Accumulate in float64 for the six-dimensional system; render tensors
    # may be float32/float16 and pose derivatives can have very different scales.
    J64, r64, w64 = J.double(), r.double(), weight.double()
    normal = J64.T @ (w64[:, None] * J64)
    precision = damping * torch.diag(normal.diagonal().clamp_min(1))
    if prior_precision is not None:
        precision = precision + prior_precision.double()
    delta = -torch.linalg.solve(normal + precision, J64.T @ (w64 * r64)).to(residual.dtype)
    pieces = []
    for vector, limit in ((delta[:3], max_translation), (delta[3:], max_rotation)):
        if limit is not None:
            if limit < 0:
                raise ValueError("Pose trust-region limits cannot be negative")
            vector = vector * (limit / vector.norm().clamp_min(1e-12)).clamp(max=1)
        pieces.append(vector)
    delta = torch.cat(pieces)
    remaining = (r + J @ delta).reshape_as(residual)
    before = (weight * r.square()).sum() / weight.sum().clamp_min(1)
    after = (weight * remaining.reshape(-1).square()).sum() / weight.sum().clamp_min(1)
    explained = ((before - after) / before.clamp_min(1e-12)).clamp(0, 1)
    return ResidualAttribution(delta, remaining, explained, before, after)


@torch.no_grad()
def finite_difference_camera_jacobian(
    render_fn: Callable[[Tensor], Tensor],
    world_to_camera: Tensor,
    translation_eps: float = 1e-4,
    rotation_eps: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Central-difference frozen-scene RGB Jacobian, requiring 13 renders.

    ``render_fn`` must be deterministic and accept one world-to-camera pose.
    Use a low-resolution render or a fixed pixel sample in the callback.
    """
    if min(translation_eps, rotation_eps) <= 0:
        raise ValueError("Finite difference steps must be positive")
    base = render_fn(world_to_camera).detach()
    derivatives = []
    for component in range(6):
        step = translation_eps if component < 3 else rotation_eps
        twist = world_to_camera.new_zeros(6)
        twist[component] = step
        plus = render_fn(se3_exp(twist) @ world_to_camera)
        minus = render_fn(se3_exp(-twist) @ world_to_camera)
        derivatives.append((plus - minus) / (2 * step))
    return base, torch.stack(derivatives, dim=-1)


@torch.no_grad()
def camera_compensated_residual(
    render_fn: Callable[[Tensor], Tensor],
    world_to_camera: Tensor,
    target: Tensor,
    weights: Tensor | None = None,
    damping: float = 1e-3,
    prior_precision: Tensor | None = None,
    max_translation: float = 0.01,
    max_rotation: float = 0.01,
    translation_eps: float = 1e-4,
    rotation_eps: float = 1e-4,
    verify_render: bool = True,
) -> ResidualAttribution:
    """Attribute RGB error to a bounded pose step and verify by rerendering.

    By default the returned remainder is the *actual* compensated render
    residual. A pose step that worsens weighted RGB MSE is rejected. This
    function does not mutate the camera; the caller controls any pose update.
    """
    base, jacobian = finite_difference_camera_jacobian(
        render_fn, world_to_camera, translation_eps, rotation_eps
    )
    residual = base - target
    result = camera_residual_attribution(
        residual, jacobian, weights, damping, prior_precision, max_translation, max_rotation
    )
    if verify_render:
        candidate = render_fn(se3_exp(result.delta) @ world_to_camera) - target
        weight = _broadcast_weights(weights, residual)
        after = (weight * candidate.square()).sum() / weight.sum().clamp_min(1)
        accepted = bool(torch.isfinite(after) & (after <= result.before_energy))
        if accepted:
            result.remaining = candidate
            result.after_energy = after
            result.explained_fraction = (
                (result.before_energy - after) / result.before_energy.clamp_min(1e-12)
            ).clamp(0, 1)
        else:
            result.delta = torch.zeros_like(result.delta)
            result.remaining = residual
            result.after_energy = result.before_energy
            result.explained_fraction = result.before_energy.new_zeros(())
        result.accepted = accepted
    return result


@dataclass
class SemanticEvidence:
    probabilities: Tensor
    weights: Tensor
    stability: Tensor
    valid_fraction: Tensor
    projection_weight: Tensor


def _sample_map(values: Tensor, locations: Tensor, pixel_protocol=LEGACY) -> Tensor:
    """Bilinear CHW sampling, with explicit source coordinate protocol."""
    if values.ndim == 2:
        values = values[None]
    height, width = values.shape[-2:]
    if resolve_protocol(pixel_protocol) == CORNER:
        normalized = sampling_grid(locations, width, height, CORNER)
    else:
        normalized = torch.stack(
            (2 * (locations[..., 0] + 0.5) / width - 1,
             2 * (locations[..., 1] + 0.5) / height - 1), dim=-1,
        )
    sampled = F.grid_sample(
        values[None], normalized[None], mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return sampled[0].permute(1, 2, 0)


def ellipse_sample_probabilities(
    probabilities: Tensor,
    uv: Tensor,
    covariance: Tensor,
    depths: Tensor | None = None,
    depth_map: Tensor | None = None,
    alpha_map: Tensor | None = None,
    depth_relative_tolerance: float = 0.05,
    depth_absolute_tolerance: float = 0.01,
    alpha_threshold: float = 0.3,
    max_projection_std: float = 8.0,
    projection_scale: float = 4.0,
    confidence_threshold: float = 0.0,
    min_valid_fraction: float = 0.5,
    teacher_confidence_map: Tensor | None = None,
    pixel_protocol=LEGACY,
) -> SemanticEvidence:
    """Sample a deterministic 9-point uncertainty ellipse with visibility gates.

    Input probabilities are ``[C,H,W]``. Out-of-frame samples, nonpositive
    point depth, surface-depth disagreement and low rendered alpha receive
    zero weight. Depth maps must be camera-space positive surface depth.
    A high variance or an ellipse crossing class boundaries lowers evidence.
    Covariance and visibility are diagnostics, not freely trainable escape
    routes; callers should stop their gradients during pseudo-supervision.
    """
    if probabilities.ndim != 3 or uv.ndim != 2 or covariance.shape != (len(uv), 2, 2):
        raise ValueError("Expected probabilities [C,H,W], uv [N,2], covariance [N,2,2]")
    if max_projection_std <= 0 or projection_scale <= 0:
        raise ValueError("Projection scales must be positive")
    num_classes, height, width = probabilities.shape
    protocol = resolve_protocol(pixel_protocol)
    if num_classes < 2:
        raise ValueError("Semantic evidence requires at least two classes")
    finite_covariance = torch.isfinite(covariance).all(-1).all(-1)
    safe_covariance = torch.nan_to_num(covariance, nan=0, posinf=0, neginf=0)
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (safe_covariance + safe_covariance.transpose(-1, -2)))
    eigenvalues = eigenvalues.clamp_min(0)
    root = eigenvectors @ torch.diag_embed(eigenvalues.sqrt())
    diagonal = 2**-0.5
    sigma_points = uv.new_tensor([
        [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
        [diagonal, diagonal], [-diagonal, diagonal],
        [diagonal, -diagonal], [-diagonal, -diagonal],
    ])
    locations = uv[:, None, :] + torch.einsum("nij,sj->nsi", root, sigma_points)
    locations = torch.nan_to_num(locations, nan=-1e8, posinf=-1e8, neginf=-1e8)
    valid = inside_bilinear_centers(locations, width, height, protocol)
    valid &= finite_covariance[:, None] & torch.isfinite(uv).all(-1)[:, None]
    if depths is not None:
        valid &= (torch.isfinite(depths) & (depths > 0))[:, None]
    if depth_map is not None:
        if depths is None:
            raise ValueError("depths are required when providing a depth_map")
        sampled_depth = _sample_map(depth_map, locations, protocol)[..., 0]
        tolerance = depth_absolute_tolerance + depth_relative_tolerance * depths.abs()
        valid &= torch.isfinite(sampled_depth) & (sampled_depth > 0)
        valid &= (sampled_depth - depths[:, None]).abs() <= tolerance[:, None]
    if alpha_map is not None:
        valid &= _sample_map(alpha_map, locations, protocol)[..., 0] >= alpha_threshold
    quadrature = uv.new_tensor([0.2] + [0.1] * 8)
    sample_weights = valid.to(uv.dtype) * quadrature
    valid_fraction = sample_weights.sum(-1)
    normalized_weight = sample_weights / valid_fraction[:, None].clamp_min(1e-12)
    samples = _sample_map(probabilities, locations, protocol).clamp_min(1e-8)
    samples = samples / samples.sum(-1, keepdim=True).clamp_min(1e-8)
    mean = (normalized_weight[..., None] * samples).sum(1)
    mean = torch.where((valid_fraction > 0)[:, None], mean, torch.full_like(mean, 1 / num_classes))
    entropy_mean = -(mean * mean.clamp_min(1e-8).log()).sum(-1)
    entropy_samples = -(samples * samples.clamp_min(1e-8).log()).sum(-1)
    js = (entropy_mean - (normalized_weight * entropy_samples).sum(-1)).clamp_min(0)
    stability = (1 - js / torch.log(uv.new_tensor(float(num_classes)))).clamp(0, 1)
    # The exponential term deliberately penalizes spatial uncertainty separately
    # from teacher class confidence and the ellipse's class stability.
    projection_weight = torch.exp(-0.5 * eigenvalues.sum(-1) / projection_scale**2)
    projection_weight *= (eigenvalues[:, -1] <= max_projection_std**2).to(uv.dtype)
    if teacher_confidence_map is None:
        confidence = mean.max(-1).values
    else:
        confidence_samples = _sample_map(teacher_confidence_map, locations, protocol)[..., 0].clamp(0, 1)
        confidence = (normalized_weight * confidence_samples).sum(-1)
    weight = valid_fraction * confidence * stability.square() * projection_weight
    weight *= ((valid_fraction >= min_valid_fraction) & (confidence >= confidence_threshold)).to(uv.dtype)
    return SemanticEvidence(mean, weight, stability, valid_fraction, projection_weight)


@dataclass
class FusedEvidence:
    probabilities: Tensor
    weights: Tensor
    agreement: Tensor
    view_count: Tensor


def fuse_multiview_evidence(
    probabilities: Tensor,
    weights: Tensor,
    min_views: int = 2,
    agreement_power: float = 2.0,
) -> FusedEvidence:
    """Fuse independent-view soft evidence with majority and JS agreement.

    Shapes are ``[V,N,C]`` and ``[V,N]``. V must index *distinct cameras*,
    not augmentation samples. A confident 50:50 disagreement is suppressed
    more strongly than uncertain but compatible evidence. Output weights
    lie in [0,1]; an empty observation set returns uniform probabilities.
    """
    if probabilities.ndim != 3 or weights.shape != probabilities.shape[:2]:
        raise ValueError("Expected probabilities [V,N,C] and weights [V,N]")
    if min_views < 1 or agreement_power <= 0:
        raise ValueError("min_views and agreement_power must be positive")
    classes = probabilities.shape[-1]
    valid = torch.isfinite(probabilities).all(-1) & torch.isfinite(weights) & (weights > 0)
    weights = torch.where(valid, weights.clamp(0, 1), torch.zeros_like(weights))
    probabilities = torch.nan_to_num(probabilities).clamp_min(1e-8)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    total = weights.sum(0)
    mean = (probabilities * weights[..., None]).sum(0) / total[:, None].clamp_min(1e-12)
    mean = torch.where((total > 0)[:, None], mean, torch.full_like(mean, 1 / classes))
    kl = (probabilities * (probabilities.log() - mean.clamp_min(1e-8).log()[None])).sum(-1)
    divergence = (kl * weights).sum(0) / total.clamp_min(1e-12)
    js_agreement = (1 - divergence / torch.log(mean.new_tensor(float(classes)))).clamp(0, 1)
    votes = F.one_hot(probabilities.argmax(-1), classes).to(weights.dtype)
    majority_fraction = (votes * weights[..., None]).sum(0).max(-1).values / total.clamp_min(1e-12)
    agreement = (js_agreement * majority_fraction).clamp(0, 1)
    view_count = (weights > 0).sum(0)
    confidence_weight = total / view_count.clamp_min(1)
    reliability = confidence_weight * agreement.pow(agreement_power)
    reliability *= (view_count >= min_views).to(reliability.dtype)
    return FusedEvidence(mean, reliability, agreement, view_count)


def geometry_supervision_gate(
    projection_weight: Tensor,
    agreement: Tensor,
    view_count: Tensor,
    rgb_structure: Tensor,
    min_views: int = 2,
    min_projection_weight: float = 0.5,
    min_agreement: float = 0.8,
) -> Tensor:
    """Conservative permission for semantic evidence to influence geometry.

    ``rgb_structure`` is externally measured normalized RGB/depth structure
    support in [0,1], not semantic confidence. Ground-truth semantic losses
    should still supervise features at pixels where this gate is zero.
    Returned gates are detached, so a learnable reliability cannot suppress
    its own supervision. A trainer can instead freeze semantic-to-geometry
    gradients completely as a conservative default.
    """
    gate = projection_weight.clamp(0, 1) * agreement.clamp(0, 1) * rgb_structure.clamp(0, 1)
    valid = (view_count >= min_views) & (projection_weight >= min_projection_weight) & (agreement >= min_agreement)
    return torch.where(valid & torch.isfinite(gate), gate, torch.zeros_like(gate)).detach()
