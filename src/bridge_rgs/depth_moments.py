"""Semantic/depth cross-moments of the existing Gaussian compositing weights.

These quantities describe mixture along a rendered ray. They are not calibrated
position uncertainty or new geometry, and do not imply a physical surface.
"""
from __future__ import annotations

import math

import torch


def semantic_depth_context(feature_sum, depth_sum, depth_square_sum,
                           depth_feature_sum, alpha, *, epsilon=1e-6, min_alpha=1e-4,
                           geometry_grad=False):
    """Return log relative depth spread plus signed feature/depth contrast.

    Inputs are unnormalized sums under ONE complete front-to-back compositing
    process, with zero moment backgrounds. Depth must already be divided by
    the frozen scene scale. All inputs use HWC grids, singleton depth/alpha.

    Geometry statistics are detached by default, while the feature and cross-feature sums
    retain gradients. geometry_grad=True also differentiates the depth/coverage
    statistics, with the same forward values and piecewise support predicate.
    For a two-depth mixture the uncompressed contrast equals
    sqrt(a*(1-a)) * (f_far-f_near), unlike either marginal mean. ``asinh`` compresses
    its growth without destroying the sign. Zero/low coverage and numerically
    degenerate depth mixtures provide zero context, never division by zero.
    """
    if type(geometry_grad) is not bool:
        raise TypeError("geometry_grad must be bool")
    if not math.isfinite(epsilon) or epsilon <= 0 or not 0 < min_alpha < 1:
        raise ValueError("Require positive finite epsilon and min_alpha in (0,1)")
    if feature_sum.ndim != 3 or depth_feature_sum.shape != feature_sum.shape:
        raise ValueError("Feature and cross-feature sums must share an HWC grid")
    scalar_shape = (*feature_sum.shape[:2], 1)
    if any(value.shape != scalar_shape for value in (depth_sum, depth_square_sum, alpha)):
        raise ValueError("Depth and alpha sums must be HxWx1 on the feature grid")
    coverage = (alpha if geometry_grad else alpha.detach()).clamp(min=min_alpha, max=1)
    mean_depth = (depth_sum if geometry_grad else depth_sum.detach()) / coverage
    second_moment = (depth_square_sum if geometry_grad else depth_square_sum.detach()) / coverage
    squared_mean = mean_depth.square()
    variance = (second_moment - squared_mean).clamp_min(0)
    # Subtracting two large moments can manufacture variance for a single layer.
    # Treat variance below their floating-point resolution as unresolvable; this
    # guard is numerical, not a calibrated geometric uncertainty estimate.
    roundoff_floor = (8 * torch.finfo(variance.dtype).eps
                      * (second_moment.abs() + squared_mean).clamp_min(1))
    supported = ((alpha.detach() >= min_alpha)
                 & (variance > roundoff_floor.clamp_min(epsilon)))
    if geometry_grad:
        # The discarded sqrt(0) branch otherwise produces 0*inf in backward.
        # Only unsupported entries change here; both outputs below remain zero
        # there. Supported values and the support decision are unchanged.
        variance = torch.where(supported, variance, torch.ones_like(variance))
    covariance = depth_feature_sum/coverage - mean_depth * (feature_sum/coverage)
    contrast = covariance / torch.sqrt(variance + epsilon)
    contrast = torch.where(supported, torch.asinh(contrast), 0)
    spread = torch.log1p(torch.sqrt(variance) / mean_depth.abs().clamp_min(min_alpha))
    spread = torch.where(supported, spread, 0)
    return torch.cat((spread, contrast), dim=-1)
