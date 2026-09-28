"""Conservative discrete splat-center termination-mass diagnostics.

This is classical ray-depth/free-space supervision machinery, not a novel
mechanism or a continuous Gaussian volume CDF. No training integration here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class ConservativeDepthBins:
    edges: Tensor
    cutoff: Tensor
    lower_index: Tensor
    covered: Tensor
    gap: Tensor


def conservative_depth_bins(depth: Tensor, depth_std: Tensor, channels=32,
                            sigma_margin=3., relative_margin=.02, near_plane=.01):
    """Detached per-view log edges from positive SfM conservative cutoffs.

    Select the largest edge <= cutoff, never interpolate upward. Samples below
    the renderer's near plane remain uncovered. A single-valued range uses one
    channel rather than manufacturing an unsupported depth range.
    """
    if depth.ndim != 1 or depth_std.shape != depth.shape:
        raise ValueError("Expected matching depth/std vectors")
    if channels not in {16, 32} or min(sigma_margin, relative_margin, near_plane) <= 0:
        raise ValueError("Use 16/32 channels and positive fixed margin/near-plane values")
    depth, depth_std = depth.detach(), depth_std.detach().to(depth)
    finite = torch.isfinite(depth) & torch.isfinite(depth_std) & (depth > 0) & (depth_std >= 0)
    cutoff = depth - torch.maximum(sigma_margin * depth_std, relative_margin * depth)
    eligible = finite & (cutoff > near_plane)
    if not bool(eligible.any()):
        return ConservativeDepthBins(depth.new_empty(0), cutoff,
                                     torch.full_like(depth, -1, dtype=torch.long), eligible,
                                     torch.full_like(depth, float("nan")))
    low, high = cutoff[eligible].min(), cutoff[eligible].max()
    if low == high:
        edges = low[None]
    else:
        # One fixed 32-edge family; the 16-edge family is a nested subset with
        # both endpoints retained, enabling monotone lower-bound comparisons.
        edges = torch.exp(torch.linspace(low.log(), high.log(), 32,
                                        device=depth.device, dtype=depth.dtype))
        # Preserve endpoint inequalities despite exp/log rounding.
        edges[0], edges[-1] = low, high
        edges = torch.unique_consecutive(edges)
        if channels == 16 and len(edges) > 16:
            subset = torch.linspace(0, len(edges)-1, 16, device=depth.device).round().long()
            edges = edges[subset]
    index = torch.searchsorted(edges, cutoff.contiguous(), right=True) - 1
    covered = eligible & (index >= 0)
    index = torch.where(covered, index, torch.full_like(index, -1))
    gap = torch.where(covered, cutoff - edges[index.clamp_min(0)],
                      torch.full_like(cutoff, float("nan")))
    if bool((gap[covered] < 0).any()):
        raise AssertionError("Selected edge is not a conservative floor")
    return ConservativeDepthBins(edges.detach(), cutoff.detach(), index.detach(), covered.detach(), gap.detach())


def alpha_termination_weights(alpha: Tensor):
    """CPU/reference alpha compositing in an already depth-sorted ray sequence."""
    if alpha.ndim < 1 or not bool(torch.isfinite(alpha).all()) or bool(((alpha < 0) | (alpha > 1)).any()):
        raise ValueError("Sorted per-ray alpha must be finite and in [0,1]")
    preceding = torch.cat((torch.ones_like(alpha[..., :1]), 1 - alpha[..., :-1]), -1)
    return alpha * torch.cumprod(preceding, -1)


def reference_front_mass(alpha: Tensor, center_depth: Tensor, edges: Tensor):
    """Discrete CDF at supplied edges, with geometry/edge decisions detached."""
    z, edges = center_depth.detach().to(alpha), edges.detach().to(alpha)
    if z.shape != alpha.shape or bool((z[..., 1:] < z[..., :-1]).any()):
        raise ValueError("alpha/depth shapes must match with increasing depth order")
    if edges.ndim != 1 or not bool(torch.isfinite(z).all() & torch.isfinite(edges).all()):
        raise ValueError("Finite center depths and edge vector required")
    indicator = (z[..., :, None] < edges).to(alpha.dtype).detach()
    return (alpha_termination_weights(alpha)[..., :, None] * indicator).sum(-2)


def front_mass_bounds(mass: Tensor, total_alpha: Tensor, edges: Tensor, cutoff: Tensor):
    """Bracket each sample's discrete center-CDF without upper interpolation.

    Below the first edge use [0, first mass]; above the last use [last mass,
    total alpha]. Geometry/thresholds are detached, while the lower mass remains
    differentiable in opacity. The upper endpoint is diagnostic, not a target.
    """
    edges, cutoff = edges.detach().to(mass), cutoff.detach().to(mass)
    if mass.ndim != 2 or mass.shape != (len(cutoff), len(edges)) or total_alpha.shape != cutoff.shape:
        raise ValueError("Expected per-target mass [N,K], alpha [N], edges [K], cutoff [N]")
    if not len(edges) or bool((edges[1:] <= edges[:-1]).any()):
        raise ValueError("Nonempty increasing edges required")
    lower_index = torch.searchsorted(edges, cutoff.contiguous(), right=True)-1
    upper_index = torch.searchsorted(edges, cutoff.contiguous(), right=False)
    row = torch.arange(len(cutoff), device=mass.device)
    lower = torch.where(lower_index >= 0, mass[row, lower_index.clamp_min(0)], mass.new_zeros(()))
    upper = torch.where(upper_index < len(edges), mass[row, upper_index.clamp_max(len(edges)-1)], total_alpha)
    return lower, upper


def sample_corner_pixel_features(features: Tensor, pixels: Tensor):
    """Bilinear HWC sampling for COLMAP/gsplat corner-origin pixel coordinates.

    Pixel [row=0,col=0] has coordinate (.5,.5), not (0,0). This samples an
    already rasterized grid, not an exact arbitrary ray. Coordinates detach.
    """
    if features.ndim != 3 or pixels.ndim != 2 or pixels.shape[-1] != 2:
        raise ValueError("Expected HWC features and Nx2 corner-based pixel coordinates")
    height, width = features.shape[:2]
    grid = 2 * pixels.detach().to(features) / features.new_tensor([width, height]) - 1
    return F.grid_sample(features.permute(2, 0, 1)[None], grid[None, None],
                         mode="bilinear", padding_mode="zeros", align_corners=False)[0, :, 0].T


def render_front_mass(scene, K, w2c, width, height, edges):
    """Full-scene gsplat pass: geometry/cameras/colors detached, opacity live.

    Rendered features are unnormalized termination mass with a zero background;
    do not divide them by total alpha. The usual rasterizer center-depth sorting
    and clipping apply. This function does not change any parameter permission.
    """
    from gsplat import rasterization

    from .model import configure_cuda

    configure_cuda()
    s = scene.splats
    K, w2c, edges = K.detach(), w2c.detach(), edges.detach()
    if edges.ndim != 1 or not 1 <= len(edges) <= 32:
        raise ValueError("Expected 1–32 finite increasing depth edges")
    if not bool(torch.isfinite(edges).all()) or bool((edges[1:] <= edges[:-1]).any()):
        raise ValueError("Expected 1–32 finite increasing depth edges")
    means = s["means"].detach()
    center_depth = means @ w2c[2, :3] + w2c[2, 3]
    colors = (center_depth[:, None] < edges[None]).to(means.dtype).detach()
    features, alpha, _ = rasterization(
        means=means, quats=s["quats"].detach(), scales=s["log_scales"].detach().exp(),
        opacities=s["opacity_logits"].sigmoid(), colors=colors,
        viewmats=w2c[None], Ks=K[None], width=width, height=height,
        backgrounds=means.new_zeros((1, len(edges))), packed=False,
        rasterize_mode="antialiased", near_plane=.01, far_plane=1e6,
        absgrad=False, render_mode="RGB", channel_chunk=32)
    return features[0], alpha[0]


class SparseFrontSupport:
    """Cached fixed-camera SfM support for an opacity-only polishing stage.

    Uses the exact diagnostic's 32-edge family. No alpha acceptance gate and no
    upper interpolation; all target decisions/confidence are detached. The
    caller must keep per-view validity and geometry/cameras fixed.
    """

    def __init__(self, depth_support):
        if depth_support.config.uv_mode != "observed":
            raise ValueError("Front support requires actual TRAIN SfM observed UVs")
        self.support = depth_support
        self.cache = {}
        self.provenance = dict(depth_support.provenance,
                               kind="opacity_only_discrete_center_front_lower_mass",
                               channels=32, sigma_margin=3., relative_margin=.02,
                               alpha_acceptance_gate=False, alpha_normalized=False,
                               interpolation="corner-origin bilinear grid approximation",
                               edges="logspace min/max conservative cutoff; strictly floor-selected",
                               near_surface_proxy="conservative CDF bracket difference around D +/- margin")

    @torch.no_grad()
    def targets(self, view_id, K, w2c, width, height, valid):
        key = int(view_id), int(width), int(height), self.support.pixel_protocol
        if key in self.cache:
            cached = self.cache[key]
            if not torch.equal(cached["K"], K.detach()) or not torch.equal(cached["pose"], w2c.detach()):
                raise ValueError("Front support cache requires fixed cameras")
            return cached
        target = self.support.targets_for_view(view_id, K, w2c, width, height, valid)
        radius = self.support.config.border_px
        invalid = (~(torch.isfinite(valid) & (valid > 0))).float()[None, None]
        eroded = 1-F.max_pool2d(F.pad(invalid, (radius,)*4, value=1), 2*radius+1, stride=1)[0, 0]
        corner_valid = sample_corner_pixel_features(eroded[..., None], target.pixels)[:, 0] >= 1-1e-6
        indices = target.point_indices[corner_valid]
        depth, pixels, confidence = target.depth[corner_valid], target.pixels[corner_valid], target.confidence[corner_valid]
        # Match the fixed diagnostic's conditional covariance calculation on CPU
        # in float64, then return detached FP32 target values to the renderer.
        covariance = self.support.covariances[indices.cpu().numpy()]
        rotation_z = w2c.detach()[2, :3].cpu().numpy().astype(np.float64)
        variance = np.einsum("i,nij,j->n", rotation_z, covariance, rotation_z)
        std = torch.tensor(np.sqrt(np.maximum(variance, 0)), device=K.device, dtype=K.dtype)
        bins = conservative_depth_bins(depth, std, channels=32)
        cached = {"K": K.detach().clone(), "pose": w2c.detach().clone(), "pixels": pixels.detach(),
                  "depth": depth.detach(), "confidence": confidence.detach(), "bins": bins,
                  "upper_near": (2*depth-bins.cutoff).detach(), "stats": dict(target.stats)}
        self.cache[key] = cached
        return cached

    def loss(self, scene, K, w2c, width, height, valid, view_id):
        target = self.targets(view_id, K, w2c, width, height, valid)
        bins = target["bins"]
        stats = {"sparse_front_targets": len(target["depth"]),
                 "sparse_front_covered": int(bins.covered.sum()), "sparse_front_view_id": int(view_id),
                 "sparse_front_lower_loss": None, "sparse_front_upper_mass": None,
                 "sparse_front_interval_width": None, "sparse_front_total_alpha": None,
                 "sparse_front_low_alpha_count": 0, "sparse_front_near_band_lower": None,
                 "sparse_front_near_band_upper": None, "sparse_front_relative_floor_gap_median": None}
        if not bool(bins.covered.any()):
            return None, stats
        image, alpha = render_front_mass(scene, K, w2c, width, height, bins.edges)
        mass = sample_corner_pixel_features(image, target["pixels"])
        total_alpha = sample_corner_pixel_features(alpha, target["pixels"])[:, 0]
        low, high = front_mass_bounds(mass, total_alpha, bins.edges, bins.cutoff)
        near_low, near_high = front_mass_bounds(mass, total_alpha, bins.edges, target["upper_near"])
        covered, weight = bins.covered, target["confidence"][bins.covered]
        loss = (low[covered]*weight).sum()/weight.sum().clamp_min(1e-8)
        with torch.no_grad():
            weighted = lambda value: float((value[covered]*weight).sum()/weight.sum().clamp_min(1e-8))
            stats.update(sparse_front_lower_loss=float(loss.detach()),
                         sparse_front_upper_mass=weighted(high),
                         sparse_front_interval_width=weighted(high-low),
                         sparse_front_total_alpha=weighted(total_alpha),
                         sparse_front_low_alpha_count=int((total_alpha[covered] < .5).sum()),
                         sparse_front_near_band_lower=weighted((near_low-high).clamp_min(0)),
                         sparse_front_near_band_upper=weighted((near_high-low).clamp_min(0)),
                         sparse_front_relative_floor_gap_median=float(
                             (bins.gap[covered]/bins.cutoff[covered]).median()))
        return loss, stats
