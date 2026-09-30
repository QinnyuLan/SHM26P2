"""Multiscale semantic correction from rendered 3D evidence only.

The decoder never accepts a photograph, label, view index or camera identity.
Its inputs are Gaussian-rendered features, color, depth, coverage and class
probabilities. It complements the explicit 3D field without replacing it.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def normalize_refiner_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize checkpoint architecture metadata and reject ignored options.

    Legacy defaults remain exactly ``{"type": "legacy"}``, so absent metadata in
    an old checkpoint loads its original module and parameter names unchanged.
    """
    values = dict(config or {})
    kind = values.pop("type", "legacy")
    if kind == "legacy":
        bound = float(values.pop("residual_bound", 1.))
        if values:
            raise ValueError(f"Unknown legacy refiner options: {sorted(values)}")
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("refiner residual_bound must be finite and positive")
        result = {"type": "legacy"}
        if bound != 1.:
            result["residual_bound"] = bound
        return result
    if kind != "multiscale":
        raise ValueError(f"Unknown refiner type: {kind!r}")
    channels = int(values.pop("channels", 64))
    bound = float(values.pop("residual_bound", 6.))
    context = values.pop("context", "pyramid_strip")
    depth_moments = values.pop("depth_moments", "off")
    if values:
        raise ValueError(f"Unknown multiscale refiner options: {sorted(values)}")
    if channels < 16 or channels % 8:
        raise ValueError("multiscale refiner channels must be a multiple of 8 and >= 16")
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("refiner residual_bound must be finite and positive")
    if context not in {"none", "pyramid", "pyramid_strip"}:
        raise ValueError("refiner context must be none, pyramid or pyramid_strip")
    if depth_moments not in {"off", "zero", "variance", "cross"}:
        raise ValueError("depth_moments must be off, zero, variance or cross")
    result = {"type": "multiscale", "channels": channels,
              "residual_bound": bound, "context": context}
    if depth_moments != "off":
        result["depth_moments"] = depth_moments
    return result


def _groups(channels: int) -> int:
    # Keep >= 2 channels per group, including batch=1/global-pool=1x1.
    return next(g for g in (8, 4, 2, 1) if channels % g == 0 and channels // g >= 2)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel, stride,
                                   padding=kernel//2, bias=False),
                         nn.GroupNorm(_groups(out_channels), out_channels), nn.SiLU())


class SeparableResidual(nn.Module):
    """Spatial mixing with a small activation/parameter budget at fine scales."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels))

    def forward(self, x):
        return F.silu(x + self.block(x))


class PanoramaContext(nn.Module):
    """Scene-level pooling and directional strips on the 1/16 feature map.

    Pooled branches supply long-range grouping for annotated cable regions;
    horizontal/vertical strips retain one spatial axis. This is generic context,
    not a bridge-coordinate template or label-specific geometric primitive.
    """
    def __init__(self, in_channels, out_channels, mode="pyramid_strip"):
        super().__init__()
        self.mode = mode
        branch_channels = max(8, out_channels // 4)
        self.bin_sizes = (1, 2, 4, 8) if mode != "none" else ()
        self.pooled = nn.ModuleList([
            ConvNormAct(in_channels, branch_channels, kernel=1) for _ in self.bin_sizes])
        self.strips = nn.ModuleList([
            ConvNormAct(in_channels, branch_channels, kernel=1) for _ in range(2)
        ]) if mode == "pyramid_strip" else nn.ModuleList()
        merged = in_channels + branch_channels * (len(self.pooled) + len(self.strips))
        self.merge = ConvNormAct(merged, out_channels, kernel=1)

    def forward(self, x):
        size = x.shape[-2:]
        branches = [x]
        for bins, project in zip(self.bin_sizes, self.pooled):
            pooled = F.adaptive_avg_pool2d(x, (min(bins, size[0]), min(bins, size[1])))
            branches.append(F.interpolate(project(pooled), size, mode="bilinear", align_corners=False))
        if self.strips:
            across_width = self.strips[0](x.mean(-1, keepdim=True))
            across_height = self.strips[1](x.mean(-2, keepdim=True))
            branches.extend([across_width.expand(-1, -1, -1, size[1]),
                             across_height.expand(-1, -1, size[0], -1)])
        return self.merge(torch.cat(branches, dim=1))


def _gradient_magnitude(x):
    dx = F.pad(x[..., 1:] - x[..., :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return (dx.square() + dy.square() + 1e-12).sqrt()


class MultiScaleRefinementHead(nn.Module):
    """FPN + panoramic context + native-resolution boundary correction.

    Zero initialization preserves the input 3D probabilities on insertion. The
    residual bound is explicit and configurable, unlike the old fixed ±1 head.
    RGB, depth and coverage are detached by default. The explicit geometry_grad
    option exposes their input derivatives without changing the forward or any
    checkpoint parameters; it does not unfreeze the head's parameters.
    """
    def __init__(self, feature_dim=16, classes=5, channels=64, residual_bound=6.,
                 context="pyramid_strip", depth_moments="off"):
        super().__init__()
        effective = normalize_refiner_config({"type": "multiscale", "channels": channels,
                                               "residual_bound": residual_bound, "context": context,
                                               "depth_moments": depth_moments})
        self.architecture_config = effective
        self.depth_moments_mode = depth_moments
        self.feature_dim = feature_dim
        self.classes = classes
        self.residual_bound = effective["residual_bound"]
        c = effective["channels"]
        # Rendered F + RGB(3) + depth(1) + alpha(1) + p3d(C) + entropy(1)
        # + RGB/depth gradient magnitude(2).
        inputs = feature_dim + classes + 8
        half_channels, detail_channels = c // 2, max(8, c // 4)
        self.detail = nn.Sequential(ConvNormAct(inputs, detail_channels),
                                    SeparableResidual(detail_channels))
        self.encoder_half = nn.Sequential(ConvNormAct(inputs, half_channels, stride=2),
                                          SeparableResidual(half_channels))
        self.encoder_quarter = nn.Sequential(ConvNormAct(half_channels, c, stride=2),
                                             SeparableResidual(c), SeparableResidual(c))
        self.encoder_eighth = nn.Sequential(ConvNormAct(c, c*2, stride=2),
                                            SeparableResidual(c*2), SeparableResidual(c*2))
        self.encoder_sixteenth = nn.Sequential(ConvNormAct(c*2, c*3, stride=2),
                                               SeparableResidual(c*3), SeparableResidual(c*3))
        self.panorama = PanoramaContext(c*3, c, effective["context"])
        self.lateral_eighth = nn.Conv2d(c*2, c, 1)
        self.lateral_quarter = nn.Conv2d(c, c, 1)
        self.decode_eighth = SeparableResidual(c)
        self.decode_quarter = SeparableResidual(c)
        self.decode_half = nn.Sequential(ConvNormAct(c + half_channels, half_channels),
                                         SeparableResidual(half_channels))
        self.fuse_detail = nn.Sequential(
            ConvNormAct(half_channels + detail_channels, half_channels),
            SeparableResidual(half_channels))
        self.out = nn.Conv2d(half_channels, classes, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        if depth_moments != "off":
            # Independent additive paths retain all pretrained stem weights and
            # their convolution shapes. Three information controls have the same
            # parameterization; only the supplied context channels differ.
            self.moment_detail = nn.Conv2d(feature_dim + 1, detail_channels, 3, padding=1,
                                           bias=False)
            self.moment_half = nn.Conv2d(feature_dim + 1, half_channels, 3, stride=2,
                                         padding=1, bias=False)
            nn.init.zeros_(self.moment_detail.weight)
            nn.init.zeros_(self.moment_half.weight)

    def evidence_tensor(self, features, rgb, depth, alpha, p3d=None, *, geometry_grad=False):
        """Convert HWC rendered evidence to one BCHW tensor; no external image."""
        if type(geometry_grad) is not bool:
            raise TypeError("geometry_grad must be bool")
        if features.ndim != 3 or rgb.shape[:2] != features.shape[:2]:
            raise ValueError("Refinement inputs must share an HWC image grid")
        features = features.permute(2, 0, 1)[None]
        rgb = (rgb if geometry_grad else rgb.detach()).permute(2, 0, 1)[None]
        depth = torch.log1p((depth if geometry_grad else depth.detach()).clamp_min(0)).permute(2, 0, 1)[None]
        depth = depth / depth.amax().clamp_min(1)
        alpha = (alpha if geometry_grad else alpha.detach()).permute(2, 0, 1)[None].clamp(0, 1)
        if p3d is None:
            p3d = features.new_full((1, self.classes, *features.shape[-2:]), 1/self.classes)
        else:
            if p3d.shape != (*rgb.shape[-2:], self.classes):
                raise ValueError("p3d must be HWC with the configured semantic class count")
            p3d = p3d.permute(2, 0, 1)[None].clamp_min(1e-7)
            p3d = p3d / p3d.sum(1, keepdim=True)
        entropy = -(p3d * p3d.log()).sum(1, keepdim=True) / math.log(self.classes)
        gray = (rgb * rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
        return torch.cat([features, rgb, depth, alpha, p3d, entropy,
                          _gradient_magnitude(gray), _gradient_magnitude(depth)], dim=1)

    def forward(self, features, rgb, depth, alpha, p3d=None, depth_moments=None, *, geometry_grad=False):
        x = self.evidence_tensor(features, rgb, depth, alpha, p3d, geometry_grad=geometry_grad)
        if self.depth_moments_mode == "off":
            detail = self.detail(x)
            half = self.encoder_half(x)
        else:
            if depth_moments is None or depth_moments.shape != (*features.shape[:2], self.feature_dim + 1):
                raise ValueError("Moment-enabled refiner requires aligned depth moment context")
            moments = depth_moments.permute(2, 0, 1)[None]
            if self.depth_moments_mode == "zero":
                moments = torch.zeros_like(moments)
            elif self.depth_moments_mode == "variance":
                moments = torch.cat((moments[:, :1], torch.zeros_like(moments[:, 1:])), dim=1)
            detail = self.detail[1](self.detail[0](x) + self.moment_detail(moments))
            half = self.encoder_half[1](self.encoder_half[0](x) + self.moment_half(moments))
        quarter = self.encoder_quarter(half)
        eighth = self.encoder_eighth(quarter)
        sixteenth = self.encoder_sixteenth(eighth)
        top = self.panorama(sixteenth)
        eighth_decoded = self.decode_eighth(self.lateral_eighth(eighth) +
            F.interpolate(top, eighth.shape[-2:], mode="bilinear", align_corners=False))
        quarter_decoded = self.decode_quarter(self.lateral_quarter(quarter) +
            F.interpolate(eighth_decoded, quarter.shape[-2:], mode="bilinear", align_corners=False))
        half_decoded = self.decode_half(torch.cat([half,
            F.interpolate(quarter_decoded, half.shape[-2:], mode="bilinear", align_corners=False)], 1))
        decoded = self.fuse_detail(torch.cat([detail,
            F.interpolate(half_decoded, x.shape[-2:], mode="bilinear", align_corners=False)], 1))
        logits = self.out(decoded)
        # Remove the softmax-invariant common offset before bounding the logits.
        logits = logits - logits.mean(1, keepdim=True)
        residual = self.residual_bound * torch.tanh(logits / self.residual_bound)
        return residual[0].permute(1, 2, 0)


def add_depth_moment_paths(source, target):
    """Copy a trained multiscale head and keep only the new additive paths zero."""
    if not isinstance(source, MultiScaleRefinementHead) or not isinstance(target, MultiScaleRefinementHead):
        raise TypeError("Depth moment warmstart requires two multiscale heads")
    if source.depth_moments_mode != "off" or target.depth_moments_mode == "off":
        raise ValueError("Only an off-to-enabled depth moment warmstart is supported")
    a, b = [dict(head.architecture_config) for head in (source, target)]
    a.pop("depth_moments", None)
    b.pop("depth_moments", None)
    if a != b or source.feature_dim != target.feature_dim or source.classes != target.classes:
        raise ValueError("Moment warmstart cannot change other head architecture settings")
    missing, unexpected = target.load_state_dict(source.state_dict(), strict=False)
    expected = {"moment_detail.weight", "moment_half.weight"}
    if set(missing) != expected or unexpected:
        raise ValueError("Unexpected moment warmstart state schema")
    with torch.no_grad():
        target.moment_detail.weight.zero_()
        target.moment_half.weight.zero_()
