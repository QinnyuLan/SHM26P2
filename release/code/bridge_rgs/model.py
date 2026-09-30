"""Explicit semantic Gaussians with view-dependent RGB and view-independent classes."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

from .refinement import MultiScaleRefinementHead, normalize_refiner_config


def configure_cuda() -> None:
    # gsplat builds once via PyTorch; the 5090 requires a Blackwell-capable toolkit.
    if "CUDA_HOME" not in os.environ and Path("/usr/local/cuda").exists():
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ.setdefault("MAX_JOBS", "4")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


class RefinementHead(nn.Module):
    """Bounded residual corrections; no image identity or test photograph input."""

    def __init__(self, feature_dim: int = 16, classes: int = 5, residual_bound: float = 1.):
        super().__init__()
        self.residual_bound = residual_bound
        channels = feature_dim + 5
        self.local = nn.Sequential(nn.Conv2d(channels, 32, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.context = nn.Sequential(nn.Conv2d(channels, 32, 3, padding=1), nn.SiLU())
        self.out = nn.Conv2d(64, classes, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features, rgb, depth, alpha, p3d=None):
        depth = torch.log1p(depth.detach().clamp_min(0))
        depth = depth / depth.amax().clamp_min(1)
        inputs = torch.cat([features, rgb.detach(), depth, alpha.detach()], -1)
        x = inputs.permute(2, 0, 1)[None]
        coarse = self.context(F.avg_pool2d(x, 4, ceil_mode=True))
        coarse = F.interpolate(coarse, x.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.out(torch.cat([self.local(x), coarse], 1))
        if self.residual_bound == 1.:
            return (2 * logits).tanh()[0].permute(1, 2, 0)
        return (self.residual_bound * (2 * logits / self.residual_bound).tanh())[0].permute(1, 2, 0)


class GaussianScene(nn.Module):
    def __init__(self, points, colors, feature_dim=16, sh_degree=3, background_points=0,
                 refiner_config=None, mip_filter_config=None):
        super().__init__()
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        if len(points) < 4:
            raise ValueError("At least four triangulated points are required")
        if colors.max() > 1:
            colors = colors / 255
        if background_points:
            # A reproducible distant shell starts sky/water coverage, then learns freely.
            center = np.median(points, axis=0)
            radius = np.percentile(np.linalg.norm(points - center, axis=-1), 95) * 4
            i = np.arange(background_points) + .5
            y = 1 - 2 * i / background_points
            a = i * math.pi * (3 - math.sqrt(5))
            xz = np.sqrt(1 - y * y)
            shell = np.stack([xz * np.cos(a), y, xz * np.sin(a)], -1) * radius + center
            points = np.concatenate([points, shell.astype(np.float32)])
            colors = np.concatenate([colors, np.tile([.55, .65, .75], (background_points, 1))])
        distances = cKDTree(points).query(points, k=4)[0][:, 1:].mean(-1)
        scale = np.clip(distances, 1e-5, np.percentile(distances, 98))
        n = len(points)
        quats = torch.zeros(n, 4)
        quats[:, 0] = 1
        self.splats = nn.ParameterDict({
            "means": nn.Parameter(torch.tensor(points)),
            "quats": nn.Parameter(quats),
            "log_scales": nn.Parameter(torch.tensor(scale).float().log()[:, None].repeat(1, 3)),
            "opacity_logits": nn.Parameter(torch.full((n,), -2.1972246)),
            "sh0": nn.Parameter((torch.tensor(colors).float() - .5)[:, None, :] / .28209479177387814),
            "sh_rest": nn.Parameter(torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3)),
            "sem_features": nn.Parameter(torch.randn(n, feature_dim) * .01),
        })
        self.semantic_decoder = nn.Linear(feature_dim, 5)
        self.register_buffer("semantic_prior_counts", torch.zeros(n, 5))
        self.refiner_config = normalize_refiner_config(refiner_config)
        if self.refiner_config["type"] == "legacy":
            self.refiner = RefinementHead(feature_dim,
                                          residual_bound=self.refiner_config.get("residual_bound", 1.))
        else:
            self.refiner = MultiScaleRefinementHead(feature_dim,
                **{key: value for key, value in self.refiner_config.items() if key != "type"})
        self.background_logits = nn.Parameter(torch.zeros(3))
        self.sh_degree = sh_degree
        self.feature_dim = feature_dim
        self.scene_scale = float(np.percentile(np.linalg.norm(points - np.median(points, axis=0), axis=-1), 90))
        from .mip_filter import normalize_config
        self.mip_filter_config = normalize_config(mip_filter_config)
        self.register_buffer("mip_filter_rho", torch.zeros(n, 1) if self.mip_filter_config else None)
        self.mip_filter_state = None

    def filtered_parameters(self):
        """Enabled filter's shared footprint; legacy/off never enters this path."""
        from .mip_filter import effective_parameters
        if self.mip_filter_config is None or self.mip_filter_state is None:
            raise ValueError("Enabled mip filter must be initialized from TRAIN cameras or restored")
        return effective_parameters(self.splats["log_scales"], self.splats["opacity_logits"], self.mip_filter_rho)

    def render(self, K, w2c, width, height, degree=None, semantics=True,
               geometry_grad=False, refine=True, absgrad=True, semantic_classifier_grad=True,
               refinement_grad_to_field=True):
        """Render a view, optionally isolating final-mask gradients to the refiner.

        Disabling ``refinement_grad_to_field`` detaches both refinement inputs
        and the log-probability base of ``probabilities``. The returned ``p3d``
        remains differentiable for an independent raw semantic loss, subject
        to the existing geometry/classifier permissions. Forward values and
        checkpoint parameters are unchanged.
        """
        configure_cuda()
        from gsplat import rasterization

        s = self.splats
        filtered = self.filtered_parameters() if self.mip_filter_config is not None else None
        degree = self.sh_degree if degree is None else min(degree, self.sh_degree)
        common = {"viewmats": w2c[None], "Ks": K[None], "width": width, "height": height,
                  "packed": False, "rasterize_mode": "antialiased", "near_plane": .01,
                  "far_plane": 1e6, "absgrad": absgrad}
        rgbd, alpha, info = rasterization(
            means=s["means"], quats=s["quats"], scales=s["log_scales"].exp() if filtered is None else filtered[0],
            opacities=s["opacity_logits"].sigmoid() if filtered is None else filtered[1], colors=torch.cat([s["sh0"], s["sh_rest"]], 1),
            sh_degree=degree, render_mode="RGB+ED",
            backgrounds=self.background_logits.sigmoid()[None], **common)
        result = {"rgb": rgbd[0, ..., :3], "depth": rgbd[0, ..., 3:4],
                  "alpha": alpha[0], "info": info}
        if semantics:
            if semantic_classifier_grad:
                logits = self.semantic_decoder(s["sem_features"])
            else:
                logits = F.linear(s["sem_features"], self.semantic_decoder.weight.detach(),
                                  self.semantic_decoder.bias.detach())
            p = logits.softmax(-1)
            values = torch.cat([p, s["sem_features"]], -1)
            with_moments = self.refiner_config.get("depth_moments", "off") != "off"
            if with_moments:
                z = (s["means"].detach() @ w2c.detach()[2, :3] + w2c.detach()[2, 3])
                z = z[:, None] / max(self.scene_scale, 1e-6)
                values = torch.cat((values, z, z.square(), z * s["sem_features"]), -1)
            bg = torch.zeros(1, values.shape[-1], device=K.device)
            bg[0, 0] = 1
            sem_common = dict(common)
            sem_common["viewmats"] = w2c.detach()[None]
            sem_common["absgrad"] = False
            geom = lambda name: s[name] if geometry_grad else s[name].detach()
            sem_filtered = None if filtered is None else tuple(v if geometry_grad else v.detach() for v in filtered)
            semantic, semantic_alpha, _ = rasterization(
                means=geom("means"), quats=geom("quats"), scales=geom("log_scales").exp() if sem_filtered is None else sem_filtered[0],
                opacities=geom("opacity_logits").sigmoid() if sem_filtered is None else sem_filtered[1], colors=values,
                backgrounds=bg, **sem_common)
            p3d = semantic[0, ..., :5].clamp_min(1e-7)
            p3d = p3d / p3d.sum(-1, keepdim=True)
            refinement_features = semantic[0, ..., 5:5+self.feature_dim]
            context_kwargs = {}
            if with_moments:
                from .depth_moments import semantic_depth_context
                start = 5 + self.feature_dim
                context = semantic_depth_context(
                    refinement_features, semantic[0, ..., start:start+1],
                    semantic[0, ..., start+1:start+2], semantic[0, ..., start+2:],
                    semantic_alpha[0])
                if not refinement_grad_to_field:
                    context = context.detach()
                context_kwargs["depth_moments"] = context
                result["depth_moments"] = context
            refinement_prior = p3d
            if not refinement_grad_to_field:
                refinement_features = refinement_features.detach()
                refinement_prior = refinement_prior.detach()
            residual = self.refiner(refinement_features, result["rgb"], result["depth"],
                                    result["alpha"], p3d=refinement_prior,
                                    **context_kwargs) if refine else torch.zeros_like(p3d)
            result.update(p3d=p3d, probabilities=(refinement_prior.log() + residual).softmax(-1),
                          residual=residual, features=refinement_features,
                          refinement_prior=refinement_prior)
        return result

    def optimizers(self, lr_scale=1.0):
        rates = {"means": 1.6e-4 * self.scene_scale, "quats": 1e-3, "log_scales": 5e-3,
                 "opacity_logits": .05, "sh0": .0025, "sh_rest": .000125, "sem_features": .01}
        opts = {k: torch.optim.Adam([v], lr=rates[k] * lr_scale, eps=1e-15)
                for k, v in self.splats.items()}
        opts["heads"] = torch.optim.Adam([
            {"params": self.semantic_decoder.parameters(), "lr": 1e-3 * lr_scale},
            {"params": self.refiner.parameters(), "lr": 3e-4 * lr_scale},
            {"params": [self.background_logits], "lr": .001 * lr_scale}])
        return opts

    @torch.no_grad()
    def render_evidence_gate(self, indices, weights, K, w2c, width, height):
        from gsplat import rasterization
        s = self.splats
        filtered = self.filtered_parameters() if self.mip_filter_config is not None else None
        full_weights = torch.zeros((len(s["means"]), 1), device=weights.device, dtype=weights.dtype)
        full_weights[indices, 0] = weights
        confidence, alpha, _ = rasterization(
            means=s["means"], quats=s["quats"],
            scales=s["log_scales"].exp() if filtered is None else filtered[0],
            opacities=s["opacity_logits"].sigmoid() if filtered is None else filtered[1],
            colors=full_weights, viewmats=w2c[None], Ks=K[None], width=width, height=height,
            packed=False, rasterize_mode="antialiased", near_plane=.01, far_plane=1e6)
        return (confidence[0, ..., 0] / alpha[0, ..., 0].clamp_min(1e-6)).clamp(0, 1) * (alpha[0, ..., 0] > .15)
