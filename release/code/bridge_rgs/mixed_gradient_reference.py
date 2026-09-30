"""RGB140-inspired signed-clone/absolute-split reference, not RGB150 reproduction.

Uses gsplat 1.5.3 screen-gradient units and Standard-style primitive operations.
The explicit cap ranks eligible parents by gradient / their own threshold. It
never recycles points or consults semantics, PCA, residuals, or camera quality.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .densification import (
    DensificationResult,
    apply_densification,
    normalized_absgrad,
    quaternion_to_matrix,
)

MODE = "rgb140_mixed"
CONVENTION = "rgb140_mixed_gsplat_normalized_v1"


@dataclass(frozen=True)
class MixedGradientConfig:
    signed_clone_threshold: float = .0002
    absolute_split_threshold: float = .0004
    small_scale_fraction: float = .01
    prune_scale_fraction: float = .1
    prune_opacity: float = .005
    max_gaussians: int = 500_000
    refine_start: int = 500
    refine_stop: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000
    reset_opacity: float = .01

    def __post_init__(self):
        for key in ("signed_clone_threshold", "absolute_split_threshold", "small_scale_fraction",
                    "prune_scale_fraction", "prune_opacity", "reset_opacity"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not self.prune_opacity < self.reset_opacity < 1:
            raise ValueError("Require prune_opacity < reset_opacity < 1")
        for key in ("max_gaussians", "refine_start", "refine_stop", "refine_every", "reset_every"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.refine_start >= self.refine_stop:
            raise ValueError("refine_start must precede refine_stop")

    def grows_at(self, step):
        return self.refine_start < step < self.refine_stop and step % self.refine_every == 0

    def resets_at(self, step):
        return 0 < step < self.refine_stop and step % self.reset_every == 0


def validate_reference_training(config):
    """Reject accidental activation of the existing geometry/semantic policies."""
    policy = MixedGradientConfig(**config.get("mixed_gradient", {}))
    if config.get("max_gaussians", policy.max_gaussians) != policy.max_gaussians:
        raise ValueError("max_gaussians must match mixed_gradient.max_gaussians")
    if config.get("parameter_scope", "all") != "all" or config.get("freeze_geometry") or config.get("freeze_rgb"):
        raise ValueError("RGB reference requires the unfrozen all-parameter scope")
    if config.get("warmstart"):
        raise ValueError("RGB reference starts from prepared initialization, not a warmstart")
    steps = config.get("steps", 30000)
    if any(config.get(key, 0) <= steps for key in ("semantic_start", "refine_start")):
        raise ValueError("RGB reference requires semantics/refiner disabled for the entire run")
    for key in ("optimize_cameras", "camera_attribution", "camera_quality_weighting", "region_rgb_weight",
                "semantic_weight", "semantic_geometry_weight", "pseudo_dir", "multiview_fusion",
                "structure_guided", "recycle_count", "foreground_allocation_fraction", "absgrad_quantile",
                "sparse_depth_weight", "sparse_front_weight", "opacity_entropy_weight", "train_labeled_only"):
        if config.get(key):
            raise ValueError(f"RGB reference forbids {key}")
    if config.get("opacity_lr", .025) != .025:
        raise ValueError("RGB140-inspired reference fixes opacity_lr=0.025")
    # These old-strategy knobs do not drive the independent policy. Reject
    # them instead of silently accepting contradictory top-level settings.
    for key in ("densify_start", "densify_stop", "densify_every", "densify_min_views", "max_splits",
                "densify_fraction", "density_budget_schedule", "absgrad_threshold", "clone_scale_fraction",
                "split_screen_radius", "split_opacity_mode", "structured_split_rule", "prune_opacity",
                "opacity_reset_every", "opacity_reset_start", "opacity_reset_stop", "opacity_reset_cap",
                "opacity_reset_pause"):
        if key in config:
            raise ValueError(f"Configure reference policy through mixed_gradient, not {key}")
    return policy


class MixedGradientState:
    def __init__(self, count, device, policy=None):
        self.policy = policy or MixedGradientConfig()
        if count > self.policy.max_gaussians:
            raise ValueError("Initial field exceeds reference cap; never silently delete to fit")
        self.signed_sum = torch.zeros(count, device=device)
        self.absolute_sum = torch.zeros(count, device=device)
        self.observations = torch.zeros(count, device=device, dtype=torch.long)

    def state_dict(self):
        return {"convention": CONVENTION, "policy": asdict(self.policy),
                "signed_sum": self.signed_sum.detach().cpu(),
                "absolute_sum": self.absolute_sum.detach().cpu(),
                "observations": self.observations.detach().cpu()}

    def load_state_dict(self, state):
        if state.get("convention") != CONVENTION or state.get("policy") != asdict(self.policy):
            raise ValueError("Mixed-gradient resume convention or policy changed")
        for key in ("signed_sum", "absolute_sum", "observations"):
            old, value = getattr(self, key), state[key]
            if value.shape != old.shape or value.dtype != old.dtype:
                raise ValueError(f"Invalid mixed-gradient window shape/dtype: {key}")
            if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError(f"Invalid mixed-gradient window values: {key}")
            old.copy_(value.to(old.device))

    def retain_grad(self, info, step):
        if step < self.policy.refine_stop:
            info["means2d"].retain_grad()

    @torch.no_grad()
    def accumulate(self, info, step):
        if step >= self.policy.refine_stop:
            return
        projected = info["means2d"]
        signed, absolute = projected.grad, getattr(projected, "absgrad", None)
        if signed is None or absolute is None:
            raise RuntimeError("Mixed-gradient reference requires retained signed and renderer absgrad")
        if signed.shape != (1, len(self.observations), 2) or absolute.shape != signed.shape:
            raise ValueError("Reference expects unpacked single-camera screen gradients")
        radii = info["radii"][0]
        visible = (radii > 0).all(-1) if radii.ndim == 2 else radii > 0
        if visible.shape != self.observations.shape or info.get("n_cameras", 1) != 1:
            raise ValueError("Reference expects unpacked single-camera visibility")
        for grad, accumulator in ((signed, self.signed_sum), (absolute, self.absolute_sum)):
            values = normalized_absgrad(grad[0], info["width"], info["height"])
            if not bool(torch.isfinite(values[visible]).all()):
                raise FloatingPointError("Nonfinite visible screen gradient")
            accumulator[visible] += values[visible]
        self.observations += visible.long()


@torch.no_grad()
def mixed_gradient_densify(parameters, state, scene_scale, step, generator=None):
    """Grow then prune; cap uses original N, no speculative prune/free slots.

    Each clone or two-child split adds exactly one primitive. Ties in the
    threshold-normalized score use original point index. Standard's stochastic
    child offsets use torch RNG, which ordinary checkpoints already preserve.
    """
    policy = state.policy
    if not policy.grows_at(step):
        raise ValueError("Not a reference densification step")
    n = len(parameters["means"])
    if n > policy.max_gaussians or state.observations.shape != (n,):
        raise ValueError("Invalid reference point count or gradient window")
    if not math.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError("scene_scale must be finite and positive")
    if any(not bool(torch.isfinite(value).all()) for value in parameters.values()):
        raise FloatingPointError("Nonfinite reference field")
    maximum_scale = parameters["log_scales"].exp().amax(-1)
    small = maximum_scale <= policy.small_scale_fraction * scene_scale
    signed = state.signed_sum / state.observations.clamp_min(1)
    absolute = state.absolute_sum / state.observations.clamp_min(1)
    priority = torch.where(small, signed / policy.signed_clone_threshold,
                           absolute / policy.absolute_split_threshold)
    eligible = (state.observations > 0) & (priority > 1)
    indices = torch.where(eligible)[0]
    order = torch.argsort(priority[indices], descending=True, stable=True)
    chosen = indices[order[:max(0, policy.max_gaussians - n)]]
    # Sort operation masks by source ID, as in gsplat's mask.nonzero().
    clone_ids = chosen[small[chosen]].sort().values
    split_ids = chosen[~small[chosen]].sort().values
    split_mask = torch.zeros(n, dtype=torch.bool, device=small.device)
    split_mask[split_ids] = True
    retained = torch.where(~split_mask)[0]
    sources = torch.cat((retained, clone_ids, split_ids.repeat(2)))
    reset = torch.arange(len(sources), device=small.device) >= len(retained)
    tensors = {key: value[sources].clone() for key, value in parameters.items()}
    if len(split_ids):
        scales = parameters["log_scales"][split_ids].exp()
        rotations = quaternion_to_matrix(parameters["quats"][split_ids])
        noise = torch.randn((2, len(split_ids), 3), device=scales.device,
                            dtype=scales.dtype, generator=generator)
        offsets = torch.einsum("nij,nj,bnj->bni", rotations, scales, noise)
        tensors["means"][-2*len(split_ids):] = (parameters["means"][split_ids] + offsets).reshape(-1, 3)
        tensors["log_scales"][-2*len(split_ids):] = (scales / 1.6).log().repeat(2, 1)
    prune = tensors["opacity_logits"].sigmoid() < policy.prune_opacity
    if step > policy.reset_every:
        prune |= tensors["log_scales"].exp().amax(-1) > policy.prune_scale_fraction * scene_scale
    if bool(prune.all()):
        raise RuntimeError("Standard pruning removed the whole field; no silent rescue")
    result = DensificationResult(
        {key: value[~prune] for key, value in tensors.items()}, sources[~prune], reset[~prune],
        split_count=len(split_ids), pruned_count=int(prune.sum()), line_splits=0, plane_splits=0,
        duplicate_count=len(clone_ids))
    diagnostics = {"density_eligible_candidates": int(eligible.sum()),
                   "density_clone_candidates": int((eligible & small).sum()),
                   "density_split_candidates": int((eligible & ~small).sum()),
                   "density_cap_rejected": int(eligible.sum()) - len(chosen),
                   "density_target_budget": policy.max_gaussians}
    return result, diagnostics


@torch.no_grad()
def standard_opacity_reset(opacity_logits, optimizer, ceiling=.01):
    """Standard cap plus zero moments for ALL opacity rows; retain Adam step."""
    if not 0 < ceiling < 1:
        raise ValueError("Opacity ceiling must be in (0,1)")
    limit = torch.logit(opacity_logits.new_tensor(ceiling))
    changed = int((opacity_logits > limit).sum())
    opacity_logits.clamp_(max=limit)
    for value in optimizer.state.get(opacity_logits, {}).values():
        if isinstance(value, torch.Tensor) and value.shape == opacity_logits.shape:
            value.zero_()
    return changed


def reference_post_step(parameters, optimizers, state, scene_scale, step):
    """The trainer and CPU contracts share grow -> prune -> reset ordering."""
    policy = state.policy
    selection, diagnostics = None, {}
    if policy.grows_at(step):
        selection, diagnostics = mixed_gradient_densify(parameters, state, scene_scale, step)
        apply_densification(parameters, selection, optimizers)
        state = MixedGradientState(len(parameters["means"]), parameters["means"].device, policy)
        diagnostics.update(splits=selection.split_count, duplicates=selection.duplicate_count,
                           pruned=selection.pruned_count)
    if policy.resets_at(step):
        diagnostics["opacity_reset_count"] = standard_opacity_reset(
            parameters["opacity_logits"], optimizers["opacity_logits"], policy.reset_opacity)
        diagnostics["opacity_reset_step"] = step
    return state, selection, diagnostics
