"""Bounded gsplat 1.5.3 MCMC recipe adapter, not a new sampling method.

Topology and Adam migration are delegated unchanged to the installed strategy.
Only the dictionary names and world/canonical length units are adapted here.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from importlib.metadata import version

import torch

MODE = "mcmc_reference"
CONVENTION = "gsplat_1.5.3_mcmc_canonical_scene_scale_v1"
ALIASES = {"log_scales": "scales", "opacity_logits": "opacities"}


@dataclass(frozen=True)
class MCMCReferenceConfig:
    cap_max: int = 500_000
    noise_lr: float = 5e5
    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    refine_every: int = 100
    min_opacity: float = .005
    opacity_reg: float = .01
    scale_reg: float = .01

    def __post_init__(self):
        for key in ("cap_max", "refine_start_iter", "refine_stop_iter", "refine_every"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.refine_start_iter >= self.refine_stop_iter:
            raise ValueError("Refinement start must precede stop")
        for key in ("noise_lr", "min_opacity", "opacity_reg", "scale_reg"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.min_opacity >= 1:
            raise ValueError("min_opacity must be below one")

    def refines_at(self, step):
        return self.refine_start_iter < step < self.refine_stop_iter and step % self.refine_every == 0

    def strategy_options(self):
        return {key: value for key, value in asdict(self).items()
                if key not in {"opacity_reg", "scale_reg"}}


def validate_reference_training(config):
    """Production recipe is fixed; smaller helper policies are for CPU contracts only."""
    policy = MCMCReferenceConfig(**config.get("mcmc_reference", {}))
    if policy != MCMCReferenceConfig():
        raise ValueError("MCMC reference recipe is fixed; policy overrides are not supported")
    if config.get("densification") != MODE:
        raise ValueError("MCMC reference requires its own densification mode")
    if config.get("max_gaussians", policy.cap_max) != policy.cap_max:
        raise ValueError("max_gaussians must match MCMC cap_max")
    if config.get("parameter_scope", "all") != "all" or config.get("freeze_geometry") or config.get("freeze_rgb"):
        raise ValueError("MCMC reference requires unfrozen RGB geometry")
    if config.get("warmstart"):
        raise ValueError("MCMC RGB reference starts from prepared initialization")
    for key, expected in (("train_scale", 1.), ("initial_opacity", .5),
                          ("initial_scale_multiplier", .1)):
        value = config.get(key, expected)
        if isinstance(value, bool) or value != expected:
            raise ValueError(f"MCMC reference fixes {key}={expected}")
    if config.get("progressive_resolution", False) is not False:
        raise ValueError("MCMC reference uses native resolution for every step")
    if config.get("independent_view_rng") is not True or config.get("view_sampling") != "shuffle":
        raise ValueError("MCMC reference requires independent shuffled TRAIN sampling")
    steps = config.get("steps", 30000)
    if any(config.get(key, 0) <= steps for key in ("semantic_start", "refine_start")):
        raise ValueError("MCMC reference requires semantics and refiner disabled")
    for key in ("optimize_cameras", "camera_attribution", "camera_quality_weighting", "region_rgb_weight",
                "semantic_weight", "semantic_geometry_weight", "semantic_initialization", "pseudo_dir",
                "multiview_fusion", "structure_guided", "recycle_count", "foreground_allocation_fraction",
                "absgrad_quantile", "sparse_depth_weight", "sparse_front_weight", "opacity_entropy_weight",
                "train_labeled_only", "mixed_gradient"):
        if config.get(key):
            raise ValueError(f"MCMC reference forbids {key}")
    mip = config.get("mip_filter")
    if mip not in (None, False, {"enabled": False}):
        raise ValueError("MCMC reference forbids Mip filtering")
    if config.get("opacity_lr", .05) != .05:
        raise ValueError("MCMC reference fixes opacity_lr=0.05")
    for key in ("densify_start", "densify_stop", "densify_every", "densify_min_views", "max_splits",
                "densify_fraction", "density_budget_schedule", "absgrad_threshold", "clone_scale_fraction",
                "split_screen_radius", "split_opacity_mode", "structured_split_rule", "prune_opacity",
                "opacity_reset_every", "opacity_reset_start", "opacity_reset_stop", "opacity_reset_cap",
                "opacity_reset_pause"):
        if key in config:
            raise ValueError(f"Configure only the MCMC recipe, not old policy field {key}")
    return policy


def canonical_noise_lr(means_lr, scene_scale):
    """Return the lr expected by the installed *world-covariance* noise kernel.

    x'=x/L, Sigma'=Sigma/L^2, lr'=lr/L.  Mapping L*Sigma'*lr'
    back to world units gives Sigma*lr/L^2.  The strategy adds noise_lr itself.
    """
    if not math.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError("scene_scale must be finite and positive")
    if not math.isfinite(means_lr) or means_lr < 0:
        raise ValueError("means_lr must be finite and nonnegative")
    return means_lr / (scene_scale * scene_scale)


def reference_regularization(scene, policy):
    """Canonical-unit recipe; diagnostics are detached tensors, not new gradients."""
    canonical_noise_lr(0., scene.scene_scale)
    opacity = policy.opacity_reg * scene.splats["opacity_logits"].sigmoid().mean()
    scale = policy.scale_reg * scene.splats["log_scales"].exp().mean() / scene.scene_scale
    return opacity + scale, {"mcmc_opacity_regularization": opacity.detach(),
                             "mcmc_scale_regularization": scale.detach()}


def _aliases(scene, optimizers):
    params, opts = {}, {}
    n = len(scene.splats["means"])
    if n == 0:
        raise ValueError("MCMC requires at least one Gaussian")
    if getattr(scene, "mip_filter_config", None) is not None:
        raise ValueError("MCMC reference does not support Mip buffers")
    for name, param in scene.splats.items():
        alias = ALIASES.get(name, name)
        if alias in params:
            raise ValueError("Splat names collide after MCMC alias mapping")
        if param.ndim == 0 or len(param) != n or not isinstance(param, torch.nn.Parameter):
            raise ValueError(f"MCMC splat {name} must be an N-row Parameter")
        if param.dtype != torch.float32 or param.device != scene.splats["means"].device:
            raise ValueError("Installed MCMC reference requires co-located FP32 splats")
        params[alias] = param
        if param.requires_grad:
            if name not in optimizers:
                raise ValueError(f"Missing splat optimizer {name}")
            opt = optimizers[name]
            if len(opt.param_groups) != 1 or len(opt.param_groups[0]["params"]) != 1:
                raise ValueError(f"MCMC requires one parameter per optimizer: {name}")
            if opt.param_groups[0]["params"][0] is not param:
                raise ValueError(f"Optimizer/Parameter identity mismatch for {name}")
            opts[alias] = opt
        elif name in optimizers:
            # Keeping an optimizer for a frozen field is safe only while unused.
            # Strategy must still rebind it after topology, so require the standard
            # RGB reference's trainable splat collection instead of silently drifting.
            raise ValueError(f"Frozen splat optimizer unsupported in MCMC: {name}")
    for alias in ("means", "quats", "scales", "opacities"):
        if alias not in params or not params[alias].requires_grad:
            raise ValueError(f"MCMC RGB geometry must be trainable: {alias}")
    return params, opts


def _zero_prior(scene):
    prior = scene.semantic_prior_counts
    if prior.shape != (len(scene.splats["means"]), 5) or torch.count_nonzero(prior).item():
        raise ValueError("MCMC reference requires an all-zero N x 5 semantic_prior_counts buffer")


def _new_strategy(policy):
    if version("gsplat") != "1.5.3":
        raise ValueError("MCMC reference is pinned to gsplat 1.5.3")
    from gsplat import MCMCStrategy
    return MCMCStrategy(**policy.strategy_options(), verbose=False)


class MCMCReferenceState:
    def __init__(self, scene_scale, initial_count, policy):
        canonical_noise_lr(0., scene_scale)
        if initial_count > policy.cap_max:
            raise ValueError("Initial field exceeds MCMC cap; never silently prune it")
        self.policy = policy
        self.scene_scale = float(scene_scale)
        self.strategy = _new_strategy(policy)
        self.strategy_state = self.strategy.initialize_state()
        self.counters = {"last_step": 0, "initial_gaussians": initial_count,
                         "current_gaussians": initial_count, "peak_gaussians": initial_count,
                         "refinement_events": 0, "relocation_events": 0, "relocated_gaussians": 0,
                         "added_gaussians": 0, "noise_steps": 0, "noise_gaussian_steps": 0}

    def state_dict(self):
        return {"convention": CONVENTION, "gsplat_version": "1.5.3",
                "policy": asdict(self.policy), "scene_scale": self.scene_scale,
                "counters": dict(self.counters),
                "binoms": self.strategy_state["binoms"].detach().cpu().clone()}

    def load_state_dict(self, saved, scene, optimizers, expected_step=None):
        if (saved.get("convention") != CONVENTION or saved.get("gsplat_version") != "1.5.3"
                or saved.get("policy") != asdict(self.policy)
                or saved.get("scene_scale") != self.scene_scale
                or scene.scene_scale != self.scene_scale):
            raise ValueError("MCMC resume convention, policy or coordinate scale changed")
        counters = saved.get("counters", {})
        if set(counters) != set(self.counters) or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counters.values()):
            raise ValueError("Invalid MCMC resume counters")
        n = len(scene.splats["means"])
        if (counters["current_gaussians"] != n or n > self.policy.cap_max
                or counters["initial_gaussians"] + counters["added_gaussians"] != n
                or counters["peak_gaussians"] != n
                or counters["noise_steps"] != counters["last_step"]
                or counters["relocation_events"] > counters["refinement_events"]
                or counters["noise_gaussian_steps"] < counters["noise_steps"] * counters["initial_gaussians"]
                or counters["noise_gaussian_steps"] > counters["noise_steps"] * n
                or counters["refinement_events"] != sum(self.policy.refines_at(i)
                    for i in range(1, counters["last_step"] + 1))):
            raise ValueError("MCMC resume counters disagree with field or schedule")
        if expected_step is not None and counters["last_step"] != expected_step:
            raise ValueError("MCMC resume step differs from checkpoint step")
        binoms = saved.get("binoms")
        expected = self.strategy.initialize_state()["binoms"]
        if (not isinstance(binoms, torch.Tensor) or binoms.dtype != expected.dtype
                or binoms.shape != expected.shape or not torch.equal(binoms.cpu(), expected)):
            raise ValueError("MCMC resume binomial table changed")
        params, opts = _aliases(scene, optimizers)
        _zero_prior(scene)
        self.strategy.check_sanity(params, opts)
        self.strategy_state = {"binoms": binoms.detach().clone().to(params["means"].device)}
        self.counters = dict(counters)


def initialize_reference(scene, optimizers, policy):
    """Bind a fresh or resumed scene; never reset parameters, Adam, or RNG."""
    params, opts = _aliases(scene, optimizers)
    _zero_prior(scene)
    state = MCMCReferenceState(scene.scene_scale, len(params["means"]), policy)
    state.strategy.check_sanity(params, opts)
    return state


@torch.no_grad()
def reference_post_step(scene, optimizers, state, step, means_lr):
    """Call after Adam, once for each one-based training step, including step 30000.

    The installed donor/dead Adam-state semantics are intentionally unchanged.
    Replacement Parameters are committed even when relocation preserves N.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step != state.counters["last_step"] + 1:
        raise ValueError("MCMC post-step calls must be consecutive and cannot repeat")
    if scene.scene_scale != state.scene_scale:
        raise ValueError("MCMC coordinate scale changed")
    params, opts = _aliases(scene, optimizers)
    before = len(params["means"])
    if before != state.counters["current_gaussians"] or before > state.policy.cap_max:
        raise ValueError("MCMC field count changed outside its strategy")
    refines = state.policy.refines_at(step)
    relocated = 0
    if refines:
        _zero_prior(scene)
        opacities = params["opacities"].sigmoid()
        if not torch.isfinite(opacities).all():
            raise FloatingPointError("Non-finite MCMC sampling opacities")
        relocated = int((opacities <= state.policy.min_opacity).sum())
        if relocated == before:
            raise ValueError("All Gaussians are dead; MCMC has no donor distribution")
    lr = canonical_noise_lr(means_lr, state.scene_scale)
    try:
        state.strategy.step_post_backward(params, opts, state.strategy_state, step, {}, lr=lr)
    finally:
        # Do not leave the model pointing at old Parameters if an upstream op fails.
        for name in scene.splats:
            scene.splats[name] = params[ALIASES.get(name, name)]
        after = len(scene.splats["means"])
        if after != before:
            scene.semantic_prior_counts = scene.semantic_prior_counts.new_zeros((after, 5))
    expected_after = min(state.policy.cap_max, int(1.05 * before)) if refines else before
    if after != expected_after or after > state.policy.cap_max:
        raise ValueError("Installed MCMC growth violated the declared cap/schedule")
    _aliases(scene, optimizers)  # Validate replacement identities, not just row count.
    added = after - before
    counters = state.counters
    counters.update(last_step=step, current_gaussians=after, peak_gaussians=after)
    counters["refinement_events"] += int(refines)
    counters["relocation_events"] += int(relocated > 0)
    counters["relocated_gaussians"] += relocated
    counters["added_gaussians"] += added
    counters["noise_steps"] += 1
    counters["noise_gaussian_steps"] += after
    return {**{f"mcmc_{k}": v for k, v in counters.items()},
            "mcmc_topology_changed": bool(relocated or added),
            "mcmc_relocated_this_step": relocated, "mcmc_added_this_step": added,
            "mcmc_passed_noise_lr": lr, "mcmc_noise_scaler": lr * state.policy.noise_lr}
