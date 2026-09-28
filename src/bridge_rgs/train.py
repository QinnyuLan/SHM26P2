"""Staged optimization. Validation cameras are always the unrefined official poses."""
from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .io import atomic_save, load_manifest, load_view, seed_everything
from .losses import (
    balanced_local_distillation,
    masked_mean,
    opacity_entropy,
    rgb_loss,
    semantic_loss,
    supervised_teacher_weights,
)
from .model import GaussianScene


class ImageCache:
    def __init__(self, limit=12, ignore_masks=False):
        self.data = OrderedDict()
        self.limit = limit
        self.ignore_masks = ignore_masks

    def get(self, view, scale):
        key = (view["name"], scale)
        if key not in self.data:
            source = dict(view, mask_path=None) if self.ignore_masks else view
            self.data[key] = load_view(source, device="cpu", scale=scale)
        self.data.move_to_end(key)
        while len(self.data) > self.limit:
            self.data.popitem(last=False)
        return {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
                for k, v in self.data[key].items()}


def pseudo_for_view(directory, view, size):
    if not directory:
        return None
    path = Path(directory) / (Path(view["name"]).stem + ".npz")
    if not path.exists():
        return None
    with np.load(path) as f:
        p = torch.tensor(f["probs"].astype(np.float32), device="cuda")
        valid = torch.tensor(f["valid"].astype(np.float32), device="cuda")
        confidence = torch.tensor(f["confidence"].astype(np.float32), device="cuda")
    p = F.interpolate(p[None], size, mode="bilinear", align_corners=False)[0]
    p = p / p.sum(0, keepdim=True).clamp_min(1e-7)
    from .coordinates import pixel_protocol, resize_discrete_tensor
    valid = resize_discrete_tensor(valid, size, pixel_protocol(view))
    confidence = F.interpolate(confidence[None, None], size, mode="bilinear", align_corners=False)[0, 0]
    return p.permute(1, 2, 0), valid, confidence


def training_class_weights(views, config=None, device="cuda"):
    config = {} if config is None else config
    if config.get("class_weights") is not None:
        weights = np.asarray(config["class_weights"], dtype=np.float32)
        if weights.shape != (5,) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("class_weights must contain five finite nonnegative values with positive sum")
        return torch.tensor(weights, device=device)
    counts = np.zeros(5, dtype=np.int64)
    for v in views:
        if v.get("mask_path"):
            mask = cv2.imread(v["mask_path"], 0)
            counts += np.bincount(mask[mask < 5], minlength=5)
    frequencies = counts / max(counts.sum(), 1)
    power = config.get("class_weight_power", .5)
    if not math.isfinite(power) or power < 0:
        raise ValueError("class_weight_power must be finite and nonnegative")
    weights = 1 / np.maximum(frequencies, .002) ** power
    weights = np.minimum(weights / weights.mean(), 3)
    return torch.tensor(weights, device=device).float()


def training_semantic_weights(views, config, device="cuda"):
    """Final and raw-field supervision share weights unless explicitly separated."""
    final = training_class_weights(views, config, device)
    if config.get("raw_class_weight_power") is None:
        return final, final
    raw_config = dict(config, class_weight_power=config["raw_class_weight_power"])
    return final, training_class_weights(views, raw_config, device)


def balanced_probability_distillation(probabilities, targets, reliability, max_class_reweight=3.0):
    """Class-mass-balanced soft KL for an unlabeled view's final prediction.

    The caller must render with a detached shared classifier and detached
    geometry. Officially labeled views never enter this teacher-only path.
    """
    if max_class_reweight < 1:
        raise ValueError("The class reweighting cap must be at least one")
    targets, reliability = targets.detach(), reliability.detach()
    classes = targets.shape[-1]
    categories = targets.argmax(-1)
    class_mass = reliability.new_zeros(classes).scatter_add_(0, categories.flatten(), reliability.flatten())
    mean_mass = class_mass.sum() / (class_mass > 0).sum().clamp_min(1)
    class_weights = (mean_mass / class_mass.clamp_min(1e-8)).clamp(max=max_class_reweight)
    kl = (targets * (targets.clamp_min(1e-7).log() - probabilities.clamp_min(1e-7).log())).sum(-1)
    return masked_mean(kl, reliability * class_weights[categories])


def optional_raw_pseudo_loss(probabilities, targets, reliability, weight):
    """A disabled loss must not create zero gradients that advance Adam moments."""
    if weight <= 0:
        return None
    kl = (targets * (targets.clamp_min(1e-6).log() - probabilities.log())).sum(-1)
    return weight * masked_mean(kl, reliability)


def multiview_evidence_enabled(config):
    """Fusion is needed for positive semantic KD or a separate geometry gate."""
    return bool(config.get("pseudo_dir") and config.get("multiview_fusion", True)
                and (config.get("multiview_weight", .05) > 0
                     or config.get("semantic_geometry_weight", 0) > 0))


def optional_multiview_loss(features, classifier, indices, targets, reliability, weight,
                           local_teacher_kd=True):
    """Skip the feature gather and classifier entirely when KD is disabled."""
    if weight <= 0:
        return None
    selected = features[indices]
    if local_teacher_kd:
        value = balanced_local_distillation(selected, classifier, targets, reliability)
    else:
        logits = classifier(selected)
        kl = (targets * (targets.clamp_min(1e-7).log() - logits.log_softmax(-1))).sum(-1)
        value = masked_mean(kl, reliability)
    return weight * value


def validate_parameter_scope(config):
    scope = config.get("parameter_scope", "all")
    if scope not in {"all", "opacity_only", "appearance_only", "refiner_only"}:
        raise ValueError("parameter_scope must be all, opacity_only, appearance_only or refiner_only")
    if scope == "opacity_only":
        if config.get("freeze_geometry") or config.get("densification", "none") != "none":
            raise ValueError("opacity_only requires mutable opacity, fixed topology and freeze_geometry:false")
        if config.get("optimize_cameras") or config.get("semantic_geometry_weight", 0):
            raise ValueError("opacity_only forbids camera updates and semantic geometry losses")
        if config.get("pseudo_dir") or config.get("multiview_fusion", False):
            raise ValueError("opacity_only polishing must not include semantic teacher supervision")
        if config.get("semantic_start", 3000) <= config.get("steps", 30000):
            raise ValueError("opacity_only requires semantic_start after the polishing budget")
    if scope == "appearance_only":
        if not config.get("freeze_geometry") or config.get("freeze_rgb", False):
            raise ValueError("appearance_only requires freeze_geometry:true and freeze_rgb:false")
        if config.get("densification", "none") != "none" or config.get("optimize_cameras"):
            raise ValueError("appearance_only requires fixed topology and cameras")
        if (config.get("semantic_geometry_weight", 0) or config.get("pseudo_dir")
                or config.get("multiview_fusion", False)):
            raise ValueError("appearance_only forbids semantic geometry and teacher supervision")
        if any(config.get(key, 3000) <= config.get("steps", 30000)
               for key in ("semantic_start", "refine_start")):
            raise ValueError("appearance_only requires semantic/refine_start after its budget")
        if config.get("sparse_depth_weight", 0) or config.get("opacity_entropy_weight", 0):
            raise ValueError("appearance_only cannot optimize frozen depth/opacity regularizers")
        if config.get("warmstart_reset_refiner", False):
            raise ValueError("appearance_only must preserve the existing refiner")
    if scope == "refiner_only":
        if not config.get("freeze_geometry") or not config.get("freeze_rgb"):
            raise ValueError("refiner_only requires frozen geometry and RGB")
        if config.get("densification", "none") != "none" or config.get("optimize_cameras"):
            raise ValueError("refiner_only requires fixed topology and cameras")
        if not config.get("train_labeled_only"):
            raise ValueError("refiner_only requires labeled-only TRAIN views")
        if config.get("pseudo_dir") or config.get("multiview_fusion"):
            raise ValueError("refiner_only currently requires pure supervised training")
        if any(config.get(key, 0) for key in ("semantic_geometry_weight", "sparse_depth_weight",
                                             "opacity_entropy_weight", "sparse_front_weight")):
            raise ValueError("refiner_only forbids geometry/opacity auxiliary losses")
        if any(config.get(key, 3000) > config.get("steps", 30000)
               for key in ("semantic_start", "refine_start")):
            raise ValueError("refiner_only requires active semantic refinement")
    return scope


def apply_parameter_scope(scene, config):
    """Apply the complete parameter allow-list for an isolated polishing stage."""
    scope = validate_parameter_scope(config)
    if scope in {"opacity_only", "appearance_only", "refiner_only"}:
        for parameter in scene.parameters():
            parameter.requires_grad_(False)
        if scope == "opacity_only":
            scene.splats["opacity_logits"].requires_grad_(True)
        elif scope == "appearance_only":
            scene.splats["sh0"].requires_grad_(True)
            scene.splats["sh_rest"].requires_grad_(True)
            scene.background_logits.requires_grad_(True)
        else:
            scene.refiner.requires_grad_(True)


def optional_sparse_front_loss(support, scene, data, pose, view_id, weight):
    """An explicit off switch performs no support lookup or additional render."""
    if weight <= 0:
        return None, {}
    if support is None:
        raise ValueError("Positive sparse_front_weight requires initialized support")
    return support.loss(scene, data["K"], pose, data["width"], data["height"], data["valid"], view_id)


def checkpoint(scene, optimizers, config, step, cameras, stats, density_state=None):
    from .coordinates import protocol_metadata
    result = {"format_version": 1, "model": scene.state_dict(), "config": config,
            "step": step, "scene_scale": scene.scene_scale,
            "feature_dim": scene.feature_dim, "sh_degree": scene.sh_degree,
            "refiner_config": scene.refiner_config,
            "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
            "training_cameras": cameras.detach().cpu(), "stats": stats,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
            "numpy_rng": np.random.get_state(), "density_state": density_state,
            "pixel_protocol": protocol_metadata(getattr(scene, "pixel_protocol", None))}
    if getattr(scene, "manifest_sha256", None) is not None:
        result["manifest_sha256"] = scene.manifest_sha256
    if scene.mip_filter_config is not None:
        from copy import deepcopy
        result["mip_filter_config"] = dict(scene.mip_filter_config)
        result["mip_filter_state"] = deepcopy(scene.mip_filter_state)
    if getattr(scene, "mcmc_reference_config", None) is not None:
        from copy import deepcopy
        result["mcmc_reference_config"] = deepcopy(scene.mcmc_reference_config)
        runtime = getattr(scene, "mcmc_reference_runtime", None)
        result["mcmc_reference_state"] = (runtime.state_dict() if runtime is not None
                                           else deepcopy(scene.mcmc_reference_state))
    return result


def restore_mcmc_metadata(scene, state):
    """Preserve strategy provenance without running MCMC during inference.

    A geometry-frozen semantic warmstart inherits this metadata unchanged.
    Active strict resume additionally validates it against live optimizers.
    """
    from copy import deepcopy

    from .mcmc_reference import CONVENTION, MODE, MCMCReferenceConfig
    policy, strategy = state.get("mcmc_reference_config"), state.get("mcmc_reference_state")
    if (policy is None) != (strategy is None):
        raise ValueError("MCMC checkpoint requires both policy and strategy state")
    if state.get("config", {}).get("densification") == MODE and policy is None:
        raise ValueError("MCMC checkpoint is missing strategy provenance")
    if policy is not None:
        from dataclasses import asdict
        if asdict(MCMCReferenceConfig(**policy)) != policy:
            raise ValueError("MCMC checkpoint policy is incomplete")
        if (strategy.get("convention") != CONVENTION or strategy.get("gsplat_version") != "1.5.3"
                or strategy.get("policy") != policy or strategy.get("scene_scale") != scene.scene_scale):
            raise ValueError("MCMC checkpoint strategy metadata disagree")
        counters = strategy.get("counters", {})
        if (not counters or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                                for value in counters.values())
                or counters.get("current_gaussians") != len(scene.splats["means"])
                or counters.get("noise_steps") != counters.get("last_step")):
            raise ValueError("MCMC checkpoint counters disagree with the field")
        binoms = strategy.get("binoms")
        if (not isinstance(binoms, torch.Tensor) or binoms.dtype != torch.float32
                or binoms.shape != (51, 51) or not bool(torch.isfinite(binoms).all())):
            raise ValueError("MCMC checkpoint binomial table is invalid")
        if state.get("config", {}).get("densification") == MODE:
            if state["config"].get("mcmc_reference") != policy:
                raise ValueError("MCMC checkpoint configuration and policy disagree")
            if counters.get("last_step") != state.get("step"):
                raise ValueError("MCMC checkpoint strategy and training steps disagree")
        scene.mcmc_reference_config = deepcopy(policy)
        scene.mcmc_reference_state = deepcopy(strategy)


def validate_mcmc_resume(saved_config, config):
    """Exact MCMC continuation cannot silently reset its view/noise schedule."""
    from .mcmc_reference import MODE
    old_mode = saved_config.get("densification", "attribution")
    new_mode = config.get("densification", "attribution")
    if MODE not in {old_mode, new_mode}:
        return
    if old_mode != new_mode:
        raise ValueError("Switching the MCMC reference strategy requires a fresh run")
    # Output/log/evaluation/checkpoint cadence can change, optimization cannot.
    defaults = {"steps": 30000, "seed": 42, "view_sampler_seed": None,
                "view_sampling": "random", "independent_view_rng": False,
                "train_scale": 1., "progressive_resolution": True,
                "sh_interval": 1000, "fixed_sh_degree": None,
                "opacity_lr": .05, "background_points": 2000,
                "initial_opacity": .5, "initial_scale_multiplier": .1,
                "init_points": None,
                "max_gaussians": 500000, "mcmc_reference": None,
                "train_labeled_only": False}
    for key, default in defaults.items():
        old, new = saved_config.get(key, default), config.get(key, default)
        if key == "view_sampler_seed":
            old = saved_config.get("seed", 42) if old is None else old
            new = config.get("seed", 42) if new is None else new
        if old != new:
            raise ValueError(f"Strict MCMC resume cannot change {key}")


def check_mcmc_finite(scene, step, gradients=False):
    """Fail visibly instead of clipping the MCMC trajectory to local bounds."""
    for name, parameter in scene.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is not None and not bool(torch.isfinite(value).all()):
            kind = "gradient" if gradients else "parameter"
            raise FloatingPointError(f"Non-finite MCMC {kind} {name} at step {step}")


def mcmc_log_scalars(stats):
    """Materialize detached regularizer diagnostics only at a log boundary."""
    for name in ("mcmc_opacity_regularization", "mcmc_scale_regularization"):
        if name in stats:
            stats[name] = float(stats[name])


def bind_training_pixel_protocol(scene, manifest, manifest_path, config, initial=None):
    """Bind an immutable input convention; a warmstart never silently migrates it."""
    import hashlib

    from .coordinates import pixel_protocol, require_matching_protocol
    protocol = pixel_protocol(manifest)
    if "pixel_protocol" in config:
        require_matching_protocol(config, manifest, "training configuration")
    with Path(manifest_path).open("rb") as handle:
        manifest_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if initial is not None:
        require_matching_protocol(initial, manifest, "resume or warmstart")
        if initial.get("manifest_sha256") and initial["manifest_sha256"] != manifest_sha:
            raise ValueError("Checkpoint manifest SHA differs; use a new isolated training run")
    scene.pixel_protocol = protocol
    scene.manifest_sha256 = manifest_sha
    config["pixel_protocol"] = protocol


def _density_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: _density_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_density_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_density_to_device(item, device) for item in value]
    return value


def initialize_density_extras(num_gaussians, num_views, device):
    return {"radii": torch.zeros(num_gaussians, device=device),
            "residual_scores": torch.zeros(num_gaussians, device=device),
            "residual_observations": torch.zeros(num_gaussians, dtype=torch.long, device=device),
            "residual_counts": torch.zeros(num_gaussians, dtype=torch.long, device=device),
            "residual_observed_views": {}, "camera_quality": torch.ones(num_views, device=device),
            "sampler_order": [], "sampler_cursor": 0, "sampler_rng_state": None, "pause_until": 0,
            "gradient_convention": "gsplat_normalized_v1"}


def restore_density_extras(stored, num_gaussians, num_views, device):
    result = initialize_density_extras(num_gaussians, num_views, device)
    if stored is None:
        return result
    result.update(_density_to_device(stored, device))
    for key in ("radii", "residual_scores", "residual_observations", "residual_counts"):
        if result[key].shape != (num_gaussians,):
            raise ValueError(f"Saved hybrid density state {key} has the wrong Gaussian count")
    if result["camera_quality"].shape != (num_views,):
        raise ValueError("Saved camera diagnostic quality has the wrong view count")
    if any(value.shape != (num_gaussians,) for value in result["residual_observed_views"].values()):
        raise ValueError("Saved residual support masks have the wrong Gaussian count")
    order = result["sampler_order"]
    if order and (len(set(order)) != len(order) or any(index < 0 or index >= num_views for index in order)
                  or not 0 <= result["sampler_cursor"] <= len(order)):
        raise ValueError("Saved shuffled-view sampler is invalid")
    return result


def initialize_view_sampler(config, extras):
    """Opt-in RNG isolation leaves historical global-NumPy resumes unchanged."""
    if not config.get("independent_view_rng", False):
        return None
    generator = np.random.default_rng(config.get("view_sampler_seed", config.get("seed", 42)))
    if extras.get("sampler_rng_state") is not None:
        generator.bit_generator.state = extras["sampler_rng_state"]
    extras["sampler_rng_state"] = generator.bit_generator.state
    return generator


def sample_training_view(population, mode, extras, generator=None):
    source = np.random if generator is None else generator
    if mode == "shuffle":
        if extras["sampler_cursor"] >= len(extras["sampler_order"]):
            extras["sampler_order"] = source.permutation(population).tolist()
            extras["sampler_cursor"] = 0
        index = extras["sampler_order"][extras["sampler_cursor"]]
        extras["sampler_cursor"] += 1
    elif mode == "random":
        offset = np.random.randint(len(population)) if generator is None else generator.integers(len(population))
        index = population[int(offset)]
    else:
        raise ValueError("view_sampling must be random or shuffle")
    if generator is not None:
        extras["sampler_rng_state"] = generator.bit_generator.state
    return index


def rendering_sh_degree(step, max_degree, config):
    if config.get("parameter_scope") == "appearance_only":
        if config.get("fixed_sh_degree") not in {None, max_degree}:
            raise ValueError("appearance_only renders the complete existing SH expansion")
        return max_degree
    if config.get("freeze_rgb"):
        return max_degree
    if config.get("fixed_sh_degree") is not None:
        degree = int(config["fixed_sh_degree"])
        if not 0 <= degree <= max_degree:
            raise ValueError("fixed_sh_degree is outside the model's SH range")
        return degree
    return min(step // config.get("sh_interval", 1000), max_degree)


def scheduled_gaussian_budget(step, config):
    maximum = config.get("max_gaussians", 600000)
    if config.get("density_budget_schedule", "constant") == "constant":
        return maximum
    if config["density_budget_schedule"] != "linear":
        raise ValueError("density_budget_schedule must be constant or linear")
    initial = min(config["initial_gaussians"], maximum)
    start, stop = config.get("densify_start", 4000), config.get("densify_stop", 22000)
    progress = min(1., max(0., (step - start) / max(1, stop - start)))
    return round(initial + progress * (maximum - initial))


def density_checkpoint_state(scores, observations, counts, observed_views, fused_target, extras=None,
                             mixed_gradient=None):
    """Preserve the capacity-allocation window and existing semantic fusion on resume."""
    result = {"scores": scores.detach().cpu(), "observations": observations.detach().cpu(),
            "counts": counts.detach().cpu(),
            "observed_views": {key: value.detach().cpu() for key, value in observed_views.items()},
            "fused_target": None if fused_target is None else tuple(value.detach().cpu() for value in fused_target),
            "extras": _density_to_device(extras, "cpu")}
    if mixed_gradient is not None:
        result["mixed_gradient"] = mixed_gradient.state_dict()
    return result


def load_scene(checkpoint_path, device="cuda"):
    from .checkpoints import load_checkpoint
    state = load_checkpoint(checkpoint_path, map_location="cpu")
    values = state["model"]
    points = values["splats.means"].numpy()
    colors = values["splats.sh0"][:, 0].numpy() * .28209479177387814 + .5
    from .mip_filter import normalize_config, restore_filter_state
    mip_config = normalize_config(state.get("mip_filter_config"))
    if normalize_config(state.get("config", {}).get("mip_filter")) != mip_config:
        raise ValueError("Checkpoint mip filter model/config metadata disagree")
    scene = GaussianScene(points, colors, state["feature_dim"], state["sh_degree"],
                          refiner_config=state.get("refiner_config"), mip_filter_config=mip_config).to(device)
    missing, unexpected = scene.load_state_dict(values, strict=False)
    if unexpected or set(missing) - {"semantic_prior_counts"}:
        raise ValueError(f"Checkpoint model schema mismatch: missing={missing}, unexpected={unexpected}")
    scene.scene_scale = state["scene_scale"]
    from .coordinates import pixel_protocol
    scene.pixel_protocol = pixel_protocol(state)
    scene.manifest_sha256 = state.get("manifest_sha256")
    restore_filter_state(scene, state)
    restore_mcmc_metadata(scene, state)
    return scene, state


def train(config, resume=None):
    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian training requires CUDA; use CPU unit tests to check geometry")
    config = dict(config)
    config.setdefault("local_teacher_kd", True)
    config.setdefault("supervised_teacher_priority", True)
    config.setdefault("camera_attribution", True)
    config.setdefault("camera_quality_weighting", True)
    config.setdefault("pseudo_refiner_weight", 0.)
    config.setdefault("independent_view_rng", False)
    config.setdefault("refiner_field_grad", True)
    config.setdefault("parameter_scope", "all")
    scope = validate_parameter_scope(config)
    save_every = config.get("save_every", 1000)
    if isinstance(save_every, bool) or not isinstance(save_every, int) or save_every < 0:
        raise ValueError("save_every must be a nonnegative integer; 0 saves only the final model")
    if scope == "opacity_only":
        config["freeze_rgb"] = True  # Retain the complete trained SH expansion from step 1.
    if scope in {"opacity_only", "appearance_only", "refiner_only"} and not (resume or config.get("warmstart")):
        raise ValueError(f"{scope} requires an existing trained scene")
    if scope == "refiner_only":
        config["refiner_field_grad"] = False
    depth_weight = float(config.get("sparse_depth_weight", 0.))
    if not math.isfinite(depth_weight) or depth_weight < 0:
        raise ValueError("sparse_depth_weight must be finite and nonnegative")
    front_weight = float(config.get("sparse_front_weight", 0.))
    if not math.isfinite(front_weight) or front_weight < 0:
        raise ValueError("sparse_front_weight must be finite and nonnegative")
    if front_weight > 0 and scope != "opacity_only":
        raise ValueError("Sparse front mass is restricted to the opacity_only parameter scope")
    entropy_weight = float(config.get("opacity_entropy_weight", 0.))
    if not math.isfinite(entropy_weight) or entropy_weight < 0:
        raise ValueError("opacity_entropy_weight must be finite and nonnegative")
    if entropy_weight > 0 and config.get("freeze_geometry"):
        raise ValueError("Opacity entropy requires unfrozen opacity parameters")
    if config["pseudo_refiner_weight"] > 0 and not config["local_teacher_kd"]:
        raise ValueError("Pseudo refiner distillation requires local_teacher_kd to detach the classifier")
    if resume and config.get("warmstart"):
        raise ValueError("Use either strict resume or a new-stage warmstart, not both")
    if config.get("freeze_geometry") and config.get("densification", "none") != "none":
        raise ValueError("A stage with frozen geometry must set densification: none")
    if config.get("freeze_geometry"):
        if config.get("optimize_cameras") or config.get("semantic_geometry_weight", 0):
            raise ValueError("Frozen geometry forbids camera updates and semantic geometry losses")
        # Geometry-frozen semantic stages retain their historical frozen RGB.
        # The explicit appearance scope trains only color with geometry fixed.
        if scope != "appearance_only":
            config["freeze_rgb"] = True
    from .refiner_crops import (
        choose_refiner_crop,
        crop_slices,
        refine_render_crop,
        validate_refiner_crop_config,
    )
    validate_refiner_crop_config(config)
    from .refiner_flip import (
        choose_refiner_horizontal_flip,
        refine_horizontal_flip,
        validate_refiner_flip_config,
        validate_refiner_flip_resume,
    )
    flip_probability = validate_refiner_flip_config(config)
    from .semantic_schedule import (
        SemanticLRSchedule,
        normalize_semantic_schedule,
        validate_semantic_schedule_resume,
    )
    normalize_semantic_schedule(config)
    from .mcmc_reference import MODE as MCMC_MODE
    from .mcmc_reference import (
        initialize_reference as initialize_mcmc,
    )
    from .mcmc_reference import (
        reference_post_step as mcmc_post_step,
    )
    from .mcmc_reference import (
        reference_regularization as mcmc_regularization,
    )
    from .mcmc_reference import (
        validate_reference_training as validate_mcmc_training,
    )
    from .mixed_gradient_reference import MODE as MIXED_MODE
    from .mixed_gradient_reference import MixedGradientState, validate_reference_training
    density_mode = config.get("densification", "attribution")
    if density_mode not in {"none", "absgrad", "attribution", "hybrid", MIXED_MODE, MCMC_MODE}:
        raise ValueError("Unknown densification strategy")
    mixed_policy = validate_reference_training(config) if density_mode == MIXED_MODE else None
    mcmc_policy = validate_mcmc_training(config) if density_mode == MCMC_MODE else None
    if mcmc_policy is not None:
        from dataclasses import asdict
        config["mcmc_reference"] = asdict(mcmc_policy)
    rgb_reference = mixed_policy is not None or mcmc_policy is not None
    from .mip_filter import (
        bind_filter_source,
        filter_stats,
        refresh_due,
        refresh_filter,
        resolve_config,
        validate_resume_step,
    )
    seed_everything(config.get("seed", 42))
    torch.set_num_threads(8)
    manifest = load_manifest(config["manifest"])
    if config.get("pseudo_dir"):
        from .teacher import verify_pseudo_provenance
        if not Path(config["pseudo_dir"]).is_dir():
            raise FileNotFoundError(f"Pseudo probability directory does not exist: {config['pseudo_dir']}")
        pseudo_provenance = verify_pseudo_provenance(config["manifest"], config["pseudo_dir"])
        exported = {str(view["name"]) for view in pseudo_provenance["views"]}
        expected = {str(view["name"]) for view in manifest["views"] if view["split"] == "train"}
        if exported != expected:
            raise ValueError("Scene training requires pseudo probabilities for every training view")
    views = [v for v in manifest["views"] if v["split"] == "train"]
    if not views:
        raise ValueError("No training views in manifest")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    init_path = config.get("init_points", str(Path(config["manifest"]).with_name("init_points.npz")))
    points = np.load(init_path)
    from .coordinates import require_matching_protocol
    initial_pixel_protocol = str(points["pixel_protocol"].item()) if "pixel_protocol" in points else None
    require_matching_protocol(manifest, initial_pixel_protocol, "initial point appearance and labels")
    saved = None
    warmstarted = None
    if resume:
        scene, saved = load_scene(resume)
        resolve_config(config, saved)
        old_density_mode = saved["config"].get("densification", "attribution")
        if MIXED_MODE in {old_density_mode, density_mode} and old_density_mode != density_mode:
            raise ValueError("Switching the RGB reference strategy requires a fresh run")
        validate_mcmc_resume(saved["config"], config)
        if saved.get("appearance_delta", {}).get("ordinary_resume_allowed") is False:
            raise ValueError("Appearance delta has no ordinary optimizer/RNG state; use an explicit warmstart")
        validate_semantic_schedule_resume(saved["config"], config)
        validate_refiner_flip_resume(saved["config"], config)
        if saved.get("checkpoint_kind") in {
                "averaged_semantics_inference_or_warmstart",
                "fullbatch_appearance_inference_or_warmstart",
                "semantic_transfer_inference_or_warmstart"}:
            raise ValueError("Derived model has no compatible optimizer state; use warmstart")
        if saved["config"].get("parameter_scope", "all") != scope:
            raise ValueError("Changing parameter_scope requires a new-stage warmstart")
        if saved["config"]["manifest"] != config["manifest"]:
            raise ValueError("Resume manifest differs: changing split requires a fresh run")
        from .refinement import normalize_refiner_config
        requested_refiner = normalize_refiner_config(config.get("refiner", scene.refiner_config))
        if requested_refiner != scene.refiner_config:
            raise ValueError("Strict resume cannot change refiner schema; use warmstart for a new stage")
        start = saved["step"] + 1
    elif config.get("warmstart"):
        scene, warmstarted = load_scene(config["warmstart"])
        resolve_config(config, warmstarted)
        if warmstarted["config"]["manifest"] != config["manifest"]:
            raise ValueError("Warmstart manifest differs: changing split risks held-out pixel leakage")
        from .refinement import normalize_refiner_config
        requested_refiner = normalize_refiner_config(config.get("refiner", scene.refiner_config))
        if scope == "appearance_only" and requested_refiner != scene.refiner_config:
            raise ValueError("appearance_only cannot replace the warmstarted refiner")
        if config.get("warmstart_add_depth_moments") and config.get("warmstart_reset_refiner"):
            raise ValueError("Cannot preserve and reset the warmstarted refiner simultaneously")
        if requested_refiner != scene.refiner_config or config.get("warmstart_reset_refiner", False):
            old_values = scene.state_dict()
            colors = old_values["splats.sh0"][:, 0].cpu().numpy() * .28209479177387814 + .5
            replacement = GaussianScene(old_values["splats.means"].cpu().numpy(), colors,
                                        scene.feature_dim, scene.sh_degree,
                                        refiner_config=requested_refiner,
                                        mip_filter_config=scene.mip_filter_config).cuda()
            preserved = {key: value for key, value in old_values.items() if not key.startswith("refiner.")}
            missing, unexpected = replacement.load_state_dict(preserved, strict=False)
            if unexpected or any(not key.startswith("refiner.") for key in missing):
                raise ValueError(f"Warmstart schema mismatch: missing={missing}, unexpected={unexpected}")
            if config.get("warmstart_add_depth_moments"):
                from .refinement import add_depth_moment_paths
                add_depth_moment_paths(scene.refiner, replacement.refiner)
            replacement.scene_scale = scene.scene_scale
            if scene.mip_filter_config is not None:
                from copy import deepcopy
                replacement.mip_filter_state = deepcopy(scene.mip_filter_state)
            restore_mcmc_metadata(replacement, warmstarted)
            scene = replacement
        elif config.get("warmstart_add_depth_moments"):
            raise ValueError("Moment warmstart requires an off-to-enabled architecture change")
        start = 1
    else:
        scene = GaussianScene(points["points"], points["colors"],
                              background_points=config.get("background_points", 2000),
                              refiner_config=config.get("refiner"),
                              mip_filter_config=resolve_config(config)).cuda()
        if mcmc_policy is not None:
            # Preserve the prepared locations/colors and deterministic shell.
            # Only the MCMC recipe's initial opacity/scale differ from baseline.
            with torch.no_grad():
                opacity = config.get("initial_opacity", .5)
                scene.splats["opacity_logits"].fill_(math.log(opacity / (1 - opacity)))
                scene.splats["log_scales"].add_(math.log(config.get("initial_scale_multiplier", .1)))
        if "semantic_counts" in points and not rgb_reference:
            # Labels observed along train-only SfM tracks give the semantic
            # field an interpretable initialization before teacher fusion.
            with torch.no_grad():
                votes = torch.tensor(points["semantic_counts"], device="cuda").float()
                classes = scene.semantic_decoder.out_features
                if votes.shape != (len(points["points"]), classes):
                    raise ValueError("Initialization semantic_counts shape disagrees with scene classes")
                if scene.feature_dim < classes:
                    raise ValueError("Semantic feature width must cover all class logits")
                scene.semantic_prior_counts[:len(votes)] = votes
                if config.get("semantic_initialization", True):
                    probabilities = (votes + .25) / (votes.sum(-1, keepdim=True) + .25 * classes)
                    scene.splats["sem_features"][:len(votes), :classes] = probabilities.log()
                    scene.semantic_decoder.weight.zero_()
                    scene.semantic_decoder.weight[:, :classes] = torch.eye(classes, device="cuda")
                    scene.semantic_decoder.bias.zero_()
        start = 1
    if scene.mip_filter_config is not None:
        if (config.get("optimize_cameras") or config.get("sparse_front_weight", 0)
                or config.get("sparse_depth_weight", 0) or config.get("opacity_entropy_weight", 0)
                or config.get("semantic_geometry_weight", 0)):
            raise ValueError("Mip reference forbids camera updates and unsynchronized auxiliary geometry paths")
        inherited_frozen = (warmstarted is not None or saved is not None) and config.get("freeze_geometry") and density_mode == "none"
        if mixed_policy is None and not inherited_frozen:
            raise ValueError("Mip reference supports mixed RGB or an inherited geometry-frozen stage only")
    if (getattr(scene, "mcmc_reference_config", None) is not None and mcmc_policy is None
            and not (config.get("freeze_geometry") and config.get("freeze_rgb") and density_mode == "none")):
        raise ValueError("Inherited MCMC field requires a geometry/RGB-frozen stage with densification:none")
    bind_training_pixel_protocol(scene, manifest, config["manifest"], config,
                                 saved if saved is not None else warmstarted)
    config["refiner"] = scene.refiner_config
    config.setdefault("initial_gaussians", saved["config"].get("initial_gaussians", len(points["points"]) + config.get("background_points", 2000))
                      if saved else len(scene.splats["means"]))
    (output / "config.json").write_text(json.dumps(config, indent=2))
    if config.get("freeze_geometry"):
        for name in ("means", "quats", "log_scales", "opacity_logits"):
            scene.splats[name].requires_grad_(False)
    if config.get("freeze_rgb"):
        for name in ("sh0", "sh_rest"):
            scene.splats[name].requires_grad_(False)
        scene.background_logits.requires_grad_(False)
    apply_parameter_scope(scene, config)
    optimizers = scene.optimizers()
    if mixed_policy is not None:
        optimizers["opacity_logits"].param_groups[0]["lr"] = .025
    if mcmc_policy is not None:
        optimizers["opacity_logits"].param_groups[0]["lr"] = .05
    semantic_schedule = SemanticLRSchedule(optimizers, config)
    cameras = torch.tensor(np.array([v["w2c"] for v in views]), device="cuda").float()
    camera_reference = cameras.clone()
    if warmstarted is not None and config.get("warmstart_training_cameras", True):
        cameras.copy_(warmstarted["training_cameras"].cuda())
    if saved:
        for k, opt in optimizers.items():
            opt.load_state_dict(saved["optimizers"][k])
        cameras.copy_(saved["training_cameras"].cuda())
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
    semantic_schedule.initialize(resume_step=saved["step"] if saved else None)
    mip_source = bind_filter_source(scene, views, cameras, scene.manifest_sha256)
    if saved:
        validate_resume_step(scene, saved["step"])
    class_weights, raw_class_weights = ((None, None) if rgb_reference
                                      else training_semantic_weights(views, config))
    cache = ImageCache(ignore_masks=rgb_reference)
    depth_support = None
    front_support = None
    if depth_weight > 0 or front_weight > 0:
        from .ray_support import SparseDepthSupport
        support_manifest = dict(manifest, init_points_path=init_path)
        depth_options = dict(config.get("sparse_depth", {}))
        colmap_images_path = depth_options.pop("colmap_images_path", None)
        depth_support = SparseDepthSupport.from_manifest(
            support_manifest, config=depth_options, colmap_images_path=colmap_images_path)
        (output / "sparse_depth_provenance.json").write_text(
            json.dumps(depth_support.provenance, indent=2))
        if front_weight > 0:
            from .ray_termination import SparseFrontSupport
            front_support = SparseFrontSupport(depth_support)
            (output / "sparse_front_provenance.json").write_text(
                json.dumps(front_support.provenance, indent=2))
    steps = config.get("steps", 30000)
    semantic_start = config.get("semantic_start", 3000)
    refine_start = config.get("refine_start", 6000)
    camera_stop = config.get("camera_stop", 12000)
    densify_start = config.get("densify_start", 4000)
    densify_stop = config.get("densify_stop", 22000)
    densify_every = config.get("densify_every", 200)
    diagnostic_every = config.get("diagnostic_every", 40)
    n = len(scene.splats["means"])
    mixed_state = MixedGradientState(n, "cuda", mixed_policy) if mixed_policy is not None else None
    if mixed_state is not None and saved:
        stored_mixed = (saved.get("density_state") or {}).get("mixed_gradient")
        if stored_mixed is None:
            raise ValueError("RGB reference resume requires its saved dual-gradient window")
        mixed_state.load_state_dict(stored_mixed)
    mcmc_state = initialize_mcmc(scene, optimizers, mcmc_policy) if mcmc_policy is not None else None
    if mcmc_state is not None:
        if saved:
            mcmc_state.load_state_dict(saved["mcmc_reference_state"], scene, optimizers,
                                       expected_step=saved["step"])
        scene.mcmc_reference_config = dict(config["mcmc_reference"])
        scene.mcmc_reference_state = mcmc_state.state_dict()
        scene.mcmc_reference_runtime = mcmc_state
    scores = torch.zeros(n, device="cuda")
    observations = torch.zeros(n, device="cuda", dtype=torch.long)
    counts = torch.zeros(n, device="cuda", dtype=torch.long)
    observed_views = {}
    fused_target = None
    extras = initialize_density_extras(n, len(views), "cuda")
    if saved and saved.get("density_state") is not None:
        window = saved["density_state"]
        if any(window[key].shape != (n,) for key in ("scores", "observations", "counts")):
            raise ValueError("Saved density window does not match Gaussian count")
        scores, observations, counts = (window[key].cuda() for key in ("scores", "observations", "counts"))
        observed_views = {int(key): value.cuda() for key, value in window["observed_views"].items()}
        if window["fused_target"] is not None:
            fused_target = tuple(value.cuda() for value in window["fused_target"])
        extras = restore_density_extras(window.get("extras"), n, len(views), "cuda")
    log = (output / "train.jsonl").open("a")
    tic = time.perf_counter()
    stats = dict(saved.get("stats", {})) if saved else {}
    if mip_source is not None:
        stats.update(filter_stats(scene))
    if saved:
        stats["resume_density_window_restored"] = saved.get("density_state") is not None
        supervised_modules = ("pseudo_dir", "multiview_fusion", "local_teacher_kd",
                              "supervised_teacher_priority", "semantic_geometry_weight", "densification",
                              "camera_attribution", "camera_quality_weighting", "pseudo_refiner_weight",
                              "refiner_field_grad", "raw_class_weight_power", "independent_view_rng")
        stats["resume_module_changes"] = [key for key in supervised_modules
                                          if saved["config"].get(key) != config.get(key)]
        stored_extras = (saved.get("density_state") or {}).get("extras") or {}
        gradient_units_changed = (config.get("densification") in {"absgrad", "hybrid"}
                                  and stored_extras.get("gradient_convention") != "gsplat_normalized_v1")
        if saved["config"].get("densification") != config.get("densification") or gradient_units_changed:
            scores.zero_()
            observations.zero_()
            counts.zero_()
            observed_views.clear()
            sampler_state = {key: extras[key] for key in ("sampler_order", "sampler_cursor", "sampler_rng_state")}
            extras = initialize_density_extras(n, len(views), "cuda")
            extras.update(sampler_state)
            stats["resume_density_window_restored"] = False
            stats["resume_density_reset_reason"] = "densification strategy or gradient units changed"
    if (not multiview_evidence_enabled(config)
            or (saved and saved["config"].get("pseudo_dir") != config.get("pseudo_dir"))):
        stats["resume_fused_target_cleared"] = fused_target is not None
        fused_target = None
    view_population = [index for index, view in enumerate(views)
                       if not config.get("train_labeled_only", False) or view.get("mask_path")]
    if not view_population:
        raise ValueError("Requested training view population is empty")
    if extras["sampler_order"] and sorted(extras["sampler_order"]) != view_population:
        extras["sampler_order"], extras["sampler_cursor"] = [], 0
        stats["resume_view_sampler_reset"] = True
    if saved and (saved["config"].get("independent_view_rng", False) != config["independent_view_rng"]
                  or saved["config"].get("view_sampler_seed", saved["config"].get("seed", 42))
                  != config.get("view_sampler_seed", config.get("seed", 42))):
        extras["sampler_order"], extras["sampler_cursor"], extras["sampler_rng_state"] = [], 0, None
        stats["resume_view_sampler_reset"] = True
    view_generator = initialize_view_sampler(config, extras)
    try:
        for step in range(start, steps + 1):
            mip_topology_changed = False
            vi = sample_training_view(view_population, config.get("view_sampling", "random"), extras, view_generator)
            view = views[vi]
            progress = step / steps
            scale = config.get("train_scale", 1.0)
            if config.get("progressive_resolution", True):
                scale *= .5 if progress < .15 else (.75 if progress < .6 else 1)
            data = cache.get(view, scale)
            if rgb_reference:
                # Counts use the field actually rendered, before post-step growth.
                stats["reference_gaussian_steps"] = stats.get("reference_gaussian_steps", 0) + n
                stats["reference_render_pixels"] = stats.get("reference_render_pixels", 0) + data["width"] * data["height"]
                stats["reference_peak_gaussians"] = max(stats.get("reference_peak_gaussians", 0), n)
            # A camera update later in this step must not mutate the reference
            # used to project the verified diagnostic residual a second time.
            pose = cameras[vi].detach().clone()
            degree = rendering_sh_degree(step, scene.sh_degree, config)
            semantic_on = step >= semantic_start
            crop = (choose_refiner_crop(
                data["height"], data["width"], config.get("refiner_crop_size", 768),
                config.get("refiner_crop_probability", 0.),
                config.get("refiner_crop_seed", config.get("seed", 42)), step,
            ) if semantic_on and step >= refine_start else None)
            flip = (choose_refiner_horizontal_flip(flip_probability, config.get("seed", 42), step)
                    if semantic_on and step >= refine_start else False)
            if crop is not None and data["mask"] is None:
                raise ValueError("Refiner crop encountered a view without GT")
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            means_lr = 1.6e-4 * scene.scene_scale * (.01 ** progress)
            optimizers["means"].param_groups[0]["lr"] = means_lr
            stats.update(semantic_schedule.apply(step))
            result = scene.render(data["K"], pose, data["width"], data["height"], degree=degree,
                                  semantics=semantic_on, refine=step >= refine_start and crop is None and not flip,
                                  refinement_grad_to_field=config["refiner_field_grad"],
                                  semantic_classifier_grad=(data["mask"] is not None or not config["local_teacher_kd"]),
                                  absgrad=mcmc_state is None)
            if mixed_state is not None:
                mixed_state.retain_grad(result["info"], step)
            final_result = refine_render_crop(scene.refiner, result, crop) if crop is not None else result
            final_mask, final_valid = data["mask"], data["valid"]
            if crop is not None:
                ys, xs = crop_slices(crop, data["height"], data["width"])
                final_mask, final_valid = final_mask[ys, xs], final_valid[ys, xs]
            if flip:
                final_result, final_mask, final_valid = refine_horizontal_flip(
                    scene.refiner, result, final_mask, final_valid)
            if flip_probability > 0:
                stats.update(refiner_horizontal_flip=flip, refiner_horizontal_flip_view=view["name"])
            if "refiner_crop_probability" in config:
                stats.update(refiner_crop=list(crop) if crop is not None else None,
                             refiner_supervised_pixels=(int(((final_valid > 0) & (final_mask != 255)).sum())
                                                        if final_mask is not None else 0),
                             refiner_crop_view=view["name"])
            loss_rgb = rgb_loss(result["rgb"], data["rgb"], data["valid"])
            # A capped foreground emphasis preserves the full-image objective.
            if data["mask"] is not None and config.get("region_rgb_weight", 0):
                region = ((data["mask"] > 0) & (data["mask"] < 5)).float()
                region = F.max_pool2d(region[None, None], 5, 1, 2)[0, 0] * data["valid"]
                loss_rgb = loss_rgb + config["region_rgb_weight"] * masked_mean(
                    (result["rgb"] - data["rgb"]).abs().mean(-1), region)
            total = loss_rgb
            if entropy_weight > 0:
                entropy = opacity_entropy(scene.splats["opacity_logits"])
                total = total + entropy_weight * entropy
                stats["opacity_entropy"] = float(entropy.detach())
            if depth_support is not None and depth_weight > 0:
                depth_loss, depth_stats = depth_support.loss(
                    result["depth"], result["alpha"], data["valid"], data["K"], pose, view["image_id"])
                stats.update(depth_stats)
                if depth_loss is not None:
                    total = total + depth_weight * depth_loss
            if front_support is not None:
                front_loss, front_stats = optional_sparse_front_loss(
                    front_support, scene, data, pose, view["image_id"], front_weight)
                stats.update(front_stats)
                if front_loss is not None:
                    total = total + front_weight * front_loss
            sem_value = torch.zeros((), device="cuda")
            if semantic_on:
                if config["pseudo_refiner_weight"] > 0:
                    # Per-view diagnostics must not retain a previous unlabeled
                    # view's values on a labeled step.
                    stats.update(pseudo_refiner_active=False, pseudo_refiner_loss=0.,
                                 pseudo_refiner_pixels=0)
                if data["mask"] is not None:
                    sem_value = semantic_loss(final_result["probabilities"], final_mask, final_valid, class_weights)
                    sem_value = sem_value + .5 * semantic_loss(result["p3d"], data["mask"], data["valid"], raw_class_weights, 0)
                    total = total + config.get("semantic_weight", .2) * sem_value
                pseudo = (pseudo_for_view(config.get("pseudo_dir"), view, (data["height"], data["width"]))
                          if data["mask"] is None else None)
                if pseudo is not None:
                    p, valid, confidence = pseudo
                    reliability = valid * data["valid"] * (confidence >= config.get("pseudo_threshold", .8))
                    raw_pseudo = optional_raw_pseudo_loss(
                        result["p3d"], p, reliability, config.get("pseudo_weight", .1))
                    if raw_pseudo is not None:
                        total = total + raw_pseudo
                    if config["pseudo_refiner_weight"] > 0 and step >= refine_start:
                        teacher_refinement = balanced_probability_distillation(
                            result["probabilities"], p, reliability,
                            config.get("pseudo_class_reweight_cap", 3.0))
                        total = total + config["pseudo_refiner_weight"] * teacher_refinement
                        stats["pseudo_refiner_loss"] = float(teacher_refinement.detach())
                        stats["pseudo_refiner_pixels"] = int((reliability > 0).sum())
                        stats["pseudo_refiner_active"] = True
                total = total + .001 * final_result["residual"].square().mean()
                if fused_target is not None:
                    indices, targets, reliability = fused_target
                    reliability, conflict = supervised_teacher_weights(
                        targets, reliability, scene.semantic_prior_counts[indices]) if config["supervised_teacher_priority"] else (
                            reliability, torch.zeros_like(reliability, dtype=torch.bool))
                    local_kd = optional_multiview_loss(
                        scene.splats["sem_features"], scene.semantic_decoder, indices,
                        targets, reliability, config.get("multiview_weight", .05),
                        config["local_teacher_kd"])
                    if local_kd is not None:
                        total = total + local_kd
                    stats["fusion_weighted_kd"] = (float(local_kd.detach())
                                                    if local_kd is not None else 0.)
                    stats["fusion_accepted_weight_sum"] = float(reliability.sum())
                    stats["fusion_accepted_weight_by_class"] = torch.zeros(
                        targets.shape[-1], device=reliability.device,
                        dtype=reliability.dtype).scatter_add_(
                            0, targets.argmax(-1), reliability).tolist()
                    stats["fusion_supervised_conflicts"] = int((conflict & (fused_target[2] > 0)).sum())
                    stats["fusion_teacher_by_class"] = torch.bincount(
                        targets[fused_target[2] > 0].argmax(-1), minlength=targets.shape[-1]).tolist()
                    stats["fusion_accepted_by_class"] = torch.bincount(
                        targets[reliability > 0].argmax(-1), minlength=targets.shape[-1]).tolist()
                    if data["mask"] is not None and config.get("semantic_geometry_weight", 0) > 0 and step % 10 == 0:
                        gate = scene.render_evidence_gate(indices, reliability, data["K"], pose, data["width"], data["height"])
                        gate *= gate >= config.get("geometry_gate_threshold", .5)
                        # RGB edges provide independent structure support; no semantics-to-pose path.
                        luminance = data["rgb"].mean(-1)
                        edge = torch.zeros_like(luminance)
                        edge[:, 1:] += (luminance[:, 1:] - luminance[:, :-1]).abs()
                        edge[1:] += (luminance[1:] - luminance[:-1]).abs()
                        gate *= (edge > .03) * data["valid"] * (result["alpha"][..., 0].detach() > .7)
                        stats["geometry_gate_pixels"] = int((gate > 0).sum())
                        stats["geometry_gate_fraction"] = float((gate > 0).float().mean())
                        geometric = scene.render(data["K"], pose, data["width"], data["height"],
                                                 degree=degree, geometry_grad=True, refine=False, absgrad=False)
                        total = total + config["semantic_geometry_weight"] * semantic_loss(
                            geometric["p3d"], data["mask"], gate, raw_class_weights, 0)
            if mcmc_state is not None:
                regularization, regularization_stats = mcmc_regularization(scene, mcmc_policy)
                total = total + regularization
                stats.update(regularization_stats)
            else:
                total = total + 1e-4 * scene.splats["log_scales"].exp().mean() / max(scene.scene_scale, 1e-6)
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            if total.requires_grad:
                total.backward()
            else:
                stats["steps_without_trainable_supervision"] = stats.get("steps_without_trainable_supervision", 0) + 1
            mcmc_finite_check = mcmc_state is not None and (step == start or step % 100 == 0 or step == steps)
            if mcmc_finite_check:
                check_mcmc_finite(scene, step, gradients=True)
            density_mode = config.get("densification", "attribution")
            if mixed_state is not None:
                mixed_state.accumulate(result["info"], step)
            if density_mode in {"absgrad", "hybrid"} and densify_start <= step < densify_stop:
                from .densification import normalized_absgrad
                means2d = result["info"]["means2d"]
                grad = getattr(means2d, "absgrad", None)
                if grad is not None:
                    values = normalized_absgrad(grad[0], data["width"], data["height"])
                    radii = result["info"]["radii"][0]
                    visible = (radii > 0).all(-1) if radii.ndim > 1 else radii > 0
                    radius = radii.max(-1).values if radii.ndim > 1 else radii
                    extras["radii"] = torch.maximum(extras["radii"], radius.float() / max(data["width"], data["height"]))
                    if density_mode == "hybrid" and config["camera_attribution"] and config["camera_quality_weighting"]:
                        values = values * extras["camera_quality"][vi]
                    scores += values * visible
                    observations += visible.long()
                    support = visible & (values >= config.get("absgrad_support_threshold", config.get("absgrad_threshold", 0)))
                    previous = observed_views.get(vi, torch.zeros_like(support))
                    counts += (support & ~previous).long()
                    observed_views[vi] = previous | support
            for opt in optimizers.values():
                opt.step()
            if mcmc_state is None:
                # Legacy policies keep their established post-Adam constraints.
                # MCMC uses the installed strategy's own relocation corrections.
                with torch.no_grad():
                    if scene.splats["quats"].requires_grad:
                        scene.splats["quats"].copy_(F.normalize(scene.splats["quats"], dim=-1))
                    if scene.splats["log_scales"].requires_grad:
                        scene.splats["log_scales"].clamp_(math.log(scene.scene_scale * 1e-6), math.log(scene.scene_scale))
                    if scene.splats["opacity_logits"].requires_grad:
                        scene.splats["opacity_logits"].clamp_(-12, 12)

            needs_camera = config.get("optimize_cameras", False) and step < camera_stop
            needs_attribution = (density_mode in {"attribution", "hybrid"}
                                 and densify_start <= step < densify_stop)
            if step % diagnostic_every == 0 and (needs_camera or needs_attribution):
                from .reliability import (
                    ResidualAttribution,
                    camera_compensated_residual,
                    projection_jacobians,
                    se3_exp,
                )
                small = cache.get(view, min(scale, 320 / view["width"]))
                with torch.no_grad():
                    render_fn = lambda candidate, small=small, degree=degree: scene.render(small["K"], candidate, small["width"], small["height"],
                                                               degree=degree, semantics=False, absgrad=False)["rgb"]
                    if config["camera_attribution"] or needs_camera:
                        diagnostic = camera_compensated_residual(
                            render_fn, pose, small["rgb"], weights=small["valid"][..., None],
                            max_translation=.001 * scene.scene_scale, max_rotation=.002,
                            translation_eps=1e-5 * scene.scene_scale, rotation_eps=1e-4,
                            prior_precision=torch.diag(torch.tensor(
                                [1 / (.005 * scene.scene_scale) ** 2] * 3 + [1 / .01 ** 2] * 3,
                                device="cuda")))
                    else:
                        raw = render_fn(pose) - small["rgb"]
                        weighted = small["valid"][..., None].expand_as(raw)
                        energy = masked_mean(raw.square(), weighted)
                        diagnostic = ResidualAttribution(pose.new_zeros(6), raw, pose.new_zeros(()), energy, energy, False)
                    # Global gauge: first training camera remains fixed. Bounded TOTAL correction.
                    if config.get("optimize_cameras", False) and vi != 0 and step < camera_stop:
                        candidate = se3_exp(diagnostic.delta) @ pose
                        displacement = (candidate[:3, 3] - camera_reference[vi, :3, 3]).norm()
                        angle = torch.acos(((torch.trace(candidate[:3, :3] @ camera_reference[vi, :3, :3].T) - 1) / 2).clamp(-1, 1))
                        if diagnostic.accepted and displacement < .005 * scene.scene_scale and angle < .01:
                            cameras[vi].copy_(candidate)
                    stats["camera_explained_fraction"] = float(diagnostic.explained_fraction)
                    if config["camera_attribution"]:
                        extras["camera_quality"][vi] = (.5 * extras["camera_quality"][vi]
                                                       + .5 * (1 - diagnostic.explained_fraction))
                    else:
                        extras["camera_quality"][vi] = 1
                    if needs_attribution:
                        # The verified remaining residual lives on the corrected
                        # projection grid, even when the actual pose is frozen.
                        diagnostic_pose = se3_exp(diagnostic.delta) @ pose if config["camera_attribution"] else pose
                        projected = projection_jacobians(scene.splats["means"], diagnostic_pose, small["K"])
                        uv = projected.uv.round().long()
                        ok = (projected.depth > .01) & (uv[:, 0] >= 0) & (uv[:, 0] < small["width"]) & (uv[:, 1] >= 0) & (uv[:, 1] < small["height"])
                        ids = ok.nonzero().flatten()
                        uv_ok = uv[ids]
                        remainder = (diagnostic.remaining if config["camera_attribution"] or not needs_camera
                                     else render_fn(pose) - small["rgb"])
                        remaining = remainder.abs().mean(-1)
                        values = remaining[uv_ok[:, 1], uv_ok[:, 0]]
                        visibility = scene.render(small["K"], diagnostic_pose, small["width"], small["height"],
                                                  degree=degree, semantics=False, absgrad=False)
                        depth = visibility["depth"][uv_ok[:, 1], uv_ok[:, 0], 0]
                        accepted = (depth - projected.depth[ids]).abs() < .05 * depth.clamp_min(.01)
                        accepted &= visibility["alpha"][uv_ok[:, 1], uv_ok[:, 0], 0] > .3
                        accepted &= small["valid"][uv_ok[:, 1], uv_ok[:, 0]].bool()
                        residual_scores = extras["residual_scores"] if density_mode == "hybrid" else scores
                        residual_observations = extras["residual_observations"] if density_mode == "hybrid" else observations
                        residual_counts = extras["residual_counts"] if density_mode == "hybrid" else counts
                        residual_seen = extras["residual_observed_views"] if density_mode == "hybrid" else observed_views
                        residual_scores[ids] += values * accepted
                        residual_observations[ids] += accepted.long()
                        support = torch.zeros(n, dtype=torch.bool, device="cuda")
                        support[ids] = (values > config.get("residual_threshold", .03)) & accepted
                        previous = residual_seen.get(vi, torch.zeros_like(support))
                        residual_counts += (support & ~previous).long()
                        residual_seen[vi] = previous | support

            reset_every = config.get("opacity_reset_every", 0)
            reset_start = config.get("opacity_reset_start", reset_every)
            reset_stop = config.get("opacity_reset_stop", densify_stop)
            if (not rgb_reference and reset_every and reset_start <= step < reset_stop
                    and step % reset_every == 0 and not config.get("freeze_geometry")):
                from .densification import reset_opacity_with_optimizer
                stats["opacity_reset_count"] = reset_opacity_with_optimizer(
                    scene.splats["opacity_logits"], optimizers["opacity_logits"],
                    config.get("opacity_reset_cap", .03))
                stats["opacity_reset_step"] = step
                extras["pause_until"] = step + config.get("opacity_reset_pause", len(views))
                scores.zero_()
                observations.zero_()
                counts.zero_()
                observed_views.clear()
                for key in ("radii", "residual_scores", "residual_observations", "residual_counts"):
                    extras[key].zero_()
                extras["residual_observed_views"].clear()

            if (densify_start <= step < densify_stop and step >= extras["pause_until"]
                    and step % densify_every == 0 and density_mode not in {"none", MIXED_MODE, MCMC_MODE}):
                from .densification import apply_densification, structure_guided_densify
                priorities = scores / observations.clamp_min(1)
                support_counts = counts
                gradient_threshold = config.get("absgrad_threshold", 0)
                if density_mode in {"hybrid", "absgrad"} and gradient_threshold > 0:
                    from .densification import adaptive_gradient_threshold
                    gradient_threshold = adaptive_gradient_threshold(
                        priorities, counts, gradient_threshold, config.get("densify_min_views", 2),
                        config.get("absgrad_quantile"), config.get("absgrad_threshold_floor", 0))
                    stats["density_absgrad_threshold"] = gradient_threshold
                if density_mode == "hybrid":
                    from .densification import hybrid_density_priority
                    priorities, support_counts = hybrid_density_priority(
                        scores, observations, counts, extras["residual_scores"],
                        extras["residual_observations"], extras["residual_counts"],
                        gradient_threshold=gradient_threshold or 8e-4,
                        residual_threshold=config.get("residual_threshold", .03),
                        min_views=config.get("densify_min_views", 2),
                        residual_weight=config.get("hybrid_residual_weight", .35))
                elif density_mode == "absgrad":
                    priorities = priorities * (priorities >= gradient_threshold)
                max_splits = config.get("max_splits", 2000)
                if config.get("densify_fraction"):
                    max_splits = min(max_splits, max(1, round(n * config["densify_fraction"])))
                active_budget = scheduled_gaussian_budget(step, config)
                # A resumed field may already exceed an earlier schedule point;
                # preserve it unless the user deliberately lowered the hard cap.
                active_budget = min(config.get("max_gaussians", 600000), max(active_budget, n))
                groups = None
                if config.get("foreground_allocation_fraction") is not None:
                    supervised = scene.semantic_prior_counts
                    mass = supervised.sum(-1)
                    purity, category = (supervised / mass[:, None].clamp_min(1)).max(-1)
                    groups = (mass >= 2) & (purity >= .8) & (category > 0)
                selection = structure_guided_densify(
                    scene.splats, priorities, support_counts,
                    max_gaussians=active_budget, max_splits=max_splits,
                    min_views=config.get("densify_min_views", 2),
                    structure_guided=config.get("structure_guided", True),
                    recycle_count=config.get("recycle_count", 0) if n >= config.get("max_gaussians", 600000) else 0,
                    contribution=observations.float() * scene.splats["opacity_logits"].detach().sigmoid(),
                    opacity_threshold=config.get("prune_opacity", .005),
                    clone_scale_threshold=(config["clone_scale_fraction"] * scene.scene_scale
                                           if config.get("clone_scale_fraction") else None),
                    screen_radii=extras["radii"], split_screen_radius=config.get("split_screen_radius"),
                    split_opacity_mode=config.get("split_opacity_mode", "transmittance"),
                    shrink_unstructured_all_axes=config.get("shrink_unstructured_all_axes", False),
                    structured_split_rule=config.get("structured_split_rule", "legacy"),
                    allocation_groups=groups,
                    foreground_fraction=config.get("foreground_allocation_fraction", .65),
                    reference_points=torch.tensor(points["points"], device="cuda").float() if config.get("structure_guided", True) else None)
                apply_densification(scene.splats, selection, list(optimizers.values()))
                scene.semantic_prior_counts = scene.semantic_prior_counts[selection.source_indices].clone()
                stats.update(splits=selection.split_count, duplicates=selection.duplicate_count,
                             pruned=selection.pruned_count, density_candidates=int((priorities > 0).sum()),
                             density_budget=max_splits, density_target_budget=active_budget,
                             density_multiview_supported=int((counts >= config.get("densify_min_views", 2)).sum()),
                             density_eligible_candidates=int(((priorities > 0) & (support_counts >= config.get("densify_min_views", 2))).sum()))
                if density_mode in {"hybrid", "absgrad"}:
                    grad_average = scores[observations > 0] / observations[observations > 0]
                    stats["density_absgrad_quantiles"] = (grad_average.quantile(grad_average.new_tensor([.5, .9, .99])).tolist()
                                                          if len(grad_average) else [])
                n = len(scene.splats["means"])
                scores = torch.zeros(n, device="cuda")
                observations = torch.zeros(n, device="cuda", dtype=torch.long)
                counts = torch.zeros(n, device="cuda", dtype=torch.long)
                observed_views.clear()
                preserved_extras = {key: extras[key] for key in ("camera_quality", "sampler_order", "sampler_cursor", "sampler_rng_state", "pause_until")}
                extras = initialize_density_extras(n, len(views), "cuda")
                extras.update(preserved_extras)
                fused_target = None

            if mixed_state is not None:
                from .mixed_gradient_reference import reference_post_step
                mixed_state, selection, diagnostics = reference_post_step(
                    scene.splats, optimizers, mixed_state, scene.scene_scale, step)
                stats.update(diagnostics)
                if selection is not None:
                    mip_topology_changed = True
                    scene.semantic_prior_counts = scene.semantic_prior_counts[selection.source_indices].clone()
                    n = len(scene.splats["means"])
                    stats["reference_peak_gaussians"] = max(stats.get("reference_peak_gaussians", 0), n)
                    scores = torch.zeros(n, device="cuda")
                    observations = torch.zeros(n, dtype=torch.long, device="cuda")
                    counts = torch.zeros(n, dtype=torch.long, device="cuda")
                    # Old strategy windows are inert here but keep checkpoint shapes valid.
                    preserved = {key: extras[key] for key in ("sampler_order", "sampler_cursor", "sampler_rng_state")}
                    extras = initialize_density_extras(n, len(views), "cuda")
                    extras.update(preserved)

            if mcmc_state is not None:
                diagnostics = mcmc_post_step(scene, optimizers, mcmc_state, step, means_lr)
                stats.update(diagnostics)
                stats["mcmc_training_view"] = view["name"]
                n = len(scene.splats["means"])
                stats["reference_peak_gaussians"] = max(stats.get("reference_peak_gaussians", 0), n)
                if len(scores) != n:
                    scores = torch.zeros(n, device="cuda")
                    observations = torch.zeros(n, dtype=torch.long, device="cuda")
                    counts = torch.zeros(n, dtype=torch.long, device="cuda")
                    preserved = {key: extras[key] for key in ("sampler_order", "sampler_cursor", "sampler_rng_state")}
                    extras = initialize_density_extras(n, len(views), "cuda")
                    extras.update(preserved)
                if mcmc_finite_check:
                    check_mcmc_finite(scene, step)
                    stats["mcmc_finite_check_step"] = step

            if (mip_source is not None and scene.splats["means"].requires_grad
                    and refresh_due(step, steps, mip_topology_changed)):
                # Raw grow/prune/reset already ran unchanged. One post-step
                # refresh handles periodic, topology and endpoint events together.
                stats.update(refresh_filter(scene, mip_source, step))

            # Recompute after any capacity change. Running fusion before a split
            # would immediately discard every target whenever periods coincide.
            if semantic_on and multiview_evidence_enabled(config) and step % config.get("fusion_every", 200) == 0:
                from .fusion import fuse_scene_evidence
                fused_target = fuse_scene_evidence(scene, views, cameras, config["pseudo_dir"],
                                                   max_points=config.get("fusion_points", 4096),
                                                   uncertainty=config.get("projection_uncertainty", True), init_path=init_path)
                stats["fused_reliable_points"] = 0 if fused_target is None else int((fused_target[2] > 0).sum())

            if (step % config.get("log_every", 20) == 0 or step == start
                    or (mcmc_state is not None and step == steps)):
                if mcmc_state is not None:
                    mcmc_log_scalars(stats)
                stats.update(step=step, loss=float(total.detach()), rgb_loss=float(loss_rgb.detach()),
                             semantic_loss=float(sem_value.detach()), gaussians=len(scene.splats["means"]),
                             elapsed_seconds=round(time.perf_counter() - tic, 2),
                             peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3))
                stats["sh_degree"] = degree
                log.write(json.dumps(stats) + "\n")
                log.flush()
                print(json.dumps(stats), flush=True)
            save = step == steps or (save_every > 0 and step % save_every == 0)
            if save:
                window = density_checkpoint_state(scores, observations, counts, observed_views, fused_target, extras, mixed_state)
                atomic_save(checkpoint(scene, optimizers, config, step, cameras, stats, window), output / "last.pt")
            if config.get("eval_every", 2000) and step % config.get("eval_every", 2000) == 0:
                from .evaluate import evaluate_scene
                metrics = evaluate_scene(scene, manifest, output / f"validation_{step:06d}",
                                         max_views=config.get("eval_max_views"), scale=config.get("eval_scale", 1.0))
                (output / f"metrics_{step:06d}.json").write_text(json.dumps(metrics, indent=2))
                # Preserve each evaluated candidate: the competition's visual normalization is unspecified.
                window = density_checkpoint_state(scores, observations, counts, observed_views, fused_target, extras, mixed_state)
                atomic_save(checkpoint(scene, optimizers, config, step, cameras, stats, window), output / f"step_{step:06d}.pt")
    finally:
        log.close()
    return output / "last.pt"
