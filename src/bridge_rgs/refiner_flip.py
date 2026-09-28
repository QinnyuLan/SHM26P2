"""Stateless GT-only horizontal augmentation of a frozen field's refiner inputs."""
from __future__ import annotations

import math

import numpy as np
import torch

PROBABILITY_KEY = "refiner_horizontal_flip_probability"


def _probability(config):
    value = config.get(PROBABILITY_KEY, 0.)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{PROBABILITY_KEY} must be a finite number in [0,1]")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{PROBABILITY_KEY} must be a finite number in [0,1]")
    return value


def validate_refiner_flip_config(config):
    probability = _probability(config)
    if probability == 0:
        return probability
    if config.get("parameter_scope", "all") != "refiner_only":
        raise ValueError("Refiner flips require parameter_scope: refiner_only")
    if any(not config.get(key, False) for key in (
            "freeze_geometry", "freeze_rgb", "train_labeled_only", "independent_view_rng")):
        raise ValueError("Refiner flips require frozen geometry/RGB, labeled-only TRAIN and independent view RNG")
    if config.get("pseudo_dir") or config.get("multiview_fusion", False):
        raise ValueError("Refiner flips permit GT-only supervision; pseudo/fusion is forbidden")
    if config.get("refiner_crop_probability", 0.):
        raise ValueError("Refiner flips and crops must be separate controlled experiments")
    if any(config.get(key, 0) for key in (
            "pseudo_weight", "pseudo_refiner_weight", "multiview_weight", "semantic_geometry_weight",
            "sparse_depth_weight", "sparse_front_weight", "opacity_entropy_weight")):
        raise ValueError("Refiner flips forbid teacher/geometry/opacity auxiliary losses")
    if config.get("optimize_cameras", False) or config.get("densification", "none") != "none":
        raise ValueError("Refiner flips require fixed cameras and topology")
    seed = config.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Refiner flips require a nonnegative integer seed")
    return probability


def validate_refiner_flip_resume(saved_config, config):
    """Legacy absence means off; an active augmentation stream cannot change."""
    old, new = _probability(saved_config), _probability(config)
    if old != new:
        raise ValueError("Strict resume cannot change refiner horizontal flip probability; use warmstart")
    if new and saved_config.get("seed", 42) != config.get("seed", 42):
        raise ValueError("Strict resume cannot change the refiner horizontal flip seed; use warmstart")


def choose_refiner_horizontal_flip(probability, seed, step):
    """No global NumPy, Python, torch, CUDA or view-sampler RNG is consumed."""
    if probability == 0:
        return False
    probability = _probability({PROBABILITY_KEY: probability})
    if (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            or not isinstance(step, int) or isinstance(step, bool) or step < 1):
        raise ValueError("Refiner flip requires nonnegative integer seed and positive integer step")
    rng = np.random.default_rng(np.random.SeedSequence([seed, step, 704291]))
    return bool(rng.random() < probability)


def refine_horizontal_flip(refiner, rendered, mask, valid):
    """Flip every HWC input plus HW GT/valid; preserve original RGB and p3d.

    Only the returned final prediction and residual live on the flipped grid.
    Field inputs are detached even if the caller accidentally enables gradients;
    training configuration additionally freezes every parameter except refiner.
    No camera mutation or second 3D render occurs.
    """
    if mask is None:
        raise ValueError("Refiner flip requires a labeled TRAIN view")
    keys = ("features", "rgb", "depth", "alpha", "refinement_prior")
    height, width = rendered["rgb"].shape[:2]
    if (mask.ndim != 2 or valid.ndim != 2 or mask.shape != (height, width)
            or valid.shape != (height, width)):
        raise ValueError("Refiner flip GT/valid must share the full rendered HW grid")
    values = [rendered[key] for key in keys]
    if any(value.ndim != 3 or value.shape[:2] != (height, width) for value in values):
        raise ValueError("Refiner flip input HWC grids differ")
    features, rgb, depth, alpha, prior = [torch.flip(value.detach(), dims=(1,)) for value in values]
    extras = {}
    if "depth_moments" in rendered:
        moments = rendered["depth_moments"]
        if moments.ndim != 3 or moments.shape[:2] != (height, width):
            raise ValueError("Refiner flip moment context grid differs")
        extras["depth_moments"] = torch.flip(moments.detach(), dims=(1,))
    residual = refiner(features, rgb, depth, alpha, p3d=prior, **extras)
    result = {"probabilities": (prior.log() + residual).softmax(-1), "residual": residual}
    return result, torch.flip(mask, dims=(1,)), torch.flip(valid, dims=(1,))
