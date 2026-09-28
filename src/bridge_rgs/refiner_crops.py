"""Stateless training crops of rendered evidence; full camera geometry is retained."""

import math

import numpy as np


def validate_refiner_crop_config(config):
    probability = float(config.get("refiner_crop_probability", 0.))
    size = config.get("refiner_crop_size", 768)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("refiner_crop_probability must be in [0,1]")
    if not isinstance(size, int) or isinstance(size, bool) or size < 32 or size % 16:
        raise ValueError("refiner_crop_size must be a patch-aligned integer >=32")
    if probability == 0:
        return
    required = ("freeze_geometry", "freeze_rgb", "train_labeled_only", "independent_view_rng")
    if any(not config.get(key, False) for key in required):
        raise ValueError("Refiner crops require frozen geometry/RGB, labeled-only views and independent view RNG")
    if config.get("pseudo_dir") or config.get("multiview_fusion", False):
        raise ValueError("Refiner crops currently permit GT-only supervision; pseudo/fusion is forbidden")
    for key in ("pseudo_weight", "pseudo_refiner_weight", "multiview_weight", "semantic_geometry_weight",
                "sparse_depth_weight", "opacity_entropy_weight", "region_rgb_weight"):
        if config.get(key, 0):
            raise ValueError(f"Refiner crops forbid mixed auxiliary losses: {key}")
    if config.get("optimize_cameras", False) or config.get("densification", "none") != "none":
        raise ValueError("Refiner crops forbid camera changes and densification")
    if config.get("parameter_scope", "all") != "all":
        raise ValueError("Refiner crops require the normal supervised semantic parameter scope")


def choose_refiner_crop(height, width, size, probability, seed, step):
    """Return (top,left,height,width), independent of all global/view RNG streams."""
    if height < 1 or width < 1 or step < 1 or seed < 0:
        raise ValueError("Invalid crop grid, step or seed")
    if not 0 <= probability <= 1:
        raise ValueError("Invalid crop probability")
    if not probability:
        return None
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step), 768031]))
    if rng.random() >= probability:
        return None
    if height < size or width < size:
        raise ValueError("Native crop size exceeds the rendered image; do not rescale or pad labels silently")
    return (int(rng.integers(height - size + 1)), int(rng.integers(width - size + 1)), size, size)


def crop_slices(crop, height, width):
    y, x, h, w = crop
    if min(y, x) < 0 or min(h, w) < 1 or y + h > height or x + w > width:
        raise ValueError("Crop lies outside rendered grid")
    return slice(y, y + h), slice(x, x + w)


def refine_render_crop(refiner, rendered, crop):
    """Apply the same head to aligned HWC evidence; p3d supervision stays external/full."""
    h, w = rendered["rgb"].shape[:2]
    ys, xs = crop_slices(crop, h, w)
    keys = ("features", "rgb", "depth", "alpha", "refinement_prior")
    if any(value.shape[:2] != (h, w) for value in (rendered[k] for k in keys)):
        raise ValueError("Rendered refiner input grids differ")
    features, rgb, depth, alpha, prior = (rendered[k][ys, xs] for k in keys)
    extras = {}
    if "depth_moments" in rendered:
        if rendered["depth_moments"].shape[:2] != (h, w):
            raise ValueError("Rendered moment grid differs from crop inputs")
        extras["depth_moments"] = rendered["depth_moments"][ys, xs]
    residual = refiner(features, rgb, depth, alpha, p3d=prior, **extras)
    return {"probabilities": (prior.log() + residual).softmax(-1), "residual": residual}
