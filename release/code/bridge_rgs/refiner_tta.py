"""Fixed two-pass horizontal test augmentation on one rendered evidence grid."""
from __future__ import annotations

import torch


@torch.inference_mode()
def horizontal_flip_average(refiner, rendered):
    """Average probabilities, undoing the second head's horizontal reflection.

    Reuses one 3D rendering. This standard inference augmentation takes no
    photograph, GT, view identifier, or camera-specific fitting parameter.
    """
    height, width = rendered["rgb"].shape[:2]
    values = [rendered[key] for key in ("features", "rgb", "depth", "alpha", "refinement_prior")]
    if any(value.ndim != 3 or value.shape[:2] != (height, width) for value in values):
        raise ValueError("Flip TTA evidence must use one aligned HWC render grid")
    features, rgb, depth, alpha, prior = [value.flip(1) for value in values]
    kwargs = {}
    if "depth_moments" in rendered:
        moments = rendered["depth_moments"]
        if moments.ndim != 3 or moments.shape[:2] != (height, width):
            raise ValueError("Flip TTA moment context grid differs")
        kwargs["depth_moments"] = moments.flip(1)
    residual = refiner(features, rgb, depth, alpha, p3d=prior, **kwargs)
    reflected = (prior.log() + residual).softmax(-1).flip(1)
    if reflected.shape != rendered["probabilities"].shape:
        raise ValueError("Flip TTA probability grids differ")
    return .5 * rendered["probabilities"] + .5 * reflected
