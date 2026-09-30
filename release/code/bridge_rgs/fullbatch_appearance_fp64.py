"""Independent FP64 image objective for fixed-geometry appearance experiments.

Renderer/warp endpoints and trainable parameters stay FP32. Only the existing
L1/SSIM image formula and its scalar accumulation use FP64; this is not an FP64
rasterizer. RMS/Armijo arithmetic remains in the original, unchanged helper.
"""
from __future__ import annotations

import torch

from .fullbatch_appearance import require
from .raw_grid import appearance_rgb_loss


def appearance_rgb_loss_fp64(prediction, target, valid):
    """Cast actual FP32 image endpoints, then use the unchanged original formula."""
    require(prediction.dtype == target.dtype == torch.float32,
            'Require actual FP32 rendered and target endpoints before FP64 image loss')
    return appearance_rgb_loss(prediction.double(), target.double(), valid)


def stream_mean_loss_fp64(items, loss_fn, *, backward, check_time=None):
    """Stream one FP64 scalar loss graph at a time into FP32 parameter gradients."""
    require(len(items) > 0, 'Empty full-batch objective')
    total = 0.
    for item in items:
        if check_time is not None:
            check_time()
        loss = loss_fn(item)
        require(loss.ndim == 0 and loss.dtype == torch.float64 and bool(torch.isfinite(loss)),
                'Finite scalar FP64 view objective required')
        total += float(loss.detach())
        if backward:
            (loss/len(items)).backward()
        del loss
        if check_time is not None:
            check_time()
    return total/len(items)
