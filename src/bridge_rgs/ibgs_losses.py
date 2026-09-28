"""Valid-domain losses for the IBGS reference port (no semantic labels).

Uses the author's Gaussian 11x11 SSIM formula. Validity is eroded for spatial
terms so undistortion padding cannot influence an accepted loss window.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def erode_valid(valid, radius):
    if valid.ndim != 2 or valid.dtype != torch.bool or radius < 0:
        raise ValueError('Expected boolean HxW validity and nonnegative radius')
    if radius == 0:
        return valid
    invalid = F.pad((~valid).float()[None, None], (radius,)*4, value=1.)
    return F.max_pool2d(invalid, 2*radius+1, stride=1)[0, 0] == 0


def gaussian_ssim_map(prediction, target):
    if prediction.ndim != 3 or prediction.shape != target.shape or prediction.shape[0] != 3:
        raise ValueError('Expected matching 3xHxW RGB tensors')
    coordinate = torch.arange(11, device=prediction.device, dtype=prediction.dtype)-5
    kernel = torch.exp(-coordinate.square()/(2*1.5**2))
    kernel = kernel/kernel.sum()
    window = (kernel[:, None]*kernel[None, :])[None, None].expand(3, 1, 11, 11).contiguous()
    x, y = prediction[None].contiguous(), target[None].contiguous()
    mx, my = (F.conv2d(t, window, padding=5, groups=3) for t in (x, y))
    vx = F.conv2d(x*x, window, padding=5, groups=3)-mx.square()
    vy = F.conv2d(y*y, window, padding=5, groups=3)-my.square()
    covariance = F.conv2d(x*y, window, padding=5, groups=3)-mx*my
    score = ((2*mx*my+.01**2)*(2*covariance+.03**2)
             / ((mx.square()+my.square()+.01**2)*(vx+vy+.03**2)))
    return score[0].mean(0)


def masked_photometric_loss(prediction, target, valid, *, ssim_weight=.2):
    if not 0 <= ssim_weight <= 1 or valid.shape != prediction.shape[-2:]:
        raise ValueError('Invalid mask shape or SSIM weight')
    valid_windows = erode_valid(valid, 5)
    if not valid.any() or (ssim_weight > 0 and not valid_windows.any()):
        raise ValueError('No valid RGB/SSIM supervision')
    l1 = (prediction-target).abs().mean(0)[valid].mean()
    dssim = (1-gaussian_ssim_map(prediction, target))[valid_windows].mean() if ssim_weight else l1*0
    return (1-ssim_weight)*l1+ssim_weight*dssim, {'l1': l1, 'dssim': dssim}


def masked_normal_loss(rendered_normal, depth_normal, depth, valid):
    active = erode_valid(valid & torch.isfinite(depth) & (depth > 0), 1)
    if not active.any():
        return rendered_normal.sum()*0
    l1 = (depth_normal-rendered_normal).abs().sum(0)[active].mean()
    cosine = (1-(depth_normal*rendered_normal).sum(0))[active].mean()
    return .4*l1+.6*cosine
