"""Fixed-camera differentiable raw-image warps for appearance-only experiments.

No renderer, optimizer or training is started here. The installed OpenCV float-map
INTER_LINEAR policy is measured once and recorded (5.0 uses continuous weights;
older builds may use a 1/32 table). Image values retain gradients, coordinates and
camera parameters do not. Clamping RGB is an explicit caller decision.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .coordinates import CORNER, pixel_protocol
from .losses import masked_mean, ssim_map


@lru_cache(maxsize=1)
def opencv_float_map_policy():
    """Identify, never silently assume, installed float-map interpolation weights."""
    image = (np.arange(7*9*3, dtype=np.float32).reshape(7, 9, 3) * .173 % 1).copy()
    mx = np.array([[.371, 2.813, 5.194], [1.039, 3.917, 6.473]], np.float32)
    my = np.array([[.537, 1.116, 3.291], [2.687, 4.019, 1.333]], np.float32)
    x, y = np.floor(mx).astype(int), np.floor(my).astype(int)
    fx, fy = mx - x.astype(np.float32), my - y.astype(np.float32)
    continuous = (image[y, x] * ((1-fx)*(1-fy))[..., None]
                  + image[y, x+1] * (fx*(1-fy))[..., None]
                  + image[y+1, x] * ((1-fx)*fy)[..., None]
                  + image[y+1, x+1] * (fx*fy)[..., None])
    base, fractions = cv2.convertMaps(mx, my, cv2.CV_16SC2, nninterpolation=False)
    table = cv2.remap(image, base, fractions, cv2.INTER_LINEAR)
    actual = cv2.remap(image, mx, my, cv2.INTER_LINEAR)
    errors = {"continuous_float32": float(np.max(np.abs(actual-continuous))),
              "opencv_5bit_table": float(np.max(np.abs(actual-table)))}
    matches = [key for key, error in errors.items() if error <= 2e-7]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown/ambiguous OpenCV float-map interpolation: {errors}")
    return matches[0]


class FixedBilinearWarp(nn.Module):
    """HWC image gather with constant-zero borders and fixed OpenCV weights."""

    def __init__(self, map_x, map_y, source_height, source_width, *, require_full_support=True):
        super().__init__()
        mx, my = np.asarray(map_x, np.float32), np.asarray(map_y, np.float32)
        if mx.ndim != 2 or mx.shape != my.shape or not np.isfinite(mx).all() or not np.isfinite(my).all():
            raise ValueError("Expected matching finite HW coordinate maps")
        if min(source_height, source_width) < 1 or max(source_height, source_width) >= 32767:
            raise ValueError("Source dimensions exceed the OpenCV fixed-map range")
        if max(float(np.abs(mx).max()), float(np.abs(my).max())) >= 32766:
            raise ValueError("Coordinates exceed the OpenCV fixed-map range")
        self.weight_policy = opencv_float_map_policy()
        if self.weight_policy == "opencv_5bit_table":
            base, fractions = cv2.convertMaps(mx, my, cv2.CV_16SC2, nninterpolation=False)
            x0, y0 = base[..., 0].astype(np.int64), base[..., 1].astype(np.int64)
            fx = (fractions & 31).astype(np.float32) / 32
            fy = (fractions >> 5).astype(np.float32) / 32
        else:
            x0, y0 = np.floor(mx).astype(np.int64), np.floor(my).astype(np.int64)
            fx, fy = mx - x0.astype(np.float32), my - y0.astype(np.float32)
        x = np.stack((x0, x0 + 1, x0, x0 + 1))
        y = np.stack((y0, y0, y0 + 1, y0 + 1))
        weights = np.stack(((1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy))
        inside = (x >= 0) & (x < source_width) & (y >= 0) & (y < source_height)
        full = (inside | (weights == 0)).all(0)
        weights = weights * inside
        if require_full_support and not full.all():
            raise ValueError("Raw output has unsupported interpolation weight; enlarge the render canvas")
        indices = (y.clip(0, source_height-1) * source_width + x.clip(0, source_width-1))
        self.source_shape = (int(source_height), int(source_width))
        self.output_shape = tuple(mx.shape)
        self.register_buffer("indices", torch.from_numpy(indices.reshape(4, -1)))
        self.register_buffer("weights", torch.from_numpy(weights.reshape(4, -1)))
        self.register_buffer("full_support", torch.from_numpy(full))
        self.map_sha256 = hashlib.sha256(mx.tobytes() + my.tobytes()).hexdigest()
        self.weights_sha256 = hashlib.sha256(weights.tobytes()).hexdigest()

    def forward(self, image):
        if image.ndim != 3 or tuple(image.shape[:2]) != self.source_shape or not image.is_floating_point():
            raise ValueError("Expected a floating HWC source image")
        if image.device != self.indices.device:
            raise ValueError("Move the fixed warp to the image device before use")
        flat = image.reshape(-1, image.shape[-1])
        weights = self.weights.to(dtype=image.dtype)
        result = flat[self.indices[0]] * weights[0, :, None]
        for i in range(1, 4):
            result = result + flat[self.indices[i]] * weights[i, :, None]
        return result.reshape(*self.output_shape, image.shape[-1])

    def receipt(self):
        return {"interpolation": "opencv_INTER_LINEAR_float_map_fixed_coordinates_torch_gather",
                "weight_policy": self.weight_policy, "opencv_version": cv2.__version__,
                "source_shape": list(self.source_shape), "output_shape": list(self.output_shape),
                "map_sha256": self.map_sha256, "weights_sha256": self.weights_sha256,
                "fully_supported_pixels": int(self.full_support.sum()),
                "output_pixels": self.full_support.numel(),
                "coordinate_gradients": False, "image_gradients": True,
                "border": "constant_zero_no_GT_fill", "clamp": "caller_explicit"}


@dataclass
class RawGridCamera:
    render_K: np.ndarray
    render_width: int
    render_height: int
    native_left: int
    native_top: int
    native_width: int
    native_height: int
    warp: FixedBilinearWarp
    protocol: str

    def crop_native(self, image):
        if tuple(image.shape[:2]) != (self.render_height, self.render_width):
            raise ValueError("Unexpected render canvas")
        return image[self.native_top:self.native_top+self.native_height,
                     self.native_left:self.native_left+self.native_width]


def build_raw_grid(K, distortion, width, height, *, protocol=CORNER):
    """Same covering canvas as official export; native pinhole K/size unchanged.

    No resized/different-K prepared canvas is accepted by this minimal contract.
    Caller must confirm its prepared metadata matches the supplied source K/size.
    """
    from .evaluate import distortion_render_grid
    protocol = pixel_protocol(protocol)
    K = np.asarray(K, np.float32)
    render_K, cw, ch, source = distortion_render_grid(K, distortion, width, height, protocol)
    shift = render_K[:2, 2] - K[:2, 2]
    rounded = np.rint(shift).astype(int)
    if not np.allclose(shift, rounded, atol=1e-5, rtol=0):
        raise ValueError("Official canvas offset must be an integer translation")
    left, top = map(int, rounded)
    if left < 0 or top < 0 or left + width > cw or top + height > ch:
        raise ValueError("Native pinhole crop lies outside the covering canvas")
    if source is None:
        mx, my = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    else:
        mx, my = source[..., 0], source[..., 1]
    warp = FixedBilinearWarp(mx, my, ch, cw, require_full_support=True)
    return RawGridCamera(render_K, cw, ch, left, top, int(width), int(height), warp, protocol)


def complete_window_support(valid, size=7):
    """Boolean HW support with complete windows, including image-border exclusion."""
    if valid.ndim != 2 or size < 1 or size % 2 != 1:
        raise ValueError("Expected HW valid and a positive odd window")
    mask = valid.detach().bool().float()
    return F.avg_pool2d(mask[None, None], size, stride=1, padding=size//2)[0, 0] == 1


def appearance_rgb_loss(prediction, target, valid):
    """0.8 valid L1 + 0.2 complete-valid-window (1 - 7box SSIM), no labels.

    Does not clamp or quantize prediction. Caller must apply an identical explicit
    pre-crop/pre-warp clamp policy to both arms. Target/support are detached.
    """
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("Expected matching RGB HWC tensors")
    if tuple(valid.shape) != tuple(prediction.shape[:2]):
        raise ValueError("Expected matching HW validity")
    target = target.detach()
    pixels = valid.detach().bool()
    windows = complete_window_support(pixels, 7)
    if not bool(pixels.any()) or not bool(windows.any()):
        raise ValueError("No complete valid RGB/SSIM support")
    l1 = masked_mean((prediction - target).abs().mean(-1), pixels)
    dssim = masked_mean(1 - ssim_map(prediction, target), windows)
    return .8 * l1 + .2 * dssim, {"l1": l1.detach(), "one_minus_ssim7": dssim.detach(),
                                 "rgb_pixels": pixels.sum(), "ssim7_centers": windows.sum()}
