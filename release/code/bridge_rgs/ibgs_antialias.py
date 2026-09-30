"""Differentiable opacity compensation for the centered-camera IBGS port.

The backend continues to use its existing Sigma_2D + .3 I footprint. Only its
opacity input changes. This is the usual determinant AA factor, not a new
renderer or a claim that IBGS has gsplat's complete forward/backward behavior.
Inputs are the actual activated scales and already normalized wxyz rotations
passed to the IBGS rasterizer. No installed or frozen source is modified.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class AAResult:
    opacity: torch.Tensor
    rho: torch.Tensor
    covariance2d: torch.Tensor
    det_original: torch.Tensor
    det_blurred: torch.Tensor
    positive_depth: torch.Tensor
    active_depth: torch.Tensor


def determinant_compensation(covariance2d, eps2d=.3):
    """Exact positive-ratio Torch derivative; safe zero branch at ratio <= 0.

    No determinant floor is added. gsplat 1.5.3 uses this forward formula but
    its handwritten VJP divides by rho + 1e-6. This function deliberately uses
    the derivative of its own positive-ratio forward, not that regularized VJP.
    """
    _require(covariance2d.ndim == 3 and covariance2d.shape[-2:] == (2, 2)
             and covariance2d.dtype in (torch.float32, torch.float64)
             and bool(torch.isfinite(covariance2d).all()), 'Finite floating Nx2x2 covariance required')
    _require(type(eps2d) in (float, int) and math.isfinite(eps2d) and eps2d > 0,
             'Positive finite covariance blur required')
    a, b, c, d = (covariance2d[:, i, j] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    det_original = a*d-b*c
    det_blurred = (a+eps2d)*(d+eps2d)-b*c
    _require(bool(torch.isfinite(det_original).all() & torch.isfinite(det_blurred).all()
                  & (det_blurred > 0).all()), 'Nonpositive/nonfinite blurred determinant; no rescue floor')
    ratio = det_original/det_blurred
    _require(bool(torch.isfinite(ratio).all()), 'Nonfinite determinant ratio')
    positive = ratio > 0
    # sqrt(clamp_min(ratio, 0)) alone produces an infinite derivative at zero,
    # allowing 0*inf NaNs even in masked slots. The explicit inactive branch is constant.
    safe = torch.where(positive, ratio, torch.ones_like(ratio))
    rho = torch.where(positive, safe.sqrt(), torch.zeros_like(ratio))
    return rho, det_original, det_blurred


def compensate_opacity(means, scales, rotations, opacities, w2c, *, width,
                       height, tanfovx, tanfovy, scale_modifier=1., eps2d=.3, near_plane=.01):
    """Return AA effective opacity with geometry gradients preserved.

    Geometry uses the IBGS centered-camera clamp +/-1.3 tan(FOV/2). Focal
    lengths are recomputed from the actual FP32 tanfov settings, as the CUDA
    backend does. w2c is row-major, not the transposed author viewmatrix.
    Slots below the explicitly bound near_plane have rho=0 and zero
    compensation gradient, avoiding overflow in culled near-zero-depth slots.
    The .01 default matches the isolated AA port; it is not a fitted cutoff.
    The backend still performs visibility/culling. No opacity or rho upper
    clipping, normalization, extra blur, or covariance/determinant floor.
    """
    _require(means.ndim == 2 and means.shape[1] == 3 and len(means) > 0,
             'Nonempty Nx3 means required')
    n = len(means)
    _require(scales.shape == (n, 3) and rotations.shape == (n, 4)
             and opacities.shape in ((n,), (n, 1)) and w2c.shape == (4, 4), 'Parameter shape mismatch')
    tensors = (means, scales, rotations, opacities, w2c)
    _require(means.dtype in (torch.float32, torch.float64)
             and all(t.dtype == means.dtype and t.device == means.device for t in tensors)
             and all(bool(torch.isfinite(t).all()) for t in tensors), 'Finite same-dtype/device tensors required')
    _require(bool((scales >= 0).all() & (opacities >= 0).all() & (opacities <= 1).all()),
             'Activated nonnegative scales and [0,1] opacities required')
    _require(bool(((rotations.square().sum(-1)-1).abs() <= 64*torch.finfo(means.dtype).eps).all()),
             'Actual normalized wxyz rotations required; do not normalize twice')
    _require(type(width) is int and type(height) is int and width > 0 and height > 0,
             'Positive integer image dimensions required')
    _require(all(type(x) in (int, float) and math.isfinite(x) and x > 0
                 for x in (tanfovx, tanfovy, scale_modifier, near_plane)), 'Positive finite raster settings required')
    w, x, y, z = rotations.unbind(-1)
    rotation = torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                            2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                            2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(n, 3, 3)
    scaled = rotation * (scales*scale_modifier).unsqueeze(-2)
    covariance_world = scaled @ scaled.transpose(-1, -2)
    camera_rotation = w2c[:3, :3]
    covariance_camera = camera_rotation @ covariance_world @ camera_rotation.T
    means_camera = means @ camera_rotation.T+w2c[:3, 3]
    positive_depth = means_camera[:, 2] > 0
    active_depth = means_camera[:, 2] >= near_plane
    camera_z = torch.where(active_depth, means_camera[:, 2], torch.ones_like(means_camera[:, 2]))
    camera_xy = torch.where(active_depth[:, None], means_camera[:, :2], torch.zeros_like(means_camera[:, :2]))
    tx, ty = (means.new_tensor(float(v)) for v in (tanfovx, tanfovy))
    fx, fy = width/(2*tx), height/(2*ty)
    limx, limy = 1.3*tx, 1.3*ty
    ratio_x = (camera_xy[:, 0]/camera_z).clamp(-limx, limx)
    ratio_y = (camera_xy[:, 1]/camera_z).clamp(-limy, limy)
    # Keep the clamped-x = z*clamp(x/z) operation explicit, like CUDA computeCov2D.
    clamped_x, clamped_y = ratio_x*camera_z, ratio_y*camera_z
    zero = torch.zeros_like(camera_z)
    jacobian = torch.stack((fx/camera_z, zero, -fx*clamped_x/camera_z.square(),
                            zero, fy/camera_z, -fy*clamped_y/camera_z.square()), -1).reshape(n, 2, 3)
    covariance2d = jacobian @ covariance_camera @ jacobian.transpose(-1, -2)
    covariance2d = torch.where(active_depth[:, None, None], covariance2d, torch.zeros_like(covariance2d))
    rho, det_original, det_blurred = determinant_compensation(covariance2d, eps2d)
    effective = opacities * (rho[:, None] if opacities.ndim == 2 else rho)
    return AAResult(effective, rho, covariance2d, det_original, det_blurred, positive_depth, active_depth)


@dataclass
class HookState:
    calls: int = 0
    last: AAResult | None = None


@contextmanager
def aa_opacity_rasterizer(rasterizer_class, *, capture_last=False, near_plane=.01):
    """Temporarily wrap author PlaneGaussianRasterizer.forward for RGB/depth.

    Construct one context for an explicitly bound renderer class. Both render
    and render_depth call the same class, so every camera gets its own rho.
    The call must use activated scales and normalized rotations, not a
    cov3D_precomp branch. The original tensor/parameter is never changed.
    capture_last defaults to False, retaining no cross-call graph for training.
    If enabled, state.last is overwritten each call; discard it after inspection.
    near_plane must match the bound backend (default: the isolated .01 port).
    The class method is restored even if projection or rendering raises.
    """
    original = rasterizer_class.forward
    _require(not getattr(original, '_ibgs_aa_opacity_hook', False), 'Nested AA hooks would double compensation')
    state = HookState()
    def forward(self, *args, **kwargs):
        _require(not args, 'Explicit rasterizer keyword arguments required')
        required = ('means3D', 'scales', 'rotations', 'opacities')
        _require(all(isinstance(kwargs.get(k), torch.Tensor) for k in required), 'Missing activated field input')
        cov = kwargs.get('cov3D_precomp')
        _require(cov is None or isinstance(cov, torch.Tensor) and cov.numel() == 0,
                 'Precomputed covariance branch is not covered by this adapter')
        settings = self.raster_settings
        result = compensate_opacity(kwargs['means3D'], kwargs['scales'], kwargs['rotations'],
            kwargs['opacities'], settings.viewmatrix.T.contiguous(),
            width=settings.image_width, height=settings.image_height,
            tanfovx=settings.tanfovx, tanfovy=settings.tanfovy, scale_modifier=settings.scale_modifier,
            near_plane=near_plane)
        state.calls += 1; state.last = result if capture_last else None
        return original(self, **dict(kwargs, opacities=result.opacity))
    forward._ibgs_aa_opacity_hook = True
    rasterizer_class.forward = forward
    try:
        yield state
    finally:
        rasterizer_class.forward = original
