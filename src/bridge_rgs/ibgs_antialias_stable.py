"""Factored FP64 opacity AA for the centered-camera IBGS port.

Actual activated inputs are cast, never reactivated or quaternion-normalized.
For B = J R_camera R_quaternion diag(scale), Cauchy--Binet gives det(B B^T)
as the sum of three squared minors. This avoids Gram-matrix subtraction.
It is the existing determinant AA formula, not a new renderer. The backend's
own FP32 covariance/conic and approximate VJP behavior are unchanged.
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
    factor2d: torch.Tensor


PRECISION_POLICY = {
    'activated_inputs': 'cast to FP64; no exp, sigmoid or quaternion renormalization',
    'projection_and_factor': 'FP64 after cast of actual activated inputs',
    'raster_scalars': 'input dtype rounded FoV and scale modifier, then FP64',
    'near_mask': 'input dtype camera z >= bound near_plane',
    'determinant': 'sum of three squared 2x2 minors of B',
    'blurred_determinant': 'D + eps2d * sum(B**2) + eps2d**2',
    'blur_constant': 'nominal FP64 eps2d (.3); not bitwise CUDA .3f',
    'zero_determinant_gradient': 'constant zero branch (rank-degenerate convention)',
    'effective_opacity': 'FP64 multiplication, cast back to input opacity dtype',
    'floor_or_clipping': False,
}


def factor_compensation(factor2d, eps2d=.3):
    """Return rho, D and H from an FP64 Nx2x3 projected covariance factor.

    Positive D uses the derivative of this forward. D=0 uses a deliberate
    zero derivative because sqrt(det) is nonsmooth at rank one. No numerical
    floor is used; nonfinite/overflow and detectable squared-minor underflow
    fail loudly. This is not gsplat's rho+1e-6 regularized handwritten VJP.
    """
    _require(factor2d.ndim == 3 and factor2d.shape[-2:] == (2, 3)
             and factor2d.dtype == torch.float64
             and bool(torch.isfinite(factor2d).all()), 'Finite FP64 Nx2x3 factor required')
    _require(type(eps2d) in (float, int) and math.isfinite(eps2d) and eps2d > 0,
             'Positive finite covariance blur required')
    u, v = factor2d.unbind(-2)
    minors = torch.stack((u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0],
                          u[:, 0]*v[:, 2]-u[:, 2]*v[:, 0],
                          u[:, 1]*v[:, 2]-u[:, 2]*v[:, 1]), -1)
    det_original = minors.square().sum(-1)
    trace = factor2d.square().sum((-2, -1))
    det_blurred = det_original + eps2d*trace + eps2d**2
    _require(bool(torch.isfinite(det_original).all() & torch.isfinite(det_blurred).all()
                  & (det_blurred > 0).all()), 'Nonpositive/nonfinite factored determinant; no rescue floor')
    positive = det_original > 0
    _require(not bool(((minors != 0).any(-1) & ~positive).any()),
             'Squared-minor underflow; no rescue floor')
    ratio = det_original/det_blurred
    _require(bool(torch.isfinite(ratio).all()) and not bool((positive & (ratio == 0)).any()),
             'Nonfinite/underflow determinant ratio')
    # Never evaluate sqrt(0) on the active autograd branch: 0*infinity is NaN.
    safe = torch.where(positive, ratio, torch.ones_like(ratio))
    rho = torch.where(positive, safe.sqrt(), torch.zeros_like(ratio))
    return rho, det_original, det_blurred


def compensate_opacity(means, scales, rotations, opacities, w2c, *, width,
                       height, tanfovx, tanfovy, scale_modifier=1., eps2d=.3, near_plane=.01):
    """Same public call as the original adapter, with an FP64 factored rho.

    Inputs are the renderer's actual activated scales/normalized wxyz/opacity;
    no FP64 exp/normalization is substituted for their FP32 activation. The
    original input-dtype z test preserves the bound near-plane branch. Culled
    slots are made safe before division and have a zero factor/gradient.
    Camera transforms, quaternion formula, centered +/-1.3 FoV Jacobian and
    factor multiplication then run in FP64. Effective opacity alone is cast
    back for the unchanged IBGS rasterizer. No far, radius or backend culling
    rules are replaced, and no backend backward accuracy is implied.
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
    input_camera = means @ w2c[:3, :3].T+w2c[:3, 3]
    _require(bool(torch.isfinite(input_camera).all()), 'Nonfinite input-dtype camera transform')
    positive_depth = input_camera[:, 2] > 0
    active_depth = input_camera[:, 2] >= near_plane
    # Cast actual activated values before all factor arithmetic. In particular,
    # do not normalize this quaternion again or exponentiate log-scales in FP64.
    means64, scales64, rotations64, w2c64 = (t.to(torch.float64) for t in (means, scales, rotations, w2c))
    w, x, y, z = rotations64.unbind(-1)
    rotation = torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                            2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                            2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(n, 3, 3)
    tx, ty, modifier = (means.new_tensor(float(v)).to(torch.float64)
                        for v in (tanfovx, tanfovy, scale_modifier))
    scaled = rotation * (scales64*modifier).unsqueeze(-2)
    camera_rotation = w2c64[:3, :3]
    means_camera = means64 @ camera_rotation.T+w2c64[:3, 3]
    _require(bool(torch.isfinite(means_camera).all()), 'Nonfinite FP64 camera transform')
    _require(not bool((active_depth & (means_camera[:, 2] <= 0)).any()),
             'Active input-dtype depth nonpositive in FP64')
    camera_z = torch.where(active_depth, means_camera[:, 2], torch.ones_like(means_camera[:, 2]))
    camera_xy = torch.where(active_depth[:, None], means_camera[:, :2], torch.zeros_like(means_camera[:, :2]))
    fx, fy = width/(2*tx), height/(2*ty)
    limx, limy = 1.3*tx, 1.3*ty
    ratio_x = (camera_xy[:, 0]/camera_z).clamp(-limx, limx)
    ratio_y = (camera_xy[:, 1]/camera_z).clamp(-limy, limy)
    zero = torch.zeros_like(camera_z)
    jacobian = torch.stack((fx/camera_z, zero, -fx*ratio_x/camera_z,
                            zero, fy/camera_z, -fy*ratio_y/camera_z), -1).reshape(n, 2, 3)
    factor2d = jacobian @ camera_rotation @ scaled
    factor2d = torch.where(active_depth[:, None, None], factor2d, torch.zeros_like(factor2d))
    rho, det_original, det_blurred = factor_compensation(factor2d, eps2d)
    effective64 = opacities.to(torch.float64) * (rho[:, None] if opacities.ndim == 2 else rho)
    effective = effective64.to(opacities.dtype)
    # This Gram matrix is diagnostic only; neither rho nor its gradient use it.
    covariance2d = factor2d @ factor2d.transpose(-1, -2)
    return AAResult(effective, rho, covariance2d, det_original, det_blurred,
                    positive_depth, active_depth, factor2d)


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
