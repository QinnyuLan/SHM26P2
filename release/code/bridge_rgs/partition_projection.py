"""Fixed-geometry EWA adapter for the experimental semantic partition shader.

The pinhole Jacobian and (w,x,y,z) quaternion convention follow installed
gsplat 1.5.3 cuda/include/Utils.cuh:persp_proj/quat_to_rotmat. In particular,
the covariance Jacobian clips x/z and y/z using the *actual principal point*;
the projected mean itself is NOT clipped. Algebraic agreement is intended,
not a claim of bitwise equivalence to CUDA FMA/rsqrt implementations.

Revision 2 explicitly transports a valid ideal conditional Gaussian to the
actual rasterizer footprint. P0 = J R_camera R_quaternion diag(scales), with
C0=P0 P0.T+0.3 I, L0=chol(C0), and B=solve(L0,P0), is prepared in FP64. For
actual Cr=solve(conic,I), Lr=chol(Cr), define P'=Lr B and noise'=0.3 T T.T,
where T=Lr L0^-1. Thus P'P'.T+noise'=Cr, A=B.T Lr^-1, V=I-B.T B. Noise'
is generally NOT 0.3 I. This is an actual-footprint-consistent surrogate, not
an assertion that the original FP32 conic-comparison gate passed. The fixed
Cholesky choice depends on screen-coordinate order; it is not a unique,
rotation-equivariant transport. Rasterization weights/culling are untouched.

Dimensionless local xi retains its unit Gaussian prior. Scale exponentiation
uses the input dtype, as in scene.render, then geometry arithmetic uses FP64.
The expensive geometry preparation is performed once per fixed camera/state;
learnable slab parameters only require matrix/vector contractions thereafter.
No renderer, filesystem, random sampling, or device launch is invoked here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .semantic_partition import MODES

EPS2D = .3
REVISION = "actual_footprint_whitened_transport_v2"


@dataclass(frozen=True)
class FrozenProjection:
    A: torch.Tensor  # [N,3,2] conditional mean map
    V: torch.Tensor  # [N,3,3] conditional covariance
    means2d: torch.Tensor  # [N,2], exact supplied means when provided
    visible: torch.Tensor  # [N], actual radii or depth-only candidates
    diagnostics: dict


def _same_tensor(value, shape, ref, name):
    if (not isinstance(value, torch.Tensor) or value.shape != shape
            or value.dtype != ref.dtype or value.device != ref.device):
        raise ValueError(f"{name}: incorrect shape/dtype/device")
    if value.requires_grad:
        raise ValueError(f"{name} must be explicitly frozen")


def _quaternion_rotation(quats):
    w, x, y, z = (quats * torch.rsqrt(quats.square().sum(-1, keepdim=True))).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


@torch.no_grad()
def prepare_projection(means, quats, log_scales, w2c, K, width, height, *,
                       radii=None, means2d=None, conics=None,
                       near_plane=.01, far_plane=1e6):
    """Cache A/V for one frozen pinhole camera, optionally binding gsplat info.

    Inputs use one common FP32/FP64 dtype. Shapes: means/log_scales[N,3],
    quats[N,4], w2c[4,4], K[3,3]. Optional gsplat metadata must be from the SAME
    camera/render: radii[N,2], means2d[N,2], conics[N,3]. Supply ``info`` arrays
    after removing their singleton camera dimension. Radii define the actual
    active mask, including opacity-aware frustum/radius culling. Without radii
    the mask is ONLY a finite positive-depth candidate set, not visibility.

    Supplied SPD conics define Cr and supplied means define the exact center.
    Their same-camera/same-scene identity is a CALLER provenance contract, not
    inferable from a cancellation-sensitive relative mean/conic threshold.
    Differences against the FP64 ideal are recorded, including counts exceeding
    the old 256-eps conic and 8-eps mean gates; those old gates are not relabelled
    as passing. Caller must not bind metadata from a different render.
    Invisible gsplat outputs may be uninitialized and are never validated or
    consumed. An active invalid geometry/metadata row fails rather than drops.
    """
    if (not isinstance(means, torch.Tensor) or means.dtype not in (torch.float32, torch.float64)
            or means.ndim != 2 or means.shape[1] != 3 or len(means) == 0):
        raise ValueError("means must be nonempty [N,3] float32/float64")
    if (isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int)
            or not isinstance(height, int) or min(width, height) <= 0
            or not 0 < near_plane < far_plane < float("inf")):
        raise ValueError("Invalid image dimensions or clipping planes")
    N = len(means)
    for value, shape, name in ((means, (N, 3), "means"), (quats, (N, 4), "quats"),
                               (log_scales, (N, 3), "log_scales"), (w2c, (4, 4), "w2c"),
                               (K, (3, 3), "K")):
        _same_tensor(value, shape, means, name)
    geometry_means, geometry_w2c, geometry_K = means.double(), w2c.double(), K.double()
    rotation = geometry_w2c[:3, :3]
    eye3 = torch.eye(3, dtype=torch.float64, device=means.device)
    camera_ok = (torch.isfinite(w2c).all() & torch.isfinite(K).all()
                 & ((rotation @ rotation.T - eye3).abs().max() <= 1e-5)
                 & (abs(torch.linalg.det(rotation)-1) <= 1e-5)
                 & ((w2c[3]-w2c.new_tensor([0., 0., 0., 1.])).abs().max() <= 1e-6)
                 & (K[0, 0] > 0) & (K[1, 1] > 0)
                 & (K[0, 1] == 0) & (K[1, 0] == 0)
                 & (K[2, 0] == 0) & (K[2, 1] == 0) & (K[2, 2] == 1))
    if not bool(camera_ok):
        raise ValueError("Require finite proper pinhole K/w2c without skew")
    camera_means = geometry_means @ rotation.T + geometry_w2c[:3, 3]
    depth = camera_means[:, 2]
    finite_geometry = (torch.isfinite(camera_means).all(-1) & torch.isfinite(quats).all(-1)
                       & torch.isfinite(log_scales).all(-1) & (quats.square().sum(-1) > 0)
                       & torch.isfinite(quats.square().sum(-1)))
    depth_valid = finite_geometry & (depth >= near_plane) & (depth <= far_plane)
    if radii is not None:
        if (not isinstance(radii, torch.Tensor) or radii.shape != (N, 2)
                or radii.device != means.device or radii.dtype not in (torch.int32, torch.int64)):
            raise ValueError("radii must be integer [N,2] on the geometry device")
        visible = (radii > 0).all(-1)
        if bool((visible & ~depth_valid).any()):
            raise ValueError("Active gsplat row has invalid geometry or clipping depth")
        mask_source = "supplied_gsplat_radii"
    else:
        if means2d is not None or conics is not None:
            raise ValueError("Binding gsplat means/conics also requires its active radii")
        visible = depth_valid
        mask_source = "depth_candidates_only_not_full_visibility"
    indices = visible.nonzero().flatten()
    count = len(indices)
    A = means.new_zeros(N, 3, 2)
    V = means.new_zeros(N, 3, 3)
    output_means = means.new_zeros(N, 2)
    for value, shape, name in ((means2d, (N, 2), "means2d"), (conics, (N, 3), "conics")):
        if value is not None:
            _same_tensor(value, shape, means, name)
    diagnostics = {"points": N, "active_points": count, "mask_source": mask_source,
                   "revision": REVISION, "geometry_compute_dtype": "float64",
                   "ideal_noise_pixel_variance": EPS2D,
                   "effective_noise": "0.3*T*T_transpose; generally anisotropic when metadata supplied",
                   "metadata_identity_contract": "caller binds same scene/camera/render",
                   "near_plane": near_plane, "far_plane": far_plane,
                   "conic_source": "supplied_gsplat_transport" if conics is not None else "ideal_cholesky",
                   "means_source": "supplied_gsplat" if means2d is not None else "pinhole_formula",
                   "mean_max_absolute_difference": 0., "conic_relative_max_difference": 0.,
                   "old_conic_gate_exceeded_rows": 0, "old_mean_gate_exceeded_rows": 0,
                   "transport_max_absolute_change_from_identity": 0.,
                   "effective_noise_max_absolute_change_from_ideal": 0.,
                   "footprint_relative_reconstruction_error": 0.,
                   "posterior_joseph_absolute_difference": 0.,
                   "conditional_total_covariance_absolute_residual": 0.}
    if count == 0:
        return FrozenProjection(A, V, output_means, visible, diagnostics)
    xyz = camera_means[indices]
    x, y, z = xyz.unbind(-1)
    fx, fy, cx, cy = geometry_K[0, 0], geometry_K[1, 1], geometry_K[0, 2], geometry_K[1, 2]
    rz = 1 / z
    tx = z * torch.clamp(x * rz, min=-(cx/fx+.3*(.5*width/fx)),
                         max=(width-cx)/fx+.3*(.5*width/fx))
    ty = z * torch.clamp(y * rz, min=-(cy/fy+.3*(.5*height/fy)),
                         max=(height-cy)/fy+.3*(.5*height/fy))
    zero = torch.zeros_like(z)
    J = torch.stack((fx*rz, zero, -fx*tx*rz.square(),
                     zero, fy*rz, -fy*ty*rz.square()), -1).reshape(-1, 2, 3)
    scale = log_scales[indices].exp().double()
    P = (J @ rotation @ _quaternion_rotation(quats[indices].double())) * scale[:, None, :]
    projected_mean = torch.stack((fx*x*rz+cx, fy*y*rz+cy), -1)
    eye2 = torch.eye(2, device=means.device, dtype=torch.float64).expand(count, 2, 2)
    C = P @ P.mT + EPS2D * eye2
    if not bool(torch.isfinite(C).all() & torch.isfinite(P).all()
                & torch.isfinite(projected_mean).all() & (scale > 0).all()):
        raise ValueError("Nonfinite/underflowed active Gaussian projection")
    chol, info = torch.linalg.cholesky_ex(C)
    if bool((info != 0).any()):
        raise ValueError("Active projected covariance is not numerically positive definite")
    B = torch.linalg.solve_triangular(chol, P, upper=False)
    ideal_A = torch.linalg.solve_triangular(chol.mT, B, upper=True).mT
    # Joseph covariance form is algebraically I-B.T B, but a sum of two Gram
    # matrices avoids subtractive negative eigenvalues at high anisotropy.
    residual_map = eye3 - B.mT @ B
    computed_V = residual_map @ residual_map.mT + EPS2D * (ideal_A @ ideal_A.mT)
    computed_V = (computed_V + computed_V.mT) * .5
    computed_A = ideal_A
    actual_C = C
    diagnostics["posterior_joseph_absolute_difference"] = float((computed_V-residual_map).abs().max())
    eps = torch.finfo(means.dtype).eps
    if conics is not None:
        actual = conics[indices].double()
        inverse = torch.stack((actual[:, 0], actual[:, 1], actual[:, 1], actual[:, 2]), -1)
        inverse = inverse.reshape(-1, 2, 2)
        expected = torch.cholesky_solve(eye2, chol)
        relative = ((inverse-expected).abs().amax((-2, -1))
                    / expected.abs().amax((-2, -1)).clamp_min(torch.finfo(torch.float64).tiny))
        if not bool(torch.isfinite(inverse).all()):
            raise ValueError("Actual conics must be finite SPD")
        conic_chol, conic_info = torch.linalg.cholesky_ex(inverse)
        if bool((conic_info != 0).any()):
            raise ValueError("Actual conics must be finite SPD")
        actual_C = torch.cholesky_solve(eye2, conic_chol)
        actual_C = (actual_C + actual_C.mT) * .5
        actual_chol, actual_info = torch.linalg.cholesky_ex(actual_C)
        if bool((actual_info != 0).any()):
            raise ValueError("Actual footprint is not numerically positive definite")
        computed_A = torch.linalg.solve_triangular(actual_chol.mT, B, upper=True).mT
        transport = torch.linalg.solve_triangular(chol.mT, actual_chol.mT, upper=True).mT
        effective_P = actual_chol @ B
        effective_noise = EPS2D * (transport @ transport.mT)
        reconstruction = effective_P @ effective_P.mT + effective_noise
        reconstruction_error = ((reconstruction-actual_C).abs().amax((-2, -1))
                                / actual_C.abs().amax((-2, -1)))
        diagnostics["conic_relative_max_difference"] = float(relative.max())
        diagnostics["old_conic_gate_exceeded_rows"] = int((relative > 256*eps).sum())
        diagnostics["transport_max_absolute_change_from_identity"] = float((transport-eye2).abs().max())
        diagnostics["effective_noise_max_absolute_change_from_ideal"] = float((effective_noise-EPS2D*eye2).abs().max())
        diagnostics["footprint_relative_reconstruction_error"] = float(reconstruction_error.max())
        if not bool(torch.isfinite(effective_noise).all() & torch.isfinite(reconstruction_error).all()):
            raise ValueError("Nonfinite transported footprint")
    if not bool(torch.isfinite(computed_V).all() & torch.isfinite(computed_A).all()):
        raise ValueError("Nonfinite conditional geometry")
    total_covariance_error = computed_A @ actual_C @ computed_A.mT + computed_V - eye3
    diagnostics["conditional_total_covariance_absolute_residual"] = float(total_covariance_error.abs().max())
    if means2d is not None:
        actual = means2d[indices].double()
        error = (actual - projected_mean).abs()
        if not bool(torch.isfinite(actual).all()):
            raise ValueError("Actual projected means must be finite")
        diagnostics["old_mean_gate_exceeded_rows"] = int((error > 8*eps*projected_mean.abs().clamp_min(1)).any(-1).sum())
        projected_mean = actual
        diagnostics["mean_max_absolute_difference"] = float(error.max())
    A[indices], V[indices], output_means[indices] = (computed_A.to(means.dtype),
                                                    computed_V.to(means.dtype), projected_mean.to(means.dtype))
    return FrozenProjection(A, V, output_means, visible, diagnostics)


def prepare_shader_coefficients(projection, normal, b, w, tau, mode="integrated"):
    """Differentiable [N,4] kx,ky,h_upper,h_lower for partition_rasterizer.

    Physical slab parameters are unit normal[N,3], b[N], w[N]>0, tau[N]>0;
    scalar b/w/tau broadcast. Normalization/positive parameterization belongs
    to the caller. Inactive rows are sanitized *before* arithmetic and return
    zero coefficients/gradients. One aggregate domain check is performed, no
    per-Gaussian eigensolver or repeated reference-module GPU synchronization.
    Conditional variance excursions up to 128 dtype eps are roundoff-clamped.
    """
    if not isinstance(projection, FrozenProjection):
        raise TypeError("Require FrozenProjection")
    if mode not in MODES:
        raise ValueError("Unknown partition mode")
    N = len(projection.visible)
    ref = projection.A
    if (not isinstance(normal, torch.Tensor) or normal.shape != (N, 3)
            or normal.device != ref.device or normal.dtype != ref.dtype):
        raise ValueError("normal must be [N,3] with projection dtype/device")
    fields = []
    for value in (b, w, tau):
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
        if value.device != ref.device or value.dtype != ref.dtype or value.shape not in ((), (N,)):
            raise ValueError("b/w/tau must be scalar or [N] with projection dtype/device")
        fields.append(value.expand(N))
    b, w, tau = fields
    visible = projection.visible
    normal = torch.where(visible[:, None], normal, normal.new_tensor([1., 0., 0.]))
    b = torch.where(visible, b, 0.)
    w = torch.where(visible, w, 1.)
    tau = torch.where(visible, tau, 1.)
    eps = torch.finfo(ref.dtype).eps
    variance = torch.einsum("ni,nij,nj->n", normal, projection.V, normal)
    valid = (torch.isfinite(normal).all() & torch.isfinite(b).all() & torch.isfinite(w).all()
             & torch.isfinite(tau).all() & ((normal.square().sum(-1)-1).abs() <= 64*eps).all()
             & (w > 0).all() & (tau > 0).all() & torch.isfinite(variance).all()
             & (variance >= -128*eps).all() & (variance <= 1+128*eps).all())
    variance = variance.clamp(0, 1)
    slope = torch.einsum("ni,nij->nj", normal, projection.A)
    if mode == "point":
        variance = torch.zeros_like(variance)
    elif mode == "marginal":
        variance = torch.ones_like(variance)
        slope = torch.zeros_like(slope)
    sd = torch.sqrt(tau.square()+variance)
    coeff = torch.cat((slope/sd[:, None], ((-b+w)/sd)[:, None], ((-b-w)/sd)[:, None]), -1)
    if not bool(valid & torch.isfinite(coeff).all() & (sd > 0).all()):
        raise ValueError("Invalid active slab parameters or conditional variance")
    return torch.where(visible[:, None], coeff, 0.).contiguous()
