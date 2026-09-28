"""Independent symmetric Gaussian OT adapter, not used by current training.

This is standard SPD Gaussian covariance transport, not a new mathematical
operator. For 2x2 C0, Cr, the SPD optimal-transport map is

    T = (Cr + sqrt(det(C0)*det(Cr))*solve(C0,I)) /
        sqrt(trace(C0*Cr) + 2*sqrt(det(C0)*det(Cr))).

It equals C0^-1/2 (C0^1/2 Cr C0^1/2)^1/2 C0^-1/2. The 2x2 identity avoids
the condition-squared intermediate eigensystem. Determinants and the positive
trace are obtained from Cholesky factors, with a common scalar normalization.
No eigenspectrum is clipped. This alternative is intentionally separate from
partition_projection.actual_footprint_whitened_transport_v2.

The operator is equivariant under orthogonal changes of screen coordinates.
This does not claim exact physical perspective integration, a rotation-invariant
full rasterizer, or measured performance improvement. Geometry is fixed and
the output stays FP64; any eventual shader cast must be audited separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

REVISION = "symmetric_gaussian_ot_2x2_v1"


def _fixed_tensor(value, tail, name):
    if (not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.float64)
            or value.ndim < len(tail) or value.shape[-len(tail):] != tail):
        raise ValueError(f"{name} must be floating (...,{','.join(map(str, tail))})")
    if value.requires_grad:
        raise ValueError(f"{name} geometry must be explicitly frozen")
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be nonempty and finite")


def _symmetric(value, name):
    # Only symmetrize arithmetic roundoff. A genuinely asymmetric input fails.
    magnitude = value.abs().amax((-2, -1), keepdim=True)
    tolerance = 64*torch.finfo(value.dtype).eps*magnitude
    if bool(((value-value.mT).abs() > tolerance).any()):
        raise ValueError(f"{name} must be symmetric")
    value = value.double()
    return (value+value.mT)*.5


def _cholesky(value, name):
    chol, info = torch.linalg.cholesky_ex(value)
    if bool((info != 0).any()) or not bool(torch.isfinite(chol).all()):
        raise ValueError(f"{name} must be numerically SPD; no eigenvalue clipping")
    return chol


@torch.no_grad()
def symmetric_ot_2x2(source_covariance, target_covariance):
    """Return FP64 SPD T with T C0 T.T = Cr, for matching (...,2,2) inputs.

    Common device and identical batch shapes are required; FP32/64 inputs may
    differ in dtype and are promoted before arithmetic. No geometry gradients,
    explicit matrix inverse, eigenvalue truncation, or caller-state mutation.
    A single positive scalar per pair normalizes both covariances; T is
    dimensionless and unchanged by this normalization.
    """
    _fixed_tensor(source_covariance, (2, 2), "source_covariance")
    _fixed_tensor(target_covariance, (2, 2), "target_covariance")
    if source_covariance.shape != target_covariance.shape or source_covariance.device != target_covariance.device:
        raise ValueError("Covariances must have identical shape and device")
    source = _symmetric(source_covariance, "source_covariance")
    target = _symmetric(target_covariance, "target_covariance")
    scale = torch.maximum(source.abs().amax((-2, -1)), target.abs().amax((-2, -1)))
    if not bool((scale > 0).all()):
        raise ValueError("Covariance scale must be positive")
    source, target = source/scale[..., None, None], target/scale[..., None, None]
    L0, Lr = _cholesky(source, "source_covariance"), _cholesky(target, "target_covariance")
    s = L0.diagonal(dim1=-2, dim2=-1).prod(-1)*Lr.diagonal(dim1=-2, dim2=-1).prod(-1)
    trace = (Lr.mT@L0).square().sum((-2, -1))
    identity = torch.eye(2, dtype=torch.float64, device=source.device).expand_as(source)
    solved = torch.cholesky_solve(identity, L0)
    T = (target+s[..., None, None]*solved)/torch.sqrt(trace+2*s)[..., None, None]
    T = (T+T.mT)*.5
    _cholesky(T, "transport")
    return T


@dataclass(frozen=True)
class TransportedConditional:
    A: torch.Tensor
    V: torch.Tensor
    transport: torch.Tensor
    effective_projection: torch.Tensor
    effective_noise: torch.Tensor
    source_covariance: torch.Tensor
    actual_covariance: torch.Tensor
    diagnostics: dict


@torch.no_grad()
def prepare_transport(P0, actual_covariance, *, noise_variance=.3):
    """Fixed (...,2,3) local projection + (...,2,2) footprint to FP64 A/V.

    xi~N(0,I), ideal noise=noise_variance*I. The target model is P'=T P0,
    noise'=noise_variance*T T.T. Its conditioning map is A=A0 T^-1 and its
    conditional covariance remains V0. V0 is evaluated in Joseph form.
    This helper neither accepts means nor changes any rasterization weights.
    Metadata same-render identity remains the caller's explicit responsibility.
    """
    _fixed_tensor(P0, (2, 3), "P0")
    _fixed_tensor(actual_covariance, (2, 2), "actual_covariance")
    if (P0.shape[:-2] != actual_covariance.shape[:-2] or P0.device != actual_covariance.device
            or isinstance(noise_variance, bool) or not isinstance(noise_variance, (int, float))
            or not 0 < noise_variance < float("inf")):
        raise ValueError("Require matching batch/device and finite positive noise_variance")
    P = P0.double()
    Cr = _symmetric(actual_covariance, "actual_covariance")
    eye2 = torch.eye(2, dtype=torch.float64, device=P.device).expand_as(Cr)
    eye3 = torch.eye(3, dtype=torch.float64, device=P.device)
    C0 = P@P.mT+noise_variance*eye2
    L0, Lr = _cholesky(C0, "source_covariance"), _cholesky(Cr, "actual_covariance")
    T = symmetric_ot_2x2(C0, Cr)
    B = torch.linalg.solve_triangular(L0, P, upper=False)
    A0 = torch.linalg.solve_triangular(L0.mT, B, upper=True).mT
    residual_map = eye3-B.mT@B
    V = residual_map@residual_map.mT+noise_variance*(A0@A0.mT)
    V = (V+V.mT)*.5
    A = torch.cholesky_solve(A0.mT, _cholesky(T, "transport")).mT
    effective_projection = T@P
    effective_noise = noise_variance*(T@T.mT)
    _cholesky(effective_noise, "effective_noise")
    reconstruction = effective_projection@effective_projection.mT+effective_noise
    error = reconstruction-Cr
    left = torch.linalg.solve_triangular(Lr, error, upper=False)
    whitened_error = torch.linalg.solve_triangular(Lr, left.mT, upper=False).mT
    total_error = A@Cr@A.mT+V-eye3
    values = (A, V, effective_projection, effective_noise, whitened_error, total_error)
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("Nonfinite transported conditional geometry")
    diagnostics = {
        "revision": REVISION, "compute_dtype": "float64", "output_dtype": "float64",
        "ideal_noise_variance": float(noise_variance),
        "effective_noise": "noise_variance*T*T_transpose; not generally isotropic",
        "eigenvalue_clipping": False,
        "footprint_relative_max_residual": float((error.abs().amax((-2, -1))/Cr.abs().amax((-2, -1))).max()),
        "footprint_whitened_max_residual": float(whitened_error.abs().max()),
        "law_total_covariance_max_residual": float(total_error.abs().max()),
        "joseph_vs_subtraction_max_residual": float((V-residual_map).abs().max()),
        "transport_max_change_from_identity": float((T-eye2).abs().max()),
        "effective_noise_max_change_from_ideal": float((effective_noise-noise_variance*eye2).abs().max()),
    }
    return TransportedConditional(A, V, T, effective_projection, effective_noise, C0, Cr, diagnostics)
