"""Exact pinhole inverse-depth reprojection, independent of a Gaussian renderer.

This is standard projective geometry, not a novel rendering method. It permits
future TRAIN-only alignment diagnostics to compare one shared ray-depth change
with independent two-dimensional view offsets. Nothing imports it in training.
"""
from __future__ import annotations

import torch


def project_inverse_depth(rays, inverse_depth, ref_to_source, focal, principal):
    """Project N reference rays and L depths into S sources.

    rays: [N,3], with z=1; inverse_depth: [N,L]; ref_to_source:
    [S,4,4], row-major source-from-reference; focal/principal: [S,2].
    All inputs use the same FP32 or FP64 dtype/device. Principal points are in
    integer array-center coordinates; no implicit half-pixel correction occurs.

    Returns uv [N,L,S,2], source_z/valid [N,L,S], and the analytic derivative
    duv_drho [N,L,S,2]. Invalid positive-depth projections are masked to zero.
    The output is differentiable through valid depths, including the quotient.
    """
    if not all(isinstance(x, torch.Tensor) for x in
               (rays, inverse_depth, ref_to_source, focal, principal)):
        raise ValueError('Tensor inputs required')
    if rays.ndim != 2 or rays.shape[-1] != 3 or inverse_depth.ndim != 2:
        raise ValueError('Expected rays [N,3] and inverse_depth [N,L]')
    n, layers = inverse_depth.shape
    if ref_to_source.ndim != 3 or ref_to_source.shape[-2:] != (4, 4):
        raise ValueError('Expected source-from-reference transforms [S,4,4]')
    sources = ref_to_source.shape[0]
    if n < 1 or layers < 1 or sources < 1 or rays.shape[0] != n:
        raise ValueError('Positive, matching dimensions required')
    if focal.shape != (sources, 2) or principal.shape != (sources, 2):
        raise ValueError('Expected source calibration [S,2]')
    if rays.dtype not in (torch.float32, torch.float64) or any(
        x.dtype != rays.dtype or x.device != rays.device
        for x in (inverse_depth, ref_to_source, focal, principal)
    ):
        raise ValueError('One FP32/FP64 dtype and device required')
    if not all(bool(torch.isfinite(x).all()) for x in (rays, ref_to_source, focal, principal)):
        raise ValueError('Finite rays and calibration required')
    if not bool((rays[:, 2] == 1).all()) or not bool((focal > 0).all()):
        raise ValueError('Ray z=1 and positive focal lengths required')
    if not bool((ref_to_source[:, 3] == rays.new_tensor([0, 0, 0, 1])).all()):
        raise ValueError('Affine source-from-reference matrices required')
    rho_live = torch.isfinite(inverse_depth) & (inverse_depth > 0)
    rho = torch.where(rho_live, inverse_depth, torch.ones_like(inverse_depth))
    rotated = torch.einsum('sij,nj->nsi', ref_to_source[:, :3, :3], rays)
    translation = ref_to_source[:, :3, 3]
    scaled_point = rotated[:, None] + rho[..., None, None]*translation
    denominator = scaled_point[..., 2]
    live = rho_live[..., None] & torch.isfinite(scaled_point).all(-1) & (denominator > 0)
    safe_denom = torch.where(live, denominator, torch.ones_like(denominator))
    uv = focal*scaled_point[..., :2]/safe_denom[..., None] + principal
    numerator = (translation[:, :2]*rotated[..., 2, None]
                 - rotated[..., :2]*translation[:, 2, None])
    derivative = focal*numerator[:, None]/safe_denom[..., None].square()
    source_z = denominator/rho[..., None]
    live = live & torch.isfinite(uv).all(-1) & torch.isfinite(derivative).all(-1) & torch.isfinite(source_z)
    return {
        'uv': torch.where(live[..., None], uv, torch.zeros_like(uv)),
        'source_z': torch.where(live, source_z, torch.zeros_like(source_z)),
        'valid': live,
        'duv_drho': torch.where(live[..., None], derivative, torch.zeros_like(derivative)),
    }


def shared_depth_interval(rays, inverse_depth, ref_to_source, focal, principal, *,
                          radius_pixels=2.0, relative_limit=0.25):
    """Bound a shared inverse-depth change in every initially valid source.

    The two return columns are signed lower and upper changes, not depths.
    The exact rational projection displacement is constrained by radius_pixels,
    rho stays within +/-relative_limit, and each valid source denominator stays
    at least half its initial value. Initially invalid sources cannot become
    evidence: use the returned center_valid mask also when projecting offsets.
    An unobserved target receives the degenerate [0,0] interval.
    """
    if not (0 < radius_pixels < float('inf') and 0 < relative_limit < 1):
        raise ValueError('Finite positive radius and relative limit in (0,1) required')
    center = project_inverse_depth(rays, inverse_depth, ref_to_source, focal, principal)
    valid = center['valid']
    rho = torch.where(torch.isfinite(inverse_depth) & (inverse_depth > 0), inverse_depth,
                      torch.zeros_like(inverse_depth))
    rotated = torch.einsum('sij,nj->nsi', ref_to_source[:, :3, :3], rays)
    translation = ref_to_source[:, :3, 3]
    denominator = rotated[:, None, :, 2] + rho[..., None]*translation[:, 2]
    denominator = torch.where(valid, denominator, torch.ones_like(denominator))
    numer = focal*(translation[:, :2]*rotated[..., 2, None]
                   - rotated[..., :2]*translation[:, 2, None])
    magnitude = torch.linalg.vector_norm(numer, dim=-1)[:, None]
    relative = relative_limit*rho
    limits = []
    for sign in (-1, 1):
        # |delta| M <= radius D0 (D0 + sign |delta| tz).
        coefficient = magnitude-sign*radius_pixels*denominator*translation[:, 2]
        safe = torch.where(coefficient > 0, coefficient, torch.ones_like(coefficient))
        pixel_limit = radius_pixels*denominator.square()/safe
        pixel_limit = torch.where(coefficient > 0, pixel_limit, torch.full_like(pixel_limit, float('inf')))
        decrease = sign*translation[:, 2] < 0
        z_safe = torch.where(decrease, translation[:, 2].abs(), torch.ones_like(translation[:, 2]))
        depth_limit = torch.where(decrease, 0.5*denominator/z_safe,
                                  torch.full_like(denominator, float('inf')))
        limit = torch.minimum(pixel_limit, depth_limit)
        limit = torch.where(valid, limit, torch.full_like(limit, float('inf'))).amin(-1)
        limit = torch.minimum(relative, limit)
        limits.append(sign*torch.where(valid.any(-1), limit, torch.zeros_like(limit)))
    return {'delta_interval': torch.stack(limits, -1), 'center_valid': valid}
