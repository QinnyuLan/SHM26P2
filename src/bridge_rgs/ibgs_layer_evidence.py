"""Detached, fixed-slot same-ID source evidence; no renderer or JIT side effects."""
from __future__ import annotations

import torch

POLICY = {
    'id': 'ibgs_source_top4_same_id_original_mass_v1',
    'layers': 4, 'sources': 4, 'chunk_pixels': 65536,
    'sampling': 'ideal bilinear at integer array centers; not CUDA texture 8-bit fractions',
    'support': 'sum(tap_weight * original_source_alphaT * same_ID * finite_positive_plane * eroded_valid)',
    'depth_gate': None, 'mass_threshold': None, 'retained_mass_normalization': False,
    'camera_delta': 'target_world_center minus source_world_center',
}


def _require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def _tensor(value, shape, device, kind='float'):
    _require(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(shape),
             'Tensor shape mismatch')
    dtype_ok = (value.dtype == torch.float32 if kind == 'float' else
                value.dtype == torch.bool if kind == 'bool' else
                value.dtype in (torch.int32, torch.int64))
    _require(dtype_ok and value.device == device and value.is_contiguous(),
             'Contiguous FP32/int32-or-int64/bool tensors on one device required')
    return value.detach()


def _camera_tensor(value, shape, device):
    value = _tensor(value, shape, device)
    _require(torch.isfinite(value).all(), 'Nonfinite camera metadata')
    return value


def _transform(points, matrix):
    # Explicit row-major operations mirror source projection, without promising CUDA FMA identity.
    return torch.stack([((points[..., 0]*matrix[k, 0]+points[..., 1]*matrix[k, 1])
                         +points[..., 2]*matrix[k, 2])+matrix[k, 3] for k in range(3)], -1)


def _unit(vector):
    norm = torch.sqrt((vector[..., 0].square()+vector[..., 1].square())+vector[..., 2].square())
    return vector/(norm[..., None]+1e-8)


def _project(points, matrix, focal, principal):
    source = _transform(points, matrix)
    inv_z = 1/(source[..., 2]+1e-8)
    uv = source[..., :2]*focal*inv_z[..., None]+principal
    return uv, source[..., 2]


def _taps(uv, height, width, active):
    inside = (active & torch.isfinite(uv).all(-1) & (uv[..., 0] >= 0)
              & (uv[..., 0] <= width-1) & (uv[..., 1] >= 0) & (uv[..., 1] <= height-1))
    safe = torch.where(inside[..., None], uv, torch.zeros_like(uv))
    floor = torch.floor(safe)
    fx, fy = (safe-floor).unbind(-1)
    weights = torch.stack(((1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy), -1)
    x, y = floor.to(torch.int64).unbind(-1)
    # Clamp repeats border taps, retaining their interpolation weights.
    xs = torch.stack((x, x+1, x, x+1), -1).clamp(0, width-1)
    ys = torch.stack((y, y, y+1, y+1), -1).clamp(0, height-1)
    return ys*width+xs, weights, inside


@torch.no_grad()
def build_layer_evidence(target_ids, target_depth, target_weights, base_rgb, sources, *,
                         ref_to_src, focal, principal, target_world_to_camera,
                         target_campos, source_campos, chunk_pixels=65536):
    """Build [H*W,4,4,7] features and [H*W,4,4] continuous support.

    Target IDs/depth/original alpha*T have shape [H,W,4], base_rgb [H,W,3].
    Exactly four source dictionaries contain ids/weights/depth [H,W,4], rgb
    [H,W,3], and bool valid [H,W] (already eroded by the caller). Grids and
    focal/principal are common, matching the original IBGS centered-camera port.
    All float inputs are actual FP32 values, matrices row-major. ref_to_src is
    the already computed [4,4,4] transform, not recomputed here. World centers
    and target w2c are explicit so world displacement is not confused with a
    camera-coordinate translation. No input receives gradients.

    source RGB is plain bilinear, regardless of valid/matched-layer weights;
    only support is masked by the eroded source valid. Nonfinite sampled
    evidence has zero support, not a fabricated finite color. Empty output
    features are zero, including nonfinite/padding target slots. Invalid
    target slots also have zero returned target_weights; valid mass is unchanged.
    """
    _require(isinstance(target_ids, torch.Tensor) and target_ids.ndim == 3
             and target_ids.shape[-1] == 4, 'Expected target HxWx4 slots')
    h, w, _ = target_ids.shape
    _require(h > 0 and w > 0 and isinstance(chunk_pixels, int) and chunk_pixels > 0,
             'Positive grid and chunk_pixels required')
    _require(len(sources) == 4, 'Exactly four uncompressed source slots required')
    device = target_ids.device
    ids = _tensor(target_ids, (h, w, 4), device, 'int').reshape(-1, 4)
    depth = _tensor(target_depth, (h, w, 4), device).reshape(-1, 4)
    mass = _tensor(target_weights, (h, w, 4), device).reshape(-1, 4)
    base = _tensor(base_rgb, (h, w, 3), device).reshape(-1, 3)
    _require(torch.isfinite(base).all(), 'Base RGB must be finite')
    _require((ids >= -1).all() and not ((mass < 0) & (ids >= 0)).any(), 'Invalid target ID or negative mass')
    target_live = ((ids >= 0) & torch.isfinite(depth) & (depth > 0)
                   & torch.isfinite(mass) & (mass > 0))
    mass = torch.where(target_live, mass, torch.zeros_like(mass))
    _require((mass.sum(-1) <= 1+1e-5).all(), 'Target original contribution mass exceeds one')
    transforms = _camera_tensor(ref_to_src, (4, 4, 4), device)
    focal = _camera_tensor(focal, (2,), device)
    principal = _camera_tensor(principal, (2,), device)
    _require((focal > 0).all(), 'Positive focal lengths required')
    world_to_camera = _camera_tensor(target_world_to_camera, (4, 4), device)
    target_center = _camera_tensor(target_campos, (3,), device)
    source_centers = _camera_tensor(source_campos, (4, 3), device)
    caches = []
    for source in sources:
        si = _tensor(source['ids'], (h, w, 4), device, 'int').reshape(-1, 4)
        sw = _tensor(source['weights'], (h, w, 4), device).reshape(-1, 4)
        sz = _tensor(source['depth'], (h, w, 4), device).reshape(-1, 4)
        srgb = _tensor(source['rgb'], (h, w, 3), device).reshape(-1, 3)
        valid = _tensor(source['valid'], (h, w), device, 'bool').reshape(-1)
        _require((si >= -1).all() and not ((sw < 0) & (si >= 0)).any(),
                 'Invalid source ID or negative original mass')
        live = (si >= 0) & torch.isfinite(sw) & (sw > 0) & torch.isfinite(sz) & (sz > 0)
        clean = torch.where(live, sw, torch.zeros_like(sw))
        _require((clean.sum(-1) <= 1+1e-5).all() and (clean <= 1).all(),
                 'Source original contribution mass exceeds one')
        for j in range(4):
            for k in range(j):
                _require(not ((si[:, j] == si[:, k]) & live[:, j] & live[:, k]).any(),
                         'Repeated live source ID would duplicate contribution mass')
        caches.append((si, clean, srgb, valid))
    n = h*w
    features = torch.zeros((n, 4, 4, 7), dtype=torch.float32, device=device)
    support = torch.zeros((n, 4, 4), dtype=torch.float32, device=device)
    inv_focal = 1/focal
    delta = target_center[None]-source_centers
    for start in range(0, n, chunk_pixels):
        end = min(start+chunk_pixels, n)
        index = torch.arange(start, end, device=device)
        xy = torch.stack((index % w, index // w), -1).to(torch.float32)
        live = target_live[start:end]
        z = torch.where(live, depth[start:end], torch.zeros_like(depth[start:end]))
        pc = torch.cat((((xy[:, None]-principal)*z[..., None])*inv_focal, z[..., None]), -1)
        shifted = pc-world_to_camera[:3, 3]
        # Original kernel uses R^T*(p_camera-t), not a newly inverted/orthogonalized R.
        pw = ((shifted[..., 0, None]*world_to_camera[0, :3]
               +shifted[..., 1, None]*world_to_camera[1, :3])
              +shifted[..., 2, None]*world_to_camera[2, :3])
        target_ray = _unit(pw-target_center)
        for source_index, (si, sw, srgb, valid) in enumerate(caches):
            uv, source_z = _project(pc, transforms[source_index], focal, principal)
            taps, beta, inside = _taps(uv, h, w, live & torch.isfinite(source_z) & (source_z > 0))
            matched = si[taps] == ids[start:end, :, None, None]
            tap_mass = torch.where(matched, sw[taps], torch.zeros_like(sw[taps])).sum(-1)
            tap_mass = torch.where(valid[taps], tap_mass, torch.zeros_like(tap_mass))
            q = (beta*tap_mass).sum(-1)
            colors = srgb[taps]
            colors = torch.where(beta[..., None] > 0, colors, torch.zeros_like(colors))
            color = (colors*beta[..., None]).sum(-2)
            source_ray = _unit(pw-source_centers[source_index])
            cosine = ((source_ray[..., 0]*target_ray[..., 0]+source_ray[..., 1]*target_ray[..., 1])
                      +source_ray[..., 2]*target_ray[..., 2])
            value = torch.cat((color-base[start:end, None],
                               delta[source_index].expand(end-start, 4, 3), cosine[..., None]), -1)
            active = inside & torch.isfinite(value).all(-1) & torch.isfinite(q) & (q > 0)
            support[start:end, :, source_index] = torch.where(active, q, torch.zeros_like(q))
            features[start:end, :, source_index] = torch.where(active[..., None], value, torch.zeros_like(value))
    return {'features': features, 'support': support, 'target_weights': mass,
            'policy': POLICY['id'], 'chunk_pixels': chunk_pixels}
