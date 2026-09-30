"""Fixed TRAIN-camera 3D smoothing reference; raw prune/reset remain unchanged.

Mip-Splatting's sampling rule and volume compensation, not a new method or a
complete reproduction of its renderer/optimizer. No image pixels are needed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time

import torch

POLICY = {'enabled': True, 'version': 1, 'depth_min': .2, 'frustum_margin': .15,
          'variance_factor': .2, 'refresh_every': 100,
          'sampling': 'min_train_z_over_global_native_max_fx',
          'unseen': 'max_seen_min_z', 'opacity_policy': 'unchanged_raw_prune_reset'}


def normalize_config(value=None):
    if value is None or value is False or value == {'enabled': False}:
        return None
    if not isinstance(value, dict) or value.get('enabled') is not True:
        raise ValueError('mip_filter must be absent/off or explicitly enabled')
    if set(value)-set(POLICY) or any(type(v) is not type(POLICY[k]) or v != POLICY[k] for k, v in value.items()):
        raise ValueError('Mip filter reference constants are fixed')
    return dict(POLICY)


def resolve_config(config, initial=None):
    """A missing stage option inherits its checkpoint; explicit migration rejects."""
    requested = normalize_config(config.get('mip_filter'))
    if initial is not None:
        previous = normalize_config(initial.get('mip_filter_config'))
        if 'mip_filter' in config and requested != previous:
            raise ValueError('Resume/warmstart cannot change the saved mip_filter protocol')
        requested = previous
    if requested is not None:
        config['mip_filter'] = requested
    return requested


def validate_buffer(rho, count):
    if (not isinstance(rho, torch.Tensor) or rho.shape != (count, 1)
            or rho.dtype != torch.float32 or not bool(torch.isfinite(rho).all())
            or bool((rho <= 0).any()) or rho.requires_grad):
        raise ValueError('Enabled mip filter requires a finite positive detached FP32 Nx1 buffer')


def effective_parameters(log_scales, opacity_logits, rho):
    """Differentiable scale/opacity transform; rho is held constant between refreshes."""
    if rho.shape != (len(log_scales), 1):
        raise ValueError('Mip filter buffer is stale after a topology change')
    scales = log_scales.exp()
    filtered = (scales.square()+rho.detach().square()).sqrt()
    # Product of bounded axis ratios avoids det(s^2) underflow for small splats.
    coefficient = (scales/filtered).prod(-1)
    return filtered, opacity_logits.sigmoid()*coefficient


def camera_source(views, cameras, manifest_sha256, *, required_count=350):
    """Bind original TRAIN native camera metadata, never resized minibatch K."""
    if len(views) != required_count or len({v['name'] for v in views}) != required_count:
        raise ValueError('Mip sampling requires exactly the fixed unique TRAIN camera population')
    if any(v['split'] != 'train' for v in views):
        raise ValueError('Mip sampling cannot use VAL cameras')
    original = torch.tensor([v.get('w2c_original', v['w2c']) for v in views], dtype=torch.float32)
    if cameras.shape != original.shape or not torch.equal(cameras.detach().cpu(), original):
        raise ValueError('Mip sampling requires unchanged original TRAIN poses')
    K = torch.tensor([v['K'] for v in views], dtype=torch.float32)
    sizes = torch.tensor([[v['width'], v['height']] for v in views], dtype=torch.float32)
    if (K.shape != (len(views), 3, 3) or not bool(torch.isfinite(K).all())
            or not bool(torch.isfinite(original).all()) or bool((K[:, [0, 1], [0, 1]] <= 0).any())
            or bool((sizes <= 0).any())):
        raise ValueError('Invalid native TRAIN camera metadata')
    records = [{'name': v['name'], 'K': k.tolist(), 'w2c': p.tolist(), 'size': s.tolist()}
               for v, k, p, s in zip(views, K, original, sizes)]
    encoded = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    metadata = {'count': len(views), 'camera_metadata_sha256': hashlib.sha256(encoded).hexdigest(),
                'manifest_sha256': manifest_sha256, 'names': [v['name'] for v in views],
                'resolution_source': 'TRAIN manifest native K/width/height; no progressive scaling'}
    return {'K': K.to(cameras.device), 'w2c': original.to(cameras.device),
            'sizes': sizes.to(cameras.device), 'metadata': metadata}


@torch.no_grad()
def compute_rho(means, source, *, point_chunk=131072):
    """FP32 author sampling rule; projection/frustum support is not occlusion."""
    if means.dtype != torch.float32 or means.ndim != 2 or means.shape[1] != 3 or not bool(torch.isfinite(means).all()):
        raise ValueError('Finite FP32 Gaussian centers required')
    K, poses, sizes = source['K'], source['w2c'], source['sizes']
    if any(x.device != means.device for x in (K, poses, sizes)) or point_chunk <= 0:
        raise ValueError('Camera tensors must share the field device')
    distance = torch.full((len(means),), 100000., dtype=torch.float32, device=means.device)
    seen = torch.zeros(len(means), dtype=torch.bool, device=means.device)
    for begin in range(0, len(means), point_chunk):
        xyz = means[begin:begin+point_chunk].detach()
        d = distance[begin:begin+len(xyz)]; supported = seen[begin:begin+len(xyz)]
        for k, pose, size in zip(K, poses, sizes):
            camera = xyz @ pose[:3, :3].T+pose[:3, 3]
            z = camera[:, 2]
            u = camera[:, 0]/z.clamp_min(.001)*k[0, 0]+k[0, 2]
            v = camera[:, 1]/z.clamp_min(.001)*k[1, 1]+k[1, 2]
            visible = ((z > .2) & (u >= -.15*size[0]) & (u <= 1.15*size[0])
                       & (v >= -.15*size[1]) & (v <= 1.15*size[1]))
            d.copy_(torch.where(visible, torch.minimum(d, z), d))
            supported.logical_or_(visible)
    if not bool(seen.any()):
        raise ValueError('No Gaussian has TRAIN frustum support; no arbitrary filter fallback')
    distance[~seen] = distance[seen].max()
    rho = (distance/K[:, 0, 0].max()*(.2**.5))[:, None]
    validate_buffer(rho, len(means))
    return rho, {'unseen_gaussians': int((~seen).sum()), 'gaussians': len(means),
                 'rho_min': float(rho.min()), 'rho_max': float(rho.max()),
                 'global_native_max_fx': float(K[:, 0, 0].max())}


def restore_filter_state(scene, checkpoint):
    """Reject enabled/missing or off/unexpected filter state; old missing fields are off."""
    if scene.mip_filter_config is None:
        if checkpoint.get('mip_filter_state') is not None:
            raise ValueError('Off checkpoint cannot carry enabled filter state')
        return
    state = checkpoint.get('mip_filter_state')
    if (not isinstance(state, dict) or not isinstance(state.get('source'), dict)
            or not isinstance(state.get('refresh_count'), int) or state['refresh_count'] < 1
            or not isinstance(state.get('last_refresh_step'), int) or state['last_refresh_step'] < 0):
        raise ValueError('Enabled checkpoint requires persisted filter source/refresh state')
    for key in ('total_refresh_wall_seconds', 'total_refresh_cuda_ms',
                'last_refresh_wall_seconds', 'last_refresh_cuda_ms'):
        value = state.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid saved mip filter cost metadata')
    source = state['source']
    if checkpoint.get('manifest_sha256') is not None and source.get('manifest_sha256') != checkpoint['manifest_sha256']:
        raise ValueError('Filter source manifest SHA differs from checkpoint')
    if (not isinstance(source.get('count'), int) or source['count'] < 1
            or len(source.get('names', [])) != source['count']
            or len(set(source['names'])) != source['count']
            or not isinstance(source.get('camera_metadata_sha256'), str)
            or len(source['camera_metadata_sha256']) != 64):
        raise ValueError('Invalid saved mip filter TRAIN source metadata')
    validate_buffer(scene.mip_filter_rho, len(scene.splats['means']))
    scene.mip_filter_state = copy.deepcopy(state)


def refresh_due(step, final_step, topology_changed=False):
    return topology_changed or step % POLICY['refresh_every'] == 0 or step == final_step


def validate_resume_step(scene, saved_step):
    # A frozen semantic stage inherits the RGB stage's cached filter and time
    # origin. Its local optimizer step may legitimately be much smaller.
    if (scene.mip_filter_config is not None and scene.splats['means'].requires_grad
            and scene.mip_filter_state['last_refresh_step'] > saved_step):
        raise ValueError('Saved mip filter refresh step is ahead of its mutable-geometry checkpoint')


def refresh_filter(scene, source, step):
    """One refresh per event; state is independent of optimizer/RNG and includes cost."""
    old = scene.mip_filter_state
    if old is not None and old['source'] != source['metadata']:
        raise ValueError('Mip filter TRAIN camera source changed')
    device = scene.splats['means'].device
    start_event = end_event = None
    if device.type == 'cuda':
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
    started = time.perf_counter()
    rho, statistics = compute_rho(scene.splats['means'], source)
    gpu_ms = 0.
    if end_event is not None:
        end_event.record(); end_event.synchronize()
        gpu_ms = float(start_event.elapsed_time(end_event))
    wall = time.perf_counter()-started
    scene.mip_filter_rho = rho
    scene.mip_filter_state = {'source': copy.deepcopy(source['metadata']),
        'refresh_count': (old['refresh_count'] if old else 0)+1, 'last_refresh_step': step,
        'total_refresh_wall_seconds': (old['total_refresh_wall_seconds'] if old else 0.)+wall,
        'total_refresh_cuda_ms': (old['total_refresh_cuda_ms'] if old else 0.)+gpu_ms,
        'last_refresh_wall_seconds': wall, 'last_refresh_cuda_ms': gpu_ms, **statistics}
    return filter_stats(scene)


def filter_stats(scene):
    return {'mip_filter_'+key: value for key, value in scene.mip_filter_state.items() if key != 'source'}


def bind_filter_source(scene, views, cameras, manifest_sha256, *, required_count=350):
    if scene.mip_filter_config is None:
        return None
    source = camera_source(views, cameras, manifest_sha256, required_count=required_count)
    if scene.mip_filter_state is None:
        refresh_filter(scene, source, 0)
    elif scene.mip_filter_state['source'] != source['metadata']:
        raise ValueError('Resume/warmstart mip filter source differs')
    return source
