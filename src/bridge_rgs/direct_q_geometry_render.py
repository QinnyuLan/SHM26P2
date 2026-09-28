"""Standard-gsplat, means-only derivative of the frozen direct-q H3 readout.

This is a separate training/probe adapter, not a replacement for direct_q_render.
The old custom semantic shader has no geometry backward. Here RGB and packed
q/features/depth moments use standard gsplat with live means. The two rasterizers'
FP32 semantic outputs are not assumed bitwise equal: a real-camera preflight
must compare raw, p3d, head output, RGB and alpha before using this adapter.
There is no straight-through substitution of custom-shader forward values.
"""
from __future__ import annotations

import torch

from .depth_moments import semantic_depth_context
from .direct_q_render import validate_q
from .refinement import MultiScaleRefinementHead


def render_direct_q_geometry(scene, q, K, w2c, width, height, *,
                             refine=True, geometry_grad=True, rasterize=None):
    """Render a frozen H3 field/head with means as its only trainable parameter.

    geometry_grad=False is a forward-identical restricted-gradient control:
    it restores the head's auxiliary/moment stop-gradients and detached depth
    attributes; footprint derivatives still reach q and feature mixtures.
    True additionally exposes SH/RGB, expected depth, alpha, live z/z²/zf and
    cross-moment statistics to the frozen head. Projection/culling/depth-sort
    and support predicates remain piecewise/non-smooth; this is not a numerical
    certificate of the installed gsplat means VJP.

    The caller freezes all other scene parameters, q and cameras. No mode,
    requires_grad flag, parameter, gradient buffer or global function is changed.
    rasterize injection is for CPU synthetic contracts only.
    """
    if type(refine) is not bool or type(geometry_grad) is not bool:
        raise TypeError('refine and geometry_grad must be bool')
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError('Positive integer canvas dimensions required')
    validate_q(q, scene)
    s = scene.splats
    means = s['means']
    if (getattr(scene, 'mip_filter_config', None) is not None
            or not isinstance(scene.refiner, MultiScaleRefinementHead)
            or scene.refiner.depth_moments_mode != 'cross'
            or scene.sh_degree != 3):
        raise ValueError('Only unfiltered SH3 H3 with multiscale cross moments is supported')
    if q.requires_grad:
        raise ValueError('q must be frozen')
    if any(p.requires_grad for p in scene.parameters() if p is not means):
        raise ValueError('All scene parameters except means must be frozen')
    if (not isinstance(K, torch.Tensor) or not isinstance(w2c, torch.Tensor)
            or K.shape != (3, 3) or w2c.shape != (4, 4)
            or any(v.requires_grad or v.dtype != means.dtype or v.device != means.device
                   for v in (K, w2c))):
        raise ValueError('Frozen, aligned K[3,3] and w2c[4,4] required')
    if not bool(torch.isfinite(K).all()) or not bool(torch.isfinite(w2c).all()):
        raise ValueError('Camera tensors must be finite')
    if rasterize is None:
        from .model import configure_cuda
        configure_cuda()
        from gsplat import rasterization as rasterize

    common = {'means': means, 'quats': s['quats'], 'scales': s['log_scales'].exp(),
              'opacities': s['opacity_logits'].sigmoid(),
              'viewmats': w2c[None], 'Ks': K[None], 'width': width, 'height': height,
              'packed': False, 'rasterize_mode': 'antialiased', 'near_plane': .01,
              'far_plane': 1e6, 'absgrad': False}
    rgbd, alpha, info = rasterize(
        colors=torch.cat((s['sh0'], s['sh_rest']), 1), sh_degree=3,
        render_mode='RGB+ED', backgrounds=scene.background_logits.sigmoid()[None], **common)
    features = s['sem_features']
    depth_means = means if geometry_grad else means.detach()
    z = (depth_means @ w2c[2, :3] + w2c[2, 3])[:, None] / max(scene.scene_scale, 1e-6)
    values = torch.cat((q, features, z, z.square(), z * features), -1)
    bg = values.new_zeros(1, values.shape[-1])
    bg[0, 0] = 1
    semantic, semantic_alpha, semantic_info = rasterize(colors=values, backgrounds=bg, **common)
    raw = semantic[0, ..., :5]
    p3d = raw.clamp_min(1e-7)
    p3d = p3d / p3d.sum(-1, keepdim=True)
    rendered_features = semantic[0, ..., 5:5+scene.feature_dim]
    start = 5 + scene.feature_dim
    moments = semantic_depth_context(
        rendered_features, semantic[0, ..., start:start+1],
        semantic[0, ..., start+1:start+2], semantic[0, ..., start+2:],
        semantic_alpha[0], geometry_grad=geometry_grad)
    rgb, depth = rgbd[0, ..., :3], rgbd[0, ..., 3:4]
    residual = scene.refiner(rendered_features, rgb, depth, alpha[0], p3d=p3d,
                             depth_moments=moments, geometry_grad=geometry_grad) \
        if refine else torch.zeros_like(p3d)
    return {'rgb': rgb, 'depth': depth, 'alpha': alpha[0], 'info': info,
            'semantic_info': semantic_info, 'raw': raw, 'p3d': p3d,
            'features': rendered_features, 'depth_moments': moments,
            'probabilities': (p3d.log()+residual).softmax(-1), 'residual': residual,
            'refinement_prior': p3d, 'direct_q_alpha': semantic_alpha[0]}
