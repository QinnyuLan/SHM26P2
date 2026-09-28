"""Camera-only direct categorical sidecar readout on an unchanged H3 context.

The caller must bind q to the exact base checkpoint and its Gaussian row order
(including background-shell rows), pixel profile, renderer source and binary.
Shape checking alone cannot establish that identity. This helper does not load
sidecars, cameras, images or labels, and does not mutate scene parameters/modes.
It is restricted to the unfiltered H3 five-class path used in the diagnostic.
"""
from __future__ import annotations

import torch

Q_ROW_SUM_ATOL = 1e-6


def validate_q(q, scene):
    """Validate the actual FP32 renderer input; never normalize or floor it.

    The tolerance allows independently cast FP64 simplex rows. q must already
    be contiguous so the same object can be supplied to both shader endpoints.
    The caller chooses whether q requires gradients; frozen-head adaptation
    should pass a detached, immutable q and train only refiner parameters.
    """
    means = scene.splats['means']
    if (not isinstance(q, torch.Tensor) or q.dtype != torch.float32
            or q.shape != (len(means), 5) or q.device != means.device
            or not q.is_contiguous()):
        raise ValueError('q must be contiguous FP32 [number_of_scene_Gaussians,5] on the scene device')
    if not bool(torch.isfinite(q).all()) or bool((q < 0).any()) or bool((q > 1).any()):
        raise ValueError('q must be finite and in [0,1]; no repair or floor is applied')
    if bool((q.detach().double().sum(-1)-1).abs().gt(Q_ROW_SUM_ATOL).any()):
        raise ValueError('q rows must sum to one within the fixed FP32 cast tolerance')


def render_direct_q(scene, q, K, w2c, width, height, *, refine=True, rasterize=None):
    """Render original evidence once, then replace only p3d and its log prior.

    `rasterize` is an optional shader injection for CPU contract tests; normal
    execution uses the same custom kernel as the passed direct-q diagnostic.
    Means/conics/opacities/tile ledger come from this very camera's original
    scene call, not a projection recomputation. [0,0,1,-1] coefficients and the
    identical q Tensor at both endpoints reproduce the frozen direct_q call.
    Residual class-0 transmittance background remains inside that rasterizer.

    Original features/RGB/depth/alpha/depth_moments are computed with no field
    gradients and passed unchanged to the head. Head execution stays in the
    caller's autograd context; its parameters may therefore be trained with q
    frozen. refine=False skips the head and uses softmax(log p3d), as H3 does.
    Only the pixel prior uses clamp_min(1e-7) and class normalization; `raw`
    remains the unclamped shader output for the affine-noise objective.
    """
    validate_q(q, scene)
    if getattr(scene, 'mip_filter_config', None) is not None:
        raise ValueError('Only the unfiltered H3 context is supported')
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError('Positive integer canvas dimensions required')
    if type(refine) is not bool:
        raise ValueError('refine must be bool')
    with torch.no_grad():
        context = scene.render(K, w2c, width, height, refine=False, absgrad=False)
    info = context['info']
    geometry = [info['means2d'][0].detach().contiguous(), info['conics'][0].detach().contiguous(),
                info['opacities'][0].detach().contiguous(), info['isect_offsets'][0].contiguous(),
                info['flatten_ids'].contiguous()]
    coefficients = q.new_tensor([0., 0., 1., -1.]).expand(len(q), -1).contiguous()
    if rasterize is None:
        from .partition_rasterizer import rasterize_semantic_partition
        rasterize = rasterize_semantic_partition
    raw, semantic_alpha = rasterize(*geometry, q, q, coefficients, width=width, height=height)
    if (raw.shape != (height, width, 5) or raw.dtype != q.dtype or raw.device != q.device
            or semantic_alpha.shape != (height, width, 1)
            or semantic_alpha.dtype != q.dtype or semantic_alpha.device != q.device):
        raise ValueError('Direct-q shader returned an incompatible camera grid')
    if not bool(torch.isfinite(raw).all()) or bool((raw < 0).any()):
        raise ValueError('Invalid raw semantic probability; no shader-output repair')
    if (not bool(torch.isfinite(semantic_alpha).all()) or bool((semantic_alpha < 0).any())
            or bool((semantic_alpha > 1).any())):
        raise ValueError('Invalid shader alpha')
    p3d = raw.clamp_min(1e-7)
    p3d = p3d/p3d.sum(-1, keepdim=True)
    kwargs = {'depth_moments': context['depth_moments']} if 'depth_moments' in context else {}
    residual = scene.refiner(context['features'], context['rgb'], context['depth'], context['alpha'],
                             p3d=p3d, **kwargs) if refine else torch.zeros_like(p3d)
    return {**context, 'original_p3d': context['p3d'], 'raw': raw, 'p3d': p3d,
            'probabilities': (p3d.log()+residual).softmax(-1), 'residual': residual,
            'refinement_prior': p3d, 'direct_q_alpha': semantic_alpha}
