"""Experimental fixed-density categorical partition on an existing scene.

This is a research candidate. Its Gaussian integration has prior art; measured
benefit and distinct scientific contribution require matched training ablations.
It does not change RGB, opacity, geometry, feature maps or depth moments.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SemanticPartitionField(nn.Module):
    """Two class endpoints and one local soft slab per frozen Gaussian.

The endpoints start equal to the original classifier. The first update can
separate them; slab derivatives initially vanish, avoiding an initial change of
the categorical field. All arms have identical initialization and parameters.
The slab center is bounded to +/-2 local standard deviations; width > .05,
softness is fixed at .3. These are experimental settings, not tuned outcomes.
"""

    def __init__(self, original_logits, mode='integrated'):
        super().__init__()
        if mode not in ('integrated', 'point', 'marginal'):
            raise ValueError('Unknown partition mode')
        if (original_logits.ndim != 2 or original_logits.shape[1] != 5
                or not original_logits.is_floating_point()
                or not bool(torch.isfinite(original_logits).all())):
            raise ValueError('Require finite N by 5 original logits')
        self.mode = mode
        self.inside_logits = nn.Parameter(original_logits.detach().clone())
        self.outside_logits = nn.Parameter(original_logits.detach().clone())
        n = len(original_logits)
        direction = original_logits.new_zeros(n, 3)
        direction[:, 0] = 1
        self.direction = nn.Parameter(direction)
        self.offset_raw = nn.Parameter(original_logits.new_zeros(n))
        self.width_raw = nn.Parameter(original_logits.new_full((n,), math.log(math.expm1(.75-.05))))

    def slab_parameters(self):
        normal = F.normalize(self.direction, dim=-1, eps=1e-8)
        offset = 2*self.offset_raw.tanh()
        width = .05+F.softplus(self.width_raw)
        tau = torch.full_like(width, .3)
        return normal, offset, width, tau

    def endpoints(self):
        return self.inside_logits.softmax(-1), self.outside_logits.softmax(-1)

    def optimizer_groups(self):
        return [{'params': [self.inside_logits, self.outside_logits], 'lr': .01},
                {'params': [self.direction, self.offset_raw, self.width_raw], 'lr': .001}]


def render_partition(scene, field, K, w2c, width, height, *, refine=True):
    """Camera-only render; RGB and contextual features are frozen computations.

The context pass still uses the original field. Only its semantic prior is
replaced for the refinement head; the final loss can differentiate through the
new prior and its use in that head. Caller controls refiner parameter freezing.
No masks, images, view identifiers or teacher outputs enter this function.
"""
    from .partition_projection import prepare_projection, prepare_shader_coefficients
    from .partition_rasterizer import rasterize_semantic_partition

    if scene.mip_filter_config is not None:
        raise ValueError('This experimental adapter currently supports the unfiltered H3 scene')
    with torch.no_grad():
        context = scene.render(K, w2c, width, height, refine=False, absgrad=False)
        info = context['info']
        projection = prepare_projection(
            scene.splats['means'].detach(), scene.splats['quats'].detach(), scene.splats['log_scales'].detach(),
            w2c, K, width, height, radii=info['radii'][0],
            means2d=info['means2d'][0], conics=info['conics'][0])
    normal, offset, half_width, softness = field.slab_parameters()
    coefficients = prepare_shader_coefficients(projection, normal, offset, half_width,
                                               softness, mode=field.mode)
    qi, qo = field.endpoints()
    raw, alpha = rasterize_semantic_partition(
        info['means2d'][0].detach().contiguous(), info['conics'][0].detach().contiguous(),
        info['opacities'][0].detach().contiguous(), info['isect_offsets'][0].contiguous(),
        info['flatten_ids'].contiguous(), qi.contiguous(), qo.contiguous(), coefficients.contiguous(),
        width=width, height=height)
    p3d = raw.clamp_min(1e-7)
    p3d = p3d/p3d.sum(-1, keepdim=True)
    kwargs = {'depth_moments': context['depth_moments']} if 'depth_moments' in context else {}
    residual = scene.refiner(context['features'], context['rgb'], context['depth'], context['alpha'],
                             p3d=p3d, **kwargs) if refine else torch.zeros_like(p3d)
    return {**context, 'original_p3d': context['p3d'], 'p3d': p3d,
            'probabilities': (p3d.log()+residual).softmax(-1), 'residual': residual,
            'refinement_prior': p3d,
            'partition_alpha': alpha, 'partition_projection_diagnostics': projection.diagnostics}
