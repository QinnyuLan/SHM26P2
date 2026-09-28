"""Fixed warm IBGS engineering reference; no changes to the external backend.

The source ablation acts only on the network's seven source-feature channels.
All source selection, texture rendering, support and photometric losses remain.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

ARMS = ('full', 'no_source')
FIELD_KEYS = ('_xyz', '_normal', '_offset', '_features_dc', '_features_rest',
              '_opacity', '_scaling', '_rotation')
SPEC = {
    'protocol': 'ibgs_warm_matched_v1', 'seed': 42, 'steps_per_arm': 6000,
    'geometry_steps': 1000, 'fusion_detached_steps': 1000, 'fusion_joint_steps': 4000,
    'gaussians': 996009, 'sh_degree': 3, 'train_cameras': 350,
    'native_resolution': True, 'renderer_sources_max': 4, 'visible_source_slots_max': 3,
    'buffer_length': 4, 'depth_error_threshold': .01,
    'base_skip': 1., 'zero_final_residual_initialization': True,
    'rgb_l1_weight': .8, 'rgb_dssim_weight': .2, 'normal_weight': .03,
    'photo_weight': .3, 'photo_dssim_weight': 1.,
    'field_eps': 1e-15, 'head_eps': 1e-8, 'adam_betas': [.9, .999],
    'weight_decay': 0., 'head_lr': .001,
    'field_lrs': {'_xyz': 1.6e-6, '_normal': .001, '_offset': 8e-7,
                  '_features_dc': .00025, '_features_rest': .0000125,
                  '_opacity': .0025, '_scaling': .0005, '_rotation': .0001},
    'scene_scaled_lr_groups': ['_xyz', '_offset'],
    'source_depth_initializations_per_arm': 350,
    'depth_update': 'target pre-update render detached, masked and stored after optimizer step',
    'camera_order': 'sorted names, NumPy default_rng(42) epoch permutations; saved once',
    'log_every': 200, 'internal_seconds_per_arm': 1800, 'external_seconds': 3900,
    'VAL_reads': 0, 'semantic_label_decodes': 0, 'densification': False,
    'adaptive_clipping': False, 'selection': 'fixed last; no validation or best checkpoint',
    'empty_source': 'base RGB + zero residual; field step retained; skip head Adam and count',
}


def stage_for_step(step):
    if type(step) is not int or not 1 <= step <= 6000:
        raise ValueError('Step must be an integer in 1..6000')
    return 'geometry' if step <= 1000 else ('fusion_detached' if step <= 2000 else 'fusion_joint')


def camera_order(names):
    names = sorted(names)
    if len(names) != 350 or len(set(names)) != 350:
        raise ValueError('Exactly 350 unique TRAIN camera names required')
    rng = np.random.default_rng(42)
    indices = []
    while len(indices) < 6000:
        indices.extend(rng.permutation(350).tolist())
    return [names[i] for i in indices[:6000]]


def load_warm_field(checkpoint, *, device='cuda'):
    """Return the official field/background, exactly as the passed preflight.

    Accept a checkpoint path or an already CPU-loaded mapping. The author's
    smallest-axis routine requires CUDA; this factory is an execution API,
    never called during CPU preparation.
    """
    from scene.gaussian_model import GaussianModel
    saved = checkpoint if isinstance(checkpoint, Mapping) else torch.load(
        checkpoint, map_location='cpu', weights_only=False)
    if saved['pixel_protocol']['id'] != 'colmap_corner_v2' or saved['sh_degree'] != 3:
        raise ValueError('Fixed corner-v2 SH3 checkpoint required')
    state = saved['model']
    if state['splats.means'].shape != (996009, 3):
        raise ValueError('Fixed 996009-point warm field required')
    field = GaussianModel(3)
    mapping = {'_xyz': 'means', '_rotation': 'quats', '_scaling': 'log_scales',
               '_opacity': 'opacity_logits', '_features_dc': 'sh0', '_features_rest': 'sh_rest'}
    for destination, key in mapping.items():
        value = state['splats.'+key].detach().to(device).clone()
        if destination == '_opacity':
            value = value[:, None]
        setattr(field, destination, torch.nn.Parameter(value))
    field.active_sh_degree = 3
    field.spatial_lr_scale = float(saved['scene_scale'])
    field._normal = torch.nn.Parameter(field.get_smallest_axis().detach().clone())
    field._offset = torch.nn.Parameter(torch.zeros(len(field._xyz), 1, device=device))
    background = state['background_logits'].sigmoid().to(device).detach().clone()
    return field, background


def load_saved_field(saved, *, device='cuda'):
    """Restore a final field without reading any image or preparing sources."""
    from scene.gaussian_model import GaussianModel
    if saved['protocol'] != SPEC['protocol'] or saved['sh_degree'] != 3:
        raise ValueError('Wrong endpoint protocol')
    field = GaussianModel(3)
    for name in FIELD_KEYS:
        setattr(field, name, torch.nn.Parameter(saved['field'][name].to(device).clone()))
    field.active_sh_degree = 3
    field.spatial_lr_scale = float(saved['scene_scale'])
    return field, saved['background'].to(device).detach().clone()


class SourceAblatedNet(torch.nn.Module):
    def __init__(self, net, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown arm')
        self.inner, self.arm = net, arm

    def forward(self, x_views, ray_dir, c_3dgs, valid_depth_mask=None):
        if x_views.ndim != 3 or x_views.shape[-1] != 7:
            raise ValueError('Expected seven source-feature channels')
        if self.arm == 'no_source':
            x_views = torch.zeros_like(x_views)
        return self.inner(x_views, ray_dir, c_3dgs, valid_depth_mask)


def initialize_fusion(net):
    layer = net.conv_decoder.final
    if not isinstance(layer, torch.nn.Conv2d) or layer.out_channels != 3:
        raise ValueError('Expected author final RGB residual convolution')
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()


def fuse_training(package, net, options, step, fuse_fn):
    stage = stage_for_step(step)
    if stage == 'geometry':
        return None
    selected = ({k: v.detach() if isinstance(v, torch.Tensor) else v
                 for k, v in package.items()} if stage == 'fusion_detached' else package)
    # All three burn arguments None enforce the author's exact unit base skip.
    # Detachment is controlled separately; upstream .5-base burn is never used.
    result = fuse_fn(selected, net, None, None, None, step, options)
    if result is None:
        # A legal unsupported view retains its actual base field path, even
        # during fusion burn-in; no invented source or head optimizer step.
        return {'image_pred': package['render'], 'residual': torch.zeros_like(package['render']),
                'valid_warp_mask': torch.zeros_like(package['render'][0]),
                'burned_in_gauss': 1., 'nb_valid_warp_level': 0, 'empty_support': True}
    if result['burned_in_gauss'] != 1.:
        raise RuntimeError('The warm reference requires a unit base skip')
    result['empty_support'] = False
    return result


def build_optimizers(field, net, scene_scale):
    if not np.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError('Positive finite scene scale required')
    groups = []
    for name in FIELD_KEYS:
        parameter = getattr(field, name)
        lr = SPEC['field_lrs'][name] * (scene_scale if name in SPEC['scene_scaled_lr_groups'] else 1.)
        groups.append({'params': [parameter], 'name': name, 'lr': lr})
    field_optimizer = torch.optim.Adam(groups, lr=0., eps=1e-15, betas=(.9, .999), weight_decay=0.)
    head_optimizer = torch.optim.Adam(net.parameters(), lr=.001, eps=1e-8,
                                      betas=(.9, .999), weight_decay=0.)
    return field_optimizer, head_optimizer


def training_losses(package, fusion, target, valid, step):
    from .ibgs_losses import erode_valid, masked_normal_loss, masked_photometric_loss
    base, _ = masked_photometric_loss(package['render'], target, valid)
    normal = masked_normal_loss(package['rendered_normal'], package['median_intersected_depth_normal'],
                                package['median_intersected_depth'][0], valid)
    h, w = target.shape[-2:]
    warped = package['warped_image'].reshape(-1, 3, h, w)[:3]
    features = package['cam_feat'].reshape(-1, 4, h, w)[:3]
    photo = base*0
    active_sources = 0
    for source_rgb, source_features in zip(warped, features, strict=True):
        active = (source_features.abs().sum(0) > 0) & valid
        if erode_valid(active, 5).any():
            photo = photo + masked_photometric_loss(source_rgb, target, active, ssim_weight=1.)[0]
            active_sources += 1
    photo = photo / max(active_sources, 1)
    fused = None
    if stage_for_step(step) == 'geometry':
        color = base
    else:
        if fusion is None:
            raise ValueError('Fusion is mandatory after step 1000')
        fused, _ = masked_photometric_loss(fusion['image_pred'], target, valid)
        color = .5*(base+fused)
    loss = color+.03*normal+.3*photo
    return loss, {'base_rgb': base, 'fused_rgb': fused, 'normal': normal,
                  'photo': photo, 'active_photo_sources': active_sources}
