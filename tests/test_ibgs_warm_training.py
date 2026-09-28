"""Small CPU contracts; also runnable beside the frozen worker/package."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE if (HERE/'bridge_rgs/ibgs_warm_training.py').exists() else HERE.parent/'src'
sys.path.insert(0, str(SNAPSHOT))
from bridge_rgs import ibgs_warm_training as warm

OFFICIAL = (HERE/'official/color_aggregation_network.py' if (HERE/'official').exists()
            else Path('/mnt/data/SHM2026/third_party/ibgs/color_aggregation_network.py'))
spec = importlib.util.spec_from_file_location('ibgs_test_official_fusion', OFFICIAL)
fusion_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion_module)


class SpyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.seen = None

    def forward(self, source, ray, own, valid=None):
        self.seen = (source, ray, own, valid)
        return (source[:, :, :3].mean(1)+ray+own)*self.weight


def package(size=16):
    base = torch.full((3, size, size), .2, requires_grad=True)
    features = torch.tensor([-2., 0., 0., 1.])[:, None, None].expand(4, size, size).clone().requires_grad_()
    return {'render': base, 'cam_feat': features,
            'warped_image': torch.full((3, size, size), .5, requires_grad=True),
            'min_depth_diff': torch.zeros(1, size, size),
            'camera_ray': torch.ones(3, size, size, requires_grad=True)}


OPTIONS = SimpleNamespace(enable_exposure_correction=False, nb_visible_src_frames=3,
                          residual_resolution_scale=1.)


def test_fixed_stages_and_strict_json():
    assert [warm.stage_for_step(s) for s in (1, 1000, 1001, 2000, 2001, 6000)] == [
        'geometry', 'geometry', 'fusion_detached', 'fusion_detached', 'fusion_joint', 'fusion_joint']
    for value in (0, 6001, True, 1.5):
        with pytest.raises(ValueError):
            warm.stage_for_step(value)
    json.dumps(warm.SPEC, allow_nan=False)


def test_saved_order_is_repeatable_complete_epochs_without_global_rng():
    names = [f'{i:03}.png' for i in range(350)]
    a = warm.camera_order(names)
    assert a == warm.camera_order(names[::-1]) and len(a) == 6000
    assert all(sorted(a[i:i+350]) == names for i in range(0, 5950, 350))
    with pytest.raises(ValueError):
        warm.camera_order(names[:-1])


def test_ablation_zeros_only_seven_network_inputs_after_support_selection():
    spy = SpyNet(); net = warm.SourceAblatedNet(spy, 'no_source')
    p = package()
    out = fusion_module.fuse_color(p, net, None, None, None, 2001, OPTIONS)
    x, ray, own, _ = spy.seen
    assert x.shape == (256, 1, 7) and torch.count_nonzero(x) == 0
    assert torch.equal(ray, p['camera_ray'].view(3, -1).T)
    assert torch.equal(own, p['render'].permute(1, 2, 0).reshape(-1, 3))
    assert out['nb_valid_warp_level'] == 1 and out['valid_warp_mask'].eq(1).all()


def test_negative_displacement_is_not_an_empty_source():
    p = package(); spy = SpyNet()
    assert p['cam_feat'].sum(0).lt(0).all()  # Counterexample to old sum>0.
    fusion_module.fuse_color(p, spy, None, None, None, 2001, OPTIONS)
    source = spy.seen[0]
    assert torch.allclose(source[..., :3], torch.full_like(source[..., :3], .3))
    assert torch.equal(source[0, 0, 3:], torch.tensor([-2., 0., 0., 1.]))


def test_training_photo_uses_same_structural_support():
    p = package()
    p['rendered_normal'] = torch.zeros_like(p['render'])
    p['median_intersected_depth_normal'] = torch.zeros_like(p['render'])
    p['median_intersected_depth'] = torch.ones_like(p['render'][:1])
    valid = torch.ones(16, 16, dtype=torch.bool)
    _, pieces = warm.training_losses(p, None, p['render'].detach(), valid, 1)
    assert pieces['active_photo_sources'] == 1 and pieces['photo'] > 0
    p['cam_feat'] = torch.zeros_like(p['cam_feat'])
    _, empty = warm.training_losses(p, None, p['render'].detach(), valid, 1)
    assert empty['active_photo_sources'] == 0 and empty['photo'] == 0


def test_zero_residual_initialization_has_unit_base_skip():
    torch.manual_seed(42)
    net = fusion_module.ColorFusionResidualNet(height=16, width=16)
    warm.initialize_fusion(net)
    p = package()
    result = warm.fuse_training(p, warm.SourceAblatedNet(net, 'full'), OPTIONS, 1001,
                                fusion_module.fuse_color)
    assert result['burned_in_gauss'] == 1.
    assert torch.equal(result['image_pred'], p['render'])
    assert torch.count_nonzero(result['residual']) == 0


def test_fusion_detachment_and_joint_gradient_routes():
    for step, expect_field_grad in ((1001, False), (2001, True)):
        p = package(); spy = SpyNet()
        result = warm.fuse_training(p, warm.SourceAblatedNet(spy, 'full'), OPTIONS,
                                    step, fusion_module.fuse_color)
        result['image_pred'].sum().backward()
        assert (p['render'].grad is not None) == expect_field_grad
        assert (p['warped_image'].grad is not None) == expect_field_grad
        assert spy.weight.grad is not None
    assert warm.fuse_training(package(), SpyNet(), OPTIONS, 1000, fusion_module.fuse_color) is None


def test_empty_source_keeps_base_field_gradient_and_no_head_path():
    p = package(); p['warped_image'] = torch.zeros_like(p['warped_image'])
    spy = SpyNet()
    result = warm.fuse_training(p, warm.SourceAblatedNet(spy, 'full'), OPTIONS,
                                1001, fusion_module.fuse_color)
    assert result['empty_support'] and result['nb_valid_warp_level'] == 0
    assert torch.equal(result['image_pred'], p['render'])
    result['image_pred'].sum().backward()
    assert p['render'].grad is not None and spy.weight.grad is None


def test_real_adam_groups_and_fresh_same_init():
    field = SimpleNamespace(**{name: torch.nn.Parameter(torch.ones(2, 3)) for name in warm.FIELD_KEYS})
    net = SpyNet()
    fo, ho = warm.build_optimizers(field, net, 5.)
    assert len(fo.param_groups) == 8 and not fo.state and not ho.state
    for group in fo.param_groups:
        name = group['name']
        assert group['lr'] == warm.SPEC['field_lrs'][name]*(5 if name in ('_xyz', '_offset') else 1)
        assert group['eps'] == 1e-15 and group['weight_decay'] == 0
        assert group['betas'] == (.9, .999)
    assert ho.param_groups[0]['eps'] == 1e-8 and ho.param_groups[0]['lr'] == .001
    assert ho.param_groups[0]['weight_decay'] == 0
