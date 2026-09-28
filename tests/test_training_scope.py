import numpy as np
import pytest
import torch

from bridge_rgs.model import GaussianScene
from bridge_rgs.train import apply_parameter_scope, rendering_sh_degree, validate_parameter_scope


def opacity_config(**overrides):
    return dict(parameter_scope="opacity_only", steps=1000, semantic_start=1000000,
                densification="none", freeze_rgb=True, **overrides)


def scene():
    points = np.array([[-.2, -.2, 3], [.2, -.2, 3], [-.2, .2, 3], [.2, .2, 3]], np.float32)
    return GaussianScene(points, np.full((4, 3), .8, np.float32))


def appearance_config(**overrides):
    config = {"parameter_scope": "appearance_only", "steps": 3000, "semantic_start": 1000000,
              "refine_start": 1000000, "densification": "none", "freeze_rgb": False,
              "freeze_geometry": True}
    config.update(overrides)
    return config


def test_appearance_scope_changes_only_sh_and_background_after_adam_step():
    model = scene()
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    apply_parameter_scope(model, appearance_config())
    allowed = {"splats.sh0", "splats.sh_rest", "background_logits"}
    assert {k for k, v in model.named_parameters() if v.requires_grad} == allowed
    sum(p.sum() for p in model.parameters() if p.requires_grad).backward()
    for optimizer in model.optimizers().values():
        optimizer.step()
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name]) == (name not in allowed)
    assert rendering_sh_degree(1, 3, appearance_config()) == 3
    assert rendering_sh_degree(1, 3, appearance_config(fixed_sh_degree=3)) == 3
    with pytest.raises(ValueError, match="complete existing SH"):
        rendering_sh_degree(1, 3, appearance_config(fixed_sh_degree=0))


@pytest.mark.parametrize("override", [
    {"freeze_geometry": False}, {"freeze_rgb": True}, {"optimize_cameras": True},
    {"densification": "hybrid"}, {"semantic_geometry_weight": .1}, {"pseudo_dir": "teacher"},
    {"multiview_fusion": True}, {"semantic_start": 3000}, {"refine_start": 1},
    {"warmstart_reset_refiner": True}, {"sparse_depth_weight": .05}, {"opacity_entropy_weight": .01},
])
def test_appearance_scope_rejects_conflicting_permissions(override):
    with pytest.raises(ValueError):
        validate_parameter_scope(appearance_config(**override))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat CUDA contract")
def test_real_rgb_gradients_reach_only_sh_and_background():
    model = scene().cuda()
    apply_parameter_scope(model, appearance_config())
    K = torch.tensor([[35., 0, 16], [0, 35., 16], [0, 0, 1]], device="cuda")
    result = model.render(K, torch.eye(4, device="cuda"), 32, 32, degree=3,
                          semantics=False, absgrad=False)
    result["rgb"].square().mean().backward()
    for name, parameter in model.named_parameters():
        if name in {"splats.sh0", "splats.sh_rest", "background_logits"}:
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        else:
            assert parameter.grad is None


def test_opacity_scope_preserves_all_other_parameters_after_adam_step():
    model = scene()
    original = {k: v.detach().clone() for k, v in model.named_parameters()}
    apply_parameter_scope(model, opacity_config())
    assert [k for k, v in model.named_parameters() if v.requires_grad] == ["splats.opacity_logits"]
    model.splats["opacity_logits"].sigmoid().sum().backward()
    for optimizer in model.optimizers().values():
        optimizer.step()
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original[name]) == (name != "splats.opacity_logits")
    assert rendering_sh_degree(1, 3, opacity_config()) == 3


@pytest.mark.parametrize("override", [{"freeze_geometry": True}, {"optimize_cameras": True},
                                      {"semantic_geometry_weight": .1}, {"pseudo_dir": "teacher"},
                                      {"multiview_fusion": True}])
def test_opacity_scope_rejects_conflicting_permissions(override):
    with pytest.raises(ValueError):
        validate_parameter_scope(opacity_config(**override))


def test_opacity_scope_rejects_densification_or_semantic_training():
    for key, value in (("densification", "hybrid"), ("semantic_start", 1)):
        config = opacity_config()
        config[key] = value
        with pytest.raises(ValueError):
            validate_parameter_scope(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat CUDA contract")
def test_real_rgb_and_depth_gradients_reach_only_opacity():
    model = scene().cuda()
    apply_parameter_scope(model, opacity_config())
    K = torch.tensor([[35., 0, 16], [0, 35., 16], [0, 0, 1]], device="cuda")
    result = model.render(K, torch.eye(4, device="cuda"), 32, 32, semantics=False, absgrad=False)
    (result["rgb"].square().mean() + .05 * result["depth"].mean()).backward()
    for name, parameter in model.named_parameters():
        if name == "splats.opacity_logits":
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        else:
            assert parameter.grad is None
