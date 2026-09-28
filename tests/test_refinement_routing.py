"""Refiner/explicit-field gradient separation through the real gsplat renderer."""

import numpy as np
import pytest
import torch

from bridge_rgs.model import GaussianScene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")


@pytest.fixture(params=[{"type": "legacy"}, {"type": "multiscale", "channels": 16}])
def scene_and_camera(request):
    torch.manual_seed(190)
    points = np.array([
        [-.25, -.25, 3.], [.25, -.25, 3.], [-.25, .25, 3.], [.25, .25, 3.2],
    ], dtype=np.float32)
    scene = GaussianScene(points, np.full_like(points, .5), feature_dim=8,
                          sh_degree=1, refiner_config=request.param).cuda()
    # Exercise feature and p3d input paths beyond the zero-output initialization.
    with torch.no_grad():
        scene.refiner.out.weight.normal_(std=.05)
        scene.refiner.out.bias.normal_(std=.02)
    K = torch.tensor([[35., 0, 16], [0, 35., 16], [0, 0, 1]], device="cuda")
    pose = torch.eye(4, device="cuda", requires_grad=True)
    return scene, K, pose


def loss_for_class(probabilities, category=2):
    return -probabilities[8:24, 8:24, category].log().mean()


def test_route_switch_and_default_are_bitwise_forward_equivalent(scene_and_camera):
    scene, K, pose = scene_and_camera
    implicit = scene.render(K, pose, 32, 32, absgrad=False)
    coupled = scene.render(K, pose, 32, 32, absgrad=False, refinement_grad_to_field=True)
    isolated = scene.render(K, pose, 32, 32, absgrad=False, refinement_grad_to_field=False)
    for key in ("rgb", "depth", "alpha", "p3d", "residual", "probabilities"):
        torch.testing.assert_close(coupled[key], implicit[key], atol=0, rtol=0)
        torch.testing.assert_close(isolated[key], implicit[key], atol=0, rtol=0)
    assert isolated["p3d"].requires_grad
    assert isolated["probabilities"].requires_grad


def test_isolated_final_loss_only_updates_refiner_then_raw_loss_updates_field(scene_and_camera):
    scene, K, pose = scene_and_camera
    # Isolation also wins when geometry is explicitly enabled for the raw loss.
    rendered = scene.render(K, pose, 32, 32, absgrad=False, geometry_grad=True,
                            refinement_grad_to_field=False)
    before = {name: parameter.detach().clone() for name, parameter in scene.named_parameters()}
    loss_for_class(rendered["probabilities"]).backward()
    for name, parameter in scene.named_parameters():
        if name.startswith("refiner."):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None, name
    assert scene.refiner.out.weight.grad.abs().sum() > 0
    assert pose.grad is None
    optimizer = torch.optim.SGD(scene.parameters(), lr=.01)
    optimizer.step()
    assert not torch.equal(scene.refiner.out.weight, before["refiner.out.weight"])
    for name, parameter in scene.named_parameters():
        if not name.startswith("refiner."):
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)

    scene.zero_grad(set_to_none=True)
    # No second render or retain_graph: the untouched raw graph remains usable.
    loss_for_class(rendered["p3d"]).backward()
    assert scene.splats["sem_features"].grad.abs().sum() > 0
    assert scene.semantic_decoder.weight.grad.abs().sum() > 0
    assert scene.semantic_decoder.bias.grad.abs().sum() > 0
    assert scene.splats["means"].grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in scene.refiner.parameters())
    assert scene.splats["sh0"].grad is None
    assert scene.splats["sh_rest"].grad is None
    assert scene.background_logits.grad is None
    assert pose.grad is None


def test_default_final_loss_still_trains_native_semantics(scene_and_camera):
    scene, K, pose = scene_and_camera
    rendered = scene.render(K, pose, 32, 32, absgrad=False)
    loss_for_class(rendered["probabilities"]).backward()
    assert scene.splats["sem_features"].grad.abs().sum() > 0
    assert scene.semantic_decoder.weight.grad.abs().sum() > 0
    assert scene.semantic_decoder.bias.grad.abs().sum() > 0
    assert scene.refiner.out.weight.grad.abs().sum() > 0
    assert scene.splats["means"].grad is None
    assert scene.splats["sh0"].grad is None
    assert pose.grad is None


def test_raw_classifier_permission_remains_independent(scene_and_camera):
    scene, K, pose = scene_and_camera
    rendered = scene.render(K, pose, 32, 32, absgrad=False,
                            semantic_classifier_grad=False, refinement_grad_to_field=False)
    loss_for_class(rendered["p3d"]).backward()
    assert scene.splats["sem_features"].grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in scene.semantic_decoder.parameters())
    assert all(parameter.grad is None for parameter in scene.refiner.parameters())


def test_disabled_refinement_keeps_raw_loss_differentiable(scene_and_camera):
    scene, K, pose = scene_and_camera
    rendered = scene.render(K, pose, 32, 32, absgrad=False, refine=False,
                            refinement_grad_to_field=False)
    assert not rendered["probabilities"].requires_grad
    assert rendered["p3d"].requires_grad
    torch.testing.assert_close(rendered["probabilities"], rendered["p3d"])
    loss_for_class(rendered["p3d"]).backward()
    assert scene.splats["sem_features"].grad.abs().sum() > 0
    assert scene.semantic_decoder.weight.grad.abs().sum() > 0
