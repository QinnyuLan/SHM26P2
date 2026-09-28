import numpy as np
import pytest
import torch

from bridge_rgs.losses import iou_scores, semantic_loss
from bridge_rgs.model import GaussianScene


def test_iou_uses_dataset_confusion_and_excludes_absent_classes():
    matrix = torch.tensor([[8, 1, 0], [2, 4, 0], [0, 0, 0]])
    result = iou_scores(matrix)
    assert result["miou_foreground"] == pytest.approx(4 / 7)
    assert result["iou"][2] is None


def test_ignored_labels_never_train_semantics():
    logits = torch.randn(12, 12, 5, requires_grad=True)
    labels = torch.full((12, 12), 255, dtype=torch.long)
    value = semantic_loss(logits.softmax(-1), labels, torch.ones(12, 12))
    value.backward()
    assert value.item() == 0
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")
def test_semantics_cannot_move_geometry_without_explicit_gate():
    generator = np.random.default_rng(21)
    points = generator.normal(size=(30, 3)).astype(np.float32) * 0.3
    points[:, 2] += 3
    scene = GaussianScene(points, np.full((30, 3), 0.4)).cuda()
    K = torch.tensor([[35.0, 0, 16], [0, 35.0, 16], [0, 0, 1]], device="cuda")
    pose = torch.eye(4, device="cuda", requires_grad=True)
    rendered = scene.render(K, pose, 32, 32, geometry_grad=False)
    loss = -rendered["probabilities"][..., 2].clamp_min(1e-6).log().mean()
    loss.backward()
    assert scene.splats["means"].grad is None
    assert scene.splats["sh0"].grad is None
    assert pose.grad is None
    assert scene.splats["sem_features"].grad.norm() > 0
    assert torch.allclose(rendered["p3d"].sum(-1), torch.ones(32, 32, device="cuda"), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")
def test_rgb_has_finite_position_and_pose_gradients():
    points = np.array(
        [[-0.2, -0.2, 3], [0.2, -0.2, 3], [-0.2, 0.2, 3], [0.2, 0.2, 3]], dtype=np.float32
    )
    scene = GaussianScene(points, np.full((4, 3), 0.8)).cuda()
    K = torch.tensor([[35.0, 0, 16], [0, 35.0, 16], [0, 0, 1]], device="cuda")
    pose = torch.eye(4, device="cuda", requires_grad=True)
    result = scene.render(K, pose, 32, 32, semantics=False)
    result["rgb"].square().mean().backward()
    assert torch.isfinite(scene.splats["means"].grad).all()
    assert torch.isfinite(pose.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA NLL layout regression")
def test_semantic_loss_hwc_layout_has_finite_cuda_backward():
    logits = torch.randn(33, 45, 5, device="cuda", requires_grad=True)
    target = torch.randint(0, 5, (33, 45), device="cuda")
    target[:2] = 255
    loss = semantic_loss(logits.softmax(-1), target, torch.ones_like(target).float())
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[2:].abs().sum() > 0
    assert logits.grad[:2].abs().sum() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat semantic geometry gate")
def test_explicit_semantic_geometry_permission_updates_scene_but_not_camera():
    points = np.array(
        [[-0.2, -0.2, 3], [0.2, -0.2, 3], [-0.2, 0.2, 3], [0.2, 0.2, 3]], dtype=np.float32
    )
    scene = GaussianScene(points, np.full((4, 3), 0.8)).cuda()
    K = torch.tensor([[35.0, 0, 16], [0, 35.0, 16], [0, 0, 1]], device="cuda")
    pose = torch.eye(4, device="cuda", requires_grad=True)
    rendered = scene.render(K, pose, 32, 32, geometry_grad=True, refine=False)
    labels = torch.ones(32, 32, device="cuda", dtype=torch.long)
    valid = torch.zeros(32, 32, device="cuda")
    valid[10:22, 10:22] = 1
    semantic_loss(rendered["p3d"], labels, valid, lovasz_weight=0).backward()
    assert torch.isfinite(scene.splats["means"].grad).all()
    assert scene.splats["means"].grad.abs().sum() > 0
    assert pose.grad is None
    assert scene.splats["sh0"].grad is None
