"""Teacher evidence must stay local and respect stable official track labels."""

import torch
from torch import nn
from torch.nn import functional as F

from bridge_rgs.losses import balanced_local_distillation, supervised_teacher_weights
from bridge_rgs.refinement import MultiScaleRefinementHead
from bridge_rgs.train import balanced_probability_distillation


def test_local_teacher_updates_features_without_global_classifier_gradient():
    classifier = nn.Linear(4, 3)
    features = torch.randn(6, 4, requires_grad=True)
    targets = torch.eye(3).repeat(2, 1)
    loss = balanced_local_distillation(features, classifier, targets, torch.ones(6))
    loss.backward()
    assert features.grad is not None and features.grad.norm() > 0
    assert classifier.weight.grad is None
    assert classifier.bias.grad is None


def test_stable_conflicting_annotations_reject_teacher_but_single_view_does_not():
    target = torch.tensor([[1., 0, 0, 0, 0]]).repeat(4, 1)
    counts = torch.tensor([[0., 0, 4, 0, 0], [0, 0, 1, 0, 0], [0, 0, 3, 1, 0], [0, 0, 0, 0, 0]])
    weights, conflict = supervised_teacher_weights(target, torch.ones(4), counts)
    torch.testing.assert_close(weights, torch.tensor([0., 1, 1, 1]))
    assert conflict.tolist() == [True, False, False, False]
    agrees = torch.tensor([[0., 0, 1, 0, 0]]).repeat(4, 1)
    weights, _ = supervised_teacher_weights(agrees, torch.ones(4), counts)
    torch.testing.assert_close(weights, torch.ones(4))


def test_class_balancing_removes_duplicate_majority_mass_with_high_cap():
    classifier = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        classifier.weight.copy_(torch.eye(2))
    features = torch.tensor([[2., -1.]]).repeat(11, 1).requires_grad_()
    targets = torch.tensor([[1., 0.]]).repeat(11, 1)
    targets[-1] = torch.tensor([0., 1.])
    balanced = balanced_local_distillation(features, classifier, targets, torch.ones(11), max_class_reweight=100)
    expected = -features[:1].log_softmax(-1).mean()
    torch.testing.assert_close(balanced, expected)


def test_rejected_teacher_batch_has_zero_gradient_and_finite_loss():
    classifier = nn.Linear(4, 3)
    features = torch.randn(6, 4, requires_grad=True)
    loss = balanced_local_distillation(features, classifier, torch.full((6, 3), 1 / 3), torch.zeros(6))
    loss.backward()
    assert torch.isfinite(loss) and loss == 0
    assert features.grad.abs().sum() == 0


def test_final_teacher_loss_updates_refiner_but_not_shared_classifier():
    features = torch.randn(16, 16, 16, requires_grad=True)
    classifier = nn.Linear(16, 5)
    refiner = MultiScaleRefinementHead(feature_dim=16, classes=5, channels=16)
    logits = F.linear(features, classifier.weight.detach(), classifier.bias.detach())
    p3d = logits.softmax(-1)
    residual = refiner(features, torch.rand(16, 16, 3), torch.ones(16, 16, 1),
                       torch.ones(16, 16, 1), p3d)
    probabilities = (p3d.log() + residual).softmax(-1)
    targets = torch.zeros_like(probabilities)
    targets[..., 2] = 1
    loss = balanced_probability_distillation(probabilities, targets, torch.ones(16, 16))
    loss.backward()
    assert classifier.weight.grad is None and classifier.bias.grad is None
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in refiner.parameters())
    assert features.grad is not None and features.grad.norm() > 0


def test_final_teacher_loss_has_no_effect_when_all_pixels_are_rejected():
    logits = torch.randn(8, 8, 5, requires_grad=True)
    loss = balanced_probability_distillation(logits.softmax(-1), torch.full_like(logits, .2), torch.zeros(8, 8))
    loss.backward()
    assert loss == 0 and logits.grad.abs().sum() == 0
