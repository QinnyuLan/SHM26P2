"""Explicitly disabled teacher losses must not advance native-field Adam state."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from bridge_rgs.train import (
    multiview_evidence_enabled,
    optional_multiview_loss,
    optional_raw_pseudo_loss,
)


def test_disabled_losses_preserve_field_and_existing_adam_momentum():
    torch.manual_seed(13)
    field = nn.Parameter(torch.randn(6, 5))
    refiner = nn.Parameter(torch.randn(6, 5))
    optimizer = torch.optim.Adam([field, refiner], lr=.01)
    (field.square().mean() + refiner.square().mean()).backward()
    optimizer.step()  # Establish nonzero native-field moments before disabling KD.
    optimizer.zero_grad(set_to_none=True)
    before = field.detach().clone()
    state = deepcopy(optimizer.state[field])

    # Invalid targets/indices prove disabled paths do not execute their math/gather.
    raw = optional_raw_pseudo_loss(field.softmax(-1), None, None, 0)
    fused = optional_multiview_loss(field, None, torch.tensor([999]), None, None, 0)
    assert raw is None and fused is None
    refiner.square().mean().backward()
    assert field.grad is None
    optimizer.step()
    torch.testing.assert_close(field, before, atol=0, rtol=0)
    for name, value in state.items():
        torch.testing.assert_close(optimizer.state[field][name], value, atol=0, rtol=0)

    # Reproduce the old zero-times-KL bug to show why grad=None matters.
    optimizer.zero_grad(set_to_none=True)
    (0 * field.log_softmax(-1).mean() + refiner.square().mean()).backward()
    assert field.grad is not None and field.grad.count_nonzero() == 0
    optimizer.step()
    assert not torch.equal(field, before)


@pytest.mark.parametrize("local", [True, False])
def test_positive_losses_retain_native_feature_and_classifier_permissions(local):
    field = nn.Parameter(torch.randn(7, 4))
    classifier = nn.Linear(4, 5)
    indices = torch.tensor([0, 2, 4])
    targets = torch.zeros(3, 5)
    targets[:, 2] = 1
    loss = optional_multiview_loss(field, classifier, indices, targets, torch.ones(3), .05, local)
    loss.backward()
    assert field.grad[indices].abs().sum() > 0
    assert field.grad[[1, 3, 5, 6]].count_nonzero() == 0
    assert (classifier.weight.grad is None) == local
    if not local:
        assert classifier.weight.grad.abs().sum() > 0

    logits = nn.Parameter(torch.randn(3, 5))
    loss = optional_raw_pseudo_loss(logits.softmax(-1), targets, torch.ones(3), .1)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().sum() > 0


def test_zero_multiview_weight_disables_fusion_but_retains_explicit_geometry_gate():
    config = {"pseudo_dir": "pseudo", "multiview_fusion": True, "multiview_weight": 0}
    assert not multiview_evidence_enabled(config)
    assert multiview_evidence_enabled({**config, "multiview_weight": .05})
    assert multiview_evidence_enabled({**config, "semantic_geometry_weight": .03})
    assert not multiview_evidence_enabled({**config, "pseudo_dir": None,
                                          "semantic_geometry_weight": .03})
    assert not multiview_evidence_enabled({**config, "multiview_fusion": False,
                                          "semantic_geometry_weight": .03})
    assert multiview_evidence_enabled({"pseudo_dir": "pseudo"})  # Preserve old default .05.
