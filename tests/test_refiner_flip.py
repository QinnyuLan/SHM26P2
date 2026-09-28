import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from bridge_rgs.losses import rgb_loss, semantic_loss
from bridge_rgs.model import GaussianScene
from bridge_rgs.refinement import MultiScaleRefinementHead
from bridge_rgs.refiner_flip import (
    PROBABILITY_KEY,
    choose_refiner_horizontal_flip,
    refine_horizontal_flip,
    validate_refiner_flip_config,
    validate_refiner_flip_resume,
)
from bridge_rgs.train import apply_parameter_scope, initialize_view_sampler, sample_training_view


def config(**overrides):
    values = {PROBABILITY_KEY: .5, "parameter_scope": "refiner_only", "freeze_geometry": True,
              "freeze_rgb": True, "train_labeled_only": True, "independent_view_rng": True,
              "seed": 42, "steps": 100, "semantic_start": 1, "refine_start": 1,
              "densification": "none", "multiview_fusion": False}
    return dict(values, **overrides)


def evidence(h=32, w=48, feature_dim=8):
    values = {key: torch.randn(h, w, channels, requires_grad=True) for key, channels in
              (("features", feature_dim), ("rgb", 3), ("depth", 1), ("alpha", 1),
               ("depth_moments", feature_dim+1))}
    values["p3d"] = torch.randn(h, w, 5).softmax(-1).requires_grad_()
    values["refinement_prior"] = values["p3d"]
    mask = torch.arange(h*w).reshape(h, w) % 5
    mask[1, 2] = 255
    valid = torch.ones(h, w)
    valid[3, 4] = 0
    return values, mask, valid


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), float("inf"), True, "0.5", None])
def test_invalid_flip_probability_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        validate_refiner_flip_config(config(**{PROBABILITY_KEY: value}))


@pytest.mark.parametrize("override", [
    {"parameter_scope": "all"}, {"freeze_geometry": False}, {"freeze_rgb": False},
    {"train_labeled_only": False}, {"independent_view_rng": False}, {"pseudo_dir": "teacher"},
    {"multiview_fusion": True}, {"refiner_crop_probability": .5}, {"pseudo_weight": .1},
    {"pseudo_refiner_weight": .1}, {"multiview_weight": .1}, {"semantic_geometry_weight": .1},
    {"sparse_depth_weight": .05}, {"sparse_front_weight": .05}, {"opacity_entropy_weight": .01},
    {"optimize_cameras": True}, {"densification": "hybrid"}, {"seed": -1},
])
def test_flip_rejects_nonisolated_training(override):
    with pytest.raises(ValueError):
        validate_refiner_flip_config(config(**override))


def test_zero_is_legacy_compatible_and_allocates_no_rng(monkeypatch):
    assert validate_refiner_flip_config({}) == 0.
    assert validate_refiner_flip_config({PROBABILITY_KEY: 0., "parameter_scope": "all"}) == 0.
    monkeypatch.setattr(np.random, "default_rng", lambda *args: pytest.fail("Off must not allocate RNG"))
    assert not choose_refiner_horizontal_flip(0., 42, 17)
    validate_refiner_flip_resume({}, {PROBABILITY_KEY: 0.})


def test_stateless_flip_preserves_all_rng_and_actual_view_sampler():
    np.random.seed(87)
    before_np = copy.deepcopy(np.random.get_state())
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    cfg = config()
    a = {"sampler_order": [], "sampler_cursor": 0, "sampler_rng_state": None}
    b = copy.deepcopy(a)
    ga, gb = initialize_view_sampler(cfg, a), initialize_view_sampler(cfg, b)
    choices, flips = [], []
    for step in range(1, 401):
        left = sample_training_view(list(range(259)), "shuffle", a, ga)
        flips.append(choose_refiner_horizontal_flip(.5, 42, step))
        right = sample_training_view(list(range(259)), "shuffle", b, gb)
        assert left == right
        choices.append(left)
    assert a == b
    assert 150 < sum(flips) < 250  # Fixed stream sanity, not a quality-selected rate.
    assert flips[201:] == [choose_refiner_horizontal_flip(.5, 42, s) for s in range(202, 401)]
    assert sorted(choices[:259]) == list(range(259))
    after_np = np.random.get_state()
    assert before_np[0] == after_np[0] and np.array_equal(before_np[1], after_np[1])
    assert before_np[2:] == after_np[2:]
    assert torch.equal(before_torch, torch.get_rng_state())
    assert before_python == random.getstate()


def test_strict_resume_cannot_change_probability_or_active_seed():
    validate_refiner_flip_resume(config(), config())
    for changed in (config(**{PROBABILITY_KEY: 0.}), config(**{PROBABILITY_KEY: 1.}), config(seed=43)):
        with pytest.raises(ValueError, match="Strict resume"):
            validate_refiner_flip_resume(config(), changed)
    with pytest.raises(ValueError, match="Strict resume"):
        validate_refiner_flip_resume({}, config())
    validate_refiner_flip_resume({}, {"seed": 123})


class CaptureHead(nn.Module):
    def forward(self, features, rgb, depth, alpha, p3d, depth_moments=None):
        self.received = (features, rgb, depth, alpha, p3d, depth_moments)
        return torch.zeros_like(p3d)


def test_full_hwc_and_moment_context_flip_aligned_without_mutation():
    rendered, mask, valid = evidence()
    original = {k: v.clone() for k, v in rendered.items()}
    original_mask, original_valid = mask.clone(), valid.clone()
    head = CaptureHead()
    result, flipped_mask, flipped_valid = refine_horizontal_flip(head, rendered, mask, valid)
    for actual, key in zip(head.received, ("features", "rgb", "depth", "alpha", "refinement_prior", "depth_moments"), strict=True):
        assert torch.equal(actual, rendered[key].flip(1))
        assert not actual.requires_grad
    assert torch.equal(flipped_mask, mask.flip(1)) and torch.equal(flipped_valid, valid.flip(1))
    assert int(flipped_mask[1, 45]) == 255 and float(flipped_valid[3, 43]) == 0
    torch.testing.assert_close(result["probabilities"], rendered["p3d"].flip(1))
    assert all(torch.equal(v, original[k]) for k, v in rendered.items())
    assert torch.equal(mask, original_mask) and torch.equal(valid, original_valid)


@pytest.mark.parametrize("mode", ["off", "zero", "variance", "cross"])
def test_final_loss_gradients_only_refiner_and_rgb_raw_losses_unchanged(mode):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    rendered, mask, valid = evidence()
    if mode == "off":
        rendered.pop("depth_moments")
    head = MultiScaleRefinementHead(8, channels=16, depth_moments=mode)
    head.out.weight.data.normal_(std=.03)
    target_rgb = torch.rand_like(rendered["rgb"])
    rgb_before = rgb_loss(rendered["rgb"], target_rgb, valid).detach()
    raw_before = semantic_loss(rendered["p3d"], mask, valid, lovasz_weight=0).detach()
    result, final_mask, final_valid = refine_horizontal_flip(head, rendered, mask, valid)
    result["probabilities"].retain_grad()
    loss = semantic_loss(result["probabilities"], final_mask, final_valid)
    loss.backward()
    assert head.out.weight.grad.abs().sum() > 0
    assert all(v.grad is None for v in rendered.values())
    assert result["probabilities"].grad[1, 45].count_nonzero() == 0
    assert result["probabilities"].grad[3, 43].count_nonzero() == 0
    assert torch.equal(rgb_before, rgb_loss(rendered["rgb"], target_rgb, valid).detach())
    assert torch.equal(raw_before, semantic_loss(rendered["p3d"], mask, valid, lovasz_weight=0).detach())


def test_off_forward_is_bitwise_original_head_and_rng():
    torch.set_num_threads(1)
    rendered, _, _ = evidence()
    head = MultiScaleRefinementHead(8, channels=16, depth_moments="cross")
    head.out.weight.data.normal_(std=.03)
    def forward():
        return head(rendered["features"], rendered["rgb"], rendered["depth"], rendered["alpha"],
                    p3d=rendered["p3d"], depth_moments=rendered["depth_moments"])
    before = forward()
    rng = torch.get_rng_state().clone()
    assert not choose_refiner_horizontal_flip(0., 42, 1)
    after = forward()
    assert torch.equal(before, after)
    assert torch.equal(rng, torch.get_rng_state())


def test_two_cpu_updates_leave_all_field_parameters_bitwise_frozen():
    torch.set_num_threads(1)
    points = np.array([[-.2,-.2,3], [.2,-.2,3], [-.2,.2,3], [.2,.2,3]], np.float32)
    scene = GaussianScene(points, np.full((4,3),.5,np.float32), feature_dim=8,
                          refiner_config={"type":"multiscale", "channels":16, "depth_moments":"cross"})
    apply_parameter_scope(scene, config())
    before = {k:v.clone() for k,v in scene.state_dict().items()}
    optimizers = scene.optimizers()
    rendered, mask, valid = evidence()
    for _ in range(2):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        result, fm, fv = refine_horizontal_flip(scene.refiner, rendered, mask, valid)
        (semantic_loss(result["probabilities"],fm,fv)+.001*result["residual"].square().mean()).backward()
        for optimizer in optimizers.values():
            optimizer.step()
    changed = [k for k,v in scene.state_dict().items() if not torch.equal(v,before[k])]
    assert changed and all(k.startswith("refiner.") for k in changed)
    for name, parameter in scene.named_parameters():
        if not name.startswith("refiner."):
            assert parameter.grad is None
            assert all(parameter not in opt.state for opt in optimizers.values())


def test_misaligned_or_missing_gt_rejected():
    rendered, mask, valid = evidence()
    with pytest.raises(ValueError, match="labeled"):
        refine_horizontal_flip(CaptureHead(), rendered, None, valid)
    with pytest.raises(ValueError, match="HW grid"):
        refine_horizontal_flip(CaptureHead(), rendered, mask[:,1:], valid)
    rendered["depth_moments"] = rendered["depth_moments"][:,1:]
    with pytest.raises(ValueError, match="moment"):
        refine_horizontal_flip(CaptureHead(), rendered, mask, valid)
