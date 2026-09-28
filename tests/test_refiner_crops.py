import copy

import numpy as np
import pytest
import torch

from bridge_rgs.losses import semantic_loss
from bridge_rgs.refinement import MultiScaleRefinementHead
from bridge_rgs.refiner_crops import (
    choose_refiner_crop,
    crop_slices,
    refine_render_crop,
    validate_refiner_crop_config,
)


def crop_config(**overrides):
    return dict(refiner_crop_probability=0.5, refiner_crop_size=768,
                freeze_geometry=True, freeze_rgb=True, train_labeled_only=True,
                independent_view_rng=True, **overrides)


def test_stateless_crop_preserves_global_rng_and_coordinates():
    np.random.seed(42)
    before = copy.deepcopy(np.random.get_state())
    torch_before = torch.get_rng_state().clone()
    values = [choose_refiner_crop(989, 1320, 768, 0.5, 42, step) for step in range(1, 101)]
    assert any(v is None for v in values) and any(v is not None for v in values)
    assert values == [choose_refiner_crop(989, 1320, 768, 0.5, 42, step) for step in range(1, 101)]
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())
    for value in values:
        if value is not None:
            y, x, h, w = value
            assert h == w == 768 and 0 <= y <= 221 and 0 <= x <= 552
    assert choose_refiner_crop(989, 1320, 768, 0, 42, 1) is None
    with pytest.raises(ValueError, match="exceeds"):
        choose_refiner_crop(32, 48, 64, 1, 42, 1)


@pytest.mark.parametrize("key,value", [
    ("pseudo_dir", "pseudo"), ("multiview_fusion", True), ("pseudo_weight", .1),
    ("pseudo_refiner_weight", .1), ("multiview_weight", .05),
    ("semantic_geometry_weight", .01), ("sparse_depth_weight", .05),
    ("opacity_entropy_weight", .01), ("region_rgb_weight", .1),
    ("optimize_cameras", True), ("densification", "hybrid"),
    ("parameter_scope", "opacity_only"),
])
def test_crop_rejects_unaligned_auxiliary_objectives(key, value):
    with pytest.raises(ValueError):
        validate_refiner_crop_config(crop_config(**{key: value}))


@pytest.mark.parametrize("key", ["freeze_geometry", "freeze_rgb", "train_labeled_only", "independent_view_rng"])
def test_crop_requires_gt_only_frozen_scene_and_separate_view_rng(key):
    config = crop_config()
    config[key] = False
    with pytest.raises(ValueError):
        validate_refiner_crop_config(config)


def test_aligned_crop_final_gradients_and_full_raw_supervision():
    torch.set_num_threads(1)
    torch.manual_seed(18)
    h, w = 64, 80
    features = torch.randn(h, w, 8, requires_grad=True)
    rgb = torch.rand(h, w, 3, requires_grad=True)
    depth = torch.ones(h, w, 1, requires_grad=True)
    alpha = torch.ones(h, w, 1, requires_grad=True)
    logits = torch.randn(h, w, 5, requires_grad=True)
    p3d = logits.softmax(-1)
    rendered = {"features": features, "rgb": rgb, "depth": depth, "alpha": alpha,
                "p3d": p3d, "refinement_prior": p3d}
    head = MultiScaleRefinementHead(8, channels=16)
    with torch.no_grad():
        head.out.weight.normal_(std=.03)
    mask = (torch.arange(h * w).reshape(h, w) % 5).long()
    mask[10, 20] = 255
    valid = torch.ones(h, w)
    valid[11, 21] = 0
    crop = (8, 16, 32, 32)
    ys, xs = crop_slices(crop, h, w)
    original_mask = mask.clone()
    output = refine_render_crop(head, rendered, crop)
    output["probabilities"].retain_grad()
    assert output["probabilities"].shape == (32, 32, 5)
    assert set(mask[ys, xs].unique().tolist()) == {0, 1, 2, 3, 4, 255}
    semantic_loss(output["probabilities"], mask[ys, xs], valid[ys, xs]).backward(retain_graph=True)
    assert features.grad[ys, xs].abs().sum() > 0
    outside = torch.ones(h, w, dtype=torch.bool)
    outside[ys, xs] = False
    assert features.grad[outside].count_nonzero() == 0
    assert logits.grad[outside].count_nonzero() == 0
    # Ignored output pixels receive no direct loss. Context inputs at those
    # positions can still influence valid neighbours through convolutions.
    assert output["probabilities"].grad[2, 4].count_nonzero() == 0
    assert output["probabilities"].grad[3, 5].count_nonzero() == 0
    assert rgb.grad is None and depth.grad is None and alpha.grad is None
    assert head.out.weight.grad.abs().sum() > 0
    logits.grad = None
    semantic_loss(rendered["p3d"], mask, valid, lovasz_weight=0).backward()
    assert logits.grad[outside].abs().sum() > 0
    assert logits.grad[10, 20].count_nonzero() == 0
    assert logits.grad[11, 21].count_nonzero() == 0
    assert torch.equal(mask, original_mask)
    assert rendered["p3d"].shape == (h, w, 5)
    with pytest.raises(ValueError, match="outside"):
        crop_slices((40, 0, 32, 32), h, w)
