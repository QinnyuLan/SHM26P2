"""CPU contracts for valid-domain matching, sampling and full auxiliary loss."""
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, SwinConfig
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentationOutput,
)

from bridge_rgs.mask2former_valid import (
    ValidSemanticTarget,
    ValidSupportMask2FormerCriterion,
    sample_mask_logits,
    sample_valid_pixels,
    semantic_targets,
    uncertainty_points,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(aux=False):
    return Mask2FormerConfig(num_labels=3, train_num_points=31, oversample_ratio=3,
                            importance_sample_ratio=.75, use_auxiliary_loss=aux, decoder_layers=3)


def target_fixture():
    labels = torch.zeros(8, 8, dtype=torch.long)
    labels[2:6, 2:6] = 1
    labels[4, 4] = 255  # ignore even though initial validity is true
    valid = torch.ones(8, 8, dtype=torch.bool)
    valid[6:, :] = False  # e.g. input padding
    valid[:, 6:] = False
    valid[0, 0] = False
    return semantic_targets([labels], [valid], 3)


def output_fixture(aux=False, low=False):
    generator = torch.Generator().manual_seed(15)
    shape = (1, 4, 4, 4) if low else (1, 4, 8, 8)
    masks = torch.randn(shape, generator=generator).requires_grad_()
    classes = torch.randn((1, 4, 4), generator=generator).requires_grad_()
    extra = [{"masks_queries_logits": torch.randn(shape, generator=generator).requires_grad_(),
              "class_queries_logits": torch.randn((1, 4, 4), generator=generator).requires_grad_()}
             for _ in range(2)] if aux else None
    return Mask2FormerForUniversalSegmentationOutput(
        masks_queries_logits=masks, class_queries_logits=classes, auxiliary_logits=extra)


def leaves(output):
    return [output.masks_queries_logits, output.class_queries_logits,
            *(v for head in output.auxiliary_logits or [] for v in head.values())]


def evaluate(criterion, output, targets):
    result = criterion(output, targets, generator=torch.Generator().manual_seed(9))
    gradients = torch.autograd.grad(result.loss, leaves(output))
    return result, gradients


def matching_equal(left, right):
    return all(torch.equal(a, b) and torch.equal(c, d)
               for (a, c), (b, d) in zip(left, right, strict=True))


def test_target_classes_use_only_effective_validity_and_keep_background():
    targets = target_fixture()
    target = targets[0]
    assert target.class_labels.tolist() == [0, 1]
    assert not target.support[4, 4]
    assert not target.support[6:].any()
    labels = torch.zeros(8, 8, dtype=torch.long)
    labels[target.mask_labels[1]] = 1
    labels[~target.support] = 999
    changed = semantic_targets([labels], [target.support], 3)[0]
    assert torch.equal(target.class_labels, changed.class_labels)
    assert torch.equal(target.mask_labels, changed.mask_labels)
    assert torch.equal(target.support, changed.support)


@pytest.mark.parametrize("aux", [False, True])
def test_ignored_target_and_unsupported_logits_do_not_change_match_loss_or_grad(aux):
    targets = target_fixture()
    output = output_fixture(aux)
    criterion = ValidSupportMask2FormerCriterion(config(aux))
    baseline, original_grads = evaluate(criterion, output, targets)
    support = targets[0].support
    # Eight-by-eight centers are exactly representable. Prediction and GT grids
    # here coincide, so these are genuinely outside ALL valid bilinear taps.
    altered = output_fixture(aux)
    with torch.no_grad():
        altered.masks_queries_logits[:, :, ~support] = 1e4
        for head in altered.auxiliary_logits or []:
            head['masks_queries_logits'][:, :, ~support] = -1e4
    binary = targets[0].mask_labels.float()
    binary[:, ~support] = float('nan')  # never read, even for matcher/aux heads
    alternative = [replace(targets[0], mask_labels=binary)]
    result, changed_grads = evaluate(criterion, altered, alternative)
    assert matching_equal(baseline.indices, result.indices)
    assert len(result.auxiliary_indices) == (2 if aux else 0)
    for left, right in zip(baseline.auxiliary_indices, result.auxiliary_indices, strict=True):
        assert matching_equal(left, right)
    assert torch.equal(baseline.loss, result.loss)
    for name in result.loss_dict:
        assert torch.equal(baseline.loss_dict[name], result.loss_dict[name])
    for first, second in zip(original_grads, changed_grads, strict=True):
        assert torch.equal(first, second)
        if first.ndim == 4:
            assert torch.count_nonzero(first[:, :, ~support]) == 0
    assert original_grads[0].abs().sum() > 0
    if aux:
        assert original_grads[2].abs().sum() > 0 and original_grads[4].abs().sum() > 0


def test_coarse_logits_obey_effective_tap_support_not_nearest_target_validity():
    # Single valid GT point (2,2) on 8x8 reaches four 4x4 prediction taps.
    support = torch.zeros(8, 8, dtype=torch.bool)
    support[2, 2] = True
    labels = torch.zeros(8, 8, dtype=torch.long)
    targets = semantic_targets([labels], [support], 3)
    point = torch.tensor([[18]])
    grid = torch.zeros(1, 4, 4, requires_grad=True)
    sampled = sample_mask_logits(grid, point, (8, 8))
    taps = torch.autograd.grad(sampled.sum(), grid)[0][0]
    expected = torch.zeros(4, 4)
    expected[:2, :2] = torch.tensor([[.0625, .1875], [.1875, .5625]])
    torch.testing.assert_close(taps, expected, rtol=0, atol=0)
    criterion = ValidSupportMask2FormerCriterion(config())
    original, gradients = evaluate(criterion, output_fixture(low=True), targets)
    altered = output_fixture(low=True)
    with torch.no_grad():
        altered.masks_queries_logits[:, :, taps == 0] = 9000
    same, other_grads = evaluate(criterion, altered, targets)
    assert torch.equal(original.loss, same.loss) and matching_equal(original.indices, same.indices)
    assert all(torch.equal(a, b) for a, b in zip(gradients, other_grads, strict=True))
    assert not gradients[0][:, :, taps == 0].any()
    # A valid interpolation tap is allowed to change loss, even though labeling
    # coarse cells using nearest-neighbor validity would mark it invalid.
    query = original.indices[0][0][0]
    with torch.no_grad():
        altered.masks_queries_logits[0, query, 0, 0] += 4
    legitimate, _ = evaluate(criterion, altered, targets)
    assert not torch.equal(original.loss, legitimate.loss)


def test_sampling_preserves_isolated_valid_pixels_and_uncertainty_domain():
    support = torch.zeros(7, 9, dtype=torch.bool)
    support[3, 4] = True
    points = sample_valid_pixels(support, 4, 31, torch.Generator().manual_seed(4))
    assert torch.equal(points, torch.full_like(points, 31))
    uncertain = uncertainty_points(torch.randn(4, 3, 5), support, 31, 3, .75)
    assert torch.equal(uncertain, points)


def test_bilinear_sampling_matches_hf_alignment_and_zero_padding_at_image_edge():
    logits = torch.tensor([[[2.]]], requires_grad=True)
    ids = torch.tensor([[0, 1, 2, 3]])
    values = sample_mask_logits(logits, ids, (2, 2))
    # For align_corners=False each GT edge center lies at +/-0.25 on the 1x1 grid.
    torch.testing.assert_close(values, torch.full((1, 4), 2 * .75**2), rtol=0, atol=0)
    points = torch.tensor([[0, 8, 20, 34]])
    tensor = torch.randn(1, 3, 4, requires_grad=True)
    coords = torch.stack(((points.remainder(7)+.5)/7, (points.div(7, rounding_mode='floor')+.5)/5), -1)
    expected = F.grid_sample(tensor[:, None], (coords*2-1)[:, :, None],
                             mode='bilinear', padding_mode='zeros', align_corners=False)[:, 0, :, 0]
    torch.testing.assert_close(sample_mask_logits(tensor, points, (5, 7)), expected, rtol=0, atol=0)


def test_no_object_classification_and_weights_match_hf_formula():
    targets = target_fixture()
    output = output_fixture(aux=True)
    cfg = config(aux=True)
    criterion = ValidSupportMask2FormerCriterion(cfg)
    result, _ = evaluate(criterion, output, targets)
    target_classes = torch.full((1, 4), 3, dtype=torch.long)
    queries, labels = result.indices[0]
    target_classes[0, queries] = targets[0].class_labels[labels]
    assert int((target_classes == 3).sum()) == 2
    assert (target_classes == 0).sum() == 1  # background is a supervised class
    expected = F.cross_entropy(output.class_queries_logits.transpose(1, 2), target_classes,
                               weight=torch.tensor([1., 1., 1., cfg.no_object_weight]))
    torch.testing.assert_close(result.loss_dict['loss_cross_entropy'], expected, rtol=0, atol=0)
    total = sum(result.loss_dict[key + suffix] * weight
                for suffix in ['', '_0', '_1'] for key, weight in criterion.weights.items())
    torch.testing.assert_close(result.loss, total)


@pytest.mark.parametrize("how", ["ignore", "invalid", "padded_batch"])
def test_all_invalid_rejected_instead_of_no_object_training(how):
    labels = [torch.full((3, 4), 255 if how == 'ignore' else 0, dtype=torch.long)]
    valid = [torch.full((3, 4), how == 'ignore', dtype=torch.bool)]
    if how == 'padded_batch':
        labels.append(torch.zeros(3, 4, dtype=torch.long));valid.append(torch.ones(3, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match='no valid'):
        semantic_targets(labels, valid, 3)
    criterion = ValidSupportMask2FormerCriterion(config())
    target = ValidSemanticTarget(torch.tensor([0]), torch.ones(1, 8, 8), torch.zeros(8, 8, dtype=torch.bool))
    with pytest.raises(ValueError, match='no valid'):
        criterion(output_fixture(), [target])


def test_auxiliary_omission_builtin_loss_and_out_of_range_class_fail_closed():
    criterion = ValidSupportMask2FormerCriterion(config(aux=True))
    with pytest.raises(ValueError, match='auxiliary'):
        criterion(output_fixture(), target_fixture())
    output = output_fixture(aux=True);output.loss = torch.tensor(1.)
    with pytest.raises(ValueError, match='without labels'):
        criterion(output, target_fixture())
    with pytest.raises(ValueError, match='class outside'):
        semantic_targets([torch.tensor([[3]])], [torch.tensor([[True]])], 3)
    with pytest.raises(ValueError, match='padded input canvas'):
        semantic_targets([torch.zeros(3, 4, dtype=torch.long), torch.zeros(4, 4, dtype=torch.long)],
                         [torch.ones(3, 4, dtype=torch.bool), torch.ones(4, 4, dtype=torch.bool)], 3)


def test_fp32_criterion_inside_autocast_preserves_differentiable_casts():
    output = output_fixture()
    output.masks_queries_logits = output.masks_queries_logits.detach().bfloat16().requires_grad_()
    output.class_queries_logits = output.class_queries_logits.detach().bfloat16().requires_grad_()
    criterion = ValidSupportMask2FormerCriterion(config())
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result, grads = evaluate(criterion, output, target_fixture())
    assert result.loss.dtype == torch.float32
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_multiple_images_keep_independent_support_and_normalize_by_all_target_masks():
    first = target_fixture()[0]
    label = torch.full((8, 8), 2, dtype=torch.long)
    valid = torch.zeros(8, 8, dtype=torch.bool)
    valid[7, 7] = True
    targets = [first, semantic_targets([label], [valid], 3)[0]]
    one = output_fixture()
    masks = one.masks_queries_logits.detach().repeat(2, 1, 1, 1).requires_grad_()
    classes = one.class_queries_logits.detach().repeat(2, 1, 1).requires_grad_()
    output = Mask2FormerForUniversalSegmentationOutput(masks_queries_logits=masks,
                                                       class_queries_logits=classes)
    criterion = ValidSupportMask2FormerCriterion(config())
    result = criterion(output, targets, generator=torch.Generator().manual_seed(55))
    assert [len(pair[0]) for pair in result.indices] == [2, 1]
    result.loss.backward()
    assert not masks.grad[0, :, ~first.support].any()
    assert not masks.grad[1, :, ~valid].any()
    assert masks.grad[1, :, 7, 7].abs().sum() > 0
    # Every second-image mask point is the single valid center, so its BCE
    # contribution uses normalization 3 (two first-image classes plus one).
    q = result.indices[1][0][0]
    expected_derivative = (masks.detach()[1, q, 7, 7].sigmoid() - 1) / 3
    bce_grad = torch.autograd.grad(
        criterion(output, targets, generator=torch.Generator().manual_seed(55))
        .loss_dict['loss_mask'], masks)[0]
    torch.testing.assert_close(bce_grad[1, q, 7, 7], expected_derivative, rtol=1e-6, atol=1e-7)


def test_real_small_hf_output_has_all_auxiliary_gradients_without_model_patch():
    backbone = SwinConfig(embed_dim=16, depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 8],
                          window_size=4, out_features=['stage1', 'stage2', 'stage3', 'stage4'])
    cfg = Mask2FormerConfig(backbone_config=backbone, num_labels=3, num_queries=4,
                           feature_size=32, mask_feature_size=32, hidden_dim=32,
                           num_attention_heads=4, encoder_layers=1, decoder_layers=3,
                           encoder_feedforward_dim=64, dim_feedforward=64,
                           train_num_points=17, use_auxiliary_loss=True)
    torch.manual_seed(33)
    model = Mask2FormerForUniversalSegmentation(cfg)
    outputs = model(torch.rand(1, 3, 64, 64), output_auxiliary_logits=True)
    assert outputs.loss is None and len(outputs.auxiliary_logits) == 2
    labels = torch.zeros(64, 64, dtype=torch.long);labels[16:48, 16:48] = 1
    valid = torch.ones(64, 64, dtype=torch.bool);valid[-8:] = False
    criterion = ValidSupportMask2FormerCriterion(cfg)
    result = criterion(outputs, semantic_targets([labels], [valid], 3),
                       generator=torch.Generator().manual_seed(12))
    assert len(result.loss_dict) == 9
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert model.class_predictor.weight.grad.abs().sum() > 0
    for stage in model.model.pixel_level_module.encoder.encoder.layers:
        grads = [p.grad for p in stage.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
