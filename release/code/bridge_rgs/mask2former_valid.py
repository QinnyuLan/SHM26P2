"""Valid-support semantic criterion for Transformers 4.57.6 Mask2Former outputs.

The HF model stays unchanged: call it WITHOUT labels and with
``output_auxiliary_logits=True``, then pass its output to this criterion. This
adapter samples valid GT pixel centers (with replacement), not the author's
continuous point distribution. Integer target gathers cannot blend ignored
labels into supervision. Prediction sampling retains HF's bilinear,
align_corners=False, zero-padding convention, including legitimate coarse-grid
neighbors of a valid point. No eroding of thin valid support is performed.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    dice_loss,
    pair_wise_dice_loss,
    pair_wise_sigmoid_cross_entropy_loss,
    sample_point,
    sigmoid_cross_entropy_loss,
)

SUPPORT_SAMPLING_ID = "valid_gt_pixel_centers_v1"


@dataclass(frozen=True)
class ValidSemanticTarget:
    """One common padded input canvas; class IDs exclude the no-object class."""

    class_labels: Tensor  # K, long
    mask_labels: Tensor  # K,H,W; values outside support are never gathered
    support: Tensor  # H,W, bool; includes label!=255 AND validity AND padding


@dataclass(frozen=True)
class ValidSupportLossOutput:
    loss: Tensor
    loss_dict: dict[str, Tensor]  # unweighted; names match the HF criterion
    indices: tuple[tuple[Tensor, Tensor], ...]
    auxiliary_indices: tuple[tuple[tuple[Tensor, Tensor], ...], ...]


def semantic_targets(
    labels: Sequence[Tensor], valid: Sequence[Tensor], num_labels: int, ignore_index: int = 255,
) -> list[ValidSemanticTarget]:
    """Build semantic masks only from the effective support, including class 0.

    Inputs must already share the model's padded canvas; this function does no
    resize or padding. All-invalid images are errors, not no-object examples.
    Changing labels outside a FIXED support has no effect. Changing a 255 label
    to a real class while marking it valid changes the support and is not an
    ignored-label perturbation.
    """
    if num_labels <= 0 or 0 <= ignore_index < num_labels:
        raise ValueError("Invalid class count or ignore index")
    if len(labels) != len(valid) or not len(labels):
        raise ValueError("A nonempty aligned label/valid batch is required")
    result = []
    for label, keep in zip(labels, valid, strict=True):
        if (label.ndim != 2 or keep.shape != label.shape or keep.dtype != torch.bool
                or label.dtype not in (torch.uint8, torch.int32, torch.int64)
                or keep.device != label.device):
            raise ValueError("Require aligned integer labels and boolean validity")
        support = keep & (label != ignore_index)
        if not bool(support.any()):
            raise ValueError("Image has no valid semantic support")
        classes = label[support].long().unique(sorted=True)
        if bool(((classes < 0) | (classes >= num_labels)).any()):
            raise ValueError("Valid target class outside the configured classes")
        masks = (label[None] == classes[:, None, None]) & support[None]
        result.append(ValidSemanticTarget(classes, masks, support))
    if len({tuple(target.support.shape) for target in result}) != 1:
        raise ValueError("Targets must use one explicitly padded input canvas")
    return result


def sample_valid_pixels(
    support: Tensor, rows: int, count: int, generator: torch.Generator | None = None,
) -> Tensor:
    """Uniform valid centers, independently per row, sampled with replacement."""
    if support.ndim != 2 or support.dtype != torch.bool or rows <= 0 or count <= 0:
        raise ValueError("Require boolean HW support and positive sampling sizes")
    valid_ids = support.flatten().nonzero().flatten()
    if not len(valid_ids):
        raise ValueError("Image has no valid semantic support")
    choices = torch.randint(len(valid_ids), (rows, count), device=support.device,
                            generator=generator)
    return valid_ids[choices]


def sample_mask_logits(logits: Tensor, pixel_ids: Tensor, canvas_hw: tuple[int, int]) -> Tensor:
    """Sample QHW logits at QP integer GT centers on the padded input canvas.

    Target labels are NOT sampled by this function. A coarse prediction logit
    can legitimately affect nearby valid centers even if its own projected
    center lies over an ignored target. Only values outside all effective
    bilinear tap footprints must have zero supervised influence.
    """
    height, width = canvas_hw
    if (logits.ndim != 3 or pixel_ids.ndim != 2 or pixel_ids.shape[0] != logits.shape[0]
            or height <= 0 or width <= 0 or pixel_ids.dtype != torch.long
            or pixel_ids.device != logits.device
            or bool(((pixel_ids < 0) | (pixel_ids >= height * width)).any())):
        raise ValueError("Invalid prediction/point/canvas alignment")
    x = (pixel_ids.remainder(width).float() + .5) / width
    y = (pixel_ids.div(width, rounding_mode="floor").float() + .5) / height
    coordinates = torch.stack((x, y), -1)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return sample_point(logits.float()[:, None], coordinates,
                            mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]


@torch.no_grad()
def uncertainty_points(
    logits: Tensor, support: Tensor, count: int, oversample_ratio: float,
    importance_sample_ratio: float, generator: torch.Generator | None = None,
) -> Tensor:
    """HF uncertainty ranking, restricted to valid GT centers in both branches."""
    if (count <= 0 or not isfinite(oversample_ratio) or oversample_ratio < 1
            or not isfinite(importance_sample_ratio) or not 0 <= importance_sample_ratio <= 1):
        raise ValueError("Invalid uncertainty sampling ratios")
    rows = logits.shape[0]
    candidates = sample_valid_pixels(support, rows, int(count * oversample_ratio), generator)
    values = sample_mask_logits(logits, candidates, tuple(support.shape))
    number = int(count * importance_sample_ratio)
    ids = (-values.abs()).topk(number, dim=1).indices
    important = candidates.gather(1, ids)
    if number == count:
        return important
    random = sample_valid_pixels(support, rows, count-number, generator)
    return torch.cat((important, random), 1)


class ValidSupportMask2FormerCriterion(nn.Module):
    """Single-process adapter with HF CE/BCE/Dice and explicit valid sampling.

    No-object classification remains HF's C-th class for unmatched queries.
    Every class present in valid pixels is a target (background included).
    Matching is recomputed for each auxiliary head on the same valid domain.
    Returned loss_dict is unweighted; .loss applies each configured weight once.
    Use criterion.to(device) alongside the model; keep criterion buffers FP32.
    """

    sampling_protocol = SUPPORT_SAMPLING_ID

    def __init__(self, config):
        super().__init__()
        self.weights = {"loss_cross_entropy": float(config.class_weight),
                        "loss_mask": float(config.mask_weight), "loss_dice": float(config.dice_weight)}
        if (config.num_labels <= 0 or config.train_num_points <= 0
                or not isfinite(config.oversample_ratio) or config.oversample_ratio < 1
                or not isfinite(config.importance_sample_ratio)
                or not 0 <= config.importance_sample_ratio <= 1
                or any(not isfinite(w) or w < 0 for w in self.weights.values())
                or not any(self.weights.values()) or not 0 < config.no_object_weight <= 1
                or config.decoder_layers < 1):
            raise ValueError("Invalid criterion configuration")
        self.hf = Mask2FormerLoss(config, self.weights)
        self.auxiliary_count = config.decoder_layers-1 if config.use_auxiliary_loss else 0

    def _validate(self, outputs, targets):
        if outputs.loss is not None:
            raise ValueError("Call the HF model without labels; its built-in loss must not run")
        masks, classes = outputs.masks_queries_logits, outputs.class_queries_logits
        if (masks.ndim != 4 or classes.ndim != 3 or masks.shape[:2] != classes.shape[:2]
                or classes.shape[-1] != self.hf.num_labels+1 or masks.shape[0] != len(targets)
                or not targets or masks.device != classes.device):
            raise ValueError("Invalid prediction batch/query/class shape")
        if self.hf.empty_weight.device != classes.device or self.hf.empty_weight.dtype != torch.float32:
            raise ValueError("Criterion buffers must be FP32 on the prediction device")
        shapes = set()
        for target in targets:
            support, labels, binary = target.support, target.class_labels, target.mask_labels
            if (support.ndim != 2 or support.dtype != torch.bool or labels.dtype != torch.long
                    or labels.ndim != 1 or binary.shape != (len(labels), *support.shape)
                    or any(t.device != classes.device for t in (support, labels, binary))):
                raise ValueError("Invalid semantic target shape/device/dtype")
            if not bool(support.any()):
                raise ValueError("Image has no valid semantic support")
            if (not len(labels) or len(labels) > masks.shape[1]
                    or not torch.equal(labels.unique(sorted=True), labels)
                    or bool(((labels < 0) | (labels >= self.hf.num_labels)).any())):
                raise ValueError("Targets require unique real classes and enough queries")
            active = binary[:, support]
            if (not bool(((active == 0) | (active == 1)).all())
                    or not bool((active.sum(0) == 1).all()) or not bool(active.any(1).all())):
                raise ValueError("Valid pixels must belong to exactly one nonempty semantic mask")
            shapes.add(tuple(support.shape))
        if len(shapes) != 1:
            raise ValueError("Targets must use one explicitly padded input canvas")
        auxiliary = outputs.auxiliary_logits
        if len(auxiliary or []) != self.auxiliary_count:
            raise ValueError("Missing or unexpected auxiliary heads; request output_auxiliary_logits=True")
        for head in [{"masks_queries_logits": masks, "class_queries_logits": classes}, *(auxiliary or [])]:
            m, c = head["masks_queries_logits"], head["class_queries_logits"]
            if (m.ndim != 4 or min(m.shape) <= 0 or m.shape[:2] != masks.shape[:2] or c.shape != classes.shape
                    or m.device != masks.device or c.device != masks.device
                    or not m.is_floating_point() or not c.is_floating_point()
                    or not bool(torch.isfinite(m).all() & torch.isfinite(c).all())):
                raise ValueError("Nonfinite or incompatible main/auxiliary logits")
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise ValueError("This adapter's mask normalization is single-process only")

    @torch.no_grad()
    def _match(self, masks, classes, targets, generator):
        indices = []
        for prediction, scores, target in zip(masks, classes, targets, strict=True):
            points = sample_valid_pixels(target.support, 1, self.hf.num_points, generator)
            pred = sample_mask_logits(prediction, points.expand(len(prediction), -1),
                                      tuple(target.support.shape))
            truth = target.mask_labels.detach().flatten(1)[:, points[0]].float()
            cost = (self.weights["loss_cross_entropy"] * -scores.softmax(-1)[:, target.class_labels]
                    + self.weights["loss_mask"] * pair_wise_sigmoid_cross_entropy_loss(pred, truth)
                    + self.weights["loss_dice"] * pair_wise_dice_loss(pred, truth))
            if not bool(torch.isfinite(cost).all()):
                raise ValueError("Nonfinite Hungarian cost; refusing silent replacement")
            query_ids, label_ids = linear_sum_assignment(cost.cpu())
            indices.append((torch.as_tensor(query_ids, device=masks.device),
                            torch.as_tensor(label_ids, device=masks.device)))
        return tuple(indices)

    def _head(self, masks, classes, targets, generator):
        masks, classes = masks.float(), classes.float()
        indices = self._match(masks, classes, targets, generator)
        losses = self.hf.loss_labels(classes, [t.class_labels for t in targets], indices)
        predictions, truths = [], []
        for prediction, target, (queries, labels) in zip(masks, targets, indices, strict=True):
            matched = prediction[queries]
            points = uncertainty_points(matched, target.support, self.hf.num_points,
                                        self.hf.oversample_ratio, self.hf.importance_sample_ratio,
                                        generator)
            predictions.append(sample_mask_logits(matched, points, tuple(target.support.shape)))
            truths.append(target.mask_labels.detach()[labels].flatten(1).gather(1, points).float())
        pred, truth = torch.cat(predictions), torch.cat(truths)
        number = sum(len(target.class_labels) for target in targets)
        losses.update(loss_mask=sigmoid_cross_entropy_loss(pred, truth, number),
                      loss_dice=dice_loss(pred, truth, number))
        return losses, indices

    def forward(self, outputs, targets: Sequence[ValidSemanticTarget], *, generator=None):
        self._validate(outputs, targets)
        with torch.autocast(device_type=outputs.masks_queries_logits.device.type, enabled=False):
            losses, indices = self._head(outputs.masks_queries_logits, outputs.class_queries_logits,
                                        targets, generator)
            auxiliary_indices = []
            for i, head in enumerate(outputs.auxiliary_logits or []):
                aux, matches = self._head(head["masks_queries_logits"], head["class_queries_logits"],
                                          targets, generator)
                losses.update({f"{key}_{i}": value for key, value in aux.items()})
                auxiliary_indices.append(matches)
            total = sum(value * self.weights[next(key for key in self.weights
                                                  if name == key or name.startswith(key+"_"))]
                        for name, value in losses.items())
        return ValidSupportLossOutput(total, losses, indices, tuple(auxiliary_indices))
