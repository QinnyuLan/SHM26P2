"""Losses and dataset-level metrics, including foreground and all-class mIoU."""
import torch
from torch.nn import functional as F


def ssim_map(x, y):
    # On our torch 2.8.0+cu128 CUDA setup, the full SSIM expression had a
    # channels-last backward mismatch against CPU/FP64 finite differences.
    # Explicit NCHW fixes that reproduced path without changing the formula.
    x = x.permute(2, 0, 1)[None].contiguous()
    y = y.permute(2, 0, 1)[None].contiguous()
    mean_x = F.avg_pool2d(x, 7, stride=1, padding=3)
    mean_y = F.avg_pool2d(y, 7, stride=1, padding=3)
    var_x = F.avg_pool2d(x * x, 7, 1, 3) - mean_x.square()
    var_y = F.avg_pool2d(y * y, 7, 1, 3) - mean_y.square()
    cov = F.avg_pool2d(x * y, 7, 1, 3) - mean_x * mean_y
    value = ((2 * mean_x * mean_y + .01 ** 2) * (2 * cov + .03 ** 2) /
             ((mean_x.square() + mean_y.square() + .01 ** 2) * (var_x + var_y + .03 ** 2)))
    return value[0].mean(0)


def masked_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1)


def opacity_entropy(logits):
    """Stable Bernoulli opacity entropy; a conventional regularizer.

    This biases primitive opacities toward the endpoints, not toward any
    particular camera or image region. Primitive opacity is not occupancy.
    """
    return -(logits.sigmoid() * F.logsigmoid(logits)
             + (-logits).sigmoid() * F.logsigmoid(-logits)).mean()


def supervised_teacher_weights(targets, reliability, semantic_counts,
                               min_observations=2, min_purity=.8):
    """Independent teacher evidence cannot override stable train-track labels.

    Counts follow Gaussian ancestry through densification, rather than being
    reassigned using Euclidean proximity across adjacent bridge components.
    """
    counts = semantic_counts.detach()
    mass = counts.sum(-1)
    purity, category = (counts / mass[:, None].clamp_min(1)).max(-1)
    stable = (mass >= min_observations) & (purity >= min_purity)
    conflict = stable & (category != targets.detach().argmax(-1))
    return reliability.detach() * (~conflict), conflict


def balanced_local_distillation(features, classifier, targets, reliability,
                               max_class_reweight=3.0):
    """Local feature-only KL with capped class-mass balancing.

    Pseudo observations have no gradient to the shared class mapping; that
    mapping is learned from official labeled images. Balancing acts on the
    accepted targets and cannot create supervision for an absent class.
    """
    targets = targets.detach()
    reliability = reliability.detach()
    logits = F.linear(features, classifier.weight.detach(),
                      None if classifier.bias is None else classifier.bias.detach())
    kl = (targets * (targets.clamp_min(1e-7).log() - logits.log_softmax(-1))).sum(-1)
    category = targets.argmax(-1)
    classes = targets.shape[-1]
    class_mass = reliability.new_zeros(classes).scatter_add_(0, category, reliability)
    present = class_mass > 0
    mean_mass = class_mass.sum() / present.sum().clamp_min(1)
    class_weights = (mean_mass / class_mass.clamp_min(1e-8)).clamp(max=max_class_reweight)
    return masked_mean(kl, reliability * class_weights[category])


def rgb_loss(prediction, target, valid):
    return .8 * masked_mean((prediction - target).abs().mean(-1), valid) + \
        .2 * masked_mean(1 - ssim_map(prediction, target), valid)


def lovasz_softmax(probabilities, labels, valid):
    """Present-class Lovasz-Softmax; labels 255 are ignored."""
    keep = valid.bool() & (labels != 255)
    p, y = probabilities[keep], labels[keep]
    losses = []
    for c in range(probabilities.shape[-1]):
        fg = (y == c).float()
        if not fg.any():
            continue
        errors, permutation = (fg - p[:, c]).abs().sort(descending=True)
        fg = fg[permutation]
        intersection = fg.sum() - fg.cumsum(0)
        union = fg.sum() + (1 - fg).cumsum(0)
        gradient = 1 - intersection / union.clamp_min(1)
        gradient = torch.cat([gradient[:1], gradient[1:] - gradient[:-1]])
        losses.append(torch.dot(errors, gradient))
    return torch.stack(losses).mean() if losses else probabilities.sum() * 0


def semantic_loss(probabilities, labels, valid, weights=None, lovasz_weight=.2):
    # PyTorch's CUDA NLL backward requires contiguous NCHW gradient buffers;
    # gsplat returns HWC, whose permute is a channels-last-strided view.
    log_probs = probabilities.clamp_min(1e-7).log().permute(2, 0, 1)[None].contiguous()
    ce = F.nll_loss(log_probs, labels[None].contiguous(), weight=weights,
                    ignore_index=255, reduction="none")[0]
    mask = valid * (labels != 255)
    loss = masked_mean(ce, mask)
    return loss + lovasz_weight * lovasz_softmax(probabilities, labels, mask) if lovasz_weight else loss


def confusion_matrix(probabilities, labels, valid, classes=5):
    pred = probabilities.argmax(-1)
    mask = valid.bool() & (labels < classes)
    return torch.bincount(labels[mask] * classes + pred[mask], minlength=classes ** 2).reshape(classes, classes)


def iou_scores(matrix):
    m = matrix.double()
    union = m.sum(0) + m.sum(1) - m.diag()
    iou = m.diag() / union.clamp_min(1)
    present = union > 0
    return {"iou": [float(x) if bool(v) else None for x, v in zip(iou, present)],
            "miou_foreground": float(iou[1:][present[1:]].mean()) if present[1:].any() else None,
            "miou_all": float(iou[present].mean()) if present.any() else None}
