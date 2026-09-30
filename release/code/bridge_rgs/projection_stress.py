"""Scoring-only helpers for a fixed TRAIN projection-noise diagnostic.

There is no 3D ground truth here. A label is a consensus of visible, original
2D TRAIN projections. None of these helpers supplies labels to evidence gates.
"""
from __future__ import annotations

import numpy as np


def select_groups(views, anchors=16, group_size=4):
    """Use camera metadata only; labeled TRAIN anchors span the first PCA axis."""
    train = sorted((v for v in views if v["split"] == "train"), key=lambda v: v["name"])
    labeled = [v for v in train if v.get("mask_path")]
    if len(labeled) < anchors or len(train) < group_size:
        raise ValueError("Insufficient distinct TRAIN cameras")

    def center(view):
        pose = np.asarray(view.get("w2c_original", view["w2c"]), dtype=np.float64)
        return -pose[:3, :3].T @ pose[:3, 3]

    centers = np.array([center(v) for v in labeled])
    _, _, vectors = np.linalg.svd(centers - centers.mean(0), full_matrices=False)
    axis = vectors[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    positions = centers @ axis
    ordered = sorted(range(len(labeled)), key=lambda i: (positions[i], labeled[i]["name"]))
    chosen = [ordered[i] for i in np.rint(np.linspace(0, len(ordered) - 1, anchors)).astype(int)]
    groups = []
    for index in chosen:
        anchor = labeled[index]
        neighbours = sorted((v for v in train if v["name"] != anchor["name"]),
                            key=lambda v: (np.linalg.norm(center(v) - centers[index]), v["name"]))
        groups.append({"anchor": anchor["name"],
                       "views": [anchor["name"]] + [v["name"] for v in neighbours[:group_size-1]]})
    return groups


def zero_mean_jitter(count, seed):
    """A fixed Gaussian draw, empirically centered; marginal variance stays one."""
    if count < 2:
        raise ValueError("At least two points required")
    noise = np.random.default_rng(seed).normal(size=(count, 2))
    return ((noise - noise.mean(0)) * np.sqrt(count / (count - 1))).astype(np.float32)


def consensus_proxy(labels, view_names, min_views=2, purity=.8, classes=5):
    """Input [V,N], invalid=-1; reject duplicated views instead of inflating votes."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or len(view_names) != len(labels) or len(set(view_names)) != len(labels):
        raise ValueError("Consensus requires distinct views and [V,N] labels")
    if min_views < 2 or not .5 < purity <= 1:
        raise ValueError("Consensus requires at least two views and majority purity")
    counts = np.stack([(labels == c).sum(0) for c in range(classes)], -1)
    mass = counts.sum(-1)
    category = counts.argmax(-1)
    fraction = counts.max(-1) / np.maximum(mass, 1)
    accepted = (mass >= min_views) & (fraction >= purity)
    return np.where(accepted, category, -1), mass, fraction


def score_predictions(proxy, prediction, weight, coverages=(.1, .25, .5, .75, 1.), classes=5):
    """Errors always include coverage; fixed-coverage ranks use weights alone.

    Zero-weight candidates cannot fill missing coverage. Ties use their original
    index. A GT-based class or boundary subset is for scoring only.
    """
    proxy, prediction, weight = map(np.asarray, (proxy, prediction, weight))
    if proxy.ndim != 1 or prediction.shape != proxy.shape or weight.shape != proxy.shape:
        raise ValueError("Scoring inputs must have identical vector shapes")
    valid = (proxy >= 0) & (proxy < classes)
    accepted = valid & np.isfinite(weight) & (weight > 0)
    population, count = int(valid.sum()), int(accepted.sum())
    errors = accepted & (proxy != prediction)
    cm = np.bincount(classes * proxy[accepted] + prediction[accepted],
                     minlength=classes**2).reshape(classes, classes)
    candidates = np.flatnonzero(accepted)
    ranked = candidates[np.argsort(-weight[candidates], kind="stable")]
    curve = []
    for fraction in coverages:
        requested = int(np.floor(population * fraction))
        reachable = requested > 0 and requested <= len(ranked)
        chosen = ranked[:requested] if reachable else np.array([], dtype=int)
        curve.append({"requested_coverage": fraction, "requested_count": requested,
                      "reachable": reachable,
                      "actual_coverage": requested / population if reachable else None,
                      "error_rate": float((prediction[chosen] != proxy[chosen]).mean())
                      if reachable else None})
    per_class = []
    for category in range(classes):
        denominator = int((valid & (proxy == category)).sum())
        n = int((accepted & (proxy == category)).sum())
        e = int((errors & (proxy == category)).sum())
        per_class.append({"class_id": category, "proxy_count": denominator,
                          "accepted": n, "errors": e,
                          "coverage": n / denominator if denominator else None,
                          "error_rate": e / n if n else None})
    return {"proxy_count": population, "accepted": count, "errors": int(errors.sum()),
            "coverage": count / population if population else None,
            "error_rate": int(errors.sum()) / count if count else None,
            "confusion_matrix": cm.tolist(), "per_class": per_class,
            "fixed_coverage_curve": curve}
