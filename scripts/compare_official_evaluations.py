"""Paired comparison restricted to the common original-grid development protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bridge_rgs.official_evaluate import (
    ANNOTATION_COUNT,
    FAMILY,
    SCORING_PROTOCOL,
    VAL_COUNT,
    require_same_official_protocol,
)


def _iou(matrices):
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    union = matrices.sum(-1) + matrices.sum(-2) - diagonal
    return np.divide(diagonal, union, out=np.full_like(diagonal, np.nan, dtype=float), where=union > 0)


def _validate(metrics):
    if metrics.get("scoring_protocol") != SCORING_PROTOCOL:
        raise ValueError("Unsupported original-grid scoring protocol")
    views = metrics["views"]
    if metrics.get("validation_views") != VAL_COUNT or len(views) != VAL_COUNT:
        raise ValueError("Require the complete fixed 50-view RGB evaluation")
    by_name = {view["name"]: view for view in views}
    if len(by_name) != len(views):
        raise ValueError("Duplicate original-grid view name")
    matrices = []
    for view in views:
        if min(view["width"], view["height"]) < 11:
            raise ValueError("Invalid original-grid view size")
        if view["rgb_pixels"] != view["width"] * view["height"]:
            raise ValueError("RGB score did not cover the complete image")
        if not all(np.isfinite(view[key]) for key in ("psnr", "ssim", "lpips")):
            raise ValueError("Nonfinite per-view RGB score")
        if "confusion_matrix" in view:
            matrix = np.asarray(view["confusion_matrix"])
            if (matrix.shape != (5, 5) or not np.issubdtype(matrix.dtype, np.integer)
                    or (matrix < 0).any() or matrix.sum() != view["semantic_pixels"]
                    or min(view["semantic_pixels"], view["semantic_ignore_pixels"]) < 0
                    or view["semantic_pixels"] + view["semantic_ignore_pixels"] != view["rgb_pixels"]):
                raise ValueError("Invalid original-grid semantic counts")
            matrices.append(matrix)
    if len(matrices) != ANNOTATION_COUNT or metrics.get("semantic_validation_views") != ANNOTATION_COUNT:
        raise ValueError("Require the complete fixed 41-view semantic evaluation")
    if not np.array_equal(np.asarray(matrices).sum(0), metrics["confusion_matrix"]):
        raise ValueError("Pooled semantic confusion differs from per-view counts")
    pooled_iou = _iou(np.asarray(matrices).sum(0))
    expected_semantic = {"iou": pooled_iou, "miou_all": np.nanmean(pooled_iou),
                         "miou_foreground": np.nanmean(pooled_iou[1:])}
    for key, expected in expected_semantic.items():
        observed = np.asarray(metrics[key], dtype=float)
        if observed.shape != np.shape(expected) or not np.allclose(observed, expected, atol=1e-12, rtol=0,
                                                                   equal_nan=True):
            raise ValueError("Aggregate semantic score differs from pooled confusion")
    for key in ("psnr", "ssim", "lpips"):
        if not np.isclose(metrics[key], np.mean([view[key] for view in views]), atol=1e-12, rtol=0):
            raise ValueError("Aggregate RGB score differs from per-view mean")
    return by_name


def paired_official_comparison(reference, candidate, repeats=5000, seed=20260926):
    """Resample views, preserving paired cameras; never admit native-grid results."""
    require_same_official_protocol(reference, candidate)
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    a, b = _validate(reference), _validate(candidate)
    if set(a) != set(b):
        raise ValueError("Original-grid camera names differ")
    names = sorted(a)
    semantic_names = [name for name in names if "confusion_matrix" in a[name]]
    if semantic_names != [name for name in names if "confusion_matrix" in b[name]]:
        raise ValueError("Original-grid annotation view names differ")
    for name in names:
        keys = ["width", "height", "rgb_pixels"]
        if name in semantic_names:
            keys += ["semantic_pixels", "semantic_ignore_pixels"]
        if any(a[name][key] != b[name][key] for key in keys):
            raise ValueError("Original-grid dimensions or scoring support differ")
    rng = np.random.default_rng(seed)
    sample = rng.integers(len(names), size=(repeats, len(names)))
    result = {
        "evaluation_family": FAMILY,
        "official_evaluation_fingerprint": reference["official_evaluation_fingerprint"],
        "scoring_protocol": SCORING_PROTOCOL,
        "views": names, "semantic_views": semantic_names,
        "bootstrap_repeats": repeats, "seed": seed,
        "scope": "paired fixed development views only; not independent seeds, bridges, peer Dev30 or blind test",
        "difference_direction": "candidate minus reference; LPIPS improves when negative",
        "metrics": {},
    }

    def add(key, baseline, new, differences):
        finite = np.asarray(differences)[np.isfinite(differences)]
        defined = bool(np.isfinite(baseline) and np.isfinite(new))
        interval = np.quantile(finite, [.025, .975]).tolist() if defined and len(finite) else None
        result["metrics"][key] = {
            "reference": float(baseline) if np.isfinite(baseline) else None,
            "candidate": float(new) if np.isfinite(new) else None,
            "difference": float(new - baseline) if defined else None,
            "paired_view_bootstrap_95_interval": interval,
            "finite_bootstrap_replicates": len(finite),
        }

    for key in ("psnr", "ssim", "lpips"):
        av, bv = [np.asarray([views[name][key] for name in names]) for views in (a, b)]
        add(key, av.mean(), bv.mean(), (bv - av)[sample].mean(1))
    sample = rng.integers(len(semantic_names), size=(repeats, len(semantic_names)))
    am, bm = [np.asarray([views[name]["confusion_matrix"] for name in semantic_names]) for views in (a, b)]
    ai, bi = _iou(am.sum(0)), _iou(bm.sum(0))
    abi, bbi = _iou(am[sample].sum(1)), _iou(bm[sample].sum(1))
    for key, columns in (("miou_all", slice(None)), ("miou_foreground", slice(1, None))):
        add(key, np.nanmean(ai[columns]), np.nanmean(bi[columns]),
            np.nanmean(bbi[:, columns], 1) - np.nanmean(abi[:, columns], 1))
    for index, name in enumerate(("background", "deck", "stay_cable", "tower", "foundation")):
        add(name + "_iou", ai[index], bi[index], bbi[:, index] - abi[:, index])
    return result


def read_completed(path):
    path = Path(path).resolve()
    payload = path.read_bytes()
    receipt_path = path.parent / "execution_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    digest = hashlib.sha256(payload).hexdigest()
    if receipt.get("status") != "completed" or receipt.get("official_metrics_sha256") != digest:
        raise ValueError("Require a completed official receipt binding these exact metric bytes")
    metrics = json.loads(payload)
    require_same_official_protocol(receipt, metrics)
    if receipt.get("scoring_protocol") != metrics.get("scoring_protocol") or receipt.get("scoring_protocol") != SCORING_PROTOCOL:
        raise ValueError("Receipt and metrics scoring protocols differ")
    if not receipt.get("inference_protocol") or receipt.get("inference_protocol") != metrics.get("inference_protocol"):
        raise ValueError("Receipt and metrics inference protocols differ")
    return metrics, {
        "path": str(path), "sha256": digest,
        "receipt": str(receipt_path), "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "checkpoint": receipt["checkpoint"], "checkpoint_sha256": receipt["checkpoint_sha256"],
        "checkpoint_pixel_protocol": receipt["checkpoint_pixel_protocol"],
        "training_manifest": receipt["training_manifest"],
        "inference_protocol": receipt["inference_protocol"],
        **({"teacher_ensemble": receipt["teacher_ensemble"]} if "teacher_ensemble" in receipt else {}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refuse to overwrite an existing paired report")
    a, a_source = read_completed(args.reference)
    b, b_source = read_completed(args.candidate)
    result = paired_official_comparison(a, b, repeats=args.repeats)
    result.update(reference_source=a_source, candidate_source=b_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["metrics"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
