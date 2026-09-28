"""Paired development-view bootstrap. Refuse incompatible evaluation protocols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def iou_from_confusion(matrices):
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    union = matrices.sum(-1) + matrices.sum(-2) - diagonal
    return np.divide(diagonal, union, out=np.full_like(diagonal, np.nan, dtype=float),
                     where=union > 0)


def paired_comparison(reference, candidate, repeats=5000, seed=20260926, rgb_only=False):
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    for key in ("protocol", "evaluation_fingerprint", "scale", "lpips_validity_policy"):
        if not reference.get(key) or reference.get(key) != candidate.get(key):
            raise ValueError(f"Incompatible or missing evaluation metadata: {key}")
    a = {v["name"]: v for v in reference["views"]}
    b = {v["name"]: v for v in candidate["views"]}
    if set(a) != set(b) or len(a) != len(reference["views"]) or len(b) != len(candidate["views"]):
        raise ValueError("Evaluation cameras differ or duplicate camera names occur")
    names = sorted(a)
    if not names:
        raise ValueError("No evaluation cameras")
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(names), size=(repeats, len(names)))
    result = {"protocol": reference["protocol"], "evaluation_fingerprint": reference["evaluation_fingerprint"],
              "views": names, "bootstrap_repeats": repeats, "seed": seed,
              "scope": "paired resampling of development views; not independent scenes or blind-test evidence",
              "difference_direction": "candidate minus reference; LPIPS improves when negative",
              "rgb_only": rgb_only,
              "metrics": {}}

    def add(key, baseline, new, boot):
        lo, hi = np.nanquantile(boot, [.025, .975])
        result["metrics"][key] = {"reference": float(baseline), "candidate": float(new),
                                  "difference": float(new-baseline),
                                  "paired_view_bootstrap_95_interval": [float(lo), float(hi)]}

    for metric in ("psnr", "ssim", "lpips"):
        if not all(a[name].get(metric) is not None and b[name].get(metric) is not None for name in names):
            continue
        av, bv = [np.array([records[name][metric] for name in names]) for records in (a, b)]
        add(metric, av.mean(), bv.mean(), (bv-av)[indices].mean(1))
    if rgb_only:
        return result
    semantic_names = [name for name in names if "confusion_matrix" in a[name]]
    if semantic_names != [name for name in names if "confusion_matrix" in b[name]]:
        raise ValueError("Semantic evaluation cameras differ")
    if semantic_names:
        sample = rng.integers(len(semantic_names), size=(repeats, len(semantic_names)))
        for matrix_key, prefix in (("confusion_matrix", ""), ("confusion_matrix_3d", "raw3d_")):
            if not all(matrix_key in records[name] for records in (a, b) for name in semantic_names):
                continue
            am, bm = [np.array([records[name][matrix_key] for name in semantic_names]) for records in (a, b)]
            ai, bi = iou_from_confusion(am.sum(0)), iou_from_confusion(bm.sum(0))
            abs_iou, bbs_iou = iou_from_confusion(am[sample].sum(1)), iou_from_confusion(bm[sample].sum(1))
            for key, columns in (("miou_all", slice(None)), ("miou_foreground", slice(1, None))):
                add(prefix+key, np.nanmean(ai[columns]), np.nanmean(bi[columns]),
                    np.nanmean(bbs_iou[:, columns], 1)-np.nanmean(abs_iou[:, columns], 1))
            for category, name in enumerate(("background", "deck", "stay_cable", "tower", "foundation")):
                add(prefix+name+"_iou", ai[category], bi[category],
                    bbs_iou[:, category]-abs_iou[:, category])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--rgb-only", action="store_true", help="Exclude untrained semantic fields in RGB stages")
    args = parser.parse_args()
    result = paired_comparison(json.loads(args.reference.read_text()),
                               json.loads(args.candidate.read_text()), args.repeats, rgb_only=args.rgb_only)
    result.update(reference_source=str(args.reference.resolve()), candidate_source=str(args.candidate.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
