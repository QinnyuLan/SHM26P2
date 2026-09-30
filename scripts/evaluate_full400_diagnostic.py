"""Evaluate the all-view fit and the historical 50-view camera list.

The all-view manifest intentionally has no validation split.  This script is
therefore a diagnostic, not a replacement for the frozen 350/50 benchmark:
it clears only the manifest-hash guard after loading the checkpoint and marks
the resulting metrics as training-overlap measurements.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from bridge_rgs.evaluate import evaluate_scene
from bridge_rgs.io import load_manifest
from bridge_rgs.train import load_scene


def _run(checkpoint: Path, manifest_path: Path, output: Path, *, select_all: bool,
         lpips: bool):
    scene, _ = load_scene(checkpoint)
    # This is deliberately a diagnostic-only escape hatch.  The pixel
    # protocol remains checked by evaluate_scene; only split lineage is reset
    # because the all-view checkpoint has no held-out split.
    scene.manifest_sha256 = None
    manifest = copy.deepcopy(load_manifest(manifest_path))
    if select_all:
        for view in manifest["views"]:
            view["split"] = "val"
    else:
        manifest["views"] = [view for view in manifest["views"]
                              if view.get("split") == "val"]
    output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_scene(scene, manifest, output, lpips_metric=lpips,
                             checkpoint_pixel_protocol=None)
    metrics["diagnostic_role"] = "all-view training fit" if select_all else (
        "historical 50/41 camera diagnostic with training overlap")
    metrics["independent_generalization"] = False
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=root / "runs/full400_semantic_moments_cross/last.pt")
    parser.add_argument("--full-manifest", type=Path, default=root / "artifacts/prepared_corner_v2_full400/manifest.json")
    parser.add_argument("--reference-manifest", type=Path, default=root / "artifacts/prepared_corner_v2/manifest.json")
    parser.add_argument("--output", type=Path, default=root / "runs/full400_evaluation")
    parser.add_argument("--no-lpips", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    all_fit = _run(checkpoint, args.full_manifest.resolve(), args.output / "all400_fit",
                   select_all=True, lpips=not args.no_lpips)
    historical = _run(checkpoint, args.reference_manifest.resolve(), args.output / "original50_diagnostic",
                      select_all=False, lpips=not args.no_lpips)
    receipt = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "all400_fit": all_fit,
        "original50_diagnostic": historical,
        "independent_generalization": False,
        "interpretation": "All 400 views were used during training; the historical 50/41 views therefore overlap with training and are diagnostic only.",
    }
    (args.output / "evaluation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
