"""Plot a verified paired comparison on the common original pixel grid."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bridge_rgs.official_evaluate import FAMILY, SCORING_PROTOCOL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Filename stem for PNG/PDF/provenance JSON")
    parser.add_argument("--title", default="Bridge-RGS: corrected-coordinate retraining on the common output grid")
    parser.add_argument("--reference-label", default="legacy support model")
    parser.add_argument("--candidate-label", default="fresh corner-v2 model")
    parser.add_argument("--inference-label", default="Single-pass inference")
    args = parser.parse_args()
    content = args.comparison.read_bytes()
    report = json.loads(content)
    if report.get("evaluation_family") != FAMILY or report.get("scoring_protocol") != SCORING_PROTOCOL:
        raise ValueError("Only the separate common original-grid comparison is supported")
    if len(report["views"]) != 50 or len(report["semantic_views"]) != 41:
        raise ValueError("Expected the complete fixed 50/41 development comparison")
    if "inputs" in report:
        for role in ("reference", "candidate"):
            report.setdefault(role + "_source", report["inputs"][role])
    for key in ("reference_source", "candidate_source"):
        source = report[key]
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Compared metric source bytes changed")
    specifications = [
        ("psnr", "RGB PSNR", "dB", 1., True, 3),
        ("ssim", "RGB SSIM", "", 1., True, 5),
        ("lpips", "RGB LPIPS", "", 1., False, 5),
        ("miou_all", "Five-class mIoU", "percentage points", 100., True, 3),
        ("miou_foreground", "Foreground mIoU", "percentage points", 100., True, 3),
        ("stay_cable_iou", "Stay-cable IoU", "percentage points", 100., True, 3),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0))
    fig.subplots_adjust(left=.05, right=.98, top=.78, bottom=.23, hspace=1.0, wspace=.27)
    for ax, (key, title, units, scale, higher, precision) in zip(axes.flat, specifications, strict=True):
        metric = report["metrics"][key]
        delta = metric["difference"] * scale
        lo, hi = np.array(metric["paired_view_bootstrap_95_interval"]) * scale
        if not np.isfinite([delta, lo, hi]).all():
            raise ValueError(f"Cannot plot an undefined endpoint: {key}")
        margin = max(abs(delta), abs(lo), abs(hi), 1e-6) * 1.24
        supported_improvement = lo > 0 if higher else hi < 0
        supported_regression = hi < 0 if higher else lo > 0
        color = "#177c67" if supported_improvement else "#b05a37" if supported_regression else "#65788f"
        ax.axvline(0, color="#949ca6", linewidth=.9, linestyle="--")
        ax.hlines(0, lo, hi, color=color, linewidth=2)
        ax.vlines([lo, hi], -.055, .055, color=color, linewidth=1.4)
        ax.plot(delta, 0, "o", color=color, markersize=7)
        ax.set(xlim=(-margin, margin), ylim=(-.5, .5), yticks=[],
               xlabel=f"Change {('(' + units + ')') if units else ''}\n{'Right' if higher else 'Left'} is better")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=32)
        baseline, candidate = metric["reference"] * scale, metric["candidate"] * scale
        ax.text(0, 1.08, f"{baseline:.{precision}f} → {candidate:.{precision}f}",
                transform=ax.transAxes, fontsize=10, color="#384355")
        ax.text(.5, .8, f"Δ {delta:+.{precision}f}", transform=ax.transAxes, ha="center", color=color, fontsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=.15)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.locator_params(axis="x", nbins=4)
        if lo == hi == delta == 0:
            ax.set_xticks([0])
            ax.text(.5, .12, "Identical output", transform=ax.transAxes,
                    ha="center", color="#65788f", fontsize=9)
    fig.suptitle(args.title,
                 x=.05, ha="left", fontsize=14, fontweight="bold")
    fig.text(.05, .915, f"Reference: {args.reference_label}    |    Candidate: {args.candidate_label}\n{args.inference_label}",
             fontsize=10, color="#465267", va="top")
    fig.text(.05, .035,
             f"Dots: candidate minus reference; bars: paired-view 95% bootstrap intervals ({report['bootstrap_repeats']:,} resamples).\n"
             "Same 50 RGB / 41 annotated development cameras; delivered uint8 PNGs on the original distorted grid.\n"
             "Different native training grids are not compared directly. This is not the peer's Dev30 or a blind test.\n"
             "Green/orange intervals exclude zero in the beneficial/adverse direction; gray intervals cross zero.",
             fontsize=9, color="#596170")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output.with_suffix("." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps({
        "comparison": str(args.comparison.resolve()), "comparison_sha256": hashlib.sha256(content).hexdigest(),
        "title": args.title, "reference_label": args.reference_label, "candidate_label": args.candidate_label,
        "inference_label": args.inference_label,
        "official_evaluation_fingerprint": report["official_evaluation_fingerprint"],
        "reference_source": report["reference_source"], "candidate_source": report["candidate_source"],
        "metrics": report["metrics"],
    }, indent=2, allow_nan=False) + "\n")
    print(args.output.with_suffix(".png").resolve())


if __name__ == "__main__":
    main()
