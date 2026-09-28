"""Plot coherent camera-only joint evaluations on the fixed development split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    registry = ROOT / "runs/optimization_results.json"
    records = {r["run"]: r for r in json.loads(registry.read_text())["records"]}
    candidates = [
        ("refined", "Initial joint model", (-2, 14)),
        ("refined_long_v2", "Longer original model", (-80, -22)),
        ("multiscale_semantic_averaged", "Old geometry + multiscale average", (15, -15)),
        ("strong_semantic_averaged", "Strong RGB + semantic average", (-12, -33)),
        ("support_split_semantic_coupled", "Corrected split + matched semantic stage", (-12, -45)),
        ("refiner_flip_tta/01_flip", "Flip-trained refiner + fixed flip TTA", (-20, -12)),
        ("teacher_renderer_transfer_support/ensemble_02_cross_0.5", "Support field + fixed DINOv3 mixture*", (-20, 32)),
    ]
    selected = []
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    fig.subplots_adjust(left=.09, right=.985, top=.93, bottom=.22)
    palette = ("#7b8ba3", "#5f96bd", "#158a72", "#7b53ae", "#d78b31", "#dd6485", "#173f73")
    for (run, label, offset), color in zip(candidates, palette):
        if run not in records:
            continue
        r = records[run]
        ax.scatter(r["psnr"], 100*r["miou_all"], s=105, color=color, zorder=4)
        ax.annotate(label, (r["psnr"], 100*r["miou_all"]), xytext=offset,
                    textcoords="offset points", fontsize=9,
                    ha="right" if offset[0] < 0 else "left",
                    arrowprops={"arrowstyle": "-", "color": color, "lw": .8})
        selected.append({"run": run, "label": label, "psnr": r["psnr"],
                         "miou_all": r["miou_all"], "ssim": r["ssim"], "lpips": r["lpips"],
                         "source": r["metrics_source"], "sha256": r["metrics_sha256"]})
    ensemble_path = ROOT / "runs/teacher_student_ensemble/metrics.json"
    if ensemble_path.is_file() and "strong_semantic_averaged" in records:
        combined = json.loads(ensemble_path.read_text())
        expected = ROOT / "runs/strong_semantic_averaged/last.pt"
        if (not combined["shared_rgb_pixels_exact"] or
                Path(combined["student_checkpoint"]).resolve() != expected.resolve()):
            raise ValueError("Ensemble does not share the expected evaluated RGB field")
        r = records["strong_semantic_averaged"]
        miou = combined["metrics"]["ensemble_0.5"]["miou"]
        ax.scatter(r["psnr"], 100*miou, s=115, color="#ca7424", marker="D", zorder=4)
        ax.annotate("Same strong RGB + DINOv3/student ensemble*",
                    (r["psnr"], 100*miou), xytext=(-12, 18), textcoords="offset points",
                    ha="right", fontsize=9, arrowprops={"arrowstyle": "-", "color": "#ca7424", "lw": .8})
        selected.append({"run": "teacher_student_ensemble", "psnr": r["psnr"],
                         "miou_all": miou, "source": str(ensemble_path.relative_to(ROOT)),
                         "sha256": hashlib.sha256(ensemble_path.read_bytes()).hexdigest()})
    ax.axhline(95, color="#a1a8b3", linestyle="--", linewidth=1)
    ax.text(27.5, 95.08, "Current semantic engineering target: 95%", color="#626b79", fontsize=9)
    ax.set(xlabel="RGB PSNR (dB); higher is better",
           ylabel="Final five-class mIoU (%); higher is better",
           title="Bridge-RGS: measured joint RGB and semantic results",
           xlim=(27.35, 30.65), ylim=(89.1, 96.0))
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.02, .03,
             "50 RGB / 41 semantic development views; native 1320 x 989; input cameras unchanged.\n"
             "Repeatedly used for development, not a blind test. Each point keeps one model's RGB and semantic scores.\n"
             "Peer Dev30 results are omitted: the exact split and held-out RGB scores are unavailable.\n"
             "*Extra large teacher at inference; verified identical shared RGB, not mixed best scores from separate fields.",
             fontsize=9, color="#596170")
    output = ROOT / "artifacts/optimization"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"joint_progress.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    payload = {"registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
               "scope": "fixed development split; not matched peer comparison", "records": selected}
    (output / "joint_progress.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(output / "joint_progress.png")


if __name__ == "__main__":
    main()
