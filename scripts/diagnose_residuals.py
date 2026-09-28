"""Controlled camera-vs-capacity diagnostic on a fixed trained Gaussian scene.

Targets are synthetic renders of this checkpoint, never claimed geometry truth.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from bridge_rgs.io import load_manifest
from bridge_rgs.reliability import camera_compensated_residual, se3_exp
from bridge_rgs.train import load_scene


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="artifacts/prepared/manifest.json")
    parser.add_argument("--output", default="runs/residual_diagnostic")
    parser.add_argument("--views", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(42)
    scene, _ = load_scene(args.checkpoint)
    views = [v for v in load_manifest(args.manifest)["views"] if v["split"] == "train"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for view in views[:: max(1, len(views) // args.views)][: args.views]:
        width = 320
        height = round(view["height"] * width / view["width"])
        K = torch.tensor(view["K"], device="cuda").float()
        K[0] *= width / view["width"]
        K[1] *= height / view["height"]
        pose = torch.tensor(view["w2c"], device="cuda").float()

        def render(candidate, K=K, width=width, height=height):
            return scene.render(K, candidate, width, height, semantics=False, absgrad=False)["rgb"]

        target = render(pose)
        twist = torch.tensor([scene.scene_scale * 0.0001, 0, 0, 0, 0.001, 0], device="cuda")
        shifted = se3_exp(twist) @ pose
        options = {
            "max_translation": scene.scene_scale * 0.002,
            "max_rotation": 0.005,
            "translation_eps": scene.scene_scale * 0.00001,
        }
        camera = camera_compensated_residual(render, shifted, target, **options)
        saved = scene.splats["opacity_logits"].clone()
        dropped = torch.rand(len(saved), device="cuda") < 0.25
        scene.splats["opacity_logits"][dropped] = -12
        try:
            capacity = camera_compensated_residual(render, pose, target, **options)
        finally:
            scene.splats["opacity_logits"].copy_(saved)
        for kind, diagnostic in [
            ("camera_perturbation", camera),
            ("opacity_capacity_removal", capacity),
        ]:
            results.append(
                {
                    "view": view["name"],
                    "case": kind,
                    "explained_fraction": float(diagnostic.explained_fraction),
                    "before_mse": float(diagnostic.before_energy),
                    "after_mse": float(diagnostic.after_energy),
                    "accepted": diagnostic.accepted,
                }
            )
    receipt = {
        "protocol": "self-rendered targets; synthetic pose shift versus random25% opacity removal; no physical geometry ground truth",
        "checkpoint": args.checkpoint,
        "observations": results,
    }
    (output / "diagnostic.json").write_text(json.dumps(receipt, indent=2))
    a = [x["explained_fraction"] for x in results if x["case"] == "camera_perturbation"]
    b = [x["explained_fraction"] for x in results if x["case"] == "opacity_capacity_removal"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([0, 1], [np.mean(a), np.mean(b)], color=["#3485ad", "#dd9156"])
    for i, values in enumerate([a, b]):
        ax.scatter(np.full(len(values), i), values, color="black", s=18, alpha=0.6)
    ax.set_xticks([0, 1], ["Camera perturbation", "Capacity removal"])
    ax.set_ylabel("Fraction of RGB residual explained by camera")
    ax.set_ylim(0, 1)
    ax.set_title("Controlled diagnostic on synthetic checkpoint renders")
    fig.tight_layout()
    fig.savefig(output / "residual_attribution.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "camera_mean": float(np.mean(a)),
                "capacity_mean": float(np.mean(b)),
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()
