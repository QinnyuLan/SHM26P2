"""Read-only diagnosis of errors unreachable by a bounded logit correction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bridge_rgs.train import load_scene


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(8)
    source_sha = digest(args.checkpoint)
    scene, checkpoint = load_scene(args.checkpoint)
    scene.eval().requires_grad_(False)
    bound = scene.refiner.residual_bound
    manifest = json.loads(args.manifest.read_text())
    train = [v for v in manifest["views"] if v["split"] == "train"]
    train_indices = {v["name"]: i for i, v in enumerate(train)}
    labeled_train = sorted([v for v in train if v.get("mask_path")], key=lambda v: v["name"])
    selected = [labeled_train[i] for i in np.linspace(0, len(labeled_train)-1, 16, dtype=int)]
    selected += [v for v in manifest["views"] if v["split"] == "val" and v.get("mask_path")]
    rows = []
    for view in selected:
        pose = (checkpoint["training_cameras"][train_indices[view["name"]]].cuda()
                if view["split"] == "train" else torch.tensor(view["w2c"], device="cuda", dtype=torch.float32))
        K = torch.tensor(view["K"], device="cuda", dtype=torch.float32)
        result = scene.render(K, pose, view["width"], view["height"], absgrad=False)
        # Ground truth is opened only after camera-only inference and is never an input.
        labels = torch.tensor(cv2.imread(view["mask_path"], cv2.IMREAD_GRAYSCALE), device="cuda", dtype=torch.long)
        valid = torch.tensor(cv2.imread(view["valid_path"], cv2.IMREAD_GRAYSCALE) > 0, device="cuda") & (labels < 5)
        prior = result["p3d"].clamp_min(1e-7).log()
        margin = prior.amax(-1) - prior.gather(-1, labels.clamp_max(4)[..., None])[..., 0]
        unreachable = margin > 2*bound + 1e-5
        pred = result["probabilities"].argmax(-1)
        wrong = pred != labels
        assert not (unreachable & ~wrong & valid).any(), "Bound invariant violated"
        raw_wrong = result["p3d"].argmax(-1) != labels
        residual = result["residual"]
        row = {"name": view["name"], "split": view["split"], "classes": []}
        for category in range(5):
            keep = valid & (labels == category)
            count = int(keep.sum())
            row["classes"].append({
                "valid_pixels": count, "raw_errors": int((keep & raw_wrong).sum()),
                "final_errors": int((keep & wrong).sum()),
                "theoretically_unreachable": int((keep & unreachable).sum()),
                "near_bound_on_any_class": int((keep & (residual.abs().amax(-1) > .95*bound)).sum()),
                "margin_quantiles": torch.quantile(margin[keep], margin.new_tensor([.5, .9, .99])).tolist() if count else None,
            })
        rows.append(row)
    totals = {}
    fields = ("valid_pixels", "raw_errors", "final_errors", "theoretically_unreachable", "near_bound_on_any_class")
    for split in ("train", "val"):
        totals[split] = []
        for category in range(5):
            selected_rows = [r["classes"][category] for r in rows if r["split"] == split]
            total = {key: sum(r[key] for r in selected_rows) for key in fields}
            total["unreachable_fraction_of_final_errors"] = total["theoretically_unreachable"] / max(1, total["final_errors"])
            totals[split].append(total)
    assert digest(args.checkpoint) == source_sha
    payload = {
        "status": "completed", "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": source_sha,
        "manifest_sha256": digest(args.manifest), "script_sha256": digest(__file__),
        "residual_bound": bound, "criterion": "max(log p3d) - log p3d[GT] > 2 * bound + 1e-5",
        "interpretation": "Necessary obstruction under any per-class correction bounded by +/-B; not a guarantee that other errors are learnable",
        "selection": "16 uniformly spaced indices in sorted labeled TRAIN names; all 41 labeled validation views",
        "scope": "Read-only development diagnosis; no optimizer, no parameter changes, no view exclusion",
        "ground_truth_used_after_prediction_only": True,
        "class_order": ["background", "deck", "stay_cable", "tower", "foundation"],
        "totals": totals, "views": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
