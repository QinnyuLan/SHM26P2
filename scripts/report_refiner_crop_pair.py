"""Audit fixed-budget crop/fullframe student refinement controls."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from compare_evaluations import paired_comparison

from bridge_rgs.refiner_crops import choose_refiner_crop
from bridge_rgs.teacher import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/refiner_crop_pair"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    arms = ["00_fullframe", "01_mixed_crop"]
    receipts = [json.loads((args.root / arm / "experiment_receipt.json").read_text()) for arm in arms]
    a, b = receipts
    if any(r["status"] != "completed" for r in receipts):
        raise ValueError("Both fixed-budget runs must complete before comparison")
    assert a["source_hashes"] == b["source_hashes"] and a["input_hashes"] == b["input_hashes"]
    assert {k for k in a["config"].keys() | b["config"].keys()
            if a["config"].get(k) != b["config"].get(k)} == {"output", "refiner_crop_probability"}
    manifest = json.loads(Path(a["input_hashes"]["manifest"]["path"]).read_text())
    views = [view for view in manifest["views"] if view["split"] == "train"]
    population = [i for i, view in enumerate(views) if view.get("mask_path")]
    generator = np.random.default_rng(42)
    sequence = []
    while len(sequence) < 3000:
        order = generator.permutation(population).tolist()
        cursor = min(len(order), 3000 - len(sequence))
        sequence.extend(order[:cursor])
    initial = torch.load(a["input_hashes"]["initial_checkpoint"]["path"], map_location="cpu", weights_only=False)
    fixed = [k for k in initial["model"] if k.startswith("splats.") and k != "splats.sem_features"] + ["background_logits"]
    rows, metrics = [], []
    for arm, receipt in zip(arms, receipts, strict=True):
        folder = args.root / arm
        for key, value in receipt["source_hashes"].items():
            assert file_sha256(folder / "source_snapshot" / key) == value
        checkpoint = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        assert checkpoint["step"] == 3000
        assert file_sha256(folder / "last.pt") == receipt["checkpoint_sha256"]
        assert all(torch.equal(initial["model"][k], checkpoint["model"][k]) for k in fixed)
        assert torch.equal(initial["training_cameras"], checkpoint["training_cameras"])
        extra = checkpoint["density_state"]["extras"]
        assert extra["sampler_order"] == order and extra["sampler_cursor"] == cursor
        assert extra["sampler_rng_state"] == generator.bit_generator.state
        config = receipt["config"]
        crops = [choose_refiner_crop(views[index]["height"], views[index]["width"],
                                     config["refiner_crop_size"], config["refiner_crop_probability"],
                                     config["refiner_crop_seed"], step)
                 for step, index in enumerate(sequence, 1)]
        logs = [json.loads(line) for line in (folder / "train.jsonl").read_text().splitlines()]
        for line in logs:
            index = line["step"] - 1
            assert line["refiner_crop_view"] == views[sequence[index]]["name"]
            assert line["refiner_crop"] == (list(crops[index]) if crops[index] else None)
        path = folder / "evaluation_native/metrics.json"
        assert file_sha256(path) == receipt["evaluation_sha256"]
        result = json.loads(path.read_text())
        metrics.append(result)
        rows.append({"arm": arm, "crop_steps": sum(crop is not None for crop in crops),
                     "fullframe_steps": sum(crop is None for crop in crops),
                     "miou_all": result["miou_all"], "miou_foreground": result["miou_foreground"],
                     "iou": result["iou"], "raw_3d": result["semantic_3d"],
                     "cable_boundary_f1": result["boundary_f1_2px"][2],
                     "psnr": result["psnr"], "ssim": result["ssim"], "lpips": result["lpips"],
                     "checkpoint_sha256": receipt["checkpoint_sha256"],
                     "elapsed_through_last_training_step": logs[-1]["elapsed_seconds"]})
        del checkpoint
    for view in metrics[0]["views"]:
        name = Path(view["name"]).stem + "_rgb.png"
        assert file_sha256(args.root / arms[0] / "evaluation_native" / name) == file_sha256(
            args.root / arms[1] / "evaluation_native" / name)
    paired = paired_comparison(metrics[0], metrics[1])
    (args.root / "paired_comparison.json").write_text(json.dumps(paired, indent=2) + "\n")
    report = {
        "status": "completed", "endpoint": 3000,
        "same_source_and_inputs": True, "only_config_differences": ["output", "refiner_crop_probability"],
        "frozen_geometry_rgb_camera_tensors_exact": True, "all_50_rgb_pngs_exact": True,
        "view_sampler_order_cursor_rng_exact": True, "logged_crop_coordinates_and_view_names_exact": True,
        "raw_p3d_supervision": "Full native GT at every step; crop does not change camera K or masks",
        "labeled_train_view_count": len(population), "training_steps_per_arm": len(sequence),
        "view_visits_min_max": [min(sequence.count(index) for index in population), max(sequence.count(index) for index in population)],
        "runs": rows, "paired": paired["metrics"],
        "scope": "Fixed interpolation development validation and update budget; engineering augmentation control, not a novel mechanism. Pixel exposure differs by the crop treatment.",
    }
    (args.root / "crop_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
