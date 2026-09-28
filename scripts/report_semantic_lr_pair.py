"""Audit the pre-registered constant versus cosine semantic convergence control."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from compare_evaluations import paired_comparison

from bridge_rgs.semantic_schedule import normalize_semantic_schedule, semantic_multiplier
from bridge_rgs.teacher import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/semantic_lr_pair"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    arms = ["00_constant", "01_cosine"]
    receipts = [json.loads((args.root / arm / "experiment_receipt.json").read_text()) for arm in arms]
    a, b = receipts
    assert all(receipt["status"] == "completed" for receipt in receipts)
    assert a["source_hashes"] == b["source_hashes"] and a["input_hashes"] == b["input_hashes"]
    ca, cb = copy.deepcopy(a["config"]), copy.deepcopy(b["config"])
    assert ca.pop("output") != cb.pop("output")
    assert ca["semantic_lr_schedule"].pop("type") == "constant"
    assert cb["semantic_lr_schedule"].pop("type") == "cosine"
    assert ca == cb
    for value in a["input_hashes"].values():
        assert file_sha256(value["path"]) == value["sha256"]
    manifest = json.loads(Path(a["input_hashes"]["manifest"]["path"]).read_text())
    views = [view for view in manifest["views"] if view["split"] == "train"]
    population = [i for i, view in enumerate(views) if view.get("mask_path")]
    assert len(population) == 259
    generator = np.random.default_rng(42)
    sequence = []
    while len(sequence) < 3000:
        order = generator.permutation(population).tolist()
        cursor = min(len(order), 3000 - len(sequence))
        sequence.extend(order[:cursor])
    initial = torch.load(a["input_hashes"]["initial_checkpoint"]["path"], map_location="cpu", weights_only=False)
    fixed = [key for key in initial["model"] if key.startswith("splats.") and key != "splats.sem_features"]
    fixed += ["background_logits", "semantic_prior_counts"]
    rows, metrics = [], []
    for arm, receipt in zip(arms, receipts, strict=True):
        folder = args.root / arm
        assert all(file_sha256(folder / "source_snapshot" / key) == value
                   for key, value in receipt["source_hashes"].items())
        checkpoint = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        assert checkpoint["step"] == 3000
        assert file_sha256(folder / "last.pt") == receipt["checkpoint_sha256"]
        assert all(torch.equal(initial["model"][key], checkpoint["model"][key]) for key in fixed)
        assert torch.equal(initial["training_cameras"], checkpoint["training_cameras"])
        extra = checkpoint["density_state"]["extras"]
        assert extra["sampler_order"] == order and extra["sampler_cursor"] == cursor
        assert extra["sampler_rng_state"] == generator.bit_generator.state
        spec = normalize_semantic_schedule(receipt["config"])
        logs = [json.loads(line) for line in (folder / "train.jsonl").read_text().splitlines()]
        for line in logs:
            multiplier = semantic_multiplier(spec, line["step"])
            assert line["semantic_lr_multiplier"] == multiplier
            for name, base in [("features", .01), ("classifier", .001), ("refiner", .0003)]:
                assert line[f"semantic_lr_{name}"] == base * multiplier
            assert line["semantic_lr_background"] == .001
            assert line["refiner_crop"] is None
            assert line["refiner_crop_view"] == views[sequence[line["step"] - 1]]["name"]
        groups = [checkpoint["optimizers"]["sem_features"]["param_groups"][0],
                  *checkpoint["optimizers"]["heads"]["param_groups"][:2]]
        for group, base in zip(groups, [.01, .001, .0003], strict=True):
            assert group["semantic_base_lr"] == base
            assert group["lr"] == base * semantic_multiplier(spec, 3000)
        assert checkpoint["optimizers"]["heads"]["param_groups"][2]["lr"] == .001
        path = folder / "evaluation_native/metrics.json"
        assert file_sha256(path) == receipt["evaluation_sha256"]
        result = json.loads(path.read_text())
        metrics.append(result)
        rows.append({"arm": arm, "miou_all": result["miou_all"],
                     "miou_foreground": result["miou_foreground"], "iou": result["iou"],
                     "raw_3d": result["semantic_3d"], "cable_boundary_f1": result["boundary_f1_2px"][2],
                     "psnr": result["psnr"], "ssim": result["ssim"], "lpips": result["lpips"],
                     "final_semantic_lr_multiplier": semantic_multiplier(spec, 3000),
                     "checkpoint_sha256": receipt["checkpoint_sha256"],
                     "elapsed_through_last_training_step": logs[-1]["elapsed_seconds"]})
        del checkpoint
    for view in metrics[0]["views"]:
        name = Path(view["name"]).stem + "_rgb.png"
        hashes = [file_sha256(args.root / arm / "evaluation_native" / name) for arm in arms]
        assert hashes[0] == hashes[1]
        assert hashes[0] == file_sha256(Path("runs/strong_semantic_averaged/evaluation_native") / name)
    paired = paired_comparison(metrics[0], metrics[1])
    (args.root / "paired_comparison.json").write_text(json.dumps(paired, indent=2) + "\n")
    report = {"status": "completed", "endpoint": 3000, "same_source_and_inputs": True,
              "only_config_differences": ["output", "semantic_lr_schedule.type"],
              "frozen_geometry_rgb_camera_tensors_exact": True,
              "all_50_rgb_pngs_exact_between_arms_and_warmstart": True,
              "view_sampler_order_cursor_rng_exact": True, "logged_learning_rates_exact": True,
              "background_lr_unchanged": True, "labeled_train_view_count": len(population),
              "view_visits_min_max": [min(sequence.count(i) for i in population), max(sequence.count(i) for i in population)],
              "runs": rows, "paired": paired["metrics"],
              "scope": "Fixed development split, one seed, same update and pixel budget. Engineering convergence control, not a novel mechanism."}
    (args.root / "lr_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
