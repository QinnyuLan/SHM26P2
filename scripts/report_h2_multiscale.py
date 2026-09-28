"""Audit and compare the four predeclared H2 arms at their fixed final step."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from compare_evaluations import paired_comparison

ARMS = ["00_gt_continuation", "01_teacher_no_fusion", "02_teacher_fixed_sampling",
        "03_teacher_projection_sampling"]
PAIRS = [(0, 1, "teacher_kd"), (1, 2, "fixed_fusion"),
         (2, 3, "projection_sampling_primary"), (0, 3, "combined_secondary")]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/h2_multiscale"))
    parser.add_argument("--plan", type=Path, default=Path("configs/generated_h2_multiscale/h2_plan.json"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    plan = json.loads(args.plan.read_text())
    receipts = [json.loads((args.root / arm / "experiment_receipt.json").read_text()) for arm in ARMS]
    base_receipt = receipts[0]
    expected_steps = plan["budget"]["steps_per_arm"]
    manifest = json.loads(Path(base_receipt["input_hashes"]["manifest"]["path"]).read_text())
    train_views = [v for v in manifest["views"] if v["split"] == "train"]
    generator = np.random.default_rng(plan["budget"]["view_sampler_seed"])
    sequence = []
    while len(sequence) < expected_steps:
        final_order = generator.permutation(len(train_views)).tolist()
        final_cursor = min(len(train_views), expected_steps - len(sequence))
        sequence.extend(final_order[:final_cursor])
    labeled_steps = sum(bool(train_views[index].get("mask_path")) for index in sequence)
    if labeled_steps != plan["budget"]["labeled_steps"]:
        raise ValueError("Planned label exposure differs from deterministic sampler")
    initial = torch.load(plan["warmstart"], map_location="cpu", weights_only=False)
    frozen_keys = [k for k in initial["model"] if k.startswith("splats.") and k != "splats.sem_features"]
    frozen_keys.append("background_logits")
    metrics, records = [], []
    expected_differences = [{"pseudo_dir", "pseudo_refiner_weight"},
                            {"multiview_fusion", "multiview_weight"},
                            {"projection_uncertainty"}]
    for i, (arm, receipt) in enumerate(zip(ARMS, receipts, strict=True)):
        folder = args.root / arm
        if receipt["status"] != "completed":
            raise ValueError(f"Run is incomplete: {arm}")
        if receipt["source_hashes"] != base_receipt["source_hashes"]:
            raise ValueError("Source snapshots differ")
        for key, digest in receipt["source_hashes"].items():
            if sha256(folder / "source_snapshot" / key) != digest:
                raise ValueError("Source snapshot changed after execution")
        for key in ("manifest", "uv_lock", "initial_checkpoint"):
            if receipt["input_hashes"][key] != base_receipt["input_hashes"][key]:
                raise ValueError(f"Common input differs: {key}")
        if i >= 1 and receipt["input_hashes"]["pseudo_provenance"] != receipts[1]["input_hashes"]["pseudo_provenance"]:
            raise ValueError("Teacher provenance differs")
        if i:
            previous = receipts[i - 1]["config"]
            different = {key for key in previous.keys() | receipt["config"].keys()
                         if key != "output" and previous.get(key) != receipt["config"].get(key)}
            if different != expected_differences[i - 1]:
                raise ValueError(f"Unexpected treatment differences: {arm}: {different}")
        else:
            different = set()
        checkpoint = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["step"] != expected_steps or sha256(folder / "last.pt") != receipt["checkpoint_sha256"]:
            raise ValueError("Checkpoint endpoint or checksum differs")
        for key in frozen_keys:
            if not torch.equal(initial["model"][key], checkpoint["model"][key]):
                raise ValueError(f"Frozen geometry/RGB changed: {arm}, {key}")
        if not torch.equal(initial["training_cameras"], checkpoint["training_cameras"]):
            raise ValueError("Training cameras changed")
        extras = checkpoint["density_state"]["extras"]
        if (extras["sampler_order"] != final_order or extras["sampler_cursor"] != final_cursor
                or extras["sampler_rng_state"] != generator.bit_generator.state):
            raise ValueError("Sampler does not match predeclared view exposure")
        evaluation_path = folder / "evaluation_native/metrics.json"
        if sha256(evaluation_path) != receipt["evaluation_sha256"]:
            raise ValueError("Evaluation checksum differs")
        evaluation = json.loads(evaluation_path.read_text())
        if evaluation["validation_views"] != 50 or evaluation["semantic_validation_views"] != 41:
            raise ValueError("Evaluation view counts differ")
        if i:
            for view in metrics[0]["views"]:
                stem = Path(view["name"]).stem
                if sha256(folder / "evaluation_native" / (stem + "_rgb.png")) != sha256(
                    args.root / ARMS[0] / "evaluation_native" / (stem + "_rgb.png")
                ):
                    raise ValueError("Frozen RGB pixels differ between arms")
        metrics.append(evaluation)
        logs = [json.loads(line) for line in (folder / "train.jsonl").read_text().splitlines()]
        pseudo_rows = [row for row in logs if "pseudo_refiner_loss" in row and
                       not train_views[sequence[row["step"] - 1]].get("mask_path")]
        acceptance = [row for row in logs if row["step"] in plan["budget"]["acceptance_log_steps"]
                      and "fusion_weighted_kd" in row]
        fields = ("fusion_weighted_kd", "fusion_accepted_weight_sum", "fusion_accepted_weight_by_class",
                  "fusion_accepted_by_class", "fusion_teacher_by_class", "fusion_supervised_conflicts")
        fusion_summary = {key: np.mean([row[key] for row in acceptance], axis=0).tolist()
                          for key in fields} if acceptance else None
        if i >= 2 and len(acceptance) != len(plan["budget"]["acceptance_log_steps"]):
            raise ValueError("Missing used-fusion evidence diagnostics")
        records.append({
            "arm": arm, "checkpoint_sha256": receipt["checkpoint_sha256"],
            "evaluation_sha256": receipt["evaluation_sha256"],
            "changed_config_keys_from_previous_arm": sorted(different),
            "psnr": evaluation["psnr"], "ssim": evaluation["ssim"], "lpips": evaluation["lpips"],
            "miou_all": evaluation["miou_all"], "miou_foreground": evaluation["miou_foreground"],
            "iou": evaluation["iou"], "raw_3d": evaluation["semantic_3d"],
            "boundary_f1_2px": evaluation["boundary_f1_2px"],
            "logged_pseudo_unlabeled_steps": [row["step"] for row in pseudo_rows],
            "logged_pseudo_loss_mean": float(np.mean([row["pseudo_refiner_loss"] for row in pseudo_rows])) if pseudo_rows else None,
            "used_fusion_diagnostic_steps": [row["step"] for row in acceptance],
            "used_fusion_mean_diagnostics": fusion_summary,
        })
        del checkpoint
    comparisons = {}
    for a, b, name in PAIRS:
        comparison = paired_comparison(metrics[a], metrics[b])
        comparison.update(reference_arm=ARMS[a], candidate_arm=ARMS[b])
        (args.root / (name + ".json")).write_text(json.dumps(comparison, indent=2) + "\n")
        comparisons[name] = comparison["metrics"]
    result = {
        "status": "completed", "plan": str(args.plan.resolve()), "plan_sha256": sha256(args.plan),
        "primary_comparison": "03_teacher_projection_sampling minus 02_teacher_fixed_sampling",
        "endpoint": expected_steps, "warmstart": plan["warmstart"],
        "warmstart_sha256": base_receipt["input_hashes"]["initial_checkpoint"]["sha256"],
        "frozen_tensor_keys": frozen_keys, "identical_rgb_pngs_all_50": True,
        "identical_source_snapshots": True, "identical_training_cameras": True,
        "identical_view_sampler_order_cursor_rng": True,
        "labeled_steps": labeled_steps, "unlabeled_steps": expected_steps - labeled_steps,
        "view_visits_min_max": [int(v) for v in (np.bincount(sequence).min(), np.bincount(sequence).max())],
        "pseudo_log_policy": "Only log rows whose current deterministic sample is unlabeled are counted; GT rows can retain stale pseudo diagnostic values",
        "fusion_log_policy": "One diagnostic row per used evidence refresh, at steps300,500,...2900; last refresh at3000 is never used by training",
        "runs": records, "comparisons": comparisons,
        "scope": "Fixed interpolation development split. RGB belongs to the old69k geometry and must not be combined with strongRGB metrics. Bootstrap samples views, not independent scenes.",
    }
    (args.root / "h2_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.root / "h2_report.json"), "runs": records}, indent=2))


if __name__ == "__main__":
    main()
