"""Inspect existing H2 2D KD logs and TRAIN calibration, without inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from audit_h2_fusion_cpu import quantiles, sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "artifacts/diagnostics/h2_pseudo_cpu.json"
    if output.exists():
        raise FileExistsError(output)
    pseudo = ROOT / "artifacts/pseudo_strong_v1"
    provenance = json.loads((pseudo / "provenance.json").read_text())
    calibration_path = pseudo / "train_confidence_calibration.json"
    assert sha(calibration_path) == provenance["train_confidence_calibration"]["sha256"]
    calibration = json.loads(calibration_path.read_text())
    manifest_path = ROOT / "artifacts/prepared/manifest.json"
    assert sha(manifest_path) == calibration["manifest_sha256"] == provenance["manifest_sha256"]
    assert calibration["checkpoint_sha256"] == provenance["checkpoint_sha256"]
    manifest = json.loads(manifest_path.read_text())
    views = [v for v in manifest["views"] if v["split"] == "train"]
    generator = np.random.default_rng(42)
    sequence = []
    while len(sequence) < 3000:
        sequence.extend(generator.permutation(len(views)).tolist())
    sequence = sequence[:3000]
    report = {"kind": "read_only_CPU_existing_2D_KD_evidence", "gpu_used": False,
              "calibration_sha256": sha(calibration_path), "provenance_sha256": sha(pseudo / "provenance.json"),
              "confidence_definition": provenance["confidence_definition"],
              "labeled_steps": sum(bool(views[i].get("mask_path")) for i in sequence),
              "unlabeled_steps": sum(not views[i].get("mask_path") for i in sequence),
              "arms": {}}
    previous_report = json.loads((ROOT / "runs/h2_multiscale/h2_report.json").read_text())
    for arm in ("01_teacher_no_fusion", "02_teacher_fixed_sampling", "03_teacher_projection_sampling"):
        run = ROOT / "runs/h2_multiscale" / arm
        receipt = json.loads((run / "experiment_receipt.json").read_text())
        assert receipt["input_hashes"]["pseudo_provenance"]["sha256"] == sha(pseudo / "provenance.json")
        assert receipt["config"]["pseudo_threshold"] == .8 and receipt["config"]["pseudo_refiner_weight"] == .1
        rows = []
        for row in map(json.loads, (run / "train.jsonl").read_text().splitlines()):
            view = views[sequence[row["step"] - 1]]
            if view.get("mask_path"):
                continue  # The persisted pseudo fields on GT steps are stale.
            rows.append({"step": row["step"], "view": view["name"],
                         "accepted_pixels": row["pseudo_refiner_pixels"],
                         "native_total_pixels": view["width"] * view["height"],
                         "accepted_fraction_of_full_grid": row["pseudo_refiner_pixels"] / (view["width"] * view["height"]),
                         "class_balanced_KL": row["pseudo_refiner_loss"],
                         "actual_weighted_KL": .1 * row["pseudo_refiner_loss"],
                         "total_loss": row["loss"], "rgb_loss": row["rgb_loss"],
                         "weighted_fusion_loss_same_step": row.get("fusion_weighted_kd", 0.)})
        existing = next(x for x in previous_report["runs"] if x["arm"] == arm)
        assert [x["step"] for x in rows] == existing["logged_pseudo_unlabeled_steps"]
        report["arms"][arm] = {
            "log_sha256": sha(run / "train.jsonl"), "valid_logged_unlabeled_rows": len(rows),
            "actual_weighted_KL_mean": float(np.mean([x["actual_weighted_KL"] for x in rows])),
            "actual_weighted_KL_quantiles": quantiles([x["actual_weighted_KL"] for x in rows]),
            "accepted_pixels_quantiles": quantiles([x["accepted_pixels"] for x in rows]),
            "accepted_fraction_of_full_grid_quantiles": quantiles([x["accepted_fraction_of_full_grid"] for x in rows]),
            "rows": rows,
        }
    t = calibration["threshold_metrics"]["0.8"]
    labeled_counts = np.asarray(calibration["labeled_train_predicted_class_support"])
    all_counts = np.asarray(calibration["all_train_predicted_class_support"])
    labeled_kept = np.asarray(t["labeled_train_accepted_predicted_counts"])
    all_kept = np.asarray(t["all_train_accepted_predicted_counts"])
    unlabeled_counts, unlabeled_kept = all_counts - labeled_counts, all_kept - labeled_kept
    assert np.all(unlabeled_counts > 0) and np.all(unlabeled_kept >= 0)
    report["calibration_at_threshold_0_8"] = {
        "class_names": calibration["class_names"],
        "labeled_train_target_class_coverage": t["labeled_train_target_class_coverage"],
        "labeled_train_predicted_class_coverage": t["labeled_train_predicted_class_coverage"],
        "labeled_train_prediction_error_rate_after_threshold": t["labeled_train_prediction_error_rate"],
        "all_train_predicted_class_coverage": t["all_train_predicted_class_coverage"],
        "unlabeled_only_count_derivation": "Exact subtraction of existing all350 minus labeled259 counts; no new inference/pixel scan; categories are teacher predictions, not GT",
        "unlabeled_only_predicted_class_support": unlabeled_counts.tolist(),
        "unlabeled_only_accepted_predicted_counts": unlabeled_kept.tolist(),
        "unlabeled_only_predicted_class_coverage": (unlabeled_kept / unlabeled_counts).tolist(),
        "unlabeled_only_all_class_coverage": float(unlabeled_kept.sum() / unlabeled_counts.sum()),
        "warning": calibration["note"],
    }
    report["interpretation_limits"] = [
        "Only 8 logged actual pseudo-loss evaluations among 785 unlabeled steps; no full-time mean claim",
        "No per-class student KL, gradients, or accepted/rejected error transitions are stored",
        "TRAIN-fit teacher accuracy on labeled259 does not establish accuracy on unlabeled91",
        "final-output-only loss is not refiner-parameter-only: refiner_field_grad=True also updates sem_features; classifier is detached",
    ]
    report["execution_script_sha256"] = sha(__file__)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
