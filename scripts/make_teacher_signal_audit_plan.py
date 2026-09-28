"""Freeze 16 uniformly indexed labeled TRAIN cameras after full pseudo verification."""

import json
from datetime import UTC, datetime
from pathlib import Path

from audit_teacher_distillation_signal import digest, fixed_views

from bridge_rgs.teacher import verify_pseudo_provenance


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def main():
    destination = Path("configs/generated_teacher_signal_audit/plan.json")
    if destination.exists():
        raise FileExistsError("Preserve the existing signal-audit plan")
    manifest_path = Path("artifacts/prepared/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    indices, views = fixed_views(manifest)
    pseudo_dir = Path("artifacts/pseudo_strong_v1")
    provenance = verify_pseudo_provenance(manifest_path, pseudo_dir, verify_files=True)
    assert len(provenance["views"]) == 350
    assert digest(provenance["checkpoint"]) == provenance["checkpoint_sha256"]
    checkpoint = Path("runs/support_split_semantic_coupled/last.pt")
    receipt_path = checkpoint.parent / "experiment_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "completed" and digest(checkpoint) == receipt["checkpoint_sha256"]
    snapshot_plan = json.loads(Path("configs/generated_refiner_flip_tta/plan.json").read_text())
    snapshot = Path(snapshot_plan["source_snapshot"])
    for key, sha in snapshot_plan["source_hashes"].items():
        assert digest(snapshot / key) == sha
    inputs = {"manifest": record(manifest_path), "student_checkpoint": record(checkpoint),
              "student_receipt": record(receipt_path), "teacher_checkpoint": record(provenance["checkpoint"]),
              "pseudo_provenance": record(pseudo_dir / "provenance.json"),
              "pseudo_confidence_calibration": record(pseudo_dir / "train_confidence_calibration.json"),
              "uv_lock": record("uv.lock")}
    for view in views:
        inputs[f"train_gt_{view['name']}"] = record(view["mask_path"])
        if view.get("valid_path"):
            inputs[f"train_valid_{view['name']}"] = record(view["valid_path"])
    plan = {"protocol": "fixed_16_labeled_train_teacher_signal_v1", "created_utc": datetime.now(UTC).isoformat(),
            "manifest": str(manifest_path.resolve()), "pseudo_dir": str(pseudo_dir.resolve()),
            "checkpoint": str(checkpoint.resolve()), "output": str(Path("runs/teacher_signal_audit16").resolve()),
            "sample_rule": "Sort the 259 labeled TRAIN names ascending, take integer floor(i*258/15) for i=0..15. No label-dependent or performance-dependent view selection.",
            "sample_indices": indices, "sample_names": [view["name"] for view in views],
            "input_hashes": inputs,
            "pseudo_files": {str((pseudo_dir / (Path(row['name']).stem + '.npz')).resolve()): row["sha256"]
                             for row in provenance["views"]},
            "full_pseudo_provenance_verified_at_plan_time": True,
            "source_snapshot": str(snapshot.resolve()), "source_hashes": snapshot_plan["source_hashes"],
            "source_tree_sha256": snapshot_plan["snapshot_tree_sha256"],
            "runner_sha256": digest("scripts/audit_teacher_distillation_signal.py"),
            "student_inference": "Native 1320x989, original TRAIN camera, standard refined output, no TTA, no input real RGB, no optimization.",
            "teacher_inference": "None: read existing FP16 TRAIN NPZ from pseudo_strong_v1; preserve every byte.",
            "scoring": "Decode GT/valid only after both predictions for each view are ready. Plan hashes may read label bytes for provenance but labels never enter prediction.",
            "threshold": .8, "confidence_bins": [0., .2, .4, .6, .8, .9, .95, 1.],
            "scope": "TRAIN in-sample correctness is optimistic; cannot establish quality of unlabeled91 or VAL gains. Bounded signal audit for a possible classical final-only soft-KD control, not novelty."}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plan": str(destination), "sha256": digest(destination), "sample_names": plan["sample_names"],
                      "runner_sha256": plan["runner_sha256"], "fully_verified_pseudo_files": 350}, indent=2))


if __name__ == "__main__":
    main()
