"""Freeze completed checkpoints and current source for one fixed three-arm TTA run."""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from evaluate_refiner_flip_tta import digest


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def main():
    output = Path("runs/refiner_flip_tta")
    plan_path = Path("configs/generated_refiner_flip_tta/plan.json")
    if output.exists() or plan_path.exists():
        raise FileExistsError("A TTA plan/output already exists; preserve it")
    manifest_path = Path("artifacts/prepared/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    views = [view for view in manifest["views"] if view["split"] == "val"]
    assert len(views) == 50 and sum(bool(view.get("mask_path")) for view in views) == 41
    inputs = {"manifest": record(manifest_path), "uv_lock": record("uv.lock"),
              "paired_comparison_script": record("scripts/compare_evaluations.py")}
    arms = {}
    for name, folder in {"base": "runs/support_split_semantic_coupled",
                         "00_control": "runs/refiner_flip/00_control", "01_flip": "runs/refiner_flip/01_flip"}.items():
        folder = Path(folder)
        receipt_path = folder / "experiment_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "completed"
        checkpoint, evaluation = folder / "last.pt", folder / "evaluation_native/metrics.json"
        assert digest(checkpoint) == receipt["checkpoint_sha256"]
        assert digest(evaluation) == receipt["evaluation_sha256"]
        assert receipt["input_hashes"]["manifest"]["sha256"] == inputs["manifest"]["sha256"]
        for role, path in (("checkpoint", checkpoint), ("evaluation", evaluation), ("experiment_receipt", receipt_path)):
            inputs[f"{name}_{role}"] = record(path)
        references = {}
        for view in views:
            for kind in ("rgb", "mask"):
                filename = f"{Path(view['name']).stem}_{kind}.png"
                references[filename] = record(evaluation.parent / filename)
        arms[name] = {"checkpoint": str(checkpoint.resolve()), "evaluation": str(evaluation.resolve()),
                      "experiment_receipt": str(receipt_path.resolve()), "reference_images": references}
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    shutil.copytree("src", snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    source_hashes = {str(path.relative_to(snapshot)): digest(path) for path in sorted(snapshot.rglob("*.py"))}
    assert source_hashes and all(digest(Path("src") / key) == value for key, value in source_hashes.items())
    snapshot_tree_hash = __import__("hashlib").sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    estimated_bytes = 3 * 80 * 1024**2
    assert shutil.disk_usage(output).free - estimated_bytes >= .7 * 1024**3
    plan = {"protocol": "fixed_refiner_horizontal_tta_v1", "created_utc": datetime.now(UTC).isoformat(),
            "manifest": str(manifest_path.resolve()), "output": str(output.resolve()),
            "input_hashes": inputs, "arms": arms, "source_snapshot": str(snapshot.resolve()),
            "source_hashes": source_hashes, "snapshot_tree_sha256": snapshot_tree_hash,
            "runner_sha256": digest("scripts/evaluate_refiner_flip_tta.py"),
            "inference": {"scale": 1.0, "lpips": True, "horizontal_flip_probability_mean": .5,
                          "teacher": False, "optimization_steps": 0},
            "expected_views": 50, "expected_semantic_views": 41,
            "native_size": [1320, 989], "expected_profile": "legacy_mixed_v1",
            "audit": "Hook each plain scene.render before fixed TTA; save and compare all50 argmax PNGs, compare fresh/native RGB and per-view/global raw CM exactly. Preserve old legacy evaluation fingerprint.",
            "gt_policy": "Standard evaluate_scene scorer reads RGB/GT/valid; only camera parameters reach scene.render and only rendered evidence reaches TTA. No claim that scoring files remain unopened.",
            "paired": "Report TTA minus plain for each of all three predeclared checkpoints; no choice of weights or training.",
            "storage": {"estimated_new_bytes": estimated_bytes, "free_bytes_when_planned": shutil.disk_usage(output).free},
            "scope": "Standard horizontal inference augmentation engineering control; fixed development split, not novel or blind-test evidence."}
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plan": str(plan_path), "plan_sha256": digest(plan_path),
                      "snapshot_tree_sha256": snapshot_tree_hash, "runner_sha256": plan["runner_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
