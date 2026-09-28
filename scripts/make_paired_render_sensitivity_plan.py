"""Prepare a non-runnable draft; lock only after both v2 stage receipts complete.

The default draft path never reads or hashes the pending checkpoint. --lock
requires an explicit completed official receipt and preserves the draft bytes.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch
from audit_paired_render_sensitivity import (
    PROFILE,
    PROTOCOL,
    digest,
    fixed_views,
    protocol_spec,
    record,
)

ROOT = Path(__file__).resolve().parents[1]


def prepare_draft(manifest_path, checkpoint_path, output):
    manifest_path, checkpoint_path = Path(manifest_path).resolve(), Path(checkpoint_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    indices, views = fixed_views(manifest)
    inputs = {"manifest": record(manifest_path), "uv_lock": record(ROOT / "uv.lock")}
    for view in views:
        for field in ("image_path", "mask_path", "valid_path"):
            if view.get(field):
                inputs[f"{view['name']}:{field}"] = record(view[field])
    return {"status": "pending_completed_checkpoint_and_official_evaluation", "protocol": PROTOCOL,
            "created_utc": datetime.now(UTC).isoformat(), "manifest": str(manifest_path),
            "pending_checkpoint_path": str(checkpoint_path), "checkpoint": None,
            "checkpoint_sha256": None, "official_receipt": None,
            "source_snapshot": None, "source_hashes": None,
            "output": str(Path(output).resolve()), "sample_indices": indices,
            "sample_names": [v["name"] for v in views], "input_hashes": inputs,
            "specification": protocol_spec(), "runner_sha256": digest(ROOT / "scripts/audit_paired_render_sensitivity.py"),
            "plan_maker_sha256": digest(__file__), "checkpoint_bytes_read_at_draft": False,
            "execution_authorization": "Root must review the completed-input locked plan before GPU execution"}


def validate_completed_inputs(draft, training_receipt, official_receipt):
    """Check completion first, before touching a potentially changing last.pt."""
    from bridge_rgs.coordinates import pixel_protocol
    from bridge_rgs.official_evaluate import FAMILY, PLAIN_INFERENCE, SCORING_PROTOCOL
    training_receipt, official_receipt = Path(training_receipt).resolve(), Path(official_receipt).resolve()
    train = json.loads(training_receipt.read_text())
    official = json.loads(official_receipt.read_text())
    if train.get("status") != "completed" or official.get("status") != "completed":
        raise ValueError("Training/native and official evaluation must both be completed before reading checkpoint")
    checkpoint = Path(draft["pending_checkpoint_path"])
    if official.get("checkpoint") != str(checkpoint):
        raise ValueError("Official receipt checkpoint path differs")
    if (official.get("evaluation_family") != FAMILY or official.get("inference_protocol") != PLAIN_INFERENCE
            or official.get("scoring_protocol") != SCORING_PROTOCOL
            or pixel_protocol(official.get("checkpoint_pixel_protocol")) != PROFILE):
        raise ValueError("Require the plain common-official corner_v2 receipt")
    manifest_sha = draft["input_hashes"]["manifest"]["sha256"]
    if (train.get("input_hashes", {}).get("manifest", {}).get("sha256") != manifest_sha
            or official.get("training_manifest", {}).get("observed_sha256") != manifest_sha):
        raise ValueError("Completed receipts bind different training manifest")
    actual_sha = digest(checkpoint)
    if train.get("checkpoint_sha256") != actual_sha or official.get("checkpoint_sha256") != actual_sha:
        raise ValueError("Completed checkpoint SHA differs")
    native_metrics = checkpoint.parent / "evaluation_native/metrics.json"
    official_metrics = official_receipt.parent / "official_metrics.json"
    if (digest(native_metrics) != train.get("evaluation_sha256")
            or digest(official_metrics) != official.get("official_metrics_sha256")):
        raise ValueError("Completed metric bytes differ")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if pixel_protocol(state) != PROFILE or state.get("manifest_sha256") != manifest_sha:
        raise ValueError("Actual checkpoint profile/manifest differs")
    if state.get("refiner_config") != protocol_spec()["refiner_config"]:
        raise ValueError("This diagnosis is registered for the fixed multiscale64/off refiner")
    if state.get("step") != 8000 or train.get("config", {}).get("steps") != 8000:
        raise ValueError("Require the fixed completed 8000-step v2 semantic endpoint")
    if digest(checkpoint) != actual_sha:
        raise ValueError("Checkpoint changed while locking")
    return {"student_checkpoint": {"path": str(checkpoint), "sha256": actual_sha},
            "training_receipt": record(training_receipt), "official_receipt": record(official_receipt),
            "native_metrics": record(native_metrics), "official_metrics": record(official_metrics)}


def lock_plan(draft_path, training_receipt, official_receipt, snapshot):
    draft_path = Path(draft_path).resolve()
    draft = json.loads(draft_path.read_text())
    if draft.get("status") != "pending_completed_checkpoint_and_official_evaluation":
        raise ValueError("Expected an unexecuted pending draft")
    if draft.get("protocol") != PROTOCOL or draft.get("specification") != protocol_spec():
        raise ValueError("Draft fixed protocol differs")
    if draft["runner_sha256"] != digest(ROOT / "scripts/audit_paired_render_sensitivity.py"):
        raise ValueError("Runner changed since the draft")
    if draft["plan_maker_sha256"] != digest(__file__):
        raise ValueError("Plan maker changed since the draft")
    for value in draft["input_hashes"].values():
        if digest(value["path"]) != value["sha256"]:
            raise ValueError(f"Draft input changed: {value['path']}")
    completed = validate_completed_inputs(draft, training_receipt, official_receipt)
    snapshot = Path(snapshot).resolve()
    if snapshot.exists():
        raise FileExistsError("Preserve existing source snapshot")
    package = ROOT / "src/bridge_rgs"
    before = {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*.py"))}
    shutil.copytree(package, snapshot / "bridge_rgs", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source_hashes = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))}
    if before != {k.removeprefix("bridge_rgs/"): v for k, v in source_hashes.items()}:
        raise ValueError("Source changed during snapshot")
    if before != {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*.py"))}:
        raise ValueError("Main source changed during snapshot")
    return {**draft, "status": "locked", "locked_utc": datetime.now(UTC).isoformat(),
            "checkpoint": completed["student_checkpoint"]["path"],
            "checkpoint_sha256": completed["student_checkpoint"]["sha256"],
            "official_receipt": completed["official_receipt"], "source_snapshot": str(snapshot),
            "source_hashes": source_hashes,
            "input_hashes": {**draft["input_hashes"], **completed, "draft_plan": record(draft_path)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/prepared_corner_v2/manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/corner_v2_semantic_coupled/last.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/paired_render_sensitivity16")
    parser.add_argument("--draft", type=Path, default=ROOT / "configs/generated_paired_render_sensitivity/draft_plan.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/generated_paired_render_sensitivity/plan.json")
    parser.add_argument("--training-receipt", type=Path, default=ROOT / "runs/corner_v2_semantic_coupled/experiment_receipt.json")
    parser.add_argument("--official-receipt", type=Path)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "runs/paired_render_sensitivity_preparation/source_snapshot")
    args = parser.parse_args()
    destination = args.plan if args.lock else args.draft
    if destination.exists():
        raise FileExistsError("Plans are immutable; preserve the existing file")
    if args.lock:
        if args.official_receipt is None:
            raise ValueError("--lock requires an explicit completed --official-receipt")
        plan = lock_plan(args.draft, args.training_receipt, args.official_receipt, args.snapshot)
    else:
        plan = prepare_draft(args.manifest, args.checkpoint, args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"plan": str(destination.resolve()), "status": plan["status"], "sha256": digest(destination),
                      "checkpoint_sha256": plan["checkpoint_sha256"], "sample_names": plan["sample_names"]}, indent=2))


if __name__ == "__main__":
    main()
