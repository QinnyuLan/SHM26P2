"""Explicit single-stage launcher and complete training-order audit for capacity v1.

Default is inspection only. --execute requires a copied entrypoint, an immutable
package matching the reviewed draft, and no other GPU process. No training loop
is duplicated: wrappers observe the existing teacher calls without random draws.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from bridge_rgs import teacher


def canonical(value) -> bytes:
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=convert).encode()


def object_sha(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_sha(tensor) -> str:
    if tensor.device.type != "cpu":
        raise ValueError("Spatial trace expects pre-CUDA CPU tensors")
    tensor = tensor.detach().contiguous()
    digest = hashlib.sha256(canonical({"shape": list(tensor.shape), "dtype": str(tensor.dtype)}))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rng_summary(device) -> dict:
    device = torch.device(device)
    value = {"cpu": tensor_sha(torch.get_rng_state()), "cuda": []}
    if device.type == "cuda":
        # Deliberately no rand/seed/synchronize, and no initialization of other devices.
        value["cuda"] = [tensor_sha(torch.cuda.get_rng_state(device))]
    return value


def checkpoint_rng(checkpoint) -> dict:
    return {"numpy": object_sha(checkpoint["numpy_generator_state"]),
            "torch": {"cpu": tensor_sha(checkpoint["torch_rng_state"].cpu()),
                      "cuda": [tensor_sha(state.cpu()) for state in
                               (checkpoint.get("cuda_rng_state") or [])]}}


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def utc() -> str:
    return datetime.now(UTC).isoformat()


class StageTrace:
    """Observe each complete training step, suppressing all terminal VAL calls."""

    def __init__(self, path: Path, *, device="cpu", prefix=(), allowed_train=None):
        self.path, self.device = Path(path), torch.device(device)
        self.handle = self.path.open("xb")
        self.digest = hashlib.sha256()
        self.steps = 0
        self.last_record = None
        for record in prefix:
            if record["step"] != self.steps + 1:
                raise ValueError("Noncontiguous resume trace prefix")
            self.digest.update(canonical(record) + b"\n")
            self.steps += 1
            self.last_record = record
        self.start_step = self.steps
        self.events = []
        self.latest_view = None
        self.eval_depth = 0
        self.allowed_train = allowed_train
        self.optimizer_id = None

    def read(self, original, reader, view):
        if self.eval_depth:
            return original(reader, view)
        if view.get("split") != "train":
            raise ValueError("Non-TRAIN read outside explicitly wrapped evaluation")
        identity = {key: view.get(key) for key in (
            "name", "split", "image_path", "mask_path", "valid_path"
        )}
        identity["domain"] = view.get("image_domain", "real_rgb")
        if self.allowed_train is not None:
            allowed = self.allowed_train.get(view["name"])
            if allowed is None or str(view["image_path"]) not in allowed:
                raise ValueError("Training RGB source is outside the locked manifest")
        result = original(reader, view)
        self.latest_view = identity
        self.events.append({"kind": "read", **identity, "shape_hw": list(result[0].shape[:2])})
        return result

    def spatial(self, kind, original, args, kwargs):
        if self.eval_depth:
            return original(*args, **kwargs)
        if self.latest_view is None:
            raise ValueError("Spatial transform was not preceded by a TRAIN read")
        rng = args[4] if len(args) > 4 else kwargs["rng"]
        before = object_sha(rng.bit_generator.state)
        result = original(*args, **kwargs)
        target = args[3] if len(args) > 3 else kwargs.get("size", kwargs.get("short_side"))
        self.events.append({
            "kind": kind, "view": self.latest_view["name"], "domain": self.latest_view["domain"],
            "target_size": target, "augment": kwargs.get("augment", True),
            "pixel_protocol": kwargs.get("pixel_protocol", "legacy_mixed_v1"),
            "input_shape_hw": list(args[0].shape[:2]),
            "output_shapes": [list(value.shape) for value in result],
            "output_sha256": [tensor_sha(value) for value in result],
            "numpy_before": before, "numpy_after": object_sha(rng.bit_generator.state),
        })
        return result

    def photo(self, original, image, strength):
        if self.eval_depth:
            return original(image, strength)
        before = rng_summary(self.device)
        result = original(image, strength)
        self.events.append({"kind": "photometric", "strength": float(strength),
                            "shape": list(image.shape), "rng_before": before,
                            "rng_after": rng_summary(self.device)})
        return result

    def step(self, original, optimizer, args, kwargs):
        if self.eval_depth or not self.events:
            raise ValueError("Optimizer step outside a traced training batch")
        if self.optimizer_id is None:
            self.optimizer_id = id(optimizer)
        if self.optimizer_id != id(optimizer):
            raise ValueError("Unexpected second optimizer in fixed teacher stage")
        before = rng_summary(self.device)
        result = original(optimizer, *args, **kwargs)
        record = {"step": self.steps + 1, "events": self.events,
                  "optimizer_rng_before": before, "optimizer_rng_after": rng_summary(self.device)}
        data = canonical(record) + b"\n"
        self.handle.write(data)
        self.handle.flush()
        if record["step"] % 1000 == 0:
            os.fsync(self.handle.fileno())
        self.digest.update(data)
        self.steps += 1
        self.last_record = record
        self.events, self.latest_view = [], None
        return result

    def evaluate(self, original, *args, **kwargs):
        if self.events:
            raise ValueError("Evaluation started with an uncommitted training trace")
        self.eval_depth += 1
        try:
            return original(*args, **kwargs)
        finally:
            self.eval_depth -= 1

    @contextlib.contextmanager
    def installed(self, module=teacher):
        originals = (module.ViewReader.read, module.aligned_crop, module.aligned_context_frame,
                     module.photometric_augment, module.evaluate_teacher, torch.optim.AdamW.step)
        module.ViewReader.read = lambda reader, view: self.read(originals[0], reader, view)
        module.aligned_crop = lambda *a, **k: self.spatial("crop", originals[1], a, k)
        module.aligned_context_frame = lambda *a, **k: self.spatial("context", originals[2], a, k)
        module.photometric_augment = lambda image, strength: self.photo(originals[3], image, strength)
        module.evaluate_teacher = lambda *a, **k: self.evaluate(originals[4], *a, **k)
        torch.optim.AdamW.step = lambda optimizer, *a, **k: self.step(originals[5], optimizer, a, k)
        try:
            yield self
        finally:
            (module.ViewReader.read, module.aligned_crop, module.aligned_context_frame,
             module.photometric_augment, module.evaluate_teacher, torch.optim.AdamW.step) = originals
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()

    def summary(self) -> dict:
        return {"start_step": self.start_step, "last_step": self.steps,
                "stream_sha256": self.digest.hexdigest(), "segment_path": str(self.path.resolve()),
                "segment_sha256": file_sha(self.path), "pending_event_count": len(self.events),
                "last_optimizer_rng_after": self.last_record["optimizer_rng_after"]
                if self.last_record else None,
                "last_numpy_after": next((event["numpy_after"] for event in
                    reversed(self.last_record["events"]) if "numpy_after" in event), None)
                if self.last_record else None}


def read_records(path: Path, *, permit_truncated_last: bool = False) -> list:
    lines = path.read_bytes().splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if permit_truncated_last and index == len(lines) - 1:
                break
            raise ValueError(f"Truncated trace line: {path}")
        records.append(json.loads(line))
    return records


def recovered_prefix(audit_root: Path, step: int, identity: dict) -> list:
    """Use immutable attempt segments; discard abandoned tail logically, never delete it."""
    history = []
    for directory in sorted(audit_root.glob("attempt_*")):
        meta = json.loads((directory / "attempt.json").read_text())
        if meta["identity"] != identity:
            raise ValueError("Resume trace plan/config/source identity changed")
        resume_step = meta["start_step"]
        if resume_step > len(history):
            raise ValueError("Resume trace has a missing prefix")
        history = history[:resume_step]
        path = directory / "steps.jsonl"
        if path.exists():
            for record in read_records(path, permit_truncated_last=True):
                if record["step"] != len(history) + 1:
                    raise ValueError("Trace step sequence is not contiguous")
                history.append(record)
    if len(history) < step:
        raise ValueError("Checkpoint is ahead of the durable trace")
    return history[:step]


def compare_stage_receipts(first: Path, second: Path) -> dict:
    values = [json.loads(Path(path).read_text()) for path in (first, second)]
    if any(value.get("status") != "completed" for value in values):
        raise ValueError("Both stage receipts must be completed")
    a, b = values
    if a["stage"] != b["stage"] or a["capacity"] == b["capacity"]:
        raise ValueError("Compare different capacities of the same stage")
    for key in ("plan_sha256", "runner_sha256", "source_sha256", "steps"):
        if a[key] != b[key]:
            raise ValueError(f"Stage comparison {key} differs")
    # Verify small JSONL bytes rather than trusting only the advertised stream digest.
    records = []
    for value in values:
        if file_sha(value["last_checkpoint"]) != value["last_checkpoint_sha256"]:
            raise ValueError("Completed endpoint checkpoint changed")
        history = recovered_prefix(Path(value["audit_root"]), value["steps"], value["trace_identity"])
        actual = hashlib.sha256(b"".join(canonical(row) + b"\n" for row in history)).hexdigest()
        if actual != value["trace"]["stream_sha256"]:
            raise ValueError("Persisted trace changed after completion")
        records.append(history)
    first_mismatch = next((index + 1 for index, (x, y) in enumerate(zip(*records, strict=True))
                           if x != y), None)
    checks = {"full_order_trace_equal": first_mismatch is None,
              "terminal_numpy_rng_equal": a["terminal_rng"]["numpy"] == b["terminal_rng"]["numpy"],
              "terminal_torch_rng_equal": a["terminal_rng"]["torch"] == b["terminal_rng"]["torch"]}
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "first_mismatch_step": first_mismatch, "steps": a["steps"],
            "stage": a["stage"], "receipts": {str(Path(p).resolve()): file_sha(p)
                                               for p in (first, second)}}


def verify_bound(item: dict) -> None:
    if file_sha(item["path"]) != item["sha256"]:
        raise ValueError(f"Input SHA differs: {item['path']}")


def verify_source(plan: dict, source_root: Path) -> dict:
    source_root = source_root.resolve()
    if not Path(__file__).resolve().is_relative_to(source_root):
        raise ValueError("Execute a copied entrypoint inside the immutable source snapshot")
    if Path(teacher.__file__).resolve() != source_root / "bridge_rgs/teacher.py":
        raise ValueError("Teacher was not imported from the selected snapshot")
    result = {}
    for name, expected in plan["source_review_sha256"].items():
        path = source_root / Path(name).relative_to("src")
        actual = file_sha(path)
        if actual != expected:
            raise ValueError(f"Snapshot differs from reviewed source: {name}")
        result[name] = actual
    return result


def verify_stage_inputs(plan: dict, capacity: str, stage: str) -> dict:
    inputs = plan["inputs"]
    for key in ("real_manifest", "render_adapt_manifest", "inference_scene",
                "selected_reference_metrics", "selected_reference_receipt"):
        verify_bound(inputs[key])
    for item in inputs["train_labels_and_validity"]:
        verify_bound(item)
    backbone = inputs["backbones"][capacity]
    verify_bound(backbone["download_receipt"])
    files = {}
    for filename, expected in backbone["download_verified_file_sha256"].items():
        if Path(filename).name != filename:
            raise ValueError("Invalid backbone filename")
        path = Path(backbone["model_dir"]) / filename
        if file_sha(path) != expected:
            raise ValueError(f"Backbone file changed: {path}")
        files[str(path)] = expected
    # Verify both domain sources once for either stage: includes TRAIN real RGB,
    # TRAIN rendered RGB and rendered VAL RGB, never real VAL photo pixels.
    from bridge_rgs.teacher_domains import validate_image_sources
    manifest = json.loads(Path(inputs["render_adapt_manifest"]["path"]).read_text())
    validate_image_sources(manifest, verify_files=True)
    return files


def verify_gpu_idle(xml: str) -> None:
    document = ET.fromstring(xml)
    if not document.findall("gpu"):
        raise ValueError("No GPU status is available")
    for gpu in document.findall("gpu"):
        listing = gpu.find("processes")
        if listing is None or (listing.text or "").strip() in {"N/A", "Not Supported"}:
            raise ValueError("GPU process status is unavailable")
    # Desktop graphics are expected on this host. Compute, mixed and unknown
    # process types remain blocking; do not interpret an unknown listing as idle.
    if any(process.findtext("type") != "G" for process in document.findall(".//process_info")):
        raise ValueError("Another GPU compute or unknown process is active")


def run_stage(plan_path: Path, key: str, source_root: Path, *, resume: bool) -> dict:
    plan = json.loads(plan_path.read_text())
    config_binding = plan["configurations"][key]
    verify_bound(config_binding)
    # Original draft bytes remain untouched; consume their byte-identical frozen copy.
    frozen_config = source_root / "draft" / Path(config_binding["path"]).name
    verify_bound({"path": str(frozen_config), "sha256": config_binding["sha256"]})
    specification = json.loads(frozen_config.read_text())
    capacity, stage = key.split("_", 1)
    config = teacher.TeacherConfig(**specification["config"])
    if not config.independent_augmentation_rng or config.eval_every != config.steps:
        raise ValueError("Require isolated augmentation RNG and terminal-only evaluation")
    source_sha = verify_source(plan, source_root)
    output = Path(specification["output_dir"]).resolve()
    if not output.is_relative_to(Path(plan["output_root"]).resolve()):
        raise ValueError("Stage output is outside fixed data-volume root")
    if shutil.disk_usage(output.parent if output.parent.exists() else Path("/mnt/data")).free < 5 << 30:
        raise ValueError("Require 5 GiB free reserve")
    # Do not initialize CUDA merely to test that the exclusive slot is free.
    xml = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True)
    verify_gpu_idle(xml)
    model_files = verify_stage_inputs(plan, capacity, stage)
    output.mkdir(parents=True, exist_ok=True)
    audit_root = output / "capacity_audit"
    audit_root.mkdir(exist_ok=True)
    with (audit_root / ".stage.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return train_locked(plan, plan_path, key, source_sha, specification,
                            config, audit_root, model_files, resume)


def train_locked(plan, plan_path, key, source_sha, specification, config,
                 audit_root, model_files, resume):
    capacity, stage = key.split("_", 1)
    output = audit_root.parent
    last = output / "last.pt"
    identity = {"plan_sha256": file_sha(plan_path), "config_sha256": object_sha(specification),
                "runner_sha256": file_sha(__file__), "source_sha256": object_sha(source_sha)}
    receipt_path = audit_root / "stage_receipt.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()).get("status") == "completed":
        raise FileExistsError("Stage is already completed")
    if bool(last.exists()) != bool(resume):
        raise ValueError("Use explicit --resume iff the fixed stage last.pt exists")
    start_step, prefix, resume_binding = 0, [], None
    if resume:
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        if checkpoint["configuration"] != asdict(config) or not 0 < checkpoint["step"] < config.steps:
            raise ValueError("Resume must preserve exact full config and unfinished fixed total steps")
        start_step = checkpoint["step"]
        prefix = recovered_prefix(audit_root, start_step, identity)
        expected_rng = checkpoint_rng(checkpoint)
        spatial = [event for event in prefix[-1]["events"] if "numpy_after" in event]
        if (prefix[-1]["optimizer_rng_after"] != expected_rng["torch"]
                or spatial[-1]["numpy_after"] != expected_rng["numpy"]):
            raise ValueError("Checkpoint RNG does not match its durable trace prefix")
        resume_binding = {"path": str(last), "sha256": file_sha(last), "step": start_step}
        del checkpoint
    elif list(audit_root.glob("attempt_*")):
        raise ValueError("Existing attempt without checkpoint; choose an explicitly reviewed recovery")
    warmstart = None
    if stage == "render_adapt":
        real_root = output.parent / "real"
        real_receipt_path = real_root / "capacity_audit/stage_receipt.json"
        real_receipt = json.loads(real_receipt_path.read_text())
        if (real_receipt.get("status") != "completed" or real_receipt["steps"] != 6000
                or real_receipt["plan_sha256"] != identity["plan_sha256"]
                or real_receipt["runner_sha256"] != identity["runner_sha256"]
                or real_receipt["source_sha256"] != source_sha
                or config.warmstart_checkpoint != str(real_root / "last.pt")
                or file_sha(config.warmstart_checkpoint) != real_receipt["last_checkpoint_sha256"]):
            raise ValueError("Stage2 must consume its own completed same-source stage1 final EMA")
        warmstart = {"path": config.warmstart_checkpoint,
                     "sha256": real_receipt["last_checkpoint_sha256"], "step": 6000}
    attempt = audit_root / f"attempt_{len(list(audit_root.glob('attempt_*'))):03d}"
    attempt.mkdir()
    write_json(attempt / "attempt.json", {"identity": identity, "start_step": start_step,
                                          "created_utc": utc(), "resume": resume_binding})
    manifest = json.loads(Path(specification["manifest"]).read_text())
    allowed = {view["name"]: {view["image_path"], *(source["path"] for source in
                view.get("image_path_sources", {}).values())}
               for view in manifest["views"] if view["split"] == "train"}
    trace = StageTrace(attempt / "steps.jsonl", device=config.device, prefix=prefix,
                       allowed_train=allowed)
    receipt = {"status": "running", "capacity": capacity, "stage": stage, "steps": config.steps,
               "plan_sha256": identity["plan_sha256"], "runner_sha256": identity["runner_sha256"],
               "source_sha256": source_sha, "trace_identity": identity,
               "audit_root": str(audit_root), "started_utc": utc(), "pid": os.getpid(),
               "configuration": specification, "warmstart": warmstart, "resume": resume_binding,
               "model_input_sha256": model_files}
    write_json(receipt_path, receipt)
    started = time.monotonic()
    try:
        with trace.installed():
            result = teacher.train_teacher(specification["manifest"], specification["model_dir"],
                                           output, config, resume=last if resume else None)
        if trace.steps != config.steps or trace.events:
            raise ValueError("Missing or uncommitted training steps in complete trace")
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        if checkpoint["step"] != config.steps or checkpoint["configuration"] != asdict(config):
            raise ValueError("Fixed endpoint checkpoint differs")
        terminal = checkpoint_rng(checkpoint)
        summary = trace.summary()
        if (summary["last_optimizer_rng_after"] != terminal["torch"]
                or summary["last_numpy_after"] != terminal["numpy"]):
            raise ValueError("Terminal checkpoint RNG differs from observed training endpoint")
        if checkpoint["provenance"]["model_weights_sha256"] != {
            Path(path).name: value for path, value in model_files.items()
            if path.endswith(".safetensors")
        }:
            raise ValueError("Trainer loaded a different backbone")
        if file_sha(plan_path) != identity["plan_sha256"]:
            raise ValueError("Plan changed during training")
        verify_bound(plan["configurations"][key])
        verify_bound({"path": str(Path(teacher.__file__).resolve().parent.parent / "draft" /
                                  Path(plan["configurations"][key]["path"]).name),
                      "sha256": plan["configurations"][key]["sha256"]})
        verify_source(plan, Path(teacher.__file__).resolve().parent.parent)
        verify_stage_inputs(plan, capacity, stage)
        receipt.update(status="completed", finished_utc=utc(), elapsed_seconds=time.monotonic()-started,
                       trace=summary, terminal_rng=terminal, result=result,
                       last_checkpoint=str(last), last_checkpoint_sha256=file_sha(last))
        write_json(receipt_path, receipt)
    except BaseException as error:
        receipt.update(status="failed", failed_utc=utc(), error=f"{type(error).__name__}: {error}",
                       trace=trace.summary(), elapsed_seconds=time.monotonic()-started)
        write_json(receipt_path, receipt)
        raise
    peer_capacity = "vit7b" if capacity == "hplus" else "hplus"
    peer = Path(plan["output_root"]) / peer_capacity / stage / "capacity_audit/stage_receipt.json"
    if peer.exists() and json.loads(peer.read_text()).get("status") == "completed":
        comparison = compare_stage_receipts(peer, receipt_path)
        pair_path = Path(plan["output_root"]) / f"{stage}_rng_comparison.json"
        write_json(pair_path, comparison)
        if comparison["status"] != "passed":
            raise ValueError("Capacity training-order/RNG audit failed; preserve both outcomes")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--plan", type=Path, required=True)
    stage.add_argument("--key", choices=("hplus_real", "hplus_render_adapt", "vit7b_real", "vit7b_render_adapt"), required=True)
    stage.add_argument("--source-snapshot", type=Path)
    stage.add_argument("--resume", action="store_true")
    stage.add_argument("--execute", action="store_true")
    compare = commands.add_parser("compare")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    args = parser.parse_args()
    if args.command == "compare":
        print(json.dumps(compare_stage_receipts(args.first, args.second), indent=2))
    elif not args.execute:
        plan = json.loads(args.plan.read_text())
        print(json.dumps({"status": "inspection_only_no_training", "stage": args.key,
                          "configuration": plan["configurations"][args.key]}, indent=2))
    else:
        if args.source_snapshot is None:
            raise ValueError("Explicit --source-snapshot is required for execution")
        result = run_stage(args.plan.resolve(), args.key, args.source_snapshot, resume=args.resume)
        print(json.dumps({"status": result["status"], "capacity": result["capacity"],
                          "stage": result["stage"], "trace": result["trace"]}, indent=2))


if __name__ == "__main__":
    main()
