"""Execute the frozen two-model original-grid plan after an explicit GPU handoff."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path("runs/official_common_grid_v1")
CORE = ("official_evaluate", "evaluate", "coordinates", "data", "losses", "train", "model", "refinement")
SECOND_ARM_MINIMUM = (300 + 256) << 20  # Remaining PNG budget plus retained free-space floor.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(UTC).isoformat()


def write_receipt(path, record):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def bind(registry, path, expected=None):
    path = Path(path).resolve()
    actual = digest(path)
    require(expected is None or actual == expected, f"SHA mismatch: {path}")
    require(str(path) not in registry or registry[str(path)] == actual, f"Input changed: {path}")
    registry[str(path)] = actual
    return actual


def require_empty_outputs(plan):
    for arm in plan["inputs"]:
        path = Path(arm["output"])
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"Refuse nonempty output: {path}")
    if Path(plan["comparison_output"]).exists():
        raise FileExistsError("Refuse existing paired comparison")


def completed_endpoints(root, plan):
    """Metadata-only first gate: never hash a checkpoint before every stage completes."""
    records = {}
    paths = [Path(arm["training_receipt"]) for arm in plan["inputs"]]
    paths.append(root / "runs/corner_v2_rgb_full/experiment_receipt.json")
    for path in paths:
        value = read_json(path)
        require(value.get("status") == "completed", f"Training has not naturally completed: {path}")
        require(value.get("finished_utc") and value.get("checkpoint_sha256") and value.get("evaluation_sha256"),
                f"Incomplete training receipt: {path}")
        records[str(path)] = value
    for stage, folder, step in (("rgb", "corner_v2_rgb_full", 30000),
                                ("semantic", "corner_v2_semantic_coupled", 8000)):
        path = root / "runs" / folder / "stage_audit.json"
        audit = read_json(path)
        require(audit.get("status") == "passed" and audit.get("stage") == stage and audit.get("step") == step,
                f"Stage audit has not passed: {path}")
        if stage == "semantic":
            require(audit.get("semantic_rgb_all50_png_and_metrics_exact_to_source") is True
                    and audit.get("semantic_frozen_rgb_geometry_keys"), "Missing frozen-geometry semantic audit")
        records[str(path)] = audit
    return records


def check_gpu_idle():
    command = ["nvidia-smi", "-q", "-x"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    document = ET.fromstring(result.stdout)
    gpus = document.findall("gpu")
    require(gpus, "No GPU reported by nvidia-smi")
    processes = document.findall(".//process_info")
    compute = [p for p in processes if p.findtext("type") != "G"]
    require(not compute, "GPU has compute or unclassified clients: " + ", ".join(
        f"pid={p.findtext('pid')} type={p.findtext('type')} {p.findtext('process_name')}" for p in compute))
    for gpu in gpus:
        listing = gpu.find("processes")
        require(listing is not None and (listing.text or "").strip() not in {"N/A", "Not Supported"},
                "GPU process list unavailable; cannot establish exclusive handoff")
    return {"checked_utc": utc(), "command": command,
            "devices": [{"uuid": g.findtext("uuid"), "name": g.findtext("product_name")} for g in gpus],
            "process_count": len(processes), "compute_process_count": 0,
            "graphics_only_processes": [{"pid": p.findtext("pid"), "name": p.findtext("process_name"),
                                         "type": "G"} for p in processes],
            "exclusivity_scope": "No other compute clients; existing graphics-only desktop clients remain"}


def check_space(root, reserved_bytes, minimum=1 << 30):
    require(reserved_bytes >= 0, "Other-job reservation must be nonnegative")
    free = shutil.disk_usage(root).free
    require(free - reserved_bytes >= minimum, "Insufficient free space after other-job reservations")
    return {"free_bytes": free, "other_jobs_reserved_bytes": reserved_bytes,
            "available_after_reservations": free - reserved_bytes, "required_bytes": minimum}


def frozen_modules(snapshot):
    require(not any(name.startswith("bridge_rgs") for name in sys.modules),
            "Runner must start in a fresh process before importing bridge_rgs")
    sys.path.insert(0, str(snapshot))
    modules = {name: importlib.import_module(f"bridge_rgs.{name}") for name in CORE}
    for name, module in modules.items():
        require(Path(module.__file__).resolve() == snapshot / "bridge_rgs" / f"{name}.py",
                f"Module escaped frozen snapshot: {name}")
    return modules


def lpips_weights(registry):
    import torch
    spec = importlib.util.find_spec("lpips")
    require(spec is not None and spec.origin, "LPIPS package unavailable")
    paths = {"alexnet_imagenet": Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth",
             "lpips_alex_v01": Path(spec.origin).parent / "weights/v0.1/alex.pth"}
    for path in paths.values():
        require(path.is_file() and path.stat().st_size > 0, f"Required local LPIPS weight missing: {path}")
    return {key: {"path": str(path.resolve()), "sha256": bind(registry, path)} for key, path in paths.items()}


def check_sources(plan, registry):
    snapshot = Path(plan["source_snapshot"])
    receipt_path = Path(plan["source_snapshot_receipt"])
    bind(registry, receipt_path, plan["source_snapshot_receipt_sha256"])
    frozen = read_json(receipt_path)
    require(set(frozen["source_hashes"]) == {str(p.relative_to(snapshot)) for p in snapshot.rglob("*.py")},
            "Source snapshot file set differs")
    tree = hashlib.sha256(json.dumps(frozen["source_hashes"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(tree == frozen["source_tree_sha256"] == plan["source_tree_sha256"], "Source tree digest differs")
    for name, sha in frozen["source_hashes"].items():
        bind(registry, snapshot / name, sha)
    for category in ("runner_hashes", "test_hashes"):
        for name, sha in frozen[category].items():
            bind(registry, receipt_path.parent / name, sha)
    return frozen


def check_training(root, plan, records, registry, official):
    training_plan_path = root / "runs/corner_v2_training_plan.json"
    bind(registry, training_plan_path)
    training_plan = read_json(training_plan_path)
    for path in records:
        bind(registry, path)
    for folder in ("corner_v2_rgb_full", "corner_v2_semantic_coupled"):
        directory = root / "runs" / folder
        receipt = records[str(directory / "experiment_receipt.json")]
        audit = records[str(directory / "stage_audit.json")]
        require(receipt["source_hashes"] == training_plan["source_hashes"], "v2 training source differs from plan")
        require(audit["source_tree_sha256"] == training_plan["source_tree_sha256"], "Stage source audit differs")
        require(audit["manifest_sha256"] == training_plan["manifest_sha256"]
                and audit["profile"] == "colmap_corner_v2", "Stage profile or manifest audit differs")
        for name, sha in receipt["source_hashes"].items():
            bind(registry, directory / "source_snapshot" / name, sha)
        bind(registry, directory / "last.pt", receipt["checkpoint_sha256"])
        require(audit["checkpoint_sha256"] == receipt["checkpoint_sha256"], "Stage checkpoint audit differs")
        bind(registry, directory / "evaluation_native/metrics.json", receipt["evaluation_sha256"])
        for value in receipt["input_hashes"].values():
            bind(registry, value["path"], value["sha256"])
    rgb = records[str(root / "runs/corner_v2_rgb_full/experiment_receipt.json")]
    sem = records[str(root / "runs/corner_v2_semantic_coupled/experiment_receipt.json")]
    require(sem["input_hashes"]["initial_checkpoint"]["sha256"] == rgb["checkpoint_sha256"], "v2 stage lineage differs")
    reference = official.read_official_reference(plan["reference_manifest"]["path"], root)
    import torch
    inputs = []
    for arm in plan["inputs"]:
        receipt = records[arm["training_receipt"]]
        sha = bind(registry, arm["checkpoint"], receipt["checkpoint_sha256"])
        require(arm["expected_checkpoint_sha256"] in (None, sha), "Planned checkpoint differs")
        manifest = receipt["input_hashes"]["manifest"]
        bind(registry, manifest["path"], arm["training_manifest_sha256"])
        require(manifest["sha256"] == arm["training_manifest_sha256"], "Training manifest receipt differs")
        state = torch.load(arm["checkpoint"], map_location="cpu", weights_only=False)
        require(state["step"] == receipt["config"]["steps"] == 8000, "Expected final 8000-step semantic endpoint")
        require(official.pixel_protocol(state) == arm["pixel_protocol"], "Checkpoint own profile differs")
        lineage = official.validate_training_lineage(state, reference)
        inputs.append(dict(arm, actual_checkpoint_sha256=sha, training_lineage=lineage))
        del state
    return reference, inputs


def audit_output(arm, reference, frozen, official, comparison, registry):
    from PIL import Image
    output = Path(arm["output"])
    receipt_path = output / "execution_receipt.json"
    receipt = read_json(receipt_path)
    metrics, _ = comparison.read_completed(output / "official_metrics.json")
    comparison._validate(metrics)
    require(receipt["checkpoint_sha256"] == arm["actual_checkpoint_sha256"], "Evaluation checkpoint differs")
    require(receipt["checkpoint_pixel_protocol"]["id"] == arm["pixel_protocol"], "Evaluation profile differs")
    require(receipt["inference_protocol"] == official.PLAIN_INFERENCE, "Expected plain inference")
    entry = receipt["entrypoint_source"]
    expected_entry = output.parent / "scripts/evaluate_official.py"
    require(Path(entry["path"]).resolve() == expected_entry, "Evaluator entrypoint path differs")
    require(entry["sha256"] == frozen["runner_hashes"]["scripts/evaluate_official.py"], "Evaluator entrypoint SHA differs")
    modules = receipt["loaded_source_modules"]
    require(set(modules) == {f"bridge_rgs.{name}" for name in CORE}, "Incomplete loaded source audit")
    snapshot = output.parent / "source_snapshot"
    for name in CORE:
        item, relative = modules[f"bridge_rgs.{name}"], f"bridge_rgs/{name}.py"
        require(Path(item["path"]).resolve() == snapshot / relative
                and item["sha256"] == frozen["source_hashes"][relative], f"Actual scorer source differs: {name}")
    require(receipt["predictions_finished_utc"] <= receipt["source_scoring_started_utc"], "GT phase preceded predictions")
    cameras = {view["name"]: view for view in reference["cameras"]}
    predictions = receipt["predictions"]
    require(len(predictions) == 50 and {p["name"] for p in predictions} == set(cameras), "Incomplete PNG predictions")
    expected_pngs = {Path(name).stem + ".png" for name in cameras}
    for kind in ("rgb", "mask"):
        require({p.name for p in (output / kind).iterdir()} == expected_pngs, "Unexpected prediction files")
    for prediction in predictions:
        camera = cameras[prediction["name"]]
        for kind, mode in (("rgb", "RGB"), ("mask", "L")):
            path = output / kind / (Path(prediction["name"]).stem + ".png")
            require(Path(prediction[kind]).resolve() == path, "Prediction path escaped expected output")
            require(digest(path) == prediction[kind + "_sha256"], "PNG changed after scoring")
            with Image.open(path) as image:
                require(image.format == "PNG" and image.mode == mode
                        and image.size == (camera["width"], camera["height"]), "Unexpected delivered PNG shape/type")
                image.verify()
    fields = ("camera", "source_image_sha256", "source_annotation_sha256", "rasterized_mask_sha256")
    source_records = receipt["source_records"]
    require(len(source_records) == 50 and {r["camera"]["name"] for r in source_records} == set(cameras),
            "Incomplete scoring source records")
    require(all(r["camera"] == cameras[r["camera"]["name"]] for r in source_records), "Scored camera changed")
    require(official.official_fingerprint([{k: r[k] for k in fields} for r in source_records])
            == metrics["official_evaluation_fingerprint"], "Scoring fingerprint is inconsistent")
    targets = {target["name"]: target for target in reference["targets"]}
    for record in source_records:
        target = targets[record["camera"]["name"]]
        for path_key, hash_key in (("source_image_path", "source_image_sha256"),
                                   ("source_annotation_path", "source_annotation_sha256")):
            require(record[path_key] == target[path_key], "Scoring data path changed")
            if target[path_key] is not None:
                # Source payload hashing is deliberately after this arm's full
                # prediction+scoring pass; the model receives none of these bytes.
                bind(registry, target[path_key], record[hash_key])
            else:
                require(record[hash_key] is None, "Unannotated view acquired a semantic target")
    return {"receipt": str(receipt_path), "receipt_sha256": bind(registry, receipt_path),
            "metrics_sha256": bind(registry, output / "official_metrics.json"),
            "fingerprint": metrics["official_evaluation_fingerprint"], "runtime_versions": receipt["runtime_versions"],
            "png_files_checked": 100}


def run_pair(root, reserved_bytes, handoff_note):
    root = Path(root).resolve()
    bundle = root / BUNDLE
    receipt_path = bundle / "pair_execution_receipt.json"
    # x mode prevents overwriting a previous failed or completed outer attempt.
    with receipt_path.open("x") as stream:
        stream.write("{}\n")
    registry = {}
    receipt = {"status": "preflight", "started_utc": utc(), "pid": os.getpid(),
               "gpu_handoff_note": handoff_note, "commands": [], "bound_input_sha256": registry}
    write_receipt(receipt_path, receipt)
    try:
        plan_path = bundle / "execution_plan.json"
        plan = read_json(plan_path)
        require(Path(plan["workspace_root"]) == root and Path(plan["source_snapshot"]) == bundle / "source_snapshot",
                "This runner only accepts the fixed local pair")
        require([a["arm"] for a in plan["inputs"]] == ["legacy_support", "corner_v2"], "Unexpected pair")
        require(all(Path(a["output"]) == bundle / a["arm"] for a in plan["inputs"]), "Unexpected output paths")
        require_empty_outputs(plan)
        # This precedes all real checkpoint hashing/loading, including legacy.
        records = completed_endpoints(root, plan)
        bind(registry, __file__)
        bind(registry, plan_path)
        frozen = check_sources(plan, registry)
        bind(registry, root / "uv.lock", frozen["uv_lock_sha256"])
        bind(registry, plan["reference_manifest"]["path"], plan["reference_manifest"]["sha256"])
        receipt["space_preflight"] = check_space(root, reserved_bytes)
        receipt["gpu_preflight"] = check_gpu_idle()
        modules = frozen_modules(Path(plan["source_snapshot"]))
        official = modules["official_evaluate"]
        receipt["lpips_weights"] = lpips_weights(registry)
        reference, inputs = check_training(root, plan, records, registry, official)
        receipt["inputs"] = inputs
        spec = importlib.util.spec_from_file_location("frozen_official_comparison", bundle / "scripts/compare_official_evaluations.py")
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        env = dict(os.environ, PYTHONPATH=plan["source_snapshot"], OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
        receipt["environment_overrides"] = {key: env[key] for key in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        receipt["python_executable"] = sys.executable

        def execute(label, command):
            log = bundle / f"{label}.log"
            item = {"label": label, "command": command, "cwd": str(root), "log": str(log), "started_utc": utc()}
            receipt["commands"].append(item)
            with log.open("x") as stream:
                process = subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
                item["pid"] = process.pid
                write_receipt(receipt_path, receipt)
                item["returncode"] = process.wait()
            item["finished_utc"] = utc()
            write_receipt(receipt_path, receipt)
            require(item["returncode"] == 0, f"{label} exited {item['returncode']}; artifacts preserved")

        receipt["status"] = "running"
        receipt["evaluations"] = []
        for arm in inputs:
            check_gpu_idle()
            # The initial 1 GiB check includes the whole pair budget; retain this
            # remaining-write check after the first arm has consumed space.
            need = (1 << 30) if not receipt["evaluations"] else SECOND_ARM_MINIMUM
            check_space(root, reserved_bytes, minimum=need)
            execute(arm["arm"], [sys.executable, str(bundle / "scripts/evaluate_official.py"), arm["checkpoint"],
                    "--output", arm["output"], "--workspace-root", str(root),
                    "--reference-manifest", plan["reference_manifest"]["path"]])
            receipt["evaluations"].append(audit_output(arm, reference, frozen, official, comparison, registry))
            write_receipt(receipt_path, receipt)
        first, second = receipt["evaluations"]
        require(first["fingerprint"] == second["fingerprint"] and first["runtime_versions"] == second["runtime_versions"],
                "The two complete evaluations do not share scoring inputs/environment")
        execute("comparison", [sys.executable, str(bundle / "scripts/compare_official_evaluations.py"),
                str(Path(inputs[0]["output"]) / "official_metrics.json"),
                str(Path(inputs[1]["output"]) / "official_metrics.json"),
                "--output", plan["comparison_output"], "--repeats", "5000"])
        receipt["comparison_sha256"] = bind(registry, plan["comparison_output"])
        for path, expected in list(registry.items()):
            bind(registry, path, expected)
        receipt.update(status="completed", finished_utc=utc(), final_input_bytes_unchanged=True)
    except BaseException as error:
        receipt.update(status="failed", finished_utc=utc(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_receipt(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True,
                        help="Only use after the root agent explicitly hands over an idle GPU")
    parser.add_argument("--reserved-bytes", type=int, required=True,
                        help="Outstanding disk writes reserved for other jobs; explicitly use 0 if none")
    parser.add_argument("--gpu-handoff-note", required=True)
    args = parser.parse_args()
    require(args.gpu_handoff_note.strip(), "Record the explicit GPU handoff")
    run_pair(ROOT, args.reserved_bytes, args.gpu_handoff_note)


if __name__ == "__main__":
    main()
