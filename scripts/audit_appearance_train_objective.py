"""Read-only all-350-TRAIN audit of the two completed appearance objectives."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

SOURCE_PLAN_SHA = "a8f889c3c9d7678cb0a1376dfd6bb1f06423e1e470f3f3853991259832869a1d"

SPEC = {
    "protocol": "appearance_all_train_objective_audit_v1",
    "arms": ["00_native", "01_original"], "checkpoints_per_arm": ["base", "matching_final"],
    "train_views": 350, "renders": 1400, "optimizer_steps": 0,
    "view_order": "first complete 350-view cycle of the original locked 3000-step order",
    "metrics": ["l1", "one_minus_ssim7", "combined_loss", "mse"],
    "primary_reduction": "equal mean over all 350 TRAIN cameras",
    "secondary_reduction": "weight each camera by its frequency in the original 3000-step order",
    "prediction": "original frozen runner: antialiased overscan -> clamp(0,1) -> native crop or raw gather",
    "objective": "original frozen appearance_rgb_loss: .8 valid L1 + .2 complete-valid-7-window SSIM loss",
    "mse": "per-pixel mean RGB squared error over the same validity as L1, FP32",
    "source_policy": "only TRAIN RGB/source_RGB/native-valid pixel files; no semantic GT or VAL pixels",
    "max_wall_seconds": 240, "maximum_output_bytes": 10 << 20,
    "minimum_output_free_bytes": 256 << 20,
    "limits": "In-sample descriptive objective audit, not new training, causal proof, or a generalization/accuracy claim.",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value, *, replace=False):
    path = Path(path)
    if replace:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    else:
        with path.open("x") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fixed_views(manifest, original_plan):
    """Keep sorted pose indices and original first-cycle enumeration distinct."""
    require(manifest.get("pixel_protocol", {}).get("id") == "colmap_corner_v2"
            if isinstance(manifest.get("pixel_protocol"), dict)
            else manifest.get("pixel_protocol") == "colmap_corner_v2", "Require original corner-v2 manifest")
    views = sorted((v for v in manifest["views"] if v["split"] == "train"), key=lambda v: v["name"])
    require(len(views) == 350 and len({v["name"] for v in views}) == 350, "Require exactly 350 unique TRAIN views")
    require([v["name"] for v in views] == original_plan["view_names"], "Original TRAIN indexing changed")
    order = original_plan["training_order"]
    require(len(order) == 3000 and all(isinstance(v, int) and 0 <= v < 350 for v in order), "Invalid original sampling order")
    require(sorted(order[:350]) == list(range(350)), "First original cycle must contain all TRAIN views exactly once")
    for view in views:
        require(np.array_equal(view["w2c"], view["w2c_original"]), "Original pose mismatch")
    return views, order[:350], Counter(order)


def summarize_pair(base, final, sample_counts):
    require(len(base) == len(final) == 350, "Pair must include all 350 views")
    require([v["name"] for v in base] == [v["name"] for v in final], "Paired views differ")
    weights = np.asarray([sample_counts[str(v["sorted_index"])] for v in base], dtype=np.float64)
    require(weights.sum() == 3000 and np.all(weights > 0), "Original frequency weights changed")
    result = {}
    for key in SPEC["metrics"]:
        first = np.asarray([v[key] for v in base], dtype=np.float64)
        second = np.asarray([v[key] for v in final], dtype=np.float64)
        require(np.isfinite(first).all() and np.isfinite(second).all(), "Nonfinite audit metric")
        change = second-first
        result[key] = {"base_equal_camera_mean": float(first.mean()), "final_equal_camera_mean": float(second.mean()),
                       "final_minus_base_equal_camera_mean": float(change.mean()),
                       "base_original_sampling_weighted_mean": float(np.average(first, weights=weights)),
                       "final_original_sampling_weighted_mean": float(np.average(second, weights=weights)),
                       "final_minus_base_original_sampling_weighted_mean": float(np.average(change, weights=weights)),
                       "improved_views": int((change < 0).sum()), "worsened_views": int((change > 0).sum()),
                       "unchanged_views": int((change == 0).sum()), "delta_quantiles_0_25_50_75_100": np.quantile(change, [0, .25, .5, .75, 1]).tolist(),
                       "lower_is_better": True}
    return result


def pixel_reader(allowed_paths, read_counts, delegate):
    allowed = set(allowed_paths)
    def guarded(path, *args, **kwargs):
        resolved = str(Path(path).resolve())
        require(resolved in allowed, f"Pixel read outside TRAIN RGB/valid allow-list: {resolved}")
        read_counts[resolved] += 1
        return delegate(path, *args, **kwargs)
    return guarded


def measured_values(prediction, target, valid, loss_function, masked_mean):
    """Invoke the unmodified training loss; MSE is a separate descriptive metric."""
    loss, stats = loss_function(prediction, target, valid)
    mse = masked_mean((prediction-target).square().mean(-1), valid)
    values = {"combined_loss": float(loss), "mse": float(mse),
              **{key: float(stats[key]) for key in ("l1", "one_minus_ssim7")}}
    require(all(np.isfinite(v) for v in values.values()), "Nonfinite TRAIN objective")
    require(abs(values["combined_loss"]-(.8*values["l1"]+.2*values["one_minus_ssim7"])) < 1e-7, "Loss mixture changed")
    return {**values, "rgb_pixels": int(stats["rgb_pixels"]), "ssim7_centers": int(stats["ssim7_centers"])}


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), "Preserve any existing audit")
    original_path = root / "runs/raw_grid_appearance_v1/plan.json"
    original = json.loads(original_path.read_text())
    require(digest(original_path) == SOURCE_PLAN_SHA, "Require the authorized original experiment plan")
    execution_path = original_path.parent / "execution_receipt.json"
    execution = json.loads(execution_path.read_text())
    require(original["status"] == "locked" and original["protocol"] == "appearance_native_vs_original_v1", "Unexpected source experiment")
    require(execution["status"] == "completed" and execution["plan_sha256"] == digest(original_path), "Source experiment not completed/bound")
    source_snapshot = Path(original["snapshot"])
    for name, sha in original["source_hashes"].items():
        require(digest(source_snapshot/name) == sha, "Source experiment snapshot changed")
    manifest_path = Path(original["manifest"])
    require(digest(manifest_path) == original["input_hashes"][str(manifest_path)], "Manifest SHA changed")
    manifest = json.loads(manifest_path.read_text())
    views, order, counts = fixed_views(manifest, original)
    inputs = {str(p): digest(p) for p in (original_path, execution_path, manifest_path, Path(original["base"]), root/"uv.lock")}
    require(inputs[original["base"]] == original["input_hashes"][original["base"]], "Base changed")
    require(inputs[str(root/"uv.lock")] == original["input_hashes"][str(root/"uv.lock")], "Original dependency lock changed")
    image_paths = set()
    for view in views:
        for key in ("image_path", "valid_path", "source_image_path"):
            path = str((root/view[key]).resolve())
            require(path in original["input_hashes"], "TRAIN input not bound in original experiment")
            inputs[path] = digest(path)
            require(inputs[path] == original["input_hashes"][path], "TRAIN pixel input changed")
            image_paths.add(path)
    arms = {}
    for arm in SPEC["arms"]:
        receipt_path = original_path.parent/arm/"training_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        row = next(v for v in execution["arms"] if v["arm"] == arm)
        require(receipt["status"] == "completed" and receipt["steps"] == 3000 and receipt["arm"] == arm,
                "Require corresponding completed final appearance checkpoint")
        require(receipt["plan_sha256"] == digest(original_path) and digest(receipt_path) == row["training_receipt_sha256"], "Training receipt not bound")
        checkpoint = Path(receipt["checkpoint"])
        require(digest(checkpoint) == receipt["checkpoint_sha256"] == row["checkpoint_sha256"], "Final delta changed")
        require(receipt["base_checkpoint_sha256"] == inputs[original["base"]], "Delta base changed")
        inputs[str(receipt_path)] = digest(receipt_path)
        inputs[str(checkpoint)] = receipt["checkpoint_sha256"]
        arms[arm] = {"base": original["base"], "matching_final": str(checkpoint), "training_receipt": str(receipt_path)}
    output.parent.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(output.parent).free >= SPEC["minimum_output_free_bytes"], "Insufficient output reserve")
    output.mkdir(parents=True)
    snapshot = output/"source_snapshot"
    shutil.copytree(source_snapshot, snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    hashes = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))}
    require(all(hashes[name] == sha for name, sha in original["source_hashes"].items()), "Original snapshot copy changed")
    plan = {"status": "locked", "specification": SPEC, "root": str(root), "output": str(output),
            "snapshot": str(snapshot), "source_hashes": hashes, "source_original_hashes": original["source_hashes"],
            "runner_sha256": digest(__file__), "input_hashes": inputs, "original_plan": str(original_path),
            "manifest": str(manifest_path), "arms": arms, "allowed_pixel_paths": sorted(image_paths),
            "view_names_sorted": [v["name"] for v in views], "audit_order": order,
            "audit_view_names": [views[i]["name"] for i in order], "original_sample_counts": dict(counts),
            "no_GPU_during_prepare": True, "execution_authorization": "Root review of this frozen plan before one read-only GPU run"}
    write_json(output/"plan.json", plan)
    return output/"plan.json"


def verify(plan):
    require(plan["status"] == "locked" and plan["specification"] == SPEC, "Require fixed locked audit")
    snapshot = Path(plan["snapshot"])
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, "Run frozen audit script")
    actual = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))}
    require(actual == plan["source_hashes"], "Audit source SHA changed")
    for name in ("model", "train", "raw_grid", "checkpoints", "evaluate", "losses", "coordinates"):
        path = Path(importlib.import_module(f"bridge_rgs.{name}").__file__).resolve()
        require(path == snapshot/"bridge_rgs"/f"{name}.py", f"Wrong actual import: {name}")
    reference = importlib.import_module("run_raw_grid_appearance")
    require(Path(reference.__file__).resolve() == snapshot/"run_raw_grid_appearance.py", "Wrong original runner import")
    for path, sha in plan["input_hashes"].items():
        require(digest(path) == sha, f"Audit input changed: {path}")
    return reference


def execute(plan_path):
    started = time.monotonic()
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    output = Path(plan["output"])
    require(not (output/"report.json").exists() and not (output/"execution_receipt.json").exists(), "No audit rerun")
    os.chdir(plan["root"])
    reference = verify(plan)
    handoff = reference.gpu_idle()
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    manifest = json.loads(Path(plan["manifest"]).read_text())
    original = json.loads(Path(plan["original_plan"]).read_text())
    views, order, counts = fixed_views(manifest, original)
    require([v["name"] for v in views] == plan["view_names_sorted"] and order == plan["audit_order"], "Audit order changed")
    require({str(k): v for k, v in counts.items()} == plan["original_sample_counts"], "Sampling frequencies changed")
    receipt = {"status": "running", "pid": os.getpid(), "plan_sha256": digest(plan_path),
               "source_hashes": plan["source_hashes"], "gpu_handoff": handoff, "render_count": 0,
               "optimizer_steps": 0, "semantic_label_pixels_decoded": 0, "val_pixels_decoded": 0}
    write_json(output/"execution_receipt.json", receipt)
    read_counts = Counter()
    original_imread = cv2.imread
    cv2.imread = pixel_reader(plan["allowed_pixel_paths"], read_counts, original_imread)
    result = {"specification": SPEC, "arms": {}, "plan_sha256": receipt["plan_sha256"]}
    try:
        from bridge_rgs.losses import masked_mean
        from bridge_rgs.raw_grid import appearance_rgb_loss
        from bridge_rgs.train import load_scene
        layouts, warps = reference.layouts_for_views(manifest, views)
        poses = torch.tensor([v["w2c_original"] for v in views], device="cuda", dtype=torch.float32)
        original_poses = reference.original_training_poses(manifest)
        base_hashes = base_metadata = None
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for arm in SPEC["arms"]:
                rows = {}
                for checkpoint_kind in SPEC["checkpoints_per_arm"]:
                    checkpoint = plan["arms"][arm][checkpoint_kind]
                    scene, state = load_scene(checkpoint)
                    scene.eval().requires_grad_(False)
                    require(torch.equal(state["training_cameras"], original_poses), "Checkpoint cameras differ from fixed manifest ordering")
                    before = {k: reference.tensor_hash(v) for k, v in scene.state_dict().items()}
                    metadata = {key: state[key] for key in ("scene_scale", "feature_dim", "sh_degree", "refiner_config", "pixel_protocol", "config")}
                    if base_hashes is None:
                        base_hashes, base_metadata = before, metadata
                    require(metadata == base_metadata and set(before) == set(base_hashes), "Scene metadata/state keys changed")
                    require(all(before[k] == base_hashes[k] for k in before if k not in reference.ALLOWED), "Nonappearance model tensors differ")
                    rows[checkpoint_kind] = []
                    for position, index in enumerate(order, 1):
                        require(time.monotonic()-started < SPEC["max_wall_seconds"], "Audit reached fixed time limit")
                        view, layout = views[index], layouts[index]
                        target, valid = reference.load_target(view, arm, plan["root"])
                        canvas = reference.render_training_rgb(scene, layout, poses[index])
                        receipt["render_count"] += 1
                        prediction = reference.predict_arm(canvas, layout, arm)
                        values = measured_values(prediction, target, valid, appearance_rgb_loss, masked_mean)
                        rows[checkpoint_kind].append({"name": view["name"], "sorted_index": index, **values})
                        if position == 1 or position % 50 == 0:
                            print(json.dumps({"arm": arm, "checkpoint": checkpoint_kind, "views": position,
                                              "renders": receipt["render_count"], "seconds": time.monotonic()-started}), flush=True)
                    require(before == {k: reference.tensor_hash(v) for k, v in scene.state_dict().items()}, "Audit modified model tensor bytes")
                    require(all(p.grad is None for p in scene.parameters()), "Read-only scene unexpectedly has gradients")
                    del scene, state, prediction, canvas, target, valid
                    gc.collect()
                    torch.cuda.empty_cache()
                result["arms"][arm] = {"views": rows, "final_minus_base": summarize_pair(rows["base"], rows["matching_final"], plan["original_sample_counts"])}
        require(receipt["render_count"] == SPEC["renders"], "Expected exactly 1400 renders")
        require(all(result["arms"][arm]["views"]["base"][i][key] == result["arms"][arm]["views"]["matching_final"][i][key]
                    for arm in SPEC["arms"] for i in range(350) for key in ("rgb_pixels", "ssim7_centers")), "Pair support changed")
        verify(plan)
        require(digest(plan_path) == receipt["plan_sha256"], "Plan changed")
        result.update(warps=warps, read_counts=dict(read_counts), model_tensors_unchanged=True,
                      cameras_and_nonappearance_tensors_exact=True, total_seconds=time.monotonic()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(output/"report.json", result)
        require(sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) <= SPEC["maximum_output_bytes"], "Output exceeded fixed budget")
        receipt.update(status="completed", seconds=time.monotonic()-started, report_sha256=digest(output/"report.json"),
                       all_inputs_and_frozen_source_unchanged=True, model_tensors_unchanged=True)
    except Exception as error:
        receipt.update(status="failed", seconds=time.monotonic()-started, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cv2.imread = original_imread
        write_json(output/"execution_receipt.json", receipt, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("/mnt/data/SHM2026/runs/appearance_train_objective_audit"))
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.root, args.output))
    else:
        execute(args.run)


if __name__ == "__main__":
    main()
