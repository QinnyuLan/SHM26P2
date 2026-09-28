"""Fixed two-arm appearance polish on native versus original distorted TRAIN RGB.

Prepare once, then execute the copied runner with its copied package on PYTHONPATH.
Only final compact color deltas are written; this runner does not support resume.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch

BASE_SHA = "a77d304f32de4356c1a4608ed5192258b6a9ced4c67a4297809f439481408d89"
MANIFEST_SHA = "91b41aeecedc352e4882eb80a1bb635d479ae432251a4c96c85a9dc110cbb327"
PROTOCOL = "appearance_native_vs_original_v1"
ALLOWED = ("splats.sh0", "splats.sh_rest", "background_logits")
ARMS = ("00_native", "01_original")
SPEC = {
    "steps": 3000, "view_seed": 42, "train_views": 350,
    "lr": {"splats.sh0": .00025, "splats.sh_rest": .0000125, "background_logits": .0001},
    "optimizer": "Adam; betas=(0.9,0.999); eps=1e-15; no weight decay or schedule",
    "trainable": list(ALLOWED), "prediction_clamp": [0., 1.],
    "loss": ".8 masked RGB L1 + .2 masked (1-7box SSIM); complete valid 7x7 SSIM centers",
    "support": "native: prepared valid; original: entire original image; no label/GT replacement",
    "renderer": "same fixed antialiased pinhole overscan; native crop or continuous-float-map four-tap warp matching installed OpenCV",
    "warp_weight_policy": "continuous_float32",
    "evaluation": "fixed final only, common original_grid_v1 all 50/41; raw geometry/semantic weights frozen",
    "forward_cv2_atol": 2e-6, "native_crop_atol": 2e-5,
    "max_training_seconds_per_arm": 1800, "checkpoint_optimizer_saved": False,
    "minimum_initial_free_bytes": 1152 << 20, "free_space_floor_bytes": 256 << 20,
    "engineering_gate": {"original_minus_native_and_base_psnr_db_min": .15,
                         "both_psnr_paired_interval_lower_positive": True,
                         "ssim_point_non_decreasing": True, "lpips_point_non_increasing": True,
                         "all5_and_cable_max_drop_from_base_pp": .20},
    "claim_boundary": "engineering objective/grid comparison, not isolated interpolation effect or novelty",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def require(value, message):
    if not value:
        raise ValueError(message)


def fixed_training_views(manifest):
    from bridge_rgs.coordinates import CORNER, pixel_protocol
    require(pixel_protocol(manifest) == CORNER, "Require corner_v2 manifest")
    views = sorted((v for v in manifest["views"] if v["split"] == "train"), key=lambda v: v["name"])
    require(len(views) == SPEC["train_views"], "Require all 350 TRAIN views")
    require(len({v["name"] for v in views}) == len(views), "Duplicate TRAIN names")
    for view in views:
        require(np.array_equal(view["w2c"], view["w2c_original"]), "This experiment fixes original cameras")
        require(all(view.get(key) for key in ("image_path", "valid_path", "source_image_path")),
                "Missing TRAIN RGB/valid source")
    return views


def training_order(count, steps, seed):
    generator = random.Random(seed)
    order = []
    while len(order) < steps:
        cycle = list(range(count))
        generator.shuffle(cycle)
        order.extend(cycle)
    return order[:steps]


def original_training_poses(manifest):
    """Checkpoint cameras use original manifest TRAIN ordering, not sorted/VAL views."""
    return torch.tensor([view["w2c_original"] for view in manifest["views"]
                         if view["split"] == "train"], dtype=torch.float32)


def tensor_hash(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    h = hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def gpu_idle():
    document = ET.fromstring(subprocess.run(["nvidia-smi", "-q", "-x"], check=True,
                                           capture_output=True, text=True).stdout)
    require(document.findall("gpu"), "No GPU")
    for gpu in document.findall("gpu"):
        listing = gpu.find("processes")
        require(listing is not None and (listing.text or "").strip() not in {"N/A", "Not Supported"},
                "Cannot determine GPU clients")
    rows = [{key: node.findtext(key) for key in ("pid", "type", "process_name")}
            for node in document.findall(".//process_info")]
    require(all(row["type"] == "G" for row in rows), "Compute GPU already occupied")
    return {"utc": utc(), "graphics_only_clients": rows, "compute_clients": 0}


def bind_inputs(plan):
    for path, expected in plan["input_hashes"].items():
        require(digest(path) == expected, f"Input changed: {path}")
    for relative, expected in plan["source_hashes"].items():
        require(digest(Path(plan["snapshot"]) / relative) == expected, f"Frozen source changed: {relative}")


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), "Refuse existing experiment directory")
    base = root / "runs/corner_v2_semantic_coupled/last.pt"
    manifest_path = root / "artifacts/prepared_corner_v2/manifest.json"
    receipt_path = root / "runs/official_common_grid_v1/corner_v2/execution_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt["status"] == "completed" and receipt["checkpoint_sha256"] == BASE_SHA,
            "Require completed original-grid v2 base evaluation")
    require(digest(base) == BASE_SHA and digest(manifest_path) == MANIFEST_SHA, "Base provenance mismatch")
    manifest = json.loads(manifest_path.read_text())
    views = fixed_training_views(manifest)
    inputs = {str(p): digest(p) for p in (base, manifest_path, receipt_path, root / "uv.lock",
              receipt_path.parent / "official_metrics.json", root / "docs/raw_grid_appearance_experiment.md",
              root / "artifacts/raw_warp_cpu_contract.json")}
    require(inputs[str(receipt_path.parent / "official_metrics.json")] == receipt["official_metrics_sha256"],
            "Base metrics changed")
    for view in views:
        for key in ("image_path", "source_image_path", "valid_path"):
            path = (root / view[key]).resolve()
            inputs[str(path)] = digest(path)
    require(shutil.disk_usage(output.parent).free >= SPEC["minimum_initial_free_bytes"], "Insufficient disk budget")
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    shutil.copytree(root / "src/bridge_rgs", snapshot / "bridge_rgs",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("run_raw_grid_appearance.py", "compare_official_evaluations.py"):
        shutil.copy2(root / "scripts" / name, snapshot / name)
    sources = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))}
    # Assert the package did not change during the copy.
    for relative, value in sources.items():
        origin = root / "src" / relative if relative.startswith("bridge_rgs/") else root / "scripts" / relative
        require(digest(origin) == value, f"Source changed while snapshotting: {relative}")
    plan = {"status": "locked", "protocol": PROTOCOL, "created_utc": utc(), "root": str(root),
            "output": str(output), "snapshot": str(snapshot), "base": str(base),
            "manifest": str(manifest_path), "base_metrics": str(receipt_path.parent / "official_metrics.json"),
            "specification": SPEC, "arms": list(ARMS), "source_hashes": sources, "input_hashes": inputs,
            "view_names": [v["name"] for v in views],
            "training_order": training_order(len(views), SPEC["steps"], SPEC["view_seed"]),
            "study_limits": "one bridge, adaptive development; not organizer scoring/peer Dev30 or independent seeds"}
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def read_rgb(path, width, height):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    require(image is not None and image.shape == (height, width, 3), f"Invalid RGB image: {path}")
    return torch.from_numpy(image[..., ::-1].copy()).to("cuda", torch.float32) / 255


def layouts_for_views(manifest, views):
    from bridge_rgs.raw_grid import build_raw_grid
    layouts, by_camera = [], {}
    for view in views:
        source = manifest["source_cameras"][str(view["camera_id"])]
        require((view["width"], view["height"]) == (source["width"], source["height"])
                and np.array_equal(np.asarray(view["K"], np.float32), np.asarray(source["K"], np.float32)),
                "Minimal experiment requires identical source/prepared K and size")
        key = str(view["camera_id"])
        if key not in by_camera:
            layout = build_raw_grid(source["K"], source["opencv_distortion"], source["width"], source["height"])
            require(layout.warp.weight_policy == SPEC["warp_weight_policy"], "Installed OpenCV float-map policy changed")
            layout.warp.to("cuda")
            by_camera[key] = layout
        layouts.append(by_camera[key])
    return layouts, {key: layout.warp.receipt() for key, layout in by_camera.items()}


def load_target(view, arm, root):
    path = view["image_path"] if arm == "00_native" else view["source_image_path"]
    rgb = read_rgb(Path(root) / path, view["width"], view["height"])
    if arm == "00_native":
        valid = cv2.imread(str(Path(root) / view["valid_path"]), cv2.IMREAD_GRAYSCALE)
        require(valid is not None and valid.shape == (view["height"], view["width"]), "Invalid native validity")
        valid = torch.from_numpy(valid > 0).to("cuda")
    else:
        valid = torch.ones(rgb.shape[:2], dtype=torch.bool, device="cuda")
    return rgb, valid


def render_training_rgb(scene, layout, pose):
    K = torch.tensor(layout.render_K, device="cuda")
    return scene.render(K, pose, layout.render_width, layout.render_height,
                        degree=scene.sh_degree, semantics=False, absgrad=False)["rgb"].clamp(0, 1)


def predict_arm(canvas, layout, arm):
    return layout.crop_native(canvas) if arm == "00_native" else layout.warp(canvas)


def smoke(scene, views, layouts, root):
    """One fixed TRAIN view; no update and no decoded labels or VAL source data."""
    from bridge_rgs.evaluate import distortion_render_grid
    from bridge_rgs.raw_grid import appearance_rgb_loss
    view, layout = views[0], layouts[0]
    pose = torch.tensor(view["w2c_original"], device="cuda", dtype=torch.float32)
    with torch.no_grad():
        canvas = render_training_rgb(scene, layout, pose)
        direct = scene.render(torch.tensor(view["K"], device="cuda"), pose,
                              view["width"], view["height"], semantics=False, absgrad=False)["rgb"].clamp(0, 1)
        crop_error = float((layout.crop_native(canvas) - direct).abs().max())
        require(crop_error <= SPEC["native_crop_atol"], "Overscan native crop differs from native render")
        manifest = json.loads((Path(root) / "artifacts/prepared_corner_v2/manifest.json").read_text())
        source = manifest["source_cameras"][str(view["camera_id"])]
        _, _, _, mapping = distortion_render_grid(source["K"], source["opencv_distortion"],
                                                  view["width"], view["height"], "colmap_corner_v2")
        array = canvas.cpu().numpy()
        expected = (array if mapping is None else cv2.remap(array, mapping[..., 0], mapping[..., 1],
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        error = float(np.max(np.abs(expected - layout.warp(canvas).cpu().numpy())))
        require(error <= SPEC["forward_cv2_atol"], "Differentiable warp differs from official OpenCV float RGB")
    gradients = {}
    for arm in ARMS:
        scene.zero_grad(set_to_none=True)
        target, valid = load_target(view, arm, root)
        prediction = predict_arm(render_training_rgb(scene, layout, pose), layout, arm)
        loss, _ = appearance_rgb_loss(prediction, target, valid)
        loss.backward()
        grads = {name: float(parameter.grad.abs().sum()) for name, parameter in scene.named_parameters()
                 if parameter.grad is not None}
        require(set(grads) == set(ALLOWED) and all(np.isfinite(v) and v > 0 for v in grads.values()),
                "Appearance gradient permission/finite/nonzero check failed")
        gradients[arm] = {"loss": float(loss.detach()), "gradient_l1": grads}
    scene.zero_grad(set_to_none=True)
    return {"view": view["name"], "native_crop_max_abs": crop_error, "cv2_float_max_abs": error,
            "gradients": gradients, "optimizer_steps": 0, "labels_decoded": 0, "val_pixels_read": 0}


def train_arm(plan, arm):
    from bridge_rgs.checkpoints import load_checkpoint, save_appearance_delta
    from bridge_rgs.raw_grid import appearance_rgb_loss
    from bridge_rgs.train import load_scene
    output = Path(plan["output"]) / arm
    require(not output.exists(), f"Refuse existing arm: {arm}")
    output.mkdir()
    receipt = {"status": "starting", "arm": arm, "protocol": PROTOCOL, "started_utc": utc(),
               "base_checkpoint_sha256": BASE_SHA, "manifest_sha256": MANIFEST_SHA,
               "specification": SPEC, "plan_sha256": plan["_plan_sha256"],
               "checkpoint_saved": "final inference delta only; no optimizer/RNG/resume state"}
    write_json(output / "training_receipt.json", receipt)
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        scene, base_state = load_scene(plan["base"])
        scene.eval()
        for name, parameter in scene.named_parameters():
            parameter.requires_grad_(name in ALLOWED)
        require({name for name, p in scene.named_parameters() if p.requires_grad} == set(ALLOWED),
                "Unexpected trainable model parameters")
        before = {name: tensor_hash(value) for name, value in scene.state_dict().items()}
        manifest = json.loads(Path(plan["manifest"]).read_text())
        views = fixed_training_views(manifest)
        require([v["name"] for v in views] == plan["view_names"], "TRAIN names changed")
        poses = torch.tensor([v["w2c_original"] for v in views], device="cuda", dtype=torch.float32)
        original_poses = original_training_poses(manifest)
        require(torch.equal(base_state["training_cameras"], original_poses), "Base cameras are not the fixed originals")
        layouts, warp_records = layouts_for_views(manifest, views)
        smoke_result = smoke(scene, views, layouts, plan["root"])
        require(before == {n: tensor_hash(v) for n, v in scene.state_dict().items()}, "Smoke modified model state")
        receipt.update(status="training", smoke=smoke_result, warps=warp_records,
                       frozen_tensor_hashes={k: v for k, v in before.items() if k not in ALLOWED})
        write_json(output / "training_receipt.json", receipt)
        named = dict(scene.named_parameters())
        optimizer = torch.optim.Adam([{"params": [named[name]], "lr": SPEC["lr"][name]} for name in ALLOWED], eps=1e-15)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        actual_names = []
        with (output / "train.jsonl").open("x") as log:
            for step, index in enumerate(plan["training_order"], 1):
                view, layout = views[index], layouts[index]
                actual_names.append(view["name"])
                target, valid = load_target(view, arm, plan["root"])
                optimizer.zero_grad(set_to_none=True)
                prediction = predict_arm(render_training_rgb(scene, layout, poses[index]), layout, arm)
                loss, stats = appearance_rgb_loss(prediction, target, valid)
                require(bool(torch.isfinite(loss)), "Nonfinite appearance loss")
                loss.backward()
                if step == 1 or step % 100 == 0:
                    require(all(named[name].grad is not None and bool(torch.isfinite(named[name].grad).all())
                                for name in ALLOWED), "Invalid appearance gradients")
                optimizer.step()
                if step == 1 or step % 100 == 0:
                    seconds = time.monotonic() - started
                    record = {"step": step, "view": view["name"], "loss": float(loss.detach()),
                              "seconds": seconds, **{k: float(v) for k, v in stats.items()}}
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                    print(json.dumps({"arm": arm, **record}), flush=True)
                    require(seconds <= SPEC["max_training_seconds_per_arm"], "Training exceeded fixed time budget")
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        state = scene.state_dict()
        after = {name: tensor_hash(value) for name, value in state.items()}
        require(all(before[k] == after[k] for k in before if k not in ALLOWED), "Frozen model tensor changed")
        require(all(before[k] != after[k] and bool(torch.isfinite(state[k]).all()) for k in ALLOWED),
                "Appearance parameter missing update or became nonfinite")
        require(shutil.disk_usage(output).free >= (400 << 20), "Insufficient compact checkpoint/evaluation space")
        checkpoint = output / "appearance.pt"
        save_appearance_delta(checkpoint, base_checkpoint=plan["base"], base_sha256=BASE_SHA,
                              model={k: state[k].detach().cpu() for k in ALLOWED}, manifest_path=plan["manifest"],
                              experiment=PROTOCOL, config={"arm": arm, **SPEC, "parameter_scope": "appearance_only",
                              "manifest": base_state["config"]["manifest"], "pixel_protocol": "colmap_corner_v2"}, step=SPEC["steps"])
        reconstructed = load_checkpoint(checkpoint)
        require(set(reconstructed["model"]) == set(state) and
                all(torch.equal(reconstructed["model"][k], v.detach().cpu()) for k, v in state.items()),
                "Compact checkpoint failed exact full-model reconstruction")
        require(torch.equal(reconstructed["training_cameras"], base_state["training_cameras"]), "Delta changed cameras")
        receipt.update(status="completed", finished_utc=utc(), steps=SPEC["steps"], training_seconds=seconds,
                       checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
                       peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                       appearance_before={k: before[k] for k in ALLOWED}, appearance_after={k: after[k] for k in ALLOWED},
                       frozen_tensors_exact=True, reconstructed_model_exact=True, original_training_cameras_exact=True,
                       view_sequence_sha256=hashlib.sha256(json.dumps(actual_names).encode()).hexdigest(),
                       labels_decoded_during_training=0, val_pixels_read_during_training=0)
        write_json(output / "training_receipt.json", receipt)
        return checkpoint
    except Exception as exc:
        receipt.update(status="failed", failed_utc=utc(), error=f"{type(exc).__name__}: {exc}")
        write_json(output / "training_receipt.json", receipt)
        raise


def execute(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    require(plan["status"] == "locked" and plan["protocol"] == PROTOCOL and plan["specification"] == SPEC,
            "Unrecognized or modified experiment specification")
    require(plan["arms"] == list(ARMS) and plan["training_order"] == training_order(350, 3000, 42),
            "Fixed arms or training sequence differs")
    snapshot = Path(plan["snapshot"])
    require(Path(__file__).resolve() == snapshot / "run_raw_grid_appearance.py", "Run the frozen runner")
    import bridge_rgs
    require(Path(bridge_rgs.__file__).resolve().parent == snapshot / "bridge_rgs", "Run the frozen package")
    os.chdir(plan["root"])
    output = Path(plan["output"])
    receipt_path = output / "execution_receipt.json"
    require(not receipt_path.exists() and all(not (output / arm).exists() for arm in ARMS), "Refuse to rerun existing experiment")
    bind_inputs(plan)
    require(shutil.disk_usage(output).free >= SPEC["minimum_initial_free_bytes"], "Insufficient initial disk budget")
    handoff = gpu_idle()
    plan["_plan_sha256"] = digest(plan_path)
    receipt = {"status": "running", "protocol": PROTOCOL, "started_utc": utc(),
               "plan_sha256": plan["_plan_sha256"], "gpu_handoff": handoff, "arms": []}
    write_json(receipt_path, receipt)
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    try:
        from compare_official_evaluations import paired_official_comparison, read_completed

        from bridge_rgs.official_evaluate import evaluate_official
        for arm in ARMS:
            checkpoint = train_arm(plan, arm)
            gc.collect()
            torch.cuda.empty_cache()
            evaluation = output / arm / "evaluation_official"
            require(shutil.disk_usage(output).free >= (556 << 20), "Insufficient full evaluation + 256 MiB reserve")
            metrics = evaluate_official(checkpoint, evaluation, workspace_root=plan["root"], entrypoint=__file__)
            receipt["arms"].append({"arm": arm, "training_receipt_sha256": digest(checkpoint.parent / "training_receipt.json"),
                                    "checkpoint_sha256": digest(checkpoint), "metrics_sha256": digest(evaluation / "official_metrics.json"),
                                    "psnr": metrics["psnr"], "ssim": metrics["ssim"], "lpips": metrics["lpips"],
                                    "miou_all": metrics["miou_all"]})
            write_json(receipt_path, receipt)
            gc.collect()
            torch.cuda.empty_cache()
        comparisons = {}
        metric_paths = {"base": Path(plan["base_metrics"]), **{arm: output / arm / "evaluation_official/official_metrics.json" for arm in ARMS}}
        loaded = {key: read_completed(path) for key, path in metric_paths.items()}
        for reference, candidate in (("base", ARMS[0]), ("base", ARMS[1]), ARMS):
            # read_completed returns the validated metrics and provenance record.
            first, second = loaded[reference][0], loaded[candidate][0]
            result = paired_official_comparison(first, second)
            result["inputs"] = {"reference": loaded[reference][1], "candidate": loaded[candidate][1]}
            path = output / f"paired_{candidate}_minus_{reference}.json"
            write_json(path, result)
            comparisons[path.name] = digest(path)
        bind_inputs(plan)
        require(digest(plan_path) == plan["_plan_sha256"], "Plan changed during execution")
        require(shutil.disk_usage(output).free >= SPEC["free_space_floor_bytes"], "Free-space reserve violated")
        receipt.update(status="completed", finished_utc=utc(), comparisons=comparisons,
                       all_bound_inputs_unchanged=True, free_disk_bytes=shutil.disk_usage(output).free)
        write_json(receipt_path, receipt)
    except Exception as exc:
        receipt.update(status="failed", failed_utc=utc(), error=f"{type(exc).__name__}: {exc}")
        write_json(receipt_path, receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/raw_grid_appearance_v1"))
    args = parser.parse_args()
    if args.prepare:
        path = prepare(args.root, args.output)
        print(json.dumps({"plan": str(path), "sha256": digest(path)}), flush=True)
    else:
        execute(args.run)


if __name__ == "__main__":
    main()
