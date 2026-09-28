"""Evaluate three predeclared fixed 0.5 horizontal refiner-TTA controls."""

import argparse
import gc
import hashlib
import importlib
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from compare_evaluations import paired_comparison

from bridge_rgs.coordinates import LEGACY, pixel_protocol
from bridge_rgs.evaluate import evaluate_scene, evaluation_fingerprint
from bridge_rgs.io import load_manifest
from bridge_rgs.train import load_scene


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(record):
    if digest(record["path"]) != record["sha256"]:
        raise ValueError(f"Input changed: {record['path']}")


def verify_plan_inputs(plan):
    for record in plan["input_hashes"].values():
        verify(record)
    for arm in plan["arms"].values():
        for record in arm["reference_images"].values():
            verify(record)


@torch.inference_mode()
def evaluate_arm(name, arm, manifest, output):
    reference = json.loads(Path(arm["evaluation"]).read_text())
    views = [view for view in manifest["views"] if view["split"] == "val"]
    assert len(views) == 50 and sum(bool(view.get("mask_path")) for view in views) == 41
    scene, state = load_scene(arm["checkpoint"])
    assert pixel_protocol(state) == pixel_protocol(manifest) == LEGACY
    assert scene.pixel_protocol == LEGACY
    assert evaluation_fingerprint(views, 1.0, LEGACY, manifest["_manifest_sha256"]) == reference["evaluation_fingerprint"]
    scene.eval().requires_grad_(False)
    before = {key: value.detach().cpu().clone() for key, value in scene.state_dict().items()}
    training_cameras = state["training_cameras"].clone()
    plain_folder = output / "plain_predictions"
    plain_folder.mkdir(parents=True)
    original_render = scene.render
    rows = []

    def audited_render(K, w2c, width, height, _render=original_render, **kwargs):
        # Standard evaluate_scene has opened scoring data, but only camera
        # parameters enter render. No GT/valid/photo is passed to this predictor.
        view = views[len(rows)]
        assert (width, height) == (view["width"], view["height"]) == (1320, 989)
        assert torch.equal(K, torch.tensor(view["K"], dtype=K.dtype, device=K.device))
        assert torch.equal(w2c, torch.tensor(view["w2c_original"], dtype=w2c.dtype, device=w2c.device))
        rendered = _render(K, w2c, width, height, **kwargs)
        plain = rendered["probabilities"].argmax(-1).byte().cpu().numpy()
        stem = Path(view["name"]).stem
        old_mask = arm["reference_images"][stem + "_mask.png"]
        old_rgb = arm["reference_images"][stem + "_rgb.png"]
        mask_reference = cv2.imread(old_mask["path"], cv2.IMREAD_UNCHANGED)
        assert mask_reference is not None and np.array_equal(plain, mask_reference)
        rgb = (rendered["rgb"].clamp(0, 1).cpu().numpy()[..., ::-1] * 255).round().astype(np.uint8)
        rgb_reference = cv2.imread(old_rgb["path"], cv2.IMREAD_COLOR)
        assert rgb_reference is not None and np.array_equal(rgb, rgb_reference)
        destination = plain_folder / (stem + "_mask.png")
        assert cv2.imwrite(str(destination), plain)
        rows.append({"name": view["name"], "pre_tta_argmax_exact_to_reference": True,
                     "fresh_rgb_exact_to_reference": True,
                     "plain_prediction_sha256": digest(destination),
                     "reference_mask_sha256": old_mask["sha256"], "reference_rgb_sha256": old_rgb["sha256"]})
        if len(rows) % 10 == 0:
            print(f"{name}: {len(rows)}/50 native cameras", flush=True)
        return rendered

    scene.render = audited_render
    torch.cuda.reset_peak_memory_stats()
    try:
        result = evaluate_scene(scene, manifest, output / "evaluation_native", scale=1.0,
                                lpips_metric=True, checkpoint_pixel_protocol=state, refiner_flip_tta=True)
    finally:
        scene.render = original_render
    assert len(rows) == 50
    assert result["validation_views"] == 50 and result["semantic_validation_views"] == 41
    assert all(torch.equal(value.detach().cpu(), before[key]) for key, value in scene.state_dict().items())
    assert torch.equal(state["training_cameras"], training_cameras)
    assert result["confusion_matrix_3d"] == reference["confusion_matrix_3d"]
    old_views = {view["name"]: view for view in reference["views"]}
    assert set(old_views) == {view["name"] for view in result["views"]}
    for row in result["views"]:
        old = old_views[row["name"]]
        assert row.get("confusion_matrix_3d") == old.get("confusion_matrix_3d")
        rgb_name = Path(row["name"]).stem + "_rgb.png"
        assert digest(output / "evaluation_native" / rgb_name) == arm["reference_images"][rgb_name]["sha256"]
        assert all(row[key] == old[key] for key in ("psnr", "ssim", "lpips"))
    comparison = paired_comparison(reference, result)
    comparison.update(reference_source=arm["evaluation"],
                      candidate_source=str((output / "evaluation_native/metrics.json").resolve()),
                      comparison="fixed horizontal TTA minus the same checkpoint's plain prediction")
    (output / "paired_tta_minus_plain.json").write_text(json.dumps(comparison, indent=2) + "\n")
    audit = {"status": "completed", "pre_tta_all_50_argmax_pngs_exact": True,
             "all_50_rgb_pngs_and_rgb_metrics_exact": True,
             "global_and_per_view_raw_confusion_exact": True, "all_model_tensors_and_cameras_unchanged": True,
             "legacy_profile_and_fingerprint_preserved": True, "views": rows,
             "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
             "metrics_sha256": digest(output / "evaluation_native/metrics.json"),
             "gt_policy": "Standard scorer opens RGB/GT/valid for scoring. Only camera parameters reach scene.render; TTA consumes rendered evidence only. No claim that scoring files remain unopened.",
             "timing_policy": "Scorer render_seconds includes the extra head and audit PNG/readback work; not an isolated deployment benchmark."}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    summary = {key: result[key] for key in ("miou_all", "miou_foreground", "iou", "boundary_f1_2px",
                                          "psnr", "ssim", "lpips", "semantic_3d")}
    summary.update(plain_miou_all=reference["miou_all"], paired=comparison["metrics"],
                   audit_sha256=digest(output / "audit.json"))
    del audited_render, scene, state, before, original_render
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/generated_refiner_flip_tta/plan.json"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    plan = json.loads(args.plan.read_text())
    assert list(plan["arms"]) == ["base", "00_control", "01_flip"]
    assert plan["protocol"] == "fixed_refiner_horizontal_tta_v1"
    assert plan["inference"] == {"scale": 1.0, "lpips": True, "horizontal_flip_probability_mean": .5,
                                  "teacher": False, "optimization_steps": 0}
    snapshot = Path(plan["source_snapshot"]).resolve()
    for key, sha in plan["source_hashes"].items():
        verify({"path": snapshot / key, "sha256": sha})
    imports = {}
    for name in ("bridge_rgs.model", "bridge_rgs.train", "bridge_rgs.refinement", "bridge_rgs.evaluate",
                 "bridge_rgs.refiner_tta", "bridge_rgs.coordinates", "bridge_rgs.io"):
        path = Path(importlib.import_module(name).__file__).resolve()
        assert path.is_relative_to(snapshot)
        imports[name] = {"path": str(path), "sha256": digest(path)}
    verify({"path": __file__, "sha256": plan["runner_sha256"]})
    verify_plan_inputs(plan)
    output = Path(plan["output"])
    receipt_path = output / "execution_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Refusing to overwrite an executed TTA experiment")
    shutil.copy2(__file__, output / "evaluate_refiner_flip_tta_executed.py")
    shutil.copy2(args.plan, output / "plan.json")
    receipt = {"status": "running", "protocol": plan["protocol"], "pid": os.getpid(),
               "plan_sha256": digest(args.plan), "actual_imports": imports,
               "source_hashes": plan["source_hashes"], "input_hashes": plan["input_hashes"],
               "runner_sha256": digest(__file__), "arms_completed": []}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        manifest = load_manifest(plan["manifest"])
        results = {}
        for name, arm in plan["arms"].items():
            original_receipt = json.loads(Path(arm["experiment_receipt"]).read_text())
            assert original_receipt["status"] == "completed"
            assert digest(arm["checkpoint"]) == original_receipt["checkpoint_sha256"]
            assert digest(arm["evaluation"]) == original_receipt["evaluation_sha256"]
            results[name] = evaluate_arm(name, arm, manifest, output / name)
            receipt["arms_completed"].append(name)
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        verify_plan_inputs(plan)
        for key, sha in plan["source_hashes"].items():
            verify({"path": snapshot / key, "sha256": sha})
        report = {"status": "completed", "protocol": plan["protocol"], "plan_sha256": digest(args.plan),
                  "all_three_predeclared_arms": results,
                  "no_training_or_weight_search": True,
                  "scope": "Standard inference augmentation engineering control. Fixed development split, one checkpoint per predeclared arm. No novelty or blind-test claim; paired views do not cover cross-seed or cross-scene uncertainty."}
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        receipt.update(status="completed", report_sha256=digest(output / "report.json"))
        print(json.dumps({key: value["miou_all"] for key, value in results.items()}), flush=True)
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
