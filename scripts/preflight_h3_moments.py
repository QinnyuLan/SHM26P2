"""Three short H3 training checks and an independent full-scene moment render."""

import argparse
import copy
import gc
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from bridge_rgs.refinement import MultiScaleRefinementHead, add_depth_moment_paths
from bridge_rgs.teacher import file_sha256
from bridge_rgs.train import load_scene

ROOT = Path(__file__).resolve().parents[1]


def error_stats(candidate, reference, valid=None):
    difference = (candidate - reference).abs()
    if valid is not None:
        difference = difference[valid.expand_as(difference)]
    return {"max_absolute": float(difference.max()), "mean_absolute": float(difference.mean()),
            "exact": torch.equal(candidate, reference)}


def verify_sources(plan, snapshot):
    for key, value in plan["source_hashes"].items():
        assert file_sha256(snapshot / key) == value
    assert file_sha256(plan["warmstart"]) == plan["warmstart_sha256"]
    assert file_sha256(plan["manifest"]) == plan["manifest_sha256"]


@torch.no_grad()
def render_audit(plan, folder):
    """Uses only one fixed TRAIN camera and frozen model tensors, never images/GT."""
    from gsplat import rasterization

    scene, initial = load_scene(plan["warmstart"])
    scene.eval().requires_grad_(False)
    manifest = json.loads(Path(plan["manifest"]).read_text())
    train_views = [view for view in manifest["views"] if view["split"] == "train"]
    index, view = next((i, view) for i, view in enumerate(train_views)
                       if view["name"] == plan["fixed_train_preflight_view"])
    assert view["split"] == "train"
    K = torch.tensor(view["K"], dtype=torch.float32, device="cuda")
    pose = initial["training_cameras"][index].cuda()
    width, height = view["width"], view["height"]
    baseline = scene.render(K, pose, width, height, absgrad=False, refinement_grad_to_field=False)
    original_head = scene.refiner
    base_config = copy.deepcopy(scene.refiner_config)
    s = scene.splats
    z = (s["means"] @ pose[2, :3] + pose[2, 3])[:, None] / max(scene.scene_scale, 1e-6)
    f = s["sem_features"]
    # A separate call, all Gaussians and their complete compositing process.
    # Black background for EVERY channel: no background color at a finite depth.
    independent_values = torch.cat([z, z.square(), z*f, f], dim=-1)
    rendered, alpha, _ = rasterization(
        means=s["means"], quats=s["quats"], scales=s["log_scales"].exp(),
        opacities=s["opacity_logits"].sigmoid(), colors=independent_values,
        backgrounds=torch.zeros(1, independent_values.shape[-1], device="cuda"),
        viewmats=pose[None], Ks=K[None], width=width, height=height,
        packed=False, rasterize_mode="antialiased", near_plane=.01, far_plane=1e6,
        absgrad=False, render_mode="RGB")
    rendered, alpha = rendered[0], alpha[0]
    d = scene.feature_dim
    depth_sum, square_sum = rendered[..., :1], rendered[..., 1:2]
    cross_sum, feature_sum = rendered[..., 2:2+d], rendered[..., 2+d:]
    assert torch.equal(alpha, baseline["alpha"])
    coverage = alpha.clamp(min=1e-4, max=1)
    mean = depth_sum / coverage
    second_moment, squared_mean = square_sum/coverage, mean.square()
    unclamped_variance = second_moment - squared_mean
    variance = unclamped_variance.clamp_min(0)
    guard = plan["numerical_guard"]
    assert guard == {"roundoff_factor": 8, "epsilon": 1e-6, "min_alpha": 1e-4}
    roundoff_floor = 8*torch.finfo(variance.dtype).eps*(second_moment.abs() + squared_mean).clamp_min(1)
    variance_floor = roundoff_floor.clamp_min(1e-6)
    supported = (alpha >= 1e-4) & (variance > variance_floor)
    covariance = cross_sum/coverage - mean*(feature_sum/coverage)
    contrast = torch.where(supported, torch.asinh(covariance/torch.sqrt(variance + 1e-6)), 0)
    spread = torch.where(supported, torch.log1p(torch.sqrt(variance)/mean.abs().clamp_min(1e-4)), 0)
    manual_context = torch.cat([spread, contrast], dim=-1)
    assert torch.isfinite(manual_context).all()
    valid = alpha >= 1e-4
    expected_depth = mean * scene.scene_scale
    depth_error = error_stats(expected_depth, baseline["depth"], valid)
    relative = ((expected_depth - baseline["depth"]).abs()/baseline["depth"].abs().clamp_min(1e-4))[valid]
    # Independent projection/matmul may round differently from CUDA's RGB+ED path.
    torch.testing.assert_close(expected_depth[valid], baseline["depth"][valid], rtol=2e-5, atol=2e-5)
    statistics = {"view": view["name"], "split": view["split"], "grid": [width, height],
                  "pixel_files_opened": False, "all_gaussians_used": len(s["means"]),
                  "moment_background_all_zero": True, "complete_alpha_exact_to_rgb": True,
                  "feature_sum_error": error_stats(feature_sum, baseline["features"]),
                  "normalized_mean_depth_to_rgb_ed_error": depth_error,
                  "mean_depth_max_relative_error": float(relative.max()),
                  "ed_comparison_valid_alpha_min": 1e-4,
                  "ed_numerical_tolerance": {"rtol": 2e-5, "atol": 2e-5},
                  "scene_scale": scene.scene_scale,
                  "numerical_guard": guard,
                  "roundoff_floor_min_max": [float(roundoff_floor.min()), float(roundoff_floor.max())],
                  "positive_fixed_epsilon_but_rejected_roundoff_fraction": float(((variance > 1e-6) & ~supported & valid).float().mean()),
                  "negative_unclamped_variance_fraction": float((unclamped_variance < 0).float().mean()),
                  "supported_variance_fraction": float(supported.float().mean()),
                  "spread_max": float(spread.max()), "cross_abs_max": float(contrast.abs().max())}
    rows, sampler_states = [], []
    for arm, mode in [("00_zero", "zero"), ("01_variance", "variance"), ("02_cross", "cross")]:
        config = dict(base_config, depth_moments=mode)
        head = MultiScaleRefinementHead(scene.feature_dim,
                                       **{key: value for key, value in config.items() if key != "type"}).cuda()
        add_depth_moment_paths(original_head, head)
        assert all(torch.equal(value, head.state_dict()[key]) for key, value in original_head.state_dict().items())
        assert all(torch.count_nonzero(head.state_dict()[key]) == 0
                   for key in ("moment_detail.weight", "moment_half.weight"))
        scene.refiner, scene.refiner_config = head, config
        inserted = scene.render(K, pose, width, height, absgrad=False, refinement_grad_to_field=False)
        initial_errors = {key: error_stats(inserted[key], baseline[key])
                          for key in ("rgb", "alpha", "depth", "features", "p3d", "residual", "probabilities")}
        assert all(initial_errors[key]["exact"] for key in ("rgb", "alpha", "depth", "p3d"))
        torch.testing.assert_close(inserted["probabilities"], baseline["probabilities"], rtol=1e-6, atol=1e-6)
        context_error = error_stats(inserted["depth_moments"], manual_context)
        torch.testing.assert_close(inserted["depth_moments"], manual_context, rtol=1e-5, atol=1e-5)
        checkpoint = torch.load(folder/arm/"last.pt", map_location="cpu", weights_only=False)
        initialization_path = folder/arm/"warmstart_initialization_audit.json"
        initialization = json.loads(initialization_path.read_text())
        assert initialization["status"] == "passed" and initialization["all_old_model_tensors_exact"]
        assert initialization["all_new_projection_weights_zero"] and initialization["only_refiner_requires_grad"]
        assert checkpoint["step"] == 2
        values = checkpoint["model"]
        for key, value in initial["model"].items():
            if not key.startswith("refiner."):
                assert torch.equal(value, values[key])
        assert torch.equal(initial["training_cameras"], checkpoint["training_cameras"])
        assert all(torch.isfinite(value).all() for value in values.values())
        changed = [key for key in initial["model"]
                   if not torch.equal(initial["model"][key], values[key])]
        assert changed and all(key.startswith("refiner.") for key in changed)
        sampler_states.append(checkpoint["density_state"]["extras"])
        scene.load_state_dict(values)
        trained = scene.render(K, pose, width, height, absgrad=False, refinement_grad_to_field=False)
        assert all(torch.equal(trained[key], baseline[key]) for key in ("rgb", "alpha", "depth", "p3d", "features"))
        logs = [json.loads(line) for line in (folder/arm/"train.jsonl").read_text().splitlines()]
        assert not any("semantic_lr_multiplier" in line for line in logs)
        assert checkpoint["optimizers"]["heads"]["param_groups"][1]["lr"] == .0003
        assert not list((folder/arm).glob("step_*.pt"))
        projection_columns = {key: values[f"refiner.{key}"].abs().sum(dim=(0, 2, 3)).tolist()
                              for key in ("moment_detail.weight", "moment_half.weight")}
        if mode == "zero":
            assert all(not any(columns) for columns in projection_columns.values())
        elif mode == "variance":
            assert all(columns[0] > 0 and not any(columns[1:]) for columns in projection_columns.values())
        else:
            assert all(columns[0] > 0 and any(columns[1:]) for columns in projection_columns.values())
        rows.append({"arm": arm, "mode": mode, "step": 2,
                     "head_parameters": sum(p.numel() for p in head.parameters()),
                     "new_projection_parameters": sum(p.numel() for name, p in head.named_parameters() if name.startswith("moment_")),
                     "initial_old_head_tensors_exact": True, "initial_new_projections_all_zero": True,
                     "real_trainer_initialization_audit_sha256": file_sha256(initialization_path),
                     "real_trainer_old_model_tensors_exact_before_first_step": True,
                     "initial_forward_errors": initial_errors, "context_to_independent_formula": context_error,
                     "only_refiner_tensors_changed": True, "old_refiner_tensor_changes": len(changed),
                     "all_parameters_finite": True, "frozen_cameras_exact": True,
                     "after_training_rgb_alpha_depth_raw_features_exact": True,
                     "projection_column_l1_after_training": projection_columns,
                     "head_lr": .0003, "peak_gpu_gb": logs[-1]["peak_gpu_gb"],
                     "checkpoint_sha256": file_sha256(folder/arm/"last.pt")})
        # The next arm always starts from the same initial frozen field and old head.
        scene.refiner, scene.refiner_config = original_head, base_config
        del head, inserted, trained, checkpoint, values
        gc.collect()
    assert len({row["head_parameters"] for row in rows}) == 1
    for key in ("sampler_order", "sampler_cursor", "sampler_rng_state"):
        assert sampler_states[0][key] == sampler_states[1][key] == sampler_states[2][key]
    return {"status": "passed", "independent_render": statistics, "runs": rows,
            "same_sampler_state": True, "same_head_parameter_count": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/generated_h3_moments_v2/h3_plan.json"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    plan_path = ROOT/args.plan
    plan = json.loads(plan_path.read_text())
    folder = ROOT/Path(plan["source_snapshot"]).parent
    snapshot = ROOT/plan["source_snapshot"]
    verify_sources(plan, snapshot)
    imports = {}
    for name in ("bridge_rgs", "bridge_rgs.model", "bridge_rgs.train",
                 "bridge_rgs.refinement", "bridge_rgs.depth_moments"):
        path = Path(importlib.import_module(name).__file__).resolve()
        assert path.is_relative_to(snapshot.resolve()), (name, str(path))
        imports[name] = {"path": str(path), "sha256": file_sha256(path)}
    receipt_path = folder/"preflight_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Refusing to overwrite H3 preflight execution")
    shutil.copy2(__file__, folder/"preflight_h3_moments_executed.py")
    worker = ROOT/"scripts/h3_training_preflight_worker.py"
    shutil.copy2(worker, folder/"h3_training_preflight_worker_executed.py")
    receipt = {"status": "running", "plan_sha256": file_sha256(plan_path),
               "script_sha256": file_sha256(__file__), "worker_script_sha256": file_sha256(worker),
               "source_hashes": plan["source_hashes"], "actual_audit_imports": imports,
               "warmstart_sha256": plan["warmstart_sha256"], "manifest_sha256": plan["manifest_sha256"],
               "input_hashes": {"initial_checkpoint": {"path": str((ROOT/plan["warmstart"]).resolve()), "sha256": plan["warmstart_sha256"]},
                                "manifest": {"path": str((ROOT/plan["manifest"]).resolve()), "sha256": plan["manifest_sha256"]}}}
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    environment = dict(os.environ, PYTHONPATH=str(snapshot), OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    try:
        for arm in ("00_zero", "01_variance", "02_cross"):
            path = plan_path.parent/f"{arm}_preflight.yaml"
            config = yaml.safe_load(path.read_text())
            assert config["steps"] == 2 and config["parameter_scope"] == "refiner_only"
            with (folder/f"{arm}.log").open("w") as log:
                subprocess.run([sys.executable, str(worker), "--config", str(path), "--snapshot", str(snapshot)],
                               cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = render_audit(plan, folder)
        verify_sources(plan, snapshot)
        (folder/"preflight_audit.json").write_text(json.dumps(report, indent=2)+'\n')
        receipt.update(status="completed", audit_sha256=file_sha256(folder/"preflight_audit.json"))
        print(json.dumps(report, indent=2))
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == "__main__":
    main()
