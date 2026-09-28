"""Forty-render diagnostic of complete appearance objectives and one transient Adam step.

CPU prepare only unless explicitly run from the frozen snapshot. No candidate
checkpoint is produced, and every in-memory tensor is restored in finally.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import signal
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

SOURCE_PLAN_SHA = "a8f889c3c9d7678cb0a1376dfd6bb1f06423e1e470f3f3853991259832869a1d"
BASE_SHA = "a77d304f32de4356c1a4608ed5192258b6a9ced4c67a4297809f439481408d89"
NEGATIVE_REPORT_SHA = "42a24e1501f397430e73842bc70c1e692ecfe4c287a2bd0d5eaa43965f47bcf4"
PARAMETERS = ("splats.sh0", "splats.sh_rest", "background_logits")
ARMS = ("00_native", "01_original")
EPSILONS = (1e-3, 5e-4, 2.5e-4)
RATES = {"splats.sh0": .00025, "splats.sh_rest": .0000125, "background_logits": .0001}
SPEC = {
    "protocol": "appearance_complete_objective_direction_and_step_v1",
    "view": "152.png", "split": "train", "arms": list(ARMS), "parameters": list(PARAMETERS),
    "epsilons": list(EPSILONS), "direction": "each color group's analytic gradient / its absmax",
    "objective": "original frozen FP32 .8 valid L1 + .2 complete-valid 7-window SSIM loss",
    "prediction": "original antialiased overscan/clamp, native crop or original float-map gather",
    "finite_difference_comparisons": 18, "renders": 40,
    "temporary_optimizer_steps_per_arm": 1, "retained_candidates": 0,
    "optimizer": {"name": "Adam", "lr": RATES, "eps": 1e-15, "betas": [.9, .999],
                  "weight_decay": 0, "fresh_state": True},
    "objective_dtype": "float32", "scalar_difference_arithmetic": "Python float over original FP32 loss scalars",
    "numerical_policy": "All epsilons and one-sided/central slopes reported; no tolerance fail is interpreted as a CUDA bug. L1/clamp nonsmoothness and FP32 resolution retained as uncertainty.",
    "internal_deadline_seconds": 55, "external_process_limit_seconds": 60,
    "output_bytes_limit": 2 << 20, "output_free_reserve_bytes": 256 << 20,
    "pixel_reads": "Only TRAIN152 prepared RGB, original RGB and native valid; no semantic labels/VAL",
    "scope": "One camera, two original objectives, one original fresh-Adam step each; no LR selection, multistep run, new candidate, or causal claim about multi-view failure.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value, *, replace=False):
    path = Path(path)
    if replace:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    else:
        with path.open("x") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fixed_first_view(manifest, original):
    views = sorted((v for v in manifest["views"] if v["split"] == "train"), key=lambda v: v["name"])
    require(len(views) == 350 and [v["name"] for v in views] == original["view_names"], "Original TRAIN indexing changed")
    view = views[original["training_order"][0]]
    require(view["name"] == SPEC["view"] and np.array_equal(view["w2c"], view["w2c_original"]), "Require original first TRAIN152 pose")
    return view


def checked_call(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Fixed complete-objective diagnostic deadline")


def normalized_direction(gradient):
    maximum = gradient.detach().abs().max()
    require(bool(torch.isfinite(gradient).all()) and bool(maximum > 0), "Finite nonzero gradient required for fixed directional diagnostic")
    direction = gradient.detach()/maximum
    derivative = float((gradient.double()*direction.double()).sum())
    return direction, derivative, float(maximum)


def numerical_record(base, plus, minus, epsilon, analytic):
    base, plus, minus = float(base), float(plus), float(minus)
    central = (plus-minus)/(2*epsilon)
    ulp_sum = float(abs(np.spacing(np.float32(plus))) + abs(np.spacing(np.float32(minus))))
    return {"epsilon": epsilon, "base_fp32_loss": base, "plus_fp32_loss": plus,
            "minus_fp32_loss": minus, "analytic_directional_derivative": analytic,
            "central_difference": central, "forward_slope": (plus-base)/epsilon,
            "backward_slope": (base-minus)/epsilon, "absolute_discrepancy": abs(central-analytic),
            "relative_discrepancy": abs(central-analytic)/max(abs(analytic), 1e-30),
            "one_ulp_scalar_resolution_in_derivative_units": ulp_sum/(2*epsilon),
            "difference_within_16_scalar_ulps": abs(plus-minus) <= 16*ulp_sum,
            "interpretation": "descriptive discrepancy, not a pass/fail CUDA correctness verdict"}


def parameter_differences(parameter, gradient, base_loss, objective, deadline):
    original = parameter.detach().clone()
    direction, analytic, maximum = normalized_direction(gradient)
    rows = []
    try:
        for epsilon in EPSILONS:
            checked_call(deadline)
            with torch.no_grad():
                positive = original + epsilon*direction
                negative = original - epsilon*direction
                parameter.copy_(positive)
                plus, plus_info = objective()
                checked_call(deadline)
                parameter.copy_(negative)
                minus, minus_info = objective()
                parameter.copy_(original)
                realized = (positive.double()-negative.double())/(2*epsilon)
                realized_analytic = float((gradient.double()*realized).sum())
                relative_roundoff = float((realized-direction.double()).square().sum().sqrt()/
                                         direction.double().square().sum().sqrt().clamp_min(1e-30))
            record = numerical_record(base_loss, plus, minus, epsilon, analytic)
            record.update(plus=plus_info, minus=minus_info,
                          realized_linearized_derivative=realized_analytic,
                          realized_direction_relative_l2_error=relative_roundoff)
            rows.append(record)
    finally:
        with torch.no_grad():
            parameter.copy_(original)
    require(torch.equal(parameter, original), "Directional probe failed exact restoration")
    return {"gradient_absmax": maximum, "direction_absmax": float(direction.abs().max()),
            "analytic_directional_derivative": analytic, "epsilons": rows,
            "parameter_restored_exact": True}


def transient_adam_step(parameters, gradients, objective, base_loss, deadline):
    """Exactly one original fresh step; restore even if post-step forward fails."""
    require(tuple(parameters) == PARAMETERS and tuple(gradients) == PARAMETERS, "Require exactly original three appearance groups in order")
    originals = {name: value.detach().clone() for name, value in parameters.items()}
    optimizer = torch.optim.Adam([{"params": [parameters[name]], "lr": RATES[name]} for name in PARAMETERS], eps=1e-15)
    require(not optimizer.state, "Fresh Adam state must be empty")
    result = None
    try:
        checked_call(deadline)
        for name in PARAMETERS:
            parameters[name].grad = gradients[name].detach().clone()
        optimizer.step()
        groups, total_linear = {}, 0.
        for name, group in zip(PARAMETERS, optimizer.param_groups, strict=True):
            value = parameters[name]
            require(bool(torch.isfinite(value).all()), "Adam produced nonfinite parameter")
            displacement = value.detach().double()-originals[name].double()
            linear = float((gradients[name].double()*displacement).sum())
            total_linear += linear
            step = float(optimizer.state[value]["step"])
            require(step == 1., "Only one fresh Adam step is allowed")
            groups[name] = {"lr": group["lr"], "eps": group["eps"], "betas": list(group["betas"]),
                            "weight_decay": group["weight_decay"], "maximize": group["maximize"],
                            "step": step, "gradient_dot_actual_displacement": linear,
                            "displacement_min": float(displacement.min()), "displacement_max": float(displacement.max()),
                            "displacement_absmax": float(displacement.abs().max()),
                            "displacement_rms": float(displacement.square().mean().sqrt()),
                            "zero_displacement_fraction": float((displacement == 0).float().mean())}
        checked_call(deadline)
        with torch.no_grad():
            after, after_info = objective()
        result = {"optimizer": SPEC["optimizer"], "groups": groups, "optimizer_steps": 1,
                  "before_fp32_loss": float(base_loss), "after_fp32_loss": float(after),
                  "actual_loss_change": float(after)-float(base_loss),
                  "gradient_dot_actual_displacement": total_linear, "after": after_info,
                  "remaining_nonlinear_change": float(after)-float(base_loss)-total_linear,
                  "restored_without_checkpoint": True}
    finally:
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(originals[name])
                value.grad = None
    require(all(torch.equal(value, originals[name]) for name, value in parameters.items()), "Adam audit failed exact restoration")
    return result


def restore_state(module, original):
    current = module.state_dict()
    require(set(current) == set(original), "Model state schema changed")
    with torch.no_grad():
        for name, value in current.items():
            value.copy_(original[name])
    for parameter in module.parameters():
        parameter.grad = None
    require(all(torch.equal(current[name], value) for name, value in original.items()), "Model state restoration failed")


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), "Preserve any prior diagnostic")
    original_path = root/"runs/raw_grid_appearance_v1/plan.json"
    require(digest(original_path) == SOURCE_PLAN_SHA, "Wrong original appearance protocol")
    original = json.loads(original_path.read_text())
    original_execution = original_path.parent/"execution_receipt.json"
    execution = json.loads(original_execution.read_text())
    require(execution["status"] == "completed" and execution["plan_sha256"] == SOURCE_PLAN_SHA, "Source experiment must be completed")
    require(original["specification"]["lr"] == RATES, "Original Adam rates differ")
    require(digest(original["base"]) == BASE_SHA, "Wrong base checkpoint")
    manifest_path = Path(original["manifest"])
    manifest = json.loads(manifest_path.read_text())
    view = fixed_first_view(manifest, original)
    negative_root = Path("/mnt/data/SHM2026/runs/appearance_train_objective_audit")
    negative_path, negative_receipt = negative_root/"report.json", negative_root/"execution_receipt.json"
    negative = json.loads(negative_receipt.read_text())
    require(negative["status"] == "completed" and negative["report_sha256"] == NEGATIVE_REPORT_SHA
            and digest(negative_path) == NEGATIVE_REPORT_SHA, "Wrong completed TRAIN-objective evidence")
    inputs = {str(p): digest(p) for p in (original_path, original_execution, Path(original["base"]),
              manifest_path, root/"uv.lock", negative_path, negative_receipt)}
    for path in (str(manifest_path), str(root/"uv.lock")):
        require(inputs[path] == original["input_hashes"][path], "Original manifest/environment changed")
    allowed = []
    for key in ("image_path", "source_image_path", "valid_path"):
        path = str((root/view[key]).resolve())
        inputs[path] = digest(path)
        require(inputs[path] == original["input_hashes"][path], "TRAIN152 pixel source changed")
        allowed.append(path)
    first_loss = {}
    for arm in ARMS:
        path = original_path.parent/arm/"train.jsonl"
        first = json.loads(path.read_text().splitlines()[0])
        require(first["step"] == 1 and first["view"] == SPEC["view"], "Original first training row differs")
        inputs[str(path)] = digest(path)
        first_loss[arm] = first["loss"]
    source = Path(original["snapshot"])
    for relative, sha in original["source_hashes"].items():
        require(digest(source/relative) == sha, "Original frozen source changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(output.parent).free >= SPEC["output_free_reserve_bytes"], "Insufficient output reserve")
    output.mkdir()
    snapshot = output/"source_snapshot"
    shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    hashes = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))}
    require(all(hashes[k] == v for k, v in original["source_hashes"].items()), "Source copy changed")
    plan = {"status": "locked", "specification": SPEC, "root": str(root), "output": str(output),
            "snapshot": str(snapshot), "source_hashes": hashes, "runner_sha256": digest(__file__),
            "input_hashes": inputs, "original_plan": str(original_path), "base": original["base"],
            "manifest": str(manifest_path), "view": view, "allowed_pixel_paths": sorted(set(allowed)),
            "original_step1_losses": first_loss,
            "execute_only_after_teacher_four_stages_and_explicit_gpu_handoff": True,
            "external_command_policy": "timeout --signal=KILL 60s; frozen runner + snapshot PYTHONPATH; no retry"}
    write_json(output/"plan.json", plan)
    return output/"plan.json"


def verify(plan):
    require(plan["status"] == "locked" and plan["specification"] == SPEC, "Fixed plan changed")
    snapshot = Path(plan["snapshot"])
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, "Use the frozen diagnostic entrypoint")
    actual = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))}
    require(actual == plan["source_hashes"], "Frozen source hashes changed")
    modules = {}
    for name in ("model", "train", "raw_grid", "losses", "checkpoints", "evaluate", "coordinates"):
        path = Path(importlib.import_module(f"bridge_rgs.{name}").__file__).resolve()
        require(path == snapshot/"bridge_rgs"/f"{name}.py", f"Wrong actual import: {name}")
        modules[name] = {"path": str(path), "sha256": digest(path)}
    reference = importlib.import_module("run_raw_grid_appearance")
    require(Path(reference.__file__).resolve() == snapshot/"run_raw_grid_appearance.py", "Wrong original runner")
    for path, sha in plan["input_hashes"].items():
        require(digest(path) == sha, f"Changed diagnostic input: {path}")
    manifest = json.loads(Path(plan["manifest"]).read_text())
    require(fixed_first_view(manifest, json.loads(Path(plan["original_plan"]).read_text())) == plan["view"], "TRAIN152 binding changed")
    return reference, modules


def run(plan_path):
    started = time.monotonic()
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    output = Path(plan["output"])
    require(not (output/"execution_receipt.json").exists(), "No diagnostic rerun")
    os.chdir(plan["root"])
    reference, modules = verify(plan)
    clients = reference.gpu_idle()
    receipt = {"status": "running", "plan_sha256": digest(plan_path), "pid": os.getpid(),
               "actual_imports": modules, "gpu_handoff": clients, "render_calls": 0,
               "numerical_discrepancies_are_not_cuda_error_verdicts": True}
    write_json(output/"execution_receipt.json", receipt)
    def alarm(*_):
        raise TimeoutError("Internal complete-objective diagnostic deadline")
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.1, SPEC["internal_deadline_seconds"]-(time.monotonic()-started)))
    scene = original_state = original_flags = original_render = before = None
    original_imread = cv2.imread
    read_counts = Counter()
    def guarded_read(path, *args, **kwargs):
        path = str(Path(path).resolve())
        require(path in plan["allowed_pixel_paths"], "Pixel read outside TRAIN152 RGB/valid")
        read_counts[path] += 1
        return original_imread(path, *args, **kwargs)
    cv2.imread = guarded_read
    rows = {}
    try:
        torch.set_num_threads(8)
        cv2.setNumThreads(8)
        from bridge_rgs.raw_grid import appearance_rgb_loss
        from bridge_rgs.train import load_scene
        scene, state = load_scene(plan["base"])
        scene.eval()
        original_flags = {name: p.requires_grad for name, p in scene.named_parameters()}
        for name, parameter in scene.named_parameters():
            parameter.requires_grad_(name in PARAMETERS)
        named = dict(scene.named_parameters())
        require({name for name, p in named.items() if p.requires_grad} == set(PARAMETERS), "Unexpected gradient permission")
        require(all(named[name].dtype == torch.float32 for name in PARAMETERS), "Original color parameters must be FP32")
        original_state = {name: value.detach().clone() for name, value in scene.state_dict().items()}
        before = {name: reference.tensor_hash(value) for name, value in original_state.items()}
        manifest = json.loads(Path(plan["manifest"]).read_text())
        require(torch.equal(state["training_cameras"], reference.original_training_poses(manifest)), "Base original camera ordering changed")
        view = plan["view"]
        layouts, warps = reference.layouts_for_views(manifest, [view])
        layout = layouts[0]
        pose = torch.tensor(view["w2c_original"], device="cuda", dtype=torch.float32)
        original_render = scene.render
        render_diagnostics = None
        def observed_render(*args, **kwargs):
            nonlocal render_diagnostics
            checked_call(started+SPEC["internal_deadline_seconds"])
            require(receipt["render_calls"] < SPEC["renders"], "Fixed render budget exceeded")
            result = original_render(*args, **kwargs)
            receipt["render_calls"] += 1
            with torch.no_grad():
                raw = result["rgb"]
                render_diagnostics = {"raw_rgb_min": float(raw.min()), "raw_rgb_max": float(raw.max()),
                    "raw_below_zero_fraction": float((raw < 0).float().mean()),
                    "raw_above_one_fraction": float((raw > 1).float().mean()),
                    "raw_exact_zero_fraction": float((raw == 0).float().mean()),
                    "raw_exact_one_fraction": float((raw == 1).float().mean())}
            return result
        scene.render = observed_render
        torch.cuda.reset_peak_memory_stats()
        for arm in ARMS:
            target, valid = reference.load_target(view, arm, plan["root"])
            def objective(arm=arm, target=target, valid=valid):
                canvas = reference.render_training_rgb(scene, layout, pose)
                prediction = reference.predict_arm(canvas, layout, arm)
                loss, stats = appearance_rgb_loss(prediction, target, valid)
                require(loss.dtype == torch.float32 and bool(torch.isfinite(loss)), "Complete original FP32 objective required")
                with torch.no_grad():
                    residual = (prediction-target).abs()[valid]
                    diagnostics = {**render_diagnostics, "l1": float(stats["l1"]),
                        "one_minus_ssim7": float(stats["one_minus_ssim7"]),
                        "rgb_pixels": int(stats["rgb_pixels"]), "ssim7_centers": int(stats["ssim7_centers"]),
                        "valid_l1_exact_zero_channel_fraction": float((residual == 0).float().mean()),
                        "valid_l1_abs_residual_le_1e_minus6_channel_fraction": float((residual <= 1e-6).float().mean()),
                        "kink_limit": "Output RGB/L1 indicators do not enumerate internal SH clamp or all SSIM nonlinearities."}
                return loss, diagnostics
            scene.zero_grad(set_to_none=True)
            base_loss, baseline_info = objective()
            require(float(base_loss) == plan["original_step1_losses"][arm], "Base forward differs from original TRAIN152 step1 logged objective")
            base_loss.backward()
            require(all(p.grad is None for name, p in named.items() if name not in PARAMETERS), "Frozen nonappearance parameter got gradient")
            require(all(named[name].grad is not None for name in PARAMETERS), "Appearance group missing gradient")
            gradients = {name: named[name].grad.detach().clone() for name in PARAMETERS}
            baseline = float(base_loss.detach())
            del base_loss
            directions = {name: parameter_differences(named[name], gradients[name], baseline, objective,
                          started+SPEC["internal_deadline_seconds"]) for name in PARAMETERS}
            step = transient_adam_step({name: named[name] for name in PARAMETERS}, gradients, objective,
                                       baseline, started+SPEC["internal_deadline_seconds"])
            restore_state(scene, original_state)
            require(before == {name: reference.tensor_hash(value) for name, value in scene.state_dict().items()}, "Inter-arm model changed")
            rows[arm] = {"baseline_fp32_loss": baseline, "baseline": baseline_info,
                         "first_training_log_forward_exact": True, "directions": directions,
                         "single_step": step, "all_tensor_bytes_restored_exact": True}
            print(json.dumps({"arm": arm, "renders": receipt["render_calls"],
                              "same_view_loss_change": step["actual_loss_change"],
                              "g_dot_delta": step["gradient_dot_actual_displacement"]}), flush=True)
        require(receipt["render_calls"] == 40 and len(rows) == 2, "Incomplete fixed diagnostic")
        require(sum(read_counts.values()) == 3 and set(read_counts) == set(plan["allowed_pixel_paths"]), "Expected only three TRAIN152 RGB/valid reads")
        verify(plan)
        require(digest(plan_path) == receipt["plan_sha256"], "Plan changed during diagnostic")
        receipt.update(status="completed", arms=rows, warps=warps, read_counts=dict(read_counts),
                       parameters_before_sha256=before, all_bound_inputs_and_sources_unchanged=True,
                       transient_optimizer_steps=2, semantic_label_pixels_decoded=0, val_pixels_decoded=0,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}", partial_arms=rows)
        raise
    finally:
        # Disable asynchronous alarm before best-effort restoration; outer 60s limit remains.
        signal.setitimer(signal.ITIMER_REAL, 0)
        cv2.imread = original_imread
        try:
            if scene is not None and original_state is not None:
                restore_state(scene, original_state)
                if original_render is not None:
                    scene.render = original_render
                for name, parameter in scene.named_parameters():
                    parameter.requires_grad_(original_flags[name])
                restored = {name: reference.tensor_hash(value) for name, value in scene.state_dict().items()}
                expected = before if before is not None else {name: reference.tensor_hash(value) for name, value in original_state.items()}
                receipt["all_model_tensors_finally_restored_exact"] = restored == expected
                require(restored == expected, "Finally model restoration differs")
        except Exception as error:
            receipt.update(status="failed", restoration_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            receipt.update(seconds=time.monotonic()-started, checkpoint_written=False)
            write_json(output/"execution_receipt.json", receipt, replace=True)
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > SPEC["output_bytes_limit"]:
            receipt.update(status="failed", output_budget_error="Diagnostic output exceeded fixed 2 MiB budget")
            write_json(output/"execution_receipt.json", receipt, replace=True)
            raise ValueError("Diagnostic output budget exceeded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("/mnt/data/SHM2026/runs/appearance_actual_objective_diagnostic"))
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.root, args.output))
    else:
        run(args.run)


if __name__ == "__main__":
    main()
