"""One bounded numerical audit of appearance CUDA derivatives, without GT or training."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

BASE_SHA = "a77d304f32de4356c1a4608ed5192258b6a9ced4c67a4297809f439481408d89"
MANIFEST_SHA = "91b41aeecedc352e4882eb80a1bb635d479ae432251a4c96c85a9dc110cbb327"
PARAMETERS = ("splats.sh0", "splats.sh_rest", "background_logits")
PATHS = ("native_raw", "overscan_clamp_crop", "overscan_clamp_raw_warp")
EPSILONS = (.01, .005, .0025)
SPEC = {
    "protocol": "appearance_cuda_directional_derivative_v1", "view": "002.png", "split": "train",
    "parameters": list(PARAMETERS), "paths": list(PATHS), "epsilons": list(EPSILONS),
    "direction": "each path/parameter's analytic gradient divided by its absolute maximum",
    "probe": "mean_RGB((1+.25sin(2pi*(x+.5)/W)+.125cos(2pi*(y+.5)/H))*[.7,1,1.3]*RGB); float64 multiply/sum",
    "loss_target": None, "finite_difference": "central; always restore original before both signs",
    "comparison": {"absolute_tolerance": 1e-7, "relative_tolerance": .02,
                   "all_eps_reported": True, "no_best_epsilon_selection": True},
    "train_steps": 0, "render_calls": 57, "internal_deadline_seconds": 55,
    "external_process_limit_seconds": 60, "output_bytes_limit": 1 << 20,
    "limits": "One camera/smooth probe/directional test, not full Jacobian, training loss, or generalization; SH/color/output clamp kinks can affect finite differences.",
}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_new(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def probe_weights(height, width, *, device="cpu"):
    x = (torch.arange(width, device=device, dtype=torch.float64) + .5) / width
    y = (torch.arange(height, device=device, dtype=torch.float64) + .5) / height
    spatial = 1 + .25 * (2 * math.pi * x).sin()[None, :] + .125 * (2 * math.pi * y).cos()[:, None]
    return spatial[..., None] * torch.tensor([.7, 1., 1.3], device=device, dtype=torch.float64)


def probe(image, weights):
    if image.shape != weights.shape:
        raise ValueError("Probe image dimensions differ")
    return (image.double() * weights).mean()


def normalized_direction(gradient):
    maximum = gradient.detach().abs().max()
    if not torch.isfinite(gradient).all() or not bool(maximum > 0):
        raise ValueError("Direction requires finite nonzero analytic gradient")
    direction = gradient.detach() / maximum
    derivative = float((gradient.double() * direction.double()).sum())
    return direction, derivative, float(maximum)


def scalar_comparison(plus, minus, epsilon, analytic):
    plus, minus, analytic = float(plus), float(minus), float(analytic)
    numerical = (plus - minus) / (2 * epsilon)
    error = abs(numerical - analytic)
    tolerance = SPEC["comparison"]
    return {"epsilon": epsilon, "plus": plus, "minus": minus, "analytic": analytic,
            "central_difference": numerical, "absolute_error": error,
            "relative_error": error / max(abs(analytic), 1e-30),
            "within_fixed_tolerance": error <= tolerance["absolute_tolerance"] +
            tolerance["relative_tolerance"] * abs(analytic)}


def checked_call(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Fixed audit deadline exceeded")


def parameter_differences(parameter, gradient, objective, deadline):
    """Restore parameter even if a render/timeout/measurement fails; never optimize."""
    original = parameter.detach().clone()
    direction, analytic, maximum = normalized_direction(gradient)
    records = []
    try:
        for epsilon in EPSILONS:
            checked_call(deadline)
            with torch.no_grad():
                positive = original + epsilon * direction
                negative = original - epsilon * direction
                parameter.copy_(positive)
                plus, plus_diagnostics = objective()
                checked_call(deadline)
                parameter.copy_(negative)
                minus, minus_diagnostics = objective()
                parameter.copy_(original)
                actual_direction = (positive.double() - negative.double()) / (2 * epsilon)
                actual_linear = float((gradient.double() * actual_direction).sum())
                roundoff = float((actual_direction - direction.double()).square().sum().sqrt()
                                 / direction.double().square().sum().sqrt().clamp_min(1e-30))
            record = scalar_comparison(plus, minus, epsilon, analytic)
            record.update(realized_step_linearized_derivative=actual_linear,
                          realized_direction_relative_l2_error=roundoff,
                          positive_output=plus_diagnostics, negative_output=minus_diagnostics)
            records.append(record)
    finally:
        with torch.no_grad():
            parameter.copy_(original)
    if not torch.equal(parameter, original):
        raise ValueError("Parameter restoration failed")
    return {"gradient_absmax": maximum, "direction_absmax": float(direction.abs().max()),
            "analytic_directional_derivative": analytic, "epsilons": records,
            "parameter_restored_exact": True}


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str(value.dtype).encode() + str(tuple(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve existing audit")
    base = root / "runs/corner_v2_semantic_coupled/last.pt"
    manifest = root / "artifacts/prepared_corner_v2/manifest.json"
    receipt = root / "runs/official_common_grid_v1/corner_v2/execution_receipt.json"
    record = json.loads(receipt.read_text())
    if (record.get("status") != "completed" or record.get("checkpoint_sha256") != BASE_SHA
            or digest(base) != BASE_SHA or digest(manifest) != MANIFEST_SHA):
        raise ValueError("Fixed completed inputs changed")
    views = [v for v in json.loads(manifest.read_text())["views"] if v["name"] == SPEC["view"]]
    if len(views) != 1 or views[0]["split"] != "train":
        raise ValueError("Require the fixed TRAIN002 camera")
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    shutil.copytree(root / "src/bridge_rgs", snapshot / "bridge_rgs",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    sources = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))}
    for relative, value in sources.items():
        origin = root / "src" / relative if relative.startswith("bridge_rgs/") else Path(__file__)
        if digest(origin) != value:
            raise ValueError("Source changed during snapshot")
    plan = {"status": "locked", "specification": SPEC, "root": str(root), "output": str(output),
            "snapshot": str(snapshot), "base": str(base), "manifest": str(manifest),
            "source_hashes": sources, "input_hashes": {str(p): digest(p) for p in
                (base, manifest, receipt, root / "uv.lock")},
            "camera": {key: views[0][key] for key in
                       ("name", "split", "width", "height", "camera_id", "K", "w2c_original")},
            "execution": "After root review: timeout --signal=KILL 60s uv run python with snapshot PYTHONPATH; no other compute client"}
    write_new(output / "plan.json", plan)
    return output / "plan.json"


def verify(plan):
    if plan["status"] != "locked" or plan["specification"] != SPEC:
        raise ValueError("Fixed audit plan differs")
    for filename, sha in plan["input_hashes"].items():
        if digest(filename) != sha:
            raise ValueError(f"Input changed: {filename}")
    snapshot = Path(plan["snapshot"])
    actual = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))}
    if actual != plan["source_hashes"] or Path(__file__).resolve() != snapshot / Path(__file__).name:
        raise ValueError("Execute the exact frozen source and runner")
    modules = {}
    for name in ("model", "evaluate", "raw_grid", "train", "checkpoints"):
        module = importlib.import_module(f"bridge_rgs.{name}")
        path = Path(module.__file__).resolve()
        if path != snapshot / "bridge_rgs" / f"{name}.py":
            raise ValueError(f"Wrong module import: {name}")
        modules[name] = {"path": str(path), "sha256": digest(path)}
    views = [v for v in json.loads(Path(plan["manifest"]).read_text())["views"]
             if v["name"] == SPEC["view"] and v["split"] == "train"]
    if len(views) != 1 or {key: views[0][key] for key in plan["camera"]} != plan["camera"]:
        raise ValueError("Locked TRAIN002 camera differs from bound manifest")
    return modules


def gpu_clients():
    xml = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, check=True).stdout
    document = ET.fromstring(xml)
    if not document.findall("gpu") or any(g.find("processes") is None for g in document.findall("gpu")):
        raise ValueError("Cannot inspect GPU clients")
    rows = [{key: p.findtext(key) for key in ("pid", "type", "process_name")}
            for p in document.findall(".//process_info")]
    if any(row["type"] != "G" for row in rows):
        raise ValueError("GPU has another compute client")
    return rows


def run(plan_path):
    start = time.monotonic()
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    output = Path(plan["output"])
    receipt_path = output / "execution_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("No automatic rerun")
    modules = verify(plan)
    clients = gpu_clients()
    receipt = {"status": "running", "plan_sha256": digest(plan_path), "pid": os.getpid(),
               "actual_imports": modules, "prelaunch_graphics_clients": clients}
    write_new(receipt_path, receipt)
    def alarm(*_):
        raise TimeoutError("Internal 55 second audit deadline")
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.1, 55 - (time.monotonic() - start)))
    scene = None
    before = None
    try:
        torch.set_num_threads(8)
        from bridge_rgs.coordinates import CORNER, pixel_protocol
        from bridge_rgs.raw_grid import build_raw_grid
        from bridge_rgs.train import load_scene
        scene, state = load_scene(plan["base"])
        if pixel_protocol(state) != CORNER or state.get("manifest_sha256") != MANIFEST_SHA:
            raise ValueError("Model protocol differs")
        scene.eval()
        for name, parameter in scene.named_parameters():
            parameter.requires_grad_(name in PARAMETERS)
        before = {name: tensor_hash(value) for name, value in scene.state_dict().items()}
        view = plan["camera"]
        manifest = json.loads(Path(plan["manifest"]).read_text())
        camera = manifest["source_cameras"][str(view["camera_id"])]
        layout = build_raw_grid(camera["K"], camera["opencv_distortion"], view["width"], view["height"])
        layout.warp.to("cuda")
        K = torch.tensor(view["K"], device="cuda", dtype=torch.float32)
        pose = torch.tensor(view["w2c_original"], device="cuda", dtype=torch.float32)
        expanded_K = torch.tensor(layout.render_K, device="cuda", dtype=torch.float32)
        weights = probe_weights(view["height"], view["width"], device="cuda")
        named = dict(scene.named_parameters())
        results = {}
        calls = 0
        for path in PATHS:
            def forward(render_path=path):
                nonlocal calls
                checked_call(start + 55)
                if render_path == "native_raw":
                    image = scene.render(K, pose, view["width"], view["height"], degree=scene.sh_degree,
                                         semantics=False, absgrad=False)["rgb"]
                    raw = image
                else:
                    raw = scene.render(expanded_K, pose, layout.render_width, layout.render_height,
                                       degree=scene.sh_degree, semantics=False, absgrad=False)["rgb"]
                    clamped = raw.clamp(0, 1)
                    image = layout.crop_native(clamped) if render_path == "overscan_clamp_crop" else layout.warp(clamped)
                calls += 1
                diagnostics = {"raw_rgb_min": float(raw.detach().min()), "raw_rgb_max": float(raw.detach().max()),
                               "raw_outside_unit_channel_fraction": float(((raw.detach() < 0) | (raw.detach() > 1)).float().mean())}
                return probe(image, weights), diagnostics
            scene.zero_grad(set_to_none=True)
            scalar, baseline = forward()
            gradients = torch.autograd.grad(scalar, [named[name] for name in PARAMETERS])
            baseline["probe"] = float(scalar.detach())
            def objective():
                value, diagnostics = forward()
                return float(value), diagnostics
            rows = {}
            for name, gradient in zip(PARAMETERS, gradients, strict=True):
                rows[name] = parameter_differences(named[name], gradient, objective, start + 55)
            results[path] = {"baseline": baseline, "parameters": rows}
        after = {name: tensor_hash(value) for name, value in scene.state_dict().items()}
        if after != before or calls != SPEC["render_calls"]:
            raise ValueError("Scene restoration or fixed render count failed")
        verify(plan)
        if digest(plan_path) != receipt["plan_sha256"]:
            raise ValueError("Plan changed")
        report = {"status": "completed", "specification": SPEC, "results": results,
                  "all_model_tensors_restored_exact": True, "input_and_source_hashes_unchanged": True,
                  "render_calls": calls, "gt_or_image_files_decoded": 0, "optimizer_steps": 0,
                  "elapsed_seconds": time.monotonic() - start, "scene_before_hashes": before,
                  "all_27_finite_differences_within_fixed_tolerance": all(
                      r["within_fixed_tolerance"] for path in results.values()
                      for group in path["parameters"].values() for r in group["epsilons"])}
        write_new(output / "report.json", report)
        receipt.update(status="completed", report_sha256=digest(output / "report.json"),
                       elapsed_seconds=report["elapsed_seconds"])
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}",
                       elapsed_seconds=time.monotonic() - start)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if scene is not None and before is not None:
            receipt["model_restored_on_exit"] = before == {
                name: tensor_hash(value) for name, value in scene.state_dict().items()}
        receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    total_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if total_bytes >= SPEC["output_bytes_limit"]:
        raise ValueError("Audit output exceeded the fixed one MiB budget")
    print(json.dumps({"status": receipt["status"], "elapsed_seconds": receipt["elapsed_seconds"],
                      "output_bytes_excluding_bytecode": total_bytes}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/appearance_cuda_gradient_audit"))
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.root, args.output))
    elif args.plan is not None:
        run(args.plan)
    else:
        parser.error("Choose --prepare or --plan")


if __name__ == "__main__":
    main()
