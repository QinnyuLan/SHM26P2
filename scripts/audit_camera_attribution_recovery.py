"""One bounded TRAIN-only camera attribution audit; prepare is CPU-only.

The immutable draft fixes all inputs. --run is reserved for the later explicit
GPU handoff; it never changes the draft and records a separate execution receipt.
No optimization, opacity interventions, semantic labels, VAL images or retries.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

PROTOCOL = "h1_production_incremental_recovery_v1"
NAMES = ("003.png", "058.png", "115.png", "170.png", "232.png", "288.png", "344.png", "400.png")
CASES = ("self_production", "real_base", "real_perturbed", "self_no_prior")
SPEC = {
    "views": list(NAMES), "width": 320, "degree": 3, "noise_multiplier": .5,
    "damping": .001, "translation_eps_scene_scale": 1e-5, "rotation_eps": 1e-4,
    "translation_limit_scene_scale": .001, "rotation_limit": .002,
    "translation_prior_std_scene_scale": .005, "rotation_prior_std": .01,
    "point_limit": 4096, "minimum_common_points": 32, "minimum_initial_pixels": 1e-4,
    "pass_ratio": .5, "pass_views": 6, "wall_seconds": 120,
    "cases": list(CASES), "renders_per_view": 30, "torch_threads": 8,
    "pixel_protocol": "legacy_mixed_v1",
    "scope": "incremental recovery diagnostic, not physical error-source identification or performance",
    "budget": "hard subprocess timeout includes imports, source/input checks, model load, renders and final checks",
    "point_support": "common projectable TRAIN SfM points; historical round-to-array valid; not visibility truth",
    "scoring": "fixed per-view projection ratios, separate rotation/translation; no confidence interval",
}
VIEW_KEYS = ("name", "split", "width", "height", "K", "w2c", "image_path", "valid_path")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.now(UTC).isoformat()


def require(value, message):
    if not value:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def selected_views(manifest, stress):
    require(manifest.get("pixel_protocol") is None, "This historical H1 audit requires the original untagged manifest")
    views = manifest["views"]
    require(len(views) == 400 and len({v["name"] for v in views}) == 400, "Expected unique historical 400 views")
    train = {v["name"]: v for v in views if v["split"] == "train"}
    require(len(train) == 350, "Expected all 350 original TRAIN views")
    injection = stress["pose_stress"]["injected_left_twists_translation_first"]
    require(set(injection) == set(train) - {"002.png"}, "Unexpected stress injection population")
    names = sorted(injection)
    chosen = [names[i * 348 // 7] for i in range(8)]
    require(chosen == list(NAMES), "Fixed TRAIN sample changed")
    return [{**{key: train[name][key] for key in VIEW_KEYS},
             "half_twist": (np.asarray(injection[name], dtype=np.float64) * .5).tolist()} for name in chosen]


def validate_twists(views, scene_scale):
    require(np.isfinite(scene_scale) and scene_scale > 0, "Invalid model scale")
    for view in views:
        twist = np.asarray(view["half_twist"], np.float64)
        require(twist.shape == (6,) and np.isfinite(twist).all(), "Invalid saved half twist")
        require(np.linalg.norm(twist[:3]) < .001 * scene_scale
                and np.linalg.norm(twist[3:]) < .002, "Half twist must fit the production one-step bounds")


def bind(plan):
    for path, expected in plan["input_hashes"].items():
        require(digest(path) == expected, f"Bound input changed: {path}")
    for relative, expected in plan["source_hashes"].items():
        require(digest(Path(plan["source_snapshot"]) / relative) == expected, f"Source changed: {relative}")


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), "Refuse an existing audit directory")
    receipt_path = root / "runs/pose_stress_clean_compensated/experiment_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("status") == "completed" and receipt.get("finished_utc"), "H1 source must be completed")
    base = root / "runs/pose_stress_clean_compensated/last.pt"
    manifest_path = root / "artifacts/prepared/manifest.json"
    stress_path = root / "artifacts/pose_stress_mild/manifest.json"
    init_path = root / "artifacts/prepared/init_points.npz"
    require(digest(base) == receipt["checkpoint_sha256"], "H1 source checkpoint changed")
    require(digest(manifest_path) == receipt["input_hashes"]["manifest"]["sha256"], "H1 clean manifest changed")
    manifest, stress = json.loads(manifest_path.read_text()), json.loads(stress_path.read_text())
    require(stress["pose_stress"]["source_manifest_sha256"] == digest(manifest_path), "Stress lineage changed")
    views = selected_views(manifest, stress)
    # CPU metadata only: never instantiate a scene or access cuda here.
    state = torch.load(base, map_location="cpu", weights_only=False, mmap=True)
    require(state["step"] == 6000 and state["sh_degree"] == 3, "Require the fixed final H1 scene")
    scale = float(state["scene_scale"])
    validate_twists(views, scale)
    original_cameras = torch.tensor([v["w2c"] for v in manifest["views"] if v["split"] == "train"])
    require(torch.equal(state["training_cameras"], original_cameras), "H1 cameras were changed")
    expected_init = json.loads((root / "artifacts/pose_stress_mild/protocol_audit.json").read_text())["init_sha256"]
    require(digest(init_path) == expected_init, "H1 clean initialization changed")
    paths = [base, manifest_path, stress_path, init_path, receipt_path, root / "uv.lock",
             root / "artifacts/pose_stress_mild/protocol_audit.json",
             root / "runs/residual_diagnostic/diagnostic.json"]
    for view in views:
        for key in ("image_path", "valid_path"):
            view[key] = str((root / view[key]).resolve())
            paths.append(Path(view[key]))
    inputs = {str(path): digest(path) for path in paths}
    origin = root / "runs/pose_stress_mild_compensated/source_snapshot"
    for relative, expected in receipt["source_hashes"].items():
        require(digest(origin / relative) == expected, "H1 immutable source changed")
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    shutil.copytree(origin / "bridge_rgs", snapshot / "bridge_rgs",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    source_hashes = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))}
    for relative, expected in receipt["source_hashes"].items():
        require(source_hashes[relative] == expected, "H1 snapshot copy changed")
    plan = {"status": "cpu_registered_draft_pending_gpu_handoff", "protocol": PROTOCOL,
            "created_utc": utc(), "workspace_root": str(root), "output": str(output),
            "source_snapshot": str(snapshot), "specification": SPEC, "source_hashes": source_hashes,
            "input_hashes": inputs, "checkpoint": str(base), "manifest": str(manifest_path),
            "init_points": str(init_path), "scene_scale": scale,
            "manifest_scene_radius": manifest["scene_radius"], "views": views,
            "historical_self_render_result": "six TRAIN cameras: camera 99.1791%, random opacity removal 0.8875%; not production-matched",
            "execution": "separate receipt; draft immutable; run only after root GPU handoff; no retry or resampling"}
    bind(plan)
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def production_options(scale, reference):
    return {"damping": .001, "max_translation": .001 * scale, "max_rotation": .002,
            "prior_precision": torch.diag(reference.new_tensor([1 / (.005 * scale) ** 2] * 3 + [1 / .01 ** 2] * 3))}


@torch.no_grad()
def solve_verified(api, render, pose, base, jacobian, target, weights, options):
    """Same production solve/re-render acceptance, reusing the fixed Jacobian."""
    residual = base - target
    unbounded = api.camera_residual_attribution(residual, jacobian, weights,
                                               **{**options, "max_translation": None, "max_rotation": None})
    result = api.camera_residual_attribution(residual, jacobian, weights, **options)
    proposed = result.delta.clone()
    candidate_pose = api.se3_exp(proposed) @ pose
    candidate = render(candidate_pose) - target
    weight = weights.expand_as(candidate)
    after = (weight * candidate.square()).sum() / weight.sum().clamp_min(1)
    accepted = bool(torch.isfinite(after) & (after <= result.before_energy))
    if not accepted:
        result.delta.zero_()
        candidate_pose = pose
        after = result.before_energy
    return candidate_pose, {
        "delta": result.delta.tolist(), "unbounded_delta": unbounded.delta.tolist(),
        "proposed_delta": proposed.tolist(), "accepted": accepted,
        "translation_clipped": bool(unbounded.delta[:3].norm() > options["max_translation"]),
        "rotation_clipped": bool(unbounded.delta[3:].norm() > options["max_rotation"]),
        "before_mse": float(result.before_energy), "after_mse": float(after),
        "explained_fraction": float(((result.before_energy - after) / result.before_energy.clamp_min(1e-12)).clamp(0, 1)),
    }


def pose_separation(first, second, scale):
    relative = first.double() @ torch.linalg.inv(second.double())
    rotation = relative[:3, :3]
    skew = torch.stack((rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1])) / 2
    angle = torch.atan2(skew.norm(), ((rotation.trace() - 1) / 2).clamp(-1, 1))
    return {"rotation_radians": float(angle), "translation_scene_scale": float(relative[:3, 3].norm() / scale)}


def common_points(api, points, pose, shifted, K, valid):
    keep = torch.ones(len(points), dtype=torch.bool, device=points.device)
    for camera in (pose, shifted):
        projection = api.projection_jacobians(points, camera, K)
        uv = projection.uv.round().long()
        ok = (projection.depth > .01) & torch.isfinite(projection.uv).all(-1)
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < valid.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < valid.shape[0])
        ids = ok.nonzero().flatten()
        ok[ids] &= valid[uv[ids, 1], uv[ids, 0]].bool()
        keep &= ok
    ids = keep.nonzero().flatten()
    if len(ids) > SPEC["point_limit"]:
        ids = ids[torch.linspace(0, len(ids) - 1, SPEC["point_limit"], device=ids.device).long()]
    return points[ids], ids


def projection_distance(api, points, first, second, K):
    if len(points) < SPEC["minimum_common_points"]:
        return None
    a, b = (api.projection_jacobians(points, pose, K) for pose in (first, second))
    # Never discard troublesome corrected points to improve the ratio.
    if not bool((a.depth > .01).all() & (b.depth > .01).all() & torch.isfinite(a.uv).all() & torch.isfinite(b.uv).all()):
        return None
    return float((a.uv - b.uv).norm(dim=-1).median())


def information(jacobian, weights, scale):
    J = jacobian.reshape(-1, 6).double()
    weight = weights.expand(jacobian.shape[:-1]).reshape(-1).double()
    normal = J.T @ (weight[:, None] * J)
    limits = J.new_tensor([.001 * scale] * 3 + [.002] * 3)
    normalized = limits[:, None] * normal * limits[None, :]
    prior = J.new_tensor([1 / (.005 * scale) ** 2] * 3 + [1 / .01 ** 2] * 3)
    regularizer = (.001 * normal.diagonal().clamp_min(1) + prior) * limits.square()
    eigenvalues = torch.linalg.eigvalsh(normalized)
    whitened = normalized / torch.sqrt(regularizer[:, None] * regularizer[None, :])
    generalized = torch.linalg.eigvalsh(whitened).clamp_min(0)
    return {"trust_scaled_information_eigenvalues": eigenvalues.tolist(),
            "regularizer_relative_information_eigenvalues": generalized.tolist(),
            "linear_mode_retention": (generalized / (1 + generalized)).tolist()}


@torch.no_grad()
def audit_view(api, render, pose, twist, target, valid, points, K, scale):
    shifted = api.se3_exp(twist) @ pose
    common, ids = common_points(api, points, pose, shifted, K, valid)
    base, J0 = api.finite_difference_camera_jacobian(render, pose, 1e-5 * scale, 1e-4)
    changed, Je = api.finite_difference_camera_jacobian(render, shifted, 1e-5 * scale, 1e-4)
    weights, options = valid[..., None], production_options(scale, pose)
    poses, cases = {}, {}
    for name, camera, rgb, jacobian, truth, prior in (
        ("self_production", shifted, changed, Je, base, options["prior_precision"]),
        ("real_base", pose, base, J0, target, options["prior_precision"]),
        ("real_perturbed", shifted, changed, Je, target, options["prior_precision"]),
        ("self_no_prior", shifted, changed, Je, base, None),
    ):
        poses[name], cases[name] = solve_verified(api, render, camera, rgb, jacobian, truth, weights,
                                                 {**options, "prior_precision": prior})
    before = projection_distance(api, common, pose, shifted, K)
    distances = {"self_production": projection_distance(api, common, pose, poses["self_production"], K),
                 "real_increment": projection_distance(api, common, poses["real_base"], poses["real_perturbed"], K),
                 "self_no_prior": projection_distance(api, common, pose, poses["self_no_prior"], K)}
    ratios = {key: value / before if value is not None and before is not None and before > SPEC["minimum_initial_pixels"] else None
              for key, value in distances.items()}
    actual = changed - base
    linear = torch.einsum("...i,i->...", J0, twist)
    denominator = (weights * actual.square()).sum()
    linear_error = float(torch.sqrt((weights * (actual - linear).square()).sum() / denominator)) if denominator > 1e-12 else None
    return {"cases": cases, "half_twist": twist.tolist(), "common_points": len(common),
            "common_point_indices_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            "initial_median_projection_pixels": before, "corrected_median_projection_pixels": distances,
            "recovery_ratios": ratios, "linear_prediction_relative_rmse": linear_error,
            "pose_separation": {"initial": pose_separation(shifted, pose, scale),
                "self_production": pose_separation(poses["self_production"], pose, scale),
                "real_increment": pose_separation(poses["real_perturbed"], poses["real_base"], scale)},
            "information_base": information(J0, weights, scale), "information_perturbed": information(Je, weights, scale)}


def summarize(records):
    require(len(records) == 8 and [r["name"] for r in records] == list(NAMES), "Incomplete/fixed-view audit cannot pass")
    result = {}
    for case in ("self_production", "real_increment", "self_no_prior"):
        values = [record["recovery_ratios"][case] for record in records]
        result[case] = {"ratios": values, "valid_views": sum(v is not None for v in values),
                        "half_recovery_views": sum(v is not None and v <= .5 for v in values)}
        result[case]["necessary_gate_passed"] = result[case]["half_recovery_views"] >= 6
    result["interpretation"] = "Only a fixed eight-view necessary diagnostic; not evidence of physical cause, novelty or H1 rendering gains"
    return result


def tensor_hash(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def worker(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    require(plan["protocol"] == PROTOCOL and plan["specification"] == SPEC, "Unexpected diagnostic protocol")
    bind(plan)
    modules = {name: importlib.import_module("bridge_rgs." + name) for name in ("io", "model", "train", "reliability")}
    snapshot = Path(plan["source_snapshot"])
    for name, module in modules.items():
        require(Path(module.__file__).resolve() == snapshot / "bridge_rgs" / f"{name}.py", "Unexpected imported source")
    torch.set_num_threads(8)
    torch.manual_seed(42)
    scene, state = modules["train"].load_scene(plan["checkpoint"])
    scene.eval()
    require(scene.scene_scale == plan["scene_scale"] and state["step"] == 6000, "Scene metadata changed")
    before = {k: tensor_hash(v) for k, v in scene.state_dict().items()}
    camera_before = tensor_hash(state["training_cameras"])
    with np.load(plan["init_points"]) as initial:
        points = torch.tensor(initial["points"], dtype=torch.float32, device="cuda")
    records = []
    result_path = Path(plan["output"]) / "audit.json"
    for view in plan["views"]:
        require(view["split"] == "train" and set(view) == set(VIEW_KEYS) | {"half_twist"}, "Camera whitelist changed")
        # The frozen historical loader only sees this whitelist, hence mask_path
        # and annotation paths cannot trigger any label decode.
        data = modules["io"].load_view({key: view[key] for key in VIEW_KEYS}, scale=320 / view["width"])
        require(data["mask"] is None and data["width"] == 320, "Unexpected label/size")
        calls = [0]

        def render(pose, data=data, calls=calls):
            calls[0] += 1
            return scene.render(data["K"], pose, data["width"], data["height"],
                                degree=3, semantics=False, absgrad=False)["rgb"]

        record = audit_view(modules["reliability"], render, data["w2c"],
                            data["w2c"].new_tensor(view["half_twist"]), data["rgb"], data["valid"], points,
                            data["K"], scene.scene_scale)
        require(calls[0] == 30, "Render budget changed")
        records.append({"name": view["name"], "render_calls": calls[0], **record})
        write_json(result_path, {"status": "partial", "protocol": PROTOCOL, "views": records})
        print(json.dumps({"completed_views": len(records), "name": view["name"], "ratios": record["recovery_ratios"]}), flush=True)
    require(before == {k: tensor_hash(v) for k, v in scene.state_dict().items()}, "Model state changed")
    require(camera_before == tensor_hash(state["training_cameras"]), "Cameras changed")
    bind(plan)
    write_json(result_path, {"status": "completed", "protocol": PROTOCOL, "views": records,
               "summary": summarize(records), "model_and_cameras_exact": True, "bound_inputs_unchanged": True,
               "labels_decoded": 0, "validation_payloads_read": 0, "optimizer_steps": 0,
               "loaded_source": {name: {"path": module.__file__, "sha256": digest(module.__file__)} for name, module in modules.items()}})


def gpu_idle():
    document = ET.fromstring(subprocess.run(["nvidia-smi", "-q", "-x"], check=True, capture_output=True, text=True).stdout)
    require(document.findall("gpu"), "No GPU")
    for gpu in document.findall("gpu"):
        listing = gpu.find("processes")
        require(listing is not None and (listing.text or "").strip() not in {"N/A", "Not Supported"}, "Unknown GPU processes")
    rows = [{key: n.findtext(key) for key in ("pid", "type", "process_name")} for n in document.findall(".//process_info")]
    require(all(row["type"] == "G" for row in rows), "GPU still has compute clients")
    return rows


def execute(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    require(plan["status"] == "cpu_registered_draft_pending_gpu_handoff" and plan["specification"] == SPEC, "Unexpected draft")
    output, snapshot = Path(plan["output"]), Path(plan["source_snapshot"])
    require(Path(__file__).resolve() == snapshot / Path(__file__).name, "Use the copied immutable runner")
    receipt_path = output / "execution_receipt.json"
    require(not receipt_path.exists() and not (output / "audit.json").exists(), "No rerun or replacement allowed")
    bind(plan)
    graphics = gpu_idle()
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(plan_path)]
    receipt = {"status": "running", "protocol": PROTOCOL, "started_utc": utc(), "command": command,
               "plan_sha256": digest(plan_path), "graphics_clients": graphics, "wall_limit_seconds": 120}
    write_json(receipt_path, receipt)
    env = {**os.environ, "PYTHONPATH": str(snapshot)}
    started = time.monotonic()
    try:
        with (output / "worker.log").open("x") as log:
            completed = subprocess.run(command, cwd=plan["workspace_root"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=SPEC["wall_seconds"], check=False)
        require(completed.returncode == 0, f"Diagnostic worker exit {completed.returncode}")
        audit_path = output / "audit.json"
        audit = json.loads(audit_path.read_text())
        require(audit["status"] == "completed" and audit["model_and_cameras_exact"], "Worker did not complete the full contract")
        summarize(audit["views"])
        require(digest(plan_path) == receipt["plan_sha256"], "Draft changed")
        receipt.update(status="completed", audit_sha256=digest(audit_path))
    except subprocess.TimeoutExpired:
        receipt.update(status="inconclusive_timeout", note="Own child killed by subprocess timeout; no subset pass, no retry or resampling")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    receipt.update(finished_utc=utc(), elapsed_seconds=time.monotonic() - started)
    write_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", type=Path)
    modes.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/h1_attribution_recovery_v1"))
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.root, args.output))
    elif args.worker:
        worker(args.worker)
    else:
        result = execute(args.run)
        print(json.dumps(result))
        if result["status"] != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
