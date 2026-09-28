"""Prepare a CPU TRAIN-only ray plan, then optionally measure its frozen-scene mass."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import bridge_rgs.ray_termination as termination_implementation
from bridge_rgs.ray_support import SparseDepthSupport
from bridge_rgs.ray_termination import (
    conservative_depth_bins,
    front_mass_bounds,
    render_front_mass,
    sample_corner_pixel_features,
)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def quantiles(array):
    array = np.asarray(array)
    array = array[np.isfinite(array)]
    return np.quantile(array, [0, .1, .5, .9, 1]).tolist() if len(array) else None


def prepare(args):
    if args.output.exists():
        raise FileExistsError("Do not overwrite the fixed TRAIN ray plan")
    args.output.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text())
    design = json.loads(args.camera_plan.read_text())
    names = [g["anchor"] for g in design["groups"]]
    if len(names) != len(set(names)) or len(names) != 16:
        raise ValueError("Exactly 16 distinct fixed TRAIN anchors required")
    views = {v["name"]: v for v in manifest["views"] if v["split"] == "train"}
    if not set(names).issubset(views):
        raise ValueError("Ray plan must not contain VAL cameras")
    support = SparseDepthSupport.from_manifest(args.manifest)
    arrays, rows, valid_hashes = {}, [], {}
    for name in names:
        view = views[name]
        K = torch.tensor(view["K"], dtype=torch.float32)
        pose = torch.tensor(view.get("w2c_original", view["w2c"]), dtype=torch.float32)
        valid = torch.tensor(cv2.imread(view["valid_path"], 0) > 0).float()
        valid_hashes[name] = digest(view["valid_path"])
        target = support.targets_for_view(view["image_id"], K, pose, view["width"], view["height"], valid)
        # Retain existing quality gates, then additionally require the bilinear
        # corner-coordinate footprint to lie in eroded valid support.
        invalid = (~valid.bool()).float()[None, None]
        radius = support.config.border_px
        eroded = 1-F.max_pool2d(F.pad(invalid, (radius,)*4, value=1), 2*radius+1, stride=1)[0, 0]
        corner_valid = sample_corner_pixel_features(eroded[..., None], target.pixels)[:, 0] >= 1-1e-6
        ids = target.point_indices[corner_valid].numpy()
        depth, pixels, confidence = target.depth[corner_valid], target.pixels[corner_valid], target.confidence[corner_valid]
        rotation_z = pose[2, :3].numpy().astype(np.float64)
        variance = np.einsum("i,nij,j->n", rotation_z, support.covariances[ids], rotation_z)
        std = torch.tensor(np.sqrt(np.maximum(variance, 0)), dtype=torch.float32)
        prefix = Path(name).stem
        arrays.update({prefix+"_pixels": pixels.numpy(), prefix+"_depth": depth.numpy(),
                       prefix+"_std": std.numpy(), prefix+"_confidence": confidence.numpy(),
                       prefix+"_point_indices": ids})
        row = {"name": name, "image_id": view["image_id"], "width": view["width"], "height": view["height"],
               "K": K.tolist(), "w2c": pose.tolist(), "support_stats": target.stats,
               "corner_valid_targets": len(ids), "depth_quantiles": quantiles(depth.numpy()),
               "relative_position_std_quantiles": quantiles((std/depth).numpy()), "bins": {}}
        for channels in (16, 32):
            bins = conservative_depth_bins(depth, std, channels=channels)
            arrays[prefix+f"_edges{channels}"] = bins.edges.numpy()
            arrays[prefix+"_cutoff"] = bins.cutoff.numpy()
            arrays[prefix+"_covered"] = bins.covered.numpy()
            row["bins"][str(channels)] = {"requested_channels": channels, "actual_channels": len(bins.edges),
                                         "covered": int(bins.covered.sum()),
                                         "empty_or_invalid_free_segment": int((~bins.covered).sum()),
                                         "edges": bins.edges.tolist(),
                                         "relative_floor_gap_quantiles": quantiles((bins.gap/bins.cutoff).numpy())}
        rows.append(row)
    np.savez_compressed(args.output / "targets.npz", **arrays)
    package = Path(termination_implementation.__file__).resolve().parent
    source_files = [package / name for name in ("ray_termination.py", "ray_support.py", "model.py")]
    inputs = {"manifest": args.manifest, "checkpoint": args.checkpoint, "camera_plan": args.camera_plan}
    plan = {"kind": "train_only_discrete_splat_center_termination_plan", "views": rows,
            "inputs": {k: {"path": str(v.resolve()), "sha256": digest(v)} for k, v in inputs.items()},
            "source_hashes": {str(p.resolve()): digest(p) for p in source_files},
            "script_sha256": digest(__file__), "support_provenance": support.provenance,
            "targets_sha256": digest(args.output / "targets.npz"), "valid_hashes": valid_hashes,
            "mask_or_rgb_pixels_read": False, "validation_pixels_read": False,
            "edge_design": "32 log-spaced cutoff extrema; nested16 rounded linspace index subset including endpoints",
            "cutoff": "SfM depth - max(3 * conditional point-position depth std, .02 * SfM depth)",
            "center_coordinate_convention": "COLMAP/gsplat corner origin; first pixel center (.5,.5)",
            "sampling": "bilinear grid approximation at UV-.5 array coordinate; no total-alpha acceptance gate",
            "legacy_validity_note": "Retains ray_support quality/validity gate, adds corner-coordinate validity; does not assert prepared OpenCV remap full-chain consistency",
            "confidence_note": "Heuristic quality weight; conditional covariance is not a calibrated probability bound"}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"prepared": True, "views": len(rows), "targets": sum(r["corner_valid_targets"] for r in rows)}))


def measure(args):
    from bridge_rgs.train import load_scene

    plan = json.loads((args.output / "plan.json").read_text())
    measurement_output = args.measurement_output or args.output
    measurement_output.mkdir(parents=True, exist_ok=True)
    if (measurement_output / "report.json").exists():
        raise FileExistsError("Do not overwrite completed diagnostic")
    for item in plan["inputs"].values():
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Prepared input SHA changed")
    for path, value in plan["source_hashes"].items():
        actual_path = Path(termination_implementation.__file__).resolve().parent / Path(path).name
        if digest(actual_path) != value:
            raise ValueError("Prepared diagnostic implementation changed")
    if digest(args.output / "targets.npz") != plan["targets_sha256"]:
        raise ValueError("Prepared targets changed")
    measured_checkpoint = args.checkpoint.resolve(strict=True)
    measured_sha = digest(measured_checkpoint)
    scene, state = load_scene(measured_checkpoint)
    manifest = json.loads(Path(plan["inputs"]["manifest"]["path"]).read_text())
    train_views = [v for v in manifest["views"] if v["split"] == "train"]
    pose_reference = torch.tensor([v.get("w2c_original", v["w2c"]) for v in train_views], dtype=torch.float32)
    if not torch.equal(state["training_cameras"], pose_reference):
        raise ValueError("Diagnostic requires the unchanged original TRAIN cameras")
    scene.eval()
    for parameter in scene.parameters():
        parameter.requires_grad_(False)
    scene.splats["opacity_logits"].requires_grad_(True)
    original = {k: v.detach().cpu().clone() for k, v in scene.state_dict().items()}
    data = np.load(args.output / "targets.npz", allow_pickle=False)
    results, raw, gradient_check = [], {}, None
    for row in plan["views"]:
        name, prefix = row["name"], Path(row["name"]).stem
        K = torch.tensor(row["K"], device="cuda", dtype=torch.float32)
        pose = torch.tensor(row["w2c"], device="cuda", dtype=torch.float32)
        edges = torch.tensor(data[prefix+"_edges32"], device="cuda")
        pixels = torch.tensor(data[prefix+"_pixels"], device="cuda")
        cutoff = torch.tensor(data[prefix+"_cutoff"], device="cuda")
        depth = torch.tensor(data[prefix+"_depth"], device="cuda")
        upper_near = 2*depth-cutoff
        covered = torch.tensor(data[prefix+"_covered"], device="cuda")
        confidence = torch.tensor(data[prefix+"_confidence"], device="cuda")
        if not len(edges):
            results.append({"name": name, "targets": len(cutoff), "status": "no_supported_free_segment"})
            continue
        with torch.set_grad_enabled(gradient_check is None):
            image, alpha = render_front_mass(scene, K, pose, row["width"], row["height"], edges)
            mass = sample_corner_pixel_features(image, pixels)
            total_alpha = sample_corner_pixel_features(alpha, pixels)[:, 0]
            low32, high32 = front_mass_bounds(mass, total_alpha, edges, cutoff)
            coarse = torch.tensor(data[prefix+"_edges16"], device="cuda")
            subset = torch.searchsorted(edges, coarse)
            if not torch.equal(edges[subset], coarse):
                raise AssertionError("16-bin family must be an exact subset")
            low16, high16 = front_mass_bounds(mass[:, subset], total_alpha, coarse, cutoff)
            near16_low, near16_high = front_mass_bounds(mass[:, subset], total_alpha, coarse, upper_near)
            near32_low, near32_high = front_mass_bounds(mass, total_alpha, edges, upper_near)
            if bool(((low32+1e-6 < low16) | (high32 > high16+1e-6)).any()):
                raise AssertionError("Finer brackets must contain less unresolved mass")
            if bool(((mass[:, 1:]+1e-6 < mass[:, :-1]) | (mass[:, 1:] > total_alpha[:, None]+1e-5)).any()):
                raise AssertionError("Invalid monotone alpha mass")
            if gradient_check is None:
                loss = (low32[covered]*confidence[covered]).sum()/confidence[covered].sum().clamp_min(1e-8)
                loss.backward()
                gradients = {k: {"finite": bool(torch.isfinite(v.grad).all()),
                                 "norm": float(v.grad.norm()), "nonzero": int(v.grad.count_nonzero())}
                             for k, v in scene.named_parameters() if v.grad is not None}
                if set(gradients) != {"splats.opacity_logits"} or not gradients["splats.opacity_logits"]["finite"]:
                    raise AssertionError("Only finite opacity gradients are allowed")
                gradient_check = {"view": name, "diagnostic_weighted_lower_mass": float(loss.detach()),
                                  "gradients": gradients, "optimizer_steps": 0}
                scene.zero_grad(set_to_none=True)
        arrays = {"alpha": total_alpha, "low16": low16, "high16": high16, "low32": low32,
                  "high32": high32, "confidence": confidence, "covered": covered,
                  "near_band_lower16": (near16_low-high16).clamp_min(0),
                  "near_band_upper16": (near16_high-low16).clamp_min(0),
                  "near_band_lower32": (near32_low-high32).clamp_min(0),
                  "near_band_upper32": (near32_high-low32).clamp_min(0)}
        arrays = {key: value.detach().cpu().numpy() for key, value in arrays.items()}
        raw.update({prefix+"_"+key: value for key, value in arrays.items()})
        result = {"name": name, "targets": len(cutoff), "covered": int(covered.sum()),
                  "low_alpha_count": int((arrays["alpha"] < .5).sum()), "alpha_quantiles": quantiles(arrays["alpha"]),
                  "mass": {}}
        for bins in (16, 32):
            low, high = arrays[f"low{bins}"], arrays[f"high{bins}"]
            valid = arrays["covered"]
            weight = arrays["confidence"][valid]
            result["mass"][str(bins)] = {"lower_quantiles": quantiles(low[valid]),
                                         "bracket_width_quantiles": quantiles((high-low)[valid]),
                                         "weighted_mean_lower": float(np.average(low[valid], weights=weight)),
                                         "weighted_near_band_lower": float(np.average(arrays[f"near_band_lower{bins}"][valid], weights=weight)),
                                         "weighted_near_band_upper": float(np.average(arrays[f"near_band_upper{bins}"][valid], weights=weight)),
                                         "fraction_lower_above_0.01": float((low[valid] > .01).mean()),
                                         "fraction_lower_above_0.1": float((low[valid] > .1).mean()),
                                         "low_alpha_lower_quantiles": quantiles(low[valid & (arrays["alpha"] < .5)]),
                                         "high_alpha_lower_quantiles": quantiles(low[valid & (arrays["alpha"] >= .5)])}
        results.append(result)
        print(json.dumps({"view": name, "targets": len(cutoff), "weighted_lower32": result["mass"]["32"]["weighted_mean_lower"]}), flush=True)
        del image, alpha, mass
    for key, value in scene.state_dict().items():
        if not torch.equal(value.cpu(), original[key]):
            raise AssertionError(f"Diagnostic changed scene state: {key}")
    if digest(measured_checkpoint) != measured_sha:
        raise AssertionError("Checkpoint changed during diagnosis")
    np.savez_compressed(measurement_output / "mass_results.npz", **raw)
    report = {"complete": True, "plan_sha256": digest(args.output / "plan.json"),
              "mass_results_sha256": digest(measurement_output / "mass_results.npz"),
              "measured_checkpoint": str(measured_checkpoint), "measured_checkpoint_sha256": measured_sha,
              "script_sha256": digest(__file__),
              "actual_source_directory": str(Path(termination_implementation.__file__).resolve().parent),
              "views": results, "gradient_check": gradient_check, "scene_bitwise_unchanged": True,
              "checkpoint_step": state["step"], "point_count": len(scene.splats["means"]),
              "validation_or_rgb_pixels_read": False, "optimizer_steps": 0,
              "scope": "Frozen-scene TRAIN diagnostic of discrete center-depth CDF; not continuous physical free-space ground truth"}
    (measurement_output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/prepared/manifest.json"))
    parser.add_argument("--camera-plan", type=Path, default=Path("runs/projection_jitter_stress/plan.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/strong_semantic_coupled/last.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/ray_termination_diagnostic"))
    parser.add_argument("--measure", action="store_true", help="Short GPU measurement of an existing CPU plan")
    parser.add_argument("--measurement-output", type=Path, help="Separate before/after result with shared fixed plan")
    args = parser.parse_args()
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    measure(args) if args.measure else prepare(args)


if __name__ == "__main__":
    main()
