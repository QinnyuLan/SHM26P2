"""Independent CPU feasibility: restrict only FPS anchors to common frusta.

V1 remains inconclusive and immutable. This script has no GPU or optimization
entry point. Geometric coverage does not establish action-gain detectability.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

V1_DIRECTORY = Path("/mnt/data/SHM2026/runs/action_transfer_probe_v1")
V2_DIRECTORY = Path("/mnt/data/SHM2026/runs/action_transfer_probe_v2")
V1_PLAN_SHA = "a6d341ea1ff91c98643a263ba4d16a00e00470ba524c7dc1b3e26437f72150b2"
PROTOCOL = "eight_region_common_frustum_action_transfer_cpu_feasibility_v2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def common_anchor_pool(v1, points, views):
    visibility = v1.frustum_visibility(points, views)
    keep = (visibility[::2].sum(0) >= 2) & (visibility[1::2].sum(0) >= 2)
    require(int(keep.sum()) >= 16, "Insufficient common-frustum anchors; do not relax the fixed rule")
    return keep, visibility


def select_anchors(v1, points, tracks, common):
    """Only the FPS population changes; local radii use the original full pool."""
    require(common.shape == (len(points),) and common.dtype == np.dtype(bool), "Invalid common-pool mask")
    anchors = v1.region_anchors(points[common], tracks[common], count=8, neighbors=16)
    for anchor in anchors:
        distances2 = np.square(points - np.asarray(anchor["anchor_xyz"])).sum(1)
        nearest = np.lexsort((tracks, distances2))[:16]
        anchor["radius"] = float(np.sqrt(distances2[nearest[-1]]))
        anchor["neighbor_track_ids"] = tracks[nearest].tolist()
        anchor["radius_reference_pool"] = "original_all_qualified_train_points"
    return anchors


def distribution(points):
    points = np.asarray(points, np.float64)
    return {"count": len(points), "minimum_xyz": points.min(0).tolist(), "maximum_xyz": points.max(0).tolist(),
            "extent_xyz": np.ptp(points, axis=0).tolist(), "median_xyz": np.median(points, axis=0).tolist(),
            "quantiles_05_25_75_95_xyz": np.quantile(points, [.05, .25, .75, .95], axis=0).tolist()}


def adam_coordinate_bound(options, steps=4):
    """Cauchy-Schwarz bound for fresh Adam, no decay, constant LR, real arithmetic.

    |m_hat| / sqrt(v_hat) <= sqrt(sum_i a_i^2/b_i), with normalized
    exponential weights a,b. Positive eps only tightens the bound. This is a
    maximum possible optimizer displacement, not an expected update or power.
    """
    beta1, beta2 = map(float, options["betas"])
    require(0 <= beta1 < 1 and 0 < beta2 < 1 and steps > 0, "Invalid Adam bound inputs")
    require(float(options.get("weight_decay", 0)) == 0 and not options.get("amsgrad", False), "Bound requires plain no-decay Adam")
    require(float(options["lr"]) > 0 and float(options["eps"]) >= 0, "Invalid Adam LR/epsilon")
    multipliers = []
    for step in range(1, steps + 1):
        powers = np.arange(step - 1, -1, -1, dtype=np.float64)
        a = (1 - beta1) * beta1 ** powers / (1 - beta1 ** step)
        b = (1 - beta2) * beta2 ** powers / (1 - beta2 ** step)
        multipliers.append(float(np.sqrt(np.sum(a * a / b))))
    return {"steps": steps, "lr": float(options["lr"]), "betas": [beta1, beta2], "eps": float(options["eps"]),
            "per_step_normalized_update_upper_bounds": multipliers,
            "coordinate_displacement_upper_bound": float(options["lr"]) * sum(multipliers),
            "assumptions": "fresh zero moments, constant LR, no weight decay; before constraints/FP32 rounding; not measured gradients"}


def capacity_bounds(model, active_keys):
    bounds = {}
    for key in active_keys:
        groups = model["optimizers"][key]["param_groups"]
        require(len(groups) == 1 and len(groups[0]["params"]) == 1, "Unexpected H1 optimizer group")
        bounds[key] = adam_coordinate_bound(groups[0])
    bounds["means_l2_upper_bound"] = math.sqrt(3) * bounds["means"]["coordinate_displacement_upper_bound"]
    bounds["scale_relative_increase_upper_bound"] = math.expm1(bounds["log_scales"]["coordinate_displacement_upper_bound"])
    bounds["opacity_global_absolute_change_upper_bound"] = .25 * bounds["opacity_logits"]["coordinate_displacement_upper_bound"]
    bounds["active_rgb_geometry_scalars_per_branch"] = 118
    bounds["interpretation"] = "Small allowed local updates do not prove no effect; no action-gain/noise-floor/detectability claim without rendering"
    return bounds


def projected_size_proxy(helper, state, indices, views, width=512):
    """Unclipped first-order 3sigma ellipse area; no image/validity/occlusion reads.

    This is only a geometric scale diagnostic. It is NOT an exact gsplat
    footprint, rasterized support, contribution estimate, or scoring mask.
    """
    points = state["splats.means"][indices].double().numpy()
    rotations = helper.quaternion_to_matrix(state["splats.quats"][indices].double()).numpy()
    scales = state["splats.log_scales"][indices].double().exp().numpy()
    covariances = (rotations * (scales ** 2)[:, None, :]) @ rotations.transpose(0, 2, 1)
    result = []
    for view in views:
        factor = min(1., width / view["width"])
        w, h = round(view["width"] * factor), round(view["height"] * factor)
        K = np.array(view["K"], np.float64)
        K[0] *= w / view["width"]
        K[1] *= h / view["height"]
        pose = np.asarray(view["w2c"], np.float64)
        pc = points @ pose[:3, :3].T + pose[:3, 3]
        rows = []
        for point, covariance in zip(pc, covariances):
            x, y, z = point
            if z <= .01:
                rows.append(None)
                continue
            J = np.array([[K[0, 0] / z, 0, -K[0, 0] * x / z ** 2],
                          [0, K[1, 1] / z, -K[1, 1] * y / z ** 2]]) @ pose[:3, :3]
            screen_covariance = J @ covariance @ J.T + .3 * np.eye(2)
            area = 9 * math.pi * math.sqrt(max(0., float(np.linalg.det(screen_covariance))))
            center = np.array([K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]])
            rows.append({"center_inside_image": bool(0 <= center[0] < w and 0 <= center[1] < h),
                         "unclipped_3sigma_ellipse_area_pixels": area,
                         "unclipped_ellipse_area_over_image": area / (w * h)})
        result.append({"name": view["name"], "width": w, "height": h, "parent_donor": rows})
    return result


def prepare(root, output):
    started = time.monotonic()
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), "Preserve an existing v2 record")
    require(output == V2_DIRECTORY, "Use the separate authorized v2 directory")
    torch.set_num_threads(8)
    old_plan_path = V1_DIRECTORY / "plan.json"
    require(digest(old_plan_path) == V1_PLAN_SHA, "Original v1 plan changed")
    old_plan = json.loads(old_plan_path.read_text())
    require(old_plan["status"] == "inconclusive_coverage", "V1 result must remain inconclusive")
    old_coverage_path = V1_DIRECTORY / "coverage.json"
    require(digest(old_coverage_path) == old_plan["coverage_sha256"], "V1 coverage changed")
    original_snapshot = Path(old_plan["source_snapshot"])
    for relative, sha in old_plan["source_hashes"].items():
        require(digest(original_snapshot / relative) == sha, "V1 source changed")
    for path, sha in old_plan["input_hashes"].items():
        require(digest(path) == sha, "V1 frozen inputs changed")
    v1_source = original_snapshot / "prepare_action_transfer_probe.py"
    v1 = load_module(v1_source, "action_probe_v1_frozen")
    manifest = json.loads(Path(old_plan["manifest"]).read_text())
    views = v1.selected_views(manifest)
    with np.load(root / "artifacts/prepared/init_points.npz") as arrays:
        points, tracks, _ = v1.qualified_tracks(arrays, [v["image_id"] for v in manifest["views"] if v["split"] == "train"])
    common, visibility = common_anchor_pool(v1, points, views)
    anchors = select_anchors(v1, points, tracks, common)
    model = torch.load(old_plan["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    state = model["model"]
    require(model["step"] == 6000 and model["sh_degree"] == 3 and len(state["splats.means"]) == 177378, "Wrong old endpoint")
    means = state["splats.means"].numpy().astype(np.float64)
    records = v1.choose_pairs(anchors, means, state["splats.opacity_logits"].sigmoid().numpy(),
                             state["splats.log_scales"].exp().numpy(), model["scene_scale"], v1.frustum_visibility(means, views))
    helper_path = original_snapshot / "action_densification.py"
    helper = load_module(helper_path, "action_probe_v2_density_helper")
    bounds = capacity_bounds(model, v1.ACTIVE_KEYS)
    for record in records:
        if record["status"] != "measurable_geometric_pair":
            continue
        parent, donor = record["parent_id"], record["donor_id"]
        contract = v1.split_contract(helper, state, parent, donor, points)
        record["cpu_action_contract"] = contract
        offset = float(np.linalg.norm(np.asarray(contract["proposed_child_means"])[0] - means[parent]))
        record["child_initial_offset_l2"] = offset
        record["four_step_means_bound_over_initial_offset"] = bounds["means_l2_upper_bound"] / max(offset, 1e-30)
        record["projected_scale_proxy"] = projected_size_proxy(helper, state, [parent, donor], views)
    summary = v1.summarize(records)
    summary["scope"] = "Independent v2 common-frustum geometric feasibility; old terminal field, two active points, four proposed steps"
    summary["v1_result_unchanged"] = "inconclusive_coverage"
    summary["limits"] = "Frustum membership is not occlusion support, residual-driven selection, camera-error diversity, or measured action sensitivity"
    del summary["limitation"]  # Replace v1 global-FPS statement with the precise v2 limitation.
    population = {"original_qualified": distribution(points), "common_frustum": distribution(points[common]),
                  "common_fraction": float(common.mean()),
                  "common_track_ids_sha256": hashlib.sha256(tracks[common].astype("<i8").tobytes()).hexdigest(),
                  "fit_frustum_count_histogram": np.bincount(visibility[::2].sum(0), minlength=5).tolist(),
                  "score_frustum_count_histogram": np.bincount(visibility[1::2].sum(0), minlength=5).tolist(),
                  "anchors": distribution(np.asarray([r["anchor_xyz"] for r in records])),
                  "radius_pool": "all original qualified TRAIN points; not the common subset"}
    old_inputs = dict(old_plan["input_hashes"])
    old_inputs[str(old_plan_path)] = V1_PLAN_SHA
    old_inputs[str(old_coverage_path)] = digest(old_coverage_path)
    require(all(digest(p) == sha for p, sha in old_inputs.items()), "Inputs changed during CPU preparation")
    files = {"prepare_action_transfer_probe_v2.py": Path(__file__).resolve(),
             "prepare_action_transfer_probe_v1.py": v1_source, "action_densification.py": helper_path,
             "test_action_transfer_probe_v2.py": root / "tests/test_action_transfer_probe_v2.py"}
    source_hashes = {name: digest(p) for name, p in files.items()}
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name, path in files.items():
        shutil.copy2(path, snapshot / name)
        require(digest(snapshot / name) == source_hashes[name], "Source copy mismatch")
    coverage = {"protocol": PROTOCOL, "summary": summary, "population": population, "regions": records,
                "four_step_optimizer_upper_bounds": bounds,
                "observed_error_difference": None, "rendered_opacity_effect": None,
                "detectability_established": False, "no_rgb_mask_decoding": True}
    write_json(output / "coverage.json", coverage)
    plan = {"protocol": PROTOCOL, "status": summary["status"], "created_utc": datetime.now(UTC).isoformat(),
            "workspace_root": str(root), "output": str(output), "v1_plan": str(old_plan_path),
            "v1_result": "inconclusive_coverage", "v1_specification_unchanged": v1.SPEC,
            "only_design_change": "FPS anchor candidate pool restricted to original qualified points inside >=2 A and >=2 B frusta; 16-neighbor radii still original full qualified pool",
            "scene_scale": float(model["scene_scale"]), "views_camera_whitelist_only": views,
            "input_hashes": old_inputs, "source_hashes": source_hashes, "source_snapshot": str(snapshot),
            "coverage_sha256": digest(output / "coverage.json"),
            "proposed_full_protocol_budget_not_executed": v1.proposed_budget(),
            "concrete_budget_for_current_geometric_pairs_not_executed": v1.proposed_budget(summary["measurable_regions"]),
            "gpu_execution_allowed": False, "gpu_worker_implemented": False,
            "decision_required": "Root review of feasibility and sensitivity limits; coverage alone is not authorization or evidence of effect"}
    write_json(output / "plan.json", plan)
    write_json(output / "cpu_preparation_receipt.json", {
        "status": "completed_cpu", "result": summary["status"], "plan_sha256": digest(output / "plan.json"),
        "coverage_sha256": digest(output / "coverage.json"), "elapsed_seconds": time.monotonic() - started,
        "input_bytes_reverified_unchanged": True, "v1_bytes_unchanged": True, "rgb_or_masks_decoded": False,
        "cuda_initialized": torch.cuda.is_initialized(), "renders": 0, "optimizer_steps": 0,
        "checkpoint_written": False, "detectability_established": False})
    return output / "plan.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=V2_DIRECTORY)
    args = parser.parse_args()
    print(prepare(args.root, args.output))
