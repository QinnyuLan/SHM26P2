"""CPU-only feasibility record for the fixed eight-region action probe.

No RGB/label decoding, CUDA entry point, optimizer, renderer, or retry policy.
Geometric coverage below six regions is inconclusive for this probe design.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

PROTOCOL = "eight_region_action_transfer_cpu_feasibility_v1"
NAMES = ("003.png", "058.png", "115.png", "170.png", "232.png", "288.png", "344.png", "400.png")
VIEW_KEYS = ("name", "image_id", "split", "width", "height", "K", "w2c")
ACTIVE_KEYS = ("means", "quats", "log_scales", "opacity_logits", "sh0", "sh_rest")
CHECKPOINT_SHA = "dd8527ea8250d4c2b7f61a0563b6b81dc90615f1ddc1a2dbf957e25c5c51231c"
SPEC = {
    "region_count": 8, "minimum_measurable_regions": 6,
    "minimum_track_observations": 3, "maximum_reprojection_error_px": 1.,
    "region_neighbor_count": 16, "opacity_floor": .005,
    "parent_scale_fraction": .002, "minimum_views_per_group": 2,
    "near_plane": .01, "fit_views": list(NAMES[::2]), "score_views": list(NAMES[1::2]),
    "selection": "TRAIN SfM median-nearest start, deterministic farthest-point sampling; ties by track ID",
    "parent": "nearest eligible Gaussian center in fixed closed ball; ties by Gaussian ID",
    "donor": "minimum fit-view frustum count times opacity in same ball; ties by Gaussian ID",
    "reuse": "no reuse of parent or donor in subsequent regions",
    "split": "existing support_constrained; preserve opacity; shrink unstructured all axes; no clone",
    "scope": "old terminal field, pure geometry selection, two active Gaussians and four proposed steps",
    "coverage_failure": "inconclusive_coverage; do not broaden regions, change cameras, or infer mechanism invalidity",
    "gpu_execution_available": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def selected_views(manifest):
    require(manifest.get("pixel_protocol") is None, "Require the historical untagged legacy manifest")
    views = manifest["views"]
    require(len(views) == 400 and len({v["name"] for v in views}) == 400, "Expected unique original 400 views")
    train = {v["name"]: v for v in views if v["split"] == "train"}
    require(len(train) == 350 and set(NAMES).issubset(train), "Fixed eight views must belong to original 350 TRAIN")
    # Deliberately discard all image/mask/annotation paths. Selection never decodes them.
    return [{key: train[name][key] for key in VIEW_KEYS} for name in NAMES]


def qualified_tracks(arrays, train_image_ids):
    points = np.asarray(arrays["points"], np.float64)
    tracks = np.asarray(arrays["track_ids"], np.int64)
    errors = np.asarray(arrays["reprojection_error"], np.float64)
    counts = np.asarray(arrays["num_observations"], np.int64)
    offsets = np.asarray(arrays["observation_offsets"], np.int64)
    image_ids = np.asarray(arrays["observation_image_ids"], np.int64)
    n = len(points)
    require(points.shape == (n, 3) and tracks.shape == errors.shape == counts.shape == (n,), "Track shape mismatch")
    require(offsets.shape == (n + 1,) and offsets[0] == 0 and offsets[-1] == len(image_ids)
            and np.array_equal(np.diff(offsets), counts), "Track offsets/counts disagree")
    require(len(np.unique(tracks)) == n, "Duplicate track IDs")
    require(set(image_ids.tolist()).issubset(set(train_image_ids)), "Track evidence contains VAL or unknown image IDs")
    for i in range(n):
        require(len(np.unique(image_ids[offsets[i]:offsets[i+1]])) == counts[i], "Duplicate observations inflate support")
    valid = (np.isfinite(points).all(1) & np.isfinite(errors) & (errors >= 0)
             & (errors <= SPEC["maximum_reprojection_error_px"])
             & (counts >= SPEC["minimum_track_observations"]))
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(tracks[indices], kind="stable")]
    require(len(indices) >= SPEC["region_neighbor_count"], "Too few qualified TRAIN points")
    return points[indices], tracks[indices], indices


def region_anchors(points, track_ids, count=8, neighbors=16):
    points = np.asarray(points, np.float64)
    track_ids = np.asarray(track_ids, np.int64)
    require(points.shape == (len(track_ids), 3) and np.isfinite(points).all(), "Invalid anchor points")
    require(len(np.unique(track_ids)) == len(track_ids) and len(points) >= max(count, neighbors), "Insufficient unique anchors")
    order = np.argsort(track_ids, kind="stable")
    points, track_ids = points[order], track_ids[order]
    cursor = int(np.square(points - np.median(points, axis=0)).sum(1).argmin())
    distance = np.full(len(points), np.inf)
    chosen, records = [], []
    for index in range(count):
        chosen.append(cursor)
        distances = np.square(points - points[cursor]).sum(1)
        nearest = np.lexsort((track_ids, distances))[:neighbors]
        radius = float(np.sqrt(distances[nearest[-1]]))
        records.append({"region": index, "anchor_track_id": int(track_ids[cursor]),
                        "anchor_xyz": points[cursor].tolist(), "radius": radius,
                        "neighbor_track_ids": track_ids[nearest].tolist()})
        distance = np.minimum(distance, distances)
        distance[chosen] = -1
        cursor = int(distance.argmax())
    return records


def frustum_visibility(points, views):
    points = np.asarray(points, np.float64)
    rows = []
    for view in views:
        require(view["split"] == "train", "Non-TRAIN camera in selection")
        pose, K = np.asarray(view["w2c"], np.float64), np.asarray(view["K"], np.float64)
        require(pose.shape == (4, 4) and K.shape == (3, 3), "Camera shape mismatch")
        camera = points @ pose[:3, :3].T + pose[:3, 3]
        projected = camera @ K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = projected[:, :2] / projected[:, 2:]
        rows.append((camera[:, 2] > SPEC["near_plane"]) & np.isfinite(uv).all(1)
                    & (uv[:, 0] >= 0) & (uv[:, 0] < view["width"])
                    & (uv[:, 1] >= 0) & (uv[:, 1] < view["height"]))
    return np.asarray(rows, bool)


def choose_pairs(anchors, means, opacity, scales, scene_scale, visibility):
    means, opacity, scales = map(np.asarray, (means, opacity, scales))
    n = len(means)
    require(means.shape == scales.shape == (n, 3) and opacity.shape == (n,), "Gaussian shape mismatch")
    require(visibility.shape == (8, n), "Need the fixed eight-camera frustum matrix")
    require(np.isfinite(scene_scale) and scene_scale > 0, "Invalid scene scale")
    fit, score = visibility[::2].sum(0), visibility[1::2].sum(0)
    finite = np.isfinite(means).all(1) & np.isfinite(scales).all(1) & np.isfinite(opacity)
    opacity_ok = finite & (opacity >= SPEC["opacity_floor"])
    large = (scales.max(1) > SPEC["parent_scale_fraction"] * scene_scale)
    fit_ok, score_ok = fit >= 2, score >= 2
    used, records = set(), []
    for anchor in anchors:
        distance2 = np.square(means - np.asarray(anchor["anchor_xyz"])).sum(1)
        pool = np.flatnonzero(distance2 <= anchor["radius"] ** 2)
        remaining = pool[~np.isin(pool, list(used))]
        candidates = remaining[opacity_ok[remaining] & large[remaining] & fit_ok[remaining] & score_ok[remaining]]
        record = {**anchor, "pool_total": len(pool), "pool_not_previously_used": len(remaining),
                  "opacity_eligible": int(opacity_ok[remaining].sum()),
                  "opacity_and_large": int((opacity_ok & large)[remaining].sum()),
                  "large_and_fit": int((opacity_ok & large & fit_ok)[remaining].sum()),
                  "large_and_score": int((opacity_ok & large & score_ok)[remaining].sum()),
                  "joint_parent_candidates": len(candidates),
                  "fit_frustum_count_histogram": np.bincount(fit[remaining], minlength=5).tolist(),
                  "score_frustum_count_histogram": np.bincount(score[remaining], minlength=5).tolist(),
                  "parent_id": None, "donor_id": None, "status": "missing_parent"}
        if len(candidates):
            parent = int(candidates[np.lexsort((candidates, distance2[candidates]))[0]])
            donors = remaining[(remaining != parent) & opacity_ok[remaining]]
            record.update(parent_id=parent, donor_candidates=len(donors), status="missing_donor")
            if len(donors):
                retention = fit[donors] * opacity[donors]
                donor = int(donors[np.lexsort((donors, retention))[0]])
                used.update((parent, donor))
                record.update(donor_id=donor, status="measurable_geometric_pair",
                              parent_distance=float(np.sqrt(distance2[parent])),
                              donor_distance=float(np.sqrt(distance2[donor])),
                              parent_opacity=float(opacity[parent]), donor_opacity=float(opacity[donor]),
                              parent_max_scale=float(scales[parent].max()), donor_max_scale=float(scales[donor].max()),
                              parent_fit_frustum_count=int(fit[parent]), parent_score_frustum_count=int(score[parent]),
                              donor_fit_frustum_count=int(fit[donor]), donor_score_frustum_count=int(score[donor]),
                              keep_opacity_parameter_sum=float(opacity[parent] + opacity[donor]),
                              split_opacity_parameter_sum=float(2 * opacity[parent]),
                              note="Opacity parameter sum is not composited alpha or integrated mass; zero-step rendered effect unmeasured")
        records.append(record)
    return records


def proposed_budget(regions=8):
    # A render means one GaussianScene.render call, not one CUDA kernel.
    branch_renders = regions * 2 * 2 * (4 + 4 + 4)
    return {"regions": regions, "camera_conditions": 2, "actions": 2,
            "fit_renders_and_backwards": regions * 2 * 2 * 4,
            "zero_step_score_renders": regions * 2 * 2 * 4,
            "post_fit_score_renders": regions * 2 * 2 * 4,
            "camera_renders": 4 * (13 + 1), "baseline_repeat_renders": 8 * 2,
            "maximum_render_calls": branch_renders + 56 + 16,
            "proposed_wall_seconds": 120,
            "wall_status": "not benchmarked or authorized; no GPU worker exists after failed coverage gate",
            "actual_render_calls": 0, "actual_optimizer_steps": 0}


def summarize(records):
    require(len(records) == SPEC["region_count"] and [r["region"] for r in records] == list(range(8)), "Incomplete fixed regions")
    measurable = sum(r["status"] == "measurable_geometric_pair" for r in records)
    return {"status": "inconclusive_coverage" if measurable < 6 else "coverage_ready_pending_separate_gpu_review",
            "regions": 8, "measurable_regions": measurable, "coverage_fraction": measurable / 8,
            "minimum_required": 6, "gpu_allowed": False,
            "scope": "Only this old-endpoint/pure-geometry/two-active-point/four-step diagnostic design",
            "not_tested": ["action utility", "instantaneous rendered opacity effect", "cross-view loss gain",
                           "camera-action interaction", "production residual-driven candidate quality"],
            "limitation": "Global geometric FPS plus fixed eight cameras can select extreme or sparsely co-observed regions; no resampling or mechanism-negative inference"}


def load_helper(path):
    name = "action_transfer_densification_helper"
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


def split_contract(helper, state, parent, donor, reference_points):
    pair = {key: state[f"splats.{key}"][[parent, donor]].clone() for key in ACTIVE_KEYS}
    before = {key: value.clone() for key, value in pair.items()}
    result = helper.structure_guided_densify(
        pair, torch.tensor([1., 0.]), torch.tensor([2, 0]), max_gaussians=2,
        max_splits=1, recycle_count=1, reference_points=torch.as_tensor(reference_points).float(),
        split_opacity_mode="preserve", structured_split_rule="support_constrained",
        shrink_unstructured_all_axes=True)
    require(result.split_count == 1 and result.duplicate_count == 0 and result.pruned_count == 1,
            "Expected exactly split parent plus delete donor")
    require(result.source_indices.tolist() == [0, 0] and all(len(v) == 2 for v in result.tensors.values()), "Budget/ancestry differs")
    require(all(torch.equal(pair[k], before[k]) for k in pair), "Helper mutated source pair")
    require(all(torch.equal(result.tensors[k], pair[k][[0, 0]]) for k in ("sh0", "sh_rest", "quats")),
            "Appearance preserve contract differs")
    alpha_delta = (result.tensors["opacity_logits"].sigmoid() - pair["opacity_logits"][[0, 0]].sigmoid()).abs().max()
    require(float(alpha_delta) <= 1e-7, "Preserve-opacity roundtrip exceeded float precision")
    rotation = helper.quaternion_to_matrix(pair["quats"][:1])[0]
    shift = result.tensors["means"] - pair["means"][0]
    mahalanobis = ((shift @ rotation) / pair["log_scales"][0].exp()).norm(dim=-1)
    require(bool((mahalanobis <= .5001).all()), "Support split exceeded parent radial bound")
    return {"total_gaussians_before_and_after": 2, "active_rgb_geometry_scalars_per_arm": sum(v.numel() for v in pair.values()),
            "split_count": 1, "delete_count": 1, "clone_count": 0,
            "source_pair_unchanged": True, "child_mahalanobis_offsets": mahalanobis.tolist(),
            "line_count": result.line_splits, "plane_count": result.plane_splits,
            "opacity_sigmoid_roundtrip_max_abs_difference": float(alpha_delta),
            "proposed_child_means": result.tensors["means"].tolist(),
            "proposed_child_scales": result.tensors["log_scales"].exp().tolist(),
            "actual_optimizer_steps": 0, "rendered_effect_measured": False}


def prepare(root, output):
    started = time.monotonic()
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), "Refuse existing output; preserve prior CPU record")
    require(output == Path("/mnt/data/SHM2026/runs/action_transfer_probe_v1"), "Use the authorized artifact directory")
    torch.set_num_threads(8)
    base = root / "runs/pose_stress_clean_compensated/last.pt"
    receipt_path = root / "runs/pose_stress_clean_compensated/experiment_receipt.json"
    manifest_path, init_path = root / "artifacts/prepared/manifest.json", root / "artifacts/prepared/init_points.npz"
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("status") == "completed" and receipt.get("finished_utc"), "H1 input is not completed")
    require(digest(base) == receipt["checkpoint_sha256"] == CHECKPOINT_SHA, "Fixed H1 input SHA differs")
    require(digest(manifest_path) == receipt["input_hashes"]["manifest"]["sha256"], "Manifest lineage changed")
    protocol_path = root / "artifacts/pose_stress_mild/protocol_audit.json"
    require(digest(init_path) == json.loads(protocol_path.read_text())["init_sha256"], "TRAIN initialization changed")
    manifest = json.loads(manifest_path.read_text())
    views = selected_views(manifest)
    model = torch.load(base, map_location="cpu", weights_only=False, mmap=True)
    require(model["step"] == 6000 and model["sh_degree"] == 3, "Unexpected H1 endpoint")
    require(len(model["model"]["splats.means"]) == 177378, "Unexpected field size")
    require(torch.equal(model["training_cameras"], torch.tensor([v["w2c"] for v in manifest["views"] if v["split"] == "train"])),
            "H1 training cameras were changed")
    inputs = [base, receipt_path, manifest_path, init_path, protocol_path, root / "uv.lock"]
    input_hashes = {str(p): digest(p) for p in inputs}
    with np.load(init_path) as arrays:
        points, tracks, indices = qualified_tracks(arrays, [v["image_id"] for v in manifest["views"] if v["split"] == "train"])
    anchors = region_anchors(points, tracks)
    state = model["model"]
    means = state["splats.means"].numpy().astype(np.float64)
    opacity, scales = state["splats.opacity_logits"].sigmoid().numpy(), state["splats.log_scales"].exp().numpy()
    visibility = frustum_visibility(means, views)
    records = choose_pairs(anchors, means, opacity, scales, model["scene_scale"], visibility)
    helper_path = root / "src/bridge_rgs/densification.py"
    helper = load_helper(helper_path)
    for record in records:
        if record["status"] == "measurable_geometric_pair":
            record["cpu_action_contract"] = split_contract(helper, state, record["parent_id"], record["donor_id"], points)
    summary = summarize(records)
    files = {"prepare_action_transfer_probe.py": Path(__file__).resolve(),
             "action_densification.py": helper_path,
             "test_action_transfer_probe.py": root / "tests/test_action_transfer_probe.py",
             "design_review.md": root / "docs/densification_action_mechanism_review.md"}
    source_hashes = {name: digest(path) for name, path in files.items()}
    historical_source = root / "runs/pose_stress_mild_compensated/source_snapshot"
    for relative, expected in receipt["source_hashes"].items():
        require(digest(historical_source / relative) == expected, "Historical H1 source changed")
    require(all(digest(path) == sha for path, sha in input_hashes.items()), "Input bytes changed during CPU preparation")
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name, path in files.items():
        shutil.copy2(path, snapshot / name)
        require(digest(snapshot / name) == source_hashes[name], "Snapshot copy mismatch")
    coverage = {"protocol": PROTOCOL, "summary": summary, "qualified_train_tracks": len(points),
                "qualified_track_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                "view_order": list(NAMES), "regions": records,
                "selection_uses_rgb_or_semantic_values": False,
                "cpu_pair_proposals_are_not_model_updates": True}
    write_json(output / "coverage.json", coverage)
    plan = {"protocol": PROTOCOL, "status": summary["status"], "specification": SPEC,
            "created_utc": datetime.now(UTC).isoformat(), "workspace_root": str(root), "output": str(output),
            "checkpoint": str(base), "scene_scale": float(model["scene_scale"]), "manifest": str(manifest_path),
            "views_camera_whitelist_only": views, "input_hashes": input_hashes, "source_hashes": source_hashes,
            "historical_source_origin": str(historical_source), "historical_source_hashes": receipt["source_hashes"],
            "source_snapshot": str(snapshot), "coverage_sha256": digest(output / "coverage.json"),
            "proposed_full_protocol_budget_not_executed": proposed_budget(),
            "gpu_execution_allowed": False, "gpu_worker_implemented": False,
            "conclusion": "Coverage gate prevents this planned probe. No action utility, opacity effect or error difference has been measured."}
    write_json(output / "plan.json", plan)
    write_json(output / "cpu_preparation_receipt.json", {
        "status": "completed_cpu", "result": summary["status"], "plan_sha256": digest(output / "plan.json"),
        "coverage_sha256": digest(output / "coverage.json"), "elapsed_seconds": time.monotonic() - started,
        "input_bytes_reverified_unchanged": True, "rgb_or_gt_decoded": False,
        "cuda_initialized": torch.cuda.is_initialized(), "renders": 0, "optimizer_steps": 0,
        "published_model_checkpoint": False,
        "future_common_visibility_sampling": "A distinct feasibility design; not an amendment or result of this probe"})
    return output / "plan.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("/mnt/data/SHM2026/runs/action_transfer_probe_v1"))
    args = parser.parse_args()
    print(prepare(args.root, args.output))
