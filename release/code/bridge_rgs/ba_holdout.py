"""CPU, TRAIN-correspondence-only bounded BA diagnostic; no file or pixel I/O.

This wraps the existing BA unchanged. Held-out tracks/observations are conditional
on the upstream SfM selection, not independently acquired calibration evidence.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .data import ColmapCamera, ColmapImage
from .geometry import (
    TrackObservation,
    TriangulatedCloud,
    bounded_bundle_adjustment,
    triangulate_dlt,
)

SPEC = {
    "protocol": "ba_track_holdout_v1",
    "fit_tracks": 500,
    "check_tracks": 5000,
    "fit_min_observations": 3,
    "check_min_observations": 6,
    "track_hash": "sha256 UTF8 ba_track_holdout_v1:track:{trackid}; ascending digest",
    "observation_hash": "sha256 UTF8 ba_track_holdout_v1:obs:{trackid}:{imageid}",
    "support_count": "ceil(2*n/3); hash-ordered remainder scores",
    "minimum_support_parallax_degrees": .5,
    "positive_depth_threshold": 0.,
    "parallax_definition": "maximum angle of support-DLT-point minus support-camera-center rays",
    "max_points": 500,
    "max_nfev": 20,
    "ba_seed": 42,
    "bootstrap_repeats": 2000,
    "bootstrap_seed": 20260927,
    "primary": "equal-track mean of mean radial native undistorted pixel error on score observations",
    "delta_direction": "candidate minus baseline; negative improves",
    "fixed_population": "baseline eligibility only; candidate failure cannot remove a track",
    "geometry_input_keys": ["point_positions", "point_track_ids", "observation_offsets",
                            "image_id", "raw_xy", "undistorted_native_xy", "saved_track_rms"],
    "candidate_validity": "finite point/projections and positive depth in all observations; parallax descriptive only",
    "coverage_minimum_tracks": 1000,
    "coverage_minimum_score_cameras": 200,
    "coverage_minimum_rows_per_camera": 5,
    "coverage_minimum_rows_per_group": 100,
    "signal_minimum_relative_mean_reduction": .01,
    "signal_ci95_upper_strictly_below": 0.,
    "signal_minimum_improving_name_groups": 3,
    "group_metric": "within-track mean of score observations in the group, then equal track mean",
}


GEOMETRY_KEYS = tuple(SPEC["geometry_input_keys"])

def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    array = np.asarray(value)
    _require(np.issubdtype(array.dtype, np.integer), f"{name} must contain integers")
    return array.astype(np.int64, copy=True)


def _rank_track_ids(ids):
    return np.array(sorted(range(len(ids)), key=lambda i: (
        hashlib.sha256(f"ba_track_holdout_v1:track:{int(ids[i])}".encode()).digest(),
        int(ids[i]))), dtype=np.int64)


def _rank_rows(track_id, rows, image_ids):
    return np.array(sorted(rows, key=lambda r: (
        hashlib.sha256(f"ba_track_holdout_v1:obs:{int(track_id)}:{int(image_ids[r])}"
                       .encode()).digest(), int(image_ids[r]))), dtype=np.int64)


def _offsets(parts):
    return np.r_[0, np.cumsum([len(x) for x in parts])].astype(np.int64)


@dataclass
class BAProblem:
    arrays: dict[str, np.ndarray]
    layout: dict[str, np.ndarray]
    images: dict[int, ColmapImage]
    cameras: dict[int, ColmapCamera]
    metadata: dict[str, Any]


def prepare_problem(arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any]) -> BAProblem:
    """Select exactly 500 fit and 5000 disjoint check tracks without label access.

    Required keys are SPEC['geometry_input_keys']; other keys are never accessed.
    All stored XYZ enter only BA initialization. Check points are re-triangulated
    from their fixed support observations for BOTH baseline and candidate.
    ``layout`` is an NPZ-ready selection ledger using original observation rows.
    """
    a = {k: np.asarray(arrays[k]).copy() for k in SPEC["geometry_input_keys"]}
    for k in ("point_track_ids", "observation_offsets", "image_id"):
        a[k] = _integer(a[k], k)
    points, ids, offsets, image_ids = (a[k] for k in
        ("point_positions", "point_track_ids", "observation_offsets", "image_id"))
    n, m = len(points), len(image_ids)
    _require(points.shape == (n, 3) and np.isfinite(points).all(), "Invalid saved points")
    _require(ids.shape == (n,) and len(np.unique(ids)) == n and (ids >= 0).all(),
             "Invalid/duplicate track IDs")
    _require(offsets.shape == (n+1,) and offsets[0] == 0 and offsets[-1] == m,
             "Invalid observation CSR extent")
    counts = np.diff(offsets)
    _require((counts >= 3).all(), "All saved tracks require >=3 observations")
    for k in ("raw_xy", "undistorted_native_xy"):
        _require(a[k].shape == (m, 2) and np.isfinite(a[k]).all(), f"Invalid {k}")
        a[k] = a[k].astype(np.float64)
    _require(a["saved_track_rms"].shape == (n,) and np.isfinite(a["saved_track_rms"]).all()
             and (a["saved_track_rms"] >= 0).all(), "Invalid saved track RMS")
    a["point_positions"] = points.astype(np.float64)
    rows_point = np.repeat(np.arange(n), counts)
    same_point = rows_point[1:] == rows_point[:-1]
    _require(np.all(image_ids[1:][same_point] > image_ids[:-1][same_point]),
             "Each saved track must have sorted unique image IDs")
    train = sorted((v for v in manifest["views"] if v["split"] == "train"),
                   key=lambda v: v["name"])
    _require(len(train) >= 3 and len({v["name"] for v in train}) == len(train)
             and len({v["image_id"] for v in train}) == len(train), "Invalid TRAIN cameras")
    images, cameras = {}, {}
    for v in train:
        cid = int(v["camera_id"])
        source = manifest["source_cameras"][str(cid)]
        if cid not in cameras:
            camera = ColmapCamera(cid, source["model"], source["width"], source["height"],
                                  np.asarray(source["params"], dtype=np.float64))
            _require(np.array_equal(camera.K, np.asarray(source["K"], dtype=np.float64))
                     and np.isfinite(camera.K).all() and min(camera.K[0, 0], camera.K[1, 1]) > 0,
                     "Source camera intrinsics differ from parameters")
            cameras[cid] = camera
        pose = np.asarray(v["w2c_original"], dtype=np.float64).copy()
        _validate_pose(pose)
        _require(np.array_equal(pose, np.asarray(v["w2c"], dtype=np.float64)),
                 "This diagnostic requires original, unchanged TRAIN cameras")
        images[int(v["image_id"])] = ColmapImage(int(v["image_id"]), v["name"], cid,
                                                pose, np.empty((0, 2)), np.empty(0, np.int64))
    _require(np.isin(image_ids, list(images)).all(), "Non-TRAIN observation in input")
    normalized = np.empty((m, 2), np.float64)
    for image_id, image in images.items():
        rows = image_ids == image_id
        K = cameras[image.camera_id].K
        normalized[rows] = (a["undistorted_native_xy"][rows] - K[:2, 2]) / K.diagonal()[:2]
    a["normalized_xy"] = normalized
    ranked = _rank_track_ids(ids)
    fit = ranked[:500]
    check = np.array([i for i in ranked[500:] if counts[i] >= 6][:5000], dtype=np.int64)
    _require(len(fit) == 500 and len(check) == 5000, "Insufficient fixed fit/check population")
    fit_rows = [np.arange(offsets[i], offsets[i+1]) for i in fit]
    all_rows, support, score = [], [], []
    for i in check:
        rows = np.arange(offsets[i], offsets[i+1])
        ordered = _rank_rows(ids[i], rows, image_ids)
        nsupport = (2 * len(rows) + 2) // 3
        all_rows.append(rows)
        support.append(ordered[:nsupport])
        score.append(ordered[nsupport:])
    layout = {
        "track_hash_order_point_indices": ranked,
        "fit_point_indices": fit, "fit_track_ids": ids[fit],
        "fit_offsets": _offsets(fit_rows), "fit_rows": np.concatenate(fit_rows),
        "check_point_indices": check, "check_track_ids": ids[check],
        "check_offsets": _offsets(all_rows), "check_rows": np.concatenate(all_rows),
        "support_offsets": _offsets(support), "support_rows": np.concatenate(support),
        "score_offsets": _offsets(score), "score_rows": np.concatenate(score),
        "score_track_indices": np.repeat(np.arange(5000), [len(x) for x in score]),
        "check_observation_track_indices": np.repeat(np.arange(5000), [len(x) for x in all_rows]),
        "train_image_ids": np.array([v["image_id"] for v in train], dtype=np.int64),
        "train_camera_ids": np.array([v["camera_id"] for v in train], dtype=np.int64),
        "train_group_ids": np.arange(len(train), dtype=np.int64) * 4 // len(train),
        "baseline_poses": np.stack([images[v["image_id"]].w2c for v in train]),
    }
    layout["score_image_ids"] = image_ids[layout["score_rows"]]
    return BAProblem(a, layout, images, cameras, {
        "train_images": [{"image_id": int(v["image_id"]), "name": v["name"]} for v in train],
        "point_count": n, "observation_count": m, "fit_tracks": 500, "check_tracks": 5000,
        "remaining_hash_rank_tracks_with_fewer_than_six_observations":
            int((counts[ranked[500:]] < 6).sum()),
        "check_support_observations": len(layout["support_rows"]),
        "check_score_observations": len(layout["score_rows"]),
        "selection_uses_labels_or_reprojection_errors": False,
    })


def _validate_pose(pose):
    _require(pose.shape == (4, 4) and np.isfinite(pose).all()
             and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
             and np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-8, rtol=0)
             and abs(np.linalg.det(pose[:3, :3]) - 1) < 1e-8, "Invalid SE(3) camera pose")


def run_fit(problem: BAProblem):
    """Return (accepted pose overrides, original BA audit); no data mutation.

    The existing solver does not export optimized fit-point positions. Its
    reported fit RMS is not an independently reconstructed after-fit residual.
    """
    a, l = problem.arrays, problem.layout
    observations = []
    for i in l["fit_point_indices"]:
        start, end = a["observation_offsets"][i:i+2]
        observations.append([TrackObservation(int(a["image_id"][r]),
                             a["normalized_xy"][r].copy(), a["raw_xy"][r].copy())
                             for r in range(start, end)])
    cloud = TriangulatedCloud(a["point_positions"][l["fit_point_indices"]].copy(),
        np.zeros((500, 3, 3)), l["fit_track_ids"].copy(),
        a["saved_track_rms"][l["fit_point_indices"]].copy(), observations,
        {"selection": SPEC["track_hash"], "check_tracks_used": 0})
    start = time.perf_counter()
    overrides, audit = bounded_bundle_adjustment(cloud, problem.images, problem.cameras,
        {i: "train" for i in problem.images}, max_points=500, max_nfev=20, seed=42)
    return overrides, dict(audit, elapsed_seconds=time.perf_counter()-start,
        fit_track_ids=l["fit_track_ids"].tolist(), check_tracks_used=0,
        optimized_fit_points_exported=False,
        fit_rms_scope="Existing solver self-report; optimized fit points are not exported")


def _reconstruct(problem, poses):
    a, l = problem.arrays, problem.layout
    points = np.full((5000, 3), np.nan)
    uv = np.full((len(l["check_rows"]), 2), np.nan)
    depth = np.full(len(l["check_rows"]), np.nan)
    residual = np.full_like(uv, np.nan)
    parallax = np.full(5000, np.nan)
    finite = np.zeros(5000, bool)
    positive = np.zeros(5000, bool)
    for i in range(5000):
        support = l["support_rows"][slice(*l["support_offsets"][i:i+2])]
        sl = slice(*l["check_offsets"][i:i+2])
        rows = l["check_rows"][sl]
        support_poses = np.stack([poses[int(j)] for j in a["image_id"][support]])
        try:
            point = triangulate_dlt(support_poses[:, :3], a["normalized_xy"][support])
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(point).all():
            continue
        points[i] = point
        all_poses = np.stack([poses[int(j)] for j in a["image_id"][rows]])
        pc = np.einsum('nij,j->ni', all_poses[:, :3, :3], point) + all_poses[:, :3, 3]
        depth[sl] = pc[:, 2]
        K = np.stack([problem.cameras[problem.images[int(j)].camera_id].K
                      for j in a["image_id"][rows]])
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            uv[sl] = np.einsum('nij,nj->ni', K, pc)[:, :2] / pc[:, 2, None]
            residual[sl] = uv[sl] - a["undistorted_native_xy"][rows]
        finite[i] = np.isfinite(uv[sl]).all() and np.isfinite(depth[sl]).all()
        positive[i] = bool((depth[sl] > 0).all())
        centers = -np.einsum('nji,nj->ni', support_poses[:, :3, :3], support_poses[:, :3, 3])
        rays = point[None] - centers
        norms = np.linalg.norm(rays, axis=1)
        if np.all(norms > 0):
            rays /= norms[:, None]
            parallax[i] = np.degrees(np.arccos(np.clip((rays @ rays.T).min(), -1., 1.)))
    # All-observation packed row -> original row lookup never changes with poses.
    row_to_packed = np.full(len(a["image_id"]), -1, np.int64)
    row_to_packed[l["check_rows"]] = np.arange(len(l["check_rows"]))
    score_packed = row_to_packed[l["score_rows"]]
    radial = np.linalg.norm(residual[score_packed], axis=1)
    per_track = np.add.reduceat(radial, l["score_offsets"][:-1]) / np.diff(l["score_offsets"])
    return {"points": points, "check_uv": uv, "check_depth": depth, "check_residual_uv": residual,
                "support_parallax_degrees": parallax, "finite": finite, "positive": positive,
                "score_radial_error": radial, "per_track_mean_radial_error": per_track}


def _summary(values):
    values = np.asarray(values, np.float64)
    if not len(values) or not np.isfinite(values).all():
        return {"count": len(values), "mean": None, "median": None, "rms": None, "p95": None}
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
                "rms": float(np.sqrt(np.mean(values**2))), "p95": float(np.percentile(values, 95))}


def evaluate(problem: BAProblem, overrides: Mapping[int, np.ndarray]):
    """Re-triangulate support-only before/after; save all selected rows, even failures."""
    _require(set(overrides) <= set(problem.images), "Non-TRAIN/unknown candidate camera")
    base_poses = {i: image.w2c.copy() for i, image in problem.images.items()}
    poses = {i: np.asarray(overrides.get(i, p), dtype=np.float64).copy() for i, p in base_poses.items()}
    for p in poses.values():
        _validate_pose(p)
    baseline = _reconstruct(problem, base_poses)
    candidate = _reconstruct(problem, poses)
    eligibility = baseline["finite"] & baseline["positive"] & (baseline["support_parallax_degrees"] >= .5)
    candidate_valid = candidate["finite"] & candidate["positive"]
    bad = eligibility & ~candidate_valid
    arrays = {f"{prefix}_{k}": v for prefix, result in [('baseline', baseline), ('candidate', candidate)]
              for k, v in result.items()}
    arrays.update(baseline_eligible=eligibility, candidate_valid=candidate_valid,
                  candidate_bad_on_fixed_population=bad,
                  candidate_poses=np.stack([poses[int(i)] for i in problem.layout["train_image_ids"]]))
    arrays["per_track_difference"] = candidate["per_track_mean_radial_error"] - baseline["per_track_mean_radial_error"]
    # Invalid candidate tracks remain in every planned denominator, even if some
    # projected error values happen to be numerically finite behind the camera.
    report = {"selected_check_tracks": 5000, "baseline_eligible_tracks": int(eligibility.sum()),
              "baseline_rejections": {"nonfinite": int((~baseline["finite"]).sum()),
                  "nonpositive_depth": int((baseline["finite"] & ~baseline["positive"]).sum()),
                  "insufficient_parallax": int((baseline["finite"] & baseline["positive"] &
                                                 ~(baseline["support_parallax_degrees"] >= .5)).sum())},
              "candidate_bad_tracks_on_fixed_population": int(bad.sum()),
              "candidate_low_parallax_on_fixed_population_descriptive":
                  int((eligibility & (candidate["support_parallax_degrees"] < .5)).sum()),
              "fixed_denominator_usable": bool(eligibility.any() and not bad.any()),
              "baseline_equal_track": _summary(baseline["per_track_mean_radial_error"][eligibility]),
              "candidate_equal_track": _summary(np.where(candidate_valid[eligibility],
                    candidate["per_track_mean_radial_error"][eligibility], np.nan)),
              "by_view": [], "by_name_group": []}
    score_keep = eligibility[problem.layout["score_track_indices"]]
    score_bad = bad[problem.layout["score_track_indices"]]
    score_ids = problem.layout["score_image_ids"]
    arrays["group_track_score_counts"] = np.zeros((4, 5000), np.int64)
    arrays["baseline_group_per_track_error"] = np.full((4, 5000), np.nan)
    arrays["candidate_group_per_track_error"] = np.full((4, 5000), np.nan)
    for group in range(4):
        ids = problem.layout["train_image_ids"][problem.layout["train_group_ids"] == group]
        mask = score_keep & np.isin(score_ids, ids)
        track_ids = problem.layout["score_track_indices"][mask]
        count = np.bincount(track_ids, minlength=5000)
        use = count > 0
        arrays["group_track_score_counts"][group] = count
        for prefix, values in [("baseline", baseline), ("candidate", candidate)]:
            sums = np.bincount(track_ids, weights=values["score_radial_error"][mask], minlength=5000)
            per_track = arrays[f"{prefix}_group_per_track_error"][group]
            per_track[use] = sums[use] / count[use]
            if prefix == "candidate":
                per_track[bad] = np.nan
        before = arrays["baseline_group_per_track_error"][group, use]
        after = arrays["candidate_group_per_track_error"][group, use]
        report["by_name_group"].append({"group": group, "score_observations": int(mask.sum()),
            "tracks": int(use.sum()), "candidate_bad_tracks": int((bad & use).sum()),
            "baseline": _summary(before), "candidate": _summary(after), "difference": _summary(after-before)})
    for image_id in problem.layout["train_image_ids"]:
        mask = score_keep & (score_ids == image_id)
        report["by_view"].append(dict(image_id=int(image_id), name=problem.images[int(image_id)].name,
                                    **_paired_summary(baseline, candidate, mask, score_bad)))
    return {"arrays": arrays, "report": report}


def _paired_summary(baseline, candidate, mask, bad):
    before = baseline["score_radial_error"][mask]
    after = np.where(bad[mask], np.nan, candidate["score_radial_error"][mask])
    return {"score_observations": int(mask.sum()), "candidate_bad_observations": int(bad[mask].sum()),
                "baseline": _summary(before), "candidate": _summary(after), "difference": _summary(after-before)}


def analyze(problem: BAProblem, evaluation, fit_audit):
    """Paired equal-track bootstrap and pose changes; no optimization or resampling choice."""
    a, r = evaluation["arrays"], evaluation["report"]
    use = a["baseline_eligible"]
    delta = a["per_track_difference"][use]
    bootstrap = None
    if r["fixed_denominator_usable"]:
        rng = np.random.default_rng(20260927)
        means = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(2000)])
        bootstrap = {"repeats": 2000, "seed": 20260927, "resampling_unit": "baseline-eligible track",
                     "mean_difference": float(delta.mean()),
                     "ci95": np.percentile(means, [2.5, 97.5]).tolist()}
    changes = []
    anchors = set(fit_audit.get("fixed_anchor_image_ids", []))
    for image_id, original, candidate in zip(problem.layout["train_image_ids"],
            problem.layout["baseline_poses"], a["candidate_poses"]):
        relative = candidate @ np.linalg.inv(original)
        rotation = Rotation.from_matrix(relative[:3, :3]).as_rotvec()
        old_center = -original[:3, :3].T @ original[:3, 3]
        new_center = -candidate[:3, :3].T @ candidate[:3, 3]
        changes.append({"image_id": int(image_id), "name": problem.images[int(image_id)].name,
            "anchor": int(image_id) in anchors, "pose_exact_original": bool(np.array_equal(original, candidate)),
            "left_rotation_vector": rotation.tolist(), "rotation_radians": float(np.linalg.norm(rotation)),
            "left_translation": relative[:3, 3].tolist(),
            "camera_center_displacement": float(np.linalg.norm(new_center-old_center))})
    return {"specification": dict(SPEC), "selection": problem.metadata, "fit": fit_audit,
            "heldout": r, "paired_bootstrap": bootstrap, "pose_changes": changes,
            "gate": _gate(r, bootstrap, fit_audit),
            "camera_coverage_descriptive": {
                "fit_observed_cameras": len(np.unique(problem.arrays["image_id"][problem.layout["fit_rows"]])),
                "optimized_cameras_solver_report": fit_audit.get("optimized_cameras"),
                "score_cameras": sum(v["score_observations"] > 0 for v in r["by_view"]),
                "score_cameras_with_nonzero_pose_change": sum(
                    v["score_observations"] > 0 and not c["pose_exact_original"]
                    for v, c in zip(r["by_view"], changes)),
                "pose_changes_are_not_optimized_parameter_count": True},
            "limitations": ["Check observations are unseen by this BA fit but passed upstream SfM selection.",
                "BA self-reported fit RMS uses unexported optimized fit points; not independently recomputed here.",
                "This is a correspondence-only camera-repair diagnostic, not an RGB or semantic performance result."]}


def _gate(report, bootstrap, fit_audit):
    coverage = {
        "eligible_tracks_ge_1000": report["baseline_eligible_tracks"] >= 1000,
        "score_cameras_ge_200_each_ge_5_rows":
            sum(v["score_observations"] >= 5 for v in report["by_view"]) >= 200,
        "all_four_groups_ge_100_score_rows":
            all(g["score_observations"] >= 100 for g in report["by_name_group"]),
    }
    before, after = report["baseline_equal_track"]["mean"], report["candidate_equal_track"]["mean"]
    relative = (before-after)/before if before is not None and before > 0 and after is not None else None
    groups_improving = sum(g["difference"]["mean"] is not None and g["difference"]["mean"] < 0
                           for g in report["by_name_group"])
    signal = {
        "ba_accepted": bool(fit_audit.get("accepted", False)),
        "all_candidates_valid_on_fixed_population": report["fixed_denominator_usable"],
        "relative_mean_reduction_ge_one_percent": relative is not None and relative >= .01,
        "paired_ci_upper_negative": bootstrap is not None and bootstrap["ci95"][1] < 0,
        "at_least_three_groups_improve": groups_improving >= 3,
    }
    coverage_passed, signal_passed = all(coverage.values()), all(signal.values())
    passed = coverage_passed and signal_passed
    return {"status": "supported_heldout_reprojection" if passed else (
                "inconclusive_coverage" if not coverage_passed else "not_supported"),
            "passed": passed, "coverage_passed": coverage_passed, "signal_passed": signal_passed,
            "coverage": coverage, "signal": signal, "relative_mean_reduction": relative,
            "improving_groups": groups_improving, "decision": "consider_matched_reference" if passed else "stop"}
