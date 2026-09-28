"""Descriptive TRAIN track-label consistency, using arrays supplied by a collector.

No image loading, optimization, renderer, or physical-surface feasibility claim.
The original/legacy pixel protocols must be audited in separate calls. Distance
and reprojection-error units are native pixels; their construction is the
collector's responsibility. World rays are camera-to-observation directions.
"""
from __future__ import annotations

import numpy as np

SPEC = {
    "protocol": "track_semantic_audit_v1", "classes": 5, "cable_class": 2,
    "no_annotation_label": -1, "ignore_label": 255,
    "strict_reprojection_error_max": 1., "strict_boundary_distance_min_exclusive": 10.,
    "minimum_track_observations": 2, "loo_minimum_other_images": 3,
    "loo_minimum_share": .8, "camera_groups": 4,
    "bootstrap_repeats": 2000, "bootstrap_seed": 20260927,
    "coverage": {"minimum_tracks": 1000, "minimum_cable_tracks": 50, "minimum_cable_images": 8},
    "signal": {"point_equal_conflict_ci_lower_exclusive": .01, "minimum_bg_cable_tracks": 50},
}


def _int_vector(value, name):
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if array.dtype.kind == "u" and array.size and array.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds int64 range")
    return array.astype(np.int64, copy=False)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _ratio(a, b):
    return float(a/b) if b else None


def _counts(points, labels, point_count):
    return np.bincount(points*5+labels, minlength=point_count*5).reshape(point_count, 5)


def track_bootstrap(minority, observations, repeats=2000, seed=20260927):
    """Resample eligible tracks, never individual observations, with replacement.

    The same sampled tracks form the weighted ratio and point-equal mean.
    Chunking bounds temporary memory; no dense observation-by-observation matrix.
    """
    minority = np.asarray(minority, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    repeats = _positive_integer(repeats, "bootstrap_repeats")
    if (minority.ndim != 1 or observations.shape != minority.shape
            or not np.isfinite(minority).all() or not np.isfinite(observations).all()
            or (minority < 0).any() or (observations < 2).any()
            or (minority > observations).any()):
        raise ValueError("Bootstrap requires aligned valid minority counts and track sizes >= 2")
    result = {"unit": "track", "tracks": int(minority.size), "repeats": repeats, "seed": int(seed),
              "weighted_conflict_95_interval": None, "point_equal_conflict_95_interval": None}
    if not minority.size:
        return result
    rng = np.random.default_rng(seed)
    weighted, equal = np.empty(repeats), np.empty(repeats)
    fraction = minority/observations
    chunk = max(1, min(64, 2_000_000//minority.size))
    for first in range(0, repeats, chunk):
        last = min(first+chunk, repeats)
        draw = rng.integers(minority.size, size=(last-first, minority.size))
        weighted[first:last] = minority[draw].sum(1)/observations[draw].sum(1)
        equal[first:last] = fraction[draw].mean(1)
    result["weighted_conflict_95_interval"] = np.quantile(weighted, [.025, .975]).tolist()
    result["point_equal_conflict_95_interval"] = np.quantile(equal, [.025, .975]).tolist()
    return result


def _loo(points, images, labels, counts, sizes):
    """Leave the query observation out before testing the consensus acceptance."""
    other = counts[points].copy()
    other[np.arange(len(points)), labels] -= 1
    other_count = sizes[points]-1
    maximum = other.max(1) if len(points) else np.empty(0, dtype=np.int64)
    # 5*max >= 4*n is exact for the fixed 0.8 threshold.
    accepted = (other_count >= 3) & (5*maximum >= 4*other_count)
    p, y = points[accepted], labels[accepted]
    prediction = other[accepted].argmax(1)
    wrong = prediction != y
    cm = np.bincount(5*y+prediction, minlength=25).reshape(5, 5)
    target_count = np.bincount(p, minlength=len(counts))
    errors = np.bincount(p, weights=wrong.astype(np.int64), minlength=len(counts))
    participating = target_count > 0
    observed_tracks = sizes > 0
    class_totals = np.bincount(labels, minlength=5)
    class_accepted = cm.sum(1)
    return {
        "acceptance": "at least 3 other distinct image IDs and other-label maximum share >= 0.8",
        "selection_bias": "acceptance depends on the other labels; minority targets can be preferentially accepted",
        "confusion_matrix_rows_target_columns_consensus": cm.tolist(),
        "known_target_observations": len(points), "accepted_targets": int(accepted.sum()),
        "accepted_target_coverage": _ratio(int(accepted.sum()), len(points)),
        "observed_tracks": int(observed_tracks.sum()), "accepted_tracks": int(participating.sum()),
        "accepted_track_coverage": _ratio(int(participating.sum()), int(observed_tracks.sum())),
        "accepted_images": int(np.unique(images[accepted]).size),
        "discordant_targets": int(wrong.sum()), "discordant_tracks": int(np.unique(p[wrong]).size),
        "conditional_observation_error": _ratio(int(wrong.sum()), int(accepted.sum())),
        "conditional_point_equal_error": float((errors[participating]/target_count[participating]).mean())
            if participating.any() else None,
        "class_target_counts": class_totals.tolist(), "class_accepted_counts": class_accepted.tolist(),
        "class_target_coverage": [_ratio(int(a), int(b)) for a, b in zip(class_accepted, class_totals)],
    }


def summarize_tracks(points, images, labels, point_count, *, bootstrap_repeats=None, seed=20260927):
    """Internal summary of known, unique track/image observations only."""
    counts = _counts(points, labels, point_count)
    sizes = counts.sum(1)
    eligible = sizes >= 2
    minority = sizes-counts.max(1)
    bg_cable = eligible & (counts[:, 0] > 0) & (counts[:, 2] > 0)
    cable = eligible & (counts[:, 2] > 0)
    represented = eligible[points]
    cable_observations = represented & (labels == 2)
    result = {
        "known_observations": len(points), "observed_tracks": int((sizes > 0).sum()),
        "tracks_ge2": int(eligible.sum()), "observations_on_tracks_ge2": int(sizes[eligible].sum()),
        "minority_observations_on_tracks_ge2": int(minority[eligible].sum()),
        "weighted_conflict": _ratio(int(minority[eligible].sum()), int(sizes[eligible].sum())),
        "point_equal_conflict": float((minority[eligible]/sizes[eligible]).mean()) if eligible.any() else None,
        "conflicted_tracks": int((eligible & (minority > 0)).sum()),
        "bg_cable_conflict_tracks": int(bg_cable.sum()),
        "bg_cable_conflict_images": int(np.unique(images[bg_cable[points]]).size),
        "cable_tracks_ge2": int(cable.sum()),
        "cable_observations_on_tracks_ge2": int(cable_observations.sum()),
        "cable_images_on_tracks_ge2": int(np.unique(images[cable_observations]).size),
        "known_label_counts": np.bincount(labels, minlength=5).tolist(),
        "loo": _loo(points, images, labels, counts, sizes),
    }
    if bootstrap_repeats is not None:
        result["bootstrap"] = track_bootstrap(minority[eligible], sizes[eligible], bootstrap_repeats, seed)
    return result


def evidence_gate(strict):
    """Predeclared resource gate, not a test of GS representational impossibility."""
    coverage = {"tracks_ge2_at_least_1000": strict["tracks_ge2"] >= 1000,
                "cable_tracks_ge2_at_least_50": strict["cable_tracks_ge2"] >= 50,
                "cable_images_at_least_8": strict["cable_images_on_tracks_ge2"] >= 8}
    interval = strict.get("bootstrap", {}).get("point_equal_conflict_95_interval")
    signal = {"point_equal_conflict_ci_lower_gt_1pct": interval is not None and interval[0] > .01,
              "bg_cable_conflict_tracks_at_least_50": strict["bg_cable_conflict_tracks"] >= 50}
    status = ("inconclusive_coverage" if not all(coverage.values()) else
              "conditional_conflict_signal_only" if all(signal.values()) else "not_supported")
    return {"status": status, "coverage_conditions": coverage, "signal_conditions": signal,
            "interpretation": "conditional observed hard-label consistency only; not a geometry, direction, or model adoption claim"}


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {"tracks": int(values.size), "point_equal_mean": float(values.mean()) if values.size else None,
            "quantiles_05_50_95": np.quantile(values, [.05, .5, .95]).tolist() if values.size else None}


def _direction_description(points, labels, rays, point_count):
    counts = _counts(points, labels, point_count)
    sizes = counts.sum(1)
    sums = np.zeros((point_count, 3), dtype=np.float64)
    np.add.at(sums, points, rays)
    eligible = sizes >= 2
    mean_pair_cos = np.zeros(point_count)
    mean_pair_cos[eligible] = ((sums[eligible]**2).sum(1)-sizes[eligible])/(sizes[eligible]*(sizes[eligible]-1))
    angle = np.rad2deg(np.arccos(np.clip(mean_pair_cos, -1, 1)))
    unanimous = eligible & (counts.max(1) == sizes)
    bg, cable = np.zeros_like(sums), np.zeros_like(sums)
    np.add.at(bg, points[labels == 0], rays[labels == 0])
    np.add.at(cable, points[labels == 2], rays[labels == 2])
    mixed = (counts[:, 0] > 0) & (counts[:, 2] > 0)
    cross_cos = (bg[mixed]*cable[mixed]).sum(1)/(counts[mixed, 0]*counts[mixed, 2])
    return {"scope": "strict observations only; descriptive, no fit or gate",
            "quantity": "degrees of acos(mean pair cosine), NOT mean pair angle or maximum angular span",
            "all_tracks_ge2": _distribution(angle[eligible]),
            "unanimous_tracks_ge2": _distribution(angle[unanimous]),
            "conflicted_tracks_ge2": _distribution(angle[eligible & ~unanimous]),
            "background_vs_cable_pairs": _distribution(np.rad2deg(np.arccos(np.clip(cross_cos, -1, 1))))}


def audit_track_labels(point_index, image_id, label, reprojection_error, boundary_distance, world_ray,
                       *, point_count, train_images, bootstrap_repeats=2000, seed=20260927):
    """Audit one pixel-label protocol; callers must not merge original and legacy.

    ``train_images`` contains ALL TRAIN ``{'image_id': int, 'name': str}`` records,
    including unannotated cameras. Group = floor(sorted-name index * 4 / count).
    -1 has no annotation, 255 has no usable known label. For these excluded rows,
    error/distance may be NaN. Rays must remain finite/nonzero and are normalized.
    Duplicate (point_index, image_id) is rejected even for excluded labels.
    """
    point_count = _positive_integer(point_count, "point_count")
    bootstrap_repeats = _positive_integer(bootstrap_repeats, "bootstrap_repeats")
    points, images, labels = (_int_vector(x, name) for x, name in
                               ((point_index, "point_index"), (image_id, "image_id"), (label, "label")))
    error, distance, rays = (np.asarray(x, dtype=np.float64) for x in
                              (reprojection_error, boundary_distance, world_ray))
    n = len(points)
    if (images.shape != (n,) or labels.shape != (n,) or error.shape != (n,)
            or distance.shape != (n,) or rays.shape != (n, 3)):
        raise ValueError("All observation arrays must be row-aligned; rays must be N by 3")
    if ((points < 0).any() or (points >= point_count).any()
            or not np.isin(labels, [-1, 0, 1, 2, 3, 4, 255]).all()):
        raise ValueError("Invalid point index or label code")
    known = (labels >= 0) & (labels < 5)
    if (not np.isfinite(error[known]).all() or not np.isfinite(distance[known]).all()
            or (error[known] < 0).any() or (distance[known] < 0).any()
            or not np.isfinite(rays).all()):
        raise ValueError("Known labels require finite nonnegative pixel error/distance; rays must be finite")
    ray_scale = np.abs(rays).max(1) if n else np.empty(0)
    if (ray_scale <= 0).any():
        raise ValueError("World rays must be nonzero")
    rays = rays/ray_scale[:, None]
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    order = np.lexsort((images, points))
    if n > 1 and ((points[order[1:]] == points[order[:-1]]) & (images[order[1:]] == images[order[:-1]])).any():
        raise ValueError("Duplicate track/image observation")
    population = list(train_images)
    if len(population) < 4:
        raise ValueError("At least four TRAIN images are required for four camera-name groups")
    for item in population:
        if (not isinstance(item, dict) or set(item) != {"image_id", "name"}
                or not isinstance(item["image_id"], (int, np.integer)) or isinstance(item["image_id"], (bool, np.bool_))
                or not isinstance(item["name"], str) or not item["name"]):
            raise ValueError("Each TRAIN image must have exactly integer image_id and nonempty name")
    population = sorted(({"image_id": int(v["image_id"]), "name": v["name"]} for v in population),
                        key=lambda v: v["name"])
    ids = [int(v["image_id"]) for v in population]
    if len(set(ids)) != len(ids) or len({v["name"] for v in population}) != len(population):
        raise ValueError("TRAIN image IDs and names must each be unique")
    if not np.isin(images, ids).all():
        raise ValueError("Observation references a non-TRAIN image ID")
    groups = {image: i*4//len(ids) for i, image in enumerate(ids)}
    row_group = np.asarray([groups[int(image)] for image in images], dtype=np.int64)
    strict = known & (error <= 1.) & (distance > 10.)

    def summary(mask, bootstrap=False):
        return summarize_tracks(points[mask], images[mask], labels[mask], point_count,
                                bootstrap_repeats=bootstrap_repeats if bootstrap else None, seed=seed)

    result = {
        "specification": dict(SPEC, bootstrap_repeats=bootstrap_repeats, bootstrap_seed=int(seed)),
        "point_count": point_count, "input_observations": n,
        "no_annotation_observations": int((labels == -1).sum()),
        "unknown_or_invalid_observations": int((labels == 255).sum()),
        "known_observations": int(known.sum()), "all": summary(known, True),
        "strict": summary(strict, True), "strata": {}, "camera_groups": [], "leave_group_out": [],
    }
    error_bins = (("le1", error <= 1), ("gt1_le2", (error > 1) & (error <= 2)), ("gt2", error > 2))
    boundary_bins = (("le3", distance <= 3), ("gt3_le10", (distance > 3) & (distance <= 10)),
                     ("gt10", distance > 10))
    for ename, emask in error_bins:
        for dname, dmask in boundary_bins:
            result["strata"][f"error_{ename}__boundary_{dname}"] = summary(known & emask & dmask)
    for group in range(4):
        selected = row_group == group
        camera_records = [dict(v, group=group) for i, v in enumerate(population) if i*4//len(ids) == group]
        result["camera_groups"].append({
            "group": group, "images": camera_records, "image_count": len(camera_records),
            "input_observations": int(selected.sum()), "known_observations": int((selected & known).sum()),
            "strict_known_observations": int((selected & strict).sum()),
            "strict_cable_observations": int((selected & strict & (labels == 2)).sum()),
        })
        dropped = summary(strict & ~selected)
        probe = evidence_gate(dropped)
        # No deletion bootstrap: these are point-estimate diagnostics, not gate reruns.
        result["leave_group_out"].append({
            "dropped_group": group, "strict": dropped,
            "coverage_conditions": probe["coverage_conditions"],
            "descriptive_signal_conditions": {
                "point_equal_conflict_gt_1pct": dropped["point_equal_conflict"] is not None
                    and dropped["point_equal_conflict"] > .01,
                "bg_cable_conflict_tracks_at_least_50": dropped["bg_cable_conflict_tracks"] >= 50,
            },
            "scope": "camera-block observation deletion only; no reconstruction, no OOF claim, no new CI or adoption gate",
        })
    result["gate"] = evidence_gate(result["strict"])
    result["direction_description"] = _direction_description(points[strict], labels[strict], rays[strict], point_count)
    return result
