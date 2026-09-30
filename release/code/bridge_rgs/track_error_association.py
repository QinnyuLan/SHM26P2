"""NumPy-only, post-hoc TRAIN track conflict / frozen H3 error association.

Exposure uses the query's ground-truth class. It is neither a deployable input
nor an out-of-fold prediction. Side-specific track means do not form a matched
within-track causal contrast: tracks may appear on one or both sides.
"""
from __future__ import annotations

import numpy as np

SPEC = {
    "protocol": "track_error_association_v1", "classes": 5, "minimum_reference_images": 3,
    "reference": "strict known observations excluding the query's entire camera group",
    "exposure": "1 - reference_count_of_query_GT_class / reference_count; exposed iff > 0",
    "estimand": "query mean within each track and side, then equal mean across present tracks per side",
    "contrast": "exposed minus compatible; larger error/Brier means worse",
    "bootstrap_repeats": 2000, "bootstrap_seed": 20260927,
    "probabilities": "cached FP32 values cast FP64; no clipping or renormalization; row sum tolerance 2e-5",
    "coverage": {"queries": 1000, "tracks": 200, "wrong_tracks": 20, "wrong_images": 4},
    "class_minimum_tracks_per_side": 20, "group_minimum_tracks_per_side": 20,
    "minimum_positive_supported_groups": 2,
}
SIDES = ("compatible", "exposed")
REFERENCE_BINS = (("3_to_5", 3, 5), ("6_to_9", 6, 9), ("ge10", 10, None))


def _integer_vector(value, name):
    a = np.asarray(value)
    if a.ndim != 1 or a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer vector")
    if a.dtype.kind == "u" and a.size and a.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds int64")
    return a.astype(np.int64, copy=False)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _mean_by_track(points, values):
    unique, inverse, counts = np.unique(points, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=values, minlength=len(unique))/counts if unique.size else np.empty(0)
    return unique, means


def _support(table, mask):
    p, error, brier = (table[k][mask] for k in ("point", "wrong", "brier"))
    _, e = _mean_by_track(p, error)
    _, b = _mean_by_track(p, brier)
    exposure_mask = mask & np.isfinite(table["exposure"])
    _, exposure = _mean_by_track(table["point"][exposure_mask], table["exposure"][exposure_mask])
    counts = table["reference_count"][mask]
    return {
        "queries": int(mask.sum()), "tracks": int(np.unique(p).size),
        "image_ids": np.unique(table["image"][mask]).tolist(),
        "images": int(np.unique(table["image"][mask]).size),
        "target_class_counts": np.bincount(table["label"][mask], minlength=5).tolist(),
        "target_group_counts": np.bincount(table["group"][mask], minlength=4).tolist(),
        "reference_count_bins": {
            name: int(((counts >= low) & ((counts <= high) if high is not None else True)).sum())
            for name, low, high in REFERENCE_BINS},
        "reference_count_min": int(counts.min()) if counts.size else None,
        "reference_count_max": int(counts.max()) if counts.size else None,
        "wrong_queries": int(error.sum()), "wrong_tracks": int(np.unique(p[error > 0]).size),
        "wrong_images": int(np.unique(table["image"][mask][error > 0]).size),
        "confusion_matrix_rows_target_columns_prediction": np.bincount(
            table["label"][mask]*5+table["prediction"][mask], minlength=25).reshape(5, 5).tolist(),
        "track_equal_error": float(e.mean()) if e.size else None,
        "track_equal_brier": float(b.mean()) if b.size else None,
        "track_equal_exposure": float(exposure.mean()) if exposure.size else None,
        "query_exposure_quantiles_0_50_100": np.quantile(table["exposure"][exposure_mask], [0, .5, 1]).tolist()
            if exposure_mask.any() else None,
    }


def joint_track_bootstrap(points, exposed, wrong, brier, *, repeats=2000, seed=20260927):
    """Both sides retain the same resampled multiplicity of every eligible track.

    A missing side has zero denominator, not a zero error. Replicates lacking a
    side are undefined and their number is reported alongside finite intervals.
    """
    repeats = _positive_integer(repeats, "bootstrap_repeats")
    points = _integer_vector(points, "bootstrap points")
    exposed, wrong, brier = np.asarray(exposed), np.asarray(wrong, np.float64), np.asarray(brier, np.float64)
    if (exposed.dtype != np.bool_ or exposed.shape != points.shape or wrong.shape != points.shape
            or brier.shape != points.shape or not np.isfinite(wrong).all() or not np.isfinite(brier).all()):
        raise ValueError("Aligned finite bootstrap values and boolean sides required")
    unique, inverse = np.unique(points, return_inverse=True)
    side_index = inverse*2+exposed.astype(np.int64)
    counts = np.bincount(side_index, minlength=len(unique)*2).reshape(-1, 2)
    present = counts > 0
    metric_means = {}
    for name, values in (("error", wrong), ("brier", brier)):
        totals = np.bincount(side_index, weights=values, minlength=len(unique)*2).reshape(-1, 2)
        metric_means[name] = np.divide(
            totals, counts, out=np.zeros(totals.shape, dtype=np.float64), where=present)
    result = {"unit": "all eligible tracks, shared multiplicities for both sides",
              "tracks": len(unique), "tracks_on_both_sides": int(present.all(1).sum()),
              "repeats": repeats, "seed": int(seed), "metrics": {}}
    distributions = {name: np.full(repeats, np.nan) for name in metric_means}
    if unique.size:
        rng = np.random.default_rng(seed)
        chunk = max(1, min(64, 1_000_000//len(unique)))
        for first in range(0, repeats, chunk):
            last = min(first+chunk, repeats)
            sample = rng.integers(len(unique), size=(last-first, len(unique)))
            denominator = present[sample].sum(1)
            valid = (denominator > 0).all(1)
            for name, means in metric_means.items():
                total = means[sample].sum(1)
                side_mean = np.divide(total, denominator, out=np.full_like(total, np.nan), where=denominator > 0)
                distributions[name][first:last][valid] = side_mean[valid, 1]-side_mean[valid, 0]
    for name, values in distributions.items():
        finite = values[np.isfinite(values)]
        result["metrics"][name] = {"difference_95_interval": np.quantile(finite, [.025, .975]).tolist()
                                  if finite.size else None,
                                  "finite_replicates": int(finite.size),
                                  "undefined_replicates": int(repeats-finite.size)}
    return result


def _comparison(table, mask, *, bootstrap_repeats=None, seed=20260927):
    result = {"sides": {name: _support(table, mask & (table["exposed"] == side))
                        for side, name in enumerate(SIDES)}}
    result["tracks_on_both_sides"] = int(np.intersect1d(
        table["point"][mask & ~table["exposed"]], table["point"][mask & table["exposed"]]).size)
    result["difference_exposed_minus_compatible"] = {}
    for metric in ("error", "brier"):
        a, b = (result["sides"][side]["track_equal_"+metric] for side in SIDES)
        result["difference_exposed_minus_compatible"][metric] = None if a is None or b is None else float(b-a)
    if bootstrap_repeats is not None:
        result["bootstrap"] = joint_track_bootstrap(
            table["point"][mask], table["exposed"][mask], table["wrong"][mask], table["brier"][mask],
            repeats=bootstrap_repeats, seed=seed)
    return result


def association_gate(report):
    support = report["cross_group_eligible"]
    coverage = {"eligible_queries_ge1000": support["queries"] >= 1000,
                "eligible_tracks_ge200": support["tracks"] >= 200,
                "wrong_tracks_ge20": support["wrong_tracks"] >= 20,
                "wrong_images_ge4": support["wrong_images"] >= 4}
    interval = report["association"]["bootstrap"]["metrics"]["brier"]["difference_95_interval"]
    clauses = {"global_brier_difference_ci_lower_positive": interval is not None and interval[0] > 0}
    for label in (0, 2):
        comparison = report["by_target_class"][str(label)]
        clauses[f"class_{label}_each_side_ge20_tracks"] = all(
            side["tracks"] >= 20 for side in comparison["sides"].values())
        difference = comparison["difference_exposed_minus_compatible"]["brier"]
        clauses[f"class_{label}_brier_difference_positive"] = difference is not None and difference > 0
    supported_groups, positive_groups = [], []
    for group in range(4):
        comparison = report["by_target_group"][str(group)]
        if all(side["tracks"] >= 20 for side in comparison["sides"].values()):
            supported_groups.append(group)
            difference = comparison["difference_exposed_minus_compatible"]["brier"]
            if difference is not None and difference > 0:
                positive_groups.append(group)
    clauses["at_least_two_supported_groups_positive"] = len(positive_groups) >= 2
    status = ("inconclusive_error_coverage" if not all(coverage.values()) else
              "conditional_error_association" if all(clauses.values()) else "not_supported")
    return {"status": status, "coverage_conditions": coverage, "association_conditions": clauses,
            "supported_target_groups": supported_groups, "positive_supported_target_groups": positive_groups,
            "scope": "post-hoc fitted-TRAIN conditional association; no causal, generalization or adoption claim"}


def _analyze(points, images, labels, strict, groups, probability, targets, point_count, repeats, seed):
    counts = np.bincount(points[strict]*20+groups[strict]*5+labels[strict],
                         minlength=point_count*20).reshape(point_count, 4, 5)
    queryable = strict[targets]
    all_reference = counts.sum(1)[points[targets]]-counts[points[targets], groups[targets]]
    rows, prob = targets[queryable], probability[queryable]
    p, y, group = points[rows], labels[rows], groups[rows]
    reference = all_reference[queryable]
    ref_count = reference.sum(1)
    target_votes = reference[np.arange(len(rows)), y]
    exposure = np.divide(ref_count-target_votes, ref_count, out=np.full(len(rows), np.nan), where=ref_count > 0)
    prediction = prob.argmax(1)
    wrong = (prediction != y).astype(np.float64)
    residual = prob.copy()
    residual[np.arange(len(rows)), y] -= 1
    brier = np.square(residual).sum(1)
    table = {"point": p, "image": images[rows], "label": y, "group": group,
             "reference_count": ref_count, "exposure": exposure,
             "exposed": target_votes < ref_count, "wrong": wrong, "brier": brier,
             "prediction": prediction}
    eligible = ref_count >= 3
    all_query_support = _support(table, np.ones(len(rows), dtype=bool))
    # No references imply undefined exposure, but error/Brier remain observed.
    all_query_support["queries_with_zero_reference"] = int((ref_count == 0).sum())
    finite_exp = np.isfinite(exposure)
    all_query_support["exposure_defined_queries"] = int(finite_exp.sum())
    _, finite_point_exp = _mean_by_track(p[finite_exp], exposure[finite_exp])
    all_query_support["track_equal_exposure"] = float(finite_point_exp.mean()) if finite_point_exp.size else None
    all_query_support["query_exposure_quantiles_0_50_100"] = np.quantile(exposure[finite_exp], [0, .5, 1]).tolist() \
        if finite_exp.any() else None
    result = {
        "input_query_rows": len(targets), "excluded_non_strict_query_rows": int((~queryable).sum()),
        "all_queryable_strict": all_query_support,
        "cross_group_eligible": _support(table, eligible),
        "queries_with_fewer_than_three_cross_group_references": int((~eligible).sum()),
        "association": _comparison(table, eligible, bootstrap_repeats=repeats, seed=seed),
        "by_target_class": {}, "by_reference_count": {}, "by_class_and_reference_count": {}, "by_target_group": {},
    }
    for label in range(5):
        result["by_target_class"][str(label)] = _comparison(table, eligible & (y == label))
    for name, low, high in REFERENCE_BINS:
        mask = eligible & (ref_count >= low) & ((ref_count <= high) if high is not None else True)
        result["by_reference_count"][name] = _comparison(table, mask)
        result["by_class_and_reference_count"][name] = {
            str(label): _comparison(table, mask & (y == label)) for label in range(5)}
    for group in range(4):
        result["by_target_group"][str(group)] = _comparison(table, eligible & (table["group"] == group))
    maximum = reference.max(1) if len(rows) else np.empty(0, dtype=np.int64)
    tied = (reference == maximum[:, None]).sum(1) > 1
    majority = reference.argmax(1)
    result["reference_hard_vote_descriptive"] = {
        "eligible_queries": int(eligible.sum()), "tied_maximum_queries": int((eligible & tied).sum()),
        "unique_majority_queries": int((eligible & ~tied).sum()),
        "unique_majority_disagrees_with_target": int((eligible & ~tied & (majority != y)).sum()),
        "smallest_class_argmax_disagrees_with_target_including_ties": int((eligible & (majority != y)).sum()),
        "rule": "ties choose the smallest class only in this descriptive count; votes do not select observations or enter the gate",
    }
    query_arrays = {
        "target_row_indices": targets.copy(), "point_index": points[targets].copy(),
        "image_id": images[targets].copy(), "label": labels[targets].copy(), "group": groups[targets].copy(),
        "query_prob": probability.copy(), "queryable_strict": queryable.copy(),
        "reference_counts": all_reference, "reference_count": all_reference.sum(1),
        "eligible": queryable & (all_reference.sum(1) >= 3),
        "exposure": np.full(len(targets), np.nan), "error": np.full(len(targets), np.nan),
        "brier": np.full(len(targets), np.nan), "exposed": np.zeros(len(targets), dtype=bool),
    }
    for name, value in (("exposure", exposure), ("error", wrong), ("brier", brier),
                        ("exposed", target_votes < ref_count)):
        query_arrays[name][queryable] = value
    return result, query_arrays


def analyze_track_error_association(point_index, image_id, label, strict, group,
                                    query_prob, target_row_indices, *, point_count, query_image_ids,
                                    duplicate_touched_point_indices=None, bootstrap_repeats=2000, seed=20260927):
    """Return ``{'report': JSONable dict, 'query_arrays': arrays for NPZ}``.

    No files are read. Input labels are not inferred or re-sampled here.

    The caller binds the fixed 16 query-image IDs, camera-name grouping and exact
    primary strict mask, including primary/legacy label agreement at queries.
    ``duplicate_touched_point_indices`` removes whole tracks from observations
    AND queries before rebuilding the separate, non-gated sensitivity analysis.
    """
    point_count = _positive_integer(point_count, "point_count")
    repeats = _positive_integer(bootstrap_repeats, "bootstrap_repeats")
    points, images, labels, groups, targets, query_ids = (_integer_vector(v, name) for v, name in (
        (point_index, "point_index"), (image_id, "image_id"), (label, "label"),
        (group, "group"), (target_row_indices, "target_row_indices"), (query_image_ids, "query_image_ids")))
    strict = np.asarray(strict)
    probability = np.asarray(query_prob, dtype=np.float64)
    n = len(points)
    if (any(a.shape != (n,) for a in (images, labels, groups, strict)) or strict.dtype != np.bool_
            or probability.shape != (len(targets), 5)):
        raise ValueError("Aligned observation vectors, boolean strict mask and Q by 5 probabilities required")
    if ((points < 0).any() or (points >= point_count).any() or (groups < 0).any() or (groups > 3).any()
            or not np.isin(labels, [-1, 0, 1, 2, 3, 4, 255]).all()
            or (strict & ((labels < 0) | (labels > 4))).any()):
        raise ValueError("Invalid point/group/label or strict mask includes an unknown label")
    if ((targets < 0).any() or (targets >= n).any() or len(np.unique(targets)) != len(targets)
            or not len(query_ids) or len(np.unique(query_ids)) != len(query_ids)
            or not np.isin(images[targets], query_ids).all()):
        raise ValueError("Target rows must be unique, in range and in the fixed query-image whitelist")
    if (not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any()
            or not np.allclose(probability.sum(1), 1., rtol=0, atol=2e-5)):
        raise ValueError("Invalid probability values/sums; no clipping or renormalization is permitted")
    order = np.lexsort((images, points))
    if n > 1 and ((points[order[1:]] == points[order[:-1]]) & (images[order[1:]] == images[order[:-1]])).any():
        raise ValueError("Duplicate point/image observation")
    order = np.argsort(images, kind="stable")
    if n > 1 and ((images[order[1:]] == images[order[:-1]]) & (groups[order[1:]] != groups[order[:-1]])).any():
        raise ValueError("The same image ID cannot belong to multiple camera groups")
    main, query_arrays = _analyze(points, images, labels, strict, groups, probability, targets, point_count, repeats, seed)
    main["gate"] = association_gate(main)
    result = {"specification": dict(SPEC, bootstrap_repeats=repeats, bootstrap_seed=int(seed)),
              "point_count": point_count, "input_observations": n, "query_image_ids": query_ids.tolist(),
              "maximum_query_probability_sum_error": float(np.abs(probability.sum(1)-1).max()) if len(targets) else None,
              "main": main, "duplicate_exclusion_sensitivity": None}
    query_arrays["duplicate_excluded"] = np.zeros(len(targets), dtype=bool)
    query_arrays["sensitivity_eligible"] = query_arrays["eligible"].copy()
    if duplicate_touched_point_indices is not None:
        excluded = _integer_vector(duplicate_touched_point_indices, "duplicate_touched_point_indices")
        if (excluded < 0).any() or (excluded >= point_count).any() or len(np.unique(excluded)) != len(excluded):
            raise ValueError("Duplicate sensitivity requires unique in-range point indices")
        retained = ~np.isin(points, excluded)
        retained_targets = retained[targets]
        remap = np.full(n, -1, dtype=np.int64)
        remap[retained] = np.arange(retained.sum())
        sensitivity, sensitivity_arrays = _analyze(
            points[retained], images[retained], labels[retained], strict[retained],
            groups[retained], probability[retained_targets], remap[targets[retained_targets]],
            point_count, None, seed)
        sensitivity.update(excluded_tracks=len(excluded), excluded_observations=int((~retained).sum()),
                           excluded_queries=int((~retained_targets).sum()), gate_evaluated=False,
                           scope="Whole-track deletion from all observations and queries; descriptive sensitivity only")
        result["duplicate_exclusion_sensitivity"] = sensitivity
        query_arrays["duplicate_excluded"] = ~retained_targets
        query_arrays["sensitivity_eligible"][:] = False
        query_arrays["sensitivity_eligible"][retained_targets] = sensitivity_arrays["eligible"]
    return {"report": result, "query_arrays": query_arrays}
