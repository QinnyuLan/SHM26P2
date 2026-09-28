"""Synthetic NumPy contracts; no dataset, label file, Torch or GPU access."""
import json

import numpy as np
import pytest

from bridge_rgs.track_semantic_audit import audit_track_labels, evidence_gate, track_bootstrap


def population(n=16):
    return [{"image_id": i+1, "name": f"{i+1:03}.png"} for i in range(n)]


def audit(points, images, labels, *, errors=None, distances=None, rays=None, point_count=None,
          train_images=None, repeats=100):
    n = len(points)
    return audit_track_labels(
        np.asarray(points, dtype=np.int64), np.asarray(images, dtype=np.int64),
        np.asarray(labels, dtype=np.int64), np.zeros(n) if errors is None else errors,
        np.full(n, 11.) if distances is None else distances,
        np.tile([0., 0., 1.], (n, 1)) if rays is None else rays,
        point_count=max(points, default=0)+1 if point_count is None else point_count,
        train_images=population() if train_images is None else train_images,
        bootstrap_repeats=repeats)


def test_minority_rate_is_not_pair_disagreement_and_has_two_weightings():
    # Sizes 4 and 2, minority counts 1 and 1; the singleton is not eligible.
    result = audit([0]*4+[1]*2+[2], [1, 2, 3, 4, 1, 2, 3], [0, 0, 0, 2, 0, 2, 4])
    summary = result["strict"]
    assert summary["tracks_ge2"] == 2
    assert summary["observations_on_tracks_ge2"] == 6
    assert summary["minority_observations_on_tracks_ge2"] == 2
    assert summary["weighted_conflict"] == pytest.approx(1/3)
    assert summary["point_equal_conflict"] == pytest.approx((1/4+1/2)/2)
    assert summary["bg_cable_conflict_tracks"] == 2
    assert summary["cable_images_on_tracks_ge2"] == 2
    assert summary["bg_cable_conflict_images"] == 4
    assert result["gate"]["status"] == "inconclusive_coverage"
    json.dumps(result, allow_nan=False)


def test_loo_removes_target_before_acceptance_and_reports_minority_selection_bias():
    result = audit([0]*4, [1, 2, 3, 4], [0, 0, 0, 2])
    loo = result["strict"]["loo"]
    assert loo["accepted_targets"] == 1
    assert loo["discordant_targets"] == loo["discordant_tracks"] == 1
    assert loo["conditional_observation_error"] == loo["conditional_point_equal_error"] == 1
    assert loo["accepted_target_coverage"] == .25
    assert loo["confusion_matrix_rows_target_columns_consensus"][2][0] == 1
    assert "minority" in loo["selection_bias"]
    # The primary conflict is .25, not the biased conditional LOO error of 1.
    assert result["strict"]["point_equal_conflict"] == .25


def test_loo_inclusive_eighty_percent_and_three_other_distinct_images():
    first = audit([0]*3, [1, 2, 3], [1, 1, 1])["strict"]["loo"]
    assert first["accepted_targets"] == 0
    second = audit([0]*6, [1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 2])["strict"]["loo"]
    assert second["accepted_targets"] == 6  # leaving a 0 gives exactly 4/5.
    assert second["conditional_observation_error"] == pytest.approx(1/6)
    assert second["confusion_matrix_rows_target_columns_consensus"][0][0] == 5


def test_unknown_and_no_annotation_are_na_not_background_or_consensus():
    result = audit([0]*5, [1, 2, 3, 4, 5], [0, 2, 255, -1, 255],
                   errors=[0, 0, np.nan, np.nan, np.nan],
                   distances=[11, 11, np.nan, np.nan, np.nan])
    assert result["unknown_or_invalid_observations"] == 2
    assert result["no_annotation_observations"] == 1
    assert result["known_observations"] == 2
    assert result["strict"]["known_label_counts"] == [1, 0, 1, 0, 0]
    assert result["strict"]["loo"]["accepted_targets"] == 0


def test_nine_strata_exact_boundary_inclusivity_and_strict():
    err = np.repeat([1., 2., 2.01], 3)
    dist = np.tile([3., 10., 10.01], 3)
    result = audit(list(range(9)), [1]*9, [0]*9, errors=err, distances=dist)
    assert len(result["strata"]) == 9
    assert all(s["known_observations"] == 1 for s in result["strata"].values())
    assert sum(s["known_observations"] for s in result["strata"].values()) == 9
    assert result["strict"]["known_observations"] == 1
    assert all("bootstrap" not in s for s in result["strata"].values())


def test_camera_name_groups_include_unannotated_cameras_and_drop_observations():
    # IDs are deliberately uncorrelated with the sorted names. All 350 count.
    cameras = [{"image_id": 1000-i, "name": f"{i:03}.png"} for i in range(350)]
    result = audit([0]*4, [1000, 912, 825, 737], [0, 0, 2, 2], train_images=list(reversed(cameras)))
    assert [g["image_count"] for g in result["camera_groups"]] == [88, 87, 88, 87]
    assert [g["known_observations"] for g in result["camera_groups"]] == [1]*4
    assert result["camera_groups"][0]["images"][0]["name"] == "000.png"
    assert result["camera_groups"][1]["images"][0]["name"] == "088.png"
    for drop in result["leave_group_out"]:
        assert drop["strict"]["known_observations"] == 3
        assert drop["strict"]["tracks_ge2"] == 1
        assert drop["strict"]["point_equal_conflict"] == pytest.approx(1/3)
        assert "bootstrap" not in drop["strict"]
        assert "no OOF" in drop["scope"]


def test_track_bootstrap_matches_independent_resampling_and_preserves_global_rng():
    np.random.seed(54)
    before = np.random.get_state()
    result = track_bootstrap([1, 1, 0], [4, 2, 7], repeats=2000, seed=20260927)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    draws = np.random.default_rng(20260927).integers(3, size=(2000, 3))
    minority, size = np.array([1., 1., 0.]), np.array([4., 2., 7.])
    weighted = minority[draws].sum(1)/size[draws].sum(1)
    equal = (minority/size)[draws].mean(1)
    np.testing.assert_array_equal(result["weighted_conflict_95_interval"], np.quantile(weighted, [.025, .975]))
    np.testing.assert_array_equal(result["point_equal_conflict_95_interval"], np.quantile(equal, [.025, .975]))


def test_gate_separates_coverage_from_signal_and_uses_strict_ci_lower():
    strict = {"tracks_ge2": 1000, "cable_tracks_ge2": 50, "cable_images_on_tracks_ge2": 8,
              "bg_cable_conflict_tracks": 50, "bootstrap": {"point_equal_conflict_95_interval": [.01001, .03]}}
    assert evidence_gate(strict)["status"] == "conditional_conflict_signal_only"
    strict["bootstrap"]["point_equal_conflict_95_interval"][0] = .01
    assert evidence_gate(strict)["status"] == "not_supported"
    strict["bootstrap"]["point_equal_conflict_95_interval"][0] = .02
    strict["bg_cable_conflict_tracks"] = 49
    assert evidence_gate(strict)["status"] == "not_supported"
    strict["cable_images_on_tracks_ge2"] = 7
    assert evidence_gate(strict)["status"] == "inconclusive_coverage"


def test_full_synthetic_gate_counts_and_ci_no_class_sampling():
    n = 1000
    points = np.repeat(np.arange(n), 2)
    images = np.ones(2*n, dtype=np.int64)
    images[1::2] = 2
    labels = np.ones(2*n, dtype=np.int64)
    for p in range(50):
        images[2*p:2*p+2] = [p % 8+1, p % 8+9]
        labels[2*p:2*p+2] = [2, 0]
    result = audit(points, images, labels, repeats=2000)
    assert result["strict"]["tracks_ge2"] == 1000
    assert result["strict"]["cable_tracks_ge2"] == 50
    assert result["strict"]["cable_images_on_tracks_ge2"] == 8
    assert result["strict"]["point_equal_conflict"] == .025
    assert result["gate"]["status"] == "conditional_conflict_signal_only"


def test_direction_summary_normalizes_rays_and_does_not_fit_labels():
    result = audit([0, 0, 1, 1], [1, 2, 1, 2], [0, 2, 1, 1],
                   rays=[[10., 0, 0], [0, 3., 0], [0, 0, 1.], [0, 0, 7.]])
    desc = result["direction_description"]
    assert desc["background_vs_cable_pairs"]["point_equal_mean"] == pytest.approx(90.)
    assert desc["conflicted_tracks_ge2"]["point_equal_mean"] == pytest.approx(90.)
    assert desc["unanimous_tracks_ge2"]["point_equal_mean"] == pytest.approx(0.)
    assert desc["all_tracks_ge2"]["point_equal_mean"] == pytest.approx(45.)
    assert "NOT mean pair angle" in desc["quantity"]


def test_empty_population_statistics_serialize_as_null_not_nan():
    result = audit([], [], [], point_count=10, rays=np.empty((0, 3)))
    assert result["strict"]["weighted_conflict"] is None
    assert result["strict"]["point_equal_conflict"] is None
    assert result["strict"]["loo"]["conditional_observation_error"] is None
    assert result["strict"]["bootstrap"]["point_equal_conflict_95_interval"] is None
    assert result["gate"]["status"] == "inconclusive_coverage"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("labels", [[0, 2], [255, -1]])
def test_duplicate_same_track_image_is_rejected_even_for_excluded_labels(labels):
    with pytest.raises(ValueError, match="Duplicate"):
        audit([0, 0], [1, 1], labels)


@pytest.mark.parametrize("which,value,match", [
    ("labels", [5], "label"), ("points", [-1], "point"), ("images", [100], "non-TRAIN"),
    ("errors", [np.nan], "finite"), ("distances", [-.1], "nonnegative"),
    ("rays", [[0, 0, 0]], "nonzero"), ("rays", [[0, 0, np.nan]], "finite"),
])
def test_invalid_inputs_fail_closed(which, value, match):
    args = {"points": [0], "images": [1], "labels": [0], "point_count": 2}
    args[which] = value
    with pytest.raises(ValueError, match=match):
        audit(**args)


def test_distinct_image_id_not_shared_calibration_id_and_array_inputs_unchanged():
    p = np.array([0, 0, 0, 0], dtype=np.int64)
    i = np.array([1, 2, 3, 4], dtype=np.int64)
    y = np.array([2, 2, 2, 2], dtype=np.int64)
    rays = np.tile([0., 0., 9.], (4, 1))
    copies = [a.copy() for a in (p, i, y, rays)]
    result = audit(p, i, y, rays=rays)
    assert result["strict"]["loo"]["accepted_targets"] == 4
    for actual, expected in zip((p, i, y, rays), copies):
        np.testing.assert_array_equal(actual, expected)


def test_noninteger_observation_identifiers_and_duplicate_names_rejected():
    with pytest.raises(ValueError, match="integer array"):
        audit_track_labels(np.array([0.]), np.array([1]), np.array([0]), [0.], [11.], [[0, 0, 1]],
                           point_count=1, train_images=population())
    cameras = population()
    cameras[1]["name"] = cameras[0]["name"]
    with pytest.raises(ValueError, match="unique"):
        audit([0], [1], [0], train_images=cameras)
