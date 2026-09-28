"""CPU-only contracts for geometric feasibility; no model fitting or rendering."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bridge_rgs import densification

spec = importlib.util.spec_from_file_location(
    "action_probe", Path(__file__).parents[1] / "scripts/prepare_action_transfer_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def cameras():
    return [{"name": name, "image_id": index + 1, "split": "train", "width": 100, "height": 100,
             "K": [[10., 0, 50], [0, 10, 50], [0, 0, 1]], "w2c": np.eye(4).tolist()}
            for index, name in enumerate(probe.NAMES)]


def track_arrays(n=20):
    return {"points": np.stack((np.arange(n), np.zeros(n), np.ones(n)), axis=1),
            "track_ids": np.arange(n)[::-1] + 10, "reprojection_error": np.zeros(n),
            "num_observations": np.full(n, 3), "observation_offsets": np.arange(n+1) * 3,
            "observation_image_ids": np.tile([2, 3, 4], n)}


def test_camera_whitelist_discards_all_image_and_label_payloads():
    views = [{**cameras()[0], "name": f"{i:03}.png", "image_id": i,
              "split": "val" if (i - 1) % 8 == 0 else "train",
              "image_path": "must-not-open", "mask_path": "must-not-open", "annotation": "secret"}
             for i in range(1, 401)]
    result = probe.selected_views({"views": views})
    assert [v["name"] for v in result] == list(probe.NAMES)
    assert all(set(v) == set(probe.VIEW_KEYS) for v in result)
    assert all(v["split"] == "train" for v in result)
    with pytest.raises(ValueError, match="legacy"):
        probe.selected_views({"views": views, "pixel_protocol": "colmap_corner_v2"})
    views[2]["split"] = "val"
    with pytest.raises(ValueError, match="TRAIN"):
        probe.selected_views({"views": views})


def test_qualified_track_selection_is_sorted_and_does_not_mutate_arrays():
    arrays = track_arrays()
    arrays["reprojection_error"][0] = 1.01
    arrays["points"][1, 0] = np.nan
    before = {k: v.copy() for k, v in arrays.items()}
    points, ids, indices = probe.qualified_tracks(arrays, {2, 3, 4})
    assert len(points) == 18 and np.all(np.diff(ids) > 0)
    np.testing.assert_array_equal(points, arrays["points"][indices])
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])


@pytest.mark.parametrize("kind", ["val", "duplicate_observation", "offset", "duplicate_track"])
def test_track_provenance_rejects_heldout_or_inflated_support(kind):
    arrays = track_arrays()
    if kind == "val":
        arrays["observation_image_ids"][0] = 99
    elif kind == "duplicate_observation":
        arrays["observation_image_ids"][0] = 3
    elif kind == "offset":
        arrays["observation_offsets"][1] += 1
    else:
        arrays["track_ids"][0] = arrays["track_ids"][1]
    with pytest.raises(ValueError):
        probe.qualified_tracks(arrays, {2, 3, 4})


def test_fps_start_tie_and_radius_are_geometry_only_and_input_order_invariant():
    points = np.array([[-3., 0, 1], [-1., 0, 1], [1., 0, 1], [3., 0, 1]])
    tracks = np.array([3, 1, 0, 2])
    records = probe.region_anchors(points, tracks, count=3, neighbors=2)
    assert records[0]["anchor_track_id"] == 0  # Median-distance tie goes to track 0.
    assert records[1]["anchor_track_id"] == 3
    assert records[0]["radius"] == 2
    permutation = [3, 2, 0, 1]
    assert records == probe.region_anchors(points[permutation], tracks[permutation], count=3, neighbors=2)
    assert records[0]["neighbor_track_ids"] == [0, 1]


def test_frustum_is_only_geometric_support_not_occlusion_and_rejects_val():
    points = np.array([[0., 0, 1], [0., 0, -.1], [100., 0, 1], [0, 0, 0]])
    view = cameras()
    visible = probe.frustum_visibility(points, view)
    assert visible.shape == (8, 4)
    assert visible[0].tolist() == [True, False, False, False]
    view[0]["split"] = "val"
    with pytest.raises(ValueError, match="TRAIN"):
        probe.frustum_visibility(points, view)


def synthetic_pair_inputs():
    return {"anchors": [{"region": i, "anchor_xyz": [0., 0, 1.], "radius": .3} for i in range(8)],
            "means": np.array([[0., 0, 1.], [.1, 0, 1.], [.2, 0, 1.], [.25, 0, 1.]]),
            "opacity": np.array([.5, .1, .5, .1]), "scales": np.ones((4, 3)) * .1,
            "scene_scale": 1., "visibility": np.ones((8, 4), bool)}


def test_two_selected_pairs_never_reuse_ids_and_missing_regions_are_not_resampled():
    inputs = synthetic_pair_inputs()
    records = probe.choose_pairs(**inputs)
    assert [(r["parent_id"], r["donor_id"]) for r in records[:2]] == [(0, 1), (2, 3)]
    assert all(r["status"] == "missing_parent" for r in records[2:])
    assert records[0]["keep_opacity_parameter_sum"] == .6
    assert records[0]["split_opacity_parameter_sum"] == 1.
    summary = probe.summarize(records)
    assert summary["status"] == "inconclusive_coverage" and summary["coverage_fraction"] == .25
    assert not summary["gpu_allowed"]
    assert "action utility" in summary["not_tested"]
    json.dumps({"records": records, "summary": summary}, allow_nan=False)


def test_parent_uses_model_scene_scale_and_both_camera_groups():
    inputs = synthetic_pair_inputs()
    inputs["scales"][:] = .003
    assert probe.choose_pairs(**inputs)[0]["parent_id"] == 0
    inputs["scene_scale"] = 2
    assert probe.choose_pairs(**inputs)[0]["status"] == "missing_parent"
    inputs["scene_scale"] = 1
    inputs["visibility"][1::2] = False
    result = probe.choose_pairs(**inputs)[0]
    assert result["large_and_fit"] == 4 and result["large_and_score"] == 0
    assert result["joint_parent_candidates"] == 0


def test_donor_is_not_selected_by_rgb_or_semantics_and_ties_use_gaussian_id():
    inputs = synthetic_pair_inputs()
    inputs["opacity"][1] = inputs["opacity"][3] = .1
    result = probe.choose_pairs(**inputs)[0]
    assert result["donor_id"] == 1
    # A donor with no A frustum support has lower proxy retention, regardless of
    # its actual, unmeasured contribution. The output reports that limitation.
    inputs["visibility"][::2, 3] = False
    result = probe.choose_pairs(**inputs)[0]
    assert result["donor_id"] == 3 and result["donor_fit_frustum_count"] == 0


def test_missing_donor_is_not_replaced_by_parent_or_outside_ball():
    inputs = synthetic_pair_inputs()
    inputs["anchors"][0]["radius"] = .01
    result = probe.choose_pairs(**inputs)[0]
    assert result["parent_id"] == 0 and result["donor_id"] is None
    assert result["status"] == "missing_donor"


def test_no_partial_region_summary_or_gpu_authorization_even_when_coverage_ready():
    with pytest.raises(ValueError, match="Incomplete"):
        probe.summarize([{"region": 0, "status": "measurable_geometric_pair"}])
    rows = [{"region": i, "status": "measurable_geometric_pair"} for i in range(8)]
    summary = probe.summarize(rows)
    assert summary["status"] == "coverage_ready_pending_separate_gpu_review"
    assert summary["gpu_allowed"] is False


def test_proposed_render_budget_counts_fits_zero_step_scoring_and_pose_checks():
    budget = probe.proposed_budget()
    assert budget["maximum_render_calls"] == 456
    assert budget["fit_renders_and_backwards"] == 128
    assert budget["zero_step_score_renders"] == budget["post_fit_score_renders"] == 128
    assert budget["camera_renders"] == 56 and budget["baseline_repeat_renders"] == 16
    assert budget["actual_render_calls"] == budget["actual_optimizer_steps"] == 0


def test_cpu_pair_proposal_has_equal_budget_preserves_attributes_and_source():
    state = {"splats.means": torch.tensor([[0., 0., 0.], [.1, .2, .3]]),
             "splats.quats": torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]]),
             "splats.log_scales": torch.tensor([[.5, .05, .005], [.02, .03, .04]]).log(),
             "splats.opacity_logits": torch.tensor([.7, .1]).logit(),
             "splats.sh0": torch.zeros(2, 1, 3), "splats.sh_rest": torch.zeros(2, 15, 3)}
    source = {k: v.clone() for k, v in state.items()}
    x = torch.linspace(-1., 1., 20)
    points = torch.stack((x, x * 0, x * 0), -1).numpy()
    result = probe.split_contract(densification, state, 0, 1, points)
    assert result["total_gaussians_before_and_after"] == 2
    assert result["active_rgb_geometry_scalars_per_arm"] == 118
    assert max(result["child_mahalanobis_offsets"]) <= .5001
    assert result["source_pair_unchanged"]
    assert not result["rendered_effect_measured"] and result["actual_optimizer_steps"] == 0
    assert all(torch.equal(state[k], v) for k, v in source.items())


def test_no_overwrite_json_or_existing_preparation(tmp_path):
    path = tmp_path / "record.json"
    probe.write_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        probe.write_json(path, {"a": 2})
    with pytest.raises(ValueError, match="existing output"):
        probe.prepare(tmp_path, tmp_path)
    with pytest.raises(ValueError, match="authorized"):
        probe.prepare(tmp_path, tmp_path / "new")


def test_entrypoint_has_no_gpu_execution_flag_and_does_not_initialize_cuda():
    assert not probe.SPEC["gpu_execution_available"]
    assert not torch.cuda.is_initialized()
