"""Pure CPU coverage and optimizer-bound tests; no inference or optimization."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from bridge_rgs import densification

ROOT = Path(__file__).parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = load(ROOT / "scripts/prepare_action_transfer_probe.py", "probe_v1_tests")
v2 = load(ROOT / "scripts/prepare_action_transfer_probe_v2.py", "probe_v2_tests")


def views():
    return [{"name": name, "split": "train", "width": 100, "height": 100,
             "K": [[10., 0, 50], [0, 10, 50], [0, 0, 1]], "w2c": np.eye(4).tolist()}
            for name in v1.NAMES]


def test_common_anchor_rule_uses_both_fixed_groups_without_pixel_payloads():
    points = np.stack((np.linspace(-1, 1, 20), np.zeros(20), np.ones(20)), axis=1)
    points = np.concatenate((points, [[0, 0, -1], [100, 0, 1]]))
    keep, visible = v2.common_anchor_pool(v1, points, views())
    assert keep.tolist() == [True] * 20 + [False] * 2
    assert visible.shape == (8, 22)
    cameras = views()
    for index in [1, 3, 5, 7]:
        cameras[index]["w2c"][0][3] = 100
    with pytest.raises(ValueError, match="do not relax"):
        v2.common_anchor_pool(v1, points, cameras)


def test_only_anchor_population_changes_radius_uses_original_full_reference():
    points = np.stack((np.arange(100, dtype=float), np.zeros(100), np.ones(100)), axis=1)
    tracks = np.arange(100) + 1000
    common = np.zeros(100, bool)
    common[::4] = True
    records = v2.select_anchors(v1, points, tracks, common)
    sparse = v1.region_anchors(points[common], tracks[common], count=8, neighbors=16)
    assert [r["anchor_track_id"] for r in records] == [r["anchor_track_id"] for r in sparse]
    assert all(r["anchor_track_id"] in tracks[common] for r in records)
    assert any(r["radius"] < s["radius"] for r, s in zip(records, sparse))
    for record in records:
        d = np.square(points - np.asarray(record["anchor_xyz"])).sum(1)
        nearest = np.lexsort((tracks, d))[:16]
        assert record["radius"] == math.sqrt(d[nearest[-1]])
        assert record["neighbor_track_ids"] == tracks[nearest].tolist()
        assert record["radius_reference_pool"] == "original_all_qualified_train_points"


def test_v2_fps_and_radii_are_input_order_invariant():
    points = np.stack((np.arange(100, dtype=float), np.zeros(100), np.ones(100)), axis=1)
    ids = np.arange(100) + 1000
    keep = (ids % 4 == 0)
    first = v2.select_anchors(v1, points, ids, keep)
    order = np.random.default_rng(20260926).permutation(len(points))
    assert first == v2.select_anchors(v1, points[order], ids[order], keep[order])


def options():
    return {"lr": .005, "betas": (.9, .999), "eps": 1e-15, "weight_decay": 0, "amsgrad": False}


def test_adam_bound_covers_constant_and_arbitrary_gradient_histories_without_optimizer():
    o = options()
    bound = v2.adam_coordinate_bound(o)
    assert bound["coordinate_displacement_upper_bound"] >= 4 * o["lr"]
    generator = np.random.default_rng(29)
    for _ in range(100):
        grads = generator.standard_normal(4) * 10 ** generator.uniform(-3, 3, 4)
        m = v = movement = 0.
        for step, gradient in enumerate(grads, 1):
            m = .9 * m + .1 * gradient
            v = .999 * v + .001 * gradient ** 2
            update = o["lr"] * (m / (1 - .9 ** step)) / (math.sqrt(v / (1 - .999 ** step)) + o["eps"])
            movement += abs(update)
        assert movement <= bound["coordinate_displacement_upper_bound"] + 1e-14
    assert bound["per_step_normalized_update_upper_bounds"][0] == pytest.approx(1)


def test_single_step_cauchy_factor_is_achievable_for_its_optimal_gradient_history():
    o = options()
    result = v2.adam_coordinate_bound(o)
    powers = np.arange(3, -1, -1, dtype=float)
    a = .1 * .9 ** powers / (1 - .9 ** 4)
    b = .001 * .999 ** powers / (1 - .999 ** 4)
    gradients = a / b
    ratio = np.dot(a, gradients) / np.sqrt(np.dot(b, gradients ** 2))
    assert ratio == pytest.approx(result["per_step_normalized_update_upper_bounds"][-1])


@pytest.mark.parametrize("key,value", [("weight_decay", .01), ("amsgrad", True), ("lr", 0), ("eps", -1), ("betas", (1, .9))])
def test_adam_bound_rejects_other_optimization_contracts(key, value):
    o = options()
    o[key] = value
    with pytest.raises(ValueError):
        v2.adam_coordinate_bound(o)


def test_capacity_bounds_preserve_checkpoint_learning_rates_not_invented_tuning():
    model = {"optimizers": {k: {"param_groups": [{**options(), "params": [0]}]} for k in v1.ACTIVE_KEYS}}
    model["optimizers"]["means"]["param_groups"][0]["lr"] = 2.6018353271484376e-5
    result = v2.capacity_bounds(model, v1.ACTIVE_KEYS)
    assert result["active_rgb_geometry_scalars_per_branch"] == 118
    assert result["means_l2_upper_bound"] < .00019
    assert result["scale_relative_increase_upper_bound"] < .021
    assert "not measured gradients" in result["means"]["assumptions"]
    json.dumps(result, allow_nan=False)


def test_projection_size_proxy_reports_geometry_only_and_no_behind_camera_area():
    state = {"splats.means": torch.tensor([[0., 0., 10.], [0., 0., -1.]]),
             "splats.quats": torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]]),
             "splats.log_scales": torch.ones(2, 3).log()}
    result = v2.projected_size_proxy(densification, state, [0, 1], views())
    area = 9 * math.pi * 1.3
    assert result[0]["parent_donor"][0]["unclipped_3sigma_ellipse_area_pixels"] == pytest.approx(area)
    assert result[0]["parent_donor"][0]["center_inside_image"]
    assert result[0]["parent_donor"][1] is None
    assert "opacity" not in json.dumps(result)


def test_budgets_keep_original_maximum_and_record_concrete_coverage_separately():
    assert v1.proposed_budget()["maximum_render_calls"] == 456
    seven = v1.proposed_budget(7)
    assert seven["maximum_render_calls"] == 408
    assert seven["fit_renders_and_backwards"] == 112
    assert not hasattr(v2, "execute") and not hasattr(v2, "worker")


def test_v2_refuses_overwrite_and_original_v1_output(tmp_path):
    with pytest.raises(ValueError, match="existing"):
        v2.prepare(tmp_path, tmp_path)
    with pytest.raises(ValueError, match="separate"):
        v2.prepare(tmp_path, tmp_path / "v1")
    assert v2.V1_DIRECTORY != v2.V2_DIRECTORY
