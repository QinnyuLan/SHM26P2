"""CPU tests for all-TRAIN scope, original indexing, and objective aggregation."""
import copy
import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from bridge_rgs.losses import masked_mean
from bridge_rgs.raw_grid import appearance_rgb_loss

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_appearance_train_objective.py"
spec = importlib.util.spec_from_file_location("appearance_train_objective_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def inputs():
    views = [{"name": f"{index:03}.png", "split": "train", "w2c": np.eye(4).tolist(),
              "w2c_original": np.eye(4).tolist()} for index in range(350)]
    order = list(range(349, -1, -1))
    source = {"pixel_protocol": {"id": "colmap_corner_v2"}, "views": list(reversed(views))}
    plan = {"view_names": [v["name"] for v in views], "training_order": order*8 + order[:200]}
    return source, plan


def test_full_train_index_is_sorted_but_enumeration_matches_original_first_cycle():
    source, plan = inputs()
    source["views"].append({"name": "val.png", "split": "val"})
    views, order, counts = audit.fixed_views(source, plan)
    assert len(views) == 350 and views[0]["name"] == "000.png"
    assert order == list(range(349, -1, -1))
    assert sum(counts.values()) == 3000 and counts[349] == 9 and counts[0] == 8
    assert all(view["split"] == "train" for view in views)


@pytest.mark.parametrize("kind", ["fewer", "duplicate", "wrong_order", "repeat_cycle", "out_of_range", "pose", "legacy"])
def test_invalid_manifest_or_training_schedule_rejected(kind):
    source, plan = inputs()
    if kind == "fewer":
        source["views"].pop()
    elif kind == "duplicate":
        source["views"][0] = copy.deepcopy(source["views"][1])
    elif kind == "wrong_order":
        plan["view_names"] = list(reversed(plan["view_names"]))
    elif kind == "repeat_cycle":
        plan["training_order"][0] = plan["training_order"][1]
    elif kind == "out_of_range":
        plan["training_order"][-1] = 350
    elif kind == "pose":
        source["views"][0]["w2c"][0][3] = .01
    elif kind == "legacy":
        source.pop("pixel_protocol")
    with pytest.raises(ValueError):
        audit.fixed_views(source, plan)


def rows():
    return [{"name": f"{i:03}.png", "sorted_index": i, **dict.fromkeys(audit.SPEC["metrics"], .1)} for i in range(350)]


def test_camera_equal_and_original_sample_weighted_changes_are_distinct():
    base = rows()
    final = copy.deepcopy(base)
    for record in final[:200]:
        for metric in audit.SPEC["metrics"]:
            record[metric] += .01
    for record in final[200:]:
        for metric in audit.SPEC["metrics"]:
            record[metric] -= .01
    counts = {str(i): 9 if i < 200 else 8 for i in range(350)}
    summary = audit.summarize_pair(base, final, counts)
    for metric in audit.SPEC["metrics"]:
        assert summary[metric]["final_minus_base_equal_camera_mean"] == pytest.approx(.5/350)
        assert summary[metric]["final_minus_base_original_sampling_weighted_mean"] == pytest.approx(6/3000)
        assert summary[metric]["improved_views"] == 150
        assert summary[metric]["worsened_views"] == 200


def test_pair_cannot_silently_drop_or_reorder_views():
    base, final = rows(), rows()
    counts = {str(i): 9 if i < 200 else 8 for i in range(350)}
    final.reverse()
    with pytest.raises(ValueError, match="differ"):
        audit.summarize_pair(base, final, counts)
    with pytest.raises(ValueError, match="350"):
        audit.summarize_pair(base[:-1], final[:-1], counts)


def test_pixel_allowlist_denies_semantic_labels_val_and_unknown_paths(tmp_path):
    train = (tmp_path/"train_rgb.png").resolve()
    counts, calls = Counter(), []
    reader = audit.pixel_reader([str(train)], counts, lambda path, *a, **k: calls.append(path))
    reader(train, 1)
    for path in (tmp_path/"train_label.png", tmp_path/"val_rgb.png", tmp_path/"valid_other.png"):
        with pytest.raises(ValueError, match="outside TRAIN"):
            reader(path, 1)
    assert dict(counts) == {str(train): 1} and calls == [train]


def test_measurement_reuses_training_loss_and_masks_mse_consistently():
    prediction = torch.full((13, 15, 3), .2)
    target = torch.full_like(prediction, .1)
    valid = torch.ones((13, 15), dtype=torch.bool)
    valid[:1] = False
    measured = audit.measured_values(prediction, target, valid, appearance_rgb_loss, masked_mean)
    exact, stats = appearance_rgb_loss(prediction, target, valid)
    assert measured["combined_loss"] == float(exact)
    assert measured["l1"] == float(stats["l1"])
    assert measured["mse"] == pytest.approx(.01)
    assert measured["rgb_pixels"] == 12*15 and measured["ssim7_centers"] == 6*9
    prediction[:1] = 100
    changed = audit.measured_values(prediction, target, valid, appearance_rgb_loss, masked_mean)
    assert changed == measured


def test_scope_is_only_two_matching_pairs_without_new_candidates():
    assert audit.SPEC["arms"] == ["00_native", "01_original"]
    assert audit.SPEC["checkpoints_per_arm"] == ["base", "matching_final"]
    assert audit.SPEC["renders"] == 2*2*350 == 1400
    assert audit.SPEC["optimizer_steps"] == 0
    assert audit.SPEC["max_wall_seconds"] == 240
