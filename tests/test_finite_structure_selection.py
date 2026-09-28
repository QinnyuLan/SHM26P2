import copy
import json

import numpy as np
import pytest

from bridge_rgs.finite_structure_selection import (
    evaluate_on_b,
    select_on_a,
    validate_fixed_support,
)


def rows(names=("A0", "A1"), *, rgb=.01, local=.01, pixels=100, ce=1., tp=50, fp=20):
    cm = np.eye(5, dtype=np.int64) * 10
    cm[0, 0], cm[0, 2], cm[2, 0], cm[2, 2] = 100-fp, fp, 100-tp, tp
    return [{"name": name, "full_rgb_mse": rgb, "local_rgb_mse": local,
             "local_pixels": pixels, "raw_ce": ce, "confusion_matrix": cm.copy()}
            | {"local_support_sha256": "a" * 64} for name in names]


def selection():
    keep = rows()
    candidates = {"action0": rows(ce=.9), "action1": rows(ce=.8),
                  "action2": rows(ce=1.2)}
    return select_on_a(keep, copy.deepcopy(keep), candidates,
                       {"action0": 2., "action1": 1., "action2": 0.})


def b_states():
    names = ("B0", "B1")
    keep = rows(names)
    candidates = {"action0": rows(names, ce=.95, tp=55),
                  "action1": rows(names, ce=.9, tp=65, rgb=.0099),
                  "action2": rows(names, ce=.01, tp=99, rgb=.001)}
    return keep, candidates


def test_shared_feasibility_blocks_rgb_worse_semantic_winner():
    keep = rows()
    candidates = {"action0": rows(ce=.5, rgb=.011), "action1": rows(ce=.9)}
    result = select_on_a(keep, keep, candidates, {"action0": 100., "action1": 1.})
    assert result["feasible"] == ["action1"]
    assert result["gradient_choice"] == result["refit_choice"] == "action1"
    assert not result["rgb_guards"]["action0"]["passed"]


def test_local_damage_is_not_hidden_by_full_image_improvement():
    keep = rows()
    result = select_on_a(keep, keep, {"action0": rows(rgb=.009, local=.011, ce=.1)},
                         {"action0": 1.})
    assert result["feasible"] == []
    assert result["gradient_choice"] == result["refit_choice"] == "keep"


def test_deterministic_ties_and_strict_ce_keep():
    keep = rows()
    tied = {"action1": rows(ce=.9), "action0": rows(ce=.9)}
    result = select_on_a(keep, keep, tied, {"action0": 1., "action1": 1.})
    assert result["gradient_choice"] == result["refit_choice"] == "action0"
    result = select_on_a(keep, keep, {"action0": rows()}, {"action0": 1.})
    assert result["gradient_choice"] == "action0" and result["refit_choice"] == "keep"


def test_strict_per_view_cap_does_not_add_repeat_tau():
    keep = rows()
    repeat = rows(rgb=.02)
    candidate = rows(rgb=.01)
    candidate[0]["full_rgb_mse"] = .01002
    candidate[1]["full_rgb_mse"] = .00998
    result = select_on_a(keep, repeat, {"action0": candidate}, {"action0": 1.})
    guard = result["rgb_guards"]["action0"]
    assert guard["aggregate"]["full_rgb_mse"]["passed"]
    assert not guard["per_view"][0]["passed"] and result["feasible"] == []


def test_aggregate_tau_uses_view_mean_repeat_difference():
    keep = rows()
    repeat = rows(rgb=.010001)
    result = select_on_a(keep, repeat, {"action0": rows(rgb=.010005)}, {"action0": 1.})
    guard = result["rgb_guards"]["action0"]["aggregate"]["full_rgb_mse"]
    assert guard["tau"] == pytest.approx(1e-5)
    assert guard["passed"] and result["feasible"] == ["action0"]


def test_zero_local_support_skips_only_local_metric():
    keep = rows(local=None, pixels=0)
    result = select_on_a(keep, keep, {"action0": rows(local=None, pixels=0, rgb=.011)},
                         {"action0": 1.})
    guard = result["rgb_guards"]["action0"]
    assert guard["aggregate"]["local_rgb_mse"]["skipped_empty_support"]
    assert not guard["passed"]
    keep[1].update(local_pixels=10, local_rgb_mse=.03)
    result = select_on_a(keep, keep, {}, {})
    assert result["summaries"]["keep"]["local_rgb_mse"] == .03
    assert result["summaries"]["keep"]["local_views"] == 1


def test_support_count_and_zero_convention_must_match():
    keep, changed = rows(), rows()
    changed[0]["local_pixels"] += 1
    with pytest.raises(ValueError, match="support counts"):
        select_on_a(keep, keep, {"action0": changed}, {"action0": 1.})
    changed[0].update(local_pixels=0, local_rgb_mse=0.)
    with pytest.raises(ValueError, match="None"):
        select_on_a(keep, keep, {"action0": changed}, {"action0": 1.})


def test_support_hash_required_and_equal_even_when_pixel_counts_match():
    keep, changed = rows(), rows()
    changed[0]["local_support_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="hashes"):
        select_on_a(keep, keep, {"action0": changed}, {"action0": 1.})
    with pytest.raises(ValueError, match="hashes"):
        validate_fixed_support(keep, changed)
    del changed[0]["local_support_sha256"]
    with pytest.raises(ValueError, match="SHA256"):
        validate_fixed_support(changed)
    assert validate_fixed_support(keep, copy.deepcopy(keep))[0]["local_pixels"] == 100


def test_local_aggregate_is_view_equal_not_pixel_count_weighted():
    keep = rows()
    keep[0].update(local_pixels=1, local_rgb_mse=.01)
    keep[1].update(local_pixels=10000, local_rgb_mse=.03)
    result = select_on_a(keep, keep, {}, {})
    assert result["summaries"]["keep"]["local_rgb_mse"] == pytest.approx(.02)


def test_fixed_gt_class_support_cannot_change_with_state():
    keep, changed = rows(), rows()
    changed[0]["confusion_matrix"][2, 0] -= 1
    changed[0]["confusion_matrix"][0, 0] += 1
    with pytest.raises(ValueError, match="GT class support"):
        select_on_a(keep, keep, {"action0": changed}, {"action0": 1.})
    with pytest.raises(ValueError, match="GT class support"):
        validate_fixed_support(keep, changed)


def test_b_uses_locked_choices_even_if_unselected_candidate_is_better():
    locked = selection()
    keep, candidates = b_states()
    before = json.dumps(locked, sort_keys=True, allow_nan=False)
    result = evaluate_on_b(locked, keep, keep, candidates)
    assert result["gradient_choice"] == "action0" and result["refit_choice"] == "action1"
    assert result["passed"] and result["status"] == "necessary_signal_present"
    assert result["summaries"]["action1"]["cable_iou"] == pytest.approx(65/120)
    assert result["summaries"]["action1"]["cable_fn"] == 70  # Two pooled views.
    assert before == json.dumps(locked, sort_keys=True, allow_nan=False)
    json.dumps(result, allow_nan=False)
    candidates["action1"] = rows(("B0", "B1"), ce=1.1)
    failed = evaluate_on_b(locked, keep, keep, candidates)
    assert not failed["passed"] and failed["refit_choice"] == "action1"


@pytest.mark.parametrize("field,value", [("full_rgb_mse", .02), ("local_rgb_mse", .02)])
def test_b_rgb_damage_rejects_better_ce_and_cable(field, value):
    keep, candidates = b_states()
    for row in candidates["action1"]:
        row[field] = value
    result = evaluate_on_b(selection(), keep, keep, candidates)
    assert not result["passed"]
    assert result["comparisons"]["keep"]["raw_ce_improved"]
    assert result["comparisons"]["keep"]["cable_iou_improved"]


def test_b_same_choice_is_not_a_selector_success():
    keep_a = rows()
    locked = select_on_a(keep_a, keep_a, {"action0": rows(ce=.9)}, {"action0": 1.})
    keep = rows(("B0", "B1"))
    result = evaluate_on_b(locked, keep, keep,
                           {"action0": rows(("B0", "B1"), ce=.5, tp=90)})
    assert not result["passed"] and result["status"] == "selection_not_distinguished"


def test_b_cable_guard_and_a_b_separation():
    keep, candidates = b_states()
    candidates["action1"] = rows(("B0", "B1"), ce=.1, tp=40)
    result = evaluate_on_b(selection(), keep, keep, candidates)
    assert not result["comparisons"]["keep"]["cable_iou_improved"]
    with pytest.raises(ValueError, match="overlaps"):
        evaluate_on_b(selection(), rows(), rows(),
                      {key: rows() for key in candidates})


@pytest.mark.parametrize("key,value", [("full_rgb_mse", np.nan), ("raw_ce", np.inf),
                                       ("local_rgb_mse", -1), ("local_pixels", True)])
def test_invalid_numbers_fail_loud(key, value):
    keep = rows()
    keep[0][key] = value
    with pytest.raises(ValueError):
        select_on_a(keep, keep, {}, {})


def test_b_gradient_reference_uses_keep_noise_and_reference_floor():
    keep, candidates = b_states()
    for row in candidates["action0"]:
        row["full_rgb_mse"] = .008
    result = evaluate_on_b(selection(), keep, keep, candidates)
    comparison = result["comparisons"]["gradient"]["rgb_guard"]
    assert comparison["aggregate"]["full_rgb_mse"]["tau"] == pytest.approx(8e-9)
    assert not comparison["passed"]  # Better than keep does not suffice.


def test_zero_full_reference_still_enforces_strict_per_view_cap():
    keep = rows(rgb=0., local=0.)
    result = select_on_a(keep, keep, {"action0": rows(rgb=1e-15, local=0.)},
                         {"action0": 1.})
    assert result["rgb_guards"]["action0"]["aggregate"]["full_rgb_mse"]["passed"]
    assert not result["rgb_guards"]["action0"]["passed"]
