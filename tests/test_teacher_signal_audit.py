"""CPU contracts for descriptive TRAIN-only distillation-signal statistics."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "scripts/audit_teacher_distillation_signal.py"
specification = importlib.util.spec_from_file_location("teacher_signal_audit", path)
audit = importlib.util.module_from_spec(specification)
specification.loader.exec_module(audit)


def probabilities(labels):
    return np.eye(5, dtype=np.float32)[labels].transpose(2, 0, 1)


def test_fixed_sample_excludes_val_and_unlabeled_and_is_order_independent():
    views = [{"name": f"{i:03d}.png", "split": "train", "mask_path": "label"} for i in range(259)]
    views += [{"name": "999.png", "split": "val", "mask_path": "label"},
              {"name": "998.png", "split": "train", "mask_path": None}]
    indices, selected = audit.fixed_views({"views": list(reversed(views))})
    assert indices == [i * 258 // 15 for i in range(16)]
    assert [view["name"] for view in selected] == [f"{i:03d}.png" for i in indices]
    assert len({view["name"] for view in selected}) == 16
    with pytest.raises(ValueError, match="259"):
        audit.fixed_views({"views": views[:-3]})


def test_kl_direction_and_fp16_probability_renormalization():
    teacher = np.array([.8, .2, 0, 0, 0], np.float16).reshape(5, 1, 1)
    student = np.array([.2, .8, 0, 0, 0], np.float32).reshape(5, 1, 1)
    assert audit.soft_kl(teacher, student).item() == pytest.approx(.6 * np.log(4), abs=1e-6)
    np.testing.assert_allclose(audit.soft_kl(teacher, teacher), 0, atol=1e-7)
    with pytest.raises(ValueError, match="Probability mass"):
        audit.soft_kl(np.zeros((5, 1, 1)), student)


def test_directional_counts_bins_threshold_equality_and_class_coverage():
    gt = np.array([[0, 1, 2, 4, 2, 4, 255]], np.uint8)
    teacher = probabilities(np.array([[0, 0, 2, 0, 2, 4, 0]]))
    student = probabilities(np.array([[1, 1, 1, 0, 2, 4, 0]]))
    confidence = np.array([[.8, .9, .6, .95, 1., .2, .8]], np.float32)
    classes = ["background", "deck", "stay_cable", "tower", "foundation"]
    result = audit.summarize_view(gt, np.ones_like(gt, bool), teacher, student, confidence, classes)
    full = result["regions"]["all"]
    assert full["all"]["pixels"] == 6
    assert full["all"]["teacher_correct_student_wrong"] == 2
    assert full["all"]["teacher_wrong_student_correct"] == 1
    assert full["all"]["both_correct"] == 2 and full["all"]["both_wrong"] == 1
    assert full["accepted_ge_0.8"]["pixels"] == 4
    assert full["accepted_ge_0.8"]["teacher_correct_student_wrong"] == 1
    assert full["accepted_ge_0.8"]["teacher_wrong_student_correct"] == 1
    assert sum(value["pixels"] for value in full["confidence_bins"].values()) == 6
    scored = audit.rates(result)
    assert scored["regions"]["all"]["accepted_coverage"] == pytest.approx(4 / 6)
    assert scored["regions"]["all"]["teacher_correction_retention_at_0.8"] == .5
    assert scored["by_gt_class"]["foundation"]["accepted_coverage"] == .5
    assert scored["by_teacher_predicted_class"]["background"]["accepted_ge_0.8"]["teacher_wrong_per_pixel"] == pytest.approx(2 / 3)
    twice = audit.rates(audit.merge_counts(result, result))
    assert twice["regions"]["all"]["all"]["pixels"] == 12
    assert twice["regions"]["all"]["accepted_coverage"] == pytest.approx(4 / 6)


def test_boundary_band_uses_only_gt_and_erodes_unknown_support():
    gt = np.zeros((17, 17), np.uint8)
    gt[:, 8:] = 2
    keep = np.ones_like(gt, bool)
    band = audit.boundary_band(gt, keep)
    assert band[8, 7] and band[8, 8]
    assert not band[0].any() and not band[-1].any()
    assert not band[:, :3].any() and not band[:, -3:].any()
    keep[8, 8] = False
    assert not audit.boundary_band(gt, keep)[6:11, 6:11].any()
