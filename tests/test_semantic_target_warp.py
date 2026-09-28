"""Synthetic target-warp contracts; no dataset pixels or probability caches."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
SCRIPT = HERE.with_name('diagnose_semantic_target_warp.py')
if not SCRIPT.exists():
    SCRIPT = HERE.parents[1]/'scripts/diagnose_semantic_target_warp.py'
spec = importlib.util.spec_from_file_location('target_warp_diagnostic', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_zero_distortion_identity_and_explicit_identity_map():
    target = np.array([[0, 0, 2, 2], [0, 1, 2, 4], [3, 1, 4, 4]], np.uint8)
    probabilities = np.eye(5, dtype=np.float32)[target]
    valid = np.full(target.shape, 255, np.uint8)
    y, x = np.indices(target.shape, dtype=np.float32)
    for back in (None, np.stack((x, y), -1)):
        result = audit.view_measurements(probabilities, target, valid, target, back)
        assert result['regions']['all']['target_disagreement'] == 0
        assert result['regions']['all']['current_errors'] == 0
        assert result['regions']['all']['oracle_miou_all_gain'] == 0
        assert result['regions']['all']['support_pixels'] == target.size
        json.dumps(result, allow_nan=False)


def test_bilinear_support_excludes_any_positive_bad_tap_not_zero_weight_outside():
    known = np.ones((2, 3), bool); known[0, 1] = False
    back = np.array([[[0., 0.], [.25, 0.], [2., 1.], [2.+1e-6, 1.], [-.1, 0.], [0., 1.]]], np.float32)
    assert audit.full_bilinear_support(known, back).tolist() == [[True, False, True, False, False, True]]
    with pytest.raises(ValueError, match='Invalid'):
        audit.full_bilinear_support(known, np.full((1, 1, 2), np.nan))


def test_argmax_happens_after_soft_warp_and_tie_uses_lowest_class():
    probabilities = np.array([[[.49, .51, 0, 0, 0], [.99, .01, 0, 0, 0]]], np.float32)
    back = np.array([[[.4, 0.]]], np.float32)
    # Nearest hard-mask transport predicts class1, soft transport predicts class0.
    assert audit.warp(probabilities, back).argmax(-1).item() == 0
    half = audit.warp(np.eye(5, dtype=np.float32)[np.array([[0, 2]])], np.array([[[.5, 0.]]], np.float32))
    assert half.argmax(-1).item() == 0


def test_oracle_repairs_only_current_error_intersected_with_target_disagreement():
    target = np.array([[0, 2, 2, 4]], np.uint8)
    prediction = np.array([[2, 0, 2, 0]], np.uint8)
    roundtrip = np.array([[0, 0, 0, 4]], np.uint8)
    result = audit.measures(target, prediction, roundtrip, np.ones_like(target, bool))
    assert result['target_disagreement'] == 2 and result['current_errors'] == 3
    assert result['error_intersection'] == 1 and result['current_correct_despite_target_disagreement'] == 1
    expected = audit.confusion(target, np.array([[2, 2, 2, 0]], np.uint8), np.ones_like(target, bool))
    np.testing.assert_array_equal(result['confusion_matrices']['oracle_intersection_only'], expected)
    assert result['oracle_miou_all_gain'] > 0


def test_unknown_native_and_original_pixels_are_not_scored_or_boundary_eroded():
    native = np.array([[0, 255, 2], [0, 2, 2]], np.uint8)
    target = native.copy(); target[1, 0] = 255
    probabilities = np.eye(5, dtype=np.float32)[np.minimum(native, 4)]
    valid = np.full(native.shape, 255, np.uint8); valid[1, 1] = 0
    result = audit.view_measurements(probabilities, native, valid, target, None)
    assert result['regions']['all']['support_pixels'] == 3
    assert result['regions']['all']['by_gt_class'][2]['support_pixels'] == 2
    # Thin structures are not eroded from scoring; the 3px band is descriptive.
    assert result['regions']['class_2_boundary3']['support_pixels'] == 3


def test_pooling_sums_counts_and_cm_instead_of_averaging_view_iou():
    a = audit.measures(np.array([[0]], np.uint8), np.array([[2]], np.uint8), np.array([[2]], np.uint8), np.array([[True]]))
    b = audit.measures(np.zeros((1, 9), np.uint8), np.zeros((1, 9), np.uint8), np.zeros((1, 9), np.uint8), np.ones((1, 9), bool))
    pooled = audit.pooled_measures([a, b])
    assert pooled['support_pixels'] == 10 and pooled['target_disagreement_fraction'] == .1
    assert pooled['scores']['actual']['iou'][0] == .9
    json.dumps(pooled, allow_nan=False)


def test_fixed_selection_uses_only_labeled_train_and_rejects_changed_list():
    labeled = [f'{i:03}.png' for i in range(1, 301) if i not in {1, 8, 13, 20}]
    # Selection must enforce the actual population/names, not merely sorted size.
    with pytest.raises(ValueError, match='259'):
        audit.fixed_views({'views': [{'name': n, 'split': 'train', 'mask_path': 'unused'} for n in labeled]})
    with pytest.raises(ValueError, match='selection'):
        audit.fixed_views({'views': [{'name': f'{i:03}.png', 'split': 'train', 'mask_path': 'unused'} for i in range(259)]})


def test_nonfinite_or_wrong_normalization_rejected():
    target = np.zeros((2, 2), np.uint8)
    p = np.ones((2, 2, 5), np.float32)
    with pytest.raises(ValueError, match='probabilities'):
        audit.view_measurements(p, target, target+255, target, None)
