import numpy as np
import pytest

from bridge_rgs.rgb_semantic_strata import class_error_sums, class_summary, paired_class_errors


def test_class_and_ignore_partition_reconcile_full_image_rgb_error():
    target = np.zeros((2, 3, 3), np.uint8)
    prediction = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    mask = np.array([[0, 1, 2], [3, 4, 255]], np.uint8)
    counts, sums = class_error_sums(prediction, target, mask)
    np.testing.assert_array_equal(counts, np.ones(6))
    expected = np.square((prediction.astype(np.float32)/255).astype(np.float64)).sum()
    assert sums.sum() == pytest.approx(expected, abs=1e-15)
    mask[1, 2] = 3
    changed_counts, changed_sums = class_error_sums(prediction, target, mask)
    assert changed_counts[3] == 2 and changed_counts[5] == 0
    assert changed_sums[3] == sums[3]+sums[5] and changed_sums[5] == 0


def test_empty_class_is_missing_and_pooled_psnr_is_not_mean_view_psnr():
    counts = np.zeros((2, 6), np.int64); counts[:, 2] = [1, 9]
    sums = np.zeros_like(counts, dtype=np.float64); sums[:, 2] = [.03, 2.7]
    summary = class_summary(counts, sums)
    assert summary['stay_cable']['pooled_mse'] == pytest.approx(.091)
    assert summary['deck']['pooled_mse'] is None
    report = paired_class_errors(counts, sums, sums, repeats=100)
    assert report['classes']['deck']['finite_replicates'] == 0
    assert report['classes']['stay_cable']['difference_mse'] == 0
    assert report['classes']['stay_cable']['psnr_95_interval'] == [0., 0.]
    assert 'ignore' not in report['classes']


def test_paired_view_bootstrap_preserves_class_population_and_direction():
    counts = np.zeros((4, 6), np.int64); counts[:, 2] = [1, 10, 100, 0]
    reference = counts*.12; candidate = reference*.25
    value = paired_class_errors(counts, reference, candidate, repeats=1000)['classes']['stay_cable']
    assert value['difference_mse'] == pytest.approx(-.03)
    assert value['difference_psnr'] == pytest.approx(10*np.log10(4))
    assert value['mse_95_interval'] == pytest.approx([-.03, -.03])
    assert 0 < value['finite_replicates'] < 1000  # All-empty resamples explicitly omitted.


def test_unknown_labels_and_nonzero_error_on_absent_class_rejected():
    image = np.zeros((2, 3, 3), np.uint8)
    with pytest.raises(ValueError):
        class_error_sums(image, image, np.full((2, 3), 6, np.uint8))
    with pytest.raises(ValueError):
        class_summary(np.zeros((2, 6), np.int64), np.ones((2, 6)))
