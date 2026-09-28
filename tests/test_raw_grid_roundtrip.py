"""CPU checks for support isolation and metric/interpolation definitions."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

spec = importlib.util.spec_from_file_location("roundtrip_diagnostic", Path(__file__).parents[1] / "scripts/diagnose_raw_grid_roundtrip.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_support_rejects_invalid_taps_and_extrapolation():
    valid = np.ones((4, 4), bool)
    valid[1, 1] = False
    x = np.array([[0., .5, 3., 3.1, np.nan]], np.float32)
    y = np.array([[0., .5, 3., 2., 1.]], np.float32)
    assert module.complete_bilinear_support(valid, x, y).tolist() == [[True, False, True, False, False]]


def test_complete_window_excludes_hole_and_outer_border():
    valid = np.ones((25, 25), bool)
    valid[12, 12] = False
    result = module.complete_windows(valid, 11)
    assert not result[:5].any() and not result[:, :5].any()
    assert not result[7:18, 7:18].any()
    assert result[5, 5]


def test_identity_and_constant_roundtrip():
    values = np.full((21, 21, 3), .4, np.float32)
    x, y = np.meshgrid(np.arange(21, dtype=np.float32), np.arange(21, dtype=np.float32))
    valid = module.complete_bilinear_support(np.ones((21, 21), bool), x, y)
    result = cv2.remap(values, x, y, cv2.INTER_LINEAR)
    score = module.score(values, result, valid, module.complete_windows(valid, 11))
    assert score['common_complete_11x11']['mse'] == 0
    assert score['common_complete_11x11']['ssim_7box'] == 1
    assert score['common_complete_11x11']['ssim_11gaussian_sigma1.5'] == 1


def test_gaussian_metric_matches_skimage_on_complete_support():
    rng = np.random.default_rng(12)
    x = rng.uniform(0, 1, (25, 31, 3))
    y = np.clip(x + rng.normal(0, .04, x.shape), 0, 1)
    _, expected = structural_similarity(x, y, data_range=1., channel_axis=-1, win_size=11,
                                        gaussian_weights=True, sigma=1.5, use_sample_covariance=False, full=True)
    actual = module.ssim_map(x, y, True)
    np.testing.assert_allclose(actual[5:-5, 5:-5], expected[5:-5, 5:-5].mean(-1), atol=1e-12)


def test_fractional_checkerboard_roundtrip_loses_edges_without_filling():
    y, x = np.indices((31, 31), dtype=np.float32)
    image = np.repeat(((x + y) % 2)[..., None], 3, axis=2)
    warped = cv2.remap(image, x + .5, y, cv2.INTER_LINEAR)
    result = cv2.remap(warped, x - .5, y, cv2.INTER_LINEAR)
    valid = module.complete_bilinear_support(np.ones((31, 31), bool), x - .5, y)
    valid[:, -2:] = False
    keep = module.complete_windows(valid, 11)
    values = module.score(image, result, valid, keep)['common_complete_11x11']
    assert values['mse'] > .2
    assert values['ssim_7box'] < .01
    # Altering unsupported outer pixels cannot change these complete-window scores.
    changed = result.copy()
    changed[~valid] = 100
    other = module.score(image, changed, valid, keep)['common_complete_11x11']
    for key in ('mse', 'ssim_7box', 'ssim_11gaussian_sigma1.5'):
        assert abs(other[key] - values[key]) < 1e-12
