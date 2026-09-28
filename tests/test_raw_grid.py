"""CPU-only fixed raw-warp and appearance loss contracts."""
import cv2
import numpy as np
import pytest
import torch

from bridge_rgs.raw_grid import (
    FixedBilinearWarp,
    appearance_rgb_loss,
    build_raw_grid,
    opencv_float_map_policy,
)


def test_float_map_policy_probe_and_explicit_table_are_distinct():
    image = (np.arange(7*9*3, dtype=np.float32).reshape(7, 9, 3) * .173 % 1).copy()
    mx = np.array([[.371, 2.813, 5.194]], np.float32)
    my = np.array([[.537, 1.116, 3.291]], np.float32)
    base, fractions = cv2.convertMaps(mx, my, cv2.CV_16SC2, nninterpolation=False)
    floating = cv2.remap(image, mx, my, cv2.INTER_LINEAR)
    table = cv2.remap(image, base, fractions, cv2.INTER_LINEAR)
    warp = FixedBilinearWarp(mx, my, 7, 9)
    np.testing.assert_allclose(warp(torch.tensor(image)), floating, atol=2e-7, rtol=0)
    assert warp.receipt()['weight_policy'] == opencv_float_map_policy()
    if opencv_float_map_policy() == 'continuous_float32':
        assert np.max(np.abs(floating-table)) > 1e-4
    else:
        np.testing.assert_array_equal(floating, table)


def test_fixed_table_matches_cv2_and_has_image_gradients():
    rng = np.random.default_rng(14)
    image = rng.uniform(-.2, 1.2, (17, 19, 3)).astype(np.float32)
    mx = rng.uniform(0, 18, (12, 13)).astype(np.float32)
    my = rng.uniform(0, 16, (12, 13)).astype(np.float32)
    warp = FixedBilinearWarp(mx, my, 17, 19)
    expected = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    values = torch.tensor(image, requires_grad=True)
    actual = warp(values)
    np.testing.assert_allclose(actual.detach(), expected, atol=2e-7, rtol=0)
    actual.sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert abs(float(values.grad.sum()) - actual.numel()) < 1e-4
    assert not list(warp.parameters()) and all(not b.requires_grad for b in warp.buffers())


def test_fixed_table_double_gradcheck():
    mx = np.array([[.37, 1.19], [.81, 1.51]], np.float32)
    my = np.array([[.51, .32], [1.67, 1.01]], np.float32)
    warp = FixedBilinearWarp(mx, my, 3, 3)
    values = torch.rand(3, 3, 2, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(warp, (values,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_unsupported_border_rejected_or_zero_weight_no_fill():
    mx = np.array([[-.5, 0., 2.]], np.float32)
    my = np.zeros_like(mx)
    with pytest.raises(ValueError, match="unsupported"):
        FixedBilinearWarp(mx, my, 1, 3)
    warp = FixedBilinearWarp(mx, my, 1, 3, require_full_support=False)
    assert warp.full_support.tolist() == [[False, True, True]]
    assert warp(torch.ones(1, 3, 1)).flatten().tolist() == [.5, 1., 1.]


def test_identity_native_crop_and_negative_distortion_overscan():
    K = np.array([[35., 0, 20.], [0, 35., 15.], [0, 0, 1]], np.float32)
    identity = build_raw_grid(K, [0., 0, 0, 0, 0], 40, 30)
    image = torch.rand(30, 40, 3)
    assert torch.equal(identity.warp(image), image)
    assert torch.equal(identity.crop_native(image), image)
    warped = build_raw_grid(K, [-.1, 0, 0, 0, 0], 40, 30)
    assert warped.render_width > 40 and warped.render_height > 30
    assert warped.warp.full_support.all()
    canvas = torch.arange(warped.render_width * warped.render_height).reshape(warped.render_height, warped.render_width, 1).float()
    crop = warped.crop_native(canvas)
    assert crop.shape == (30, 40, 1)
    assert crop[0, 0] == canvas[warped.native_top, warped.native_left]


def test_loss_keeps_image_edges_for_l1_but_requires_full_ssim_windows():
    target = torch.zeros(15, 17, 3, requires_grad=True)
    prediction = torch.zeros_like(target, requires_grad=True)
    with torch.no_grad():
        prediction[0, 0] = 1
    loss, stats = appearance_rgb_loss(prediction, target, torch.ones(15, 17, dtype=torch.bool))
    assert int(stats['rgb_pixels']) == 15*17 and int(stats['ssim7_centers']) == 9*11
    assert stats['l1'] > 0  # Full RGB border is not replaced with the target.
    loss.backward()
    assert prediction.grad is not None and target.grad is None


def test_invalid_values_never_enter_loss_or_ssim_windows():
    values = torch.rand(21, 21, 3)
    valid = torch.ones(21, 21, dtype=torch.bool)
    valid[:2] = False
    changed = values.clone()
    changed[~valid] = 2
    loss, _ = appearance_rgb_loss(changed, values, valid)
    assert abs(float(loss)) < 1e-7
    with pytest.raises(ValueError, match="No complete"):
        appearance_rgb_loss(values, values, torch.zeros_like(valid))
