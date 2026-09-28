import pytest
import torch

from bridge_rgs.ibgs_losses import (
    erode_valid,
    gaussian_ssim_map,
    masked_normal_loss,
    masked_photometric_loss,
)


def test_invalid_padding_cannot_change_loss_or_gradient():
    torch.manual_seed(24)
    target = torch.rand(3, 32, 40, dtype=torch.float64)
    prediction = torch.rand_like(target).requires_grad_()
    valid = torch.ones(32, 40, dtype=torch.bool)
    valid[:3] = False
    valid[12:15, 16:19] = False
    original = masked_photometric_loss(prediction, target, valid)[0]
    gradient = torch.autograd.grad(original, prediction)[0]
    changed = prediction.detach().clone()
    changed[:, ~valid] = 100.
    changed.requires_grad_()
    target2 = target.clone()
    target2[:, ~valid] = -100.
    second = masked_photometric_loss(changed, target2, valid)[0]
    gradient2 = torch.autograd.grad(second, changed)[0]
    torch.testing.assert_close(original, second, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(gradient, gradient2, atol=1e-12, rtol=1e-12)
    assert not gradient[:, ~valid].any()


def test_identity_ssim_and_loss():
    image = torch.linspace(0, 1, 3*24*24).reshape(3, 24, 24)
    valid = torch.ones(24, 24, dtype=torch.bool)
    torch.testing.assert_close(gaussian_ssim_map(image, image), torch.ones_like(valid).float())
    assert masked_photometric_loss(image, image, valid)[0].item() == 0


def test_eroded_source_support_implies_all_bilinear_rgb_taps_valid():
    valid = torch.ones(12, 16, dtype=torch.bool)
    valid[:2] = False
    valid[6, 7] = False
    safe_depth = erode_valid(valid, 1)
    # If any bilinear depth tap survives a 3x3 erosion, all 2x2 RGB taps
    # around that same continuous coordinate are valid.
    for y in range(11):
        for x in range(15):
            if safe_depth[y:y+2, x:x+2].any():
                assert valid[y:y+2, x:x+2].all()


def test_normal_loss_empty_support_is_differentiable_zero():
    normal = torch.randn(3, 12, 12, requires_grad=True)
    value = masked_normal_loss(normal, normal.detach(), torch.zeros(12, 12),
                               torch.ones(12, 12, dtype=torch.bool))
    value.backward()
    assert value.item() == 0 and not normal.grad.any()


def test_empty_valid_ssim_rejected():
    image = torch.zeros(3, 8, 8)
    with pytest.raises(ValueError, match='supervision'):
        masked_photometric_loss(image, image, torch.ones(8, 8, dtype=torch.bool))


def test_photometric_gradient_finite_difference():
    torch.manual_seed(1)
    image = torch.rand(3, 16, 17, dtype=torch.float64, requires_grad=True)
    target = torch.rand_like(image)
    valid = torch.ones(16, 17, dtype=torch.bool)
    direction = torch.randn_like(image)*.01
    value = masked_photometric_loss(image, target, valid)[0]
    analytic = (torch.autograd.grad(value, image)[0]*direction).sum()
    h = 1e-5
    numeric = (masked_photometric_loss(image+h*direction, target, valid)[0]
               - masked_photometric_loss(image-h*direction, target, valid)[0])/(2*h)
    torch.testing.assert_close(analytic, numeric, atol=1e-9, rtol=1e-5)
