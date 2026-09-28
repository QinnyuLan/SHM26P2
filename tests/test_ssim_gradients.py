"""Regression for SSIM's reproduced torch2.8 CUDA channels-last backward mismatch."""
import pytest
import torch
from torch.nn import functional as F

from bridge_rgs.losses import ssim_map


def images(cropped, dtype=torch.float64, device='cpu'):
    generator = torch.Generator().manual_seed(20260927)
    shape = (43, 118, 3) if cropped else (37, 53, 3)
    x = .2+.6*torch.rand(shape, generator=generator, dtype=dtype)
    y = (x+.025*torch.randn(shape, generator=generator, dtype=dtype)).clamp(.01, .99)
    x, y = x.to(device), y.to(device)
    if cropped:
        x, y = x[3:40, 6:112:2], y[3:40, 6:112:2]
        assert not x.is_contiguous() and not y.is_contiguous()
    return x.detach().requires_grad_(True), y.detach().requires_grad_(True)


def convolution_reference(x, y):
    """Independent box-window implementation: grouped convolution, not avg_pool."""
    x, y = x.permute(2, 0, 1)[None].contiguous(), y.permute(2, 0, 1)[None].contiguous()
    kernel = x.new_full((3, 1, 7, 7), 1/49)
    pool = lambda z: F.conv2d(z, kernel, padding=3, groups=3)
    mx, my = pool(x), pool(y)
    vx, vy, covariance = pool(x*x)-mx*mx, pool(y*y)-my*my, pool(x*y)-mx*my
    value = (2*mx*my+.01**2)*(2*covariance+.03**2)
    value = value/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return value[0].mean(0)


def old_layout_forward(x, y):
    """Former forward values only; never assert its affected backward is correct."""
    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    mx, my = F.avg_pool2d(x, 7, 1, 3), F.avg_pool2d(y, 7, 1, 3)
    vx = F.avg_pool2d(x*x, 7, 1, 3)-mx.square()
    vy = F.avg_pool2d(y*y, 7, 1, 3)-my.square()
    covariance = F.avg_pool2d(x*y, 7, 1, 3)-mx*my
    return (((2*mx*my+.01**2)*(2*covariance+.03**2)) /
            ((mx.square()+my.square()+.01**2)*(vx+vy+.03**2)))[0].mean(0)


def objective(image_map):
    return (1-image_map[3:-3, 3:-3]).mean()


@pytest.mark.parametrize('cropped', [False, True])
def test_cpu_ssim_map_and_gradients_match_independent_convolution(cropped):
    x, y = images(cropped)
    actual = ssim_map(x, y)
    reference = convolution_reference(x, y)
    torch.testing.assert_close(actual, reference, atol=2e-13, rtol=2e-13)
    gradients = torch.autograd.grad(objective(actual), (x, y))
    expected = torch.autograd.grad(objective(reference), (x, y))
    for gradient, ref in zip(gradients, expected, strict=True):
        torch.testing.assert_close(gradient, ref, atol=1e-14, rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA gradient regression')
@pytest.mark.parametrize('cropped', [False, True])
def test_cuda_ssim_fp64_full_gradient_cpu_reference_and_independent_direction_fd(cropped):
    cpu_x, cpu_y = images(cropped)
    reference = convolution_reference(cpu_x, cpu_y)
    expected = torch.autograd.grad(objective(reference), (cpu_x, cpu_y))
    x, y = images(cropped, device='cuda')
    actual = ssim_map(x, y)
    gradients = torch.autograd.grad(objective(actual), (x, y))
    torch.testing.assert_close(actual.detach().cpu(), reference.detach(), atol=2e-13, rtol=2e-13)
    for gradient, ref in zip(gradients, expected, strict=True):
        torch.testing.assert_close(gradient.cpu(), ref, atol=1e-14, rtol=1e-10)
    # The direction comes from an independent fixed RNG, never from either gradient.
    generator = torch.Generator().manual_seed(51927)
    direction = torch.randn(cpu_x.shape, generator=generator, dtype=torch.float64)
    direction = (direction/direction.abs().max()).cuda()
    epsilon = 1e-5
    with torch.no_grad():
        plus = objective(ssim_map(x+epsilon*direction, y))
        minus = objective(ssim_map(x-epsilon*direction, y))
    finite_difference = (plus-minus)/(2*epsilon)
    analytic = (gradients[0]*direction).sum()
    torch.testing.assert_close(analytic, finite_difference, atol=1e-10, rtol=1e-6)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA forward compatibility'))])
@pytest.mark.parametrize('cropped', [False, True])
def test_fp32_forward_matches_former_layout_for_contiguous_and_crop(device, cropped):
    x, y = images(cropped, dtype=torch.float32, device=device)
    with torch.no_grad():
        actual, previous = ssim_map(x, y), old_layout_forward(x, y)
    torch.testing.assert_close(actual, previous, atol=1e-6, rtol=1e-6)
