"""Mixed precision boundary and unchanged-formula contracts, CPU only."""
import pytest
import torch

from bridge_rgs.fullbatch_appearance import PassExpired
from bridge_rgs.fullbatch_appearance_fp64 import appearance_rgb_loss_fp64, stream_mean_loss_fp64
from bridge_rgs.raw_grid import appearance_rgb_loss


def test_fp64_objective_exact_same_formula_with_fp32_gradient_return_and_crop():
    torch.manual_seed(4)
    storage = torch.rand(12, 16, 3, dtype=torch.float32, requires_grad=True)
    prediction = storage[:, 1:15]
    target = torch.rand_like(prediction, requires_grad=True)
    valid = torch.ones(prediction.shape[:2], dtype=torch.bool)
    loss, parts = appearance_rgb_loss_fp64(prediction, target, valid)
    expected, _ = appearance_rgb_loss(prediction.double(), target.double(), valid)
    assert loss.dtype == torch.float64 and torch.equal(loss, expected)
    actual_grad, = torch.autograd.grad(loss, storage, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, storage)
    assert actual_grad.dtype == torch.float32 and torch.equal(actual_grad, expected_grad)
    assert parts['l1'].dtype == parts['one_minus_ssim7'].dtype == torch.float64
    assert target.grad is None and bool(actual_grad[:, 1:15].abs().sum())
    assert not bool(actual_grad[:, [0, 15]].any())


def test_stream_matches_equal_mean_parameter_gradient_and_loss():
    parameter = torch.tensor([.2, .3], dtype=torch.float32, requires_grad=True)
    items = [1., 2., 4.]
    def objective(item):
        return (parameter.double()-item).square().sum()
    expected = torch.stack([objective(item) for item in items]).mean()
    expected_grad, = torch.autograd.grad(expected, parameter)
    value = stream_mean_loss_fp64(items, objective, backward=True)
    assert value == float(expected.detach())
    assert parameter.grad.dtype == torch.float32
    torch.testing.assert_close(parameter.grad, expected_grad, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('invalid', [torch.tensor(1., dtype=torch.float32), torch.tensor(float('nan'), dtype=torch.float64),
                                    torch.ones(2, dtype=torch.float64)])
def test_stream_rejects_wrong_loss_dtype_shape_or_finiteness(invalid):
    with pytest.raises(ValueError, match='FP64'):
        stream_mean_loss_fp64([1], lambda _: invalid, backward=False)


def test_stream_deadline_propagates_without_finishing_partial_pass():
    calls, checks = [], []
    def check():
        checks.append(1)
        if len(checks) == 2:
            raise PassExpired('fixed deadline')
    def objective(item):
        calls.append(item)
        return torch.tensor(float(item), dtype=torch.float64)
    with pytest.raises(PassExpired):
        stream_mean_loss_fp64([1, 2], objective, backward=False, check_time=check)
    assert calls == [1]


def test_objective_rejects_non_fp32_endpoints():
    value = torch.ones(8, 8, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match='FP32'):
        appearance_rgb_loss_fp64(value, value, torch.ones(8, 8, dtype=torch.bool))
