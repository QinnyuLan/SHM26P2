import importlib.util
import time
from pathlib import Path

import pytest
import torch

SPEC_PATH = Path(__file__).resolve().parents[1] / "scripts/audit_appearance_cuda_gradients.py"
spec = importlib.util.spec_from_file_location("appearance_cuda_gradient_audit", SPEC_PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_probe_is_fixed_spatial_rgb_linear_function_with_double_accumulation():
    weights = audit.probe_weights(11, 13)
    assert weights.shape == (11, 13, 3) and weights.dtype == torch.double
    assert weights.min() > 0
    torch.testing.assert_close(weights[..., 2] / weights[..., 1], torch.full((11, 13), 1.3, dtype=torch.double))
    values = torch.full((11, 13, 3), .4, requires_grad=True)
    scalar = audit.probe(values, weights)
    scalar.backward()
    assert scalar.dtype == torch.double
    torch.testing.assert_close(values.grad.double(), weights / values.numel(), atol=1e-9, rtol=1e-6)
    torch.testing.assert_close(audit.probe(values * 2, weights), scalar * 2)


def test_direction_is_absmax_normalized_not_search_selected():
    gradient = torch.tensor([1., -3., 2.])
    direction, derivative, maximum = audit.normalized_direction(gradient)
    torch.testing.assert_close(direction, gradient / 3)
    assert float(direction.abs().max()) == 1 and maximum == 3
    assert derivative == pytest.approx(14 / 3)


@pytest.mark.parametrize("value", [torch.zeros(3), torch.tensor([float("nan")]), torch.tensor([float("inf")])])
def test_invalid_gradient_direction_rejected(value):
    with pytest.raises(ValueError):
        audit.normalized_direction(value)


def test_all_predefined_epsilons_and_exact_restoration_for_smooth_function():
    parameter = torch.nn.Parameter(torch.tensor([.3, -.7, 1.2], dtype=torch.double))
    original = parameter.detach().clone()
    function = lambda: ((parameter.square().sum()), {})
    gradient, = torch.autograd.grad(function()[0], parameter)
    rows = audit.parameter_differences(parameter, gradient, function, time.monotonic() + 10)
    assert [r["epsilon"] for r in rows["epsilons"]] == [.01, .005, .0025]
    assert all(r["within_fixed_tolerance"] for r in rows["epsilons"])
    assert all(r["relative_error"] < 1e-12 for r in rows["epsilons"])
    assert torch.equal(parameter, original)


def test_restore_even_when_objective_or_deadline_fails():
    parameter = torch.nn.Parameter(torch.tensor([.5, 1.]))
    original = parameter.detach().clone()
    def fail():
        assert not torch.equal(parameter, original)
        raise RuntimeError("render failure")
    with pytest.raises(RuntimeError, match="render failure"):
        audit.parameter_differences(parameter, torch.ones(2), fail, time.monotonic() + 10)
    assert torch.equal(parameter, original)
    with pytest.raises(TimeoutError):
        audit.parameter_differences(parameter, torch.ones(2), fail, time.monotonic() - 1)
    assert torch.equal(parameter, original)


def test_negative_derivative_mismatch_is_not_hidden_by_relative_error():
    row = audit.scalar_comparison(-.01, .01, .01, 1.)
    assert row["central_difference"] == -1 and row["relative_error"] == 2
    assert row["within_fixed_tolerance"] is False


def test_fixed_budget_has_no_training_or_gt_target():
    assert audit.SPEC["loss_target"] is None
    assert audit.SPEC["train_steps"] == 0
    assert audit.SPEC["render_calls"] == 3 + 3 * 3 * 3 * 2
    assert audit.SPEC["external_process_limit_seconds"] == 60
    assert audit.SPEC["output_bytes_limit"] == 1 << 20
