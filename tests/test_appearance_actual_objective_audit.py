"""CPU contracts for fixed directional probes, one transient step and restoration."""
import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/audit_appearance_actual_objective.py"
spec = importlib.util.spec_from_file_location("appearance_actual_objective_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_protocol_has_only_two_original_goals_eighteen_differences_and_forty_renders():
    assert audit.ARMS == ("00_native", "01_original")
    assert audit.EPSILONS == (1e-3, 5e-4, 2.5e-4)
    assert audit.SPEC["finite_difference_comparisons"] == 2*3*3 == 18
    assert audit.SPEC["renders"] == 2*(1+3*3*2+1) == 40
    assert audit.SPEC["temporary_optimizer_steps_per_arm"] == 1
    assert audit.SPEC["retained_candidates"] == 0


def test_direction_normalization_and_analytic_double_accumulation():
    gradient = torch.tensor([1., -3., 2.])
    direction, derivative, maximum = audit.normalized_direction(gradient)
    assert torch.equal(direction, gradient/3)
    assert derivative == pytest.approx(14/3) and maximum == 3


@pytest.mark.parametrize("gradient", [torch.zeros(3), torch.tensor([float("nan")]), torch.tensor([float("inf")])])
def test_zero_or_nonfinite_gradient_cannot_be_silently_replaced(gradient):
    with pytest.raises(ValueError):
        audit.normalized_direction(gradient)


def test_original_fp32_objective_scalars_and_no_automatic_numerical_verdict():
    value = float(np.float32(.03))
    result = audit.numerical_record(value, value, value, .001, 1.)
    assert result["central_difference"] == 0
    assert result["difference_within_16_scalar_ulps"]
    assert result["relative_discrepancy"] == 1
    assert "pass" not in result and "within_fixed_tolerance" not in result
    assert "not a pass/fail" in result["interpretation"]


def test_all_epsilons_use_original_base_and_restore_exactly():
    parameter = torch.nn.Parameter(torch.tensor([.3, -.7, 1.2]))
    original = parameter.detach().clone()
    gradient = 2*original
    calls = []
    def objective():
        calls.append(parameter.detach().clone())
        return parameter.square().sum(), {"dtype": str(parameter.dtype)}
    result = audit.parameter_differences(parameter, gradient, original.square().sum(), objective, time.monotonic()+10)
    direction = gradient/gradient.abs().max()
    assert len(calls) == 6 and torch.equal(parameter, original)
    assert [r["epsilon"] for r in result["epsilons"]] == list(audit.EPSILONS)
    for i, epsilon in enumerate(audit.EPSILONS):
        assert torch.equal(calls[2*i], original+epsilon*direction)
        assert torch.equal(calls[2*i+1], original-epsilon*direction)
    assert all(row["relative_discrepancy"] < .001 for row in result["epsilons"])


@pytest.mark.parametrize("failure", ["objective", "deadline"])
def test_directional_probe_restores_on_failure(failure):
    parameter = torch.nn.Parameter(torch.tensor([.3, -.7]))
    original = parameter.detach().clone()
    def objective():
        raise RuntimeError("intentional forward failure")
    deadline = time.monotonic()-1 if failure == "deadline" else time.monotonic()+10
    with pytest.raises((RuntimeError, TimeoutError)):
        audit.parameter_differences(parameter, original, .5, objective, deadline)
    assert torch.equal(parameter, original)


def parameter_groups():
    return {name: torch.nn.Parameter(torch.tensor([.3, -.5], dtype=torch.float32)) for name in audit.PARAMETERS}


def test_one_original_fresh_adam_step_reports_actual_displacement_and_restores(monkeypatch):
    parameters = parameter_groups()
    original = {name: p.detach().clone() for name, p in parameters.items()}
    gradients = {name: 2*p.detach() for name, p in parameters.items()}
    calls, steps = [], []
    original_step = torch.optim.Adam.step
    def step(optimizer, *args, **kwargs):
        assert not optimizer.state
        steps.append(optimizer)
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    def objective():
        calls.append(1)
        return sum(p.square().sum() for p in parameters.values()), {}
    base = float(sum(p.square().sum() for p in original.values()))
    result = audit.transient_adam_step(parameters, gradients, objective, base, time.monotonic()+10)
    assert len(steps) == len(calls) == result["optimizer_steps"] == 1
    assert result["actual_loss_change"] < 0 and result["gradient_dot_actual_displacement"] < 0
    for name, group in result["groups"].items():
        assert group["lr"] == audit.RATES[name] and group["eps"] == 1e-15
        assert group["betas"] == [.9, .999] and group["weight_decay"] == 0
        assert group["step"] == 1 and group["displacement_absmax"] > 0
        assert torch.equal(parameters[name], original[name]) and parameters[name].grad is None


def test_post_adam_failure_restores_all_parameter_groups():
    parameters = parameter_groups()
    original = {name: p.detach().clone() for name, p in parameters.items()}
    def objective():
        raise RuntimeError("after optimizer")
    with pytest.raises(RuntimeError, match="after optimizer"):
        audit.transient_adam_step(parameters, {k: torch.ones_like(p) for k, p in parameters.items()},
                                 objective, 1., time.monotonic()+10)
    assert all(torch.equal(p, original[k]) and p.grad is None for k, p in parameters.items())


def test_full_state_restoration_includes_buffers_and_clears_grads():
    model = torch.nn.Linear(2, 2)
    model.register_buffer("count", torch.tensor(3))
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(2)
        model.count.add_(1)
    model.weight.grad = torch.ones_like(model.weight)
    audit.restore_state(model, original)
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
    assert model.weight.grad is None


def test_fixed_camera_must_match_original_first_training_index():
    views = [{"name": f"{i:03}.png", "split": "train", "w2c": np.eye(4).tolist(),
              "w2c_original": np.eye(4).tolist()} for i in range(350)]
    manifest = {"views": list(reversed(views))}
    original = {"view_names": [v["name"] for v in views], "training_order": [152]}
    assert audit.fixed_first_view(manifest, original)["name"] == "152.png"
    original["training_order"] = [151]
    with pytest.raises(ValueError, match="TRAIN152"):
        audit.fixed_first_view(manifest, original)
