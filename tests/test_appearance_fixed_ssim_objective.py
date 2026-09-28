"""CPU revision contract: one approved source change, fixed original probe budget."""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load("audit_appearance_fixed_ssim_objective")
legacy = load("audit_appearance_actual_objective")


def test_numerical_schedule_and_optimizer_functions_are_unchanged():
    def functions(path):
        return {node.name: ast.dump(node) for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef)}
    old = functions(Path(legacy.__file__))
    new = functions(Path(audit.__file__))
    for name in ("normalized_direction", "numerical_record", "parameter_differences",
                 "transient_adam_step", "restore_state", "fixed_first_view"):
        assert old[name] == new[name]
    assert audit.ARMS == legacy.ARMS and audit.PARAMETERS == legacy.PARAMETERS
    assert audit.EPSILONS == legacy.EPSILONS and audit.RATES == legacy.RATES
    changed = {key for key in audit.SPEC.keys() | legacy.SPEC.keys()
               if audit.SPEC.get(key) != legacy.SPEC.get(key)}
    assert changed == {"protocol", "revision", "baseline_max_old_fp32_scalar_ulps", "objective"}
    assert audit.SPEC["renders"] == 40


@pytest.mark.parametrize("old", [.03350527957081795, -.03350527957081795, 0.])
def test_baseline_uses_old_fp32_spacing_and_fixed_sixteen_ulp_gate(old):
    old = np.float32(old)
    same = audit.baseline_forward_comparison(old, old)
    assert same["forward_exact"] and same["within_predeclared_limit"]
    spacing = float(abs(np.spacing(old)))
    for count in (1, 16, 17):
        new = np.float32(float(old) + count * spacing)
        result = audit.baseline_forward_comparison(old, new)
        assert not result["forward_exact"]
        assert result["absolute_difference_in_old_ulps"] == count
        assert result["within_predeclared_limit"] == (count <= 16)
        assert "gradient correctness" in result["interpretation"]


@pytest.mark.parametrize("old,new", [(float("nan"), .1), (.1, float("inf")), (.1, .2)])
def test_baseline_rejects_nonfinite_or_not_fp32_scalars(old, new):
    with pytest.raises(ValueError):
        audit.baseline_forward_comparison(old, new)


def test_source_delta_rejects_any_other_change_or_unapproved_loss():
    old = {"bridge_rgs/losses.py": "old", "bridge_rgs/model.py": "same", "old_runner.py": "same"}
    new = {**old, "bridge_rgs/losses.py": audit.FIXED_LOSS_SHA, "new_runner.py": "runner"}
    delta = audit.check_source_delta(old, new, "new_runner.py")
    assert set(delta) == {"bridge_rgs/losses.py"}
    with pytest.raises(ValueError, match="Only approved"):
        audit.check_source_delta(old, {**new, "bridge_rgs/model.py": "changed"}, "new_runner.py")
    with pytest.raises(ValueError, match="Unreviewed"):
        audit.check_source_delta(old, {**new, "bridge_rgs/losses.py": "other"}, "new_runner.py")
    with pytest.raises(ValueError, match="file set"):
        audit.check_source_delta(old, {**new, "extra.py": "unknown"}, "new_runner.py")
    with pytest.raises(ValueError, match="file set"):
        audit.check_source_delta(old, {k: v for k, v in new.items() if k != "old_runner.py"}, "new_runner.py")
