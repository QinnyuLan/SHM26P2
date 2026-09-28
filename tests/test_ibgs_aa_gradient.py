import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
WORKER = HERE/'check_ibgs_aa_gradient.py'
if not WORKER.exists():
    WORKER = HERE.parent/'scripts/check_ibgs_aa_gradient.py'
spec = importlib.util.spec_from_file_location('ibgs_aa_gradient_test', WORKER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_actual_fp32_step_and_strict_json():
    x = np.array([.11, .27], np.float32); direction = np.array([.3, -.4], np.float32)
    h = .002; plus, minus = x+h*direction, x-h*direction
    g = np.array([.7, -.2], np.float32)
    row = m.fd_report(g, plus, minus, float(plus.astype(float)@g), float(minus.astype(float)@g), h)
    assert row['passed'] and row['absolute_error'] < 1e-12
    assert row['actual_direction'] != direction.astype(float).tolist()
    json.dumps(row, allow_nan=False)


def test_zero_signal_inconclusive_and_cap_active_mismatch():
    zero = m.fd_report([0.], [.1], [-.1], 0., 0., .002)
    assert not zero['measurable'] and not zero['passed']
    cap = m.fd_report([.01], [6.002], [5.998], .99, .99, .002)
    assert cap['measurable'] and not cap['passed'] and cap['finite_difference'] == 0
    with pytest.raises(ValueError):
        m.fd_report([np.nan], [.1], [-.1], 0., 0., .002)


def test_fixed_counts_no_sensitivity_selection_and_safe_write(tmp_path):
    assert m.SPEC['forward_calls'] == (1+4*2*3)+(1+2*3)+1 == 33
    assert m.SPEC['primary_h'] == .002 and m.SPEC['sensitivity_h'] == [.001, .004]
    with pytest.raises(ValueError):
        m.write(tmp_path/'invalid.json', {'x': float('nan')})
    assert not (tmp_path/'invalid.json').exists()
