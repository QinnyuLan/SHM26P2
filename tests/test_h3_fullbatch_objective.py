"""CPU contracts for the fixed two-view objective probe; no real data/GPU."""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from bridge_rgs import fullbatch_appearance as solver

spec = importlib.util.spec_from_file_location('h3_objective_probe', Path(__file__).parents[1]/'scripts/audit_h3_fullbatch_objective.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def manifest():
    views = []
    for i in range(1, 401):
        views.append({'name': f'{i:03}.png', 'split': 'val' if (i-1) % 8 == 0 else 'train',
                      'camera_id': 1, 'width': 8, 'height': 8, 'K': np.eye(3).tolist(),
                      'w2c': np.eye(4).tolist(), 'w2c_original': np.eye(4).tolist(),
                      'source_image_path': f'raw/{i:03}.png', 'mask_path': 'must-not-read',
                      'valid_path': 'must-not-read', 'image_path': 'must-not-read'})
    # Actual held-out camera split is not every eighth name; reserve target names.
    # Build any 350-name TRAIN set whose index175 is205 (29 VAL precede205).
    excluded = set(range(1, 201, 7))
    excluded |= set(range(301, 322))
    assert len(excluded) == 50
    for v in views:
        v['split'] = 'val' if int(v['name'][:3]) in excluded else 'train'
    return {'views': list(reversed(views))}


def test_views_sort_names_but_keep_checkpoint_camera_index_and_whitelist():
    data = manifest()
    chosen, names = probe.fixed_views(data)
    assert [v['name'] for v in chosen] == ['002.png', '205.png']
    assert all(names[v['camera_index']] == v['name'] for v in chosen)
    assert all(not any(k in v for k in ('mask_path', 'valid_path', 'image_path')) for v in chosen)
    data['pixel_protocol'] = 'colmap_corner_v2'
    with pytest.raises(ValueError, match='legacy'):
        probe.fixed_views(data)


def parameters():
    return {k: torch.nn.Parameter(torch.linspace(.1, .5, 32).reshape(8, 4)) for k in probe.KEYS}


def test_realized_displacement_not_nominal_controls_fd_gate():
    theta = torch.tensor([1.e6], dtype=torch.float32)
    g, direction = torch.ones(1), torch.ones(1)
    epsilon = .1
    plus, minus = theta+epsilon, theta-epsilon
    row = probe.finite_difference_record(float(theta), float(plus), float(minus), epsilon,
                                         g, theta, plus, minus, direction)
    assert row['nominal_analytic'] == 1.
    assert row['realized_analytic'] == 1.25
    assert row['realized_relative_discrepancy'] == 0.
    assert row['passed']
    assert row['realized_direction_relative_l2'] == .25
    json.dumps(row, allow_nan=False)


def test_scalar_record_uses_float_not_numpy_array_protocol():
    class FloatOnlyScalar:
        def __init__(self, value):
            self.value = value

        def __float__(self):
            return self.value

        def __array__(self, *args, **kwargs):
            raise AssertionError('CUDA scalar must not enter NumPy array protocol')

    theta, gradient, direction = torch.zeros(1), torch.ones(1), torch.ones(1)
    epsilon = .001
    positive, negative = theta+epsilon, theta-epsilon
    row = probe.finite_difference_record(FloatOnlyScalar(0.), FloatOnlyScalar(float(positive)),
                                         FloatOnlyScalar(float(negative)), epsilon,
                                         gradient, theta, positive, negative, direction)
    assert row['passed'] and row['realized_relative_discrepancy'] == 0.
    assert row['one_ulp_derivative_resolution'] > 0
    json.dumps(row, allow_nan=False)


def test_toy_exact_20_calls_nine_fd_and_one_uncommitted_rms_step():
    p = parameters()
    before = {k: v.detach().clone() for k, v in p.items()}
    calls = 0
    def objective():
        nonlocal calls
        calls += 1
        prediction = torch.stack(tuple(p.values())).sum(0)
        return prediction.square().mean(), prediction
    base, image = objective()
    gradients = torch.autograd.grad(base, [image, *p.values()])
    image_g = gradients[0].detach()
    gs = dict(zip(probe.KEYS, [g.detach() for g in gradients[1:]], strict=True))
    deadline = time.monotonic()+30
    fd = {k: probe.probe_group(p[k], gs[k], float(base.detach()), objective, deadline, image_g) for k in p}
    step = probe.transient_rms(p, gs, float(base.detach()), objective, deadline)
    assert calls == 20 and step['committed_steps'] == 0 and step['passed']
    assert all(torch.equal(p[k], before[k]) for k in p)
    assert all(r['passed'] for g in fd.values() for r in g['epsilons'])
    row = {'finite_differences': fd, 'temporary_rms_step': step, 'baseline': {'warp_cv2_max_abs_difference': 0.}}
    assert probe.technical_gate([row, row])['passed']
    json.dumps([row, row], allow_nan=False)


def test_fd_restores_on_negative_prediction_failure():
    p = torch.nn.Parameter(torch.tensor([.2, .4]))
    original = p.detach().clone()
    calls = 0
    def bad():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('injected')
        return p.square().sum(), p
    with pytest.raises(RuntimeError, match='injected'):
        probe.probe_group(p, 2*original, float(original.square().sum()), bad, time.monotonic()+30)
    assert calls == 2 and torch.equal(p, original)


def test_rms_always_rolls_back_including_exception_and_rejected_step():
    p = parameters()
    original = {k: v.detach().clone() for k, v in p.items()}
    gs = {k: torch.ones_like(v) for k, v in p.items()}
    def bad():
        raise RuntimeError('injected')
    with pytest.raises(RuntimeError, match='injected'):
        probe.transient_rms(p, gs, 1., bad, time.monotonic()+30)
    assert all(torch.equal(p[k], original[k]) for k in p)
    result = probe.transient_rms(p, gs, 1., lambda: (torch.tensor(2.), None), time.monotonic()+30)
    assert not result['passed'] and not result['helper_armijo_passed']
    assert all(torch.equal(p[k], original[k]) for k in p)


def test_fixed_solver_and_complete_gate_no_subset_or_scalar_ulp_exemption():
    assert solver.RATES == probe.SPEC['solver']['rates']
    assert solver.EPS == probe.SPEC['solver']['eps'] and solver.BETA2 == probe.SPEC['solver']['beta2']
    with pytest.raises(ValueError, match='both'):
        probe.technical_gate([])
    with pytest.raises(TimeoutError):
        probe.check_time(time.monotonic()-1)
    p = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match='nonzero'):
        probe.probe_group(p, torch.zeros(1), 0., lambda: None, time.monotonic()+1)
    assert not torch.cuda.is_initialized()
