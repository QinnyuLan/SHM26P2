"""Small verifier contracts; the frozen 48-case audit is deliberately not run."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import roots_hermitenorm

from bridge_rgs import semantic_partition as core

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_semantic_partition.py'
_spec = importlib.util.spec_from_file_location('semantic_partition_independent_audit', SCRIPT)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def toy():
    return {'name': 'unit_test_only', 'P': np.array([[.7, .2, .1], [.1, .6, -.1]]),
                'noise': np.array([[.5, .05], [.05, .4]]),
                'normal': np.array([.6, 0., .8]), 'b': .3, 'w': .8, 'tau': 1.,
                'delta': np.array([.2, -.4])}


def test_fixed_case_population_is_deterministic_without_data():
    first, second = audit.generate_cases(), audit.generate_cases()
    assert len(first) == len(second) == 48
    assert len({c['name'] for c in first}) == 48
    for left, right in zip(first, second):
        for key in left:
            assert np.array_equal(left[key], right[key])
        assert abs(np.linalg.norm(left['normal']) - 1) < 1e-14
    assert audit.SPEC['gauss_hermite_order'] == 512
    gradient_cases = [first[i] for i in audit.SPEC['gradient_case_indices']]
    assert len({c['name'].split('/')[0] for c in gradient_cases if np.any(c['delta'])}) == 8
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('mode', ['integrated', 'point', 'marginal'])
def test_independent_scalar_quadrature_toy(mode):
    case = toy()
    numerical, error = audit.reference_quad(case, mode)
    analytical = audit.reference_closed(case, mode)
    actual = float(audit.torch_gate(core, case, mode, torch))
    assert error < 1e-10
    assert numerical == pytest.approx(analytical, abs=1e-11)
    assert actual == pytest.approx(numerical, abs=1e-11)


@pytest.mark.parametrize('mode', ['integrated', 'point', 'marginal'])
def test_independent_finite_difference_chart_toy(mode):
    result = audit.check_gradients(core, toy(), mode, torch)
    assert result['passed']
    assert len(result['analytic']) == 16
    json.dumps(result, allow_nan=False)


def test_screen_mass_integrates_normalized_2d_distribution():
    case = toy()
    nodes_weights = roots_hermitenorm(32)
    prior_mass = audit.reference_closed(case, 'marginal')
    for mode in ['integrated', 'marginal']:
        actual = audit.hermite_screen_mass(core, case, mode, torch, nodes_weights)
        assert actual == pytest.approx(prior_mass, abs=1e-12)
    # Point mode is a control, not required to have the same prior mass.
    actual_point = audit.hermite_screen_mass(core, case, 'point', torch, nodes_weights)
    _, conditional_variance = audit.conditional_scalar(case)
    std = np.sqrt(case['tau'] ** 2 + 1 - conditional_variance)
    expected_point = audit.normal_interval((-case['b'] - case['w']) / std,
                                          (-case['b'] + case['w']) / std)
    assert actual_point == pytest.approx(expected_point, abs=1e-12)
    assert abs(actual_point - prior_mass) > .01


@pytest.mark.parametrize('mode', ['integrated', 'point', 'marginal'])
def test_units_and_slab_sign_symmetry(mode):
    case = toy()
    assert audit.check_units(core, case, mode, torch)['passed']
    reversed_case = dict(case, normal=-case['normal'], b=-case['b'])
    assert audit.reference_quad(case, mode)[0] == pytest.approx(
        audit.reference_quad(reversed_case, mode)[0], abs=1e-12)


def test_fixed_visibility_compositor_against_independent_integrals():
    result = audit.check_reference_compositing(core, torch)
    assert result['passed']
    assert result['all_zero_weight_ray_background_exact']
    json.dumps(result, allow_nan=False)


def test_gradient_gate_uses_absolute_near_zero_not_relative_noise():
    actual = np.array([1., 0., 1e-8])
    assert audit.gradient_comparison(actual, np.array([1.00001, 1e-9, 1.1e-8]))['passed']
    assert not audit.gradient_comparison(actual, np.array([1.001, 0., 1e-8]))['passed']
    assert not audit.gradient_comparison(actual, np.array([1., 1e-6, 1e-8]))['passed']


def make_plan(tmp_path):
    source_root = tmp_path / 'source_snapshot'
    module = source_root / 'bridge_rgs/semantic_partition.py'
    module.parent.mkdir(parents=True)
    module.write_text('# synthetic provenance placeholder; never imported\n')
    output = tmp_path / 'output'
    plan = {'protocol': audit.SPEC['protocol'], 'specification': audit.SPEC,
                'output': str(output), 'source_root': str(source_root),
                'source_hashes': {str(SCRIPT): audit.sha256(SCRIPT),
                               str(module): audit.sha256(module)}}
    plan_path = tmp_path / 'plan.json'
    audit.write_json(plan_path, plan)
    return plan_path, output, module


def test_plan_spec_worker_and_core_are_bound_before_execution(tmp_path):
    plan_path, output, module = make_plan(tmp_path)
    digest = audit.sha256(plan_path)
    assert audit.validate_plan(plan_path, digest, output)['specification'] == audit.SPEC
    with pytest.raises(ValueError, match='Plan SHA'):
        audit.validate_plan(plan_path, '0' * 64, output)
    with pytest.raises(ValueError, match='Output'):
        audit.validate_plan(plan_path, digest, output / 'other')
    module.write_text('# changed\n')
    with pytest.raises(ValueError, match='source mismatch'):
        audit.validate_plan(plan_path, digest, output)


def test_nonempty_output_refused_before_numerical_or_module_execution(tmp_path, monkeypatch):
    plan_path, output, _ = make_plan(tmp_path)
    output.mkdir()
    (output / 'retained_failure.json').write_text('{}')
    monkeypatch.setattr(audit, 'run_numerical', lambda *_: pytest.fail('Must not run'))
    with pytest.raises(ValueError, match='non-empty'):
        audit.execute(plan_path, audit.sha256(plan_path), output)
    assert not torch.cuda.is_initialized()


def test_five_point_formula_polynomial_and_input_unchanged():
    vector = np.array([.7, -.3])
    original = vector.copy()
    derivative = audit.finite_difference(lambda x: x[0] ** 4 + 3 * x[1] ** 3,
                                        vector, 1e-4)
    assert np.allclose(derivative, [4 * vector[0] ** 3, 9 * vector[1] ** 2], atol=1e-10)
    assert np.array_equal(vector, original)
