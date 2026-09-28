import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _load():
    here = Path(__file__).resolve().parent
    path = here/'optimize_raw_simplex.py'
    if not path.is_file():
        path = here.parent/'scripts/optimize_raw_simplex.py'
    spec = importlib.util.spec_from_file_location('raw_simplex_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load()


def test_initialization_is_explicit_and_master_not_fp32():
    q32 = np.array([[.1, .2, .2, .2, .3]], dtype=np.float32)
    saved = q32.copy()
    master, report = runner.initial_master(q32)
    np.testing.assert_allclose(master.sum(axis=1), 1, atol=1e-15)
    assert master.dtype == np.float64
    assert report['original_max_row_sum_error'] > 0
    np.testing.assert_array_equal(q32, saved)
    assert not report['base_full_affine_objective_measured']


def test_actual_displacement_casts_endpoints_before_double_subtraction():
    q = np.array([[.2, .3, .5]])
    proposal = q + np.array([[1e-10, -1e-10, 0.]])
    np.testing.assert_array_equal(runner.actual_displacement(q, proposal), 0)
    assert np.any(proposal - q != 0)


def test_armijo_requires_actual_negative_dot_and_strict_full_decrease():
    g = np.array([[1., 2.]])
    d = np.array([[.1, -.1]])
    assert runner.armijo_decision(1., .99, g, d)['accepted']
    assert not runner.armijo_decision(1., 1., g, d)['accepted']
    assert not runner.armijo_decision(1., 1. - 1e-7, g, d)['accepted']
    assert not runner.armijo_decision(1., .9, g, -d)['accepted']
    assert not runner.armijo_decision(1., .9, g, np.zeros_like(d))['accepted']


def _quadratic_callback(target, calls):
    def full(q, *, gradient, score, phase):
        actual = q.astype(np.float32).astype(np.float64)
        calls.append({'q': q.copy(), 'phase': phase, 'gradient': gradient, 'score': score})
        result = {'complete': True, 'objective': float(.5 * np.square(actual - target).sum())}
        if gradient:
            result['gradient'] = actual - target
        if score:
            result['metrics'] = {'description': 'synthetic'}
        return result
    return full


def test_solver_reserves_final_pass_counts_and_preserves_master():
    q = np.array([[.2, .3, .5]])
    target = np.array([[.65, .2, .15]])
    calls = []
    output, result = runner.solve_projected(
        q, _quadratic_callback(target, calls), max_passes=7,
        max_accepted=3, max_backtracks=4, gap_tolerance=0,
    )
    assert result['complete_passes'] == len(calls) <= 7
    assert calls[0]['phase'] == 'baseline' and calls[-1]['phase'] == 'final'
    assert calls[0]['score'] and calls[-1]['score'] and calls[-1]['gradient']
    assert result['endpoint']['objective'] < result['baseline']['objective']
    assert result['accepted_updates'] <= 3
    assert not result['convergence_certified']
    np.testing.assert_array_equal(calls[-1]['q'], output)
    assert output.dtype == np.float64
    assert np.any(output != output.astype(np.float32).astype(np.float64))


def test_two_pass_budget_can_only_measure_without_any_trial():
    calls = []
    q = np.array([[.5, .5]])
    output, result = runner.solve_projected(
        q, _quadratic_callback(np.array([[.8, .2]]), calls), max_passes=2,
    )
    assert result['stop_reason'] == 'pass_budget'
    assert result['accepted_updates'] == 0
    np.testing.assert_array_equal(q, output)
    assert [row['phase'] for row in calls] == ['baseline', 'final']


def test_partial_trial_cannot_be_accepted_or_mutate_q():
    q = np.array([[.5, .5]])
    accepted = []
    def callback(value, *, gradient, score, phase):
        if phase == 'trial':
            return {'complete': False, 'objective': -100.}
        return {'complete': True, 'objective': 1., 'gradient': np.array([[1., 2.]])}
    with pytest.raises(ValueError, match='Partial'):
        runner.solve_projected(q, callback, on_accept=lambda *args: accepted.append(args))
    assert not accepted
    np.testing.assert_array_equal(q, [[.5, .5]])


def test_callback_timeout_does_not_accept_trial():
    accepted = []
    def callback(value, *, gradient, score, phase):
        if phase == 'trial':
            raise TimeoutError('partial pass')
        return {'complete': True, 'objective': 1., 'gradient': np.array([[1., 2.]])}
    with pytest.raises(TimeoutError):
        runner.solve_projected(np.array([[.5, .5]]), callback,
                               on_accept=lambda *args: accepted.append(args))
    assert accepted == []


def test_failed_backtracking_is_not_convergence_and_final_still_measured():
    phases = []
    def callback(q, *, gradient, score, phase):
        phases.append(phase)
        result = {'complete': True, 'objective': 1.}
        if gradient:
            result['gradient'] = np.array([[1., 2.]])
        return result
    _, report = runner.solve_projected(np.array([[.5, .5]]), callback, max_backtracks=3)
    assert report['stop_reason'] == 'backtracking_exhausted'
    assert phases == ['baseline', 'trial', 'trial', 'trial', 'final']
    assert report['accepted_updates'] == 0 and not report['convergence_certified']


def test_constant_gradient_stops_without_a_proposal():
    phases = []
    def callback(q, *, gradient, score, phase):
        phases.append(phase)
        return {'complete': True, 'objective': 1., 'gradient': np.full_like(q, 4.)}
    q = np.array([[.4, .6]])
    out, report = runner.solve_projected(q, callback)
    assert report['stop_reason'] == 'zero_tangent_gradient'
    assert phases == ['baseline', 'final']
    assert out.tobytes() == q.tobytes()


def test_full_report_actual_synthetic_solve_strict_json(tmp_path):
    q = np.array([[.2, .3, .5]])
    out, result = runner.solve_projected(
        q, _quadratic_callback(np.array([[.65, .2, .15]]), []),
        max_passes=7, max_accepted=3, max_backtracks=4, gap_tolerance=0,
    )
    _, initialization = runner.initial_master(np.array([[.1, .2, .2, .2, .3]], np.float32))
    result['initialization'] = initialization
    result['empty_class_metrics'] = runner.confusion_metrics(np.diag([7, 0, 2, 0, 3]))
    text = json.dumps(result, allow_nan=False)
    assert json.loads(text)['complete_passes'] <= 7
    runner.write(tmp_path/'report.json', result)
    assert runner.read(tmp_path/'report.json') == json.loads(text)
    assert out.dtype == np.float64


def test_view_gradient_cast_before_mean_preserves_tiny_values():
    tiny = np.nextafter(np.float32(0), np.float32(1))
    per_view = np.array([[tiny, -tiny]], np.float32)
    accumulator = np.zeros((1, 2), np.float64)
    for _ in range(259):
        runner.accumulate_view_gradient(accumulator, per_view)
    np.testing.assert_array_equal(accumulator / 259, per_view.astype(np.float64))
    assert np.all(per_view / np.float32(259) == 0)
    with pytest.raises(ValueError):
        runner.accumulate_view_gradient(accumulator, per_view.astype(np.float64))


def test_exclusive_attempt_claim_and_existing_endpoint_reject(tmp_path):
    runner.claim_attempt(tmp_path, 'a' * 64)
    assert runner.read(tmp_path/'execution_started.json')['plan_sha256'] == 'a' * 64
    with pytest.raises(ValueError, match='Previous'):
        runner.claim_attempt(tmp_path, 'a' * 64)
    another = tmp_path/'old'; another.mkdir()
    (another/'final_q_delta.npz').write_bytes(b'old endpoint')
    with pytest.raises(ValueError, match='Previous'):
        runner.claim_attempt(another, 'a' * 64)
    assert not (another/'execution_started.json').exists()


def _preflight_records():
    plan = {'specification': {'protocol': 'simplex_scene_preflight_v2', 'names': ['002.png'],
                              'counts': {'scene': 1}, 'numerics': {'tf32': False}},
            'expected_gsplat_binary': {'path': '/fake/frozen.so', 'sha256': 'binary'}}
    receipt = {'status': 'completed', 'numerical_status': 'passed', 'plan_sha256': 'plan',
               'analysis_sha256': 'analysis', 'inputs_and_sources_unchanged': True,
               'numerics_restored': True, 'restoration': {'state_exact': True,
                                                        'flags_modes_gradients_restored': True},
               'counts': plan['specification']['counts'], 'numerics_actual': plan['specification']['numerics'],
               'actual_gsplat_binary': plan['expected_gsplat_binary']}
    launch = {'status': 'completed', 'natural_completion': True, 'exit_code': 0,
              'execution_receipt_sha256': 'receipt', 'plan_sha256': 'plan'}
    analysis = {'numerical_status': 'passed', 'plan_sha256': 'plan', 'specification': plan['specification'],
                'records': [{'name': '002.png', 'passed': True, 'gates': {'fd': True}}],
                'counts': plan['specification']['counts']}
    return plan, receipt, launch, analysis


def test_preflight_needs_numerical_pass_natural_exit_and_restoration():
    for field in ('numeric', 'natural', 'restored', 'hash', 'failed_v1', 'false_record'):
        plan, receipt, launch, analysis = _preflight_records()
        runner.validate_preflight(plan, receipt, launch, analysis, plan_sha='plan',
                                  receipt_sha='receipt', analysis_sha='analysis')
        if field == 'numeric':
            receipt['numerical_status'] = 'failed_or_inconclusive'
        elif field == 'natural':
            launch['natural_completion'] = False
        elif field == 'restored':
            receipt['restoration']['state_exact'] = False
        elif field == 'hash':
            launch['execution_receipt_sha256'] = 'wrong'
        elif field == 'failed_v1':
            plan['specification']['protocol'] = 'simplex_scene_preflight_v1'
        else:
            analysis['records'][0]['gates']['fd'] = False
        with pytest.raises(ValueError):
            runner.validate_preflight(plan, receipt, launch, analysis, plan_sha='plan',
                                      receipt_sha='receipt', analysis_sha='analysis')


def test_write_serializes_before_open_and_atomic_replace(tmp_path):
    path = tmp_path/'receipt.json'
    runner.write(path, {'status': 'running'})
    old = path.read_bytes()
    for invalid in ({'bad': np.bool_(True)}, {'bad': np.nan}):
        with pytest.raises((TypeError, ValueError)):
            runner.write(path, invalid, replace=True)
        assert path.read_bytes() == old
    runner.write(path, {'status': 'completed'}, replace=True)
    assert runner.read(path) == {'status': 'completed'}
    assert list(tmp_path.iterdir()) == [path]
