import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PATH = HERE/'diagnose_simplex_direction.py'
if not PATH.exists():
    PATH = HERE.parent/'scripts/diagnose_simplex_direction.py'
spec = importlib.util.spec_from_file_location('simplex_direction', PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_affine_derivative_matches_independent_matrix_chain_and_fd():
    W = np.array([[.5, .25], [.15, .7], [0., 0.]], np.float64)
    q = np.array([[.2, .3, .5], [.6, .1, .3]], np.float64)
    v = np.array([[0., 1., 0.], [0., 0., 1.]], np.float64)
    labels = np.array([1, 2, 2], np.uint8)
    cw = np.array([.5, 1.2, 1.7, 1., 1.])
    background = np.column_stack((1-W.sum(1), np.zeros((3, 2))))
    current = ((W@q+background)[np.arange(3), labels]).astype(np.float32)
    vertex = ((W@v+background)[np.arange(3), labels]).astype(np.float32)
    # Use exact endpoint line (including FP32 endpoint rounding), no finite-h log loss in the primary test.
    result = worker.cached_terms(current, vertex, labels, cw, 0., upstream=True)
    a = (1-worker.SPEC['delta'])*current.astype(np.float64)+worker.SPEC['delta']/5
    b = (1-worker.SPEC['delta'])*(vertex.astype(np.float64)-current.astype(np.float64))
    expected = -np.sum(cw[labels]*b/a)
    assert result['derivative_sum'] == pytest.approx(expected, abs=1e-15)
    h = 1e-6
    plus = worker.cached_terms(current, vertex, labels, cw, .3+h)['loss_sum']
    minus = worker.cached_terms(current, vertex, labels, cw, .3-h)['loss_sum']
    derivative = worker.cached_terms(current, vertex, labels, cw, .3)['derivative_sum']
    assert derivative == pytest.approx((plus-minus)/(2*h), abs=2e-9)
    assert result['derivative_pixel'][-1] == 0  # zero-W foreground: fixed noise has no q gradient


def test_stream_means_upstream_rounding_and_bin_closure(tmp_path):
    records = []
    values = [([0., .2], [.1, .15], [2, 0]), ([.4, .6, .5], [.3, .8, .2], [1, 3, 4])]
    weights = np.array([.5, .9, 1.7, 1.1, 1.3], np.float64)
    expected_b = expected_b32 = expected_f = 0.
    for index, (a, v, y) in enumerate(values):
        a, v, y = np.array(a, np.float32), np.array(v, np.float32), np.array(y, np.uint8)
        paths = {}
        for name, array in (('a', a), ('v', v), ('labels', y)):
            paths[name] = worker.save_array(tmp_path, f'{index}.{name}.npy', array)
        records.append({'name': str(index), 'pixels': len(a), **paths})
        p = (1-worker.SPEC['delta'])*a.astype(np.float64)+worker.SPEC['delta']/5
        delta_raw = v.astype(np.float64)-a.astype(np.float64)
        upstream = -weights[y]*(1-worker.SPEC['delta'])/len(a)/p
        expected_b += np.sum(upstream*delta_raw)/2
        expected_b32 += np.sum(upstream.astype(np.float32).astype(np.float64)*delta_raw)/2
        expected_f += np.mean(-weights[y]*np.log(p))/2
    report = worker.line_summary(records, weights, 0., bins=True)
    assert report['F'] == pytest.approx(expected_f, abs=1e-15)
    assert report['B64'] == pytest.approx(expected_b, rel=1e-15)
    assert report['B32upstream'] == pytest.approx(expected_b32, rel=1e-15)
    bins = report['probability_bins']
    assert sum(bins['pixel_counts']) == 5
    assert sum(bins['positive'])+sum(bins['negative']) == pytest.approx(report['B64'])
    assert sum(bins['absolute']) >= abs(report['B64'])
    json.dumps(report, allow_nan=False)


def test_gate_rejects_wrong_adjoint_even_with_correct_sign_and_signal():
    assert worker.direction_gate(-.1, -.1, -.1)['passed']
    assert not worker.direction_gate(-.1, -.02, -.02)['passed']
    assert not worker.direction_gate(-1e-8, -1e-8, -1e-8)['passed']
    assert not worker.direction_gate(.1, .1, .1)['passed']
    assert not worker.direction_gate(-.1, -.2, -.1)['passed']  # upstream cast not assumed harmless
    with pytest.raises(ValueError):
        worker.direction_gate(np.nan, -.1, -.1)


def test_single_convex_line_search_interior_and_endpoint():
    value, report = worker.fixed_line_search(lambda t: 2*(t-.123))
    assert value == pytest.approx(.123, abs=1e-15)
    assert report['iterations'] == 64
    value, report = worker.fixed_line_search(lambda t: -.5)
    assert value == 1. and report['iterations'] == 0
    with pytest.raises(ValueError):
        worker.fixed_line_search(lambda t: t+.1)


def test_actual_candidate_must_strictly_decrease_and_match_cache():
    assert worker.candidate_gate(.1, .09, .09)['passed']
    assert not worker.candidate_gate(.1, .09, .1)['passed']
    assert not worker.candidate_gate(.1, .09, .099)['passed']
    assert not worker.candidate_gate(.1, .1001, .0999)['passed']


def test_strict_json_failure_does_not_create_or_truncate_report(tmp_path):
    path = tmp_path/'receipt.json'
    with pytest.raises(TypeError):
        worker.write(path, {'bad': np.bool_(True)})
    assert not path.exists()
    worker.write(path, {'status': 'running'})
    with pytest.raises(ValueError):
        worker.write(path, {'bad': np.inf}, replace=True)
    assert worker.read(path) == {'status': 'running'}


def test_parent_requires_natural_completion_and_independent_audit():
    plan = {'specification': {'protocol': 'raw_simplex_fullbatch_v1'}}
    receipt = {'status': 'completed', 'plan_sha256': worker.PARENT_PLAN_SHA,
               'state_unchanged_before_restore': True, 'inputs_sources_unchanged': True}
    launch = {'status': 'completed', 'natural_completion': True, 'exit_code': 0,
              'execution_receipt_sha256': 'receipt', 'plan_sha256': worker.PARENT_PLAN_SHA}
    audit = {'status': 'passed', 'plan_sha256': worker.PARENT_PLAN_SHA}
    worker.validate_parent(plan, receipt, launch, audit, worker.PARENT_PLAN_SHA, 'receipt')
    audit['status'] = 'failed'
    with pytest.raises(ValueError):
        worker.validate_parent(plan, receipt, launch, audit, worker.PARENT_PLAN_SHA, 'receipt')


def test_background_is_not_removed_or_assigned_to_q_in_line():
    # Identical fixed background adds to both endpoints but cancels only in b, not in p.
    y = np.array([0], np.uint8); weights = np.ones(5)
    a, v = np.array([.8], np.float32), np.array([.9], np.float32)
    result = worker.cached_terms(a, v, y, weights, 0., upstream=True)
    expected = -(1-worker.SPEC['delta'])*(float(v[0])-float(a[0]))/(
        (1-worker.SPEC['delta'])*float(a[0])+worker.SPEC['delta']/5)
    assert result['derivative_sum'] == pytest.approx(expected)


@pytest.mark.parametrize('mutation', [None, 'numerical_spec', 'nonzero_work', 'modified_source'])
def test_workflow_revision_only_accepts_bound_zero_work_cache_failure(monkeypatch, mutation):
    snapshot = worker.FAILED_V1/'source_snapshot'
    original = {PATH.name: 'oldworker', 'bridge_rgs/model.py': 'model'}
    old = {'specification': {**worker.SPEC, 'protocol': 'simplex_direction_diagnostic_v1'},
           'source_snapshot': str(snapshot), 'source_hashes': original}
    receipt = {'status': 'failed', 'error': 'ValueError: Frozen contract changed',
               'counts': {'scene': 0, 'shader': 0, 'vjp': 0},
               'plan_sha256': worker.FAILED_V1_HASHES['plan.json']}
    launch = {'status': 'failed', 'exit_code': 1, 'natural_completion': False,
              'execution_receipt_sha256': worker.FAILED_V1_HASHES['execution_receipt.json'],
              'plan_sha256': worker.FAILED_V1_HASHES['plan.json']}
    actual = {**original, **{f'.pytest_cache/{i}': str(i) for i in range(4)}}
    records = dict(zip(worker.FAILED_V1_HASHES, (old, receipt, launch), strict=True))
    monkeypatch.setattr(worker, 'sha', lambda p: worker.FAILED_V1_HASHES[Path(p).name])
    monkeypatch.setattr(worker, 'read', lambda p: records[Path(p).name])
    monkeypatch.setattr(worker, 'files', lambda p: actual)
    if mutation == 'numerical_spec':
        old['specification']['bisection_iterations'] = 65
    elif mutation == 'nonzero_work':
        receipt['counts']['shader'] = 1
    elif mutation == 'modified_source':
        actual['bridge_rgs/model.py'] = 'changed'
    if mutation is not None:
        with pytest.raises(ValueError):
            worker.workflow_revision()
    else:
        inputs, revision = worker.workflow_revision()
        assert inputs[str(snapshot/PATH.name)] == 'oldworker'
        assert len(inputs) == 8
        assert revision['numerical_specification_unchanged']
        assert 'no:cacheprovider' in revision['frozen_test_rule']
        json.dumps(revision, allow_nan=False)
