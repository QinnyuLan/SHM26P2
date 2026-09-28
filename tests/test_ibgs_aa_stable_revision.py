"""Small CPU contracts for the explicit revision wrapper, without real payloads."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
PATH = HERE/'check_ibgs_aa_stable_revision.py'
if not PATH.exists():
    PATH = HERE.parent/'scripts/check_ibgs_aa_stable_revision.py'
spec = importlib.util.spec_from_file_location('stable_revision_tested', PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_explicit_provider_is_restored_on_success_and_exception():
    old, new = (lambda: 'old'), (lambda: 'factored64')
    legacy, stable = SimpleNamespace(aa_opacity_rasterizer=old), SimpleNamespace(aa_opacity_rasterizer=new)
    report = {}
    with worker.inject_provider(legacy, stable, report):
        assert legacy.aa_opacity_rasterizer() == 'factored64'
        assert report['provider_restored'] is False
    assert legacy.aa_opacity_rasterizer is old and report['provider_restored'] is True
    with pytest.raises(RuntimeError, match='render failure'), worker.inject_provider(legacy, stable, report):
        raise RuntimeError('render failure')
    assert legacy.aa_opacity_rasterizer is old and report['provider_restored'] is True


def test_literal_policy_does_not_execute_module(tmp_path):
    source = tmp_path/'provider.py'
    source.write_text("raise RuntimeError('must not import')\nPRECISION_POLICY = {'dtype': 'FP64', 'floor': False}\n")
    assert worker.precision_policy(source) == {'dtype': 'FP64', 'floor': False}


def test_failure_state_requires_fixed_61_step_identity():
    value = {'status': 'failed', 'step': 61, 'arm': 'full', 'protocol': 'ibgs_aa_warm_matched_v1', 'sh_degree': 3}
    worker.validate_failure_state(value)
    for key, bad in [('status', 'completed'), ('step', 62), ('arm', 'no_source')]:
        with pytest.raises(ValueError):
            worker.validate_failure_state({**value, key: bad})


def test_failure_raw_gradient_scope_and_count():
    report = {'raster_calls': 1, 'backward': 1, 'antialias_calls': 1,
              'field_parameters_unchanged': True, 'hook_restored': True,
              'gradient_summary': {k: {'finite': True, 'l2': 0.} for k in worker.RAW_KEYS}}
    worker.validate_mode('failure', report)
    assert '_normal' not in report['gradient_summary'] and '_offset' not in report['gradient_summary']
    for bad in ({**report, 'raster_calls': 2}, {**report, 'field_parameters_unchanged': False}):
        with pytest.raises(ValueError):
            worker.validate_mode('failure', bad)
    bad = copy.deepcopy(report); bad['gradient_summary']['_xyz']['finite'] = False
    with pytest.raises(ValueError):
        worker.validate_mode('failure', bad)


def test_original_gradient_gate_does_not_promote_cap_counterexample():
    report = {'forward_calls': 33, 'hook_calls': 33, 'backward_calls': 3,
              'primary_unsaturated_passed': True, 'numerical_status': 'passed',
              'zero_control': {'rho_zero': True, 'finite_zero_opacity_gradient': True, 'background_exact': True},
              'hook_restored': True, 'numeric_flags_restored': True,
              'cap_derivative_mismatch_observed': True}
    worker.validate_mode('gradient', report)
    with pytest.raises(ValueError):
        worker.validate_mode('gradient', {**report, 'primary_unsaturated_passed': False})
    with pytest.raises(ValueError):
        worker.validate_mode('gradient', {**report, 'forward_calls': 32})
