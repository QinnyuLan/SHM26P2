"""Small CPU-only lineage/scope gates; no scene or target decoding."""
import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/preflight_ibgs_aa_port.py'
if not PATH.exists():
    PATH = Path(__file__).with_name('preflight_ibgs_aa_port.py')
spec = importlib.util.spec_from_file_location('aa_port_preflight_tested', PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_compatibility_needs_natural_success_and_matching_lineage():
    execution = {'status': 'completed', 'plan_sha256': 'p', 'field_unchanged': True,
                 'hook_restored': True, 'prediction_barrier_complete': True,
                 'render_calls': 16, 'antialias_calls': 16, 'backward': 0, 'optimizer_steps': 0}
    launch = {'status': 'completed', 'natural_completion': True, 'exit_code': 0,
              'plan_sha256': 'p', 'execution_receipt_sha256': 'e'}
    review = {'status': 'passed', 'plan_sha256': 'p', 'execution_receipt_sha256': 'e'}
    worker.validate_compatibility(execution, launch, review, 'p', 'e')
    for bad in ({**launch, 'exit_code': 1}, {**launch, 'execution_receipt_sha256': 'wrong'}):
        with pytest.raises(ValueError):
            worker.validate_compatibility(execution, bad, review, 'p', 'e')
    with pytest.raises(ValueError):
        worker.validate_compatibility({**execution, 'antialias_calls': 15}, launch, review, 'p', 'e')


def test_fixed_source_bank_excludes_self_and_heldout():
    names = [f'{i:03}.png' for i in range(350)]
    contract = {'train_rows': [{'name': n, 'split': 'train'} for n in names],
                'neighbors': {'views': [
                    {'name': '002.png', 'neighbors_4': [{'name': n} for n in names[3:7]]},
                    {'name': '041.png', 'neighbors_4': [{'name': n} for n in names[42:46]]}]}}
    _, sources = worker.selected_sources(contract)
    assert sources == names[3:7] + names[42:46]
    changed = copy.deepcopy(contract)
    changed['neighbors']['views'][0]['neighbors_4'][0]['name'] = '002.png'
    with pytest.raises(ValueError):
        worker.selected_sources(changed)
    contract['train_rows'][3]['split'] = 'val'
    with pytest.raises(ValueError):
        worker.selected_sources(contract)


def test_success_requires_counts_gradients_and_restoration():
    report = {'counts': {'depth_renders': 8, 'target_renders': 2, 'backward': 2, 'optimizer_steps': 0},
              'antialias_calls': 10, 'field_parameters_unchanged': True, 'network_parameters_unchanged': True,
              'hook_restored': True, 'numerics_restored': True, 'rgb_decodes': 10,
              'source_depth_updates': 8, 'valid_decodes': 1,
              'target_rows': [{'name': n, 'fusion_gradients_finite': True,
                               'gradient_summary': {'_xyz': {'finite': True}}} for n in worker.TARGETS]}
    worker.validate_result(report)
    for key, value in [('antialias_calls', 9), ('hook_restored', False), ('network_parameters_unchanged', False)]:
        with pytest.raises(ValueError):
            worker.validate_result({**report, key: value})
    changed = copy.deepcopy(report)
    changed['target_rows'][1]['gradient_summary']['_xyz']['finite'] = False
    with pytest.raises(ValueError):
        worker.validate_result(changed)
