"""Synthetic admission and unchanged-control contracts; no real run is prepared."""
import ast
import copy
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name, filename):
    path = HERE/filename
    if not path.exists():
        path = HERE.parent/('tests' if filename.startswith('test_') else 'scripts')/filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


worker, WORKER_PATH = load('proposal_contract_worker', 'train_continuous_geometry_proposals.py')
old, OLD_PATH = load('old_continuous_contract_worker', 'train_continuous_semantic_geometry.py')
contracts, _ = load('continuous_existing_contracts', 'test_continuous_geometry_train.py')


def evidence():
    digests = dict(worker.EVIDENCE_SHA)
    common_plan = {'base_checkpoint': 'base', 'manifest': 'manifest', 'q_delta': 'q',
                   'numerics': {'tf32': False}, 'expected_gsplat_binary': {'sha256': 'binary'}}
    common_exec = {'status': 'completed', 'sources_inputs_unchanged': True, 'numerics_restored': True,
                   'restoration': {'state_exact': True, 'flags_modes_gradients_restored': True}}
    result = []
    for name in ('preflight', 'chain'):
        result.append({'plan': copy.deepcopy(common_plan),
            'execution': {**copy.deepcopy(common_exec), 'plan_sha256': digests[name+'_plan']},
            'launch': {'status': 'completed', 'natural_completion': True, 'exit_code': 0,
                       'plan_sha256': digests[name+'_plan'], 'execution_receipt_sha256': digests[name+'_execution']},
            'review': {'status': 'passed', 'plan_sha256': digests[name+'_plan']}})
    preflight, chain = result
    preflight['execution'].update(numerical_status='not_passed', main_count=12, main_measurable=12,
        main_passed=3, counts={'raster': 56, 'means_vjp': 4, 'target_decodes': 6})
    preflight['review'].update(numerical_status='not_passed', main_passed=3)
    chain['execution'].update(diagnostic_status='completed_localization_only',
        counts={'native_rasters': 3, 'replay_rasters': 3, 'backward_sweeps': 1, 'target_decodes': 2},
        replay={'endpoint_replay_allowed': True, 'baseline_four_channel_and_alpha_exact': True})
    chain['review'].update(execution_receipt_sha256=digests['chain_execution'], before_after_bound_hashes_exact=True,
                          original_numerical_gate={'status': 'not_passed', 'passed': 3, 'total': 12})
    return preflight, chain, digests


def test_numerical_worker_functions_and_fixed_budget_are_unchanged():
    def definitions(path):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
    current, previous = definitions(WORKER_PATH), definitions(OLD_PATH)
    for name in ('batch_schedule', 'accept_candidate', 'evaluate_view_objectives', 'run_attempts', 'perform'):
        assert current[name] == previous[name]
    changed = {k for k in set(old.SPEC) | set(worker.SPEC) if old.SPEC.get(k) != worker.SPEC.get(k)}
    assert changed == {'protocol', 'gradient_interpretation', 'adoption'}
    assert worker.SPEC['expected_counts']['raster'] == 14504
    assert worker.SPEC['expected_counts']['means_vjp'] == 6216


def test_new_admission_preserves_old_failed_gate_even_when_record_audit_passes(monkeypatch, tmp_path):
    preflight, chain, digests = evidence()
    accepted_evidence = worker.validate_evidence(preflight, chain, digests)
    assert accepted_evidence['independent_record_audits_passed']
    assert accepted_evidence['old_preflight_numerical_status'] == 'not_passed'
    assert accepted_evidence['old_fd_passed'] == 3 and accepted_evidence['old_fd_count'] == 12
    assert not accepted_evidence['old_fd_gate_passed'] and not accepted_evidence['derivative_certified']
    records = {key: preflight[role] for key, role in (
        ('plan.json', 'plan'), ('execution_receipt.json', 'execution'), ('launch_receipt.json', 'launch'))}
    monkeypatch.setattr(old, 'read', lambda path: records[Path(path).name])
    output = Path('/mnt/data/continuous_proposal_synthetic_old_prepare_must_not_create')
    assert not output.exists()
    with pytest.raises(ValueError, match='preflight must pass'):
        old.prepare(output, tmp_path)
    assert not output.exists()


@pytest.mark.parametrize('corruption', ['audit_failed', 'renamed_old_pass', 'incomplete_chain',
                                       'wrong_sha', 'different_geometry', 'unsafe_replay'])
def test_evidence_admission_rejects_corruption(corruption):
    preflight, chain, digests = evidence()
    if corruption == 'audit_failed':
        chain['review']['status'] = 'failed'
    elif corruption == 'renamed_old_pass':
        preflight['execution']['numerical_status'] = 'passed'
    elif corruption == 'incomplete_chain':
        chain['launch']['exit_code'] = 1
    elif corruption == 'wrong_sha':
        digests['preflight_execution'] = '0'*64
    elif corruption == 'different_geometry':
        chain['plan']['base_checkpoint'] = 'different'
    elif corruption == 'unsafe_replay':
        chain['execution']['replay']['endpoint_replay_allowed'] = False
    with pytest.raises(ValueError):
        worker.validate_evidence(preflight, chain, digests)


def test_existing_real_reduction_controller_and_rollback_contracts_on_new_entrypoint(monkeypatch):
    monkeypatch.setattr(contracts, 'worker', worker)
    contracts.test_fixed_148_schedule_has_four_complete_rounds_and_does_not_consume_global_rng()
    contracts.test_actual_reduction_is_equal_view_not_pooled_pixels_for_losses_and_gradients()
    contracts.test_each_fp32_vjp_is_promoted_before_accumulation()
    contracts.test_all_fixed_attempts_use_baseline_candidate_and_real_adam_without_extra_evaluations()
    contracts.test_actual_semantic_improvement_is_rejected_when_rgb_worsens_and_adam_restores()
    for case in ('ce_equal', 'batch_rgb', 'single_view_rgb'):
        contracts.test_trust_rejects_each_independent_condition_without_rejecting_finite_controls(case)
    contracts.test_trust_exact_boundaries_relative_and_absolute_rgb_allowance()
    for failure in ('exception', 'partial', 'bad_mean', 'nan'):
        contracts.test_candidate_failure_rolls_back_existing_moments_and_does_not_emit_completed_attempt(failure)
