"""Small CPU contracts; no preparation, model loading, images or CUDA."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
WORKER = HERE/'train_ibgs_aa_stable_matched.py'
if not WORKER.exists():
    WORKER = HERE.parent/'scripts/train_ibgs_aa_stable_matched.py'
spec = importlib.util.spec_from_file_location('stable_warm_contract_worker', WORKER)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
SCOPE_PATH = HERE/'train_ibgs_aa_warm_matched.py'
if not SCOPE_PATH.exists():
    SCOPE_PATH = worker.FAILED_AA/'source_snapshot/train_ibgs_aa_warm_matched.py'
OLD_PATH = HERE/'train_ibgs_warm_matched.py'
if not OLD_PATH.exists():
    OLD_PATH = worker.PARENT/'source_snapshot/train_ibgs_warm_matched.py'
POLICY_PATH = HERE/'bridge_rgs/ibgs_antialias_stable.py'
if not POLICY_PATH.exists():
    POLICY_PATH = worker.ROOT/'src/bridge_rgs/ibgs_antialias_stable.py'
SCOPE = worker.load_scope(SCOPE_PATH)
OLD = SCOPE.load_worker(OLD_PATH)
POLICY = worker.precision_policy(POLICY_PATH)


def test_only_explicit_identity_precision_and_timeout_change():
    original = worker.read(worker.PARENT/'plan.json')['specification']
    before = copy.deepcopy(original)
    revised = worker.revised_specification(original, POLICY)
    assert original == before
    assert {k for k in revised if revised[k] != original.get(k)} == {
        'protocol', 'renderer_profile', 'aa_provider', 'precision_policy',
        'internal_seconds_per_arm', 'external_seconds'}
    assert revised['steps_per_arm'] == 6000 and revised['seed'] == 42
    assert (revised['geometry_steps'], revised['fusion_detached_steps'], revised['fusion_joint_steps']) == (1000, 1000, 4000)
    assert (revised['internal_seconds_per_arm'], revised['external_seconds']) == (3600, 7500)
    json.dumps(revised, allow_nan=False)


def test_actual_original_checkpoint_is_required_not_failed_field():
    plan = worker.read(worker.PARENT/'plan.json')
    worker.validate_initial_checkpoint(plan, OLD)
    for checkpoint, digest in ((str(worker.FAILED_AA/'full/failure.pt'), worker.INITIAL_SHA),
                                (worker.INITIAL, 'different')):
        broken = copy.deepcopy(plan); broken['checkpoint'] = checkpoint
        broken['input_hashes'][checkpoint] = digest
        with pytest.raises(RuntimeError, match='original'):
            worker.validate_initial_checkpoint(broken, OLD)


def evidence(mode):
    plan = {'mode': mode, 'renderer_profile': worker.PROFILE, 'aa_provider': worker.PROVIDER,
            'aa_module_sha256': 'module', 'precision_policy': POLICY,
            'source_hashes': {'bridge_rgs/ibgs_antialias_stable.py': 'module'},
            'binary': {'sha256': worker.BINARY_SHA}}
    report = {'numerical_status': 'passed', 'provider_restored': True,
              'provider_binding': {'module': worker.PROVIDER, 'callable_module': worker.PROVIDER,
                                   'sha256': 'module', 'precision_policy': POLICY},
              'optimizer_steps': 0, 'data_reads': 0, 'hook_restored': True}
    if mode == 'compatibility':
        report.update(render_calls=16, antialias_calls=16, backward=0, predictions=[{}]*16,
                      field_unchanged=True, prediction_barrier_complete=True)
    elif mode == 'gradient':
        report.update(primary_unsaturated_passed=True, positive_case=[{'finite_differences': [{'passed': True}]}]*4,
                      zero_control={k: True for k in ('rho_zero', 'finite_zero_opacity_gradient', 'background_exact')},
                      forward_calls=33, hook_calls=33, backward_calls=3, numeric_flags_restored=True,
                      cap_derivative_mismatch_observed=True)
    elif mode == 'preflight':
        report.update(counts={'depth_renders': 8, 'target_renders': 2, 'backward': 2, 'optimizer_steps': 0},
                      antialias_calls=10, field_parameters_unchanged=True, network_parameters_unchanged=True,
                      numerics_restored=True, rgb_decodes=10, valid_decodes=1, source_depth_updates=8,
                      target_rows=[{'name': name, 'fusion_gradients_finite': True,
                                    'gradient_summary': {str(i): {'finite': True} for i in range(8)}}
                                   for name in ('002.png', '041.png')])
    else:
        plan['diagnostic_failure_checkpoint'] = {'sha256': 'abbe8973ebc7649e99b162f9d878129232fe5475ffe8672f2e4fc9b48c5b5d8f'}
        report.update(raster_calls=1, backward=1, field_parameters_unchanged=True, antialias_calls=1,
                      camera='004.png', checkpoint_step=61,
                      gradient_summary={str(i): {'finite': True} for i in range(6)})
    return plan, report


@pytest.mark.parametrize('mode', worker.MODES)
def test_all_four_new_checks_require_actual_stable_provider(mode):
    plan, report = evidence(mode)
    worker.validate_revision(mode, plan, report, 'module', POLICY, OLD, SCOPE)
    for key, value in (('numerical_status', 'not_passed'), ('provider_restored', False)):
        bad = copy.deepcopy(report); bad[key] = value
        with pytest.raises(RuntimeError):
            worker.validate_revision(mode, plan, bad, 'module', POLICY, OLD, SCOPE)
    bad = copy.deepcopy(report); bad['provider_binding']['callable_module'] = 'bridge_rgs.ibgs_antialias'
    with pytest.raises(RuntimeError):
        worker.validate_revision(mode, plan, bad, 'module', POLICY, OLD, SCOPE)
    bad = copy.deepcopy(plan); bad['binary']['sha256'] = 'old binary'
    with pytest.raises(RuntimeError):
        worker.validate_revision(mode, bad, report, 'module', POLICY, OLD, SCOPE)


def test_completed_receipt_alone_cannot_replace_natural_exit(tmp_path):
    OLD.write(tmp_path/'plan.json', {})
    report = {'status': 'completed', 'numerical_status': 'passed', 'plan_sha256': OLD.sha(tmp_path/'plan.json')}
    OLD.write(tmp_path/'execution_receipt.json', report)
    launch = {'status': 'completed', 'natural_completion': True, 'exit_code': 0,
              'plan_sha256': report['plan_sha256'], 'execution_receipt_sha256': OLD.sha(tmp_path/'execution_receipt.json')}
    OLD.write(tmp_path/'launch_receipt.json', launch)
    SCOPE.completed_evidence(tmp_path, OLD, numerical=True)
    launch['exit_code'] = 1
    (tmp_path/'launch_receipt.json').write_text(json.dumps(launch))
    with pytest.raises(RuntimeError):
        SCOPE.completed_evidence(tmp_path, OLD, numerical=True)


def test_metadata_scope_restores_when_inherited_arm_fails():
    assert OLD.sha(worker.FAILED_AA/'source_snapshot/train_ibgs_aa_warm_matched.py') == worker.SCOPE_SHA
    original = SCOPE.PROTOCOL, SCOPE.PROFILE, SCOPE.renderer_metadata
    with pytest.raises(RuntimeError), worker.inherited_scope(SCOPE):
        assert SCOPE.PROTOCOL == worker.PROTOCOL and SCOPE.renderer_metadata is worker.renderer_metadata
        raise RuntimeError('synthetic inherited failure')
    assert (SCOPE.PROTOCOL, SCOPE.PROFILE, SCOPE.renderer_metadata) == original


def test_frozen_endpoint_serializer_records_stable_provider_and_precision(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', list)
    field = SimpleNamespace(_xyz=torch.nn.Parameter(torch.tensor([[1., 2., 3.]])), spatial_lr_scale=2.)
    inner = torch.nn.Linear(2, 3)
    field_opt = torch.optim.Adam([field._xyz], lr=.01)
    head_opt = torch.optim.Adam(inner.parameters(), lr=.02)
    plan = {'specification': worker.revised_specification({}, POLICY), 'camera_order_sha256': 'order',
            'data_contract': 'contract', 'data_contract_sha256': 'contractsha', 'coordinate_profile': 'colmap_corner_v2',
            'aa_module_sha256': 'module', 'binary': {'sha256': worker.BINARY_SHA}, 'precision_policy': POLICY}
    with worker.inherited_scope(SCOPE):
        SCOPE.save_endpoint(OLD, SimpleNamespace(FIELD_KEYS=('_xyz',)), SimpleNamespace(calls=6350), tmp_path/'last.pt',
            field=field, net=SimpleNamespace(inner=inner), field_opt=field_opt, head_opt=head_opt,
            background=torch.tensor([.2, .3, .4]), arm='full', step=6000, plan=plan,
            plan_sha='plan', neighbors={}, counts={}, status='completed')
    saved = torch.load(tmp_path/'last.pt', weights_only=False, map_location='cpu')
    assert saved['protocol'] == worker.PROTOCOL and saved['renderer_profile'] == worker.PROFILE
    assert saved['aa_provider'] == worker.PROVIDER and saved['precision_policy'] == POLICY
    assert saved['aa_module_sha256'] == 'module' and saved['renderer_binary']['sha256'] == worker.BINARY_SHA
    assert saved['counts']['aa_opacity_calls'] == 6350 and saved['step'] == 6000
    assert torch.equal(saved['field']['_xyz'], field._xyz.detach())
