"""CPU-only endpoint verification: no renderer/model load of real artifacts."""
import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from bridge_rgs.fullbatch_appearance import inference_checkpoint

spec = importlib.util.spec_from_file_location(
    'h3_endpoint', Path(__file__).parents[1]/'scripts/audit_fullbatch_appearance_h3_endpoint.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fullpass(index, role, loss, *, on_time=True):
    return {'pass': index, 'role': role, 'loss': loss, 'complete': True,
            'renders': 350, 'seconds': 1., 'on_time': on_time}


def transaction(alpha, loss, *, slope=-.1, accepted=False, passes=2):
    return {'accepted_step_before': 0, 'alpha': alpha, 'baseline': 1., 'candidate': loss,
            'g_dot_d': -.1, 'g_dot_actual_delta': slope,
            'displacement_absmax': dict.fromkeys(audit.KEYS, .01),
            'accepted': accepted, 'completed_passes': passes}


def receipt(*, passes=2, partial=0, accepted=1, loss=.99, seconds=4., reason='complete_pass_or_time_budget'):
    return {'complete_passes': passes, 'partial_renders': partial, 'accepted_steps': accepted,
            'renderer_calls': 350*passes+partial, 'last_accepted_train_loss': loss,
            'optimization_seconds': seconds, 'stop_reason': reason}


def test_scalar_armijo_acceptance_with_rejected_first_alpha():
    rows = [fullpass(1, 'baseline_gradient', 1.), fullpass(2, 'trial_alpha_1.0', 1.01),
            transaction(1., 1.01), fullpass(3, 'trial_alpha_0.5', .99),
            transaction(.5, .99, slope=-.05, accepted=True, passes=3)]
    result = audit.audit_passes(rows, receipt(passes=3))
    assert result['accepted_steps'] == 1 and result['renderer_calls'] == 1050
    assert result['transactions'][1]['decrease_floor'] == 1e-5
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('mutation', ['false_accept', 'skip_alpha', 'wrong_count', 'wrong_last'])
def test_log_tampering_rejected(mutation):
    rows = [fullpass(1, 'baseline_gradient', 1.), fullpass(2, 'trial_alpha_1.0', .99),
            transaction(1., .99, accepted=True)]
    record = receipt()
    if mutation == 'false_accept':
        rows[1]['loss'] = rows[2]['candidate'] = 1.01
    elif mutation == 'skip_alpha':
        rows[1]['role'] = 'trial_alpha_0.5'
        rows[2]['alpha'] = .5
    elif mutation == 'wrong_count':
        record['renderer_calls'] += 1
    else:
        record['last_accepted_train_loss'] = .98
    with pytest.raises(ValueError):
        audit.audit_passes(rows, record)


def test_partial_trial_not_accepted_or_counted_as_complete():
    rows = [fullpass(1, 'baseline_gradient', 1.),
            {'role': 'trial_alpha_1.0', 'complete': False, 'renders': 14, 'seconds': 1., 'loss': None},
            transaction(1., None, passes=1)]
    result = audit.audit_passes(rows, receipt(passes=1, partial=14, accepted=0, loss=None, seconds=361.,
                                             reason='time_budget_trial_pass_discarded'))
    assert result['complete_passes'] == 1 and result['renderer_calls'] == 364


def fixture_checkpoint():
    base = {'format_version': 1, 'config': {'manifest': 'old.json'}, 'step': 8000,
            'scene_scale': 2., 'feature_dim': 16, 'sh_degree': 3,
            'refiner_config': {'type': 'multiscale'},
            'model': {**{k: torch.zeros(2, 3) for k in audit.KEYS},
                      'splats.means': torch.ones(2, 3), 'buffer': torch.zeros(2, dtype=torch.int64)},
            'training_cameras': torch.eye(4)[None], 'optimizers': {'old': 'must not survive'},
            'numpy_rng': 'must not survive'}
    plan = {'base': {'path': 'base.pt', 'sha256': audit.BASE_SHA},
            'manifest': {'path': 'old.json', 'sha256': 'observed'},
            'source_hashes': {'source.py': 'source'},
            'specification': {'protocol': 'conditional_fullbatch_appearance_h3_v2'}}
    end = {'accepted_steps': 1, 'complete_passes': 2, 'last_accepted_train_loss': .99,
           'stop_reason': 'complete_pass_or_time_budget', 'training_camera_mapping': {'exact': True}}
    metadata = {'protocol': plan['specification']['protocol'], 'base': plan['base'],
                'manifest_observed': plan['manifest'], 'source_files_sha256': plan['source_hashes'],
                'stage_config': plan['specification'], 'base_manifest_declared_sha256': None, **end}
    model = copy.deepcopy(base['model'])
    model['splats.sh0'] += .001
    candidate = inference_checkpoint(base, model, metadata)
    return base, candidate, plan, end


def test_inference_checkpoint_preserves_historical_absence_and_exact_frozen_tensors():
    base, candidate, plan, end = fixture_checkpoint()
    result = audit.compare_checkpoint(base, candidate, plan, end)
    assert result['changed_keys'] == ['splats.sh0']
    assert result['historical_absent_fields_still_absent'] == ['manifest_sha256', 'pixel_protocol']
    assert 'optimizers' not in candidate and 'numpy_rng' not in candidate


@pytest.mark.parametrize('mutation', ['geometry', 'camera', 'invent_profile', 'old_optimizer', 'wrong_kind'])
def test_forbidden_endpoint_mutations_rejected(mutation):
    base, candidate, plan, end = fixture_checkpoint()
    if mutation == 'geometry':
        candidate['model']['splats.means'][0, 0] += .1
    elif mutation == 'camera':
        candidate['training_cameras'][0, 0, 0] += .1
    elif mutation == 'invent_profile':
        candidate['pixel_protocol'] = 'legacy_mixed_v1'
    elif mutation == 'old_optimizer':
        candidate['optimizers'] = {}
    else:
        candidate['checkpoint_kind'] = 'ordinary'
    with pytest.raises(ValueError):
        audit.compare_checkpoint(base, candidate, plan, end)


def test_unfinished_launch_rejects_before_any_checkpoint_load(tmp_path, monkeypatch):
    out = tmp_path/'run'; out.mkdir()
    plan_path = tmp_path/'plan.json'
    plan_path.write_text(json.dumps({'output': str(out)}))
    plan_sha = audit.sha(plan_path)
    (out/'execution_receipt.json').write_text(json.dumps({'status': 'completed', 'plan_sha256': plan_sha}))
    launch = tmp_path/'launch.json'
    launch.write_text(json.dumps({'status': 'running', 'plan_sha256': plan_sha}))
    def forbid(*args, **kwargs):
        raise AssertionError('checkpoint touched before completed gate')
    monkeypatch.setattr(torch, 'load', forbid)
    with pytest.raises(ValueError, match='Natural outer exit0'):
        audit.audit(plan_path, plan_sha, launch, tmp_path/'audit.json')
    assert not torch.cuda.is_initialized()


def protocol_fixture(fp64):
    name = 'run_fullbatch_appearance_fp64.py' if fp64 else 'run_fullbatch_appearance_v2.py'
    source = (Path(__file__).parents[1]/'scripts'/name).read_text()
    node = next(x for x in ast.parse(source).body if isinstance(x, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'SPEC' for t in x.targets))
    numerical = ast.literal_eval(node.value)
    contract = audit.PROTOCOLS[numerical['protocol']]
    return {'specification': numerical, 'status': contract['status'],
            'protocol_document': {'path': contract['document']},
            'source_hashes': {'bridge_rgs/fullbatch_appearance_fp64.py': audit.FP64_HELPER_SHA}}


@pytest.mark.parametrize('fp64', [False, True])
def test_exact_separate_precision_contracts(fp64):
    plan = protocol_fixture(fp64)
    assert audit.validate_protocol(plan)['objective_dtype'] == ('float64' if fp64 else 'float32')
    plan['specification']['optimization_seconds'] += 1
    with pytest.raises(ValueError, match='specification'):
        audit.validate_protocol(plan)


def test_fp64_cannot_reuse_old_gate_or_different_helper():
    plan = protocol_fixture(True)
    contract = audit.validate_protocol(plan)
    old = {'specification': {'protocol': audit.PROTOCOLS['conditional_fullbatch_appearance_h3_v2']['diagnostic_protocol']}}
    receipt = {'status': 'completed', 'technical_gate_passed': True, 'plan_sha256': 'bound'}
    with pytest.raises(ValueError, match='different precision'):
        audit.validate_diagnostic_precision(contract, old, receipt, 'bound')
    new = {'specification': {'protocol': contract['diagnostic_protocol']}}
    receipt['technical_gate_passed'] = False
    with pytest.raises(ValueError, match='failed diagnostic'):
        audit.validate_diagnostic_precision(contract, new, receipt, 'bound')
    plan['source_hashes']['bridge_rgs/fullbatch_appearance_fp64.py'] = 'other'
    with pytest.raises(ValueError, match='implementation'):
        audit.validate_protocol(plan)
