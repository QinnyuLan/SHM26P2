"""Synthetic CPU contracts only; never touch real run artifacts or pixels."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import ndtr

HERE = Path(__file__).resolve()
SCRIPT = HERE.with_name('audit_partition_mass_arrangement.py')
if not SCRIPT.exists():
    SCRIPT = HERE.parents[1]/'scripts/audit_partition_mass_arrangement.py'
spec = importlib.util.spec_from_file_location('partition_mass_audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_perpendicular_ties_parallel_basis_and_sign():
    normal = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]], dtype=torch.float64)
    normal /= normal.norm(dim=-1, keepdim=True)
    result = audit.perpendicular_normal(normal)
    torch.testing.assert_close(result.norm(dim=-1), torch.ones(4, dtype=torch.float64))
    torch.testing.assert_close((normal*result).sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-15, rtol=0)
    assert torch.equal(result[0], torch.tensor([0., 0., 1.], dtype=torch.float64))
    assert torch.equal(result[1], torch.tensor([0., 0., -1.], dtype=torch.float64))
    assert torch.equal(audit.perpendicular_normal(-normal), -result)
    with pytest.raises(ValueError, match='unit'):
        audit.perpendicular_normal(normal*2)


def test_qbar_is_current_endpoints_analytic_prior_not_old_classifier():
    from bridge_rgs.partition_field import SemanticPartitionField
    from bridge_rgs.semantic_partition import normal_interval_probability
    field = SemanticPartitionField(torch.zeros(3, 5), mode='integrated')
    with torch.no_grad():
        field.inside_logits.copy_(torch.tensor([[2., -1., 0., 1., -2.]]).expand(3, -1))
        field.outside_logits.copy_(torch.tensor([[-1., 1., 2., -2., 0.]]).expand(3, -1))
        field.offset_raw.copy_(torch.tensor([0., .4, -.6]))
        field.width_raw.copy_(torch.tensor([-.8, .2, 1.]))
        field.direction.copy_(torch.tensor([[1., 2., 3.], [0., 1., 0.], [-2., 1., 1.]]))
    before = copy.deepcopy(field.state_dict())
    choices, diagnostics = audit.readouts(field, normal_interval_probability)
    qi, qo = field.endpoints()
    _, b, w, tau = field.slab_parameters()
    b, w, tau = (x.detach().double().numpy() for x in (b, w, tau))
    occupancy = ndtr((-b+w)/np.sqrt(1+tau*tau))-ndtr((-b-w)/np.sqrt(1+tau*tau))
    expected = (occupancy[:, None]*qi.detach().double().numpy()+(1-occupancy[:, None])*qo.detach().double().numpy()).astype(np.float32)
    qbar = choices['own_qbar'].endpoints()[0]
    np.testing.assert_array_equal(qbar.numpy(), expected)
    assert torch.equal(qbar, choices['own_qbar'].endpoints()[1])
    assert diagnostics['same_endpoints_b_w_tau'] is True
    assert diagnostics['qbar_cast_max_error'] < 3e-8
    assert diagnostics['perpendicular_dot_absmax'] < 1e-7
    assert not torch.allclose(qbar, torch.full_like(qbar, .2))
    assert all(torch.equal(v, before[k]) for k, v in field.state_dict().items())
    assert all(not t.requires_grad for choice in choices.values() for t in (*choice.endpoints(), *choice.slab_parameters()))
    json.dumps(diagnostics, allow_nan=False)


def test_constant_endpoints_remove_gate_but_preserve_unconditional_mass():
    from bridge_rgs.partition_field import SemanticPartitionField
    from bridge_rgs.semantic_partition import normal_interval_probability
    torch.manual_seed(31)
    field = SemanticPartitionField(torch.randn(2, 5))
    with torch.no_grad():
        field.outside_logits.add_(torch.randn(2, 5))
    choices, _ = audit.readouts(field, normal_interval_probability)
    qi, qo = choices['own_qbar'].endpoints()
    for gate in (torch.zeros(2, 1), torch.ones(2, 1), torch.tensor([[.2], [.8]])):
        torch.testing.assert_close(gate*qi+(1-gate)*qo, qi, rtol=0, atol=3e-8)
    assert choices['own_qbar'].mode == choices['perpendicular'].mode == 'integrated'


def test_score_same_known_valid_support_and_weighted_pixel_denominator():
    prob = np.array([[[.7, .1, .1, .05, .05], [.1, .2, .3, .2, .2], [.2]*5],
                     [[0., .1, .7, .1, .1], [.2]*5, [.2]*5]], dtype=np.float32)
    mask = np.array([[0, 2, 255], [0, 4, 1]], np.uint8)
    valid = np.array([[1, 1, 1], [1, 0, 0]], np.uint8)
    result = audit.score(prob, mask, valid)
    y = np.array([0, 2, 0]); p = prob.reshape(-1, 5)[[0, 1, 3]].astype(np.float64)
    weights = np.asarray(audit.WEIGHTS, np.float32).astype(np.float64)
    expected = (-np.log(np.maximum(p[np.arange(3), y], 1e-7))*weights[y]).sum()/3
    assert result['weighted_ce'] == pytest.approx(expected, rel=0, abs=1e-15)
    assert result['brier'] == pytest.approx(np.square(p-np.eye(5)[y]).sum()/3)
    assert result['pixels'] == 3 and np.asarray(result['confusion_matrix']).sum() == 3
    assert result['confusion_matrix'][0][2] == 1
    with pytest.raises(ValueError, match='Empty'):
        audit.score(prob, mask, np.zeros_like(valid))
    invalid = prob.copy(); invalid[0, 0] *= .8
    with pytest.raises(ValueError, match='probabilities'):
        audit.score(invalid, mask, valid)


def synthetic_rows():
    keys = ['old_p3d']+[case+'.'+kind for case in audit.CASES for kind in ('raw', 'scene')]
    return [{'name': name, 'scores': {key: {'weighted_ce': .5+i/100, 'brier': .4+i/100,
                    'confusion_matrix': (np.eye(5, dtype=int)*(i+1)).tolist()} for key in keys}}
            for i, name in enumerate(audit.NAMES)]


def test_paired_view_bootstrap_sign_count_weighted_cm_and_missing_classes():
    rows = synthetic_rows()
    for row in rows:
        row['scores']['own_qbar.raw']['weighted_ce'] += .2
        row['scores']['own_qbar.raw']['brier'] -= .1
    result = audit.summarize(rows)
    pair = result['paired_differences']['own_qbar.raw minus learned.raw']
    assert pair['weighted_ce']['difference'] == pytest.approx(.2)
    assert pair['weighted_ce']['paired_camera_95_interval'] == pytest.approx([.2, .2])
    assert pair['brier']['paired_camera_95_interval'] == pytest.approx([-.1, -.1])
    assert pair['miou_all']['difference'] == 0
    assert np.asarray(result['aggregate']['learned.raw']['confusion_matrix'])[0, 0] == sum(range(1, 17))
    for row in rows:
        for value in row['scores'].values():
            value['confusion_matrix'][-1][-1] = 0
    missing = audit.summarize(rows)
    assert missing['paired_differences']['own_qbar.raw minus learned.raw']['foundation']['difference'] is None
    assert missing['adoption_gate'] is None
    json.dumps(missing, allow_nan=False)


def test_barrier_rejects_before_target_reader_even_import(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, 'imread', lambda *_: pytest.fail('Target read before barrier'))
    with pytest.raises(ValueError, match='All predictions'):
        audit.score_saved({'targets': []}, [], 47)


def fake_producer(tmp_path):
    folder = tmp_path/'producer'; (folder/'integrated').mkdir(parents=True)
    plan = {'specification': {'candidate': 'integrated', 'steps': 2000, 'class_weights_fp32': audit.WEIGHTS}}
    audit.write(folder/'plan.json', plan)
    digest = audit.sha(folder/'plan.json')
    stage = {'status': 'completed', 'arm': 'integrated', 'steps': 2000,
             'delta_path': str(folder/'integrated/final_delta.pt'), 'delta_sha256': 'not yet read'}
    audit.write(folder/'integrated/training_receipt.json', stage)
    execution = {'status': 'completed', 'bound_sources_inputs_unchanged': True, 'plan_sha256': digest, 'training': [stage]}
    audit.write(folder/'execution_receipt.json', execution)
    audit.write(folder/'launch_receipt.json', {'status': 'completed', 'exit_code': 0, 'natural_completion': True,
                 'plan_sha256': digest, 'execution_receipt_sha256': audit.sha(folder/'execution_receipt.json')})
    audit.write(folder/'independent_cpu_review.json', {'status': 'passed', 'plan_sha256': digest,
                 'inputs_sources_outputs_unchanged': True, 'evaluation': {'adoption_passed': False}})
    return folder


def test_completion_does_not_require_performance_success_or_read_endpoint(tmp_path):
    folder = fake_producer(tmp_path)
    # Missing delta is intentional: the completion check must not open it.
    _, stage = audit.completed_producer(folder)
    assert stage['arm'] == 'integrated'
    launch = audit.read(folder/'launch_receipt.json'); launch['exit_code'] = 1
    audit.write(folder/'launch_receipt.json', launch, replace=True)
    with pytest.raises(ValueError, match='naturally'):
        audit.completed_producer(folder)


def test_failed_audit_blocks_prepare_before_endpoint_hash(tmp_path):
    folder = fake_producer(tmp_path)
    report = audit.read(folder/'independent_cpu_review.json'); report['status'] = 'failed'
    audit.write(folder/'independent_cpu_review.json', report, replace=True)
    with pytest.raises(ValueError, match='independent audit'):
        audit.completed_producer(folder)


def test_fixed_budget_and_no_automatic_gate_or_execution():
    assert audit.SPEC['scene_calls'] == 3*len(audit.NAMES) == 48
    assert audit.SPEC['teacher_calls'] == audit.SPEC['optimizer_steps'] == audit.SPEC['backwards'] == 0
    assert audit.SPEC['selection_or_adoption_gate'] is None
    assert 'internal_seconds' not in audit.SPEC
    assert audit.CASES == ('learned', 'own_qbar', 'perpendicular')
    assert audit.PRODUCER.name == 'semantic_partition_matched_v2'
