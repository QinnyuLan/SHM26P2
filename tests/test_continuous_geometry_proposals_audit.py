"""Independent checker corruption examples, using synthetic records only."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PATH = HERE/'audit_continuous_geometry_proposals.py'
if not PATH.exists():
    PATH = HERE.parent/'scripts/audit_continuous_geometry_proposals.py'
spec = importlib.util.spec_from_file_location('independent_geometry_checker', PATH)
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def record(names, ce=1., rgb=1., means='0'*64, cm=False):
    rows = [{'name': n, 'rgb_mse': rgb, 'raw_ce': ce, 'rgb_pixels': 3, 'semantic_pixels': 2} for n in names]
    result = {'complete': True, 'names': names, 'rows': rows, 'rgb_mse': rgb, 'raw_ce': ce, 'means_sha256': means}
    if cm:
        matrix = np.zeros((5, 5), np.int64); matrix[0, 0] = 1; matrix[2, 2] = 1
        for row in rows:
            row['confusion_matrix'] = matrix.tolist()
        result.update(confusion_matrix=(matrix*len(names)).tolist(), iou=[1., None, 1., None, None], miou_all=1.)
    return result


def attempt(index, names, before, after, accepted, already_accepted):
    gates = {'semantic_decreases': after['raw_ce'] < before['raw_ce'],
             'batch_rgb_protected': after['rgb_mse'] <= before['rgb_mse']+max(1e-6*before['rgb_mse'], 1e-12),
             'every_view_rgb_protected': all(b['rgb_mse'] <= 1.001*a['rgb_mse']
                 for a, b in zip(before['rows'], after['rows'], strict=True))}
    return {'attempt': index, 'round': 0, 'names': names, 'baseline': before, 'candidate': after,
        'accepted': accepted, 'accepted_optimizer_steps': already_accepted+int(accepted),
        'acceptance': {'criteria': gates, 'control_finite_commit': False, 'rgb_tolerance': max(1e-6*before['rgb_mse'], 1e-12),
                      'candidate_minus_baseline_rgb': after['rgb_mse']-before['rgb_mse'],
                      'candidate_minus_baseline_semantic': after['raw_ce']-before['raw_ce']},
        'proposal': {'mode': 'semantic', 'optimizer_proposed_step': already_accepted+1, 'adam_betas': [.9, .999],
                     'adam_eps': 1e-15, 'weight_decay': 0., 'first_order_RGB_guarantee': False,
                     'learning_rate': 1.6e-6, 'mahalanobis_cap': 1/64, 'actual_mahalanobis_max': .01,
                     'adam_proposed_nonzero_point_count': 1, 'scaled_point_count': 0,
                     'fp32_overcap_restored_point_count': 0, 'cap_rounded_to_before_point_count': 0,
                     'actual_nonzero_point_count': 1, 'actual_nonzero_coordinate_count': 1,
                     'rgb_dot_actual_delta': .001, 'semantic_dot_actual_delta': -.1}}


def example():
    names = [str(j) for j in range(7)]
    original = record(names, cm=True)
    before = record(names)
    rejected = record(names, means='1'*64)
    accepted = record(names, ce=.9, means='2'*64)
    rows = [attempt(1, names, before, rejected, False, 0), attempt(2, names, before, accepted, True, 0)]
    batches = [{'round': 0, 'names': names}]*2
    support = {r['name']: r for r in original['rows']}
    return rows, batches, support


def test_actual_gate_and_rejected_means_adam_hash_chain():
    rows, batches, support = example()
    result = audit.check_trace(rows, batches, 'trust', '0'*64, support, 2, 1.)
    assert result['accepted'] == result['rejected'] == 1
    assert result['last_means_sha256'] == '2'*64
    assert result['sum_accepted_recorded_max_step'] == .01
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('failure', ['gate', 'rollback_hash', 'adam_step', 'cap', 'dot'])
def test_corrupt_trace_is_rejected(failure):
    rows, batches, support = example()
    if failure == 'gate':
        rows[0]['accepted'] = True
    elif failure == 'rollback_hash':
        rows[1]['baseline']['means_sha256'] = '1'*64
    elif failure == 'adam_step':
        rows[1]['proposal']['optimizer_proposed_step'] = 2
    elif failure == 'cap':
        rows[0]['proposal']['actual_mahalanobis_max'] = 1/64+1e-10
    else:
        rows[0]['proposal']['rgb_dot_actual_delta'] = float('nan')
    with pytest.raises(ValueError):
        audit.check_trace(rows, batches, 'trust', '0'*64, support, 2, 1.)


def test_equal_view_loss_and_class_support_not_pooled_pixels():
    r = record(['a', 'b'], cm=True)
    r['rows'][0]['rgb_mse'] = .2; r['rows'][1]['rgb_mse'] = .8; r['rgb_mse'] = .5
    r['rows'][1]['rgb_pixels'] = 300
    result = audit.check_rows(r, ['a', 'b'], cm=True)
    assert result['rgb_mse'] == .5 and result['cable_iou'] == 1.
    wrong = copy.deepcopy(r); wrong['rgb_mse'] = (.2*3+.8*300)/303
    with pytest.raises(ValueError, match='Equal-view'):
        audit.check_rows(wrong, ['a', 'b'], cm=True)
    wrong = copy.deepcopy(r); wrong['rows'][0]['confusion_matrix'][2][2] = 0
    with pytest.raises(ValueError, match='confusion'):
        audit.check_rows(wrong, ['a', 'b'], cm=True)


def test_independent_rotated_anisotropic_cumulative_metric():
    base = np.zeros((1, 3), np.float32); endpoint = np.array([[.125, 0, 0]], np.float32)
    q = np.array([[.5, .5, .5, .5]])  # local z maps to world x.
    delta, radius, distance = audit.cumulative_displacement(base, endpoint, q, np.log([[2., 4., 8.]]))
    assert np.array_equal(delta, endpoint.astype(np.float64))
    assert radius == pytest.approx(1/64) and distance == .125


def test_schedule_fixed_four_rounds_without_global_rng_mutation():
    names = [f'{j:03d}.png' for j in range(259)]
    before = audit.random.getstate(); rows = audit.schedule(names)
    assert audit.random.getstate() == before and len(rows) == 148
    for epoch in range(4):
        assert sorted(n for r in rows if r['round'] == epoch for n in r['names']) == names


def test_natural_completion_checked_before_endpoint_or_checkpoint_load(tmp_path, monkeypatch):
    import torch
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Premature checkpoint/endpoint read'))
    plan = tmp_path/'plan.json'; plan.write_text(json.dumps({'output': str(tmp_path)}))
    (tmp_path/'launch_receipt.json').write_text(json.dumps({'status': 'running'}))
    (tmp_path/'execution_receipt.json').write_text(json.dumps({'status': 'running'}))
    with pytest.raises(ValueError, match='Natural completed'):
        audit.audit(plan, audit.sha(plan))
