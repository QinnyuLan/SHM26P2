"""Actual draft controller/reduction with synthetic CPU objectives only."""
import copy
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from bridge_rgs.continuous_geometry import MeansTransaction

HERE = Path(__file__).resolve().parent
SCRIPT = HERE/'train_continuous_semantic_geometry.py'
if not SCRIPT.exists():
    SCRIPT = HERE.parent/'scripts/train_continuous_semantic_geometry.py'
spec = importlib.util.spec_from_file_location('continuous_train_test', SCRIPT)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

NAMES = [f'view{i:03d}' for i in range(259)]


def counters():
    return {'means_vjp': 0, 'training_view_pairs': 0, 'description_view_pairs': 0}


def transaction():
    means = torch.nn.Parameter(torch.zeros(1, 3))
    return MeansTransaction(means, torch.tensor([[1., 0, 0, 0]]), torch.zeros(1, 3), 1.)


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    else:
        assert left == right


def batch(rgb, ce):
    rgb = np.broadcast_to(rgb, (7,)); ce = np.broadcast_to(ce, (7,))
    rows = [{'name': name, 'rgb_mse': float(r), 'raw_ce': float(s)}
            for name, r, s in zip(NAMES[:7], rgb, ce, strict=True)]
    return {'complete': True, 'names': NAMES[:7], 'rows': rows,
            'rgb_mse': float(np.mean(rgb)), 'raw_ce': float(np.mean(ce))}


class ToyObjectives:
    """Quadratic RGB and Bernoulli CE, with independent graphs and varying supports."""
    def __init__(self, tx, rgb_target=-1.):
        self.tx, self.rgb_target = tx, rgb_target
        self.calls, self.counts = [], counters()

    def __call__(self, names, *, gradient):
        self.calls.append((list(names), gradient, self.tx.means.detach().clone()))
        def view(name):
            index = NAMES.index(name)
            rgb = (self.tx.means[0, 0].double()-self.rgb_target).square()
            ce = torch.nn.functional.softplus(self.tx.means[0, 0].double())
            return rgb, ce, {'rgb_pixels': index+1, 'semantic_pixels': index % 11+1}
        return worker.evaluate_view_objectives(names, self.tx.means, view, self.counts, gradient=gradient)


def test_fixed_148_schedule_has_four_complete_rounds_and_does_not_consume_global_rng():
    state = random.getstate()
    schedule = worker.batch_schedule(NAMES)
    assert random.getstate() == state and schedule == worker.batch_schedule(NAMES)
    assert len(schedule) == 148 and all(len(row['names']) == 7 for row in schedule)
    # Fixed Python Random(42) first seven indices, retaining RNG between rounds.
    heads = [[238, 9, 103, 60, 5, 79, 236], [226, 256, 58, 127, 29, 92, 79],
             [78, 70, 4, 14, 124, 174, 122], [228, 194, 186, 112, 58, 251, 219]]
    for epoch in range(4):
        rows = schedule[37*epoch:37*(epoch+1)]
        assert {r['round'] for r in rows} == {epoch}
        assert sorted(n for r in rows for n in r['names']) == NAMES
        assert rows[0]['names'] == [NAMES[i] for i in heads[epoch]]
    for invalid in (NAMES[::-1], NAMES[:-1], NAMES[:-1]+[NAMES[0]]):
        with pytest.raises(ValueError, match='sorted259'):
            worker.batch_schedule(invalid)


def test_actual_reduction_is_equal_view_not_pooled_pixels_for_losses_and_gradients():
    means = torch.nn.Parameter(torch.tensor([[.25, 0, 0]]))
    rgb_targets = [np.linspace(-.8, .2+i/5, 3*(i+1)).reshape(-1, 3) for i in range(7)]
    labels = [np.arange(2*i+1) % 2 for i in range(7)]
    seen = []
    def view(name):
        i = NAMES.index(name); seen.append(name)
        rgb = (means[0, 0].double()-torch.from_numpy(rgb_targets[i])).square().mean()
        # Binary logistic CE has a closed-form independent NumPy gradient.
        z = means[0, 0].double()
        ce = (torch.nn.functional.softplus(z)-torch.from_numpy(labels[i])*z).mean()
        return rgb, ce, {'rgb_pixels': i+1, 'semantic_pixels': 2*i+1}
    counts = counters()
    actual = worker.evaluate_view_objectives(NAMES[:7], means, view, counts, gradient=True)
    rgb_values = np.array([np.mean((.25-t)**2) for t in rgb_targets])
    ce_values = np.array([np.logaddexp(0, .25)-np.mean(y)*.25 for y in labels])
    rgb_grad = np.array([2*np.mean(.25-t) for t in rgb_targets], np.float32).astype(np.float64).mean()
    ce_grad = np.array([1/(1+np.exp(-.25))-np.mean(y) for y in labels], np.float32).astype(np.float64).mean()
    assert seen == NAMES[:7] and counts == {'means_vjp': 14, 'training_view_pairs': 7, 'description_view_pairs': 0}
    assert actual['rgb_mse'] == pytest.approx(rgb_values.mean(), abs=1e-15)
    assert actual['raw_ce'] == pytest.approx(ce_values.mean(), abs=1e-15)
    assert actual['rgb_mse'] != pytest.approx(np.average(rgb_values, weights=np.arange(1, 8)))
    assert actual['raw_ce'] != pytest.approx(np.average(ce_values, weights=np.arange(1, 14, 2)))
    np.testing.assert_allclose(actual['g_rgb'].numpy(), [[rgb_grad, 0, 0]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(actual['g_semantic'].numpy(), [[ce_grad, 0, 0]], rtol=0, atol=1e-15)
    assert actual['g_rgb'].dtype == actual['g_semantic'].dtype == torch.float64
    assert means.grad is None and torch.equal(means, torch.tensor([[.25, 0, 0]]))


def test_each_fp32_vjp_is_promoted_before_accumulation():
    means = torch.nn.Parameter(torch.zeros(1, 3))
    slopes = [2**24, 1, -(2**24), 3, 4, 5, 6]
    def view(name):
        slope = slopes[NAMES.index(name)]
        return (100+means[0, 0].double()*slope, 100-means[0, 0].double()*slope,
                {'rgb_pixels': 1, 'semantic_pixels': 1})
    result = worker.evaluate_view_objectives(NAMES[:7], means, view, counters(), gradient=True)
    assert float(result['g_rgb'][0, 0]) == 19/7
    assert float(result['g_semantic'][0, 0]) == -19/7


@pytest.mark.parametrize('case', ['ce_equal', 'batch_rgb', 'single_view_rgb'])
def test_trust_rejects_each_independent_condition_without_rejecting_finite_controls(case):
    baseline = batch(1., 1.)
    if case == 'ce_equal':
        candidate, failed = batch(1., 1.), 'semantic_decreases'
    elif case == 'batch_rgb':
        candidate, failed = batch(1+2e-6, .9), 'batch_rgb_protected'
    else:
        candidate, failed = batch([1.002, .998, 1, 1, 1, 1, 1], .9), 'every_view_rgb_protected'
    accepted, details = worker.accept_candidate('trust', baseline, candidate)
    assert not accepted and not details['criteria'][failed]
    assert sum(not v for v in details['criteria'].values()) == 1
    for arm in ('joint', 'pcgrad'):
        assert worker.accept_candidate(arm, baseline, candidate)[0]


def test_trust_exact_boundaries_relative_and_absolute_rgb_allowance():
    baseline = batch(1., 1.)
    accepted, details = worker.accept_candidate('trust', baseline, batch(1+1e-6, .9))
    assert accepted and details['rgb_tolerance'] == 1e-6
    # Absolute allowance dominates at tiny MSE, while per-view guard still applies.
    baseline = batch(1e-9, 1.)
    assert worker.accept_candidate('trust', baseline, batch(1.0005e-9, .9))[0]
    assert worker.accept_candidate('trust', baseline, batch(1.0005e-9, .9))[1]['rgb_tolerance'] == 1e-12
    assert not worker.accept_candidate('trust', batch(0., 1.), batch(1e-13, .9))[0]


def test_all_fixed_attempts_use_baseline_candidate_and_real_adam_without_extra_evaluations():
    tx = transaction(); objective = ToyObjectives(tx); schedule = worker.batch_schedule(NAMES)
    emitted = []
    result = worker.run_attempts('joint', schedule, tx, objective, emitted.append)
    assert result['attempts'] == result['accepted'] == 148 and result['rejected'] == 0
    assert result['stop_reason'] == 'fixed_148_attempt_schedule_finished' and not result['converged_claimed']
    assert len(objective.calls) == 296 and len(emitted) == 148
    assert objective.counts == {'means_vjp': 2072, 'training_view_pairs': 2072, 'description_view_pairs': 0}
    for i, row in enumerate(emitted):
        before, after = objective.calls[2*i:2*i+2]
        assert before[:2] == (schedule[i]['names'], True)
        assert after[:2] == (schedule[i]['names'], False)
        assert row['attempt'] == row['accepted_optimizer_steps'] == row['proposal']['optimizer_proposed_step'] == i+1
        # Independent linearization of the toy quadratic at the actual FP32 step.
        x = float(before[2][0, 0]); delta = float((after[2].double()-before[2].double())[0, 0])
        rgb_g = float(np.float32(2*(x+1)))
        assert row['proposal']['rgb_dot_actual_delta'] == pytest.approx(rgb_g*delta, abs=1e-15)
    assert int(tx.optimizer.state[tx.means]['step']) == 148 and tx.pending is None
    json.dumps(result, allow_nan=False)


def test_actual_semantic_improvement_is_rejected_when_rgb_worsens_and_adam_restores():
    tx = transaction(); objective = ToyObjectives(tx, rgb_target=1.)
    before = tx.means.detach().clone()
    result = worker.run_attempts('trust', worker.batch_schedule(NAMES)[:1], tx, objective)
    row = result['history'][0]
    assert row['candidate']['raw_ce'] < row['baseline']['raw_ce']
    assert row['candidate']['rgb_mse'] > row['baseline']['rgb_mse']
    assert result['accepted'] == 0 and result['rejected'] == 1
    assert torch.equal(tx.means, before) and tx.optimizer.state_dict()['state'] == {}
    assert tx.pending is None and tx.means.grad is None


@pytest.mark.parametrize('failure', ['exception', 'partial', 'bad_mean', 'nan'])
def test_candidate_failure_rolls_back_existing_moments_and_does_not_emit_completed_attempt(failure):
    tx = transaction(); objective = ToyObjectives(tx)
    schedule = worker.batch_schedule(NAMES)[:1]
    worker.run_attempts('joint', schedule, tx, objective)
    before, state = tx.means.detach().clone(), copy.deepcopy(tx.optimizer.state_dict())
    emitted = []
    def evaluate(names, *, gradient):
        if not gradient and failure == 'exception':
            raise RuntimeError('Synthetic interrupted candidate')
        result = objective(names, gradient=gradient)
        if not gradient:
            if failure == 'partial':
                result['complete'] = False
            elif failure == 'bad_mean':
                result['rgb_mse'] += .1
            elif failure == 'nan':
                result['rows'][0]['raw_ce'] = float('nan')
        return result
    with pytest.raises((RuntimeError, ValueError)):
        worker.run_attempts('joint', schedule, tx, evaluate, emitted.append)
    assert emitted == [] and torch.equal(tx.means, before)
    assert_state_equal(tx.optimizer.state_dict(), state)
    assert tx.pending is None and tx.means.grad is None


def test_failed_numerical_preflight_blocks_prepare_before_any_output(monkeypatch, tmp_path):
    output = Path('/mnt/data/continuous_geometry_synthetic_prepare_must_not_create')
    assert not output.exists()
    records = {'plan.json': {}, 'execution_receipt.json': {
        'status': 'completed', 'numerical_status': 'not_passed', 'main_count': 12,
        'main_passed': 3, 'main_measurable': 12}, 'launch_receipt.json': {
        'status': 'completed', 'natural_completion': True, 'exit_code': 0}}
    monkeypatch.setattr(worker, 'read', lambda path: records[Path(path).name])
    with pytest.raises(ValueError, match='preflight must pass'):
        worker.prepare(output, tmp_path)
    assert not output.exists()
