"""Real route/controller contracts with synthetic CPU graphs, no scene or pixels."""
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
SCRIPT = HERE/'train_scene_geometry_routes.py'
if not SCRIPT.exists():
    SCRIPT = HERE.parent/'scripts/train_scene_geometry_routes.py'
spec = importlib.util.spec_from_file_location('scene_routes_test', SCRIPT)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
NAMES = [f'view{i:03d}' for i in range(259)]


def counters():
    return {'means_vjp': 0, 'training_view_pairs': 0, 'description_view_pairs': 0}


def transaction():
    return MeansTransaction(torch.nn.Parameter(torch.zeros(1, 3)),
                            torch.tensor([[1., 0, 0, 0]]), torch.zeros(1, 3), 1.)


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_state_equal(x, y)
    else:
        assert a == b


def test_schedule_fixed_complete_and_global_rng_untouched():
    rng = random.getstate()
    schedule = worker.batch_schedule(NAMES)
    assert random.getstate() == rng and schedule == worker.batch_schedule(NAMES)
    assert len(schedule) == 148 and all(len(b['names']) == 7 for b in schedule)
    heads = [[238, 9, 103, 60, 5, 79, 236], [226, 256, 58, 127, 29, 92, 79],
             [78, 70, 4, 14, 124, 174, 122], [228, 194, 186, 112, 58, 251, 219]]
    for i, first in enumerate(heads):
        rows = schedule[i*37:(i+1)*37]
        assert {b['round'] for b in rows} == {i}
        assert sorted(n for b in rows for n in b['names']) == NAMES
        assert rows[0]['names'] == [NAMES[j] for j in first]
    for bad in (NAMES[::-1], NAMES[:-1], NAMES[:-1]+[NAMES[0]]):
        with pytest.raises(ValueError, match='sorted259'):
            worker.batch_schedule(bad)


def test_shared_graph_three_vjps_equal_view_not_pixel_weighted():
    means = torch.nn.Parameter(torch.tensor([[.25, 0, 0]]))
    targets = [np.linspace(-.8, .2+i/5, 3*(i+1)).reshape(-1, 3) for i in range(7)]
    labels = [np.arange(2*i+1) % 2 for i in range(7)]
    seen = []

    def view(name):
        i = NAMES.index(name); seen.append(name)
        x = means[0, 0].double(); z = x.square()+x
        rgb = (z-torch.from_numpy(targets[i])).square().mean()
        prior = (torch.nn.functional.softplus(z)-torch.from_numpy(labels[i])*z).mean()
        evidence = .1*(2*z+1).square()
        return rgb, prior+evidence, prior+evidence.detach(), {
            'raw_ce': float(prior.detach()), 'rgb_pixels': i+1, 'semantic_pixels': 2*i+1}

    counts = counters()
    result = worker.evaluate_view_objectives(NAMES[:7], means, view, counts, gradient=True)
    x = .25; z = x*x+x; dz = 2*x+1
    rgb = np.array([np.mean((z-t)**2) for t in targets])
    prior = np.array([np.logaddexp(0, z)-y.mean()*z for y in labels])
    scene = prior+.1*(2*z+1)**2
    gr = np.array([2*np.mean(z-t)*dz for t in targets], np.float32).astype(np.float64).mean()
    gp_rows = np.array([(1/(1+np.exp(-z))-y.mean())*dz for y in labels])
    gp = gp_rows.astype(np.float32).astype(np.float64).mean()
    gf = (gp_rows+.4*(2*z+1)*dz).astype(np.float32).astype(np.float64).mean()
    assert seen == NAMES[:7]
    assert counts == {'means_vjp': 21, 'training_view_pairs': 7, 'description_view_pairs': 0}
    for key, values in [('rgb_mse', rgb), ('raw_ce', prior), ('scene_ce', scene)]:
        assert result[key] == pytest.approx(values.mean(), abs=1e-15)
    assert result['rgb_mse'] != pytest.approx(np.average(rgb, weights=np.arange(1, 8)))
    assert result['scene_ce'] != pytest.approx(np.average(scene, weights=np.arange(1, 14, 2)))
    for key, expected in [('g_rgb', gr), ('g_full', gf), ('g_prior', gp)]:
        assert result[key].dtype == torch.float64
        np.testing.assert_allclose(result[key].numpy(), [[expected, 0, 0]], atol=1e-15, rtol=0)
    summary = result['path_gradients']
    assert summary['l2']['context'] == pytest.approx(abs(gf-gp), abs=1e-15)
    assert summary['dots']['prior_context'] == pytest.approx(gp*(gf-gp), abs=1e-15)
    assert means.grad is None and torch.equal(means, torch.tensor([[.25, 0, 0]]))


def test_fp32_vjps_promoted_before_sum_and_forward_mismatch_refused():
    means = torch.nn.Parameter(torch.zeros(1, 3))
    slopes = [2**24, 1, -(2**24), 3, 4, 5, 6]

    def view(name):
        z = means[0, 0].double()*slopes[NAMES.index(name)]
        return 100+z, 100+2*z, 100-z+3*z.detach(), {'raw_ce': 100.}

    result = worker.evaluate_view_objectives(NAMES[:7], means, view, counters(), gradient=True)
    assert float(result['g_rgb'][0, 0]) == 19/7
    assert float(result['g_full'][0, 0]) == 38/7
    assert float(result['g_prior'][0, 0]) == -19/7
    counts = counters()

    def mismatch(name):
        rgb, full, prior, meta = view(name)
        return rgb, full, prior+1e-4, meta

    with pytest.raises(ValueError, match='forward differs'):
        worker.evaluate_view_objectives(NAMES[:7], means, mismatch, counts, gradient=True)
    assert counts == counters()


def test_prior_readout_stops_five_context_routes_but_both_prior_paths_are_live():
    weights = torch.tensor([.4, -.3, .7, .1, -.8], dtype=torch.float64)
    tail = torch.tensor([.2, .4, -.5, .7, -.1], dtype=torch.float64)

    class ToyHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(weights.clone(), requires_grad=False)
            self.route_flags = []

        def forward(self, features, rgb, depth, alpha, *, p3d, depth_moments, geometry_grad):
            assert geometry_grad is True
            contexts = [features, rgb, depth, alpha, depth_moments]
            self.route_flags.append([v.requires_grad for v in contexts]+[p3d.requires_grad])
            evidence = sum((i+1)*v.mean() for i, v in enumerate(contexts))
            entropy = -(p3d*p3d.log()).sum(-1, keepdim=True)
            return evidence*self.weights+.7*p3d+.2*entropy*tail

    keys = ['features', 'rgb', 'depth', 'alpha', 'depth_moments']
    context = {k: torch.tensor([.05*(i+1)], dtype=torch.float64, requires_grad=True)
               for i, k in enumerate(keys)}
    p = torch.tensor([.1, .2, .3, .15, .25], dtype=torch.float64, requires_grad=True)
    context['p3d'] = p; head = ToyHead()
    residual = head(*(context[k] for k in keys[:4]), p3d=p,
                    depth_moments=context['depth_moments'], geometry_grad=True)
    full = (p.log()+residual).softmax(-1)
    prior = worker.prior_readout(head, context)
    assert torch.equal(full, prior)
    assert head.route_flags == [[True]*6, [False]*5+[True]]
    inputs = [context[k] for k in keys]+[p]
    full_g = torch.autograd.grad(full[2], inputs)
    prior_g = torch.autograd.grad(prior[2], inputs, allow_unused=True)
    assert prior_g[:5] == (None,)*5
    # Independent analytic softmax/log/entropy derivative, including both prior paths.
    prob = prior.detach().numpy(); ps = p.detach().numpy()
    glogit = prob[2]*(np.eye(5)[2]-prob)
    expected_p = glogit/ps+.7*glogit+.2*np.dot(glogit, tail.numpy())*(-np.log(ps)-1)
    np.testing.assert_allclose(prior_g[5].numpy(), expected_p, atol=2e-16, rtol=1e-14)
    assert torch.equal(full_g[5], prior_g[5])
    for i, value in enumerate(full_g[:5]):
        assert float(value) == pytest.approx((i+1)*np.dot(glogit, weights.numpy()), abs=2e-16)
    assert head.weights.grad is None and not head.weights.requires_grad


class ToyObjectives:
    """The same quadratic forward, different semantic route; no renderer mocks."""
    def __init__(self, tx):
        self.tx = tx; self.counts = counters(); self.calls = []

    def __call__(self, names, *, gradient):
        self.calls.append((list(names), gradient, self.tx.means.detach().clone()))

        def view(name):
            x = self.tx.means[0, 0].double()
            rgb = .0001*(x+1).square()
            full = (1-2*x).square()
            # Same value: prior derivative has the opposite sign, context supplies -3x.
            prior = (1+x-3*x.detach()).square()
            return rgb, full, prior, {'raw_ce': 1.}

        return worker.evaluate_view_objectives(names, self.tx.means, view, self.counts, gradient=gradient)


@pytest.mark.parametrize('arm,sign', [('full', 1), ('prior_only', -1)])
def test_both_routes_run_148_real_adam_attempts_with_actual_displacement_and_json(arm, sign):
    tx = transaction(); objective = ToyObjectives(tx); emitted = []
    schedule = worker.batch_schedule(NAMES)
    result = worker.run_attempts(arm, schedule, tx, objective, emitted.append)
    assert result['attempts'] == result['accepted'] == 148 and result['rejected'] == 0
    assert not result['converged_claimed'] and result['stop_reason'] == 'fixed_148_attempt_schedule_finished'
    assert len(objective.calls) == 296 and len(emitted) == 148
    assert objective.counts == {'means_vjp': 3108, 'training_view_pairs': 2072, 'description_view_pairs': 0}
    assert float(tx.means.detach()[0, 0]) * sign > 0
    for i, row in enumerate(emitted):
        before, after = objective.calls[2*i:2*i+2]
        assert before[:2] == (schedule[i]['names'], True)
        assert after[:2] == (schedule[i]['names'], False)
        x = float(before[2][0, 0]); delta = float((after[2].double()-before[2].double())[0, 0])
        expected = {'rgb': float(np.float32(.0002*(x+1))),
                    'full': float(np.float32(-4*(1-2*x))),
                    'prior': float(np.float32(2*(1-2*x)))}
        expected['context'] = expected['full']-expected['prior']
        for key, gradient in expected.items():
            assert row['path_gradients']['actual_dots'][key] == pytest.approx(gradient*delta, abs=1e-15)
        selected = expected['full' if arm == 'full' else 'prior']
        assert row['proposal']['semantic_dot_actual_delta'] == pytest.approx(selected*delta, abs=1e-15)
        assert row['attempt'] == row['accepted_optimizer_steps'] == row['proposal']['optimizer_proposed_step'] == i+1
    # The prior-only route genuinely increases this toy's full objective and still commits.
    if arm == 'prior_only':
        assert all(r['candidate']['scene_ce'] > r['baseline']['scene_ce'] for r in emitted)
    assert int(tx.optimizer.state[tx.means]['step']) == 148 and tx.pending is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('failure', ['exception', 'partial', 'bad_mean', 'nan'])
def test_candidate_failure_restores_value_moments_step_and_emits_nothing(failure):
    tx = transaction(); objective = ToyObjectives(tx); schedule = worker.batch_schedule(NAMES)[:1]
    worker.run_attempts('full', schedule, tx, objective)
    before = tx.means.detach().clone(); state = copy.deepcopy(tx.optimizer.state_dict()); emitted = []

    def evaluate(names, *, gradient):
        if not gradient and failure == 'exception':
            raise RuntimeError('Interrupted candidate')
        result = objective(names, gradient=gradient)
        if not gradient:
            if failure == 'partial':
                result['complete'] = False
            elif failure == 'bad_mean':
                result['scene_ce'] += .1
            elif failure == 'nan':
                result['rows'][0]['scene_ce'] = float('nan')
        return result

    with pytest.raises((RuntimeError, ValueError)):
        worker.run_attempts('full', schedule, tx, evaluate, emitted.append)
    assert emitted == [] and torch.equal(tx.means, before)
    assert_state_equal(tx.optimizer.state_dict(), state)
    assert tx.pending is None and tx.means.grad is None
