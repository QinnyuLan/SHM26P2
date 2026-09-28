"""Synthetic CPU transactions, independent separable CE/RGB objectives only."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/diagnose_finite_structure_refit.py'
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parent/'diagnose_finite_structure_refit.py'
spec = importlib.util.spec_from_file_location('finite_refit_controller_test', SCRIPT)
worker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = worker
spec.loader.exec_module(worker)


def initial():
    return np.array([[.5, .5]], np.float64), (np.zeros((1, 1, 3), np.float32),
                                            np.zeros((1, 2, 3), np.float32))


class Objective:
    """CE of a fixed positive mixture and independent quadratic SH loss.

    Deliberately reversed gradients exercise rejection; no real renderer,
    checkpoint, image, or target is accessed. Cache release deletes only the
    target payload, retaining the independent SH gradient object.
    """
    def __init__(self, *, bad_q=False, bad_sh=False, constant=False, partial=None):
        self.bad_q, self.bad_sh, self.constant, self.partial = bad_q, bad_sh, constant, partial
        self.calls, self.records, self.releases = [], [], []

    def q_value(self, q):
        q32 = q.astype(np.float32).astype(np.float64)
        return 1. if self.constant else float(-np.log(q32@np.array([.6, .4])).mean())

    def q_gradient(self, q):
        q32 = q.astype(np.float32).astype(np.float64)
        weights = np.array([.4, .6] if self.bad_q else [.6, .4])
        return -np.ones_like(q) if self.constant else -weights[None]/(q32@weights)[:, None]/len(q)

    def sh_value(self, values):
        return 1. if self.constant else float(sum(np.mean((v.astype(np.float64)-.02)**2) for v in values))

    def sh_gradient(self, values):
        return tuple(np.zeros_like(v, dtype=np.float64) if self.constant else
                     (-1 if self.bad_sh else 1)*2*(v.astype(np.float64)-.02)/v.size for v in values)

    def __call__(self, q, sh, *, gradient_q, gradient_sh, phase):
        self.calls.append({'q': q.copy(), 'sh': tuple(v.copy() for v in sh),
                           'gq': gradient_q, 'gs': gradient_sh, 'phase': phase})
        row = {'complete': phase != self.partial, 'q_ce': self.q_value(q), 'rgb_mse': self.sh_value(sh),
               'q32_sha256': worker.q_sha(q), 'sh_sha256': worker.sh_sha(sh),
               'target_cache': {'q32_sha256': worker.q_sha(q), 'payload': True}}
        if gradient_q:
            row['q_gradient'] = self.q_gradient(q)
        if gradient_sh:
            row['sh_gradient'] = self.sh_gradient(sh)
        self.records.append(row)
        return row

    def release(self, row):
        assert row['target_cache']['payload'], 'Double release or use of released q cache'
        row['target_cache']['payload'] = False
        self.releases.append(row)

    def fw(self, q, current, vertex, endpoint):
        assert current['target_cache']['payload'] and endpoint['target_cache']['payload']
        # A deliberately bad, but finite, exact-cache FW proposal must reject.
        proposal = np.zeros_like(q); proposal[:, 1] = 1.
        return proposal, {'passed': True, 'cached_objective': self.q_value(proposal)}


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


def test_rejected_sh_restores_parameter_moments_and_adam_step():
    _, values = initial(); tx = worker.SHTransaction(values)
    gradients = tuple(np.ones_like(v, dtype=np.float64) for v in values)
    tx.propose(gradients); tx.resolve(True)
    accepted_values = tx.values(); accepted_state = copy.deepcopy(tx.optimizer.state_dict())
    assert all(float(s['step']) == 1 for s in accepted_state['state'].values())
    tx.propose(tuple(g*7 for g in gradients)); tx.resolve(False)
    assert all(np.array_equal(a, b) for a, b in zip(accepted_values, tx.values(), strict=True))
    assert_state_equal(accepted_state, tx.optimizer.state_dict())
    assert tx.pending is None and all(p.grad is None for p in tx.params)


@pytest.mark.parametrize('bad_q,bad_sh', [(True, False), (False, True)])
def test_component_acceptance_is_independent_and_rejected_gradients_are_not_reused(bad_q, bad_sh):
    q0, sh0 = initial(); objective = Objective(bad_q=bad_q, bad_sh=bad_sh)
    q, sh, report = worker.solve_refit(q0, sh0, objective, objective.fw, release=objective.release)
    if bad_q:
        assert np.array_equal(q, q0) and report['accepted']['q_em'] == 0
        assert objective.sh_value(sh) < objective.sh_value(sh0) and report['accepted']['sh'] > 0
    else:
        assert objective.q_value(q) < objective.q_value(q0) and report['accepted']['q_em'] == 20
        assert all(np.array_equal(a, b) for a, b in zip(sh, sh0, strict=True))
        assert report['accepted']['sh'] == 0
    # Every next EM proposal must use the accepted component state, including
    # the first slot after a rejected FW trial, not its trial gradient/cache.
    accepted_q = q0.copy(); accepted_sh = sh0
    # Separate literal Adam reference, with a gradient recomputed from the
    # accepted quadratic state each time; it has no access to worker caches.
    expected_params = [torch.nn.Parameter(torch.from_numpy(v.copy())) for v in sh0]
    expected_adam = torch.optim.Adam([{'params': [expected_params[0]], 'lr': .0025},
                                     {'params': [expected_params[1]], 'lr': .000125}],
                                    betas=(.9, .999), eps=1e-15, weight_decay=0)
    for call in objective.calls:
        if not call['phase'].startswith('em_'):
            continue
        responsibility = accepted_q * (-objective.q_gradient(accepted_q))
        expected_q = responsibility/responsibility.sum(-1, keepdims=True)
        np.testing.assert_allclose(call['q'], expected_q, rtol=0, atol=2e-16)
        state_before = copy.deepcopy(expected_adam.state_dict())
        for p, g in zip(expected_params, objective.sh_gradient(accepted_sh), strict=True):
            p.grad = torch.as_tensor(g, dtype=torch.float32)
        expected_adam.step(); expected_adam.zero_grad(set_to_none=True)
        for actual, expected in zip(call['sh'], expected_params, strict=True):
            np.testing.assert_array_equal(actual, expected.detach().numpy())
        if objective.q_value(call['q']) < objective.q_value(accepted_q):
            accepted_q = call['q']
        if objective.sh_value(call['sh']) < objective.sh_value(accepted_sh):
            accepted_sh = call['sh']
        else:
            expected_adam.load_state_dict(state_before)
            with torch.no_grad():
                for p, v in zip(expected_params, accepted_sh, strict=True):
                    p.copy_(torch.from_numpy(v))
    assert report['accepted']['q_fw'] == 0
    assert report['q_final_repeat'] == report['rgb_final_repeat'] == 0
    assert len(objective.releases) == 26


def test_fixed_noop_schedule_and_derived_q_rgb_vjp_budgets():
    q0, sh0 = initial(); objective = Objective(constant=True)
    def forbidden_fw(*_):
        pytest.fail('Zero-signal LMO must use its fixed no-op slot')
    q, sh, report = worker.solve_refit(q0, sh0, objective, forbidden_fw, release=objective.release)
    assert np.array_equal(q, q0) and all(np.array_equal(a, b) for a, b in zip(sh, sh0, strict=True))
    assert report['complete_passes'] == len(objective.calls) == 26
    assert report['accepted'] == {'q_em': 0, 'q_fw': 0, 'sh': 0}
    assert sum(r['gq'] for r in objective.calls)*5*8 == 960
    assert sum(r['gs'] for r in objective.calls)*5*8 == 840
    phases = ['baseline']
    for block in range(2):
        phases += [f'em_{block}_{i}' for i in range(10)]+[f'fw_vertex_{block}', f'fw_candidate_{block}']
    assert [r['phase'] for r in objective.calls] == phases+['final']
    assert all(not r['gs'] for r in objective.calls if r['phase'].startswith('fw'))
    assert json.loads(json.dumps(report, allow_nan=False))['complete_passes'] == 26


def test_partial_em_pass_rolls_back_pending_sh_and_cannot_commit():
    q, sh = initial(); objective = Objective(partial='em_0_0'); tx = worker.SHTransaction(sh)
    with pytest.raises(ValueError, match='Incomplete'):
        worker.solve_refit(q, sh, objective, objective.fw, sh_optimizer=tx)
    assert [r['phase'] for r in objective.calls] == ['baseline', 'em_0_0']
    assert all(np.array_equal(a, b) for a, b in zip(tx.values(), sh, strict=True))
    assert tx.optimizer.state_dict()['state'] == {} and tx.pending is None


@pytest.mark.parametrize('phase', ['fw_vertex_0', 'fw_candidate_0'])
def test_partial_fw_stops_before_further_slots_or_final_selection(phase):
    q, sh = initial(); objective = Objective(partial=phase)
    with pytest.raises(ValueError, match='Incomplete'):
        worker.solve_refit(q, sh, objective, objective.fw, release=objective.release)
    assert objective.calls[-1]['phase'] == phase
    assert not any(r['phase'] == 'final' or r['phase'].startswith('em_1') for r in objective.calls)


def test_measurable_fw_numerical_failure_is_not_relabelled_noop():
    q, sh = initial(); objective = Objective(bad_q=True)
    def failed_alignment(q, *_):
        return q.copy(), {'passed': False, 'reason': 'synthetic-adjoint-mismatch'}
    with pytest.raises(ValueError, match='numerical alignment'):
        worker.solve_refit(q, sh, objective, failed_alignment, release=objective.release)
    assert objective.calls[-1]['phase'] == 'fw_vertex_0'
    assert not any(r['phase'] in ('fw_candidate_0', 'final') for r in objective.calls)


def test_pixel_barrier_requires_unchanged_write_once_selection_before_any_b_unique_read(tmp_path):
    paths = {}
    for name in ('a_image', 'a_mask', 'b_image', 'b_mask', 'shared_valid'):
        path = tmp_path/name; path.write_text('synthetic '+name); paths[name] = str(path)
    def view(prefix):
        return {'image_path': paths[prefix+'_image'], 'mask_path': paths[prefix+'_mask'],
                'valid_path': paths['shared_valid']}
    plan = {'A': [view('a')], 'B': [view('b')],
            'deferred_B_hashes': {paths[k]: worker.sha(paths[k]) for k in ('b_image', 'b_mask')}}
    barrier = worker.PixelBarrier(plan)
    barrier.allow(paths['a_image']); barrier.allow(paths['shared_valid'])
    for key in ('b_image', 'b_mask'):
        with pytest.raises(ValueError, match='before immutable'):
            barrier.allow(paths[key])
    choice = tmp_path/'immutable_A_selection.json'
    worker.write(choice, {'gradient_choice': 'keep', 'refit_choice': 'action0'})
    with pytest.raises(FileExistsError):
        worker.write(choice, {'refit_choice': 'action1'})
    with pytest.raises(ValueError, match='missing/changed'):
        barrier.open_b(choice, '0'*64)
    assert not barrier.opened
    barrier.open_b(choice, worker.sha(choice))
    barrier.allow(paths['b_image']); barrier.allow(paths['b_mask'])
    with pytest.raises(ValueError, match='outside fixed TRAIN'):
        barrier.allow(tmp_path/'another_view')
    with pytest.raises(ValueError, match='missing/changed'):
        barrier.open_b(choice, worker.sha(choice))
