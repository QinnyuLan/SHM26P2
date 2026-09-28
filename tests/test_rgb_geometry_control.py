"""Synthetic matched RGB-only control contracts; no GPU/data/preparation."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
PARENT_SOURCE = Path('/mnt/data/SHM2026/runs/continuous_geometry_proposals_v1/source_snapshot')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


WORKER_PATH = HERE/'train_rgb_geometry_control.py'
HELPER_PATH = HERE/'bridge_rgs/continuous_geometry_rgb.py'
if not WORKER_PATH.exists():
    WORKER_PATH = HERE.parent/'scripts/train_rgb_geometry_control.py'
    HELPER_PATH = HERE.parent/'src/bridge_rgs/continuous_geometry_rgb.py'
OLD_SOURCE = HERE if (HERE/'train_continuous_geometry_proposals.py').exists() else PARENT_SOURCE
worker = load('rgb_geometry_worker_contract', WORKER_PATH)
old_worker = load('old_geometry_worker_contract', OLD_SOURCE/'train_continuous_geometry_proposals.py')
helper = load('rgb_geometry_helper_contract', HELPER_PATH)
old_helper = load('old_geometry_helper_contract', OLD_SOURCE/'bridge_rgs/continuous_geometry.py')


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def transaction(module):
    means = torch.nn.Parameter(torch.zeros((2, 3), dtype=torch.float32))
    return module.MeansTransaction(means, torch.tensor([[1., 0, 0, 0]]).repeat(2, 1), torch.zeros((2, 3)), 1.)


def same_state(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same_state(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            same_state(x, y)
    else:
        assert a == b


def test_inherited_control_reduction_and_real_render_paths_unchanged():
    now = functions(WORKER_PATH); old = functions(OLD_SOURCE/'train_continuous_geometry_proposals.py')
    for name in ('batch_schedule', 'accept_candidate', 'evaluate_view_objectives', 'run_attempts'):
        assert now[name] == old[name]
    # Only the helper import is different in the complete real rendering body.
    assert now['perform'].replace('continuous_geometry_rgb', 'continuous_geometry') == old['perform']
    current_helper = functions(HELPER_PATH); inherited_helper = functions(OLD_SOURCE/'bridge_rgs/continuous_geometry.py')
    for name in ('MeansTransaction', 'fixed_metric', 'cap_displacement', '_metric_norm'):
        assert current_helper[name] == inherited_helper[name]


def test_same_148_batch_order_and_exact_independent_budget():
    names = [f'{i:03d}.png' for i in range(259)]
    assert worker.batch_schedule(names) == old_worker.batch_schedule(names)
    assert worker.ARMS == ('rgb',)
    assert worker.SPEC['expected_counts'] == {
        'batch_attempts': 148, 'training_view_pairs': 2072,
        'description_view_pairs': 518, 'raster': 5180, 'means_vjp': 2072}
    assert worker.SPEC['internal_seconds'] == 300 and worker.SPEC['external_seconds'] == 360


def test_rgb_proposals_ignore_semantic_gradient_but_report_its_true_dot():
    rgb = torch.tensor([[2., -1., 4.], [-3., 2., 1.]], dtype=torch.float64)
    sem = torch.tensor([[-8., 2., -4.], [1., 7., 2.]], dtype=torch.float64)
    first, second, old = transaction(helper), transaction(helper), transaction(old_helper)
    for scale in (1., -.4):
        before = first.means.detach().double().clone()
        a, report = first.propose(scale*rgb, sem, 'rgb')
        b, report_other = second.propose(scale*rgb, -sem*19, 'rgb')
        c, _ = old.propose(scale*rgb, torch.zeros_like(sem), 'joint')
        assert torch.equal(a, b) and torch.equal(a, c)
        same_state(first.optimizer.state_dict(), old.optimizer.state_dict())
        same_state(first.optimizer.state_dict(), second.optimizer.state_dict())
        actual = a.double()-before
        assert report['semantic_dot_actual_delta'] == float((sem*actual).sum())
        assert report_other['semantic_dot_actual_delta'] == float((-19*sem*actual).sum())
        assert report['weighted_semantic_gradient_l2'] > 0 and report['mode'] == 'rgb'
        assert report['combined_gradient_l2'] == float((scale*rgb).norm())
        json.dumps(report, allow_nan=False)
        first.resolve(True); second.resolve(True); old.resolve(True)


def test_finite_rgb_control_commits_even_if_both_objectives_worsen():
    names = [str(i) for i in range(7)]
    def record(value):
        return {'complete': True, 'names': names, 'rows': [
            {'name': n, 'rgb_mse': value, 'raw_ce': value} for n in names], 'rgb_mse': value, 'raw_ce': value}
    accepted, report = worker.accept_candidate('rgb', record(1.), record(2.))
    assert accepted and report['control_finite_commit']
    assert not any(report['criteria'].values())
    with pytest.raises(ValueError, match='Unknown arm'):
        worker.accept_candidate('trust', record(1.), record(.5))


def test_rgb_mode_reject_keeps_original_adam_rollback_contract():
    tx = transaction(helper); base = tx.means.detach().clone()
    tx.propose(torch.ones_like(base), torch.full_like(base, -10.), 'rgb')
    tx.resolve(False)
    assert torch.equal(tx.means, base) and tx.optimizer.state_dict()['state'] == {}
    assert tx.means.grad is None and tx.pending is None
