"""Contracts for the isolated continuation and unchanged frozen tensors."""
import importlib.util
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve()
SCRIPT = HERE.parent / 'run_projective_deck_pooling.py'
if not SCRIPT.exists():
    SCRIPT = HERE.parents[1] / 'scripts/run_projective_deck_pooling.py'
spec = importlib.util.spec_from_file_location('deck_runner', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_population_never_uses_val_and_requires_all_train():
    views = [{'name': f'{i:03}.png', 'split': 'train', 'mask_path': 'x' if i < 259 else None}
             for i in range(350)]
    views += [{'name': f'val{i}.png', 'split': 'val', 'mask_path': 'truth'} for i in range(50)]
    chosen = runner.training_population({'views': list(reversed(views))})
    assert len(chosen) == 259 and all(v['split'] == 'train' for v in chosen)
    assert [v['name'] for v in chosen] == sorted(v['name'] for v in chosen)
    with pytest.raises(ValueError, match='population'):
        runner.training_population({'views': views[1:]})


def test_frozen_digest_detects_values_shapes_and_scope():
    state = {'splats.x': torch.arange(4.), 'refiner.weight': torch.ones(3)}
    frozen = runner.state_digest(state)
    head = runner.state_digest(state, refiner=True)
    state['refiner.weight'].add_(1)
    assert runner.state_digest(state) == frozen
    assert runner.state_digest(state, refiner=True) != head
    state['splats.x'] = state['splats.x'].reshape(2, 2)
    assert runner.state_digest(state) != frozen


def test_omitting_frozen_objectives_preserves_head_gradient():
    from bridge_rgs.losses import semantic_loss
    torch.manual_seed(2)
    logits = torch.randn(9, 11, 5, requires_grad=True)
    residual = logits.tanh()
    target = torch.randint(0, 5, (9, 11))
    valid = torch.ones(9, 11)
    target[0] = 255
    weights = torch.tensor([.7, 1., 1.1, 1.2, 1.4])
    objective = semantic_loss(logits.softmax(-1), target, valid, weights) + .001*residual.square().mean()
    frozen_raw = semantic_loss(torch.randn(9, 11, 5).softmax(-1), target, valid, weights, 0)
    actual = torch.autograd.grad(objective + .5*frozen_raw + torch.tensor(4.), logits, retain_graph=True)[0]
    simplified = torch.autograd.grad(objective, logits)[0]
    torch.testing.assert_close(actual, simplified, rtol=0, atol=0)


def test_predeclared_controls_and_budget_are_matched():
    assert runner.ARMS == ('original', 'per_camera', 'projective', 'wrong')
    assert runner.SPEC['steps_per_arm'] == 2000
    assert runner.SPEC['candidate'] == 'projective'
    assert runner.SPEC['gain_thresholds'] == {'original': .0015, 'per_camera': .001, 'wrong': .001, 'E': .002}
