"""CPU contracts for completed-only access and independent fixed training replay."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from bridge_rgs.mask2former_training import ShuffledViews, poly_learning_rate, training_sample


def helper():
    path = Path(__file__).resolve().parents[1]/'scripts/audit_mask2former_reference_endpoint.py'
    spec = importlib.util.spec_from_file_location('m2f_endpoint_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_running_endpoint_rejected_without_checkpoint_access(tmp_path, monkeypatch):
    module = helper()
    (tmp_path/'execution_receipt.json').write_text(json.dumps({'status': 'running', 'completed_step': 1000}))
    monkeypatch.setattr(module.torch, 'load', lambda *a, **k: pytest.fail('must not load active last'))
    with pytest.raises(ValueError, match='checkpoint remains unopened'):
        module.completed_receipt(tmp_path)
    (tmp_path/'execution_receipt.json').write_text(json.dumps({'status': 'completed', 'completed_step': 6000,
                                                             'successful_updates': 5999}))
    with pytest.raises(ValueError):
        module.completed_receipt(tmp_path)


def test_independent_replay_matches_real_sampler_augment_draws_and_terminal_states():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        module = helper()
        names = [f'{i:03d}.png' for i in range(259)]
        expected, sampler_state, augmentation_state = module.replay_trace(names, 20260805, total=12)
        sampler, aug = ShuffledViews(names, 20260805), np.random.default_rng(20260806)
        rgb, labels, valid = np.zeros((989, 1320, 3), np.uint8), np.zeros((989, 1320), np.uint8), np.ones((989, 1320), bool)
        actual = []
        for step in range(1, 13):
            name = sampler.next()
            _, _, _, transform = training_sample(rgb, labels, valid, step, aug)
            actual.append({'step': step, 'view': name, **transform})
        assert actual == expected
        assert module.canonical(sampler_state) == module.canonical(sampler.state_dict())
        assert module.canonical(augmentation_state) == module.canonical(aug.bit_generator.state)
        whole, final, _ = module.replay_trace(names, 20260805)
        assert len(whole) == 6000 and final['cursor'] == 6000 % 259
        assert len({row['view'] for row in whole}) == 259
        assert sorted(row['view'] for row in whole[:259]) == names
    finally:
        torch.set_num_threads(old_threads)


def test_optimizer_contract_checks_all_parameter_steps_and_moments():
    module = helper()
    params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(4)]
    declared = [{'lr': lr, 'weight_decay': decay, 'names': [f'weight{i}']}
                for i, (lr, decay) in enumerate([(1e-5, .05), (1e-5, 0), (1e-4, .05), (1e-4, 0)])]
    opt = torch.optim.AdamW([{'params': [p], 'lr': row['lr'], 'weight_decay': row['weight_decay']}
                             for p, row in zip(params, declared, strict=True)])
    for step in [1, 2]:
        poly_learning_rate(opt, step)
        for p in params:
            p.grad = torch.ones_like(p)
        opt.step()
    state = {'model': {f'weight{i}': p.detach() for i, p in enumerate(params)}, 'optimizer': opt.state_dict()}
    assert module.optimizer_contract(state, declared, step=2)['all_adam_parameter_steps'] == 2
    state['optimizer']['state'][0]['step'] = torch.tensor(1.)
    with pytest.raises(ValueError, match='Skipped update'):
        module.optimizer_contract(state, declared, step=2)


def test_stage_probe_reads_only_named_pretrained_tensors_and_rejects_zero_delta(tmp_path):
    module = helper()
    names = [f'model.pixel_level_module.encoder.encoder.layers.{i}.blocks.0.attention.self.query.weight' for i in range(4)]
    source = {name: torch.ones(2, 2) for name in names}
    save_file(source, tmp_path/'model.safetensors')
    state = {'model': {name: value+.25 for name, value in source.items()}}
    receipt = {'stage_update_probes': {f'stage{i+1}': {'finite': True, 'delta_l1': 1.} for i in range(4)}}
    audit = module.stage_deltas(state, receipt, tmp_path)
    assert all(row['changed_elements'] == 4 for row in audit.values())
    state['model'][names[0]] = source[names[0]]
    with pytest.raises(ValueError, match='delta'):
        module.stage_deltas(state, receipt, tmp_path)
