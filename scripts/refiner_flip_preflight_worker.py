"""Real-trainer flip preflight with read-only hooks and compact audit-only saving."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import torch
import yaml


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--expected', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    config = yaml.safe_load(args.config.read_text())
    expected = json.loads(args.expected.read_text())
    snapshot = args.snapshot.resolve()
    from bridge_rgs import train as training
    from bridge_rgs.model import GaussianScene
    modules = {}
    for name in ('bridge_rgs', 'bridge_rgs.train', 'bridge_rgs.model', 'bridge_rgs.refinement'):
        path = Path(importlib.import_module(name).__file__).resolve()
        assert path.is_relative_to(snapshot)
        modules[name] = {'path': str(path), 'sha256': digest(path)}
    initial = torch.load(config['warmstart'], map_location='cpu', mmap=True, weights_only=False)
    assert len(initial['model']) == 131
    output = Path(config['output'])
    if output.exists():
        raise FileExistsError(output)
    report = {'status': 'running', 'steps': config['steps'], 'actual_imports': modules,
              'config_sha256': digest(args.config), 'expected_plan_sha256': digest(args.expected),
              'warmstart_sha256': digest(config['warmstart']), 'initialization': None,
              'view_sequence': [], 'flip_selector': [], 'renders': []}
    original_optimizer = GaussianScene.optimizers
    original_render = GaussianScene.render
    original_sampler = training.sample_training_view
    original_save = training.atomic_save
    flip_module = None
    if (snapshot/'bridge_rgs/refiner_flip.py').exists():
        import bridge_rgs.refiner_flip as flip_module
        original_choose = flip_module.choose_refiner_horizontal_flip
        def audited_choose(probability, seed, step):
            value = original_choose(probability, seed, step)
            assert value == expected['flips'][step-1]
            report['flip_selector'].append({'step': step, 'flip': value})
            return value
        flip_module.choose_refiner_horizontal_flip = audited_choose

    def audited_optimizer(scene, *args, **kwargs):
        actual = scene.state_dict()
        assert set(actual) == set(initial['model'])
        assert all(torch.equal(v.detach().cpu(), initial['model'][k]) for k, v in actual.items())
        assert scene.refiner_config == initial['refiner_config']
        assert all(p.requires_grad == k.startswith('refiner.') for k, p in scene.named_parameters())
        result = original_optimizer(scene, *args, **kwargs)
        assert all(not opt.state for opt in result.values())
        assert result['heads'].param_groups[1]['lr'] == .0003
        report['initialization'] = {'original_tensor_count': len(actual), 'all_131_tensors_exact': True,
                                    'architecture_exact': True, 'fresh_adam_states_empty': True,
                                    'only_refiner_trainable': True}
        return result

    def audited_sampler(*args, **kwargs):
        value = original_sampler(*args, **kwargs)
        index = len(report['view_sequence'])
        assert value == expected['view_indices'][index]
        report['view_sequence'].append(value)
        return value

    def audited_render(scene, K, w2c, width, height, **kwargs):
        step = len(report['renders'])+1
        assert len(report['view_sequence']) == step
        assert kwargs['refine'] == (not expected['flips'][step-1])
        vi = report['view_sequence'][-1]
        assert torch.equal(w2c.detach().cpu(), initial['training_cameras'][vi])
        assert kwargs.get('degree') == 3
        result = original_render(scene, K, w2c, width, height, **kwargs)
        report['renders'].append({'step': step, 'view_index': vi, 'head_inside_render': kwargs['refine'],
                                  'effective_flip': not kwargs['refine'], 'native_size': [width,height],
                                  'field_hashes': {key: tensor_sha(result[key]) for key in
                                                   ('rgb', 'depth', 'alpha', 'features', 'p3d')}})
        return result

    def audit_save(state, path):
        assert path.name == 'last.pt' and state['step'] == config['steps']
        assert set(state['model']) == set(initial['model'])
        changed = [k for k, v in state['model'].items() if not torch.equal(v.detach().cpu(), initial['model'][k])]
        assert changed and all(k.startswith('refiner.') for k in changed)
        assert torch.equal(state['training_cameras'], initial['training_cameras'])
        assert all(torch.isfinite(v).all() for v in state['model'].values())
        assert all(state[k] == initial[k] for k in ('refiner_config', 'scene_scale', 'feature_dim', 'sh_degree'))
        for name, opt in state['optimizers'].items():
            if name != 'heads':
                assert not opt['state']
        head_opt = state['optimizers']['heads']
        refiner_ids = set(head_opt['param_groups'][1]['params'])
        assert set(head_opt['state']).issubset(refiner_ids)
        assert head_opt['param_groups'][1]['lr'] == .0003
        for value in head_opt['state'].values():
            assert int(value['step']) == config['steps']
        compact = {'kind': 'preflight_audit_only_not_inference_or_resume_checkpoint',
                   'refiner_model': {k: cpu(v) for k, v in state['model'].items() if k.startswith('refiner.')},
                   'heads_optimizer': cpu(head_opt), 'torch_rng': state['torch_rng'], 'cuda_rng': state['cuda_rng'],
                   'numpy_rng': state['numpy_rng'], 'sampler': {k: state['density_state']['extras'][k] for k in
                      ('sampler_order', 'sampler_cursor', 'sampler_rng_state')},
                   'loss_log_stats': state['stats'], 'config': state['config']}
        torch.save(compact, path.with_name('audit_state.pt'))
        report['final'] = {'changed_model_keys': changed, 'only_refiner_changed': True,
                           'field_classifier_cameras_exact': True, 'metadata_exact': True,
                           'all_parameters_finite': True, 'only_refiner_adam_state': True,
                           'head_lr': .0003, 'compact_audit_bytes': path.with_name('audit_state.pt').stat().st_size,
                           'full_checkpoint_written': False}

    GaussianScene.optimizers, GaussianScene.render = audited_optimizer, audited_render
    training.sample_training_view, training.atomic_save = audited_sampler, audit_save
    try:
        training.train(config)
        assert len(report['renders']) == config['steps']
        if flip_module is not None:
            assert len(report['flip_selector']) == config['steps']
        else:
            assert not any(expected['flips'])
        logs = [json.loads(x) for x in (output/'train.jsonl').read_text().splitlines()]
        assert [x['step'] for x in logs] == list(range(1, config['steps']+1))
        if config.get('refiner_horizontal_flip_probability', 0) > 0:
            assert [x['refiner_horizontal_flip'] for x in logs] == expected['flips']
        assert not any('semantic_lr_multiplier' in x for x in logs)
        report.update(status='passed', peak_gpu_gb=max(x['peak_gpu_gb'] for x in logs),
                      compact_audit_sha256=digest(output/'audit_state.pt'))
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        GaussianScene.optimizers, GaussianScene.render = original_optimizer, original_render
        training.sample_training_view, training.atomic_save = original_sampler, original_save
        if flip_module is not None:
            flip_module.choose_refiner_horizontal_flip = original_choose
        output.mkdir(parents=True, exist_ok=True)
        (output/'worker_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status':report['status'], 'steps':report['steps'], 'peak_gpu_gb':report['peak_gpu_gb']}))


if __name__ == '__main__':
    main()
