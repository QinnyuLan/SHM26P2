"""CPU-only audit of the completed, fixed610-step Mip filter smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

PLAN_SHA = '9f7d7a6693251e7f8347bc25724af02a1650c2f2f4eab7b2e84bbb9927fac22c'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def finite_tree(value, label):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite '+label)
        return 1
    if isinstance(value, dict):
        return sum(finite_tree(v, f'{label}.{k}') for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sum(finite_tree(v, f'{label}.{i}') for i, v in enumerate(value))
    return 0


def audit(directory):
    from bridge_rgs import mip_filter
    from bridge_rgs.train import load_scene

    root = Path(directory).resolve(); snapshot = root/'source_snapshot'
    if Path(mip_filter.__file__).resolve() != snapshot/'bridge_rgs/mip_filter.py':
        raise ValueError('Use the smoke frozen PYTHONPATH')
    plan = json.loads((root/'plan.json').read_text())
    launch = json.loads((root/'launch_receipt.json').read_text())
    assert digest(root/'plan.json') == PLAN_SHA == launch['plan_sha256']
    assert launch['status'] == 'completed' and launch['exit_code'] == 0
    assert {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')} == plan['source_hashes']
    for path, record in plan['inputs'].items():
        assert digest(path) == record['sha256'] and Path(path).stat().st_size == record['bytes']
    config = yaml.safe_load((root/'training_config.yaml').read_text())
    assert config == plan['config']
    checkpoint = root/'last.pt'; before_sha = digest(checkpoint)
    scene, state = load_scene(checkpoint, device='cpu')
    torch.set_num_threads(8)
    assert state['step'] == 610 and state['mip_filter_config'] == mip_filter.POLICY
    for key, value in config.items():
        assert state['config'][key] == (mip_filter.POLICY if key == 'mip_filter' else value), key
    assert state['config'] == json.loads((root/'config.json').read_text())
    assert state['config']['eval_every'] == 0 and state['config']['semantic_start'] > 610
    finite_counts = {k: finite_tree(state[k], k) for k in ('model', 'optimizers', 'density_state')}
    manifest_path = Path(config['manifest']).resolve()
    manifest = json.loads(manifest_path.read_text())
    views = [v for v in manifest['views'] if v['split'] == 'train']
    source = mip_filter.camera_source(views, state['training_cameras'], digest(manifest_path))
    saved_filter = state['mip_filter_state']; saved_rho = state['model']['mip_filter_rho']
    assert source['metadata'] == saved_filter['source'] and len(views) == 350
    assert saved_filter['refresh_count'] == 8 and saved_filter['last_refresh_step'] == 610
    assert saved_filter['gaussians'] == len(state['model']['splats.means']) == len(saved_rho)
    rho, rho_stats = mip_filter.compute_rho(scene.splats['means'], source)
    absolute = (rho-saved_rho).abs(); relative = absolute/saved_rho
    rho_comparison = {'max_absolute_error': float(absolute.max()), 'max_relative_error': float(relative.max()),
                      'mean_absolute_error': float(absolute.mean()), 'bitwise_equal': torch.equal(rho, saved_rho),
                      'interpretation': 'CPU versus saved GPU arithmetic; reported without choosing or changing a tolerance',
                      'unseen_CPU': rho_stats['unseen_gaussians'], 'unseen_GPU': saved_filter['unseen_gaussians']}
    logs = [json.loads(line) for line in (root/'train.jsonl').read_text().splitlines()]
    assert [x['step'] for x in logs] == [1]+list(range(10, 611, 10))
    refresh_records = []; previous_count = 0
    for row in logs:
        step = row['step']; expected = 1+step//100+(step == 610)
        assert row['mip_filter_refresh_count'] == expected
        assert row['mip_filter_last_refresh_step'] == (610 if step == 610 else step//100*100)
        assert row['semantic_loss'] == 0 and row['sh_degree'] == 0
        if expected != previous_count:
            refresh_records.append(row); previous_count = expected
    assert len(refresh_records) == 8
    gpu_total = sum(r['mip_filter_last_refresh_cuda_ms'] for r in refresh_records)
    wall_total = sum(r['mip_filter_last_refresh_wall_seconds'] for r in refresh_records)
    assert gpu_total == saved_filter['total_refresh_cuda_ms'] and wall_total == saved_filter['total_refresh_wall_seconds']
    event = next(r for r in logs if r['step'] == 600)
    initial_count = logs[0]['gaussians']; n = logs[-1]['gaussians']
    assert all(r['gaussians'] == initial_count for r in logs if r['step'] < 600)
    assert n == initial_count+event['splits']+event['duplicates']-event['pruned']
    assert event['splits'] > 0 and event['duplicates'] > 0
    assert n == len(saved_rho) and state['density_state']['mixed_gradient']['observations'].shape == (n,)
    groups = {}
    required = {'means', 'quats', 'log_scales', 'opacity_logits', 'sh0'}
    for name, optimizer in state['optimizers'].items():
        records = []
        for group_index, group in enumerate(optimizer['param_groups']):
            for parameter_id in group['params']:
                value = optimizer['state'].get(parameter_id)
                if value is None:
                    continue
                records.append({'group': group_index, 'step': int(value['step']),
                                'nonzero_exp_avg': int(torch.count_nonzero(value['exp_avg'])),
                                'nonzero_exp_avg_sq': int(torch.count_nonzero(value['exp_avg_sq'])),
                                'lr': group['lr']})
                assert int(value['step']) == 610
        if name in required:
            assert len(records) == 1 and records[0]['nonzero_exp_avg'] > 0
        if name == 'sem_features':
            assert not records
        if name == 'heads':
            assert len(records) == 1 and records[0]['group'] == 2 and records[0]['nonzero_exp_avg'] > 0
        groups[name] = records
    assert not groups['sh_rest'] or all(r['nonzero_exp_avg'] == 0 for r in groups['sh_rest'])
    assert not any(p.name.startswith(('validation', 'evaluation', 'metrics')) for p in root.iterdir())
    prior = Path('/mnt/data/SHM2026/runs/rgb140_inspired_mixed_500k/source_snapshot/bridge_rgs/mixed_gradient_reference.py')
    assert digest(prior) == digest(snapshot/'bridge_rgs/mixed_gradient_reference.py')
    assert digest(checkpoint) == before_sha and not torch.cuda.is_initialized()
    report = {'status': 'passed', 'scope': 'CPU-only completed endpoint audit; GPU/CPU rho difference reported separately',
              'plan_sha256': PLAN_SHA, 'checkpoint_sha256': before_sha,
              'launch_receipt_sha256': digest(root/'launch_receipt.json'), 'audit_script_sha256': digest(__file__),
              'source_40_files_exact': True, 'inputs_exact': True, 'config_exact': True,
              'step': 610, 'gaussians': n, 'finite_tensor_counts': finite_counts,
              'camera_population_original350_exact': True, 'filter_source_exact': True,
              'rho_shape': list(saved_rho.shape), 'rho_CPU_comparison': rho_comparison,
              'refresh_count': 8, 'refresh_steps_from_logs': [r['mip_filter_last_refresh_step'] for r in refresh_records],
              'refresh_total_cuda_ms': gpu_total, 'refresh_total_wall_seconds': wall_total,
              'raw_mixed_module_unchanged': True, 'raw_opacity_reset_exercised': False,
              'reset_note': '610 smoke precedes first reset3000; formula/order preserved by byte-identical mixed module and CPU contracts',
              'topology600': {k: event[k] for k in ('splits', 'duplicates', 'pruned')},
              'optimizer_gradient_update_evidence': groups,
              'SH_note': 'Degree0 throughout610; sh_rest zero moments are expected, not a failed gradient contract',
              'no_validation_artifacts': True,
              'validation_evidence_limit': 'eval_every0, frozen CLI train path and artifact inventory; no syscall-level file-read trace',
              'elapsed_training_seconds': logs[-1]['elapsed_seconds'], 'elapsed_launcher_seconds': launch['elapsed_seconds'],
              'cuda_initialized_by_audit': False}
    target = root/'cpu_endpoint_audit.json'
    with target.open('x') as f: json.dump(report, f, indent=2, allow_nan=False); f.write('\n')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', required=True)
    print(audit(parser.parse_args().run))
