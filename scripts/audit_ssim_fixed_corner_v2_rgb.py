"""CPU-only final audit of the fixed 30k SSIM-layout replay; never starts training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from compare_evaluations import paired_comparison

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('/mnt/data/SHM2026/runs/ssim_fixed_corner_v2_rgb_full')
SUPPORT = Path('/mnt/data/SHM2026/runs/ssim_fixed_corner_v2_supporting_audit')
PLAN_SHA = '81b378c4d58b586d22d4eb3718a9f678f8a6650d5d0003b4f5e01c28b19e6b6e'
SUPPORT_SHA = 'cc2c0fa8a42089196a246993ecd6a16bf9e308aa6e1e0a9932ce998aba318a8d'
LOSS_SHA = '1e114069c2fdb181e243d7cb8a986659b2e229d2ec49c34821cb45013d9cf2b2'
PAIRED_HELPER_SHA = 'bbb2482fad1203c642216bdad799ae790895bf4582df6105ef6fb25263a5c90c'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    torch.set_num_threads(8)
    report_path = OUTPUT/'stage_audit.json'
    paired_path = OUTPUT/'paired_native_rgb_minus_old_rgb.json'
    require(not report_path.exists() and not paired_path.exists(), 'Preserve any existing endpoint audit')
    require(digest(OUTPUT/'plan.json') == PLAN_SHA, 'Changed locked replay plan')
    require(digest(ROOT/'scripts/compare_evaluations.py') == PAIRED_HELPER_SHA, 'Paired helper changed')
    plan = read(OUTPUT/'plan.json')
    receipt = read(OUTPUT/'experiment_receipt.json')
    require(receipt['status'] == 'completed', 'Wait for natural training and native-evaluation completion')
    config = plan['config']
    require(receipt['config'] == config and receipt['resume'] is None, 'Wrong actual experiment config/resume')
    require(digest(OUTPUT/'replay_config.yaml') == plan['config_sha256'], 'Replay YAML changed')
    require(yaml.safe_load((OUTPUT/'replay_config.yaml').read_text()) == config, 'Replay YAML values changed')
    old = Path(plan['reference_directory'])
    old_receipt = read(old/'experiment_receipt.json')
    require(old_receipt['status'] == 'completed', 'Old endpoint is incomplete')
    changed = {k for k in config.keys() | old_receipt['config'].keys()
               if config.get(k) != old_receipt['config'].get(k)}
    require(changed == {'output'}, 'Replay changes more than the output path')
    require(receipt['source_hashes'] == plan['source_hashes'], 'Runtime source differs from plan')
    actual = {str(p.relative_to(OUTPUT/'source_snapshot')): digest(p)
              for p in sorted((OUTPUT/'source_snapshot').rglob('*.py'))}
    require(actual == plan['source_hashes'], 'Actual frozen package changed')
    require(set(actual) == set(old_receipt['source_hashes']), 'Package file set differs')
    require({k for k in actual if actual[k] != old_receipt['source_hashes'][k]} == {'bridge_rgs/losses.py'}
            and actual['bridge_rgs/losses.py'] == LOSS_SHA, 'Intervention is not the single approved loss repair')
    for path, value in plan['inputs'].items():
        require(digest(path) == value['sha256'], f'Bound input changed: {path}')
    for value in receipt['input_hashes'].values():
        require(digest(value['path']) == value['sha256'], 'Runtime input binding changed')
    checkpoint = OUTPUT/'last.pt'
    require(digest(checkpoint) == receipt['checkpoint_sha256'], 'Final checkpoint SHA mismatch')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(state['step'] == config['steps'] == 30000, 'Not the fixed final 30k endpoint')
    for key, value in config.items():
        require(state['config'][key] == value, f'Checkpoint config mismatch: {key}')
    old_effective = read(old/'config.json')
    expected_effective = {**old_effective, 'output': config['output']}
    require(state['config'] == expected_effective == read(OUTPUT/'config.json'), 'Effective defaults/init count differ')
    require(all(bool(torch.isfinite(t).all()) for t in state['model'].values()), 'Nonfinite model state')
    manifest_path = ROOT/config['manifest']
    manifest = read(manifest_path)
    require(state['manifest_sha256'] == digest(manifest_path), 'Checkpoint manifest hash differs')
    require(state['pixel_protocol']['id'] == config['pixel_protocol'] == 'colmap_corner_v2', 'Wrong pixel profile')
    train_views = [v for v in manifest['views'] if v['split'] == 'train']
    cameras = torch.tensor(np.asarray([v['w2c_original'] for v in train_views]), dtype=torch.float32)
    require(len(train_views) == 350 and torch.equal(state['training_cameras'], cameras), 'Original TRAIN cameras changed')
    require(config['save_every'] == 5000 and config['eval_every'] == 0, 'Checkpoint/evaluation schedule changed')
    require(not list(OUTPUT.glob('step_*.pt')) and not list(OUTPUT.glob('metrics_*.json'))
            and not list(OUTPUT.glob('validation_*')), 'Unexpected intermediate checkpoint/validation')

    require(digest(SUPPORT/'supporting_audit.json') == SUPPORT_SHA, 'Supporting evidence changed')
    supporting = read(SUPPORT/'supporting_audit.json')
    for value in supporting['evidence'].values():
        require(digest(value['path']) == value['sha256'], 'Supporting record changed')
    sampler = read(SUPPORT/'sampler_reconstruction.json')
    trace_record = sampler['reconstructed_trace']
    require(digest(trace_record['path']) == trace_record['sha256'], 'Reconstructed trace changed')
    trace = read(trace_record['path'])
    require(trace['view_names'] == [v['name'] for v in train_views], 'Sampler population order differs')
    generator = np.random.RandomState(42)
    indices = []
    while len(indices) < 30000:
        cycle = generator.permutation(350).tolist()
        indices.extend(cycle)
    require(indices[:30000] == trace['indices'], 'Reconstructed order differs')
    require(all(np.array_equal(x, y) for x, y in zip(generator.get_state(), state['numpy_rng'], strict=True)),
            'Checkpoint NumPy terminal state differs from old reconstructed path')
    extras = state['density_state']['extras']
    require(extras['sampler_order'] == cycle and extras['sampler_cursor'] == 250
            and extras['sampler_rng_state'] is None, 'Checkpoint sampler terminal state differs')
    require(state['config']['independent_view_rng'] is False, 'Original sampler mode changed')

    metrics_path = OUTPUT/'evaluation_native/metrics.json'
    require(digest(metrics_path) == receipt['evaluation_sha256'], 'Native final metrics SHA mismatch')
    metrics = read(metrics_path)
    old_metrics_path = old/'evaluation_native/metrics.json'
    require(digest(old_metrics_path) == old_receipt['evaluation_sha256'], 'Old RGB native metrics changed')
    old_metrics = read(old_metrics_path)
    expected_names = {v['name'] for v in manifest['views'] if v['split'] == 'val'}
    require(metrics['validation_views'] == len(metrics['views']) == 50
            and metrics['semantic_validation_views'] == 41, 'Incomplete fixed native endpoint evaluation')
    require({v['name'] for v in metrics['views']} == expected_names, 'Native VAL names differ')
    require(metrics['scale'] == 1 and metrics['pixel_protocol']['id'] == 'colmap_corner_v2'
            and metrics['evaluation_fingerprint'] == old_metrics['evaluation_fingerprint'], 'Incompatible native protocol')
    require(all((v['width'], v['height']) == (1320, 989) for v in metrics['views']), 'Wrong native dimensions')
    require(all(np.isfinite(v[k]) for v in metrics['views'] for k in ('psnr', 'ssim', 'lpips')), 'Nonfinite RGB metric')
    paired = paired_comparison(old_metrics, metrics, rgb_only=True)
    require(set(paired['metrics']) == {'psnr', 'ssim', 'lpips'}, 'RGB-only comparison leaked semantic ranking')
    paired['sources'] = {str(p): digest(p) for p in [old_metrics_path, old/'experiment_receipt.json',
                                                   metrics_path, OUTPUT/'experiment_receipt.json']}
    pending_path = OUTPUT/'pending_semantic_config.yaml'
    require(digest(pending_path) == plan['pending_semantic_config_sha256'], 'Pending semantic config changed')
    pending = yaml.safe_load(pending_path.read_text())
    old_semantic = read(ROOT/'runs/corner_v2_semantic_coupled/experiment_receipt.json')['config']
    require({k for k in pending.keys() | old_semantic.keys() if pending.get(k) != old_semantic.get(k)}
            == {'output', 'warmstart'}, 'Pending semantic changes more than its two paths')
    require(Path(pending['warmstart']) == checkpoint and pending['steps'] == 8000, 'Wrong pending semantic source/budget')
    report = {'status': 'passed', 'stage': 'rgb', 'cpu_only': True, 'plan_sha256': PLAN_SHA,
              'checkpoint_sha256': receipt['checkpoint_sha256'], 'step': state['step'],
              'source_and_input_hashes_verified': True, 'only_source_change': 'bridge_rgs/losses.py',
              'effective_config': state['config'], 'all_model_tensors_finite': True, 'original_train_cameras_exact': True,
              'profile': state['pixel_protocol'], 'manifest_sha256': state['manifest_sha256'],
              'sampler_terminal_numpy_and_last_cycle_exact': True,
              'reconstructed_trace': trace_record,
              'sampler_scope': 'Full order is reconstructed from unchanged implementation and seed, not directly observed. Matching terminal NumPy state and final permutation/cursor are supporting evidence, not standalone proof of every runtime sample.',
              'no_intermediate_validation_or_step_checkpoints': True,
              'native_fingerprint': metrics['evaluation_fingerprint'],
              'rgb_metrics': {k: metrics[k] for k in ('psnr', 'ssim', 'lpips')},
              'gaussians': metrics['gaussians'], 'pending_semantic_only_path_changes': True,
              'semantic_stage_started_by_this_audit': False,
              'rgb_stage_semantic_scope': 'Untrained semantic endpoint: no semantic ranking or performance claim.',
              'audit_runner_sha256': digest(__file__),
              'paired_helper_sha256': digest(ROOT/'scripts/compare_evaluations.py')}
    with paired_path.open('x') as stream:
        stream.write(json.dumps(paired, indent=2, allow_nan=False)+'\n')
    report['paired_native_rgb'] = {'path': str(paired_path), 'sha256': digest(paired_path)}
    with report_path.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'report': str(report_path), 'sha256': digest(report_path),
                      'rgb_metrics': report['rgb_metrics'], 'paired_metrics': paired['metrics']}, indent=2))


if __name__ == '__main__':
    main()
