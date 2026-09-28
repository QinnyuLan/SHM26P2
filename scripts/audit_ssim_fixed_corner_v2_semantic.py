"""CPU-only fixed 8k semantic endpoint audit; no training or image scoring."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from compare_evaluations import paired_comparison

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('/mnt/data/SHM2026/runs/ssim_fixed_corner_v2_semantic_coupled')
PLAN_SHA = 'b1df1dad8c28434f876f55faa9962417c271cf51dc554b2183d3c90d34fb10ac'
RGB_SHA = 'bd5097e3aa5d0a8fcc6c326c7f653447c4683979727a9277fdde73c4eaf16239'
LOSS_SHA = '1e114069c2fdb181e243d7cb8a986659b2e229d2ec49c34821cb45013d9cf2b2'
HELPER_SHA = 'bbb2482fad1203c642216bdad799ae790895bf4582df6105ef6fb25263a5c90c'


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
    report_path, paired_path = OUTPUT/'stage_audit.json', OUTPUT/'paired_native_minus_old_semantic.json'
    trace_path = OUTPUT/'reconstructed_semantic_view_order.json'
    require(not any(p.exists() for p in (report_path, paired_path, trace_path)), 'Preserve existing audit artifacts')
    require(digest(OUTPUT/'plan.json') == PLAN_SHA, 'Changed locked semantic plan')
    require(digest(ROOT/'scripts/compare_evaluations.py') == HELPER_SHA, 'Paired helper changed')
    plan, receipt = read(OUTPUT/'plan.json'), read(OUTPUT/'experiment_receipt.json')
    require(receipt['status'] == 'completed', 'Wait for natural semantic training and native-evaluation completion')
    config = plan['config']
    require(receipt['config'] == config and receipt['resume'] is None, 'Wrong actual config/resume')
    require(yaml.safe_load((OUTPUT/'replay_config.yaml').read_text()) == config, 'Replay YAML values changed')
    old = Path(plan['reference'])
    old_receipt = read(old/'experiment_receipt.json')
    require(old_receipt['status'] == 'completed', 'Old semantic endpoint incomplete')
    require({k for k in config.keys() | old_receipt['config'].keys()
             if config.get(k) != old_receipt['config'].get(k)} == {'output', 'warmstart'}, 'Unexpected config intervention')
    actual = {str(p.relative_to(OUTPUT/'source_snapshot')): digest(p)
              for p in sorted((OUTPUT/'source_snapshot').rglob('*.py'))}
    require(actual == plan['source_hashes'] == receipt['source_hashes'], 'Actual frozen source differs')
    require(set(actual) == set(old_receipt['source_hashes'])
            and {k for k in actual if actual[k] != old_receipt['source_hashes'][k]} == {'bridge_rgs/losses.py'}
            and actual['bridge_rgs/losses.py'] == LOSS_SHA, 'Source intervention is not the approved loss repair')
    for path, value in plan['inputs'].items():
        require(digest(path) == value['sha256'], f'Bound input changed: {path}')
    for value in receipt['input_hashes'].values():
        require(digest(value['path']) == value['sha256'], 'Runtime input changed')
    checkpoint, initial = OUTPUT/'last.pt', Path(config['warmstart'])
    require(digest(checkpoint) == receipt['checkpoint_sha256'], 'Final semantic checkpoint SHA mismatch')
    require(digest(initial) == RGB_SHA == receipt['input_hashes']['initial_checkpoint']['sha256'], 'Wrong new 30k warmstart')
    rgb_receipt, rgb_audit = read(initial.parent/'experiment_receipt.json'), read(initial.parent/'stage_audit.json')
    require(rgb_receipt['status'] == 'completed' and rgb_audit['status'] == 'passed'
            and rgb_receipt['checkpoint_sha256'] == rgb_audit['checkpoint_sha256'] == RGB_SHA, 'RGB endpoint not audited')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    before = torch.load(initial, map_location='cpu', weights_only=False)
    require(state['step'] == config['steps'] == 8000, 'Not fixed 8k endpoint')
    expected = {**read(old/'config.json'), 'output': config['output'], 'warmstart': config['warmstart'],
                'initial_gaussians': len(before['model']['splats.means'])}
    require(state['config'] == expected == read(OUTPUT/'config.json'), 'Effective config/defaults differ')
    require(all(bool(torch.isfinite(t).all()) for t in state['model'].values()), 'Nonfinite model state')
    frozen = [k for k in before['model'] if k.startswith('splats.') and k != 'splats.sem_features']
    frozen += ['background_logits', 'semantic_prior_counts']
    require(all(state['model'][k].dtype == before['model'][k].dtype
                and torch.equal(state['model'][k], before['model'][k]) for k in frozen), 'Frozen geometry/RGB/prior changed')
    require(torch.equal(state['training_cameras'], before['training_cameras']), 'Warmstart cameras changed')
    require(all(state[k] == before[k] for k in ('feature_dim', 'sh_degree', 'scene_scale', 'pixel_protocol', 'manifest_sha256')),
            'Render metadata changed')
    manifest_path = ROOT/config['manifest']
    manifest = read(manifest_path)
    require(state['manifest_sha256'] == digest(manifest_path)
            and state['pixel_protocol']['id'] == config['pixel_protocol'] == 'colmap_corner_v2', 'Wrong data/profile binding')
    train = [v for v in manifest['views'] if v['split'] == 'train']
    population = [i for i, view in enumerate(train) if view.get('mask_path')]
    require(len(train) == 350 and len(population) == 259 and config['train_labeled_only'], 'Wrong supervised TRAIN population')
    original_cameras = torch.tensor(np.asarray([v['w2c_original'] for v in train]), dtype=torch.float32)
    require(torch.equal(state['training_cameras'], original_cameras), 'Original camera poses changed')
    require(config['independent_view_rng'] and config['view_sampler_seed'] == 42
            and config['view_sampling'] == 'shuffle', 'Changed sampler contract')
    generator, order = np.random.default_rng(42), []
    while len(order) < 8000:
        cycle = generator.permutation(population).tolist()
        order.extend(cycle)
    extras = state['density_state']['extras']
    require(extras['sampler_rng_state'] == generator.bit_generator.state
            and extras['sampler_order'] == cycle and extras['sampler_cursor'] == 8000 % 259, 'Sampler terminal mismatch')
    require(config['save_every'] == 2000 and config['eval_every'] == 0
            and not list(OUTPUT.glob('step_*.pt')) and not list(OUTPUT.glob('metrics_*.json'))
            and not list(OUTPUT.glob('validation_*')), 'Intermediate checkpoint/validation schedule changed')
    metrics_path, old_metrics_path = OUTPUT/'evaluation_native/metrics.json', old/'evaluation_native/metrics.json'
    rgb_metrics_path = initial.parent/'evaluation_native/metrics.json'
    require(digest(metrics_path) == receipt['evaluation_sha256']
            and digest(old_metrics_path) == old_receipt['evaluation_sha256']
            and digest(rgb_metrics_path) == rgb_receipt['evaluation_sha256'], 'Metric receipt binding differs')
    metrics, old_metrics, rgb_metrics = read(metrics_path), read(old_metrics_path), read(rgb_metrics_path)
    names = {v['name'] for v in manifest['views'] if v['split'] == 'val'}
    require(metrics['validation_views'] == len(metrics['views']) == 50 and metrics['semantic_validation_views'] == 41
            and {v['name'] for v in metrics['views']} == names, 'Incomplete fixed 50/41 evaluation')
    require(metrics['scale'] == 1 and metrics['pixel_protocol']['id'] == 'colmap_corner_v2'
            and metrics['evaluation_fingerprint'] == old_metrics['evaluation_fingerprint'], 'Incompatible native protocol')
    require(all((v['width'], v['height']) == (1320, 989) for v in metrics['views']), 'Wrong native dimensions')
    require(all(metrics[k] == rgb_metrics[k] for k in ('psnr', 'ssim', 'lpips', 'evaluation_fingerprint')), 'RGB metrics changed')
    png_hashes = {}
    for view in metrics['views']:
        name = Path(view['name']).stem+'_rgb.png'
        sha = digest(OUTPUT/'evaluation_native'/name)
        require(sha == digest(initial.parent/'evaluation_native'/name), f'RGB PNG changed: {name}')
        png_hashes[name] = sha
    paired = paired_comparison(old_metrics, metrics)
    paired['sources'] = {str(p): digest(p) for p in (metrics_path, old_metrics_path,
                                                   OUTPUT/'experiment_receipt.json', old/'experiment_receipt.json')}
    trace = {'steps': 8000, 'labeled_train_population': population, 'train_names': [v['name'] for v in train],
             'indices': order[:8000], 'scope': 'Reconstructed from frozen sampler and seed; not a directly observed per-step trace.'}
    with trace_path.open('x') as stream:
        stream.write(json.dumps(trace, separators=(',', ':'))+'\n')
    with paired_path.open('x') as stream:
        stream.write(json.dumps(paired, indent=2, allow_nan=False)+'\n')
    report = {'status': 'passed', 'stage': 'semantic', 'cpu_only': True, 'step': 8000, 'plan_sha256': PLAN_SHA,
              'checkpoint_sha256': receipt['checkpoint_sha256'], 'warmstart_sha256': RGB_SHA,
              'source_and_input_hashes_verified': True, 'all_model_tensors_finite': True,
              'frozen_rgb_geometry_prior_tensors_exact': frozen, 'original_cameras_and_render_metadata_exact': True,
              'manifest_sha256': state['manifest_sha256'], 'profile': state['pixel_protocol'],
              'native_fingerprint': metrics['evaluation_fingerprint'], 'all50_rgb_png_sha256': png_hashes,
              'native_rgb_metrics_exact_to_new30k': True, 'labeled_train_count': 259,
              'sampler_terminal_state_order_cursor_match': True, 'sampler_cursor': extras['sampler_cursor'],
              'reconstructed_trace': {'path': str(trace_path), 'sha256': digest(trace_path)},
              'sampler_scope': 'Terminal state and cycle/cursor support the reconstructed 8k order; no claim of direct observation of all runtime views.',
              'no_intermediate_validation_or_step_checkpoints': True,
              'metrics': {k: metrics[k] for k in ('psnr', 'ssim', 'lpips', 'miou_all', 'miou_foreground', 'iou', 'semantic_3d')},
              'paired_native': {'path': str(paired_path), 'sha256': digest(paired_path)},
              'scope': 'Fixed semantic followthrough on the new RGB geometry; no official-grid scoring here or automatic model adoption.',
              'runner_sha256': digest(__file__), 'paired_helper_sha256': HELPER_SHA}
    with report_path.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'report': str(report_path), 'sha256': digest(report_path), 'metrics': report['metrics'],
                      'paired_metrics': paired['metrics']}, indent=2))


if __name__ == '__main__':
    main()
