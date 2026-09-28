"""CPU-only fixed-endpoint provenance and frozen-RGB audit for corner-v2 stages."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['rgb', 'semantic'])
    args = parser.parse_args()
    torch.set_num_threads(8)
    plan = json.loads((ROOT/'runs/corner_v2_training_plan.json').read_text())
    config_path = 'configs/corner_v2_rgb_full.yaml' if args.stage == 'rgb' else 'configs/corner_v2_semantic_coupled.yaml'
    config = yaml.safe_load((ROOT/config_path).read_text())
    assert digest(ROOT/config_path) == plan['config_sha256'][config_path]
    folder = ROOT/config['output']
    receipt = json.loads((folder/'experiment_receipt.json').read_text())
    assert receipt['status'] == 'completed' and receipt['config'] == config
    assert receipt['source_hashes'] == plan['source_hashes']
    for name, sha in receipt['source_hashes'].items():
        assert digest(folder/'source_snapshot'/name) == sha
    for value in receipt['input_hashes'].values():
        assert digest(value['path']) == value['sha256']
    assert receipt['input_hashes']['manifest']['sha256'] == plan['manifest_sha256']
    assert receipt['input_hashes']['init_points']['sha256'] == plan['init_points_sha256']
    assert digest(folder/'last.pt') == receipt['checkpoint_sha256']
    state = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
    assert state['step'] == config['steps'] == (30000 if args.stage == 'rgb' else 8000)
    # Trainer records normalized architecture, explicit defaults and seed count.
    # All user-declared values must remain fixed; these derived fields are logged.
    for key, value in config.items():
        if key == 'refiner':
            assert all(state['config'][key][name] == item for name, item in value.items())
        else:
            assert state['config'][key] == value
    expected_extra = {'local_teacher_kd', 'supervised_teacher_priority', 'camera_attribution',
                      'camera_quality_weighting', 'pseudo_refiner_weight', 'independent_view_rng',
                      'refiner_field_grad', 'parameter_scope', 'refiner', 'initial_gaussians'}
    assert state['config'].keys() - config.keys() <= expected_extra
    assert state['pixel_protocol']['id'] == config['pixel_protocol'] == 'colmap_corner_v2'
    assert state['manifest_sha256'] == plan['manifest_sha256']
    assert all(torch.isfinite(value).all() for value in state['model'].values())
    assert config['eval_every'] == 0
    assert not list(folder.glob('step_*.pt')) and not list(folder.glob('metrics_*.json'))
    metric_path = folder/'evaluation_native/metrics.json'
    assert digest(metric_path) == receipt['evaluation_sha256']
    metrics = json.loads(metric_path.read_text())
    grid = json.loads((ROOT/'runs/corner_v2_preparation/grid_audit.json').read_text())
    assert metrics['pixel_protocol']['id'] == 'colmap_corner_v2'
    assert metrics['evaluation_fingerprint'] == grid['native_fingerprints']['corner_v2']
    assert metrics['evaluation_fingerprint'] != grid['native_fingerprints']['legacy']
    assert metrics['validation_views'] == 50 and metrics['semantic_validation_views'] == 41
    assert metrics['scale'] == 1.0 and metrics['lpips'] is not None
    assert all((v['width'], v['height']) == (1320, 989) for v in metrics['views'])
    frozen = None
    if args.stage == 'rgb':
        assert not config.get('warmstart') and 'initial_checkpoint' not in receipt['input_hashes']
    else:
        source = ROOT/config['warmstart']
        source_receipt = json.loads((source.parent/'experiment_receipt.json').read_text())
        assert source_receipt['status'] == 'completed'
        assert digest(source) == receipt['input_hashes']['initial_checkpoint']['sha256'] == source_receipt['checkpoint_sha256']
        before = torch.load(source, map_location='cpu', weights_only=False)
        frozen = [key for key in before['model'] if key.startswith('splats.') and key != 'splats.sem_features'] + ['background_logits', 'semantic_prior_counts']
        assert all(state['model'][key].dtype == before['model'][key].dtype and torch.equal(state['model'][key], before['model'][key]) for key in frozen)
        assert torch.equal(state['training_cameras'], before['training_cameras'])
        assert all(state[key] == before[key] for key in ('feature_dim', 'sh_degree', 'scene_scale'))
        reference = json.loads((source.parent/'evaluation_native/metrics.json').read_text())
        assert all(reference[key] == metrics[key] for key in ('psnr', 'ssim', 'lpips', 'evaluation_fingerprint'))
        for view in metrics['views']:
            rgb = Path(view['name']).stem+'_rgb.png'
            assert digest(source.parent/'evaluation_native'/rgb) == digest(folder/'evaluation_native'/rgb)
    log = [json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines()]
    report = {'status': 'passed', 'stage': args.stage, 'step': state['step'], 'checkpoint_sha256': receipt['checkpoint_sha256'],
              'source_tree_sha256': plan['source_tree_sha256'], 'profile': 'colmap_corner_v2',
              'manifest_sha256': state['manifest_sha256'], 'native_fingerprint': metrics['evaluation_fingerprint'],
              'effective_configuration': state['config'],
              'no_intermediate_step_checkpoints_or_validation': True,
              'fresh_rgb_from_new_seed': args.stage == 'rgb',
              'semantic_frozen_rgb_geometry_keys': frozen,
              'semantic_rgb_all50_png_and_metrics_exact_to_source': True if args.stage == 'semantic' else None,
              'gaussians': metrics['gaussians'], 'elapsed_through_last_training_step': log[-1]['elapsed_seconds'],
              'peak_allocated_gpu_gb': log[-1]['peak_gpu_gb'],
              'semantic_metrics_scope': ('Not a trained semantic endpoint; no semantic selection at RGB stage'
                                         if args.stage == 'rgb' else 'Fixed 8000-step supervised endpoint'),
              'metrics': {key: metrics[key] for key in ('psnr', 'ssim', 'lpips', 'miou_all', 'miou_foreground', 'iou', 'semantic_3d')},
              'interpretation': 'Separate corrected-coordinate native protocol; no direct legacy-native paired bootstrap or causal gain claim.'}
    (folder/'stage_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
