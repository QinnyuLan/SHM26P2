"""CPU audit of the fixed horizontal flip pair, with paired development-view CIs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from compare_evaluations import paired_comparison


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/refiner_flip'))
    args = parser.parse_args()
    torch.set_num_threads(8)
    plan_path = Path('configs/generated_refiner_flip/flip_plan.json')
    plan = json.loads(plan_path.read_text())
    launch = json.loads((args.root/'launch_receipt.json').read_text())
    source = json.loads(Path('runs/refiner_flip_preflight/source_provenance.json').read_text())
    assert digest(plan_path) == launch['plan_sha256']
    arms = ['00_control', '01_flip']
    receipts = [json.loads((args.root/arm/'experiment_receipt.json').read_text()) for arm in arms]
    assert all(r['status'] == 'completed' for r in receipts)
    assert receipts[0]['source_hashes'] == receipts[1]['source_hashes'] == source['source_hashes']
    assert receipts[0]['input_hashes'] == receipts[1]['input_hashes']
    configs = [r['config'] for r in receipts]
    assert {key for key in configs[0].keys() | configs[1].keys()
            if configs[0].get(key) != configs[1].get(key)} == {'output', 'refiner_horizontal_flip_probability'}
    for path, sha in launch['config_sha256'].items():
        assert digest(path) == sha
        locked = yaml.safe_load(Path(path).read_text())
        arm = Path(locked['output']).name
        assert locked == receipts[arms.index(arm)]['config']
    for value in receipts[0]['input_hashes'].values():
        assert digest(value['path']) == value['sha256']
    warmstart = Path(plan['warmstart'])
    assert digest(warmstart) == plan['warmstart_sha256']
    initial = torch.load(warmstart, map_location='cpu', weights_only=False)
    baseline_folder = warmstart.parent/'evaluation_native'
    baseline = json.loads((baseline_folder/'metrics.json').read_text())
    manifest = json.loads(Path(plan['manifest']).read_text())
    views = [v for v in manifest['views'] if v['split'] == 'train']
    population = [i for i, v in enumerate(views) if v.get('mask_path')]
    assert len(population) == plan['train_labeled_population'] == 259
    generator, sequence = np.random.default_rng(42), []
    while len(sequence) < 3000:
        order = generator.permutation(population).tolist()
        cursor = min(len(order), 3000-len(sequence))
        sequence.extend(order[:cursor])
    assert sequence == plan['stateless_plan']['view_indices']
    names = [views[index]['name'] for index in sequence]
    assert names == plan['stateless_plan']['view_names']
    planned_flags = [bool(np.random.default_rng(np.random.SeedSequence([42, step, 704291])).random() < .5)
                     for step in range(1, 3001)]
    assert planned_flags == plan['stateless_plan']['flip_flags']
    frozen = [k for k in initial['model'] if not k.startswith('refiner.')]
    rows, results, random_states = [], [], []
    for arm, receipt, probability in zip(arms, receipts, [0., .5], strict=True):
        folder = args.root/arm
        config = receipt['config']
        assert config['steps'] == 3000 and config['save_every'] == config['eval_every'] == 0
        assert config['refiner_horizontal_flip_probability'] == probability
        assert config['parameter_scope'] == 'refiner_only' and config['refiner']['depth_moments'] == 'off'
        assert not config['warmstart_reset_refiner'] and not config['warmstart_add_depth_moments']
        assert 'semantic_lr_schedule' not in config
        assert all(digest(folder/'source_snapshot'/key) == sha for key, sha in receipt['source_hashes'].items())
        assert digest(folder/'last.pt') == receipt['checkpoint_sha256']
        checkpoint = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
        assert checkpoint['step'] == 3000 and checkpoint['model'].keys() == initial['model'].keys()
        assert len(checkpoint['model']) == plan['warmstart_model_tensor_count'] == 131
        assert all(checkpoint['model'][key].dtype == tensor.dtype
                   and checkpoint['model'][key].shape == tensor.shape for key, tensor in initial['model'].items())
        assert all(torch.isfinite(tensor).all() for tensor in checkpoint['model'].values())
        assert all(torch.equal(checkpoint['model'][key], initial['model'][key]) for key in frozen)
        assert torch.equal(checkpoint['training_cameras'], initial['training_cameras'])
        assert checkpoint['training_cameras'].dtype == initial['training_cameras'].dtype
        assert all(checkpoint[key] == initial[key] for key in ('scene_scale', 'feature_dim', 'sh_degree', 'refiner_config'))
        changed = [key for key in initial['model'] if not torch.equal(initial['model'][key], checkpoint['model'][key])]
        assert changed and all(key.startswith('refiner.') for key in changed)
        for name, optimizer in checkpoint['optimizers'].items():
            if name != 'heads':
                assert not optimizer['state']
            else:
                refiner_parameters = set(optimizer['param_groups'][1]['params'])
                assert set(optimizer['state']) == refiner_parameters
                assert optimizer['param_groups'][1]['lr'] == .0003
                assert all(float(state['step']) == 3000 for state in optimizer['state'].values())
        extra = checkpoint['density_state']['extras']
        assert extra['sampler_order'] == order and extra['sampler_cursor'] == cursor
        assert extra['sampler_rng_state'] == generator.bit_generator.state
        random_states.append({key: checkpoint[key] for key in ('numpy_rng', 'torch_rng', 'cuda_rng')})
        logs = [json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines()]
        assert [line['step'] for line in logs] == [1] + list(range(100, 3001, 100))
        for line in logs:
            step = line['step']
            assert line['refiner_crop'] is None and line['refiner_crop_view'] == names[step-1]
            assert line['refiner_supervised_pixels'] > 0 and line['sh_degree'] == 3
            assert 'semantic_lr_multiplier' not in line
            if probability:
                assert line['refiner_horizontal_flip'] == planned_flags[step-1]
                assert line['refiner_horizontal_flip_view'] == names[step-1]
            else:
                assert 'refiner_horizontal_flip' not in line
        metric_path = folder/'evaluation_native/metrics.json'
        assert digest(metric_path) == receipt['evaluation_sha256']
        result = json.loads(metric_path.read_text())
        assert result['scale'] == 1.0 and result['lpips'] is not None
        assert len(result['views']) == 50
        assert sum('confusion_matrix' in view for view in result['views']) == 41
        assert result['confusion_matrix_3d'] == baseline['confusion_matrix_3d']
        old_views = {view['name']: view for view in baseline['views']}
        assert len(old_views) == 50
        for view in result['views']:
            old = old_views[view['name']]
            assert (view['width'], view['height']) == (1320, 989)
            assert view.get('confusion_matrix_3d') == old.get('confusion_matrix_3d')
            png = Path(view['name']).stem+'_rgb.png'
            assert digest(folder/'evaluation_native'/png) == digest(baseline_folder/png)
        results.append(result)
        rows.append({'arm': arm, 'flip_probability': probability,
                     'planned_flip_steps': sum(planned_flags) if probability else 0,
                     'miou_all': result['miou_all'], 'miou_foreground': result['miou_foreground'],
                     'iou': result['iou'], 'raw_3d': result['semantic_3d'],
                     'cable_boundary_f1': result['boundary_f1_2px'][2],
                     'psnr': result['psnr'], 'ssim': result['ssim'], 'lpips': result['lpips'],
                     'changed_refiner_tensors': len(changed), 'frozen_tensors': len(frozen),
                     'elapsed_through_last_training_step': logs[-1]['elapsed_seconds'],
                     'peak_training_allocated_gpu_gb': logs[-1]['peak_gpu_gb'],
                     'checkpoint_sha256': receipt['checkpoint_sha256']})
        del checkpoint
    assert exact(random_states[0], random_states[1])
    pairs = {}
    for reference, candidate, name in [(results[0], results[1], 'flip_minus_control'),
                                       (baseline, results[0], 'control_minus_warmstart'),
                                       (baseline, results[1], 'flip_minus_warmstart')]:
        paired = paired_comparison(reference, candidate)
        (args.root/f'paired_{name}.json').write_text(json.dumps(paired, indent=2)+'\n')
        pairs[name] = paired['metrics']
    report = {'status': 'completed', 'fixed_endpoint': 3000, 'primary_pair': 'flip_minus_control',
              'report_script_sha256': digest(__file__),
              'comparison_script_sha256': digest(Path(__file__).with_name('compare_evaluations.py')),
              'source_tree_sha256': plan['snapshot_tree_sha256'], 'same_source_and_inputs': True,
              'only_config_differences': ['output', 'refiner_horizontal_flip_probability'],
              'all_131_model_keys_and_architecture_preserved': True,
              'all_nonrefiner_field_classifier_prior_camera_tensors_exact_to_warmstart': True,
              'all_50_rgb_pngs_and_41_raw3d_confusion_matrices_exact_to_warmstart': True,
              'frozen_optimizer_states_empty_head_step3000': True,
              'torch_cuda_numpy_rng_exact_between_arms': True,
              'sampler_final_order_cursor_rng_exact_to_fixed3000_plan': True,
              'all_logged_view_names_and_flip_flags_exact_to_stateless_plan': True,
              'full_step_audit_limit': 'Formal logs at step1 and every100; full sequence reconstructed from immutable sampler and final saved state. Eight-step preflight directly checked each step.',
              'train_population': 259, 'visits_per_view_min_max': [11, 12],
              'same_full_native_supervised_pixel_budget': True,
              'warmstart_metrics_sha256': digest(baseline_folder/'metrics.json'),
              'warmstart_miou_all': baseline['miou_all'], 'runs': rows, 'paired': pairs,
              'limitations': ['Standard augmentation engineering control, development-adaptive selection; not a novelty claim.',
                              'One seed and one bridge; paired view bootstrap is not across-seed or across-scene uncertainty.',
                              'All41 semantic validation views including235 retained; no intermediate validation or checkpoint selection.',
                              'Same-source repeat shows comparable CUDA numerical drift to old/new off difference; precise kernel cause and long-run variance not isolated.']}
    (args.root/'flip_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'runs': rows, 'paired': pairs['flip_minus_control']}, indent=2))


if __name__ == '__main__':
    main()
