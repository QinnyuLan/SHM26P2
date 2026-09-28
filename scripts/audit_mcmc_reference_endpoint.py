"""CPU-only audit of the naturally completed fixed MCMC smoke/full endpoint.

No rasterizer, checkpoint mutation, or access to a still-running checkpoint.
Cost is independently reconstructed from the declared deterministic growth rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path('/home/sky/workspace/SHM2026')
POLICY = {'cap_max': 500000, 'noise_lr': 500000., 'refine_start_iter': 500,
              'refine_stop_iter': 25000, 'refine_every': 100, 'min_opacity': .005,
              'opacity_reg': .01, 'scale_reg': .01}
CONVENTION = 'gsplat_1.5.3_mcmc_canonical_scene_scale_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require_completed(directory):
    """This gate precedes every stat/hash/load of last.pt."""
    experiment = read(directory / 'experiment_receipt.json')
    launch = read(directory / 'launch_receipt.json')
    require(experiment.get('status') == 'completed' and experiment.get('finished_utc')
            and launch.get('status') == 'completed' and launch.get('exit_code') == 0
            and launch.get('finished_utc'), 'Wait for natural completed exit0; checkpoint unopened')
    return experiment, launch


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        require(value.device.type == 'cpu' and bool(torch.isfinite(value).all()),
                'Nonfinite or non-CPU checkpoint tensor')
        return 1
    if isinstance(value, dict):
        return sum(finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(finite_tensors(v) for v in value)
    return 0


def expected_schedule(steps, initial=62000, width=1320, height=989):
    """Independent integer N recurrence; render before, noise after growth."""
    require(steps in (610, 30000) and initial == 62000, 'Only fixed smoke/full recipe is supported')
    n = initial
    rendered = noise = 0
    refinements = 0
    records = {}
    events = []
    first_cap_step = None
    for step in range(1, steps + 1):
        rendered += n
        before = n
        refines = 500 < step < 25000 and step % 100 == 0
        if refines:
            n = min(500000, int(1.05 * n))
            refinements += 1
            events.append({'step': step, 'before': before, 'after': n, 'added': n - before})
            if n == 500000 and first_cap_step is None:
                first_cap_step = step
        noise += n
        if step == 1 or step % 100 == 0 or step == steps:
            records[step] = {'last_step': step, 'initial_gaussians': initial,
                'current_gaussians': n, 'peak_gaussians': n, 'refinement_events': refinements,
                'added_gaussians': n-initial, 'noise_steps': step, 'noise_gaussian_steps': noise,
                'rendered_gaussian_steps': rendered, 'rendered_pixels': step*width*height,
                'added_this_step': n-before}
    return records, {'initial_N': initial, 'final_N': n, 'peak_N': n,
        'primary_training_renders': steps, 'primary_render_pixels': steps*width*height,
        'render_before_topology_gaussian_steps': rendered,
        'noise_after_topology_gaussian_steps': noise, 'refinement_events': refinements,
        'first_cap_step': first_cap_step, 'events': events,
        'cost_scope': 'Training only. One RGB+ED raster per step. No diagnostic/semantic raster. '
                      'Gaussian-step and pixel counts are workload proxies, not FLOPs or elapsed time.'}


def audit_logs(rows, steps, scene_scale, state):
    expected, cost = expected_schedule(steps)
    require([r['step'] for r in rows] == list(expected), 'Missing/duplicate/unordered step1/100/final logs')
    relocated = relocation_events = 0
    for row in rows:
        step, values = row['step'], expected[row['step']]
        for name, value in values.items():
            if name not in ('rendered_gaussian_steps', 'rendered_pixels', 'added_this_step'):
                require(row['mcmc_' + name] == value, f'MCMC counter differs: {step}/{name}')
        require(row['gaussians'] == values['current_gaussians']
                and row['reference_gaussian_steps'] == values['rendered_gaussian_steps']
                and row['reference_render_pixels'] == values['rendered_pixels']
                and row['reference_peak_gaussians'] == values['peak_gaussians'], 'Training cost counter differs')
        count = row['mcmc_relocated_this_step']
        require(isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                'Invalid relocation count')
        refines = 500 < step < 25000 and step % 100 == 0
        before = values['current_gaussians'] - values['added_this_step']
        require(count < before and (refines or count == 0), 'Unexpected/all-dead relocation')
        relocated += count
        relocation_events += int(count > 0)
        require(row['mcmc_relocated_gaussians'] == relocated
                and row['mcmc_relocation_events'] == relocation_events
                and row['mcmc_added_this_step'] == values['added_this_step']
                and row['mcmc_topology_changed'] == bool(count or values['added_this_step']),
                'Relocation/addition accounting differs')
        passed_lr = 1.6e-4 * (.01 ** (step / steps)) / scene_scale
        require(math.isclose(row['mcmc_passed_noise_lr'], passed_lr, rel_tol=1e-12)
                and math.isclose(row['mcmc_noise_scaler'], passed_lr * 5e5, rel_tol=1e-12),
                'Canonical covariance-noise LR differs')
        require(row['mcmc_finite_check_step'] == step and row['semantic_loss'] == 0
                and row['sh_degree'] == min(3, step // 1000), 'Finite/semantic/SH schedule differs')
        require(all(math.isfinite(row[key]) for key in ('loss', 'rgb_loss', 'elapsed_seconds',
                    'peak_gpu_gb', 'mcmc_opacity_regularization', 'mcmc_scale_regularization')),
                'Nonfinite scalar diagnostic')
    counters = state['mcmc_reference_state']['counters']
    for key, value in counters.items():
        require(rows[-1]['mcmc_' + key] == value, 'Saved MCMC counter differs from final log')
    for key in ('step', 'gaussians', 'reference_gaussian_steps', 'reference_render_pixels',
                'reference_peak_gaussians', 'mcmc_finite_check_step'):
        require(state['stats'][key] == rows[-1][key], 'Saved endpoint stats are stale')
    cost.update(relocation_events=relocation_events, relocated_gaussians=relocated,
                last_logged_training_seconds=rows[-1]['elapsed_seconds'],
                peak_allocated_GiB_rounded=max(r['peak_gpu_gb'] for r in rows))
    return cost


def audit_sampler(views, rows, extras, steps):
    """Reconstruct independent default_rng shuffle, never consume saved RNG."""
    generator = np.random.default_rng(42)
    by_step = {r['step']: r['mcmc_training_view'] for r in rows}
    order, cursor = [], 0
    for step in range(1, steps + 1):
        if cursor >= len(order):
            order, cursor = generator.permutation(len(views)).tolist(), 0
        index = order[cursor]
        cursor += 1
        if step in by_step:
            require(views[index]['name'] == by_step[step], 'Logged view differs from fixed shuffle')
    require(extras['sampler_order'] == order and extras['sampler_cursor'] == cursor
            and extras['sampler_rng_state'] == generator.bit_generator.state,
            'Saved independent view sampler differs from reconstructed endpoint')
    return {'seed': 42, 'population': len(views), 'completed_full_cycles': steps//len(views),
            'cursor': cursor, 'logged_names_and_endpoint_state_match': True,
            'evidence_limit': 'Full order is reconstructed from source/seed; only logged boundaries '
                              'are independently observed, not an every-step view trace.'}


def audit_state(state, config, manifest, manifest_sha, initial_count):
    n, steps = len(state['model']['splats.means']), config['steps']
    require(state['step'] == steps and all(state['config'].get(k) == v for k, v in config.items()),
            'Checkpoint effective configuration/step differs')
    require(state['manifest_sha256'] == manifest_sha and state['pixel_protocol']['id'] == 'colmap_corner_v2',
            'Checkpoint manifest/profile differs')
    require(state.get('mip_filter_config') is None and state.get('mip_filter_state') is None
            and 'mip_filter_rho' not in state['model'], 'Unexpected Mip state')
    require(state['mcmc_reference_config'] == POLICY, 'MCMC policy changed')
    strategy = state['mcmc_reference_state']
    require(strategy['policy'] == POLICY and strategy['convention'] == CONVENTION
            and strategy['gsplat_version'] == '1.5.3' and strategy['scene_scale'] == state['scene_scale'],
            'Strategy metadata differs')
    expected_binoms = torch.zeros(51, 51, dtype=torch.float32)
    for i in range(51):
        for j in range(i + 1):
            expected_binoms[i, j] = math.comb(i, j)
    require(strategy['binoms'].dtype == torch.float32 and torch.equal(strategy['binoms'], expected_binoms),
            'Binomial table differs')
    shape_tails = {'means': (3,), 'quats': (4,), 'log_scales': (3,), 'opacity_logits': (), 'sh0': (1, 3),
                       'sh_rest': ((state['sh_degree']+1)**2-1, 3), 'sem_features': (state['feature_dim'],)}
    require({key.removeprefix('splats.') for key in state['model'] if key.startswith('splats.')}
            == set(shape_tails), 'Unexpected splat fields')
    for name, tail in shape_tails.items():
        parameter = state['model']['splats.' + name]
        require(parameter.shape == (n, *tail) and parameter.dtype == torch.float32,
                'Splat shape/dtype mismatch: ' + name)
        optimizer = state['optimizers'][name]
        require(len(optimizer['param_groups']) == 1 and len(optimizer['param_groups'][0]['params']) == 1,
                'Invalid splat Adam parameter grouping')
        group = optimizer['param_groups'][0]
        require(group['eps'] == 1e-15 and tuple(group['betas']) == (.9, .999), 'Adam convention changed')
        identifier = group['params'][0]
        require(set(optimizer['state']) <= {identifier}, 'Orphaned Adam parameter state')
        entry = optimizer['state'].get(identifier, {})
        if name != 'sem_features':
            require(set(entry) == {'step', 'exp_avg', 'exp_avg_sq'}
                    and int(entry['step']) == steps, 'Missing/stale active Adam moments: ' + name)
            require(entry['exp_avg'].shape == parameter.shape and entry['exp_avg_sq'].shape == parameter.shape,
                    'Topology/Adam moment shape mismatch: ' + name)
        else:
            require(not entry, 'Unsupervised semantic feature received optimizer state')
    require(state['optimizers']['opacity_logits']['param_groups'][0]['lr'] == .05,
            'Opacity LR changed')
    expected_lr = 1.6e-6 * state['scene_scale']
    require(math.isclose(state['optimizers']['means']['param_groups'][0]['lr'], expected_lr, rel_tol=1e-12),
            'Final means LR/horizon differs')
    prior = state['model']['semantic_prior_counts']
    require(prior.shape == (n, 5) and not bool(prior.any()), 'Semantic prior is populated or stale')
    density = state['density_state']
    for key in ('scores', 'observations', 'counts'):
        require(density[key].shape == (n,) and not bool(density[key].any()), 'Active/stale old density window')
    require(density['observed_views'] == {} and density['fused_target'] is None
            and 'mixed_gradient' not in density, 'Old strategy/fusion state is active')
    extras = density['extras']
    for key in ('radii', 'residual_scores', 'residual_observations', 'residual_counts'):
        require(extras[key].shape == (n,) and not bool(extras[key].any()), 'Stale/active residual density buffer')
    require(extras['residual_observed_views'] == {} and extras['pause_until'] == 0,
            'Unexpected residual observations or reset pause')
    views = [v for v in manifest['views'] if v['split'] == 'train']
    require(len(views) == len({v['name'] for v in views}) == 350, 'TRAIN population changed')
    cameras = torch.tensor(np.asarray([v['w2c_original'] for v in views]), dtype=torch.float32)
    require(torch.equal(state['training_cameras'], cameras), 'Original TRAIN cameras changed')
    require(torch.equal(extras['camera_quality'], torch.ones(350)), 'Unexpected camera quality weighting')
    require(strategy['counters']['initial_gaussians'] == initial_count, 'Wrong initial N')
    require(state['torch_rng'].dtype == torch.uint8 and state['cuda_rng'].dtype == torch.uint8,
            'Missing RNG state')
    return {key: finite_tensors(state[key]) for key in ('model', 'optimizers', 'density_state',
                                                     'mcmc_reference_state', 'training_cameras')}


def verify_bound(directory, plan, receipt):
    source = Path(plan['source_snapshot'])
    hashes = {str(p.relative_to(source)): digest(p) for p in source.rglob('*.py')}
    require(hashes == plan['source_hashes'] == receipt['source_hashes'], 'Frozen runtime sources changed')
    require(digest(directory/'training_config.yaml') == plan['config_sha256'], 'Config bytes changed')
    for path, record in plan['inputs'].items():
        require(Path(path).stat().st_size == record['bytes'] and digest(path) == record['sha256'],
                'Bound input changed: ' + path)
    return hashes


def audit(directory, expected_plan_sha256, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    require(not output.exists(), 'Refuse overwriting audit')
    receipt, launch = require_completed(directory)
    require(not torch.cuda.is_initialized(), 'CPU-only endpoint audit')
    torch.set_num_threads(8)
    plan_path = directory/'plan.json'
    require(digest(plan_path) == expected_plan_sha256 == launch['plan_sha256'], 'Plan SHA differs')
    plan, config = read(plan_path), read(directory/'plan.json')['config']
    require(config == receipt['config'] == yaml.safe_load((directory/'training_config.yaml').read_text())
            and receipt['resume'] is None, 'Runtime configuration/resume differs')
    smoke = plan['smoke']
    steps = 610 if smoke else 30000
    require(config['steps'] == steps and receipt['completed_commands'] == (1 if smoke else 2)
            and len(receipt['commands']) == (1 if smoke else 2), 'Incomplete fixed execution sequence')
    require(config['densification'] == 'mcmc_reference' and config['mcmc_reference'] == POLICY
            and config['max_gaussians'] == 500000 and config['background_points'] == 2000
            and config['initial_opacity'] == .5 and config['initial_scale_multiplier'] == .1
            and config['train_scale'] == 1 and config['progressive_resolution'] is False
            and config['seed'] == config['view_sampler_seed'] == 42
            and config['independent_view_rng'] is True and config['view_sampling'] == 'shuffle'
            and config['log_every'] == 100 and config['eval_every'] == 0
            and config['semantic_start'] > steps and config['refine_start'] > steps,
            'Unexpected MCMC reference recipe')
    for key in ('warmstart', 'optimize_cameras', 'camera_attribution', 'camera_quality_weighting',
                'region_rgb_weight', 'semantic_weight', 'semantic_geometry_weight', 'semantic_initialization',
                'pseudo_dir', 'multiview_fusion', 'structure_guided', 'recycle_count', 'sparse_depth_weight',
                'sparse_front_weight', 'opacity_entropy_weight', 'train_labeled_only', 'mixed_gradient', 'mip_filter'):
        require(not config.get(key), 'Unexpected active component: ' + key)
    require(plan['validation_during_training'] is False and plan['mask_supervision'] is False
            and plan['native_evaluation_after_training'] == (not smoke), 'Evaluation/mask scope changed')
    source_hashes = verify_bound(directory, plan, receipt)
    source = Path(plan['source_snapshot'])/'bridge_rgs'
    train_source = (source/'train.py').read_text()
    for clause in ('source = dict(view, mask_path=None) if self.ignore_masks else view',
                   'ImageCache(ignore_masks=rgb_reference)', 'if "semantic_counts" in points and not rgb_reference',
                   'absgrad=mcmc_state is None', 'if mcmc_state is None:'):
        require(clause in train_source, 'Audited no-mask/MCMC source branch missing: ' + clause)
    manifest_path = (ROOT/config['manifest']).resolve()
    manifest = read(manifest_path)
    require(str(manifest_path) in plan['inputs'], 'Manifest not bound')
    views = [v for v in manifest['views'] if v['split'] == 'train']
    train_inputs = {str(Path(v[k]).resolve()) for v in views for k in ('image_path', 'valid_path')}
    require(train_inputs <= set(plan['inputs']) and not any('/masks/' in p for p in plan['inputs']),
            'Missing TRAIN RGB/valid bytes or mask input bound')
    require({(v['width'], v['height']) for v in views} == {(1320, 989)}, 'Native dimensions changed')
    initial_path = (ROOT/manifest['init_points_path']).resolve()
    require(str(initial_path) in plan['inputs'], 'Initialization not bound')
    with np.load(initial_path) as initial:
        initial_count = len(initial['points']) + 2000
    require(initial_count == 62000, 'Wrong initialization count')
    checkpoint_path = directory/'last.pt'  # Only reached after the completed gate.
    checkpoint_sha = digest(checkpoint_path)
    require(checkpoint_sha == receipt['checkpoint_sha256'], 'Checkpoint bytes differ from receipt')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    require(state['config'] == read(directory/'config.json'), 'Saved effective config differs')
    finite = audit_state(state, config, manifest, digest(manifest_path), initial_count)
    rows = [json.loads(line) for line in (directory/'train.jsonl').read_text().splitlines()]
    cost = audit_logs(rows, steps, state['scene_scale'], state)
    require(len(state['model']['splats.means']) == cost['final_N'], 'Final field count differs')
    sampler = audit_sampler(views, rows, state['density_state']['extras'], steps)
    native = None
    artifacts = [plan_path, directory/'training_config.yaml', directory/'config.json', checkpoint_path,
                 directory/'train.jsonl', directory/'experiment_receipt.json', directory/'launch_receipt.json']
    if not smoke:
        metrics_path = directory/'evaluation_native/metrics.json'
        require(digest(metrics_path) == receipt['evaluation_sha256'], 'Native metrics bytes differ')
        metrics = read(metrics_path)
        val = [v for v in manifest['views'] if v['split'] == 'val']
        require(metrics['validation_views'] == len(metrics['views']) == len(val) == 50
                and metrics['scale'] == 1 and metrics['gaussians'] == cost['final_N']
                and metrics['pixel_protocol']['id'] == 'colmap_corner_v2'
                and {v['name'] for v in metrics['views']} == {v['name'] for v in val}
                and all((v['width'], v['height']) == (1320, 989) for v in metrics['views']),
                'Incomplete/wrong native endpoint evaluation')
        require(all(math.isfinite(v[key]) for v in metrics['views'] for key in ('psnr', 'ssim', 'lpips')),
                'Nonfinite RGB metric')
        native = {key: metrics[key] for key in ('psnr', 'ssim', 'lpips', 'evaluation_fingerprint')}
        artifacts.append(metrics_path)
    require(digest(checkpoint_path) == checkpoint_sha, 'Checkpoint changed during CPU audit')
    verify_bound(directory, plan, receipt)
    require(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA')
    result = {'status': 'passed', 'audit_utc': datetime.now(UTC).isoformat(),
        'directory': str(directory), 'smoke': smoke, 'step': steps, 'plan_sha256': expected_plan_sha256,
        'checkpoint_sha256': checkpoint_sha, 'source_hashes': source_hashes,
        'auditor_path': str(Path(__file__).resolve()), 'auditor_sha256': digest(__file__),
        'source_and_input_endpoint_hashes_match': True, 'finite_tensor_counts': finite,
        'original350_train_cameras_exact': True, 'semantic_prior_zero': True, 'mip_disabled': True,
        'parameter_and_adam_shapes_match': True, 'view_sampler': sampler, 'cost': cost,
        'native_rgb_metrics': native, 'launch_elapsed_seconds': launch['elapsed_seconds'],
        'evidence_limits': [('No-mask and no-extra-render claims use frozen source/config plus logs; '
                            'not a runtime file-open/rasterizer trace.'),
                            ('Saved Parameter dimensions/Adam IDs checked; CPU audit cannot retroactively '
                            'prove every live optimizer identity. Helper contracts and smoke cover migration.'),
                            ('No semantic quality conclusion from this RGB-only training. '
                            'Full-run official RGB gate is a separate later evaluation.')],
        'artifact_hashes': {str(p): digest(p) for p in artifacts}}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.directory/'endpoint_audit.json'
    result = audit(args.directory, args.expected_plan_sha256, output)
    print(json.dumps({'status': result['status'], 'output': str(output.resolve()),
                      'sha256': digest(output), 'cost': result['cost']}, indent=2))


if __name__ == '__main__':
    main()
