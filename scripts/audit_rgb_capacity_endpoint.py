"""Read-only CPU audit for the fixed 500k-to-1M capacity control.

Uses the already-tested RGB endpoint cost reconstruction. Never reads an active
checkpoint, decodes pixels, renders, or conflates native and official metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from audit_rgb140_reference_endpoint import digest, finite_tensors, read, reconstruct_cost, require

PLAN_SHA = '40191687627d54da91b78c8b819e83bc04b04b986ae6568efe042c6dbcd447f3'
REFERENCE_PLAN_SHA = '81b378c4d58b586d22d4eb3718a9f678f8a6650d5d0003b4f5e01c28b19e6b6e'
HELPER_SHA = '8d174a9ef90fc42a0261cc867bc88ea2f2a30a132404e03eed96b3063d24cf08'
PAIR_HELPER_SHA = 'bbb2482fad1203c642216bdad799ae790895bf4582df6105ef6fb25263a5c90c'
DEFAULT_RUN = Path('/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1')


def completed(directory):
    """Only these small receipts are opened before the natural-completion gate."""
    launch = read(directory/'launch_receipt.json')
    require(launch.get('status') == 'completed' and launch.get('exit_code') == 0
            and launch.get('plan_sha256') == PLAN_SHA, 'Wait for natural outer exit0; checkpoint unopened')
    records = {stage: read(directory/f'{stage}_receipt.json') for stage in ('train', 'native')}
    for stage, value in records.items():
        require(value.get('status') == 'completed' and value.get('plan_sha256') == PLAN_SHA
                and value.get('stage') == stage and value.get('original_render_restored') is True,
                'Both train and native must complete; checkpoint unopened')
        require(digest(directory/f'{stage}_receipt.json') == launch[f'{stage}_receipt_sha256'],
                'Stage receipt changed after natural completion')
    return launch, records


def audit_growth(rows):
    n, events, fills = 62000, [], 0
    for row in rows:
        step = row['step']
        event = (500 <= step < 24000 and step % 100 == 0
                 and not any(reset <= step < reset+400 for reset in (6000, 12000, 18000)))
        if event:
            cap = round(62000+(step-500)/23500*(1000000-62000))
            allocation = min(12000, max(1, round(n*.06)))
            require(row['density_target_budget'] == cap and row['density_budget'] == allocation,
                    'Linear growth schedule/allocation differs')
            added = row['splits']+row['duplicates']
            require(0 <= added <= allocation and 0 <= row['pruned'] <= n
                    and row['gaussians'] == n+added-row['pruned'] <= cap,
                    'Grow/prune accounting or capacity differs')
            fills += row['gaussians'] == cap
            events.append({'step': step, 'before': n, 'after': row['gaussians'], 'budget': cap,
                           'added': added, 'pruned': row['pruned']})
        else:
            require(row['gaussians'] == n, 'Point count changed outside the original growth window')
        n = row['gaussians']
    require(len(events) == 223 and events[-1]['step'] == 23900 and n <= 996009,
            'Wrong event count/final scheduled cap')
    return {'events': events, 'events_filled_to_scheduled_budget': fills,
            'last_growth_step': 23900, 'final_scheduled_cap': 996009, 'actual_final_N': n}


def observed_cost(rows, dimensions):
    cost = reconstruct_cost(rows, dimensions, diagnostic_every=20)
    width, height = dimensions[0]
    by_step = {r['step']: r['gaussians'] for r in rows}
    n, product = 62000, 0
    for step in range(1, 30001):
        scale = .5 if step/30000 < .15 else (.75 if step/30000 < .6 else 1.)
        product += n*max(8, round(width*scale))*max(8, round(height*scale))
        if 500 <= step < 24000 and step % 20 == 0:
            small = min(scale, 320/width)
            product += 2*n*max(8, round(width*small))*max(8, round(height*small))
        n = by_step.get(step, n)
    expected = {'render_calls': cost['all_training_and_diagnostic_renders'],
                'gaussian_render_sum': cost['all_render_gaussian_sum'],
                'canvas_pixel_sum': cost['all_render_pixels'], 'gaussian_canvas_pixel_sum': product,
                'peak_pre_render_gaussians': cost['peak_logged_post_update_or_primary_N']}
    return expected, {k: v for k, v in cost.items() if k != 'boundary_totals'}


def audit_state(state, reference, config, manifest):
    expected = dict(reference['config'], max_gaussians=1000000, output=config['output'])
    require(state['config'] == expected and state['step'] == 30000
            and state['format_version'] == reference['format_version'], 'Wrong effective config/terminal step')
    require(set(state) == set(reference), 'Checkpoint resume schema changed')
    n = len(state['model']['splats.means'])
    require(0 < n <= 996009 and state['stats']['gaussians'] == n, 'Invalid endpoint Gaussian count')
    require(set(state['model']) == set(reference['model']), 'Model schema changed')
    for key, value in state['model'].items():
        old = reference['model'][key]
        shape = ((n, *old.shape[1:]) if key.startswith('splats.') or key == 'semantic_prior_counts' else old.shape)
        require(tuple(value.shape) == tuple(shape) and value.dtype == old.dtype, 'Model shape/dtype differs: '+key)
        if key.startswith(('semantic_decoder.', 'refiner.')):
            require(torch.equal(value, old), 'Disabled semantic head changed')
    for key in ('feature_dim', 'sh_degree', 'scene_scale', 'refiner_config', 'pixel_protocol', 'manifest_sha256'):
        require(state[key] == reference[key], 'Shared representation/profile metadata differs: '+key)
    views = [v for v in manifest['views'] if v['split'] == 'train']
    cameras = torch.tensor(np.asarray([v['w2c_original'] for v in views]), dtype=torch.float32)
    require(len(views) == 350 and torch.equal(cameras, state['training_cameras'])
            and torch.equal(cameras, reference['training_cameras']), 'TRAIN cameras are not identical')
    require(set(state['optimizers']) == set(reference['optimizers']), 'Optimizer schema differs')
    for name, optimizer in state['optimizers'].items():
        old = reference['optimizers'][name]
        require(optimizer['param_groups'] == old['param_groups'] and set(optimizer['state']) == set(old['state']),
                'Adam groups/state IDs differ: '+name)
        for entry in optimizer['state'].values():
            require(set(entry) == {'step', 'exp_avg', 'exp_avg_sq'} and float(entry['step']) == 30000,
                    'Incomplete/stale Adam moment contract: '+name)
            shape = state['model']['background_logits'].shape if name == 'heads' else state['model']['splats.'+name].shape
            require(entry['exp_avg'].shape == entry['exp_avg_sq'].shape == shape
                    and entry['exp_avg'].dtype == entry['exp_avg_sq'].dtype == torch.float32,
                    'Adam moment shape/dtype differs: '+name)
    density = state['density_state']; old_density = reference['density_state']
    require(set(density) == set(old_density) and density['fused_target'] is None, 'Density/fusion schema differs')
    extras = density['extras']
    require(set(extras) == set(old_density['extras']), 'Density extras schema differs')
    for container, old, keys in ((density, old_density, ('scores', 'observations', 'counts')),
                                 (extras, old_density['extras'], ('radii', 'residual_scores', 'residual_observations', 'residual_counts'))):
        for key in keys:
            require(container[key].shape == (n,) and container[key].dtype == old[key].dtype
                    and bool((container[key] >= 0).all()), 'Invalid density window: '+key)
    for support in (density['observed_views'], extras['residual_observed_views']):
        require(all(isinstance(k, int) and 0 <= k < 350 and v.shape == (n,) and v.dtype == torch.bool
                    for k, v in support.items()), 'Wrong density view-mask schema')
    require(torch.equal(extras['camera_quality'], torch.ones(350))
            and extras['pause_until'] == 18400 and extras['gradient_convention'] == 'gsplat_normalized_v1',
            'Camera weighting/reset/gradient convention changed')
    generator = np.random.RandomState(42)
    for _ in range(86):
        order = generator.permutation(350).tolist()
    require(all(np.array_equal(a, b) for a, b in zip(generator.get_state(), state['numpy_rng'], strict=True))
            and extras['sampler_order'] == order and extras['sampler_cursor'] == 250
            and extras['sampler_rng_state'] is None and state['config']['independent_view_rng'] is False,
            'Original shuffle terminal RNG/order/cursor differs')
    for key in ('torch_rng', 'cuda_rng'):
        require(state[key].shape == reference[key].shape and state[key].dtype == torch.uint8,
                'Missing/incompatible RNG resume state')
    return {key: finite_tensors(state[key]) for key in ('model', 'optimizers', 'density_state', 'training_cameras')}


def audit_runtime_imports(stage, record, snapshot, expected_sources):
    """Attest the actual frozen CLI dependency graph for each completed stage."""
    common = {'bridge_rgs.cli', 'bridge_rgs.model', 'bridge_rgs.io', 'bridge_rgs.losses'}
    specific = {'train': {'bridge_rgs.train', 'bridge_rgs.densification'},
                'native': {'bridge_rgs.train', 'bridge_rgs.evaluate'}}
    require(stage in specific, 'Unknown runtime stage')
    # The historical 30-module package loads images via io, not the later data
    # module. Still hash EVERY attested import, including non-core dependencies.
    for name, value in record['actual_imports'].items():
        path = Path(value['path']); rel = str(path.relative_to(snapshot))
        require(value['sha256'] == expected_sources[rel] == digest(path), 'Runtime source differs: '+name)
    require(common | specific[stage] <= record['actual_imports'].keys(),
            'Core runtime imports are not attested for '+stage)


def audit(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    require(not output.exists(), 'Preserve existing endpoint audit')
    launch, records = completed(directory)  # BEFORE any checkpoint stat, hash, or load.
    require(not torch.cuda.is_initialized(), 'CPU-only audit')
    torch.set_num_threads(8)
    helper = Path(__file__).with_name('audit_rgb140_reference_endpoint.py')
    require(digest(helper) == HELPER_SHA and digest(helper.with_name('compare_evaluations.py')) == PAIR_HELPER_SHA,
            'Reused CPU reconstruction helper changed')
    require(digest(directory/'plan.json') == PLAN_SHA, 'Wrong frozen capacity plan')
    plan = read(directory/'plan.json'); config = plan['config']
    reference_path = Path(plan['reference_directory']); snapshot = Path(plan['source_snapshot'])
    require(digest(reference_path/'plan.json') == REFERENCE_PLAN_SHA, 'Wrong reference plan')
    old_plan = read(reference_path/'plan.json')
    require({k for k in config.keys() | old_plan['config'].keys() if config.get(k) != old_plan['config'].get(k)}
            == {'output', 'max_gaussians'} and config['max_gaussians'] == 1000000,
            'Capacity comparison changed other settings')
    require(yaml.safe_load(Path(plan['config_path']).read_text()) == config
            and digest(plan['config_path']) == plan['config_sha256'], 'Bound YAML differs')
    expected_sources = old_plan['source_hashes']
    require(len(expected_sources) == 30 and plan['source_hashes'] == expected_sources
            == {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')}, 'Package not byte-identical')
    require(digest(plan['runner']) == plan['runner_sha256'], 'Frozen observer/runner changed')
    for path, value in plan['input_hashes'].items():
        require(digest(path) == value, 'Bound input changed: '+path)
    require(records['train']['arguments'] == ['bridge_rgs.cli', 'train', '--config', plan['config_path']]
            and records['native']['arguments'] == ['bridge_rgs.cli', 'evaluate', '--checkpoint',
                str(Path(plan['training_output'])/'last.pt'), '--manifest', config['manifest'], '--output',
                str(Path(plan['training_output'])/'evaluation_native'), '--lpips'], 'Actual CLI arguments differ')
    for stage, record in records.items():
        require(record['environment'] == plan['environment'], 'Runtime environment differs')
        audit_runtime_imports(stage, record, snapshot, expected_sources)
    training = Path(plan['training_output']); checkpoint = training/'last.pt'
    require(digest(checkpoint) == launch['checkpoint_sha256'], 'Final model bytes differ')
    require(read(reference_path/'experiment_receipt.json')['status'] == 'completed'
            and digest(reference_path/'last.pt') == read(reference_path/'stage_audit.json')['checkpoint_sha256'],
            'Reference model is incomplete/changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    old_state = torch.load(reference_path/'last.pt', map_location='cpu', weights_only=False, mmap=True)
    manifest_path = Path(plan['root'])/config['manifest']; manifest = read(manifest_path)
    require(digest(manifest_path) == state['manifest_sha256'], 'Checkpoint manifest differs')
    finite_counts = audit_state(state, old_state, config, manifest)
    require(read(training/'config.json') == state['config'], 'Saved effective config differs')
    rows = [json.loads(line) for line in (training/'train.jsonl').read_text().splitlines()]
    views = [v for v in manifest['views'] if v['split'] == 'train']
    require([v['name'] for v in views] == plan['train_names_in_manifest_order'], 'TRAIN population/order differs')
    expected_counts, cost = observed_cost(rows, [(v['width'], v['height']) for v in views])
    growth = audit_growth(rows)
    require(records['train']['counts'] == expected_counts and cost['final_N'] == len(state['model']['splats.means']),
            'Observed train costs and independent reconstruction disagree')
    metrics_path = training/'evaluation_native/metrics.json'
    require(digest(metrics_path) == launch['native_metrics_sha256'], 'Native metrics changed')
    metrics, old_metrics = read(metrics_path), read(reference_path/'evaluation_native/metrics.json')
    require(metrics['evaluation_fingerprint'] == old_metrics['evaluation_fingerprint']
            and metrics['pixel_protocol']['id'] == 'colmap_corner_v2' and metrics['scale'] == 1
            and metrics['validation_views'] == len(metrics['views']) == 50 and metrics['semantic_validation_views'] == 41
            and sorted(v['name'] for v in metrics['views']) == plan['val_names_sorted']
            and metrics['gaussians'] == cost['final_N'], 'Incompatible/incomplete native endpoint')
    n = cost['final_N']; pixels = sum(v['width']*v['height'] for v in metrics['views'])
    expected_native = {'render_calls': 50, 'gaussian_render_sum': n*50, 'canvas_pixel_sum': pixels,
                       'gaussian_canvas_pixel_sum': n*pixels, 'peak_pre_render_gaussians': n}
    require(records['native']['counts'] == expected_native, 'Native observer counts differ')
    require(all(np.isfinite(v[key]) for v in metrics['views'] for key in ('psnr', 'ssim', 'lpips')),
            'Nonfinite native RGB metrics')
    require(not list(training.glob('step_*.pt')) and not list(training.glob('validation_*'))
            and not list(training.glob('metrics_*.json')), 'Unexpected intermediate evaluation/checkpoint selection')
    for path, value in plan['input_hashes'].items():
        require(digest(path) == value, 'Input changed during audit: '+path)
    require(expected_sources == {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')},
            'Source changed during audit')
    require(digest(checkpoint) == launch['checkpoint_sha256'] and not torch.cuda.is_initialized(), 'Endpoint changed/CUDA initialized')
    result = {'status': 'passed', 'cpu_only': True, 'plan_sha256': PLAN_SHA,
              'checkpoint_sha256': launch['checkpoint_sha256'], 'source_hashes': expected_sources,
              'only_config_differences': ['output', 'max_gaussians'], 'step': 30000, 'gaussians': n,
              'all_model_optimizer_density_tensors_finite': True, 'finite_tensor_counts': finite_counts,
              'parameter_optimizer_density_rng_resume_schema_checked': True, 'original_350_train_cameras_exact': True,
              'source_input_bytes_exact': True, 'growth': growth, 'reconstructed_cost': cost,
              'train_observer_counts_exact_to_reconstruction': True, 'native_observer_counts_exact': True,
              'sampler_terminal_exact': True, 'sampler_limit': 'Reconstructed original global NumPy shuffle; terminal state corroborates but is not an observed per-step view trace.',
              'native_fingerprint': metrics['evaluation_fingerprint'],
              'native_rgb': {k: metrics[k] for k in ('psnr', 'ssim', 'lpips')},
              'untrained_semantics_not_ranked': True, 'official_rgb_gate_not_evaluated': True,
              'audit_runner_sha256': digest(__file__), 'cost_helper_sha256': HELPER_SHA,
              'limitations': 'CPU checks terminal resume-state schema, not a live CUDA replay. No RGB/semantic accuracy or official adoption gate is inferred.',
              'artifact_hashes': {str(p): digest(p) for p in (directory/'plan.json', directory/'launch_receipt.json',
                  directory/'train_receipt.json', directory/'native_receipt.json', checkpoint, metrics_path, training/'train.jsonl')}}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    target = args.output or args.run/'cpu_endpoint_audit.json'
    result = audit(args.run, target)
    print(json.dumps({'status': result['status'], 'output': str(target), 'sha256': digest(target)}))
