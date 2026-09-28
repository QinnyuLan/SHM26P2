"""CPU-only completed RGB endpoint audit; no active checkpoint or GPU access."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from compare_evaluations import paired_comparison

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = Path('/mnt/data/SHM2026/runs/rgb140_inspired_mixed_500k')
REFERENCE = Path('/mnt/data/SHM2026/runs/ssim_fixed_corner_v2_rgb_full')
PLAN_SHA = 'd67e9f44ab20335b4d51868ec5a42bac9a8e52f92685fb39512d2ca4961de1f9'
OLD_PLAN_SHA = '81b378c4d58b586d22d4eb3718a9f678f8a6650d5d0003b4f5e01c28b19e6b6e'
OLD_TRAIN_SHA = '99c27c929253b17e71753a6a2afb080d1cdc0e464ce8f2550d78d8177c3e04d0'
NEW_TRAIN_SHA = 'f1ce3f49d0c3fa73da2422d1786cabdf841098f21db7a94bca575f691f8ce3a3'
IO_SHA = '08fafa9f20ee07cddd867db4d4c02a33e706435ec40a13a15e22db4a7600a65d'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def require_completed(directory):
    """Must run before stat/hash/load of last.pt, even when a partial file exists."""
    receipt = read(directory/'experiment_receipt.json')
    require(receipt.get('status') == 'completed', 'Wait for completed training and native evaluation; checkpoint unopened')
    launch = read(directory/'launch_receipt.json')
    require(launch.get('status') == 'completed', 'Wait for root natural-exit confirmation; checkpoint unopened')
    for key in ['exit_code', 'observed_exit_code', 'returncode']:
        require(key not in launch or launch[key] == 0, 'Launch did not naturally succeed')
    return receipt, launch


def reconstruct_cost(rows, dimensions, *, total_steps=30000, initial_count=62000,
                     change_every=100, diagnostic_every=None, diagnostic_start=500, diagnostic_stop=24000):
    """Post-step N logs reconstruct render-before-grow N, never interpolate N."""
    expected_steps = [1] + list(range(change_every, total_steps+1, change_every))
    require([r['step'] for r in rows] == expected_steps, 'Missing/duplicate/unordered N boundary log')
    require(len(set(dimensions)) == 1, 'Unequal image dimensions require exact view-order reconstruction')
    width, height = dimensions[0]
    by_step = {r['step']: r for r in rows}
    require(by_step[1]['gaussians'] == initial_count, 'Step1 changed initial N')
    n, total_n, total_pixels, diagnostic_n, diagnostic_pixels, diagnostics = initial_count, 0, 0, 0, 0, 0
    phases = {}
    boundary_totals = {}
    for step in range(1, total_steps+1):
        scale = .5 if step/total_steps < .15 else (.75 if step/total_steps < .6 else 1.)
        w, h = max(8, round(width*scale)), max(8, round(height*scale))
        total_n += n
        total_pixels += w*h
        phase = phases.setdefault(str(scale), {'steps': 0, 'width': w, 'height': h, 'render_pixels': 0})
        phase['steps'] += 1
        phase['render_pixels'] += w*h
        if diagnostic_every and diagnostic_start <= step < diagnostic_stop and step % diagnostic_every == 0:
            dw, dh = max(8, round(width*min(scale, 320/width))), max(8, round(height*min(scale, 320/width)))
            # Audited old hybrid source: raw residual render + visibility render,
            # both no-grad and before this step's post-optimizer densification.
            diagnostics += 2
            diagnostic_n += 2*n
            diagnostic_pixels += 2*dw*dh
        if step in by_step:
            post = by_step[step]['gaussians']
            require(isinstance(post, int) and post > 0, 'Invalid post-step field count')
            n = post
            boundary_totals[step] = (total_n, total_pixels)
    return {'primary_training_renders': total_steps, 'primary_gaussian_steps': total_n,
            'primary_render_pixels': total_pixels, 'mean_primary_render_N': total_n/total_steps,
            'final_N': n, 'peak_logged_post_update_or_primary_N': max(r['gaussians'] for r in rows),
            'resolution_phases': phases, 'diagnostic_renders': diagnostics,
            'diagnostic_gaussian_render_sum': diagnostic_n, 'diagnostic_render_pixels': diagnostic_pixels,
            'all_training_and_diagnostic_renders': total_steps+diagnostics,
            'all_render_gaussian_sum': total_n+diagnostic_n,
            'all_render_pixels': total_pixels+diagnostic_pixels,
            'boundary_totals': boundary_totals}


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), 'Nonfinite checkpoint tensor')
        return 1
    if isinstance(value, dict):
        return sum(finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(finite_tensors(v) for v in value)
    return 0


def bound_run(directory, expected_plan_sha, *, mixed):
    receipt, launch = require_completed(directory)  # First guard; never open active last.
    plan_path = directory/'plan.json'
    require(digest(plan_path) == expected_plan_sha == launch['plan_sha256'], 'Plan/launch SHA changed')
    plan, config = read(plan_path), receipt['config']
    require(config == plan['config'] and receipt['resume'] is None, 'Actual config/resume differs')
    config_path = directory/('training_config.yaml' if mixed else 'replay_config.yaml')
    require(digest(config_path) == plan['config_sha256'] and yaml.safe_load(config_path.read_text()) == config,
            'Config YAML differs from plan')
    expected_sources = {'bridge_rgs/'+k.removeprefix('bridge_rgs/'): v for k, v in plan['source_hashes'].items()}
    actual_sources = {str(p.relative_to(directory/'source_snapshot')): digest(p)
                      for p in (directory/'source_snapshot').rglob('*.py')}
    require(actual_sources == receipt['source_hashes'] == expected_sources, 'Runtime package does not match frozen plan')
    source = directory/'source_snapshot/bridge_rgs'
    require(digest(source/'train.py') == (NEW_TRAIN_SHA if mixed else OLD_TRAIN_SHA)
            and digest(source/'io.py') == IO_SHA, 'Unreviewed N/schedule/resize source')
    for path, value in plan['inputs'].items():
        require(digest(path) == value['sha256'], f'Planned input changed: {path}')
    for value in receipt['input_hashes'].values():
        require(digest(value['path']) == value['sha256'], 'Runtime input changed')
    require(config['steps'] == 30000 and config['max_gaussians'] == 500000 and config['eval_every'] == 0
            and config['log_every'] == 100 and config['progressive_resolution'] and config['train_scale'] == 1
            and not config.get('warmstart') and not config.get('optimize_cameras')
            and config['semantic_start'] > 30000 and config['refine_start'] > 30000,
            'Unexpected RGB-only 30k protocol')
    if mixed:
        require(config['densification'] == 'rgb140_mixed' and not config['semantic_initialization']
                and config['region_rgb_weight'] == 0 and config['mixed_gradient']['refine_every'] == 100
                and plan['train_mask_pixels_read'] is False, 'Not the no-mask mixed reference')
        text = (source/'train.py').read_text()
        for clause in ['source = dict(view, mask_path=None) if self.ignore_masks else view',
                       'ImageCache(ignore_masks=mixed_policy is not None)',
                       '((None, None) if mixed_policy is not None',
                       'if "semantic_counts" in points and mixed_policy is None']:
            require(clause in text, 'Audited no-mask source branch missing')
    else:
        require(config['densification'] == 'hybrid' and config['densify_every'] == 100
                and config['diagnostic_every'] == 20 and not config['camera_attribution']
                and not config['camera_quality_weighting'], 'Old reconstruction source assumptions differ')
    checkpoint = directory/'last.pt'
    require(digest(checkpoint) == receipt['checkpoint_sha256'], 'Endpoint checkpoint SHA differs')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    require(state['step'] == 30000 and state['config'] == read(directory/'config.json'), 'Not exact effective30k endpoint')
    require(all(state['config'][key] == value for key, value in config.items()), 'Checkpoint protocol differs')
    finite_count = (finite_tensors(state['model']) + finite_tensors(state['optimizers'])
                    + finite_tensors(state['density_state']))
    if mixed:
        require(not bool(state['model']['semantic_prior_counts'].any()), 'Disabled semantic priors unexpectedly populated')
        require(state['density_state']['mixed_gradient']['policy'] == config['mixed_gradient'],
                'Endpoint dual-gradient policy differs')
    manifest_path = ROOT/config['manifest']
    manifest = read(manifest_path)
    train = [v for v in manifest['views'] if v['split'] == 'train']
    require(len(train) == len({v['name'] for v in train}) == 350, 'Not fixed350 TRAIN population')
    cameras = torch.tensor(np.asarray([v['w2c_original'] for v in train]), dtype=torch.float32)
    require(torch.equal(state['training_cameras'], cameras), 'Original TRAIN camera changed')
    require(state['pixel_protocol']['id'] == 'colmap_corner_v2'
            and state['manifest_sha256'] == digest(manifest_path), 'Checkpoint profile/manifest differs')
    initial = np.load(ROOT/manifest['init_points_path'])
    initial_n = len(initial['points']) + config['background_points']
    initial.close()
    rows = [json.loads(line) for line in (directory/'train.jsonl').read_text().splitlines()]
    require(all(np.isfinite(row[key]) for row in rows for key in ['loss', 'rgb_loss', 'elapsed_seconds', 'peak_gpu_gb']),
            'Nonfinite logged training scalar')
    cost = reconstruct_cost(rows, [(v['width'], v['height']) for v in train], initial_count=initial_n,
                            diagnostic_every=None if mixed else 20)
    require(cost['final_N'] == len(state['model']['splats.means']), 'Final logged/checkpoint N differs')
    if mixed:
        for row in rows:
            reconstructed = cost['boundary_totals'][row['step']]
            require(reconstructed == (row['reference_gaussian_steps'], row['reference_render_pixels']),
                    'Direct cumulative counters differ from post-N reconstruction')
        require(cost['peak_logged_post_update_or_primary_N'] == rows[-1]['reference_peak_gaussians'],
                'Direct peak N differs from logged N')
        require(all(state['stats'][key] == rows[-1][key] for key in
                    ['reference_gaussian_steps', 'reference_render_pixels', 'reference_peak_gaussians']),
                'Checkpoint cumulative counters differ from final log')
        cost['measurement_kind'] = 'Direct cumulative primary counters; independently cross-checked using source+post-step logs'
    else:
        cost['measurement_kind'] = 'Reconstructed from SHA-bound historical source+all100-step post-update N logs; not historical direct instrumentation'
    del cost['boundary_totals']
    cost.update(last_logged_training_seconds=rows[-1]['elapsed_seconds'],
                training_peak_allocated_GiB_rounded=max(row['peak_gpu_gb'] for row in rows),
                experiment_seconds_including_native_evaluation=(datetime.fromisoformat(receipt['finished_utc'])
                                                                 - datetime.fromisoformat(receipt['started_utc'])).total_seconds(),
                scope='Render counts, Gaussian sums and pixel sums are cost proxies, not exact FLOPs; primary backward vs no-grad diagnostics differ. Excludes final evaluation renders. Peak N excludes transient pre-prune construction tensors.')
    metrics_path = directory/'evaluation_native/metrics.json'
    require(digest(metrics_path) == receipt['evaluation_sha256'], 'Native metrics receipt SHA differs')
    metrics = read(metrics_path)
    val = [v for v in manifest['views'] if v['split'] == 'val']
    require(metrics['validation_views'] == len(metrics['views']) == 50 and metrics['scale'] == 1
            and {v['name'] for v in metrics['views']} == {v['name'] for v in val}
            and all((v['width'], v['height']) == (1320, 989) for v in metrics['views'])
            and metrics['pixel_protocol']['id'] == 'colmap_corner_v2', 'Incomplete native RGB grid')
    require(all(np.isfinite(v[k]) for v in metrics['views'] for k in ['psnr', 'ssim', 'lpips']), 'Nonfinite RGB metric')
    require(metrics['gaussians'] == cost['final_N'], 'Metrics do not score final field count')
    result = {'directory': str(directory), 'checkpoint_sha256': receipt['checkpoint_sha256'], 'step': 30000,
              'all_model_optimizer_density_tensors_finite': True, 'finite_tensor_count': finite_count,
              'original_train_cameras_exact': True, 'train_view_names': [v['name'] for v in train],
              'train_no_mask_loading': bool(mixed),
              'no_mask_evidence_kind': 'Pinned source/config branches, not a full runtime pixel-open trace' if mixed else 'Old run used label initialization/region weighting',
              'native_rgb_metrics': {k: metrics[k] for k in ['psnr', 'ssim', 'lpips']},
              'cost': cost, 'source_train_sha256': digest(source/'train.py'),
              'source_io_sha256': IO_SHA, 'source_and_input_hashes_verified': True,
              'artifact_hashes': {str(p): digest(p) for p in [plan_path, config_path, checkpoint, metrics_path,
                                  directory/'experiment_receipt.json', directory/'launch_receipt.json', directory/'train.jsonl']}}
    del state
    return result, metrics, manifest_path


def audit(output):
    output = Path(output).resolve()
    require(not output.exists(), 'Do not overwrite an existing audit')
    # Guard candidate before inspecting either large endpoint, regardless of caller env.
    require_completed(CANDIDATE)
    require(not torch.cuda.is_initialized(), 'This audit must be CPU-only')
    torch.set_num_threads(8)
    candidate, new_metrics, manifest_path = bound_run(CANDIDATE, PLAN_SHA, mixed=True)
    reference, old_metrics, old_manifest_path = bound_run(REFERENCE, OLD_PLAN_SHA, mixed=False)
    require(manifest_path == old_manifest_path, 'Training manifests differ')
    from bridge_rgs import evaluate, io
    expected_eval = CANDIDATE/'source_snapshot/bridge_rgs/evaluate.py'
    require(Path(evaluate.__file__).resolve() == expected_eval, 'Use candidate frozen PYTHONPATH for fingerprint reconstruction')
    manifest = io.load_manifest(manifest_path)
    views = [v for v in manifest['views'] if v['split'] == 'val']
    fingerprint = evaluate.evaluation_fingerprint(views, 1.0, 'colmap_corner_v2', digest(manifest_path))
    require(fingerprint == old_metrics['evaluation_fingerprint'] == new_metrics['evaluation_fingerprint'],
            'Native RGB fingerprint mismatch')
    paired = paired_comparison(old_metrics, new_metrics, rgb_only=True)
    require(set(paired['metrics']) == {'psnr', 'ssim', 'lpips'}, 'Untrained semantic metrics must not be compared')
    ratios = {key: candidate['cost'][key]/reference['cost'][key] for key in
              ['final_N', 'peak_logged_post_update_or_primary_N', 'primary_gaussian_steps', 'primary_render_pixels',
               'all_render_gaussian_sum', 'all_render_pixels', 'last_logged_training_seconds',
               'training_peak_allocated_GiB_rounded', 'experiment_seconds_including_native_evaluation']}
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
    report = {'status': 'passed', 'cpu_only': True, 'candidate': candidate, 'reference': reference,
              'native_fingerprint': fingerprint, 'candidate_over_reference_cost_ratios': ratios,
              'paired_native_rgb': paired, 'script_sha256': digest(__file__),
              'paired_helper_sha256': digest(Path(__file__).with_name('compare_evaluations.py')),
              'scope': 'RGB-only two multi-factor recipes on same corner-v2 350/50; not SEM386/RGB150 exact, not isolated densification effect. Semantic heads untrained and excluded. Official RGB is a separate root evaluation.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=CANDIDATE/'endpoint_audit.json')
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({'status': result['status'], 'native_rgb': result['candidate']['native_rgb_metrics'],
                      'cost_ratios': result['candidate_over_reference_cost_ratios']}, indent=2))


if __name__ == '__main__':
    main()
