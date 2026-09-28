"""CPU-only endpoint audit of the fixed 30k TRAIN-derived Mip filter reference."""
from __future__ import annotations

import argparse
import difflib
import json
import struct
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from audit_rgb140_reference_endpoint import (
    digest,
    finite_tensors,
    read,
    reconstruct_cost,
    require,
    require_completed,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('/mnt/data/SHM2026/runs/rgb140_mip_filter_500k')
PLAN_SHA = '6fcd1723900960058400afd75307793710cc6fcf82ee1f42cbc7bae5c652b41e'
MIXED_SHA = 'f551eae520de72043dc4e3d7c690f47c3a71292b0733d9eb49fcacd1d907db18'


def audit_refreshes(rows, saved):
    """Every 100-step post-update event is logged; step1 exposes initialization."""
    require([r['step'] for r in rows] == [1] + list(range(100, 30001, 100)),
            'Missing or duplicate 100-step log')
    gpu_total = wall_total = 0.
    for row in rows:
        step = row['step']
        require(row['mip_filter_refresh_count'] == 1 + step // 100,
                'Periodic/topology/final refresh was missed or duplicated')
        require(row['mip_filter_last_refresh_step'] == step // 100 * 100,
                'Filter is not refreshed after this update')
        for key in ('last_refresh_cuda_ms', 'last_refresh_wall_seconds'):
            require(np.isfinite(row['mip_filter_' + key]) and row['mip_filter_' + key] >= 0,
                    'Nonfinite or negative filter cost')
        gpu_total += row['mip_filter_last_refresh_cuda_ms']
        wall_total += row['mip_filter_last_refresh_wall_seconds']
        require(gpu_total == row['mip_filter_total_refresh_cuda_ms']
                and wall_total == row['mip_filter_total_refresh_wall_seconds'],
                'Direct refresh totals differ from independent sum of event costs')
        require(row['mip_filter_gaussians'] == row['gaussians'], 'Stale topology-sized filter')
    require(saved['refresh_count'] == 301 and saved['last_refresh_step'] == 30000,
            'Incomplete final filter state')
    require(gpu_total == saved['total_refresh_cuda_ms']
            and wall_total == saved['total_refresh_wall_seconds'], 'Saved refresh cost differs')
    return {'count': 301, 'steps': [0] + list(range(100, 30001, 100)),
            'total_cuda_ms': gpu_total, 'total_wall_seconds': wall_total,
            'initialization_cuda_ms': rows[0]['mip_filter_last_refresh_cuda_ms'],
            'initialization_wall_seconds': rows[0]['mip_filter_last_refresh_wall_seconds'],
            'loop_cuda_ms': gpu_total - rows[0]['mip_filter_last_refresh_cuda_ms'],
            'loop_wall_seconds': wall_total - rows[0]['mip_filter_last_refresh_wall_seconds'],
            'evidence': 'Initial cache exposed by step1; every subsequent refresh has a post-step100 record. '
                        'Topology events fall on the same boundaries and are counted once.'}


def audit_topology(rows, policy):
    n = rows[0]['gaussians']
    events, resets = [], []
    for row in rows:
        step = row['step']
        if policy.grows_at(step):
            before = n
            require(all(isinstance(row[k], int) and row[k] >= 0 for k in ('splits', 'duplicates', 'pruned')),
                    'Invalid topology event count')
            n += row['splits'] + row['duplicates'] - row['pruned']
            events.append({'step': step, 'before': before, 'after': n,
                           **{k: row[k] for k in ('splits', 'duplicates', 'pruned')}})
        require(n == row['gaussians'], 'Post-step N not explained by the fixed raw policy')
        if policy.resets_at(step):
            require(row['opacity_reset_step'] == step, 'Missing scheduled raw opacity reset')
            require(0 <= row['opacity_reset_count'] <= n, 'Invalid reset count')
            resets.append({'step': step, 'count': row['opacity_reset_count']})
        elif resets:
            require(row['opacity_reset_step'] == resets[-1]['step'], 'Unexpected opacity reset')
        else:
            require('opacity_reset_step' not in row, 'Unexpected early opacity reset')
    require([r['step'] for r in events] == list(range(600, 15000, 100)), 'Wrong topology schedule')
    require([r['step'] for r in resets] == [3000, 6000, 9000, 12000], 'Wrong reset schedule')
    return {'events': events, 'resets': resets, 'raw_policy_unchanged': True,
            'reset_interpretation': 'Raw sigmoid opacity cap .01 is unchanged; filtered rendered opacity '
                                    'cap is .01 times compensation, not the author effective-opacity reset.'}


def png_size(path):
    with path.open('rb') as stream:
        header = stream.read(24)
    require(header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR', 'Invalid PNG')
    return struct.unpack('>II', header[16:24])


def audit_proposal_incident(directory, planned_inputs):
    """A restored endpoint hash does not establish uninterrupted immutability."""
    path = directory / 'proposal_edit_incident.json'
    incident = read(path)
    original, preserved = Path(incident['file']), Path(incident['preserved_copy'])
    require(incident['expected_preregistered_sha256'] == planned_inputs[str(original)]['sha256']
            == incident['restored_sha256'] == digest(original), 'Bound proposal has not been restored exactly')
    require(digest(preserved) == incident['transient_sha256'], 'Preserved transient document differs')
    diff_path = directory / 'transient_proposal_edit.diff'
    expected_diff = ''.join(difflib.unified_diff(original.read_text().splitlines(True),
                                                preserved.read_text().splitlines(True),
                                                fromfile='preregistered-restored', tofile='transient-edit'))
    require(diff_path.read_text() == expected_diff, 'Preserved incident diff differs from both bound versions')
    return {'incident': incident, 'incident_sha256': digest(path),
            'diff_path': str(diff_path), 'diff_sha256': digest(diff_path),
            'preserved_copy_sha256': digest(preserved), 'restored_endpoint_sha256': digest(original),
            'preregistered_and_endpoint_document_bytes_match': True,
            'continuous_document_immutability': False,
            'interpretation': 'The bound proposal was temporarily edited and restored. Endpoint hash '
                              'verification does not hide or negate that event. This human-readable '
                              'document is not a training runtime input; the event alone is not evidence '
                              'of a numerical source/configuration change. Numerical source/config/input '
                              'endpoint hashes are checked separately, not continuously observed.'}


def audit(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    require(not output.exists(), 'Refusing to overwrite an endpoint audit')
    receipt, launch = require_completed(directory)  # Before touching any checkpoint.
    require(launch.get('exit_code') == 0, 'Explicit natural exit0 is required')
    require(not torch.cuda.is_initialized(), 'CPU-only audit')
    torch.set_num_threads(8)
    plan_path = directory / 'plan.json'
    require(digest(plan_path) == PLAN_SHA == launch['plan_sha256'], 'Fixed plan changed')
    plan, config = read(plan_path), receipt['config']
    require(config == plan['config'] and receipt['resume'] is None, 'Actual config/resume differs')
    config_path = directory / 'training_config.yaml'
    require(digest(config_path) == plan['config_sha256']
            and yaml.safe_load(config_path.read_text()) == config, 'Config bytes/value changed')
    require(config['steps'] == 30000 and config['eval_every'] == 0 and config['save_every'] == 5000
            and config['semantic_start'] > 30000 and config['refine_start'] > 30000,
            'Not the fixed RGB-only endpoint')
    snapshot = directory / 'source_snapshot'
    actual_sources = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')}
    require(actual_sources == plan['source_hashes'] == receipt['source_hashes'] and len(actual_sources) == 40,
            'Runtime source differs from frozen smoke package')
    parent_sources = {str(p.relative_to(plan['source_parent'])): digest(p)
                      for p in Path(plan['source_parent']).rglob('*.py')}
    require(parent_sources == actual_sources, 'Parent snapshot differs')
    for path, record in plan['inputs'].items():
        require(digest(path) == record['sha256'] and Path(path).stat().st_size == record['bytes'],
                f'Planned input changed: {path}')
    for record in receipt['input_hashes'].values():
        require(digest(record['path']) == record['sha256'], 'Runtime input hash differs')
    incident = audit_proposal_incident(directory, plan['inputs'])
    control = Path(plan['primary_comparison'])
    old_receipt, _ = require_completed(control)
    differences = {key for key in set(config) | set(old_receipt['config'])
                   if config.get(key) != old_receipt['config'].get(key)}
    require(differences == {'output', 'mip_filter'}, 'More than filter/output differs from primary config')
    require(digest(snapshot / 'bridge_rgs/mixed_gradient_reference.py') == MIXED_SHA
            == digest(control / 'source_snapshot/bridge_rgs/mixed_gradient_reference.py'),
            'Raw prune/reset/grow implementation changed')
    text = (snapshot / 'bridge_rgs/train.py').read_text()
    for clause in ('source = dict(view, mask_path=None) if self.ignore_masks else view',
                   'ImageCache(ignore_masks=mixed_policy is not None)',
                   '((None, None) if mixed_policy is not None',
                   'if "semantic_counts" in points and mixed_policy is None'):
        require(clause in text, 'Audited TRAIN no-mask branch missing')
    require(plan['train_mask_pixels_read'] is False and plan['validation_during_training'] is False,
            'Different training pixel-access protocol')
    from bridge_rgs import evaluate, io, mip_filter
    from bridge_rgs.mixed_gradient_reference import validate_reference_training
    from bridge_rgs.train import load_scene
    for module in (evaluate, io, mip_filter):
        require(Path(module.__file__).resolve() == snapshot / 'bridge_rgs' / Path(module.__file__).name,
                'Use the candidate frozen PYTHONPATH')
    policy = validate_reference_training(config)
    checkpoint = directory / 'last.pt'
    checkpoint_sha = digest(checkpoint)
    require(checkpoint_sha == receipt['checkpoint_sha256'], 'Checkpoint receipt SHA differs')
    scene, state = load_scene(checkpoint, device='cpu')
    require(state['step'] == 30000 and state['config'] == read(directory / 'config.json'), 'Wrong saved endpoint')
    require(all(state['config'][k] == (mip_filter.POLICY if k == 'mip_filter' else v)
                for k, v in config.items()), 'Saved effective config differs')
    require(state['mip_filter_config'] == mip_filter.POLICY, 'Filter policy differs')
    finite_counts = {k: finite_tensors(state[k]) for k in ('model', 'optimizers', 'density_state')}
    require(not bool(state['model']['semantic_prior_counts'].any()), 'No-mask field contains label priors')
    require(state['density_state']['mixed_gradient']['policy'] == config['mixed_gradient'], 'Raw policy state changed')
    manifest_path = ROOT / config['manifest']
    manifest = io.load_manifest(manifest_path)
    views = [v for v in manifest['views'] if v['split'] == 'train']
    source = mip_filter.camera_source(views, state['training_cameras'], digest(manifest_path))
    require(len(views) == 350 and source['metadata'] == state['mip_filter_state']['source'],
            'Original TRAIN camera/native-K/filter provenance differs')
    require(state['manifest_sha256'] == digest(manifest_path)
            and state['pixel_protocol']['id'] == 'colmap_corner_v2', 'Checkpoint data profile differs')
    train_rgb = {str(Path(v['image_path']).resolve()) for v in views}
    require(train_rgb <= set(plan['inputs']) and not any('/masks/' in p for p in plan['inputs']),
            'TRAIN RGB coverage or no-mask planned input contract differs')
    n = len(state['model']['splats.means'])
    rho = state['model']['mip_filter_rho']
    mip_filter.validate_buffer(rho, n)
    for name in ('signed_sum', 'absolute_sum', 'observations'):
        require(state['density_state']['mixed_gradient'][name].shape == (n,), 'Stale mixed topology buffer')
    rows = [json.loads(line) for line in (directory / 'train.jsonl').read_text().splitlines()]
    require(all(np.isfinite(r[k]) for r in rows for k in ('loss', 'rgb_loss', 'elapsed_seconds', 'peak_gpu_gb')),
            'Nonfinite training scalar')
    require(all(r['semantic_loss'] == 0 and r['sh_degree'] == min(3, r['step'] // 1000) for r in rows),
            'Unplanned semantic or SH schedule')
    with np.load(ROOT / manifest['init_points_path']) as initial:
        initial_n = len(initial['points']) + config['background_points']
    cost = reconstruct_cost(rows, [(v['width'], v['height']) for v in views], initial_count=initial_n)
    require(cost['final_N'] == n <= config['max_gaussians'], 'Endpoint/cap N differs')
    for row in rows:
        require(cost['boundary_totals'][row['step']] == (row['reference_gaussian_steps'], row['reference_render_pixels']),
                'Direct cumulative counters differ from source+post-step-N reconstruction')
    require(cost['peak_logged_post_update_or_primary_N'] == rows[-1]['reference_peak_gaussians'], 'Peak N differs')
    for key in ('reference_gaussian_steps', 'reference_render_pixels', 'reference_peak_gaussians'):
        require(state['stats'][key] == rows[-1][key], 'Checkpoint cost differs from final log')
    del cost['boundary_totals']
    refreshes = audit_refreshes(rows, state['mip_filter_state'])
    topology = audit_topology(rows, policy)
    recomputed, rho_stats = mip_filter.compute_rho(scene.splats['means'], source)
    absolute, relative = (recomputed - rho).abs(), (recomputed - rho).abs() / rho
    rho_difference = {'max_absolute': float(absolute.max()), 'max_relative': float(relative.max()),
                      'mean_absolute': float(absolute.mean()), 'bitwise_equal': torch.equal(recomputed, rho),
                      'unseen_CPU': rho_stats['unseen_gaussians'],
                      'unseen_GPU': state['mip_filter_state']['unseen_gaussians'],
                      'scope': 'Saved GPU versus independently recomputed CPU arithmetic; numerical difference '
                               'reported without a post-hoc tolerance or bitwise-equivalence claim.'}
    cost.update(training_seconds=rows[-1]['elapsed_seconds'],
                peak_allocated_GiB_rounded=max(r['peak_gpu_gb'] for r in rows),
                train_plus_native_evaluation_seconds=(datetime.fromisoformat(receipt['finished_utc'])
                                                     - datetime.fromisoformat(receipt['started_utc'])).total_seconds(),
                loop_filter_wall_fraction_of_training=refreshes['loop_wall_seconds'] / rows[-1]['elapsed_seconds'],
                scope='Direct render Gaussian/pixel totals independently reconstructed; not FLOPs. '
                      'Excludes final evaluation, transient topology allocations and CPU audit cost. '
                      'Training timer begins after initialization; loop filter fraction excludes the initial refresh.')
    native_path = directory / 'evaluation_native/metrics.json'
    require(digest(native_path) == receipt['evaluation_sha256'], 'Native metrics receipt SHA differs')
    metrics = read(native_path)
    val = [v for v in manifest['views'] if v['split'] == 'val']
    fingerprint = evaluate.evaluation_fingerprint(val, 1., 'colmap_corner_v2', digest(manifest_path))
    require(metrics['validation_views'] == len(metrics['views']) == 50 and metrics['semantic_validation_views'] == 41
            and metrics['scale'] == 1 and metrics['gaussians'] == n
            and metrics['pixel_protocol']['id'] == 'colmap_corner_v2'
            and {v['name'] for v in metrics['views']} == {v['name'] for v in val}
            and metrics['evaluation_fingerprint'] == fingerprint, 'Native endpoint protocol/fingerprint differs')
    primary_native = control / 'evaluation_native/metrics.json'
    require(digest(primary_native) == old_receipt['evaluation_sha256']
            and read(primary_native)['evaluation_fingerprint'] == fingerprint, 'Primary native protocol differs')
    png_hashes = {}
    for row in metrics['views']:
        require((row['width'], row['height']) == (1320, 989)
                and all(np.isfinite(row[k]) for k in ('psnr', 'ssim', 'lpips')), 'Incomplete native RGB record')
        path = directory / 'evaluation_native' / (Path(row['name']).stem + '_rgb.png')
        require(png_size(path) == (1320, 989), 'RGB PNG dimensions differ')
        png_hashes[path.name] = digest(path)
    require(not any(directory.glob('validation_*')) and not any(directory.glob('metrics_*.json'))
            and not any(directory.glob('step_*.pt')), 'Unexpected intermediate evaluation/checkpoint')
    require(digest(checkpoint) == checkpoint_sha and not torch.cuda.is_initialized(), 'Checkpoint changed or CUDA initialized')
    report = {'status': 'passed', 'cpu_only': True, 'plan_sha256': PLAN_SHA,
              'checkpoint_sha256': checkpoint_sha, 'step': 30000, 'gaussians': n,
              'source_count': len(actual_sources), 'source_and_input_endpoint_hashes_match_plan': True,
              'proposal_edit_incident': incident,
              'config_differences_from_primary_control': sorted(differences), 'finite_tensor_counts': finite_counts,
              'original_TRAIN_cameras_native_K_exact': True, 'filter_source_exact': True,
              'train_no_mask_evidence': 'Frozen source/config plus zero semantic priors; not syscall-level pixel-open tracing.',
              'no_intermediate_VAL_evidence': 'eval_every0 and artifact inventory; final wrapper native evaluation is separate.',
              'rho_shape': list(rho.shape), 'rho_CPU_comparison': rho_difference,
              'filter_refreshes': refreshes, 'topology_and_raw_reset': topology, 'cost': cost,
              'native_rgb_metrics': {k: metrics[k] for k in ('psnr', 'ssim', 'lpips')},
              'native_evaluation_fingerprint': fingerprint, 'native_RGB_PNG_sha256': png_hashes,
              'semantic_score_scope': 'Untrained semantic heads; values deliberately excluded from results/ranking.',
              'comparison_scope': 'Only endpoint integrity and native protocol compatibility; root performs score pairs and official evaluation.',
              'artifact_sha256': {str(p): digest(p) for p in (plan_path, config_path, native_path,
                  directory / 'experiment_receipt.json', directory / 'launch_receipt.json', directory / 'train.jsonl',
                  directory / 'proposal_edit_incident.json', Path(incident['incident']['preserved_copy']),
                  Path(incident['diff_path']),
                  Path(__file__), Path(__file__).with_name('audit_rgb140_reference_endpoint.py'))}}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.output or args.run / 'cpu_endpoint_audit.json')
    print(json.dumps({key: result[key] for key in ('status', 'checkpoint_sha256', 'gaussians', 'rho_CPU_comparison')}, indent=2))
