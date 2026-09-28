"""Independent NumPy recomputation of saved two-TRAIN probe evidence only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np

PLAN_SHA = '499e7b76cab849dc9515438c853fd12d9d4e3aa24456211c06778480b0051f7e'
EXEC_SHA = 'bd8a14883622e195a06d5aa47d5e53ea5af4f04c7154def4b0b8223ddc6dfd8c'


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def losses(data, prefix, weights, delta):
    valid, y = data['valid'], data['labels']
    mask = valid & (y < 5)
    check(valid.dtype == np.bool_ and y.dtype == np.uint8 and mask.any() and valid.any(), 'Target schema/support')
    rgb = np.clip(data[prefix+'_rgb'], 0, 1).astype(np.float64)[valid]
    result = [float(np.mean(np.square(rgb-data['target_rgb'][valid])))]
    for key in ('raw', 'probabilities'):
        values = data[prefix+'_'+key]
        check(values.dtype == np.float32 and values.shape == (*y.shape, 5)
              and np.isfinite(values).all() and (values >= 0).all(), 'Probability evidence')
        labels = y[mask].astype(np.int64)
        selected = values[mask][np.arange(len(labels)), labels].astype(np.float64)
        # Original positive affine noise uses total noise delta, distributed over 5 classes.
        result.append(float(np.mean(-weights[labels]*np.log((1-delta)*selected+delta/5))))
    return result


def audit(folder):
    start = time.monotonic()
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicitly disable CUDA')
    folder = Path(folder).resolve()
    plan, execution, launch = [read(folder/n) for n in ('plan.json', 'execution_receipt.json', 'launch_receipt.json')]
    check(sha(folder/'plan.json') == PLAN_SHA and sha(folder/'execution_receipt.json') == EXEC_SHA, 'Fixed experiment identity')
    check(execution['status'] == launch['status'] == 'completed' and launch['natural_completion'] is True
          and launch['exit_code'] == 0 and launch['plan_sha256'] == execution['plan_sha256'] == PLAN_SHA
          and launch['execution_receipt_sha256'] == EXEC_SHA, 'Natural completion chain')
    spec = plan['specification']
    check(spec['names'] == [v['name'] for v in execution['views']] == ['002.png', '041.png']
          and spec['affine_noise'] == 5e-7 and spec['training_commits'] == 0, 'Fixed experiment scope')
    snapshot = Path(plan['source_snapshot'])
    paths = {str(p.relative_to(snapshot)): p for p in snapshot.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    check(set(paths) == set(plan['source_hashes']), 'Source inventory changed')
    bindings = {str(paths[k]): v for k, v in plan['source_hashes'].items()}
    target_paths = {v[k] for v in plan['views'] for k in ('image_path', 'mask_path', 'valid_path')}
    parent = read('/mnt/data/SHM2026/runs/continuous_geometry_proposals_v1/plan.json')
    check(all(plan['input_hashes'][p] == parent['input_hashes'][p] for p in target_paths), 'Inherited target SHA differs')
    # No original target bytes/hash reads: labels/valid/RGB truth come exclusively from saved NPZ.
    bindings.update({p: h for p, h in plan['input_hashes'].items() if p not in target_paths})
    bindings.update(plan['installed_sources'])
    binary = plan['expected_gsplat_binary']; bindings[binary['path']] = binary['sha256']
    bindings[str(folder/'plan.json')] = PLAN_SHA
    bindings[str(folder/'execution_receipt.json')] = EXEC_SHA
    bindings[str(folder/'launch_receipt.json')] = sha(folder/'launch_receipt.json')
    check(launch['runner_sha256'] == plan['source_hashes']['probe_readout_geometry.py'], 'Runner SHA differs')
    for name, entry in execution['actual_imports'].items():
        path = Path(entry['path'])
        check(path.is_relative_to(snapshot)
              and plan['source_hashes'].get(str(path.relative_to(snapshot))) == entry['sha256'], 'Unbound import: '+name)
    for name, version in plan['runtime_versions'].items():
        check(importlib.metadata.version(name) == version, 'Runtime differs: '+name)
    check(execution['numerics_actual'] == plan['numerics'] and execution['numerics_restored']
          and execution['sources_inputs_unchanged'], 'Runtime numerical/state assertions')
    check(all(execution['counts'][k] == v for k, v in spec['expected_counts'].items())
          and execution['counts']['low_level_raster_calls'] == 18, 'Fixed call counts')
    restoration = execution['restoration']
    check(restoration['state_before'] == restoration['state_after'] and restoration['state_exact']
          and restoration['flags_modes_gradients_restored'], 'Saved runtime restoration records differ')
    for row in execution['views']:
        path = folder/(Path(row['name']).stem+'_evidence.npz')
        check(str(path) == row['evidence']['path'], 'Unexpected evidence location')
        bindings[str(path)] = row['evidence']['sha256']
    check(all(sha(p) == h for p, h in bindings.items()), 'Source/input/output byte binding failed')

    differences = []; rows = []; base_hash = restoration['state_before']['splats.means']
    def close(actual, expected, description):
        a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
        check(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), description+' shape/finite')
        differences.extend(np.abs(a-b).ravel().tolist())
        check(np.allclose(a, b, rtol=2e-12, atol=2e-14), description+' differs')
    weights = np.asarray(plan['class_weights'], np.float32).astype(np.float64)
    for recorded in execution['views']:
        with np.load(recorded['evidence']['path'], allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        check(all(np.isfinite(v).all() for v in data.values()), 'Nonfinite saved array')
        base, candidate = data['base_means'], data['candidate_means']
        check(base.shape == candidate.shape == (498136, 3) and base.dtype == candidate.dtype == np.float32, 'Means schema')
        check(hashlib.sha256(base.tobytes()).hexdigest() == base_hash, 'Base means differ from restoration')
        recomputed = {p: losses(data, p, weights, spec['affine_noise']) for p in ('old', 'baseline', 'candidate')}
        for key, actual in recomputed.items():
            close(actual, recorded[key+'_losses'], key+' losses')
        delta = candidate.astype(np.float64)-base.astype(np.float64)
        dots = []
        for name in ('rgb', 'raw', 'scene'):
            gradient = data['gradient_'+name]
            check(gradient.shape == base.shape and gradient.dtype == np.float32
                  and np.isfinite(gradient).all() and np.count_nonzero(gradient), 'Saved VJP invalid')
            dots.append(float(np.sum(gradient.astype(np.float64)*delta, dtype=np.float64)))
        close(dots, recorded['gradient_actual_dots'], 'Saved-gradient actual-displacement dots')
        close([dots[0], dots[2]], [recorded['proposal']['rgb_dot_actual_delta'], recorded['proposal']['semantic_dot_actual_delta']], 'Transaction dots')
        changes = np.subtract(recomputed['candidate'], recomputed['baseline'])
        close(changes, recorded['actual_loss_changes'], 'Actual loss changes')
        close(np.max(abs(delta)), recorded['proposal']['actual_absmax_displacement'], 'Actual displacement max')
        check(np.count_nonzero(np.any(delta != 0, axis=1)) == recorded['proposal']['actual_nonzero_point_count']
              and np.count_nonzero(delta) == recorded['proposal']['actual_nonzero_coordinate_count'], 'Actual displacement counts')
        forward = {}
        for key in ('rgb', 'raw', 'probabilities'):
            a, b = data['old_'+key], data['baseline_'+key]
            value = {'exact': bool(np.array_equal(a, b)), 'max_abs': float(np.max(abs(a.astype(np.float64)-b.astype(np.float64))))}
            check(value == recorded['forward_comparison'][key], 'Saved forward difference: '+key)
            if key != 'rgb':
                value['argmax_differences'] = int(np.count_nonzero(a.argmax(-1) != b.argmax(-1)))
                check(value['argmax_differences'] == recorded['forward_mask_differences'][key], 'Argmax difference')
            forward[key] = value
        rows.append({'name': recorded['name'], 'loss_order': ['RGB_MSE', 'raw_affine_CE', 'scene_affine_CE'],
                     'losses': recomputed, 'actual_loss_changes': changes.tolist(), 'gradient_actual_dots': dots,
                     'forward_comparison': forward,
                     'actual_nonzero_points': int(np.count_nonzero(np.any(delta != 0, axis=1)))})
    check(all(sha(p) == h for p, h in bindings.items()), 'Audit inputs changed during independent read')
    return {'status': 'passed', 'plan_sha256': PLAN_SHA, 'execution_receipt_sha256': EXEC_SHA,
            'auditor_source_sha256': sha(__file__), 'bound_files_verified_before_after': len(bindings),
            'source_files': len(paths), 'original_target_files_not_read': len(target_paths),
            'compared_scalar_count': len(differences), 'maximum_scalar_absolute_difference': max(differences),
            'counts': execution['counts'], 'views': rows,
            'restoration_saved_hashes_exact': True, 'numerics_and_restore_records_consistent': True,
            'elapsed_seconds': time.monotonic()-start, 'GPU_calls': 0, 'renders': 0, 'original_GT_decodes_or_hash_reads': 0,
            'limits': ['Recomputed saved-image objectives and saved-VJP dot products; no independent renderer/VJP execution.',
                       'Depth/alpha/features/moments were not saved separately: their equality is a bound runtime assertion, not independently remeasured.',
                       'Restoration/flags/optimizer rollback/call counts are checked as saved runtime records; no access to the finished process memory.',
                       'Mahalanobis covariance and proposed Adam moments were not saved; cap and optimizer proposal are not independently reconstructed.',
                       'Two TRAIN views and temporary steps establish no VAL benefit, convergence, adoption or general derivative certificate.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    path = args.run/'independent_cpu_review.json'
    payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
    with path.open('x') as stream:
        stream.write(payload)
    print(json.dumps({'status': result['status'], 'path': str(path), 'sha256': sha(path),
                      'elapsed_seconds': result['elapsed_seconds'], 'max_difference': result['maximum_scalar_absolute_difference']}))
