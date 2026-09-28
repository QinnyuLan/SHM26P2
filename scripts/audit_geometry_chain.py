"""Independent NumPy audit of saved geometry-chain arrays; never renders or decodes images."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np

PLAN_SHA = '059bf2e5c4ca558a6252ec8053f2759a6f7ae49174bedbacaa345973acfff43b'
EXEC_SHA = '386831daf98b1b3ebc0b28e5086a75709aaa599e175dd18cb9f880bb0214ff57'
ATTRS = ('means2d', 'conics', 'opacities', 'colors')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dot(g, p, m, h):
    terms = np.asarray(g, np.float64)*((p.astype(np.float64)-m.astype(np.float64))/(2*h))
    require(np.isfinite(terms).all(), 'Nonfinite secant product')
    return {'value': float(np.sum(terms)), 'absolute_product_sum': float(np.sum(abs(terms)))}


def image_identity(r0, rp, rm, target, valid, adjoint, h):
    base, plus, minus = [np.asarray(x[..., :3], np.float64) for x in (r0, rp, rm)]
    b, p, m = [np.minimum(np.maximum(x, 0), 1)[valid] for x in (base, plus, minus)]
    y = target[valid]
    f0, fp, fm = [float(np.sum((x-y)**2)/x.size) for x in (b, p, m)]
    loss_fd = (fp-fm)/(2*h)
    image_dot = dot(adjoint, plus, minus, h)['value']
    # Separate quadratic endpoint remainders around the same clamped baseline.
    linear = float(np.sum(2*(b-y)*(p-m))/(b.size*2*h))
    remainder = float((np.sum((p-b)**2)-np.sum((m-b)**2))/(b.size*2*h))
    branch = lambda x: (x > 1).astype(np.int8)-(x < 0).astype(np.int8)
    return {
        'baseline': f0, 'plus': fp, 'minus': fm, 'C_raw_image_vjp': image_dot,
        'D_loss_fd': loss_fd, 'C_clamped_image_exact': linear,
        'quadratic_remainder': remainder, 'quadratic_identity_residual': loss_fd-linear-remainder,
        'raw_to_clamped_C_difference': linear-image_dot,
        'clamp_crossing_valid_channels_plus': int(np.sum((branch(base) != branch(plus))[valid])),
        'clamp_crossing_valid_channels_minus': int(np.sum((branch(base) != branch(minus))[valid])),
    }


def audit(folder):
    started = time.monotonic()
    folder = Path(folder).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicitly disable CUDA')
    plan, result, launch = [read(folder/k) for k in ('plan.json', 'execution_receipt.json', 'launch_receipt.json')]
    require(sha(folder/'plan.json') == PLAN_SHA and sha(folder/'execution_receipt.json') == EXEC_SHA,
            'Unexpected completed experiment')
    require(launch['status'] == result['status'] == 'completed' and launch['natural_completion']
            and launch['exit_code'] == 0 and launch['execution_receipt_sha256'] == EXEC_SHA
            and launch['plan_sha256'] == result['plan_sha256'] == PLAN_SHA, 'Completion chain failed')
    require(result['diagnostic_status'] == 'completed_localization_only', 'Wrong scientific status')
    require(plan['specification']['protocol'] == 'continuous_geometry_chain_v1'
            and plan['specification']['view'] == '041.png' and plan['specification']['amplitude'] == 1/128,
            'Unexpected fixed diagnostic')
    snapshot = Path(plan['source_snapshot'])
    actual_paths = {str(p.relative_to(snapshot)) for p in snapshot.rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    require(actual_paths == set(plan['source_hashes']), 'Source inventory changed')
    bound = {str(snapshot/k): v for k, v in plan['source_hashes'].items()}
    bound.update(plan['input_hashes']); bound.update(plan['installed_sources'])
    binary = plan['expected_gsplat_binary']; bound[binary['path']] = binary['sha256']
    bound.update({str(folder/'plan.json'): PLAN_SHA, str(folder/'execution_receipt.json'): EXEC_SHA,
                  str(folder/'launch_receipt.json'): sha(folder/'launch_receipt.json')})
    require(all(plan['source_hashes'].get(k) == v for k, v in plan['inherited_source_hashes'].items()),
            'Inherited source changed')
    require(launch['runner_sha256'] == plan['source_hashes']['diagnose_geometry_chain.py'], 'Entry source mismatch')
    for name, version in plan['runtime_versions'].items():
        require(importlib.metadata.version(name) == version, 'Runtime version mismatch: '+name)
    for name, record in result['actual_imports'].items():
        path = Path(record['path'])
        require(path.is_relative_to(snapshot) and record['sha256'] == plan['source_hashes'].get(str(path.relative_to(snapshot))),
                'Unbound actual import: '+name)
    require(result['actual_gsplat_binary'] == binary and result['sources_inputs_unchanged'], 'Binary/input record mismatch')
    require(result['numerics_actual'] == plan['numerics'] and result['numerics_restored'], 'Numerical mode mismatch')
    names = {'target_and_metric', 'baseline', 'gradients', 'direction', 'plus', 'minus',
             'replay_baseline', 'replay_plus', 'replay_minus'}
    require(set(result['arrays']) == names, 'Unexpected array set')
    for name, record in result['arrays'].items():
        require(Path(record['path']) == folder/(name+'.npz'), 'Unexpected saved array path')
        bound[record['path']] = record['sha256']
    for path, digest in bound.items():
        require(sha(path) == digest, 'Before audit hash changed: '+path)
    arrays = {}
    for name, record in result['arrays'].items():
        with np.load(record['path'], allow_pickle=False) as archive:
            require(set(archive.files) == set(record['arrays']), 'NPZ inventory changed')
            arrays[name] = {k: archive[k] for k in archive.files}
        for key, value in arrays[name].items():
            require({'shape': list(value.shape), 'dtype': str(value.dtype)} == record['arrays'][key], 'NPZ header mismatch')
            require(np.isfinite(value).all(), 'Saved array not finite: '+name+'/'+key)
    checks = {'scalar_count': 0, 'max_absolute_discrepancy': 0.}
    def compare(actual, expected, path=''):
        if isinstance(expected, dict):
            require(set(actual) == set(expected), 'Result keys mismatch: '+path)
            for key in expected:
                compare(actual[key], expected[key], path+'/'+key)
        elif expected is None or isinstance(expected, (str, bool, int)):
            require(actual == expected, 'Discrete result mismatch: '+path)
        else:
            error = abs(float(actual)-float(expected))
            require(np.isfinite(actual) and error <= 2e-13+2e-10*abs(expected), 'Numeric result mismatch: '+path)
            checks['scalar_count'] += 1
            checks['max_absolute_discrepancy'] = max(checks['max_absolute_discrepancy'], error)
    b, p, m, g, metric = [arrays[k] for k in ('baseline', 'plus', 'minus', 'gradients', 'target_and_metric')]
    h = 1/128
    valid = metric['valid']; target = metric['target_rgb_u8'].astype(np.float64)/255
    require(valid.dtype == bool and valid.any() and metric['target_rgb_u8'].dtype == np.uint8, 'Wrong saved target/support')
    for state in (b, p, m):
        require(np.array_equal(state['valid_projection'], np.all(state['radii'] > 0, axis=1)), 'Invalid projection definition')
        require(np.array_equal(state['backgrounds'], b['backgrounds']), 'Background changed')
        for key in ATTRS:
            require(np.count_nonzero(state[key][~state['valid_projection']]) == 0, 'Invalid saved storage not sanitized')
        ids = state['flatten_ids']; offsets = state['isect_offsets'].ravel()
        require(ids.ndim == 1 and len(ids) > 0 and ids.min() >= 0 and ids.max() < len(state['means']), 'Bad ledger IDs')
        require(state['valid_projection'][ids].all() and offsets[0] == 0
                and np.all(np.diff(offsets) >= 0) and offsets[-1] <= len(ids), 'Invalid saved tile ledger')
    A = dot(g['means'], p['means'], m['means'], h)
    pieces = {}
    common = b['valid_projection'] & p['valid_projection'] & m['valid_projection']
    for key in ATTRS:
        grad = g[key][0]; active = (grad != 0).reshape(len(grad), -1).any(axis=1)
        missing = int(np.sum(active & ~common)); selected = active & common
        value = dot(grad[selected], p[key][selected], m[key][selected], h)
        pieces[key] = {'value': None if missing else value['value'], 'common_valid_value': value['value'],
            'absolute_product_sum': value['absolute_product_sum'], 'nonzero_gradient_invalid_endpoint_rows': missing,
            'common_valid_rows': int(common.sum()), 'nonzero_gradient_common_valid_rows': int(selected.sum()),
            'complete': not bool(missing)}
    B = sum(v['value'] for v in pieces.values()) if all(v['complete'] for v in pieces.values()) else None
    image = image_identity(b['raw4'], p['raw4'], m['raw4'], target, valid, g['image'][0, ..., :3], h)
    chain = {'A_means': A, 'B_components': pieces, 'B_complete': B, **image,
             'A_minus_B': None if B is None else A['value']-B,
             'B_minus_C': None if B is None else B-image['C_raw_image_vjp'],
             'C_minus_D': image['C_raw_image_vjp']-image['D_loss_fd']}
    compare(chain, result['chain'], 'chain')
    # MSE -> raw4 derivative can be recomputed without renderer backward.
    upstream = np.zeros_like(g['image'][0]); raw_rgb = b['raw4'][..., :3].astype(np.float64)
    analytic_rgb = (2*(np.clip(raw_rgb, 0, 1)-target)/(3*int(valid.sum()))).astype(np.float32)
    analytic_rgb *= (valid[..., None] & (raw_rgb >= 0) & (raw_rgb <= 1))
    upstream[..., :3] = analytic_rgb
    upstream_error = float(np.max(abs(upstream.astype(np.float64)-g['image'][0])))
    require(np.allclose(upstream, g['image'][0], rtol=2e-7, atol=1e-14), 'Saved image VJP is not expected MSE adjoint')
    require(np.count_nonzero(g['image'][..., 3]) == 0 and np.count_nonzero(g['colors'][..., 3]) == 0,
            'RGB-only loss unexpectedly differentiated depth')
    rotation, scale = metric['rotation'], metric['scales']
    require((scale > 0).all() and np.max(abs(rotation.transpose(0, 2, 1)@rotation-np.eye(3))) < 1e-10,
            'Invalid fixed covariance factors')
    local = (rotation.transpose(0, 2, 1)@g['means'].astype(np.float64)[..., None])[..., 0]
    lengths = np.sqrt(np.sum((local*scale)**2, axis=1)); denominator = float(lengths.max())
    rebuilt = (rotation@(local*scale**2)[..., None])[..., 0]/denominator
    direction = arrays['direction']['direction']
    require(np.allclose(rebuilt, direction, rtol=1e-10, atol=2e-15), 'Covariance direction reconstruction mismatch')
    compare({'normalizer': denominator, 'max_mahalanobis': float(np.max(lengths/denominator)),
             'nonzero_rows': int(np.count_nonzero(lengths))}, result['direction_reconstruction'], 'direction')
    quantization = {}
    for name, state, sign in (('plus', p, 1), ('minus', m, -1)):
        require(np.array_equal(state['means'], (b['means'].astype(np.float64)+sign*h*direction).astype(np.float32)),
                'Actual FP32 endpoint mismatch')
        delta = state['means'].astype(np.float64)-b['means'].astype(np.float64)
        dlocal = (rotation.transpose(0, 2, 1)@delta[..., None])[..., 0]
        radius = np.sqrt(np.sum((dlocal/scale)**2, axis=1)); over = radius > h
        row_dot = np.sum(g['means'].astype(np.float64)*delta, axis=1)
        quantization[name] = {'rows_exceeding_ideal_h': int(over.sum()), 'maximum_mahalanobis': float(radius.max()),
            'all_rows_gradient_dot': float(row_dot.sum()), 'exceeding_rows_gradient_dot': float(row_dot[over].sum()),
            'exceeding_rows_absolute_gradient_dot': float(abs(row_dot[over]).sum())}
    compare(quantization, result['quantization'], 'quantization')
    unique = np.unique(b['flatten_ids'])
    require(b['valid_projection'][unique].all(), 'Invalid baseline referenced row')
    replay = {'referenced_unique_rows': len(unique), 'invalid_plus': int(np.sum(~p['valid_projection'][unique])),
              'invalid_minus': int(np.sum(~m['valid_projection'][unique]))}
    replay['endpoint_replay_allowed'] = replay['invalid_plus'] == replay['invalid_minus'] == 0
    require(replay['endpoint_replay_allowed'], 'Unexpected missing replay prerequisites')
    rb, rp, rm = [arrays[k] for k in ('replay_baseline', 'replay_plus', 'replay_minus')]
    replay['baseline_four_channel_and_alpha_exact'] = all(np.array_equal(rb[k], b[k]) for k in ('raw4', 'alpha'))
    require(replay['baseline_four_channel_and_alpha_exact'], 'Baseline replay changed')
    replay['fixed_ledger'] = image_identity(rb['raw4'], rp['raw4'], rm['raw4'], target, valid, g['image'][0, ..., :3], h)
    replay['natural_minus_fixed_loss_fd'] = image['D_loss_fd']-replay['fixed_ledger']['D_loss_fd']
    replay['scope'] = 'tile inclusion/order jointly frozen; alpha thresholds, caps, early termination remain active'
    compare(replay, result['replay'], 'replay')
    ledger = {name: {'offsets_exact_baseline': bool(np.array_equal(state['isect_offsets'], b['isect_offsets'])),
        'ordered_flatten_ids_exact_baseline': bool(np.array_equal(state['flatten_ids'], b['flatten_ids'])),
        'isect_count': len(state['flatten_ids']),
        'active_row_change_count': int(np.sum(state['valid_projection'] != b['valid_projection']))}
        for name, state in (('plus', p), ('minus', m))}
    compare(ledger, result['ledger'], 'ledger')
    counts = {'native_rasters': 3, 'replay_rasters': 3, 'backward_sweeps': 1, 'target_decodes': 2}
    require(result['counts'] == counts, 'Unexpected execution counts')
    require(all(result[k] == 0 for k in ('optimizer_steps', 'head_calls', 'teacher_calls', 'val_views', 'semantic_rasters', 'label_decodes')),
            'Unexpected extra work')
    restoration = result['restoration']
    require(restoration['state_exact'] and restoration['flags_modes_gradients_restored']
            and restoration['state_before'] == restoration['state_after'], 'Restore receipt inconsistent')
    require(hashlib.sha256(b['means'].tobytes()).hexdigest() == restoration['state_before']['splats.means'],
            'Saved baseline means not restoration baseline')
    prior = read(plan['prior_execution'])
    prior_view, = [v for v in prior['views'] if v['name'] == '041.png']
    prior_row, = [v for v in prior['records'] if v['view'] == '041.png' and v['amplitude'] == h
                  and v['direction_loss'] == v['measured_loss'] == 'rgb_mse']
    compare({'bytewise_direction_verifiable': False,
        'reason': 'original run did not save full gradient or endpoint arrays; this is a reconstructed instance',
        'baseline_loss_difference': image['baseline']-prior_view['baseline'][0],
        'normalizer_difference': denominator-prior_view['directions']['rgb_mse']['normalizer'],
        'analytic_difference': A['value']-prior_row['analytic_actual_fp32_displacement'],
        'loss_fd_difference': image['D_loss_fd']-prior_row['central_difference']}, result['prior_comparison'], 'prior')
    require(prior['numerical_status'] == 'not_passed' and prior['main_passed'] == 3 and prior['main_count'] == 12,
            'Old numerical failure must remain preserved')
    for path, digest in bound.items():
        require(sha(path) == digest, 'After audit hash changed: '+path)
    return {
        'status': 'passed', 'meaning': 'independent saved-array and provenance consistency audit; not a numerical gate pass',
        'run': str(folder), 'plan_sha256': PLAN_SHA, 'execution_receipt_sha256': EXEC_SHA,
        'launch_receipt_sha256': sha(folder/'launch_receipt.json'), 'audit_script_sha256': sha(__file__),
        'source_count': len(plan['source_hashes']), 'input_count': len(plan['input_hashes']),
        'installed_source_count': len(plan['installed_sources']), 'array_file_count': len(arrays),
        'before_after_bound_hashes_exact': True, 'comparison': checks,
        'A': A['value'], 'B': B, 'C': image['C_raw_image_vjp'], 'D': image['D_loss_fd'],
        'A_minus_B': chain['A_minus_B'], 'B_minus_C': chain['B_minus_C'], 'C_minus_D': chain['C_minus_D'],
        'natural_image_identity': image, 'fixed_ledger_image_identity': replay['fixed_ledger'],
        'natural_minus_fixed_loss_fd': replay['natural_minus_fixed_loss_fd'], 'B_components': pieces,
        'quantization': quantization, 'ledger': ledger, 'baseline_intersection_count': len(b['flatten_ids']),
        'replay_reference_validity': {k: v for k, v in replay.items() if k not in ('fixed_ledger', 'scope')},
        'saved_image_adjoint_max_absolute_error': upstream_error,
        'covariance_direction_max_absolute_error': float(abs(rebuilt-direction).max()),
        'restoration_receipt_consistent': True, 'counts_receipt_and_output_inventory_consistent': counts,
        'original_numerical_gate': {'status': 'not_passed', 'passed': 3, 'total': 12},
        'new_renders': 0, 'new_GT_image_decodes': 0, 'GPU_calls': 0,
        'limitations': [
            'Means and intermediate VJPs are saved measurements, not independently regenerated CUDA derivatives.',
            'Fixed-ledger raster outputs are audited as saved arrays; no independent raster replay was executed.',
            'Runtime work counts and state/flag restoration use bound producer records; output inventory is corroboration, not execution tracing.',
            'Old failed preflight did not save complete directions: same-instance bytewise replay cannot be established.',
            'Invalid projected attributes were saved as zero; uninitialized original storage cannot be reconstructed.',
            'One reconstructed TRAIN041 direction only; no new calibration gate, general CUDA correctness or training admission.',
        ],
        'elapsed_seconds': time.monotonic()-started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.run/'independent_cpu_review.json'
    require(not output.exists(), 'Refuse overwriting previous audit')
    result = audit(args.run)
    payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
    with output.open('x') as handle:
        handle.write(payload)
    print(json.dumps({'status': result['status'], 'path': str(output), 'sha256': sha(output),
                      'elapsed_seconds': result['elapsed_seconds'], 'comparison': result['comparison']}))


if __name__ == '__main__':
    main()
