"""CPU-only independent audit of saved continuous-means preflight scalars.

No producer functions, model loading, rendering, or image/label decoding.
Saved high-dimensional gradients/endpoints do not exist: their dot products,
Mahalanobis norms and reduction mass are checked for consistency, not recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import time
from pathlib import Path

PLAN_SHA = '6a79de1060458df432b8df8b37edb0a5c6d2a2f9028cc2d4f37f901303787845'
N = 498136
NAMES = ['002.png', '041.png']
LOSSES = ['rgb_mse', 'raw_affine_ce']
AMPLITUDES = [1/64, 1/128, 1/256]


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def check_finite(value):
    if isinstance(value, dict):
        for v in value.values():
            check_finite(v)
    elif isinstance(value, list):
        for v in value:
            check_finite(v)
    elif isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite saved scalar')


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected), 'Nonfinite '+label)
    error = abs(actual-expected)
    require(error <= 1e-15+1e-11*max(abs(actual), abs(expected)), label+' differs')
    return error


def check_record(row, baseline, repeat):
    """Recompute FD/errors/floors/gates from scalar endpoints; dots from one sides."""
    h, lp, lm = row['amplitude'], row['loss_plus'], row['loss_minus']
    analytic = row['analytic_actual_fp32_displacement']
    require(h in AMPLITUDES and lp >= 0 and lm >= 0, 'Invalid step/loss')
    fd = (lp-lm)/(2*h)
    absolute = abs(fd-analytic)
    relative = absolute/abs(analytic) if analytic else None
    scalar_floor = (math.ulp(lp)+math.ulp(lm))/(2*h)
    repeat_floor = abs(baseline-repeat)/h
    # This term cannot be independently recomputed without abs(gradient*direction).
    reduction_bound = row['dot_reduction_bound']
    gamma = (3*N*math.ulp(1.))/(1-3*N*math.ulp(1.))
    require(reduction_bound >= 0 and reduction_bound+1e-300 >= gamma*abs(analytic)*(1-1e-12),
            'Reduction bound is smaller than the triangle-inequality lower bound')
    floor = max(scalar_floor, repeat_floor, reduction_bound)
    measurable = abs(analytic) > 10*floor and analytic != 0
    passed = measurable and relative <= .05
    errors = []
    for key, expected in (('central_difference', fd), ('absolute_error', absolute),
                          ('scalar_spacing_floor', scalar_floor), ('repeat_floor', repeat_floor), ('floor', floor)):
        errors.append(close(row[key], expected, key))
    if relative is None:
        require(row['relative_error'] is None, 'Zero derivative relative error')
    else:
        errors.append(close(row['relative_error'], relative, 'relative_error'))
    require(row['measurable'] is bool(measurable) and row['passed'] is bool(passed), 'Stored numerical gate differs')
    positive, negative = row['one_sided']['plus'], row['one_sided']['minus']
    for side, loss in ((positive, lp), (negative, lm)):
        errors.append(close(side['actual_loss_change'], loss-baseline, 'one-sided loss change'))
        require(side['actual_max_mahalanobis'] >= 0 and side['cast_error_rms'] >= 0, 'Invalid displacement norm')
    dot_identity = (positive['analytic_actual_change']-negative['analytic_actual_change'])/(2*h)
    loss_identity = (positive['actual_loss_change']-negative['actual_loss_change'])/(2*h)
    dot_error = close(analytic, dot_identity, 'central vs one-sided actual-dot identity')
    loss_error = close(fd, loss_identity, 'central vs one-sided actual-loss identity')
    require(isinstance(row['changed_coordinates'], int) and 0 < row['changed_coordinates'] <= 3*N
            and row['actual_direction_rms'] > 0 and row['quantization_direction_rms'] >= 0,
            'Degenerate recorded displacement')
    # The two measured losses share a displacement; the worker's FP32 cast is
    # not the later optimizer's enforced cap. Report overshoots, do not reject them.
    return {'view': row['view'], 'direction_loss': row['direction_loss'], 'measured_loss': row['measured_loss'],
            'amplitude': h, 'analytic_actual_dot': analytic, 'recomputed_FD': fd,
            'relative_error': relative, 'measurable': bool(measurable), 'passed': bool(passed),
            'max_scalar_absolute_difference': max(errors), 'actual_dot_identity_absolute_difference': dot_error,
            'loss_identity_absolute_difference': loss_error,
            'recorded_actual_max_mahalanobis': max(positive['actual_max_mahalanobis'], negative['actual_max_mahalanobis'])}


def audit(run):
    require(sha(run/'plan.json') == PLAN_SHA, 'Unexpected fixed plan')
    plan, launch, receipt = (read(run/name) for name in ('plan.json', 'launch_receipt.json', 'execution_receipt.json'))
    require(plan['output'] == str(run) and launch['status'] == receipt['status'] == 'completed'
            and launch['exit_code'] == 0 and launch['natural_completion'] is True, 'Natural completion required')
    require(launch['plan_sha256'] == receipt['plan_sha256'] == PLAN_SHA
            and launch['execution_receipt_sha256'] == sha(run/'execution_receipt.json'), 'Completed receipt chain')
    snapshot = Path(plan['source_snapshot']); sources = tree(snapshot)
    require(sources == plan['source_hashes'], 'Frozen source tree changed')
    entry = snapshot/'diagnose_continuous_semantic_geometry.py'
    require(launch['runner_sha256'] == sha(entry) == sources[entry.name], 'Runner SHA mismatch')
    require(all(sources[k] == h for k, h in plan['inherited_source_hashes'].items()), 'Inherited package changed')
    spec = plan['specification']
    expected = {'protocol': 'continuous_semantic_geometry_preflight_v1', 'names': NAMES, 'point_count': N,
                'losses': LOSSES, 'affine_noise': 5e-7, 'amplitudes': AMPLITUDES,
                'loss_dtype': 'float64', 'renderer_parameter_dtype': 'float32', 'fd_relative': .05,
                'signal_multiplier': 10., 'internal_seconds': 120, 'external_seconds': 180,
                'counts': {'raster': 56, 'means_vjp': 4, 'target_decodes': 6}}
    require(all(spec[k] == v for k, v in expected.items()), 'Frozen numerical specification')
    require(receipt['counts'] == expected['counts'] and len(receipt['records']) == 24, 'Actual call counts')
    for key in ('optimizer_steps', 'head_calls', 'teacher_calls', 'val_views'):
        require(spec[key] == receipt[key] == 0, 'Forbidden work')
    require(plan['prepare_checkpoint_loads'] == plan['prepare_pixel_decodes'] == 0, 'Prepare scope')
    inputs = {**plan['input_hashes'], **plan['installed_sources']}
    binary = plan['expected_gsplat_binary']; inputs[binary['path']] = binary['sha256']
    for path, digest in inputs.items():
        require(sha(path) == digest, 'Bound input/source/binary changed: '+path)
    for name, version in plan['runtime_versions'].items():
        require(importlib.metadata.version(name) == version, 'Installed version changed: '+name)
    require(receipt['actual_gsplat_binary'] == binary, 'Actual loaded renderer binary')
    for name, record in receipt['actual_imports'].items():
        path = Path(record['path']); rel = str(path.relative_to(snapshot))
        require(name.startswith('bridge_rgs') and sources[rel] == record['sha256'], 'Nonfrozen imported module')
    required_imports = {'bridge_rgs.train', 'bridge_rgs.model', 'bridge_rgs.semantic_assignment', 'bridge_rgs.densification'}
    require(required_imports <= receipt['actual_imports'].keys(), 'Missing expected source import')
    require(receipt['sources_inputs_unchanged'] is True and receipt['numerics_actual'] == plan['numerics']
            and receipt['numerics_restored'] is True, 'Source/flag completion checks')
    restored = receipt['restoration']
    require(restored['state_exact'] is True and restored['flags_modes_gradients_restored'] is True
            and restored['state_before'] == restored['state_after'], 'Recorded scene restoration')
    require(receipt['base_means_sha256'] == restored['state_before']['splats.means'], 'Baseline means runtime identity')
    tests = read(run/'frozen_cpu_tests.json')
    require(tests['status'] == 'passed' and tests['exit_code'] == 0 and tests['plan_sha256'] == PLAN_SHA
            and tests['source_hashes_unchanged'] and tests['source_files'] == len(sources)
            and 'no:cacheprovider' in tests['command'], 'Frozen test/source guard')
    for marker in ('execution_started.json', 'launch_started.json'):
        require(read(run/marker)['plan_sha256'] == PLAN_SHA, 'Started marker binding')
    views = {v['name']: v for v in receipt['views']}
    require(list(views) == [v['name'] for v in plan['views']] == NAMES, 'Fixed TRAIN views')
    for v in plan['views']:
        require(v['split'] == 'train', 'Non-TRAIN view')
        for key in ('image_path', 'mask_path', 'valid_path'):
            require(v[key] in plan['input_hashes'], 'Unbound target path')
        saved = views[v['name']]
        require(0 < saved['known_pixels'] <= saved['rgb_valid_pixels'] <= v['width']*v['height'], 'Recorded support counts')
        for direction in saved['directions'].values():
            require(direction['normalizer'] > 0 and direction['max_mahalanobis'] == 1.
                    and 0 < direction['nonzero_rows'] <= N, 'Recorded direction construction')
    check_finite(receipt)
    expected_rows = [(v, d, h, m) for v in NAMES for d in LOSSES for h in AMPLITUDES for m in LOSSES]
    require([(r['view'], r['direction_loss'], r['amplitude'], r['measured_loss']) for r in receipt['records']]
            == expected_rows, 'Fixed direction/amplitude matrix')
    recomputed = []
    for row in receipt['records']:
        view = views[row['view']]; index = LOSSES.index(row['measured_loss'])
        recomputed.append(check_record(row, view['baseline'][index], view['repeat'][index]))
    for i in range(0, len(receipt['records']), 2):
        a, b = receipt['records'][i:i+2]
        for key in ('changed_coordinates', 'actual_direction_rms', 'quantization_direction_rms'):
            require(a[key] == b[key], 'Measured losses did not share exact displacement metadata')
        for side in ('plus', 'minus'):
            for key in ('actual_max_mahalanobis', 'cast_error_rms'):
                require(a['one_sided'][side][key] == b['one_sided'][side][key], 'One-sided displacement metadata differs')
    own = [r for r in recomputed if r['direction_loss'] == r['measured_loss']]
    passed = sum(r['passed'] for r in own); measurable = sum(r['measurable'] for r in own)
    status = 'passed' if passed == 12 else 'not_passed'
    require(receipt['main_count'] == len(own) == 12 and receipt['main_passed'] == passed
            and receipt['main_measurable'] == measurable and receipt['numerical_status'] == status
            and receipt['cross_derivatives_are_descriptive'] is True, 'Overall numerical gate mismatch')
    close(receipt['maximum_main_relative_error'], max(r['relative_error'] for r in own), 'Maximum FD error')
    observed = {str(run/name): sha(run/name) for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json',
                'frozen_cpu_tests.json', 'execution_started.json', 'launch_started.json', 'run.log')}
    require(tree(snapshot) == sources, 'Sources changed during audit')
    for path, digest in {**inputs, **observed}.items():
        require(sha(path) == digest, 'Input/output changed during audit')
    return {'status': 'passed', 'plan_sha256': PLAN_SHA, 'source_count': len(sources), 'input_and_binary_count': len(inputs),
            'source_hashes': sources, 'observed_outputs': observed, 'counts': receipt['counts'],
            'producer_elapsed_seconds': receipt['elapsed_seconds'], 'outer_elapsed_seconds': launch['elapsed_seconds'],
            'numerical_status': status, 'main_passed': passed, 'main_count': 12, 'main_measurable': measurable,
            'own_direction_checks': own, 'cross_direction_checks_recomputed': 12,
            'maximum_scalar_absolute_difference': max(r['max_scalar_absolute_difference'] for r in recomputed),
            'maximum_actual_dot_identity_absolute_difference': max(r['actual_dot_identity_absolute_difference'] for r in recomputed),
            'maximum_loss_identity_absolute_difference': max(r['loss_identity_absolute_difference'] for r in recomputed),
            'recorded_maximum_one_sided_mahalanobis': max(r['recorded_actual_max_mahalanobis'] for r in recomputed),
            'restoration_record_consistent': True, 'dot_reduction_bound_independently_recomputed': False,
            'limitations': [
                'The audit passes when saved evidence and failed numerical gates agree; it does not pass the preflight.',
                'Recomputed scalar endpoint FD, scalar/repeat floors, relative errors, signal gates and one-sided identities only.',
                'No saved full gradients or displaced means: high-dimensional dots, absolute-product reduction mass, covariance directions, norm/cast statistics and alpha equality are not independently reconstructed.',
                'Reduction bound is inherited and checked only against a triangle-inequality lower bound; the overall floor is not a rigorous renderer error bound.',
                'Runtime scene/flags restoration is internally consistent and base file SHA unchanged, but process-local state cannot be independently re-observed after exit.',
                'Recorded FP32 displacement may exceed ideal h; this preflight does not apply the later optimizer cap.',
                'No GPU, render, optimizer, new RGB/label decode, VAL, or kernel-cause/performance claim.']}


def self_test():
    h = 1/64; baseline, repeat, lp, lm = 1., 1., 1.+h, 1.-h
    dot = 1.; bound = (3*N*math.ulp(1.))/(1-3*N*math.ulp(1.))
    floor = max(bound, (math.ulp(lp)+math.ulp(lm))/(2*h))
    row = {'amplitude': h, 'loss_plus': lp, 'loss_minus': lm,
           'analytic_actual_fp32_displacement': dot, 'central_difference': 1., 'absolute_error': 0.,
           'relative_error': 0., 'scalar_spacing_floor': (math.ulp(lp)+math.ulp(lm))/(2*h),
           'repeat_floor': 0., 'dot_reduction_bound': bound, 'floor': floor, 'measurable': True, 'passed': True,
           'changed_coordinates': 1, 'actual_direction_rms': 1., 'quantization_direction_rms': 0.,
           'view': 'synthetic', 'direction_loss': 'rgb_mse', 'measured_loss': 'rgb_mse',
           'one_sided': {'plus': {'actual_max_mahalanobis': h, 'analytic_actual_change': h, 'actual_loss_change': h, 'cast_error_rms': 0.},
                         'minus': {'actual_max_mahalanobis': h, 'analytic_actual_change': -h, 'actual_loss_change': -h, 'cast_error_rms': 0.}}}
    assert check_record(row, baseline, repeat)['passed']
    row['one_sided']['plus']['analytic_actual_change'] += .01
    try:
        check_record(row, baseline, repeat)
    except ValueError:
        pass
    else:
        raise AssertionError('Broken actual-dot identity accepted')
    print('Synthetic FD and negative one-sided identity contracts passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path); parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    require(args.run is not None and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only run required')
    run = args.run.resolve(); output = run/'independent_cpu_review.json'
    require(not output.exists(), 'Do not overwrite an audit')
    digest = sha(__file__); started = time.monotonic()
    with (run/'independent_audit_started.json').open('x') as stream:
        json.dump({'auditor_sha256': digest, 'plan_sha256': PLAN_SHA}, stream)
    report = {'status': 'failed', 'auditor_path': str(Path(__file__).resolve()), 'auditor_sha256': digest}
    def expired(*_):
        raise TimeoutError('Independent CPU audit exceeded 120 seconds')
    previous = signal.signal(signal.SIGALRM, expired); signal.alarm(120)
    try:
        report.update(audit(run))
        require(sha(__file__) == digest, 'Auditor changed during execution')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        report['elapsed_seconds'] = time.monotonic()-started
        payload = json.dumps(report, indent=2, allow_nan=False)+'\n'
        with output.open('x') as stream:
            stream.write(payload)


if __name__ == '__main__':
    main()
