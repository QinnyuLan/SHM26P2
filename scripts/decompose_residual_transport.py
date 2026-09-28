"""One frozen CPU-only decomposition of already-audited TRAIN transport caches."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
PARENT = Path('/mnt/data/SHM2026/runs/residual_transport_probe_v1')
PARENT_PLAN_SHA = '061586d6309693a41c66b720159076e9bb52d6c1d7d605ec5d939a51402b297b'
PARENT_AUDIT_SHA = 'c2476de5dac9951b3f4318c1f55711f903fbf9dde5eea1c866d7cec877ce2b33'
NAMES = ['002.png', '021.png', '041.png', '059.png', '079.png', '100.png',
         '118.png', '137.png', '156.png', '176.png', '200.png', '220.png',
         '241.png', '259.png', '278.png', '300.png']
ARMS = ('zero', 'source_residual', 'render_only', 'full_photo')
SPEC = {'protocol': 'residual_transport_decomposition_v1', 'targets': NAMES,
        'arms': list(ARMS), 'folds': 'A even / B odd; other-fold coefficient only',
        'definition': 's=cached source residual; d=M*(same-weight warp(Rs)-Rt); full=s+d',
        'support': 'parent common valid and source_count exact; unsupported corrections zero',
        'fit': 'each nonzero arm two independent [0,1] scalars, equal-view full-valid unclipped MSE',
        'wrong': 'parent report reference only; no new fit',
        'expansion': 'five unclipped MSE change terms at heldout full_photo coefficient',
        'internal_seconds': 90, 'external_seconds': 120, 'scene_calls': 0,
        'original_image_decodes': 0, 'semantic_labels': 0, 'VAL': 0, 'model_updates': 0,
        'adoption_gate': None}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(payload)


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def load_helper(path):
    spec = importlib.util.spec_from_file_location('frozen_transport_math', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parent_metadata():
    plan, execution, launch, audit, audit_launch, analysis = (read(PARENT/n) for n in
        ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json',
         'independent_audit_launch_receipt.json', 'analysis.json'))
    require(sha(PARENT/'plan.json') == PARENT_PLAN_SHA and sha(PARENT/'independent_cpu_review.json') == PARENT_AUDIT_SHA,
            'Fixed parent identity')
    require(execution['status'] == launch['status'] == 'completed'
            and launch['natural_completion'] and launch['exit_code'] == 0
            and execution['plan_sha256'] == launch['plan_sha256'] == PARENT_PLAN_SHA
            and launch['execution_receipt_sha256'] == sha(PARENT/'execution_receipt.json'), 'Parent naturally completed')
    require(audit['status'] == 'passed' and audit['plan_sha256'] == PARENT_PLAN_SHA
            and audit['execution_receipt_sha256'] == sha(PARENT/'execution_receipt.json')
            and audit_launch['status'] == 'completed' and audit_launch['natural_completion']
            and audit_launch['exit_code'] == 0 and audit_launch['report_sha256'] == PARENT_AUDIT_SHA,
            'Parent independent audit must complete')
    require(execution['analysis_sha256'] == sha(PARENT/'analysis.json')
            and [r['name'] for r in analysis['views']] == NAMES
            and [r['name'] for r in plan['camera_selection']] == NAMES, 'Fixed parent analysis/list')
    return plan, execution, analysis


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output')
    parent, execution, analysis = parent_metadata()
    helper = Path(parent['source_snapshot'])/'bridge_rgs/residual_transport.py'
    require(sha(helper) == parent['source_hashes']['bridge_rgs/residual_transport.py'], 'Unchanged parent helper')
    records = execution['renders']+execution['transports']+[r['target'] for r in analysis['views']]
    require(len(execution['renders']) == 78 and len(execution['transports']) == 16
            and len({r['path'] for r in records}) == 110, 'Exactly 78+16+16 caches')
    inputs = {str(PARENT/n): sha(PARENT/n) for n in ('plan.json', 'execution_receipt.json',
        'launch_receipt.json', 'independent_cpu_review.json', 'independent_audit_launch_receipt.json', 'analysis.json')}
    inputs.update({r['path']: r['sha256'] for r in records})
    inputs[str(helper)] = sha(helper)
    require(all(sha(p) == digest for p, digest in inputs.items()), 'Parent cached inputs changed')
    snapshot = output/'source_snapshot'; snapshot.mkdir(parents=True)
    for source, name in ((Path(__file__), 'decompose_residual_transport.py'), (helper, 'residual_transport.py'),
                        (ROOT/'tests/test_residual_transport_decomposition.py', 'test_residual_transport_decomposition.py'),
                        (ROOT/'docs/residual_transport_decomposition_protocol.md', 'protocol.md'),
                        (ROOT/'uv.lock', 'uv.lock')):
        shutil.copy2(source, snapshot/name)
    plan = {'specification': SPEC, 'output': str(output), 'parent': str(PARENT),
            'source_snapshot': str(snapshot), 'source_hashes': tree(snapshot), 'input_hashes': inputs,
            'camera_selection': parent['camera_selection'], 'renders': execution['renders'],
            'transports': execution['transports'], 'targets': [{'name': r['name'], **r['target']} for r in analysis['views']],
            'runtime': {n: importlib.metadata.version(n) for n in ('numpy', 'opencv-python-headless')},
            'python': sys.version, 'prepare_array_loads': 0}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json'),
                      'source_files': len(plan['source_hashes']), 'input_files': len(inputs)}), flush=True)


def verify(plan):
    require(plan['specification'] == SPEC and tree(Path(plan['source_snapshot'])) == plan['source_hashes'], 'Source/spec changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/'decompose_residual_transport.py', 'Frozen entry only')
    require(sys.version == plan['python'], 'Python changed')
    require(all(importlib.metadata.version(n) == v for n, v in plan['runtime'].items()), 'CPU runtime changed')
    require(all(sha(p) == digest for p, digest in plan['input_hashes'].items()), 'Bound inputs changed')


def load_arrays(record):
    with np.load(record['path'], allow_pickle=False) as stream:
        return {k: stream[k] for k in stream.files}


def decompose_arrays(base, source_residual, warped_render, valid, source_count):
    base, source_residual, warped_render = (np.asarray(x, np.float64) for x in (base, source_residual, warped_render))
    require(base.ndim == 3 and base.shape[-1] == 3 and source_residual.shape == warped_render.shape == base.shape,
            'HWC3 common color grid')
    valid, source_count = np.asarray(valid), np.asarray(source_count)
    require(valid.dtype == bool and valid.shape == source_count.shape == base.shape[:2]
            and np.issubdtype(source_count.dtype, np.integer) and np.all(source_count >= 0)
            and np.array_equal(valid, source_count > 0), 'Exact binary support/count')
    require(all(np.isfinite(x).all() for x in (base, source_residual, warped_render)), 'Finite components')
    require(np.all(source_residual[~valid] == 0), 'Cached s must be zero outside support')
    d = np.where(valid[..., None], warped_render-base, 0.)
    full = source_residual+d
    require(np.isfinite(full).all(), 'Finite summed correction')
    return {'render_only': d, 'full_photo': full, 'valid': valid, 'source_count': source_count}


def prediction_barrier(records, names=NAMES):
    require([r['name'] for r in records] == list(names) and len(set(names)) == len(names)
            and all(r.get('path') and r.get('sha256') for r in records), 'All fixed decompositions precede targets')


def cross_fit(rows, helper):
    require([r['name'] for r in rows] == NAMES
            and [r['fold'] for r in rows] == ['A' if i % 2 == 0 else 'B' for i in range(16)], 'Fixed folds')
    return {arm: {fold: helper.fit_shrinkage([r['statistics'][arm] for r in rows if r['fold'] == fold])
                  for fold in ('A', 'B')} for arm in ARMS[1:]}


def coefficient_for(row, arm, fits):
    require(arm in ARMS and row['fold'] in ('A', 'B'), 'Fixed arm/fold')
    other = 'B' if row['fold'] == 'A' else 'A'
    return (0., None) if arm == 'zero' else (fits[arm][other]['coefficient'], other)


def metrics(base, truth, correction, coefficient, valid):
    prediction = np.asarray(base, np.float64)+coefficient*correction
    out = {}
    for name, value in (('unclipped', prediction), ('clipped', np.clip(prediction, 0, 1))):
        error = (value-np.asarray(truth, np.float64))[valid]
        require(error.size > 0 and np.isfinite(error).all(), 'Nonempty finite full-valid scoring')
        mse = float(np.mean(error**2))
        require(np.isfinite(mse), 'Finite MSE')
        out[name+'_MSE'] = mse
        out[name+'_log10_MSE'] = float(np.log10(mse)) if mse > 0 else None
    return out


def five_term_expansion(base, truth, s, d, coefficient, valid):
    e = (np.asarray(base, np.float64)-np.asarray(truth, np.float64))[valid]
    s, d = s[valid], d[valid]
    require(e.size and np.isfinite(e).all() and np.isfinite(s).all() and np.isfinite(d).all(), 'Finite expansion')
    c = float(coefficient)
    terms = {'linear_source': float(2*c*np.mean(e*s)), 'linear_render': float(2*c*np.mean(e*d)),
             'quadratic_source': float(c*c*np.mean(s*s)), 'quadratic_render': float(c*c*np.mean(d*d)),
             'cross_source_render': float(2*c*c*np.mean(s*d))}
    direct = float(np.mean((e+c*(s+d))**2)-np.mean(e**2))
    total = float(sum(terms.values()))
    tolerance = float(64*np.finfo(np.float64).eps*max(1., abs(direct), sum(abs(x) for x in terms.values())))
    require(abs(total-direct) <= tolerance, 'Unclipped five-term identity failed')
    return {'terms': terms, 'sum': total, 'direct_MSE_change': direct,
            'absolute_error': abs(total-direct), 'tolerance': tolerance}


def summarize(rows):
    result = {}
    for fold in ('all', 'A', 'B'):
        subset = rows if fold == 'all' else [r for r in rows if r['fold'] == fold]
        result[fold] = {}
        for arm in ARMS:
            result[fold][arm] = {'views': len(subset)}
            for domain in ('unclipped', 'clipped'):
                mse = [r['metrics'][arm][domain+'_MSE'] for r in subset]
                logs = [r['metrics'][arm][domain+'_log10_MSE'] for r in subset]
                mean_log = float(np.mean(logs)) if all(x is not None for x in logs) else None
                result[fold][arm].update({domain+'_MSE': float(np.mean(mse)),
                    domain+'_mean_log10_MSE': mean_log,
                    domain+'_mean_PSNR_dB': -10*mean_log if mean_log is not None else None,
                    domain+'_zero_MSE_views': sum(x == 0 for x in mse)})
    return result


def run(plan, report):
    output = Path(plan['output']); arrays_dir = output/'arrays'; arrays_dir.mkdir()
    helper = load_helper(Path(plan['source_snapshot'])/'residual_transport.py')
    require('torch' not in sys.modules, 'CPU only, no Torch import')
    render_records = {r['name']: r for r in plan['renders']}
    transport_records = {r['name']: r for r in plan['transports']}
    target_records = {r['name']: r for r in plan['targets']}
    # One target and its four sources at a time; avoid loading the entire bank.
    for selection in plan['camera_selection']:
        name = selection['name']; target = load_arrays(render_records[name])
        sources = []
        for record in selection['sources']:
            source = load_arrays(render_records[record['name']])
            source['residual'] = source['rgb']  # Geometry unchanged; same helper/table/weights.
            sources.append(source)
        warped = helper.transport_residual(target['depth'], target['alpha'], target['valid'],
                                           target['K'], target['w2c'], sources)
        old = load_arrays(transport_records[name])
        require(np.array_equal(warped['valid'], old['valid'])
                and np.array_equal(warped['source_count'], old['source_count']), 'Parent support/count changed')
        components = decompose_arrays(target['rgb'], old['true_residual'], warped['true_residual'],
                                      old['valid'], old['source_count'])
        path = arrays_dir/('decomposition_'+name+'.npz')
        with path.open('xb') as stream:
            np.savez(stream, **components)
        report['decompositions'].append({'name': name, 'path': str(path), 'sha256': sha(path),
                                         'common_pixels': int(old['valid'].sum()), 'support_count_exact': True})
    prediction_barrier(report['decompositions'])
    report['prediction_barrier_complete'] = True
    print(json.dumps({'phase': 'all_16_decompositions_saved', 'target_cache_loads': 0}), flush=True)
    decompositions = {r['name']: r for r in report['decompositions']}
    rows = []
    # Targets are first accessed here. Only sufficient statistics survive each view.
    for selection in plan['camera_selection']:
        name = selection['name']; prediction_barrier(report['decompositions'])
        base = load_arrays(render_records[name]); target = load_arrays(target_records[name])
        report['target_cache_loads'] += 1
        require(np.array_equal(base['valid'], target['valid']), 'Target full-valid grid changed')
        old = load_arrays(transport_records[name]); new = load_arrays(decompositions[name])
        corrections = {'source_residual': old['true_residual'], **{a: new[a] for a in ARMS[2:]}}
        stats = {a: helper.shrink_statistics(base['rgb'], target['rgb'], corrections[a], target['valid']) for a in ARMS[1:]}
        rows.append({'name': name, 'fold': selection['fold'], 'statistics': stats})
    fits = cross_fit(rows, helper)
    old_analysis = read(Path(plan['parent'])/'analysis.json')
    require(fits['source_residual'] == old_analysis['fits']['true'], 'Original source-residual fits changed')
    for row in rows:
        name = row['name']; base = load_arrays(render_records[name]); target = load_arrays(target_records[name])
        report['target_cache_loads'] += 1
        old = load_arrays(transport_records[name]); new = load_arrays(decompositions[name])
        corrections = {'zero': np.zeros_like(old['true_residual']), 'source_residual': old['true_residual'],
                       **{a: new[a] for a in ARMS[2:]}}
        row['metrics'] = {}
        for arm in ARMS:
            c, fold = coefficient_for(row, arm, fits)
            row['metrics'][arm] = {'coefficient': c, 'fit_fold': fold,
                **metrics(base['rgb'], target['rgb'], corrections[arm], c, target['valid'])}
        full_c, _ = coefficient_for(row, 'full_photo', fits)
        row['full_photo_expansion'] = five_term_expansion(base['rgb'], target['rgb'], corrections['source_residual'],
                                                         corrections['render_only'], full_c, target['valid'])
    summary = summarize(rows)
    for new_arm, old_arm in (('zero', 'zero'), ('source_residual', 'true')):
        for name in ('clipped_MSE', 'unclipped_MSE'):
            require(summary['all'][new_arm][name] == old_analysis['equal_view_mean'][old_arm][name], 'Parent metric changed')
    analysis = {'scope': 'cached conditional TRAIN decomposition; scalar OOF only, no adoption gate',
                'fits': fits, 'views': rows, 'equal_view_summary': summary,
                'old_wrong_reference': {'source': str(Path(plan['parent'])/'analysis.json'),
                    'sha256': sha(Path(plan['parent'])/'analysis.json'), 'fits': old_analysis['fits']['wrong'],
                    'equal_view_mean': old_analysis['equal_view_mean']['wrong'], 'refit': False},
                'photo_identity_limit': 'full=s+d from cached FP32 Is-Rs; not original Is byte reconstruction'}
    write(output/'analysis.json', analysis)
    report['analysis_sha256'] = sha(output/'analysis.json')
    report['helper_sha256'] = sha(helper.__file__)
    require('torch' not in sys.modules and report['target_cache_loads'] == 32, 'CPU cache-only fixed counts')


def execute(path, expected):
    require(sha(path) == expected, 'Plan hash changed')
    plan = read(path); output = Path(plan['output'])
    require(not any((output/n).exists() for n in ('execution_started.json', 'execution_receipt.json', 'analysis.json')), 'One attempt only')
    write(output/'execution_started.json', {'plan_sha256': expected})
    start = time.monotonic()
    report = {'status': 'running', 'plan_sha256': expected, 'decompositions': [],
              'target_cache_loads': 0, 'original_image_decodes': 0, 'scene_calls': 0,
              'semantic_GT': 0, 'VAL': 0, 'model_updates': 0, 'cuda_used': False}
    def expired(*_):
        raise TimeoutError('Fixed 90-second CPU budget exhausted')
    old = signal.signal(signal.SIGALRM, expired); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan); run(plan, report); verify(plan)
        report.update(status='completed', source_inputs_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)
        report['elapsed_seconds'] = time.monotonic()-start
        write(output/'execution_receipt.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path); mode.add_argument('--execute', type=Path)
    parser.add_argument('--expected-plan-sha256'); args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'CPU and no bytecode required')
    prepare(args.prepare) if args.prepare else execute(args.execute, args.expected_plan_sha256)
