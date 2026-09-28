"""Independent NumPy reduction of completed direction caches, never a renderer."""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

ATOL, RTOL = 1e-10, 1e-11


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check_hashes(mapping):
    for path, digest in mapping.items():
        require(sha(path) == digest, f'Hash mismatch: {path}')


def compare(actual, saved, label, errors):
    a, b = np.asarray(actual, np.float64), np.asarray(saved, np.float64)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label)
    error = np.abs(a-b)
    require(np.all(error <= ATOL+RTOL*np.abs(b)), f'Numeric mismatch: {label}')
    errors['values'] += a.size
    errors['maximum_absolute_error'] = max(errors['maximum_absolute_error'], float(error.max(initial=0)))


def view_terms(a, v, labels, weights, specification, gamma=None, bracket=None):
    """Direct full-view arithmetic, distinct from the producer's chunk reducer."""
    require(a.dtype == v.dtype == np.float32 and a.shape == v.shape == labels.shape
            and a.ndim == 1 and len(a) > 0 and labels.dtype == np.uint8, 'Cache dtype/grid')
    require(np.isfinite(a).all() and np.isfinite(v).all()
            and (a >= 0).all() and (v >= 0).all() and (labels < 5).all(), 'Cache domain')
    count, delta = len(a), specification['delta']
    raw_a, raw_v = a.astype(np.float64), v.astype(np.float64)
    p0, p1 = (1-delta)*raw_a+delta/5, (1-delta)*raw_v+delta/5
    difference = raw_v-raw_a
    dp = (1-delta)*difference
    w = weights[labels]
    pixel_derivative = -w*dp/p0
    upstream32 = (-w*(1-delta)/p0/count).astype(np.float32)
    output = {'F': float(np.mean(-w*np.log(p0))), 'F_vertex': float(np.mean(-w*np.log(p1))),
              'B64': float(np.mean(pixel_derivative)),
              'B32upstream': float(np.dot(upstream32.astype(np.float64), difference)),
              'absolute_B_pixel_mass': float(np.mean(np.abs(pixel_derivative))),
              'derivative_at_one': float(np.mean(-w*dp/p1))}
    edges = [*specification['bins'][:-1], np.inf]
    bins = {key: [] for key in ('pixel_counts', 'positive', 'negative', 'absolute')}
    for lo, hi in pairwise(edges):
        keep = (p0 >= lo) & (p0 < hi)
        terms = pixel_derivative[keep]
        bins['pixel_counts'].append(int(keep.sum()))
        bins['positive'].append(float(terms[terms > 0].sum()/count))
        bins['negative'].append(float(terms[terms < 0].sum()/count))
        bins['absolute'].append(float(np.abs(terms).sum()/count))
    output['bins'] = bins
    if gamma is not None:
        require(0 <= gamma <= 1, 'Candidate outside fixed segment')
        probability = p0*(1-gamma)+p1*gamma
        output['F_candidate'] = float(np.mean(-w*np.log(probability)))
        output['derivative_at_gamma'] = float(np.mean(-w*dp/probability))
        if bracket is not None:
            output['bracket_derivatives'] = [float(np.mean(-w*dp/(p0*(1-t)+p1*t))) for t in bracket]
    return output


def gate_values(a, b64, b32, spec):
    def pair(x, y):
        return abs(x-y) <= spec['absolute_tolerance']+spec['relative_tolerance']*max(abs(x), abs(y))
    require(np.isfinite([a, b64, b32]).all(), 'Nonfinite derivative')
    return {'A_vs_B32upstream': bool(pair(a, b32)), 'B32upstream_vs_B64': bool(pair(b32, b64)),
            'measurable': bool(min(abs(a), abs(b64), abs(b32)) >
                               spec['signal_multiplier']*spec['absolute_tolerance']),
            'all_negative': bool(max(a, b64, b32) < 0)}


def reduce_gradients(records, shape):
    aggregate = np.zeros(shape, np.float64)
    for record in records:
        g = np.load(record['gradient'], mmap_mode='r', allow_pickle=False)
        require(g.dtype == np.float32 and g.shape == shape and np.isfinite(g).all(), 'Bad g32 cache')
        aggregate += g.astype(np.float64)
    aggregate /= len(records)
    vertex = np.zeros(shape, np.float64)
    vertex[np.arange(shape[0]), np.argmin(aggregate, axis=1)] = 1
    return aggregate, vertex


def audit(run, expected):
    require(sha(run/'plan.json') == expected, 'Wrong plan')
    plan, launch, receipt = (read(run/name) for name in ('plan.json', 'launch_receipt.json', 'execution_receipt.json'))
    require(launch['status'] == 'completed' and launch['natural_completion'] is True and launch['exit_code'] == 0
            and launch['plan_sha256'] == expected
            and launch['execution_receipt_sha256'] == sha(run/'execution_receipt.json'), 'No bound natural completion')
    require(receipt['status'] == 'completed' and receipt['plan_sha256'] == expected
            and receipt['inputs_sources_unchanged'] is True, 'Incomplete producer')
    require(sha(run/'analysis.json') == receipt['analysis_sha256'], 'Analysis hash mismatch')
    report = read(run/'analysis.json')
    spec = plan['specification']
    require(spec['protocol'] in ('simplex_direction_diagnostic_v1', 'simplex_direction_diagnostic_v2')
            and spec['views'] == 259, 'Wrong contract')
    snapshot = Path(plan['source_snapshot'])
    actual_source = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    require(actual_source == plan['source_hashes'], 'Source tree mismatch')
    bindings = {**plan['input_hashes'], **plan['installed_sources']}
    binary = plan['expected_gsplat_binary']; bindings[binary['path']] = binary['sha256']
    check_hashes(bindings)
    require(report['actual_gsplat_binary'] == binary, 'Actual binary differs')
    for item in report['actual_imports'].values():
        path = Path(item['path'])
        require(path.is_relative_to(snapshot) and actual_source[str(path.relative_to(snapshot))] == item['sha256'],
                'Actual module not bound')
    records = report['cache_records']
    names = [v['name'] for v in plan['training_views']]
    require(len(names) == len(set(names)) == 259 and names == sorted(names)
            and all(v['split'] == 'train' for v in plan['training_views'])
            and [r['name'] for r in records] == names, 'Wrong fixed TRAIN cache order')
    expected_cache = {record[key] for record in records for key in ('a', 'v', 'labels', 'gradient')}
    if 'diagnostic_candidate' in report:
        expected_cache.add(report['diagnostic_candidate'])
    require(set(report['cache_hashes']) == expected_cache
            and {str(p) for p in (run/'cache').iterdir()} == expected_cache, 'Cache inventory differs')
    check_hashes(report['cache_hashes'])
    with np.load(plan['endpoint'], allow_pickle=False) as saved:
        master, q32 = saved['q_master'], saved['q_renderer']
    require(master.dtype == np.float64 and q32.dtype == np.float32
            and master.ndim == 2 and master.shape[1] == 5 and master.shape == q32.shape
            and np.isfinite(master).all() and (master >= 0).all()
            and np.max(np.abs(master.sum(1)-1)) <= 1e-12
            and master.astype(np.float32).tobytes() == q32.tobytes(), 'Endpoint simplex/cast differs')
    gradient, vertex = reduce_gradients(records, master.shape)
    displacement = vertex.astype(np.float32).astype(np.float64)-q32.astype(np.float64)
    errors = {'values': 0, 'maximum_absolute_error': 0.}
    a_full = float(np.sum(gradient*displacement))
    compare(a_full, report['A_from_full_gradient'], 'A full', errors)
    weights = np.asarray(plan['class_weights'], np.float32).astype(np.float64)
    gamma = report.get('gamma')
    bracket = report.get('line_search', {}).get('bracket')
    rows = []
    for record, stored in zip(records, report['derivative_at_zero']['rows'], strict=True):
        arrays = [np.load(record[key], mmap_mode='r', allow_pickle=False) for key in ('a', 'v', 'labels')]
        require(len(arrays[0]) == record['pixels'], 'Saved support differs')
        values = view_terms(*arrays, weights, spec, gamma, bracket)
        g = np.load(record['gradient'], mmap_mode='r', allow_pickle=False)
        values['A'] = float(np.sum(g.astype(np.float64)*displacement))
        compare(values['A'], record['A'], 'Per-view A', errors)
        compare(values['F'], record['F_current'], 'Current view F', errors)
        compare(values['F_vertex'], record['F_vertex'], 'Vertex view F', errors)
        require(stored['name'] == record['name'] and stored['pixels'] == record['pixels'], 'Summary order differs')
        for key in ('A', 'F', 'B64', 'B32upstream', 'absolute_B_pixel_mass'):
            compare(values[key], stored[key], f'Per-view {key}', errors)
        values.update(name=record['name'], pixels=record['pixels'])
        rows.append(values)
    totals = {key: float(np.mean([r[key] for r in rows]))
              for key in ('A', 'F', 'F_vertex', 'B64', 'B32upstream', 'absolute_B_pixel_mass', 'derivative_at_one')}
    for key in ('F', 'B64', 'B32upstream'):
        compare(totals[key], report['derivative_at_zero'][key], key, errors)
    compare(totals['A'], report['A_from_per_view'], 'A mean', errors)
    compare(totals['F'], report['baseline_objective'], 'Baseline F', errors)
    compare(totals['F_vertex'], report['vertex_objective'], 'Vertex F', errors)
    bins = {key: np.sum([r['bins'][key] for r in rows], axis=0) / (1 if key == 'pixel_counts' else 259)
            for key in ('pixel_counts', 'positive', 'negative', 'absolute')}
    for key, value in bins.items():
        compare(value, report['derivative_at_zero']['probability_bins'][key], 'Bin '+key, errors)
    compare(float(bins['positive'].sum()+bins['negative'].sum()), totals['B64'], 'Bin closure', errors)
    decisions = gate_values(totals['A'], totals['B64'], totals['B32upstream'], spec)
    saved_gate = report['direction_gate']
    for key in ('A_vs_B32upstream', 'B32upstream_vs_B64'):
        require(decisions[key] == saved_gate['comparisons'][key]['passed'], 'Derivative comparison gate differs')
    require(all(decisions.values()) == saved_gate['passed'] and decisions['measurable'] == saved_gate['measurable']
            and decisions['all_negative'] == saved_gate['all_negative'], 'Direction gate differs')
    candidate = None
    if not saved_gate['passed']:
        require(gamma is None and report['numerical_status'] == 'direction_inconsistent_or_unmeasurable',
                'A failed direction gate must stop before line search')
    else:
        require(gamma is not None, 'A passed direction gate must record its fixed line result')
    if gamma is not None:
        require(saved_gate['passed'], 'Candidate followed a failed direction gate')
        candidate = {'gamma': gamma, 'F_cached': float(np.mean([r['F_candidate'] for r in rows])),
                     'derivative_at_gamma': float(np.mean([r['derivative_at_gamma'] for r in rows]))}
        compare(candidate['F_cached'], report['cached_candidate']['F'], 'Cached candidate F', errors)
        compare(candidate['derivative_at_gamma'], report['cached_candidate']['B64'], 'Root derivative', errors)
        compare(totals['B64'], report['line_search']['derivative_at_zero'], 'Search derivative at zero', errors)
        compare(totals['derivative_at_one'], report['line_search']['derivative_at_one'], 'Search derivative at one', errors)
        if bracket is not None:
            require(0 <= bracket[0] <= gamma <= bracket[1] <= 1
                    and report['line_search']['iterations'] == 64, 'Root bracket contract')
            candidate['bracket_derivatives'] = np.mean([r['bracket_derivatives'] for r in rows], axis=0).tolist()
        else:
            require(gamma == 1 and totals['derivative_at_one'] <= 0, 'Wrong endpoint line minimizer')
        q_candidate = (1-gamma)*master+gamma*vertex
        rounding = q_candidate.astype(np.float32).astype(np.float64)-(q32.astype(np.float64)+gamma*displacement)
        compare(float(np.max(np.abs(rounding))), report['line_input_rounding_max'], 'Candidate cast deviation', errors)
        if 'diagnostic_candidate' in report:
            saved = np.load(report['diagnostic_candidate'], allow_pickle=False)
            require(saved.dtype == np.float64 and saved.tobytes() == q_candidate.tobytes(), 'Candidate q differs')
            actual_rows = report['actual_candidate_rows']
            require([r['name'] for r in actual_rows] == names, 'Actual candidate F population differs')
            actual = float(np.mean([r['F'] for r in actual_rows]))
            compare(actual, report['actual_candidate_objective'], 'Actual F row mean', errors)
            predicted_gain, actual_gain = totals['F']-candidate['F_cached'], totals['F']-actual
            tolerance = spec['absolute_tolerance']+spec['relative_tolerance']*max(abs(predicted_gain), abs(actual_gain))
            confirmed = predicted_gain > 0 and actual_gain > 0 and abs(predicted_gain-actual_gain) <= tolerance
            confirmation = report['candidate_confirmation']
            compare(predicted_gain, confirmation['predicted_decrease'], 'Predicted gain', errors)
            compare(actual_gain, confirmation['actual_decrease'], 'Actual gain', errors)
            require(bool(confirmed) == confirmation['passed'], 'Candidate confirmation gate differs')
            candidate.update(actual_F_from_saved_rows=actual, confirmed=bool(confirmed))
            require(report['numerical_status'] == ('direction_consistent_candidate_decreased'
                    if confirmed else 'candidate_not_confirmed'), 'Candidate status differs')
        else:
            require(np.array_equal(q_candidate.astype(np.float32), q32)
                    and report['numerical_status'] == 'fp32_stagnation', 'Missing actual candidate pass')
    count = 3 if 'actual_candidate_rows' in report else 2
    expected_counts = {'scene': 259*count, 'gsplat': 518*count, 'shader': 259*count,
                       'vjp': 259, 'target_decodes': 518, 'complete_passes': count}
    require(report['counts'] == expected_counts
            and receipt['counts'] == {**expected_counts, 'total_raster': 777*count}, 'Runtime counts differ')
    require(all(report[key] == 0 for key in ('val_views', 'teacher_calls', 'model_updates', 'head_calls')),
            'Excluded runtime activity')
    require(report['state_unchanged_before_restore'] and report['numerics_restored']
            and report['restoration']['state_exact'] and report['restoration']['flags_modes_gradients_restored'],
            'Missing restoration attestation')
    check_hashes(bindings); check_hashes(report['cache_hashes'])
    check_hashes({str(snapshot/name): digest for name, digest in actual_source.items()})
    return {'status': 'passed', 'plan_sha256': expected, 'comparison_errors': errors,
            'comparison_atol': ATOL, 'comparison_rtol': RTOL, 'source_count': len(actual_source),
            'input_count': len(plan['input_hashes']), 'cache_count': len(expected_cache), 'counts': expected_counts,
            'totals': totals, 'direction_gate_recomputed': decisions,
            'probability_bins_recomputed': {k: v.tolist() for k, v in bins.items()},
            'candidate': candidate, 'producer_numerical_status': report['numerical_status'],
            'lmo_vertex_sha256': hashlib.sha256(vertex.tobytes()).hexdigest(),
            'analysis_sha256': receipt['analysis_sha256'],
            'limits': ['Gradients are saved g32 values: this does not independently rerun W-transpose.',
                       'The fixed-cache scalar line is independently reduced; 64-step search was not rerun.',
                       'Actual candidate F is checked by averaging saved rows, not by re-rendering.',
                       'No GPU, new pixels, models, or producer numerical helpers were used.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    output = args.run/'independent_cpu_review.json'
    require(not output.exists(), 'Refuse overwrite of a previous audit')
    start = time.monotonic(); report = {'status': 'failed', 'plan_sha256': args.expected_plan_sha256}
    def timeout(*_):
        raise TimeoutError('Independent CPU audit 90-second limit')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(90)
    try:
        report = audit(args.run.resolve(), args.expected_plan_sha256)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-start, checker_path=str(Path(__file__).resolve()),
                      checker_sha256=sha(__file__))
        with output.open('x') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
