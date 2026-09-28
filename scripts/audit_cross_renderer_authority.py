"""Independent CPU audit with direct centered-design ridge fits; no analyzer import."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.linalg import cho_factor, cho_solve

NAMES = ['002.png', '021.png', '041.png', '059.png', '079.png', '100.png', '118.png',
         '137.png', '156.png', '176.png', '200.png', '220.png', '241.png', '259.png', '278.png', '300.png']
KINDS = ('S', 'T1', 'T2', 'Tm')
READERS = ('baseline', 'candidate', 'permuted')
CASES = ('both_correct', 'S_only_correct', 'Tm_only_correct', 'both_wrong')


def need(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, message, atol=1e-10, rtol=1e-7):
    need(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True), message)


def norm(value):
    value = np.asarray(value, np.float64)
    return value/value.sum(-1, keepdims=True)


def entropy(value):
    return -(value*np.log(np.maximum(value, 1e-12))).sum(-1)


def disagreement(first, second):
    p, q = norm(first), norm(second)
    # Entropy identity is independent of the collector analyzer's KL expression.
    return np.maximum(0, entropy((p+q)*.5)-.5*(entropy(p)+entropy(q)))


def permute(values, view_index):
    h, w = values.shape
    groups = {}
    for y in range(0, h, 32):
        for x in range(0, w, 32):
            groups.setdefault((min(32, h-y), min(32, w-x)), []).append((y, x))
    rng = np.random.default_rng(np.random.SeedSequence([20260926, view_index]))
    result = np.empty_like(values)
    moved = 0
    metadata = []
    for (bh, bw), positions in sorted(groups.items()):
        order = rng.permutation(len(positions))
        for j, (y, x) in enumerate(positions):
            sy, sx = positions[order[j]]
            result[y:y+bh, x:x+bw] = values[sy:sy+bh, sx:sx+bw]
            moved += int(j != order[j])*bh*bw
        metadata.append({'shape': [bh, bw], 'block_count': len(positions),
                         'destination_to_source': order.tolist(), 'fixed_blocks': int((order == np.arange(len(order))).sum())})
    return result, {'groups': metadata, 'moved_pixels': moved, 'total_pixels': h*w, 'moved_fraction': moved/(h*w)}


def compact(s, t, d, shuffled, y):
    s, t = norm(s), norm(t)
    onehot = np.eye(5)[y]
    z = ((t-onehot)**2).sum(-1)-((s-onehot)**2).sum(-1)
    return {'pair': 5*s.argmax(-1)+t.argmax(-1), 'hs': entropy(s), 'ht': entropy(t),
            'd': np.asarray(d), 'shuffled': np.asarray(shuffled), 'z': z}


def design(values, reader):
    n = len(values['z'])
    x = np.zeros((n, 75 if reader == 'baseline' else 100), dtype=np.float64)
    row, pair = np.arange(n), values['pair']
    x[row, 3*pair] = 1
    x[row, 3*pair+1] = values['hs']
    x[row, 3*pair+2] = values['ht']
    if reader != 'baseline':
        x[row, 75+pair] = values['d' if reader == 'candidate' else 'shuffled']
    return x


def direct_fit(values, reader):
    """Center actual X first, avoiding the analyzer's raw second-moment subtraction."""
    x = design(values, reader)
    mean, std = x.mean(0), x.std(0)
    active = std > 1e-12
    scale = np.where(active, std, 1)
    x -= mean
    x /= scale
    x[:, ~active] = 0
    z = values['z']
    centered_z = z-z.mean()
    lhs = x.T@x/len(z)
    lhs.flat[::len(mean)+1] += .001
    beta = cho_solve(cho_factor(lhs, lower=True), x.T@centered_z/len(z))
    coefficient = beta/scale
    return {'mean': mean, 'std': std, 'active': active, 'beta': beta, 'coefficient': coefficient,
            'intercept': float(z.mean()-mean@coefficient), 'n': len(z)}


def predict(values, reader, model):
    return design(values, reader)@model['coefficient']+model['intercept']


def load_view(view, index):
    arrays = {key: np.load(view[key]['path'], mmap_mode='r', allow_pickle=False) for key in KINDS}
    # This function is called only after the entire 64-array preflight succeeds.
    y = np.array(Image.open(view['mask']['path']))
    valid_image = np.array(Image.open(view['valid']['path']))
    shape = arrays['S'].shape[:2]
    need(y.shape == valid_image.shape == shape and y.dtype == np.uint8, 'GT shape/dtype')
    need(np.isin(y, [0, 1, 2, 3, 4, 255]).all() and np.isin(valid_image, [0, 1, 255]).all(), 'GT/valid IDs')
    valid = (y < 5) & (valid_image > 0)
    d = np.empty(shape, np.float64)
    for row in range(0, shape[0], 31):
        d[row:row+31] = disagreement(arrays['T1'][row:row+31], arrays['T2'][row:row+31])
    shuffled, metadata = permute(d, index)
    return arrays, y, valid, d, shuffled, metadata


def evaluate(arrays, y, valid, d, shuffled, models, old):
    counts = np.zeros(5, np.int64)
    cases = np.zeros((5, 4), np.int64)
    sse = {key: np.zeros(5) for key in READERS}
    zsum = np.zeros(5)
    cms = {key: np.zeros((5, 5), np.int64) for key in ('S', 'Tm', 'half', *['route_'+r for r in READERS])}
    pred_diffs = {key: 0. for key in READERS}
    old_route_differences = {key: 0 for key in READERS}
    pair_counts = np.zeros((5, 25), np.int64)
    pair_sse = {key: np.zeros((5, 25)) for key in READERS}
    pair_abs_prediction = {key: np.zeros(25) for key in READERS}
    standard_feature_absmax = {key: np.zeros(75 if key == 'baseline' else 100) for key in READERS}
    for row in range(0, len(y), 31):
        support = valid[row:row+31]
        labels = y[row:row+31][support]
        s, t = norm(arrays['S'][row:row+31][support]), norm(arrays['Tm'][row:row+31][support])
        values = compact(s, t, d[row:row+31][support], shuffled[row:row+31][support], labels)
        pair_index = 25*labels.astype(np.int64)+values['pair']
        pair_counts += np.bincount(pair_index, minlength=125).reshape(5, 25)
        ps, pt = s.argmax(-1), t.argmax(-1)
        predictions = {'S': ps, 'Tm': pt, 'half': (s+t).argmax(-1)}
        counts += np.bincount(labels, minlength=5)
        zsum += np.bincount(labels, weights=values['z'], minlength=5)
        code = np.where(ps == labels, np.where(pt == labels, 0, 1), np.where(pt == labels, 2, 3))
        cases += np.bincount(4*labels.astype(np.int64)+code, minlength=20).reshape(5, 4)
        for key in READERS:
            x = design(values, key)
            model = models[key]
            zhat = x@model['coefficient']+model['intercept']
            other = x@np.asarray(old[key]['raw_coefficient'])+old[key]['intercept']
            pred_diffs[key] = max(pred_diffs[key], float(np.max(np.abs(zhat-other), initial=0)))
            old_route_differences[key] += int(((zhat > 0) != (other > 0)).sum())
            sse[key] += np.bincount(labels, weights=(zhat-values['z'])**2, minlength=5)
            pair_sse[key] += np.bincount(pair_index, weights=(zhat-values['z'])**2, minlength=125).reshape(5, 25)
            np.maximum.at(pair_abs_prediction[key], values['pair'], abs(zhat))
            scaled = (x-model['mean'])/np.where(model['active'], model['std'], 1)
            scaled[:, ~model['active']] = 0
            standard_feature_absmax[key] = np.maximum(standard_feature_absmax[key], np.max(abs(scaled), axis=0, initial=0))
            predictions['route_'+key] = np.where(zhat > 0, ps, pt)
        for key, ids in predictions.items():
            cms[key] += np.bincount(labels.astype(np.int64)*5+ids, minlength=25).reshape(5, 5)
    return {'class_pixels': counts.tolist(), 'valid_pixels': int(counts.sum()),
        'mse': {k: float(v.sum()/counts.sum()) for k, v in sse.items()},
        'class_mse': {k: [float(v[c]/counts[c]) if counts[c] else None for c in range(5)] for k, v in sse.items()},
        'class_mean_relative_brier': [float(zsum[c]/counts[c]) if counts[c] else None for c in range(5)],
        'class_case_counts': cases.tolist(), 'case_counts': dict(zip(CASES, map(int, cases.sum(0)), strict=True)),
        'confusion_matrices': {k: v.tolist() for k, v in cms.items()},
        'extrapolation_diagnostics': {'gt_class_by_pair_counts': pair_counts.tolist(),
            'gt_class_by_pair_sse': {k: v.tolist() for k, v in pair_sse.items()},
            'pair_abs_prediction_max': {k: v.tolist() for k, v in pair_abs_prediction.items()},
            'standardized_feature_absmax': {k: v.tolist() for k, v in standard_feature_absmax.items()},
            'scope': 'Posthoc descriptive diagnostics on the same fixed outputs; no ridge/threshold changes'},
        'direct_vs_saved_prediction_absmax': pred_diffs, 'direct_vs_saved_routing_sign_differences': old_route_differences}


def independent_summary(rows):
    rng = np.random.default_rng(20260926)
    weights = np.stack([np.bincount(rng.integers(0, 16, 16), minlength=16) for _ in range(5000)])
    comparisons = {}
    joint = set(range(5))
    for key in ('baseline', 'permuted'):
        original = np.array([r['mse'][key] for r in rows])
        candidate = np.array([r['mse']['candidate'] for r in rows])
        delta = original-candidate
        interval = np.quantile(weights@delta/16, [.025, .975])
        relative = float(delta.mean()/original.mean()) if original.mean() > 0 else None
        class_gains = []
        for c in range(5):
            gains = [r['class_mse'][key][c]-r['class_mse']['candidate'][c] for r in rows if r['class_mse'][key][c] is not None]
            value = float(np.mean(gains)) if gains else None
            class_gains.append({'class_id': c, 'present_views': len(gains), 'view_equal_mean_gain': value})
            if value is None or value <= 0:
                joint.discard(c)
        clauses = {'relative_gain_at_least_1_percent': relative is not None and relative >= .01,
                   'paired_lower_positive': bool(interval[0] > 0), 'positive_views_at_least_10': int((delta > 0).sum()) >= 10}
        comparisons[key] = {'reference_mean_mse': float(original.mean()), 'candidate_mean_mse': float(candidate.mean()),
            'mean_mse_gain': float(delta.mean()), 'relative_mse_gain': relative, 'paired_view_bootstrap_95_gain': interval.tolist(),
            'positive_views': int((delta > 0).sum()), 'per_view_gain': delta.tolist(), 'class_gains': class_gains, 'clauses': clauses}
    support = {}
    for case in ('S_only_correct', 'Tm_only_correct'):
        n = [r['case_counts'][case] for r in rows]
        support[case] = {'pixels': sum(n), 'views_with_events': sum(v > 0 for v in n), 'enough': sum(n) >= 100 and sum(v > 0 for v in n) >= 4}
    enough = all(v['enough'] for v in support.values())
    passed = enough and len(joint) >= 2 and all(all(v['clauses'].values()) for v in comparisons.values())
    return {'comparisons': comparisons, 'error_support': support, 'joint_positive_classes': sorted(joint),
        'at_least_two_joint_positive_classes': len(joint) >= 2, 'resource_gate_passed': bool(passed),
        'status': 'inconclusive_error_support' if not enough else 'necessary_increment_present' if passed else 'necessary_increment_not_met'}


def nested_compare(actual, expected, prefix=''):
    if isinstance(actual, dict):
        for key, value in actual.items():
            nested_compare(value, expected[key], prefix+'/'+key)
    elif isinstance(actual, list):
        need(len(actual) == len(expected), prefix+' length')
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            nested_compare(a, b, prefix+f'/{i}')
    elif isinstance(actual, float):
        close(actual, expected, prefix)
    else:
        need(actual == expected, prefix)


def audit(run, analysis, expected_plan):
    start = time.monotonic()
    run, analysis = Path(run).resolve(), Path(analysis).resolve()
    out = run/'independent_cpu_review.json'
    need(not out.exists(), 'Never overwrite audit')
    need(sha(run/'plan.json') == expected_plan, 'Locked plan')
    plan, receipt, launch = [read(run/n) for n in ('plan.json', 'execution_receipt.json', 'launch_receipt.json')]
    ar = read(analysis/'execution_receipt.json')
    al = read(analysis/'launch_receipt.json')
    need(receipt['status'] == ar['status'] == launch['status'] == 'completed' and launch['natural_completion']
         and type(launch['exit_code']) is int and launch['exit_code'] == 0
         and launch['execution_receipt_sha256'] == sha(run/'execution_receipt.json'), 'Both computations complete before any GT reads')
    need(al['status'] == 'completed' and al['natural_completion'] is True and al['exit_code'] == 0
         and al['execution_receipt_sha256'] == sha(analysis/'execution_receipt.json'), 'Analysis natural exit')
    need(receipt['inputs_and_sources_unchanged'] and receipt['plan_sha256'] == expected_plan, 'Collector source binding')
    need(ar['input_bytes_unchanged'] and ar['gpu_used'] is False and ar['input_sha256'] == sha(run/'analysis_input.json')
         and ar['analysis_sha256'] == sha(analysis/'analysis.json'), 'Analysis binding')
    inp, original = read(run/'analysis_input.json'), read(analysis/'analysis.json')
    need(inp['specification'] == original['specification'] == ar['specification'] == plan['analysis_specification'], 'Analysis fixed specification')
    need(inp['prediction_receipt']['sha256'] == sha(run/'execution_receipt.json'), 'Prediction receipt binding')
    snapshot = Path(plan['source_snapshot'])
    tree = {str(f.relative_to(snapshot)): sha(f) for f in snapshot.rglob('*') if f.is_file() and f.suffix in {'.py', '.md'}}
    need(tree == plan['source_hashes'] and ar['script_sha256'] == tree['analyze_cross_renderer_authority.py'], 'Frozen inventory')
    for path, digest in plan['input_hashes'].items():
        need(sha(path) == digest, 'Bound collector input '+path)
    for record in receipt['actual_imports'].values():
        path = Path(record['path'])
        need(path.is_relative_to(snapshot) and record['sha256'] == tree[str(path.relative_to(snapshot))], 'Actual import')
    need(receipt['scene_renders'] == 16 and receipt['teacher_predict_image_calls'] == 48
         and receipt['gt_payload_reads'] == receipt['real_rgb_payload_reads'] == receipt['optimizer_steps'] == 0, 'Fixed GPU budget and isolation')
    need([v['name'] for v in inp['views']] == [v['name'] for v in plan['views']] == NAMES, 'Fixed 16 TRAIN names')
    need(all(v['split'] == 'train' for v in inp['views']), 'TRAIN only')
    old_targets = read(plan['target_sha_source']['path'])['input_hashes']
    need(sha(plan['target_sha_source']['path']) == plan['target_sha_source']['sha256'], 'Historical target source')
    probability_paths = []
    for view, locked in zip(inp['views'], plan['views'], strict=True):
        for key in ('mask', 'valid'):
            need(view[key] == locked[key] and view[key]['sha256'] == old_targets[view[key]['path']], 'TRAIN label provenance')
        for key in KINDS:
            record = view[key]
            need(sha(record['path']) == record['sha256'], 'Probability bytes')
            value = np.load(record['path'], mmap_mode='r', allow_pickle=False)
            need(value.shape == (989, 1320, 5) and record['shape'] == list(value.shape)
                 and value.dtype == np.float32 and record['dtype'] == 'float32', 'Fixed probability grid')
            for row in range(0, len(value), 31):
                chunk = value[row:row+31]
                need(np.isfinite(chunk).all() and chunk.min() >= 0 and chunk.max() <= 1
                     and np.max(abs(chunk.sum(-1, dtype=np.float64)-1)) <= 5e-6, 'Probability contract')
            probability_paths.append(record['path'])
    need(len(set(probability_paths)) == 64, '64 unique predictions before any target bytes')
    # Only below this barrier are mask/valid bytes opened or hashed.
    compact_views = []
    for i, view in enumerate(inp['views']):
        for key in ('mask', 'valid'):
            need(sha(view[key]['path']) == view[key]['sha256'], 'Actual target SHA')
        arrays, y, valid, d, shuffled, meta = load_view(view, i)
        support = valid[::4, ::4]
        data = compact(arrays['S'][::4, ::4][support], arrays['Tm'][::4, ::4][support],
                       d[::4, ::4][support], shuffled[::4, ::4][support], y[::4, ::4][support])
        compact_views.append(data)
        need(len(data['z']) == original['views'][i]['fit_grid_pixels'] and meta == original['views'][i]['permutation'], 'Fit pixels and independent block permutation')
    models, fit_diffs = {}, {}
    for fold in range(4):
        data = {key: np.concatenate([v[key] for i, v in enumerate(compact_views) if i % 4 != fold]) for key in compact_views[0]}
        models[fold], fit_diffs[fold] = {}, {}
        for key in READERS:
            model = direct_fit(data, key)
            models[fold][key] = model
            old = original['fold_models'][str(fold)][key]
            need(model['n'] == old['n_train_pixels'] and np.array_equal(model['active'], old['active']), 'Ridge training support')
            for current, recorded in [('mean', 'mean'), ('std', 'std'), ('beta', 'standardized_coefficient'), ('coefficient', 'raw_coefficient')]:
                close(model[current], old[recorded], 'Independent direct-design ridge '+current, atol=1e-7, rtol=1e-6)
            close(model['intercept'], old['intercept'], 'Ridge intercept', atol=1e-7, rtol=1e-6)
            fit_diffs[fold][key] = {'raw_coefficient_max_abs': float(np.max(abs(model['coefficient']-old['raw_coefficient']))),
                                   'intercept_abs': abs(model['intercept']-old['intercept']), 'fit_pixels': model['n']}
        print(json.dumps({'independent_fold_done': fold}), flush=True)
    rows = []
    for i, view in enumerate(inp['views']):
        arrays, y, valid, d, shuffled, _ = load_view(view, i)
        result = evaluate(arrays, y, valid, d, shuffled, models[i % 4], original['fold_models'][str(i % 4)])
        # Route sign changes, if any, are a numerical audit failure rather than hidden.
        need(not any(result['direct_vs_saved_routing_sign_differences'].values()), 'Direct ridge route differs near zero')
        check = {k: v for k, v in result.items() if not k.startswith('direct_vs_') and k != 'extrapolation_diagnostics'}
        nested_compare(check, original['views'][i], f'view{i}')
        result.update(name=view['name'], fold=i % 4)
        rows.append(result)
    summary = independent_summary(rows)
    nested_compare(summary, original['summary'], 'summary')
    for view in inp['views']:
        for key in (*KINDS, 'mask', 'valid'):
            need(sha(view[key]['path']) == view[key]['sha256'], 'Prediction/target end hash')
    report = {'status': 'passed', 'script_sha256': sha(__file__), 'plan_sha256': expected_plan,
        'collector_receipt_sha256': sha(run/'execution_receipt.json'), 'collector_launch_sha256': sha(run/'launch_receipt.json'),
        'analysis_receipt_sha256': sha(analysis/'execution_receipt.json'), 'analysis_sha256': sha(analysis/'analysis.json'),
        'analysis_launch_sha256': sha(analysis/'launch_receipt.json'),
        'verified': {'source_files': len(tree), 'bound_inputs': len(plan['input_hashes']), 'probabilities': 64,
                     'TRAIN_views': 16, 'new_model_inferences': 0, 'independent_direct_ridge_fits': 12,
                     'independent_confusion_matrices': 96, 'bootstrap_repeats': 5000, 'seed': 20260926},
        'direct_fit_numerical_differences': fit_diffs, 'views': rows, 'summary': summary,
        'limits': ['Independent CPU direct centered design+Cholesky, no original analyzer function import.',
                  'Fit equivalence comparisons use predeclared atol1e-7/rtol1e-6 for coefficients and atol1e-10/rtol1e-7 for losses; differences are retained.',
                  'Every pixel comparison uses prepared legacy TRAIN known+valid support, not original-grid VAL.',
                  'Base models fit these cameras; cross-fitting applies only to the fixed reader, not to base predictions.',
                  'Ridge/Brier gains and routing confusion are descriptive, not a new deployed system or innovation proof.'],
        'elapsed_seconds': time.monotonic()-start, 'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    with out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'output': str(out), 'sha256': sha(out), 'seconds': report['elapsed_seconds'], 'summary': summary}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    audit(args.run, args.analysis, args.expected_plan_sha256)
