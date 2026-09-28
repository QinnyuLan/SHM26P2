#!/usr/bin/env python3
"""Fixed CPU-only camera cross-fit of relative semantic error, never model training.

Reads labels only after a completed collector receipt and all 64 probability files
have been verified. All fitting/scoring is on the prescribed 16 TRAIN cameras.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from PIL import Image

FIXED_NAMES = (
    '002.png', '021.png', '041.png', '059.png', '079.png', '100.png', '118.png',
    '137.png', '156.png', '176.png', '200.png', '220.png', '241.png', '259.png',
    '278.png', '300.png',
)
SPEC = {
    'protocol': 'cross_renderer_authority_analysis_v1',
    'views': list(FIXED_NAMES), 'fold': 'sorted view index modulo 4', 'folds': 4,
    'probability_keys': ['S', 'T1', 'T2', 'Tm'], 'class_count': 5,
    'probability_contract': 'HWC float32; finite, [0,1], sum within 5e-6; statistics normalize FP64',
    'target': 'Brier(Tm,y)-Brier(S,y), summed over 5 classes, positive favors S',
    'features': '25 onehot predicted-class pairs tensor [1,H(S),H(Tm)], 75 columns',
    'additional_feature': '25 onehot predicted-class pairs times JS(T1,T2), 25 columns',
    'log_epsilon': 1e-12, 'readout_dimensions': {'baseline': 75, 'candidate': 100, 'permuted': 100},
    'routing_description': 'predicted z>0 selects S, otherwise Tm; descriptive only, no gate',
    'fit_stride': 4, 'fit_support': 'row/column 0::4, valid>0 and known y; pooled pixels',
    'ridge': 0.001, 'ridge_objective': 'mean squared error + ridge*sum(standardized slopes squared)',
    'intercept_penalized': False,
    'standardization': 'fold TRAIN pooled-pixel population mean/std, std<=1e-12 column set to zero',
    'block_size': 32, 'permutation_seed': 20260926,
    'permutation': 'within-camera actual-shape block groups, independent SeedSequence(seed,index)',
    'edge_policy': 'partial rectangles permuted only among identical shapes; singleton unchanged',
    'evaluation': 'full valid known pixels; per-camera MSE, then equal camera mean',
    'bootstrap_repeats': 5000, 'bootstrap_seed': 20260926,
    'minimum_unique_correct_pixels_each_direction': 100,
    'minimum_unique_correct_views_each_direction': 4,
    'minimum_relative_mse_gain_each_comparison': 0.01,
    'minimum_positive_gain_views_each_comparison': 10,
    'minimum_joint_positive_classes': 2,
    'gate': 'candidate improves baseline and block-permuted control; paired 95% lower>0',
    'scope': 'reader camera cross-fitting on base-model in-fit TRAIN; not model OOF or novel-view proof',
}
COLS = {'baseline': np.arange(75), 'candidate': np.arange(100),
        'permuted': np.r_[np.arange(75), np.arange(100, 125)]}
CASES = ('both_correct', 'S_only_correct', 'Tm_only_correct', 'both_wrong')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def checked_record(record):
    path = Path(record['path'])
    if not path.is_absolute() or not path.is_file() or sha256(path) != record['sha256']:
        raise ValueError(f'Invalid bound file: {path}')
    return path


def validate_prediction_input(value):
    """Never accesses any target path. Complete probability verification first."""
    if value.get('status') != 'predictions_completed' or value.get('protocol') != SPEC['protocol']:
        raise ValueError('Completed collector input required')
    if value.get('specification') != SPEC:
        raise ValueError('Fixed analysis specification differs')
    receipt_path = checked_record(value['prediction_receipt'])
    if json.loads(receipt_path.read_text()).get('status') != 'completed':
        raise ValueError('Prediction execution receipt is not completed')
    launch = json.loads((receipt_path.parent / 'launch_receipt.json').read_text())
    if (launch.get('status') != 'completed' or launch.get('natural_completion') is not True
            or type(launch.get('exit_code')) is not int or launch['exit_code'] != 0
            or launch.get('execution_receipt_sha256') != value['prediction_receipt']['sha256']):
        raise ValueError('Natural collector exit0 receipt is required before targets')
    views = value['views']
    if [view['name'] for view in views] != list(FIXED_NAMES):
        raise ValueError('Expected the fixed sorted 16-camera list')
    for view in views:
        if view.get('split') != 'train':
            raise ValueError('Only TRAIN views are allowed')
        shape = None
        for key in SPEC['probability_keys']:
            record = view[key]
            array = np.load(checked_record(record), mmap_mode='r', allow_pickle=False)
            if array.ndim != 3 or array.shape[-1] != 5 or array.dtype != np.float32:
                raise ValueError('Expected HWC5 FP32 probability file')
            if list(array.shape) != record.get('shape') or record.get('dtype') != 'float32':
                raise ValueError('Probability header differs from collector declaration')
            if shape is not None and array.shape != shape:
                raise ValueError('Per-view probability canvas mismatch')
            shape = array.shape
            for start in range(0, shape[0], 32):
                block = array[start:start + 32]
                if not np.isfinite(block).all() or block.min() < 0 or block.max() > 1:
                    raise ValueError('Probability values violate the fixed contract')
                if np.max(np.abs(block.sum(-1, dtype=np.float64) - 1)) > 5e-6:
                    raise ValueError('Probability sums violate the fixed contract')
    return views


def targets(view, shape):
    """Called only after validate_prediction_input completes for every camera."""
    mask_path, valid_path = checked_record(view['mask']), checked_record(view['valid'])
    with Image.open(mask_path) as image:
        y = np.array(image)
    with Image.open(valid_path) as image:
        valid = np.array(image)
    if y.shape != shape or valid.shape != shape or y.dtype != np.uint8:
        raise ValueError('Legacy target/support canvas mismatch')
    if not np.isin(y, [0, 1, 2, 3, 4, 255]).all():
        raise ValueError('Unknown labels must be encoded as 255')
    if not np.isin(valid, [0, 1, 255]).all():
        raise ValueError('Validity map must be binary')
    return y, (valid > 0) & (y != 255)


def normalize_probabilities(p):
    p = np.asarray(p, dtype=np.float64)
    mass = p.sum(-1, keepdims=True)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(mass <= 0):
        raise ValueError('Finite nonnegative probability with positive mass required')
    return p / mass


def entropy(p):
    p = normalize_probabilities(p)
    return -(p * np.log(np.maximum(p, 1e-12))).sum(-1)


def js_divergence(first, second):
    first, second = normalize_probabilities(first), normalize_probabilities(second)
    mixture = (first + second) * .5
    log_m = np.log(np.maximum(mixture, 1e-12))
    first_term = (first * (np.log(np.maximum(first, 1e-12)) - log_m)).sum(-1)
    second_term = (second * (np.log(np.maximum(second, 1e-12)) - log_m)).sum(-1)
    return np.maximum(0.0, .5 * (first_term + second_term))


def block_permutation(array, view_index, block_size=32, seed=20260926):
    """Shuffle whole equally shaped rectangles, including irregular edge groups."""
    h, w = array.shape
    groups = {}
    for row in range(0, h, block_size):
        for col in range(0, w, block_size):
            shape = (min(block_size, h - row), min(block_size, w - col))
            groups.setdefault(shape, []).append((row, col))
    rng = np.random.default_rng(np.random.SeedSequence([seed, view_index]))
    result = np.empty_like(array)
    records = []
    moved_pixels = 0
    for (bh, bw), positions in sorted(groups.items()):
        order = rng.permutation(len(positions))
        for destination, source_index in zip(positions, order, strict=True):
            source = positions[int(source_index)]
            r, c = destination
            a, b = source
            result[r:r + bh, c:c + bw] = array[a:a + bh, b:b + bw]
            moved_pixels += (source != destination) * bh * bw
        records.append({'shape': [bh, bw], 'block_count': len(positions),
                        'destination_to_source': order.tolist(),
                        'fixed_blocks': int(np.sum(order == np.arange(len(order))))})
    return result, {'groups': records, 'moved_pixels': int(moved_pixels),
                    'total_pixels': h * w, 'moved_fraction': moved_pixels / (h * w)}


def features_and_target(s, tm, d, permuted, labels):
    s, tm = normalize_probabilities(s), normalize_probabilities(tm)
    labels = np.asarray(labels, dtype=np.int64)
    pair = 5 * s.argmax(-1) + tm.argmax(-1)
    rows = np.arange(len(labels))
    x = np.zeros((len(labels), 125), dtype=np.float64)
    x[rows, 3 * pair] = 1
    x[rows, 3 * pair + 1] = entropy(s)
    x[rows, 3 * pair + 2] = entropy(tm)
    x[rows, 75 + pair], x[rows, 100 + pair] = d, permuted
    z = (tm * tm).sum(-1) - (s * s).sum(-1)
    z -= 2 * (tm[rows, labels] - s[rows, labels])
    return x, z


def empty_moments(dimension=125):
    return {'n': 0, 'sx': np.zeros(dimension), 'sxx': np.zeros((dimension, dimension)),
            'sz': 0., 'sxz': np.zeros(dimension), 'szz': 0.}


def add_moments(state, x, z):
    state['n'] += len(z)
    state['sx'] += x.sum(0)
    state['sxx'] += x.T @ x
    state['sz'] += float(z.sum())
    state['sxz'] += x.T @ z
    state['szz'] += float(z @ z)


def sum_moments(states):
    result = empty_moments(len(states[0]['sx']))
    for state in states:
        for key in result:
            result[key] += state[key]
    return result


def fit_ridge(state, columns, ridge=0.001):
    """Exact centered normal equations for mean-MSE ridge, not sum-MSE ridge."""
    if not state['n']:
        raise ValueError('No TRAIN-fold fit pixels')
    cols = np.asarray(columns)
    n = state['n']
    mean, ymean = state['sx'][cols] / n, state['sz'] / n
    covariance = state['sxx'][np.ix_(cols, cols)] / n - np.outer(mean, mean)
    covariance = (covariance + covariance.T) * .5
    std = np.sqrt(np.maximum(0, np.diag(covariance)))
    active = std > 1e-12
    scales = np.where(active, std, 1.)
    normal = covariance / np.outer(scales, scales)
    normal[~active, :] = 0
    normal[:, ~active] = 0
    rhs = (state['sxz'][cols] / n - mean * ymean) / scales
    rhs[~active] = 0
    beta = np.linalg.solve(normal + ridge * np.eye(len(cols)), rhs)
    # Raw-space form used in streaming evaluation. Inactive columns have no effect.
    coef = beta / scales
    intercept = ymean - float(mean @ coef)
    return {'columns': cols.tolist(), 'mean': mean.tolist(), 'std': std.tolist(),
            'active': active.tolist(), 'standardized_coefficient': beta.tolist(),
            'raw_coefficient': coef.tolist(), 'intercept': intercept,
            'n_train_pixels': int(n), 'ridge': ridge,
            'centered_standardized_effective_rank': int(np.linalg.matrix_rank(normal))}


def predict(model, x):
    return x[:, model['columns']] @ np.asarray(model['raw_coefficient']) + model['intercept']


def load_view(view, index):
    arrays = {key: np.load(view[key]['path'], mmap_mode='r', allow_pickle=False)
              for key in SPEC['probability_keys']}
    h, w = arrays['S'].shape[:2]
    d = np.empty((h, w), dtype=np.float64)
    for start in range(0, h, 32):
        d[start:start + 32] = js_divergence(arrays['T1'][start:start + 32],
                                         arrays['T2'][start:start + 32])
    permuted, metadata = block_permutation(d, index)
    y, valid = targets(view, (h, w))
    if not valid.any() or not valid[::4, ::4].any():
        raise ValueError('Empty fixed full or stride4 known-valid support')
    return arrays, d, permuted, y, valid, metadata


def view_fit_moments(arrays, d, permuted, y, valid):
    state = empty_moments()
    for row in range(0, len(y), 32):
        support = valid[row:row + 32:4, ::4]
        x, z = features_and_target(arrays['S'][row:row + 32:4, ::4][support],
                                   arrays['Tm'][row:row + 32:4, ::4][support],
                                   d[row:row + 32:4, ::4][support],
                                   permuted[row:row + 32:4, ::4][support],
                                   y[row:row + 32:4, ::4][support])
        add_moments(state, x, z)
    return state


def evaluate_view(arrays, d, permuted, y, valid, models):
    count = np.zeros(5, dtype=np.int64)
    sse = {key: np.zeros(5) for key in COLS}
    zsum = np.zeros(5)
    cases = np.zeros((5, 4), dtype=np.int64)
    cms = {key: np.zeros((5, 5), dtype=np.int64) for key in ['S', 'Tm', 'half', 'route_baseline', 'route_candidate', 'route_permuted']}
    for row in range(0, len(y), 32):
        support = valid[row:row + 32]
        labels = y[row:row + 32][support].astype(np.int64)
        s, tm = arrays['S'][row:row + 32][support], arrays['Tm'][row:row + 32][support]
        x, z = features_and_target(s, tm, d[row:row + 32][support],
                                   permuted[row:row + 32][support], labels)
        count += np.bincount(labels, minlength=5)
        zsum += np.bincount(labels, weights=z, minlength=5)
        readouts = {key: predict(model, x) for key, model in models.items()}
        for key, predicted_z in readouts.items():
            error = predicted_z - z
            sse[key] += np.bincount(labels, weights=error * error, minlength=5)
        sn, tn = normalize_probabilities(s), normalize_probabilities(tm)
        predictions = {'S': sn.argmax(-1), 'Tm': tn.argmax(-1),
                       'half': (sn + tn).argmax(-1)}
        for key, predicted_z in readouts.items():
            predictions['route_' + key] = np.where(predicted_z > 0,
                                                   predictions['S'], predictions['Tm'])
        ok_s, ok_tm = predictions['S'] == labels, predictions['Tm'] == labels
        case = np.where(ok_s, np.where(ok_tm, 0, 1), np.where(ok_tm, 2, 3))
        cases += np.bincount(4 * labels + case, minlength=20).reshape(5, 4)
        for key, pred in predictions.items():
            cms[key] += np.bincount(5 * labels + pred, minlength=25).reshape(5, 5)
    return {'valid_pixels': int(count.sum()), 'class_pixels': count.tolist(),
            'mse': {key: float(values.sum() / count.sum()) for key, values in sse.items()},
            'class_mse': {key: [float(values[c] / count[c]) if count[c] else None
                               for c in range(5)] for key, values in sse.items()},
            'class_mean_relative_brier': [float(zsum[c] / count[c]) if count[c] else None
                                          for c in range(5)],
            'case_order': list(CASES), 'class_case_counts': cases.tolist(),
            'case_counts': dict(zip(CASES, map(int, cases.sum(0)), strict=True)),
            'confusion_matrices': {key: value.tolist() for key, value in cms.items()}}


def summarize(views, bootstrap_repeats=5000, bootstrap_seed=20260926):
    if len(views) != 16:
        raise ValueError('Fixed 16 view summary required')
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, 16, size=(bootstrap_repeats, 16))
    comparisons = {}
    positive_classes = []
    for reference in ('baseline', 'permuted'):
        base = np.array([view['mse'][reference] for view in views])
        candidate = np.array([view['mse']['candidate'] for view in views])
        gains = base - candidate
        ci = np.quantile(gains[indices].mean(1), [.025, .975])
        relative = float(gains.mean() / base.mean()) if base.mean() > 0 else None
        class_gains = []
        for c in range(5):
            values = [view['class_mse'][reference][c] - view['class_mse']['candidate'][c]
                      for view in views if view['class_mse'][reference][c] is not None]
            class_gains.append({'class_id': c, 'present_views': len(values),
                                'view_equal_mean_gain': float(np.mean(values)) if values else None})
        clauses = {'relative_gain_at_least_1_percent': relative is not None and relative >= .01,
                   'paired_lower_positive': bool(ci[0] > 0),
                   'positive_views_at_least_10': int((gains > 0).sum()) >= 10}
        comparisons[reference] = {'reference_mean_mse': float(base.mean()),
                                  'candidate_mean_mse': float(candidate.mean()),
                                  'mean_mse_gain': float(gains.mean()),
                                  'relative_mse_gain': relative,
                                  'paired_view_bootstrap_95_gain': ci.tolist(),
                                  'positive_views': int((gains > 0).sum()),
                                  'per_view_gain': gains.tolist(), 'class_gains': class_gains,
                                  'clauses': clauses}
    for c in range(5):
        pair = [comparisons[r]['class_gains'][c]['view_equal_mean_gain']
                for r in ('baseline', 'permuted')]
        if all(value is not None and value > 0 for value in pair):
            positive_classes.append(c)
    error_support = {}
    for case in ('S_only_correct', 'Tm_only_correct'):
        counts = [view['case_counts'][case] for view in views]
        error_support[case] = {'pixels': sum(counts), 'views_with_events': sum(x > 0 for x in counts),
                               'enough': sum(counts) >= 100 and sum(x > 0 for x in counts) >= 4}
    enough = all(item['enough'] for item in error_support.values())
    passed = enough and len(positive_classes) >= 2 and all(
        all(item['clauses'].values()) for item in comparisons.values())
    return {'status': ('inconclusive_error_support' if not enough else
                       'necessary_increment_present' if passed else 'necessary_increment_not_met'),
            'resource_gate_passed': bool(passed), 'comparisons': comparisons,
            'error_support': error_support, 'joint_positive_classes': positive_classes,
            'at_least_two_joint_positive_classes': len(positive_classes) >= 2,
            'gain_direction': 'reference reader MSE minus candidate reader MSE; positive improves',
            'bootstrap_repeats': bootstrap_repeats, 'bootstrap_seed': bootstrap_seed,
            'interpretation': SPEC['scope']}


def fit_fold_models(states):
    models = {}
    for fold in range(4):
        state = sum_moments([item for i, item in enumerate(states) if i % 4 != fold])
        models[fold] = {key: fit_ridge(state, cols) for key, cols in COLS.items()}
    return models


def analyze(value):
    views = validate_prediction_input(value)
    states, permutation_records = [], []
    for index, view in enumerate(views):
        arrays, d, permuted, y, valid, metadata = load_view(view, index)
        states.append(view_fit_moments(arrays, d, permuted, y, valid))
        permutation_records.append(metadata)
    models = fit_fold_models(states)
    rows = []
    for index, view in enumerate(views):
        arrays, d, permuted, y, valid, _ = load_view(view, index)
        row = evaluate_view(arrays, d, permuted, y, valid, models[index % 4])
        row.update(name=view['name'], fold=index % 4, fit_grid_pixels=int(states[index]['n']),
                   permutation=permutation_records[index])
        rows.append(row)
    pair_counts = {str(fold): sum_moments([item for i, item in enumerate(states)
                                         if i % 4 != fold])['sx'][np.arange(0, 75, 3)]
                   for fold in range(4)}
    pair_report = {fold: {'counts': counts.astype(int).tolist(),
                          'unseen_pair_ids': np.flatnonzero(counts == 0).tolist(),
                          'rare_pair_ids_count_1_to_99': np.flatnonzero((counts > 0) & (counts < 100)).tolist()}
                   for fold, counts in pair_counts.items()}
    return {'specification': SPEC, 'fold_models': models, 'fold_pair_support': pair_report,
            'views': rows, 'summary': summarize(rows),
            'limitations': [
                'Only the low-dimensional diagnostic readout is cross-fitted. Base models fit TRAIN.',
                'Pixel and neighboring-view errors are correlated; 16 views are not independent bridges.',
                'Different reconstructed fields change geometry, color and training; not pure appearance causality.',
                'MSE predicts relative Brier error, not deployment IoU. No semantic gate or renderer was updated.',
                'A negative result only rejects this fixed pair-conditioned linear readout and resource gate.',
                'Error-support and improvement gates are resource decisions, not novelty or generalization proof.',
            ]}


def run(input_path, expected_input_sha256, output):
    started = time.monotonic()
    path, output = Path(input_path).resolve(), Path(output).resolve()
    if sha256(path) != expected_input_sha256:
        raise ValueError('Analysis input SHA mismatch')
    value = json.loads(path.read_text())
    launch_path = Path(value['prediction_receipt']['path']).parent / 'launch_receipt.json'
    launch_sha = sha256(launch_path)
    if output.exists():
        raise FileExistsError('Analysis output must be a new directory')
    output.mkdir(parents=True)
    receipt = {'status': 'running', 'input': str(path), 'input_sha256': expected_input_sha256,
               'script_sha256': sha256(__file__), 'specification': SPEC, 'gpu_used': False,
               'collector_launch_receipt': {'path': str(launch_path), 'sha256': launch_sha}}
    write_json(output / 'execution_receipt.json', receipt)
    try:
        result = analyze(value)
        # Recheck bytes after scoring. This does not decode additional labels.
        if sha256(path) != expected_input_sha256:
            raise ValueError('Input JSON changed')
        checked_record(value['prediction_receipt'])
        if sha256(launch_path) != launch_sha or sha256(__file__) != receipt['script_sha256']:
            raise ValueError('Collector launch receipt or analysis source changed')
        for view in value['views']:
            for key in (*SPEC['probability_keys'], 'mask', 'valid'):
                checked_record(view[key])
        write_json(output / 'analysis.json', result)
        receipt.update(status='completed', analysis_sha256=sha256(output / 'analysis.json'),
                       input_bytes_unchanged=True)
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic() - started,
                       peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        write_json(output / 'execution_receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--expected-input-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.expected_input_sha256, args.output)


if __name__ == '__main__':
    main()
