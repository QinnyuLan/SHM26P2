"""Independent saved-record audit; no renderer, producer imports, or pixel decoding."""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import numpy as np

ATOL = 1e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def close(actual, expected, label):
    require(np.isfinite([actual, expected]).all()
            and abs(actual - expected) <= ATOL * max(1., abs(expected)), label)


def hash_map(mapping):
    for path, digest in mapping.items():
        require(sha(path) == digest, f'SHA mismatch: {path}')


def cm_metrics(value):
    cm = np.asarray(value)
    require(cm.shape == (5, 5) and cm.dtype.kind in 'iu' and (cm >= 0).all(), 'Bad CM')
    cm = cm.astype(np.int64)
    intersection = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - intersection
    iou = np.divide(intersection, union, out=np.zeros(5, np.float64), where=union > 0)
    return cm, [float(v) if n else None for v, n in zip(iou, union, strict=True)], (
        float(iou[union > 0].mean()) if (union > 0).any() else None)


def check_metrics(record, cm):
    _, iou, mean = cm_metrics(cm)
    require(record['confusion_matrix'] == np.asarray(cm).tolist(), 'Pooled CM differs')
    for actual, expected in zip(record['iou'], iou, strict=True):
        if expected is None:
            require(actual is None, 'Absent class IoU must be null')
        else:
            close(actual, expected, 'IoU differs')
    if mean is None:
        require(record['miou_all'] is None, 'Empty mIoU must be null')
    else:
        close(record['miou_all'], mean, 'mIoU differs')


def score_summary(full_pass):
    output = {}
    for kind in ('raw', 'scene', 'original_raw', 'original_scene'):
        cms, ces, briers = [], [], []
        for row in full_pass['rows']:
            item = row[kind]
            cm, _, _ = cm_metrics(item['confusion_matrix'])
            require(int(cm.sum()) == item['pixels'] and item['pixels'] > 0, 'CM support differs')
            check_metrics(item, cm)
            require(np.isfinite([item['weighted_ce'], item['brier']]).all()
                    and item['weighted_ce'] >= 0 and item['brier'] >= 0, 'Bad descriptive score')
            cms.append(cm); ces.append(item['weighted_ce']); briers.append(item['brier'])
        total = np.sum(cms, axis=0)
        stored = full_pass['metrics'][kind]
        check_metrics(stored, total)
        close(stored['weighted_ce_view_mean'], float(np.mean(ces)), 'CE view mean differs')
        close(stored['brier_view_mean'], float(np.mean(briers)), 'Brier view mean differs')
        output[kind] = {'confusion_matrix': total.tolist(), 'miou_all': cm_metrics(total)[2],
                        'weighted_ce_view_mean': float(np.mean(ces)),
                        'brier_view_mean': float(np.mean(briers))}
    close(full_pass['original_affine_objective'],
          float(np.mean([r['original_affine_objective'] for r in full_pass['rows']])),
          'Original affine scalar mean differs')
    return output


def check_trace(passes, analysis, names, spec):
    require(2 <= len(passes) <= spec['max_full_passes'], 'Pass budget exceeded')
    require(passes[0]['phase'] == 'baseline' and passes[-1]['phase'] == 'final', 'Endpoint phases')
    sums = {key: 0 for key in ('scene', 'gsplat', 'shader', 'vjp', 'head', 'target_decodes',
                              'complete_passes', 'partial_pass_views')}
    for number, item in enumerate(passes):
        phase = item['phase']
        require(phase in ('baseline', 'gradient', 'trial', 'final'), 'Unknown phase')
        require((phase == 'baseline') == (number == 0)
                and (phase == 'final') == (number == len(passes)-1), 'Repeated endpoint pass')
        require(item['complete'] is True and item['views'] == len(names)
                and [r['name'] for r in item['rows']] == names, 'Incomplete/order-changing pass')
        close(item['objective'], float(np.mean([r['objective'] for r in item['rows']])),
              'Per-view objective mean differs')
        expected = {'scene': len(names), 'gsplat': 2*len(names), 'shader': len(names),
                    'vjp': len(names) if phase != 'trial' else 0,
                    'head': 2*len(names) if phase in ('baseline', 'final') else 0,
                    'target_decodes': 2*len(names) if number == 0 else 0,
                    'complete_passes': 1, 'partial_pass_views': 0}
        require(item['counts'] == expected, 'Per-pass call counts differ')
        for key in sums:
            sums[key] += expected[key]
    require(analysis['baseline'] == passes[0], 'Analysis baseline != complete-pass log')
    require({k: v for k, v in analysis['endpoint'].items() if k != 'gap'} == passes[-1],
            'Analysis endpoint != complete-pass log')
    pointer, accepted, attempt, previous_eta, last_eta = 1, 0, 0, None, None
    current = passes[0]
    accepted_hash = current['q32_sha256']
    accepting = []
    for row in analysis['history']:
        if pointer < len(passes)-1 and passes[pointer]['phase'] == 'gradient':
            require(attempt == 0 and accepted > 0, 'Unexpected gradient pass')
            current = passes[pointer]; pointer += 1
            require(current['q32_sha256'] == accepted_hash, 'Gradient q differs from accepted q')
        require(row['accepted_updates_before'] == accepted and row['trial_index'] == attempt,
                'Trial/accepted update order differs')
        require(0 <= attempt < spec['max_backtracks'] and np.isfinite(row['eta']) and row['eta'] > 0,
                'Trial budget/step invalid')
        if attempt:
            close(row['eta'], previous_eta*.5, 'Backtrack does not halve eta')
        elif last_eta is not None:
            close(row['eta'], last_eta*2, 'New round does not double accepted eta')
        previous_eta = row['eta']
        dot = row['actual_gradient_dot_displacement']
        require(np.isfinite(dot), 'Nonfinite recorded directional derivative')
        if 'trial_skipped' in row:
            require(row['trial_skipped'] == 'nonnegative_actual_direction' and dot >= 0
                    and row['accepted'] is False, 'Skipped trial invalid')
        else:
            require(pointer < len(passes)-1 and passes[pointer]['phase'] == 'trial', 'Missing full trial')
            trial = passes[pointer]; pointer += 1
            close(row['trial_objective'], trial['objective'], 'Trial objective differs')
            bound = current['objective'] + spec['armijo']*dot
            decrease = current['objective'] - trial['objective']
            decision = bool(dot < 0 and decrease > 0 and trial['objective'] <= bound)
            close(row['armijo_bound'], bound, 'Armijo scalar bound differs')
            close(row['full_objective_decrease'], decrease, 'Recorded decrease differs')
            require(row['accepted'] is decision, 'Acceptance differs from full-F scalar Armijo')
            if decision:
                accepted += 1; last_eta = row['eta']; accepted_hash = trial['q32_sha256']
                accepting.append({'number': accepted, 'before': current['objective'],
                                  'after': trial['objective'], 'actual_dot': dot,
                                  'armijo_bound': bound, 'complete_pass': pointer})
        require(row['complete_passes'] == pointer, 'Trial complete-pass counter differs')
        attempt = 0 if row['accepted'] else attempt+1
    # A complete new gradient can discover a small gap without any new trial.
    if pointer < len(passes)-1 and passes[pointer]['phase'] == 'gradient':
        require(attempt == 0 and accepted > 0
                and passes[pointer]['q32_sha256'] == accepted_hash, 'Unexpected trailing gradient')
        pointer += 1
    require(pointer == len(passes)-1 and passes[-1]['q32_sha256'] == accepted_hash,
            'Unaccounted pass or final q differs')
    require(accepted == analysis['accepted_updates'] <= spec['max_accepted']
            and analysis['complete_passes'] == len(passes), 'Global count mismatch')
    require(analysis['convergence_certified'] is False, 'Execution cannot certify convergence')
    reason = analysis['stop_reason']
    require(reason in ('accepted_budget', 'pass_budget', 'zero_tangent_gradient',
                       'approximate_gap_small', 'backtracking_exhausted', 'fp32_stagnation'), 'Unknown stop')
    if reason == 'accepted_budget':
        require(accepted == spec['max_accepted'], 'Incorrect accepted-budget stop')
    if reason == 'pass_budget':
        require(len(passes) >= spec['max_full_passes']-1, 'Incorrect pass-budget stop')
    return sums, accepting


def check_q(master, renderer):
    require(master.dtype == np.float64 and renderer.dtype == np.float32
            and master.ndim == 2 and master.shape[1] == 5 and master.shape == renderer.shape,
            'Bad q shapes/dtypes')
    require(np.isfinite(master).all() and (master >= 0).all() and (master <= 1).all(), 'Bad master q')
    error = float(np.max(np.abs(master.sum(-1)-1)))
    require(error <= 1e-12, 'Master is not simplex within declared FP64 tolerance')
    require(renderer.tobytes() == master.astype(np.float32).tobytes(), 'Renderer q != exact FP32 cast')
    return {'shape': list(master.shape), 'master_row_sum_max_abs_error': error,
            'renderer_row_sum_max_abs_error': float(np.max(np.abs(renderer.astype(np.float64).sum(-1)-1))),
            'renderer_sha256': hashlib.sha256(renderer.tobytes()).hexdigest(),
            'zero_components': int(np.count_nonzero(master == 0))}


def audit(run, expected):
    plan_path = run/'plan.json'
    require(sha(plan_path) == expected, 'Plan SHA differs')
    plan = read(plan_path)
    # Do not inspect metrics/q until the producer has naturally stopped.
    launch, receipt = read(run/'launch_receipt.json'), read(run/'execution_receipt.json')
    require(launch['status'] == 'completed' and launch['natural_completion'] is True
            and launch['exit_code'] == 0 and launch['plan_sha256'] == expected
            and launch['execution_receipt_sha256'] == sha(run/'execution_receipt.json'), 'No bound natural completion')
    require(receipt['status'] == 'completed' and receipt['plan_sha256'] == expected
            and receipt['inputs_sources_unchanged'] is True, 'Producer completion/provenance failed')
    snapshot = Path(plan['source_snapshot'])
    actual_files = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    require(actual_files == plan['source_hashes'], 'Source tree differs')
    bindings = {**plan['input_hashes'], **plan['installed_sources']}
    binary = plan['expected_gsplat_binary']; bindings[binary['path']] = binary['sha256']
    hash_map(bindings)
    require(receipt['actual_gsplat_binary'] == binary, 'Loaded gsplat binary differs')
    for item in receipt['actual_imports'].values():
        path = Path(item['path'])
        require(path.is_relative_to(snapshot)
                and actual_files[str(path.relative_to(snapshot))] == item['sha256'], 'Loaded source mismatch')
    names = [v['name'] for v in plan['training_views']]
    require(len(names) == len(set(names)) == 259 and names == sorted(names)
            and all(v['split'] == 'train' for v in plan['training_views']), 'Wrong TRAIN population')
    manifest = read(plan['manifest'])
    train = [v for v in manifest['views'] if v['split'] == 'train']
    camera_bytes = np.asarray([v['w2c_original'] for v in train], np.float32).tobytes()
    require(len(train) == 350 and [v['name'] for v in train] == plan['all_training_camera_names']
            and hashlib.sha256(camera_bytes).hexdigest() == receipt['training_camera_sha256'],
            'Original TRAIN camera metadata SHA differs')
    population = sorted((v for v in manifest['views'] if v['split'] == 'train' and v.get('mask_path')),
                        key=lambda v: v['name'])
    require([v['name'] for v in population] == names, 'Manifest population differs')
    for actual, original in zip(plan['training_views'], population, strict=True):
        for key in ('K', 'w2c_original', 'width', 'height'):
            require(actual[key] == original[key], f'Camera metadata differs: {key}')
        for key in ('mask_path', 'valid_path'):
            require(Path(actual[key]).resolve() == Path(original[key]).resolve(), 'TRAIN target path differs')
    analysis_path, trace_path = run/'analysis.json', run/'passes.jsonl'
    require(sha(analysis_path) == receipt['analysis_sha256'], 'Analysis SHA differs')
    analysis = read(analysis_path)
    require(analysis['plan_sha256'] == expected
            and analysis['base_checkpoint_sha256'] == plan['specification']['base_sha256']
            and analysis['manifest_sha256'] == plan['specification']['manifest_sha256'], 'Result lineage differs')
    passes = [json.loads(line) for line in trace_path.read_text().splitlines()]
    counts, accepted = check_trace(passes, analysis, names, plan['specification'])
    counts['total_raster'] = counts['gsplat'] + counts['shader']
    require(receipt['counts'] == counts and receipt['accepted_updates'] == analysis['accepted_updates']
            and receipt['stop_reason'] == analysis['stop_reason'], 'Receipt summary differs')
    for key in ('val_views', 'teacher_calls', 'rgb_payload_reads', 'model_optimizer_steps', 'production_checkpoint_writes'):
        require(receipt[key] == 0, f'Nonzero excluded activity: {key}')
    require(receipt['state_unchanged_before_restore'] is True and receipt['numerics_restored'] is True
            and receipt['restoration']['state_exact'] is True
            and receipt['restoration']['flags_modes_gradients_restored'] is True
            and receipt['restoration']['state_before'] == receipt['restoration']['state_after'], 'Restoration attestation differs')
    summaries = {phase: score_summary(passes[index]) for phase, index in (('baseline', 0), ('final', -1))}
    delta_path = Path(analysis['q_delta']['path'])
    require(delta_path.resolve() == (run/'final_q_delta.npz').resolve()
            and sha(delta_path) == analysis['q_delta']['sha256'], 'Endpoint q SHA differs')
    with np.load(delta_path, allow_pickle=False) as arrays:
        require(set(arrays.files) == {'q_master', 'q_renderer'}, 'Unexpected q arrays')
        master, renderer = arrays['q_master'], arrays['q_renderer']
    q_report = check_q(master, renderer)
    require(q_report['renderer_sha256'] == passes[-1]['q32_sha256'], 'Final renderer q differs from final pass')
    close(q_report['renderer_row_sum_max_abs_error'], passes[-1]['q32_max_row_sum_error'], 'Cast error differs')
    artifacts = {str(p): sha(p) for p in (plan_path, run/'launch_receipt.json', run/'execution_receipt.json',
                                         analysis_path, trace_path, delta_path)}
    if accepted:
        saved_path = run/'last_accepted_q_master.npy'; state_path = run/'last_accepted_state.json'
        saved, state = np.load(saved_path, allow_pickle=False), read(state_path)
        require(np.array_equal(saved, master) and state['accepted_updates'] == len(accepted)
                and sha(saved_path) == state['q_master_sha256'], 'Last accepted q differs from endpoint')
        artifacts.update({str(p): sha(p) for p in (saved_path, state_path)})
    hash_map(bindings); hash_map(artifacts)
    hash_map({str(snapshot/name): digest for name, digest in actual_files.items()})
    return {'status': 'passed', 'plan_sha256': expected, 'source_count': len(actual_files),
            'input_count': len(plan['input_hashes']), 'counts': counts, 'accepted': accepted,
            'baseline_objective': passes[0]['objective'], 'final_objective': passes[-1]['objective'],
            'stop_reason': analysis['stop_reason'], 'reported_endpoint_gap': analysis['endpoint']['gap'],
            'convergence_certified': False, 'metrics_from_saved_cms_and_scalars': summaries,
            'final_q': q_report, 'bound_artifacts': artifacts, 'scalar_comparison_atol': ATOL,
            'limits': ['No saved gradients or prediction pixels: W-transpose, F, gap and pixel CM were not recomputed.',
                       'Objective checks independently average saved per-view scalars; Armijo uses recorded gradient-dot-displacement.',
                       'State restoration/counters are source-and-receipt attestations, not independent renderer observations.',
                       'No model, GPU, RGB/mask/valid decoder, or producer math/scorer was imported.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    output = args.run/'independent_cpu_review.json'
    require(not output.exists(), 'Do not overwrite a prior audit')
    start = time.monotonic()
    def expired(*_):
        raise TimeoutError('Independent CPU audit 60-second limit')
    signal.signal(signal.SIGALRM, expired); signal.alarm(60)
    report = {'status': 'failed', 'plan_sha256': args.expected_plan_sha256}
    try:
        report = audit(args.run.resolve(), args.expected_plan_sha256)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-start, checker_path=str(Path(__file__).resolve()),
                      checker_sha256=sha(__file__))
        payload = json.dumps(report, indent=2, allow_nan=False)+'\n'
        with output.open('x') as handle:
            handle.write(payload)


if __name__ == '__main__':
    main()
