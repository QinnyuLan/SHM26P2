"""Independent saved-record audit; no producer imports, rendering or pixel decoding.

Run only after natural completion and the producer's immutable A barrier opened B.
Intermediate VJPs/Adam states and released target arrays are not reconstructible;
this audit checks their recorded transaction identities, not their numeric content.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np

STATES = ['keep', 'action0', 'action1', 'action2', 'action3']
A_NAMES = [f'{x:03d}.png' for x in (2, 41, 79, 118, 156, 200, 241, 278)]
B_NAMES = [f'{x:03d}.png' for x in (21, 59, 100, 137, 176, 220, 259, 300)]
N = 498136
MAX_DIFFERENCE = 0.


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ahash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def same(actual, expected, label='value'):
    global MAX_DIFFERENCE
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label+' keys')
        for key in expected:
            same(actual[key], expected[key], label+'.'+str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label+' length')
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            same(a, b, f'{label}[{i}]')
    elif isinstance(expected, float):
        require(isinstance(actual, (float, int)) and not isinstance(actual, bool), label+' type')
        error = abs(actual-expected)
        require(math.isfinite(actual) and math.isfinite(expected)
                and error <= 1e-12+1e-11*abs(expected), label+' numeric mismatch')
        MAX_DIFFERENCE = max(MAX_DIFFERENCE, error)
    else:
        require(actual == expected and not (isinstance(expected, bool) and type(actual) is not bool), label+' mismatch')


def subset(actual, expected, label):
    for key, value in expected.items():
        same(actual[key], value, label+'.'+key)


def tree(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def finite(value):
    if isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, list):
        for v in value:
            finite(v)
    elif isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite saved scalar')


def rows_ok(rows, names, reference=None):
    require([r['name'] for r in rows] == names, 'Wrong row population/order')
    for i, row in enumerate(rows):
        finite(row)
        cm = np.asarray(row['confusion_matrix'])
        require(cm.shape == (5, 5) and cm.dtype.kind in 'iu' and np.all(cm >= 0), 'Invalid CM')
        require(cm.sum() > 0 and row['raw_ce'] >= 0 and row['full_rgb_mse'] >= 0, 'Empty/invalid objective')
        require(isinstance(row['local_pixels'], int) and 0 <= row['local_pixels'] <= row['rgb_valid_pixels'], 'Local count')
        require((row['local_rgb_mse'] is None) == (row['local_pixels'] == 0), 'Empty-local convention')
        if row['local_pixels']:
            require(row['local_rgb_mse'] >= 0, 'Negative local MSE')
        for key in ('local_support_sha256', 'rgb_valid_sha256'):
            require(isinstance(row[key], str) and len(row[key]) == 64
                    and all(c in '0123456789abcdef' for c in row[key]), 'Missing support SHA')
        if reference is not None:
            for key in ('name', 'local_pixels', 'local_support_sha256', 'rgb_valid_pixels', 'rgb_valid_sha256'):
                same(row[key], reference[i][key], key)
            require(np.array_equal(cm.sum(1), np.asarray(reference[i]['confusion_matrix']).sum(1)), 'GT support changed')


def mean(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return math.fsum(v/len(values) for v in values) if values else None


def summary(rows):
    cm = np.asarray([r['confusion_matrix'] for r in rows], np.int64).sum(0)
    tp = int(cm[2, 2]); fp = int(cm[:, 2].sum())-tp; fn = int(cm[2].sum())-tp
    return {'raw_ce': mean(rows, 'raw_ce'), 'full_rgb_mse': mean(rows, 'full_rgb_mse'),
            'local_rgb_mse': mean(rows, 'local_rgb_mse'), 'local_views': sum(r['local_pixels'] > 0 for r in rows),
            'confusion_matrix': cm.tolist(), 'cable_tp': tp, 'cable_fp': fp, 'cable_fn': fn,
            'cable_iou': tp/(tp+fp+fn) if tp+fp+fn else None,
            'cable_precision': tp/(tp+fp) if tp+fp else None,
            'cable_recall': tp/(tp+fn) if tp+fn else None}


def guard(reference, candidate, keep, repeat):
    aggregate = {}
    for key in ('full_rgb_mse', 'local_rgb_mse'):
        ref, new = mean(reference, key), mean(candidate, key)
        if ref is None:
            aggregate[key] = {'skipped_empty_support': True, 'passed': True}
        else:
            noise = abs(mean(keep, key)-mean(repeat, key))
            tau = max(10*noise, 1e-6*ref, 1e-12)
            aggregate[key] = {'reference': ref, 'candidate': new, 'difference': new-ref,
                              'keep_repeat_difference': noise, 'tau': tau,
                              'skipped_empty_support': False, 'passed': bool(new-ref <= tau)}
    per_view = [{'name': a['name'], 'reference': a['full_rgb_mse'], 'candidate': b['full_rgb_mse'],
                 'limit': 1.001*a['full_rgb_mse'], 'passed': b['full_rgb_mse'] <= 1.001*a['full_rgb_mse']}
                for a, b in zip(reference, candidate, strict=True)]
    return {'aggregate': aggregate, 'per_view': per_view,
            'passed': all(x['passed'] for x in aggregate.values()) and all(x['passed'] for x in per_view)}


def decision_a(states, repeat, scores):
    keep = states['keep']; keys = sorted(k for k in states if k != 'keep')
    summaries = {k: summary(v) for k, v in states.items()}
    guards = {k: guard(keep, states[k], keep, repeat) for k in keys}
    feasible = [k for k in keys if guards[k]['passed']]
    gradient = min(feasible, key=lambda k: (-scores[k], k)) if feasible else 'keep'
    refit = min(feasible, key=lambda k: (summaries[k]['raw_ce'], k)) if feasible else 'keep'
    if summaries[refit]['raw_ce'] >= summaries['keep']['raw_ce']:
        refit = 'keep'
    return {'feasible': feasible, 'gradient_choice': gradient, 'refit_choice': refit,
            'rgb_guards': guards, 'summaries': summaries}


def decision_b(selection, states, repeat):
    g, r = selection['gradient_choice'], selection['refit_choice']
    summaries = {k: summary(v) for k, v in states.items()}
    comparisons = {}
    for role, ref in (('keep', 'keep'), ('gradient', g)):
        new, old = summaries[r], summaries[ref]
        diff = new['cable_iou']-old['cable_iou'] if new['cable_iou'] is not None and old['cable_iou'] is not None else None
        comparisons[role] = {'reference_id': ref, 'candidate_id': r,
            'raw_ce_difference': new['raw_ce']-old['raw_ce'], 'cable_iou_difference': diff,
            'raw_ce_improved': new['raw_ce'] < old['raw_ce'], 'cable_iou_improved': diff is not None and diff > 0,
            'rgb_guard': guard(states[ref], states[r], states['keep'], repeat)}
    passed = g != r and all(c['raw_ce_improved'] and c['cable_iou_improved'] and c['rgb_guard']['passed'] for c in comparisons.values())
    status = 'necessary_signal_present' if passed else 'selection_not_distinguished' if g == r else 'specified_action_not_supported'
    return {'gradient_choice': g, 'refit_choice': r, 'summaries': summaries, 'comparisons': comparisons,
            'choices_differ': g != r, 'passed': bool(passed), 'status': status}


def check_fit(result, logged, names, reference):
    phases = ['baseline']
    for block in range(2):
        phases += [f'em_{block}_{i}' for i in range(10)]+[f'fw_vertex_{block}', f'fw_candidate_{block}']
    phases += ['final']
    passes = result['passes']
    require(len(passes) == result['complete_passes'] == len(logged) == 26, '26 passes required')
    same([p['phase'] for p in passes], phases, 'phase order')
    for p, log in zip(passes, logged, strict=True):
        # approximate_gap is added after the on-disk pass record; its gradient is not saved.
        same({k: v for k, v in p.items() if k != 'approximate_gap'},
             {k: v for k, v in log.items() if k not in ('state', 'target_cache')}, 'pass log')
        require(p['complete'] is True, 'Partial pass')
        rows_ok(p['rows'], names, reference)
        same(p['q_ce'], mean(p['rows'], 'raw_ce'), 'pass CE')
        same(p['rgb_mse'], mean(p['rows'], 'full_rgb_mse'), 'pass RGB')
        same(log['target_cache']['q32_sha256'], p['q32_sha256'], 'cache q identity')
    same(result['baseline'], passes[0], 'baseline alias'); same(result['final'], passes[-1], 'final alias')
    require(len(result['history']) == 22 and result['conditional_optimum_claimed'] is False, 'Fit history/claim')
    q, sh = passes[0], passes[0]
    accepted = {'q_em': 0, 'q_fw': 0, 'sh': 0}; index = 1; history = 0
    for block in range(2):
        for slot in range(10):
            trial = passes[index]; h = result['history'][history]
            qa, sa = trial['q_ce'] < q['q_ce'], trial['rgb_mse'] < sh['rgb_mse']
            same(h, {'kind': 'em_sh', 'block': block, 'slot': slot, 'q_base': q['q_ce'], 'q_trial': trial['q_ce'],
                     'q_accepted': qa, 'rgb_base': sh['rgb_mse'], 'rgb_trial': trial['rgb_mse'], 'sh_accepted': sa,
                     'q_stagnant': trial['q32_sha256'] == q['q32_sha256'],
                     'sh_stagnant': trial['sh_sha256'] == sh['sh_sha256']}, 'EM transaction')
            if qa:
                q = trial; accepted['q_em'] += 1
            if sa:
                sh = trial; accepted['sh'] += 1
            index += 1; history += 1
        vertex, trial = passes[index:index+2]; h = result['history'][history]; line = h['line']
        same(vertex['sh_sha256'], sh['sh_sha256'], 'FW keeps accepted SH')
        same(trial['sh_sha256'], sh['sh_sha256'], 'FW keeps accepted SH')
        if line['passed']:
            z = line['zero']; a, b, c = line['A'], z['B64'], z['B32upstream']
            require(min(abs(a), abs(b), abs(c)) > 2e-6 and max(a, b, c) < 0, 'FW measurable descent')
            for x, y in ((a, c), (c, b)):
                require(abs(x-y) <= 2e-7+5e-4*max(abs(x), abs(y)), 'FW direction agreement')
            require(line['gate']['passed'] is True and 0 <= line['gamma'] <= 1, 'FW gate/gamma')
            same(z['F'], q['q_ce'], 'FW current cache CE')
            predicted, actual = q['q_ce']-line['cached_objective'], q['q_ce']-trial['q_ce']
            tol = 2e-7+5e-4*max(abs(predicted), abs(actual))
            same(h['confirmation'], {'predicted_decrease': predicted, 'actual_decrease': actual, 'tolerance': tol}, 'FW confirmation')
            require(abs(predicted-actual) <= tol, 'FW cached/actual CE')
            qa = actual > 0 and predicted > 0
        else:
            require(line['reason'] == 'no_measurable_negative_LMO' and line['A'] >= -2e-6
                    and h['confirmation'] is None, 'Unexpected FW failure')
            same(vertex['q32_sha256'], q['q32_sha256'], 'No-direction vertex')
            same(trial['q32_sha256'], q['q32_sha256'], 'No-direction trial')
            qa = False
        subset(h, {'kind': 'fw', 'block': block, 'q_accepted': qa,
                   'reason': 'accepted' if qa else 'finite_noop_or_nondecrease'}, 'FW history')
        if qa:
            q = trial; accepted['q_fw'] += 1
        index += 2; history += 1
    same(result['accepted'], accepted, 'Accepted counts')
    same(passes[-1]['q32_sha256'], q['q32_sha256'], 'Final accepted q')
    same(passes[-1]['sh_sha256'], sh['sh_sha256'], 'Final accepted SH')
    same(result['q_final_repeat'], passes[-1]['q_ce']-q['q_ce'], 'Final q repeat')
    same(result['rgb_final_repeat'], passes[-1]['rgb_mse']-sh['rgb_mse'], 'Final RGB repeat')
    return accepted


def audit(run, expected):
    run = Path(run).resolve(); plan = read(run/'plan.json')
    launch, receipt = read(run/'launch_receipt.json'), read(run/'execution_receipt.json')
    require(sha(run/'plan.json') == expected == launch['plan_sha256'] == receipt['plan_sha256'], 'Plan binding')
    require(launch.get('status', 'completed') == receipt['status'] == 'completed'
            and launch['exit_code'] == 0 and launch['natural_completion'] is True, 'Natural completion required')
    require(launch['execution_receipt_sha256'] == sha(run/'execution_receipt.json')
            and receipt['B_unique_inputs_opened'] is True, 'No completed B access barrier')
    require(plan['output'] == str(run) and plan['specification']['protocol'] == 'finite_structure_refit_v1', 'Run/protocol')
    require([v['name'] for v in plan['A']] == A_NAMES and [v['name'] for v in plan['B']] == B_NAMES, 'Fixed split')
    subset(plan['specification'], {'point_count': N, 'K': 64, 'parents': 256, 'actions': 4, 'editable_rows': 2048,
           'em_blocks': 2, 'em_slots_per_block': 10, 'passes_per_state': 26, 'affine_noise': 5e-7,
           'sh_rates': [.0025, .000125], 'sh_eps': 1e-15, 'sh_betas': [.9, .999],
           'internal_seconds': 600, 'external_seconds': 660}, 'Fixed numerical specification')
    snapshot = Path(plan['source_snapshot']); sources = tree(snapshot)
    same(sources, plan['source_hashes'], 'Source tree')
    for key, digest in plan['inherited_source_hashes'].items():
        require(sources[key] == digest, 'Inherited source changed')
    # Only reached after completion+B barrier: no premature B input bytes/hashes.
    inputs = {**plan['input_hashes'], **plan['deferred_B_hashes'], **plan['installed_sources']}
    inputs[plan['expected_gsplat_binary']['path']] = plan['expected_gsplat_binary']['sha256']
    for path, digest in inputs.items():
        require(sha(path) == digest, 'Input changed: '+path)
    require(receipt['sources_inputs_unchanged'] and receipt['numerics_restored'], 'Provenance restoration')
    same(receipt['numerics_actual'], plan['numerics'], 'Numerics')
    same(receipt['actual_gsplat_binary'], plan['expected_gsplat_binary'], 'Loaded gsplat')
    for record in receipt['actual_imports'].values():
        path = Path(record['path']); rel = str(path.relative_to(snapshot))
        require(sources[rel] == record['sha256'], 'Actual imported source')
    restored = receipt['restoration']
    require(restored['state_exact'] and restored['flags_modes_gradients_restored'], 'State restoration')
    same(restored['state_before'], restored['state_after'], 'Restored tensor hashes')
    counts = {'scene': 1144, 'gsplat': 1152, 'shader': 1144, 'q_vjp': 960, 'rgb_vjp': 840,
              'control_vjp': 8, 'target_decodes': 48, 'complete_fit_passes': 130, 'partial_pass_views': 0,
              'total_raster': 2296}
    same(receipt['counts'], counts, 'Call counts')
    for key in ('val_views', 'head_calls', 'teacher_calls', 'production_checkpoint_writes'):
        require(receipt[key] == 0, 'Forbidden work: '+key)
    observed = {str(run/x): sha(run/x) for x in ('plan.json', 'launch_receipt.json', 'execution_receipt.json')}
    def load(name):
        p = run/name; observed[str(p)] = sha(p)
        return read(p)
    analysis, choice, library = load('analysis.json'), load('immutable_A_selection.json'), load('library.json')
    require(observed[str(run/'analysis.json')] == receipt['analysis_sha256'], 'Analysis binding')
    require(receipt['A_selection'] == {'path': str(run/'immutable_A_selection.json'),
            'sha256': observed[str(run/'immutable_A_selection.json')]}, 'A selection binding')
    require(choice['B_unique_inputs_opened'] is False and choice['source_plan_sha256'] == expected, 'Choice barrier record')
    # The producer writes analysis before its finally block derives total_raster.
    same(analysis['counts'], {k: v for k, v in counts.items() if k != 'total_raster'}, 'Analysis counts')
    require(analysis['counts']['gsplat']+analysis['counts']['shader'] == counts['total_raster'], 'Derived raster count')
    require(analysis['conditional_optimum_claimed'] is False and analysis['VAL_or_performance_adoption'] is False, 'Overclaim')
    for path, digest in plan['coverage_lineage'].items():
        require(sha(path) == digest and inputs[path] == digest, 'Coverage chain')
    require(library == read(plan['coverage_library']), 'Exact previous coverage library')
    require(library['status'] == 'ready' and library['eligible_parents'] >= 256, 'Library coverage')
    parents, donors, U = (np.asarray(library[k], np.int64) for k in ('parents', 'donors', 'U'))
    require(len(set(parents)) == 256 and len(set(donors)) == 64 and len(set(U)) == 2048
            and not set(parents) & set(donors) and set(parents) | set(donors) <= set(U)
            and U.min() >= 0 and U.max() < N, 'Action library supports')
    for i in range(4):
        same(library['actions'][f'action{i}'], parents[i::4].tolist(), 'Round-robin parents')
    logs = [json.loads(line) for line in (run/'passes.jsonl').read_text().splitlines()]
    observed[str(run/'passes.jsonl')] = sha(run/'passes.jsonl')
    require(len(logs) == 130 and [p['state'] for p in logs] == [s for s in STATES for _ in range(26)], 'Pass order/count')
    with np.load(plan['q_delta'], allow_pickle=False) as z:
        qbase = z['q_master'].copy()
        require(np.array_equal(z['q_renderer'], qbase.astype(np.float32)), 'Parent q cast')
    require(qbase.dtype == np.float64 and qbase.shape == (N, 5), 'Parent master')
    accepted, fit = {}, {}
    for state in STATES:
        fit[state] = load(state+'_refit.json')
    reference_A = fit['keep']['baseline']['rows']
    for state in STATES:
        result = fit[state]; accepted[state] = check_fit(result, [x for x in logs if x['state'] == state], A_NAMES, reference_A)
        meta = analysis['state_metadata'][state]
        mapping = run/(state+'_mapping.npz'); observed[str(mapping)] = sha(mapping)
        with np.load(mapping, allow_pickle=False) as z:
            source, editable = z['source_indices'], z['editable_indices']
        removed = np.concatenate((np.asarray(library['actions'].get(state, []), np.int64), donors)) if state != 'keep' else []
        source_expected = np.arange(N) if state == 'keep' else np.concatenate((np.setdiff1d(np.arange(N), removed),
                               library['actions'][state], library['actions'][state]))
        require(np.array_equal(source, source_expected) and np.array_equal(editable, np.flatnonzero(np.isin(source, U)))
                and len(editable) == 2048, 'Same-N source/active mapping')
        subset(meta, {'source_indices_sha256': ahash(source), 'editable_indices_sha256': ahash(editable),
                     'q_base_sha256': ahash(qbase[source]), 'point_count': N}, 'State metadata')
        for key, digest in meta['fixed_state_hashes'].items():
            if key in restored['state_before'] and (state == 'keep' or not key.startswith('splats.') and key != 'semantic_prior_counts'):
                require(digest == restored['state_before'][key], 'Unchanged head/background tensor')
        delta = analysis['state_deltas'][state]; path = Path(delta['path'])
        require(path == run/(state+'_delta.npz') and sha(path) == delta['sha256'], 'Delta provenance')
        observed[str(path)] = delta['sha256']
        with np.load(path, allow_pickle=False) as z:
            q, q32, sh = z['q_master'], z['q_renderer'], (z['sh0'], z['sh_rest'])
            require(np.array_equal(z['editable_indices'], editable), 'Delta edit indices')
        require(q.dtype == np.float64 and q.shape == (2048, 5) and np.isfinite(q).all() and (q >= 0).all()
                and np.max(abs(q.sum(1)-1)) <= 1e-10 and np.array_equal(q32, q.astype(np.float32)), 'Simplex/cast')
        require(sh[0].shape == (2048, 1, 3) and sh[1].shape == (2048, 15, 3)
                and all(x.dtype == np.float32 and np.isfinite(x).all() for x in sh), 'SH delta')
        local_sha, sh_hashes = ahash(q32), dict(zip(('sh0', 'sh_rest'), map(ahash, sh), strict=True))
        full = qbase[source].astype(np.float32); full[editable] = q32
        subset(delta, {'q32_sha256': local_sha, 'sh_sha256': sh_hashes}, 'Delta content')
        subset(result['final'], {'q32_sha256': local_sha, 'full_q32_sha256': ahash(full), 'sh_sha256': sh_hashes}, 'Final content')
        for endpoint in ('zero', 'final'):
            row = analysis['B_states'][state][endpoint]
            subset(row, {'state': state, 'phase': 'B_'+endpoint,
                'q32_sha256': ahash(full) if endpoint == 'final' else ahash(qbase[source].astype(np.float32)),
                'sh_sha256': sh_hashes if endpoint == 'final' else result['baseline']['sh_sha256'],
                'delta_sha256': delta['sha256'] if endpoint == 'final' else None}, 'B endpoint identity')
    same(analysis['state_deltas'], choice['state_deltas'], 'Choice locks endpoint hashes')
    same(analysis['A']['selection'], choice['selection'], 'Immutable choice')
    same(analysis['A']['repeat'], choice['keep_repeat'], 'Immutable repeat')
    for repeat, final, delta in ((analysis['A']['repeat'], fit['keep']['final'], analysis['state_deltas']['keep']),
                                  (analysis['B_repeat'], analysis['B_states']['keep']['final'], analysis['state_deltas']['keep'])):
        subset(repeat, {'state': 'keep', 'sh_sha256': final['sh_sha256'],
                       'q32_sha256': final.get('full_q32_sha256', final['q32_sha256']),
                       'delta_sha256': delta['sha256']}, 'Fitted keep repeat identity')
    A = {s: fit[s]['final']['rows'] for s in STATES}
    rows_ok(analysis['A']['repeat']['rows'], A_NAMES, reference_A)
    subset(choice['selection'], {'protocol': 'finite_structure_selection_v1', 'phase': 'A_choices_locked',
                                'candidate_ids': STATES[1:], 'view_names': A_NAMES,
                                'gradient_scores': receipt['gradient_scores']}, 'A selection scope')
    require(all(math.isfinite(v) and v >= 0 for v in receipt['gradient_scores'].values()), 'Gradient selection scores')
    subset(choice['selection'], decision_a(A, analysis['A']['repeat']['rows'], receipt['gradient_scores']), 'Independent A')
    B = {s: analysis['B_states'][s]['final']['rows'] for s in STATES}; refB = B['keep']
    for state in STATES:
        for endpoint in ('zero', 'final'):
            rows_ok(analysis['B_states'][state][endpoint]['rows'], B_NAMES, refB)
    rows_ok(analysis['B_repeat']['rows'], B_NAMES, refB)
    verdict = decision_b(choice['selection'], B, analysis['B_repeat']['rows'])
    subset(analysis['B'], verdict, 'Independent B')
    subset(analysis['B'], {'protocol': 'finite_structure_selection_v1', 'phase': 'B_evaluation_no_reselection'}, 'B scope')
    same(receipt['scientific_status'], verdict['status'], 'Scientific status')
    for row in reference_A+refB:
        path = run/f"support_{row['name']}.npy"; observed[str(path)] = sha(path)
        support = np.load(path, allow_pickle=False)
        require(support.dtype == np.bool_ and ahash(support) == row['local_support_sha256']
                == analysis['support_hashes'][row['name']] and support.sum() >= row['local_pixels'], 'Fixed support bytes')
    releases = {r['folder']: r for r in receipt['cache_releases']}
    require(len(releases) == len(receipt['cache_releases']) == 130, 'Cache release count/uniqueness')
    for log in logs:
        cache = log['target_cache']; folder = cache['folder']
        same(releases[folder], {'folder': folder, 'q32_sha256': log['q32_sha256'], 'files': 16}, 'Cache release identity')
        require(len(cache['hashes']) == 16 and len(cache['records']) == 8 and not Path(folder).exists(), 'Cache lifecycle')
        for row, record in zip(log['rows'], cache['records'], strict=True):
            require(record['name'] == row['name'] and record['pixels'] == np.asarray(row['confusion_matrix']).sum(), 'Cache support count')
    require(not list((run/'temporary').iterdir()), 'Unreleased cache')
    recovery = run/'recovery_preparation.json'
    if recovery.exists():
        observed[str(recovery)] = sha(recovery)
    same(tree(snapshot), sources, 'Source after audit')
    for path, digest in {**inputs, **observed}.items():
        require(sha(path) == digest, 'Bound bytes changed during audit')
    return {'status': 'passed', 'plan_sha256': expected, 'observed_outputs': observed, 'source_count': len(sources),
            'input_count': len(inputs), 'counts': counts, 'accepted': accepted, 'gradient_choice': verdict['gradient_choice'],
            'refit_choice': verdict['refit_choice'], 'B_status': verdict['status'], 'B_passed': verdict['passed'],
            'B_comparisons': verdict['comparisons'], 'max_scalar_absolute_difference': MAX_DIFFERENCE,
            'coverage_library_exact': True, 'limitations': [
                'No render, new label decode, CM recomputation from pixels, or VJP numerical audit.',
                'Deleted target cache bytes and intermediate gradients/Adam moments cannot be reconstructed; checked recorded identities/lifecycle and bound source contracts.',
                'Frozen state restoration and transformed geometry rely on saved runtime hashes; no saved full per-action scene.',
                'A-before-B timing relies on bound producer barrier and immutable choice record, not a fresh operating-system access trace.',
                'TRAIN finite-budget action evidence only; neither conditional optimality nor VAL performance.']}


def self_test():
    def rows(prefix, ce=1., tp=50, rgb=.01):
        cm = np.eye(5, dtype=np.int64)*10; cm[2, 2] = tp; cm[2, 0] = 100-tp; cm[0, 2] = 20
        return [{'name': prefix+str(i), 'raw_ce': ce, 'full_rgb_mse': rgb, 'local_rgb_mse': None,
                 'local_pixels': 0, 'rgb_valid_pixels': 1000, 'local_support_sha256': 'a'*64,
                 'rgb_valid_sha256': 'b'*64, 'confusion_matrix': cm.tolist()} for i in range(2)]
    A = {'keep': rows('A'), 'action0': rows('A', .9), 'action1': rows('A', .8)}
    choice = decision_a(A, A['keep'], {'action0': 2., 'action1': 1.})
    require(choice['gradient_choice'] == 'action0' and choice['refit_choice'] == 'action1', 'Synthetic selection')
    B = {'keep': rows('B'), 'action0': rows('B', .95, 55), 'action1': rows('B', .8, 65)}
    rows_ok(B['action1'], ['B0', 'B1'], B['keep'])
    require(decision_b(choice, B, B['keep'])['passed'], 'Synthetic positive transfer')
    bad = copy.deepcopy(B); bad['action1'][0]['full_rgb_mse'] = .010011
    require(not decision_b(choice, bad, B['keep'])['passed'], 'Strict per-view rejection')
    bad['action1'][0]['confusion_matrix'][2][0] -= 1
    try:
        rows_ok(bad['action1'], ['B0', 'B1'], B['keep'])
    except ValueError:
        pass
    else:
        raise ValueError('Changed GT support accepted')
    require(not guard(B['keep'], B['keep'], B['keep'], B['keep'])['aggregate']['full_rgb_mse']['skipped_empty_support'], 'Full metric required')
    json.dumps(decision_b(choice, B, B['keep']), allow_nan=False)
    # Independent saved-record fixture exercises retained-component identities,
    # strict accept/reject, no-op FW, and immutable pass-log reconciliation.
    passes, history, logged = [], [], []
    def record(phase, ce, mse, qid, sid):
        rr = rows('A', ce=ce, rgb=mse)
        p = {'complete': True, 'phase': phase, 'q_ce': ce, 'rgb_mse': mse, 'rows': rr,
             'q32_sha256': qid, 'sh_sha256': {'sh0': sid, 'sh_rest': sid}}
        passes.append(p)
        logged.append({'state': 'keep', **copy.deepcopy(p), 'target_cache': {'q32_sha256': qid}})
        return p
    q = sh = record('baseline', 1., 1., 'q0', 's0')
    for block in range(2):
        for slot in range(10):
            qa, sa = slot % 2 == 0, slot % 2 == 1
            trial = record(f'em_{block}_{slot}', q['q_ce']+(-.01 if qa else .01),
                           sh['rgb_mse']+(-.01 if sa else .01), f'q{block}{slot}', f's{block}{slot}')
            history.append({'kind': 'em_sh', 'block': block, 'slot': slot,
                'q_base': q['q_ce'], 'q_trial': trial['q_ce'], 'q_accepted': qa,
                'rgb_base': sh['rgb_mse'], 'rgb_trial': trial['rgb_mse'], 'sh_accepted': sa,
                'q_stagnant': trial['q32_sha256'] == q['q32_sha256'],
                'sh_stagnant': trial['sh_sha256'] == sh['sh_sha256']})
            if qa:
                q = trial
            if sa:
                sh = trial
        for label in ('vertex', 'candidate'):
            record(f'fw_{label}_{block}', q['q_ce'], sh['rgb_mse'], q['q32_sha256'], sh['sh_sha256']['sh0'])
        history.append({'kind': 'fw', 'block': block, 'q_accepted': False, 'confirmation': None,
                        'line': {'passed': False, 'reason': 'no_measurable_negative_LMO', 'A': 0.},
                        'reason': 'finite_noop_or_nondecrease'})
    record('final', q['q_ce'], sh['rgb_mse'], q['q32_sha256'], sh['sh_sha256']['sh0'])
    result = {'passes': passes, 'complete_passes': 26, 'baseline': passes[0], 'final': passes[-1],
              'history': history, 'accepted': {'q_em': 10, 'q_fw': 0, 'sh': 10},
              'q_final_repeat': 0., 'rgb_final_repeat': 0., 'conditional_optimum_claimed': False}
    check_fit(result, logged, ['A0', 'A1'], passes[0]['rows'])
    broken = copy.deepcopy(result); broken['history'][1]['q_base'] += .1
    try:
        check_fit(broken, logged, ['A0', 'A1'], passes[0]['rows'])
    except ValueError:
        pass
    else:
        raise ValueError('Uncommitted trial cache accepted as next base')
    print('Independent arithmetic contracts passed (no producer imports/data/GPU).')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path); parser.add_argument('--expected-plan-sha256')
    parser.add_argument('--output', type=Path, help='New report path within run; preserve any earlier failed audit')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    require(args.run is not None and args.expected_plan_sha256, 'Run and expected SHA required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only environment required')
    output = args.output or args.run/'independent_cpu_review.json'
    require(output.resolve().parent == args.run.resolve(), 'Keep audit report in the run directory')
    require(not output.exists(), 'Refuse existing audit output')
    started = time.monotonic()
    with output.with_suffix('.started.json').open('x') as handle:
        json.dump({'auditor_sha256': sha(__file__), 'plan_sha256': args.expected_plan_sha256}, handle)
    report = {'status': 'failed', 'auditor_path': str(Path(__file__).resolve()), 'auditor_sha256': sha(__file__)}
    def expired(*_):
        raise TimeoutError('Independent CPU audit exceeded 180 seconds')
    previous = signal.signal(signal.SIGALRM, expired); signal.alarm(180)
    try:
        report.update(audit(args.run, args.expected_plan_sha256))
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        report['elapsed_seconds'] = time.monotonic()-started
        payload = json.dumps(report, indent=2, allow_nan=False)+'\n'
        with output.open('x') as handle:
            handle.write(payload)


if __name__ == '__main__':
    main()
