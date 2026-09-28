"""Independent saved-record audit of completed EM/FW; no renderer or target decoder."""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

# Previously independently tested I/O, simplex and CM arithmetic, not producer code.
import audit_raw_simplex_result as basic
import numpy as np

require, close, read, sha = basic.require, basic.close, basic.read, basic.sha


def direction_gate(line, spec):
    gate = line['direction_gate']
    a, b64, b32 = line['A'], line['cache_at_zero']['B64'], line['cache_at_zero']['B32upstream']
    require(np.isfinite([a, b64, b32]).all(), 'Nonfinite saved derivatives')
    passed = min(abs(a), abs(b64), abs(b32)) > spec['signal_multiplier']*spec['absolute_tolerance']
    passed &= max(a, b64, b32) < 0
    for name, x, y in [('A_vs_B32upstream', a, b32), ('B32upstream_vs_B64', b32, b64)]:
        tolerance = spec['absolute_tolerance']+spec['relative_tolerance']*max(abs(x), abs(y))
        item = gate['comparisons'][name]
        close(item['left'], x, 'Direction left'); close(item['right'], y, 'Direction right')
        close(item['absolute_error'], abs(x-y), 'Direction error')
        close(item['tolerance'], tolerance, 'Direction tolerance')
        require(item['passed'] is bool(abs(x-y) <= tolerance), 'Direction comparison decision')
        passed &= abs(x-y) <= tolerance
    require(gate['passed'] is bool(passed) and line['passed'] is bool(passed), 'Direction gate decision')
    return bool(passed)


def check_control(passes, analysis, spec):
    """Replay decisions using only saved complete objectives and identities."""
    require(2 <= len(passes) <= spec['max_full_passes'], 'Pass budget')
    require(passes[0]['phase'] == 'baseline' and passes[-1]['phase'] == 'final', 'Endpoint phases')
    current = passes[0]
    pointer, hindex, accepted = 1, 0, {'em': 0, 'fw': 0}
    history, accepted_rows = analysis['history'], []
    expected_stop = 'fixed_two_block_schedule_finished'

    def consume(phase):
        nonlocal pointer
        require(pointer < len(passes)-1 and passes[pointer]['phase'] == phase, 'Unexpected/missing full pass')
        item = passes[pointer]; pointer += 1
        return item

    def event(kind, block, slot=None):
        nonlocal hindex
        require(hindex < len(history), 'Missing decision')
        row = history[hindex]; hindex += 1
        require(row['kind'] == kind and row['block'] == block
                and (slot is None or row['slot'] == slot), 'Decision order')
        close(row['base_objective'], current['objective'], 'Current complete objective changed')
        require(row['gap_at_current'] == current['gap'], 'Current gradient/gap association')
        return row

    def commit(row, trial):
        nonlocal current
        require(trial['objective'] < current['objective'], 'Accepted objective did not decrease')
        accepted_rows.append({'kind': row['kind'], 'block': row['block'], 'before': current['objective'],
                              'after': trial['objective'], 'q32_sha256': trial['q32_sha256']})
        accepted[row['kind']] += 1
        current = trial

    for block in range(spec['em_blocks']):
        for slot in range(spec['em_slots_per_block']):
            row = event('em', block, slot)
            if row['reason'] == 'fp32_stagnation_skip_remaining_em_slots':
                require(row['accepted'] is False and row['actual_direction_l2'] == 0, 'Bad EM stagnation')
                break
            trial = consume(f'em_{block}_{slot}')
            close(row['trial_objective'], trial['objective'], 'EM trial F')
            require(row['gap_at_trial'] == trial['gap'], 'EM trial gradient association')
            decrease = trial['objective'] < current['objective']
            require(row['accepted'] is bool(decrease), 'EM acceptance')
            require(row['reason'] == ('actual_full_objective_decreased' if decrease
                    else 'nondecrease_skip_remaining_em_slots'), 'EM reason')
            if decrease:
                commit(row, trial)
            else:
                break
        consume(f'fw_vertex_{block}')
        row = event('fw', block)
        line = row['line']
        passed = direction_gate(line, spec)
        if not passed:
            require(row['accepted'] is False and row['reason'] == 'fixed_direction_gate_not_met', 'FW stop reason')
        elif row['reason'] == 'fp32_stagnation':
            require(row['accepted'] is False, 'FW stagnation accepted')
        else:
            trial = consume(f'fw_candidate_{block}')
            close(row['trial_objective'], trial['objective'], 'FW trial F')
            require(row['gap_at_trial'] == trial['gap'], 'FW trial gradient association')
            predicted, actual = current['objective']-line['cached_objective'], current['objective']-trial['objective']
            tolerance = spec['absolute_tolerance']+spec['relative_tolerance']*max(abs(predicted), abs(actual))
            decision = predicted > 0 and actual > 0 and abs(predicted-actual) <= tolerance
            close(row['predicted_decrease'], predicted, 'FW predicted decrease')
            close(row['actual_decrease'], actual, 'FW actual decrease')
            close(row['agreement_tolerance'], tolerance, 'FW agreement tolerance')
            require(row['accepted'] is bool(decision) and row['reason'] == ('actual_decrease_confirmed'
                    if decision else 'actual_candidate_not_confirmed'), 'FW acceptance')
            if decision:
                commit(row, trial)
        if not row['accepted']:
            expected_stop = 'fw_stopped_'+row['reason']; break
        if max(current['gap']['directional'], current['gap']['centered']) <= spec['gap_tolerance']:
            expected_stop = 'approximate_gap_small_after_accepted_fw'; break
    require(hindex == len(history) and pointer == len(passes)-1, 'Extra decisions/passes after stop')
    require(passes[-1]['q32_sha256'] == current['q32_sha256'], 'Final not last accepted q')
    close(analysis['final_repeat_objective_difference'], passes[-1]['objective']-current['objective'], 'Final repeat F')
    require(analysis['stop_reason'] == expected_stop and analysis['accepted_updates'] == accepted,
            'Stop/update count mismatch')
    require(analysis['complete_passes'] == len(passes) and analysis['convergence_certified'] is False
            and analysis['performance_adoption'] is False, 'Completion is not convergence/adoption')
    return accepted_rows


def check_passes(log, analysis, names):
    passes = analysis['passes']
    require(len(log) == len(passes), 'Pass log length')
    sums = {k: 0 for k in ('scene', 'gsplat', 'shader', 'vjp', 'head', 'target_decodes',
                           'complete_passes', 'partial_pass_views')}
    for i, (saved, item) in enumerate(zip(log, passes, strict=True)):
        require({k: v for k, v in saved.items() if k != 'target_cache'} ==
                {k: v for k, v in item.items() if k != 'gap'}, 'Analysis vs durable pass log')
        phase = item['phase']; score = phase in ('baseline', 'final')
        vertex = phase.startswith('fw_vertex_')
        require(item['complete'] is True and item['views'] == len(names)
                and [r['name'] for r in item['rows']] == names, 'Incomplete/name-changing pass')
        require(all(np.isfinite(r['objective']) and r['objective'] >= 0 for r in item['rows']), 'Bad per-view F')
        close(item['objective'], float(np.mean([r['objective'] for r in item['rows']])), 'Complete F mean')
        expected = {'scene': len(names), 'gsplat': 2*len(names), 'shader': len(names),
                    'vjp': 0 if vertex else len(names), 'head': 2*len(names) if score else 0,
                    'target_decodes': 2*len(names) if i == 0 else 0,
                    'complete_passes': 1, 'partial_pass_views': 0}
        require(item['counts'] == expected and ('metrics' in item) == score
                and ('gap' in item) != vertex, 'Per-pass counts/scoring scope')
        if phase != 'final':
            require(saved['target_cache']['q32_sha256'] == item['q32_sha256'], 'Cache q identity')
        for k in sums:
            sums[k] += expected[k]
    require(analysis['baseline'] == passes[0] and analysis['endpoint'] == passes[-1], 'Endpoint aliases')
    return sums


def audit(run, expected):
    require(sha(run/'plan.json') == expected, 'Wrong plan')
    plan, launch, receipt = (read(run/name) for name in ('plan.json', 'launch_receipt.json', 'execution_receipt.json'))
    require(launch['status'] == 'completed' and launch['natural_completion'] is True and launch['exit_code'] == 0
            and launch['plan_sha256'] == expected
            and launch['execution_receipt_sha256'] == sha(run/'execution_receipt.json'), 'No bound natural completion')
    require(receipt['status'] == 'completed' and receipt['plan_sha256'] == expected
            and receipt['inputs_sources_unchanged'] is True, 'Producer incomplete')
    spec = plan['specification']
    require(spec['protocol'] == 'raw_simplex_em_fw_v1' and spec['em_blocks'] == 2
            and spec['em_slots_per_block'] == 10 and spec['max_full_passes'] == 26, 'Wrong fixed budget')
    snapshot = Path(plan['source_snapshot'])
    sources = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*')
               if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    require(sources == plan['source_hashes'], 'Snapshot changed')
    require(all(sources[k] == v for k, v in plan['inherited_source_hashes'].items()), 'Inherited source changed')
    bound = {**plan['input_hashes'], **plan['installed_sources']}
    binary = plan['expected_gsplat_binary']; bound[binary['path']] = binary['sha256']
    basic.hash_map(bound)
    require(receipt['actual_gsplat_binary'] == binary, 'Loaded binary changed')
    for item in receipt['actual_imports'].values():
        p = Path(item['path'])
        require(p.is_relative_to(snapshot) and sources[str(p.relative_to(snapshot))] == item['sha256'], 'Actual source changed')
    require(sha(run/'analysis.json') == receipt['analysis_sha256'], 'Analysis hash')
    analysis = read(run/'analysis.json')
    log = [json.loads(line) for line in (run/'passes.jsonl').read_text().splitlines()]
    names = [v['name'] for v in plan['training_views']]
    require(len(names) == len(set(names)) == 259 and names == sorted(names)
            and all(v['split'] == 'train' for v in plan['training_views']), 'TRAIN scope')
    counts = check_passes(log, analysis, names)
    accepts = check_control(analysis['passes'], analysis, spec)
    require(receipt['counts'] == {**counts, 'total_raster': counts['gsplat']+counts['shader']}, 'Total counts')
    require(receipt['accepted_updates'] == analysis['accepted_updates']
            and receipt['stop_reason'] == analysis['stop_reason'], 'Receipt summary mismatch')
    scores = {phase: basic.score_summary(analysis[key]) for phase, key in [('baseline', 'baseline'), ('final', 'endpoint')]}
    initial = np.load(plan['endpoint'], allow_pickle=False)
    initial_check = basic.check_q(initial, initial.astype(np.float32))
    require(initial_check['renderer_sha256'] == analysis['baseline']['q32_sha256'], 'Starting candidate identity')
    initial_difference = analysis['baseline']['objective']-plan['endpoint_expected_objective']
    require(abs(initial_difference) <= spec['absolute_tolerance'], 'Initial candidate F reproduction gate')
    delta = analysis['q_delta']; require(sha(delta['path']) == delta['sha256'], 'Final delta hash')
    with np.load(delta['path'], allow_pickle=False) as arrays:
        require(set(arrays.files) == {'q_master', 'q_renderer'}, 'Delta keys')
        q = basic.check_q(arrays['q_master'], arrays['q_renderer'])
        master = arrays['q_master'].copy()
    require(q['shape'] == initial_check['shape']
            and q['renderer_sha256'] == analysis['endpoint']['q32_sha256'], 'Final q identity')
    if accepts:
        state = read(run/'last_accepted_state.json'); path = run/'last_accepted_q_master.npy'
        require(sha(path) == state['q_master_sha256'] and np.load(path, allow_pickle=False).tobytes() == master.tobytes(),
                'Last accepted master differs from final')
        require(state['ordinary_resume_supported'] is False and state['production_checkpoint'] is False, 'Q-only format')
    require(analysis['production_checkpoint'] is False and analysis['ordinary_resume_supported'] is False,
            'Not a production/resumable checkpoint')
    for key in ('val_views', 'teacher_calls', 'model_updates', 'production_checkpoint_writes'):
        require(receipt[key] == 0, 'Excluded activity')
    require(receipt['state_unchanged_before_restore'] and receipt['restoration']['state_exact']
            and receipt['restoration']['flags_modes_gradients_restored'] and receipt['numerics_restored'], 'Restoration evidence')
    # The released pixel caches cannot be recomputed. Audit only their ownership/inventory attestations.
    cache_records = {p['target_cache']['folder']: p['target_cache'] for p in log if 'target_cache' in p}
    require(len(receipt['cache_releases']) == len(cache_records), 'Cache release count')
    for item in receipt['cache_releases']:
        original = cache_records.pop(item['folder'])
        require(item['hashes'] == original['hashes'] and item['q32_sha256'] == original['q32_sha256']
                and item['deleted_files'] == len(original['hashes']) == 518
                and Path(item['folder']).parent == run/'temporary_target_cache', 'Cache ownership/release record')
    require(not cache_records and not list((run/'temporary_target_cache').iterdir()), 'Unexpected remaining cache')
    basic.hash_map(bound); basic.hash_map({str(snapshot/k): v for k, v in sources.items()})
    return {'status': 'passed', 'plan_sha256': expected, 'execution_receipt_sha256': sha(run/'execution_receipt.json'),
            'analysis_sha256': receipt['analysis_sha256'], 'source_count': len(sources), 'input_count': len(plan['input_hashes']),
            'counts': counts, 'accepted_updates': analysis['accepted_updates'], 'acceptances': accepts,
            'stop_reason': analysis['stop_reason'], 'convergence_certified': False, 'q': q, 'scores': scores,
            'initial_F': analysis['baseline']['objective'], 'final_F': analysis['endpoint']['objective'],
            'initial_F_vs_bound_candidate_difference': initial_difference,
            'F_decrease': analysis['baseline']['objective']-analysis['endpoint']['objective'],
            'final_repeat_F_difference': analysis['final_repeat_objective_difference'],
            'scalar_tolerance': basic.ATOL, 'helper_path': str(Path(basic.__file__).resolve()), 'helper_sha256': sha(basic.__file__),
            'limitations': ['CM and complete objectives are independently aggregated from saved records, not new predictions.',
                            'No saved gradients or released target caches were regenerated: W-transpose, gap, and FW line search are not independently recomputed.',
                            'Restoration and zero-excluded-activity fields are checked attestations, not a replay.',
                            'No GPU execution, checkpoint deserialization, target/image decoding, or new predictions; bound file bytes are read only for hashing.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path); parser.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args(); output = args.run/'independent_cpu_review.json'
    require(not output.exists(), 'Refuse audit overwrite')
    start = time.monotonic(); report = {'status': 'failed', 'plan_sha256': args.expected_plan_sha256}
    def expired(*_):
        raise TimeoutError('Independent audit 60-second limit')
    signal.signal(signal.SIGALRM, expired); signal.alarm(60)
    try:
        report = audit(args.run.resolve(), args.expected_plan_sha256)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'; raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-start, checker_sha256=sha(__file__))
        with output.open('x') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
