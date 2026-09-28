"""Independent cached CPU decomposition replay; reuse only our audited warp code."""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import audit_residual_transport as previous
import numpy as np

ARMS = ('zero', 'source_residual', 'render_only', 'full_photo')
HELPER_SHA = '9d94c5245441f384646a6c74793345f82978f91421a8e23c60aa59d9c3cd5404'
check, read, sha = previous.check, previous.read, previous.sha


def components(base, source_residual, warped, support):
    d = (warped-base.astype(np.float64))*support[..., None]
    return {'render_only': d, 'full_photo': source_residual+d}


def score(base, truth, correction, coefficient, valid):
    prediction = base.astype(np.float64)+coefficient*correction
    result = {}
    for domain, value in [('unclipped', prediction), ('clipped', np.clip(prediction, 0, 1))]:
        error = (value-truth.astype(np.float64))[valid]
        mse = float(np.sum(error*error)/error.size)
        result[domain+'_MSE'] = mse
        result[domain+'_log10_MSE'] = float(np.log10(mse)) if mse > 0 else None
    return result


def expansion(base, truth, s, d, c, valid):
    e = (base.astype(np.float64)-truth.astype(np.float64))[valid]
    s, d = s[valid], d[valid]; scale = e.size
    terms = dict(zip(('linear_source', 'linear_render', 'quadratic_source', 'quadratic_render', 'cross_source_render'),
                     map(float, (2*c*np.sum(e*s)/scale, 2*c*np.sum(e*d)/scale,
                                 c*c*np.sum(s*s)/scale, c*c*np.sum(d*d)/scale, 2*c*c*np.sum(s*d)/scale)), strict=True))
    direct = float((np.sum((e+c*(s+d))**2)-np.sum(e*e))/scale)
    total = sum(terms.values())
    tolerance = 64*np.finfo(np.float64).eps*max(1., abs(direct), sum(abs(v) for v in terms.values()))
    check(abs(total-direct) <= tolerance, 'Independent five-term polynomial identity')
    return {'terms': terms, 'sum': total, 'direct_MSE_change': direct,
            'absolute_error': abs(total-direct), 'tolerance': float(tolerance)}


def summarize(rows):
    summary = {}
    for label in ('all', 'A', 'B'):
        subset = [r for r in rows if label == 'all' or r['fold'] == label]
        summary[label] = {}
        for arm in ARMS:
            result = {'views': len(subset)}
            for domain in ('unclipped', 'clipped'):
                values = [r['metrics'][arm][domain+'_MSE'] for r in subset]
                logs = [r['metrics'][arm][domain+'_log10_MSE'] for r in subset]
                mean_log = float(sum(logs)/len(logs)) if all(x is not None for x in logs) else None
                result.update({domain+'_MSE': float(sum(values)/len(values)),
                               domain+'_mean_log10_MSE': mean_log,
                               domain+'_mean_PSNR_dB': -10*mean_log if mean_log is not None else None,
                               domain+'_zero_MSE_views': sum(v == 0 for v in values)})
            summary[label][arm] = result
    return summary


def run_audit(run, expected):
    output = run/'independent_cpu_review.json'; check(not output.exists(), 'No audit overwrite')
    report = {'status': 'failed'}; start = time.monotonic(); audit = previous.Audit()
    def expired(*_):
        raise TimeoutError('Independent cached CPU replay 90s deadline')
    signal.signal(signal.SIGALRM, expired); signal.alarm(90)
    try:
        check(sha(previous.__file__) == HELPER_SHA, 'Independent warp helper source')
        check(sha(run/'plan.json') == expected, 'Frozen plan')
        plan, execution, launch = (read(run/n) for n in ('plan.json', 'execution_receipt.json', 'launch_receipt.json'))
        check(execution['status'] == launch['status'] == 'completed' and launch['natural_completion'] is True
              and launch['exit_code'] == 0 and launch['plan_sha256'] == execution['plan_sha256'] == expected
              and launch['execution_receipt_sha256'] == sha(run/'execution_receipt.json'), 'Natural completion required')
        check(plan['specification']['protocol'] == 'residual_transport_decomposition_v1'
              and plan['specification']['targets'] == previous.NAMES and plan['specification']['arms'] == list(ARMS), 'Unchanged mathematical protocol')
        check(previous.tree(Path(plan['source_snapshot'])) == plan['source_hashes'] and len(plan['source_hashes']) == 5, 'Frozen5 source files')
        for path, digest in plan['input_hashes'].items():
            audit.bind(path, digest)
        parent = Path(plan['parent']); parent_plan = read(parent/'plan.json'); parent_analysis = read(parent/'analysis.json')
        parent_review = read(parent/'independent_cpu_review.json')
        check(parent_review['status'] == 'passed' and parent_review['plan_sha256'] == sha(parent/'plan.json')
              and parent_review['execution_receipt_sha256'] == sha(parent/'execution_receipt.json'), 'Original independent audit lineage')
        check(plan['camera_selection'] == parent_plan['camera_selection']
              and [r['name'] for r in plan['targets']] == previous.NAMES, 'No new views/source selection')
        check(execution['prediction_barrier_complete'] and execution['source_inputs_unchanged']
              and execution['target_cache_loads'] == 32 and not execution['cuda_used'], 'Saved barrier/count/CPU attestations')
        check(all(execution[k] == 0 for k in ('original_image_decodes', 'scene_calls', 'semantic_GT', 'VAL', 'model_updates')), 'Cache-only scope')
        check(execution['helper_sha256'] == plan['source_hashes']['residual_transport.py'], 'Producer used unchanged parent math')
        audit.bind(str(run/'analysis.json'), execution['analysis_sha256']); saved = read(run/'analysis.json')
        check([r['name'] for r in execution['decompositions']] == previous.NAMES == [r['name'] for r in saved['views']], 'All16 decomposition barrier')
        records = {r['name']: r for r in plan['renders']}; old_records = {r['name']: r for r in plan['transports']}
        targets = {r['name']: r for r in plan['targets']}; replayed = []; rows = []
        previous.cv2.setNumThreads(4)
        for choice, record in zip(plan['camera_selection'], execution['decompositions'], strict=True):
            name = choice['name']; base = audit.arrays(records[name]); sources = []
            for source_record in choice['sources']:
                source = audit.arrays(records[source_record['name']]); source['residual'] = source['rgb']; sources.append(source)
            warped, _ = previous.replay(base, sources); old = audit.arrays(old_records[name]); fresh = audit.arrays(record)
            check(record['support_count_exact'] and record['common_pixels'] == int(old['valid'].sum()), 'Support attestation/count')
            audit.close(warped['valid'], old['valid'], name+'.same_support')
            audit.close(warped['source_count'], old['source_count'], name+'.same_sources')
            recomputed = {**components(base['rgb'], old['true_residual'], warped['true_residual'], old['valid']),
                          'valid': old['valid'], 'source_count': old['source_count']}
            audit.close(recomputed, fresh, name+'.decomposition')
            replayed.append({'name': name, 'fold': choice['fold'], 'base': base, 's': old['true_residual'],
                             'd': fresh['render_only'], 'full': fresh['full_photo']})
        # Match the score barrier: all predictions above verified before saved target RGB below.
        for case in replayed:
            name = case['name']; base = case['base']; truth = audit.arrays(targets[name])
            check(np.array_equal(truth['valid'], base['valid']), 'Full-target valid identity')
            corrections = {'source_residual': case['s'], 'render_only': case['d'], 'full_photo': case['full']}
            stats = {arm: previous.sufficient(base['rgb'], truth['rgb'], correction, truth['valid']) for arm, correction in corrections.items()}
            rows.append({'name': name, 'fold': case['fold'], 'statistics': stats}); case['truth'] = truth['rgb']
        fits = {arm: {fold: previous.fit([r['statistics'][arm] for r in rows if r['fold'] == fold]) for fold in 'AB'} for arm in ARMS[1:]}
        audit.close(fits, saved['fits'], 'All6 fitted scalars')
        audit.close(fits['source_residual'], parent_analysis['fits']['true'], 'Unchanged source fits')
        for row, case, expected_row in zip(rows, replayed, saved['views'], strict=True):
            base, truth = case['base'], case['truth']; other = 'B' if row['fold'] == 'A' else 'A'
            corrections = {'zero': np.zeros_like(case['s']), 'source_residual': case['s'], 'render_only': case['d'], 'full_photo': case['full']}
            row['metrics'] = {}
            for arm, correction in corrections.items():
                c = 0. if arm == 'zero' else fits[arm][other]['coefficient']
                row['metrics'][arm] = {'coefficient': c, 'fit_fold': None if arm == 'zero' else other,
                                       **score(base['rgb'], truth, correction, c, base['valid'])}
            row['full_photo_expansion'] = expansion(base['rgb'], truth, case['s'], case['d'], fits['full_photo'][other]['coefficient'], base['valid'])
            audit.close(row, expected_row, row['name']+'.metrics_and_five_terms')
        summary = summarize(rows); audit.close(summary, saved['equal_view_summary'], 'All/A/B summary')
        old_wrong = saved['old_wrong_reference']
        check(old_wrong['refit'] is False and old_wrong['source'] == str(parent/'analysis.json')
              and old_wrong['sha256'] == sha(parent/'analysis.json'), 'Wrong reference only, no refit')
        audit.close(old_wrong['fits'], parent_analysis['fits']['wrong'], 'Old wrong fits')
        audit.close(old_wrong['equal_view_mean'], parent_analysis['equal_view_mean']['wrong'], 'Old wrong metrics')
        report.update(status='passed', plan_sha256=expected, execution_receipt_sha256=sha(run/'execution_receipt.json'),
                      source_files=5, bound_files=len(audit.hashes), decomposition_replays=16, fits=fits,
                      views=rows, equal_view_summary=summary, max_absolute_error=audit.max_error,
                      compared_entries=int(audit.comparisons), absolute_tolerance=previous.ATOL,
                      producer_target_cache_loads_attested=32, original_image_decodes=0, GPU_execution=False,
                      limitations=['No producer mathematical functions imported; saved source render RGB is transported using our previous independent warp.',
                                   'No new rendering, original image/label decoding, model loading, source selection or fitting population.',
                                   'Runtime target-load count and chronology are saved attestations, not reconstructed from pixels.',
                                   'full=s+d is an algebraic identity on cached FP32 source residuals; not original-photo byte reconstruction.',
                                   'Earlier v1 package-metadata preparation failure stays failed; successful v2 retains the same numerical protocol.',
                                   'Scalar cross-fit on TRAIN is not field OOF or held-out-view generalization; no adoption gate.'])
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        signal.alarm(0); report['elapsed_seconds'] = time.monotonic()-start; report['auditor_sha256'] = sha(__file__)
        with output.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path); parser.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args(); run_audit(args.run, args.expected_plan_sha256)
