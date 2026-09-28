"""Thin CPU audit of two scene-route endpoints; no GT/model decoding or rendering."""
from __future__ import annotations

import argparse
import json
import signal
import time
from datetime import datetime
from pathlib import Path

import audit_continuous_geometry_evaluation as previous
import numpy as np

common = previous.common
check, read, sha = common.check, common.read, common.sha
KINDS = previous.KINDS
COUNTS = {key: value//2 for key, value in previous.COUNTS.items()
          if key not in ('GT_RGB_payload_reads', 'annotation_payload_reads')}
COUNTS.update(GT_RGB_payload_reads=50, annotation_payload_reads=41, native_rasterize_to_pixels_calls=300)
ARMS = ('full', 'prior_only')


def prediction_population(records, names):
    indexed = {(r['arm'], r['name']): r for r in records}
    check(len(names) == 50 and names == sorted(set(names)) and len(records) == len(indexed) == 100
          and set(indexed) == {(a, n) for a in ARMS for n in names}
          and all(r['teacher_recomputed'] is True and set(r['masks']) == set(KINDS)
                  and set(r['soft']) == {'raw', 'scene', 'teacher'} for r in records),
          'Complete100 two-route predictions')
    return indexed


def pair_definitions():
    return ([(a, 'baseline', k) for a in ARMS for k in KINDS]
            + [('prior_only', 'full', k) for k in KINDS]
            + [(a, 'E', 'joint') for a in ARMS])


def predictions(run, plan, execution, audit):
    audit.record(execution['predictions_receipt']); saved = read(execution['predictions_receipt']['path'])
    check(Path(execution['predictions_receipt']['path']) == run/'evaluation/predictions_receipt.json'
          and saved['status'] == 'all_300_masks_and_100_RGB_before_GT'
          and saved['annotation_payload_reads'] == saved['GT_RGB_payload_reads'] == 0
          and saved['reference_evaluation'] == plan['prior_evaluation']
          and saved['prior_baseline_identity'] == plan['prior_baseline_identity'], 'Saved300 barrier/reference identity')
    check(saved['finished_utc'] == execution['prediction_finished_utc']
          and datetime.fromisoformat(saved['finished_utc']) <= datetime.fromisoformat(execution['scoring_started_utc']),
          'Prediction barrier chronology')
    for key in ('scene_calls', 'direct_q_shader_calls', 'head_calls', 'teacher_predict_calls',
                'teacher_backbone_forwards', 'native_gsplat_high_level_calls', 'native_rasterize_to_pixels_calls'):
        check(saved['counts'][key] == COUNTS[key], 'Prediction work attestation')
    records = prediction_population(saved['records'], [v['name'] for v in plan['views']])
    for view in plan['views']:
        name, camera = view['name'], view['camera']; w, h = camera['width'], camera['height']
        K, cw, ch, grid = common.original_grid(camera); boundary = plan['baseline_references'][name]['boundary']
        check([cw, ch] == boundary['canvas'] and np.array_equal(K, np.asarray(boundary['canvas_K'], np.float32)), 'Legacy canvas grid')
        for arm in ARMS:
            directory = run/'evaluation'/arm; record = records[arm, name]
            for key, folder, shape in [('rgb', 'rgb', (h, w, 3)), ('teacher_input_rgb', 'canvas_rgb', (ch, cw, 3))]:
                check(Path(record[key]['path']) == directory/folder/name, 'Own endpoint RGB path')
                audit.png(record[key], shape)
            soft = {}
            for kind in ('raw', 'scene', 'teacher'):
                bound = record['soft'][kind]
                check(Path(bound['path']) == directory/(kind+'_soft')/(Path(name).stem+'.npy')
                      and bound['shape'] == [ch, cw, 5] and bound['dtype'] == 'float32', 'Soft output schema')
                soft[kind] = audit.probabilities(bound, (ch, cw, 5))
            for kind in KINDS:
                bound = record['masks'][kind]
                check(Path(bound['path']) == directory/(kind+'_mask')/name, 'Own mask path')
                mask = audit.png(bound, (h, w)); check((mask < 5).all(), 'Mask class range')
                probability = (soft['scene']+soft['teacher'])*np.float32(.5) if kind == 'joint' else soft[kind]
                check(np.array_equal(common.mask_from_soft(probability, grid), mask), 'Independent probability/mix warp')
            del soft, probability, mask
    return {'new_RGB_PNGs': 100, 'new_canvas_PNGs': 100, 'new_soft_arrays': 300, 'new_mask_PNGs': 300,
            'independent_soft_to_mask_reconstructions': 300, 'baseline_new_renders': 0,
            'baseline_identity': 'Previously audited identity, inherited hash-bound evidence; no new baseline inference.'}


def statistics(plan, execution, audit):
    metrics = {}
    for key, binding in plan['reference_metrics'].items():
        audit.record(binding); arm, kind = key.rsplit('_', 1); metrics[arm, kind] = read(binding['path'])
    check(set(metrics) == {('baseline', k) for k in KINDS} | {('E', 'joint')}, 'Fixed baseline and E references')
    check(all(v['official_evaluation_fingerprint'] == plan['reference_fingerprint'] for v in metrics.values()),
          'Reference original-grid fingerprint')
    reference = metrics['baseline', 'raw']; summary = {}
    check(set(execution['metrics']) == {a+'_'+k for a in ARMS for k in KINDS}, 'Six new fixed readouts')
    for arm in ARMS:
        summary[arm] = {}
        for kind in KINDS:
            binding = execution['metrics'][arm+'_'+kind]; audit.record(binding); value = read(binding['path'])
            check(value['official_evaluation_fingerprint'] == plan['reference_fingerprint'] == reference['official_evaluation_fingerprint']
                  and value['validation_views'] == 50 and value['semantic_validation_views'] == 41, 'Matched population')
            provenance = value['provenance']
            check(provenance['base_checkpoint_sha256'] == plan['input_hashes'][plan['base_checkpoint']]
                  and provenance['q'] == plan['q_delta'] and provenance['means_delta'] == plan['arms'][arm]['means_delta']
                  and provenance['teacher'] == (plan['teacher'] if kind == 'joint' else None), 'New endpoint identity')
            summary[arm][kind] = previous.saved_metrics(value, plan['views'], reference, audit); metrics[arm, kind] = value
            if kind != 'raw':
                for row, raw in zip(value['views'], metrics[arm, 'raw']['views'], strict=True):
                    check(all(row[k] == raw[k] for k in ('name', 'width', 'height', 'rgb_pixels', 'psnr', 'ssim', 'lpips')),
                          'Own RGB scores identical across readouts')
    definitions = pair_definitions()
    check(set(execution['comparisons']) == {f'paired_{a}_minus_{b}_{k}' for a, b, k in definitions}, 'All11 fixed pair directions')
    pairs = {}
    for candidate, reference_arm, kind in definitions:
        key = f'paired_{candidate}_minus_{reference_arm}_{kind}'; bound = execution['comparisons'][key]
        audit.record(bound); saved = read(bound['path'])
        check(saved['candidate_arm'] == candidate and saved['reference_arm'] == reference_arm
              and saved['readout'] == kind and saved['descriptive_only'] is True
              and saved['bootstrap_repeats'] == 5000 and saved['seed'] == 20260926
              and saved['views'] == [v['name'] for v in plan['views']]
              and saved['official_evaluation_fingerprint'] == plan['reference_fingerprint'], 'Paired identity/direction')
        independent = common.independent_bootstrap(metrics[reference_arm, kind], metrics[candidate, kind])
        audit.close(independent, saved['metrics'], key); pairs[key] = independent
    return {'metrics': summary, 'pairs': pairs, 'saved_new_CM_reaggregated': 246,
            'bootstrap_comparisons': 11, 'adoption_gate': None}

def run_audit(run, expected):
    output = run/'independent_cpu_review.json'; check(not output.exists(), 'No audit overwrite')
    start = time.monotonic(); result = {'status': 'failed'}; audit = common.Audit()
    def expired(*_):
        raise TimeoutError('CPU audit deadline')
    signal.signal(signal.SIGALRM, expired); signal.alarm(270)
    try:
        check(not common.torch.cuda.is_initialized(), 'No CUDA'); common.cv2.setNumThreads(4); common.torch.set_num_threads(4)
        check(sha(run/'plan.json') == expected, 'Frozen plan identity')
        plan, execution, launch = (read(run/n) for n in ('plan.json', 'execution_receipt.json', 'launch_receipt.json'))
        check(launch['status'] == execution['status'] == 'completed' and launch['natural_completion'] is True
              and launch['exit_code'] == 0 and launch['plan_sha256'] == execution['plan_sha256'] == expected
              and launch['execution_receipt_sha256'] == sha(run/'execution_receipt.json'), 'Natural completion required')
        check(plan['specification']['protocol'] == 'scene_geometry_routes_evaluation_v1'
              and plan['specification']['arms'] == list(ARMS) and execution['adoption_gate'] is None
              and execution['new_baseline_renders'] == execution['optimizer_steps'] == execution['head_adaptation_steps'] == 0,
              'Fixed additional control only')
        check(execution['numerics_actual'] == previous.NUMERICS and execution['numerics_restored'] is True
              and execution['sources_inputs_unchanged'] is True, 'Numerical/source evidence')
        check(all(execution['counts'][k] == n for k, n in COUNTS.items()), 'Fixed runtime counts')
        check(set(execution['restoration']) == set(ARMS)
              and all(execution['restoration'][a]['state_exact']
                      and execution['restoration'][a]['flags_modes_gradients_restored'] for a in ARMS), 'Runtime restoration evidence')
        snapshot = Path(plan['source_snapshot'])
        check(len(plan['source_hashes']) == 70 and len(plan['input_hashes']) == 1970
              and common.tree(snapshot) == plan['source_hashes']
              and all(plan['source_hashes'][k] == v for k, v in plan['inherited_source_hashes'].items()), 'Inherited frozen source70/input1970')
        for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json'):
            audit.bind(run/name)
        for path, digest in {**{str(snapshot/k): v for k, v in plan['source_hashes'].items()},
                             **plan['input_hashes'], **plan['installed_sources']}.items():
            audit.bind(path, digest)
        for record in execution['actual_imports'].values():
            audit.record(record); path = Path(record['path'])
            check(path.is_relative_to(snapshot) and plan['source_hashes'][str(path.relative_to(snapshot))] == record['sha256'], 'Actual import identity')
        for record in execution['actual_gsplat_binary'].values():
            audit.record(record)
        check(plan['expected_gsplat_binary'] in execution['actual_gsplat_binary'].values(), 'Actual binary identity')
        for name in ('training_plan', 'training_audit', 'prior_evaluation', 'prior_audit'):
            audit.record(plan[name])
        train_review = read(plan['training_audit']['path']); old_review = read(plan['prior_audit']['path'])
        check(train_review['status'] == old_review['status'] == 'passed'
              and train_review['plan_sha256'] == plan['training_plan']['sha256']
              and old_review['execution_receipt_sha256'] == plan['prior_evaluation']['sha256'], 'Completed audit lineages')
        check(set(plan['arms']) == set(ARMS), 'Two trained endpoint identities')
        for arm in ARMS:
            stage_binding = plan['arms'][arm]['receipt']; audit.record(stage_binding)
            stage = read(stage_binding['path']); delta = plan['arms'][arm]['means_delta']; audit.record(delta)
            check(stage['status'] == 'completed' and stage['arm'] == arm
                  and stage['plan_sha256'] == plan['training_plan']['sha256']
                  and stage['attempts'] == stage['accepted'] == 148 and stage['rejected'] == 0
                  and stage['final_means_sha256'] == train_review['arms'][arm]['endpoint_mean_sha256']
                  and stage['means_delta_sha256'] == delta['sha256']
                  and stage['base_means_sha256'] == plan['base_means_sha256']
                  and stage['q_renderer_sha256'] == plan['optimized_q_renderer_sha256']
                  and stage['frozen_state_hashes'] == plan['training_frozen_state_hashes'], 'Own trained endpoint lineage')
        check(execution['prior_evaluation'] == plan['prior_evaluation'] and execution['prior_audit'] == plan['prior_audit']
              and execution['baseline_identity_reused'] == plan['prior_baseline_identity'], 'Reused baseline provenance')
        result['predictions'] = predictions(run, plan, execution, audit)
        result['statistics'] = statistics(plan, execution, audit)
        audit.final_hash_check(); check(not common.torch.cuda.is_initialized(), 'No CUDA after audit')
        result.update(status='passed', plan_sha256=expected, execution_receipt_sha256=sha(run/'execution_receipt.json'),
            source_count=70, input_count=1970, unique_bound_artifacts=len(audit.bound), max_numeric_error=audit.max_error,
            scalar_absolute_tolerance=1e-12, mask_tolerance='exact', CUDA_initialized=False,
            runtime_counts_attested=execution['counts'], producer_elapsed_seconds=execution['elapsed_seconds'],
            scope=['Reconstructed300 masks from saved soft arrays; hashed100 RGB/100 canvas and all bound prior predictions.',
                   'Independent246 saved-CM aggregation, own RGB score means and11 paired5000 bootstrap comparisons.',
                   'No GPU, checkpoint deserialization, GT decoding, rerender or fresh teacher inference.',
                   'Saved CM and PSNR/SSIM/LPIPS values cannot be independently regenerated here.',
                   'Runtime counts/restoration and prior baseline identity remain bound completed-run evidence.',
                   'Bound target/model file bytes may be read for hashing only; no new adoption gate.'])
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'; raise
    finally:
        signal.alarm(0); result.update(elapsed_seconds=time.monotonic()-start, checker_sha256=sha(__file__),
            inherited_auditor_sha256=sha(previous.__file__), shared_independent_helper_sha256=sha(common.__file__))
        with output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run', type=Path)
    parser.add_argument('--expected-plan-sha256', required=True); args = parser.parse_args()
    run_audit(args.run.resolve(), args.expected_plan_sha256)
