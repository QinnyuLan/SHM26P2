"""CPU audit of saved four-endpoint predictions, without GT or model decoding."""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from datetime import datetime
from pathlib import Path

import audit_semantic_partition_matched as common
import numpy as np

ARMS = ('baseline', 'joint', 'pcgrad', 'trust')
KINDS = ('raw', 'scene', 'joint')
NUMERICS = {'cudnn_allow_tf32': True, 'matmul_allow_tf32': False,
            'matmul_precision': 'highest', 'cudnn_benchmark': False}
COUNTS = {'scene_calls': 200, 'direct_q_shader_calls': 200, 'head_calls': 200,
          'teacher_predict_calls': 200, 'teacher_backbone_forwards': 2800,
          'native_gsplat_high_level_calls': 400, 'GT_RGB_payload_reads': 50,
          'annotation_payload_reads': 41, 'RGB_scoring_calls': 200, 'confusion_matrices': 492}
check, read, sha = common.check, common.read, common.sha


def population(records, views, identities):
    names = [v['name'] for v in views]
    check(len(names) == 50 and names == sorted(set(names)), 'Fixed50 cameras')
    indexed = {(r['arm'], r['name']): r for r in records}
    check(len(records) == len(indexed) == 200
          and set(indexed) == {(a, n) for a in ARMS for n in names}, 'Complete200 predictions')
    check(all(set(r['masks']) == set(KINDS) and set(r['soft']) == {'raw', 'scene', 'teacher'}
              and r['teacher_recomputed'] is True for r in records), 'Prediction schema')
    check(len(identities) == 50 and {r['name'] for r in identities} == set(names)
          and all(r['canvas_rgb_exact'] and r['original_rgb_exact'] and r['teacher_soft_exact']
                  and set(r['mask_exact']) == set(KINDS) and all(r['mask_exact'].values())
                  for r in identities), 'Complete baseline identity')
    return indexed


def saved_metrics(value, views, support_reference, audit):
    """Reaggregate saved CM and RGB scores; never derive CM from target pixels."""
    rows = {r['name']: r for r in value['views']}
    support = {r['name']: r for r in support_reference['views']}
    names = [v['name'] for v in views]
    check(len(rows) == len(value['views']) == 50 and set(rows) == set(names), 'Metric population')
    matrices = []
    for view in views:
        row, ref = rows[view['name']], support[view['name']]
        camera = view['camera']; width, height = camera['width'], camera['height']
        check(row['width'] == width and row['height'] == height and row['rgb_pixels'] == width*height,
              'RGB score dimensions')
        check(np.isfinite([row[k] for k in ('psnr', 'ssim', 'lpips')]).all(), 'Finite RGB scores')
        annotated = view['source_annotation_path'] is not None
        check(('confusion_matrix' in row) == annotated, 'Semantic population')
        if annotated:
            cm, old = np.asarray(row['confusion_matrix']), np.asarray(ref['confusion_matrix'])
            check(cm.shape == (5, 5) and cm.dtype.kind in 'iu' and (cm >= 0).all(), 'Valid saved CM')
            check(np.array_equal(cm.sum(1), old.sum(1)), 'Fixed per-class GT support')
            check(int(cm.sum()) == row['semantic_pixels'] == ref['semantic_pixels']
                  and row['semantic_ignore_pixels'] == ref['semantic_ignore_pixels']
                  and row['semantic_pixels']+row['semantic_ignore_pixels'] == width*height, 'CM denominator')
            matrices.append(cm)
    check(len(matrices) == 41, 'Fixed41 semantic views')
    pooled = np.sum(matrices, axis=0); values = common.iou(pooled)
    check(np.array_equal(pooled, value['confusion_matrix']), 'Pooled CM mismatch')
    for key, number in [('iou', values), ('miou_all', np.nanmean(values)),
                        ('miou_foreground', np.nanmean(values[1:]))]:
        audit.close(value[key], number, key)
    for key in ('psnr', 'ssim', 'lpips'):
        audit.close(value[key], np.mean([rows[n][key] for n in names]), key+' mean')
    return {k: value[k] for k in ('miou_all', 'miou_foreground', 'iou', 'psnr', 'ssim', 'lpips')}


def inspect_predictions(run, plan, execution, audit):
    audit.record(execution['predictions_receipt']); prediction = read(execution['predictions_receipt']['path'])
    check(Path(execution['predictions_receipt']['path']) == run/'evaluation/predictions_receipt.json'
          and prediction['status'] == 'all_600_masks_and_200_RGB_before_GT'
          and prediction['annotation_payload_reads'] == prediction['GT_RGB_payload_reads'] == 0, 'GT barrier receipt')
    check(prediction['finished_utc'] == execution['prediction_finished_utc']
          and datetime.fromisoformat(prediction['finished_utc']) <= datetime.fromisoformat(execution['scoring_started_utc']),
          'Prediction-before-scoring chronology')
    for key in ('scene_calls', 'direct_q_shader_calls', 'head_calls', 'teacher_predict_calls',
                'teacher_backbone_forwards', 'native_gsplat_high_level_calls'):
        check(prediction['counts'][key] == COUNTS[key], 'Prediction call evidence')
    views = plan['views']; records = population(prediction['records'], views, prediction['baseline_identity'])
    for index, view in enumerate(views):
        name, camera = view['name'], view['camera']; w, h = camera['width'], camera['height']
        K, cw, ch, grid = common.original_grid(camera); ref = plan['baseline_references'][name]
        check([cw, ch] == ref['boundary']['canvas']
              and np.array_equal(K, np.asarray(ref['boundary']['canvas_K'], np.float32)), 'Original legacy grid')
        for arm in ARMS:
            record = records[arm, name]; directory = run/'evaluation'/arm
            check(Path(record['rgb']['path']) == directory/'rgb'/name
                  and Path(record['teacher_input_rgb']['path']) == directory/'canvas_rgb'/name, 'Own-arm RGB paths')
            audit.png(record['rgb'], (h, w, 3)); audit.png(record['teacher_input_rgb'], (ch, cw, 3))
            soft = {}
            for kind in ('raw', 'scene', 'teacher'):
                binding = record['soft'][kind]
                check(Path(binding['path']) == directory/(kind+'_soft')/(Path(name).stem+'.npy')
                      and binding['shape'] == [ch, cw, 5] and binding['dtype'] == 'float32', 'Own-arm probability path/grid')
                soft[kind] = audit.probabilities(binding, (ch, cw, 5))
            for kind in KINDS:
                binding = record['masks'][kind]
                check(Path(binding['path']) == directory/(kind+'_mask')/name, 'Own-arm mask path')
                mask = audit.png(binding, (h, w)); check((mask < 5).all(), 'Known predicted classes')
                probability = (soft['scene']+soft['teacher'])*np.float32(.5) if kind == 'joint' else soft[kind]
                check(np.array_equal(common.mask_from_soft(probability, grid), mask), 'Independent five-channel soft/mix warp')
            if arm == 'baseline':
                for key, reference in [('rgb', ref['original_rgb']), ('teacher_input_rgb', ref['canvas_rgb'])]:
                    audit.record(reference)
                    check(record[key]['sha256'] == reference['sha256'], 'Baseline H3 RGB byte identity')
                for kind in KINDS:
                    audit.record(ref['masks'][kind])
                    check(record['masks'][kind]['sha256'] == ref['masks'][kind]['sha256'], 'Baseline optimized-q-zero masks')
                old = audit.probabilities(ref['teacher_soft'], (ch, cw, 5))
                check(np.array_equal(soft['teacher'], old), 'Fresh baseline teacher probability identity')
                del old
            del soft, probability, mask
        if index in (0, 24, 49):
            print(json.dumps({'audited_views': index+1, 'mask_reconstructions': (index+1)*12}), flush=True)
    return {'RGB_PNGs': 200, 'canvas_RGB_PNGs': 200, 'soft_arrays': 600,
            'mask_PNGs': 600, 'independent_soft_to_mask_reconstructions': 600,
            'baseline_mask_bytes_exact': 150, 'baseline_H3_RGB_bytes_exact': 50,
            'baseline_canvas_bytes_exact': 50, 'baseline_teacher_probabilities_exact': 50,
            'teacher_and_renderer_calls': 'Saved runtime evidence only; not independently replayed.'}


def inspect_metrics(plan, execution, audit):
    audit.record(plan['reference_E']); reference = read(plan['reference_E']['path'])
    fingerprint_payload = {'evaluation_family': plan['evaluation_family'], 'scoring_protocol': plan['scoring_protocol'],
        'views': [{k: v[k] for k in ('camera', 'source_image_sha256', 'source_annotation_sha256', 'rasterized_mask_sha256')}
                  for v in plan['views']]}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    check(fingerprint == plan['reference_fingerprint'] == reference['official_evaluation_fingerprint'], 'Same official fingerprint')
    check(set(execution['metrics']) == {a+'_'+k for a in ARMS for k in KINDS}, 'Twelve fixed metrics')
    metrics, summary = {}, {}
    for arm in ARMS:
        for kind in KINDS:
            key = arm+'_'+kind; audit.record(execution['metrics'][key]); value = read(execution['metrics'][key]['path'])
            check(value['official_evaluation_fingerprint'] == fingerprint and value['validation_views'] == 50
                  and value['semantic_validation_views'] == 41, 'Metric protocol')
            provenance = value['provenance']
            check(provenance['base_checkpoint_sha256'] == plan['input_hashes'][plan['base_checkpoint']]
                  and provenance['q'] == plan['q_delta']
                  and provenance['means_delta'] == (None if arm == 'baseline' else plan['arms'][arm]['means_delta'])
                  and provenance['teacher'] == (plan['teacher'] if kind == 'joint' else None), 'Metric endpoint provenance')
            metrics[arm, kind] = value; summary[key] = saved_metrics(value, plan['views'], reference, audit)
            if kind != 'raw':
                for row, raw in zip(value['views'], metrics[arm, 'raw']['views'], strict=True):
                    check(all(row[k] == raw[k] for k in ('name', 'width', 'height', 'rgb_pixels', 'psnr', 'ssim', 'lpips')),
                          'Same own-arm RGB across readouts')
            if arm == 'baseline':
                audit.record(plan['baseline_metrics'][kind]); old = read(plan['baseline_metrics'][kind]['path'])
                check(value['confusion_matrix'] == old['confusion_matrix'], 'Baseline pooled CM exact')
                old_rows = {r['name']: r for r in old['views']}
                check(all(row.get('confusion_matrix') == old_rows[row['name']].get('confusion_matrix') for row in value['views']),
                      'Baseline per-view CM exact')
    pairs = {}; expected_pairs = {f'paired_{a}_minus_baseline_{k}' for a in ARMS[1:] for k in KINDS}
    expected_pairs |= {f'paired_{a}_minus_E_joint' for a in ARMS[1:]}
    check(set(execution['comparisons']) == expected_pairs, 'Twelve fixed paired comparisons')
    for arm in ARMS[1:]:
        for kind, ref_arm in [(k, 'baseline') for k in KINDS]+[('joint', 'E')]:
            key = f'paired_{arm}_minus_{ref_arm}_{kind}'; binding = execution['comparisons'][key]
            audit.record(binding); saved = read(binding['path'])
            independent = common.independent_bootstrap(reference if ref_arm == 'E' else metrics['baseline', kind], metrics[arm, kind])
            check(saved['candidate_arm'] == arm and saved['reference_arm'] == ref_arm and saved['readout'] == kind
                  and saved['descriptive_only'] is True and saved['bootstrap_repeats'] == 5000 and saved['seed'] == 20260926
                  and saved['views'] == [v['name'] for v in plan['views']]
                  and saved['official_evaluation_fingerprint'] == fingerprint, 'Pair direction/population')
            audit.close(independent, saved['metrics'], key); pairs[key] = independent
    return {'metrics': summary, 'pairs': pairs, 'saved_perview_CM_reaggregated': 492,
            'independent_bootstrap_comparisons': 12, 'adoption_gate': None}


def run_audit(run, expected, seconds=570):
    output = run/'independent_cpu_review.json'; check(not output.exists(), 'No audit overwrite')
    audit = common.Audit(); result = {'status': 'failed'}; start = time.monotonic()
    def expired(*_):
        raise TimeoutError('Independent CPU audit deadline')
    signal.signal(signal.SIGALRM, expired); signal.alarm(seconds)
    try:
        check(not common.torch.cuda.is_initialized(), 'No CUDA'); common.torch.set_num_threads(4); common.cv2.setNumThreads(4)
        check(sha(run/'plan.json') == expected, 'Wrong frozen plan')
        plan, execution, launch = (read(run/n) for n in ('plan.json', 'execution_receipt.json', 'launch_receipt.json'))
        check(launch['status'] == execution['status'] == 'completed' and launch['natural_completion'] is True
              and launch['exit_code'] == 0 and launch['plan_sha256'] == execution['plan_sha256'] == expected
              and launch['execution_receipt_sha256'] == sha(run/'execution_receipt.json'), 'Natural completion required')
        check(plan['specification']['protocol'] == 'continuous_geometry_proposal_evaluation_v1'
              and plan['specification']['arms'] == list(ARMS) and plan['specification']['adoption_gate'] is None
              and execution['adoption_gate'] is None and execution['optimizer_steps'] == execution['head_adaptation_steps'] == 0,
              'Fixed descriptive evaluation only')
        check(execution['numerics_actual'] == NUMERICS and execution['numerics_restored'] is True
              and execution['sources_inputs_unchanged'] is True, 'Numerics/bound inputs')
        check(all(execution['counts'][k] == n for k, n in COUNTS.items())
              and execution['counts']['native_rasterize_to_pixels_calls'] > 0, 'Fixed reported work')
        check(set(execution['restoration']) == set(ARMS)
              and all(r['state_exact'] and r['flags_modes_gradients_restored'] for r in execution['restoration'].values()),
              'Runtime state restoration evidence')
        snapshot = Path(plan['source_snapshot'])
        check(len(plan['source_hashes']) == 67 and len(plan['input_hashes']) == 331
              and common.tree(snapshot) == plan['source_hashes'], 'Frozen67 sources/331 inputs')
        check(all(plan['source_hashes'][k] == v for k, v in plan['inherited_source_hashes'].items()), 'Recovery package unchanged')
        for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json'):
            audit.bind(run/name)
        for path, digest in {**{str(snapshot/k): v for k, v in plan['source_hashes'].items()},
                             **plan['input_hashes'], **plan['installed_sources']}.items():
            audit.bind(path, digest)
        audit.record(plan['expected_gsplat_binary'])
        for record in execution['actual_imports'].values():
            audit.record(record); path = Path(record['path'])
            check(path.is_relative_to(snapshot) and plan['source_hashes'][str(path.relative_to(snapshot))] == record['sha256'], 'Actual frozen import')
        for record in execution['actual_gsplat_binary'].values():
            audit.record(record)
        check(plan['expected_gsplat_binary'] in execution['actual_gsplat_binary'].values(), 'Actual binary identity')
        audit.record(plan['training_plan']); audit.record(plan['training_audit'])
        review = read(plan['training_audit']['path'])
        check(review['status'] == 'passed' and review['plan_sha256'] == plan['training_plan']['sha256'], 'Completed TRAIN audit binding')
        result['prediction_checks'] = inspect_predictions(run, plan, execution, audit)
        result['statistics'] = inspect_metrics(plan, execution, audit)
        audit.final_hash_check(); check(not common.torch.cuda.is_initialized(), 'No CUDA after audit')
        result.update(status='passed', plan_sha256=expected, execution_receipt_sha256=sha(run/'execution_receipt.json'),
            source_count=67, input_count=331, unique_bound_artifacts=len(audit.bound), max_numeric_error=audit.max_error,
            scalar_tolerance_absolute=1e-12, masks_tolerance='exact integer equality', CUDA_initialized=False,
            producer_runtime_counts=execution['counts'], producer_elapsed_seconds=execution['elapsed_seconds'],
            producer_peak_cuda_allocated_bytes=execution['peak_cuda_allocated_bytes'],
            scope=['Independently reconstructed all saved probability warps/mixtures and baseline byte identities.',
                   'Reaggregated saved per-view CM and RGB scores; independently resampled paired bootstrap.',
                   'No GPU execution, checkpoint deserialization, GT/image-target decoding or new prediction.',
                   'Bound file bytes are read for hashing; only saved predicted RGB/masks/probabilities are decoded.',
                   'CM cannot be independently regenerated from GT here; PSNR/SSIM/LPIPS values are not independently rescored.',
                   'Runtime renderer/teacher/backbone/scorer counts and restoration are bound attestations, not re-executed evidence.',
                   'No adoption gate exists for this fixed descriptive evaluation.'])
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'; raise
    finally:
        signal.alarm(0); result.update(elapsed_seconds=time.monotonic()-start, checker_sha256=sha(__file__),
                                      reused_independent_helper_sha256=sha(common.__file__))
        with output.open('x') as handle:
            json.dump(result, handle, indent=2, allow_nan=False); handle.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run', type=Path)
    parser.add_argument('--expected-plan-sha256', required=True); args = parser.parse_args()
    run_audit(args.run.resolve(), args.expected_plan_sha256)
