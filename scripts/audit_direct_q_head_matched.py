"""Small CPU adapter over the independent matched-field auditor; no GT decoding."""
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
import torch

ARMS = ('original_q', 'optimized_q')
READOUTS = ('original_q_zero', 'optimized_q_zero')+ARMS
KINDS = ('joint', 'scene', 'raw')
THRESHOLDS = {'original_q': .001, 'E': .002}
check, read, sha = common.check, common.read, common.sha
TRAIN_NUMERICS = {'cudnn_allow_tf32': False, 'matmul_allow_tf32': False,
                  'matmul_precision': 'highest', 'cudnn_benchmark': False}
EVAL_NUMERICS = dict(TRAIN_NUMERICS, cudnn_allow_tf32=True)


def recovery_numerics(parent, parent_launch, diagnostic, actual):
    """A failed evaluation stays failed; only its completed training is reused."""
    check(parent['status'] == parent_launch['status'] == 'failed' and parent_launch['exit_code'] == 1,
          'Original failed attempt must remain failed')
    check(parent['new_val_annotation_payload_reads'] == 0 and 'evaluation' not in parent
          and parent['numerics_actual'] == TRAIN_NUMERICS and parent['amp_enabled'] is False
          and parent['numerics_restored'] is True, 'Failed-parent numerical/scope evidence')
    check([r['arm'] for r in parent['training']] == list(ARMS)
          and all(r['status'] == 'completed' and r['steps'] == 2000 for r in parent['training']),
          'Two completed training endpoints required')
    check(diagnostic['status'] == 'completed' and diagnostic['hypothesis_confirmed'] is True
          and diagnostic['scene_calls'] == 3 and diagnostic['direct_q_shader_calls'] == 1
          and diagnostic['target_reads'] == diagnostic['training_updates'] == 0
          and all(diagnostic[k] is True for k in ('state_exact', 'q_exact', 'inputs_unchanged', 'numerics_restored')),
          'Target-free numerical diagnosis required')
    rows = diagnostic['records']
    check([r['label'] for r in rows] == ['native_tf32_false', 'native_tf32_true', 'original_q_tf32_true'],
          'Fixed numerical diagnosis cases')
    check(rows[0]['actual_numerics'] == TRAIN_NUMERICS and rows[0]['E_mask_pixel_differences'] > 0
          and rows[0]['E_mask_bytes_exact'] is False, 'Original mismatch evidence')
    check(all(r['actual_numerics'] == EVAL_NUMERICS and r['E_mask_pixel_differences'] == 0
              and r['E_mask_bytes_exact'] is True for r in rows[1:])
          and rows[1]['false_vs_true_p3d_exact'] is True
          and rows[2]['native_true_soft_exact'] is True and rows[2]['native_true_p3d_exact'] is True,
          'Historical-policy/native-adapter reproduction')
    check(len({r['p3d_sha256'] for r in rows}) == 1
          and rows[1]['joint_mask']['sha256'] == rows[2]['joint_mask']['sha256']
          and rows[1]['soft']['sha256'] == rows[2]['soft']['sha256'], 'Diagnosis output identities')
    check(actual == EVAL_NUMERICS, 'Evaluation must use historical numerical policy')
    return {'parent_status': 'failed', 'training_numerics': TRAIN_NUMERICS,
            'evaluation_numerics': EVAL_NUMERICS, 'new_training_updates': 0,
            'scope': 'Numeric-policy recovery of evaluation only; original failed execution is not reclassified.'}


def recovery_lineage(plan, execution, audit):
    """Resolve unchanged training separately from the new evaluation directory."""
    audit.record(plan['training_plan']); parent_path = Path(plan['training_plan']['path'])
    parent_run = parent_path.parent; parent_plan = read(parent_path)
    parent, launch = (read(parent_run/n) for n in ('execution_receipt.json', 'launch_receipt.json'))
    check(plan['training_plan'] == execution['training_plan'] and plan['training_lineage'] == execution['training_lineage']
          and plan['numerical_recovery'] == execution['numerical_recovery']
          and execution['training_updates'] == 0 and execution['original_training_wrapper_status'] == 'failed'
          and execution['numerics_restored'] is True, 'Evaluation-only receipt lineage')
    check(launch['plan_sha256'] == parent['plan_sha256'] == plan['training_plan']['sha256']
          and launch['execution_receipt_sha256'] == sha(parent_run/'execution_receipt.json'), 'Preserved failure SHA chain')
    audit.record(plan['numerical_recovery']); diagnosis_path = Path(plan['numerical_recovery']['path'])
    diagnostic = read(diagnosis_path); diagnostic_run = diagnosis_path.parent
    diagnostic_plan = read(diagnostic_run/'plan.json'); diagnostic_launch = read(diagnostic_run/'launch_receipt.json')
    check(diagnostic_launch['natural_completion'] is True and diagnostic_launch['exit_code'] == 0
          and diagnostic_launch['execution_receipt_sha256'] == sha(diagnosis_path)
          and diagnostic_launch['plan_sha256'] == diagnostic['plan_sha256'] == sha(diagnostic_run/'plan.json'),
          'Naturally completed numerical diagnosis')
    expected_paths = {str(parent_run/n) for n in ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'wiring_preflight.json')}
    expected_paths |= {str(diagnostic_run/n) for n in ('plan.json', 'execution_receipt.json', 'launch_receipt.json')}
    check(set(plan['training_lineage']) == expected_paths, 'Complete failed-parent/diagnostic lineage')
    for path, record in plan['training_lineage'].items():
        check(path == record['path'] and plan['input_hashes'][path] == record['sha256'], 'Bound lineage record')
        audit.record(record)
    changed = {'output', 'source_snapshot', 'source_hashes', 'input_hashes', 'specification', 'numerics',
               'execution_budget', 'status', 'prepare_checkpoint_loads', 'prepare_target_decodes'}
    check(all(plan[k] == value for k, value in parent_plan.items() if k not in changed), 'Unchanged training/scoring inputs')
    check(all(plan['input_hashes'][k] == v for k, v in parent_plan['input_hashes'].items()), 'Original input inheritance')
    for bound_plan, sources_key in ((parent_plan, 'source_hashes'), (diagnostic_plan, 'sources')):
        snapshot = Path(bound_plan.get('source_snapshot', bound_plan.get('snapshot')))
        check(common.tree(snapshot) == bound_plan[sources_key], 'Preserved lineage source tree')
        for name, digest in bound_plan[sources_key].items():
            audit.bind(snapshot/name, digest)
    check(all(plan['source_hashes'][k] == v for k, v in parent_plan['source_hashes'].items()), 'Training source copied exactly')
    for key, value in diagnostic_plan['inputs'].items():
        audit.bind(key, value)
    for record in parent['actual_imports'].values():
        path = Path(record['path']); snapshot = Path(parent_plan['source_snapshot']); audit.record(record)
        check(path.is_relative_to(snapshot) and parent_plan['source_hashes'][str(path.relative_to(snapshot))] == record['sha256'],
              'Original actual import provenance')
    for record in diagnostic['actual_imports'].values():
        path = Path(record['path']); snapshot = Path(diagnostic_plan['snapshot']); audit.record(record)
        check(path.is_relative_to(snapshot) and diagnostic_plan['sources'][str(path.relative_to(snapshot))] == record['sha256'],
              'Diagnosis actual import provenance')
    evidence = recovery_numerics(parent, launch, diagnostic, execution['numerics_actual'])
    spec = plan['specification']
    check(spec['training_unchanged'] is True and spec['training_updates'] == 0
          and spec['training_numerics'] == TRAIN_NUMERICS and spec['evaluation_numerics'] == EVAL_NUMERICS
          and spec['training_plan_sha256'] == plan['training_plan']['sha256']
          and spec['diagnostic_execution_sha256'] == plan['numerical_recovery']['sha256'], 'Recovery policy binding')
    for key in ('evaluation', 'gain_thresholds', 'class_guard', 'bootstrap'):
        check(spec[key] == parent_plan['specification'][key], 'No scoring/selection change')
    evidence.update(training_plan=plan['training_plan'], numerical_recovery=plan['numerical_recovery'])
    return parent_run, parent_plan, parent, evidence


def gate_clauses(pairs):
    check(set(pairs) == set(THRESHOLDS), 'Fixed candidate references')
    out = {}
    for reference, threshold in THRESHOLDS.items():
        values = pairs[reference]
        out[reference+'_miou_gain'] = values['miou_all']['difference'] >= threshold
        out[reference+'_miou_ci_lower_positive'] = values['miou_all']['paired_view_bootstrap_95_interval'][0] > 0
        for name in common.CLASSES:
            out[reference+'_'+name+'_guard'] = values[name+'_iou']['difference'] >= (-.001 if name == 'stay_cable' else -.002)
    return {k: bool(v) for k, v in out.items()}


def saved_metrics(value, reference, audit):
    """Reaggregate saved per-view CM; does not independently derive CM from GT."""
    baseline = {r['name']: r for r in reference['views']}
    rows = {r['name']: r for r in value['views']}
    check(len(rows) == len(value['views']) == 50 and set(rows) == set(baseline), 'Metric population')
    matrices = []
    for name, row in rows.items():
        old = baseline[name]
        for key in ('name', 'width', 'height', 'rgb_pixels', 'psnr', 'ssim', 'lpips'):
            check(row[key] == old[key], 'E RGB score reuse differs')
        check(('confusion_matrix' in row) == ('confusion_matrix' in old), 'Semantic view population')
        if 'confusion_matrix' not in row:
            continue
        cm = np.asarray(row['confusion_matrix'])
        check(cm.shape == (5, 5) and cm.dtype.kind in 'iu' and (cm >= 0).all(), 'Bad saved CM')
        check(int(cm.sum()) == row['semantic_pixels'] == old['semantic_pixels']
              and row['semantic_ignore_pixels'] == old['semantic_ignore_pixels'], 'Saved CM support')
        matrices.append(cm)
    check(len(matrices) == 41, 'Annotated population')
    pooled = np.sum(matrices, axis=0); iou = common.iou(pooled)
    check(np.array_equal(pooled, value['confusion_matrix']), 'Pooled CM')
    for key, expected in [('iou', iou), ('miou_all', np.nanmean(iou)), ('miou_foreground', np.nanmean(iou[1:]))]:
        audit.close(value[key], expected, 'Saved '+key)
    for key in ('psnr', 'ssim', 'lpips'):
        audit.close(value[key], np.mean([r[key] for r in rows.values()]), 'Saved RGB mean')
    return {'miou_all': float(np.nanmean(iou)), 'iou': iou.tolist()}


def training(run, plan, execution, audit):
    manifest = read(plan['manifest']); train = [v for v in manifest['views'] if v['split'] == 'train']
    labeled = sorted((v for v in train if v.get('mask_path')), key=lambda v: v['name'])
    order = common.expected_order()
    check(len(train) == 350 and len(labeled) == 259 and labeled == plan['training_views']
          and order == plan['training_order'], 'TRAIN order/population')
    names = [labeled[i]['name'] for i in order]
    names_sha = hashlib.sha256(json.dumps(names).encode()).hexdigest()
    base = torch.load(plan['base_checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    head = {k: v for k, v in base['model'].items() if k.startswith('refiner.')}
    nonhead = {k: v for k, v in base['model'].items() if not k.startswith('refiner.')}
    head_sha, nonhead_sha = common.tensor_digest(head), common.tensor_digest(nonhead)
    cameras = base['training_cameras'].numpy()
    check(np.array_equal(cameras, np.asarray([v['w2c_original'] for v in train], np.float32)), 'Original TRAIN cameras')
    camera_sha = hashlib.sha256(cameras.tobytes()).hexdigest()
    audit.record(plan['q_delta'])
    with np.load(plan['q_delta']['path'], allow_pickle=False) as arrays:
        master, renderer = arrays['q_master'], arrays['q_renderer']
        check(master.dtype == np.float64 and renderer.dtype == np.float32 and master.shape == renderer.shape == (498136, 5)
              and np.isfinite(master).all() and (master >= 0).all()
              and np.max(np.abs(master.sum(1)-1)) <= 1e-12
              and master.astype(np.float32).tobytes() == renderer.tobytes(), 'Bound q sidecar/cast')
        optimized_sha = hashlib.sha256(renderer.tobytes()).hexdigest()
    check(optimized_sha == plan['optimized_q_renderer_sha256'], 'Optimized q identity')
    receipts, rng, summaries = [], [], []
    for index, arm in enumerate(ARMS):
        directory = run/arm; audit.bind(directory/'training_receipt.json')
        receipt = read(directory/'training_receipt.json'); receipts.append(receipt)
        check(receipt == execution['training'][index] and receipt['status'] == 'completed' and receipt['arm'] == arm,
              'Training receipt binding')
        check(all(receipt[k] == 2000 for k in ('steps', 'scene_calls', 'direct_q_shader_calls', 'optimizer_steps', 'backwards'))
              and receipt['val_payload_reads'] == receipt['real_rgb_payload_reads'] == 0, 'Training counts/scope')
        audit.bind(directory/'steps.jsonl', receipt['training_log_sha256'])
        rows = [json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
        check(len(rows) == 2000 and [r['step'] for r in rows] == list(range(1, 2001))
              and [r['view'] for r in rows] == names and receipt['sample_names_sha256'] == names_sha, 'Actual sample sequence')
        values = np.asarray([[r[k] for k in ('loss', 'final', 'residual_square')] for r in rows])
        check(np.isfinite(values).all() and np.allclose(values[:, 0], values[:, 1]+.001*values[:, 2], rtol=3e-7, atol=1e-9),
              'Head-only objective components')
        check(receipt['refiner_initial_sha256'] == head_sha and receipt['frozen_nonrefiner_sha256'] == nonhead_sha
              and receipt['training_camera_sha256'] == camera_sha, 'Initialization/frozen evidence')
        audit.bind(receipt['delta_path'], receipt['delta_sha256'])
        check(Path(receipt['delta_path']) == directory/'final_delta.pt', 'Final-only endpoint')
        delta = torch.load(receipt['delta_path'], map_location='cpu', mmap=True, weights_only=False)
        check(delta['format'] == 'direct_q_head_matched_delta_v1' and 'model' not in delta and 'partition_state' not in delta
              and delta['arm'] == arm and delta['step'] == 2000 and delta['specification'] == plan['specification']
              and delta['plan_sha256'] == sha(run/'plan.json') and delta['base_checkpoint_sha256'] == common.BASE_SHA
              and delta['manifest_sha256'] == common.MANIFEST_SHA and delta['refiner_config'] == base['refiner_config']
              and delta['pixel_protocol'] == 'legacy_mixed_v1' and delta['ordinary_resume_supported'] is False
              and delta['production_checkpoint'] is False, 'Delta scope/provenance')
        final = {'refiner.'+k: v for k, v in delta['refiner_state'].items()}
        check(set(final) == set(head) and all(v.shape == head[k].shape and v.dtype == head[k].dtype for k, v in final.items()),
              'Refiner-only tensor schema')
        check(common.tensor_digest(final) == receipt['refiner_final_sha256'] != head_sha, 'Refiner update hash')
        check(delta['training_camera_sha256'] == camera_sha and delta['q_sha256'] == receipt['q_initial_sha256']
              == receipt['q_final_sha256'], 'Fixed q/cameras')
        expected_q = execution['wiring_preflight'][arm+'_q_sha256']
        check(delta['q_sha256'] == expected_q and (arm != 'optimized_q' or expected_q == optimized_sha), 'Preflight/endpoint q')
        common.inspect_adam(delta['stage_optimizer_state'], delta['refiner_state'], None, 'refiner_only')
        check(receipt['parameter_count_refiner'] == sum(v.numel() for v in final.values()), 'Head parameter count')
        rng.append((delta['stage_torch_rng'].clone(), delta['stage_cuda_rng'].clone()))
        summaries.append({'arm': arm, 'delta_sha256': receipt['delta_sha256'], 'q_sha256': expected_q,
                          'elapsed_seconds': receipt['elapsed_seconds'], 'peak_cuda_allocated_bytes': receipt['peak_cuda_allocated_bytes']})
    check(receipts[0]['optimizer_groups'] == receipts[1]['optimizer_groups'], 'Matched optimizer groups')
    return {'arms': summaries, 'sample_order_exact': True, 'same_head_initialization': True,
            'end_rng_equal': all(torch.equal(a, b) for a, b in zip(rng[0], rng[1], strict=True)),
            'original_q_limit': 'GPU classifier q identity is attested by native-exact preflight and unchanged q hashes, not reconstructed with CPU softmax.'}


def evaluation(run, plan, execution, audit):
    directory = run/'evaluation'; audit.bind(directory/'execution_receipt.json')
    result = read(directory/'execution_receipt.json')
    check(result == execution['evaluation'] and result['plan_sha256'] == sha(run/'plan.json')
          and result['status'] == 'completed' and result['scene_calls'] == 250
          and result['direct_q_shader_calls'] == 200 and result['new_masks'] == 600
          and result['annotation_payload_reads'] == 41 and result['teacher_calls'] == result['gt_rgb_payload_reads'] == 0,
          'Evaluation scope')
    audit.bind(directory/'predictions_receipt.json', result['predictions_receipt_sha256'])
    predictions = read(directory/'predictions_receipt.json')
    check(predictions['status'] == 'all_600_masks_before_GT' and predictions['annotation_payload_reads'] == 0
          and predictions['scene_calls'] == 250 and predictions['direct_q_shader_calls'] == 200,
          'Prediction barrier')
    check(predictions['predictions_finished_utc'] == result['predictions_finished_utc']
          and datetime.fromisoformat(result['predictions_finished_utc']) <= datetime.fromisoformat(result['scoring_started_utc'])
          < datetime.fromisoformat(result['scoring_finished_utc']), 'Barrier chronology')
    previous_plan, previous, e = (read(plan[k]) for k in ('prior_plan_path', 'prior_receipt_path', 'prior_e_metrics'))
    views = previous_plan['views']; names = sorted(v['name'] for v in views)
    check(len(names) == len(set(names)) == 50 and sum(v['source_annotation_path'] is not None for v in views) == 41,
          'Official fixed views')
    check(previous_plan['teachers']['old']['sha256'] == common.TEACHER_SHA, 'Teacher cache identity')
    fingerprint_payload = {'evaluation_family': previous_plan['evaluation_family'], 'scoring_protocol': previous_plan['scoring_protocol'],
                           'views': [{k: v[k] for k in ('camera', 'source_image_sha256', 'source_annotation_sha256', 'rasterized_mask_sha256')}
                                     for v in sorted(views, key=lambda v: v['name'])]}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    check(fingerprint == previous_plan['reference_fingerprint'] == e['official_evaluation_fingerprint'], 'Fingerprint')
    old = {(r['arm'], r['name']): r for r in previous['predictions']}
    records = {(r['arm'], r['name']): r for r in predictions['records']}
    check(len(records) == len(predictions['records']) == 200 and set(records) == {(a, n) for a in READOUTS for n in names},
          'All fixed readouts')
    check(len(predictions['baseline_E_exact']) == 50 and len(predictions['original_q_zero_native_exact']) == 50
          and predictions['original_q_zero_native_exact'] == result['original_q_zero_native_exact']
          and all(all(r['checks'].values()) for r in result['original_q_zero_native_exact']), 'Native identity evidence')
    metrics, scores = {}, {}
    for arm in READOUTS:
        for kind in KINDS:
            binding = result['metrics'][arm+'_'+kind]; audit.record(binding); value = read(binding['path'])
            check(value['official_evaluation_fingerprint'] == fingerprint and value['validation_views'] == 50
                  and value['semantic_validation_views'] == 41, 'Metric identity')
            metrics[arm, kind] = value; scores[arm+'_'+kind] = saved_metrics(value, e, audit)
    for i, view in enumerate(views):
        name, camera = view['name'], view['camera']; w, h = camera['width'], camera['height']
        K, cw, ch, grid = common.original_grid(camera); cached = previous['h3_canvases'][name]
        check([cw, ch] == cached['boundary']['canvas'] and np.array_equal(K, np.asarray(cached['boundary']['canvas_K'], np.float32)),
              'Legacy teacher canvas')
        a, old_e = old['A', name], old['E', name]
        teacher = audit.probabilities(a['teacher_soft_canvas'], (ch, cw, 5))
        e_mask = {'path': old_e['mask'], 'sha256': old_e['mask_sha256']}
        baseline = predictions['baseline_E_exact'][i]
        check(Path(baseline['path']) == directory/'baseline/joint_mask'/name and baseline['sha256'] == e_mask['sha256'], 'E50 mask bytes')
        audit.png(baseline, (h, w)); audit.record(e_mask)
        masks = {}
        for arm in READOUTS:
            record = records[arm, name]
            check(record['teacher_input_rgb_exact'] and record['teacher_soft_canvas'] == a['teacher_soft_canvas']
                  and record['rgb_reference'] == {'path': old_e['rgb'], 'sha256': old_e['rgb_sha256']}, 'Input RGB/teacher provenance')
            audit.record(record['rgb_reference'])
            for folder, reference in [('canvas_rgb', cached['canvas_rgb']), ('h3_rgb', {'path': a['rgb'], 'sha256': a['rgb_sha256']})]:
                audit.bind(directory/arm/folder/name, reference['sha256']); audit.record(reference)
            final = audit.probabilities(record['soft_canvas'], (ch, cw, 5))
            for kind in KINDS:
                mask = audit.png(record['masks'][kind], (h, w)); check((mask < 5).all(), 'Unknown predicted class')
                masks[arm, kind] = mask
                if kind != 'raw':
                    soft = final if kind == 'scene' else (final+teacher)*np.float32(.5)
                    check(np.array_equal(common.mask_from_soft(soft, grid), mask), 'Independent five-channel blend/warp')
            del final
        check(records['original_q_zero', name]['masks']['joint']['sha256'] == e_mask['sha256'], 'Original-zero joint E reproduction')
        for arm in ARMS:
            check(records[arm, name]['masks']['raw']['sha256'] == records[arm+'_zero', name]['masks']['raw']['sha256'], 'Frozen q raw mask changed')
    pairs = {}
    definitions = [(ref, 'optimized_q', ref, False) for ref in THRESHOLDS]
    definitions += [('optimized_q_zero_minus_original_q_zero', 'optimized_q_zero', 'original_q_zero', True),
                    ('optimized_q_minus_optimized_q_zero', 'optimized_q', 'optimized_q_zero', True)]
    for key, candidate, reference, secondary in definitions:
        bindings = result['secondary_comparisons'] if secondary else result['comparisons']
        audit.record(bindings[key]); saved = read(bindings[key]['path'])
        independent = common.independent_bootstrap(e if reference == 'E' else metrics[reference, 'joint'], metrics[candidate, 'joint'])
        check(saved['candidate_arm'] == candidate and saved['reference_arm'] == reference and saved['bootstrap_repeats'] == 5000
              and saved['seed'] == 20260926 and saved['views'] == names and saved['official_evaluation_fingerprint'] == fingerprint,
              'Fixed paired comparison')
        check(not secondary or saved['descriptive_only'] is True, 'Zero-step cannot become candidate')
        for metric, value in independent.items():
            audit.close(value, saved['metrics'][metric], 'Pair '+key+'/'+metric)
        pairs[key] = independent
    clauses = gate_clauses({k: pairs[k] for k in THRESHOLDS}); audit.bind(directory/'system_gate.json', result['gate_sha256'])
    gate = read(directory/'system_gate.json')
    check(gate == result['gate'] and gate['clauses'] == clauses and gate['thresholds'] == THRESHOLDS
          and gate['passed'] == all(clauses.values()), 'Fixed14 gates')
    return {'scores': scores, 'pairs': pairs, 'gate_passed': gate['passed'], 'clauses': clauses,
            'saved_CM_reaggregated': 492, 'GT_annotations_decoded': 0, 'prediction_masks_checked': 650,
            'soft_scene_joint_masks_reconstructed': 400, 'fixed_q_raw_zero_endpoint_pairs_exact': 100,
            'scope': 'CM/support from saved metrics only; no GT rasterization or independent pixel-to-GT CM claim. RGB scores reused, not rescored.'}


def run_audit(run, expected, seconds=600):
    output = run/'independent_cpu_review.json'; check(not output.exists(), 'No overwrite')
    started = time.monotonic(); audit = common.Audit(); result = {'status': 'failed'}
    def timeout(*_):
        raise TimeoutError('Independent CPU audit deadline')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(seconds)
    try:
        check(not torch.cuda.is_initialized(), 'No CUDA'); torch.set_num_threads(4); common.cv2.setNumThreads(4)
        check(sha(run/'plan.json') == expected, 'Wrong plan')
        plan, execution, launch = (read(run/n) for n in ('plan.json', 'execution_receipt.json', 'launch_receipt.json'))
        check(launch['status'] == 'completed' and launch['natural_completion'] is True and launch['exit_code'] == 0
              and launch['plan_sha256'] == expected and launch['execution_receipt_sha256'] == sha(run/'execution_receipt.json'), 'Natural completion required')
        check(execution['status'] == 'completed' and execution['plan_sha256'] == expected
              and execution['bound_sources_inputs_unchanged'] and execution['new_val_annotation_payload_reads'] == 41, 'Complete producer')
        spec = plan['specification']; recovery = spec['protocol'] == 'direct_q_head_evaluation_recovery_v1'
        check((recovery and spec['readouts'] == list(READOUTS)
               or spec['protocol'] == 'direct_q_head_matched_v1' and spec['arms'] == list(ARMS))
              and spec['candidate'] == 'optimized_q' and spec['gain_thresholds'] == THRESHOLDS, 'Fixed candidate')
        snapshot = Path(plan['source_snapshot']); check(common.tree(snapshot) == plan['source_hashes'], 'Frozen tree')
        for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json'):
            audit.bind(run/name)
        for relative, digest in plan['source_hashes'].items():
            audit.bind(snapshot/relative, digest)
        for path, digest in {**plan['input_hashes'], **plan['installed_sources']}.items():
            audit.bind(path, digest)
        check(all(plan['source_hashes']['bridge_rgs/'+k] == v for k, v in plan['inherited_package_hashes'].items()), 'Inherited package')
        for record in execution['actual_imports'].values():
            audit.record(record); p = Path(record['path'])
            check(p.is_relative_to(snapshot) and plan['source_hashes'][str(p.relative_to(snapshot))] == record['sha256'], 'Actual import')
        for record in execution['loaded_gsplat_binary_sources'].values():
            audit.record(record)
        check(plan['expected_gsplat_binary'] in execution['loaded_gsplat_binary_sources'].values(), 'Actual gsplat binary')
        if recovery:
            training_run, training_plan, training_execution, result['recovery'] = recovery_lineage(plan, execution, audit)
        else:
            training_run, training_plan, training_execution = run, plan, execution
        wiring = training_execution['wiring_preflight']
        check(wiring['passed'] and all(all(c.values()) for c in wiring['checks'].values()), 'Wiring preflight')
        result['training'] = training(training_run, training_plan, training_execution, audit)
        result['evaluation'] = evaluation(run, plan, execution, audit)
        audit.final_hash_check(); check(not torch.cuda.is_initialized() and common.tree(snapshot) == plan['source_hashes'], 'Final CPU/source state')
        result.update(status='passed', plan_sha256=expected, source_count=len(plan['source_hashes']), input_count=len(plan['input_hashes']),
                      max_numeric_difference=audit.max_error, unique_bound_artifacts=len(audit.bound), CUDA_initialized=False,
                      helper_sha256=sha(common.__file__), checkpoint_reads='CPU tensor dictionaries only; no model construction or inference',
                      target_scope='Bound label/image bytes may be hashed; no GT decoding, GT rescoring, RGB metric recomputation or new predictions.')
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'; raise
    finally:
        signal.alarm(0); result.update(elapsed_seconds=time.monotonic()-started, checker_sha256=sha(__file__))
        with output.open('x') as handle:
            json.dump(result, handle, indent=2, allow_nan=False); handle.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run', type=Path)
    parser.add_argument('--expected-plan-sha256', required=True); args = parser.parse_args()
    run_audit(args.run.resolve(), args.expected_plan_sha256)
