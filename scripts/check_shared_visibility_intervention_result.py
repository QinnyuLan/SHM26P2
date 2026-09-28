"""Independent CPU audit of saved shared-opacity predictions; never invokes the worker or renderer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

NAMES = ['002.png', '043.png', '084.png', '125.png', '167.png', '215.png', '257.png', '300.png']
ORDER = ['base', 'minus', 'plus'] * 2
KEYS = ('rgb_full', 'ce_full', 'rgb_mass', 'ce_mass')
BASE_SHA = '391a0577450f7f458b75cb9e2fcf926953142b516728ea3600ef2ef064d72c13'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def compare_tree(actual, expected, path='result'):
    """Discrete values exact; alternate FP64 reduction order allowed at 1e-12."""
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'Different fields: ' + path)
        for key in expected:
            compare_tree(actual[key], expected[key], path + '.' + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'Different length: ' + path)
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            compare_tree(left, right, f'{path}[{index}]')
    elif isinstance(expected, float):
        require(np.isfinite(actual) and np.isfinite(expected)
                and abs(actual - expected) <= 1e-12 * max(1., abs(expected)), 'Numeric mismatch: ' + path)
    else:
        require(actual == expected, 'Discrete mismatch: ' + path)


def effects(values):
    first, second = np.asarray(values[:3], np.float64), np.asarray(values[3:], np.float64)
    require(first.shape == second.shape == (3,) and np.isfinite([first, second]).all(), 'Bad repeated measurements')
    mean = (first + second) * .5
    tau = float(10 * max(float(np.max(np.abs(first - second))),
                         32 * np.finfo(np.float64).eps * max(1., float(np.max(np.abs([first, second]))))))
    return {'base': float(mean[0]), 'minus': float(mean[1]), 'plus': float(mean[2]),
            'minus_gain': float(mean[0] - mean[1]), 'plus_gain': float(mean[0] - mean[2]),
            'repeat_tolerance': tau, 'minus_relative_gain': float((mean[0] - mean[1]) / mean[0]) if mean[0] > 0 else None}


def metrics_from_arrays(prediction, mass, truth, labels, valid, class_weights):
    support = valid & (labels < 5)
    require(valid.any() and support.any(), 'No labeled/valid support')
    rgb, raw, alpha = [prediction[k] for k in ('rgb', 'raw', 'alpha')]
    require(all(np.isfinite(x).all() for x in (rgb, raw, alpha, mass)) and (raw >= 0).all(), 'Nonfinite/negative evidence')
    w = mass[support].astype(np.float64)
    total = np.sum(w, dtype=np.float64)
    square_total = np.sum(w * w, dtype=np.float64)
    neff = float(total * total / square_total) if square_total > 0 else 0.
    coverage = {'mass_sum': float(total), 'effective_pixels': neff, 'measurable': bool(total >= 64 and neff >= 256)}
    mse = np.sum((rgb.astype(np.float64) - truth.astype(np.float64)) ** 2, axis=2) / 3
    y = labels[support].astype(np.int64)
    py = np.take_along_axis(raw[support].astype(np.float64), y[:, None], axis=1)[:, 0]
    loss_ce = -np.log((1 - 5e-7) * py + 5e-7 / 5) * np.asarray(class_weights, np.float64)[y]
    ids = prediction['raw_ids']
    require(np.isin(ids, [0, 1, 2, 3, 4]).all(), 'Invalid saved semantic ID')
    # Common positive normalization does not change real-arithmetic class order;
    # use actual saved FP32 p3d IDs for CM, and disclose any near-tie discrepancy.
    raw_argmax = np.maximum(raw, np.float32(1e-7)).argmax(-1)
    cm = np.zeros((5, 5), np.int64)
    np.add.at(cm, (y, ids[support]), 1)
    result = {'rgb_full': float(mse[valid].mean()), 'ce_full': float(loss_ce.mean()),
              'rgb_mass': float(np.sum(w * mse[support], dtype=np.float64) / total) if total else None,
              'ce_mass': float(np.sum(w * loss_ce, dtype=np.float64) / total) if total else None,
              'cm': cm.tolist(), 'alpha_mean': float(alpha[valid].astype(np.float64).mean())}
    return result, coverage, int(np.count_nonzero(raw_argmax != ids))


def pooled_metrics(matrix):
    matrix = np.asarray(matrix, np.int64)
    diagonal = matrix.diagonal()
    union = matrix.sum(0) + matrix.sum(1) - diagonal
    iou = [float(diagonal[i] / union[i]) if union[i] else None for i in range(5)]
    return {'confusion_matrix': matrix.tolist(), 'iou': iou,
            'miou': float(np.mean([x for x in iou if x is not None])) if any(union) else None,
            'false_positive': (matrix.sum(0) - diagonal).tolist(),
            'false_negative': (matrix.sum(1) - diagonal).tolist()}


def independent_summary(rows):
    measured = [r for r in rows if r['coverage']['measurable']]
    aggregates = {}
    for key in KEYS:
        chosen = rows if key in ('rgb_full', 'ce_full') else measured
        aggregates[key] = effects([float(np.mean([r['predictions'][i][key] for r in chosen]))
                                   for i in range(6)]) if chosen else None
    per_view = []
    for row in rows:
        per_view.append({key: effects([p[key] for p in row['predictions']]) for key in KEYS
                         if all(p[key] is not None for p in row['predictions'])})
    joint = sum(all(per_view[i][key]['minus_gain'] > per_view[i][key]['repeat_tolerance']
                    for key in ('rgb_mass', 'ce_mass')) for i, row in enumerate(rows) if row['coverage']['measurable'])
    minimum_joint = max(2, math.ceil(.75 * len(measured)))
    clauses = {'coverage': len(measured) >= 2, 'joint_views': joint >= minimum_joint}
    for key in ('rgb_mass', 'ce_mass'):
        e = aggregates[key]
        clauses[key + '_gain'] = bool(e and e['minus_gain'] > e['repeat_tolerance']
                                      and e['minus_relative_gain'] is not None and e['minus_relative_gain'] >= .001)
        clauses[key + '_direction'] = bool(e and e['minus_gain'] - e['plus_gain'] > e['repeat_tolerance'])
    e = aggregates['rgb_full']
    clauses['full_rgb_no_regression'] = e['minus_gain'] >= -e['repeat_tolerance']
    e = aggregates['ce_full']
    clauses['full_semantic_improves'] = e['minus_gain'] > e['repeat_tolerance']
    decision = 'inconclusive_coverage' if len(measured) < 2 else (
        'necessary_interference_signal_present' if all(clauses.values()) else 'specified_intervention_not_supported')
    return {'decision': decision, 'clauses': clauses, 'covered_views': [r['name'] for r in measured],
            'joint_improving_views': joint, 'required_joint_views': minimum_joint, 'effects': aggregates,
            'pooled_raw': {case: pooled_metrics(sum((np.asarray(row['predictions'][i]['cm'], np.int64)
                                                   for row in rows), np.zeros((5, 5), np.int64)))
                           for i, case in enumerate(ORDER[:3])},
            'cm_scope': 'First independent prediction per state; repeats used for numerical loss floor, never double-count GT'}, per_view


def audit(plan_path, expected_sha, output, recovery=False):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    require(not output.exists(), 'Do not overwrite a CPU audit')
    require(not torch.cuda.is_initialized(), 'CPU-only checker')
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    require(digest(plan_path) == expected_sha, 'Caller-bound plan changed')
    recovery_plan, recovery_path = (read(plan_path), plan_path) if recovery else (None, None)
    if recovery:
        recovery_snapshot = Path(recovery_plan['source_snapshot'])
        require({str(p.relative_to(recovery_snapshot)): digest(p) for p in recovery_snapshot.rglob('*.py')}
                == recovery_plan['source_hashes'], 'CPU recovery source changed')
        for path, sha in recovery_plan['input_hashes'].items():
            require(digest(path) == sha, 'CPU recovery bound input changed: ' + path)
        plan_path = Path(recovery_plan['original_plan'])
        require(digest(plan_path) == recovery_plan['original_plan_sha256'], 'Original failed-attempt plan changed')
    plan, run = read(plan_path), plan_path.parent
    receipt = read(run / 'execution_receipt.json')
    measurement_path = recovery_path.parent / 'recovered_measurements.json' if recovery else run / 'audit.json'
    worker = read(measurement_path)
    if recovery:
        require(not (run / 'audit.json').exists(), 'Do not insert or replace the missing original runtime audit')
        recovery_receipt = read(recovery_path.parent / 'execution_receipt.json')
        require(recovery_receipt['status'] == 'completed_cpu_recovery_only'
                and recovery_receipt['plan_sha256'] == expected_sha
                and recovery_receipt['recovered_measurements_sha256'] == digest(measurement_path),
                'CPU recovery result is not bound to its completed receipt')
        require(worker['recovery_plan_sha256'] == expected_sha
                and worker['original_execution']['receipt_sha256'] == digest(run / 'execution_receipt.json'),
                'CPU reconstruction provenance differs')
        require(receipt['status'] == 'failed' and receipt['exit_code'] == 1,
                'CPU recovery must preserve the original failed execution')
        require(worker['status'] == 'recovered_measurements_only', 'Not a saved-array-only CPU reconstruction')
        require(worker['original_execution']['status'] == 'failed'
                and worker['original_execution']['exit_code'] == 1, 'Recovery disguises original execution failure')
        require(all(worker[k] == 0 for k in ('new_renders', 'new_backward', 'new_optimizer_steps')),
                'Recovery performed a new render or optimization')
        require(all(worker['runtime_evidence'].get(k) is None for k in
                    ('all_state_restored_exact', 'camera_unchanged', 'executed_raster_calls',
                     'prediction_before_gt_runtime_timestamps')), 'Missing runtime evidence must remain unavailable')
    else:
        require(receipt['status'] == worker['status'] == 'completed' and receipt['exit_code'] == 0,
                'Only naturally completed diagnostic is auditable without explicit recovery mode')
        require(worker['plan_sha256'] == expected_sha
                and digest(measurement_path) == receipt['audit_sha256'], 'Broken completion binding')
    require(receipt['plan_sha256'] == digest(plan_path), 'Original execution/plan binding differs')
    fixed = {'protocol': 'shared_visibility_intervention_v1', 'names': NAMES, 'order': ORDER,
             'points': 498136, 'group_points': 3004, 'distance_scene_scale': .01,
             'distance': .16261470794677735, 'logit_shift': math.log(2), 'width': 1320, 'height': 989,
             'degree': 3, 'scene_calls': 48, 'contribution_calls': 8, 'raster_calls': 104,
             'backwards': 0, 'optimizer_steps': 0, 'delta': 5e-7, 'mass_atol': 5e-6,
             'minimum_mass': 64., 'minimum_effective_pixels': 256., 'minimum_covered_views': 2,
             'joint_fraction': .75, 'minimum_relative_gain': .001, 'repeat_floor_multiplier': 10.,
             'fp64_floor_multiplier': 32., 'timeout_seconds': 120}
    require(all(plan['specification'][k] == v for k, v in fixed.items()), 'Fixed numerical protocol changed')
    snapshot = Path(plan['source_snapshot'])
    sources = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')}
    require(sources == plan['source_hashes'] == receipt['source_hashes'], 'Frozen source changed')
    require({str(p.relative_to(snapshot / 'bridge_rgs')): digest(p) for p in (snapshot / 'bridge_rgs').rglob('*.py')}
            == plan['inherited_package_hashes'], 'Inherited numerical package changed')
    for path, sha in plan['input_hashes'].items():
        require(digest(path) == sha, 'Bound input changed: ' + path)
    if not recovery:
        for key in ('scene_calls', 'contribution_calls', 'raster_calls', 'backwards', 'optimizer_steps'):
            require(worker[key] == fixed[key], 'Execution budget differs')
        require(worker['predictions_finished_utc'] <= worker['scoring_started_utc'], 'Prediction/GT phase order differs')
        require(worker['camera_unchanged'] and all(worker['restoration'].get(k) is True for k in
                ('all_tensors_restored_exact', 'all_flags_restored', 'all_modes_restored')), 'Runtime state restoration not certified')
        require(worker['no_parameter_gradients'], 'Unexpected parameter gradients')
    require(digest(plan['checkpoint']) == BASE_SHA, 'Wrong completed base field')
    state = torch.load(plan['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    if not recovery:
        require(worker['restoration']['tensor_hashes_before'] == worker['restoration']['tensor_hashes_after']
                == {k: tensor_digest(v) for k, v in state['model'].items()}, 'Restored state digest differs from original checkpoint')
    manifest = read(plan['manifest'])
    train = sorted((v for v in manifest['views'] if v['split'] == 'train'), key=lambda v: v['name'])
    require(len(train) == 350 and len({v['name'] for v in train}) == 350, 'Wrong TRAIN camera set')
    cameras = state['training_cameras']
    require(tensor_digest(cameras) == plan['training_cameras_sha256']
            and torch.equal(cameras, torch.tensor([v['w2c_original'] for v in train], dtype=torch.float32)), 'Original cameras differ')
    require(state['step'] == 8000 and state['sh_degree'] == 3 and not state.get('mip_filter_config'), 'Wrong base architecture/profile')
    require(float(state['scene_scale']) * .01 == fixed['distance'], 'Wrong world threshold')
    means = state['model']['splats.means'].double().numpy()
    poses = cameras.double().numpy()
    centers = np.stack([-p[:3, :3].T @ p[:3, 3] for p in poses])
    nearest_camera = cKDTree(centers).query(means, workers=8)[0]
    with np.load(plan['init_points'], allow_pickle=False) as init:
        nearest_init = cKDTree(init['points'].astype(np.float64)).query(means, workers=8)[0]
    ids = np.where((nearest_camera < fixed['distance']) & (nearest_init > fixed['distance']))[0].astype(np.int64)
    saved_ids = np.load(plan['group_ids'], allow_pickle=False)
    require(len(means) == 498136 and saved_ids.dtype == np.int64 and len(ids) == 3004
            and np.array_equal(ids, saved_ids), 'Independent geometry-only G selection differs')
    original_logits = state['model']['splats.opacity_logits']
    reconstructed_logit_hashes = {}
    for case, sign in [('base', 0), ('minus', -1), ('plus', 1)]:
        logits = original_logits.clone()
        if sign:
            logits[torch.from_numpy(ids)] += sign * math.log(2)
        reconstructed_logit_hashes[case] = tensor_digest(logits)
        if not recovery:
            require(reconstructed_logit_hashes[case] == worker['interventions'][case]['logits_sha256'],
                    'Recorded intervention differs from independently shifted original FP32 logits')
    counts = np.asarray(plan['class_counts'], np.float64)
    weights = np.maximum(counts / counts.sum(), .002) ** (-.25)
    weights = np.minimum(weights / weights.mean(), 3).astype(np.float32)
    require(np.array_equal(weights, np.asarray(plan['class_weights'], np.float32)), 'Class weight formula differs')
    weight_plan = read(plan['weight_source']['plan'])
    require(digest(plan['weight_source']['plan']) == plan['weight_source']['plan_sha256']
            and np.array_equal(weights, np.asarray(weight_plan['class_weights'], np.float32)), 'Class weight provenance differs')
    array_list = recovery_plan['prediction_arrays'] if recovery else worker['prediction_arrays']
    array_records = {record['path']: record for record in array_list}
    require(len(array_records) == len(array_list), 'Duplicate prediction-array binding')
    require(set(array_records) == {str(p) for p in (run / 'predictions').glob('*.npy')}, 'Unbound or missing saved array')
    for path, record in array_records.items():
        require(digest(path) == record['sha256'], 'Saved prediction changed')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        require(list(array.shape) == record['shape'] and str(array.dtype) == record['dtype'], 'Saved shape/dtype differs')
    require([v['name'] for v in plan['views']] == [r['name'] for r in worker['rows']] == NAMES, 'Wrong fixed8 order')
    if not recovery:
        require(len(worker['trace']) == 48, 'Incomplete scene-call trace')
    recomputed_rows, raw_id_discrepancies, alpha_contracts = [], [], []
    expected_arrays = set()
    for vi, view in enumerate(plan['views']):
        require(view['split'] == 'train' and train[view['camera_index']]['name'] == view['name'], 'Non-TRAIN/misaligned camera')
        original = train[view['camera_index']]
        require(view['K'] == original['K'] and all(Path(view[k]).resolve() == Path(original[k]).resolve()
                for k in ('image_path', 'mask_path', 'valid_path')), 'View target or intrinsics changed')
        truth = cv2.imread(view['image_path'], cv2.IMREAD_COLOR)[..., ::-1].copy().astype(np.float32) / 255
        labels = cv2.imread(view['mask_path'], cv2.IMREAD_UNCHANGED)
        valid = cv2.imread(view['valid_path'], cv2.IMREAD_GRAYSCALE) > 0
        require(truth.shape == (989, 1320, 3) and labels.shape == valid.shape == (989, 1320), 'Wrong native target grid')
        mass_path = run / 'predictions' / f"{view['name']}.mass.npy"
        expected_arrays.add(str(mass_path))
        mass = np.load(mass_path, mmap_mode='r', allow_pickle=False)
        require(mass.dtype == np.float32 and mass.shape == (989, 1320) and (mass >= 0).all(), 'Invalid fixed mass')
        contribution = {}
        for key in ('contribution_total', 'contribution_alpha'):
            path = run / 'predictions' / f"{view['name']}.{key}.npy"
            expected_arrays.add(str(path))
            contribution[key] = np.load(path, mmap_mode='r', allow_pickle=False)
            require(contribution[key].shape == (989, 1320) and contribution[key].dtype == np.float32
                    and np.isfinite(contribution[key]).all(), 'Invalid contribution evidence')
        alpha = contribution['contribution_alpha']
        full_mass_error = float(np.max(np.abs(contribution['contribution_total'].astype(np.float64) - alpha)))
        require(full_mass_error <= 5e-6 and float(np.max(mass.astype(np.float64) - alpha)) <= 5e-6,
                'Forward contribution mass/alpha contract differs')
        row = {'name': view['name'], 'predictions': []}
        for index, case in enumerate(ORDER):
            paths = {key: run / 'predictions' / f"{view['name']}.{index}.{case}.{key}.npy"
                     for key in ('rgb', 'raw', 'alpha', 'raw_ids', 'rgb_alpha')}
            expected_arrays.update(str(p) for p in paths.values())
            prediction = {key: np.load(path, mmap_mode='r', allow_pickle=False) for key, path in paths.items()}
            for key, shape in [('rgb', (989, 1320, 3)), ('raw', (989, 1320, 5)), ('alpha', (989, 1320)),
                               ('raw_ids', (989, 1320)), ('rgb_alpha', (989, 1320))]:
                require(prediction[key].shape == shape and prediction[key].dtype == (np.uint8 if key == 'raw_ids' else np.float32), 'Prediction layout differs')
            require(np.isfinite(prediction['rgb_alpha']).all(), 'Nonfinite RGB alpha')
            rgb_semantic_error = float(np.max(np.abs(prediction['rgb_alpha'].astype(np.float64) - prediction['alpha'])))
            require(rgb_semantic_error <= 5e-6, 'RGB/semantic saved alpha mismatch')
            if index == 0:
                require(float(np.max(mass.astype(np.float64) - prediction['alpha'])) <= 5e-6, 'Group mass exceeds base alpha')
                mass_contract = {'full_mass_alpha_max_error': full_mass_error,
                                 'scene_alpha_max_error': float(np.max(np.abs(alpha.astype(np.float64) - prediction['alpha']))),
                                 'rgb_alpha_max_error': float(np.max(np.abs(alpha.astype(np.float64) - prediction['rgb_alpha']))),
                                 'mass_min': float(mass.min()), 'mass_max': float(mass.max())}
                require(max(mass_contract[k] for k in ('scene_alpha_max_error', 'rgb_alpha_max_error')) <= 5e-6,
                        'Contribution/scene alpha mismatch')
                compare_tree(mass_contract, worker['rows'][vi]['mass_contract'], view['name'] + '.mass_contract')
                alpha_contracts.append({'name': view['name'], **mass_contract})
            values, cover, raw_ties = metrics_from_arrays(prediction, mass, truth, labels, valid, weights)
            row['predictions'].append(values)
            row['coverage'] = cover
            compare_tree(values, worker['rows'][vi]['predictions'][index], f'{view["name"]}.{index}')
            raw_id_discrepancies.append({'name': view['name'], 'index': index, 'clamped_raw_vs_saved_p3d_argmax_pixels': raw_ties})
            if not recovery:
                trace = worker['trace'][vi * 6 + index]
                require(trace['name'] == view['name'] and trace['case'] == case and trace['repeat'] == index // 3
                        and trace['scene_calls'] == vi * 6 + index + 1
                        and trace['raster_calls'] == vi * 13 + (index + 1) * 2 + 1
                        and 0 <= trace['rgb_semantic_alpha_max_error'] <= 5e-6, 'Scene/raster/alpha trace differs')
                require(rgb_semantic_error == trace['rgb_semantic_alpha_max_error'], 'Reported alpha error differs from saved arrays')
        compare_tree(cover, worker['rows'][vi]['coverage'], view['name'] + '.coverage')
        contract = worker['rows'][vi]['mass_contract']
        require(all(0 <= contract[k] <= 5e-6 for k in
                    ('full_mass_alpha_max_error', 'scene_alpha_max_error', 'rgb_alpha_max_error')), 'Forward alpha contract failed')
        recomputed_rows.append(row)
    require(expected_arrays == set(array_records) and len(expected_arrays) == 264, 'Saved-array budget/schema differs')
    summary, per_view = independent_summary(recomputed_rows)
    compare_tree(summary, worker['summary'], 'summary')
    for row, reported, effect in zip(recomputed_rows, worker['rows'], per_view, strict=True):
        compare_tree(effect, reported['effects'], row['name'] + '.effects')
        row['effects'] = effect
    if not recovery:
        require(receipt['decision'] == summary['decision'], 'Receipt decision differs')
    else:
        require(recovery_receipt['decision'] == summary['decision'], 'CPU recovery receipt decision differs')
    require(not torch.cuda.is_initialized() and digest(plan['checkpoint']) == BASE_SHA, 'GPU used or persistent checkpoint changed')
    report = {'status': 'passed_cpu_reconstruction' if recovery else 'passed', 'cpu_only': True,
              'plan_sha256': digest(plan_path), 'recovery_plan_sha256': expected_sha if recovery else None,
              'measurements_sha256': digest(measurement_path), 'execution_receipt_sha256': digest(run / 'execution_receipt.json'),
              'checker_sha256': digest(__file__), 'source_input_endpoint_hashes_exact': True,
              'saved_prediction_sets': 48, 'scene_calls': None if recovery else 48,
              'raster_calls': None if recovery else 104, 'optimizer_steps': None if recovery else 0,
              'backwards': None if recovery else 0, 'new_CPU_audit_renders': 0,
              'group_ids_independently_recomputed_exact': True, 'group_count': 3004, 'group_ids_sha256': digest(plan['group_ids']),
              'reconstructed_FP32_logit_intervention_sha256': reconstructed_logit_hashes,
              'intervention_runtime_hashes_match': None if recovery else True,
              'saved_array_count': len(array_records), 'saved_array_hashes_verified': True,
              'runtime_restoration_attestation': None if recovery else worker['restoration'],
              'persistent_base_checkpoint_unchanged': True,
              'evidence_limits': ('Original GPU attempt failed during final JSON serialization. Saved predictions permit CPU '
                                 'measurement reconstruction, not retroactive success. Runtime tensor/flag/mode restoration hashes, '
                                 'camera/call counters and phase timestamps were not saved and remain unavailable. The original '
                                 'checkpoint bytes still match their input SHA. Frozen control flow implies48scene/104raster/0optimizer, '
                                 'but this is not a recovered runtime trace. All three alpha paths are independently checked from arrays.'
                                 if recovery else
                                 'Runtime call counts and full state/flag/mode restoration are worker attestations tied to frozen source; '
                                 'recorded before/after state hashes also match the original checkpoint. CPU checker does not rerun a '
                                 'renderer or directly observe historical GPU state. All three alpha paths are recomputed from saved arrays.'),
              'numeric_comparison': 'Independent FP64 reductions; tolerance1e-12*max(1,abs(reported)); discrete gate/CM exact.',
              'independent_alpha_contracts': alpha_contracts,
              'raw_argmax_diagnostics': raw_id_discrepancies,
              'rows': recomputed_rows, 'summary': summary,
              'scope': 'Fixed8 TRAIN, fixedG/logit shifts only; no VAL/generalization, new method or automatic model adoption claim.'}
    if recovery:
        report['original_execution'] = {'status': receipt['status'], 'exit_code': receipt['exit_code'],
                                         'receipt': str(run / 'execution_receipt.json'),
                                         'worker_log_sha256': digest(run / 'worker.log')}
        report['recovery_execution_receipt_sha256'] = digest(recovery_path.parent / 'execution_receipt.json')
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--plan', type=Path)
    modes.add_argument('--recovery-plan', type=Path)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    selected = args.recovery_plan or args.plan
    result = audit(selected, args.expected_plan_sha256, args.output or selected.parent / 'independent_cpu_result_audit.json',
                   recovery=args.recovery_plan is not None)
    print(json.dumps({'status': result['status'], 'decision': result['summary']['decision']}, indent=2))
