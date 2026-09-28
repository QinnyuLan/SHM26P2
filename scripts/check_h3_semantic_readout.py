"""Independent CPU/NumPy verification of the completed fixed four-render diagnostic."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

RUN = Path('/mnt/data/SHM2026/runs/h3_semantic_readout_diagnostic_v1')
PLAN_SHA = 'cd0e6336a3df4e7d01890926f6590f10f493993a435a6640f5efc7e54055e81f'
RECEIPT_SHA = 'fadabf80c0f1ddd501f9774ea50b01697ae67b4a9208aef67bd1cca81c3be88c'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def score(p, y, valid):
    keep = valid & (y != 255)
    pred = p.argmax(-1)
    cm = np.bincount((y[keep]*5+pred[keep]).astype(np.int64), minlength=25).reshape(5, 5)
    truth_probability = p[keep][np.arange(int(keep.sum())), y[keep]]
    ce = float(-np.log(np.maximum(truth_probability.astype(np.float64), 1e-7)).mean())
    ce_log32 = float(-np.log(np.maximum(truth_probability, np.float32(1e-7))).sum(dtype=np.float64)/keep.sum())
    return cm, ce, ce_log32


def iou(cm):
    union = cm.sum(0)+cm.sum(1)-cm.diagonal()
    values = cm.diagonal()/np.maximum(union, 1)
    return {'iou': values.tolist(), 'miou_all': float(values[union > 0].mean()),
            'miou_foreground': float(values[1:][union[1:] > 0].mean())}


def check_iou(actual, expected):
    a = actual['iou']+[actual['miou_all'], actual['miou_foreground']]
    b = expected['iou']+[expected['miou_all'], expected['miou_foreground']]
    assert np.allclose(a, b, rtol=0, atol=2e-15)  # float64 reduction order only; CM must be exact.


def corrections(a, b, y, valid):
    keep = valid & (y < 5)
    raw, first, second = a['p3d'].argmax(-1), a['probabilities'].argmax(-1), b['probabilities'].argmax(-1)
    corrected = keep & (y == 2) & (raw == 0) & (first == 2)
    return {'original_cable_raw_bg_corrected_by_head': int(corrected.sum()),
            'correction_retained_after_projection': int((corrected & (second == 2)).sum()),
            'correction_lost_after_projection': int((corrected & (second != 2)).sum()),
            'new_cable_corrections': int((keep & (y == 2) & (first != 2) & (second == 2)).sum()),
            'new_cable_false_positives': int((keep & (y != 2) & (first != 2) & (second == 2)).sum()),
            'all_class_new_errors': int((keep & (first == y) & (second != y)).sum()),
            'all_class_new_corrections': int((keep & (first != y) & (second == y)).sum()),
            'final_argmax_changes': int((keep & (first != second)).sum())}


def main():
    torch.set_num_threads(8)
    assert not torch.cuda.is_initialized()
    assert sha(RUN/'plan.json') == PLAN_SHA and sha(RUN/'execution_receipt.json') == RECEIPT_SHA
    plan = json.loads((RUN/'plan.json').read_text())
    result = json.loads((RUN/'execution_receipt.json').read_text())
    launch = json.loads((RUN/'launch_receipt.json').read_text())
    assert launch['status'] == 'completed' and launch['exit_code'] == 0
    assert launch['plan_sha256'] == PLAN_SHA and launch['execution_receipt_sha256'] == RECEIPT_SHA
    assert result['status'] == 'completed' and result['all_invariants_passed'] and result['numerical_gate_passed']
    assert result['plan_sha256'] == PLAN_SHA and result['source_hashes'] == plan['source_hashes']
    assert result['input_hashes'] == plan['input_hashes'] and result['all_bound_inputs_and_sources_unchanged']
    assert (result['scene_renders'], result['raster_calls'], result['backward_calls'], result['optimizer_steps']) == (4, 8, 0, 0)
    assert not result['checkpoint_written'] and result['real_rgb_pixels_decoded'] == result['val_pixels_decoded'] == 0
    paths = dict(plan['input_hashes'])
    snapshot = Path(plan['source_snapshot'])
    for path, expected in plan['source_hashes'].items():
        paths[str(snapshot/path)] = expected
    for path, expected in paths.items():
        assert sha(path) == expected, path
    for entry in result['actual_imports'].values():
        p = Path(entry['path'])
        assert p.is_relative_to(snapshot) and plan['source_hashes'][str(p.relative_to(snapshot))] == entry['sha256'] == sha(p)
    manifest = json.loads(Path(plan['manifest']).read_text())
    train = [v for v in manifest['views'] if v['split'] == 'train']
    assert [v['name'] for v in train] == plan['training_camera_names']
    assert [v['name'] for v in sorted(train, key=lambda x: x['name'])[::175]][:2] == ['002.png', '205.png']
    state = torch.load(plan['base'], map_location='cpu', mmap=True, weights_only=False)
    cameras = state['training_cameras'].numpy()
    expected_cameras = np.array([v['w2c_original'] for v in train], dtype=np.float32)
    assert np.array_equal(cameras, expected_cameras)
    restoration = result['restoration']
    hashes = {k: array_sha(v.detach().numpy()) for k, v in state['model'].items()}
    assert hashes == restoration['tensor_hashes_before'] == restoration['tensor_hashes_after']
    assert array_sha(cameras) == plan['training_camera_sha256'] == restoration['camera_hash_before'] == restoration['camera_hash_after']
    assert all(restoration[k] for k in ('all_model_tensors_finally_restored_exact', 'flags_restored_exact', 'modes_restored_exact',
                                      'cameras_restored_exact', 'no_parameter_gradients', 'non_feature_tensors_unchanged_before_restore',
                                      'cameras_unchanged_before_restore'))
    expected_reads = [v[k] for v in plan['views'] for k in ('mask_path', 'valid_path')]
    assert result['pixel_scope']['predictions_completed'] and result['pixel_scope']['reads'] == expected_reads
    assert set(expected_reads) == set(plan['allowed_pixel_paths']) and len(expected_reads) == 4
    assert plan['specification']['ce']['weights'] == [1., 1., 1., 1., 1.]
    data, records = {}, {}
    for record in result['predictions']:
        name, condition = record['name'], record['condition']
        assert name in ('002.png', '205.png') and condition in ('original', 'projected')
        assert sha(record['path']) == record['sha256']
        with np.load(record['path'], allow_pickle=False) as file:
            assert set(file.files) == {'p3d', 'probabilities'}
            arrays = {key: file[key] for key in file.files}
        for key, p in arrays.items():
            assert p.shape == (989, 1320, 5) and p.dtype == np.float32 and np.isfinite(p).all()
            assert array_sha(p) == record['tensor_hashes'][key]
        data[name, condition], records[name, condition] = arrays, record
    assert len(data) == 4
    rows, pooled = [], {(condition, key): np.zeros((5, 5), np.int64) for condition in ('original', 'projected') for key in ('p3d', 'probabilities')}
    for view, stored in zip(plan['views'], result['views'], strict=True):
        name = view['name']
        assert view['split'] == 'train' and train[view['camera_index']]['name'] == name == stored['name']
        a, b = data[name, 'original'], data[name, 'projected']
        difference = abs(a['p3d'].astype(np.float64)-b['p3d'].astype(np.float64))
        error, argmax_changes = float(difference.max()), int((a['p3d'].argmax(-1) != b['p3d'].argmax(-1)).sum())
        assert error == stored['invariance']['p3d_max_absolute'] and error <= 1e-6
        assert argmax_changes == stored['invariance']['raw_argmax_changes'] == 0
        for key in ('rgb', 'depth', 'alpha'):
            assert records[name, 'original']['tensor_hashes'][key] == records[name, 'projected']['tensor_hashes'][key]
            assert stored['invariance']['exact'][key]
        labels = cv2.imread(view['mask_path'], cv2.IMREAD_GRAYSCALE).astype(np.int64)
        valid = cv2.imread(view['valid_path'], cv2.IMREAD_GRAYSCALE) > 0
        assert labels.shape == valid.shape == (989, 1320)
        labels[~valid] = 255
        row = {'name': name, 'p3d_max_absolute': error, 'raw_argmax_changes': argmax_changes, 'scores': {}}
        for condition in ('original', 'projected'):
            row['scores'][condition] = {}
            for key in ('p3d', 'probabilities'):
                cm, ce64, ce_log32 = score(data[name, condition][key], labels, valid)
                ref = stored['scores'][condition][key]
                assert np.array_equal(cm, ref['confusion_matrix'])
                assert int(cm.sum()) == ref['valid_pixels']
                max_diff = max(abs(ce64-ref['ce']), abs(ce_log32-ref['ce']))
                assert max_diff < 1e-6  # CPU scoring roundoff check only; not a changed renderer acceptance gate.
                check_iou(iou(cm), ref['iou'])
                pooled[condition, key] += cm
                row['scores'][condition][key] = {'ce_numpy_float64': ce64, 'ce_numpy_log32_sum64': ce_log32,
                    'original_fp32_ce': ref['ce'], 'ce_recompute_max_abs_error': max_diff,
                    'confusion_matrix': cm.tolist(), 'iou': iou(cm)}
        row['correction_counts'] = corrections(a, b, labels, valid)
        assert row['correction_counts'] == stored['correction_counts']
        rows.append(row)
    aggregate = {}
    for (condition, key), cm in pooled.items():
        assert np.array_equal(cm, result['summary'][condition][key]['pooled_confusion_matrix'])
        check_iou(iou(cm), result['summary'][condition][key]['pooled_iou'])
        aggregate.setdefault(condition, {})[key] = {'confusion_matrix': cm.tolist(), 'iou': iou(cm),
            'equal_view_mean_ce_numpy64': sum(r['scores'][condition][key]['ce_numpy_float64'] for r in rows)/2}
    counts = {k: sum(row['correction_counts'][k] for row in rows) for k in rows[0]['correction_counts']}
    # Descriptive localization only, not another selection or acceptance rule.
    oldcm, newcm = pooled['original', 'probabilities'], pooled['projected', 'probabilities']
    changes = {'background_to_cable_false_positive_increase': int(newcm[0, 2]-oldcm[0, 2]),
               'cable_true_positive_change': int(newcm[2, 2]-oldcm[2, 2]),
               'cable_correction_retention_fraction': counts['correction_retained_after_projection']/counts['original_cable_raw_bg_corrected_by_head']}
    for path, expected in paths.items():
        assert sha(path) == expected
    assert not torch.cuda.is_initialized()
    audit = {'status': 'passed_independent_cpu_review', 'plan_sha256': PLAN_SHA, 'execution_receipt_sha256': RECEIPT_SHA,
             'launcher_sha256': sha(RUN/'launch_receipt.json'), 'auditor_sha256': sha(__file__),
             'natural_exit_code': launch['exit_code'], 'outer_seconds': launch['elapsed_seconds'],
             'source_input_import_camera_checkpoint_restore_hashes_match': True,
             'four_npz_hash_shape_dtype_checks': True, 'p3d_full_image_gate_independently_recomputed': True,
             'rgb_depth_alpha_gate': 'runtime captured before/after tensor SHA agree; arrays were not retained, so independent pixel recomputation is unavailable',
             'runtime_contract': '4 scenes/8 rasters/0 backward/0 optimizer and restore flags/modes are persisted worker evidence, not an independent rerender',
             'pixel_scope': {'worker_reads': expected_reads, 'auditor_reads': expected_reads, 'real_rgb_and_val_decodes': 0},
             'rows': rows, 'pooled': aggregate, 'correction_counts': counts, 'descriptive_changes': changes,
             'interpretation': 'Only this frozen head on two TRAIN views: OOD joint null-feature/cross-moment dependency, with retained cable corrections and increased false positives; no information necessity, generalization or retraining upper bound.',
             'new_renders': 0, 'new_backwards': 0, 'new_optimizer_steps': 0, 'cuda_initialized': False}
    with (RUN/'independent_cpu_review.json').open('x') as f:
        json.dump(audit, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'audit': str(RUN/'independent_cpu_review.json'), 'sha256': sha(RUN/'independent_cpu_review.json'),
                      'changes': changes, 'corrections': counts,
                      'max_ce_error': max(s['ce_recompute_max_abs_error'] for row in rows for condition in row['scores'].values() for s in condition.values())}))


if __name__ == '__main__':
    main()
