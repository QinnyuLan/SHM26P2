"""CPU-only independent delivered-PNG, 50/41 paired, and system-adoption audit.

No model load or scoring/paired implementation import. Recompute PNG PSNR,
polygon draw-order masks, confusion matrices and paired intervals with NumPy;
SSIM/LPIPS retain the hash-bound existing scoring values, without rescoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

FINGERPRINT = '21a2f19c5d4d703403a0107402d5dd5e006dcd98e5f8e023820c388c3e69a85d'
TEACHER_SHA = '00f5b84ac9a56c39512c5b8e43f70110397923feea1a2b4c78bdffd1f5524bff'
CLASSES = ('background', 'deck', 'stay_cable', 'tower', 'foundation')
RGB_KEYS = ('psnr', 'ssim', 'lpips')
GATE = {'psnr_min_gain_db': .15, 'psnr_paired_95_lower': '>0', 'ssim_min_gain': 0.,
        'lpips_max_gain': 0., 'miou_all_min_gain': -.002, 'stay_cable_iou_min_gain': -.002}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def iou(matrix):
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    union = matrix.sum(-1)+matrix.sum(-2)-diagonal
    return np.divide(diagonal, union, out=np.full_like(diagonal, np.nan, dtype=float), where=union > 0)


def rasterize(content, width, height):
    """Direct draw-order equivalent: known classes 0..4, unknown polygons 255."""
    data = json.loads(content)
    require((data['imageWidth'], data['imageHeight']) == (width, height), 'Annotation grid differs')
    image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(image)
    for shape in data['shapes']:
        require(shape.get('shape_type', 'polygon') == 'polygon' and len(shape['points']) >= 3,
                'Unsupported polygon annotation')
        label = shape['label'].strip()
        draw.polygon([tuple(point) for point in shape['points']], fill=CLASSES.index(label) if label in CLASSES else 255)
    return np.asarray(image, dtype=np.uint8)


def bound_metrics(path):
    path = Path(path)
    metrics, receipt = read(path), read(path.parent/'execution_receipt.json')
    require(receipt['status'] == 'completed' and sha(path) == receipt['official_metrics_sha256'],
            'Incomplete/unbound official metrics')
    for key in ('evaluation_family', 'scoring_protocol', 'official_evaluation_fingerprint', 'inference_protocol'):
        require(metrics[key] == receipt[key], 'Receipt/metrics protocol differs')
    require(metrics['evaluation_family'] == 'official_original_pixel_grid'
            and metrics['scoring_protocol']['id'] == 'official_original_grid_v1'
            and metrics['official_evaluation_fingerprint'] == FINGERPRINT, 'Wrong common 50/41 scoring protocol')
    views = metrics['views']
    require(metrics['validation_views'] == len(views) == len({v['name'] for v in views}) == 50,
            'Require all unique 50 views')
    matrices = []
    for view in views:
        require(view['rgb_pixels'] == view['width']*view['height']
                and all(np.isfinite(view[k]) for k in RGB_KEYS), 'Wrong RGB support/finite score')
        if 'confusion_matrix' in view:
            matrix = np.asarray(view['confusion_matrix'])
            require(matrix.shape == (5, 5) and np.issubdtype(matrix.dtype, np.integer) and (matrix >= 0).all()
                    and matrix.sum() == view['semantic_pixels']
                    and view['semantic_pixels']+view['semantic_ignore_pixels'] == view['rgb_pixels'], 'Invalid semantic support')
            matrices.append(matrix)
    require(metrics['semantic_validation_views'] == len(matrices) == 41
            and np.array_equal(sum(matrices), metrics['confusion_matrix']), 'Wrong 41-view pooled confusion')
    values = iou(sum(matrices))
    require(np.allclose(values, metrics['iou'], atol=1e-12, rtol=0)
            and abs(values.mean()-metrics['miou_all']) < 1e-12
            and abs(values[1:].mean()-metrics['miou_foreground']) < 1e-12, 'Wrong pooled IoU')
    for key in RGB_KEYS:
        require(abs(np.mean([v[key] for v in views])-metrics[key]) < 1e-12, 'Wrong RGB mean')
    return metrics, receipt


def independent_pair(reference, candidate):
    a, b = [{v['name']: v for v in m['views']} for m in (reference, candidate)]
    require(set(a) == set(b), 'Different RGB view populations')
    names = sorted(a)
    semantic = [name for name in names if 'confusion_matrix' in a[name]]
    require(len(names) == 50 and len(semantic) == 41
            and semantic == [name for name in names if 'confusion_matrix' in b[name]], 'Wrong paired 50/41 sets')
    for name in names:
        keys = ('width', 'height', 'rgb_pixels') + (('semantic_pixels', 'semantic_ignore_pixels') if name in semantic else ())
        require(all(a[name][k] == b[name][k] for k in keys), 'Paired grids/support differ')
    rng = np.random.default_rng(20260926)
    counts = np.stack([np.bincount(rng.integers(50, size=50), minlength=50) for _ in range(5000)])
    result = {}
    def add(key, old, new, differences):
        finite = differences[np.isfinite(differences)]
        result[key] = {'reference': float(old), 'candidate': float(new), 'difference': float(new-old),
                       'paired_view_bootstrap_95_interval': np.percentile(finite, [2.5, 97.5], method='linear').tolist(),
                       'finite_bootstrap_replicates': len(finite)}
    for key in RGB_KEYS:
        av, bv = [np.asarray([v[name][key] for name in names]) for v in (a, b)]
        add(key, av.mean(), bv.mean(), counts@(bv-av)/50)
    counts = np.stack([np.bincount(rng.integers(41, size=41), minlength=41) for _ in range(5000)])
    ac, bc = [np.asarray([v[name]['confusion_matrix'] for name in semantic]) for v in (a, b)]
    ai, bi = iou(ac.sum(0)), iou(bc.sum(0))
    ab, bb = iou((counts@ac.reshape(41, 25)).reshape(-1, 5, 5)), iou((counts@bc.reshape(41, 25)).reshape(-1, 5, 5))
    add('miou_all', np.nanmean(ai), np.nanmean(bi), np.nanmean(bb, 1)-np.nanmean(ab, 1))
    add('miou_foreground', np.nanmean(ai[1:]), np.nanmean(bi[1:]), np.nanmean(bb[:, 1:], 1)-np.nanmean(ab[:, 1:], 1))
    for index, name in enumerate(CLASSES):
        add(name+'_iou', ai[index], bi[index], bb[:, index]-ab[:, index])
    return result


def clauses(metrics):
    return {'psnr_gain_at_least_0_15_dB': metrics['psnr']['difference'] >= .15,
            'psnr_paired_95_lower_positive': metrics['psnr']['paired_view_bootstrap_95_interval'][0] > 0,
            'ssim_point_not_lower': metrics['ssim']['difference'] >= 0,
            'lpips_point_not_higher': metrics['lpips']['difference'] <= 0,
            'miou_all_drop_at_most_0_20pp': metrics['miou_all']['difference'] >= -.002,
            'cable_drop_at_most_0_20pp': metrics['stay_cable_iou']['difference'] >= -.002}


def png_audit(metrics, receipt):
    require(receipt['predictions_finished_utc'] <= receipt['source_scoring_started_utc'], 'Source scoring before predictions complete')
    predictions, sources, views = ({v['name']: v for v in rows} for rows in
                                  (receipt['predictions'], receipt['source_records'], metrics['views']))
    require(len(predictions) == len(sources) == 50 and set(predictions) == set(sources) == set(views), 'Incomplete PNG/source sets')
    rgb_hashes, psnr_error, matrices, fingerprint_rows = {}, [], [], []
    for name in sorted(views):
        view, prediction, source = views[name], predictions[name], sources[name]
        for kind in ('rgb', 'mask'):
            require(sha(prediction[kind]) == prediction[kind+'_sha256'], 'Prediction bytes changed')
        rgb = cv2.imread(prediction['rgb'], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(prediction['mask'], cv2.IMREAD_UNCHANGED)
        require(rgb is not None and rgb.dtype == np.uint8 and rgb.shape == (view['height'], view['width'], 3)
                and mask is not None and mask.dtype == np.uint8 and mask.shape == rgb.shape[:2]
                and np.isin(mask, np.arange(5)).all(), 'Invalid delivered PNG arrays')
        require(sha(source['source_image_path']) == source['source_image_sha256'], 'GT RGB changed')
        truth = cv2.imread(source['source_image_path'], cv2.IMREAD_UNCHANGED)
        require(truth is not None and truth.dtype == np.uint8 and truth.shape == rgb.shape, 'GT RGB array mismatch')
        difference = (rgb.astype(np.float32)/255).astype(np.float64)-(truth.astype(np.float32)/255).astype(np.float64)
        psnr = float(-10*np.log10(max(float(np.mean(difference*difference)), 1e-12)))
        error = abs(psnr-view['psnr'])
        require(error < 1e-10, 'Independent PNG PSNR differs')
        psnr_error.append(error)
        rgb_hashes[name] = prediction['rgb_sha256']
        if source['source_annotation_path'] is not None:
            content = Path(source['source_annotation_path']).read_bytes()
            require(hashlib.sha256(content).hexdigest() == source['source_annotation_sha256'], 'GT annotation changed')
            target = rasterize(content, view['width'], view['height'])
            require(hashlib.sha256(target.tobytes()).hexdigest() == source['rasterized_mask_sha256'], 'Independent rasterization differs')
            valid = target != 255
            matrix = np.bincount(target[valid].astype(np.int64)*5+mask[valid], minlength=25).reshape(5, 5)
            require(np.array_equal(matrix, view['confusion_matrix']), 'Independent delivered-mask confusion differs')
            matrices.append(matrix)
        else:
            require('confusion_matrix' not in view, 'Unlabeled view acquired semantic scores')
        fingerprint_rows.append({key: source[key] for key in
                                 ('camera', 'source_image_sha256', 'source_annotation_sha256', 'rasterized_mask_sha256')})
    payload = {'evaluation_family': metrics['evaluation_family'], 'scoring_protocol': metrics['scoring_protocol'],
               'views': sorted(fingerprint_rows, key=lambda x: x['camera']['name'])}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(fingerprint == FINGERPRINT and len(matrices) == 41 and np.array_equal(sum(matrices), metrics['confusion_matrix']),
            'Independent common fingerprint/pooled confusion differs')
    return {'rgb_hashes': rgb_hashes, 'max_psnr_absolute_discrepancy': max(psnr_error),
            'confusion_recomputed_from_41_prediction_pngs_and_original_polygons': True}


def audit(plan_path, expected_training_sha, expected_evaluation_sha, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    require(not output.exists(), 'Preserve any previous audit')
    require(sha(plan_path) == expected_training_sha, 'Wrong immutable training plan')
    plan = read(plan_path)
    run = Path(plan['output'])
    preparation = run/'official_preparation'
    evaluation_path = preparation/'plan.json'
    evaluation, launch = read(evaluation_path), read(preparation/'launch_receipt.json')
    require(sha(evaluation_path) == expected_evaluation_sha == launch['plan_sha256']
            and launch['status'] == 'completed' and launch.get('bound_inputs_postchecked') is True,
            'Evaluation/comparison has not completed')
    require(set(launch['stages']) == {'plain', 'teacher'}
            and all(v['status'] == 'completed' and v['exit_code'] == 0 for v in launch['stages'].values()), 'Missing successful natural evaluation exits')
    require(evaluation['adoption_branch'] == 'teacher' and evaluation['bootstrap_repeats'] == 5000
            and evaluation['bootstrap_seed'] == 20260926 and evaluation['adoption_gate'] == GATE, 'Fixed branch or gate changed')
    for path, expected in evaluation['bound_inputs'].items():
        require(sha(path) == expected, 'Bound evaluation source/input changed: '+path)
    require(evaluation['bound_inputs'][str(plan_path)] == expected_training_sha, 'Evaluation uses another training run')
    training = read(run/'execution_receipt.json')
    require(training['status'] == 'completed' and training['accepted_steps'] > 0
            and sha(run/'final.pt') == training['checkpoint_sha256'], 'No completed changed candidate')
    cv2.setNumThreads(8)
    comparisons, png_results, pair_shas, metric_shas = {}, {}, {}, {}
    for branch in ('plain', 'teacher'):
        reference_path = Path(evaluation['reference_metrics'][branch])
        candidate_path = run/f'official_{branch}'/'official_metrics.json'
        reference, ref_receipt = bound_metrics(reference_path)
        candidate, receipt = bound_metrics(candidate_path)
        require(receipt['checkpoint_sha256'] == training['checkpoint_sha256']
                and Path(receipt['checkpoint']).resolve() == run/'final.pt', 'Prediction from another field')
        if branch == 'teacher':
            teacher = receipt['teacher_ensemble']
            require(teacher['teacher_checkpoint_sha256'] == TEACHER_SHA and teacher['teacher_weight'] == .5
                    and teacher['inference_renderer_sha256'] == training['checkpoint_sha256']
                    and teacher['inference'] == {'tile_size': 768, 'stride': 512, 'flip': True,
                                                 'context_weight': .25, 'context_short_side': 768}, 'Teacher inference protocol changed')
            for path, expected in teacher['input_files_sha256'].items():
                require(sha(path) == expected, 'Teacher input changed')
        else:
            require('teacher_ensemble' not in receipt, 'Plain branch unexpectedly used teacher')
        for item in [receipt['entrypoint_source'], *receipt['loaded_source_modules'].values()]:
            require(sha(item['path']) == item['sha256'], 'Actual evaluated source changed')
        for name, item in receipt['loaded_source_modules'].items():
            require(Path(item['path']).resolve() == Path(plan['source_snapshot'])/(name.replace('.', '/')+'.py'),
                    'Inference/scoring package escaped frozen snapshot')
        pair_path = run/f'paired_official_{branch}.json'
        pair = read(pair_path)
        require(pair['bootstrap_repeats'] == 5000 and pair['seed'] == 20260926
                and pair['difference_direction'] == 'candidate minus reference; LPIPS improves when negative', 'Wrong paired direction/protocol')
        for role, path, r in [('reference', reference_path, ref_receipt), ('candidate', candidate_path, receipt)]:
            binding = pair[role+'_source']
            require(binding['path'] == str(path) and binding['sha256'] == sha(path)
                    and binding['receipt_sha256'] == sha(path.parent/'execution_receipt.json')
                    and binding['checkpoint_sha256'] == r['checkpoint_sha256'], 'Pair endpoint binding differs')
        independent = independent_pair(reference, candidate)
        for key, fields in independent.items():
            for field, value in fields.items():
                require(np.allclose(value, pair['metrics'][key][field], atol=1e-12, rtol=0), 'Independent paired interval differs')
        png_results[branch] = {'candidate': png_audit(candidate, receipt), 'reference': png_audit(reference, ref_receipt)}
        comparisons[branch], pair_shas[branch] = independent, sha(pair_path)
        metric_shas[branch] = {'candidate': sha(candidate_path), 'reference': sha(reference_path)}
    require(png_results['plain']['candidate']['rgb_hashes'] == png_results['teacher']['candidate']['rgb_hashes'],
            'New plain/teacher outputs use different RGB')
    gate_path = run/'system_adoption_gate.json'
    gate = read(gate_path)
    expected = clauses(comparisons['teacher'])
    require(sha(gate_path) == launch['system_adoption_gate_sha256'] and gate['clauses'] == expected
            and gate['passed'] == all(expected.values()) and gate['plain_branch_is_descriptive_only'] is True
            and gate['fifty_rgb_pngs_identical'] is True and gate['evaluation_plan_sha256'] == expected_evaluation_sha
            and gate['candidate_metrics_sha256'] == metric_shas['teacher']['candidate'], 'Independent system adoption gate differs')
    for key, fields in comparisons['teacher'].items():
        for field, value in fields.items():
            require(np.allclose(value, gate['metrics'][key][field], atol=1e-12, rtol=0), 'Gate uses different metric evidence')
    result = {'status': 'passed', 'cpu_only': True, 'auditor_sha256': sha(__file__),
              'training_plan_sha256': expected_training_sha, 'evaluation_plan_sha256': expected_evaluation_sha,
              'evaluation_launch_sha256': sha(preparation/'launch_receipt.json'), 'checkpoint_sha256': training['checkpoint_sha256'],
              'metrics_sha256': metric_shas, 'pair_sha256': pair_shas, 'paired': comparisons,
              'png_checks': png_results, 'system_adoption_gate': {'passed': all(expected.values()), 'clauses': expected},
              'scoring_scope': 'PNG PSNR and semantic confusion recomputed. Existing SSIM/LPIPS not rescored; source and metric bytes bound.',
              'statistical_scope': 'Paired fixed development views, not selection-adjusted, independent seeds, exact peer Dev30 or innovation evidence.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-plan', required=True, type=Path)
    parser.add_argument('--expected-training-plan-sha256', required=True)
    parser.add_argument('--expected-evaluation-plan-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.training_plan, args.expected_training_plan_sha256, args.expected_evaluation_plan_sha256, args.output)
    print(json.dumps({'status': result['status'], 'gate': result['system_adoption_gate'], 'output_sha256': sha(args.output)}))
