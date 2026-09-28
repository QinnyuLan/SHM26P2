"""Score fresh camera-only joint exports after a natural completed render barrier."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def score(path):
    start = time.monotonic(); plan = read(path); output = Path(plan['output'])
    run = Path(plan['render_output']); rendered = read(run/'execution_receipt.json')
    launch = read(output/'render_launch_receipt.json')
    require(rendered['status'] == launch['status'] == 'completed' and launch['exit_code'] == 0
            and launch['natural_completion'] and launch['execution_receipt_sha256'] == sha(run/'execution_receipt.json')
            and launch['plan_sha256'] == sha(path) and rendered['bundle_sha256'] == plan['bundle_sha256']
            and rendered['camera_json_sha256'] == plan['cameras_sha256'], 'Natural bound rendering completion required')
    require(sha(__file__) == plan['frozen_inputs'][str(Path(__file__).resolve())], 'Frozen scorer required')
    require(all(sha(p) == h for p, h in plan['frozen_inputs'].items()), 'Changed frozen input')
    snapshot = Path(plan['scoring_snapshot'])
    require(all(sha(snapshot/p) == h for p, h in plan['scoring_source_hashes'].items()), 'Changed scoring source')
    sys.path.insert(0, str(snapshot))
    official = importlib.import_module('bridge_rgs.official_evaluate')
    compare = importlib.import_module('compare_official_evaluations')
    records = {r['name']: r for r in rendered['predictions']}
    expected = {v['name'] for v in plan['views']} | {'365.png', 'novel_shifted.png'}
    require(len(records) == len(rendered['predictions']) == 52 and set(records) == expected,
            'Complete unique 52-camera output barrier required')
    require(rendered['scene_renders'] == 156 and rendered['teacher_calls'] == rendered['selector_calls'] == 52,
            'Unexpected inference counts')
    require(all(sha(r[k]) == r[k+'_sha256'] for r in records.values() for k in ('rgb', 'mask')), 'Output changed')
    identity = {'RGB_exact': [], 'mask_exact': []}
    for name, reference in plan['reference_predictions'].items():
        for key, role in (('rgb', 'rgb'), ('mask', 'semantic')):
            old = reference[role]
            require(sha(old[key]) == old[key+'_sha256'] == records[name][key+'_sha256'],
                    f'Fresh {key} differs for {name}')
            identity['RGB_exact' if key == 'rgb' else 'mask_exact'].append(name)
    for key in ('rgb', 'mask'):
        require(records['365.png'][key+'_sha256'] == records['001.png'][key+'_sha256'], 'Rename changes output')
        require(records['novel_shifted.png'][key+'_sha256'] != records['001.png'][key+'_sha256'],
                'Novel pose failed to change output')
    for item in records.values():
        mask = cv2.imread(item['mask'], cv2.IMREAD_UNCHANGED)
        require(mask.shape == (989, 1320) and mask.dtype == np.uint8 and (mask < 5).all(), 'Invalid ID mask')
    write(output/'joint_prediction_review.json', {'status': 'passed', 'plan_sha256': sha(path), **identity,
        'renamed_rgb_and_mask_exact': True, 'novel_pose_changed_both_outputs': True, 'gt_reads_before_review': 0})
    report = {'status': 'running', 'plan_sha256': sha(path), 'worker_sha256': sha(__file__),
        'RGB_GT_reads': 0, 'annotation_reads': 0, 'lpips_calls': 0, 'scene_renders': 0, 'teacher_calls': 0,
        'render_receipt_sha256': sha(run/'execution_receipt.json'), 'prediction_review_sha256': sha(output/'joint_prediction_review.json')}
    try:
        flags = plan['scoring_numerics']
        torch.backends.cudnn.allow_tf32 = flags['cudnn_tf32']; torch.backends.cudnn.benchmark = flags['benchmark']
        torch.set_float32_matmul_precision(flags['matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32 = flags['matmul_tf32']
        torch.set_num_threads(8); cv2.setNumThreads(8)
        torch.cuda.reset_peak_memory_stats(); perceptual = official._lpips('cuda'); rows = []
        for view in plan['views']:
            name = view['name']; camera = view['camera']; item = records[name]
            content = Path(view['source_image_path']).read_bytes(); report['RGB_GT_reads'] += 1
            require(hashlib.sha256(content).hexdigest() == view['source_image_sha256'], 'GT RGB changed')
            target = official._decode_rgb(content, camera['width'], camera['height'], 'target')
            prediction = official._decode_rgb(Path(item['rgb']).read_bytes(), camera['width'], camera['height'], 'prediction')
            truth = None
            if view['source_annotation_path'] is not None:
                content = Path(view['source_annotation_path']).read_bytes(); report['annotation_reads'] += 1
                require(hashlib.sha256(content).hexdigest() == view['source_annotation_sha256'], 'Annotation changed')
                truth = official.rasterize_official_annotation(content, camera['width'], camera['height'])
                require(hashlib.sha256(truth.tobytes()).hexdigest() == view['rasterized_mask_sha256'], 'Rasterizer differs')
            mask = cv2.imread(item['mask'], cv2.IMREAD_UNCHANGED)
            value = official.score_official_arrays(prediction, mask, target, truth, perceptual, 'cuda')
            report['lpips_calls'] += 1
            rows.append({'name': name, 'width': camera['width'], 'height': camera['height'],
                         'rgb_sha256': item['rgb_sha256'], 'mask_sha256': item['mask_sha256'], **value})
        fingerprint = official.official_fingerprint([{k: v[k] for k in
            ('camera', 'source_image_sha256', 'source_annotation_sha256', 'rasterized_mask_sha256')} for v in plan['views']])
        require(fingerprint == plan['reference_fingerprint'] and report['RGB_GT_reads'] == report['lpips_calls'] == 50
                and report['annotation_reads'] == 41, 'Wrong scoring population')
        matrix = sum(np.array(r['confusion_matrix'], np.int64) for r in rows if 'confusion_matrix' in r)
        iou = compare._iou(matrix)
        metrics = {'evaluation_family': official.FAMILY, 'official_evaluation_fingerprint': fingerprint,
            'scoring_protocol': official.SCORING_PROTOCOL, 'validation_views': 50, 'semantic_validation_views': 41,
            'views': rows, 'confusion_matrix': matrix.tolist(), 'iou': iou.tolist(),
            'miou_all': float(np.nanmean(iou)), 'miou_foreground': float(np.nanmean(iou[1:])),
            **{k: float(np.mean([r[k] for r in rows])) for k in ('psnr', 'ssim', 'lpips')},
            'scope': 'Fresh three-field camera-only engineering export; repeated development views; no semantic gain over E'}
        compare._validate(metrics); write(output/'official_metrics.json', metrics)
        pairs = {}
        for role, ref in plan['references'].items():
            pair = compare.paired_official_comparison(read(ref), metrics, repeats=5000, seed=20260926)
            write(output/f'paired_minus_{role}.json', pair); pairs[role] = pair
        old = read(plan['references']['E'])
        require(metrics['confusion_matrix'] == old['confusion_matrix'], 'New semantic confusion differs from E')
        gate = pairs['E']['metrics']
        clauses = {'psnr_plus_0_15': gate['psnr']['difference'] >= .15,
                   'psnr_paired_lower_positive': gate['psnr']['paired_view_bootstrap_95_interval'][0] > 0,
                   'ssim_not_lower': gate['ssim']['difference'] >= 0,
                   'lpips_not_higher': gate['lpips']['difference'] <= 0,
                   'semantic_exact': metrics['confusion_matrix'] == old['confusion_matrix']}
        write(output/'joint_gate.json', {'passed': all(clauses.values()), 'clauses': clauses,
            'scope': 'Engineering adoption conditions; not research novelty or blind-test evidence'})
        report.update(status='completed', official_metrics_sha256=sha(output/'official_metrics.json'),
                      gate_sha256=sha(output/'joint_gate.json'), gate_passed=all(clauses.values()))
        print(json.dumps({k: metrics[k] for k in ('psnr', 'ssim', 'lpips', 'miou_all', 'miou_foreground', 'iou')}))
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-start
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        write(output/'score_execution_receipt.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    score(parser.parse_args().plan)
