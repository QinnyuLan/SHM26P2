"""Independent CPU review: delivered pixels, direct polygon labels and paired CIs."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def iou(cm):
    diagonal = np.diagonal(cm, axis1=-2, axis2=-1)
    union = cm.sum(-1)+cm.sum(-2)-diagonal
    return np.divide(diagonal, union, out=np.full_like(diagonal, np.nan, dtype=float), where=union > 0)


def pixels(path, mode=None):
    with Image.open(path) as image:
        return np.array(image.convert(mode) if mode else image)


def review(run):
    started = time.perf_counter(); run = Path(run)
    plan = read(run/'plan.json'); execution = read(run/'score_execution_receipt.json')
    launch = read(run/'score_launch_receipt.json')
    assert execution['status'] == launch['status'] == 'completed' and launch['exit_code'] == 0
    assert launch['natural_completion'] and launch['execution_receipt_sha256'] == sha(run/'score_execution_receipt.json')
    assert execution['plan_sha256'] == launch['plan_sha256'] == sha(run/'plan.json')
    assert execution['official_metrics_sha256'] == sha(run/'official_metrics.json')
    rendered = read(run/'render/execution_receipt.json'); preds = {r['name']: r for r in rendered['predictions']}
    assert len(preds) == 52 and all(sha(r[k]) == r[k+'_sha256'] for r in preds.values() for k in ('rgb', 'mask'))
    metrics = read(run/'official_metrics.json'); rows = {r['name']: r for r in metrics['views']}
    errors = []; semantic_count = 0
    for view in plan['views']:
        name = view['name']; row = rows[name]; item = preds[name]
        assert sha(view['source_image_path']) == view['source_image_sha256']
        predicted, target = pixels(item['rgb'], 'RGB'), pixels(view['source_image_path'], 'RGB')
        a, b = [(p.astype(np.float32)/255).astype(np.float64) for p in (predicted, target)]
        psnr = -10*np.log10(max(np.square(a-b).sum()/a.size, 1e-12))
        errors.append(abs(float(psnr)-row['psnr']))
        if view['source_annotation_path'] is not None:
            semantic_count += 1
            assert sha(view['source_annotation_path']) == view['source_annotation_sha256']
            annotation = read(view['source_annotation_path'])
            labels = ['background', 'deck', 'stay_cable', 'tower', 'foundation']
            canvas = Image.new('L', (annotation['imageWidth'], annotation['imageHeight']), 0)
            draw = ImageDraw.Draw(canvas)
            # One direct draw with unknown=255; independent of official two-pass masking.
            for shape in annotation['shapes']:
                assert shape.get('shape_type', 'polygon') == 'polygon' and len(shape['points']) >= 3
                label = shape['label'].strip(); value = labels.index(label) if label in labels else 255
                draw.polygon([tuple(p) for p in shape['points']], fill=value)
            truth = np.array(canvas)
            assert hashlib.sha256(truth.tobytes()).hexdigest() == view['rasterized_mask_sha256']
            mask = pixels(item['mask']); keep = truth != 255
            cm = np.zeros(25, dtype=np.int64)
            codes, counts = np.unique(truth[keep].astype(np.int64)*5+mask[keep], return_counts=True)
            cm[codes] = counts
            assert cm.reshape(5, 5).tolist() == row['confusion_matrix']
    assert semantic_count == 41 and len(rows) == 50
    names = sorted(rows); sem_names = [n for n in names if 'confusion_matrix' in rows[n]]
    rng = np.random.default_rng(20260926)
    draws = rng.integers(0, 50, (5000, 50))
    rgb_counts = np.stack([np.bincount(d, minlength=50)/50 for d in draws])
    draws = rng.integers(0, 41, (5000, 41))
    sem_counts = np.stack([np.bincount(d, minlength=41) for d in draws])
    for key in ('psnr', 'ssim', 'lpips'):
        errors.append(abs(np.mean([rows[n][key] for n in names])-metrics[key]))
    matrices = np.array([rows[n]['confusion_matrix'] for n in sem_names])
    assert matrices.sum(0).tolist() == metrics['confusion_matrix']
    errors.extend(np.abs(iou(matrices.sum(0))-metrics['iou']))
    new_iou = iou((sem_counts@matrices.reshape(41, 25)).reshape(5000, 5, 5))
    for role, reference in plan['references'].items():
        assert sha(reference) == plan['frozen_inputs'][reference]
        ref = {r['name']: r for r in read(reference)['views']}
        pair = read(run/f'paired_minus_{role}.json')['metrics']
        for key in ('psnr', 'ssim', 'lpips'):
            delta = np.array([rows[n][key]-ref[n][key] for n in names])
            errors.append(abs(delta.mean()-pair[key]['difference']))
            errors.extend(np.abs(np.quantile(rgb_counts@delta, [.025, .975])-pair[key]['paired_view_bootstrap_95_interval']))
        old_cm = np.array([ref[n]['confusion_matrix'] for n in sem_names])
        old_iou = iou((sem_counts@old_cm.reshape(41, 25)).reshape(5000, 5, 5))
        point = iou(matrices.sum(0))-iou(old_cm.sum(0))
        delta = new_iou-old_iou
        for key, columns in [('miou_all', list(range(5))), ('miou_foreground', [1, 2, 3, 4])]+[
                (name+'_iou', [i]) for i, name in enumerate(['background', 'deck', 'stay_cable', 'tower', 'foundation'])]:
            errors.append(abs(np.nanmean(point[columns])-pair[key]['difference']))
            interval = np.quantile(np.nanmean(delta[:, columns], 1), [.025, .975])
            errors.extend(np.abs(interval-pair[key]['paired_view_bootstrap_95_interval']))
    assert max(errors) < 1e-12
    result = {'status': 'passed', 'plan_sha256': sha(run/'plan.json'), 'worker_sha256': sha(__file__),
        'score_receipt_sha256': sha(run/'score_execution_receipt.json'), 'RGB_GT_reads': 50, 'annotation_reads': 41,
        'independent_direct_draw_masks_exact': 41, 'independent_confusions_exact': 41,
        'max_absolute_error': float(max(errors)), 'gpu_calls': 0, 'elapsed_seconds': time.perf_counter()-started,
        'scope': 'CPU pixel PSNR, independently drawn labels/confusions, saved perceptual means and paired statistics; no SSIM/LPIPS pixel rescoring'}
    with (run/'independent_cpu_review.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    review(parser.parse_args().run)
