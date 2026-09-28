"""Independent CPU review for the fixed F semantic teacher candidate."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/mnt/data/SHM2026/runs/f_semantic_teacher_candidate_v1')
F = Path('/mnt/data/SHM2026/runs/ibgs_joint_replacement_v2')
MULTI = Path('/mnt/data/SHM2026/runs/multifield_h3_teacher_v1')


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def iou(matrix):
    diag = np.diagonal(matrix, axis1=-2, axis2=-1).astype(float)
    union = matrix.sum(-1) + matrix.sum(-2) - diag
    return np.divide(diag, union, out=np.full_like(diag, np.nan), where=union > 0)


def main():
    receipt = json.loads((ROOT/'prediction_receipt.json').read_text())
    candidate = json.loads((ROOT/'candidate_metrics.json').read_text())
    plan = json.loads((MULTI/'plan.json').read_text())
    views = {v['name']: v for v in plan['views']}
    names = sorted(views)
    rows = {}
    for name in names:
        view = views[name]
        pred_path = ROOT/'mask'/name
        pred = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        if view['source_annotation_path'] is None:
            continue
        # Annotation JSON is rendered by the project scorer; import it without
        # importing any CUDA module or initializing a GPU.
        sys.path.insert(0, '/mnt/data/SHM2026/runs/matched_rgb_teacher_adaptation_v1/evaluation_v2/source_snapshot')
        import bridge_rgs.official_evaluate as official
        content = Path(view['source_annotation_path']).read_bytes()
        truth = official.rasterize_official_annotation(content, view['camera']['width'], view['camera']['height'])
        keep = truth != 255
        matrix = np.bincount(5*truth[keep].astype(np.int64)+pred[keep].astype(np.int64), minlength=25).reshape(5, 5)
        rows[name] = {'confusion_matrix': matrix.tolist(), 'semantic_pixels': int(keep.sum()),
                      'semantic_ignore_pixels': int((~keep).sum()), 'mask_sha256': sha(pred_path)}
    matrices = np.asarray([rows[n]['confusion_matrix'] for n in sorted(rows)], np.int64)
    pooled = matrices.sum(0); values = iou(pooled)
    checks = {
        'prediction_barrier_50': len(receipt['records']) == 50,
        'annotated_views_41': len(rows) == 41,
        'candidate_confusion_exact': pooled.tolist() == candidate['confusion_matrix'],
        'candidate_iou_exact': np.allclose(values, candidate['iou'], atol=1e-12, rtol=0),
        'all_masks_uint8_class_range': all((cv2.imread(str(ROOT/'mask'/n), cv2.IMREAD_UNCHANGED) < 5).all() for n in names),
        'no_gpu': True,
    }
    report = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
              'recomputed_confusion_matrix': pooled.tolist(), 'recomputed_iou': values.tolist(),
              'candidate_metrics_sha256': sha(ROOT/'candidate_metrics.json'),
              'pair_sha256': sha(ROOT/'paired_candidate_minus_F.json'), 'views': rows}
    (ROOT/'independent_cpu_review.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if report['status'] != 'passed':
        raise SystemExit('independent review failed')
    print(json.dumps({'status': report['status'], 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
