"""Saved-evidence corruption contracts; synthetic arrays only."""
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
path = Path(__file__).resolve().parents[1]/'scripts/audit_continuous_geometry_evaluation.py'
spec = importlib.util.spec_from_file_location('geometry_evaluation_audit', path)
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def fixture():
    views = [{'name': f'{i:03d}.png', 'camera': {'width': 3, 'height': 2},
              'source_annotation_path': 'synthetic' if i < 41 else None} for i in range(50)]
    matrix = np.diag([2, 1, 1, 1, 1])
    rows = [{'name': v['name'], 'width': 3, 'height': 2, 'rgb_pixels': 6,
             'psnr': 20., 'ssim': .8, 'lpips': .2,
             **({'confusion_matrix': matrix.tolist(), 'semantic_pixels': 6, 'semantic_ignore_pixels': 0}
                if v['source_annotation_path'] else {})} for v in views]
    value = {'views': rows, 'confusion_matrix': (41*matrix).tolist(), 'iou': [1.]*5,
             'miou_all': 1., 'miou_foreground': 1., 'psnr': 20., 'ssim': .8, 'lpips': .2}
    return views, value


def test_saved_metric_reaggregation_rejects_denominator_and_rgb_corruption():
    views, reference = fixture()
    assert audit.saved_metrics(reference, views, reference, audit.common.Audit())['miou_all'] == 1.
    for mutation in ('support', 'rgb', 'pooled'):
        changed = copy.deepcopy(reference)
        if mutation == 'support':
            changed['views'][0]['confusion_matrix'][0][0] += 1
        elif mutation == 'rgb':
            changed['psnr'] += .01
        else:
            changed['confusion_matrix'][0][1] += 1
        with pytest.raises(AssertionError):
            audit.saved_metrics(changed, views, reference, audit.common.Audit())


def test_prediction_barrier_rejects_duplicate_and_unproven_baseline():
    views, _ = fixture()
    records = [{'arm': arm, 'name': v['name'], 'masks': dict.fromkeys(audit.KINDS),
                'soft': dict.fromkeys(('raw', 'scene', 'teacher')), 'teacher_recomputed': True}
               for arm in audit.ARMS for v in views]
    identity = [{'name': v['name'], 'canvas_rgb_exact': True, 'original_rgb_exact': True,
                 'teacher_soft_exact': True, 'mask_exact': dict.fromkeys(audit.KINDS, True)} for v in views]
    assert len(audit.population(records, views, identity)) == 200
    with pytest.raises(AssertionError, match='Complete200'):
        audit.population(records[:-1]+[records[0]], views, identity)
    identity[0]['teacher_soft_exact'] = False
    with pytest.raises(AssertionError, match='baseline identity'):
        audit.population(records, views, identity)


def test_reused_independent_bootstrap_direction_and_constant_shift():
    _, reference = fixture(); candidate = copy.deepcopy(reference)
    for row in candidate['views']:
        row['psnr'] += 2.
    pairs = audit.common.independent_bootstrap(reference, candidate)
    assert pairs['psnr']['difference'] == 2.
    assert pairs['psnr']['paired_view_bootstrap_95_interval'] == [2., 2.]
    assert pairs['miou_all']['difference'] == 0.
    assert pairs['miou_all']['paired_view_bootstrap_95_interval'] == [0., 0.]
