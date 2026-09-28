"""Only thin two-route auditor schema changes; no run outputs or target pixels."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('scene_geometry_evaluation_audit', SCRIPTS/'audit_scene_geometry_evaluation.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def test_two_route_population_rejects_duplicate_wrongarm_and_stale_teacher():
    names = [f'{i:03d}.png' for i in range(50)]
    rows = [{'arm': arm, 'name': name, 'masks': dict.fromkeys(audit.KINDS),
             'soft': dict.fromkeys(('raw', 'scene', 'teacher')), 'teacher_recomputed': True}
            for arm in ('full', 'prior_only') for name in names]
    assert len(audit.prediction_population(rows, names)) == 100
    for kind in ('duplicate', 'wrongarm', 'teacher'):
        broken = copy.deepcopy(rows)
        if kind == 'duplicate':
            broken[-1] = broken[0]
        elif kind == 'wrongarm':
            broken[-1]['arm'] = 'rgb'
        else:
            broken[-1]['teacher_recomputed'] = False
        with pytest.raises(AssertionError, match='Complete100'):
            audit.prediction_population(broken, names)


def test_eleven_fixed_pair_directions_and_two_arm_counts():
    expected = {('full', 'baseline', k) for k in ('raw', 'scene', 'joint')}
    expected |= {('prior_only', 'baseline', k) for k in ('raw', 'scene', 'joint')}
    expected |= {('prior_only', 'full', k) for k in ('raw', 'scene', 'joint')}
    expected |= {('full', 'E', 'joint'), ('prior_only', 'E', 'joint')}
    actual = audit.pair_definitions()
    assert len(actual) == len(set(actual)) == 11 and set(actual) == expected
    assert audit.COUNTS == {'scene_calls': 100, 'direct_q_shader_calls': 100,
        'head_calls': 100, 'teacher_predict_calls': 100, 'teacher_backbone_forwards': 1400,
        'native_gsplat_high_level_calls': 200, 'native_rasterize_to_pixels_calls': 300,
        'GT_RGB_payload_reads': 50, 'annotation_payload_reads': 41,
        'RGB_scoring_calls': 100, 'confusion_matrices': 246}
