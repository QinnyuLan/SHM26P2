"""Synthetic corruption contracts; never open an experiment output."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'independent_simplex_audit', Path(__file__).parents[1]/'scripts/audit_raw_simplex_result.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    def full(phase, objective, digest):
        score = phase in ('baseline', 'final')
        return {'phase': phase, 'complete': True, 'views': 2,
                'rows': [{'name': name, 'objective': objective} for name in ('a', 'b')],
                'objective': objective, 'q32_sha256': digest,
                'counts': {'scene': 2, 'gsplat': 4, 'shader': 2, 'vjp': 0 if phase == 'trial' else 2,
                           'head': 4 if score else 0, 'target_decodes': 4 if phase == 'baseline' else 0,
                           'complete_passes': 1, 'partial_pass_views': 0}}
    passes = [full('baseline', 1., 'old'), full('trial', .9, 'new'), full('final', .9, 'new')]
    spec = {'max_full_passes': 80, 'max_accepted': 1, 'max_backtracks': 8, 'armijo': 1e-4}
    row = {'accepted_updates_before': 0, 'trial_index': 0, 'eta': 1.,
           'actual_gradient_dot_displacement': -.2, 'trial_objective': .9,
           'armijo_bound': 1.-2e-5, 'full_objective_decrease': 1.-.9,
           'accepted': True, 'complete_passes': 2}
    analysis = {'baseline': copy.deepcopy(passes[0]),
                'endpoint': {**copy.deepcopy(passes[-1]), 'gap': {}}, 'history': [row],
                'accepted_updates': 1, 'complete_passes': 3,
                'convergence_certified': False, 'stop_reason': 'accepted_budget'}
    return passes, analysis, spec


def test_trace_acceptance_and_complete_counts():
    passes, analysis, spec = fixture()
    totals, accepted = m.check_trace(passes, analysis, ['a', 'b'], spec)
    assert totals['scene'] == 6 and totals['vjp'] == 4 and totals['head'] == 8
    assert accepted[0]['after'] < accepted[0]['before']


@pytest.mark.parametrize('broken', ['names', 'count', 'armijo', 'acceptance', 'q', 'convergence'])
def test_corrupted_trace_rejected(broken):
    passes, analysis, spec = fixture()
    if broken == 'names':
        passes[1]['rows'].reverse()
    elif broken == 'count':
        passes[1]['counts']['vjp'] = 2
    elif broken == 'armijo':
        analysis['history'][0]['armijo_bound'] = 2.
    elif broken == 'acceptance':
        analysis['history'][0]['accepted'] = False
    elif broken == 'q':
        passes[-1]['q32_sha256'] = 'wrong'
        analysis['endpoint']['q32_sha256'] = 'wrong'
    else:
        analysis['convergence_certified'] = True
    with pytest.raises(ValueError):
        m.check_trace(passes, analysis, ['a', 'b'], spec)


def test_master_and_actual_renderer_cast_checked_separately():
    master = np.array([[.1, .2, .3, .15, .25]], np.float64)
    renderer = master.astype(np.float32)
    assert m.check_q(master, renderer)['shape'] == [1, 5]
    renderer[0, 0] = np.nextafter(renderer[0, 0], np.float32(1))
    with pytest.raises(ValueError, match='FP32 cast'):
        m.check_q(master, renderer)
    with pytest.raises(ValueError, match='simplex'):
        m.check_q(master*.9, (master*.9).astype(np.float32))


def test_confusion_support_and_missing_classes():
    cm = np.array([[2, 1, 0, 0, 0], [0, 3, 0, 0, 0], [0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
    _, iou, mean = m.cm_metrics(cm)
    assert iou == [2/3, 3/4, None, 1., None]
    record = {'confusion_matrix': cm.tolist(), 'iou': iou, 'miou_all': mean}
    m.check_metrics(record, cm)
    record['iou'][0] = .9
    with pytest.raises(ValueError):
        m.check_metrics(record, cm)
