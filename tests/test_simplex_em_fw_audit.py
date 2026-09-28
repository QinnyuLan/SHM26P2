import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_simplex_em_fw as audit


def fixture():
    spec = {'max_full_passes': 26, 'em_blocks': 1, 'em_slots_per_block': 1,
            'gap_tolerance': 1e-5, 'absolute_tolerance': 2e-7,
            'relative_tolerance': 5e-4, 'signal_multiplier': 10}
    gap = {'directional': .1, 'centered': .1}
    def point(phase, objective, identity):
        return {'phase': phase, 'objective': objective, 'q32_sha256': identity, 'gap': gap.copy()}
    passes = [point('baseline', 1., 'a'), point('em_0_0', 1.1, 'b'),
              point('fw_vertex_0', 2., 'v'), point('fw_candidate_0', .95, 'c'), point('final', .95, 'c')]
    comparison = {'left': -.2, 'right': -.2, 'absolute_error': 0.,
                  'tolerance': 2e-7+5e-4*.2, 'passed': True}
    line = {'passed': True, 'A': -.2, 'cached_objective': .95,
            'cache_at_zero': {'B64': -.2, 'B32upstream': -.2},
            'direction_gate': {'passed': True, 'comparisons': {
                'A_vs_B32upstream': comparison.copy(), 'B32upstream_vs_B64': comparison.copy()}}}
    report = {'history': [
        {'kind': 'em', 'block': 0, 'slot': 0, 'base_objective': 1., 'gap_at_current': gap.copy(),
         'gap_at_trial': gap.copy(), 'trial_objective': 1.1, 'accepted': False,
         'reason': 'nondecrease_skip_remaining_em_slots'},
        {'kind': 'fw', 'block': 0, 'base_objective': 1., 'gap_at_current': gap.copy(), 'line': line,
         'gap_at_trial': gap.copy(), 'trial_objective': .95, 'predicted_decrease': .05,
         'actual_decrease': .05, 'agreement_tolerance': 2e-7+5e-4*.05,
         'accepted': True, 'reason': 'actual_decrease_confirmed'}],
        'final_repeat_objective_difference': 0., 'stop_reason': 'fixed_two_block_schedule_finished',
        'accepted_updates': {'em': 0, 'fw': 1}, 'complete_passes': 5,
        'convergence_certified': False, 'performance_adoption': False}
    return passes, report, spec


def test_rejected_em_retains_current_for_one_confirmed_fw():
    passes, report, spec = fixture()
    accepted = audit.check_control(passes, report, spec)
    assert accepted == [{'kind': 'fw', 'block': 0, 'before': 1., 'after': .95, 'q32_sha256': 'c'}]


@pytest.mark.parametrize('corruption', ['false_decrease', 'wrong_final', 'extra_pass', 'false_convergence'])
def test_control_rejects_corrupt_record(corruption):
    passes, report, spec = fixture()
    if corruption == 'false_decrease':
        passes[3]['objective'] = 1.01
    elif corruption == 'wrong_final':
        passes[-1]['q32_sha256'] = 'rejected'
    elif corruption == 'extra_pass':
        passes.insert(-1, copy.deepcopy(passes[2])); report['complete_passes'] += 1
    else:
        report['convergence_certified'] = True
    with pytest.raises(ValueError):
        audit.check_control(passes, report, spec)


def test_bad_complete_mean_or_q_cache_cannot_pass_log_check():
    gap = {'directional': .1, 'centered': .1}
    counts = {'scene': 1, 'gsplat': 2, 'shader': 1, 'vjp': 1, 'head': 2,
              'target_decodes': 2, 'complete_passes': 1, 'partial_pass_views': 0}
    baseline = {'phase': 'baseline', 'complete': True, 'views': 1,
                'q32_sha256': 'a', 'rows': [{'name': 'TRAIN', 'objective': 1.}],
                'objective': 1., 'counts': counts, 'metrics': {}}
    final = copy.deepcopy(baseline); final['phase'] = 'final'; final['counts']['target_decodes'] = 0
    log = [copy.deepcopy(baseline), copy.deepcopy(final)]
    log[0]['target_cache'] = {'q32_sha256': 'a'}
    baseline['gap'] = gap.copy(); final['gap'] = gap.copy()
    analysis = {'passes': [baseline, final], 'baseline': baseline, 'endpoint': final}
    audit.check_passes(log, analysis, ['TRAIN'])
    bad_log = copy.deepcopy(log); bad_log[0]['target_cache']['q32_sha256'] = 'other'
    with pytest.raises(ValueError, match='Cache q identity'):
        audit.check_passes(bad_log, analysis, ['TRAIN'])
    baseline['objective'] = log[0]['objective'] = .9
    with pytest.raises(ValueError, match='Complete F mean'):
        audit.check_passes(log, analysis, ['TRAIN'])
