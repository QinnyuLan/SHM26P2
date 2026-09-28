import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PATH = HERE/'optimize_simplex_em_fw.py'
if not PATH.exists():
    PATH = HERE.parent/'scripts/optimize_simplex_em_fw.py'
spec = importlib.util.spec_from_file_location('simplex_em_fw', PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class LinearCallbacks:
    """Tiny deterministic callback fixture tests control flow, not EM benefit."""
    def __init__(self, *, reject_em=False, fail_fw=False, partial=False):
        self.calls, self.released, self.accepted = [], [], []
        self.reject_em, self.fail_fw, self.partial = reject_em, fail_fw, partial
        self.coefficients = np.array([[1., 1.01]])

    def objective(self, q):
        return float(2-np.sum(q.astype(np.float32).astype(np.float64)*self.coefficients))

    def full(self, q, *, gradient, score, phase, cache_targets):
        self.calls.append((phase, gradient, score, q.copy()))
        result = {'complete': not (self.partial and phase.startswith('em_')),
                  'objective': self.objective(q), 'phase': phase, 'q32_sha256': worker.q32_sha(q)}
        if self.reject_em and phase.startswith('em_'):
            result['objective'] += 1
        if gradient:
            result['gradient'] = -self.coefficients.copy()
        if cache_targets:
            result['target_cache'] = {'phase': phase, 'q32_sha256': worker.q32_sha(q)}
        return result

    def fw(self, q, current, vertex, vertex_result):
        assert current['target_cache']['phase'] not in self.released
        if self.fail_fw:
            return None, {'passed': False}
        candidate = .9*q+.1*vertex
        return candidate, {'passed': True, 'cached_objective': self.objective(candidate)}

    def release(self, result):
        key = result['target_cache']['phase']
        assert key not in self.released
        self.released.append(key)

    def accept(self, q, row):
        self.accepted.append((q.copy(), row['kind']))

    def run(self, q=None):
        return worker.solve_em_fw(np.array([[.8, .2]]) if q is None else q,
                                  self.full, self.fw, release_cache=self.release, on_accept=self.accept)


def test_fixed_schedule_max26_passes_full_gradient_at_every_candidate_and_strict_json():
    callbacks = LinearCallbacks()
    q, report = callbacks.run()
    assert report['complete_passes'] == len(callbacks.calls) == 26
    assert report['accepted_updates'] == {'em': 20, 'fw': 2}
    assert sum(c[1] for c in callbacks.calls) == 24
    assert [c[0] for c in callbacks.calls if c[2]] == ['baseline', 'final']
    assert q.dtype == np.float64
    assert all('gap' in p for p in report['passes'] if not p['phase'].startswith('fw_vertex'))
    assert report['endpoint']['objective'] < report['baseline']['objective']
    assert not report['convergence_certified'] and not report['performance_adoption']
    json.dumps(report, allow_nan=False)


def test_rejected_em_skips_block_uses_last_accepted_gradient_and_cache():
    callbacks = LinearCallbacks(reject_em=True)
    q, report = callbacks.run()
    assert report['complete_passes'] == 8
    assert report['accepted_updates'] == {'em': 0, 'fw': 2}
    assert [h['kind'] for h in report['history']] == ['em', 'fw', 'em', 'fw']
    assert all(h['reason'] == 'nondecrease_skip_remaining_em_slots'
               for h in report['history'] if h['kind'] == 'em')
    np.testing.assert_allclose(q, [[.648, .352]])


def test_failed_fw_ends_entire_schedule_and_keeps_last_accepted():
    callbacks = LinearCallbacks(reject_em=True, fail_fw=True)
    q, report = callbacks.run()
    assert report['complete_passes'] == 4
    assert report['accepted_updates'] == {'em': 0, 'fw': 0}
    assert report['stop_reason'] == 'fw_stopped_fixed_direction_gate_not_met'
    np.testing.assert_array_equal(q, [[.8, .2]])


def test_incomplete_em_pass_is_fatal_and_never_reaches_fw():
    callbacks = LinearCallbacks(partial=True)
    with pytest.raises(ValueError, match='Incomplete'):
        callbacks.run()
    assert [c[0] for c in callbacks.calls] == ['baseline', 'em_0_0']
    assert not callbacks.accepted


def test_zero_locked_em_moves_to_scheduled_fw_without_useless_em_render():
    callbacks = LinearCallbacks(fail_fw=True)
    q, report = callbacks.run(np.array([[1., 0.]]))
    assert [c[0] for c in callbacks.calls] == ['baseline', 'fw_vertex_0', 'final']
    assert report['history'][0]['reason'] == 'fp32_stagnation_skip_remaining_em_slots'
    np.testing.assert_array_equal(q, [[1., 0.]])


@pytest.mark.parametrize('bad_kind', ['nondecrease', 'cache_mismatch'])
def test_fw_actual_failure_not_committed_or_retried(bad_kind):
    callbacks = LinearCallbacks(reject_em=True)
    original_full, original_fw = callbacks.full, callbacks.fw
    def full(q, **kwargs):
        result = original_full(q, **kwargs)
        if kwargs['phase'].startswith('fw_candidate') and bad_kind == 'nondecrease':
            result['objective'] += 1
        return result
    def fw(*args):
        candidate, metadata = original_fw(*args)
        if bad_kind == 'cache_mismatch':
            metadata['cached_objective'] -= .01
        return candidate, metadata
    callbacks.full, callbacks.fw = full, fw
    q, report = callbacks.run()
    assert report['stop_reason'] == 'fw_stopped_actual_candidate_not_confirmed'
    assert report['complete_passes'] == 5
    assert not callbacks.accepted
    np.testing.assert_array_equal(q, [[.8, .2]])


@pytest.mark.parametrize('where', ['pass', 'cache'])
def test_q32_identity_mismatch_rejected_before_any_step(where):
    callbacks = LinearCallbacks()
    original = callbacks.full
    def bad(q, **kwargs):
        result = original(q, **kwargs)
        if where == 'pass':
            result['q32_sha256'] = 'wrong'
        else:
            result['target_cache']['q32_sha256'] = 'wrong'
        return result
    callbacks.full = bad
    with pytest.raises(ValueError, match='identity'):
        callbacks.run()
    assert len(callbacks.calls) == 1 and not callbacks.accepted


def test_direct_q_hook_preserves_exact_output_identity_and_restores_on_failure():
    output = (np.array([.3, .7], np.float32), np.array([.8], np.float32))
    def original(q):
        assert q == 'fixed'
        return output
    adapter = SimpleNamespace(direct_q=original)
    seen = []
    with worker.capture_direct_q(adapter, seen.append):
        assert adapter.direct_q('fixed') is output
    assert seen[0] is output[0] and adapter.direct_q is original
    def fail(_):
        raise RuntimeError('simulated cache write failure')
    with pytest.raises(RuntimeError, match='cache write'), worker.capture_direct_q(adapter, fail):
        adapter.direct_q('fixed')
    assert adapter.direct_q is original
