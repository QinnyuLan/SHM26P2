"""Pure CPU wrapper contracts; does not build or initialize CUDA."""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKER = HERE/'check_ibgs_layer_select.py'
if not WORKER.exists():
    WORKER = HERE.parent/'scripts/check_ibgs_layer_select.py'
spec = importlib.util.spec_from_file_location('selector_check_contract', WORKER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_summary_preserves_empty_first_slot_zero_low_and_original_weights():
    selected = {'weights': np.array([[0., 0., .25, .1], [.2, .3, .1, .05]], np.float32),
                'depth': np.array([[0., 0., 2., 4.], [1., 2., 3., 4.]], np.float32),
                'ordinals': np.array([[0, 0, 2, 3], [5, 4, 6, 7]], np.int32)}
    result = m.summarize_selected(selected, [.4, .3], [3, 7])
    np.testing.assert_array_equal(result['median_low'], [0, 4])
    np.testing.assert_array_equal(result['median_high'], [3, 7])
    np.testing.assert_allclose(result['median_weight'], [.35, .65], atol=1e-7)
    np.testing.assert_allclose(result['median_depth'], [.9/.35, 1.3/.65], atol=3e-7)


def test_saved_trace_to_slots_preserves_ring_order_and_weight_tie_ordinal():
    trace = {'xy': np.array([[0, 0], [1, 0]]), 'offsets': np.array([0, 5, 5]),
             'median_slots': np.array([[2, 1, 3, -1], [-1]*4]),
             'ids': np.array([9, 8, 7, 6, 5]), 'order': np.arange(1, 6),
             'w': np.array([.1, .2, .2, .3, .8], np.float32),
             'z': np.array([1., 2., 3., 4., np.inf], np.float32)}
    result = m.reference_slots(trace)
    assert result['median4']['ids'][0].tolist() == [7, 8, 6, -1]
    assert result['top4']['ids'][0].tolist() == [6, 8, 7, 9]
    assert result['top4']['ordinals'][0].tolist() == [4, 2, 3, 1]
    assert np.all(result['median4']['ids'][1] == -1)
    assert np.all(result['top4']['depth'][1] == 0)


def test_numeric_tolerance_never_relaxes_discrete_or_nonfinite_and_cases_fixed():
    assert m.numerical_comparison([1.], [1.+1e-5])['passed']
    bad = m.numerical_comparison([3, 2], [3, 3], discrete=True)
    assert not bad['passed'] and bad['mismatches'] == 1 and bad['examples'][0]['index'] == [1]
    assert not m.numerical_comparison([np.nan], [np.nan])['passed']
    assert [c['name'] for c in m.synthetic_inputs()] == m.SPEC['synthetics']
    assert m.SPEC['sample_rays'] == 16*512*9
    assert m.SPEC['render_calls'] == m.SPEC['backward'] == m.SPEC['GT_decodes'] == 0
