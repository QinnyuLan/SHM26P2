"""Synthetic-only contracts: no actual model/camera/pixels or GPU."""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKER = HERE/'diagnose_ibgs_layer_sources.py'
if not WORKER.exists():
    WORKER = HERE.parent/'scripts/diagnose_ibgs_layer_sources.py'
spec = importlib.util.spec_from_file_location('layer_sources_test', WORKER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_fp32_projection_uses_plane_depth_and_original_matrix_order():
    xy = np.array([[1., 2.], [3., 4.]], np.float32)
    z = np.array([2., 4.], np.float32)
    focal = np.array([2., 3.], np.float32); principal = np.array([1., 1.], np.float32)
    matrix = np.eye(4, dtype=np.float32); matrix[0, 3] = .5; matrix[2, 3] = 1.
    uv, result_z = m.project_query(xy, z, matrix, focal, principal)
    point = np.column_stack(((xy[:, 0]-1)*z*np.float32(.5),
                             (xy[:, 1]-1)*z*np.float32(1/3), z))
    point[:, 0] += .5; point[:, 2] += 1
    expected = point[:, :2]*focal*(np.float32(1)/(point[:, 2]+np.float32(1e-8)))[:, None]+principal
    np.testing.assert_array_equal(uv, expected)
    np.testing.assert_array_equal(result_z, [3., 5.])
    assert uv.dtype == result_z.dtype == np.float32


def test_texture_coordinate_rounding_clamped_taps_and_invalid_domain():
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)+1
    uv = np.array([[.0099, 0], [3., 2.], [-.01, 1], [np.inf, 1]], np.float32)
    sampled = m.bilinear_query(depth, uv)
    actual_fraction = float(np.float32(uv[0, 0]+np.float32(.5)))-.5
    assert sampled['ideal'][0] == 1+actual_fraction
    assert sampled['quantized'][0] == np.float32(1+3/256)
    np.testing.assert_array_equal(sampled['taps'][1], [[3, 2]]*4)
    assert sampled['ideal'][1] == sampled['quantized'][1] == 12
    assert sampled['inside'].tolist() == [True, True, False, False]
    assert np.isnan(sampled['ideal'][2:]).all()


def test_dual_gate_ambiguity_invalid_z_and_original_median_support():
    depth = np.array([[1., 2.], [1., 2.]], np.float32)
    p = m.depth_predicate(depth, np.array([[.0099, 0], [0, 0], [0, 0]], np.float32),
                          np.array([1., -1., 1.], np.float32))
    assert p['ideal_gate'].tolist() == [True, False, True]
    assert p['quantized_gate'].tolist() == [False, False, True]
    assert p['ambiguous'].tolist() == [True, False, False]
    table = {'center_xy': np.array([[0, 0], [1, 1]]), 'candidate_center_index': np.array([0, 0, 1]),
             'median_mask': np.array([False, True, False]), 'w': np.array([.4, .3, .2], np.float32)}
    support = m.median_support(table, p)
    # Good candidate G_F does not establish any original median-supported source.
    np.testing.assert_array_equal(support['ideal'], [0., 0.])
    np.testing.assert_array_equal(support['quantized'], [0., 0.])


def test_source_id_weight_uses_full_rgb_list_not_source_top4():
    rows = [{'ids': np.array([7, 8]), 'w': np.array([.02, .3]), 'alpha': np.array([.04, .6]),
             'T': np.array([.5, .5]), 'z': np.array([-1., 5.])}]
    values, present = m.source_identity_weights(rows, np.array([[0, 0, -1, 0], [0, 0, 0, 0]]), [7, 9])
    np.testing.assert_array_equal(present, [[True, True, False, True], [False]*4])
    assert values['w'][0, 0] == .02 and values['z'][0, 0] == -1
    assert np.isnan(values['w'][0, 2]) and np.all(values['w'][1] == 0)
    # Repeated clamp tap remains one physical record but its weights add normally.
    assert np.dot(values['w'][0, [0, 1]], [.4, .6]) == .02


def test_candidate_union_keeps_non_narrow_and_empty_eligible_centers():
    trace = {'validity': np.array([True, True, False]), 'is_center': np.ones(3, bool),
             'offsets': np.array([0, 3, 3, 4]), 'xy': np.array([[0, 0], [1, 1], [2, 2]]),
             'cell_id': np.arange(3), 'ids': np.array([4, 5, 6, 7]), 'z': np.array([1., 2., np.inf, 3.]),
             'median_mask': np.array([True, False, True, True]), 'top4_mask': np.array([False, True, False, True]),
             'order': np.arange(4), 'w': np.array([.1, .4, .2, .3])}
    summaries = [{'comparison': {'status': 'summary_consistent_reconstruction'},
                  'finite_cpu_arithmetic': True, 'production': {'median_depth': 2.},
                  'summary': {'median_depth': 2.}} for _ in range(3)]
    def footprint(conic):
        return np.full(len(conic), 4.), np.full(len(conic), 1.), np.ones(len(conic), bool)
    result = m.candidate_table(trace, summaries, np.zeros((8, 4)), footprint)
    assert result['center_ray_index'].tolist() == [0, 1]
    assert result['ids'].tolist() == [4, 5]
    assert not result['narrow_front'].any()
    assert result['candidate_center_index'].tolist() == [0, 0]
