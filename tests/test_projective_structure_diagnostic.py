"""Synthetic protocol contracts; no real axes or target pixels are inspected."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bridge_rgs import structure_axes as geometry

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/diagnose_projective_structure_axes.py'
if not SCRIPT.exists():
    SCRIPT = Path(__file__).with_name('diagnose_projective_structure_axes.py')
_spec = importlib.util.spec_from_file_location('projective_structure_diagnostic', SCRIPT)
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


def cameras_fixture(count=350):
    return [{'name': f'{i+1:03d}.png', 'image_id': i+1, 'camera_id': 1, 'split': 'train',
             'width': 1320, 'height': 989, 'K': [[925.7016189, 0, 660], [0, 925.7016189, 494.5], [0, 0, 1]],
             'w2c': np.eye(4).tolist(), 'w2c_original': np.eye(4).tolist(),
             'mask_path': f'/nonexistent/masks/{i}.png' if i < 259 else None,
             'valid_path': '/nonexistent/valid.png', 'image_path': f'/nonexistent/rgb/{i}.png'}
            for i in range(count)]


def patch_names(monkeypatch, cameras):
    monkeypatch.setattr(worker, 'NAMES', tuple(cameras[i*258//15]['name'] for i in range(16)))


def test_camera_groups_precision_and_train_gate(monkeypatch):
    cameras = cameras_fixture(); patch_names(monkeypatch, cameras)
    rows = worker.camera_metadata({'views': cameras})
    assert np.bincount([r['group'] for r in rows]).tolist() == [88, 87, 88, 87]
    assert rows[0]['K'][0][0] == float(np.float32(925.7016189))
    assert rows[0]['w2c_original'] == cameras[0]['w2c_original']
    cameras[0]['split'] = 'val'
    with pytest.raises(ValueError, match='350'):
        worker.camera_metadata({'views': cameras})


def test_candidate_thresholds_are_fixed_and_do_not_relabel():
    counts = np.array([[0, 3, 0, 0, 0], [0, 2, 0, 0, 0], [1, 9, 0, 0, 0],
                       [1, 8, 0, 0, 0], [0, 0, 0, 3, 0], [0, 3, 0, 0, 0]], np.float32)
    arrays = {'semantic_counts': counts, 'num_observations': np.array([3]*6),
              'reprojection_error': np.array([3, 0, 1, 0, 0, 3.0001])}
    original = counts.copy()
    assert worker.candidate_mask(arrays, 1).tolist() == [True, False, True, False, False, False]
    assert np.array_equal(counts, original)


def test_whole_point_group_deletion_no_vote_subtraction():
    arrays = {'points': np.zeros((3, 3)), 'observation_image_ids': np.array([1, 2, 2, 3, 3, 4]),
              'observation_offsets': np.array([0, 2, 4, 6]), 'num_observations': np.array([2, 2, 2])}
    cameras = [{'image_id': i+1, 'group': i} for i in range(4)]
    touched = worker.point_camera_groups(arrays, cameras)
    assert touched.tolist() == [[True, True, False, False], [False, True, True, False], [False, False, True, True]]
    assert np.flatnonzero(~touched[:, 1]).tolist() == [2]
    arrays['observation_image_ids'][0] = 99
    with pytest.raises(ValueError, match='Non-TRAIN'):
        worker.point_camera_groups(arrays, cameras)


def test_global_constant_camera_weighting_not_valid_pixel_weighting():
    first = np.tile([1., 0.], (1000, 1)); second = np.array([[0., 1.]])
    third = np.array([[0., 1.]])
    result = worker.equal_camera_constant([(first, np.ones(1000, bool)),
        (second, np.ones(1, bool)), (third, np.ones(1, bool))], geometry)
    assert np.allclose(result['direction'], [0, 1])
    assert result['camera_count'] == 3


def test_axis_blocks_use_all_candidates_before_usable(monkeypatch):
    points = np.arange(60, dtype=float).reshape(20, 3)
    seen = []
    class Helper:
        def local_line_evidence(self, p, neighbors, minimum_linearity):
            return {'usable': np.arange(20)<16, 'directions': np.tile([1., 0, 0], (20, 1)),
                    'linearity': np.ones(20)}
        def spatial_blocks(self, p, bins):
            seen.append(p.copy())
            return np.arange(20)%4, {'bins': bins}
        orientation_blocks = staticmethod(geometry.orientation_blocks)
        principal_axis = staticmethod(geometry.principal_axis)
        bootstrap_axis = staticmethod(geometry.bootstrap_axis)
    base, saved = worker.estimate_axis(points, np.arange(20), Helper(), True, 3)
    assert np.array_equal(seen[0], points)
    assert base['usable_points'] == 16 and base['occupied_blocks'] == 4
    assert saved['bootstrap_axes'].shape == (256, 3)
    assert not worker.axis_gates(base, [dict(base, angle_to_base_degrees=0.)]*4)['passed']


def test_insufficient_axis_is_reportable_not_execution_failure():
    row, arrays = worker.estimate_axis(np.zeros((5, 3)), np.arange(5), geometry)
    assert row['axis'] is None and row['candidate_points'] == 5
    assert arrays['candidate_indices'].tolist() == list(range(5))
    assert not worker.axis_gates(row, [dict(row, angle_to_base_degrees=None)]*4)['passed']


def test_field_controls_and_support_contract(monkeypatch):
    names = tuple(f'{i}.png' for i in range(16)); monkeypatch.setattr(worker, 'NAMES', names)
    cameras = [{'name': n, 'width': 64, 'height': 48, 'K': [[90, 0, 32], [0, 90, 24], [0, 0, 1]],
                'w2c_original': np.eye(4).tolist()} for n in names]
    result = worker.field_diagnostics({1: np.array([1., 0, 0]), 3: np.array([0., 1, 0])}, cameras, geometry)
    assert result['valid_fraction_per_axis'] == {'1': 1., '3': 1.}
    assert result['axes_collinear_fraction_all_centers'] == 0
    assert not result['gates']['one_axis_at_least_8_views_median_fanout_5degrees']
    for row in result['views']:
        for axis in row['axes'].values():
            assert axis['sample_valid_fraction']['projective'] < 1
            assert axis['pairwise_common_sample_valid_fraction']['projective']['wrong_pose'] == axis['sample_valid_fraction']['projective']
    json.dumps(result, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item(), allow_nan=False)


def test_prepare_never_opens_pixel_payloads_and_copies_minimal_package(tmp_path, monkeypatch):
    root = tmp_path/'workspace'; root.mkdir()
    cameras = cameras_fixture(); patch_names(monkeypatch, cameras)
    prep = root/'artifacts/prepared'; prep.mkdir(parents=True)
    manifest = {'views': cameras, 'init_points_path': str(prep/'init_points.npz'),
                'geometry_audit_path': str(prep/'geometry_audit.json')}
    (prep/'manifest.json').write_text(json.dumps(manifest))
    (prep/'init_points.npz').write_bytes(b'not read during preparation')
    (prep/'geometry_audit.json').write_text('{}')
    legacy = root/'legacy'; legacy.mkdir()
    for name in ('prepare.py', 'data.py', 'geometry.py'):
        (legacy/name).write_text('# provenance only\n')
    for path in ('uv.lock', 'src/bridge_rgs/structure_axes.py', 'src/bridge_rgs/__init__.py',
                 'scripts/diagnose_projective_structure_axes.py', 'tests/test_projective_structure_diagnostic.py',
                 'tests/test_structure_axes.py', 'docs/'+worker.DOCUMENT):
        p = root/path; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# synthetic\n')
    monkeypatch.setattr(worker, 'ROOT', root); monkeypatch.setattr(worker, 'LEGACY_SOURCE', legacy)
    monkeypatch.setattr(worker, '__file__', str(root/'scripts/diagnose_projective_structure_axes.py'))
    path = worker.prepare(tmp_path/'out')
    plan = json.loads(path.read_text())
    assert plan['prepare_pixel_payload_reads'] == 0 and not plan['axis_estimation_during_prepare']
    assert len(plan['source_hashes']) == 6
    assert 'bridge_rgs/structure_axes.py' in plan['source_hashes']
    assert 'test_structure_axes.py' in plan['source_hashes']
    assert not any('/nonexistent/' in k for k in plan['input_hashes'])
    with pytest.raises(ValueError, match='Fresh'):
        worker.prepare(tmp_path/'out')


def test_collinear_gate_uses_common_valid_denominator(monkeypatch):
    names = tuple(f'{i}.png' for i in range(16)); monkeypatch.setattr(worker, 'NAMES', names)
    cameras = [{'name': n, 'width': 64, 'height': 48, 'K': np.eye(3).tolist(),
                'w2c_original': np.eye(4).tolist()} for n in names]
    class Helper:
        feature_centers = staticmethod(geometry.feature_centers)
        canonical_axis = staticmethod(geometry.canonical_axis)
        best_constant_direction = staticmethod(geometry.best_constant_direction)
        unoriented_degrees = staticmethod(geometry.unoriented_degrees)
        line_samples = staticmethod(geometry.line_samples)
        @staticmethod
        def project_direction(K, rotation, axis, centers, minimum_sine):
            valid = np.zeros(centers.shape[:2], bool); valid[0, 0] = True
            direction = np.zeros_like(centers); direction[0, 0] = [1, 0]
            return direction, valid, valid.astype(float)
    result = worker.field_diagnostics({1: [1, 0, 0], 3: [0, 1, 0]}, cameras, Helper())
    assert result['axes_collinear_fraction_all_centers'] < .25
    assert result['axes_collinear_fraction_common_valid'] == 1
    assert result['gates']['collinear_at_most_25percent'] is False
