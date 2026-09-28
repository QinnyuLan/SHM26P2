"""Synthetic CPU contracts for the minimal TRAIN metadata port."""
import json

import numpy as np
import pytest

from bridge_rgs.ibgs_data_contract import (
    build_train_contract,
    neighbor_contract,
    point_cloud_descriptor,
)


def make_manifest(tmp_path):
    (tmp_path/'valid').write_bytes(b'not decoded as image')
    views = []
    for i in range(3):
        (tmp_path/f'{i}.rgb').write_bytes(bytes([i]))
        pose = np.eye(4); pose[0, 3] = -.1*i
        views.append({'name': f'{i:03}.png', 'split': 'train', 'image_id': i+1, 'camera_id': 1,
                      'width': 5, 'height': 3, 'K': [[4, 0, 2.5], [0, 4, 1.5], [0, 0, 1]],
                      'w2c': pose.tolist(), 'w2c_original': pose.tolist(),
                      'image_path': f'{i}.rgb', 'valid_path': 'valid', 'mask_path': 'must_not_open_mask'})
    views.append({'name': 'val.png', 'split': 'val', 'image_path': 'missing_VAL', 'mask_path': 'missing_label'})
    np.savez(tmp_path/'init.npz', points=np.zeros((2, 3), np.float32), colors=np.ones((2, 3), np.float32),
             semantic_counts=np.zeros((2, 5), np.float32))
    manifest = {'pixel_protocol': {'id': 'colmap_corner_v2'}, 'views': views,
                'init_points_path': 'init.npz', 'scene_radius': 5.}
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(manifest))
    return path, manifest


def test_metadata_and_hashes_train_only_no_decode_or_copy(tmp_path, monkeypatch):
    path, _ = make_manifest(tmp_path)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('No NPZ arrays should be deserialized'))
    before = set(tmp_path.iterdir()); result = build_train_contract(path, required_train_count=3)
    assert set(tmp_path.iterdir()) == before
    assert len(result['pixel_hashes']) == 4 and result['shared_valid_files'] == 1
    assert result['heldout_payloads_read'] == result['image_or_mask_decodes'] == 0
    assert [r['name'] for r in result['train_rows']] == ['000.png', '001.png', '002.png']
    assert all('mask_path' not in r for r in result['train_rows'])
    assert result['initial_point_cloud']['point_count'] == 2
    assert result['initial_point_cloud']['semantic_arrays_loaded'] is False
    assert result['neighbors']['coverage']['at_least_4'] == 0
    json.dumps(result, allow_nan=False)


def test_actual_rotation_center_and_fp32_metadata(tmp_path):
    path, manifest = make_manifest(tmp_path); theta = .123456789
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    center = np.array([.2, -.3, 1.]); pose = np.eye(4); pose[:3, :3] = R; pose[:3, 3] = -R@center
    manifest['views'][0].update(w2c=pose.tolist(), w2c_original=pose.tolist()); path.write_text(json.dumps(manifest))
    row = build_train_contract(path, required_train_count=3)['train_rows'][0]
    np.testing.assert_allclose(row['camera_center'], center, atol=1e-15)
    np.testing.assert_array_equal(row['ibgs_R'], R.T)
    np.testing.assert_array_equal(row['w2c_fp32'], pose.astype(np.float32).astype(np.float64))


def test_heldout_alias_population_and_mismatched_grid_fail(tmp_path):
    path, manifest = make_manifest(tmp_path)
    with pytest.raises(ValueError, match='population'):
        build_train_contract(path)
    manifest['views'][-1]['image_path'] = '0.rgb'; path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='held-out'):
        build_train_contract(path, required_train_count=3)
    manifest['views'][-1]['image_path'] = 'missing_VAL'; manifest['views'][1]['width'] = 6
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='share native'):
        build_train_contract(path, required_train_count=3)


def row(name, distance, angle=0, index=0):
    a = np.deg2rad(angle)
    return {'name': name, 'index': index, 'split': 'train', 'camera_center': [distance, 0, 0],
            'forward_world': [np.sin(a), 0, np.cos(a)]}


def test_default_strict_filters_self_exclusion_ties_and_no_rescue():
    rows = [row('target', 0), row('minimum', .01), row('maximum', 1.5), row('behind', .5, 31),
            row('z', 1, 10, 4), row('a', 1, 10, 5), row('first_angle', 1, 0, 6), row('far', 2, 0, 7)]
    result = neighbor_contract(rows, 5); target = result['views'][0]
    assert [r['name'] for r in target['neighbors_4']] == ['first_angle', 'a', 'z']
    assert target['default_candidate_count'] == 3
    assert target['threshold_requirement_4']['strict_upper_distance_must_exceed'] == 1.5
    assert target['threshold_requirement_4']['distance_over_scene_radius'] == .3
    assert target['threshold_requirement_8']['strict_upper_distance_must_exceed'] is None
    assert result['filters']['maximum_distance_over_scene_radius'] == .3
    bad = [dict(rows[0], split='val')]+rows[1:]
    with pytest.raises(ValueError, match='Only TRAIN'):
        neighbor_contract(bad, 5)


def test_eight_neighbors_are_fixed_prefix_of_metadata_order():
    rows = [row('target', 0)]+[row(f's{i}', .1*i, index=i) for i in range(1, 11)]
    target = neighbor_contract(rows, 5)['views'][0]
    assert [r['index'] for r in target['neighbors_4']] == [1, 2, 3, 4]
    assert [r['index'] for r in target['neighbors_8']] == list(range(1, 9))


def test_point_fields_shape_and_object_refusal(tmp_path):
    bad = tmp_path/'bad.npz'; np.savez(bad, points=np.zeros((2, 3), np.float32), colors=np.zeros((1, 3), np.float32))
    with pytest.raises(ValueError, match='Nx3'):
        point_cloud_descriptor(bad)
    obj = tmp_path/'object.npz'; np.savez(obj, points=np.array([object()], dtype=object))
    with pytest.raises(ValueError, match='Object'):
        point_cloud_descriptor(obj)
