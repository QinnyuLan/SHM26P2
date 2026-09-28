"""Synthetic CPU contracts; no real images, labels, observations or model loads."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve()
SCRIPT = HERE.parent/'collect_train_track_semantics.py'
if not SCRIPT.exists(): SCRIPT = HERE.parents[1]/'scripts/collect_train_track_semantics.py'
spec = importlib.util.spec_from_file_location('track_collector', SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def fixture():
    K = [[100., 0., 4.], [0., 100., 4.], [0., 0., 1.]]
    views = [{'image_id': i, 'name': f'{i:03}.png', 'split': 'train', 'camera_id': 1,
              'K': K, 'w2c_original': np.eye(4).tolist(), 'width': 9, 'height': 9}
             for i in (2, 3, 4)]
    arrays = {'points': np.array([[0., 0., 2.], [.02, 0., 2.]], np.float32),
              'track_ids': np.array([31, 17]), 'num_observations': np.array([3, 2]),
              'observation_image_ids': np.array([2, 3, 4, 2, 4]),
              'observation_offsets': np.array([0, 3, 5])}
    camera = {'model': 'SIMPLE_RADIAL', 'width': 9, 'height': 9, 'params': [100., 4., 4., 0.],
              'K': K, 'opencv_distortion': [0., 0., 0., 0., 0.]}
    return arrays, views, camera


def test_recovery_keeps_saved_csr_last_duplicate_and_skips_val(tmp_path):
    arrays, views, _ = fixture()
    lines = ['1 1 0 0 0 0 0 0 1 001.png', 'DO NOT PARSE VAL OBSERVATIONS']
    for v in views:
        lines.extend([f"{v['image_id']} 1 0 0 0 0 0 0 1 {v['name']}",
                      '3 4 31 4.25 4.5 31 5.5 4 17 2 2 99999'])
    path = tmp_path/'images.txt'; path.write_text('\n'.join(lines)+'\n')
    xy, points, by_view, report = collector.recover_observations(path, arrays, views)
    assert points.tolist() == [0, 0, 0, 1, 1]
    np.testing.assert_array_equal(xy, [[4.25, 4.5]]*3+[[5.5, 4.]]*2)
    assert by_view[2].tolist() == [0, 3] and by_view[3].tolist() == [1]
    assert report['selected_pair_duplicate_keypoints'] == 3
    assert report['nontrain_observation_lines_not_parsed'] == 1
    bad = path.read_text().replace('4.25 4.5 31', '4.25 4.5 99').replace('3 4 31', '3 4 99')
    path.write_text(bad)
    with pytest.raises(ValueError, match='keypoint missing'):
        collector.recover_observations(path, arrays, views)


def test_layout_rejects_val_and_duplicate_observed_views():
    arrays, views, _ = fixture()
    bad = copy.deepcopy(arrays); bad['observation_image_ids'][0] = 1
    with pytest.raises(ValueError, match='Held-out'):
        collector.observation_layout(bad, views)
    bad = copy.deepcopy(arrays); bad['observation_image_ids'][1] = 2
    with pytest.raises(ValueError, match='Repeated'):
        collector.observation_layout(bad, views)
    views[0]['split'] = 'val'
    with pytest.raises(ValueError, match='TRAIN'):
        collector.observation_layout(arrays, views)


def test_camera_and_pose_provenance_refusal(tmp_path):
    arrays, views, camera = fixture()
    path = tmp_path/'cameras.txt'; path.write_text('1 SIMPLE_RADIAL 9 9 100 4 4 0\n')
    collector.validate_source_cameras(path, {'1': camera})
    path.write_text('1 SIMPLE_RADIAL 9 9 99 4 4 0\n')
    with pytest.raises(ValueError, match='calibration'):
        collector.validate_source_cameras(path, {'1': camera})
    images = tmp_path/'images.txt'
    images.write_text('2 1 0 0 0 1 0 0 1 002.png\n4 4 31 5 4 17\n')
    with pytest.raises(ValueError, match='Original pose'):
        collector.recover_observations(images, arrays, views)


def test_floor_and_legacy_ties_and_negative_near_border_are_distinct():
    xy = np.array([[.5, 1.5], [1.51, .51], [-.49, 4.], [8.99, 8.99], [9., 1.]])
    primary, inside = collector.pixel_indices(xy, 9, 9, 'floor')
    legacy, legacy_inside = collector.pixel_indices(xy, 9, 9, 'rint')
    assert primary.tolist() == [[0, 1], [1, 0], [-1, 4], [8, 8], [9, 1]]
    assert legacy.tolist() == [[0, 2], [2, 1], [0, 4], [9, 9], [9, 1]]
    assert inside.tolist() == [True, True, False, True, False]
    assert legacy_inside.tolist() == [True, True, True, False, False]


def test_observed_ray_is_not_camera_to_saved_point_direction():
    arrays, views, camera = fixture()
    v = views[0]
    g = collector.geometry_for_view(np.array([[6., 4.]]), arrays['points'][1:].astype(np.float64), v, camera)
    np.testing.assert_allclose(g['reprojection_error'], [1.], atol=1e-7)
    np.testing.assert_allclose(g['viewing_ray_world'], np.array([[.02, 0., 1.]])/np.linalg.norm([.02, 0., 1.]))
    assert not np.allclose(g['viewing_ray_world'], g['point_direction_world'])
    np.testing.assert_allclose(np.linalg.norm(g['viewing_ray_world'], axis=1), 1.)
    # World direction must apply R transpose, not R, under column convention.
    rotated = copy.deepcopy(v); rotated['w2c_original'] = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.],
                                                                  [0., 0., 1., 0.], [0., 0., 0., 1.]]).tolist()
    g = collector.geometry_for_view(np.array([[5., 4.]]), np.array([[0., -.02, 2.]]), rotated, camera)
    np.testing.assert_allclose(g['reprojection_error'], 0., atol=1e-12)
    assert g['viewing_ray_world'][0, 1] < 0
    with pytest.raises(ValueError, match='nonpositive'):
        collector.geometry_for_view(np.array([[4., 4.]]), np.array([[0., 0., -1.]]), v, camera)


def test_official_polygon_ignore_and_later_known_override():
    data = {'imageWidth': 9, 'imageHeight': 9, 'imageData': 'never decoded', 'shapes': [
        {'label': 'stay_cable', 'points': [[0, 0], [8, 0], [8, 8], [0, 8]]},
        {'label': 'unknown', 'points': [[2, 2], [6, 2], [6, 6], [2, 6]]},
        {'label': 'deck', 'points': [[3, 3], [5, 3], [5, 5], [3, 5]]},
    ]}
    mask = collector.rasterize_annotation(json.dumps(data), 9, 9)
    assert mask[1, 1] == 2 and mask[2, 2] == 255 and mask[4, 4] == 1


def test_distances_include_other_class_ignore_valid_holes_and_exterior():
    mask = np.full((25, 25), 2, np.uint8); support = np.ones_like(mask, bool)
    indices = np.array([[12, 12], [0, 0], [24, 12], [25, 12]])
    inside = np.array([True, True, True, False])
    a = collector.sample_labels_and_distance(mask, support, indices, inside)
    np.testing.assert_array_equal(a['boundary_distance'][:3], [13., 1., 1.])
    assert a['label'][3] == 255 and np.isnan(a['boundary_distance'][3])
    mask[12, 14] = 255
    b = collector.sample_labels_and_distance(mask, support, indices, inside)
    assert b['boundary_distance'][0] == 2.
    mask[12, 14] = 0
    c = collector.sample_labels_and_distance(mask, support, indices, inside)
    assert c['boundary_distance'][0] == 2.
    support[12, 13] = False
    d = collector.sample_labels_and_distance(mask, support, indices, inside)
    assert d['boundary_distance'][0] == 1.
    support[12, 12] = False
    e = collector.sample_labels_and_distance(mask, support, indices, inside)
    assert e['sampled_label'][0] == 2 and e['label'][0] == 255 and not e['valid'][0]


def test_fixed_strata_endpoint_inclusivity_and_na():
    np.testing.assert_array_equal(collector.strata([0, 1, 1.01, 2, 2.01, np.nan], [1, 2]), [0, 0, 1, 1, 2, -1])
    np.testing.assert_array_equal(collector.strata([0, 3, 3.01, 10, 10.01, np.nan], [3, 10]), [0, 0, 1, 1, 2, -1])


def test_synthetic_collection_preserves_unlabeled_na_without_reading_rgb_or_val(tmp_path, monkeypatch):
    arrays, views, camera = fixture()
    arrays['reprojection_error'] = np.zeros(2, np.float32)
    init = tmp_path/'init.npz'; np.savez(init, **arrays)
    cameras = tmp_path/'cameras.txt'; cameras.write_text('1 SIMPLE_RADIAL 9 9 100 4 4 0\n')
    lines = ['1 1 0 0 0 0 0 0 1 001.png', 'VAL IS NEVER PARSED']
    bindings = {}
    for view in views:
        view['w2c'] = view['w2c_original']
        view['image_path'] = '/RGB_MUST_NOT_OPEN'
        view['source_image_path'] = '/RAW_RGB_MUST_NOT_OPEN'
        lines.extend([f"{view['image_id']} 1 0 0 0 0 0 0 1 {view['name']}", '4 4 31 5 4 17'])
        if view['image_id'] == 3:
            view.update(mask_path=None, source_annotation_path=None, valid_path='/UNLABELED_VALID_MUST_NOT_OPEN')
            continue
        annotation = tmp_path/f"{view['image_id']}.json"
        annotation.write_text(json.dumps({'imageWidth': 9, 'imageHeight': 9, 'shapes': []}))
        mask, valid = tmp_path/f"{view['image_id']}_mask.png", tmp_path/f"{view['image_id']}_valid.png"
        Image.fromarray(np.zeros((9, 9), np.uint8)).save(mask)
        Image.fromarray(np.full((9, 9), 255, np.uint8)).save(valid)
        view.update(source_annotation_path=str(annotation), mask_path=str(mask), valid_path=str(valid))
        for p in (annotation, mask, valid): bindings[str(p)] = collector.sha(p)
    images = tmp_path/'images.txt'; images.write_text('\n'.join(lines)+'\n')
    val = dict(views[0], image_id=1, name='001.png', split='val', mask_path='/VAL_MUST_NOT_OPEN',
               source_annotation_path='/VAL_ANNOTATION_MUST_NOT_OPEN')
    manifest = {'class_names': list(collector.CLASS_NAMES), 'views': views+[val], 'source_cameras': {'1': camera}}
    synthetic_spec = dict(collector.SPEC, points=2, observations=5, train_views=3, labeled_train_views=2)
    monkeypatch.setattr(collector, 'SPEC', synthetic_spec)
    monkeypatch.setattr(collector, 'verify', lambda _: manifest)
    plan = {'output': str(tmp_path), 'init_points': str(init), 'colmap_images': str(images),
            'colmap_cameras': str(cameras), 'source_hashes': {}, 'input_hashes': bindings}
    (tmp_path/'plan.json').write_text('{}')
    report = collector.collect(plan)
    assert report['status'] == 'completed' and report['raw_annotation_payload_reads'] == 2
    assert report['rgb_payload_reads'] == report['val_pixel_payload_reads'] == report['gpu_calls'] == 0
    with np.load(tmp_path/'observations.npz') as observations:
        unlabeled = observations['image_id'] == 3
        assert np.all(observations['primary_label'][unlabeled] == -1)
        assert np.all(observations['legacy_label'][unlabeled] == -1)
        assert np.isnan(observations['primary_boundary_distance'][unlabeled]).all()
        np.testing.assert_array_equal(observations['image_id'], arrays['observation_image_ids'])
        assert observations['viewing_ray_world'].shape == observations['point_direction_world'].shape == (5, 3)
    with pytest.raises(ValueError, match='overwrite'):
        collector.collect(plan)
