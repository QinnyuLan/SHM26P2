"""Camera-only export: inputs and exact pixel operations, no neural mocks of scores."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('multifield_bundle',
    Path(__file__).resolve().parents[1]/'scripts/render_multifield_bundle.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def camera():
    return {'name': 'novel.png', 'K': [[400, 0, 300], [0, 400, 200], [0, 0, 1]],
            'w2c': np.eye(4).tolist(), 'distortion': [0, 0, 0, 0], 'width': 600, 'height': 400}


def test_new_camera_without_dataset_ids_or_paths():
    result = m.camera_record(camera(), 7)
    assert result['image_id'] == result['camera_id'] == 7
    assert result['split'] == 'test'
    assert set(result) == {'name', 'image_id', 'camera_id', 'split', 'K', 'w2c', 'distortion', 'width', 'height'}


@pytest.mark.parametrize('extra', ['source_image_path', 'rgb', 'mask', 'annotation', 'valid_path'])
def test_target_or_image_inputs_rejected(extra):
    with pytest.raises(ValueError):
        m.camera_record({**camera(), extra: '/must/not/open'}, 0)


@pytest.mark.parametrize('value', ['../x.png', '/tmp/x.png', 'x.jpg', '', '.png/../x.png'])
def test_path_escape_rejected(value):
    with pytest.raises(ValueError):
        m.camera_record({**camera(), 'name': value}, 0)


def test_bad_rotation_rejected():
    value = camera()
    value['w2c'][0][0] = -1
    with pytest.raises(ValueError, match='proper rotation'):
        m.camera_record(value, 0)


def test_uint8_mean_all_possible_values_matches_integer_half_even():
    a, b = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8))
    actual = m.mean_rgb(np.repeat(a[..., None], 3, axis=-1), np.repeat(b[..., None], 3, axis=-1))[..., 0]
    total = a.astype(np.uint16)+b
    floor = total // 2
    expected = floor + ((total % 2 == 1) & (floor % 2 == 1))
    np.testing.assert_array_equal(actual, expected)


def test_soft_probabilities_are_warped_before_argmax():
    p = np.zeros((2, 2, 5), np.float32)
    p[..., 0] = [[.51, .01], [.51, .01]]
    p[..., 1] = 1-p[..., 0]
    t = np.zeros_like(p)
    t[..., 0] = [[.99, .49], [.99, .49]]
    t[..., 1] = 1-t[..., 0]
    class Teacher:
        def blend(self, rgb, probabilities):
            assert rgb.min() == 0 and rgb.max() == 1
            return .5 * probabilities + .5 * t
    back = np.array([[[.25, 0], [.75, 1]]], np.float32)
    rgb = np.array([[[-1, 0, 2]]*2]*2, np.float32)
    result = m.semantic_mask(rgb, p, Teacher(), back)
    expected = cv2.remap(.5*p+.5*t, back[..., 0], back[..., 1], cv2.INTER_LINEAR).argmax(-1)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result, [[0, 1]])


def test_zero_coverage_is_not_silently_background():
    class Teacher:
        def blend(self, rgb, probabilities):
            return probabilities
    with pytest.raises(ValueError, match='Uncovered'):
        m.semantic_mask(np.zeros((2, 2, 3)), np.zeros((2, 2, 5)), Teacher(), None)
