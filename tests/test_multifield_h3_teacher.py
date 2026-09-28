"""CPU contracts for the actual-soft fusion, fresh E output, and target-read barrier."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve()
SCRIPT = HERE.parent/'evaluate_multifield_h3_teacher.py'
if not SCRIPT.exists():
    SCRIPT = HERE.parents[1]/'scripts/evaluate_multifield_h3_teacher.py'
spec = importlib.util.spec_from_file_location('multifield_tested', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_uint8_mean_no_overflow_and_ties_to_even():
    a = np.array([0, 1, 2, 254, 255], np.uint8)
    b = np.array([1, 2, 3, 255, 255], np.uint8)
    np.testing.assert_array_equal(m.average_rgb(a, b), [0, 2, 2, 254, 255])


def test_soft_probabilities_blend_before_warp_not_hard_ids():
    a = np.array([[[.51, .49, 0, 0, 0], [.99, .01, 0, 0, 0]]], np.float32)
    b = np.array([[[.01, .99, 0, 0, 0], [.49, .51, 0, 0, 0]]], np.float32)
    mix = m.blend_soft(a, b)
    np.testing.assert_array_equal(mix, .5*a+.5*b)
    np.testing.assert_array_equal(mix.argmax(-1), [[1, 0]])
    back = np.array([[[.75, 0.]]], np.float32)
    actual = m.base.output_probabilities(mix.transpose(2, 0, 1), back)
    expected = cv2.remap(.5*a+.5*b, back[..., 0], back[..., 1], cv2.INTER_LINEAR)
    np.testing.assert_array_equal(actual, expected)
    assert actual.argmax(-1).item() == 0
    with pytest.raises(ValueError, match='HWC FP32'):
        m.blend_soft(a.argmax(-1).astype(np.float32), b.argmax(-1).astype(np.float32))


@pytest.mark.parametrize('mutation', ['missing_e', 'duplicate', 'unknown_arm'])
def test_250_prediction_barrier_rejects_incomplete_or_duplicate_sets(mutation):
    views = [{'name': f'{i:03d}.png'} for i in range(50)]
    predictions = [{'arm': a, 'name': v['name']} for a in m.ARMS for v in views]
    m.prediction_barrier(predictions, views)
    if mutation == 'missing_e':
        predictions = [r for r in predictions if r['arm'] != 'E']
    elif mutation == 'duplicate':
        predictions[-1] = predictions[0]
    else:
        predictions[-1] = {'arm': 'F', 'name': views[-1]['name']}
    with pytest.raises(ValueError, match='250 unique'):
        m.prediction_barrier(predictions, views)


def test_e_is_written_from_new_a_array_and_byte_exact_composite(tmp_path):
    rgb = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    source = m.save_png(tmp_path/'fresh_rgb.png', rgb[..., ::-1])
    a = np.arange(20, dtype=np.uint8).reshape(4, 5) % 5
    saved_mask, saved_rgb = m.engineering_control(tmp_path, '001.png', a, source)
    np.testing.assert_array_equal(cv2.imread(saved_mask['path'], cv2.IMREAD_UNCHANGED), a)
    assert Path(saved_rgb['path']).read_bytes() == Path(source['path']).read_bytes()
    assert saved_rgb['sha256'] == source['sha256']
    with pytest.raises(ValueError, match='overwrite'):
        m.engineering_control(tmp_path, '001.png', a, source)


def test_soft_cache_is_fp32_exact_and_hashed(tmp_path):
    p = np.full((4, 5, 5), .2, np.float32)
    record = m.save_soft(tmp_path/'p.npy', p)
    np.testing.assert_array_equal(np.load(record['path'], allow_pickle=False), p)
    assert record['sha256'] == m.sha(record['path']) and record['dtype'] == 'float32'
    with pytest.raises(ValueError, match='FP32'):
        m.save_soft(tmp_path/'wrong.npy', p.astype(np.float64))


def test_mismatched_canvas_is_rejected_without_relabeling():
    class Official:
        @staticmethod
        def pixel_protocol(state):
            return state['id']
        @staticmethod
        def distortion_render_grid(*args):
            return np.eye(3, dtype=np.float32), 4, 3, None
    camera = {'K': np.eye(3).tolist(), 'distortion': [0., 0.], 'width': 4, 'height': 3}
    info = {'canvas': [4, 3], 'canvas_K': np.eye(3).tolist()}
    m.verify_canvas(camera, {'id': m.base.LEGACY}, info, Official)
    for state, bad_info in [({'id': 'colmap_corner_v2'}, info),
                             ({'id': m.base.LEGACY}, dict(info, canvas=[5, 3]))]:
        with pytest.raises(ValueError, match='exact legacy canvas'):
            m.verify_canvas(camera, state, bad_info, Official)
