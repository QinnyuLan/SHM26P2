"""Small CPU contracts for the explicit original-grid to legacy teacher adapter."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve()
SCRIPT = HERE.parent/'evaluate_rgb_teacher_transfer.py'
if not SCRIPT.is_file():
    SCRIPT = HERE.parents[1]/'scripts/evaluate_rgb_teacher_transfer.py'
spec = importlib.util.spec_from_file_location('rgb_teacher_transfer_tested', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def camera(distortion=None):
    return {'width': 31, 'height': 23, 'K': [[16., 0., 15.], [0., 16., 11.], [0., 0., 1.]],
                'distortion': [0.]*5 if distortion is None else distortion}


def test_zero_distortion_preserves_rgb_and_probabilities():
    from bridge_rgs.evaluate import distortion_render_grid
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(23, 31, 3), dtype=np.uint8)
    mx, my, back, info = m.adapter_maps(camera(), distortion_render_grid)
    assert np.array_equal(m.input_canvas(rgb, mx, my), rgb)
    assert back is None and info['canvas'] == [31, 23]
    assert info['canvas_incomplete_source_support_pixels'] == 0
    prob = rng.random((5, 23, 31), dtype=np.float32)
    assert np.array_equal(m.output_probabilities(prob, back), prob.transpose(1, 2, 0))


def test_distorted_adapter_uses_canvas_k_without_half_pixel_and_black_border():
    from bridge_rgs.evaluate import distortion_render_grid
    cam = camera([.2, 0., 0., 0., 0.])
    mx, my, back, info = m.adapter_maps(cam, distortion_render_grid)
    expected_x, expected_y = cv2.initUndistortRectifyMap(np.asarray(cam['K'], np.float32),
        np.asarray(cam['distortion'], np.float32), None, np.asarray(info['canvas_K'], np.float32),
        tuple(info['canvas']), cv2.CV_32FC1)
    np.testing.assert_array_equal(mx, expected_x)
    np.testing.assert_array_equal(my, expected_y)
    assert back is not None and info['canvas_incomplete_source_support_pixels'] > 0
    canvas = m.input_canvas(np.full((23, 31, 3), 255, np.uint8), mx, my)
    assert canvas.min() == 0 and canvas.max() == 255
    assert info['original_incomplete_roundtrip_footprint_pixels'] > 0


def test_soft_probability_warp_precedes_argmax():
    prob = np.zeros((5, 1, 2), np.float32)
    prob[0, 0] = [.51, .01]
    prob[1, 0] = [.49, .99]
    back = np.array([[[.5, 0.]]], np.float32)
    got = m.output_probabilities(prob, back)
    np.testing.assert_allclose(got[0, 0, :2], [.26, .74])
    assert got.argmax(-1).item() == 1
    assert cv2.remap(prob.argmax(0).astype(np.float32), back[..., 0], back[..., 1], cv2.INTER_LINEAR).item() == .5


def test_explicit_sources_cannot_relabel_corner_checkpoint():
    source = {'input_grid': 'original_distorted_uint8_png', 'renderer_profiles': ['colmap_corner_v2']*2}
    declaration = m.input_declaration(m.ARMS[1], source)
    assert declaration['renderer_profiles'] == source['renderer_profiles']
    assert declaration['output_teacher_grid'] == m.LEGACY
    assert declaration['pure_teacher_weight'] == 1 and not declaration['uses_h3_semantic_readout']
    with pytest.raises(ValueError, match='relabel'):
        m.input_declaration(m.ARMS[1], dict(source, renderer_profiles=[m.LEGACY]*2))
    with pytest.raises(ValueError, match='Explicit'):
        m.input_declaration(m.ARMS[1], dict(source, input_grid='native'))


def test_fixed_gate_units_and_strict_confidence_bound():
    rgb = {'clauses': {'psnr_gain_at_least_0_15_dB': True, 'psnr_paired_95_lower_positive': True,
                           'ssim_point_not_lower': True, 'lpips_point_not_higher': True}}
    pair = {'metrics': {'miou_all': {'difference': .002, 'paired_view_bootstrap_95_interval': [.0001, .003]},
                             'stay_cable_iou': {'difference': -.001}}}
    assert all(m.adoption_clauses(rgb, pair).values())
    pair['metrics']['miou_all']['paired_view_bootstrap_95_interval'][0] = 0.
    assert not all(m.adoption_clauses(rgb, pair).values())


def test_fingerprint_uses_actual_frozen_four_key_contract():
    from bridge_rgs.official_evaluate import official_fingerprint
    path = Path('/mnt/data/SHM2026/runs/official_selected_ensemble_v1/cross_teacher/execution_receipt.json')
    receipt = m.read(path)
    records = [m.fingerprint_record(row) for row in receipt['source_records']]
    assert all(set(r) == {'camera', 'source_image_sha256', 'source_annotation_sha256', 'rasterized_mask_sha256'} for r in records)
    assert official_fingerprint(records) == receipt['official_evaluation_fingerprint']
