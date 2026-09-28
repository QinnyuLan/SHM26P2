"""CPU contracts; synthetic values only, no actual camera rendering/pixel data."""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location('h3_readout', Path(__file__).parents[1]/'scripts/audit_h3_semantic_readout.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_fixed_selection_uses_train_sort_index_not_filename175():
    excluded = set(range(1, 201, 7)) | set(range(301, 322))
    rows = [{'name': f'{i:03}.png', 'split': 'val' if i in excluded else 'train', 'width': 1320, 'height': 989,
             'K': np.eye(3).tolist(), 'w2c_original': np.eye(4).tolist(), 'mask_path': '/never-mask',
             'valid_path': '/never-valid', 'image_path': '/never-real-rgb'} for i in range(400, 0, -1)]
    chosen, names = probe.fixed_views({'views': rows})
    assert [v['name'] for v in chosen] == ['002.png', '205.png']
    assert all(names[v['camera_index']] == v['name'] for v in chosen)
    assert all('image_path' not in v for v in chosen)
    with pytest.raises(ValueError, match='legacy'):
        probe.fixed_views({'views': rows, 'pixel_protocol': 'colmap_corner_v2'})


def test_class_difference_projection_preserves_probabilities_and_linear_cross_moment():
    generator = torch.Generator().manual_seed(42)
    features = torch.randn(17, 16, generator=generator)
    weight = torch.randn(5, 16, generator=generator)
    d = weight.double()[1:]-weight.double()[0]
    _, _, vh = torch.linalg.svd(d, full_matrices=True)
    projection = vh[:4].T@vh[:4]
    candidate = probe.project_features(features, projection.tolist())
    assert not torch.equal(features, candidate)
    assert torch.allclose((features@weight.T).softmax(-1), (candidate@weight.T).softmax(-1), atol=1e-6, rtol=0)
    mass = torch.rand(6, 17, generator=generator).double()
    z = torch.arange(17).double()[:, None]
    assert torch.allclose((mass@(features.double()*z))@projection,
                          mass@(features.double()@projection*z), atol=1e-12, rtol=0)


def test_invariance_strict_rgb_bitwise_and_full_image_p3d_gate():
    original = {'rgb': torch.zeros(3, 4, 3), 'depth': torch.ones(3, 4, 1), 'alpha': torch.ones(3, 4, 1),
                'p3d': torch.ones(3, 4, 5)/5}
    projected = {k: v.clone() for k, v in original.items()}
    assert probe.invariance(original, projected)['passed']
    projected['p3d'][0, 0, 1] += 2e-6
    assert not probe.invariance(original, projected)['passed']
    projected = {k: v.clone() for k, v in original.items()}
    projected['rgb'][2, 3, 0] = -0.0  # Signed-zero differs bitwise though torch.equal would accept it.
    assert not probe.invariance(original, projected)['passed']
    json.dumps(probe.invariance(original, projected), allow_nan=False)


def test_all_state_flags_modes_and_cameras_restored_on_exception():
    scene = torch.nn.Sequential(torch.nn.Linear(16, 5), torch.nn.BatchNorm1d(5))
    scene[1].eval()
    scene[0].bias.requires_grad_(False)
    scene.register_buffer('extra', torch.ones(3))
    before = {k: v.clone() for k, v in scene.state_dict().items()}
    cameras = torch.eye(4)[None]
    report = {}
    with pytest.raises(RuntimeError, match='synthetic'), probe.preserved_scene(scene, cameras, report):
        scene[0].weight.fill_(4)
        scene.extra.fill_(2)
        cameras.zero_()
        raise RuntimeError('synthetic')
    assert all(torch.equal(before[k], v) for k, v in scene.state_dict().items())
    assert scene.training and not scene[1].training and not scene[0].bias.requires_grad
    assert torch.equal(cameras, torch.eye(4)[None])
    assert all(report[k] for k in ('all_model_tensors_finally_restored_exact', 'flags_restored_exact',
                                   'modes_restored_exact', 'cameras_restored_exact', 'no_parameter_gradients'))
    json.dumps(report, allow_nan=False)


def test_pixel_gate_denies_early_or_real_rgb_and_restores_reader(tmp_path):
    allowed = tmp_path/'mask.png'
    cv2.imwrite(str(allowed), np.zeros((2, 2), np.uint8))
    old = cv2.imread
    evidence = {}
    with probe.pixel_guard([allowed], evidence):
        with pytest.raises(ValueError, match='Pixel decode'):
            cv2.imread(str(allowed))
        evidence['predictions_completed'] = True
        with pytest.raises(ValueError, match='Pixel decode'):
            cv2.imread(str(tmp_path/'real_rgb.png'))
        assert cv2.imread(str(allowed)).shape == (2, 2, 3)
    assert cv2.imread is old and len(evidence['reads']) == 1


def test_descriptive_ce_valid_pixel_denominator_and_correction_accounting():
    p = torch.tensor([[[.4, .1, .3, .1, .1], [.1, .1, .6, .1, .1], [.9, .025, .025, .025, .025]]])
    labels = torch.tensor([[2, 2, 255]])
    valid = torch.tensor([[True, True, False]])
    score = probe.score_prediction(p, labels, valid)
    assert score['ce'] == pytest.approx(float(-torch.log(torch.tensor([.3, .6])).mean()))
    assert score['valid_pixels'] == 2 and sum(map(sum, score['confusion_matrix'])) == 2
    original = {'p3d': p, 'probabilities': p.flip(2)}
    original['probabilities'] = torch.tensor([[[0., 0., 1., 0., 0.], [0., 0., 1., 0., 0.], [1., 0., 0., 0., 0.]]])
    candidate = {'probabilities': p}
    counts = probe.correction_counts(original, candidate, labels, valid)
    assert counts['original_cable_raw_bg_corrected_by_head'] == 1
    assert counts['correction_retained_after_projection'] == 0 and counts['correction_lost_after_projection'] == 1
    json.dumps({'score': score, 'counts': counts}, allow_nan=False)
