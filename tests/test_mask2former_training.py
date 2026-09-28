"""CPU contracts for fixed training, continuation and native soft-score tiling."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from transformers.models.swin.modeling_swin import SwinDropPath

from bridge_rgs.mask2former_training import (
    FORMAT,
    INFERENCE_PROTOCOL,
    TRAIN_PROTOCOL,
    ShuffledViews,
    normalize_rgb,
    poly_learning_rate,
    predict_tiled,
    query_scores,
    stage_gradient_status,
    tile_origins,
    training_sample,
    validate_config,
    validate_trained_state,
)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config():
    return yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/mask2former_reference_v1.yaml').read_text())


def test_config_rejects_unregistered_formula_and_paths():
    fixed = config()
    assert validate_config(fixed) == fixed
    fixed['protocol']['horizontal_flip_probability'] = .25
    with pytest.raises(ValueError):
        validate_config(fixed)
    fixed = config()
    fixed['manifest'] = 'relative.json'
    with pytest.raises(ValueError):
        validate_config(fixed)


def test_sampler_exact_resume_and_independent_complete_permutations():
    names = ['002.png', '003.png', '004.png']
    first = ShuffledViews(names, 42)
    prefix = [first.next() for _ in range(4)]
    assert sorted(prefix[:3]) == names
    state = first.state_dict()
    other = ShuffledViews(names, 100)
    other.load_state_dict(state)
    assert [first.next() for _ in range(15)] == [other.next() for _ in range(15)]
    state['names'][0] = 'bad.png'
    with pytest.raises(ValueError):
        other.load_state_dict(state)


def test_crop_context_flip_and_rng_resume_share_exact_valid_support():
    labels = np.broadcast_to((np.arange(1320)//264).astype(np.uint8), (989, 1320)).copy()
    rgb = np.repeat((labels*40)[..., None], 3, -1)
    valid = np.ones((989, 1320), bool)
    valid[:50] = False
    labels[500:503] = 255
    original = labels.copy()
    rng = np.random.default_rng(63)
    saved = copy.deepcopy(rng.bit_generator.state)
    first = training_sample(rgb, labels, valid, 1, rng)
    rng.bit_generator.state = saved
    replay = training_sample(rgb, labels, valid, 1, rng)
    assert first[3] == replay[3]
    assert all(torch.equal(a, b) for a, b in zip(first[:3], replay[:3], strict=True))
    full = training_sample(rgb, labels, valid, 2, rng)
    assert first[0].shape == (1, 3, 768, 768) and full[0].shape == (1, 3, 768, 1056)
    assert not full[0][..., 1025:].any() and not full[2][:, 1025:].any()
    assert torch.all(full[1][:, 1025:] == 255) and torch.all(full[1][~full[2]] == 255)
    assert np.array_equal(labels, original)
    with pytest.raises(ValueError, match='all-invalid'):
        training_sample(rgb, labels, np.zeros_like(valid), 1, rng)


def test_poly_schedule_uses_original_group_lr_through_optimizer_resume():
    p = nn.Parameter(torch.ones(2))
    opt = torch.optim.AdamW([{'params': [p], 'lr': 1e-5}])
    assert poly_learning_rate(opt, 1) == 1
    poly_learning_rate(opt, 3000)
    saved = copy.deepcopy(opt.state_dict())
    multiplier = poly_learning_rate(opt, 3001)
    resumed = torch.optim.AdamW([p], lr=7)
    resumed.load_state_dict(saved)
    assert poly_learning_rate(resumed, 3001) == multiplier
    assert opt.param_groups[0]['lr'] == resumed.param_groups[0]['lr']
    assert 0 < poly_learning_rate(opt, 6000) < .001
    with pytest.raises(ValueError):
        poly_learning_rate(opt, 6001)


def test_real_swin_drop_path_zero_gradient_is_legal_finite_probe():
    weight = nn.Parameter(torch.ones(1, 3, 4))
    drop = SwinDropPath(.5).train()
    torch.manual_seed(0)
    drop(weight).sum().backward()
    assert weight.grad is not None and not weight.grad.any()
    finite, nonzero = stage_gradient_status({'stage4': weight})
    assert finite == {'stage4': True} and nonzero == {'stage4': False}
    weight.grad[0, 0, 0] = float('nan')
    assert stage_gradient_status({'stage4': weight})[0] == {'stage4': False}


def test_query_scores_upsample_logits_before_sigmoid_and_omit_null_without_early_argmax():
    classes = torch.tensor([[[0., 1., 2., 3., 4., 5.], [5., 4., 3., 2., 1., 0.]]])
    masks = torch.tensor([[[[-2., 2.]], [[1., -1.]]]])
    scores = query_scores(classes.bfloat16(), masks.bfloat16(), (2, 3))
    masks_up = torch.nn.functional.interpolate(masks, (2, 3), mode='bilinear', align_corners=False).sigmoid()
    expected = sum(classes.softmax(-1)[0, q, :5, None, None]*masks_up[0, q] for q in range(2))
    torch.testing.assert_close(scores[0], expected)
    assert scores.dtype == torch.float32 and scores.shape == (1, 5, 2, 3)
    assert (scores > 0).all()


class FixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []
    def forward(self, pixels, output_auxiliary_logits):
        assert output_auxiliary_logits is False
        self.calls.append(tuple(pixels.shape))
        return SimpleNamespace(class_queries_logits=torch.tensor([[[1., 2., 3., 4., 5., 6.]]]),
                               masks_queries_logits=torch.zeros(1, 1, 2, 2))


@pytest.mark.parametrize('hw', [(31, 33), (768, 768), (989, 1320)])
def test_tiling_covers_native_edges_preserves_soft_class_mass_and_has_no_tta(hw):
    model = FixedModel()
    out = predict_tiled(model, np.zeros((*hw, 3), np.uint8))
    expected = torch.tensor([1., 2., 3., 4., 5.]).softmax(-1).numpy()
    assert out.shape == (*hw, 5) and out.dtype == np.float32
    np.testing.assert_allclose(out, np.broadcast_to(expected, out.shape), atol=1e-6)
    np.testing.assert_allclose(out.sum(-1), 1, atol=1e-6)
    assert len(model.calls) == len(tile_origins(hw[0]))*len(tile_origins(hw[1]))
    assert all(s[-1] % 32 == 0 and s[-2] % 32 == 0 for s in model.calls)
    assert INFERENCE_PROTOCOL['flip'] is False and INFERENCE_PROTOCOL['context_blend'] is False


def test_normalization_pads_after_normalizing_without_double_rescaling():
    image = np.full((3, 5, 3), 255, np.uint8)
    values = normalize_rgb(image)
    torch.testing.assert_close(values[0, :, 0, 0], (1-torch.tensor([.485, .456, .406]))/torch.tensor([.229, .224, .225]))
    assert not values[..., 3:, :].any() and not values[..., 5:].any()


def runner():
    path = Path(__file__).resolve().parents[1]/'scripts/train_mask2former_reference.py'
    spec = importlib.util.spec_from_file_location('reference_training_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_resume_restores_adam_sampler_aug_points_and_torch_rng_on_cpu(monkeypatch):
    script = runner()
    cfg = config()
    model = nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    opt.step()
    views = ShuffledViews(['a', 'b'], 42)
    name = views.next()
    aug = np.random.default_rng(12)
    aug.random()
    points = torch.Generator().manual_seed(45)
    torch.rand(2, generator=points)
    state = {'format': FORMAT, 'step': 1, 'training_config': cfg,
             'provenance': {'source_hashes': {'dummy': 'same'}, 'input_hashes': {}},
             'model': copy.deepcopy(model.state_dict()), 'optimizer': copy.deepcopy(opt.state_dict()),
             'sampler_state': views.state_dict(), 'augmentation_rng_state': copy.deepcopy(aug.bit_generator.state),
             'point_rng_state': points.get_state(), 'torch_rng_state': torch.get_rng_state(),
             'cuda_rng_state': torch.tensor([1, 2], dtype=torch.uint8),
             'trace': [{'step': 1, 'view': name}], 'loss_logs': []}
    expected = (views.next(), aug.random(), torch.rand(4, generator=points), torch.rand(4))
    got_cuda = []
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda value: got_cuda.append(value.clone()))
    new_model = nn.Linear(2, 1)
    new_opt = torch.optim.AdamW(new_model.parameters())
    new_views, new_aug, new_points = ShuffledViews(['a', 'b'], 4), np.random.default_rng(8), torch.Generator()
    script.restore_training(state, new_model, new_opt, new_views, new_aug, new_points, cfg,
                            {'source_hashes': {'dummy': 'same'}, 'input_hashes': {}})
    assert (new_views.next(), new_aug.random()) == expected[:2]
    assert torch.equal(torch.rand(4, generator=new_points), expected[2])
    assert torch.equal(torch.rand(4), expected[3]) and torch.equal(got_cuda[0], state['cuda_rng_state'])
    for key, value in model.state_dict().items():
        assert torch.equal(new_model.state_dict()[key], value)
    assert new_opt.state_dict()['state'].keys() == state['optimizer']['state'].keys()
    altered = copy.deepcopy(cfg)
    altered['protocol']['steps'] = 6001
    with pytest.raises(ValueError, match='Resume'):
        script.restore_training(state, new_model, new_opt, new_views, new_aug, new_points, altered,
                                {'source_hashes': {'dummy': 'same'}, 'input_hashes': {}})


def test_inference_rejects_intermediate_endpoint_or_wrong_pixel_profile():
    state = {'format': FORMAT, 'step': 6000, 'training_completed': True,
             'pixel_protocol': 'legacy_mixed_v1', 'class_names': ['background', 'deck', 'stay_cable', 'tower', 'foundation'],
             'training_config': config(), 'model_config': {'num_labels': 5, 'num_queries': 100}}
    assert validate_trained_state(state).num_labels == 5
    for key, value in [('step', 1000), ('pixel_protocol', 'colmap_corner_v2'), ('training_completed', False)]:
        bad = dict(state, **{key: value})
        with pytest.raises(ValueError):
            validate_trained_state(bad)
    assert TRAIN_PROTOCOL['validation_during_training'] is False


def evaluation_runner(monkeypatch):
    folder = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location('reference_evaluation_test', folder/'evaluate_mask2former_reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_camera_predictor_only_uses_rendered_rgb_and_soft_scores_before_warp(monkeypatch):
    import cv2
    module = evaluation_runner(monkeypatch)
    rgb = torch.tensor([[[-.2, .2, 1.2], [.4, .5, .6], [.8, .9, 1.]],
                        [[.3, .2, .1], [.9, .1, .2], [.5, .6, .7]]])
    observed = {}
    class Scene:
        def __init__(self):
            self.splats = {'means': torch.zeros(1, 3)}
        def render(self, K, pose, width, height, **kwargs):
            observed['render'] = (width, height, kwargs)
            return {'rgb': rgb.clone()}
    source = np.array([[[.5, 0], [1.5, 0]], [[.5, 1], [1.5, 1]]], np.float32)
    monkeypatch.setattr(module, 'distortion_render_grid', lambda *a: (np.eye(3, dtype=np.float32), 3, 2, source))
    probability = np.zeros((2, 3, 5), np.float32)
    probability[:, 0, 1] = 1
    probability[:, 1:, 2] = 1
    def prediction(model, image):
        np.testing.assert_array_equal(image, (rgb.clamp(0, 1).numpy()*255).round().astype(np.uint8))
        return probability
    monkeypatch.setattr(module, 'predict_tiled', prediction)
    camera = {'name': '001.png', 'image_id': 1, 'camera_id': 1, 'split': 'val',
              'K': np.eye(3).tolist(), 'w2c': np.eye(4).tolist(), 'width': 2, 'height': 2, 'distortion': [0]*5}
    delivered, mask, _ = module.predict_camera(Scene(), object(), camera, {})
    expected_rgb = cv2.remap(rgb.clamp(0, 1).numpy(), source[..., 0], source[..., 1], cv2.INTER_LINEAR)
    expected_scores = cv2.remap(probability, source[..., 0], source[..., 1], cv2.INTER_LINEAR)
    np.testing.assert_array_equal(delivered, (expected_rgb*255).round().astype(np.uint8))
    np.testing.assert_array_equal(mask, expected_scores.argmax(-1))
    assert observed['render'] == (3, 2, {'degree': 3, 'semantics': False, 'refine': False, 'absgrad': False})
    with pytest.raises(ValueError, match='whitelist'):
        module.predict_camera(Scene(), object(), dict(camera, source_image_path='never-read.png'), {})


def test_evaluation_rejects_incomplete_training_before_loading_model(tmp_path, monkeypatch):
    import json
    module = evaluation_runner(monkeypatch)
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps({'source_snapshot': str(Path(module.__file__).parent), 'configuration': {}}))
    checkpoint = tmp_path/'last.pt'
    checkpoint.write_bytes(b'not loaded')
    (tmp_path/'execution_receipt.json').write_text(json.dumps({'status': 'running'}))
    monkeypatch.setattr(module, 'verify_plan', lambda plan: None)
    monkeypatch.setattr(module, 'load_trained_reference', lambda *a: pytest.fail('must reject before model load'))
    output = tmp_path/'evaluation'
    with pytest.raises(ValueError, match='completed fixed6000'):
        module.evaluate(plan, checkpoint, output)
    assert not output.exists()
