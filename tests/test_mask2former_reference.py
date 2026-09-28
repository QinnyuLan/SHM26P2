"""CPU-only contracts for the bounded Swin-L preflight, without downloads."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, SwinConfig

from bridge_rgs import mask2former_reference as ref


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small(classes=5):
    backbone = SwinConfig(embed_dim=16, depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 8],
                          window_size=4, drop_path_rate=.2,
                          out_features=['stage1', 'stage2', 'stage3', 'stage4'])
    cfg = Mask2FormerConfig(backbone_config=backbone, num_labels=classes, num_queries=6,
                           feature_size=32, mask_feature_size=32, hidden_dim=32,
                           num_attention_heads=4, encoder_layers=1, decoder_layers=3,
                           encoder_feedforward_dim=64, dim_feedforward=64, train_num_points=17)
    return Mask2FormerForUniversalSegmentation(cfg)


def test_transplant_reuses_every_other_tensor_and_retains_whole_new_six_row_head():
    source, target = small(7).state_dict(), small(5)
    before = {k: v.clone() for k, v in target.state_dict().items() if k in ref.REPLACED_KEYS}
    audit = ref.transplant_ade_state(target, source, pretrained_classes=7)
    after = target.state_dict()
    assert audit['target_classes'] == 5 and audit['reinitialized_keys'] == sorted(ref.REPLACED_KEYS)
    for name, tensor in after.items():
        assert torch.equal(tensor, before[name] if name in before else source[name])
    assert target.class_predictor.weight.shape == (6, 32)
    assert torch.equal(target.criterion.empty_weight, torch.tensor([1., 1., 1., 1., 1., .1]))


@pytest.mark.parametrize('corruption', ['missing', 'extra', 'shape', 'dtype', 'nan', 'old_head_shape'])
def test_transplant_rejects_unlisted_change_before_mutating_target(corruption):
    source, target = small(7).state_dict(), small(5)
    before = copy.deepcopy(target.state_dict())
    key = next(k for k, v in source.items() if k not in ref.REPLACED_KEYS and v.is_floating_point())
    if corruption == 'missing':
        del source[key]
    elif corruption == 'extra':
        source['unknown'] = torch.tensor(1.)
    elif corruption == 'shape':
        source[key] = source[key].flatten()[:1]
    elif corruption == 'dtype':
        source[key] = source[key].double()
    elif corruption == 'nan':
        source[key].view(-1)[0] = float('nan')
    else:
        source['class_predictor.bias'] = torch.zeros(9)
    with pytest.raises(ValueError):
        ref.transplant_ade_state(target, source, pretrained_classes=7)
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in before.items())


def test_author_groups_cover_once_and_classify_hf_backbone_norm_embedding_position():
    model = small()
    optimizer, rows = ref.author_optimizer(model)
    mapping = {name: (r['lr'], r['weight_decay']) for r in rows for name in r['names']}
    assert set(mapping) == {name for name, _ in model.named_parameters()}
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids) == len(set(ids)) == len(mapping)
    prefix = 'model.pixel_level_module.encoder.'
    for name, (lr, decay) in mapping.items():
        assert lr == (1e-5 if name.startswith(prefix) else 1e-4)
        if 'relative_position_bias_table' in name:
            assert decay == 0
    assert mapping['class_predictor.weight'] == (1e-4, .05)
    assert mapping['class_predictor.bias'] == (1e-4, .05)
    assert mapping[prefix+'embeddings.norm.weight'] == (1e-5, 0)
    embedding_names = [name+'.weight' for name, module in model.named_modules()
                       if isinstance(module, torch.nn.Embedding)]
    assert embedding_names and all(mapping[name][1] == 0 for name in embedding_names)
    assert all(g['betas'] == (.9, .999) and g['eps'] == 1e-8 for g in optimizer.param_groups)
    next(model.parameters()).requires_grad_(False)
    with pytest.raises(ValueError, match='all model parameters'):
        ref.author_optimizer(model)


def test_swin_recomputation_preserves_stochastic_forward_and_parameter_gradients():
    original = small().model.pixel_level_module.encoder.train()
    copied = copy.deepcopy(original)
    shell = SimpleNamespace(model=SimpleNamespace(pixel_level_module=SimpleNamespace(encoder=copied)))
    names = ref.enable_swin_checkpointing(shell)
    assert len(names) == 4
    calls = [0, 0, 0, 0]
    for i, stage in enumerate(copied.encoder.layers):
        def hook(module, args, i=i):
            calls[i] += 1
        stage.register_forward_pre_hook(hook)
    image = torch.rand(1, 3, 64, 64)
    losses = []
    for encoder in [original, copied]:
        torch.manual_seed(52)
        features = encoder(image).feature_maps
        loss = sum(feature.square().mean() for feature in features)
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    assert all(count >= 2 for count in calls)
    for a, b in zip(original.parameters(), copied.parameters(), strict=True):
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_fixed_crop_context_keeps_ignore_and_hf_normalized_zero_padding():
    rgb = np.full((989, 1320, 3), 128, np.uint8)
    label = np.zeros((989, 1320), np.uint8)
    label[:, 660:] = 2
    valid = np.ones((989, 1320), bool)
    valid[:200] = False
    label[500:510] = 255
    original = label.copy()
    rows = ref.fixed_preflight_inputs(rgb, label, valid)
    assert np.array_equal(label, original)
    assert rows['crop']['padded_hw'] == [768, 768]
    assert rows['context']['padded_hw'] == [768, 1056]
    assert rows['context']['visible_hw'] == [768, 1025]
    for row in rows.values():
        assert torch.all(row['labels'][~row['valid']] == 255)
        assert set(row['labels'].unique().tolist()) == {0, 2, 255}
    assert not rows['context']['pixels'][..., 1025:].any()
    assert not rows['context']['valid'][:, 1025:].any()
    assert not rows['crop']['valid'][:90].any()


def test_verify_completed_model_requires_pinned_receipt_and_actual_bytes(tmp_path, monkeypatch):
    for name in ref.MODEL_FILES:
        (tmp_path/name).write_bytes(name.encode())
    monkeypatch.setattr(ref, 'WEIGHT_SHA', ref.digest(tmp_path/'model.safetensors'))
    monkeypatch.setattr(ref, 'WEIGHT_BYTES', (tmp_path/'model.safetensors').stat().st_size)
    receipt = {'status': 'completed', 'provider': 'HuggingFace', 'model_id': ref.MODEL_ID,
               'revision': ref.REVISION, 'model_dir': str(tmp_path),
               'files_sha256': {name: ref.digest(tmp_path/name) for name in ref.MODEL_FILES}}
    path = tmp_path/'download_provenance.json'
    path.write_text(json.dumps(receipt))
    assert len(ref.verify_model(tmp_path)) == 4
    (tmp_path/'config.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA changed'):
        ref.verify_model(tmp_path)
    receipt['revision'] = 'main'
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='pinned'):
        ref.verify_model(tmp_path)


def test_preflight_schedule_and_gpu_unknown_are_bounded_and_fail_closed(monkeypatch):
    path = Path(__file__).resolve().parents[1]/'scripts/preflight_mask2former_reference.py'
    spec = importlib.util.spec_from_file_location('mask2former_preflight_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SPEC['steps'] == ['crop', 'context', 'crop', 'context']
    assert module.SPEC['maximum_render_calls'] == 1
    assert module.SPEC['accuracy_scoring'] is False
    assert module.SPEC['amp'] == 'cuda bfloat16; FP32 parameters/optimizer/criterion; no loss scaling'
    for value in ['123\n', 'N/A\n', 'Not Supported\n']:
        monkeypatch.setattr(module.subprocess, 'run', lambda *a, value=value, **k: SimpleNamespace(stdout=value))
        with pytest.raises(ValueError, match='clients present or query unknown'):
            module.gpu_idle()
