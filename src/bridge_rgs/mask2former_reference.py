"""Strict initialization and author-style optimization for the CPU/GPU preflight.

No model downloads, training loop, or site-packages patches live here. The
valid-support criterion remains a separate, explicitly versioned adapter.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

MODEL_ID = "facebook/mask2former-swin-large-ade-semantic"
REVISION = "aa25c92404a40599614215e76514c79b427c7527"
WEIGHT_SHA = "b143c144341c15b4f20165cc6d2c9305fb1b66792f68a6e0e06d2b20dc063b14"
WEIGHT_BYTES = 866052064
MODEL_FILES = {"model.safetensors", "config.json", "preprocessor_config.json"}
REPLACED_KEYS = frozenset({"class_predictor.weight", "class_predictor.bias", "criterion.empty_weight"})
CLASS_NAMES = ["background", "deck", "stay_cable", "tower", "foundation"]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_model(directory):
    directory = Path(directory).resolve()
    path = directory / "download_provenance.json"
    receipt = json.loads(path.read_text())
    hashes = receipt.get('files_sha256', {})
    if (receipt.get('status') != 'completed' or receipt.get('provider') != 'HuggingFace'
            or receipt.get('model_id') != MODEL_ID or receipt.get('revision') != REVISION
            or receipt.get('model_dir') != str(directory) or set(hashes) != MODEL_FILES
            or hashes.get('model.safetensors') != WEIGHT_SHA):
        raise ValueError('Require the completed pinned Swin-L ADE download')
    for name, expected in hashes.items():
        if digest(directory/name) != expected:
            raise ValueError(f'Model file SHA changed: {name}')
    if (directory/'model.safetensors').stat().st_size != WEIGHT_BYTES:
        raise ValueError('Weight byte count changed')
    return {str(directory/name): sha for name, sha in hashes.items()} | {str(path): digest(path)}


def transplant_ade_state(model, source, pretrained_classes=150):
    """Load every tensor exactly except enumerated C+1 head/CE buffer changes.

    Reinitialize all six class rows, including no-object, using the model's
    normal HF initialization. Do not infer or copy ADE class correspondences.
    Validated entirely before mutation; all reused tensors must retain dtype.
    """
    target = model.state_dict()
    if set(source) != set(target):
        raise ValueError(f'State key mismatch: missing={sorted(set(target)-set(source))}; '
                         f'unexpected={sorted(set(source)-set(target))}')
    expected_shapes = {'class_predictor.weight': (pretrained_classes+1, model.config.hidden_dim),
                       'class_predictor.bias': (pretrained_classes+1,),
                       'criterion.empty_weight': (pretrained_classes+1,)}
    for name, value in source.items():
        expected = expected_shapes[name] if name in REPLACED_KEYS else tuple(target[name].shape)
        if (tuple(value.shape) != expected or value.dtype != target[name].dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f'State shape/dtype/finite mismatch: {name}')
    initial = {name: target[name].clone() for name in REPLACED_KEYS}
    loaded = model.load_state_dict({k: v for k, v in source.items() if k not in REPLACED_KEYS},
                                   strict=False)
    if set(loaded.missing_keys) != REPLACED_KEYS or loaded.unexpected_keys:
        raise ValueError('Unexpected strict-load result')
    actual = model.state_dict()
    if any(not torch.equal(actual[k], v) for k, v in source.items() if k not in REPLACED_KEYS):
        raise ValueError('Reused pretrained tensor differs after loading')
    if any(not torch.equal(actual[k], v) for k, v in initial.items()):
        raise ValueError('New class head or CE buffer changed during loading')
    return {'loaded_tensor_count': len(source)-len(REPLACED_KEYS),
            'reinitialized_keys': sorted(REPLACED_KEYS), 'pretrained_classes': pretrained_classes,
            'target_classes': model.config.num_labels, 'reused_tensors_exact': True}


def load_reference(directory):
    hashes = verify_model(directory)
    cfg = Mask2FormerConfig.from_json_file(str(Path(directory)/'config.json'))
    b = cfg.backbone_config
    if (cfg.num_labels != 150 or cfg.num_queries != 100 or cfg.hidden_dim != 256
            or b.model_type != 'swin' or b.embed_dim != 192 or b.depths != [2, 2, 18, 2]
            or b.num_heads != [6, 12, 24, 48] or b.window_size != 12):
        raise ValueError('Expected fixed ADE Swin-L architecture')
    cfg = copy.deepcopy(cfg)
    cfg.num_labels = 5
    cfg.id2label = dict(enumerate(CLASS_NAMES))
    cfg.label2id = {name: i for i, name in enumerate(CLASS_NAMES)}
    model = Mask2FormerForUniversalSegmentation(cfg)
    audit = transplant_ade_state(model, load_file(str(Path(directory)/'model.safetensors'), device='cpu'))
    if verify_model(directory) != hashes:
        raise ValueError('Pretrained model changed during load')
    return model, audit


def author_optimizer(model):
    """Group HF parameter paths by the author's LR/decay rules, not substrings.

    Backbone LR .1x, AdamW beta(.9,.999)/eps1e-8; norm, embedding and position
    parameters have zero decay. The runner separately applies global clip .01.
    Four preflight steps use constant LR; no 6k scheduler is implied.
    """
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
                  nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                  nn.LayerNorm, nn.LocalResponseNorm)
    backbone = 'model.pixel_level_module.encoder.'
    groups, names, seen = {}, {}, set()
    for module_name, module in model.named_modules():
        for leaf_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                raise ValueError('Swin-L reference requires all model parameters trainable')
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            name = f'{module_name}.{leaf_name}' if module_name else leaf_name
            lr = 1e-5 if name.startswith(backbone) else 1e-4
            no_decay = (isinstance(module, (*norm_types, nn.Embedding))
                        or 'relative_position_bias_table' in leaf_name or 'absolute_pos_embed' in leaf_name)
            decay = 0. if no_decay else .05
            groups.setdefault((lr, decay), []).append(parameter)
            names.setdefault((lr, decay), []).append(name)
    if seen != {id(p) for p in model.parameters()}:
        raise ValueError('Optimizer does not cover each model parameter exactly once')
    optimizer = torch.optim.AdamW([{'params': parameters, 'lr': lr, 'weight_decay': decay}
                                   for (lr, decay), parameters in groups.items()],
                                  betas=(.9, .999), eps=1e-8)
    audit = [{'lr': lr, 'weight_decay': decay, 'names': names[(lr, decay)],
              'parameters': sum(p.numel() for p in parameters)}
             for (lr, decay), parameters in groups.items()]
    return optimizer, audit


def enable_swin_checkpointing(model):
    encoder = model.model.pixel_level_module.encoder
    encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False, 'preserve_rng_state': True})
    layers = encoder.encoder.layers
    if len(layers) != 4 or not all(layer.gradient_checkpointing for layer in layers):
        raise ValueError('All four Swin stages must actually enable checkpointing')
    return [f'model.pixel_level_module.encoder.encoder.layers.{i}' for i in range(4)]


def fixed_preflight_inputs(rgb, labels, valid):
    """Center crop / Pillow legacy full context; normalize then zero-pad like HF."""
    if (rgb.shape != (989, 1320, 3) or rgb.dtype != np.uint8 or labels.shape != (989, 1320)
            or labels.dtype != np.uint8 or valid.shape != labels.shape or valid.dtype != bool):
        raise ValueError('Require aligned native TRAIN002 uint8 RGB/label and bool validity')
    if not np.isin(labels, [0, 1, 2, 3, 4, 255]).all():
        raise ValueError('Unexpected original five-class mask')
    labels = labels.copy()
    labels[~valid] = 255
    rows = [('crop', rgb[110:878, 276:1044], labels[110:878, 276:1044],
             valid[110:878, 276:1044]),
            ('context', np.asarray(Image.fromarray(rgb).resize((1025, 768), Image.Resampling.BILINEAR)),
             np.asarray(Image.fromarray(labels).resize((1025, 768), Image.Resampling.NEAREST)),
             np.asarray(Image.fromarray(valid).resize((1025, 768), Image.Resampling.NEAREST)))]
    result = {}
    for mode, image, mask, keep in rows:
        height, width = mask.shape
        padding = (0, (-width) % 32, 0, (-height) % 32)
        pixels = torch.from_numpy(image.copy()).permute(2, 0, 1).float()/255
        mean = pixels.new_tensor([.485, .456, .406])[:, None, None]
        std = pixels.new_tensor([.229, .224, .225])[:, None, None]
        pixels = F.pad((pixels-mean)/std, padding)[None]
        label = F.pad(torch.from_numpy(mask.copy()).long(), padding, value=255)
        support = F.pad(torch.from_numpy(keep.copy()).bool(), padding, value=False) & (label != 255)
        result[mode] = {'pixels': pixels, 'labels': label, 'valid': support,
                        'visible_hw': [height, width], 'padded_hw': list(label.shape)}
    return result
