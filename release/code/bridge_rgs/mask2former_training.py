"""Fixed rendered-RGB Mask2Former reference protocol; no Gaussian/teacher fusion."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

from .mask2former_reference import CLASS_NAMES

FORMAT = 'rendered_mask2former_reference_v1'
TRAIN_PROTOCOL = {
    'steps': 6000, 'seed': 20260805, 'batch_size': 1,
    'odd_input': 'uniform native crop768', 'even_input': 'legacy full context1025x768',
    'horizontal_flip_probability': .5, 'poly_power': .9,
    'optimizer': 'AdamW lr1e-4 backbone0.1x decay0.05 norm_embedding_position0 beta0.9_0.999 eps1e-8',
    'gradient_clip': .01, 'autocast_dtype': 'bfloat16', 'loss_scaling': False,
    'valid_support': 'valid_gt_pixel_centers_v1', 'swin_checkpoint_stages': 4,
    'checkpoint_every': 1000, 'loss_log_every': 100, 'hard_budget_seconds': 2400,
    'validation_during_training': False, 'pixel_protocol': 'legacy_mixed_v1',
}
INFERENCE_PROTOCOL = {
    'id': 'rendered_mask2former_tile768_stride512_no_tta_v1',
    'tile_size': 768, 'stride': 512, 'flip': False, 'context_blend': False,
    'input': 'uint8 RGB from this fixed scene pinhole canvas; ImageNet normalization; normalized zero pad',
    'scores': 'FP32 bilinear mask-logit upsample to padded tile, sigmoid, query sum with class softmax excluding no-object',
    'overlap': 'uniform sum raw class scores / overlap count; then normalize five-class sum per pixel',
    'warp': 'original-grid bilinear soft probabilities before argmax, shared official helper',
}


def validate_config(config):
    required = {'protocol', 'base_checkpoint', 'manifest', 'model_dir', 'cache_receipt',
                'output', 'selected_reference_receipt'}
    if set(config) != required or config['protocol'] != TRAIN_PROTOCOL:
        raise ValueError('Require the fixed rendered Mask2Former configuration')
    for name in required - {'protocol'}:
        if not Path(config[name]).is_absolute():
            raise ValueError(f'Use an absolute {name} path')
    return copy.deepcopy(config)


class ShuffledViews:
    """Independent permutation RNG and exact cursor for full continuation state."""
    def __init__(self, names, seed):
        self.names = list(names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError('Require nonempty unique view names')
        self.rng = np.random.default_rng(seed)
        self.order, self.cursor = [], 0

    def next(self):
        if self.cursor == len(self.order):
            self.order = self.rng.permutation(self.names).tolist()
            self.cursor = 0
        value = self.order[self.cursor]
        self.cursor += 1
        return value

    def state_dict(self):
        return {'names': self.names.copy(), 'order': self.order.copy(), 'cursor': self.cursor,
                'rng': copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        if (state['names'] != self.names or sorted(state['order']) != sorted(self.names)
                or not 0 <= state['cursor'] <= len(self.names)):
            raise ValueError('Sampler resume view set/cursor differs')
        self.order, self.cursor = state['order'].copy(), state['cursor']
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])


def normalize_rgb(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('Expected uint8 HWC RGB')
    pixels = torch.from_numpy(image.copy()).permute(2, 0, 1).float()/255
    mean = pixels.new_tensor([.485, .456, .406])[:, None, None]
    std = pixels.new_tensor([.229, .224, .225])[:, None, None]
    h, w = image.shape[:2]
    return F.pad((pixels-mean)/std, (0, (-w) % 32, 0, (-h) % 32))[None]


def training_sample(rgb, labels, valid, step, rng):
    """One crop/context and flip draw, solely from the independent TRAIN RNG."""
    if (rgb.shape != (989, 1320, 3) or rgb.dtype != np.uint8 or labels.shape != rgb.shape[:2]
            or labels.dtype != np.uint8 or valid.shape != labels.shape or valid.dtype != bool
            or not np.isin(labels, [0, 1, 2, 3, 4, 255]).all() or step < 1):
        raise ValueError('Expected aligned native bridge TRAIN image/labels/valid')
    labels = labels.copy()
    labels[~valid] = 255
    if step % 2:
        y, x = int(rng.integers(0, 989-768+1)), int(rng.integers(0, 1320-768+1))
        rgb, labels, valid = (a[y:y+768, x:x+768] for a in (rgb, labels, valid))
        transform = {'mode': 'crop', 'x': x, 'y': y, 'visible_hw': [768, 768]}
    else:
        rgb = np.asarray(Image.fromarray(rgb).resize((1025, 768), Image.Resampling.BILINEAR))
        labels = np.asarray(Image.fromarray(labels).resize((1025, 768), Image.Resampling.NEAREST))
        valid = np.asarray(Image.fromarray(valid).resize((1025, 768), Image.Resampling.NEAREST))
        transform = {'mode': 'context', 'visible_hw': [768, 1025]}
    flip = bool(rng.random() < .5)
    if flip:
        rgb, labels, valid = (np.flip(a, 1) for a in (rgb, labels, valid))
    h, w = labels.shape
    padding = (0, (-w) % 32, 0, (-h) % 32)
    target = F.pad(torch.from_numpy(labels.copy()).long(), padding, value=255)
    support = F.pad(torch.from_numpy(valid.copy()).bool(), padding, value=False) & (target != 255)
    if not support.any():
        raise ValueError('Fixed sampling produced an all-invalid sample; stop without resampling')
    return normalize_rgb(rgb), target, support, dict(transform, flip=flip)


def poly_learning_rate(optimizer, step, total=6000):
    if not 1 <= step <= total:
        raise ValueError('Step outside declared poly schedule')
    multiplier = (1-(step-1)/total)**.9
    for group in optimizer.param_groups:
        if 'reference_base_lr' not in group:
            group['reference_base_lr'] = group['lr']
        group['lr'] = group['reference_base_lr']*multiplier
    return multiplier


def stage_gradient_status(probes):
    """A dropped residual branch has a valid zero gradient, not a failed update."""
    finite = {name: p.grad is not None and bool(torch.isfinite(p.grad).all())
              for name, p in probes.items()}
    nonzero = {name: p.grad is not None and bool((p.grad != 0).any())
               for name, p in probes.items()}
    return finite, nonzero


def tile_origins(size, tile=768, stride=512):
    if size < 1 or not 0 < stride <= tile:
        raise ValueError('Invalid tile geometry')
    positions = list(range(0, max(size-tile+1, 1), stride))
    last = max(0, size-tile)
    if positions[-1] != last:
        positions.append(last)
    return positions


def query_scores(class_logits, mask_logits, padded_hw):
    """Standard query semantic scores; no-object omitted, no argmax in this path."""
    if class_logits.ndim != 3 or class_logits.shape[-1] != 6 or mask_logits.ndim != 4:
        raise ValueError('Expected batch/query six-class logits and mask logits')
    classes = class_logits.float().softmax(-1)[..., :5]
    masks = F.interpolate(mask_logits.float(), size=padded_hw, mode='bilinear', align_corners=False).sigmoid()
    return torch.einsum('bqc,bqhw->bchw', classes, masks)


@torch.inference_mode()
def predict_tiled(model, rgb):
    """Uniform overlapping scores then class normalization; no flip/context TTA."""
    device = next(model.parameters()).device
    h, w = rgb.shape[:2]
    sums, counts = torch.zeros((5, h, w), device=device), torch.zeros((h, w), device=device)
    for y in tile_origins(h):
        for x in tile_origins(w):
            image = rgb[y:y+768, x:x+768]
            pixels = normalize_rgb(image).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                output = model(pixels, output_auxiliary_logits=False)
            scores = query_scores(output.class_queries_logits, output.masks_queries_logits, pixels.shape[-2:])[0]
            ih, iw = image.shape[:2]
            sums[:, y:y+ih, x:x+iw] += scores[:, :ih, :iw]
            counts[y:y+ih, x:x+iw] += 1
    if not bool((counts > 0).all()):
        raise ValueError('Uncovered tile pixels')
    scores = sums/counts[None]
    mass = scores.sum(0, keepdim=True)
    if not bool(torch.isfinite(scores).all()) or not bool((mass > 0).all()):
        raise ValueError('Nonfinite/zero-mass query score')
    return (scores/mass).permute(1, 2, 0).cpu().numpy()


def validate_trained_state(state):
    if (state.get('format') != FORMAT or state.get('step') != TRAIN_PROTOCOL['steps']
            or state.get('training_completed') is not True or state.get('pixel_protocol') != 'legacy_mixed_v1'
            or state.get('class_names') != CLASS_NAMES):
        raise ValueError('Require fixed final6000 Mask2Former reference checkpoint')
    validate_config(state['training_config'])
    config = Mask2FormerConfig.from_dict(state['model_config'])
    if config.num_labels != 5 or config.num_queries != 100:
        raise ValueError('Unexpected final class/query schema')
    return config


def load_trained_reference(checkpoint, device='cuda'):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = validate_trained_state(state)
    model = Mask2FormerForUniversalSegmentation(config)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval().requires_grad_(False), state
