"""Read only a completed fixed6000 endpoint, replay CPU sampling, audit provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

PLAN_PATH = Path('/mnt/data/SHM2026/runs/mask2former_reference_v1/preparation/plan.json')
PLAN_SHA = 'de7fc9b821e3700066a9db18107155d1cd6a46ad38f4873085f473384f0e7600'
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def completed_receipt(directory):
    receipt = read(Path(directory)/'execution_receipt.json')
    require(receipt.get('status') == 'completed' and receipt.get('completed_step') == 6000
            and receipt.get('successful_updates') == 6000, 'Wait for completed6000; checkpoint remains unopened')
    require(receipt.get('resume_checkpoint') is None, 'This audit binds the fresh fixed reference, not a resumed variant')
    return receipt


def replay_trace(names, seed, total=6000):
    """Independent NumPy reproduction of frozen view shuffle/crop/flip draws."""
    require(len(names) == len(set(names)) == 259 and names == sorted(names), 'Require sorted259 labeled TRAIN names')
    views, augmentation = np.random.default_rng(seed), np.random.default_rng(seed+1)
    order, cursor, trace = [], 0, []
    for step in range(1, total+1):
        if cursor == len(order):
            order, cursor = views.permutation(names).tolist(), 0
        name = order[cursor]
        cursor += 1
        if step % 2:
            y, x = int(augmentation.integers(0, 222)), int(augmentation.integers(0, 553))
            transform = {'mode': 'crop', 'x': x, 'y': y, 'visible_hw': [768, 768]}
        else:
            transform = {'mode': 'context', 'visible_hw': [768, 1025]}
        trace.append({'step': step, 'view': name, **transform, 'flip': bool(augmentation.random() < .5)})
    sampler = {'names': names.copy(), 'order': order, 'cursor': cursor, 'rng': views.bit_generator.state}
    return trace, sampler, augmentation.bit_generator.state


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), 'Nonfinite endpoint tensor')
        return 1
    if isinstance(value, dict):
        return sum(finite_tensors(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(finite_tensors(v) for v in value)
    return 0


def optimizer_contract(state, groups, step=6000):
    optimizer = state['optimizer']
    require(len(optimizer['param_groups']) == len(groups) == 4, 'Expected four author LR/decay groups')
    all_ids, names = [], []
    for actual, declared in zip(optimizer['param_groups'], groups, strict=True):
        require(len(actual['params']) == len(declared['names']), 'Optimizer name/ID alignment differs')
        require(actual['reference_base_lr'] == declared['lr'] and actual['weight_decay'] == declared['weight_decay']
                and actual['betas'] == (.9, .999) and actual['eps'] == 1e-8,
                'Optimizer hyperparameters differ')
        require(np.isclose(actual['lr'], declared['lr']*(1-(step-1)/6000)**.9, rtol=1e-12, atol=0),
                'Final poly LR differs')
        for identity, name in zip(actual['params'], declared['names'], strict=True):
            require(name in state['model'] and identity in optimizer['state'], 'Missing optimizer/model parameter')
            value, slot = state['model'][name], optimizer['state'][identity]
            require(value.dtype == torch.float32 and float(slot['step']) == step, 'Skipped update or non-FP32 parameter')
            for key in ['exp_avg', 'exp_avg_sq']:
                require(slot[key].shape == value.shape and slot[key].dtype == torch.float32, 'Adam moment shape/dtype differs')
            all_ids.append(identity)
            names.append(name)
    require(len(set(all_ids)) == len(all_ids) == len(optimizer['state'])
            and set(all_ids) == set(optimizer['state']) and len(set(names)) == len(names),
            'Optimizer coverage duplicates or orphan state')
    return {'parameter_tensors': len(names), 'parameters': sum(state['model'][name].numel() for name in names),
            'all_adam_parameter_steps': step, 'all_moments_shape_dtype_verified': True}


def stage_deltas(state, receipt, model_dir):
    """Only official four query tensors are decoded, not a new model initialization."""
    rows = {}
    with safe_open(str(Path(model_dir)/'model.safetensors'), framework='pt', device='cpu') as source:
        for index in range(4):
            stage = f'stage{index+1}'
            name = f'model.pixel_level_module.encoder.encoder.layers.{index}.blocks.0.attention.self.query.weight'
            initial, final = source.get_tensor(name), state['model'][name]
            require(initial.shape == final.shape and initial.dtype == final.dtype == torch.float32,
                    'Swin stage probe schema differs')
            delta = float((final-initial).abs().sum())
            expected = receipt['stage_update_probes'][stage]
            require(expected['finite'] is True and np.isfinite(delta) and delta > 0
                    and np.isclose(delta, expected['delta_l1'], rtol=1e-5, atol=1e-5),
                    'Stage final/pretrained delta differs from receipt')
            rows[stage] = {'parameter': name, 'cpu_delta_l1': delta,
                           'runtime_delta_l1': expected['delta_l1'], 'changed_elements': int((final != initial).sum())}
    return rows


def audit(output):
    require(not torch.cuda.is_initialized(), 'CPU-only audit required')
    # The plan itself is immutable and safe to read before the mutable checkpoint guard.
    require(digest(PLAN_PATH) == PLAN_SHA, 'Frozen plan SHA differs')
    plan = read(PLAN_PATH)
    cfg, folder = plan['configuration'], Path(plan['configuration']['output'])
    receipt = completed_receipt(folder)  # No last.pt stat/hash/load before this line.
    output = Path(output).resolve()
    require(not output.exists(), 'Do not overwrite existing endpoint audit')
    torch.set_num_threads(8)
    require(receipt['plan_sha256'] == PLAN_SHA and receipt['configuration'] == cfg
            and receipt['source_hashes'] == plan['source_hashes'] and receipt['all_inputs_sources_unchanged'] is True,
            'Runtime plan/config/source binding differs')
    require(cfg['protocol']['steps'] == 6000 and cfg['protocol']['seed'] == 20260805
            and cfg['protocol']['validation_during_training'] is False
            and receipt['validation_during_training'] is False, 'Training protocol changed')
    snapshot = Path(plan['source_snapshot'])
    source_hashes = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')}
    require(source_hashes == plan['source_hashes'], 'Frozen source bytes changed')
    for path, sha in plan['input_hashes'].items():
        require(digest(path) == sha, f'Bound source/input changed: {path}')
    checkpoint, trace_path, log_path = folder/'last.pt', folder/'trace.jsonl', folder/'train.jsonl'
    require(receipt['checkpoint'] == str(checkpoint) and digest(checkpoint) == receipt['checkpoint_sha256']
            and digest(trace_path) == receipt['trace_sha256'] and digest(log_path) == receipt['log_sha256'],
            'Endpoint/trace/log SHA differs')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    require(state['format'] == 'rendered_mask2former_reference_v1' and state['step'] == 6000
            and state['training_completed'] is True and state['training_config'] == cfg
            and state['pixel_protocol'] == 'legacy_mixed_v1', 'Wrong final checkpoint contract')
    provenance = state['provenance']
    require(provenance['input_hashes'] == plan['input_hashes'] and provenance['source_hashes'] == source_hashes
            and provenance['source_snapshot'] == plan['source_snapshot']
            and provenance['base_checkpoint'] == cfg['base_checkpoint']
            and provenance['base_sha256'] == digest(cfg['base_checkpoint'])
            and provenance['manifest'] == cfg['manifest'] and provenance['manifest_sha256'] == digest(cfg['manifest']),
            'Checkpoint dependency provenance differs')
    manifest = read(cfg['manifest'])
    names = sorted(v['name'] for v in manifest['views'] if v['split'] == 'train' and v.get('mask_path'))
    cache = read(cfg['cache_receipt'])
    require(sorted(plan['train_views']) == names == cache['view_names'] and cache['status'] == 'completed'
            and cache['checkpoint_sha256'] == provenance['base_sha256']
            and cache['manifest_sha256'] == provenance['manifest_sha256'], '259 same-field labeled TRAIN cache binding differs')
    expected, sampler, augmentation = replay_trace(names, cfg['protocol']['seed'])
    observed = [json.loads(line) for line in trace_path.read_text().splitlines()]
    require(observed == state['trace'] == expected, 'Observed full6000 view/crop/context/flip trace differs')
    require(canonical(state['sampler_state']) == canonical(sampler)
            and canonical(state['augmentation_rng_state']) == canonical(augmentation), 'Terminal independent sampler/augmentation RNG differs')
    logs = [json.loads(line) for line in log_path.read_text().splitlines()]
    require(logs == state['loss_logs'] and [row['step'] for row in logs] == [1]+list(range(100, 6001, 100)),
            'Incomplete100-step logging')
    for row in logs:
        require(all(np.isfinite(row[k]) for k in ['loss', 'grad_norm_before_clip', 'lr_multiplier', 'elapsed_seconds'])
                and all(row['stage_gradient_probe_finite'].values()), 'Logged nonfinite gradient/loss')
        require({k: row[k] for k in expected[row['step']-1]} == expected[row['step']-1], 'Loss log is not aligned to trace')
    tensors = finite_tensors(state['model'])+finite_tensors(state['optimizer'])
    require(state['class_names'] == manifest['class_names'] and state['model']['class_predictor.weight'].shape[0] == 6
            and state['model']['class_predictor.bias'].shape == (6,), 'Five semantic plus no-object classifier differs')
    optimizer = optimizer_contract(state, receipt['optimizer_groups'])
    probes = stage_deltas(state, receipt, cfg['model_dir'])
    rng_shapes = {}
    for key in ['point_rng_state', 'torch_rng_state', 'cuda_rng_state']:
        value = state[key]
        require(value.device.type == 'cpu' and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0,
                f'Invalid stored RNG: {key}')
        rng_shapes[key] = list(value.shape)
    require(len(receipt['checkpointed_stages']) == 4 and not torch.cuda.is_initialized(), 'Stage checkpointing/CPU contract differs')
    # Recheck endpoint/source inputs after all CPU reads; never mutate checkpoint or cache.
    for path, sha in plan['input_hashes'].items():
        require(digest(path) == sha, 'Input changed during audit')
    require(digest(checkpoint) == receipt['checkpoint_sha256'], 'Checkpoint changed during audit')
    report = {'status': 'passed', 'cpu_only': True, 'plan': str(PLAN_PATH), 'plan_sha256': PLAN_SHA,
              'checkpoint': str(checkpoint), 'checkpoint_sha256': receipt['checkpoint_sha256'],
              'receipt_sha256': digest(folder/'execution_receipt.json'), 'script_sha256': digest(__file__),
              'source_hashes': source_hashes, 'input_hashes_verified': len(plan['input_hashes']),
              'train_view_names': names, 'steps': 6000, 'full_observed_trace_matches_independent_replay': True,
              'terminal_sampler_and_augmentation_rng_exact': True, 'trace_sha256': receipt['trace_sha256'],
              'crop_updates': 3000, 'context_updates': 3000, 'flipped_updates': sum(row['flip'] for row in expected),
              'all_model_optimizer_tensors_finite': True, 'finite_tensor_count': tensors,
              'optimizer': optimizer, 'stage_probe_deltas': probes, 'stored_rng_shapes': rng_shapes,
              'elapsed_seconds': receipt['elapsed_seconds'],
              'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in logs),
              'peak_reserved_bytes': max(row['peak_reserved_bytes'] for row in logs),
              'loss_first': logs[0]['loss'], 'loss_last': logs[-1]['loss'],
              'no_skip_evidence': 'Contiguous6000 post-step observed trace, source step-after-Adam ordering, all Adam parameter step counters=6000; zero drop-path probe gradients are legal.',
              'limits': 'No GPU or accuracy scoring. CPU RNG replay covers views/augmentation, not point-generator or CUDA dropout bitwise replay. Four stage probes are fixed query weights; complete stage audit was preflight only. TRAIN losses are different sampled views and not a quality benchmark.'}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('/mnt/data/SHM2026/runs/mask2former_reference_v1/endpoint_audit.json'))
    args = parser.parse_args()
    report = audit(args.output)
    print(json.dumps({'status': report['status'], 'steps': report['steps'],
                      'checkpoint_sha256': report['checkpoint_sha256'], 'stage_probe_deltas': report['stage_probe_deltas']}, indent=2))


if __name__ == '__main__':
    main()
