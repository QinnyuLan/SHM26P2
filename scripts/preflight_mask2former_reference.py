"""Prepare a fixed four-update TRAIN render preflight; worker requires GPU handoff."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from bridge_rgs.mask2former_reference import (
    author_optimizer,
    digest,
    enable_swin_checkpointing,
    fixed_preflight_inputs,
    load_reference,
    verify_model,
)

BASE_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
VALID_SHA = '44d0a621b67b8b05e38a54a12fe8f9de748f4f393747a9f87050949af59d3c00'
SPEC = {'protocol': 'swinl_ade_valid_support_four_update_preflight_v1',
        'view': '002.png', 'split': 'train', 'pixel_protocol': 'legacy_mixed_v1',
        'seed': 20260927, 'steps': ['crop', 'context', 'crop', 'context'],
        'padded_hw': {'crop': [768, 768], 'context': [768, 1056]},
        'visible_hw': {'crop': [768, 768], 'context': [768, 1025]},
        'render': 'fresh selected H3 field; native checkpoint TRAIN camera; clamp and round uint8 RGB',
        'crop': 'center x276 y110; no random transform or flip',
        'context': 'Pillow bilinear RGB / nearest labels+valid, legacy 1025x768; normalized zero pad',
        'sampling': 'valid_gt_pixel_centers_v1', 'optimizer': 'AdamW author LR/decay groups; constant 4-step LR',
        'gradient_clip': .01, 'amp': 'cuda bfloat16; FP32 parameters/optimizer/criterion; no loss scaling',
        'checkpointing': 'all four Swin stages; non-reentrant; preserve RNG',
        'process_seconds': 600, 'maximum_render_calls': 1, 'accuracy_scoring': False,
        'checkpoint_or_probability_written': False,
        'limits': 'One TRAIN view, four optimizer attempts; any skipped update fails. Not 6k training or validation.'}


def write_json(path, value, replace=False):
    path = Path(path)
    if replace:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)
    else:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')


def tensor_hash(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


def gpu_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True, timeout=5)
    if result.stdout.strip():
        raise ValueError(f'GPU compute clients present or query unknown: {result.stdout.strip()}')


def dependency_sources():
    modules = ['transformers.models.mask2former.modeling_mask2former',
               'transformers.models.swin.modeling_swin', 'transformers.modeling_layers']
    return {str(Path(importlib.import_module(name).__file__).resolve()):
            digest(importlib.import_module(name).__file__) for name in modules}


def prepare(root, output, model_dir):
    """CPU-only binding; no scene/model forward or tensor allocation on CUDA."""
    root, output, model_dir = Path(root).resolve(), Path(output).resolve(), Path(model_dir).resolve()
    if output.exists() or not output.is_relative_to(Path('/mnt/data')):
        raise ValueError('Use a new /mnt/data preflight directory')
    model_files = verify_model(model_dir)
    base, manifest_path = root/'runs/h3_moments/02_cross/last.pt', root/'artifacts/prepared/manifest.json'
    if digest(base) != BASE_SHA or digest(root/'src/bridge_rgs/mask2former_valid.py') != VALID_SHA:
        raise ValueError('Selected H3 or accepted valid criterion changed')
    from bridge_rgs.coordinates import LEGACY, pixel_protocol
    manifest = json.loads(manifest_path.read_text())
    state = torch.load(base, map_location='cpu', weights_only=False, mmap=True)
    train = sorted((v for v in manifest['views'] if v['split'] == 'train'), key=lambda v: v['name'])
    names = [v['name'] for v in train]
    if (pixel_protocol(state) != LEGACY or pixel_protocol(manifest) != LEGACY
            or len(names) != 350 or len(set(names)) != 350 or state['config'].get('optimize_cameras')
            or (root/state['config']['manifest']).resolve() != manifest_path
            or not torch.equal(state['training_cameras'], torch.tensor([v['w2c'] for v in train], dtype=torch.float32))):
        raise ValueError('Expected legacy selected field and exact sorted TRAIN camera binding')
    view = train[names.index(SPEC['view'])]
    if not view.get('mask_path') or (view['height'], view['width']) != (989, 1320):
        raise ValueError('Fixed labeled native TRAIN002 required')
    files = [base, manifest_path, root/'uv.lock', Path(view['mask_path']), Path(view['valid_path'])]
    bindings = {str(p): digest(p) for p in files} | model_files | dependency_sources()
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(root/'src/bridge_rgs', snapshot/'bridge_rgs',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    source = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')}
    plan = {'status': 'cpu_locked_waiting_root_gpu_handoff', 'specification': SPEC,
            'root': str(root), 'output': str(output), 'source_snapshot': str(snapshot),
            'base': str(base), 'base_sha256': BASE_SHA, 'manifest': str(manifest_path),
            'manifest_sha_observed': digest(manifest_path), 'base_declared_manifest_sha': state.get('manifest_sha256'),
            'model_dir': str(model_dir), 'view': view, 'training_camera_names': names,
            'training_cameras_sha256': tensor_hash(state['training_cameras']),
            'input_hashes': bindings, 'source_hashes': source,
            'gpu_execution': 'Only after explicit root handoff; timeout 660s around this frozen worker'}
    write_json(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    if plan['specification'] != SPEC or plan['status'] != 'cpu_locked_waiting_root_gpu_handoff':
        raise ValueError('Preflight specification/status changed')
    snapshot = Path(plan['source_snapshot'])
    if Path(__file__).resolve() != snapshot/Path(__file__).name:
        raise ValueError('Run the frozen worker')
    actual = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')}
    if actual != plan['source_hashes']:
        raise ValueError('Frozen source changed')
    for path, expected in plan['input_hashes'].items():
        if digest(path) != expected:
            raise ValueError(f'Bound input/dependency changed: {path}')
    imported = {}
    for name in ['mask2former_reference', 'mask2former_valid', 'train', 'model', 'coordinates']:
        path = Path(importlib.import_module('bridge_rgs.'+name).__file__).resolve()
        if path != snapshot/'bridge_rgs'/f'{name}.py':
            raise ValueError(f'Wrong package import: {path}')
        imported[name] = {'path': str(path), 'sha256': digest(path)}
    return imported


def render_inputs(plan):
    """Only camera-rendered RGB plus two fixed TRAIN label/valid files are read."""
    from bridge_rgs.train import load_scene
    scene, state = load_scene(plan['base'])
    scene.eval().requires_grad_(False)
    before = {k: tensor_hash(v) for k, v in scene.state_dict().items()}
    view = plan['view']
    pose = state['training_cameras'][plan['training_camera_names'].index(view['name'])].cuda()
    camera_sha = tensor_hash(state['training_cameras'])
    if camera_sha != plan['training_cameras_sha256']:
        raise ValueError('TRAIN camera tensor changed')
    with torch.no_grad():
        rgb = scene.render(torch.tensor(view['K'], device='cuda').float(), pose, 1320, 989,
                           degree=3, semantics=False, refine=False, absgrad=False)['rgb']
        raw_hash = tensor_hash(rgb)
        image = np.round(rgb.clamp(0, 1).cpu().numpy()*255).astype(np.uint8)
    after = {k: tensor_hash(v) for k, v in scene.state_dict().items()}
    if before != after or tensor_hash(state['training_cameras']) != camera_sha:
        raise ValueError('Frozen field or cameras changed during RGB rendering')
    audit = {'checkpoint_sha256': plan['base_sha256'], 'renderer_calls': 1,
             'raw_float_rgb_sha256': raw_hash, 'uint8_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
             'scene_state_before_after_sha256': before, 'all_scene_tensors_and_cameras_exact': True,
             'rgb_input_source': 'fresh selected scene render; no real RGB or other cache decoded'}
    labels = cv2.imread(view['mask_path'], cv2.IMREAD_UNCHANGED)
    valid = cv2.imread(view['valid_path'], cv2.IMREAD_GRAYSCALE) > 0
    inputs = fixed_preflight_inputs(image, labels, valid)
    del scene, state, rgb, pose
    gc.collect()
    torch.cuda.empty_cache()
    return inputs, audit


def gradient_summary(parameters):
    gradients = [p.grad for p in parameters if p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in gradients)
    return {'gradient_tensors': len(gradients), 'missing_gradient_tensors': len(parameters)-len(gradients),
            'finite': finite,
            'gradient_l1': sum(float(g.abs().sum()) for g in gradients) if finite else None}


def worker(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    output = Path(plan['output'])/'execution_receipt.json'
    if output.exists():
        raise FileExistsError('No automatic retry or overwrite of preflight')
    imported = verify(plan)
    gpu_idle()
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    torch.manual_seed(SPEC['seed'])
    torch.cuda.manual_seed_all(SPEC['seed'])
    started = time.monotonic()
    report = {'status': 'running', 'plan_sha256': digest(plan_path), 'actual_imports': imported,
              'pid': os.getpid(), 'optimizer_attempts': 0, 'successful_updates': 0, 'steps': [],
              'gpu': torch.cuda.get_device_name(), 'preflight_only': True}
    write_json(output, report)
    original_read = cv2.imread
    allowed = {str(Path(plan['view'][k]).resolve()) for k in ['mask_path', 'valid_path']}
    reads = []
    def guarded(path, *args, **kwargs):
        resolved = str(Path(path).resolve())
        if resolved not in allowed:
            raise ValueError(f'Forbidden pixel read: {resolved}')
        reads.append(resolved)
        return original_read(path, *args, **kwargs)
    cv2.imread = guarded
    try:
        inputs, report['render'] = render_inputs(plan)
        model, report['initialization'] = load_reference(plan['model_dir'])
        report['checkpointed_stages'] = enable_swin_checkpointing(model)
        model = model.cuda().train()
        optimizer, report['optimizer_groups'] = author_optimizer(model)
        from bridge_rgs.mask2former_valid import ValidSupportMask2FormerCriterion, semantic_targets
        criterion = ValidSupportMask2FormerCriterion(model.config).cuda()
        if not torch.cuda.is_bf16_supported():
            raise ValueError('This fixed preflight requires CUDA BF16 support')
        report['precision'] = {'autocast_dtype': 'bfloat16', 'parameters_optimizer_criterion': 'float32',
                               'loss_scaling': False, 'automatic_overflow_retry': False}
        generator = torch.Generator(device='cuda').manual_seed(SPEC['seed']+1)
        stages = {f'stage{i+1}': list(stage.parameters())
                  for i, stage in enumerate(model.model.pixel_level_module.encoder.encoder.layers)}
        stage_ids = {id(p) for parameters in stages.values() for p in parameters}
        stages['other_trainable'] = [p for p in model.parameters() if id(p) not in stage_ids]
        report['model_parameters'] = sum(p.numel() for p in model.parameters())
        report['peak_includes'] = 'Full optimizer, separate criterion and FP32 before-update parameter clones; scene freed before training'
        for number, mode in enumerate(SPEC['steps'], 1):
            if time.monotonic()-started >= SPEC['process_seconds']:
                raise TimeoutError('Internal 600s preflight limit')
            row = inputs[mode]
            if row['padded_hw'] != SPEC['padded_hw'][mode] or row['visible_hw'] != SPEC['visible_hw'][mode]:
                raise ValueError('Predefined crop/context size differs')
            pixels = row['pixels'].cuda()
            targets = semantic_targets([row['labels'].cuda()], [row['valid'].cuda()], 5)
            optimizer.zero_grad(set_to_none=True)
            before = {key: [p.detach().clone() for p in parameters] for key, parameters in stages.items()}
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            tick = time.monotonic()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                predictions = model(pixels, output_auxiliary_logits=True)
            loss = criterion(predictions, targets, generator=generator).loss
            loss.backward()
            torch.cuda.synchronize()
            forward_backward = time.monotonic()-tick
            gradient = {key: gradient_summary(parameters) for key, parameters in stages.items()}
            torch.cuda.synchronize()
            update_tick = time.monotonic()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), SPEC['gradient_clip'])
            nonfinite = (not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm))
                         or any(not value['finite'] for value in gradient.values()))
            if not nonfinite:
                optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.monotonic()-tick
            update_seconds = time.monotonic()-update_tick
            changes = {key: {'parameter_delta_l1': sum(float((p.detach()-old).abs().sum())
                                                      for p, old in zip(parameters, before[key], strict=True)),
                             'changed_tensors': sum(not torch.equal(p.detach(), old)
                                                    for p, old in zip(parameters, before[key], strict=True))}
                       for key, parameters in stages.items()}
            step = {'step': number, 'mode': mode, 'padded_hw': row['padded_hw'],
                    'loss': float(loss.detach()) if bool(torch.isfinite(loss)) else None,
                    'seconds_forward_backward_clip_step_including_gradient_audit': elapsed,
                    'forward_backward_seconds': forward_backward,
                    'clip_update_seconds': update_seconds,
                    'training_compute_seconds_excluding_audit': forward_backward+update_seconds,
                    'before_update_clones_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
                    'grad_norm_before_clip': float(norm) if bool(torch.isfinite(norm)) else None,
                    'autocast_dtype': 'bfloat16', 'loss_scaling': False,
                    'nonfinite_loss_or_gradients': nonfinite,
                    'optimizer_step_executed': not nonfinite,
                    'gradients': gradient, 'changes': changes,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            report['optimizer_attempts'] += 1
            success = (not nonfinite and bool(torch.isfinite(loss))
                       and all(v['finite'] and v['gradient_l1'] > 0 for v in gradient.values())
                       and all(v['parameter_delta_l1'] > 0 for v in changes.values()))
            report['successful_updates'] += int(success)
            report['steps'].append(step)
            write_json(output, report, replace=True)
            if not success:
                raise ValueError('Skipped/invalid update or unchanged stage; no AMP retry')
            if time.monotonic()-started >= SPEC['process_seconds']:
                raise TimeoutError('Internal 600s limit reached after complete update; no retry')
            del predictions, loss, before, pixels, targets
        verify(plan)
        if reads != [plan['view']['mask_path'], plan['view']['valid_path']]:
            raise ValueError('Pixel-read allowlist/count changed')
        report.update(status='completed', all_inputs_sources_unchanged=True, pixel_reads=reads,
                      elapsed_seconds=time.monotonic()-started,
                      metric_policy='No accuracy scoring; no VAL or real RGB model input; no saved trained candidate')
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}',
                      elapsed_seconds=time.monotonic()-started, pixel_reads=reads)
        raise
    finally:
        cv2.imread = original_read
        write_json(output, report, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    ready = commands.add_parser('prepare')
    ready.add_argument('--root', type=Path, default=Path.cwd())
    ready.add_argument('--output', type=Path, required=True)
    ready.add_argument('--model-dir', type=Path, required=True)
    run = commands.add_parser('worker')
    run.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.root, args.output, args.model_dir))
    else:
        worker(args.plan)


if __name__ == '__main__':
    main()
