"""Fixed 6000-update rendered RGB reference; prepare is CPU-only, train requires handoff."""
from __future__ import annotations

import argparse
import copy
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
import yaml

from bridge_rgs.mask2former_reference import (
    CLASS_NAMES,
    author_optimizer,
    digest,
    enable_swin_checkpointing,
    load_reference,
    verify_model,
)
from bridge_rgs.mask2former_training import (
    FORMAT,
    TRAIN_PROTOCOL,
    ShuffledViews,
    poly_learning_rate,
    stage_gradient_status,
    training_sample,
    validate_config,
)
from bridge_rgs.mask2former_valid import ValidSupportMask2FormerCriterion, semantic_targets

BASE_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
PREFLIGHT = Path('/mnt/data/SHM2026/preflight/mask2former_reference_v1/plan.json')
PREFLIGHT_SHA = 'a36c55bd41e899a41d0cbeb01556f3e893c4e2202492ae6cec11d047eeeef291'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify_hashes(bindings):
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError(f'Bound file changed: {path}')


def cache_records(config):
    """Check every PNG hash and exact labeled-TRAIN set, without reading GT pixels."""
    manifest = json.loads(Path(config['manifest']).read_text())
    views = sorted((v for v in manifest['views'] if v['split'] == 'train' and v.get('mask_path')),
                   key=lambda v: v['name'])
    names = [v['name'] for v in views]
    receipt = json.loads(Path(config['cache_receipt']).read_text())
    if (len(names) != 259 or len(set(names)) != 259 or receipt.get('status') != 'completed'
            or receipt.get('view_names') != names or receipt.get('checkpoint_sha256') != BASE_SHA
            or receipt.get('checkpoint') != config['base_checkpoint']
            or receipt.get('manifest_sha256') != digest(config['manifest'])
            or receipt.get('manifest') != config['manifest']
            or receipt.get('pixel_protocol') != 'legacy_mixed_v1'
            or len(receipt.get('records', [])) != 259):
        raise ValueError('Require completed exact259 TRAIN cache from selected legacy H3')
    if not receipt.get('frozen_tensors_verified') or not receipt.get('source_hashes'):
        raise ValueError('Cache lacks frozen-field/source audit')
    verify_hashes(receipt['source_hashes'])
    records = {r['name']: r for r in receipt['records']}
    if sorted(records) != names:
        raise ValueError('Cache records missing/duplicated view')
    result, bindings = {}, {config['cache_receipt']: digest(config['cache_receipt'])}
    for view in views:
        record = records[view['name']]
        path = Path(record['image_path'])
        if (not path.is_absolute() or (record['width'], record['height']) != (1320, 989)
                or (view['width'], view['height']) != (1320, 989) or digest(path) != record['sha256']):
            raise ValueError('Cache image dimensions/source hash mismatch')
        row = dict(view, rendered_image_path=str(path))
        result[view['name']] = row
        for key in ('rendered_image_path', 'mask_path', 'valid_path'):
            value = str(Path(row[key]).resolve())
            row[key] = value
            bindings[value] = digest(value)
    return result, bindings


def prepare(config_path, output, root):
    """Freeze only after the reviewed exact-scene TRAIN cache exists."""
    config_path, output, root = Path(config_path).resolve(), Path(output).resolve(), Path(root).resolve()
    config = validate_config(yaml.safe_load(config_path.read_text()))
    if output.exists() or Path(config['output']).exists() or not output.is_relative_to('/mnt/data'):
        raise ValueError('Use fresh data-volume preparation/training paths')
    if digest(PREFLIGHT) != PREFLIGHT_SHA or digest(config['base_checkpoint']) != BASE_SHA:
        raise ValueError('Reviewed preflight or selected H3 source changed')
    parent = json.loads(PREFLIGHT.read_text())
    previous = Path(parent['source_snapshot'])
    verify_hashes({str(previous/name): sha for name, sha in parent['source_hashes'].items()})
    rows, files = cache_records(config)
    files |= verify_model(config['model_dir'])
    files |= {str(config_path): digest(config_path), config['base_checkpoint']: BASE_SHA,
              config['manifest']: digest(config['manifest']), str(root/'uv.lock'): digest(root/'uv.lock'),
              config['selected_reference_receipt']: digest(config['selected_reference_receipt']),
              str(PREFLIGHT): PREFLIGHT_SHA}
    files |= {path: sha for path, sha in parent['input_hashes'].items()
              if 'site-packages' in path}
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(previous/'bridge_rgs', snapshot/'bridge_rgs',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['mask2former_training.py']:
        shutil.copy2(root/'src/bridge_rgs'/name, snapshot/'bridge_rgs'/name)
    for name in ['train_mask2former_reference.py', 'evaluate_mask2former_reference.py',
                 'render_mask2former_train_cache.py']:
        shutil.copy2(root/'scripts'/name, snapshot/name)
    sources = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')}
    plan = {'status': 'cpu_locked_waiting_root_gpu_handoff', 'configuration': config,
            'config_path': str(config_path), 'source_snapshot': str(snapshot), 'source_hashes': sources,
            'source_parent': {'plan': str(PREFLIGHT), 'sha256': PREFLIGHT_SHA},
            'input_hashes': files, 'train_views': rows, 'root': str(root),
            'execution': 'Only after root GPU handoff. External timeout2400s; cooperative stop2280s preserves latest safe checkpoint.'}
    verify_hashes(files)
    write_json(output/'plan.json', plan)
    return output/'plan.json'


def verify_plan(plan):
    validate_config(plan['configuration'])
    snapshot = Path(plan['source_snapshot'])
    if (plan['status'] != 'cpu_locked_waiting_root_gpu_handoff'
            or Path(__file__).resolve() != snapshot/Path(__file__).name):
        raise ValueError('Run the reviewed frozen training script')
    actual = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')}
    if actual != plan['source_hashes']:
        raise ValueError('Frozen source changed')
    verify_hashes(plan['input_hashes'])
    for name in ['mask2former_training', 'mask2former_reference', 'mask2former_valid', 'train', 'model']:
        actual_path = Path(importlib.import_module('bridge_rgs.'+name).__file__).resolve()
        if actual_path != snapshot/'bridge_rgs'/f'{name}.py':
            raise ValueError(f'Unbound actual import: {actual_path}')


def checkpoint_state(model, optimizer, step, sampler, augmentation, points, config, plan, trace, logs):
    return {'format': FORMAT, 'pixel_protocol': 'legacy_mixed_v1', 'class_names': CLASS_NAMES,
            'step': step, 'training_completed': step == TRAIN_PROTOCOL['steps'],
            'model_config': model.config.to_dict(), 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'training_config': config,
            'sampler_state': sampler.state_dict(), 'augmentation_rng_state': copy.deepcopy(augmentation.bit_generator.state),
            'point_rng_state': points.get_state(), 'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(), 'trace': trace.copy(), 'loss_logs': logs.copy(),
            'provenance': {'base_checkpoint': config['base_checkpoint'], 'base_sha256': BASE_SHA,
                           'manifest': config['manifest'], 'manifest_sha256': digest(config['manifest']),
                           'input_hashes': plan['input_hashes'], 'source_hashes': plan['source_hashes'],
                           'source_snapshot': plan['source_snapshot'],
                           'data_scope': 'Only259 labeled TRAIN rendered RGB; no real RGB, pseudo labels or validation selection'}}


def restore_training(state, model, optimizer, sampler, augmentation, points, config, plan):
    previous = copy.deepcopy(state['training_config'])
    previous['output'] = config['output']
    if (state.get('format') != FORMAT or previous != config or not 1 <= state['step'] < 6000
            or state['provenance']['source_hashes'] != plan['source_hashes']):
        raise ValueError('Resume protocol/source changed or checkpoint already final')
    # A new output/config path is allowed; all actual dataset/weight inputs remain bound.
    old_inputs = state['provenance']['input_hashes']
    verify_hashes(old_inputs)
    if any(plan['input_hashes'].get(path, sha) != sha for path, sha in old_inputs.items()):
        raise ValueError('Resume input SHA differs')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    sampler.load_state_dict(state['sampler_state'])
    augmentation.bit_generator.state = copy.deepcopy(state['augmentation_rng_state'])
    points.set_state(state['point_rng_state'].cpu())
    torch.set_rng_state(state['torch_rng_state'].cpu())
    torch.cuda.set_rng_state(state['cuda_rng_state'].cpu())
    if len(state['trace']) != state['step'] or state['trace'][-1]['step'] != state['step']:
        raise ValueError('Resume trace is incomplete')
    return state['step'], copy.deepcopy(state['trace']), copy.deepcopy(state['loss_logs'])


def save_checkpoint(path, value):
    temporary = Path(path).with_suffix('.tmp.pt')
    torch.save(value, temporary)
    temporary.replace(path)


def train(plan_path, resume=None):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    verify_plan(plan)
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                           text=True, capture_output=True, check=True, timeout=5)
    if query.stdout.strip():
        raise ValueError('GPU is occupied or query unknown; handoff required')
    config, rows = plan['configuration'], plan['train_views']
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(plan['root'])
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    seed = TRAIN_PROTOCOL['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    receipt = {'status': 'running', 'plan': str(plan_path), 'plan_sha256': digest(plan_path),
               'pid': os.getpid(), 'configuration': config, 'completed_step': 0,
               'resume_checkpoint': str(Path(resume).resolve()) if resume else None,
               'resume_checkpoint_sha256': digest(resume) if resume else None,
               'validation_during_training': False, 'source_hashes': plan['source_hashes']}
    receipt_path = output/'execution_receipt.json'
    write_json(receipt_path, receipt)
    original_read = cv2.imread
    allowed = {str(row[key]) for row in rows.values() for key in ['rendered_image_path', 'mask_path', 'valid_path']}
    def guarded(path, *args, **kwargs):
        if str(Path(path).resolve()) not in allowed:
            raise ValueError('Pixel read outside locked rendered TRAIN259 support')
        return original_read(path, *args, **kwargs)
    cv2.imread = guarded
    started = time.monotonic()
    try:
        if not torch.cuda.is_bf16_supported():
            raise ValueError('This fixed reference requires BF16 support')
        model, receipt['initialization'] = load_reference(config['model_dir'])
        receipt['checkpointed_stages'] = enable_swin_checkpointing(model)
        model = model.cuda().train()
        optimizer, receipt['optimizer_groups'] = author_optimizer(model)
        criterion = ValidSupportMask2FormerCriterion(model.config).cuda()
        sampler = ShuffledViews(sorted(rows), seed)
        augmentation = np.random.default_rng(seed+1)
        points = torch.Generator(device='cuda').manual_seed(seed+2)
        torch.manual_seed(seed+3)
        torch.cuda.manual_seed_all(seed+3)
        completed, trace, logs = 0, [], []
        if resume:
            completed, trace, logs = restore_training(torch.load(resume, map_location='cpu', weights_only=False),
                                                      model, optimizer, sampler, augmentation, points, config, plan)
        stage_samples = {f'stage{i+1}': stage.blocks[0].attention.self.query.weight
                         for i, stage in enumerate(model.model.pixel_level_module.encoder.encoder.layers)}
        before_stages = {name: p.detach().clone() for name, p in stage_samples.items()}
        receipt['stage_audit_scope'] = 'One fixed attention query weight per stage; full-stage gradients/updates were verified in preflight'
        write_json(receipt_path, receipt)
        with (output/'trace.jsonl').open('x') as trace_file, (output/'train.jsonl').open('x') as log_file:
            for row in trace:
                trace_file.write(json.dumps(row)+'\n')
            for row in logs:
                log_file.write(json.dumps(row)+'\n')
            for step in range(completed+1, 6001):
                if time.monotonic()-started >= 2280:
                    raise TimeoutError('Fixed budget stopping before new step; previous periodic checkpoint retained')
                tick = time.monotonic()
                name = sampler.next()
                row = rows[name]
                rgb = cv2.imread(row['rendered_image_path'], cv2.IMREAD_COLOR)[..., ::-1]
                labels = cv2.imread(row['mask_path'], cv2.IMREAD_UNCHANGED)
                valid = cv2.imread(row['valid_path'], cv2.IMREAD_GRAYSCALE) > 0
                pixels, label, valid, transform = training_sample(rgb, labels, valid, step, augmentation)
                targets = semantic_targets([label.cuda()], [valid.cuda()], 5)
                multiplier = poly_learning_rate(optimizer, step)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    predictions = model(pixels.cuda(), output_auxiliary_logits=True)
                loss = criterion(predictions, targets, generator=points).loss
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .01, error_if_nonfinite=True)
                stage_ok, stage_nonzero = stage_gradient_status(stage_samples)
                # At batch1, legitimate stochastic depth can zero a whole block.
                # Only existence/finiteness is a per-step contract; nonzero is diagnostic.
                if not bool(torch.isfinite(loss)) or not all(stage_ok.values()):
                    raise ValueError('Nonfinite loss or missing/nonfinite stage gradient; no AMP skip/retry')
                optimizer.step()
                torch.cuda.synchronize()
                completed = step
                event = {'step': step, 'view': name, **transform}
                trace.append(event)
                trace_file.write(json.dumps(event)+'\n')
                if step == 1 or step % 100 == 0:
                    record = dict(event, loss=float(loss.detach()), lr_multiplier=multiplier,
                                  grad_norm_before_clip=float(norm), step_seconds=time.monotonic()-tick,
                                  stage_gradient_probe_finite=stage_ok,
                                  stage_gradient_probe_nonzero=stage_nonzero,
                                  elapsed_seconds=time.monotonic()-started,
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    logs.append(record)
                    log_file.write(json.dumps(record, allow_nan=False)+'\n')
                    log_file.flush()
                    trace_file.flush()
                    receipt.update(completed_step=step, elapsed_seconds=time.monotonic()-started)
                    write_json(receipt_path, receipt)
                if step % 1000 == 0:
                    save_checkpoint(output/'last.pt', checkpoint_state(model, optimizer, completed, sampler,
                                    augmentation, points, config, plan, trace, logs))
                del predictions, loss, pixels, targets
        verify_plan(plan)
        changes = {name: {'finite': bool(torch.isfinite(p).all()),
                          'delta_l1': float((p.detach()-before_stages[name]).abs().sum())}
                   for name, p in stage_samples.items()}
        if not all(value['finite'] and value['delta_l1'] > 0 for value in changes.values()):
            raise ValueError('Stage update probe failed')
        receipt.update(status='completed', completed_step=completed, successful_updates=6000,
                       checkpoint=str(output/'last.pt'), checkpoint_sha256=digest(output/'last.pt'),
                       trace_sha256=digest(output/'trace.jsonl'), log_sha256=digest(output/'train.jsonl'),
                       stage_update_probes=changes, all_inputs_sources_unchanged=True,
                       elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}',
                       elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        cv2.imread = original_read
        write_json(receipt_path, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    ready = commands.add_parser('prepare')
    ready.add_argument('--config', type=Path, required=True)
    ready.add_argument('--output', type=Path, required=True)
    ready.add_argument('--root', type=Path, default=Path.cwd())
    run = commands.add_parser('train')
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.config, args.output, args.root))
    else:
        train(args.plan, args.resume)


if __name__ == '__main__':
    main()
