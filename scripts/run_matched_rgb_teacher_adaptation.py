"""Fixed 2k selected/composite-domain teacher continuation; no validation reads."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from run_teacher_capacity_stage import (
    StageTrace,
    checkpoint_rng,
    file_sha,
    read_records,
    utc,
    verify_gpu_idle,
    write_json,
)

from bridge_rgs import teacher
from bridge_rgs.teacher_domains import (
    MULTI_COMPONENT_DOMAIN,
    validate_image_sources,
    verify_common_original_warmstart,
)

ROOT = Path('/home/sky/workspace/SHM2026')
ARMS = ('selected', 'composite')
WARMSTART = ROOT/'runs/teacher_render_adapt_v1/best.pt'
WARMSTART_SHA = '00f5b84ac9a56c39512c5b8e43f70110397923feea1a2b4c78bdffd1f5524bff'
ORIGINAL_SHA = '551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa'
SPEC = {'id': 'matched_legacy_composite_teacher_head_adaptation_v1', 'arms': list(ARMS), 'steps': 2000,
        'domain_steps': {'real': 1000, 'rendered': 1000}, 'validation_reads': False,
        'internal_seconds_per_arm': 900, 'external_seconds_per_arm': 960,
        'selection': 'Exactly last.pt EMA at 2000; no VAL selection, no retry or extra steps',
        'evaluation_input': 'Both endpoints on the same actual composite original RGB, via fixed legacy adapter',
        'scope': 'Matched engineering domain continuation; TRAIN geometry and historic teacher saw these TRAIN views; not out-of-fit render adaptation or innovation'}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def source_files(path):
    return {str(p.relative_to(path)): file_sha(p) for p in sorted(path.rglob('*.py'))}


def fixed_config():
    old = read(ROOT/'configs/teacher_render_adapt_v1.json')['config']
    old.update(warmstart_checkpoint=str(WARMSTART), evaluate_validation=False, independent_augmentation_rng=True,
               checkpoint_every=500, eval_every=2000)
    return asdict(teacher.TeacherConfig(**old))


def checked_config(settings):
    config = teacher.TeacherConfig(**settings)
    expected = {'steps': 2000, 'crop_size': 768, 'channels': 192, 'lr': 3e-5, 'weight_decay': .01,
        'warmup_steps': 100, 'ema_decay': .99, 'seed': 20260926, 'context_start': 1,
        'context_probability': .5, 'context_short_side': 768, 'gradient_clip': 1., 'adapter_rank': 0,
        'render_mix_probability': .5, 'evaluate_validation': False, 'independent_augmentation_rng': True,
        'checkpoint_every': 500, 'eval_every': 2000, 'cpu_threads': 8, 'device': 'cuda',
        'warmstart_checkpoint': str(WARMSTART)}
    require(all(getattr(config, key) == value for key, value in expected.items()), 'Fixed 2k teacher configuration differs')
    require(config.consistency_start > config.steps, 'No unlabeled consistency in this continuation')
    return config


def prepare(parent, cache_receipt):
    """CPU-only; called after cache completion and review, not during cache render."""
    parent, cache_receipt = Path(parent).resolve(), Path(cache_receipt).resolve()
    require(not torch.cuda.is_initialized() and not (parent/'plan.json').exists()
            and not (parent/'source_snapshot').exists(), 'New CPU plan required')
    cache = read(cache_receipt)
    require(cache['status'] == 'completed' and cache['id'] == MULTI_COMPONENT_DOMAIN
            and cache['original_manifest_sha256'] == ORIGINAL_SHA, 'Completed fixed cache required')
    original_path = ROOT/'artifacts/prepared/manifest.json'
    require(file_sha(original_path) == ORIGINAL_SHA and file_sha(WARMSTART) == WARMSTART_SHA, 'Fixed source changed')
    original = read(original_path)
    require(len(original['views']) == 400 and sum(v['split'] == 'train' for v in original['views']) == 350
            and sum(v['split'] == 'train' and bool(v.get('mask_path')) for v in original['views']) == 259, 'Original split changed')
    parent.mkdir(parents=True, exist_ok=True)
    bindings = {str(cache_receipt): file_sha(cache_receipt), str(original_path): ORIGINAL_SHA, str(WARMSTART): WARMSTART_SHA}
    manifests, configurations = {}, {}
    for arm in ARMS:
        manifest = copy.deepcopy(original)
        manifest['image_source_protocol'] = {'id': MULTI_COMPONENT_DOMAIN, 'pixel_protocol': 'legacy_mixed_v1',
            'original_manifest': str(original_path), 'original_manifest_sha256': ORIGINAL_SHA,
            'cache_receipt': {'path': str(cache_receipt), 'sha256': file_sha(cache_receipt)}, 'train_domain': arm}
        records = {row['name']: row for row in cache['records'][f'train_{arm}']+cache['records']['val_composite']}
        for view in manifest['views']:
            row = records[view['name']]
            view['image_path_sources'] = {'rendered': {'path': row['image_path'], 'sha256': row['sha256']}}
            train = view['split'] == 'train'
            view['image_domain'] = 'mixed_real_rendered' if train else 'rendered_rgb'
            if train:
                view['image_path_sources']['real'] = {'path': view['image_path'], 'sha256': file_sha(view['image_path'])}
                for path in (view['image_path'], view.get('mask_path'), view.get('valid_path'), row['image_path']):
                    if path:
                        bindings[str(Path(path).resolve())] = file_sha(path)
            else:
                view['image_path'] = row['image_path']
        validate_image_sources(manifest)
        teacher.verify_teacher_render_protocol(manifest, manifest['image_source_protocol'])
        path = parent/f'manifest_{arm}.json'
        require(not path.exists(), 'Manifest already exists')
        write_json(path, manifest)
        manifests[arm] = str(path)
        settings = {'manifest': str(path), 'model_dir': str(ROOT/'models/dinov3-vith16plus'),
                    'output_dir': str(parent/arm), 'config': fixed_config()}
        checked_config(settings['config'])
        config_path = parent/f'config_{arm}.json'
        require(not config_path.exists(), 'Configuration already exists')
        write_json(config_path, settings)
        configurations[arm] = {'path': str(config_path), 'sha256': file_sha(config_path)}
        bindings[str(path)] = file_sha(path)
        bindings[str(config_path)] = file_sha(config_path)
    initial = torch.load(WARMSTART, map_location='cpu', weights_only=False)
    warmstart_preflight = {arm: verify_common_original_warmstart(initial, read(path)['image_source_protocol'])
                          for arm, path in manifests.items()}
    old_manifest = initial['provenance']['manifest']
    bindings[old_manifest] = file_sha(old_manifest)
    old_domain = initial['provenance']['image_source_protocol']
    for split in ('train', 'val'):
        item = old_domain[f'{split}_render_receipt']
        require(file_sha(item['path']) == item['sha256'], 'Historical teacher receipt changed')
        bindings[item['path']] = item['sha256']
    model_dir = ROOT/'models/dinov3-vith16plus'
    for path in [model_dir/'config.json', *model_dir.glob('*.safetensors'), *model_dir.glob('*.safetensors.index.json'), ROOT/'uv.lock', ROOT/'configs/teacher_render_adapt_v1.json']:
        bindings[str(path)] = file_sha(path)
    for component in cache['components'].values():
        bindings[component['checkpoint']] = component['checkpoint_sha256']
    snapshot = parent/'source_snapshot'
    package = Path(teacher.__file__).resolve().parent
    shutil.copytree(package, snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    scripts = [Path(__file__), ROOT/'scripts/run_teacher_capacity_stage.py']
    tests = [ROOT/'tests/test_teacher_multi_component.py', ROOT/'tests/test_matched_teacher_launcher.py']
    for path in scripts+tests:
        shutil.copy2(path, snapshot/path.name)
        bindings[str(path.resolve())] = file_sha(path)
    plan = {'status': 'locked_pending_root_gpu_handoff', 'specification': SPEC, 'root': str(ROOT),
        'output_root': str(parent), 'source_snapshot': str(snapshot), 'source_hashes': source_files(snapshot),
        'input_hashes': bindings, 'manifests': manifests, 'configurations': configurations,
        'warmstart': {'path': str(WARMSTART), 'sha256': WARMSTART_SHA},
        'warmstart_common_original_preflight': warmstart_preflight,
        'external_timeout_seconds_per_arm': 960, 'prepare_val_pixel_reads': 0,
        'env': {'PYTHONPATH': str(snapshot), 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'}}
    write_json(parent/'plan.json', plan)
    return parent/'plan.json'


def normalized_trace(record):
    """Discard ONLY controlled RGB bytes/path; retain target/valid hashes and RNG."""
    value = copy.deepcopy(record)
    for event in value['events']:
        if event['kind'] == 'read':
            event.pop('image_path')
        elif event['kind'] in {'crop', 'context'}:
            require(len(event['output_sha256']) == 3, 'Expected RGB/target/valid spatial trace')
            event['output_sha256'][0] = '<controlled-RGB>'
    return value


def compare_receipts(first, second):
    receipts = [read(path) for path in (first, second)]
    require([r['arm'] for r in receipts] == list(ARMS) and all(r['status'] == 'completed' for r in receipts), 'Both completed arms required')
    for key in ('plan_sha256', 'configuration', 'warmstart', 'source_hashes', 'steps'):
        require(receipts[0][key] == receipts[1][key], f'Matched {key} differs')
    traces = []
    for receipt in receipts:
        summary = receipt['trace']
        require(file_sha(summary['segment_path']) == summary['segment_sha256'], 'Trace changed')
        require(file_sha(receipt['last_checkpoint']) == receipt['last_checkpoint_sha256'], 'Endpoint changed')
        records = read_records(Path(summary['segment_path']))
        require([r['step'] for r in records] == list(range(1, 2001)), 'Incomplete step sequence')
        traces.append([normalized_trace(r) for r in records])
    mismatch = next((i+1 for i, (a, b) in enumerate(zip(*traces, strict=True)) if a != b), None)
    checks = {'ordered_view_domain_targets_spatial_and_photo_rng_equal': mismatch is None,
              'terminal_rng_equal': receipts[0]['terminal_rng'] == receipts[1]['terminal_rng']}
    return {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks, 'first_mismatch_step': mismatch,
            'ignored_difference': 'Only RGB source path and RGB spatial tensor SHA; full raw traces remain stored',
            'steps': 2000, 'receipt_sha256': {str(Path(p).resolve()): file_sha(p) for p in (first, second)}}


def verify(plan):
    snapshot = Path(plan['source_snapshot'])
    require(plan['specification'] == SPEC and source_files(snapshot) == plan['source_hashes'], 'Frozen source/specification changed')
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Execute frozen launcher only')
    require(all(file_sha(p) == value for p, value in plan['input_hashes'].items()), 'Input changed')
    actual = {}
    for name, mod in list(sys.modules.items()):
        if name == 'bridge_rgs' or name.startswith('bridge_rgs.') or name == 'run_teacher_capacity_stage':
            path = Path(mod.__file__).resolve()
            require(path.is_relative_to(snapshot) and file_sha(path) == plan['source_hashes'][str(path.relative_to(snapshot))], 'Import escaped snapshot')
            actual[name] = {'path': str(path), 'sha256': file_sha(path)}
    return actual


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return True


def execute(plan_path, arm):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    require(arm in ARMS, 'Unknown arm')
    actual = verify(plan)
    bound = plan['configurations'][arm]
    require(file_sha(bound['path']) == bound['sha256'], 'Config changed')
    settings = read(bound['path'])
    config = checked_config(settings['config'])
    output = Path(settings['output_dir'])
    require(output == Path(plan['output_root'])/arm and not output.exists(), 'Fresh fixed arm output required')
    require(settings['manifest'] == plan['manifests'][arm], 'Arm manifest differs')
    manifest = read(settings['manifest'])
    require(manifest['image_source_protocol']['train_domain'] == arm, 'Wrong cache domain')
    verify_gpu_idle(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=5))
    os.chdir(plan['root'])
    output.mkdir()
    allowed = {v['name']: {s['path'] for s in v['image_path_sources'].values()} for v in manifest['views'] if v['split'] == 'train'}
    trace = StageTrace(output/'steps.jsonl', device='cuda', allowed_train=allowed)
    receipt = {'status': 'failed', 'arm': arm, 'steps': 2000, 'plan_sha256': file_sha(plan_path),
        'configuration': asdict(config), 'warmstart': plan['warmstart'], 'source_hashes': plan['source_hashes'],
        'input_hashes': plan['input_hashes'], 'actual_imports': actual, 'started_utc': utc(), 'pid': os.getpid(),
        'validation_pixel_reads': 0, 'no_retry': True, 'exit_status_scope': 'Root outer receipt must separately confirm natural process exit 0'}
    started = time.monotonic()
    old_handler = signal.getsignal(signal.SIGALRM)
    def deadline(*_):
        raise TimeoutError('Fixed 900s training limit; no automatic resume/retry')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(900)
    try:
        with trace.installed():
            result = teacher.train_teacher(settings['manifest'], settings['model_dir'], output, config)
        require(trace.steps == 2000 and not trace.events, 'Incomplete 2000 optimizer updates')
        last = output/'last.pt'
        state = torch.load(last, map_location='cpu', weights_only=False)
        require(state['step'] == 2000 and state['configuration'] == asdict(config), 'Endpoint changed')
        require(state['validation'] is None and result['validation'] is None and not (output/'best.pt').exists(), 'Unexpected validation/selection')
        require(state['domain_counts'] == {'real': 1000, 'rendered': 1000}, 'Domain schedule changed')
        require(finite_tree({k: state[k] for k in ('decoder', 'ema_decoder', 'optimizer')}), 'Nonfinite endpoint')
        require(state['provenance']['warmstart']['sha256'] == WARMSTART_SHA, 'Wrong initial EMA')
        terminal, summary = checkpoint_rng(state), trace.summary()
        require(summary['last_optimizer_rng_after'] == terminal['torch'] and summary['last_numpy_after'] == terminal['numpy'], 'Checkpoint/observed RNG differ')
        receipt.update(status='completed', last_checkpoint=str(last), last_checkpoint_sha256=file_sha(last),
                       trace=summary, terminal_rng=terminal, result=result, actual_imports=verify(plan), finished_utc=utc())
    except BaseException as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        receipt['trace'] = trace.summary()
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        receipt['elapsed_seconds'] = time.monotonic()-started
        receipt['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        receipt['inputs_sources_unchanged'] = source_files(Path(plan['source_snapshot'])) == plan['source_hashes'] and all(file_sha(p) == h for p, h in plan['input_hashes'].items())
        if not receipt['inputs_sources_unchanged']:
            receipt['status'] = 'failed'
        write_json(output/'stage_receipt.json', receipt)
    require(receipt['status'] == 'completed', 'Endpoint invariance failed')
    print(json.dumps({'status': receipt['status'], 'arm': arm, 'receipt': str(output/'stage_receipt.json')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path)
    action.add_argument('--run', type=Path)
    action.add_argument('--compare', nargs=2, type=Path)
    p.add_argument('--cache-receipt', type=Path)
    p.add_argument('--arm', choices=ARMS)
    args = p.parse_args()
    if args.prepare:
        require(args.cache_receipt is not None, 'Cache receipt required')
        print(prepare(args.prepare, args.cache_receipt))
    elif args.run:
        execute(args.run, args.arm)
    else:
        print(json.dumps(compare_receipts(*args.compare), indent=2))
