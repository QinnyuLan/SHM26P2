"""Freeze/run one 1M-cap engineering control from the completed 500k v2 recipe.

Only max_gaussians and the output path change. The entire training package is
copied byte-for-byte from the completed reference, never from current src.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path('/home/sky/workspace/SHM2026')
REFERENCE = Path('/mnt/data/SHM2026/runs/ssim_fixed_corner_v2_rgb_full')
REFERENCE_PLAN_SHA = '81b378c4d58b586d22d4eb3718a9f678f8a6650d5d0003b4f5e01c28b19e6b6e'
REFERENCE_CONFIG_SHA = 'b897e47518b4e8b8120b3b930c3c4c83df6c8004dd2e5a0a46423ac1c8c4be1c'
MANIFEST_SHA = '91b41aeecedc352e4882eb80a1bb635d479ae432251a4c96c85a9dc110cbb327'
INIT_SHA = '544917a1697f9363168ad1632f8f9370f7ff1efa0ef240091501b60e45b158c2'
PROTOCOL = 'rgb_hybrid_capacity_1m_reference_v1'
TIMEOUT = 2400


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(UTC).isoformat()


def write(path, data, *, replace=False):
    payload = json.dumps(data, indent=2, allow_nan=False)+'\n'
    if replace:
        temporary = Path(path).with_suffix('.tmp')
        temporary.write_text(payload)
        temporary.replace(path)
    else:
        with Path(path).open('x') as stream:
            stream.write(payload)


def tree(snapshot):
    return {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob('*.py'))}


def exact_config_difference(reference, candidate, output):
    require(set(reference) == set(candidate), 'Configuration keys changed')
    differences = {key for key in reference if reference[key] != candidate[key]}
    require(differences == {'output', 'max_gaussians'}, 'Only output and max_gaussians may change')
    require(reference['max_gaussians'] == 500000 and candidate['max_gaussians'] == 1000000
            and candidate['output'] == str(Path(output).resolve()), 'Wrong capacity/output intervention')
    require(candidate['steps'] == 30000 and candidate['seed'] == 42
            and candidate.get('warmstart') is None and candidate.get('resume') is None,
            'Require fresh fixed 30k seed42 training')
    return {key: {'reference': reference[key], 'candidate': candidate[key]} for key in sorted(differences)}


def verify_reference():
    require(sha(REFERENCE/'plan.json') == REFERENCE_PLAN_SHA
            and sha(REFERENCE/'replay_config.yaml') == REFERENCE_CONFIG_SHA, 'Reference plan/config changed')
    plan, launch, experiment, audit = [read(REFERENCE/name) for name in
                                     ('plan.json', 'launch_receipt.json', 'experiment_receipt.json', 'stage_audit.json')]
    require(launch['status'] == experiment['status'] == 'completed' and launch['observed_exit_code'] == 0
            and audit['status'] == 'passed' and audit['step'] == 30000
            and launch['plan_sha256'] == audit['plan_sha256'] == REFERENCE_PLAN_SHA
            and launch['stage_audit_sha256'] == sha(REFERENCE/'stage_audit.json'), 'Reference endpoint not completed/audited')
    require(sha(REFERENCE/'last.pt') == launch['checkpoint_sha256'] == experiment['checkpoint_sha256']
            == audit['checkpoint_sha256'], 'Reference endpoint bytes differ')
    config = yaml.safe_load((REFERENCE/'replay_config.yaml').read_text())
    require(config == plan['config'] == experiment['config'], 'Reference config declarations differ')
    require(tree(REFERENCE/'source_snapshot') == plan['source_hashes'] == experiment['source_hashes']
            and len(plan['source_hashes']) == 30, 'Actual reference package differs')
    for binding in experiment['input_hashes'].values():
        require(sha(binding['path']) == binding['sha256'], 'Reference manifest/init/environment changed')
    return plan, config, audit


def split_inputs(manifest):
    require(manifest['pixel_protocol']['id'] == 'colmap_corner_v2', 'Require the same corner-v2 manifest')
    train = [v for v in manifest['views'] if v['split'] == 'train']
    val = [v for v in manifest['views'] if v['split'] == 'val']
    require(len(train) == len({v['name'] for v in train}) == 350
            and len(val) == len({v['name'] for v in val}) == 50
            and not ({v['name'] for v in train} & {v['name'] for v in val}), 'Wrong fixed 350/50 split')
    require(sum(bool(v.get('mask_path')) for v in train) == 259
            and sum(bool(v.get('mask_path')) for v in val) == 41, 'Wrong annotation population')
    files = lambda views: {str((ROOT/v[key]).resolve()) for v in views
                           for key in ('image_path', 'valid_path', 'mask_path') if v.get(key)}
    return train, val, files(train), files(val)


def runtime_environment(snapshot):
    return {'PYTHONPATH': str(snapshot), 'PYTHONDONTWRITEBYTECODE': '1',
            'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8',
            'TORCH_CUDA_ARCH_LIST': '12.0', 'MAX_JOBS': '4'}


def prepare(output):
    output = Path(output).resolve()
    require(not output.exists(), 'Preserve existing run; no overwrite/retry')
    reference, config, audit = verify_reference()
    candidate = dict(config, output=str(output/'training'), max_gaussians=1000000)
    diff = exact_config_difference(config, candidate, output/'training')
    manifest_path = (ROOT/config['manifest']).resolve()
    require(sha(manifest_path) == MANIFEST_SHA, 'Manifest changed')
    manifest = read(manifest_path)
    train, val, train_files, val_files = split_inputs(manifest)
    init_path = (ROOT/manifest['init_points_path']).resolve()
    require(sha(init_path) == INIT_SHA, 'Initialization changed')
    for path in train_files | {str(init_path), str(manifest_path)}:
        require(sha(path) == reference['inputs'][path]['sha256'], 'TRAIN/init bytes differ from the completed reference')
    import numpy as np
    with np.load(init_path, allow_pickle=False) as points:
        require(points['points'].shape == points['colors'].shape == (60000, 3), 'Expected same 60000 input points/colors')
        initialization = {'seed': 42, 'foreground_points': 60000, 'background_shell_points': config['background_points'],
                          'expected_initial_gaussians': 62000, 'fresh_no_resume_no_warmstart': True,
                          'npz_arrays': {key: {'shape': list(points[key].shape), 'dtype': str(points[key].dtype),
                                              'array_bytes_sha256': hashlib.sha256(points[key].tobytes()).hexdigest()}
                                         for key in points.files}}
    document = ROOT/'docs/rgb_capacity_1m_protocol.md'
    require(document.is_file(), 'Reviewed capacity protocol document missing')
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(REFERENCE/'source_snapshot'/'bridge_rgs', snapshot/'bridge_rgs',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    require(tree(snapshot) == reference['source_hashes'], 'Copied package not exact')
    runner = output/'run_rgb_capacity_reference.py'
    shutil.copy2(__file__, runner)
    shutil.copy2(document, output/'protocol_at_lock.md')
    config_path = output/'training_config.yaml'
    config_path.write_text(yaml.safe_dump(candidate, sort_keys=False))
    paths = train_files | val_files | {str(manifest_path), str(init_path), str(ROOT/'uv.lock'), str(ROOT/'pyproject.toml'),
                                     str(runner), str(output/'protocol_at_lock.md'), str(config_path)}
    paths.update(str(REFERENCE/name) for name in ('plan.json', 'replay_config.yaml', 'launch_receipt.json',
                                                'experiment_receipt.json', 'stage_audit.json', 'config.json'))
    bindings = {path: sha(path) for path in sorted(paths)}
    plan = {'protocol': PROTOCOL, 'status': 'locked_pending_root_gpu_authorization', 'created_utc': utc(),
            'root': str(ROOT), 'output': str(output), 'training_output': candidate['output'],
            'reference_directory': str(REFERENCE), 'reference_plan_sha256': REFERENCE_PLAN_SHA,
            'config': candidate, 'config_path': str(config_path), 'config_sha256': sha(config_path),
            'exact_config_difference': diff, 'source_snapshot': str(snapshot), 'source_hashes': tree(snapshot),
            'source_origin': str(REFERENCE/'source_snapshot'), 'source_package_byte_identical': True,
            'runner': str(runner), 'runner_sha256': sha(runner), 'input_hashes': bindings,
            'environment': runtime_environment(snapshot),
            'training_pixel_paths': sorted(train_files), 'native_evaluation_pixel_paths': sorted(val_files),
            'train_names_in_manifest_order': [v['name'] for v in train], 'val_names_sorted': sorted(v['name'] for v in val),
            'initialization': initialization, 'time_limit_seconds': TIMEOUT, 'automatic_retry': False,
            'evaluation': 'fixed final native 50/41 with original package; RGB only interpretation; official evaluation separate',
            'inherited_train_mask_use': 'region RGB emphasis .15; semantic prior quota .65; semantic loss is off',
            'sampling_caveat': 'Same original global NumPy shuffle/RNG implementation, not promised same view sequence. Capacity may change RNG consumption.',
            'observer': 'Wrap GaussianScene.render: only count calls, pre-call N and canvas pixels; no RNG/tensor mutation, exact return and arguments.',
            'reference_cost': {'training_seconds': 1115.77, 'peak_allocated_GiB_logged': 1.463,
                               'final_gaussians': audit['gaussians'], 'checkpoint_bytes': (REFERENCE/'last.pt').stat().st_size},
            'budget_rationale': '2400s total child budget includes train, native scoring and checks, about 2.15x prior training time; no guarantee against OOM.',
            'disk_estimate_bytes': 4_000_000_000, 'smoke': 'No new smoke: exact existing package, same initialization/recipe, capacity limit only.',
            'scope': 'Engineering capacity comparison; increased compute/storage, not matched-compute or academic innovation.'}
    write(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan_path):
    plan = read(plan_path)
    require(plan['protocol'] == PROTOCOL and plan['status'] == 'locked_pending_root_gpu_authorization'
            and plan['time_limit_seconds'] == TIMEOUT and plan['automatic_retry'] is False, 'Unregistered bounded capacity protocol')
    require(Path(__file__).resolve() == Path(plan['runner']).resolve() and sha(__file__) == plan['runner_sha256'], 'Use the frozen runner')
    require(tree(Path(plan['source_snapshot'])) == plan['source_hashes'], 'Frozen package changed')
    require(plan['environment'] == runtime_environment(Path(plan['source_snapshot'])), 'Fixed runtime environment changed')
    reference, config, _ = verify_reference()
    require(plan['source_hashes'] == reference['source_hashes'], 'Candidate package differs from reference')
    require(yaml.safe_load(Path(plan['config_path']).read_text()) == plan['config']
            and sha(plan['config_path']) == plan['config_sha256'], 'Candidate configuration changed')
    require(exact_config_difference(config, plan['config'], plan['training_output']) == plan['exact_config_difference'], 'Configuration difference changed')
    for path, expected in plan['input_hashes'].items():
        require(sha(path) == expected, 'Bound input changed: '+path)
    return plan


def gpu_idle():
    result = subprocess.run(['nvidia-smi', '-q', '-x'], check=True, capture_output=True, text=True, timeout=5)
    state = ET.fromstring(result.stdout)
    require(state.findall('gpu'), 'GPU query missing')
    for gpu in state.findall('gpu'):
        listing = gpu.find('processes')
        require(listing is not None and (listing.text or '').strip() not in {'N/A', 'Not Supported'}, 'GPU query unavailable')
    require(all(p.findtext('type') == 'G' for p in state.findall('.//process_info')), 'GPU compute client present')


def render_observer(original, counts):
    @functools.wraps(original)
    def observed(scene, K, w2c, width, height, *args, **kwargs):
        n, pixels = len(scene.splats['means']), int(width)*int(height)
        counts['render_calls'] += 1
        counts['gaussian_render_sum'] += n
        counts['canvas_pixel_sum'] += pixels
        counts['gaussian_canvas_pixel_sum'] += n*pixels
        counts['peak_pre_render_gaussians'] = max(counts['peak_pre_render_gaussians'], n)
        return original(scene, K, w2c, width, height, *args, **kwargs)
    return observed


def worker(plan_path, stage):
    plan = verify(plan_path)
    os.chdir(plan['root'])
    import torch

    from bridge_rgs import cli, model
    snapshot = Path(plan['source_snapshot']).resolve()
    counts = dict.fromkeys(('render_calls', 'gaussian_render_sum', 'canvas_pixel_sum', 'gaussian_canvas_pixel_sum',
                            'peak_pre_render_gaussians'), 0)
    for name, module in list(sys.modules.items()):
        if name == 'bridge_rgs' or name.startswith('bridge_rgs.'):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(snapshot) and plan['source_hashes'].get(str(path.relative_to(snapshot))) == sha(path),
                    'Actual package import escaped frozen snapshot')
    out = Path(plan['output'])
    record_path = out/f'{stage}_receipt.json'
    require(not record_path.exists(), 'No stage retry')
    arguments = (['bridge_rgs.cli', 'train', '--config', plan['config_path']] if stage == 'train' else
                 ['bridge_rgs.cli', 'evaluate', '--checkpoint', str(Path(plan['training_output'])/'last.pt'),
                  '--manifest', plan['config']['manifest'], '--output', str(Path(plan['training_output'])/'evaluation_native'), '--lpips'])
    record = {'status': 'running', 'stage': stage, 'plan_sha256': sha(plan_path), 'started_utc': utc(),
              'arguments': arguments, 'environment': plan['environment'],
              'render_observer_counts_include_calls_started_not_completed_on_failure': True}
    write(record_path, record)
    original, argv, started = model.GaussianScene.render, sys.argv, time.monotonic()
    model.GaussianScene.render = render_observer(original, counts)
    try:
        sys.argv = arguments
        cli.main()
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        model.GaussianScene.render, sys.argv = original, argv
        imports = {}
        for name, module in list(sys.modules.items()):
            if name == 'bridge_rgs' or name.startswith('bridge_rgs.'):
                path = Path(module.__file__).resolve()
                require(path.is_relative_to(snapshot) and plan['source_hashes'].get(str(path.relative_to(snapshot))) == sha(path), 'Runtime source mismatch')
                imports[name] = {'path': str(path), 'sha256': sha(path)}
        record.update(finished_utc=utc(), elapsed_seconds=time.monotonic()-started, counts=counts,
                      actual_imports=imports, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), original_render_restored=True)
        write(record_path, record, replace=True)


def stages(plan_path):
    plan = verify(plan_path)
    env = dict(os.environ, **plan['environment'])
    out = Path(plan['output'])
    for stage in ('train', 'native'):
        command = [sys.executable, plan['runner'], '--worker', str(plan_path), '--stage', stage]
        with (out/f'{stage}.log').open('x') as log:
            subprocess.run(command, cwd=plan['root'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    verify(plan_path)
    return 0


def run(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = verify(plan_path)
    out, started = Path(plan['output']), time.monotonic()
    receipt_path = out/'launch_receipt.json'
    require(not receipt_path.exists() and not Path(plan['training_output']).exists(), 'No overwrite/resume/retry')
    gpu_idle()
    command = ['timeout', '--signal=TERM', '--kill-after=10s', f'{TIMEOUT}s', sys.executable, plan['runner'], '--stages', str(plan_path)]
    record = {'status': 'running', 'plan_sha256': sha(plan_path), 'started_utc': utc(), 'command': command,
              'time_limit_seconds': TIMEOUT, 'automatic_retry': False}
    write(receipt_path, record)
    try:
        with (out/'wrapper.log').open('x') as log:
            process = subprocess.Popen(command, cwd=plan['root'], stdout=log, stderr=subprocess.STDOUT)
            record['timeout_pid'] = process.pid
            write(receipt_path, record, replace=True)
            code = process.wait()
        record.update(status='completed' if code == 0 else 'failed', exit_code=code)
        require(code == 0, 'Capacity run failed or timed out; no automatic retry')
        verify(plan_path)
        for stage in ('train', 'native'):
            require(read(out/f'{stage}_receipt.json')['status'] == 'completed', 'Missing complete stage')
        training = Path(plan['training_output'])
        record.update(checkpoint_sha256=sha(training/'last.pt'), native_metrics_sha256=sha(training/'evaluation_native/metrics.json'),
                      train_receipt_sha256=sha(out/'train_receipt.json'), native_receipt_sha256=sha(out/'native_receipt.json'),
                      inputs_and_source_after_exact=True)
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record.update(finished_utc=utc(), elapsed_seconds=time.monotonic()-started)
        write(receipt_path, record, replace=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', type=Path)
    mode.add_argument('--stages', type=Path, help=argparse.SUPPRESS)
    mode.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--stage', choices=('train', 'native'), help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        require(args.output is not None, '--prepare requires --output')
        print(prepare(args.output))
    elif args.run:
        print(json.dumps(run(args.run), indent=2))
    elif args.stages:
        stages(args.stages)
    else:
        require(args.stage is not None, 'Worker requires fixed stage')
        worker(args.worker, args.stage)


if __name__ == '__main__':
    main()
