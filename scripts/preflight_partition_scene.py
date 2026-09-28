"""Fixed two-TRAIN-camera integration check without image/label reads or training."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path('/home/sky/workspace/SHM2026')
CKPT = ROOT/'runs/h3_moments/02_cross/last.pt'
CKPT_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
MANIFEST = ROOT/'artifacts/prepared/manifest.json'
SPEC = {'protocol': 'partition_scene_preflight_v2', 'names': ['002.png', '118.png'],
        'modes': ['marginal', 'point', 'integrated'], 'grid': 'native legacy_mixed_v1',
        'initial_probability_tolerance': 3e-6, 'alpha_tolerance': 2e-6,
        'rgb_tolerance': 0., 'simplex_tolerance': 3e-6,
        'endpoint_perturbation': '.05*sin(flattened endpoint index*.37), opposite endpoint signs',
        'probe': 'separate raw and full-final mean(sum(probability * [0,.3,-.2,.7,-.4])) VJPs from same render',
        'probe_modes': ['point', 'integrated'], 'optimizer_steps': 0,
        'source_image_reads': 0, 'label_reads': 0, 'val_views': 0,
        'internal_seconds': 480, 'external_seconds': 540, 'retries': 0,
        'scope': 'integration, finite gradients and runtime only; no accuracy or innovation claim'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*.py')) if '__pycache__' not in p.parts}


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Refuse existing experiment')
    if sha(CKPT) != CKPT_SHA:
        raise ValueError('Wrong H3 checkpoint')
    manifest = json.loads(MANIFEST.read_text())
    views = [next(v for v in manifest['views'] if v['name'] == name) for name in SPEC['names']]
    if not all(v['split'] == 'train' for v in views):
        raise ValueError('Only two fixed TRAIN cameras')
    cameras = [{k: v[k] for k in ('name', 'split', 'K', 'w2c_original', 'width', 'height')} for v in views]
    snapshot = output/'source_snapshot'
    shutil.copytree(ROOT/'src/bridge_rgs', snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    shutil.copy2(ROOT/'uv.lock', snapshot/'uv.lock')
    (snapshot/'tests').mkdir()
    for name in ('test_semantic_partition.py', 'test_partition_projection.py', 'test_partition_field.py'):
        shutil.copy2(ROOT/'tests'/name, snapshot/'tests'/name)
    gs = Path(importlib.util.find_spec('gsplat').origin).parent
    sources = source_tree(snapshot)
    installed = {str(p): sha(p) for p in [gs/'cuda/_wrapper.py', gs/'rendering.py',
                 gs/'cuda/csrc/RasterizeToPixels3DGSFwd.cu', gs/'cuda/include/Utils.cuh']}
    prior = {}
    for relative in ('partition_rasterizer_synthetic_v1/execution_receipt.json',
                     'semantic_partition_numerical_v1/execution/execution_receipt.json'):
        path = Path('/mnt/data/SHM2026/runs')/relative
        value = json.loads(path.read_text())
        if value['status'] not in ('passed', 'completed'):
            raise ValueError('Numerical preflight not passed')
        if 'numerical_status' in value and value['numerical_status'] != 'passed':
            raise ValueError('Numerical report failed despite worker completion')
        prior[str(path)] = sha(path)
        run_root = path.parent if path.parent.name != 'execution' else path.parent.parent
        launch_path = run_root/'launch_receipt.json'
        launch = json.loads(launch_path.read_text())
        if launch['exit_code'] != 0 or not launch['natural_completion']:
            raise ValueError('Numerical worker did not complete naturally')
        if launch['execution_receipt_sha256'] != sha(path):
            raise ValueError('Numerical launch does not bind this execution receipt')
        prior[str(launch_path)] = sha(launch_path)
    plan = {'specification': SPEC, 'output': str(output), 'snapshot': str(snapshot),
            'sources': sources, 'uv_lock_sha256': sha(snapshot/'uv.lock'),
            'inputs': {str(CKPT): CKPT_SHA, str(MANIFEST): sha(MANIFEST), **prior},
            'installed_sources': installed, 'cameras': cameras,
            'runtime_versions': {p: importlib.metadata.version(p) for p in ('torch', 'triton', 'gsplat', 'numpy')},
            'allowed_existing_gpu_executable': '/usr/share/rustdesk/rustdesk'}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    if plan['specification'] != SPEC or source_tree(Path(plan['snapshot'])) != plan['sources']:
        raise ValueError('Specification or snapshot changed')
    for p, digest in {**plan['inputs'], **plan['installed_sources']}.items():
        if sha(p) != digest:
            raise ValueError(f'Bound source/input changed: {p}')
    if sha(Path(plan['snapshot'])/'uv.lock') != plan['uv_lock_sha256']:
        raise ValueError('Lock changed')
    for p, version in plan['runtime_versions'].items():
        if importlib.metadata.version(p) != version:
            raise ValueError('Runtime changed')


def execute(path, expected):
    if sha(path) != expected:
        raise ValueError('Plan SHA mismatch')
    plan = json.loads(Path(path).read_text())
    verify(plan)
    output, snapshot = Path(plan['output']), Path(plan['snapshot'])
    if Path(__file__).resolve() != snapshot/Path(__file__).name:
        raise ValueError('Execute frozen worker only')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True)
    for row in apps.splitlines():
        pid = int(row.split(',')[0])
        if Path(f'/proc/{pid}/exe').resolve().as_posix() != plan['allowed_existing_gpu_executable']:
            raise ValueError(f'Unexpected GPU user {pid}; do not terminate it')
    started = {'plan_sha256': expected, 'pid': os.getpid(), 'gpu_processes_before': apps}
    write(output/'execution_started.json', started)
    receipt = {**started, 'status': 'running', 'label_reads': 0, 'image_reads': 0, 'optimizer_steps': 0}
    beginning = time.perf_counter()
    def alarm(_signal, _frame):
        raise TimeoutError('Fixed real-scene preflight deadline')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(SPEC['internal_seconds'])
    try:
        torch.set_num_threads(4)
        torch.manual_seed(20260927)
        sys.path.insert(0, str(snapshot))
        from bridge_rgs.partition_field import SemanticPartitionField, render_partition
        from bridge_rgs.train import load_scene
        scene, state = load_scene(CKPT)
        scene.eval().requires_grad_(False)
        scene.refiner.requires_grad_(True)
        if scene.pixel_protocol != 'legacy_mixed_v1':
            raise ValueError('Wrong coordinate protocol')
        with torch.no_grad():
            logits = scene.semantic_decoder(scene.splats['sem_features'])
        records = []
        for camera in plan['cameras']:
            K = torch.tensor(camera['K'], dtype=torch.float32, device='cuda')
            w2c = torch.tensor(camera['w2c_original'], dtype=torch.float32, device='cuda')
            width, height = camera['width'], camera['height']
            with torch.no_grad():
                baseline = scene.render(K, w2c, width, height, absgrad=False)
            for mode in SPEC['modes']:
                field = SemanticPartitionField(logits, mode=mode)
                torch.cuda.synchronize()
                start = time.perf_counter()
                initial = render_partition(scene, field, K, w2c, width, height)
                torch.cuda.synchronize()
                forward_seconds = time.perf_counter()-start
                maximum = lambda value: float(value.detach().abs().max())
                errors = {'p3d': maximum(initial['p3d']-baseline['p3d']),
                          'probabilities': maximum(initial['probabilities']-baseline['probabilities']),
                          'alpha': maximum(initial['partition_alpha']-baseline['alpha']),
                          'rgb': maximum(initial['rgb']-baseline['rgb']),
                          'simplex': maximum(initial['p3d'].sum(-1)-1)}
                del initial
                gradients = None
                raw_gradients = None
                head_gradient = None
                backward_seconds = None
                if mode in SPEC['probe_modes']:
                    with torch.no_grad():
                        delta = .05*torch.sin(torch.arange(logits.numel(), device='cuda').reshape_as(logits)*.37)
                        field.inside_logits.add_(delta)
                        field.outside_logits.sub_(delta)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    rendered = render_partition(scene, field, K, w2c, width, height)
                    contrast = logits.new_tensor([0., .3, -.2, .7, -.4])
                    raw_probe = (rendered['p3d']*contrast).sum(-1).mean()
                    raw_vjp = torch.autograd.grad(raw_probe, tuple(field.parameters()), retain_graph=True)
                    raw_gradients = {name: {'finite': bool(torch.isfinite(g).all()), 'norm': float(g.norm())}
                                     for (name, _p), g in zip(field.named_parameters(), raw_vjp, strict=True)}
                    scene.refiner.zero_grad(set_to_none=True)
                    final_probe = (rendered['probabilities']*contrast).sum(-1).mean()
                    final_probe.backward()
                    torch.cuda.synchronize()
                    backward_seconds = time.perf_counter()-start
                    gradients = {name: {'finite': bool(p.grad is not None and torch.isfinite(p.grad).all()),
                                         'norm': float(p.grad.norm()) if p.grad is not None else None}
                                 for name, p in field.named_parameters()}
                    head_gradient = {'finite': all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                                  for p in scene.refiner.parameters()),
                                     'squared_norm': sum(float(p.grad.square().sum()) for p in scene.refiner.parameters()
                                                         if p.grad is not None)}
                    del rendered, raw_vjp, raw_probe, final_probe
                gates = {'initial_p3d': errors['p3d'] <= SPEC['initial_probability_tolerance'],
                         'initial_final': errors['probabilities'] <= SPEC['initial_probability_tolerance'],
                         'alpha': errors['alpha'] <= SPEC['alpha_tolerance'],
                         'rgb': errors['rgb'] <= SPEC['rgb_tolerance'],
                         'simplex': errors['simplex'] <= SPEC['simplex_tolerance'],
                         'gradients': gradients is None or all(v['finite'] and v['norm'] > 0 for v in gradients.values()),
                         'raw_gradients': raw_gradients is None or all(v['finite'] and v['norm'] > 0 for v in raw_gradients.values()),
                         'head_gradient': head_gradient is None or head_gradient['finite'] and head_gradient['squared_norm'] > 0}
                record = {'name': camera['name'], 'mode': mode, 'errors': errors, 'gradients': gradients,
                          'raw_gradients': raw_gradients, 'head_gradient': head_gradient,
                          'forward_seconds': forward_seconds, 'forward_backward_seconds': backward_seconds,
                          'gates': gates, 'passed': all(gates.values())}
                records.append(record)
                print(json.dumps(record, allow_nan=False), flush=True)
                del field
            del baseline
        verify(plan)
        report = {'plan_sha256': expected, 'specification': SPEC, 'records': records,
                  'gaussians': len(logits), 'checkpoint_step': state.get('step'),
                  'status': 'passed' if all(r['passed'] for r in records) else 'failed_numerical_contract'}
        write(output/'analysis.json', report)
        actual_imports = {name: {'path': str(Path(module.__file__).resolve()), 'sha256': sha(module.__file__)}
                          for name, module in sys.modules.items()
                          if name.startswith('bridge_rgs') and getattr(module, '__file__', None)}
        for record in actual_imports.values():
            p = Path(record['path'])
            if not p.is_relative_to(snapshot) or plan['sources'][str(p.relative_to(snapshot))] != record['sha256']:
                raise ValueError('A loaded bridge module is not frozen')
        from gsplat.cuda._backend import _C
        binary = {'path': str(Path(_C.__file__).resolve()), 'sha256': sha(_C.__file__)}
        receipt.update(status=report['status'], inputs_and_sources_unchanged=True,
                       analysis_sha256=sha(output/'analysis.json'),
                       peak_memory_bytes=torch.cuda.max_memory_allocated(),
                       actual_package_imports=actual_imports, actual_gsplat_binary=binary)
        if report['status'] != 'passed':
            raise RuntimeError('Real-scene preflight failed')
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        receipt['elapsed_seconds'] = time.perf_counter()-beginning
        write(output/'execution_receipt.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare')
    parser.add_argument('--execute')
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if bool(args.prepare) == bool(args.execute):
        parser.error('Choose prepare or execute')
    if args.prepare:
        prepare(args.prepare)
    elif not args.expected_plan_sha256:
        parser.error('Expected plan SHA required')
    else:
        execute(args.execute, args.expected_plan_sha256)
