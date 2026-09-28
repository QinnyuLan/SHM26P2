"""Bounded synthetic CUDA audit; no bridge data, training or quality claims."""
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

import numpy as np
import torch

ROOT = Path('/home/sky/workspace/SHM2026')
SPEC = {'protocol': 'partition_rasterizer_synthetic_v1', 'seed': 20260927,
        'cases': ['regular', 'partial_tiles', 'empty_intersections', 'early_stop',
                  'threshold_neighbors', 'central_tiny_interval', 'tails'],
        'forward_absolute_tolerance': 2e-6, 'alpha_absolute_tolerance': 2e-6,
        'gradient_absolute_tolerance': 2e-5, 'gradient_relative_l2_tolerance': 5e-4,
        'simplex_absolute_tolerance': 2e-6,
        'directional_fd_absolute_tolerance': 2e-4, 'directional_fd_relative_tolerance': 2e-3,
        'directional_fd_step': 1e-3, 'directional_fd_cases': ['regular', 'partial_tiles'],
        'internal_seconds': 240, 'external_seconds': 300, 'retries': 0,
        'scene_reads': 0, 'optimizer_steps': 0, 'real_image_renders': 0,
        'scope': 'FP32 Triton vs independent FP64 fixed-visibility reference and gsplat constant endpoints'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def cases():
    rng = np.random.default_rng(SPEC['seed'])
    output = []
    for name in SPEC['cases']:
        width, height, n = (35, 19, 7) if name == 'partial_tiles' else (16, 16, 6)
        means = np.column_stack((rng.uniform(0, width, n), rng.uniform(0, height, n)))
        conics = np.tile([.035, .004, .06], (n, 1))
        opacity = rng.uniform(.1, .8, n)
        coeff = np.column_stack((rng.uniform(-.2, .2, (n, 2)), np.ones(n), -np.ones(n)))
        qin, qout = rng.dirichlet(np.ones(5), n), rng.dirichlet(np.ones(5), n)
        tile_count = ((width+15)//16)*((height+15)//16)
        ledger = [list(range(n)) for _ in range(tile_count)]
        if name == 'partial_tiles':
            ledger[1] = []
            ledger[-1] = [0, 2, 4]
        elif name == 'empty_intersections':
            ledger = [[] for _ in range(tile_count)]
        elif name == 'early_stop':
            means[:] = [8.5, 8.5]
            conics[:] = [1e-6, 0, 1e-6]
            opacity[:] = [.99, .99, .9, .5, .4, .3]
        elif name == 'threshold_neighbors':
            means[:] = [8.5, 8.5]
            conics[:] = [100., 0, 100.]
            center = np.float32(1/255)
            opacity[:] = [np.nextafter(center, np.float32(0)), center,
                          np.nextafter(center, np.float32(1)), .999, .9, .7]
        elif name == 'central_tiny_interval':
            coeff[:] = [0, 0, 1e-8, -1e-8]
        elif name == 'tails':
            coeff[:, :2] = 0
            coeff[:, 2:] = np.array([[8.1, 7.9], [-7.9, -8.1], [5.1, 4.9],
                                     [-4.9, -5.1], [.1, -.1], [3., -3.]])
        lengths = [len(row) for row in ledger]
        offsets = np.cumsum([0]+lengths[:-1], dtype=np.int32).reshape((height+15)//16, (width+15)//16)
        ids = np.array([g for row in ledger for g in row], dtype=np.int32)
        floats = [means, conics, opacity, qin, qout, coeff]
        floats = [torch.from_numpy(np.asarray(x, dtype=np.float32)) for x in floats]
        output.append({'name': name, 'width': width, 'height': height,
                       'values': (*floats[:3], torch.from_numpy(offsets), torch.from_numpy(ids), *floats[3:])})
    return output


def reference(values, width, height):
    """Independent dense FP64 compositing; full T and branch state retained."""
    means, conics, opacity, offsets, ids, qin, qout, coeff = values
    assert all(not x.is_cuda for x in values)
    y, x = torch.meshgrid(torch.arange(height, dtype=torch.float64)+.5,
                          torch.arange(width, dtype=torch.float64)+.5, indexing='ij')
    out = torch.zeros(height, width, 5, dtype=torch.float64)
    alpha = torch.zeros(height, width, 1, dtype=torch.float64)
    flat_offsets = offsets.flatten().tolist()+[len(ids)]
    for tile in range(offsets.numel()):
        yy, xx = divmod(tile, offsets.shape[1])
        rows, cols = slice(yy*16, min((yy+1)*16, height)), slice(xx*16, min((xx+1)*16, width))
        px, py = x[rows, cols], y[rows, cols]
        trans = torch.ones_like(px)
        done = torch.zeros_like(px, dtype=torch.bool)
        result = torch.zeros(*px.shape, 5, dtype=torch.float64)
        for g in ids[flat_offsets[tile]:flat_offsets[tile+1]].tolist():
            dx, dy = means[g, 0]-px, means[g, 1]-py
            a, b, c = conics[g]
            sigma = .5*(a*dx.square()+c*dy.square())+b*dx*dy
            opacity_at_pixel = (opacity[g]*torch.exp(-sigma)).clamp_max(.999)
            supported = (~done) & (sigma >= 0) & (opacity_at_pixel >= 1/255)
            next_trans = trans*(1-opacity_at_pixel)
            stop = supported & (next_trans <= 1e-4)
            use = supported & ~stop
            weight = torch.where(use, trans*opacity_at_pixel, 0.)
            affine = -coeff[g, 0]*dx-coeff[g, 1]*dy
            hi, lo = affine+coeff[g, 2], affine+coeff[g, 3]
            # erf difference is independent of the shader's tail-specific erfc.
            gate = .5*(torch.erf(hi/(2**.5))-torch.erf(lo/(2**.5)))
            result = result+weight[..., None]*(qout[g]+gate[..., None]*(qin[g]-qout[g]))
            trans = torch.where(use, next_trans, trans)
            done = done | stop
        bg = torch.tensor([1., 0, 0, 0, 0], dtype=torch.float64)
        out[rows, cols] = result+trans[..., None]*bg
        alpha[rows, cols, 0] = 1-trans
    return out, alpha


def check_case(case, shader):
    from gsplat.cuda._wrapper import rasterize_to_pixels
    cpu = tuple(x.double() if x.is_floating_point() else x for x in case['values'])
    gpu = tuple(x.cuda().requires_grad_(i >= 5) if x.is_floating_point() else x.cuda()
                for i, x in enumerate(case['values']))
    cpu = tuple(x.requires_grad_(i >= 5) if x.is_floating_point() else x for i, x in enumerate(cpu))
    width, height = case['width'], case['height']
    p, a = shader(*gpu, width=width, height=height)
    expected, ea = reference(cpu, width, height)
    # A deterministic signed linear loss probes every pixel/class.
    loss_weights = torch.cos(torch.arange(p.numel(), dtype=torch.float64)*.37).reshape(p.shape)/p.shape[0]/p.shape[1]
    if len(case['values'][4]):
        cg = torch.autograd.grad((expected*loss_weights).sum(), cpu[5:])
    else:
        cg = tuple(torch.zeros_like(v) for v in cpu[5:])
    gg = torch.autograd.grad((p*loss_weights.cuda().float()).sum(), gpu[5:])
    gradients = []
    for name, actual, target in zip(('qin', 'qout', 'coefficients'), gg, cg, strict=True):
        difference = actual.detach().cpu().double()-target
        absolute = float(difference.abs().max())
        relative = float(difference.norm()/target.norm().clamp_min(1e-12))
        passed = absolute <= SPEC['gradient_absolute_tolerance'] and (
            target.norm() < 1e-7 or relative <= SPEC['gradient_relative_l2_tolerance'])
        gradients.append({'name': name, 'maximum_absolute_error': absolute,
                          'relative_l2_error': relative, 'passed': bool(passed)})
    g = tuple(v.detach() for v in gpu)
    const, ca = shader(*g[:5], g[5], g[5], g[7], width=width, height=height)
    gs, gsalpha = rasterize_to_pixels(g[0][None], g[1][None], g[5][None], g[2][None],
        width, height, 16, g[3][None], g[4], backgrounds=torch.tensor([[1., 0, 0, 0, 0]], device='cuda'))
    maximum = lambda x: float(x.detach().abs().max().cpu())
    errors = {'reference_forward': maximum(p.cpu().double()-expected),
              'reference_alpha': maximum(a.cpu().double()-ea),
              'gsplat_constant_forward': maximum(const-gs[0]),
              'gsplat_alpha': maximum(ca-gsalpha[0]),
              'simplex': maximum(p.sum(-1)-1)}
    gates = {k: v <= SPEC['alpha_absolute_tolerance' if 'alpha' in k else
                         'simplex_absolute_tolerance' if k == 'simplex' else 'forward_absolute_tolerance']
             for k, v in errors.items()}
    fd = None
    if case['name'] in SPEC['directional_fd_cases']:
        direction = torch.sin(torch.arange(g[7].numel(), device='cuda')*.71).reshape(g[7].shape)
        direction /= direction.norm()
        h = SPEC['directional_fd_step']
        pp, _ = shader(*g[:7], (g[7]+h*direction).contiguous(), width=width, height=height)
        pm, _ = shader(*g[:7], (g[7]-h*direction).contiguous(), width=width, height=height)
        numerical = float(((pp-pm).double()*loss_weights.cuda()).sum()/(2*h))
        analytic = float((gg[2]*direction).sum())
        difference = abs(numerical-analytic)
        passed = difference <= SPEC['directional_fd_absolute_tolerance']+SPEC['directional_fd_relative_tolerance']*abs(analytic)
        fd = {'analytic': analytic, 'numerical': numerical, 'absolute_error': difference, 'passed': passed}
    return {'case': case['name'], 'width': width, 'height': height, 'errors': errors,
            'gates': gates, 'gradients': gradients, 'directional_fd': fd,
            'passed': all(gates.values()) and all(r['passed'] for r in gradients) and (fd is None or fd['passed'])}


def source_paths():
    gsroot = Path(importlib.util.find_spec('gsplat').origin).parent
    return [ROOT/'src/bridge_rgs/partition_rasterizer.py', Path(__file__).resolve(), ROOT/'uv.lock',
            gsroot/'cuda/_wrapper.py', gsroot/'cuda/csrc/RasterizeToPixels3DGSFwd.cu',
            gsroot/'cuda/include/Utils.cuh']


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Refuse existing output')
    snapshot = output/'source_snapshot'
    snapshot.mkdir(parents=True)
    copied = []
    for path in source_paths():
        target = snapshot/path.name
        shutil.copy2(path, target)
        copied.append({'original': str(path), 'snapshot': str(target), 'sha256': sha(target)})
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'triton', 'gsplat', 'numpy')}
    plan = {'specification': SPEC, 'source_root': str(snapshot), 'sources': copied,
            'runtime_versions': versions, 'output': str(output),
            'cpu_threads': 4, 'cuda_arch': '12.0', 'maximum_task_gpu_processes': 0,
            'unrelated_gpu_executable_allowed': '/usr/share/rustdesk/rustdesk'}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    if plan['specification'] != SPEC:
        raise ValueError('Changed specification')
    for source in plan['sources']:
        if sha(source['snapshot']) != source['sha256']:
            raise ValueError('Frozen source changed')
    if any(importlib.metadata.version(p) != v for p, v in plan['runtime_versions'].items()):
        raise ValueError('Runtime changed')


def execute(plan_path, expected_sha):
    if sha(plan_path) != expected_sha:
        raise ValueError('Plan digest differs')
    plan = json.loads(Path(plan_path).read_text())
    verify(plan)
    output, snapshot = Path(plan['output']), Path(plan['source_root'])
    if Path(__file__).resolve() != snapshot/Path(__file__).name:
        raise ValueError('Only frozen worker can execute')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True)
    for row in apps.splitlines():
        pid = int(row.split(',')[0])
        if Path(f'/proc/{pid}/exe').resolve().as_posix() != plan['unrelated_gpu_executable_allowed']:
            raise ValueError(f'Unexpected GPU process {pid}; leave it untouched')
    started = {'pid': os.getpid(), 'plan_sha256': expected_sha, 'gpu_processes_before': apps}
    write(output/'execution_started.json', started)
    start = time.perf_counter()
    receipt = {**started, 'status': 'running'}
    def timeout(_signal, _frame):
        raise TimeoutError('Synthetic CUDA preflight deadline')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(SPEC['internal_seconds'])
    try:
        torch.set_num_threads(plan['cpu_threads'])
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError('Unexpected GPU architecture')
        sys.path.insert(0, str(snapshot))
        shader_module = importlib.import_module('partition_rasterizer')
        if Path(shader_module.__file__).resolve() != snapshot/'partition_rasterizer.py':
            raise ValueError('Wrong loaded shader')
        results = []
        for case in cases():
            result = check_case(case, shader_module.rasterize_semantic_partition)
            results.append(result)
            print(json.dumps(result), flush=True)
        torch.cuda.synchronize()
        verify(plan)
        report = {'status': 'passed' if all(r['passed'] for r in results) else 'failed_numerical_contract',
                  'specification': SPEC, 'results': results, 'plan_sha256': expected_sha,
                  'scope': 'Synthetic correctness only; no measured bridge quality gain or novelty.'}
        write(output/'analysis.json', report)
        receipt.update(status=report['status'], source_hashes_unchanged=True,
                       analysis_sha256=sha(output/'analysis.json'),
                       max_memory_allocated_bytes=torch.cuda.max_memory_allocated())
        if report['status'] != 'passed':
            raise RuntimeError('Synthetic numerical contract failed')
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        receipt['elapsed_seconds'] = time.perf_counter()-start
        write(output/'execution_receipt.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare')
    parser.add_argument('--execute')
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if bool(args.prepare) == bool(args.execute):
        parser.error('Choose one of --prepare or --execute')
    if args.prepare:
        prepare(args.prepare)
    else:
        if not args.expected_plan_sha256:
            parser.error('Execution needs plan SHA')
        execute(args.execute, args.expected_plan_sha256)
