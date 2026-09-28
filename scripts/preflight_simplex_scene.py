"""Fixed H3 direct-simplex operator preflight; no real image/label reads or updates."""
from __future__ import annotations

import argparse
import difflib
import hashlib
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
OLD = Path('/mnt/data/SHM2026/runs/semantic_partition_matched_v2')
PRIOR = Path('/mnt/data/SHM2026/runs/partition_scene_preflight_v3')
FAILED_V1 = Path('/mnt/data/SHM2026/runs/simplex_scene_preflight_v1')
FAILED_V1_PLAN_SHA = '725ebd4d9f04815085aa3a9725886b6a60823c1958e1a0695cfb44501c9fee8a'
OLD_PLAN_SHA = 'f10d266083999abd1635ea8cb91e7407247dafba113f525a6ec7c790e3e4cc1d'
CKPT = ROOT/'runs/h3_moments/02_cross/last.pt'
CKPT_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
MANIFEST = ROOT/'artifacts/prepared/manifest.json'
MANIFEST_SHA = '551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa'
SPEC = {
    'protocol': 'simplex_scene_preflight_v2', 'names': ['002.png', '118.png'],
    'profile': 'legacy_mixed_v1', 'q0': 'original classifier FP32 softmax',
    'q1': [1/15, 2/15, 3/15, 4/15, 5/15],
    'midpoint': 'FP64 mean of FP32 q0/q1, cast FP32',
    'direction': 'FP64 q1 minus q0; each endpoint independently cast FP32',
    'coefficients': [0., 0., 1., -1.], 'same_leaf_endpoints': True,
    'probe_class_vectors': [[.3, -.2, .1, .8, -.4], [-.6, .2, .7, -.1, .3], [.1, .9, -.3, .2, -.5]],
    'probe_spatial': ['1', '1+x/width', '1+y/height+.25*((x//16+y//16)%2)'],
    'probe_dtype': 'FP32 including division by H*W; scalar products FP64',
    'adjoint_absolute': 2e-7, 'adjoint_relative': 5e-4,
    'minimum_signal_probes': 2, 'signal_floor_multiplier': 10.,
    'raw_identity_absolute': 3e-6, 'production_p3d_absolute': 3e-6,
    'midpoint_affine_absolute': 2e-6, 'alpha_absolute': 2e-6, 'simplex_absolute': 3e-6,
    'synthetic_labels': '(x//32+2*(y//32))%5', 'noise_delta': 5e-7,
    'ce_class_weights': [1., 1., 1., 1., 1.], 'primary_epsilon': .01,
    'sensitivity_epsilons': [.005, .0025], 'ce_absolute': 2e-6, 'ce_relative': .005,
    'counts': {'scene': 2, 'gsplat': 4, 'shader': 18, 'total_raster': 22, 'vjp': 8},
    'internal_seconds': 90, 'external_seconds': 120, 'retries': 0,
    'optimizer_steps': 0, 'label_reads': 0, 'image_reads': 0,
    'numerics': {'cudnn_allow_tf32': False, 'matmul_allow_tf32': False,
                 'matmul_precision': 'highest', 'cudnn_benchmark': False},
    'scope': 'two-view direct-q operator wiring/calibration, not a full adjoint certificate or convergence result',
}
ENV = {'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
       'TORCH_CUDA_ARCH_LIST': '12.0', 'MAX_JOBS': '4'}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    serialized = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as handle:
        handle.write(serialized)


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.md', '.lock')}


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def numerical_flags(value=None):
    old = {'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
           'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
           'matmul_precision': torch.get_float32_matmul_precision(),
           'cudnn_benchmark': torch.backends.cudnn.benchmark}
    if value is not None:
        torch.set_float32_matmul_precision(value['matmul_precision'])
        torch.backends.cudnn.allow_tf32 = value['cudnn_allow_tf32']
        torch.backends.cuda.matmul.allow_tf32 = value['matmul_allow_tf32']
        torch.backends.cudnn.benchmark = value['cudnn_benchmark']
    return old


def simplex_error(q):
    return {'minimum': float(q.min()), 'maximum': float(q.max()),
            'row_sum_max_abs_error': float((q.double().sum(-1)-1).abs().max()),
            'finite': bool(torch.isfinite(q).all())}


def simplex_points(q0):
    q1 = q0.new_tensor(SPEC['q1']).expand_as(q0).clone()
    qm = ((q0.double()+q1.double())/2).float().contiguous()
    return q1, qm, q1.double()-q0.double()


def central_points(qm, direction, epsilon):
    return ((qm.double()+epsilon*direction).float().contiguous(),
            (qm.double()-epsilon*direction).float().contiguous())


def fixed_probe(height, width, index, device):
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                          torch.arange(width, device=device, dtype=torch.float32), indexing='ij')
    spatial = (torch.ones_like(x), 1+x/width,
               1+y/height+.25*((x//16+y//16)%2))[index]
    classes = torch.tensor(SPEC['probe_class_vectors'][index], device=device, dtype=torch.float32)
    return (spatial[..., None]*classes/(height*width)).contiguous()


def synthetic_labels(height, width, device):
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    return (x//32+2*(y//32)) % 5


def dot64(left, right):
    return float((left.double()*right.double()).sum())


def adjoint_result(analytic, secant):
    scale = max(abs(analytic), abs(secant))
    tolerance = SPEC['adjoint_absolute']+SPEC['adjoint_relative']*scale
    return {'analytic': analytic, 'forward_secant': secant, 'absolute_error': abs(analytic-secant),
            'tolerance': tolerance, 'signal': scale,
            'measurable': scale > SPEC['signal_floor_multiplier']*SPEC['adjoint_absolute'],
            'within_tolerance': abs(analytic-secant) <= tolerance}


def adjoint_gate(rows):
    count = sum(row['measurable'] for row in rows)
    return {'signal_probe_count': count, 'sufficient_signal': count >= SPEC['minimum_signal_probes'],
            'passed': count >= SPEC['minimum_signal_probes'] and all(row['within_tolerance'] for row in rows)}


def fd_result(epsilon, gradient, plus, minus, loss_plus, loss_minus):
    direction = (plus.double()-minus.double())/(2*epsilon)
    products = gradient.double()*direction
    analytic = float(products.sum())
    secant = (loss_plus-loss_minus)/(2*epsilon)
    scalar_floor = (abs(float(np.spacing(loss_plus)))+abs(float(np.spacing(loss_minus))))/(2*epsilon)
    reduction_floor = float(float(products.abs().sum())*np.finfo(np.float64).eps*products.numel())
    floor = max(scalar_floor, reduction_floor)
    tolerance = SPEC['ce_absolute']+SPEC['ce_relative']*max(abs(analytic), abs(secant))
    measurable = bool(abs(analytic) > SPEC['signal_floor_multiplier']*floor)
    return {'epsilon': epsilon, 'primary': epsilon == SPEC['primary_epsilon'],
            'loss_plus': loss_plus, 'loss_minus': loss_minus, 'analytic_actual_displacement': analytic,
            'central_difference': secant, 'absolute_error': abs(analytic-secant), 'tolerance': tolerance,
            'relative_error': abs(analytic-secant)/abs(analytic) if analytic else None,
            'fp64_scalar_resolution_floor': scalar_floor, 'fp64_reduction_bound': reduction_floor,
            'measurable': measurable, 'within_tolerance': abs(analytic-secant) <= tolerance,
            'passed': measurable and abs(analytic-secant) <= tolerance,
            'actual_direction_max_abs': float(direction.abs().max()),
            'actual_direction_row_sum_max_abs': float(direction.sum(-1).abs().max()),
            'plus_simplex': simplex_error(plus), 'minus_simplex': simplex_error(minus)}


def validate_prior(plan, receipt, launch, plan_sha, receipt_sha):
    if receipt['status'] != 'passed' or receipt['plan_sha256'] != plan_sha:
        raise ValueError('Prior real integration not passed')
    if (launch.get('status', 'completed') != 'completed' or launch['exit_code'] != 0
            or not launch['natural_completion'] or launch['plan_sha256'] != plan_sha
            or launch['execution_receipt_sha256'] != receipt_sha):
        raise ValueError('Prior did not complete naturally with bound receipt')
    if not receipt['inputs_and_sources_unchanged'] or not plan['sources']:
        raise ValueError('Prior provenance missing')


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Refuse existing output')
    if sha(OLD/'plan.json') != OLD_PLAN_SHA or sha(CKPT) != CKPT_SHA or sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError('Fixed predecessor/checkpoint/manifest changed')
    if sha(FAILED_V1/'plan.json') != FAILED_V1_PLAN_SHA:
        raise ValueError('Failed v1 plan changed')
    failed_plan = json.loads((FAILED_V1/'plan.json').read_text())
    failed_receipt = json.loads((FAILED_V1/'execution_receipt.json').read_text())
    failed_launch = json.loads((FAILED_V1/'launch_receipt.json').read_text())
    if (failed_receipt['status'] != 'failed' or failed_launch['exit_code'] != 1
            or failed_launch['plan_sha256'] != FAILED_V1_PLAN_SHA
            or failed_launch['execution_receipt_sha256'] != sha(FAILED_V1/'execution_receipt.json')
            or failed_receipt['error'] != 'TypeError: Object of type bool is not JSON serializable'):
        raise ValueError('Expected preserved v1 serialization failure')
    if {k: v for k, v in failed_plan['specification'].items() if k != 'protocol'} != {
            k: v for k, v in SPEC.items() if k != 'protocol'}:
        raise ValueError('Numerical specification must not change in serialization repair')
    old = json.loads((OLD/'plan.json').read_text())
    prior_plan = json.loads((PRIOR/'plan.json').read_text())
    prior_receipt = json.loads((PRIOR/'execution_receipt.json').read_text())
    validate_prior(prior_plan, prior_receipt, json.loads((PRIOR/'launch_receipt.json').read_text()),
                   sha(PRIOR/'plan.json'), sha(PRIOR/'execution_receipt.json'))
    package = Path(old['source_snapshot'])/'bridge_rgs'
    expected_package = {key.removeprefix('bridge_rgs/'): value for key, value in old['source_hashes'].items()
                        if key.startswith('bridge_rgs/')}
    if tree(package) != expected_package:
        raise ValueError('Old package changed or has unbound source additions')
    views = json.loads(MANIFEST.read_text())['views']
    cameras = []
    for name in SPEC['names']:
        view = next(v for v in views if v['name'] == name)
        if view['split'] != 'train':
            raise ValueError('Non-TRAIN camera')
        cameras.append({k: view[k] for k in ('name', 'split', 'K', 'w2c_original', 'width', 'height')})
    snapshot = output/'source_snapshot'
    shutil.copytree(package, snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in (Path(__file__), ROOT/'tests/test_simplex_scene_preflight.py',
                 ROOT/'docs/simplex_scene_preflight_protocol.md', ROOT/'uv.lock'):
        shutil.copy2(path, snapshot/path.name)
    inputs = {str(CKPT): CKPT_SHA, str(MANIFEST): MANIFEST_SHA, str(OLD/'plan.json'): OLD_PLAN_SHA}
    for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json'):
        inputs[str(PRIOR/name)] = sha(PRIOR/name)
    for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'analysis.json'):
        inputs[str(FAILED_V1/name)] = sha(FAILED_V1/name)
    repair_files = {}
    for name in (Path(__file__).name, 'test_simplex_scene_preflight.py', 'simplex_scene_preflight_protocol.md'):
        old_path, new_path = FAILED_V1/'source_snapshot'/name, snapshot/name
        inputs[str(old_path)] = sha(old_path)
        repair_files[name] = {'old_sha256': sha(old_path), 'new_sha256': sha(new_path),
                             'unified_diff': ''.join(difflib.unified_diff(
                                 old_path.read_text().splitlines(keepends=True),
                                 new_path.read_text().splitlines(keepends=True), fromfile='v1/'+name, tofile='v2/'+name))}
    repair_path = output/'serialization_revision.json'
    write(repair_path, {'kind': 'JSON scalar/write repair only; old failed run not reclassified',
                        'failed_v1_plan_sha256': FAILED_V1_PLAN_SHA, 'files': repair_files})
    inputs[str(repair_path)] = sha(repair_path)
    binary = prior_receipt['actual_gsplat_binary']
    if sha(binary['path']) != binary['sha256']:
        raise ValueError('Installed gsplat binary changed')
    plan = {'specification': SPEC, 'output': str(output), 'snapshot': str(snapshot), 'sources': tree(snapshot),
            'inputs': inputs, 'cameras': cameras, 'environment': ENV,
            'installed_sources': prior_plan['installed_sources'], 'expected_gsplat_binary': binary,
            'inherited_package': str(package), 'inherited_package_hashes': tree(package),
            'runtime_versions': {p: importlib.metadata.version(p) for p in ('torch', 'triton', 'gsplat', 'numpy')},
            'prepare_label_reads': 0, 'prepare_image_reads': 0,
            'serialization_revision': {'path': str(repair_path), 'sha256': sha(repair_path)},
            'allowed_existing_gpu_executable': '/usr/share/rustdesk/rustdesk'}
    verify(plan)
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    if plan['specification'] != SPEC or tree(plan['snapshot']) != plan['sources']:
        raise ValueError('Frozen source/specification changed')
    if tree(Path(plan['snapshot'])/'bridge_rgs') != plan['inherited_package_hashes']:
        raise ValueError('Package inheritance changed')
    for path, digest in {**plan['inputs'], **plan['installed_sources'],
                         plan['expected_gsplat_binary']['path']: plan['expected_gsplat_binary']['sha256']}.items():
        if sha(path) != digest:
            raise ValueError(f'Bound input changed: {path}')
    for name, version in plan['runtime_versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Runtime version changed')


def capture_scene(scene):
    return {'hashes': {k: tensor_hash(v) for k, v in scene.state_dict().items()},
            'flags': {k: p.requires_grad for k, p in scene.named_parameters()},
            'gradients': {k: None if p.grad is None else p.grad.clone() for k, p in scene.named_parameters()},
            'modes': {k: m.training for k, m in scene.named_modules()}}


def restore_scene(scene, captured):
    for name, param in scene.named_parameters():
        param.requires_grad_(captured['flags'][name])
        param.grad = captured['gradients'][name]
    for name, module in scene.named_modules():
        module.training = captured['modes'][name]
    after = {k: tensor_hash(v) for k, v in scene.state_dict().items()}
    return {'state_before': captured['hashes'], 'state_after': after,
            'state_exact': after == captured['hashes'],
            'flags_modes_gradients_restored': all(p.requires_grad == captured['flags'][k]
                and ((p.grad is None and captured['gradients'][k] is None)
                     or (p.grad is not None and torch.equal(p.grad, captured['gradients'][k])))
                for k, p in scene.named_parameters())
                and all(m.training == captured['modes'][k] for k, m in scene.named_modules())}


def capture_context(scene, K, w2c, width, height, counters):
    """Unchanged scene call, intercept original semantic raw channels without extra raster."""
    import gsplat
    captured, original = {}, gsplat.rasterization
    def record_raw(*args, **kwargs):
        counters['gsplat'] += 1
        result = original(*args, **kwargs)
        if kwargs.get('render_mode') != 'RGB+ED':
            if captured:
                raise ValueError('Expected exactly one original semantic raster')
            captured.update(raw=result[0][0, ..., :5], alpha=result[1][0])
        return result
    gsplat.rasterization = record_raw
    try:
        counters['scene'] += 1
        with torch.no_grad():
            context = scene.render(K, w2c, width, height, refine=False, absgrad=False)
    finally:
        gsplat.rasterization = original
    if set(captured) != {'raw', 'alpha'}:
        raise ValueError('Original semantic raw capture missing')
    return context, captured


def direct_q(info, q, rasterize, *, width, height):
    """Same q Tensor passed twice; autograd adds both endpoint VJPs. No renormalization."""
    if not q.is_contiguous():
        raise ValueError('Single contiguous q required')
    geometry = [info['means2d'][0].detach().contiguous(), info['conics'][0].detach().contiguous(),
                info['opacities'][0].detach().contiguous(), info['isect_offsets'][0].contiguous(),
                info['flatten_ids'].contiguous()]
    coefficients = q.new_tensor(SPEC['coefficients']).expand(len(q), -1).contiguous()
    return rasterize(*geometry, q, q, coefficients, width=width, height=height)


def run_view(scene, camera, counters, rasterize, affine_ce):
    K = torch.tensor(camera['K'], dtype=torch.float32, device='cuda')
    w2c = torch.tensor(camera['w2c_original'], dtype=torch.float32, device='cuda')
    width, height = camera['width'], camera['height']
    camera_hashes = [tensor_hash(K), tensor_hash(w2c)]
    context, captured = capture_context(scene, K, w2c, width, height, counters)
    preserved = {k: tensor_hash(context[k]) for k in ('rgb', 'depth', 'alpha', 'features', 'depth_moments')
                 if k in context}
    info = context['info']
    with torch.no_grad():
        q0 = scene.semantic_decoder(scene.splats['sem_features']).softmax(-1).detach().contiguous()
    q1, qm, direction = simplex_points(q0)
    def render(q):
        counters['shader'] += 1
        return direct_q(info, q, rasterize, width=width, height=height)
    with torch.no_grad():
        raw0, alpha0 = render(q0)
        raw1, alpha1 = render(q1)
    qm.requires_grad_(True)
    rawm, alpham = render(qm)
    maximum = lambda x: float(x.detach().abs().max())
    p0 = raw0.clamp_min(1e-7)
    p0 = p0/p0.sum(-1, keepdim=True)
    errors = {'production_p3d': maximum(p0-context['p3d']),
              'original_raw5': maximum(raw0-captured['raw']),
              'midpoint_affine': maximum(rawm.double()-(raw0.double()+raw1.double())/2),
              'alpha_vs_rgb': maximum(alpha0-context['alpha']),
              'alpha_vs_semantic': maximum(alpha0-captured['alpha']),
              'alpha_independent_of_q': max(maximum(alpha0-alpha1), maximum(alpha0-alpham))}
    adjoint = []
    for index in range(3):
        probe = fixed_probe(height, width, index, qm.device)
        scalar = (rawm.double()*probe.double()).sum()
        grad, = torch.autograd.grad(scalar, (qm,), retain_graph=True)
        counters['vjp'] += 1
        if not bool(torch.isfinite(grad).all()):
            raise ValueError('Nonfinite direct-q VJP')
        adjoint.append({'probe': index, **adjoint_result(dot64(grad, direction), dot64(probe, raw1.double()-raw0.double()))})
    labels = synthetic_labels(height, width, qm.device)
    valid = torch.ones_like(labels, dtype=torch.bool)
    weights = qm.new_tensor(SPEC['ce_class_weights'])
    loss = affine_ce(rawm, labels, valid, weights, SPEC['noise_delta'])
    grad, = torch.autograd.grad(loss, (qm,))
    counters['vjp'] += 1
    if not bool(torch.isfinite(grad).all()):
        raise ValueError('Nonfinite CE VJP')
    fd = []
    for epsilon in [SPEC['primary_epsilon'], *SPEC['sensitivity_epsilons']]:
        plus, minus = central_points(qm.detach(), direction, epsilon)
        with torch.no_grad():
            rp, _ = render(plus)
            lp = float(affine_ce(rp, labels, valid, weights, SPEC['noise_delta']))
            rm, _ = render(minus)
            lm = float(affine_ce(rm, labels, valid, weights, SPEC['noise_delta']))
        fd.append(fd_result(epsilon, grad, plus, minus, lp, lm))
    simplexes = {key: simplex_error(q) for key, q in (('q0', q0), ('q1', q1), ('qm', qm.detach()))}
    context_exact = all(tensor_hash(context[k]) == digest for k, digest in preserved.items())
    cameras_exact = camera_hashes == [tensor_hash(K), tensor_hash(w2c)]
    gates = {'raw_identity': errors['original_raw5'] <= SPEC['raw_identity_absolute'],
             'production_identity': errors['production_p3d'] <= SPEC['production_p3d_absolute'],
             'affine_midpoint': errors['midpoint_affine'] <= SPEC['midpoint_affine_absolute'],
             'alpha': max(errors[k] for k in ('alpha_vs_rgb', 'alpha_vs_semantic', 'alpha_independent_of_q')) <= SPEC['alpha_absolute'],
             'simplex': all(v['finite'] and v['minimum'] >= 0 and v['row_sum_max_abs_error'] <= SPEC['simplex_absolute'] for v in simplexes.values()),
             'adjoint': adjoint_gate(adjoint)['passed'], 'primary_ce': fd[0]['passed'],
             'context_exact': context_exact, 'camera_exact': cameras_exact}
    return {'name': camera['name'], 'errors': errors, 'simplex_cast_errors': simplexes,
            'same_leaf': qm.is_leaf, 'same_leaf_arguments': True, 'coefficients': SPEC['coefficients'],
            'fixed_background': 'T e_bg independent of q; cancels in forward secant, derivative zero',
            'adjoint': adjoint, 'adjoint_signal': adjoint_gate(adjoint), 'ce_at_midpoint': float(loss.detach()),
            'ce_fd': fd, 'context_hashes': preserved, 'camera_hashes': camera_hashes,
            'gates': gates, 'passed': all(gates.values())}


def start_attempt(output, expected):
    """Exclusive marker precedes verification/GPU work; failed attempts remain consumed."""
    output = Path(output)
    if any((output/name).exists() for name in ('execution_started.json', 'execution_receipt.json', 'analysis.json')):
        raise ValueError('This plan already has an attempt or output; no retry/overwrite')
    write(output/'execution_started.json', {'plan_sha256': expected, 'pid': os.getpid(),
                                          'started_unix_seconds': time.time()})


def execute(path, expected):
    if sha(path) != expected:
        raise ValueError('Plan SHA mismatch')
    plan = json.loads(Path(path).read_text())
    output, snapshot = Path(plan['output']), Path(plan['snapshot'])
    if Path(__file__).resolve() != snapshot/Path(__file__).name:
        raise ValueError('Execute frozen worker only')
    start_attempt(output, expected)
    counts = {'scene': 0, 'gsplat': 0, 'shader': 0, 'vjp': 0}
    receipt = {'plan_sha256': expected, 'status': 'running', 'pid': os.getpid(),
               'label_reads': 0, 'image_reads': 0, 'optimizer_steps': 0, 'counts': counts}
    started, scene, captured, previous_flags = time.perf_counter(), None, None, numerical_flags()
    def deadline(_signum, _frame):
        raise TimeoutError('Internal preflight deadline; no retry')
    previous_alarm = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan)
        for key, value in plan['environment'].items():
            if os.environ.get(key) != value:
                raise ValueError(f'Execution environment mismatch: {key}')
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                       '--format=csv,noheader'], text=True, timeout=5)
        for row in apps.splitlines():
            pid = int(row.split(',')[0])
            if str(Path(f'/proc/{pid}/exe').resolve()) != plan['allowed_existing_gpu_executable']:
                raise ValueError(f'Unexpected GPU client {pid}')
        receipt['gpu_processes_before'] = apps
        receipt['gpu_shared_with_desktop'] = bool(apps.strip())
        torch.set_num_threads(4)
        numerical_flags(SPEC['numerics'])
        receipt['numerics_original'] = previous_flags
        receipt['numerics_actual'] = numerical_flags()
        sys.path.insert(0, str(snapshot))
        from bridge_rgs.partition_rasterizer import rasterize_semantic_partition
        from bridge_rgs.semantic_assignment import affine_raw_ce
        from bridge_rgs.train import load_scene
        scene, state = load_scene(CKPT)
        captured = capture_scene(scene)
        scene.eval().requires_grad_(False)
        if scene.pixel_protocol != SPEC['profile'] or scene.mip_filter_config is not None:
            raise ValueError('Wrong H3 profile/filter')
        records = [run_view(scene, camera, counts, rasterize_semantic_partition, affine_raw_ce) for camera in plan['cameras']]
        counts['total_raster'] = counts['gsplat']+counts['shader']
        if counts != SPEC['counts']:
            raise ValueError('Call budget mismatch')
        actual = {name: {'path': str(Path(module.__file__).resolve()), 'sha256': sha(module.__file__)}
                  for name, module in sys.modules.items() if name.startswith('bridge_rgs') and getattr(module, '__file__', None)}
        for record in actual.values():
            p = Path(record['path'])
            if not p.is_relative_to(snapshot) or plan['sources'].get(str(p.relative_to(snapshot))) != record['sha256']:
                raise ValueError('Non-frozen package import')
        from gsplat.cuda._backend import _C
        binary = {'path': str(Path(_C.__file__).resolve()), 'sha256': sha(_C.__file__)}
        if binary != plan['expected_gsplat_binary']:
            raise ValueError('Actual gsplat binary differs')
        verify(plan)
        report = {'specification': SPEC, 'plan_sha256': expected, 'records': records,
                  'numerical_status': 'passed' if all(r['passed'] for r in records) else 'failed_or_inconclusive',
                  'counts': counts, 'gaussians': len(scene.splats['means']), 'checkpoint_step': state['step']}
        write(output/'analysis.json', report)
        receipt.update(status='completed', numerical_status=report['numerical_status'], analysis_sha256=sha(output/'analysis.json'),
                       actual_package_imports=actual, actual_gsplat_binary=binary, inputs_and_sources_unchanged=True,
                       peak_memory_allocated=torch.cuda.max_memory_allocated(), peak_memory_reserved=torch.cuda.max_memory_reserved())
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            if scene is not None and captured is not None:
                receipt['restoration'] = restore_scene(scene, captured)
                if not receipt['restoration']['state_exact'] or not receipt['restoration']['flags_modes_gradients_restored']:
                    receipt['status'] = 'failed_restoration'
            numerical_flags(previous_flags)
            receipt['numerics_restored'] = numerical_flags() == previous_flags
            receipt['elapsed_seconds'] = time.perf_counter()-started
            receipt['counts']['total_raster'] = counts['gsplat']+counts['shader']
            write(output/'execution_receipt.json', receipt)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_alarm)
    if receipt['status'] != 'completed':
        raise RuntimeError('Preflight failed; keep result without retry')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path)
    group.add_argument('--run', type=Path)
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    else:
        execute(args.run, args.expected_plan_sha256)
