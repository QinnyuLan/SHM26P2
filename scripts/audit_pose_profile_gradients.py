"""Fixed TRAIN-only gradients of known pose/geometry interventions.

A regularized linear profile application, not new GLS or full marginal likelihood.
Preparation and numerical contracts are CPU-only. GPU execution needs a later
explicit handoff; no optimizer, real RGB/semantic reads, or candidate checkpoints.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

H1_PLAN = Path('runs/h1_attribution_recovery_v1/plan.json')
H1_SHA = 'b46a07939a3f0c96beb03b9f84825ed2d8a452d0d24dc8b443219e61190a9f6e'
V2 = Path('/mnt/data/SHM2026/runs/action_transfer_probe_v2')
V2_SHA = '1bbb5ffc17d2c41bb3a17bc61a6c8b3f616916951e535a92f769bad8a8269a79'
COVERAGE_SHA = 'be3066a7d04ba9d4d46d2f86fe15eb09fbe3fa25aa3cf1d0580c8eb005c92166'
NAMES = ('003.png', '058.png', '115.png', '170.png', '232.png', '288.png', '344.png', '400.png')
GEOMETRY_NAMES = ('003.png', '058.png', '115.png', '170.png', '344.png')
ARMS = ('raw', 'profile', 'permuted')
SPEC = {
    'protocol': 'frozen_pose_profile_means_gradient_diagnostic_v1',
    'views': list(NAMES), 'geometry_gate_views': list(GEOMETRY_NAMES),
    'width': 320, 'sh_degree': 3, 'repeats': 2,
    'translation_fd_scene_scale': 1e-5, 'rotation_fd': 1e-4,
    'translation_prior_std_scene_scale': .005, 'rotation_prior_std': .01,
    'damping': .001, 'local_displacement_scene_scale': .001,
    'permutation_seed': 20260926, 'permutation': 'fixed valid whitened scalar rows, shared by both conditions',
    'region': 'v2 region0 closed ball, all original 22 means, no opacity/size/error selection',
    'pca_sign': 'largest absolute component positive; ties choose first xyz coordinate',
    'loss': '(weighted residual squared + delta quadratic penalty) / (2 sumW)',
    'regularizer': 'prior_precision + .001 diag(diag(JtWJ).clamp_min(1)); true J only, fixed across arms',
    'delta': 'unconstrained exact six-dimensional solve; no trust clipping or nonlinear acceptance',
    'gradient': 'all shared Gaussian means only, same world units; no mixed-parameter norm',
    'pose_case': 'theta0 at exp(half_twist)T, J here; target R(theta0,T), correct field change zero',
    'geometry_case': 'theta0+d at T, J here; target R(theta0,T), known repair u=-actual_FP32_d',
    'noise_multiplier': 10., 'fp32_floor_ulps': 32.,
    'pose_required_views': 6, 'pose_norm_ratio_max': .5,
    'geometry_required_measurable_views': 5, 'geometry_required_preserved_views': 4,
    'geometry_direction_retention_min': .5,
    'renders_per_view': 33, 'backwards_per_view': 14,
    'total_renders': 264, 'total_backwards': 112,
    'budget_formula': '8 * [1 target + 2 null + 2*(13 FD + 2 graph renders)]; 8*(2 null + 2*2*3 backward)',
    'worker_wall_seconds': 180, 'torch_threads': 8,
    'rgb_or_semantic_pixels_read': 0, 'optimizer_steps': 0,
    'interpretation': 'necessary fixed-field/one-region gradient diagnostic only; not rendering gains or novelty',
}
VIEW_KEYS = ('name', 'split', 'width', 'height', 'K', 'w2c', 'valid_path', 'half_twist')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data, *, replace=False):
    path = Path(path)
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if replace:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(text)
        temporary.replace(path)
    else:
        with path.open('x') as stream:
            stream.write(text)


def tensor_hash(value):
    value = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def whitened_system(jacobian, weights, scene_scale, permutation):
    """Detached N x 6 / 6 x 6 state. No N x N operator is ever constructed."""
    require(jacobian.shape[-1] == 6 and scene_scale > 0, 'Six columns and positive scale required')
    shape = jacobian.shape[:-1]
    w = torch.broadcast_to(weights.detach(), shape).reshape(-1).double()
    require(bool(torch.isfinite(w).all() & (w >= 0).all()) and bool(w.sum() > 0), 'Invalid weights')
    index = (w > 0).nonzero().flatten()
    sqrt_w = w[index].sqrt()
    j = jacobian.detach().reshape(-1, 6)[index].double()*sqrt_w[:, None]
    require(bool(torch.isfinite(j).all()), 'Active Jacobian is not finite')
    permutation = torch.as_tensor(permutation, device=j.device, dtype=torch.long)
    require(permutation.shape == (len(index),) and
            torch.equal(permutation.sort().values, torch.arange(len(index), device=j.device)), 'Invalid row permutation')
    normal = j.T @ j
    prior = j.new_tensor([1/(.005*scene_scale)**2]*3+[1/.01**2]*3)
    precision = torch.diag(prior + .001*normal.diagonal().clamp_min(1))
    permuted = j[permutation]
    require(torch.allclose(permuted.T @ permuted, normal, atol=1e-9, rtol=1e-10), 'Whitened normal not preserved')
    cholesky = torch.linalg.cholesky(normal+precision)
    return {'index': index, 'sqrt_w': sqrt_w, 'sum_w': w.sum(),
            'j': j, 'permuted_j': permuted, 'normal': normal,
            'precision': precision, 'cholesky': cholesky}


def profile_losses(residual, system):
    """Envelope derivative through r only; the optimal nuisance is detached."""
    r = residual.reshape(-1)[system['index']].double()*system['sqrt_w']
    require(bool(torch.isfinite(r).all()), 'Active residual is not finite')
    result = {'raw': r.square().sum()/(2*system['sum_w'])}
    for name, key in (('profile', 'j'), ('permuted', 'permuted_j')):
        j = system[key]
        delta = -torch.cholesky_solve((j.T@r)[:, None], system['cholesky'])[:, 0].detach()
        remaining = r+j@delta
        # Positive sum avoids subtracting two almost equal residual energies.
        result[name] = (remaining.square().sum()+delta@system['precision']@delta)/(2*system['sum_w'])
    return result


def raw_loss(residual, weights):
    w = torch.broadcast_to(weights.detach(), residual.shape).double()
    return (w*residual.double().square()).sum()/(2*w.sum())


def fixed_permutation(size, view_index):
    return np.random.default_rng(SPEC['permutation_seed']+view_index).permutation(size)


def local_geometry(means, region, views, scene_scale):
    """Selection depends only on the already fixed CPU region and original means."""
    m = means.detach().cpu().double().numpy()
    ids = np.flatnonzero(np.linalg.norm(m-np.asarray(region['anchor_xyz']), axis=1) <= region['radius'])
    require(len(ids) == 22, 'Fixed v2 region must contain exactly the original 22 Gaussians')
    centered = m[ids]-m[ids].mean(0)
    eigenvalues, vectors = np.linalg.eigh(centered.T@centered/len(ids))
    require(eigenvalues[-1] > eigenvalues[-2]*(1+1e-8), 'PCA leading direction is ambiguous')
    axis = vectors[:, -1]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    amplitude = .001*scene_scale
    shifted = means.detach().cpu().clone()
    shifted[ids] += torch.as_tensor(amplitude*axis, dtype=shifted.dtype)
    displacement = shifted.double()-means.detach().cpu().double()
    support = []
    for view in views:
        row = {'name': view['name']}
        for label, points in (('original', m[ids]), ('displaced', shifted[ids].double().numpy())):
            pose, k = np.asarray(view['w2c']), np.asarray(view['K'])
            pc = points@pose[:3, :3].T+pose[:3, 3]
            q = pc@k.T
            uv = q[:, :2]/q[:, 2:]
            inside = (pc[:, 2] > .01) & (uv[:, 0] >= 0) & (uv[:, 0] < view['width'])
            inside &= (uv[:, 1] >= 0) & (uv[:, 1] < view['height'])
            row[f'{label}_positive_depth_center_in_frame'] = int(inside.sum())
        support.append(row)
    expected = [r['name'] for r in support if r['original_positive_depth_center_in_frame'] > 0]
    require(expected == list(GEOMETRY_NAMES), 'Predeclared geometry support set changed')
    return {'indices': ids.tolist(), 'axis': axis.tolist(), 'eigenvalues': eigenvalues.tolist(),
            'amplitude': amplitude, 'actual_fp32_displacement': displacement[ids].tolist(),
            'repair_direction_l2': float(displacement.norm()), 'per_view_center_support': support,
            'support_interpretation': 'center frusta only, not actual visibility or numerical measurability'}


@contextlib.contextmanager
def preserved_scene(scene):
    """Restore tensor bytes, buffers, gradient permissions and gradients on every exit."""
    state = {name: value.detach().clone() for name, value in scene.state_dict().items()}
    flags = {name: p.requires_grad for name, p in scene.named_parameters()}
    gradients = {name: None if p.grad is None else p.grad.detach().clone() for name, p in scene.named_parameters()}
    try:
        yield state
    finally:
        current = scene.state_dict()
        require(set(current) == set(state), 'State schema changed during diagnostic')
        with torch.no_grad():
            for name, value in current.items():
                value.copy_(state[name])
        for name, p in scene.named_parameters():
            p.requires_grad_(flags[name])
            p.grad = gradients[name]
        require(all(torch.equal(current[k], v) for k, v in state.items()), 'Scene restoration failed')


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), 'Preserve every prior plan/result; no overwrite')
    h1_path, v2_path, coverage_path = root/H1_PLAN, V2/'plan.json', V2/'coverage.json'
    require(digest(h1_path) == H1_SHA and digest(v2_path) == V2_SHA
            and digest(coverage_path) == COVERAGE_SHA, 'Fixed source protocols changed')
    h1, v2 = json.loads(h1_path.read_text()), json.loads(v2_path.read_text())
    coverage = json.loads(coverage_path.read_text())
    previous = h1_path.parent/'execution_receipt.json'
    previous_record = json.loads(previous.read_text())
    require(previous_record['status'] == 'completed' and previous_record['plan_sha256'] == H1_SHA,
            'H1 reference diagnostic must be completed against the fixed source plan')
    checkpoint = Path(h1['checkpoint'])
    require(digest(checkpoint) == h1['input_hashes'][str(checkpoint)], 'Fixed H1 checkpoint changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    require(state['step'] == 6000 and state['sh_degree'] == 3, 'Wrong frozen H1 field')
    require(float(state['scene_scale']) == h1['scene_scale'] == v2['scene_scale'], 'Scene scale mismatch')
    views = [{key: view[key] for key in VIEW_KEYS} for view in h1['views']]
    require([v['name'] for v in views] == list(NAMES) and all(v['split'] == 'train' for v in views), 'Fixed TRAIN views changed')
    geometry = local_geometry(state['model']['splats.means'], coverage['regions'][0], views, h1['scene_scale'])
    paths = [h1_path, previous, v2_path, coverage_path, checkpoint, Path(h1['manifest']),
             root/'artifacts/pose_stress_mild/manifest.json', root/'uv.lock']
    paths.extend(Path(v['valid_path']) for v in views)
    inputs = {str(p): digest(p) for p in paths}
    reused_h1_paths = (str(checkpoint), h1['manifest'], str(root/'uv.lock'),
                      str(root/'artifacts/pose_stress_mild/manifest.json'),
                      *(v['valid_path'] for v in views))
    for path in reused_h1_paths:
        require(inputs[path] == h1['input_hashes'][path], 'Historical input changed')
    origin = Path(h1['source_snapshot'])
    for relative, sha in h1['source_hashes'].items():
        require(digest(origin/relative) == sha, 'Frozen H1 source changed')
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(origin/'bridge_rgs', snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    shutil.copy2(root/'tests/test_pose_profile_gradients.py', snapshot/'test_pose_profile_gradients.py')
    sources = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob('*.py'))}
    require(all(sources[k] == v for k, v in h1['source_hashes'].items() if k.startswith('bridge_rgs/')), 'Old production source copy changed')
    plan = {'status': 'cpu_locked_pending_explicit_gpu_handoff', 'created_utc': datetime.now(UTC).isoformat(),
            'specification': SPEC, 'root': str(root), 'output': str(output), 'source_snapshot': str(snapshot),
            'source_hashes': sources, 'input_hashes': inputs, 'checkpoint': str(checkpoint),
            'manifest': h1['manifest'], 'scene_scale': h1['scene_scale'], 'views': views, 'geometry': geometry,
            'checkpoint_sha256': digest(checkpoint), 'gpu_authorized': False,
            'timing_basis': 'Prior same field/8views/240 no-backward render audit 6.726s; 264 renders +112 backward unmeasured. 180s child wall, no retries.',
            'cpu_preparation_rgb_valid_or_semantic_decodes': 0, 'cpu_cuda_initialized': torch.cuda.is_initialized()}
    require(not plan['cpu_cuda_initialized'], 'CPU prepare initialized CUDA')
    write_json(output/'plan.json', plan)
    return output/'plan.json'


def norm(value):
    return float(value.double().norm())


def weighted_rms(value, weights):
    w = torch.broadcast_to(weights, value.shape).double()
    return float(((value.double().square()*w).sum()/w.sum()).sqrt())


def gradient_repeats(render, pose, target, weights, means, system=None):
    rows = []
    for _ in range(2):
        prediction = render(pose)
        residual = prediction-target
        losses = {'raw': raw_loss(residual, weights)} if system is None else profile_losses(residual, system)
        gradients = {}
        for index, (name, loss) in enumerate(losses.items()):
            gradients[name] = torch.autograd.grad(loss, means, retain_graph=index < len(losses)-1)[0].detach()
            require(bool(torch.isfinite(gradients[name]).all()), 'Nonfinite means gradient')
        rows.append({'gradient': gradients, 'residual': residual.detach(),
                     'prediction_rms': weighted_rms(prediction.detach(), weights),
                     'loss': {name: float(value.detach()) for name, value in losses.items()}})
    return rows


def condition_summary(repeats, null, target, weights, repair):
    """All floors, ratios and direction signs are descriptive or preregistered."""
    gradients = {arm: (repeats[0]['gradient'][arm].double()+repeats[1]['gradient'][arm].double())/2 for arm in ARMS}
    norms = {arm: norm(g) for arm, g in gradients.items()}
    gradient_repeat = {arm: norm(repeats[0]['gradient'][arm].double()-repeats[1]['gradient'][arm].double()) for arm in ARMS}
    null_norms = [norm(r['gradient']['raw']) for r in null]
    fp32 = SPEC['fp32_floor_ulps']*torch.finfo(torch.float32).eps
    gradient_floor = max(*null_norms, *gradient_repeat.values(), fp32*max(norms.values()))
    null_rms = [weighted_rms(r['residual'], weights) for r in null]
    repeat_rms = weighted_rms(repeats[0]['residual'].double()-repeats[1]['residual'].double(), weights)
    residual_rms = sum(weighted_rms(r['residual'], weights) for r in repeats)/2
    rgb_floor = max(*null_rms, repeat_rms, fp32*max(1., weighted_rms(target, weights), *(r['prediction_rms'] for r in repeats)))
    dots = {arm: float(-(g*repair.double()).sum()) for arm, g in gradients.items()}
    null_dots = [abs(float((r['gradient']['raw'].double()*repair.double()).sum())) for r in null]
    dot_repeat = [abs(float(((repeats[0]['gradient'][arm].double()-repeats[1]['gradient'][arm].double())*repair.double()).sum())) for arm in ARMS]
    dot_rounding = fp32*max(float((g*repair.double()).abs().sum()) for g in gradients.values())
    dot_floor = max(*null_dots, *dot_repeat, dot_rounding)
    multiplier = SPEC['noise_multiplier']
    measurable = residual_rms > multiplier*rgb_floor and norms['raw'] > multiplier*gradient_floor
    directional_measurable = measurable and dots['raw'] > multiplier*dot_floor
    ratio = lambda numerator, denominator: numerator/denominator if denominator > 0 else None
    pose_pass = measurable and all(
        norms['profile'] <= SPEC['pose_norm_ratio_max']*norms[control]
        and norms[control]-norms['profile'] > multiplier*gradient_floor for control in ('raw', 'permuted'))
    geometry_pass = (directional_measurable and dots['profile'] > 0
                     and dots['profile'] >= SPEC['geometry_direction_retention_min']*dots['raw'])
    return {'residual_rms': residual_rms, 'residual_noise_floor': rgb_floor,
            'raw_signal_measurable': measurable, 'gradient_norm': norms,
            'gradient_noise_floor': gradient_floor, 'gradient_repeat_difference_norm': gradient_repeat,
            'null_gradient_norm': null_norms, 'null_residual_rms': null_rms,
            'profile_over_raw_norm': ratio(norms['profile'], norms['raw']),
            'profile_over_permuted_norm': ratio(norms['profile'], norms['permuted']),
            'repair_descent_strength_minus_g_dot_u': dots, 'directional_noise_floor': dot_floor,
            'geometry_direction_measurable': directional_measurable,
            'profile_direction_retention': ratio(dots['profile'], dots['raw']),
            'permuted_direction_retention': ratio(dots['permuted'], dots['raw']),
            'pose_specific_suppression_gate': bool(pose_pass),
            'geometry_repair_retention_gate': bool(geometry_pass),
            'losses_not_a_benefit_metric': [r['loss'] for r in repeats]}


def summarize(records):
    require([r['name'] for r in records] == list(NAMES), 'All eight views must be reported')
    pose_measurable = sum(r['pose']['raw_signal_measurable'] for r in records)
    pose_pass = sum(r['pose']['pose_specific_suppression_gate'] for r in records)
    geometric = [r for r in records if r['name'] in GEOMETRY_NAMES]
    geometry_measurable = sum(r['geometry']['geometry_direction_measurable'] for r in geometric)
    geometry_pass = sum(r['geometry']['geometry_repair_retention_gate'] for r in geometric)
    if pose_measurable < 6 or geometry_measurable != 5:
        status = 'inconclusive_measurability'
    elif pose_pass >= 6 and geometry_pass >= 4:
        status = 'necessary_gradient_gates_passed'
    else:
        status = 'necessary_gradient_gates_not_met'
    return {'status': status, 'pose_measurable_views': pose_measurable, 'pose_gate_views': pose_pass,
            'geometry_predeclared_views': list(GEOMETRY_NAMES), 'geometry_measurable_views': geometry_measurable,
            'geometry_retention_views': geometry_pass,
            'scope': 'One fixed self-rendered field and one fixed region only; no scene-performance or novelty result'}


def verify(plan):
    require(plan['specification'] == SPEC, 'Fixed specification changed')
    snapshot = Path(plan['source_snapshot'])
    files = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob('*.py'))}
    require(files == plan['source_hashes'], 'Snapshot changed')
    for path, sha in plan['input_hashes'].items():
        require(digest(path) == sha, f'Input changed: {path}')


def load_camera_only(view):
    import cv2
    require(set(view) == set(VIEW_KEYS) and view['split'] == 'train', 'Camera whitelist changed')
    valid = cv2.imread(view['valid_path'], cv2.IMREAD_GRAYSCALE)
    require(valid is not None and valid.shape == (view['height'], view['width']), 'Wrong native valid dimensions')
    width, height = 320, max(8, round(view['height']*320/view['width']))
    valid = cv2.resize(valid, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    k = np.asarray(view['K'], np.float32).copy()
    k[0] *= width/view['width']
    k[1] *= height/view['height']
    return torch.tensor(k, device='cuda'), torch.tensor(view['w2c'], device='cuda').float(), torch.tensor(valid, device='cuda').float()[..., None], width, height


def worker(plan_path):
    import cv2
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Use immutable runner')
    verify(plan)
    modules = {name: importlib.import_module('bridge_rgs.'+name) for name in ('train', 'model', 'reliability', 'refinement')}
    for name, module in modules.items():
        require(Path(module.__file__).resolve() == Path(plan['source_snapshot'])/'bridge_rgs'/f'{name}.py', 'Wrong actual import')
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    torch.manual_seed(42)
    read_paths = []
    allowed = {str(Path(v['valid_path']).resolve()) for v in plan['views']}
    original_read = cv2.imread
    def guarded_read(path, *args, **kwargs):
        path = str(Path(path).resolve())
        require(path in allowed, 'Only fixed TRAIN validity masks may be decoded')
        read_paths.append(path)
        return original_read(path, *args, **kwargs)
    cv2.imread = guarded_read
    scene, state, records, before, cameras_hash = None, None, [], None, None
    report = {'status': 'running', 'views': records, 'renders': 0, 'backwards': 0,
              'optimizer_steps': 0, 'rgb_and_semantic_decodes': 0, 'checkpoint_written': False}
    output = Path(plan['output'])/'audit.json'
    try:
        scene, state = modules['train'].load_scene(plan['checkpoint'])
        scene.eval()
        cameras_hash = tensor_hash(state['training_cameras'])
        before = {k: tensor_hash(v) for k, v in scene.state_dict().items()}
        with preserved_scene(scene) as original:
            for name, p in scene.named_parameters():
                p.requires_grad_(name == 'splats.means')
            means = scene.splats['means']
            ids = torch.tensor(plan['geometry']['indices'], device=means.device)
            actual_d = torch.tensor(plan['geometry']['actual_fp32_displacement'], device=means.device, dtype=torch.float64)
            changed = original['splats.means'].clone()
            changed[ids] = (original['splats.means'][ids].double()+actual_d).to(means.dtype)
            repair = torch.zeros_like(means, dtype=torch.float64)
            repair[ids] = original['splats.means'][ids].double()-changed[ids].double()
            require(torch.equal(repair[ids], -actual_d), 'Realized displacement differs from CPU-locked bytes')
            for vi, view in enumerate(plan['views']):
                k, pose, weights, width, height = load_camera_only(view)
                start_renders, start_backwards = report['renders'], report['backwards']
                def render(camera, k=k, width=width, height=height):
                    require(report['renders'] < SPEC['total_renders'], 'Render budget exhausted')
                    result = scene.render(k, camera, width, height, degree=3, semantics=False, absgrad=False)['rgb']
                    report['renders'] += 1
                    return result
                with torch.no_grad():
                    means.copy_(original['splats.means'])
                    target = render(pose).detach()
                null = gradient_repeats(render, pose, target, weights, means)
                report['backwards'] += 2
                permutation = fixed_permutation(int((weights > 0).sum())*3, vi)
                row = {'name': view['name'], 'permutation_sha256': hashlib.sha256(permutation.tobytes()).hexdigest()}
                for condition in ('pose', 'geometry'):
                    with torch.no_grad():
                        means.copy_(original['splats.means'] if condition == 'pose' else changed)
                    camera = (modules['reliability'].se3_exp(pose.new_tensor(view['half_twist']))@pose if condition == 'pose' else pose)
                    _, jacobian = modules['reliability'].finite_difference_camera_jacobian(render, camera,
                        1e-5*plan['scene_scale'], 1e-4)
                    system = whitened_system(jacobian, weights, plan['scene_scale'], permutation)
                    repeated = gradient_repeats(render, camera, target, weights, means, system)
                    report['backwards'] += 6
                    row[condition] = condition_summary(repeated, null, target, weights, repair)
                    p = system['precision'].diagonal().sqrt()
                    spectrum = torch.linalg.eigvalsh(system['normal']/p[:, None]/p[None, :])
                    row[condition]['information_spectrum'] = spectrum.tolist()
                    row[condition]['precision_diagonal'] = system['precision'].diagonal().tolist()
                    row[condition]['normal_permutation_relative_error'] = norm(system['permuted_j'].T@system['permuted_j']-system['normal'])/max(norm(system['normal']), 1e-30)
                    del repeated, jacobian, system
                with torch.no_grad():
                    means.copy_(original['splats.means'])
                require(report['renders']-start_renders == 33 and report['backwards']-start_backwards == 14, 'Per-view budget changed')
                records.append(row)
                write_json(output, report, replace=output.exists())
                print(json.dumps({'completed_view': view['name'], 'render_count': report['renders']}), flush=True)
            require(all(p.grad is None for p in scene.parameters()), 'autograd.grad must not accumulate parameter gradients')
        require(before == {k: tensor_hash(v) for k, v in scene.state_dict().items()}, 'Final scene bytes changed')
        require(cameras_hash == tensor_hash(state['training_cameras']), 'Saved cameras changed')
        require(report['renders'] == 264 and report['backwards'] == 112 and len(read_paths) == 8, 'Incomplete budget/read contract')
        verify(plan)
        report.update(status='completed', summary=summarize(records), tensor_restoration_exact=True,
                      saved_cameras_exact=True, all_bound_inputs_sources_unchanged=True,
                      validity_reads=read_paths,
                      actual_imports={name: {'path': module.__file__, 'sha256': digest(module.__file__)} for name, module in modules.items()})
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        cv2.imread = original_read
        if scene is not None and before is not None:
            restored = before == {k: tensor_hash(v) for k, v in scene.state_dict().items()}
            report['tensor_restoration_exact'] = restored
            report['saved_cameras_exact'] = cameras_hash == tensor_hash(state['training_cameras'])
            if not restored or not report['saved_cameras_exact']:
                report.update(status='failed', restoration_error='Final model or saved camera bytes differ')
        write_json(output, report, replace=output.exists())


def gpu_idle():
    result = subprocess.run(['nvidia-smi', '-q', '-x'], check=True, capture_output=True, text=True, timeout=5)
    document = ET.fromstring(result.stdout)
    require(document.findall('gpu'), 'No CUDA GPU')
    for gpu in document.findall('gpu'):
        processes = gpu.find('processes')
        require(processes is not None and (processes.text or '').strip() not in {'N/A', 'Not Supported'},
                'Missing or unavailable GPU client information')
    rows = [{k: row.findtext(k) for k in ('pid', 'type', 'process_name')} for row in document.findall('.//process_info')]
    require(all(row['type'] == 'G' for row in rows), 'Another compute or unknown GPU client is present')
    return rows


def execute(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Use immutable runner')
    output = Path(plan['output'])
    require(not (output/'execution_receipt.json').exists() and not (output/'audit.json').exists(), 'No retry or overwrite')
    verify(plan)
    graphics = gpu_idle()
    receipt = {'status': 'running', 'plan_sha256': digest(plan_path), 'graphics_clients': graphics,
               'started_utc': datetime.now(UTC).isoformat(), 'wall_seconds': SPEC['worker_wall_seconds'],
               'authorization': 'operator supplies explicit handoff flag; plan remains CPU registration'}
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(plan_path)]
    receipt['command'] = command
    write_json(output/'execution_receipt.json', receipt)
    start = time.monotonic()
    try:
        with (output/'worker.log').open('x') as log:
            result = subprocess.run(command, cwd=plan['root'], env={**os.environ, 'PYTHONPATH': plan['source_snapshot']},
                                    stdout=log, stderr=subprocess.STDOUT, timeout=SPEC['worker_wall_seconds'], check=False)
        require(result.returncode == 0, f'Diagnostic child exit {result.returncode}')
        report = json.loads((output/'audit.json').read_text())
        require(report['status'] == 'completed' and report['tensor_restoration_exact'], 'Incomplete diagnostic')
        verify(plan)
        require(digest(plan_path) == receipt['plan_sha256'], 'Plan changed')
        receipt.update(status='completed', report_sha256=digest(output/'audit.json'), summary=report['summary'])
    except subprocess.TimeoutExpired:
        receipt.update(status='inconclusive_timeout', restoration_contract='Child killed; in-memory finally not guaranteed; immutable inputs rechecked; no checkpoint writes')
        verify(plan)
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-start, finished_utc=datetime.now(UTC).isoformat())
        write_json(output/'execution_receipt.json', receipt, replace=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--run', type=Path)
    modes.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--confirmed-gpu-handoff', action='store_true')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('/mnt/data/SHM2026/runs/pose_profile_gradient_v1'))
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.root, args.output))
    elif args.worker:
        worker(args.worker)
    else:
        require(args.confirmed_gpu_handoff, 'GPU execution requires later explicit root handoff')
        result = execute(args.run)
        print(json.dumps(result))
        if result['status'] != 'completed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
