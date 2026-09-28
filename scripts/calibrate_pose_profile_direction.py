"""Independent fixed-interval numerical calibration; never rewrites the old probe.

CPU preparation only by default. Five original TRAIN cameras, the same 22 means,
three predeclared steps toward repair, no optimizer or real RGB/semantic reads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ORIGINAL = Path('/mnt/data/SHM2026/runs/pose_profile_gradient_v1')
ORIGINAL_HASHES = {
    'plan.json': '0b5872b2b89fa1e9ac49ac37d919c4ca8d2bdbbd806dfc2441b492eee8006048',
    'audit.json': '7b4184a2446e7ab718f884ee905b1c75ffebd68155dbd1d0ab289d91a9237f9a',
    'execution_receipt.json': 'd39276f426cb3ccc743a30f48357381e99c46935d141353ec8eac315d77008be',
}
REFERENCE_SHA = 'e59ff68a141f567fac921837a5b3266b5cbb34c2ec736e707b12f7548e2e1468'
reference_path = Path(__file__).with_name('audit_pose_profile_gradients.py')
if hashlib.sha256(reference_path.read_bytes()).hexdigest() != REFERENCE_SHA:
    raise ValueError('Original algebra/loading/restoration helper changed')
module_spec = importlib.util.spec_from_file_location('fixed_profile_reference', reference_path)
ref = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(ref)

SPEC = {
    'protocol': 'pose_profile_fixed_interval_numerical_calibration_v1',
    'views': list(ref.GEOMETRY_NAMES), 'fractions': [1/8, 1/16, 1/32],
    'judged_fractions': [1/16, 1/32], 'repeats': 2,
    'geometry': 'Original 22 IDs, original realized FP32 displacement; repair interval only',
    'jacobian': 'At theta_g,T once per view; original eps, Lambda, valid weights and permutation',
    'profile': 'Re-solve delta at every residual with frozen J/Lambda; original envelope backward',
    'chain': 'A=g_means dot actual_FP32_step; B=g_whitened_r dot delta_whitened_r; D=actual_loss_difference; D=B+L(delta_r)',
    'arithmetic': 'Original FP32 renderer/residual subtraction, FP64 losses and inner products',
    'noise_multiplier': 10., 'relative_chain_tolerance': .10,
    'quadratic_identity_relative_tolerance': 1e-9,
    'geometry_required_reliable_views': 5, 'geometry_required_retained_views': 4,
    'geometry_retention_min': .5,
    'absolute_full_image_rms_gate': False,
    'budget': '5*(1 target+2 null+13 camera-FD+2 base+3*2 step renders); 5*(2 null+2*3 means backwards)',
    'total_renders': 120, 'total_backwards': 40, 'optimizer_steps': 0,
    'worker_wall_seconds': 180,
    'interpretation': 'Independent directional calibration only; old inconclusive remains unchanged',
}


def interval_steps(original, changed):
    """Round each endpoint once to FP32, then use the actual displacement in dots."""
    ref.require(original.dtype == changed.dtype == torch.float32, 'Require original FP32 means')
    repair = original.double()-changed.double()
    rows = []
    for fraction in SPEC['fractions']:
        endpoint = (changed.double()+fraction*repair).float()
        actual = endpoint.double()-changed.double()
        ref.require(bool(((endpoint >= torch.minimum(original, changed)) &
                          (endpoint <= torch.maximum(original, changed))).all()), 'Step leaves repair interval')
        ref.require(bool(actual.norm() > 0), 'FP32 step rounded entirely to zero')
        rows.append({'fraction': fraction, 'endpoint': endpoint, 'actual': actual,
                     'relative_quantization_error': float((actual-fraction*repair).norm()/(fraction*repair).norm())})
    return rows


def whiten(value, system):
    return value.detach().reshape(-1)[system['index']].double()*system['sqrt_w']


def image_direction_terms(base_residual, step_residual, system, arm):
    """Independent closed-form image derivative and positive quadratic remainder."""
    r0, dr = whiten(base_residual, system), whiten(step_residual, system)-whiten(base_residual, system)
    if arm == 'raw':
        image_gradient = r0/system['sum_w']
    else:
        j = system['j' if arm == 'profile' else 'permuted_j']
        delta = -torch.cholesky_solve((j.T@r0)[:, None], system['cholesky'])[:, 0]
        image_gradient = (r0+j@delta)/system['sum_w']
    terms = image_gradient*dr
    # Re-solving delta for dr gives the exact remainder of the fixed quadratic.
    remainder = ref.profile_losses(step_residual.double()-base_residual.double(), system)[arm]
    return float(terms.sum()), float(terms.abs().sum()), float(remainder)


def summarize_step(base, endpoints, system, actual_step, null_losses):
    """Compare the three chain links, not repeatability alone or profile energy."""
    rows = {}
    for arm in ref.ARMS:
        samples = []
        for b, e in zip(base, endpoints, strict=True):
            products = b['gradient'][arm].double()*actual_step
            a = float(products.sum())
            b_term, b_abs, q = image_direction_terms(b['residual'], e['residual'], system, arm)
            samples.append({'parameter_linear_change': a, 'image_linear_change': b_term,
                            'quadratic_remainder': q, 'loss_change': e['loss'][arm]-b['loss'][arm],
                            'parameter_sum_abs_products': float(products.abs().sum()),
                            'image_sum_abs_products': b_abs,
                            'base_loss': b['loss'][arm], 'step_loss': e['loss'][arm]})
        mean = {key: sum(s[key] for s in samples)/2 for key in samples[0]}
        repeat_loss = max(abs(samples[0]['base_loss']-samples[1]['base_loss']),
                          abs(samples[0]['step_loss']-samples[1]['step_loss']),
                          abs(samples[0]['loss_change']-samples[1]['loss_change']), *null_losses)
        eps64 = 32*torch.finfo(torch.float64).eps
        loss_floor = max(repeat_loss, eps64*(abs(mean['base_loss'])+abs(mean['step_loss'])))
        a_floor = max(abs(samples[0]['parameter_linear_change']-samples[1]['parameter_linear_change']),
                      32*torch.finfo(torch.float32).eps*mean['parameter_sum_abs_products'])
        b_floor = max(abs(samples[0]['image_linear_change']-samples[1]['image_linear_change']),
                      eps64*mean['image_sum_abs_products'])
        floor = max(loss_floor, a_floor, b_floor)
        a, b, observed = (mean[k] for k in ('parameter_linear_change', 'image_linear_change', 'loss_change'))
        signal = all(abs(value) > SPEC['noise_multiplier']*floor for value in (a, b, observed))
        relative = lambda left, right: abs(left-right)/max(abs(left), abs(right), 1e-300)
        chain_error = relative(a, b)
        fd_error = relative(a, observed)
        identity_error = abs(observed-b-mean['quadratic_remainder'])
        identity_scale = abs(observed)+abs(b)+mean['quadratic_remainder']
        identity_ok = identity_error <= max(eps64*identity_scale, SPEC['quadratic_identity_relative_tolerance']*identity_scale)
        agreement = (chain_error <= SPEC['relative_chain_tolerance'] and
                     fd_error <= SPEC['relative_chain_tolerance'] and identity_ok)
        rows[arm] = {**mean, 'repeats': samples, 'loss_floor': loss_floor,
                     'parameter_floor': a_floor, 'image_floor': b_floor, 'combined_floor': floor,
                     'measurable_change': signal, 'parameter_vs_image_relative_error': chain_error,
                     'parameter_vs_actual_loss_relative_error': fd_error,
                     'quadratic_identity_absolute_error': identity_error,
                     'quadratic_identity_ok': identity_ok, 'chain_agreement': agreement,
                     'direction_reliable': signal and agreement}
    return rows


def summarize_view(step_rows):
    ref.require([s['fraction'] for s in step_rows] == SPEC['fractions'], 'All fixed steps required')
    judged = [r for r in step_rows if r['fraction'] in SPEC['judged_fractions']]
    measurable = all(row['arms'][arm]['measurable_change'] for row in judged for arm in ref.ARMS)
    reliable = all(row['arms'][arm]['direction_reliable'] for row in judged for arm in ref.ARMS)
    raw_descent = all(row['arms']['raw']['loss_change'] < 0 and
                      row['arms']['raw']['parameter_linear_change'] < 0 for row in judged)
    retained = reliable and raw_descent and all(
        row['arms']['profile']['loss_change'] < 0 and
        -row['arms']['profile']['loss_change'] >= SPEC['geometry_retention_min']*(-row['arms']['raw']['loss_change']) and
        -row['arms']['profile']['parameter_linear_change'] >= SPEC['geometry_retention_min']*(-row['arms']['raw']['parameter_linear_change'])
        for row in judged)
    return {'changes_measurable': measurable, 'direction_reliable': reliable and raw_descent,
            'profile_repair_retained': retained,
            'largest_step': 'reported only, predeclared; not substituted for either judged step'}


def summarize(records):
    ref.require([r['name'] for r in records] == SPEC['views'], 'All original five views required')
    measurable = sum(r['summary']['changes_measurable'] for r in records)
    reliable = sum(r['summary']['direction_reliable'] for r in records)
    retained = sum(r['summary']['profile_repair_retained'] for r in records)
    status = ('inconclusive_numerical_signal' if measurable != 5 else
              'direction_calibration_not_met' if reliable != 5 or retained < 4 else
              'independent_direction_calibration_supported')
    return {'status': status, 'measurable_views': measurable, 'reliable_views': reliable,
            'retained_views': retained, 'old_probe_result': 'inconclusive_measurability',
            'training_authorized': False,
            'limit': 'Finite one-sided secants on one fixed region, not exact derivatives or global scene performance'}


def verify(plan):
    ref.require(plan['specification'] == SPEC, 'Calibration specification changed')
    snapshot = Path(plan['source_snapshot'])
    actual = {str(p.relative_to(snapshot)): ref.digest(p) for p in sorted(snapshot.rglob('*.py'))}
    ref.require(actual == plan['source_hashes'], 'Frozen source changed')
    for path, sha in plan['input_hashes'].items():
        ref.require(ref.digest(path) == sha, f'Bound input changed: {path}')


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    ref.require(not output.exists(), 'No overwrite or retry of previous calibration')
    for name, sha in ORIGINAL_HASHES.items():
        ref.require(ref.digest(ORIGINAL/name) == sha, 'Original probe artifact changed')
    old = json.loads((ORIGINAL/'plan.json').read_text())
    receipt = json.loads((ORIGINAL/'execution_receipt.json').read_text())
    ref.require(receipt['status'] == 'completed' and receipt['plan_sha256'] == ORIGINAL_HASHES['plan.json']
                and receipt['report_sha256'] == ORIGINAL_HASHES['audit.json'], 'Original run not bound')
    ref.verify(old)
    state = torch.load(old['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    original = state['model']['splats.means'][old['geometry']['indices']].clone()
    d = torch.tensor(old['geometry']['actual_fp32_displacement'], dtype=torch.float64)
    changed = (original.double()+d).float()
    ref.require(torch.equal(changed.double()-original.double(), d), 'Original displacement changed')
    steps = interval_steps(original, changed)
    step_records = [{'fraction': r['fraction'], 'actual_fp32_step': r['actual'].tolist(),
                     'relative_quantization_error': r['relative_quantization_error']} for r in steps]
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(old['source_snapshot'], snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    shutil.copy2(root/'tests/test_pose_profile_calibration.py', snapshot/'test_pose_profile_calibration.py')
    sources = {str(p.relative_to(snapshot)): ref.digest(p) for p in sorted(snapshot.rglob('*.py'))}
    ref.require(all(sources[k] == sha for k, sha in old['source_hashes'].items()), 'Original source copy changed')
    inputs = {**old['input_hashes'], **{str(ORIGINAL/name): sha for name, sha in ORIGINAL_HASHES.items()}}
    plan = {'status': 'cpu_locked_pending_explicit_gpu_handoff', 'created_utc': datetime.now(UTC).isoformat(),
            'specification': SPEC, 'root': str(root), 'output': str(output), 'source_snapshot': str(snapshot),
            'source_hashes': sources, 'input_hashes': inputs, 'checkpoint': old['checkpoint'],
            'checkpoint_sha256': old['checkpoint_sha256'], 'scene_scale': old['scene_scale'],
            'views': [v for v in old['views'] if v['name'] in SPEC['views']],
            'original_view_indices': {v['name']: i for i, v in enumerate(old['views'])},
            'geometry': old['geometry'], 'steps': step_records,
            'gpu_authorized': False, 'cpu_cuda_initialized': torch.cuda.is_initialized(),
            'timing_basis': 'Original 264 renders/112 backwards worker including loading/checks completed in 5.447s; new 120/40, cap180s, no retry',
            'pixel_decodes_during_preparation': 0}
    ref.require(not plan['cpu_cuda_initialized'], 'CPU preparation initialized CUDA')
    ref.write_json(output/'plan.json', plan)
    verify(plan)
    return output/'plan.json'


def worker(plan_path):
    import cv2
    plan = json.loads(Path(plan_path).read_text())
    ref.require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Use frozen runner')
    verify(plan)
    modules = {name: importlib.import_module('bridge_rgs.'+name) for name in ('train', 'model', 'reliability', 'refinement')}
    for name, module in modules.items():
        ref.require(Path(module.__file__).resolve() == Path(plan['source_snapshot'])/'bridge_rgs'/f'{name}.py', 'Wrong actual import')
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    torch.manual_seed(42)
    allowed = {str(Path(v['valid_path']).resolve()) for v in plan['views']}
    reads, original_read = [], cv2.imread
    def guarded_read(path, *args, **kwargs):
        path = str(Path(path).resolve())
        ref.require(path in allowed, 'Only original TRAIN validity masks may be decoded')
        reads.append(path)
        return original_read(path, *args, **kwargs)
    cv2.imread = guarded_read
    output = Path(plan['output'])/'audit.json'
    report = {'status': 'running', 'views': [], 'renders': 0, 'backwards': 0,
              'optimizer_steps': 0, 'rgb_and_semantic_decodes': 0, 'checkpoint_written': False}
    scene, state, before, camera_hash = None, None, None, None
    try:
        scene, state = modules['train'].load_scene(plan['checkpoint'])
        scene.eval()
        before = {k: ref.tensor_hash(v) for k, v in scene.state_dict().items()}
        camera_hash = ref.tensor_hash(state['training_cameras'])
        with ref.preserved_scene(scene) as original_state:
            for name, p in scene.named_parameters():
                p.requires_grad_(name == 'splats.means')
            means = scene.splats['means']
            ids = torch.tensor(plan['geometry']['indices'], device=means.device)
            original = original_state['splats.means']
            changed = original.clone()
            d = torch.tensor(plan['geometry']['actual_fp32_displacement'], device=means.device, dtype=torch.float64)
            changed[ids] = (original[ids].double()+d).float()
            ref.require(torch.equal(changed[ids].double()-original[ids].double(), d), 'Original FP32 perturbation differs')
            steps = interval_steps(original[ids], changed[ids])
            for row, locked in zip(steps, plan['steps'], strict=True):
                ref.require(row['fraction'] == locked['fraction'] and torch.equal(row['actual'].cpu(),
                            torch.tensor(locked['actual_fp32_step'], dtype=torch.float64)), 'Locked FP32 steps differ')
            for view in plan['views']:
                k, pose, weights, width, height = ref.load_camera_only(view)
                start_renders, start_backwards = report['renders'], report['backwards']
                def render(camera, k=k, width=width, height=height):
                    ref.require(report['renders'] < SPEC['total_renders'], 'Render budget exceeded')
                    result = scene.render(k, camera, width, height, degree=3, semantics=False, absgrad=False)['rgb']
                    report['renders'] += 1
                    return result
                with torch.no_grad():
                    means.copy_(original)
                    target = render(pose).detach()
                null = ref.gradient_repeats(render, pose, target, weights, means)
                report['backwards'] += 2
                with torch.no_grad():
                    means.copy_(changed)
                _, jacobian = modules['reliability'].finite_difference_camera_jacobian(render, pose, 1e-5*plan['scene_scale'], 1e-4)
                permutation = ref.fixed_permutation(int((weights > 0).sum())*3, plan['original_view_indices'][view['name']])
                system = ref.whitened_system(jacobian, weights, plan['scene_scale'], permutation)
                base = ref.gradient_repeats(render, pose, target, weights, means, system)
                report['backwards'] += 6
                step_rows = []
                for step in steps:
                    with torch.no_grad():
                        means.copy_(changed)
                        means[ids] = step['endpoint']
                        endpoints = []
                        for _ in range(2):
                            residual = render(pose)-target
                            losses = ref.profile_losses(residual, system)
                            endpoints.append({'residual': residual.detach(), 'loss': {a: float(v) for a, v in losses.items()}})
                    actual = torch.zeros_like(means, dtype=torch.float64)
                    actual[ids] = step['actual']
                    arms = summarize_step(base, endpoints, system, actual, [n['loss']['raw'] for n in null])
                    step_rows.append({'fraction': step['fraction'], 'actual_step_l2': ref.norm(actual),
                                      'relative_quantization_error': step['relative_quantization_error'], 'arms': arms})
                rw = whiten(base[0]['residual'], system)
                energy = rw.square()
                effective = float(energy.sum().square()/energy.square().sum()) if bool(energy.sum() > 0) else 0.
                row = {'name': view['name'], 'steps': step_rows, 'summary': summarize_view(step_rows),
                       'permutation_sha256': hashlib.sha256(permutation.tobytes()).hexdigest(),
                       'precision_diagonal': system['precision'].diagonal().tolist(),
                       'original_global_rms_descriptive_only': ref.weighted_rms(base[0]['residual'], weights),
                       'base_repeat_rms': ref.weighted_rms(base[0]['residual'].double()-base[1]['residual'].double(), weights),
                       'null_rms': [ref.weighted_rms(n['residual'], weights) for n in null],
                       'residual_energy_effective_scalar_count': effective,
                       'valid_scalar_count': len(system['index']),
                       'residual_energy_effective_fraction': effective/len(system['index']),
                       'energy_concentration_limit': 'Inverse participation of residual energy, not a visibility/occlusion label'}
                with torch.no_grad():
                    means.copy_(original)
                ref.require(report['renders']-start_renders == 24 and report['backwards']-start_backwards == 8, 'Per-view budget changed')
                report['views'].append(row)
                ref.write_json(output, report, replace=output.exists())
                print(json.dumps({'completed_view': view['name'], 'renders': report['renders']}), flush=True)
            ref.require(all(p.grad is None for p in scene.parameters()), 'Parameter grads accumulated')
        ref.require(report['renders'] == 120 and report['backwards'] == 40 and len(reads) == 5, 'Incomplete budget/read contract')
        verify(plan)
        report.update(status='completed', summary=summarize(report['views']), validity_reads=reads,
                      all_bound_inputs_sources_unchanged=True,
                      actual_imports={name: {'path': m.__file__, 'sha256': ref.digest(m.__file__)} for name, m in modules.items()})
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        cv2.imread = original_read
        if before is not None:
            report['tensor_restoration_exact'] = before == {k: ref.tensor_hash(v) for k, v in scene.state_dict().items()}
            report['saved_cameras_exact'] = camera_hash == ref.tensor_hash(state['training_cameras'])
            if not report['tensor_restoration_exact'] or not report['saved_cameras_exact']:
                report.update(status='failed', restoration_error='Model/camera bytes differ')
        ref.write_json(output, report, replace=output.exists())


def execute(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    ref.require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Use immutable runner')
    output = Path(plan['output'])
    ref.require(not (output/'execution_receipt.json').exists() and not (output/'audit.json').exists(), 'No retries/overwrite')
    verify(plan)
    graphics = ref.gpu_idle()
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(plan_path)]
    receipt = {'status': 'running', 'plan_sha256': ref.digest(plan_path), 'graphics_clients': graphics,
               'started_utc': datetime.now(UTC).isoformat(), 'command': command,
               'wall_seconds': SPEC['worker_wall_seconds'], 'old_probe_result_unchanged': True}
    ref.write_json(output/'execution_receipt.json', receipt)
    start = time.monotonic()
    try:
        with (output/'worker.log').open('x') as log:
            result = subprocess.run(command, cwd=plan['root'], env={**os.environ, 'PYTHONPATH': plan['source_snapshot']},
                                    stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False)
        ref.require(result.returncode == 0, f'Worker exited {result.returncode}')
        report = json.loads((output/'audit.json').read_text())
        ref.require(report['status'] == 'completed' and report['tensor_restoration_exact'] and report['saved_cameras_exact'], 'Incomplete/restoration failed')
        receipt.update(status='completed', report_sha256=ref.digest(output/'audit.json'), summary=report['summary'])
    except subprocess.TimeoutExpired:
        receipt.update(status='inconclusive_timeout', restoration_contract='Hard kill cannot guarantee in-memory finally; no checkpoints written')
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            verify(plan)
            ref.require(ref.digest(plan_path) == receipt['plan_sha256'], 'Plan changed')
            receipt['bound_inputs_and_sources_unchanged'] = True
        except (OSError, ValueError, KeyError) as error:
            receipt.update(status='failed', source_error=f'{type(error).__name__}: {error}')
        receipt.update(elapsed_seconds=time.monotonic()-start, finished_utc=datetime.now(UTC).isoformat())
        ref.write_json(output/'execution_receipt.json', receipt, replace=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', type=Path)
    mode.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('/mnt/data/SHM2026/runs/pose_profile_numerical_calibration_v1'))
    parser.add_argument('--confirmed-gpu-handoff', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.root, args.output))
    elif args.worker:
        worker(args.worker)
    else:
        ref.require(args.confirmed_gpu_handoff, 'Await explicit root GPU handoff')
        result = execute(args.run)
        print(json.dumps(result))
        if result['status'] != 'completed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
