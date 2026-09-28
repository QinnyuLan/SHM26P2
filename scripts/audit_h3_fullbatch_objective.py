"""Fixed H3 two-TRAIN-view, forty-render full-original RGB objective diagnostic.

--prepare is CPU-only. --run requires a separately authorized frozen snapshot.
No candidate is committed, no production checkpoint is written, no VAL is read.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

BASE_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
LOSS_SHA = '1e114069c2fdb181e243d7cb8a986659b2e229d2ec49c34821cb45013d9cf2b2'
SOLVER_SHA = '10c00b1728e7d1e3773219f18b961065996d8c3a31824700333d8987bfefa746'
KEYS = ('splats.sh0', 'splats.sh_rest', 'background_logits')
EPSILONS = (1e-3, 5e-4, 2.5e-4)
SPEC = {
    'protocol': 'h3_full_original_objective_rms_armijo_precheck_v1',
    'train_sorted_indices': [0, 175], 'view_names': ['002.png', '205.png'],
    'pixel_protocol': 'legacy_mixed_v1', 'sh_degree': 3,
    'parameters': list(KEYS), 'epsilons': list(EPSILONS),
    'direction': 'per-group FP32 analytic gradient divided by absmax',
    'objective': 'FP32 .8 full-original RGB L1 + .2 complete 7-window (1-SSIM)',
    'prediction': 'legacy overscan RGB -> clamp[0,1] -> fixed float-map gather',
    'renders_per_view': 20, 'render_calls': 40, 'backward_calls': 2,
    'solver': {'beta1': 0., 'beta2': .999, 'eps': 1e-8,
               'rates': dict(zip(KEYS, [2.5e-4, 1.25e-5, 1e-4], strict=True)),
               'alpha': 1., 'fresh_state_per_view': True, 'committed_steps': 0},
    'fd_max_relative_error': .05, 'warp_max_abs_error': 2e-6,
    'step_actual_predicted_ratio_bounds': [.9, 1.1],
    'step_rule': 'g_dot_actual_delta<0 and helper Armijo including decrease floor',
    'deadline_seconds': 120, 'pixel_reads': 'two original TRAIN RGB only; no masks/valid/VAL',
    'failure_policy': 'report all fixed comparisons; no changed epsilon/view/threshold or retry; no CUDA-bug inference',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def write(path, value, replace=False):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    if replace:
        tmp = path.with_suffix('.tmp')
        tmp.write_text(payload)
        tmp.replace(path)
    else:
        with path.open('x') as stream:
            stream.write(payload)


def fixed_views(manifest):
    require(manifest.get('pixel_protocol', 'legacy_mixed_v1') == 'legacy_mixed_v1', 'Require legacy manifest')
    training = [v for v in manifest['views'] if v['split'] == 'train']
    require(len(training) == len({v['name'] for v in training}) == 350, 'Require unique 350 TRAIN cameras')
    ordered = sorted(training, key=lambda v: v['name'])
    chosen = [ordered[i] for i in SPEC['train_sorted_indices']]
    require([v['name'] for v in chosen] == SPEC['view_names'], 'Fixed TRAIN names changed')
    names = [v['name'] for v in training]
    keys = ('name', 'split', 'camera_id', 'width', 'height', 'K', 'w2c', 'w2c_original', 'source_image_path')
    return [{**{k: v[k] for k in keys}, 'camera_index': names.index(v['name'])} for v in chosen], names


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Fixed H3 objective diagnostic deadline')


def finite_difference_record(base, plus, minus, epsilon, gradient, original, positive, negative, direction):
    # CUDA scalar tensors expose __array__ but cannot be read by NumPy directly.
    base, plus, minus = float(base), float(plus), float(minus)
    central = (float(plus)-float(minus))/(2*epsilon)
    dp = (positive.double()-original.double())/epsilon
    dm = (original.double()-negative.double())/epsilon
    realized = (dp+dm)/2
    analytic = float((gradient.double()*realized).sum())
    nominal = float((gradient.double()*direction.double()).sum())
    error = abs(central-analytic)/max(abs(analytic), 1e-30)
    ulps = float(abs(np.spacing(np.float32(plus)))+abs(np.spacing(np.float32(minus))))
    return {'epsilon': epsilon, 'base_fp32_loss': float(base), 'plus_fp32_loss': float(plus),
            'minus_fp32_loss': float(minus), 'central_difference': central,
            'forward_difference': (float(plus)-float(base))/epsilon,
            'backward_difference': (float(base)-float(minus))/epsilon,
            'nominal_analytic': nominal, 'realized_analytic': analytic,
            'realized_plus_analytic': float((gradient.double()*dp).sum()),
            'realized_minus_analytic': float((gradient.double()*dm).sum()),
            'realized_relative_discrepancy': error,
            'realized_direction_relative_l2': float((realized-direction.double()).norm()/direction.double().norm().clamp_min(1e-30)),
            'plus_delta_absmax': float((positive-original).abs().max()),
            'minus_delta_absmax': float((negative-original).abs().max()),
            'one_ulp_derivative_resolution': ulps/(2*epsilon),
            'signal_within_16_scalar_ulps': bool(abs(float(plus)-float(minus)) <= 16*ulps),
            'passed': bool(np.isfinite(error) and abs(analytic) > 0 and float(plus) != float(minus)
                           and error <= SPEC['fd_max_relative_error'])}


def probe_group(parameter, gradient, base_loss, objective, deadline, image_gradient=None):
    original = parameter.detach().clone()
    require(bool(torch.isfinite(gradient).all()) and bool(gradient.abs().max() > 0), 'Finite nonzero group gradient required')
    direction = gradient.detach()/gradient.abs().max()
    rows = []
    try:
        for epsilon in EPSILONS:
            check_time(deadline)
            with torch.no_grad():
                positive, negative = original+epsilon*direction, original-epsilon*direction
                parameter.copy_(positive)
                plus, plus_prediction = objective()
                check_time(deadline)
                parameter.copy_(negative)
                minus, minus_prediction = objective()
                parameter.copy_(original)
                row = finite_difference_record(base_loss, plus, minus, epsilon, gradient,
                                               original, positive, negative, direction)
                if image_gradient is not None:
                    jd = (plus_prediction.double()-minus_prediction.double())/(2*epsilon)
                    row['image_gradient_dot_prediction_secant'] = float((image_gradient.double()*jd).sum())
                    row['prediction_secant_rms'] = float(jd.square().mean().sqrt())
                rows.append(row)
    finally:
        with torch.no_grad():
            parameter.copy_(original)
    require(torch.equal(parameter, original), 'FD parameter restoration failed')
    return {'gradient_absmax': float(gradient.abs().max()), 'gradient_rms': float(gradient.double().square().mean().sqrt()),
            'epsilons': rows, 'parameter_restored_exact': True}


def transient_rms(parameters, gradients, base_loss, objective, deadline):
    from bridge_rgs import fullbatch_appearance as solver
    originals = {k: p.detach().clone() for k, p in parameters.items()}
    optimizer = solver.RMSArmijo(parameters)
    proposal = optimizer.propose(gradients)
    check_time(deadline)
    with optimizer.candidate(proposal, gradients, 1.) as trial:
        with torch.no_grad():
            after, _ = objective()
        actual = float(base_loss)-float(after)
        predicted = -trial.actual_slope
        ratio = actual/predicted if predicted > 0 else None
        passed = bool(solver.armijo(float(base_loss), float(after), trial.actual_slope)
                      and ratio is not None and .9 <= ratio <= 1.1)
        result = {'before_fp32_loss': float(base_loss), 'after_fp32_loss': float(after),
                  'nominal_slope': proposal.analytic_slope, 'actual_slope': trial.actual_slope,
                  'actual_decrease': actual, 'predicted_decrease': predicted,
                  'actual_to_predicted_decrease_ratio': ratio,
                  'decrease_floor': solver.decrease_floor(float(base_loss)),
                  'helper_armijo_passed': bool(solver.armijo(float(base_loss), float(after), trial.actual_slope)),
                  'displacement_absmax': trial.displacement_absmax, 'passed': passed}
        # Deliberately never call trial.accept: even a passing candidate rolls back.
    require(optimizer.accepted_steps == 0 and all(not bool(v.any()) for v in optimizer.second_moment.values()),
            'Diagnostic must not commit RMS state')
    require(all(torch.equal(p, originals[k]) for k, p in parameters.items()), 'RMS candidate not rolled back')
    result.update(parameters_restored_exact=True, committed_steps=0)
    return result


def technical_gate(rows):
    require(len(rows) == 2, 'Need both fixed views')
    clauses = {'all_18_fd_pass': all(e['passed'] for r in rows for g in r['finite_differences'].values() for e in g['epsilons']),
               'both_rms_steps_pass': all(r['temporary_rms_step']['passed'] for r in rows),
               'both_official_warps_pass': all(r['baseline']['warp_cv2_max_abs_difference'] <= SPEC['warp_max_abs_error'] for r in rows)}
    require(all(set(r['finite_differences']) == set(KEYS) and
                all(len(g['epsilons']) == 3 for g in r['finite_differences'].values()) for r in rows), 'Incomplete FD matrix')
    return {'passed': all(clauses.values()), 'clauses': clauses,
            'scope': 'Two fixed TRAIN per-view local contracts, not a 350-view descent or accuracy guarantee'}


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), 'Preserve any previous diagnostic')
    base, manifest_path = root/'runs/h3_moments/02_cross/last.pt', root/'artifacts/prepared/manifest.json'
    require(digest(base) == BASE_SHA, 'Wrong selected H3 checkpoint')
    require(digest(root/'src/bridge_rgs/losses.py') == LOSS_SHA, 'Require approved contiguous SSIM fix')
    require(digest(root/'src/bridge_rgs/fullbatch_appearance.py') == SOLVER_SHA, 'RMS helper changed')
    manifest = json.loads(manifest_path.read_text())
    views, names = fixed_views(manifest)
    document = root/'docs/fullbatch_appearance_h3_protocol.md'
    paths = [base, manifest_path, root/'uv.lock', root/'pyproject.toml', document]
    allowed = [str((root/v['source_image_path']).resolve()) for v in views]
    paths += [Path(p) for p in allowed]
    inputs = {str(p): digest(p) for p in paths}
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(root/'src/bridge_rgs', snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    shutil.copy2(document, snapshot/'fullbatch_appearance_h3_protocol.md')
    sources = {str(p.relative_to(snapshot)): digest(p) for p in sorted(p for p in snapshot.rglob('*') if p.is_file() and p.suffix in {'.py', '.md'})}
    plan = {'status': 'locked_pending_root_gpu_authorization', 'specification': SPEC,
            'root': str(root), 'output': str(output), 'source_snapshot': str(snapshot), 'source_hashes': sources,
            'base': str(base), 'base_sha256': BASE_SHA, 'manifest': str(manifest_path),
            'training_camera_names': names, 'views': views, 'allowed_pixel_paths': allowed, 'input_hashes': inputs,
            'external_timeout_seconds': 180, 'retained_candidate_checkpoints': 0}
    write(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    require(plan['status'] == 'locked_pending_root_gpu_authorization' and plan['specification'] == SPEC, 'Plan contract differs')
    snapshot = Path(plan['source_snapshot']).resolve()
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Run the frozen entrypoint')
    actual = {str(p.relative_to(snapshot)): digest(p) for p in sorted(p for p in snapshot.rglob('*') if p.is_file() and p.suffix in {'.py', '.md'})}
    require(actual == plan['source_hashes'], 'Frozen source bytes differ')
    require(actual['bridge_rgs/losses.py'] == LOSS_SHA and actual['bridge_rgs/fullbatch_appearance.py'] == SOLVER_SHA,
            'Numerical helper contract changed')
    require(digest(plan['base']) == plan['base_sha256'] == BASE_SHA, 'Wrong H3 base')
    for path, expected in plan['input_hashes'].items():
        require(digest(path) == expected, f'Changed input {path}')
    require(fixed_views(json.loads(Path(plan['manifest']).read_text())) == (plan['views'], plan['training_camera_names']),
            'View selection changed')
    for name in ('train', 'model', 'raw_grid', 'losses', 'fullbatch_appearance', 'evaluate', 'coordinates'):
        importlib.import_module('bridge_rgs.'+name)
    loaded = {}
    for name, module in list(sys.modules.items()):
        if name == 'bridge_rgs' or name.startswith('bridge_rgs.'):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(snapshot) and actual.get(str(path.relative_to(snapshot))) == digest(path), 'Unfrozen package import')
            loaded[name] = {'path': str(path), 'sha256': digest(path)}
    return loaded


def run(plan_path):
    started = time.monotonic()
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    output = Path(plan['output'])
    require(not (output/'execution_receipt.json').exists(), 'No retry/overwrite')
    os.chdir(plan['root'])
    loaded = verify(plan)
    receipt = {'status': 'running', 'plan_sha256': digest(plan_path), 'render_calls': 0, 'backward_calls': 0,
               'source_hashes': plan['source_hashes'], 'input_hashes': plan['input_hashes'], 'actual_imports': loaded,
               'base_sha256': BASE_SHA, 'profile': 'legacy_mixed_v1',
               'semantic_label_pixels_decoded': 0, 'val_pixels_decoded': 0,
               'views': [], 'checkpoint_written': False, 'committed_optimizer_steps': 0,
               'numerical_gate_failure_is_not_cuda_bug_evidence': True}
    write(output/'execution_receipt.json', receipt)
    scene = original = flags = modes = cameras_before = base = None
    imread = cv2.imread
    reads = Counter()
    deadline = started+SPEC['deadline_seconds']
    def alarm(*_):
        raise TimeoutError('Fixed 120-second diagnostic limit')
    old_handler = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.1, deadline-time.monotonic()))
    def guarded_read(filename, *args, **kwargs):
        name = str(Path(filename).resolve())
        require(name in plan['allowed_pixel_paths'], 'Read outside fixed two original TRAIN RGB')
        reads[name] += 1
        return imread(filename, *args, **kwargs)
    cv2.imread = guarded_read
    try:
        from bridge_rgs.coordinates import LEGACY, pixel_protocol
        from bridge_rgs.evaluate import distortion_render_grid
        from bridge_rgs.raw_grid import appearance_rgb_loss, build_raw_grid
        from bridge_rgs.train import load_scene
        torch.set_num_threads(8)
        cv2.setNumThreads(8)
        scene, base = load_scene(plan['base'])
        require(pixel_protocol(base) == LEGACY and base['sh_degree'] == 3, 'Wrong base profile/SH')
        original = {k: v.detach().clone() for k, v in scene.state_dict().items()}
        receipt['tensor_hashes_before'] = {k: tensor_hash(v) for k, v in original.items()}
        flags = {k: p.requires_grad for k, p in scene.named_parameters()}
        modes = {k: m.training for k, m in scene.named_modules()}
        cameras_before = base['training_cameras'].detach().clone()
        require(cameras_before.shape == (350, 4, 4), 'TRAIN camera array mismatch')
        scene.eval()
        named = dict(scene.named_parameters())
        for k, p in named.items():
            p.requires_grad_(k in KEYS)
        parameters = {k: named[k] for k in KEYS}
        manifest = json.loads(Path(plan['manifest']).read_text())
        require((Path(plan['root'])/base['config']['manifest']).resolve() == Path(plan['manifest']).resolve(), 'Base manifest lineage differs')
        receipt['training_cameras_sha256'] = tensor_hash(cameras_before)
        for view in plan['views']:
            check_time(deadline)
            camera = manifest['source_cameras'][str(view['camera_id'])]
            require(np.array_equal(camera['K'], view['K']) and camera['width'] == view['width'] and camera['height'] == view['height'], 'Original/prepared camera grid differs')
            pose = cameras_before[view['camera_index']].detach().to('cuda')
            image_path = str((Path(plan['root'])/view['source_image_path']).resolve())
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            require(image is not None and image.shape == (view['height'], view['width'], 3), 'Unexpected TRAIN RGB shape')
            target = torch.from_numpy(image[..., ::-1].copy()).to('cuda', torch.float32)/255
            layout = build_raw_grid(camera['K'], camera['opencv_distortion'], camera['width'], camera['height'], protocol=LEGACY)
            layout.warp.to('cuda')
            K = torch.tensor(layout.render_K, device='cuda')
            valid = torch.ones_like(target[..., 0], dtype=torch.bool)
            baseline_info = {}
            per_view_calls = 0
            def objective(baseline=False, K=K, pose=pose, layout=layout, target=target,
                          valid=valid, camera=camera, view=view, baseline_info=baseline_info):
                nonlocal per_view_calls
                check_time(deadline)
                require(receipt['render_calls'] < 40 and per_view_calls < 20, 'Render budget exceeded')
                canvas = scene.render(K, pose, layout.render_width, layout.render_height,
                                      degree=3, semantics=False, refine=False, absgrad=False)['rgb'].clamp(0, 1)
                receipt['render_calls'] += 1
                per_view_calls += 1
                prediction = layout.warp(canvas)
                loss, components = appearance_rgb_loss(prediction, target, valid)
                require(bool(torch.isfinite(loss)), 'Nonfinite objective')
                if baseline:
                    rk, cw, ch, mapping = distortion_render_grid(np.asarray(camera['K'], np.float32), camera['opencv_distortion'], camera['width'], camera['height'], LEGACY)
                    require(np.array_equal(rk, layout.render_K) and (cw, ch) == (layout.render_width, layout.render_height), 'Official overscan differs')
                    array = canvas.detach().cpu().numpy()
                    official = array if mapping is None else cv2.remap(array, mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    difference = float(np.max(np.abs(prediction.detach().cpu().numpy()-official)))
                    baseline_info.update(loss=float(loss.detach()), components={k: float(v) for k, v in components.items()},
                                         warp_cv2_max_abs_difference=difference, prediction_sha256=tensor_hash(prediction),
                                         warp=layout.warp.receipt(), camera_index=view['camera_index'],
                                         camera_max_abs_vs_original=float((pose.cpu()-torch.tensor(view['w2c_original'])).abs().max()))
                return loss, prediction
            loss, prediction = objective(baseline=True)
            gradients_all = torch.autograd.grad(loss, [prediction, *parameters.values()])
            receipt['backward_calls'] += 1
            image_gradient = gradients_all[0].detach()
            gradients = {k: v.detach() for k, v in zip(KEYS, gradients_all[1:], strict=True)}
            base_loss = float(loss.detach())
            del loss, prediction, gradients_all
            fd = {k: probe_group(parameters[k], gradients[k], base_loss, objective, deadline, image_gradient) for k in KEYS}
            step = transient_rms(parameters, gradients, base_loss, objective, deadline)
            require(per_view_calls == 20 and all(torch.equal(scene.state_dict()[k], v) for k, v in original.items()), 'Per-view state/call contract failed')
            require(all(p.grad is None for p in scene.parameters()), 'Unexpected accumulated parameter gradient')
            receipt['views'].append({'name': view['name'], 'render_calls': per_view_calls, 'baseline': baseline_info,
                                     'finite_differences': fd, 'temporary_rms_step': step})
        check_time(deadline)
        require(receipt['render_calls'] == 40 and receipt['backward_calls'] == 2, 'Incomplete fixed budget')
        receipt.update(status='completed', technical_gate=technical_gate(receipt['views']), technical_gate_passed=technical_gate(receipt['views'])['passed'])
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        cv2.imread = imread
        try:
            if scene is not None and original is not None:
                with torch.no_grad():
                    for k, value in scene.state_dict().items():
                        value.copy_(original[k])
                for k, p in scene.named_parameters():
                    p.requires_grad_(flags[k])
                    p.grad = None
                for k, m in scene.named_modules():
                    m.training = modes[k]
                receipt['tensor_hashes_after'] = {k: tensor_hash(v) for k, v in scene.state_dict().items()}
                receipt['all_model_tensors_finally_restored_exact'] = receipt['tensor_hashes_after'] == receipt['tensor_hashes_before']
                receipt['all_flags_restored'] = all(p.requires_grad == flags[k] for k, p in scene.named_parameters())
                receipt['all_modes_restored'] = all(m.training == modes[k] for k, m in scene.named_modules())
                receipt['training_cameras_unchanged'] = torch.equal(base['training_cameras'], cameras_before)
                require(all(receipt[k] for k in ('all_model_tensors_finally_restored_exact', 'all_flags_restored', 'all_modes_restored', 'training_cameras_unchanged')), 'Restoration failed')
            receipt['inputs_after_exact'] = all(digest(p) == h for p, h in plan['input_hashes'].items())
            receipt['source_after_exact'] = all(digest(Path(plan['source_snapshot'])/p) == h for p, h in plan['source_hashes'].items())
            receipt['all_bound_inputs_and_sources_unchanged'] = receipt['inputs_after_exact'] and receipt['source_after_exact']
            require(receipt['all_bound_inputs_and_sources_unchanged'], 'Input/source bytes changed')
        except Exception as error:
            receipt.update(status='failed', restoration_or_binding_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            receipt.update(seconds=time.monotonic()-started, pixel_reads=dict(reads), cuda_initialized=torch.cuda.is_initialized())
            write(output/'execution_receipt.json', receipt, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--run', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('/mnt/data/SHM2026/runs/h3_fullbatch_objective_precheck_v1'))
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.root, args.output))
    else:
        run(args.run)


if __name__ == '__main__':
    main()
