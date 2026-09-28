"""Conditional full-TRAIN appearance solver; preparation code, no plan creation.

Default prints the proposed contract only. --run requires a later reviewed,
immutable source snapshot/plan and explicit bound review of the 40-render audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
DIAGNOSTIC_PLAN_SHA = 'c8db5d457049844b346727ebf83bfc2ac8d65971c0903d7feb33d94cb675b5d3'
SPEC = {'protocol': 'conditional_fullbatch_appearance_v1', 'train_views': 350,
        'pixel_protocol': 'legacy_mixed_v1', 'sh_degree': 3,
        'beta1': 0., 'beta2': .999, 'eps': 1e-8,
        'rates': {'splats.sh0': .00025, 'splats.sh_rest': .0000125, 'background_logits': .0001},
        'alphas': [1., .5, .25], 'armijo_c1': .0001,
        'decrease_floor': 'max(1e-7,1e-5*abs(baseline))',
        'max_accepted_steps': 20, 'max_complete_passes': 40, 'optimization_seconds': 360.,
        'objective': '350-view equal mean: .8 full original RGB L1 + .2 complete 7-window SSIM loss',
        'prediction': 'legacy official pinhole/overscan -> clamp[0,1] -> fixed original-grid float gather',
        'deadline': 'check before/after every view; discard partial gradient/trial; no partial acceptance',
        'cpu_threads': 8, 'semantics_or_VAL_pixels_read': 0,
        'checkpoint': 'one standalone full inference checkpoint; no optimizer/RNG; ordinary resume forbidden'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data, *, replace=False):
    payload = json.dumps(data, indent=2, allow_nan=False)+'\n'
    if replace:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(payload)
        temporary.replace(path)
    else:
        with path.open('x') as stream:
            stream.write(payload)


def bound(item):
    require(sha(item['path']) == item['sha256'], f"Changed bound input: {item['path']}")


def diagnostic_gate(plan):
    for key in ('diagnostic_receipt', 'diagnostic_review'):
        bound(plan[key])
    receipt, review = read(plan['diagnostic_receipt']['path']), read(plan['diagnostic_review']['path'])
    require(receipt['status'] == 'completed' and receipt['plan_sha256'] == DIAGNOSTIC_PLAN_SHA
            and receipt['render_calls'] == 40 and receipt['all_model_tensors_finally_restored_exact'],
            'The fixed 40-render diagnostic has not completed successfully')
    require(review.get('decision') == 'allow_fullbatch_appearance'
            and review.get('diagnostic_receipt_sha256') == plan['diagnostic_receipt']['sha256']
            and review.get('full_objective_fd_and_single_step_review_passed') is True
            and bool(review.get('reason')), 'Explicit reviewed FD/one-step interpretation is required')
    # Numerical FD tolerances are not converted to an automatic CUDA verdict.


def verify_plan(path):
    plan = read(path)
    require(plan['status'] == 'locked_authorized_after_actual_objective_diagnostic'
            and plan['specification'] == SPEC, 'No approved locked full-batch experiment plan')
    snapshot = Path(plan['source_snapshot']).resolve()
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Use the later reviewed immutable runner')
    for relative, expected in plan['source_hashes'].items():
        bound({'path': snapshot/relative, 'sha256': expected})
    require(plan['base']['sha256'] == BASE_SHA, 'Wrong fixed H3 base')
    for key in ('base', 'manifest'):
        bound(plan[key])
    for filename, expected in plan['input_hashes'].items():
        bound({'path': filename, 'sha256': expected})
    diagnostic_gate(plan)
    return plan


def fixed_views(manifest, base, root, bindings, training_camera_names):
    import numpy as np
    import torch

    from bridge_rgs.coordinates import LEGACY, pixel_protocol
    require(pixel_protocol(base) == pixel_protocol(manifest) == LEGACY, 'Require the historical legacy profile')
    training = [v for v in manifest['views'] if v['split'] == 'train']
    names = [v['name'] for v in training]
    require(training_camera_names == names, 'TRAIN camera name/index order differs from bound manifest')
    views = sorted(training, key=lambda v: v['name'])
    require(len(views) == 350 and len({v['name'] for v in views}) == 350, 'Require all unique 350 TRAIN views')
    cameras = base['training_cameras']
    require(cameras.shape == (350, 4, 4) and cameras.dtype == torch.float32
            and bool(torch.isfinite(cameras).all()), 'Wrong base TRAIN camera schema')
    by_name = {name: cameras[index].detach().clone() for index, name in enumerate(names)}
    original = torch.tensor([v['w2c_original'] for v in training], dtype=torch.float32)
    mapping = {'names_in_checkpoint_index_order': names,
               'source': 'base.training_cameras indexed by bound original manifest TRAIN order',
               'camera_bytes_sha256': hashlib.sha256(cameras.numpy().tobytes()).hexdigest(),
               'different_from_original_pose_count': int((cameras != original).any(dim=2).any(dim=1).sum()),
               'max_abs_difference_from_original_pose': float((cameras-original).abs().max())}
    for view in views:
        camera = manifest['source_cameras'][str(view['camera_id'])]
        require(np.array_equal(view['K'], camera['K']) and view['width'] == camera['width']
                and view['height'] == camera['height'], 'Prepared/source image grid differs')
        path = str((Path(root)/view['source_image_path']).resolve())
        require(path in bindings, 'Every original TRAIN RGB must have a bound SHA')
    return views, by_name, mapping


def gpu_idle():
    result = subprocess.run(['nvidia-smi', '-q', '-x'], check=True, capture_output=True, text=True, timeout=5)
    document = ET.fromstring(result.stdout)
    require(document.findall('gpu'), 'GPU status missing')
    for gpu in document.findall('gpu'):
        listing = gpu.find('processes')
        require(listing is not None and (listing.text or '').strip() not in {'N/A', 'Not Supported'}, 'GPU status unavailable')
    require(all(p.findtext('type') == 'G' for p in document.findall('.//process_info')), 'GPU compute slot is occupied')


def run(path):
    path = Path(path).resolve()
    plan = verify_plan(path)
    output = Path(plan['output']).resolve()
    require(not output.exists(), 'Never overwrite/retry an existing experiment')
    gpu_idle()
    os.chdir(plan['root'])
    import cv2
    import torch

    from bridge_rgs import fullbatch_appearance as solver
    from bridge_rgs.raw_grid import appearance_rgb_loss, build_raw_grid
    from bridge_rgs.train import load_scene
    snapshot = Path(plan['source_snapshot']).resolve()
    for name, module in list(sys.modules.items()):
        if name == 'bridge_rgs' or name.startswith('bridge_rgs.'):
            source = Path(module.__file__).resolve()
            require(source.is_relative_to(snapshot), 'Non-frozen package import')
            relative = str(source.relative_to(snapshot))
            require(plan['source_hashes'].get(relative) == sha(source), 'Imported source SHA missing/different')
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    output.mkdir(parents=True)
    receipt = {'status': 'running', 'plan_sha256': sha(path), 'renderer_calls': 0,
               'complete_passes': 0, 'partial_renders': 0, 'accepted_steps': 0, 'checkpoint_written': False,
               'source_and_input_policy': 'fixed TRAIN original RGB only; no semantic labels/VAL'}
    write(output/'execution_receipt.json', receipt)
    scene = original = original_flags = None
    imread = cv2.imread
    started = time.monotonic()
    try:
        scene, base = load_scene(plan['base']['path'])
        scene.eval()
        original = {key: value.detach().clone() for key, value in scene.state_dict().items()}
        original_flags = {key: p.requires_grad for key, p in scene.named_parameters()}
        for key, parameter in scene.named_parameters():
            parameter.requires_grad_(key in solver.KEYS)
        named = dict(scene.named_parameters())
        parameters = {key: named[key] for key in solver.KEYS}
        manifest = read(plan['manifest']['path'])
        require((Path(plan['root'])/base['config']['manifest']).resolve() == Path(plan['manifest']['path']).resolve(),
                'Observed manifest path must match the original base configuration')
        require(base['sh_degree'] == SPEC['sh_degree'], 'Fixed SH degree differs')
        views, cameras, mapping = fixed_views(manifest, base, plan['root'], plan['input_hashes'],
                                              plan['training_camera_names'])
        receipt['training_camera_mapping'] = mapping
        allowed = {str((Path(plan['root'])/v['source_image_path']).resolve()) for v in views}
        def guarded_read(filename, *args, **kwargs):
            require(str(Path(filename).resolve()) in allowed, 'Pixel read outside original TRAIN RGB')
            return imread(filename, *args, **kwargs)
        cv2.imread = guarded_read
        layouts = {}
        for view in views:
            key = str(view['camera_id'])
            if key not in layouts:
                camera = manifest['source_cameras'][key]
                layout = build_raw_grid(camera['K'], camera['opencv_distortion'], camera['width'], camera['height'],
                                        protocol='legacy_mixed_v1')
                layout.warp.to('cuda')
                layouts[key] = layout
        def objective(view):
            filename = str((Path(plan['root'])/view['source_image_path']).resolve())
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            require(image is not None and image.shape == (view['height'], view['width'], 3), 'Wrong original RGB dimensions')
            target = torch.from_numpy(image[..., ::-1].copy()).to('cuda', torch.float32)/255
            layout = layouts[str(view['camera_id'])]
            canvas = scene.render(torch.tensor(layout.render_K, device='cuda'),
                                  cameras[view['name']].to('cuda'),
                                  layout.render_width, layout.render_height, degree=3,
                                  semantics=False, absgrad=False)['rgb'].clamp(0, 1)
            receipt['renderer_calls'] += 1
            prediction = layout.warp(canvas)
            return appearance_rgb_loss(prediction, target, torch.ones_like(target[..., 0], dtype=torch.bool))[0]
        optimizer = solver.RMSArmijo(parameters)
        budget = solver.PassBudget(time.monotonic)
        last_loss, reason = None, 'accepted_step_limit'
        torch.cuda.reset_peak_memory_stats()
        with (output/'passes.jsonl').open('x') as log:
            def full_pass(backward, role):
                require(budget.passes < budget.max_passes, 'Complete-pass budget exhausted')
                before, calls_before = time.monotonic(), receipt['renderer_calls']
                try:
                    with torch.set_grad_enabled(backward):
                        loss = solver.stream_mean_loss(views, objective, backward=backward,
                                                       check_time=budget.check_time)
                except solver.PassExpired:
                    # Throw away any partial accumulated gradient. A trial's enclosing
                    # transaction restores its candidate before the solver exits.
                    scene.zero_grad(set_to_none=True)
                    count = receipt['renderer_calls']-calls_before
                    receipt['partial_renders'] += count
                    log.write(json.dumps({'role': role, 'complete': False, 'renders': count,
                                          'seconds': time.monotonic()-before, 'loss': None})+'\n')
                    log.flush()
                    return None, False
                on_time = budget.completed()
                receipt['complete_passes'] = budget.passes
                record = {'pass': budget.passes, 'role': role, 'loss': loss, 'complete': True,
                          'renders': receipt['renderer_calls']-calls_before,
                          'seconds': time.monotonic()-before, 'on_time': on_time}
                log.write(json.dumps(record, allow_nan=False)+'\n')
                log.flush()
                return loss, on_time
            while optimizer.accepted_steps < SPEC['max_accepted_steps']:
                if not budget.can_start():
                    reason = 'complete_pass_or_time_budget'; break
                scene.zero_grad(set_to_none=True)
                baseline, on_time = full_pass(True, 'baseline_gradient')
                if not on_time:
                    reason = 'time_budget_gradient_pass_discarded'; break
                if last_loss is not None and baseline > last_loss+solver.decrease_floor(last_loss):
                    raise ValueError('Accepted baseline replay increased beyond the fixed guard')
                require(all(p.grad is None for k, p in named.items() if k not in solver.KEYS), 'Frozen gradient path opened')
                require(all(p.grad is not None for p in parameters.values()), 'Missing appearance gradient')
                gradients = {k: p.grad.detach().clone() for k, p in parameters.items()}
                require(all(torch.isfinite(g).all() for g in gradients.values()), 'Nonfinite full gradient')
                if not any(bool(g.any()) for g in gradients.values()):
                    reason = 'zero_full_gradient'; break
                proposal = optimizer.propose(gradients)
                accepted = False
                for alpha in solver.ALPHAS:
                    if not budget.can_start():
                        reason = 'complete_pass_or_time_budget'; break
                    with optimizer.candidate(proposal, gradients, alpha) as trial:
                        if trial.actual_slope < 0:
                            value, on_time = full_pass(False, f'trial_alpha_{alpha}')
                            if on_time:
                                accepted = trial.accept(baseline, value)
                        else:
                            value, on_time = None, True
                        record = {'accepted_step_before': optimizer.accepted_steps, 'alpha': alpha,
                                  'baseline': baseline, 'candidate': value, 'g_dot_d': proposal.analytic_slope,
                                  'g_dot_actual_delta': trial.actual_slope,
                                  'displacement_absmax': trial.displacement_absmax, 'accepted': accepted,
                                  'completed_passes': budget.passes}
                        log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
                    if accepted:
                        last_loss = value
                        receipt['accepted_steps'] = optimizer.accepted_steps
                        break
                    if not on_time:
                        reason = 'time_budget_trial_pass_discarded'; break
                if not accepted:
                    if reason == 'accepted_step_limit':
                        reason = 'three_fixed_trials_rejected'
                    break
        optimization_seconds = time.monotonic()-budget.started
        require(receipt['renderer_calls'] == budget.passes*SPEC['train_views']+receipt['partial_renders'],
                'Render/pass accounting differs')
        scene.zero_grad(set_to_none=True)
        require(all(torch.equal(value, original[key]) for key, value in scene.state_dict().items()
                    if key not in solver.KEYS), 'Frozen model tensor changed')
        verify_plan(path)
        require(sha(path) == receipt['plan_sha256'], 'Plan changed during solver')
        if optimizer.accepted_steps:
            metadata = {'protocol': SPEC['protocol'], 'base': plan['base'], 'manifest_observed': plan['manifest'],
                        'base_manifest_declared_sha256': base.get('manifest_sha256'),
                        'source_files_sha256': plan['source_hashes'], 'stage_config': SPEC,
                        'training_camera_mapping': mapping,
                        'accepted_steps': optimizer.accepted_steps, 'complete_passes': budget.passes,
                        'last_accepted_train_loss': last_loss, 'stop_reason': reason}
            result = solver.inference_checkpoint(base, scene.state_dict(), metadata)
            final = output/'final.pt'
            solver.save_full_inference(result, final)
            reloaded = torch.load(final, map_location='cpu', weights_only=False)
            require(all(torch.equal(reloaded['model'][k], v) for k, v in result['model'].items())
                    and torch.equal(reloaded['training_cameras'], base['training_cameras']), 'Saved model differs')
            receipt.update(checkpoint_written=True, checkpoint=str(final), checkpoint_sha256=sha(final))
        receipt.update(status='completed', stop_reason=reason, last_accepted_train_loss=last_loss,
                       optimization_seconds=optimization_seconds,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       frozen_tensors_exact=True, base_training_cameras_exact=True,
                       ordinary_resume_allowed=False)
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        cv2.imread = imread
        try:
            if scene is not None and original is not None:
                with torch.no_grad():
                    for key, value in scene.state_dict().items():
                        value.copy_(original[key])
                for key, parameter in scene.named_parameters():
                    if original_flags is not None:
                        parameter.requires_grad_(original_flags[key])
                    parameter.grad = None
                restored = all(torch.equal(v, original[k]) for k, v in scene.state_dict().items())
                receipt['in_memory_base_restored_exact'] = restored
                require(restored, 'Process base restoration failed')
        except BaseException as error:
            receipt.update(status='failed', restoration_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            receipt['elapsed_seconds_including_setup_save'] = time.monotonic()-started
            write(output/'execution_receipt.json', receipt, replace=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.run:
        print(json.dumps(run(args.run)))
    else:
        print(json.dumps({'status': 'code_preparation_only_no_plan_no_GPU', 'specification': SPEC}, indent=2))


if __name__ == '__main__':
    main()
