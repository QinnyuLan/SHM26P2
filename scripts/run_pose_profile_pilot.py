"""Isolated fixed 1050-step means-only raw/profile pilot; CPU preparation first.

No production trainer changes. Execution requires the independent numerical
calibration to pass and a separate root GPU handoff. No resume or model selection.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
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

import numpy as np
import torch

REFERENCE_SHA = 'e59ff68a141f567fac921837a5b3266b5cbb34c2ec736e707b12f7548e2e1468'
reference_path = Path(__file__).with_name('audit_pose_profile_gradients.py')
if hashlib.sha256(reference_path.read_bytes()).hexdigest() != REFERENCE_SHA:
    raise ValueError('Frozen profile helper changed')
_spec = importlib.util.spec_from_file_location('pilot_profile_reference', reference_path)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)
ORIGIN = Path('/mnt/data/SHM2026/runs/pose_profile_gradient_v1/plan.json')
ORIGIN_SHA = '0b5872b2b89fa1e9ac49ac37d919c4ca8d2bdbbd806dfc2441b492eee8006048'
CALIBRATION = Path('/mnt/data/SHM2026/runs/pose_profile_numerical_calibration_v1/plan.json')
CALIBRATION_SHA = 'd14423c9772793648d2e401af946418ce9c12cde00f4d54f514157faa7892f69'
KEY = 'splats.means'
ARMS = ('raw', 'profile')
CAMERA_KEYS = ('name', 'split', 'width', 'height', 'K', 'w2c', 'image_path', 'valid_path')
SPEC = {
    'protocol': 'mild_pose_means_only_raw_profile_pilot_v1',
    'steps': 1050, 'train_views': 350, 'cycles': 3, 'seed': 42, 'width': 320,
    'sh_degree': 3, 'arms': list(ARMS), 'active_parameter': KEY,
    'sampler': 'independent np.default_rng(42), three successive permutations of manifest TRAIN order',
    'optimizer': 'fresh Adam; fixed base final means lr/betas/eps; no inherited states or scheduler',
    'loss': 'valid-weighted RGB L2 / (2 sum of RGB scalar weights); profile exact frozen-J regularized minimum',
    'translation_fd_scene_scale': 1e-5, 'rotation_fd': 1e-4,
    'regularizer': 'original helper prior_precision + .001 diag(normal.diag.clamp_min(1))',
    'profile_relinearization': 'every training view at current means, 13 no-grad FD renders then 1 graph render',
    'pose_update': False, 'densification': False, 'semantic_training': False,
    'rgb_region_weights': False, 'ssim': False, 'extra_regularization': False,
    'raw_train_renders': 1050, 'profile_train_renders': 14700, 'total_backwards': 2100,
    'evaluation': 'after both fixed last checkpoints; original frozen native full50 evaluator, scale1 and LPIPS',
    'evaluation_renders': 150, 'paired_bootstrap_repeats': 5000,
    'checkpoint': 'full inference/warmstart only, no optimizer/RNG/density state, no resume',
    'total_wall_seconds': 600,
    'interpretation': 'mature-field mild-stress pilot only, unequal render/compute budget; not clean/stress interaction or superiority to BA',
}


def schedule(count=350):
    ref.require(count == SPEC['train_views'], 'This pilot has exactly 350 TRAIN views')
    rng = np.random.default_rng(SPEC['seed'])
    return np.concatenate([rng.permutation(count) for _ in range(SPEC['cycles'])]).tolist()


def adam_settings(state):
    groups = state['optimizers']['means']['param_groups']
    ref.require(len(groups) == 1 and len(groups[0]['params']) == 1, 'Unexpected base means optimizer')
    group = groups[0]
    ref.require(group.get('weight_decay', 0) == 0 and not group.get('amsgrad', False)
                and not group.get('maximize', False) and not group.get('capturable', False)
                and not group.get('differentiable', False) and group.get('fused') is None
                and group.get('foreach') is None and not group.get('decoupled_weight_decay', False),
                'Base Adam nondefault controls require a new protocol')
    result = {'lr': float(group['lr']), 'betas': list(group['betas']), 'eps': float(group['eps'])}
    ref.require(result['lr'] > 0 and result['eps'] > 0, 'Nonpositive Adam settings')
    return result


def means_optimizer(scene, settings):
    for name, parameter in scene.named_parameters():
        parameter.requires_grad_(name == KEY)
        parameter.grad = None
    active = [(name, p) for name, p in scene.named_parameters() if p.requires_grad]
    ref.require([name for name, _ in active] == [KEY], 'Only Gaussian means may have gradients')
    optimizer = torch.optim.Adam([active[0][1]], lr=settings['lr'],
                                 betas=tuple(settings['betas']), eps=settings['eps'])
    ref.require(not optimizer.state, 'Adam must start with empty moments')
    return optimizer


def inference_checkpoint(base, model, plan, arm, stage_receipt):
    """Explicit whitelist: no historical optimizer, RNG, density, step or config fiction."""
    fields = ('format_version', 'scene_scale', 'feature_dim', 'sh_degree', 'refiner_config',
              'pixel_protocol', 'manifest_sha256')
    result = {key: copy.deepcopy(base[key]) for key in fields if key in base}
    result['model'] = {key: value.detach().cpu().clone() for key, value in model.items()}
    # This is the actual fixed stage camera array, not the clean base camera array.
    result['training_cameras'] = torch.tensor([v['w2c'] for v in plan['views']], dtype=torch.float32)
    result.update(step=SPEC['steps'], checkpoint_kind='pose_profile_pilot_inference_or_warmstart',
                  ordinary_resume_allowed=False,
                  config={'manifest': plan['stress_manifest'], 'output': str(Path(plan['output'])/arm),
                          'steps': SPEC['steps'], 'parameter_scope': 'pose_profile_means_only_pilot',
                          'seed': SPEC['seed'], 'refiner': copy.deepcopy(base.get('refiner_config', {'type': 'legacy'}))},
                  pose_profile_pilot={'specification': SPEC, 'arm': arm,
                                      'base_checkpoint': plan['checkpoint'], 'base_sha256': plan['checkpoint_sha256'],
                                      'base_step': base['step'], 'base_config': copy.deepcopy(base['config']),
                                      'observed_training_manifest_sha256': plan['input_hashes'][plan['stress_manifest']],
                                      'source_hashes': plan['source_hashes'], 'adam': plan['adam'],
                                      'stage_receipt': {**copy.deepcopy(stage_receipt), 'status': 'training_budget_completed'},
                                      'ordinary_resume_allowed': False})
    # The old base has no manifest hash. If a future caller supplies one, it cannot
    # silently identify the new mild-stage configuration as the clean source.
    if 'manifest_sha256' in result:
        result['pose_profile_pilot']['base_manifest_sha256'] = result.pop('manifest_sha256')
    return result


def save_new_checkpoint(value, path):
    """No overwrite even if a destination appears between preparation and publish."""
    path = Path(path)
    ref.require(not path.exists(), 'Preserve existing checkpoint')
    temporary = path.with_suffix('.tmp')
    with temporary.open('xb') as stream:
        torch.save(value, stream)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def view_contract(clean, stressed):
    ref.require(len(clean['views']) == len(stressed['views']) == 400, 'Fixed 400-camera manifest required')
    ref.require([v['name'] for v in clean['views']] == [v['name'] for v in stressed['views']], 'Camera order differs')
    for a, b in zip(clean['views'], stressed['views'], strict=True):
        for key in ('name', 'split', 'image_path', 'mask_path', 'valid_path', 'K', 'width', 'height', 'w2c_original'):
            ref.require(a.get(key) == b.get(key), f'Pose-only stress changed {key}')
        if a['split'] == 'val':
            ref.require(a['w2c'] == b['w2c'], 'VAL camera changed')
    views = [{key: v[key] for key in CAMERA_KEYS} for v in stressed['views'] if v['split'] == 'train']
    ref.require(len(views) == 350 and len({v['name'] for v in views}) == 350, 'Exactly 350 unique TRAIN views required')
    return views


def verify(plan):
    ref.require(plan['specification'] == SPEC and plan['indices'] == schedule(), 'Fixed protocol or sequence differs')
    source = Path(plan['source_snapshot'])
    hashes = {str(p.relative_to(source)): ref.digest(p) for p in sorted(source.rglob('*.py'))}
    ref.require(hashes == plan['source_hashes'], 'Source bytes changed')
    for path, sha in plan['input_hashes'].items():
        ref.require(ref.digest(path) == sha, f'Input changed: {path}')


def calibration_gate():
    ref.require(ref.digest(CALIBRATION) == CALIBRATION_SHA, 'Fixed calibration plan changed')
    receipt_path, audit_path = CALIBRATION.parent/'execution_receipt.json', CALIBRATION.parent/'audit.json'
    ref.require(receipt_path.is_file() and audit_path.is_file(), 'Independent calibration has not completed')
    receipt, audit = json.loads(receipt_path.read_text()), json.loads(audit_path.read_text())
    ref.require(receipt['status'] == 'completed' and receipt['plan_sha256'] == CALIBRATION_SHA
                and receipt['report_sha256'] == ref.digest(audit_path), 'Calibration receipt/source mismatch')
    ref.require(audit['status'] == 'completed' and audit['tensor_restoration_exact'] and audit['saved_cameras_exact']
                and audit['summary'] == receipt['summary'], 'Calibration integrity failed')
    ref.require(audit['summary']['status'] == 'independent_direction_calibration_supported'
                and audit['summary']['reliable_views'] == 5 and audit['summary']['retained_views'] >= 4,
                'Independent numerical calibration did not support this pilot')
    return {str(CALIBRATION): CALIBRATION_SHA, str(receipt_path): ref.digest(receipt_path), str(audit_path): ref.digest(audit_path)}


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    ref.require(not output.exists(), 'Never overwrite an existing pilot')
    ref.require(ref.digest(ORIGIN) == ORIGIN_SHA and ref.digest(CALIBRATION) == CALIBRATION_SHA, 'Fixed predecessor plan changed')
    origin = json.loads(ORIGIN.read_text())
    ref.verify(origin)
    clean_path, stress_path = Path(origin['manifest']), root/'artifacts/pose_stress_mild/manifest.json'
    clean_sha, stress_sha = ref.digest(clean_path), ref.digest(stress_path)
    ref.require(clean_sha == origin['input_hashes'].get(str(clean_path)), 'Historical clean manifest bytes changed')
    ref.require(stress_sha == origin['input_hashes'].get(str(stress_path)), 'Historical mild stress manifest bytes changed')
    clean, stressed = json.loads(clean_path.read_text()), json.loads(stress_path.read_text())
    views = view_contract(clean, stressed)
    base = torch.load(origin['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    ref.require(base['step'] == 6000 and len(base['model'][KEY]) == 177378 and base['sh_degree'] == 3, 'Wrong mature field')
    settings = adam_settings(base)
    ref.require(torch.equal(base['training_cameras'], torch.tensor(
        [v['w2c'] for v in clean['views'] if v['split'] == 'train'], dtype=torch.float32)), 'Base clean camera mapping differs')
    # These files already exist locally. Preparation hashes but does not decode them.
    alexnet = Path.home()/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'
    lpips_weights = Path(importlib.util.find_spec('lpips').origin).parent/'weights/v0.1/alex.pth'
    ref.require(alexnet.is_file() and lpips_weights.is_file(), 'LPIPS weights must already be cached; no downloads')
    inputs = dict(origin['input_hashes'])
    inputs.update({str(ORIGIN): ORIGIN_SHA, str(CALIBRATION): CALIBRATION_SHA,
                   str(clean_path): clean_sha, str(stress_path): stress_sha})
    ref.require(str(clean_path) in inputs and str(stress_path) in inputs, 'Both manifests must be explicitly bound')
    for path in (alexnet, lpips_weights):
        inputs[str(path)] = ref.digest(path)
    for view in clean['views']:
        keys = ('image_path', 'valid_path', 'mask_path') if view['split'] == 'val' else ('image_path', 'valid_path')
        for key in keys:
            if view.get(key):
                path = str(Path(view[key]).resolve())
                if path not in inputs:
                    inputs[path] = ref.digest(path)
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(origin['source_snapshot'], snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in (Path(__file__), root/'scripts/compare_evaluations.py', root/'tests/test_pose_profile_pilot.py'):
        shutil.copy2(path, snapshot/path.name)
    sources = {str(p.relative_to(snapshot)): ref.digest(p) for p in sorted(snapshot.rglob('*.py'))}
    ref.require(all(sources[k] == v for k, v in origin['source_hashes'].items()), 'Old source copy changed')
    plan = {'status': 'cpu_prepared_pending_calibration_and_explicit_root_handoff',
            'created_utc': datetime.now(UTC).isoformat(), 'root': str(root), 'output': str(output),
            'source_snapshot': str(snapshot), 'source_hashes': sources, 'input_hashes': inputs,
            'checkpoint': origin['checkpoint'], 'checkpoint_sha256': origin['checkpoint_sha256'],
            'clean_manifest': str(clean_path), 'stress_manifest': str(stress_path),
            'views': views, 'indices': schedule(), 'adam': settings, 'scene_scale': origin['scene_scale'],
            'specification': SPEC, 'pixel_decodes_during_preparation': 0,
            'cpu_cuda_initialized': torch.cuda.is_initialized(), 'gpu_authorized': False,
            'lpips_weights': [str(alexnet), str(lpips_weights)]}
    ref.require(not plan['cpu_cuda_initialized'], 'CPU preparation initialized CUDA')
    ref.write_json(output/'plan.json', plan)
    verify(plan)
    return output/'plan.json'


@contextlib.contextmanager
def guarded_reads(paths):
    import cv2
    allowed = {str(Path(p).resolve()) for p in paths}
    original = cv2.imread
    reads = []
    def checked(path, *args, **kwargs):
        resolved = str(Path(path).resolve())
        ref.require(resolved in allowed, 'Pixel decode outside fixed stage allowlist')
        reads.append(resolved)
        return original(path, *args, **kwargs)
    cv2.imread = checked
    try:
        yield reads
    finally:
        cv2.imread = original


def load_modules(plan):
    modules = {name: importlib.import_module('bridge_rgs.'+name) for name in ('train', 'model', 'io', 'reliability', 'evaluate')}
    for name, module in modules.items():
        ref.require(Path(module.__file__).resolve() == Path(plan['source_snapshot'])/'bridge_rgs'/f'{name}.py', 'Unexpected actual import')
    return modules


def train_arm(plan, arm):
    import cv2
    ref.require(arm in ARMS, 'Unknown arm')
    output = Path(plan['output'])/arm
    ref.require(not output.exists(), 'No arm resume/retry')
    output.mkdir()
    modules = load_modules(plan)
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    torch.manual_seed(SPEC['seed'])
    report = {'status': 'running', 'arm': arm, 'steps': 0, 'training_renders': 0, 'backwards': 0,
              'active_parameters': [KEY], 'semantics_trained': False, 'ordinary_resume_allowed': False}
    scene, state, original = None, None, None
    started = time.monotonic()
    try:
        scene, state = modules['train'].load_scene(plan['checkpoint'])
        scene.eval()
        original_cameras = state['training_cameras'].clone()
        with ref.preserved_scene(scene) as original:
            optimizer = means_optimizer(scene, plan['adam'])
            report['fresh_adam_empty'] = not bool(optimizer.state)
            means = scene.splats['means']
            allowed = [v[k] for v in plan['views'] for k in ('image_path', 'valid_path')]
            with guarded_reads(allowed) as reads, (output/'train.jsonl').open('x') as log:
                optimization_start = time.monotonic()
                for step, index in enumerate(plan['indices'], 1):
                    view = plan['views'][index]
                    data = modules['io'].load_view({**view, 'mask_path': None}, scale=320/view['width'])
                    ref.require(data['mask'] is None and data['width'] == 320, 'Training pixel contract changed')
                    optimizer.zero_grad(set_to_none=True)
                    def render(camera, data=data):
                        value = scene.render(data['K'], camera, data['width'], data['height'],
                                             degree=3, semantics=False, absgrad=False)['rgb']
                        report['training_renders'] += 1
                        return value
                    weights = data['valid'][..., None]
                    if arm == 'profile':
                        _, jacobian = modules['reliability'].finite_difference_camera_jacobian(
                            render, data['w2c'], 1e-5*plan['scene_scale'], 1e-4)
                        system = ref.whitened_system(jacobian, weights, plan['scene_scale'],
                                                     torch.arange(int((weights > 0).sum())*3, device=means.device))
                    prediction = render(data['w2c'])
                    residual = prediction-data['rgb']
                    raw = ref.raw_loss(residual, weights)
                    loss = raw if arm == 'raw' else ref.profile_losses(residual, system)['profile']
                    ref.require(bool(torch.isfinite(loss)), 'Nonfinite loss')
                    loss.backward()
                    report['backwards'] += 1
                    ref.require(means.grad is not None and bool(torch.isfinite(means.grad).all()), 'Invalid means gradient')
                    ref.require(all(p.grad is None for name, p in scene.named_parameters() if name != KEY), 'Frozen gradient appeared')
                    before = means.detach().clone()
                    gradient = means.grad.detach().double()
                    optimizer.step()
                    change = means.detach().double()-before.double()
                    report['steps'] = step
                    record = {'step': step, 'view': view['name'], 'index': index,
                              'raw_l2': float(raw.detach()), 'objective': float(loss.detach()),
                              'gradient_l2': ref.norm(gradient), 'actual_update_l2': ref.norm(change),
                              'gradient_dot_actual_update': float((gradient*change).sum()),
                              'displacement_from_base_l2': ref.norm(means.detach().double()-original[KEY].double()),
                              'renders_so_far': report['training_renders']}
                    log.write(json.dumps(record, allow_nan=False)+'\n')
                    if step % 50 == 0 or step == SPEC['steps']:
                        log.flush()
                        print(json.dumps(record), flush=True)
                    del loss, raw, prediction, residual, gradient, change, before, data
                    if arm == 'profile':
                        del jacobian, system
                torch.cuda.synchronize()
                report.update(optimization_seconds=time.monotonic()-optimization_start, pixel_decodes=len(reads))
            expected = SPEC['raw_train_renders' if arm == 'raw' else 'profile_train_renders']
            ref.require(report['steps'] == 1050 and report['training_renders'] == expected and report['backwards'] == 1050,
                        'Training budget incomplete')
            scene.zero_grad(set_to_none=True)
            ref.require(all(torch.equal(v, original[k]) for k, v in scene.state_dict().items() if k != KEY), 'Frozen tensor changed')
            verify(plan)
            ref.require(torch.equal(state['training_cameras'], original_cameras), 'Source camera array changed')
            report.update(frozen_tensors_exact=True, source_camera_array_unchanged=True,
                          means_changed=not torch.equal(means, original[KEY]),
                          final_displacement_l2=ref.norm(means.detach().double()-original[KEY].double()),
                          optimizer_steps=int(optimizer.state[means]['step']))
            result = inference_checkpoint(state, scene.state_dict(), plan, arm, report)
            save_new_checkpoint(result, output/'last.pt')
            reloaded = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
            ref.require(all(torch.equal(reloaded['model'][k], v) for k, v in result['model'].items()), 'Saved model changed')
            report.update(checkpoint=str(output/'last.pt'), checkpoint_sha256=ref.digest(output/'last.pt'))
        report.update(status='completed', in_memory_source_restored=True,
                      source_modules={name: {'path': m.__file__, 'sha256': ref.digest(m.__file__)} for name, m in modules.items()})
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if scene is not None and original is not None:
            report['in_memory_source_restored'] = all(torch.equal(v, original[k]) for k, v in scene.state_dict().items())
        report['elapsed_seconds_including_load_save'] = time.monotonic()-started
        ref.write_json(output/'training_receipt.json', report)


def evaluate_endpoints(plan):
    modules = load_modules(plan)
    manifest = modules['io'].load_manifest(plan['clean_manifest'])
    allowed = [v[k] for v in manifest['views'] if v['split'] == 'val'
               for k in ('image_path', 'valid_path', 'mask_path') if v.get(k)]
    results = {}
    with guarded_reads(allowed):
        for arm, checkpoint in [('base', plan['checkpoint']), *[(a, str(Path(plan['output'])/a/'last.pt')) for a in ARMS]]:
            output = Path(plan['output'])/arm/'evaluation_native'
            ref.require(not output.exists(), 'Do not overwrite endpoint evaluation')
            scene, _ = modules['train'].load_scene(checkpoint)
            metrics = modules['evaluate'].evaluate_scene(scene, manifest, output, scale=1., lpips_metric=True)
            ref.require(metrics['validation_views'] == 50 and metrics['lpips'] is not None, 'Full RGB evaluation required')
            ref.require(all(np.isfinite(row[key]) for row in metrics['views'] for key in ('psnr', 'ssim', 'lpips')),
                        'Nonfinite endpoint RGB metrics')
            ref.require(len(list(output.glob('*_rgb.png'))) == len(list(output.glob('*_mask.png'))) == 50,
                        'Incomplete native PNG output')
            results[arm] = metrics
            del scene
            torch.cuda.empty_cache()
    compare_spec = importlib.util.spec_from_file_location('frozen_compare', Path(plan['source_snapshot'])/'compare_evaluations.py')
    compare = importlib.util.module_from_spec(compare_spec)
    compare_spec.loader.exec_module(compare)
    pairs = {}
    for baseline, candidate in (('base', 'raw'), ('base', 'profile'), ('raw', 'profile')):
        pairs[f'{candidate}_minus_{baseline}'] = compare.paired_comparison(
            results[baseline], results[candidate], repeats=5000, rgb_only=True)
    ref.write_json(Path(plan['output'])/'paired_rgb.json', pairs)
    return {arm: ref.digest(Path(plan['output'])/arm/'evaluation_native/metrics.json') for arm in results}


def worker(plan_path, phase):
    plan = json.loads(Path(plan_path).read_text())
    ref.require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Use frozen runner')
    verify(plan)
    calibration_gate()
    if phase in ARMS:
        train_arm(plan, phase)
    else:
        for arm in ARMS:
            record = json.loads((Path(plan['output'])/arm/'training_receipt.json').read_text())
            ref.require(record['status'] == 'completed' and record['steps'] == 1050 and record['frozen_tensors_exact']
                        and ref.digest(record['checkpoint']) == record['checkpoint_sha256'], 'Incomplete arm')
        evaluation_started = time.monotonic()
        metrics = evaluate_endpoints(plan)
        ref.write_json(Path(plan['output'])/'evaluation_receipt.json', {'status': 'completed', 'metrics_sha256': metrics,
                       'paired_rgb_sha256': ref.digest(Path(plan['output'])/'paired_rgb.json'), 'rgb_only_claims': True,
                       'elapsed_seconds': time.monotonic()-evaluation_started, 'native_renders': 150})
    verify(plan)


def remaining_budget(deadline):
    seconds = deadline-time.monotonic()
    if seconds <= 0:
        raise TimeoutError('Fixed 600 second pilot budget exhausted; no retry')
    return seconds


def execute(plan_path):
    started = time.monotonic()
    deadline = started+SPEC['total_wall_seconds']
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    ref.require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Use frozen runner')
    output = Path(plan['output'])
    ref.require(not (output/'execution_receipt.json').exists(), 'No reruns/resume')
    verify(plan)
    calibration = calibration_gate()
    graphics = ref.gpu_idle()
    ref.require(shutil.disk_usage(output).free >= 3*1024**3, 'Keep 3 GiB for two checkpoints/native outputs')
    receipt = {'status': 'running', 'started_utc': datetime.now(UTC).isoformat(), 'plan_sha256': ref.digest(plan_path),
               'calibration_input_hashes': calibration, 'graphics_clients': graphics, 'commands': [],
               'total_train_renders': 15750, 'total_train_backwards': 2100, 'native_evaluation_renders': 150,
               'total_wall_seconds': SPEC['total_wall_seconds']}
    ref.write_json(output/'execution_receipt.json', receipt)
    try:
        for phase in (*ARMS, 'evaluate'):
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(plan_path), '--phase', phase]
            receipt['commands'].append(command)
            with (output/f'{phase}_worker.log').open('x') as log:
                child = subprocess.run(command, cwd=plan['root'], env={**os.environ, 'PYTHONPATH': plan['source_snapshot']},
                                       stdout=log, stderr=subprocess.STDOUT, timeout=remaining_budget(deadline), check=False)
            ref.require(child.returncode == 0, f'{phase} worker failed: {child.returncode}')
        verify(plan)
        ref.require(ref.digest(plan_path) == receipt['plan_sha256'] and calibration_gate() == calibration, 'Bound plan/calibration changed')
        remaining_budget(deadline)
        receipt.update(status='completed', evaluation_receipt_sha256=ref.digest(output/'evaluation_receipt.json'),
                       training_receipt_sha256={a: ref.digest(output/a/'training_receipt.json') for a in ARMS})
    except (subprocess.TimeoutExpired, TimeoutError):
        receipt.update(status='inconclusive_timeout', retry_allowed=False,
                       limitation='Hard timeout does not guarantee child finally; immutable disk inputs and all existing products retained')
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            verify(plan)
            ref.require(ref.digest(plan_path) == receipt['plan_sha256'], 'Plan changed')
            receipt['bound_inputs_sources_unchanged'] = True
        except (OSError, ValueError, KeyError) as error:
            receipt.update(status='failed', source_error=f'{type(error).__name__}: {error}')
        if time.monotonic() > deadline and receipt['status'] == 'completed':
            receipt.update(status='inconclusive_timeout', retry_allowed=False,
                           limitation='Total 600s including final source verification exceeded; preserve outputs, no retry')
        receipt.update(elapsed_seconds=time.monotonic()-started, finished_utc=datetime.now(UTC).isoformat())
        ref.write_json(output/'execution_receipt.json', receipt, replace=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', type=Path)
    mode.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--phase', choices=(*ARMS, 'evaluate'), help=argparse.SUPPRESS)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('/mnt/data/SHM2026/runs/pose_profile_means_pilot_v1'))
    parser.add_argument('--confirmed-gpu-handoff', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.root, args.output))
    elif args.worker:
        ref.require(args.phase is not None, 'Worker phase required')
        worker(args.worker, args.phase)
    else:
        ref.require(args.confirmed_gpu_handoff, 'Await root handoff and passing independent calibration')
        receipt = execute(args.run)
        print(json.dumps(receipt))
        if receipt['status'] != 'completed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
