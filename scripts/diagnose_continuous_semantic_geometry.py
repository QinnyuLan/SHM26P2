"""Two fixed TRAIN views: standard-gsplat means derivatives, no optimization."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
PARENT = Path('/mnt/data/SHM2026/runs/raw_simplex_em_fw_v1')
RGB_SOURCE = Path('/mnt/data/SHM2026/runs/matched_rgb_teacher_adaptation_v1/plan.json')
PARENT_HASHES = {
    'plan.json': '29b066604846598a679a7fb0f1d25d88fce452cc25019adb3b552ad5a760fdcc',
    'execution_receipt.json': 'b129e790bb514dd1180e41978264af8b8ed6ee2ce9f297f24e2f17e749b9e7f0',
    'launch_receipt.json': '049643d84509a633b9cf42344100d56e779fead4596b070ae05725f072181422',
    'independent_cpu_review.json': '9c2dcfc8daf108545f06c4c9fda5ccd93ddca007af667f7752b37f01fe67f743',
    'final_q_delta.npz': 'fd735da6ee25b5e5ee4343f6dc48331cbedc7d717b75c9e86fa66285f64a63d5',
}
SPEC = {
    'protocol': 'continuous_semantic_geometry_preflight_v1',
    'names': ['002.png', '041.png'], 'point_count': 498136,
    'geometry': 'original H3; only independent means leaf is differentiated',
    'q': 'audited EM/FW q_renderer exactly; no renormalization or feature mapping',
    'renderer': 'two standard gsplat calls: SH RGB and explicit fixed five-channel q',
    'losses': ['rgb_mse', 'raw_affine_ce'], 'affine_noise': 5e-7,
    'direction': 'Sigma_i g_i / max_j sqrt(g_j^T Sigma_j g_j); FP64 construction',
    'amplitudes': [1/64, 1/128, 1/256],
    'endpoints': 'independently cast FP32(base64 +/- h direction64)',
    'loss_dtype': 'float64', 'renderer_parameter_dtype': 'float32',
    'fd_relative': .05, 'signal_multiplier': 10.,
    'floor': 'max(repeat scalar difference/h, FP64 scalar-spacing/h, FP64 dot gamma_n bound)',
    'gate': 'all 12 own-loss FD measurable and relative error <= .05; cross-loss descriptive',
    'counts': {'raster': 56, 'means_vjp': 4, 'target_decodes': 6},
    'internal_seconds': 120, 'external_seconds': 180,
    'optimizer_steps': 0, 'head_calls': 0, 'teacher_calls': 0, 'val_views': 0,
    'scope': 'objective-specific wiring calibration, not capacity or performance evidence',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as handle:
        handle.write(payload)


def files(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def covariance_direction(gradient, rotation, scales):
    """Common scalar normalization; each row has Mahalanobis length <= one."""
    g = np.asarray(gradient, np.float64)
    R = np.asarray(rotation, np.float64); s = np.asarray(scales, np.float64)
    require(g.ndim == 2 and g.shape[1] == 3 and s.shape == g.shape
            and R.shape == (len(g), 3, 3) and len(g) > 0, 'Invalid direction shapes')
    require(all(np.isfinite(x).all() for x in (g, R, s)) and (s > 0).all(), 'Nonfinite/invalid covariance input')
    require(np.max(np.abs(np.swapaxes(R, 1, 2) @ R-np.eye(3))) < 1e-10, 'Rotation not orthonormal')
    local = np.einsum('nji,nj->ni', R, g)*s
    length = np.linalg.norm(local, axis=1)
    denominator = float(length.max())
    require(np.isfinite(denominator) and denominator > 0, 'Zero/nonfinite own geometry gradient')
    direction = np.einsum('nij,nj->ni', R, s*local)/denominator
    require(np.isfinite(direction).all(), 'Nonfinite covariance-scaled direction')
    return direction, {'normalizer': denominator, 'max_mahalanobis': float((length/denominator).max()),
                       'nonzero_rows': int(np.count_nonzero(length))}


def displaced(base, direction, amplitude):
    require(base.dtype == np.float32 and direction.dtype == np.float64
            and base.shape == direction.shape and amplitude > 0, 'Invalid displacement contract')
    plus = (base.astype(np.float64)+amplitude*direction).astype(np.float32)
    minus = (base.astype(np.float64)-amplitude*direction).astype(np.float32)
    require(np.isfinite(plus).all() and np.isfinite(minus).all(), 'Nonfinite FP32 endpoints')
    require(not np.array_equal(plus, minus), 'FP32 endpoints collapsed')
    return plus, minus


def fd_record(gradient, plus, minus, h, loss_plus, loss_minus, baseline, repeat):
    require(h > 0 and all(np.isfinite(v) for v in (loss_plus, loss_minus, baseline, repeat)), 'Invalid FD scalars')
    actual = (plus.astype(np.float64)-minus.astype(np.float64))/(2*h)
    products = np.asarray(gradient, np.float64)*actual
    require(np.isfinite(products).all(), 'Nonfinite actual displacement dot')
    analytic = float(products.sum(dtype=np.float64)); fd = float((loss_plus-loss_minus)/(2*h))
    eps = np.finfo(np.float64).eps
    gamma = products.size*eps/(1-products.size*eps)
    scalar_floor = float((abs(np.spacing(loss_plus))+abs(np.spacing(loss_minus)))/(2*h))
    repeat_floor = float(abs(baseline-repeat)/h)
    dot_floor = float(gamma*np.abs(products).sum(dtype=np.float64))
    floor = max(scalar_floor, repeat_floor, dot_floor)
    measurable = bool(abs(analytic) > SPEC['signal_multiplier']*floor and analytic != 0)
    relative = float(abs(fd-analytic)/abs(analytic)) if analytic else None
    passed = bool(measurable and relative <= SPEC['fd_relative'])
    return {'amplitude': h, 'loss_plus': loss_plus, 'loss_minus': loss_minus,
            'analytic_actual_fp32_displacement': analytic, 'central_difference': fd,
            'absolute_error': float(abs(fd-analytic)), 'relative_error': relative,
            'floor': floor, 'repeat_floor': repeat_floor, 'scalar_spacing_floor': scalar_floor,
            'dot_reduction_bound': dot_floor, 'measurable': measurable, 'passed': passed,
            'changed_coordinates': int(np.count_nonzero(plus != minus)),
            'actual_direction_rms': float(np.sqrt(np.mean(actual*actual))),
            'quantization_direction_rms': None}


def actual_displacement_stats(base, plus, minus, direction, rotation, scales, h, gradient, losses):
    """Report one-sided actual movements too; only central FD enters the gate."""
    result = {}
    for key, endpoint, loss in zip(('plus', 'minus'), (plus, minus), losses[1:], strict=True):
        change = endpoint.astype(np.float64)-base.astype(np.float64)
        local = np.einsum('nji,nj->ni', rotation, change)/scales
        sign = 1 if key == 'plus' else -1
        result[key] = {'actual_max_mahalanobis': float(np.linalg.norm(local, axis=1).max()),
                       'analytic_actual_change': float(np.sum(gradient.astype(np.float64)*change)),
                       'actual_loss_change': float(loss-losses[0]),
                       'cast_error_rms': float(np.sqrt(np.mean((change-sign*h*direction)**2)))}
    return result


def summarize(records):
    own = [r for r in records if r['direction_loss'] == r['measured_loss']]
    require(len(records) == 24 and len(own) == 12, 'Incomplete fixed FD experiment')
    return {'main_count': len(own), 'main_passed': sum(r['passed'] for r in own),
            'main_measurable': sum(r['measurable'] for r in own),
            'numerical_status': 'passed' if all(r['passed'] for r in own) else 'not_passed',
            'maximum_main_relative_error': max((r['relative_error'] for r in own
                                                if r['relative_error'] is not None), default=None),
            'cross_derivatives_are_descriptive': True}


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Use a new data-disk directory')
    inputs = {str(PARENT/k): v for k, v in PARENT_HASHES.items()}
    require(all(sha(p) == h for p, h in inputs.items()), 'Completed parent lineage changed')
    parent, receipt, launch, audit = (read(PARENT/k) for k in
        ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json'))
    require(receipt['status'] == launch['status'] == 'completed' and launch['natural_completion']
            and launch['exit_code'] == 0 and audit['status'] == 'passed'
            and receipt['plan_sha256'] == launch['plan_sha256'] == PARENT_HASHES['plan.json']
            and launch['execution_receipt_sha256'] == PARENT_HASHES['execution_receipt.json'], 'Audited natural parent completion required')
    source = Path(parent['source_snapshot'])
    require(files(source) == parent['source_hashes'], 'Parent frozen source changed')
    views = {v['name']: v for v in read(parent['manifest'])['views'] if v['split'] == 'train'}
    selected = [views[n] for n in SPEC['names']]
    require(len(views) == 350, 'Wrong TRAIN camera population')
    expected = {**parent['input_hashes'], **read(RGB_SOURCE)['input_hashes']}
    inputs.update({str(RGB_SOURCE): sha(RGB_SOURCE), str(ROOT/'uv.lock'): sha(ROOT/'uv.lock')})
    for key in ('base_checkpoint', 'manifest'):
        inputs[parent[key]] = parent['input_hashes'][parent[key]]
    for view in selected:
        for key in ('image_path', 'mask_path', 'valid_path'):
            require(view[key] in expected, 'Missing inherited TRAIN input hash')
            inputs[view[key]] = expected[view[key]]
    require(all(sha(p) == h for p, h in inputs.items()), 'Bound input changed')
    snapshot = output/'source_snapshot'
    shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in (Path(__file__), ROOT/'tests/test_continuous_semantic_geometry.py',
                 ROOT/'docs/continuous_semantic_geometry_protocol.md'):
        shutil.copy2(path, snapshot/path.name)
    plan = {k: parent[k] for k in ('base_checkpoint', 'manifest', 'all_training_camera_names',
            'expected_gsplat_binary', 'installed_sources', 'runtime_versions', 'environment', 'numerics', 'class_weights')}
    plan.update(specification=SPEC, output=str(output), source_snapshot=str(snapshot), source_hashes=files(snapshot),
                inherited_source_hashes=parent['source_hashes'], input_hashes=inputs,
                q_delta=str(PARENT/'final_q_delta.npz'), views=selected,
                status='prepared_pending_root_execution', prepare_checkpoint_loads=0, prepare_pixel_decodes=0)
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC and files(plan['source_snapshot']) == plan['source_hashes'], 'Frozen contract changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Frozen entry only')
    binary = plan['expected_gsplat_binary']
    for path, digest in {**plan['input_hashes'], **plan['installed_sources'], binary['path']: binary['sha256']}.items():
        require(sha(path) == digest, 'Input/source changed: '+path)
    for key, value in plan['environment'].items():
        require(os.environ.get(key) == value, 'Environment changed: '+key)
    for key, value in plan['runtime_versions'].items():
        require(importlib.metadata.version(key) == value, 'Runtime changed: '+key)


def standard_pair(scene, means, q, view, rasterization, counts):
    """No custom rasterizer and no detached means/projection metadata."""
    import torch
    s = scene.splats
    common = {'means': means, 'quats': s['quats'].detach(), 'scales': s['log_scales'].detach().exp(),
              'opacities': s['opacity_logits'].detach().sigmoid(),
              'viewmats': torch.tensor(view['w2c_original'], dtype=torch.float32, device=means.device)[None],
              'Ks': torch.tensor(view['K'], dtype=torch.float32, device=means.device)[None],
              'width': view['width'], 'height': view['height'], 'packed': False,
              'rasterize_mode': 'antialiased', 'near_plane': .01, 'far_plane': 1e6, 'absgrad': False}
    rgb, alpha_rgb, _ = rasterization(colors=torch.cat([s['sh0'], s['sh_rest']], 1).detach(),
        sh_degree=3, render_mode='RGB+ED', backgrounds=scene.background_logits.detach().sigmoid()[None], **common)
    counts['raster'] += 1
    raw, alpha_semantic, _ = rasterization(colors=q, backgrounds=q.new_tensor([[1., 0., 0., 0., 0.]]), **common)
    counts['raster'] += 1
    require(torch.equal(alpha_rgb, alpha_semantic), 'RGB/semantic shared alpha differs')
    return rgb[0, ..., :3], raw[0]


def perform(plan, report, deadline):
    import cv2
    import torch
    from gsplat import rasterization
    require(not any(n.startswith('bridge_rgs') for n in sys.modules), 'No live bridge package imports')
    adapter = importlib.import_module('preflight_simplex_scene')
    from bridge_rgs.densification import quaternion_to_matrix
    from bridge_rgs.semantic_assignment import affine_raw_ce
    from bridge_rgs.train import load_scene
    torch.set_num_threads(4)
    previous = adapter.numerical_flags(plan['numerics'])
    report['numerics_actual'] = adapter.numerical_flags()
    require(report['numerics_actual'] == plan['numerics'], 'Numerical flags differ')
    scene = captured = None
    allowed = {v[k] for v in plan['views'] for k in ('image_path', 'mask_path', 'valid_path')}
    original_imread = cv2.imread
    def guarded_imread(path, *args, **kwargs):
        require(str(path) in allowed, 'Only fixed two TRAIN target grids may be decoded')
        return original_imread(path, *args, **kwargs)
    cv2.imread = guarded_imread
    counts = report['counts']; records = []
    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError('Fixed preflight deadline; no retries')
    try:
        scene, state = load_scene(plan['base_checkpoint'])
        captured = adapter.capture_scene(scene)
        scene.eval().requires_grad_(False)
        require(scene.sh_degree == 3 and scene.pixel_protocol == 'legacy_mixed_v1'
                and scene.mip_filter_config is None and len(scene.splats['means']) == SPEC['point_count'], 'Wrong base field')
        train = [v for v in read(plan['manifest'])['views'] if v['split'] == 'train']
        require([v['name'] for v in train] == plan['all_training_camera_names']
                and np.array_equal(state['training_cameras'].cpu().numpy(),
                                   np.asarray([v['w2c_original'] for v in train], np.float32)), 'Original TRAIN cameras differ')
        with np.load(plan['q_delta'], allow_pickle=False) as data:
            q32 = data['q_renderer'].copy()
            require(np.array_equal(q32, data['q_master'].astype(np.float32)), 'q master/cast identity differs')
        require(q32.dtype == np.float32 and q32.shape == (SPEC['point_count'], 5)
                and np.isfinite(q32).all() and (q32 >= 0).all()
                and np.max(np.abs(q32.astype(np.float64).sum(1)-1)) <= 1e-6, 'Invalid frozen q')
        q = torch.from_numpy(q32).cuda()
        base = scene.splats['means'].detach().cpu().numpy().copy()
        R = quaternion_to_matrix(scene.splats['quats'].detach().double()).cpu().numpy()
        scales = scene.splats['log_scales'].detach().double().exp().cpu().numpy()
        weights = torch.tensor(plan['class_weights'], dtype=torch.float32, device='cuda')
        report['q32_sha256'] = array_sha(q32); report['base_means_sha256'] = array_sha(base)
        report['views'] = []
        torch.cuda.reset_peak_memory_stats()
        for view in plan['views']:
            tick()
            rgb_np = cv2.imread(view['image_path'], cv2.IMREAD_COLOR)
            mask_np = cv2.imread(view['mask_path'], cv2.IMREAD_UNCHANGED)
            valid_np = cv2.imread(view['valid_path'], cv2.IMREAD_UNCHANGED)
            counts['target_decodes'] += 3
            shape = (view['height'], view['width'])
            require(rgb_np is not None and mask_np is not None and valid_np is not None
                    and rgb_np.shape == (*shape, 3) and mask_np.shape == valid_np.shape == shape
                    and np.isin(mask_np, [0, 1, 2, 3, 4, 255]).all(), 'Invalid fixed TRAIN pixels')
            target_rgb = torch.from_numpy(cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)).cuda().double()/255
            labels = torch.from_numpy(mask_np.astype(np.int64)).cuda()
            valid = torch.from_numpy(valid_np > 0).cuda()
            require(bool(valid.any()) and bool((valid & (labels < 5)).any()), 'Empty target support')
            def evaluate(value, gradient=False, view=view, valid=valid, target_rgb=target_rgb, labels=labels):
                tick()
                means = torch.from_numpy(value.copy()).cuda().requires_grad_(gradient)
                with torch.set_grad_enabled(gradient):
                    rgb, raw = standard_pair(scene, means, q, view, rasterization, counts)
                    losses = ((rgb.clamp(0, 1).double()[valid]-target_rgb[valid]).square().mean(),
                              affine_raw_ce(raw, labels, valid, weights, SPEC['affine_noise']))
                values = [float(loss.detach()) for loss in losses]
                require(np.isfinite(values).all(), 'Nonfinite objective')
                gradients = []
                if gradient:
                    for loss in losses:
                        g, = torch.autograd.grad(loss, means)
                        counts['means_vjp'] += 1
                        require(bool(torch.isfinite(g).all()), 'Nonfinite means VJP')
                        gradients.append(g.detach().cpu().numpy())
                return values, gradients
            baseline, gradients = evaluate(base, True)
            repeat, _ = evaluate(base)
            view_report = {'name': view['name'], 'baseline': baseline, 'repeat': repeat,
                           'rgb_valid_pixels': int(valid.sum()), 'known_pixels': int((valid & (labels < 5)).sum()),
                           'directions': {}}
            for j, name in enumerate(SPEC['losses']):
                direction, stats = covariance_direction(gradients[j], R, scales)
                view_report['directions'][name] = stats
                for h in SPEC['amplitudes']:
                    plus, minus = displaced(base, direction, h)
                    fp, _ = evaluate(plus); fm, _ = evaluate(minus)
                    for k, measured in enumerate(SPEC['losses']):
                        row = fd_record(gradients[k], plus, minus, h, fp[k], fm[k], baseline[k], repeat[k])
                        actual = (plus.astype(np.float64)-minus.astype(np.float64))/(2*h)
                        row['quantization_direction_rms'] = float(np.sqrt(np.mean((actual-direction)**2)))
                        row['one_sided'] = actual_displacement_stats(base, plus, minus, direction, R, scales,
                            h, gradients[k], (baseline[k], fp[k], fm[k]))
                        row.update(view=view['name'], direction_loss=name, measured_loss=measured)
                        records.append(row)
            report['views'].append(view_report)
            print(json.dumps({'completed_view': view['name'], 'counts': counts}), flush=True)
        require(counts == SPEC['counts'], 'Fixed numerical call budget differs')
        report['records'] = records; report.update(summarize(records))
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    finally:
        cv2.imread = original_imread
        try:
            if scene is not None and captured is not None:
                report['restoration'] = adapter.restore_scene(scene, captured)
                require(report['restoration']['state_exact'] and report['restoration']['flags_modes_gradients_restored'], 'Full restoration failed')
        finally:
            adapter.numerical_flags(previous)
            report['numerics_after_restore'] = adapter.numerical_flags()
            report['numerics_restored'] = report['numerics_after_restore'] == previous
            require(report['numerics_restored'], 'Numerical flags restoration failed')
        report['actual_imports'] = {}
        for name, module in list(sys.modules.items()):
            if name.startswith('bridge_rgs') and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve()
                require(path.is_relative_to(plan['source_snapshot'])
                        and plan['source_hashes'].get(str(path.relative_to(plan['source_snapshot']))) == sha(path), 'Nonfrozen import')
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}
        if counts['raster']:
            from gsplat.cuda._backend import _C
            binary = {'path': str(Path(_C.__file__).resolve()), 'sha256': sha(_C.__file__)}
            require(binary == plan['expected_gsplat_binary'], 'Loaded renderer binary differs')
            report['actual_gsplat_binary'] = binary


def execute(path, expected):
    require(sha(path) == expected, 'Wrong plan SHA')
    plan = read(path); output = Path(plan['output'])
    require(not any((output/k).exists() for k in ('execution_started.json', 'execution_receipt.json')), 'Refuse previous attempt')
    write(output/'execution_started.json', {'plan_sha256': expected, 'pid': os.getpid(), 'time': time.time()})
    report = {'status': 'running', 'plan_sha256': expected, 'counts': dict.fromkeys(SPEC['counts'], 0),
              'optimizer_steps': 0, 'head_calls': 0, 'teacher_calls': 0, 'val_views': 0}
    started = time.monotonic()
    def expired(*_):
        raise TimeoutError('Fixed120s budget')
    previous = signal.signal(signal.SIGALRM, expired); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan); sys.path.insert(0, plan['source_snapshot'])
        old = importlib.import_module('optimize_raw_simplex')
        report['gpu_before'] = old.gpu_inventory()
        perform(plan, report, started+SPEC['internal_seconds'])
        verify(plan)
        report.update(status='completed', sources_inputs_unchanged=True)
    except BaseException as error:
        report.update(status='inconclusive_timeout' if isinstance(error, TimeoutError) else 'failed',
                      error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        report['elapsed_seconds'] = time.monotonic()-started
        write(output/'execution_receipt.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path); action.add_argument('--run', type=Path)
    action.add_argument('--print-spec', action='store_true')
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.print_spec:
        print(json.dumps(SPEC, indent=2)); return
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode generation')
    if args.prepare:
        prepare(args.prepare)
    else:
        execute(args.run, args.expected_plan_sha256)


if __name__ == '__main__':
    main()
