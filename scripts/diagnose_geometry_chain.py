"""One reconstructed failed means direction: localize its rendering-chain secant."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
PARENT = Path('/mnt/data/SHM2026/runs/continuous_geometry_preflight_v1')
PARENT_PLAN_SHA = '6a79de1060458df432b8df8b37edb0a5c6d2a2f9028cc2d4f37f901303787845'
PARENT_EXEC_SHA = '90ef418aecdbe909ca802ffecdd1c8bcd136c760af9fb5760ab44a3aef3b908e'
PARENT_AUDIT_SHA = '0f51b71d006deba716f5470a0d63f2038c837b9b4972abed5146a7c4348e4718'
ATTRS = ('means2d', 'conics', 'opacities', 'colors')
SPEC = {
    'protocol': 'continuous_geometry_chain_v1', 'view': '041.png',
    'direction': 'reconstructed original RGB-own covariance-normalized gradient', 'amplitude': 1/128,
    'loss': 'FP64 valid-pixel mean MSE of clamp(RGB,0,1)',
    'native_render_mode': 'RGB+ED; retain exact four-channel raster colors/background',
    'chain': ['A:means actual secant VJP', 'B:continuous raster-input actual secant VJP',
              'C:raw raster image actual secant VJP', 'D:actual loss central difference'],
    'replay': 'baseline offsets/flatten IDs; all referenced endpoint projections must remain valid',
    'invalid_projection': 'no endpoint replay; partial B explicitly unavailable as full chain',
    'maximum_native_rasters': 3, 'maximum_replay_rasters': 3, 'backward_sweeps': 1,
    'internal_seconds': 120, 'external_seconds': 180,
    'optimizer_steps': 0, 'semantic_rasters': 0, 'teacher_calls': 0, 'head_calls': 0, 'val_views': 0,
    'claim': 'localization only; no new pass gate, training admission, or revision of old 3/12 result',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as f:
        f.write(payload)


def files(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def to_numpy(value):
    return value.detach().cpu().contiguous().numpy()


def dot_secant(gradient, plus, minus, h):
    g, p, m = (np.asarray(v, np.float64) for v in (gradient, plus, minus))
    require(g.shape == p.shape == m.shape and h > 0
            and all(np.isfinite(v).all() for v in (g, p, m)), 'Invalid finite secant arrays')
    products = g*((p-m)/(2*h))
    return {'value': float(products.sum(dtype=np.float64)),
            'absolute_product_sum': float(np.abs(products).sum(dtype=np.float64))}


def mse_decomposition(raw0, raw_plus, raw_minus, target, valid, gradient_image, h):
    """Exact clamped-image quadratic identity, separate from raw-image VJP C."""
    r0, rp, rm, y = (np.asarray(x, np.float64) for x in (raw0, raw_plus, raw_minus, target))
    valid = np.asarray(valid, bool)
    require(r0.shape == rp.shape == rm.shape == y.shape and r0.shape == (*valid.shape, 3)
            and valid.any() and h > 0 and all(np.isfinite(x).all() for x in (r0, rp, rm, y)), 'Bad image grids')
    p0, pp, pm, yt = (np.clip(x, 0, 1)[valid] for x in (r0, rp, rm, y))
    f0, fp, fm = (float(np.mean((p-yt)**2)) for p in (p0, pp, pm))
    D = (fp-fm)/(2*h)
    C = dot_secant(gradient_image, rp, rm, h)['value']
    clipped_C = float(np.mean(2*(p0-yt)*(pp-pm))/(2*h))
    quadratic = float(np.mean((pp-pm)*(pp+pm-2*p0))/(2*h))
    branch = lambda x: np.where(x < 0, -1, np.where(x > 1, 1, 0))
    return {'baseline': f0, 'plus': fp, 'minus': fm, 'C_raw_image_vjp': C, 'D_loss_fd': D,
            'C_clamped_image_exact': clipped_C, 'quadratic_remainder': quadratic,
            'quadratic_identity_residual': float(D-clipped_C-quadratic),
            'raw_to_clamped_C_difference': float(clipped_C-C),
            'clamp_crossing_valid_channels_plus': int(np.count_nonzero((branch(r0) != branch(rp))[valid])),
            'clamp_crossing_valid_channels_minus': int(np.count_nonzero((branch(r0) != branch(rm))[valid]))}


def reference_validity(flatten_ids, valid0, valid_plus, valid_minus):
    ids = np.asarray(flatten_ids, np.int64)
    v0, vp, vm = (np.asarray(v, bool).reshape(-1) for v in (valid0, valid_plus, valid_minus))
    require(v0.shape == vp.shape == vm.shape and ids.ndim == 1
            and len(ids) > 0 and ids.min() >= 0 and ids.max() < len(v0), 'Invalid baseline references')
    unique = np.unique(ids)
    require(v0[unique].all(), 'Baseline list references invalid projection')
    return {'referenced_unique_rows': len(unique), 'invalid_plus': int((~vp[unique]).sum()),
            'invalid_minus': int((~vm[unique]).sum()),
            'endpoint_replay_allowed': bool(vp[unique].all() and vm[unique].all())}


def intermediate_secant(gradient, plus, minus, valid0, valid_plus, valid_minus, h):
    """Never multiply or interpret invalid fused-projection storage."""
    g = np.asarray(gradient)
    plus, minus = np.asarray(plus), np.asarray(minus)
    v0, vp, vm = (np.asarray(v, bool) for v in (valid0, valid_plus, valid_minus))
    require(g.ndim >= 1 and g.shape == plus.shape == minus.shape
            and v0.shape == vp.shape == vm.shape == (len(g),)
            and np.isfinite(g).all() and h > 0, 'Invalid intermediate secant shape/gradient')
    common = v0 & vp & vm
    active = np.any(g != 0, axis=tuple(range(1, g.ndim))) if g.ndim > 1 else g != 0
    missing = active & ~common
    selected = active & common
    partial = dot_secant(g[selected], plus[selected], minus[selected], h)
    return {'value': partial['value'] if not missing.any() else None,
            'common_valid_value': partial['value'], 'absolute_product_sum': partial['absolute_product_sum'],
            'nonzero_gradient_invalid_endpoint_rows': int(missing.sum()),
            'common_valid_rows': int(common.sum()), 'nonzero_gradient_common_valid_rows': int(selected.sum()),
            'complete': bool(not missing.any())}


@contextlib.contextmanager
def capture_raster(module):
    """Patch only a Python call boundary during this independent diagnostic."""
    original = module.rasterize_to_pixels
    signature = inspect.signature(original)
    captured = {}
    def wrapper(*args, **kwargs):
        require(not captured, 'Expected exactly one unchunked native raster')
        bound = signature.bind(*args, **kwargs); bound.apply_defaults()
        captured['inputs'] = dict(bound.arguments)
        result = original(*args, **kwargs)
        captured['output'] = result
        return result
    module.rasterize_to_pixels = wrapper
    try:
        yield captured
    finally:
        module.rasterize_to_pixels = original


def standard_rgb(scene, means, view, rasterization):
    """Exactly the RGB call of the frozen two-task preflight, without its q call."""
    import torch
    s = scene.splats
    return rasterization(means=means, quats=s['quats'].detach(), scales=s['log_scales'].detach().exp(),
        opacities=s['opacity_logits'].detach().sigmoid(), colors=torch.cat([s['sh0'], s['sh_rest']], 1).detach(),
        sh_degree=3, render_mode='RGB+ED', backgrounds=scene.background_logits.detach().sigmoid()[None],
        viewmats=torch.tensor(view['w2c_original'], dtype=torch.float32, device=means.device)[None],
        Ks=torch.tensor(view['K'], dtype=torch.float32, device=means.device)[None],
        width=view['width'], height=view['height'], packed=False, rasterize_mode='antialiased',
        near_plane=.01, far_plane=1e6, absgrad=False)


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Use new data-disk output')
    require(sha(PARENT/'plan.json') == PARENT_PLAN_SHA and sha(PARENT/'execution_receipt.json') == PARENT_EXEC_SHA,
            'Original numerical result changed')
    require(sha(PARENT/'independent_cpu_review.json') == PARENT_AUDIT_SHA
            and read(PARENT/'independent_cpu_review.json')['status'] == 'passed', 'Parent independent audit changed')
    parent, receipt, launch = (read(PARENT/k) for k in ('plan.json', 'execution_receipt.json', 'launch_receipt.json'))
    require(receipt['status'] == launch['status'] == 'completed' and launch['natural_completion']
            and launch['exit_code'] == 0 and receipt['numerical_status'] == 'not_passed'
            and receipt['main_passed'] == 3 and receipt['main_count'] == 12
            and launch['execution_receipt_sha256'] == PARENT_EXEC_SHA
            and launch['plan_sha256'] == receipt['plan_sha256'] == PARENT_PLAN_SHA, 'Expected preserved failed numerical gate')
    source = Path(parent['source_snapshot'])
    require(files(source) == parent['source_hashes'], 'Original frozen source changed')
    view, = [v for v in parent['views'] if v['name'] == SPEC['view']]
    inputs = {str(PARENT/k): sha(PARENT/k) for k in
              ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json')}
    for path in (parent['base_checkpoint'], parent['manifest'], parent['q_delta'], view['image_path'], view['valid_path'], str(ROOT/'uv.lock')):
        inputs[path] = parent['input_hashes'][path]
    require(all(sha(p) == h for p, h in inputs.items()), 'Bound input changed')
    snapshot = output/'source_snapshot'
    shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in (Path(__file__), ROOT/'tests/test_geometry_chain.py', ROOT/'docs/geometry_chain_diagnostic.md'):
        shutil.copy2(path, snapshot/path.name)
    plan = {k: parent[k] for k in ('base_checkpoint', 'manifest', 'all_training_camera_names', 'q_delta',
            'expected_gsplat_binary', 'installed_sources', 'runtime_versions', 'environment', 'numerics')}
    plan.update(output=str(output), source_snapshot=str(snapshot), source_hashes=files(snapshot), input_hashes=inputs,
                specification=SPEC, inherited_source_hashes=parent['source_hashes'], view=view,
                prior_execution=str(PARENT/'execution_receipt.json'), status='prepared_pending_root_execution')
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC and files(plan['source_snapshot']) == plan['source_hashes'], 'Frozen contract changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Frozen entry required')
    binary = plan['expected_gsplat_binary']
    for path, digest in {**plan['input_hashes'], **plan['installed_sources'], binary['path']: binary['sha256']}.items():
        require(sha(path) == digest, 'Input/dependency changed: '+path)
    for key, value in plan['runtime_versions'].items():
        require(importlib.metadata.version(key) == value, 'Runtime changed: '+key)
    for key, value in plan['environment'].items():
        require(os.environ.get(key) == value, 'Environment changed: '+key)


def perform(plan, report, deadline):
    import cv2
    import torch
    from gsplat import rasterization
    rendering = importlib.import_module('gsplat.rendering')
    low_level = rendering.rasterize_to_pixels
    require(not any(n.startswith('bridge_rgs') for n in sys.modules), 'No live bridge imports')
    old = importlib.import_module('diagnose_continuous_semantic_geometry')
    adapter = importlib.import_module('preflight_simplex_scene')
    from bridge_rgs.densification import quaternion_to_matrix
    from bridge_rgs.train import load_scene
    torch.set_num_threads(4)
    previous = adapter.numerical_flags(plan['numerics'])
    report['numerics_actual'] = adapter.numerical_flags()
    require(report['numerics_actual'] == plan['numerics'], 'Numerical flags changed')
    scene = captured = None
    output = Path(plan['output']); report['arrays'] = {}; view = plan['view']; h = SPEC['amplitude']
    original_imread = cv2.imread
    def guarded_imread(path, *args, **kwargs):
        require(str(path) in (view['image_path'], view['valid_path']), 'Only 041 RGB/valid; no mask or VAL')
        return original_imread(path, *args, **kwargs)
    cv2.imread = guarded_imread
    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError('Fixed chain-diagnostic deadline')
    def save(name, **arrays):
        tick(); path = output/(name+'.npz')
        require(not path.exists(), 'No array replacement')
        np.savez(path, **arrays)
        report['arrays'][name] = {'path': str(path), 'sha256': sha(path),
            'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()}}
    try:
        torch.cuda.reset_peak_memory_stats()
        report['memory_scope'] = 'peak allocated bytes since immediately before scene loading; includes loaded model'
        scene, state = load_scene(plan['base_checkpoint']); captured = adapter.capture_scene(scene)
        scene.eval().requires_grad_(False)
        require(scene.pixel_protocol == 'legacy_mixed_v1' and scene.sh_degree == 3
                and scene.mip_filter_config is None, 'Wrong fixed H3 field')
        train = [v for v in read(plan['manifest'])['views'] if v['split'] == 'train']
        require([v['name'] for v in train] == plan['all_training_camera_names']
                and np.array_equal(to_numpy(state['training_cameras']), np.asarray([v['w2c_original'] for v in train], np.float32)), 'TRAIN cameras changed')
        target_u8 = cv2.imread(view['image_path'], cv2.IMREAD_COLOR)
        report['counts']['target_decodes'] += 1
        valid_u8 = cv2.imread(view['valid_path'], cv2.IMREAD_UNCHANGED)
        report['counts']['target_decodes'] += 1
        require(target_u8 is not None and valid_u8 is not None
                and target_u8.shape == (view['height'], view['width'], 3)
                and valid_u8.shape == target_u8.shape[:2], 'Wrong target grid')
        target_u8 = cv2.cvtColor(target_u8, cv2.COLOR_BGR2RGB)
        valid_np = valid_u8 > 0; require(valid_np.any(), 'Empty RGB support')
        target = torch.from_numpy(target_u8).cuda().double()/255
        valid = torch.from_numpy(valid_np).cuda()
        base = to_numpy(scene.splats['means']).copy()
        R = to_numpy(quaternion_to_matrix(scene.splats['quats'].detach().double()))
        scales = to_numpy(scene.splats['log_scales'].detach().double().exp())
        save('target_and_metric', target_rgb_u8=target_u8, valid=valid_np, rotation=R, scales=scales)
        states = {}; gradients = {}
        def native(name, means_np, gradient=False):
            tick(); means = torch.from_numpy(means_np.copy()).cuda().requires_grad_(gradient)
            with torch.set_grad_enabled(gradient), capture_raster(rendering) as cap:
                rgbd, alpha, info = standard_rgb(scene, means, view, rasterization)
            report['counts']['native_rasters'] += 1
            inp = cap['inputs']; raw4, native_alpha = cap['output']
            require(not inp['packed'] and not inp['absgrad'] and inp['masks'] is None
                    and raw4.shape[-1] == inp['colors'].shape[-1] == inp['backgrounds'].shape[-1] == 4
                    and torch.equal(raw4[..., :3], rgbd[..., :3]) and torch.equal(alpha, native_alpha), 'Unexpected raster call shape/path')
            raster_call = {k: int(inp[k]) for k in ('image_width', 'image_height', 'tile_size')}
            raster_call.update(packed=False, absgrad=False, masks=None, channels=4)
            if 'raster_call' in report:
                require(report['raster_call'] == raster_call, 'Raster dimensions/settings changed')
            else:
                report['raster_call'] = raster_call
            loss = (raw4[0, ..., :3].clamp(0, 1).double()[valid]-target[valid]).square().mean()
            if gradient:
                items = [means, *(inp[k] for k in ATTRS), raw4]
                values = torch.autograd.grad(loss, items)
                report['counts']['backward_sweeps'] += 1
                gradients.update({k: to_numpy(v) for k, v in zip(('means', *ATTRS, 'image'), values, strict=True)})
                require(all(np.isfinite(v).all() for v in gradients.values()), 'Nonfinite chain gradients')
            valid_rows = (info['radii'] > 0).all(-1)
            safe = {}
            for k in ATTRS:
                tensor = inp[k].detach()
                mask = valid_rows if tensor.ndim == valid_rows.ndim else valid_rows[..., None]
                safe[k] = to_numpy(torch.where(mask, tensor, torch.zeros_like(tensor)))[0]
                require(np.isfinite(safe[k]).all(), 'Nonfinite valid raster attributes')
            ledger = {k: to_numpy(inp[k]) for k in ('isect_offsets', 'flatten_ids')}
            arrays = {**safe, **ledger, 'means': means_np.copy(), 'valid_projection': to_numpy(valid_rows)[0],
                      'radii': to_numpy(info['radii'])[0], 'raw4': to_numpy(raw4)[0],
                      'alpha': to_numpy(native_alpha)[0], 'backgrounds': to_numpy(inp['backgrounds'])}
            save(name, **arrays)
            detached = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
            states[name] = {'arrays': arrays, 'inputs': detached, 'loss': float(loss.detach())}
            return states[name]
        baseline = native('baseline', base, True)
        save('gradients', **gradients)
        direction, stats = old.covariance_direction(gradients['means'], R, scales)
        plus, minus = old.displaced(base, direction, h)
        save('direction', direction=direction)
        for name, values in (('plus', plus), ('minus', minus)):
            native(name, values)
        a0, ap, am = (states[k]['arrays'] for k in ('baseline', 'plus', 'minus'))
        A = dot_secant(gradients['means'], plus, minus, h)
        pieces = {k: intermediate_secant(gradients[k][0], ap[k], am[k], a0['valid_projection'],
                                        ap['valid_projection'], am['valid_projection'], h) for k in ATTRS}
        B = sum(v['value'] for v in pieces.values()) if all(v['complete'] for v in pieces.values()) else None
        image = mse_decomposition(a0['raw4'][..., :3], ap['raw4'][..., :3], am['raw4'][..., :3],
                                  target_u8.astype(np.float64)/255, valid_np, gradients['image'][0, ..., :3], h)
        report['chain'] = {'A_means': A, 'B_components': pieces, 'B_complete': B,
            **image, 'A_minus_B': A['value']-B if B is not None else None,
            'B_minus_C': B-image['C_raw_image_vjp'] if B is not None else None,
            'C_minus_D': image['C_raw_image_vjp']-image['D_loss_fd']}
        report['direction_reconstruction'] = stats
        prior = read(plan['prior_execution'])
        prior_view, = [v for v in prior['views'] if v['name'] == SPEC['view']]
        prior_row, = [r for r in prior['records'] if r['view'] == SPEC['view'] and r['amplitude'] == h
                     and r['direction_loss'] == r['measured_loss'] == 'rgb_mse']
        report['prior_comparison'] = {'bytewise_direction_verifiable': False,
            'reason': 'original run did not save full gradient or endpoint arrays; this is a reconstructed instance',
            'baseline_loss_difference': baseline['loss']-prior_view['baseline'][0],
            'normalizer_difference': stats['normalizer']-prior_view['directions']['rgb_mse']['normalizer'],
            'analytic_difference': A['value']-prior_row['analytic_actual_fp32_displacement'],
            'loss_fd_difference': image['D_loss_fd']-prior_row['central_difference']}
        report['quantization'] = {}
        for name, endpoint in (('plus', plus), ('minus', minus)):
            delta = endpoint.astype(np.float64)-base.astype(np.float64)
            lengths = np.linalg.norm(np.einsum('nji,nj->ni', R, delta)/scales, axis=1)
            row_dot = np.sum(gradients['means'].astype(np.float64)*delta, axis=1)
            over = lengths > h
            report['quantization'][name] = {'rows_exceeding_ideal_h': int(over.sum()), 'maximum_mahalanobis': float(lengths.max()),
                'all_rows_gradient_dot': float(row_dot.sum()), 'exceeding_rows_gradient_dot': float(row_dot[over].sum()),
                'exceeding_rows_absolute_gradient_dot': float(np.abs(row_dot[over]).sum())}
        check = reference_validity(a0['flatten_ids'], a0['valid_projection'], ap['valid_projection'], am['valid_projection'])
        report['replay'] = check
        report['ledger'] = {name: {'offsets_exact_baseline': bool(np.array_equal(states[name]['arrays']['isect_offsets'], a0['isect_offsets'])),
                                  'ordered_flatten_ids_exact_baseline': bool(np.array_equal(states[name]['arrays']['flatten_ids'], a0['flatten_ids'])),
                                  'isect_count': int(states[name]['arrays']['flatten_ids'].size),
                                  'active_row_change_count': int(np.count_nonzero(states[name]['arrays']['valid_projection'] != a0['valid_projection']))}
                            for name in ('plus', 'minus')}
        def replay(name):
            tick(); args = dict(states['baseline']['inputs'])
            for key in ATTRS:
                args[key] = states[name]['inputs'][key]
            with torch.no_grad():
                result, alpha = low_level(**args)
            report['counts']['replay_rasters'] += 1
            array = to_numpy(result)[0]
            save('replay_'+name, raw4=array, alpha=to_numpy(alpha)[0])
            return array, to_numpy(alpha)[0]
        repeated, repeated_alpha = replay('baseline')
        require(np.array_equal(repeated, a0['raw4']) and np.array_equal(repeated_alpha, a0['alpha']), 'Baseline replay must be exact, including raw depth and alpha')
        report['replay']['baseline_four_channel_and_alpha_exact'] = True
        if check['endpoint_replay_allowed']:
            replay_plus, _ = replay('plus'); replay_minus, _ = replay('minus')
            fixed = mse_decomposition(repeated[..., :3], replay_plus[..., :3], replay_minus[..., :3],
                target_u8.astype(np.float64)/255, valid_np, gradients['image'][0, ..., :3], h)
            report['replay']['fixed_ledger'] = fixed
            report['replay']['natural_minus_fixed_loss_fd'] = image['D_loss_fd']-fixed['D_loss_fd']
            report['replay']['scope'] = 'tile inclusion/order jointly frozen; alpha thresholds, caps, early termination remain active'
        else:
            report['replay']['scope'] = 'endpoint replay skipped: no intersection restriction or invalid attribute substitution'
        require(report['counts'] == {'native_rasters': 3, 'replay_rasters': 3 if check['endpoint_replay_allowed'] else 1,
                                     'backward_sweeps': 1, 'target_decodes': 2}, 'Unexpected fixed call counts')
        report['diagnostic_status'] = 'completed_localization_only'
    finally:
        report['peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        cv2.imread = original_imread
        require(rendering.rasterize_to_pixels is low_level, 'Raster hook restoration failed')
        try:
            if scene is not None and captured is not None:
                report['restoration'] = adapter.restore_scene(scene, captured)
                require(report['restoration']['state_exact'] and report['restoration']['flags_modes_gradients_restored'], 'Scene restoration failed')
        finally:
            adapter.numerical_flags(previous)
            report['numerics_restored'] = adapter.numerical_flags() == previous
            require(report['numerics_restored'], 'Numerics restoration failed')
        report['actual_imports'] = {}
        for name, module in list(sys.modules.items()):
            if (name.startswith('bridge_rgs') or name in ('diagnose_continuous_semantic_geometry',
                    'preflight_simplex_scene', 'optimize_raw_simplex')) and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve()
                require(path.is_relative_to(plan['source_snapshot'])
                        and plan['source_hashes'].get(str(path.relative_to(plan['source_snapshot']))) == sha(path), 'Nonfrozen local import')
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}
        if report['counts']['native_rasters']:
            from gsplat.cuda._backend import _C
            binary = {'path': str(Path(_C.__file__).resolve()), 'sha256': sha(_C.__file__)}
            require(binary == plan['expected_gsplat_binary'], 'Renderer binary changed')
            report['actual_gsplat_binary'] = binary


def execute(path, expected):
    require(sha(path) == expected, 'Wrong plan SHA')
    plan = read(path); output = Path(plan['output'])
    require(not any((output/k).exists() for k in ('execution_started.json', 'execution_receipt.json')), 'Refuse previous attempt')
    write(output/'execution_started.json', {'plan_sha256': expected, 'pid': os.getpid(), 'time': time.time()})
    report = {'status': 'running', 'plan_sha256': expected, 'optimizer_steps': 0, 'head_calls': 0,
              'teacher_calls': 0, 'val_views': 0, 'semantic_rasters': 0, 'label_decodes': 0,
              'counts': dict.fromkeys(('native_rasters', 'replay_rasters', 'backward_sweeps', 'target_decodes'), 0)}
    started = time.monotonic()
    def expired(*_):
        raise TimeoutError('Fixed chain120s deadline')
    previous = signal.signal(signal.SIGALRM, expired); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan); sys.path.insert(0, plan['source_snapshot'])
        report['gpu_before'] = importlib.import_module('optimize_raw_simplex').gpu_inventory()
        perform(plan, report, started+SPEC['internal_seconds'])
        verify(plan); report.update(status='completed', sources_inputs_unchanged=True)
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
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode')
    if args.prepare:
        prepare(args.prepare)
    else:
        execute(args.run, args.expected_plan_sha256)


if __name__ == '__main__':
    main()
