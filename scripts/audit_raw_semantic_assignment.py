"""One fixed TRAIN sweep: approximate raw-color EM versus feature-only Adam.

Prepare is CPU only. Execute needs a separate GPU handoff and the frozen package.
No production checkpoint or VAL scoring is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch

from bridge_rgs.semantic_assignment import affine_raw_ce, realize_probabilities, waterfill_counts

BASE_SHA = '391a0577450f7f458b75cb9e2fcf926953142b516728ea3600ef2ef064d72c13'
MANIFEST_SHA = '91b41aeecedc352e4882eb80a1bb635d479ae432251a4c96c85a9dc110cbb327'
SPEC = {'protocol': 'train259_raw_assignment_one_sweep_v1', 'seed': 42,
        'views': 259, 'width': 1320, 'height': 989, 'degree': 3,
        'pixel_protocol': 'colmap_corner_v2', 'delta': 5e-7, 'epsilon': 1e-5,
        'class_weight_power': .25, 'class_frequency_floor': .002, 'class_weight_max': 3.,
        'adam': {'lr': .01, 'eps': 1e-15, 'betas': [.9, .999]},
        'phases': ['initial_em_e', 'em_endpoint', 'adam_sweep', 'adam_endpoint'],
        'scene_calls': 1036, 'raster_calls': 2072, 'color_backwards': 259,
        'adam_updates': 259, 'm_steps': 1, 'timeout_seconds': 600,
        'probability_realization_atol': 5e-6,
        'em_decrease_absolute_tolerance': 1e-6, 'em_decrease_relative_tolerance': 1e-4,
        'objective': 'FP64 affine-noise raw weighted CE; valid pixel mean per view, equal mean of259views',
        'scope': 'TRAIN diagnostic only; no base full-objective pass, no VAL, retries, adoption or production checkpoint',
        'domain_difference': 'EM target q>=epsilon; Adam may leave this bounded simplex',
        'vjp': 'Approximate gsplat color VJP; no per-ray scale correction; actual endpoint objective decides numerical decrease'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_hash(x):
    a = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def write_json(path, value, replace=False):
    path = Path(path)
    if replace:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)
    else:
        with path.open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')


def semantic_state(scene):
    return {k: tensor_hash(v) for k, v in scene.state_dict().items() if k != 'splats.sem_features'}


@contextmanager
def capture_raw(module, detach_colors=False):
    """EM has an isolated colors leaf; Adam preserves the original feature graph."""
    original = module.rasterization
    capture = {'calls': 0, 'semantic_calls': 0}

    def wrapped(*args, **kwargs):
        capture['calls'] += 1
        if kwargs.get('render_mode', 'RGB') == 'RGB+ED':
            return original(*args, **kwargs)
        capture['semantic_calls'] += 1
        if args:
            raise ValueError('Expected the frozen renderer keyword contract')
        if detach_colors:
            kwargs = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            kwargs['colors'] = kwargs['colors'].requires_grad_(True)
        with torch.enable_grad() if detach_colors else nullcontext():
            result = original(**kwargs)
            capture.update(raw=result[0][0, ..., :5], alpha=result[1][0, ..., 0], colors=kwargs['colors'])
        return result
    module.rasterization = wrapped
    try:
        yield capture
    finally:
        module.rasterization = original


def em_feature_candidate(initial_features, weight, bias, counts, qold):
    """Realize only active rows: zero responsibility leaves FP32 feature bytes intact."""
    targets, active = waterfill_counts(counts, qold, SPEC['epsilon'])
    result = initial_features.clone()
    mapped, audit = realize_probabilities(initial_features[active], weight, bias, targets[active])
    result[active] = mapped
    if not torch.equal(result[~active], initial_features[~active]):
        raise ValueError('Zero responsibility features changed')
    return result, targets, active, audit


def scores(cm):
    union = cm.sum(0)+cm.sum(1)-np.diag(cm)
    iou = np.divide(np.diag(cm), union, out=np.zeros(5), where=union > 0)
    return {'confusion_matrix': cm.tolist(), 'iou': [float(v) if u else None for v, u in zip(iou, union)],
            'miou_all': float(iou[union > 0].mean()), 'miou_foreground': float(iou[1:][union[1:] > 0].mean())}


def metric_row(rendered, capture, target, valid, weights, loss_value):
    """Descriptive normalized CE/CM alongside the distinct primary raw objective."""
    with torch.no_grad():
        keep = valid.bool() & (target >= 0) & (target < 5)
        labels = target[keep].long()
        raw_gt = capture['raw'].detach()[keep].double().gather(1, labels[:, None])[:, 0]
        noise_fraction = (SPEC['delta']/5)/((1-SPEC['delta'])*raw_gt+SPEC['delta']/5)
        out = {'noise_ce': loss_value, 'valid_pixels': int(keep.sum()),
               'zero_alpha_foreground_pixels': int(((capture['alpha'][keep] == 0) & (labels != 0)).sum()),
               'noise_fraction_mean': float(noise_fraction.mean()), 'noise_fraction_max': float(noise_fraction.max()),
               'noise_dominant_pixels': int((noise_fraction >= .5).sum()),
               'raw_min': float(capture['raw'].detach().min()), 'raw_max': float(capture['raw'].detach().max())}
        for key, result_key in (('raw', 'p3d'), ('final', 'probabilities')):
            p = rendered[result_key].detach()[keep]
            cm = torch.bincount(labels*5+p.argmax(-1), minlength=25).reshape(5, 5)
            out[key+'_cm'] = cm.cpu().numpy()
            true = p.double().gather(1, labels[:, None])[:, 0]
            out[key+'_normalized_ce'] = float(-(true.clamp_min(1e-7).log()*weights.double()[labels]).mean())
        return out


def aggregate(rows):
    out = {key: float(np.mean([r[key] for r in rows]))
           for key in ('noise_ce', 'raw_normalized_ce', 'final_normalized_ce', 'noise_fraction_mean')}
    for key in ('valid_pixels', 'zero_alpha_foreground_pixels', 'noise_dominant_pixels'):
        out[key] = sum(r[key] for r in rows)
    out.update(noise_fraction_max=max(r['noise_fraction_max'] for r in rows),
               raw_min=min(r['raw_min'] for r in rows), raw_max=max(r['raw_max'] for r in rows), views=len(rows))
    for key in ('raw', 'final'):
        out[key] = scores(sum((r[key+'_cm'] for r in rows), np.zeros((5, 5), np.int64)))
    return out


def prepare(root, output):
    from bridge_rgs.coordinates import pixel_protocol

    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(Path('/mnt/data')):
        raise ValueError('Use a new data-disk directory')
    checkpoint = Path('/mnt/data/SHM2026/runs/ssim_fixed_corner_v2_semantic_coupled/last.pt')
    manifest_path = root/'artifacts/prepared_corner_v2/manifest.json'
    if digest(checkpoint) != BASE_SHA or digest(manifest_path) != MANIFEST_SHA:
        raise ValueError('Fixed source checkpoint/manifest differs')
    manifest = json.loads(manifest_path.read_text())
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    train = sorted((v for v in manifest['views'] if v['split'] == 'train'), key=lambda v: v['name'])
    labeled = [v for v in train if v.get('mask_path')]
    if (len(train) != 350 or len(labeled) != 259 or len({v['name'] for v in train}) != 350
            or pixel_protocol(state) != SPEC['pixel_protocol'] or pixel_protocol(manifest) != SPEC['pixel_protocol']
            or state['manifest_sha256'] != MANIFEST_SHA
            or state['config'].get('raw_class_weight_power') != .25
            or state['config'].get('class_weight_power') != .25 or state['config'].get('class_weights') is not None
            or not torch.equal(state['training_cameras'], torch.tensor([v['w2c'] for v in train], dtype=torch.float32))):
        raise ValueError('TRAIN population/camera/profile/weights contract differs')
    order = np.random.default_rng(SPEC['seed']).permutation(len(labeled))
    views = [labeled[i] for i in order]
    inputs = {str(checkpoint): BASE_SHA, str(manifest_path): MANIFEST_SHA, str(root/'uv.lock'): digest(root/'uv.lock')}
    gsplat_root = Path(importlib.util.find_spec('gsplat').origin).parent
    for relative in ('rendering.py', 'cuda/_wrapper.py', 'cuda/csrc/RasterizeToPixels3DGSFwd.cu',
                     'cuda/csrc/RasterizeToPixels3DGSBwd.cu'):
        path = gsplat_root/relative
        inputs[str(path)] = digest(path)
    counts = np.zeros(5, np.int64)
    for v in labeled:
        if (v['width'], v['height']) != (1320, 989) or not v.get('valid_path'):
            raise ValueError('Native TRAIN mask/valid required')
        # No real RGB file is opened, hashed, or supplied to the predictor.
        for key in ('mask_path', 'valid_path'):
            inputs[v[key]] = digest(v[key])
        mask = cv2.imread(v['mask_path'], cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape != (989, 1320):
            raise ValueError('Invalid TRAIN mask grid')
        counts += np.bincount(mask[mask < 5], minlength=5)
    weights = np.maximum(counts/counts.sum(), .002)**(-.25)
    weights = np.minimum(weights/weights.mean(), 3).astype(np.float32)
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    old_package = checkpoint.parent/'source_snapshot/bridge_rgs'
    shutil.copytree(old_package, snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    parent_sources = {str(p.relative_to(old_package)): digest(p) for p in old_package.rglob('*.py')}
    shutil.copy2(root/'src/bridge_rgs/semantic_assignment.py', snapshot/'bridge_rgs/semantic_assignment.py')
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    shutil.copy2(root/'docs/raw_semantic_assignment_proposal.md', output/'proposal_at_lock.md')
    inputs[str(output/'proposal_at_lock.md')] = digest(output/'proposal_at_lock.md')
    plan = {'status': 'cpu_locked_waiting_separate_gpu_authorization', 'specification': SPEC,
            'checkpoint': str(checkpoint), 'manifest': str(manifest_path), 'output': str(output),
            'source_snapshot': str(snapshot), 'input_hashes': inputs,
            'source_hashes': {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')},
            'parent_source_package': str(old_package), 'parent_package_hashes': parent_sources,
            'views': views, 'ordered_names': [v['name'] for v in views],
            'training_camera_names': [v['name'] for v in train],
            'training_cameras_sha256': tensor_hash(state['training_cameras']),
            'train_class_counts': counts.tolist(), 'class_weights': weights.tolist(),
            'weight_source': 'Original259 TRAIN mask counts, same old train.py formula; valid used by objective, not class counts',
            'storage_budget_bytes': 400*1024**2, 'free_at_prepare_bytes': shutil.disk_usage(output).free}
    if plan['free_at_prepare_bytes'] < plan['storage_budget_bytes']:
        raise ValueError('Insufficient disk budget')
    write_json(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    snapshot = Path(plan['source_snapshot'])
    if plan['specification'] != SPEC or Path(__file__).resolve() != snapshot/Path(__file__).name:
        raise ValueError('Use the frozen entrypoint and fixed specification')
    if {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')} != plan['source_hashes']:
        raise ValueError('Frozen source changed')
    for name in ('train', 'model', 'coordinates', 'refinement', 'semantic_assignment'):
        if Path(importlib.import_module('bridge_rgs.'+name).__file__).resolve() != snapshot/'bridge_rgs'/f'{name}.py':
            raise ValueError('Wrong package imported')
    for relative, sha in plan['parent_package_hashes'].items():
        if digest(snapshot/'bridge_rgs'/relative) != sha:
            raise ValueError('Existing package must be byte-identical to391a package')
    for path, sha in plan['input_hashes'].items():
        if digest(path) != sha:
            raise ValueError(f'Bound input changed: {path}')


def q_audit(actual, target=None):
    a = np.asarray(actual, np.float64)
    out = {'min_q': float(a.min()), 'components_below_epsilon': int((a < SPEC['epsilon']).sum()),
           'zero_components': int((a == 0).sum()), 'row_sum_max_error': float(abs(a.sum(-1)-1).max())}
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Actual softmax not finite/nonnegative')
    if target is not None:
        out['target_min_q'] = float(np.min(target))
        out['target_max_absolute_error'] = float(abs(a-target).max(initial=0))
        if out['target_max_absolute_error'] > SPEC['probability_realization_atol']:
            raise ValueError('Fixed feature realization tolerance failed')
    return out


def execute(plan_path):
    plan_path = Path(plan_path).resolve(); plan = json.loads(plan_path.read_text()); verify(plan)
    output = Path(plan['output']); receipt_path = output/'execution_receipt.json'
    receipt = {'status': 'running', 'plan_sha256': digest(plan_path), 'specification': SPEC,
               'scene_calls': 0, 'raster_calls': 0, 'color_backwards': 0, 'adam_updates': 0, 'm_steps': 0,
               'phases': {}, 'production_checkpoint_written': False}
    write_json(receipt_path, receipt)
    start = time.monotonic(); scene = None; original = None; before = None; flags = None
    try:
        import gsplat

        from bridge_rgs.train import load_scene

        torch.set_num_threads(8); cv2.setNumThreads(8)
        scene, state = load_scene(plan['checkpoint'])
        flags = {k: p.requires_grad for k, p in scene.named_parameters()}
        training_flags = {k: module.training for k, module in scene.named_modules()}
        scene.eval().requires_grad_(False)
        parameter = scene.splats['sem_features']; original = parameter.detach().cpu().clone()
        before = {k: tensor_hash(v) for k, v in scene.state_dict().items()}
        camera_hash = tensor_hash(state['training_cameras'])
        if camera_hash != plan['training_cameras_sha256']:
            raise ValueError('TRAIN camera tensor changed')
        fixed_state = semantic_state(scene)
        W = scene.semantic_decoder.weight.detach().cpu(); b = scene.semantic_decoder.bias.detach().cpu()
        with torch.no_grad():
            base_q = scene.semantic_decoder(parameter).softmax(-1).cpu().numpy()
        norm_q = base_q.astype(np.float64); norm_q /= norm_q.sum(-1, keepdims=True)
        target_q0 = (1-5*SPEC['epsilon'])*norm_q+SPEC['epsilon']
        initial, initialization = realize_probabilities(original, W, b, target_q0)
        with torch.no_grad():
            parameter.copy_(initial.cuda())
            qold = scene.semantic_decoder(parameter).softmax(-1).detach()
        old_numpy = qold.cpu().numpy()
        initial_hash = tensor_hash(parameter); qold_hash = tensor_hash(qold)
        receipt['initialization'] = dict(initialization, actual=q_audit(old_numpy, target_q0),
            initial_features_sha256=initial_hash, actual_qold_sha256=qold_hash,
            base_to_normalized_max_abs=float(abs(norm_q-base_q).max()),
            base_to_actual_q_max_abs=float(abs(old_numpy-base_q).max()),
            full_base_objective_measured=False)
        np.savez(output/'shared_initialization.npz', features=initial.numpy(), base_q=base_q,
                 target_q0=target_q0, actual_qold=old_numpy)
        weights = torch.tensor(plan['class_weights'], dtype=torch.float32, device='cuda')
        count_accumulator = np.zeros_like(old_numpy, dtype=np.float64)
        rgb_hashes = {}; trace_digest = hashlib.sha256(); optimizer = None
        trace_path = output/'view_trace.jsonl'
        with trace_path.open('x') as trace:
            for phase in SPEC['phases']:
                if phase == 'em_endpoint':
                    counts = count_accumulator*old_numpy.astype(np.float64)/SPEC['views']
                    if not np.isfinite(counts).all() or (counts < 0).any():
                        raise ValueError('EM counts invalid; no clipping or rescue')
                    candidate, targets, active, mapping = em_feature_candidate(initial, W, b, counts, old_numpy)
                    with torch.no_grad():
                        parameter.copy_(candidate.cuda())
                        realized = scene.semantic_decoder(parameter).softmax(-1).cpu().numpy()
                    active_q = q_audit(realized[active], targets[active]) if active.any() else None
                    if not torch.equal(parameter.detach().cpu()[~active], initial[~active]):
                        raise ValueError('Inactive features changed')
                    receipt['em_candidate'] = dict(mapping, active_rows=int(active.sum()), zero_rows=int((~active).sum()),
                        actual=q_audit(realized), active_realization=active_q, zero_rows_features_exact=True)
                    receipt['m_steps'] += 1
                    np.savez(output/'em_candidate.npz', features=candidate.numpy(), targets=targets, counts=counts,
                             active_rows=active, actual_probabilities=realized)
                if phase == 'adam_sweep':
                    with torch.no_grad():
                        parameter.copy_(initial.cuda())
                        check_q = scene.semantic_decoder(parameter).softmax(-1)
                    if tensor_hash(parameter) != initial_hash or tensor_hash(check_q) != qold_hash:
                        raise ValueError('Adam shared initialization differs')
                    parameter.requires_grad_(True)
                    optimizer = torch.optim.Adam([parameter], **{**SPEC['adam'], 'betas': tuple(SPEC['adam']['betas'])})
                    if optimizer.state:
                        raise ValueError('Adam must start fresh')
                    receipt['adam_initialization'] = {'features_sha256': initial_hash, 'qold_sha256': qold_hash, 'fresh_state': True}
                if phase == 'adam_endpoint':
                    parameter.requires_grad_(False)
                    with torch.no_grad():
                        actual = scene.semantic_decoder(parameter).softmax(-1).cpu().numpy()
                    receipt['adam_candidate'] = q_audit(actual)
                    np.savez(output/'adam_candidate.npz', features=parameter.detach().cpu().numpy(), actual_probabilities=actual)
                torch.cuda.synchronize(); phase_start = time.monotonic(); rows = []; observed = []
                for index, view in enumerate(plan['views']):
                    if receipt['scene_calls'] >= SPEC['scene_calls'] or time.monotonic()-start > SPEC['timeout_seconds']:
                        raise RuntimeError('Fixed call/time budget exhausted')
                    camera_index = plan['training_camera_names'].index(view['name'])
                    pose = state['training_cameras'][camera_index].cuda()
                    K = torch.tensor(view['K'], dtype=torch.float32, device='cuda')
                    with (nullcontext() if phase == 'adam_sweep' else torch.no_grad()), capture_raw(gsplat, phase == 'initial_em_e') as capture:
                        rendered = scene.render(K, pose, 1320, 989, degree=3, semantics=True,
                                                refine=phase != 'adam_sweep', absgrad=False)
                    receipt['scene_calls'] += 1; receipt['raster_calls'] += capture['calls']
                    if capture['calls'] != 2 or capture['semantic_calls'] != 1:
                        raise ValueError('Unexpected rasterization call count')
                    # Only TRAIN targets are decoded. No real RGB or VAL pixels.
                    mask = cv2.imread(view['mask_path'], cv2.IMREAD_UNCHANGED)
                    valid_np = cv2.imread(view['valid_path'], cv2.IMREAD_GRAYSCALE)
                    if mask is None or valid_np is None or mask.shape != (989, 1320) or valid_np.shape != mask.shape:
                        raise ValueError('TRAIN target/valid grid differs')
                    target = torch.as_tensor(mask.astype(np.int64), device='cuda')
                    valid = torch.as_tensor(valid_np > 0, device='cuda')
                    loss = affine_raw_ce(capture['raw'], target, valid, weights, SPEC['delta'])
                    loss_value = float(loss.detach())
                    if phase == 'initial_em_e':
                        if not torch.equal(capture['colors'][:, :5].detach(), qold):
                            raise ValueError('E-pass actual qold changed')
                        grad = torch.autograd.grad(loss, capture['colors'])[0]
                        if not bool(torch.isfinite(grad).all()) or bool((grad[:, 5:] != 0).any()) or bool((grad[:, :5] > 0).any()):
                            raise ValueError('Unexpected EM gradient sign or auxiliary channel gradient')
                        count_accumulator += -grad[:, :5].detach().cpu().numpy().astype(np.float64)
                        receipt['color_backwards'] += 1
                    if phase == 'adam_sweep':
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                            raise ValueError('Feature Adam gradient missing/nonfinite')
                        if any(p.grad is not None for name, p in scene.named_parameters() if name != 'splats.sem_features'):
                            raise ValueError('Frozen parameter has gradient')
                        optimizer.step(); optimizer.zero_grad(set_to_none=True)
                        if not bool(torch.isfinite(parameter).all()):
                            raise ValueError('Nonfinite feature after Adam')
                        receipt['adam_updates'] += 1
                    else:
                        rows.append(metric_row(rendered, capture, target, valid, weights, loss_value))
                    rgb_hash = tensor_hash(rendered['rgb'])
                    if phase == 'initial_em_e':
                        rgb_hashes[view['name']] = rgb_hash
                    elif rgb_hash != rgb_hashes[view['name']]:
                        raise ValueError('RGB changed despite frozen field')
                    observed.append(view['name'])
                    record = {'phase': phase, 'index': index, 'name': view['name'], 'camera_index': camera_index,
                              'scene_call': receipt['scene_calls'], 'noise_ce': loss_value, 'rgb_sha256': rgb_hash}
                    line = (json.dumps(record, sort_keys=True, allow_nan=False)+'\n').encode()
                    trace.write(line.decode()); trace.flush(); trace_digest.update(line)
                    del rendered, capture, loss, target, valid, pose, K
                torch.cuda.synchronize()
                if observed != plan['ordered_names'] or semantic_state(scene) != fixed_state or tensor_hash(state['training_cameras']) != camera_hash:
                    raise ValueError('Order, frozen field, or cameras changed')
                receipt['phases'][phase] = {'wall_seconds': time.monotonic()-phase_start, 'ordered_names': observed,
                                           'metrics': aggregate(rows) if rows else None,
                                           'features_sha256': tensor_hash(parameter), 'nonfeature_state_exact': True}
                write_json(receipt_path, receipt, replace=True)
        for key in ('scene_calls', 'raster_calls', 'color_backwards', 'adam_updates', 'm_steps'):
            if receipt[key] != SPEC[key]:
                raise ValueError(f'Final budget differs: {key}')
        initial_ce = receipt['phases']['initial_em_e']['metrics']['noise_ce']
        final_ce = receipt['phases']['em_endpoint']['metrics']['noise_ce']
        tolerance = max(SPEC['em_decrease_absolute_tolerance'], SPEC['em_decrease_relative_tolerance']*initial_ce)
        receipt['em_actual_objective_check'] = {'initial_noise_ce': initial_ce, 'endpoint_noise_ce': final_ce,
            'decrease': initial_ce-final_ce, 'fixed_tolerance': tolerance,
            'decision': 'numerical_decrease_only_no_model_adoption' if initial_ce-final_ce > tolerance else 'unresolved_not_adopted_no_retry'}
        receipt['trace_sha256'] = trace_digest.hexdigest()
        if receipt['trace_sha256'] != digest(trace_path):
            raise ValueError('Trace digest differs')
        receipt['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        receipt['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
        verify(plan)
        receipt['status'] = 'completed'
    except Exception as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        if scene is not None and original is not None:
            with torch.no_grad():
                scene.splats['sem_features'].copy_(original.to(scene.splats['sem_features'].device))
            for name, parameter in scene.named_parameters():
                parameter.grad = None
                parameter.requires_grad_(flags[name])
            for name, module in scene.named_modules():
                module.training = training_flags[name]
            receipt['finally_flags_restored'] = all(p.requires_grad == flags[name] for name, p in scene.named_parameters())
            receipt['finally_all_state_restored_exact'] = before == {k: tensor_hash(v) for k, v in scene.state_dict().items()}
            receipt['finally_camera_exact'] = camera_hash == tensor_hash(state['training_cameras'])
            if not receipt['finally_all_state_restored_exact'] or not receipt['finally_camera_exact']:
                receipt['status'] = 'failed'
        receipt['wall_seconds'] = time.monotonic()-start
        receipt['output_sha256'] = {p.name: digest(p) for p in output.glob('*.npz')}
        write_json(receipt_path, receipt, replace=True)
    if receipt['status'] != 'completed':
        raise RuntimeError('Final state restoration failed')
    return receipt


def main():
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare'); group.add_argument('--execute'); group.add_argument('--verify')
    args = parser.parse_args()
    if args.prepare:
        print(prepare(Path.cwd(), args.prepare))
    elif args.verify:
        verify(json.loads(Path(args.verify).read_text())); print('Frozen CPU import/source/input verification passed')
    else:
        print(json.dumps(execute(args.execute), indent=2))


if __name__ == '__main__':
    main()
