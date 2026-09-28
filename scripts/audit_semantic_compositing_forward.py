"""Approximate VJP proposal, followed by actual frozen-renderer existence check.

CPU prepare only until root GPU handoff. v1 remains unchanged; this distinct
protocol uses row-rescaled VJP only to propose one semantic assignment.
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import time
from pathlib import Path

import audit_semantic_compositing_lp as core
import cv2
import numpy as np
import torch
from scipy import sparse

PARENT_PLAN_SHA = 'd642e3cfd3b5df4c4a61051539990b30e086c9332bb1a4985e7baf7a80d85fad'
SPEC = dict(core.SPEC, protocol='sixteen_train_rescaled_vjp_actual_forward_v2',
            weight_interpretation='VJP is approximate; per-ray alpha/sum(VJP) restores only common scale; original5e-6 reconstruction gate retained',
            certificate_scope='Row-rescaled VJP proxy only; never renderer infeasibility',
            interior_epsilon=1e-5, feature_probability_absolute_tolerance=5e-6,
            actual_forward_positive_margin=1e-4, max_scene_renders=16,
            actual_forward='Original scene.render with only temporary union semantic features changed; same8 cameras, full SH3/refine=False; no checkpoint saved',
            cm_policy='Original8 TRAIN raw CM before/after are descriptive only; no fitting or selection from full CM')


def recover_row(weights, alpha):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not np.isfinite(weights).all() or (weights < 0).any() or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Invalid VJP row or alpha')
    mass = float(weights.sum())
    if (mass == 0) != (alpha == 0):
        raise ValueError('Cannot restore scale with inconsistent zero mass')
    scale = float(alpha/mass) if mass else 1.
    corrected = weights*scale
    return corrected, {'original_vjp_mass': mass, 'forward_alpha': float(alpha),
                       'original_signed_mass_error': mass-alpha, 'common_scale_factor': scale,
                       'corrected_mass_error': float(corrected.sum()-alpha)}


def ray_errors(weights, colors, alpha, raw, p3d):
    reconstructed = np.asarray(weights, np.float64) @ np.asarray(colors, np.float64)
    reconstructed[0] += 1-alpha
    normalized = np.maximum(reconstructed, 1e-7); normalized /= normalized.sum()
    return {'raw_max_absolute_error': float(np.max(abs(reconstructed-raw))),
            'p3d_max_absolute_error': float(np.max(abs(normalized-p3d)))}


def feature_candidate(current, weight, bias, q, epsilon=1e-5):
    """Minimum-norm feature delta for full-rank class differences; CPU FP64."""
    current = torch.as_tensor(current, dtype=torch.float64, device='cpu')
    weight = torch.as_tensor(weight, dtype=torch.float64, device='cpu')
    bias = torch.as_tensor(bias, dtype=torch.float64, device='cpu')
    q = torch.as_tensor(q, dtype=torch.float64, device='cpu')
    if (q.shape != (len(current), 5) or weight.shape != (5, current.shape[1]) or bias.shape != (5,)
            or not all(bool(torch.isfinite(v).all()) for v in (current, weight, bias, q))
            or bool((q < 0).any()) or not torch.allclose(q.sum(-1), torch.ones(len(q), dtype=q.dtype), atol=1e-10, rtol=0)
            or not 0 < epsilon < .2):
        raise ValueError('Invalid feature candidate input')
    D = weight[1:]-weight[0]
    if int(torch.linalg.matrix_rank(D)) != 4:
        raise ValueError('Fixed class-difference decoder must have rank4')
    interior = (1-5*epsilon)*q+epsilon
    target = interior[:, 1:].log()-interior[:, :1].log()
    old = current @ D.T+(bias[1:]-bias[0])
    delta = (target-old) @ torch.linalg.pinv(D).T
    candidate64 = current+delta
    candidate = candidate64.float()
    predicted = (candidate @ weight.float().T+bias.float()).softmax(-1).double()
    error = float((predicted-interior).abs().max()) if len(q) else 0.
    if error > SPEC['feature_probability_absolute_tolerance']:
        raise ValueError('CPU FP32 feature realization exceeds fixed probability tolerance')
    norms = delta.norm(dim=-1).numpy()
    audit = {'epsilon': epsilon, 'decoder_difference_rank': 4,
             'target_interior_min_q': float(interior.min()) if interior.numel() else None,
             'cpu_float32_realized_min_q': float(predicted.min()) if predicted.numel() else None,
             'cpu_float32_probability_max_abs_error': error,
             'delta_l2_min_median_max': np.quantile(norms, [0, .5, 1]).tolist() if len(norms) else None,
             'delta_max_abs': float(delta.abs().max()) if delta.numel() else 0.,
             'delta_rms': float(delta.square().mean().sqrt()) if delta.numel() else 0.,
             'candidate_feature_max_abs': float(candidate.abs().max()) if candidate.numel() else 0.}
    return candidate, interior.numpy(), audit


def margin(probabilities, label):
    p = np.asarray(probabilities, np.float64)
    return float(p[label]-np.max(np.delete(p, label)))


def classify_actual(margins):
    values = np.asarray(margins, np.float64)
    if not np.isfinite(values).all() or len(values) != 16:
        raise ValueError('Exactly sixteen finite actual margins required')
    return ('better_shared_assignment_exhibited_on_fixed16_train_rays'
            if values.min() > SPEC['actual_forward_positive_margin'] else 'unresolved_no_renderer_infeasibility_claim')


def raw_cm(p3d, view):
    target = cv2.imread(view['mask_path'], cv2.IMREAD_UNCHANGED)
    valid = cv2.imread(view['valid_path'], cv2.IMREAD_GRAYSCALE)
    if target is None or valid is None or target.shape != p3d.shape[:2] or valid.shape != target.shape:
        raise ValueError('Fixed TRAIN score grid differs')
    keep = (valid > 0) & (target < 5)
    return np.bincount((target[keep]*5+p3d.argmax(-1)[keep]).astype(np.int64), minlength=25).reshape(5, 5)


def cm_scores(cm):
    cm = np.asarray(cm, np.int64); union = cm.sum(0)+cm.sum(1)-np.diag(cm)
    iou = np.divide(np.diag(cm), union, out=np.zeros(5), where=union > 0)
    return {'confusion_matrix': cm.tolist(), 'iou': [float(x) if u else None for x, u in zip(iou, union)],
            'miou_all': float(iou[union > 0].mean()) if (union > 0).any() else None}


def prepare(output):
    parent = Path('/mnt/data/SHM2026/runs/semantic_compositing_lp_v1')
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(Path('/mnt/data')):
        raise ValueError('Use a new data-disk output directory')
    if core.digest(parent/'plan.json') != PARENT_PLAN_SHA:
        raise ValueError('Fixed16 parent plan changed')
    original = json.loads((parent/'plan.json').read_text())
    failed = json.loads((parent/'execution_receipt.json').read_text())
    if failed['status'] != 'failed' or failed['plan_sha256'] != PARENT_PLAN_SHA:
        raise ValueError('Original failed attempt must remain recorded')
    inputs = dict(original['input_hashes'])
    for p in (parent/'plan.json', parent/'execution_receipt.json', parent/'backward_precision_source_audit.json'):
        inputs[str(p)] = core.digest(p)
    for p, sha in inputs.items():
        if core.digest(p) != sha:
            raise ValueError(f'Bound input changed: {p}')
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(Path(original['source_snapshot']), snapshot)
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    sources = {str(p.relative_to(snapshot)): core.digest(p) for p in snapshot.rglob('*.py')}
    plan = dict(original, specification=SPEC, output=str(output), source_snapshot=str(snapshot),
                source_hashes=sources, input_hashes=inputs,
                parent_plan=str(parent/'plan.json'), parent_plan_sha256=PARENT_PLAN_SHA,
                correction_note='New approximate-proposal plus actual-forward protocol; v1 gate/failure/source preserved. No ray/target change.')
    core.write_json(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    snapshot = Path(plan['source_snapshot'])
    if plan['specification'] != SPEC or Path(__file__).resolve() != snapshot/Path(__file__).name:
        raise ValueError('Use the frozen v2 entrypoint and fixed specification')
    if Path(core.__file__).resolve() != snapshot/'audit_semantic_compositing_lp.py':
        raise ValueError('LP helper must be the unchanged v1 frozen file')
    if {str(p.relative_to(snapshot)): core.digest(p) for p in snapshot.rglob('*.py')} != plan['source_hashes']:
        raise ValueError('Source changed')
    for name in ('train', 'model', 'coordinates', 'refinement'):
        if Path(importlib.import_module('bridge_rgs.'+name).__file__).resolve() != snapshot/'bridge_rgs'/f'{name}.py':
            raise ValueError('Wrong package loaded')
    for path, sha in plan['input_hashes'].items():
        if core.digest(path) != sha:
            raise ValueError(f'Input changed: {path}')


def execute(plan_path):
    plan_path = Path(plan_path).resolve(); plan = json.loads(plan_path.read_text()); verify(plan)
    output = Path(plan['output']); receipt_path = output/'execution_receipt.json'
    receipt = {'status': 'running', 'plan_sha256': core.digest(plan_path), 'specification': SPEC,
               'scene_renders': 0, 'color_backwards': 0, 'rays': []}
    core.write_json(receipt_path, receipt); start = time.monotonic()
    scene = None; original_features = None; union_gpu = None; before = None
    try:
        import gsplat

        from bridge_rgs.train import load_scene

        torch.set_num_threads(8); cv2.setNumThreads(8)
        scene, state = load_scene(plan['checkpoint']); scene.eval().requires_grad_(False)
        before = {k: core.tensor_hash(v) for k, v in scene.state_dict().items()}
        camera_hash = core.tensor_hash(state['training_cameras'])
        if camera_hash != plan['training_cameras_sha256']:
            raise ValueError('Checkpoint TRAIN cameras changed')
        (output/'rows').mkdir()
        rows, ids, raw_data, data, alphas, labels, rgb_hashes, cms = [], [], [], [], [], [], {}, []
        for sample in plan['samples']:
            view = sample['view']; pose = state['training_cameras'][plan['training_camera_names'].index(view['name'])].cuda()
            with torch.no_grad(), core.differentiable_semantic_colors(gsplat) as captured:
                rendered = scene.render(torch.tensor(view['K'], dtype=torch.float32, device='cuda'), pose,
                                        1320, 989, degree=3, semantics=True, refine=False, absgrad=False)
            receipt['scene_renders'] += 1
            if len(captured) != 1 or any(p.requires_grad or p.grad is not None for p in scene.parameters()):
                raise ValueError('Only a single semantic color leaf may receive gradients')
            record = captured[0]; colors, raw, alpha = record['colors'], record['raw'], record['alpha']
            rgb_hashes[view['name']] = core.tensor_hash(rendered['rgb'])
            cms.append({'name': view['name'], 'before': raw_cm(rendered['p3d'].cpu().numpy(), view).tolist()})
            for index, ray in enumerate(sample['rays']):
                y, x = ray['y'], ray['x']
                grad = torch.autograd.grad(raw[0, y, x, 0], colors, retain_graph=index == 0)[0]
                receipt['color_backwards'] += 1
                if not bool(torch.isfinite(grad).all()) or bool((grad[:, 1:] != 0).any()):
                    raise ValueError('Invalid color-only derivative')
                W = grad[:, 0].detach().cpu().numpy(); P = colors[:, :5].detach().cpu().numpy()
                raw_pixel = raw[0, y, x, :5].detach().cpu().numpy(); p3d = rendered['p3d'][y, x].cpu().numpy()
                a = float(alpha[0, y, x, 0]); corrected, recovery = recover_row(W, a)
                positive = np.flatnonzero(W > 0); row = len(receipt['rays'])
                row_path = output/'rows'/f'{row:02d}.npz'
                np.savez(row_path, gaussian_ids=positive, raw_vjp=W[positive], corrected_W=corrected[positive],
                         probabilities=P[positive], alpha=a, raw=raw_pixel, p3d=p3d)
                audit = {'name': view['name'], **ray, 'row_file': str(row_path), 'row_sha256': core.digest(row_path),
                         'recovery': recovery, 'uncorrected_errors': ray_errors(W, P, a, raw_pixel, p3d),
                         'corrected_errors': ray_errors(corrected, P, a, raw_pixel, p3d)}
                receipt['rays'].append(audit)
                core.write_json(receipt_path, receipt, replace=True)  # Preserve the failing ray if any prerequisite rejects.
                audit['prerequisite'] = core.verify_ray(corrected, P, a, raw_pixel, p3d)
                previous = float(np.max(abs(p3d-np.asarray(ray['baseline_p3d']))))
                if previous > core.SPEC['reconstruction_absolute_tolerance']:
                    raise ValueError('Original selected p3d changed')
                audit['previous_p3d_max_abs_difference'] = previous
                rows.extend([row]*len(positive)); ids.extend(positive.tolist())
                raw_data.extend(W[positive].tolist()); data.extend(corrected[positive].tolist())
                alphas.append(a); labels.append(ray['target']); del grad
            del captured, record, colors, raw, alpha, rendered, pose
        union = np.unique(ids); columns = np.searchsorted(union, ids)
        W = sparse.csr_matrix((data, (rows, columns)), shape=(16, len(union)), dtype=np.float64)
        Wraw = sparse.csr_matrix((raw_data, (rows, columns)), shape=W.shape, dtype=np.float64)
        sparse.save_npz(output/'corrected_proxy_weights.npz', W); sparse.save_npz(output/'raw_vjp_weights.npz', Wraw)
        report, q, dual = core.solve_lp(W, alphas, labels)
        report['certificate']['scope'] = 'Row-rescaled VJP proxy only; not original-renderer infeasibility'
        receipt.update(proxy_lp=report, row_overlap=core.row_overlap(W), unique_gaussians=len(union))
        if not report['success']:
            raise ValueError('LP failed to generate the one prescribed candidate; no alternate solver/retry')
        union_gpu = torch.tensor(union, dtype=torch.long, device='cuda')
        original_features = scene.splats['sem_features'][union_gpu].detach().clone()
        candidate, interior, feature_audit = feature_candidate(original_features.cpu(), scene.semantic_decoder.weight.cpu(),
                                                               scene.semantic_decoder.bias.cpu(), q, SPEC['interior_epsilon'])
        np.savez(output/'candidate.npz', gaussian_ids=union, q=q, q_interior=interior,
                 features=candidate.numpy(), lambda_normalized=dual)
        with torch.no_grad():
            scene.splats['sem_features'][union_gpu] = candidate.cuda()
            realized = scene.semantic_decoder(scene.splats['sem_features'][union_gpu]).softmax(-1).cpu().numpy()
        feature_audit['cuda_probability_max_abs_error'] = float(np.max(abs(realized-interior), initial=0))
        feature_audit['cuda_realized_min_q'] = float(realized.min()) if realized.size else None
        receipt['feature_reparameterization'] = feature_audit
        if feature_audit['cuda_probability_max_abs_error'] > SPEC['feature_probability_absolute_tolerance']:
            raise ValueError('Actual decoder did not realize the prescribed interior q')
        unchanged = {k: core.tensor_hash(v) for k, v in scene.state_dict().items() if k != 'splats.sem_features'}
        if unchanged != {k: v for k, v in before.items() if k != 'splats.sem_features'}:
            raise ValueError('Non-feature scene tensors changed')
        actual = []
        with torch.no_grad():
            for sample, cm_record in zip(plan['samples'], cms):
                view = sample['view']; pose = state['training_cameras'][plan['training_camera_names'].index(view['name'])].cuda()
                rendered = scene.render(torch.tensor(view['K'], dtype=torch.float32, device='cuda'), pose,
                                        1320, 989, degree=3, semantics=True, refine=False, absgrad=False)
                receipt['scene_renders'] += 1
                if core.tensor_hash(rendered['rgb']) != rgb_hashes[view['name']]:
                    raise ValueError('Original-renderer RGB changed')
                p3d = rendered['p3d'].cpu().numpy()
                cm_record['after'] = raw_cm(p3d, view).tolist()
                for ray in sample['rays']:
                    p = p3d[ray['y'], ray['x']]
                    actual.append({'name': view['name'], 'role': ray['role'], 'y': ray['y'], 'x': ray['x'],
                                   'target': ray['target'], 'p3d': p.tolist(), 'prediction': int(p.argmax()),
                                   'original_margin': margin(ray['baseline_p3d'], ray['target']),
                                   'actual_margin': margin(p, ray['target'])})
                del rendered, p3d, pose
        if core.tensor_hash(state['training_cameras']) != camera_hash:
            raise ValueError('Original camera tensor changed')
        receipt.update(actual_forward=actual, actual_min_margin=min(v['actual_margin'] for v in actual),
                       actual_conclusion=classify_actual([v['actual_margin'] for v in actual]),
                       native_rgb_exact_all8=True, raw_cm_per_view=cms,
                       raw_cm_all8={key: cm_scores(sum((np.array(v[key]) for v in cms), np.zeros((5, 5), np.int64))) for key in ('before', 'after')})
        verify(plan)
        receipt.update(status='completed', output_hashes={name: core.digest(output/name) for name in ('raw_vjp_weights.npz', 'corrected_proxy_weights.npz', 'candidate.npz')},
                       limits=['Actual positive margin only proves existence on these16 TRAIN probes, not full-image quality/3D truth/generalization.',
                               'Proxy LP bounds cannot prove infeasibility of the real renderer; actual forward failing the fixed1e-4 gate is unresolved.',
                               'No production checkpoint, geometry, RGB, decoder, camera, optimizer or selected model is changed.'])
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if scene is not None and original_features is not None:
            with torch.no_grad():
                scene.splats['sem_features'][union_gpu] = original_features
        if scene is not None and before is not None:
            receipt['all_scene_tensors_restored_exact'] = before == {k: core.tensor_hash(v) for k, v in scene.state_dict().items()}
            if not receipt['all_scene_tensors_restored_exact']:
                receipt.update(status='failed', restoration_error='Scene tensors did not restore exactly')
        receipt['elapsed_seconds'] = time.monotonic()-start
        core.write_json(receipt_path, receipt, replace=True)
    if receipt['status'] != 'completed':
        raise RuntimeError('Diagnostic failed or restoration failed')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path)
    mode.add_argument('--execute', type=Path)
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.prepare))
    else:
        print(json.dumps({'status': execute(args.execute)['status']}))


if __name__ == '__main__':
    main()
