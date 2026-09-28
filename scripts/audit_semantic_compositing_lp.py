"""Sixteen fixed TRAIN rays: original-compositor weights and a simplex LP.

Preparation and tests are CPU-only. Frozen --execute requires root GPU handoff.
No scene, decoder, geometry, or production probability is optimized or saved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linprog

BASE_SHA = '391a0577450f7f458b75cb9e2fcf926953142b516728ea3600ef2ef064d72c13'
PARENT_PLAN_SHA = '084cdc53b9c115ba698afae618564730ef13a048382e0ce0dfa24edb3f9cfc37'
SPEC = {'protocol': 'sixteen_train_rays_complete_compositing_simplex_v1',
        'views': 8, 'rays_per_view': 2, 'width': 1320, 'height': 989, 'degree': 3,
        'pixel_protocol': 'colmap_corner_v2', 'refine': False,
        'selection': 'Median row-major valid GT cable/raw-background error; nearest valid GT background/raw-background correct pixel, squared distance ties row-major; no distance cutoff',
        'coordinates': 'Zero-based native array row y and column x; no camera/UV coordinate modification',
        'color_channel_for_weight_derivative': 0, 'sparse_keep': 'Every W_i > 0 exactly; no threshold/top-K',
        'max_scene_renders': 8, 'max_color_backwards': 16, 'scene_parameters_detached': True,
        'reconstruction_absolute_tolerance': 5e-6, 'mass_absolute_tolerance': 5e-6,
        'certificate_absolute_tolerance': 1e-6,
        'lp': 'Independent five-class simplex per contributing Gaussian, shared across all16 rays; maximize common correct-class versus every competitor margin',
        'solver': 'scipy linprog HiGHS; success alone is not a certificate',
        'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9,
        'lp_time_limit_seconds': 60,
        'overlap': 'Every ray pair sum_i min(Wa_i,Wb_i), also divided by smaller row mass; report same-view and cross-view separately',
        'limits': 'Few TRAIN probes only; no VAL/real RGB, scene update, adoption, 3D ground-truth or novelty claim'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def tensor_hash(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


def write_json(path, value, replace=False):
    path = Path(path)
    if replace:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)
    else:
        with path.open('x') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')


def select_rays(probabilities, target, valid):
    if probabilities.shape != (*target.shape, 5) or valid.shape != target.shape:
        raise ValueError('Selection grids differ')
    predicted = probabilities.argmax(-1)
    error = np.argwhere(valid & (target == 2) & (predicted == 0))
    background = np.argwhere(valid & (target == 0) & (predicted == 0))
    if not len(error) or not len(background):
        raise ValueError('Fixed rule requires both cable error and correct background; no fallback')
    anchor = error[len(error)//2]
    squared_distance = ((background.astype(np.int64)-anchor)**2).sum(-1)
    neighbor = background[int(squared_distance.argmin())]
    result = []
    for role, pixel, label in [('cable_error', anchor, 2), ('background_correct', neighbor, 0)]:
        y, x = map(int, pixel)
        interior = cv2.erode(((target == label) & valid).astype(np.uint8), np.ones((7, 7), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0)
        result.append({'role': role, 'y': y, 'x': x, 'target': label, 'baseline_prediction': 0,
                       'gt_boundary3_or_invalid_neighbor': not bool(interior[y, x]),
                       'baseline_p3d': probabilities[y, x].astype(float).tolist()})
    return result, {'cable_error_candidates': len(error), 'background_correct_candidates': len(background),
                    'selected_error_rank': len(error)//2, 'pair_distance_squared': int(squared_distance.min()),
                    'pair_distance_pixels': float(np.sqrt(squared_distance.min()))}


@contextmanager
def differentiable_semantic_colors(module):
    """Detach every renderer input, then attach only the semantic color leaf."""
    original, records = module.rasterization, []

    def wrapped(*args, **kwargs):
        if args:
            raise ValueError('Expected named rasterizer inputs')
        if kwargs.get('render_mode') == 'RGB+ED':
            return original(**kwargs)
        if records or kwargs.get('sh_degree') is not None:
            raise ValueError('Expected one plain semantic color pass')
        detached = {key: value.detach() if isinstance(value, torch.Tensor) else value
                    for key, value in kwargs.items()}
        colors = detached['colors'].requires_grad_(True)
        if colors.ndim != 2 or colors.shape[1] < 5:
            raise ValueError('Expected one semantic color per Gaussian')
        with torch.enable_grad():
            output = original(**detached)
        records.append({'colors': colors, 'raw': output[0], 'alpha': output[1].detach(),
                        'background': detached['backgrounds']})
        return output
    module.rasterization = wrapped
    try:
        yield records
    finally:
        module.rasterization = original


def verify_ray(weights, colors, alpha, raw, p3d):
    """Full float64 reconstruction from all positive FP32 color derivatives."""
    weights, colors, raw, p3d = (np.asarray(x, dtype=np.float64) for x in (weights, colors, raw, p3d))
    if (weights.ndim != 1 or colors.shape != (len(weights), 5) or raw.shape != (5,) or p3d.shape != (5,)
            or not all(np.isfinite(x).all() for x in (weights, colors, raw, p3d))
            or (weights < 0).any() or not np.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError('Invalid nonnegative complete weight/ray contract')
    if (colors < 0).any() or not np.allclose(colors.sum(-1), 1., atol=2e-6):
        raise ValueError('Current Gaussian class probabilities are invalid')
    residual = np.array([1-alpha, 0., 0., 0., 0.])
    reconstructed = weights @ colors + residual
    normalized = np.maximum(reconstructed, 1e-7)
    normalized /= normalized.sum()
    mass_error = abs(weights.sum()-alpha)
    raw_error, p3d_error = float(np.max(abs(reconstructed-raw))), float(np.max(abs(normalized-p3d)))
    if mass_error > SPEC['mass_absolute_tolerance'] or max(raw_error, p3d_error) > SPEC['reconstruction_absolute_tolerance']:
        raise ValueError(f'Compositor reconstruction prerequisite failed: mass={mass_error}, raw={raw_error}, p3d={p3d_error}')
    return {'weights_sum': float(weights.sum()), 'alpha': float(alpha), 'mass_absolute_error': float(mass_error),
            'raw_max_absolute_error': raw_error, 'p3d_max_absolute_error': p3d_error,
            'positive_weights': int((weights > 0).sum()),
            'min_positive_weight': float(weights[weights > 0].min()) if (weights > 0).any() else None,
            'raw': raw.tolist(), 'reconstructed_raw': reconstructed.tolist(), 'p3d': p3d.tolist()}


def build_constraints(weights, alpha, labels):
    """γ + W(q_competitor − q_target) <= fixed residual class difference."""
    weights = sparse.csr_matrix(weights, dtype=np.float64)
    rows, columns, values, rhs, pairs = [], [], [], [], []
    for ray, label in enumerate(labels):
        start, end = weights.indptr[ray:ray+2]
        ids, w = weights.indices[start:end], weights.data[start:end]
        for competitor in range(5):
            if competitor == label:
                continue
            row = len(rhs)
            rows.extend([row]*(2*len(w)+1))
            columns.extend((ids*5+competitor).tolist()+(ids*5+label).tolist()+[weights.shape[1]*5])
            values.extend(w.tolist()+(-w).tolist()+[1.])
            rhs.append((1-alpha[ray])*(int(label == 0)-int(competitor == 0)))
            pairs.append((ray, int(label), competitor))
    A = sparse.csr_matrix((values, (rows, columns)), shape=(len(rhs), weights.shape[1]*5+1))
    N = weights.shape[1]
    eq = sparse.csr_matrix((np.ones(N*5), (np.repeat(np.arange(N), 5), np.arange(N*5))), shape=(N, N*5+1))
    return A, np.asarray(rhs), eq, pairs


def certificate(weights, alpha, labels, q, lambdas):
    """Recompute both bounds using complete W, independently of HiGHS pruning.

    Any normalized nonnegative lambda gives a dual upper bound. q is projected
    onto feasible simplexes. Report the gap, never just the solver objective.
    """
    weights = sparse.csr_matrix(weights, dtype=np.float64)
    q, lambdas = np.asarray(q, np.float64), np.asarray(lambdas, np.float64)
    if q.shape != (weights.shape[1], 5) or lambdas.shape != (len(labels)*4,) or not np.isfinite(q).all() or not np.isfinite(lambdas).all():
        raise ValueError('Invalid candidate primal/dual')
    q = np.maximum(q, 0.)
    sums = q.sum(-1, keepdims=True)
    q[sums[:, 0] == 0] = .2
    q /= q.sum(-1, keepdims=True)
    lambdas = np.maximum(lambdas, 0.)
    if lambdas.sum() == 0:
        lambdas[:] = 1.
    lambdas /= lambdas.sum()
    prediction = np.asarray(weights @ q)
    prediction[:, 0] += 1-alpha
    margins, residual_differences = [], []
    # Dual coefficient for each Gaussian/class, from the original complete W.
    scores = np.zeros((weights.shape[1], 5), np.float64)
    for ray, label in enumerate(labels):
        start, end = weights.indptr[ray:ray+2]
        ids, w = weights.indices[start:end], weights.data[start:end]
        for competitor in range(5):
            if competitor == label:
                continue
            multiplier = lambdas[len(margins)]
            scores[ids, label] += multiplier*w
            scores[ids, competitor] -= multiplier*w
            margins.append(prediction[ray, label]-prediction[ray, competitor])
            residual_differences.append((1-alpha[ray])*(int(label == 0)-int(competitor == 0)))
    lower = float(min(margins))
    upper = float(lambdas @ np.asarray(residual_differences) + scores.max(-1).sum())
    gap = upper-lower
    tolerance = SPEC['certificate_absolute_tolerance']
    if gap < -tolerance:
        raise ValueError('Full-W primal/dual bounds inconsistent')
    conclusion = ('positive_margin_assignment_exists_for_probe' if lower > tolerance else
                  'no_nonnegative_common_margin_within_full_W_relaxation' if upper < -tolerance else
                  'near_zero_or_unresolved_bound_only')
    return {'primal_margin_lower_bound': lower, 'dual_margin_upper_bound': upper, 'upper_minus_lower': gap,
            'floating_absolute_tolerance': tolerance, 'simplex_sum_max_error': float(np.max(abs(q.sum(-1)-1), initial=0)),
            'simplex_min_q': float(np.min(q, initial=0)), 'lambda_sum': float(lambdas.sum()),
            'lambda_min': float(lambdas.min()), 'per_ray_min_margin': np.asarray(margins).reshape(-1, 4).min(-1).tolist(),
            'prediction': prediction.tolist(), 'conclusion': conclusion}, q, lambdas


def solve_lp(weights, alpha, labels):
    weights = sparse.csr_matrix(weights, dtype=np.float64)
    alpha, labels = np.asarray(alpha, np.float64), np.asarray(labels, np.int64)
    if (weights.shape[0] != len(labels) or alpha.shape != labels.shape or len(labels) == 0
            or (weights.data < 0).any() or not np.isfinite(weights.data).all()
            or not np.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any()
            or (labels < 0).any() or (labels > 4).any()):
        raise ValueError('Invalid LP input')
    A, b, eq, pairs = build_constraints(weights, alpha, labels)
    objective = np.zeros(A.shape[1]); objective[-1] = -1.
    result = linprog(objective, A_ub=A, b_ub=b, A_eq=eq if len(eq.indptr) > 1 else None,
                     b_eq=np.ones(weights.shape[1]) if weights.shape[1] else None,
                     bounds=[(0., None)]*(A.shape[1]-1)+[(None, None)], method='highs',
                     options={'primal_feasibility_tolerance': SPEC['primal_feasibility_tolerance'],
                              'dual_feasibility_tolerance': SPEC['dual_feasibility_tolerance'],
                              'time_limit': SPEC['lp_time_limit_seconds']})
    q = result.x[:-1].reshape(-1, 5) if result.x is not None else np.full((weights.shape[1], 5), .2)
    dual = -result.ineqlin.marginals if result.ineqlin.marginals is not None else np.ones(len(pairs))
    bounds, q, dual = certificate(weights, alpha, labels, q, dual)
    report = {'success': bool(result.success), 'status': int(result.status), 'message': str(result.message),
              'iterations': int(result.nit), 'solver_gamma': float(result.x[-1]) if result.x is not None else None,
              'variables': A.shape[1], 'shared_gaussians': weights.shape[1], 'constraints': len(b),
              'complete_weight_nonzeros': int(weights.nnz), 'certificate': bounds,
              'note': 'Bounds are recomputed with every original positive W, including coefficients HiGHS may ignore.'}
    return report, q, dual



def row_overlap(weights):
    weights = sparse.csr_matrix(weights, dtype=np.float64)
    n = weights.shape[0]
    mass = np.asarray(weights.sum(-1)).ravel()
    overlap = np.zeros((n, n))
    normalized = np.zeros((n, n))
    within, across = [], []
    for i in range(n):
        for j in range(i, n):
            value = float(weights[i].minimum(weights[j]).sum())
            overlap[i, j] = overlap[j, i] = value
            denominator = min(mass[i], mass[j])
            normalized[i, j] = normalized[j, i] = value/denominator if denominator > 0 else 0.
            if i != j:
                (within if i//2 == j//2 else across).append((i, j))
    def summarize(pairs):
        values = np.array([overlap[i, j] for i, j in pairs])
        fractions = np.array([normalized[i, j] for i, j in pairs])
        return {'pairs': len(pairs), 'positive_overlap_pairs': int((values > 0).sum()),
                'overlap_min_median_max': np.quantile(values, [0, .5, 1]).tolist() if len(values) else None,
                'normalized_overlap_min_median_max': np.quantile(fractions, [0, .5, 1]).tolist() if len(values) else None}
    return {'row_mass': mass.tolist(), 'overlap_matrix': overlap.tolist(),
            'normalized_by_smaller_row_mass': normalized.tolist(),
            'same_view': summarize(within), 'cross_view': summarize(across),
            'zero_mass_rows': np.flatnonzero(mass == 0).tolist(),
            'interpretation': 'Distinct views with negligible shared W only form a joint16-ray LP, not evidence of strong cross-view constraint conflict.'}


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    parent = Path('/mnt/data/SHM2026/runs/raw_semantic_mass_diagnostic_v2')
    if output.exists() or not output.is_relative_to(Path('/mnt/data')):
        raise ValueError('Use a new data-disk output directory')
    if digest(parent/'plan.json') != PARENT_PLAN_SHA:
        raise ValueError('Parent diagnostic plan changed')
    plan0 = json.loads((parent/'plan.json').read_text())
    receipt = json.loads((parent/'execution_receipt.json').read_text())
    if receipt['status'] != 'completed' or receipt['plan_sha256'] != PARENT_PLAN_SHA or plan0['checkpoint_sha256'] != BASE_SHA:
        raise ValueError('Completed fixed eight-view parent required')
    paths = [parent/'plan.json', parent/'execution_receipt.json', Path(plan0['checkpoint']), Path(plan0['manifest']), root/'uv.lock']
    inputs = {str(p): digest(p) for p in paths}
    if inputs[plan0['checkpoint']] != BASE_SHA:
        raise ValueError('Fixed field changed')
    selected = []
    for view, record in zip(plan0['views'], receipt['records']):
        if view['name'] != record['name'] or view['split'] != 'train' or digest(record['path']) != record['sha256']:
            raise ValueError('Parent TRAIN sample/array changed')
        for key in ('mask_path', 'valid_path'):
            if digest(view[key]) != plan0['input_hashes'][view[key]]:
                raise ValueError('Bound TRAIN labels changed')
            inputs[view[key]] = digest(view[key])
        inputs[record['path']] = record['sha256']
        with np.load(record['path'], allow_pickle=False) as values:
            rays, audit = select_rays(values['p3d'], cv2.imread(view['mask_path'], cv2.IMREAD_UNCHANGED),
                                      cv2.imread(view['valid_path'], cv2.IMREAD_GRAYSCALE) > 0)
        selected.append({'view': view, 'rays': rays, 'selection_audit': audit,
                         'parent_array': record['path'], 'parent_array_sha256': record['sha256']})
    if len(selected) != 8 or any(len(v['rays']) != 2 for v in selected):
        raise ValueError('Expected exactly eight pairs')
    state = torch.load(plan0['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    decoder = state['model']['semantic_decoder.weight'].double()
    class_difference = decoder[1:]-decoder[0]
    rank = int(torch.linalg.matrix_rank(class_difference))
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(Path(plan0['source_snapshot'])/'bridge_rgs', snapshot/'bridge_rgs')
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    sources = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')}
    plan = {'status': 'cpu_locked_waiting_gpu_authorization', 'specification': SPEC,
            'output': str(output), 'source_snapshot': str(snapshot), 'input_hashes': inputs, 'source_hashes': sources,
            'checkpoint': plan0['checkpoint'], 'checkpoint_sha256': BASE_SHA, 'manifest': plan0['manifest'],
            'sample_names': plan0['sample_names'], 'samples': selected,
            'training_camera_names': plan0['training_camera_names'], 'training_cameras_sha256': plan0['training_cameras_sha256'],
            'decoder_class_difference_rank': rank, 'decoder_class_difference_singular_values': torch.linalg.svdvals(class_difference).tolist(),
            'decoder_note': 'Rank4 means free16D per-Gaussian features can represent arbitrary interior class probabilities for this fixed linear decoder; the LP ignores feature/refiner regularization and uses the simplex closure.',
            'parent_plan': str(parent/'plan.json'), 'parent_plan_sha256': PARENT_PLAN_SHA}
    write_json(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    snapshot = Path(plan['source_snapshot'])
    if plan['specification'] != SPEC or Path(__file__).resolve() != snapshot/Path(__file__).name:
        raise ValueError('Use fixed specification and frozen entrypoint')
    sources = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')}
    if sources != plan['source_hashes']:
        raise ValueError('Frozen source changed')
    for name in ('train', 'model', 'coordinates', 'refinement'):
        if Path(importlib.import_module('bridge_rgs.'+name).__file__).resolve() != snapshot/'bridge_rgs'/f'{name}.py':
            raise ValueError('Incorrect imported package')
    for path, sha in plan['input_hashes'].items():
        if digest(path) != sha:
            raise ValueError(f'Input changed: {path}')


def execute(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text()); verify(plan)
    output = Path(plan['output']); receipt_path = output/'execution_receipt.json'
    receipt = {'status': 'running', 'plan_sha256': digest(plan_path), 'specification': SPEC}
    write_json(receipt_path, receipt)
    start = time.monotonic()
    try:
        import gsplat

        from bridge_rgs.train import load_scene

        torch.set_num_threads(8); cv2.setNumThreads(8)
        scene, state = load_scene(plan['checkpoint']); scene.eval().requires_grad_(False)
        before = {k: tensor_hash(v) for k, v in scene.state_dict().items()}
        if tensor_hash(state['training_cameras']) != plan['training_cameras_sha256']:
            raise ValueError('Camera binding changed')
        row_ids, ids, data, alpha_values, targets, audits = [], [], [], [], [], []
        all_ids = []
        for sample in plan['samples']:
            view = sample['view']
            pose = state['training_cameras'][plan['training_camera_names'].index(view['name'])].cuda()
            with torch.no_grad(), differentiable_semantic_colors(gsplat) as captured:
                rendered = scene.render(torch.tensor(view['K'], dtype=torch.float32, device='cuda'),
                                        pose, 1320, 989, degree=3, semantics=True, refine=False, absgrad=False)
            if len(captured) != 1:
                raise ValueError('Expected one captured semantic pass')
            record = captured[0]
            colors = record['colors']; raw = record['raw']; alpha = record['alpha']
            if any(p.requires_grad or p.grad is not None for p in scene.parameters()):
                raise ValueError('Scene parameters must remain completely detached')
            for index, ray in enumerate(sample['rays']):
                y, x = ray['y'], ray['x']
                gradient = torch.autograd.grad(raw[0, y, x, 0], colors, retain_graph=index == 0)[0]
                if not bool(torch.isfinite(gradient).all()) or bool((gradient[:, 1:] != 0).any()):
                    raise ValueError('Color derivative must be finite and confined to selected channel')
                weights = gradient[:, 0].detach().cpu().numpy()
                probabilities = colors[:, :5].detach().cpu().numpy()
                raw_pixel = raw[0, y, x, :5].detach().cpu().numpy()
                p3d = rendered['p3d'][y, x].detach().cpu().numpy()
                a = float(alpha[0, y, x, 0])
                audit = verify_ray(weights, probabilities, a, raw_pixel, p3d)
                baseline_error = float(np.max(abs(p3d-np.asarray(ray['baseline_p3d']))))
                if baseline_error > SPEC['reconstruction_absolute_tolerance']:
                    raise ValueError('Captured forward differs from original selected prediction')
                audit.update(name=view['name'], **ray, previous_p3d_max_abs_difference=baseline_error)
                positive = np.flatnonzero(weights > 0)
                row = len(audits)
                row_ids.extend([row]*len(positive)); ids.extend(positive.tolist()); data.extend(weights[positive].tolist())
                all_ids.append(positive); alpha_values.append(a); targets.append(ray['target']); audits.append(audit)
                del gradient
            del record, captured, colors, raw, alpha, rendered, pose
        if len(audits) != 16 or before != {k: tensor_hash(v) for k, v in scene.state_dict().items()} or tensor_hash(state['training_cameras']) != plan['training_cameras_sha256']:
            raise ValueError('Incomplete probes or mutated scene/cameras')
        del scene, state
        torch.cuda.empty_cache()
        print('Eight TRAIN renders and sixteen color backwards complete; GPU scene freed, CPU LP follows.', flush=True)
        union = np.unique(np.concatenate(all_ids))
        columns = np.searchsorted(union, ids)
        W = sparse.csr_matrix((np.asarray(data, np.float64), (row_ids, columns)), shape=(16, len(union)))
        sparse.save_npz(output/'complete_weights.npz', W)
        np.savez(output/'ray_bindings.npz', gaussian_ids=union, alpha=np.asarray(alpha_values), target=np.asarray(targets))
        report, q, dual = solve_lp(W, alpha_values, targets)
        np.savez(output/'lp_assignment.npz', gaussian_ids=union, q=q, normalized_lambda=dual)
        verify(plan)
        receipt.update(status='completed', scene_renders=8, color_backwards=16, rays=audits,
                       unique_gaussians=len(union), frozen_tensors_verified=True, model_tensor_hashes=before,
                       lp=report, row_overlap=row_overlap(W), output_hashes={name: digest(output/name) for name in ('complete_weights.npz', 'ray_bindings.npz', 'lp_assignment.npz')},
                       limits=['W is complete for the implemented original rasterizer, including its existing early-termination behavior; no top-K or new threshold.',
                               'Positive margin proves only these TRAIN probe labels permit a better shared assignment; it is not a whole-image or 3D-truth result.',
                               'Near-zero or wide primal/dual intervals are not an infeasibility claim; solver success alone is not the certificate.',
                               'Standard alpha compositing and LP diagnostics are not presented as novel.'])
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic()-start
        write_json(receipt_path, receipt, replace=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path)
    mode.add_argument('--execute', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.root, args.prepare))
    else:
        print(json.dumps({'status': execute(args.execute)['status']}))


if __name__ == '__main__':
    main()
