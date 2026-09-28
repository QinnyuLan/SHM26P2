"""Fixed eight TRAIN views: raw semantic probability/mass and four-bias diagnosis.

--prepare is CPU-only. --execute requires a separate root GPU handoff and must
run this same frozen entrypoint with its snapshot package on PYTHONPATH.
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
from scipy.optimize import minimize
from scipy.special import logsumexp

BASE_SHA = '391a0577450f7f458b75cb9e2fcf926953142b516728ea3600ef2ef064d72c13'
SPEC = {'protocol': 'eight_train_raw_probability_mass_bias_v1', 'seed': 20260927,
        'views': 8, 'fit_indices': [0, 2, 4, 6], 'check_indices': [1, 3, 5, 7],
        'width': 1320, 'height': 989, 'degree': 3, 'refine': False,
        'pixel_protocol': 'colmap_corner_v2', 'fit_pixels_per_view': 50000,
        'class_weight_power': .25, 'class_frequency_floor': .002,
        'class_weight_max': 3., 'bias_background': 0., 'bias_bounds': [-6., 6.],
        'fit': 'L-BFGS-B weighted CE; mean over sampled pixels, no IoU objective',
        'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-7,
        'threshold_alpha': .5, 'alpha_boundary_tolerance': 2e-6, 'quantiles': [0., .1, .25, .5, .75, .9, 1.],
        'maximum_scene_renders': 8, 'source_model_updated': False,
        'scope': 'TRAIN descriptive diagnosis only; no VAL, real RGB, adoption or generalization claim'}


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


def fixed_views(manifest):
    names = [v['name'] for v in manifest['views']]
    population = sorted((v for v in manifest['views'] if v['split'] == 'train' and v.get('mask_path')),
                        key=lambda v: v['name'])
    if len(names) != len(set(names)) or len(population) != 259:
        raise ValueError('Expected 259 unique labeled TRAIN views')
    indices = [i * 258 // 7 for i in range(8)]
    return indices, [population[i] for i in indices]


def class_weights(counts):
    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape != (5,) or (counts < 0).any() or counts.sum() <= 0:
        raise ValueError('Invalid TRAIN class counts')
    frequency = counts / counts.sum()
    weights = 1 / np.maximum(frequency, SPEC['class_frequency_floor']) ** SPEC['class_weight_power']
    return np.minimum(weights / weights.mean(), SPEC['class_weight_max']).astype(np.float32)


def objective(bias4, logp, labels, weights):
    """Analytic CPU float64 CE/value+gradient; background fixes the additive gauge."""
    logits = logp + np.r_[0., bias4]
    normalizer = logsumexp(logits, axis=1)
    selected_weight = weights[labels]
    value = np.mean((normalizer - logits[np.arange(len(labels)), labels]) * selected_weight)
    probabilities = np.exp(logits - normalizer[:, None])
    probabilities[np.arange(len(labels)), labels] -= 1
    gradient = np.mean(probabilities * selected_weight[:, None], axis=0)[1:]
    return float(value), gradient


def fit_bias(arrays, weights):
    rng = np.random.default_rng(SPEC['seed'])
    lp, ys, samples = [], [], []
    for name, probabilities, labels, keep in arrays:
        indices = np.flatnonzero(keep)
        if len(indices) == 0:
            raise ValueError('Empty valid fit image')
        if len(indices) > SPEC['fit_pixels_per_view']:
            indices = rng.choice(indices, SPEC['fit_pixels_per_view'], replace=False)
        lp.append(np.log(np.maximum(probabilities.reshape(-1, 5)[indices].astype(np.float64), 1e-7)))
        ys.append(labels.ravel()[indices].astype(np.int64))
        samples.append({'name': name, 'pixels': len(indices),
                        'flat_indices_sha256': hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()})
    logp, labels = np.concatenate(lp), np.concatenate(ys)
    result = minimize(objective, np.zeros(4), args=(logp, labels, weights.astype(np.float64)),
                      method='L-BFGS-B', jac=True, bounds=[tuple(SPEC['bias_bounds'])]*4,
                      options={key: SPEC[key] for key in ('maxiter', 'ftol', 'gtol')})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise ValueError('Nonfinite bias fit')
    return np.r_[0., result.x], {'samples': samples, 'iterations': int(result.nit),
                                'success': bool(result.success), 'status': int(result.status),
                                'message': str(result.message), 'weighted_ce_before': objective(np.zeros(4), logp, labels, weights)[0],
                                'weighted_ce_after': float(result.fun), 'bias': np.r_[0., result.x].tolist()}


def confusion(target, prediction, keep):
    return np.bincount((target[keep]*5 + prediction[keep]).astype(np.int64), minlength=25).reshape(5, 5)


def cm_scores(matrix):
    matrix = np.asarray(matrix, np.int64)
    union = matrix.sum(0)+matrix.sum(1)-np.diag(matrix)
    iou = np.divide(np.diag(matrix), union, out=np.zeros(5, np.float64), where=union > 0)
    return {'confusion_matrix': matrix.tolist(), 'iou': [float(v) if n else None for v, n in zip(iou, union)],
            'miou_all': float(iou[union > 0].mean()) if (union > 0).any() else None,
            'miou_foreground': float(iou[1:][union[1:] > 0].mean()) if (union[1:] > 0).any() else None}


def summarize(probabilities, alpha, labels, keep, bias):
    if probabilities.shape != (*labels.shape, 5) or alpha.shape != labels.shape or keep.shape != labels.shape:
        raise ValueError('Mismatched diagnostic grids')
    if (not np.isfinite(probabilities).all() or not np.isfinite(alpha).all()
            or (probabilities <= 0).any() or not np.allclose(probabilities.sum(-1), 1, atol=2e-6)
            or alpha.min() < -1e-6 or alpha.max() > 1+1e-6):
        raise ValueError('Invalid semantic probability or alpha')
    keep = keep & (labels < 5)
    raw = probabilities.argmax(-1)
    calibrated = (np.log(np.maximum(probabilities.astype(np.float64), 1e-7))+bias).argmax(-1)
    rows = []
    for category in range(5):
        mask = keep & (labels == category)
        n = int(mask.sum())
        rows.append({'class_id': category, 'pixels': n,
                     'semantic_alpha_le_half': int((mask & (alpha <= .5)).sum()),
                     'semantic_alpha_below_half_minus_tolerance': int((mask & (alpha < .5-SPEC['alpha_boundary_tolerance'])).sum()),
                     'semantic_alpha_half_boundary_band': int((mask & (np.abs(alpha-.5) <= SPEC['alpha_boundary_tolerance'])).sum()),
                     'p_true_le_half': int((mask & (probabilities[..., category] <= .5)).sum()),
                     'p_background_gt_p_true': int((mask & (probabilities[..., 0] > probabilities[..., category])).sum()),
                     'p_true_quantiles': np.quantile(probabilities[..., category][mask], SPEC['quantiles']).tolist() if n else None,
                     'alpha_quantiles': np.quantile(alpha[mask], SPEC['quantiles']).tolist() if n else None})
    return {'raw': cm_scores(confusion(labels, raw, keep)),
            'calibrated': cm_scores(confusion(labels, calibrated, keep)), 'by_gt_class': rows,
            'valid_pixels': int(keep.sum())}


@contextmanager
def capture_semantic_alpha(module):
    """Observe the two existing passes; no new pass, altered color, or model edit."""
    original, records = module.rasterization, []

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        records.append({'render_mode': kwargs.get('render_mode', 'RGB'), 'alpha': result[1]})
        return result
    module.rasterization = wrapped
    try:
        yield records
    finally:
        module.rasterization = original


def prepare(root, output):
    from bridge_rgs.coordinates import CORNER, pixel_protocol

    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(Path('/mnt/data')):
        raise ValueError('Use a new /mnt/data output directory')
    checkpoint = Path('/mnt/data/SHM2026/runs/ssim_fixed_corner_v2_semantic_coupled/last.pt')
    manifest_path = root/'artifacts/prepared_corner_v2/manifest.json'
    if digest(checkpoint) != BASE_SHA:
        raise ValueError('Fixed joint checkpoint changed')
    manifest = json.loads(manifest_path.read_text())
    indices, views = fixed_views(manifest)  # Names fixed before any label is decoded.
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    train = sorted((v for v in manifest['views'] if v['split'] == 'train'), key=lambda v: v['name'])
    if (len(train) != 350 or pixel_protocol(state) != CORNER or pixel_protocol(manifest) != CORNER
            or state.get('manifest_sha256') != digest(manifest_path)
            or state['config'].get('class_weight_power') != .25
            or state['config'].get('raw_class_weight_power') != .25
            or state['config'].get('class_weights') is not None
            or not torch.equal(state['training_cameras'], torch.tensor([v['w2c'] for v in train], dtype=torch.float32))):
        raise ValueError('Fixed checkpoint/profile/TRAIN camera or class weighting contract differs')
    counts = np.zeros(5, np.int64)
    inputs = {str(checkpoint): BASE_SHA, str(manifest_path): digest(manifest_path),
              str(root/'uv.lock'): digest(root/'uv.lock')}
    for view in train:
        if view.get('mask_path'):
            path = Path(view['mask_path'])
            inputs[str(path)] = digest(path)
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != (989, 1320):
                raise ValueError('Invalid native TRAIN label grid')
            counts += np.bincount(mask[mask < 5], minlength=5)
    for view in views:
        if (view['width'], view['height']) != (1320, 989) or not view.get('valid_path'):
            raise ValueError('Native diagnostic grid and validity required')
        inputs[view['valid_path']] = digest(view['valid_path'])
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    source_package = checkpoint.parent/'source_snapshot/bridge_rgs'
    shutil.copytree(source_package, snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    sources = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob('*.py'))}
    plan = {'status': 'cpu_locked_waiting_gpu_authorization', 'specification': SPEC,
            'root': str(root), 'output': str(output), 'source_snapshot': str(snapshot),
            'checkpoint': str(checkpoint), 'checkpoint_sha256': BASE_SHA,
            'manifest': str(manifest_path), 'sample_indices': indices, 'views': views,
            'sample_names': [v['name'] for v in views], 'training_camera_names': [v['name'] for v in train],
            'training_cameras_sha256': tensor_hash(state['training_cameras']),
            'train_class_counts': counts.tolist(), 'class_weights': class_weights(counts).tolist(),
            'weight_source': 'Original train.py formula: all259 TRAIN mask class counts, no valid weighting; FP32 weights',
            'input_hashes': inputs, 'source_hashes': sources, 'parent_source_package': str(source_package),
            'execution': 'Only after root GPU handoff; 8 native renders then CPU fit. No production parameters written.'}
    write_json(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    if plan['specification'] != SPEC:
        raise ValueError('Fixed specification changed')
    snapshot = Path(plan['source_snapshot'])
    if Path(__file__).resolve() != snapshot/Path(__file__).name:
        raise ValueError('Use the frozen entrypoint')
    actual = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')}
    if actual != plan['source_hashes']:
        raise ValueError('Source changed')
    for name in ('train', 'model', 'coordinates', 'refinement'):
        if Path(importlib.import_module('bridge_rgs.'+name).__file__).resolve() != snapshot/'bridge_rgs'/f'{name}.py':
            raise ValueError('Package was not loaded from this snapshot')
    for path, sha in plan['input_hashes'].items():
        if digest(path) != sha:
            raise ValueError(f'Input changed: {path}')


def execute(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    verify(plan)
    output = Path(plan['output'])
    receipt_path = output/'execution_receipt.json'
    receipt = {'status': 'running', 'plan_sha256': digest(plan_path), 'specification': SPEC}
    write_json(receipt_path, receipt)
    start = time.monotonic()
    try:
        import gsplat

        from bridge_rgs.train import load_scene

        torch.set_num_threads(8)
        cv2.setNumThreads(8)
        scene, state = load_scene(plan['checkpoint'])
        scene.eval().requires_grad_(False)
        before = {k: tensor_hash(v) for k, v in scene.state_dict().items()}
        if tensor_hash(state['training_cameras']) != plan['training_cameras_sha256']:
            raise ValueError('Camera tensor changed')
        records = []
        (output/'arrays').mkdir()
        with torch.inference_mode():
            for index, view in enumerate(plan['views']):
                pose = state['training_cameras'][plan['training_camera_names'].index(view['name'])].cuda()
                K = torch.tensor(view['K'], dtype=torch.float32, device='cuda')
                with capture_semantic_alpha(gsplat) as passes:
                    rendered = scene.render(K, pose, 1320, 989, degree=3, semantics=True,
                                            refine=False, absgrad=False)
                if len(passes) != 2 or [p['render_mode'] for p in passes] != ['RGB+ED', 'RGB']:
                    raise ValueError('Unexpected semantic rendering passes')
                p3d = rendered['p3d'].cpu().numpy()
                alpha = passes[1]['alpha'][0, ..., 0].cpu().numpy()
                alpha_delta = float(np.abs(alpha-rendered['alpha'][..., 0].cpu().numpy()).max())
                path = output/'arrays'/f"{Path(view['name']).stem}.npz"
                np.savez(path, p3d=p3d, semantic_alpha=alpha)
                records.append({'name': view['name'], 'role': 'fit' if index in SPEC['fit_indices'] else 'train_check',
                                'path': str(path), 'sha256': digest(path),
                                'semantic_rgb_alpha_max_abs_diff': alpha_delta})
                del rendered, passes, p3d, alpha
        if before != {k: tensor_hash(v) for k, v in scene.state_dict().items()} or tensor_hash(state['training_cameras']) != plan['training_cameras_sha256']:
            raise ValueError('Frozen scene/cameras changed')
        del scene, state, K, pose
        torch.cuda.empty_cache()
        print('All 8 TRAIN renders complete; model freed. CPU fitting follows.', flush=True)
        # All diagnostic renders have completed before reading these label/valid
        # arrays. Prepare read only TRAIN labels to reconstruct the original weights.
        arrays = []
        for record, view in zip(records, plan['views']):
            values = np.load(record['path'], allow_pickle=False)
            target = cv2.imread(view['mask_path'], cv2.IMREAD_UNCHANGED)
            valid = cv2.imread(view['valid_path'], cv2.IMREAD_GRAYSCALE)
            if target is None or valid is None or target.shape != (989, 1320) or valid.shape != target.shape:
                raise ValueError('TRAIN target/valid grid mismatch')
            arrays.append((view['name'], values['p3d'], target, (valid > 0) & (target < 5), values['semantic_alpha']))
            values.close()
        bias, fitting = fit_bias([v[:4] for i, v in enumerate(arrays) if i in SPEC['fit_indices']], np.asarray(plan['class_weights']))
        per_view = []
        for record, (name, p, labels, keep, alpha) in zip(records, arrays):
            per_view.append({'name': name, 'role': record['role'], 'statistics': summarize(p, alpha, labels, keep, bias)})
        pooled = {}
        for role in ('fit', 'train_check', 'all_eight_train'):
            selected = [v for v in per_view if role == 'all_eight_train' or v['role'] == role]
            pooled[role] = {kind: cm_scores(sum((np.array(v['statistics'][kind]['confusion_matrix']) for v in selected), np.zeros((5, 5), np.int64)))
                            for kind in ('raw', 'calibrated')}
        verify(plan)
        for record in records:
            if digest(record['path']) != record['sha256']:
                raise ValueError('Prediction array changed')
        receipt.update(status='completed', records=records, fitting=fitting, per_view=per_view, pooled=pooled,
                       frozen_model_camera_verified=True, model_tensor_hashes=before, renders=8,
                       limits=['Bias is a standard rendered-field calibration, not a change to Gaussian geometry or a novel method.',
                               'In ideal mixing total semantic alpha <= .5 obstructs an all-Gaussians-foreground assignment; clamp/normalization/FP32 effects are separated with a fixed 2e-6 boundary band. Current p_c is not a cable geometry oracle.',
                               'Check views remain TRAIN views used for model training; they are held out only from bias fitting.',
                               'Positive scalar temperature alone cannot change argmax; no temperature or validation tuning performed.'])
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
    mode.add_argument('--prepare', type=Path, metavar='NEW_OUTPUT')
    mode.add_argument('--execute', type=Path, metavar='PLAN')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.root, args.prepare))
    else:
        result = execute(args.execute)
        print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    main()
