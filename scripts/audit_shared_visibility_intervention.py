"""Fixed 8-TRAIN, forward-only shared-opacity diagnostic. CPU prepare; no retries."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

RUNS = Path('/mnt/data/SHM2026/runs')
DEFAULT_OUTPUT = RUNS / 'shared_visibility_intervention_v1'
BASE_SHA = '391a0577450f7f458b75cb9e2fcf926953142b516728ea3600ef2ef064d72c13'
MANIFEST_SHA = '91b41aeecedc352e4882eb80a1bb635d479ae432251a4c96c85a9dc110cbb327'
INIT_SHA = '544917a1697f9363168ad1632f8f9370f7ff1efa0ef240091501b60e45b158c2'
NAMES = ['002.png', '043.png', '084.png', '125.png', '167.png', '215.png', '257.png', '300.png']
ORDER = ['base', 'minus', 'plus', 'base', 'minus', 'plus']
SPEC = {'protocol': 'shared_visibility_intervention_v1', 'names': NAMES, 'order': ORDER,
        'points': 498136, 'group_points': 3004, 'distance_scene_scale': .01,
        'distance': .16261470794677735, 'logit_shift': math.log(2), 'width': 1320, 'height': 989,
        'degree': 3, 'scene_calls': 48, 'contribution_calls': 8, 'raster_calls': 104,
        'backwards': 0, 'optimizer_steps': 0, 'delta': 5e-7,
        'mass_atol': 5e-6, 'minimum_mass': 64., 'minimum_effective_pixels': 256.,
        'minimum_covered_views': 2, 'joint_fraction': .75, 'minimum_relative_gain': .001,
        'repeat_floor_multiplier': 10., 'fp64_floor_multiplier': 32., 'timeout_seconds': 120,
        'timeout_grace_seconds': 5, 'minimum_free_bytes': 3*1024**3+256*1024**2, 'target_read_policy': 'All48 scene predictions before any TRAIN pixel decode',
        'scope': 'Current fixed assignments only; not physical geometry truth, novelty, or model adoption'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, replace=False):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False)+'\n'
    if replace:
        tmp = path.with_suffix('.tmp'); tmp.write_text(text); tmp.replace(path)
    else:
        with path.open('x') as stream:
            stream.write(text)


def tensor_hash(tensor):
    a = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def tree(path):
    return {str(p.relative_to(path)): digest(p) for p in sorted(Path(path).rglob('*.py'))}


def group_ids(means, poses, initial, distance):
    """Euclidean distances in the original shared world frame; no pixels/errors."""
    p, T = np.asarray(means, np.float64), np.asarray(poses, np.float64)
    cameras = -np.einsum('nji,nj->ni', T[:, :3, :3], T[:, :3, 3])
    near = cKDTree(cameras).query(p, workers=8)[0] < distance
    far = cKDTree(np.asarray(initial, np.float64)).query(p, workers=8)[0] > distance
    return np.flatnonzero(near & far).astype(np.int64)


def shifted_logits(original, ids, case):
    require(case in {'base', 'minus', 'plus'}, 'Unknown intervention')
    result = original.clone()
    if case != 'base':
        result[ids] += (-1 if case == 'minus' else 1)*SPEC['logit_shift']
    return result


def weights_from_counts(counts):
    counts = np.asarray(counts, np.float64)
    require(counts.shape == (5,) and (counts >= 0).all() and counts.sum() > 0, 'Bad class counts')
    weights = np.maximum(counts/counts.sum(), .002)**(-.25)
    return np.minimum(weights/weights.mean(), 3).astype(np.float32)


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists() and output.is_relative_to(Path('/mnt/data')), 'Require new data-disk output')
    origin_path = RUNS/'raw_semantic_assignment_v1/plan.json'
    receipt_path = origin_path.parent/'execution_receipt.json'
    origin, receipt = read(origin_path), read(receipt_path)
    require(receipt['status'] == 'completed' and receipt['plan_sha256'] == digest(origin_path)
            and receipt['finally_all_state_restored_exact'], 'Assignment origin incomplete/unbound')
    require(tree(origin['source_snapshot']) == origin['source_hashes'], 'Origin source changed')
    sample_path = RUNS/'raw_semantic_mass_diagnostic_v1/plan.json'
    sample = read(sample_path)
    require(sample['sample_names'] == NAMES, 'Historical camera sample differs')
    checkpoint, manifest_path = Path(origin['checkpoint']), Path(origin['manifest'])
    init_path = root/'artifacts/prepared_corner_v2/init_points.npz'
    for path, sha in [(checkpoint, BASE_SHA), (manifest_path, MANIFEST_SHA), (init_path, INIT_SHA)]:
        require(digest(path) == sha, 'Fixed input changed: '+str(path))
    manifest = read(manifest_path)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    train = sorted((v for v in manifest['views'] if v['split'] == 'train'), key=lambda v: v['name'])
    require(len(train) == 350 and [v['name'] for v in train] == origin['training_camera_names'], 'Camera population differs')
    require(torch.equal(state['training_cameras'], torch.tensor([v['w2c'] for v in train], dtype=torch.float32)), 'Camera poses differ')
    require(tensor_hash(state['training_cameras']) == origin['training_cameras_sha256'], 'Camera origin hash differs')
    require(state['manifest_sha256'] == MANIFEST_SHA and state['step'] == 8000 and state['sh_degree'] == 3,
            'Expected completed fixed semantic field')
    require(not state.get('mip_filter_config'), 'This diagnostic is for the original unfiltered field')
    require(state['model']['splats.means'].shape[0] == SPEC['points'], 'Point count differs')
    require(float(state['scene_scale'])*.01 == SPEC['distance'], 'Scene scale differs')
    with np.load(init_path) as init:
        ids = group_ids(state['model']['splats.means'].numpy(), state['training_cameras'].numpy(), init['points'], SPEC['distance'])
    require(len(ids) == SPEC['group_points'], 'Fixed geometry group differs; do not retune')
    weights = weights_from_counts(origin['train_class_counts'])
    require(np.array_equal(weights, np.asarray(origin['class_weights'], np.float32))
            and np.array_equal(weights, np.asarray(sample['class_weights'], np.float32)), 'Weight provenance differs')
    views = []
    inputs = {str(p): digest(p) for p in [checkpoint, manifest_path, init_path, origin_path, receipt_path, sample_path, root/'uv.lock']}
    # Reuse the completed assignment's exact installed renderer bindings.
    for path, sha in origin['input_hashes'].items():
        if '/gsplat/' in path or path.endswith('/uv.lock'):
            require(digest(path) == sha, 'Previously bound runtime changed'); inputs[path] = sha
    for name in NAMES:
        index = [v['name'] for v in train].index(name); v = train[index]
        require(v['mask_path'] and v['valid_path'] and (v['width'], v['height']) == (1320, 989), 'Require labeled native TRAIN')
        view = {k: v[k] for k in ('name', 'split', 'width', 'height', 'K')}
        view['camera_index'] = index
        for key in ('image_path', 'mask_path', 'valid_path'):
            path = str((root/Path(v[key])).resolve())
            view[key] = path; inputs[path] = digest(path)
            if key != 'image_path':
                require(inputs[path] == origin['input_hashes'][path], 'Previously bound TRAIN target changed')
        views.append(view)
    require(shutil.disk_usage(output.parent).free >= SPEC['minimum_free_bytes'], 'Insufficient data-disk prediction space')
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(Path(origin['source_snapshot'])/'bridge_rgs', snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    shutil.copy2(root/'docs/next_shared_geometry_hypothesis.md', output/'proposal_at_lock.md')
    shutil.copy2(root/'tests/test_shared_visibility_intervention.py', snapshot/'test_shared_visibility_intervention.py')
    np.save(output/'group_ids.npy', ids, allow_pickle=False)
    for path in [output/'proposal_at_lock.md', snapshot/'test_shared_visibility_intervention.py', output/'group_ids.npy']:
        inputs[str(path)] = digest(path)
    parent_hashes = tree(Path(origin['source_snapshot'])/'bridge_rgs')
    require(tree(snapshot/'bridge_rgs') == parent_hashes, 'Inherited package changed')
    plan = {'status': 'prepared_not_started_waiting_gpu_handoff', 'specification': SPEC,
            'created_utc': datetime.now(UTC).isoformat(), 'output': str(output), 'source_snapshot': str(snapshot),
            'checkpoint': str(checkpoint), 'manifest': str(manifest_path), 'init_points': str(init_path),
            'group_ids': str(output/'group_ids.npy'), 'views': views, 'source_hashes': tree(snapshot),
            'input_hashes': inputs, 'inherited_package_hashes': parent_hashes,
            'training_cameras_sha256': tensor_hash(state['training_cameras']),
            'class_weights': weights.tolist(), 'class_counts': origin['train_class_counts'],
            'weight_source': {'plan': str(origin_path), 'plan_sha256': digest(origin_path),
                              'formula': 'All259 original TRAIN counts; power.25/frequency floor.002/weight cap3; valid not used in counts'},
            'cpu_prepare_pixels_decoded': 0, 'group_selection': 'Geometry-only nearest TRAIN camera <distance and nearest init SfM >distance',
            'execution': 'Single attempt only; no retries, resampling, production checkpoint, backward, optimizer, VAL or teacher'}
    write(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan, imports=False):
    snapshot = Path(plan['source_snapshot'])
    require(plan['specification'] == SPEC and Path(__file__).resolve() == snapshot/Path(__file__).name, 'Use frozen entrypoint/spec')
    require(tree(snapshot) == plan['source_hashes'], 'Frozen source changed')
    require(tree(snapshot/'bridge_rgs') == plan['inherited_package_hashes'], 'Inherited package changed')
    for path, sha in plan['input_hashes'].items():
        require(digest(path) == sha, 'Bound input changed: '+path)
    if imports:
        for name in ('train', 'model', 'coordinates', 'refinement'):
            require(Path(importlib.import_module('bridge_rgs.'+name).__file__).resolve() == snapshot/'bridge_rgs'/f'{name}.py', 'Import escaped frozen package')


@contextmanager
def capture_scene(module):
    original = module.rasterization
    capture = {'calls': 0, 'semantic_calls': 0}
    def wrapped(*args, **kwargs):
        require(not args, 'Expected keyword rasterization')
        capture['calls'] += 1
        result = original(**kwargs)
        if kwargs.get('render_mode', 'RGB') != 'RGB+ED':
            capture['semantic_calls'] += 1
            capture.update(raw=result[0][0, ..., :5], alpha=result[1][0, ..., 0], kwargs=kwargs)
        return result
    module.rasterization = wrapped
    try:
        yield capture
    finally:
        module.rasterization = original


def contribution_arguments(kwargs, ids):
    """Full-scene colors changed only; preserve ordering, footprint and opacity."""
    require(kwargs.get('render_mode', 'RGB') == 'RGB', 'Expected raw semantic raster')
    out = dict(kwargs)
    colors = torch.ones((kwargs['means'].shape[0], 2), dtype=kwargs['means'].dtype, device=kwargs['means'].device)
    colors[:, 0] = 0; colors[ids, 0] = 1
    out.update(colors=colors, backgrounds=colors.new_zeros((1, 2)))
    return out


def verify_mass(rendered, alpha, original_alpha, rgb_alpha):
    mass, total = rendered[..., 0], rendered[..., 1]
    error = float(np.max(np.abs(total.astype(np.float64)-alpha)))
    original_error = float(np.max(np.abs(alpha.astype(np.float64)-original_alpha)))
    rgb_error = float(np.max(np.abs(alpha.astype(np.float64)-rgb_alpha)))
    require(np.isfinite(rendered).all() and np.isfinite(alpha).all() and np.isfinite(original_alpha).all() and np.isfinite(rgb_alpha).all(), 'Nonfinite mass')
    require(error <= SPEC['mass_atol'] and original_error <= SPEC['mass_atol'] and rgb_error <= SPEC['mass_atol']
            and mass.min() >= -SPEC['mass_atol'] and np.max(mass-alpha) <= SPEC['mass_atol'], 'inconclusive_render_contract: full-compositing mass mismatch')
    require(mass.min() >= 0, 'Negative forward mass cannot be silently clipped')
    return mass.copy(), {'full_mass_alpha_max_error': error, 'scene_alpha_max_error': original_error, 'rgb_alpha_max_error': rgb_error,
                         'mass_min': float(mass.min()), 'mass_max': float(mass.max())}


def coverage(mass, keep):
    w = np.asarray(mass, np.float64)[keep]
    total, squared = float(w.sum()), float(np.dot(w, w))
    effective = total*total/squared if squared else 0.
    return {'mass_sum': total, 'effective_pixels': effective,
            'measurable': total >= SPEC['minimum_mass'] and effective >= SPEC['minimum_effective_pixels']}


def score_prediction(prediction, mass, rgb, labels, valid, weights):
    keep = valid & (labels >= 0) & (labels < 5)
    require(valid.any() and keep.any(), 'Empty RGB or labeled support')
    rendered, raw = prediction['rgb'], prediction['raw']
    require(np.isfinite(rendered).all() and np.isfinite(raw).all() and (raw >= 0).all(), 'Invalid prediction')
    rgb_error = np.mean((rendered.astype(np.float64)-rgb.astype(np.float64))**2, axis=-1)
    y = labels[keep].astype(np.int64)
    selected = raw[keep].astype(np.float64)[np.arange(len(y)), y]
    ce = -np.log((1-SPEC['delta'])*selected+SPEC['delta']/5)*np.asarray(weights, np.float64)[y]
    w = np.asarray(mass, np.float64)[keep]; denom = float(w.sum())
    predicted = prediction['raw_ids'][keep].astype(np.int64)
    cm = np.bincount(5*y+predicted, minlength=25).reshape(5, 5)
    return {'rgb_full': float(rgb_error[valid].mean()), 'ce_full': float(ce.mean()),
            'rgb_mass': float(np.dot(w, rgb_error[keep])/denom) if denom else None,
            'ce_mass': float(np.dot(w, ce)/denom) if denom else None,
            'cm': cm.tolist(), 'alpha_mean': float(prediction['alpha'].astype(np.float64)[valid].mean())}


def effect(values):
    a = np.asarray(values, np.float64).reshape(2, 3)
    require(np.isfinite(a).all(), 'Nonfinite loss')
    floor = SPEC['repeat_floor_multiplier']*max(float(np.abs(a[0]-a[1]).max()),
        SPEC['fp64_floor_multiplier']*np.finfo(np.float64).eps*max(1., float(np.abs(a).max())))
    means = a.mean(0)
    return {'base': float(means[0]), 'minus': float(means[1]), 'plus': float(means[2]),
            'minus_gain': float(means[0]-means[1]), 'plus_gain': float(means[0]-means[2]),
            'repeat_tolerance': float(floor), 'minus_relative_gain': float((means[0]-means[1])/means[0]) if means[0] > 0 else None}


def cm_scores(cm):
    cm = np.asarray(cm, np.int64); union = cm.sum(0)+cm.sum(1)-np.diag(cm)
    iou = np.divide(np.diag(cm), union, out=np.zeros(5, np.float64), where=union > 0)
    return {'confusion_matrix': cm.tolist(), 'iou': [float(x) if u else None for x, u in zip(iou, union)],
            'miou': float(iou[union > 0].mean()) if (union > 0).any() else None,
            'false_positive': (cm.sum(0)-np.diag(cm)).tolist(), 'false_negative': (cm.sum(1)-np.diag(cm)).tolist()}


def summarize(rows):
    measured = [r for r in rows if r['coverage']['measurable']]
    summaries = {}
    for key in ('rgb_full', 'ce_full', 'rgb_mass', 'ce_mass'):
        chosen = rows if key.endswith('full') else measured
        summaries[key] = effect(np.mean([[p[key] for p in r['predictions']] for r in chosen], axis=0)) if chosen else None
    for r in rows:
        r['effects'] = {key: effect([p[key] for p in r['predictions']]) for key in ('rgb_mass', 'ce_mass')
                        if all(p[key] is not None for p in r['predictions'])}
        r['effects'].update({key: effect([p[key] for p in r['predictions']]) for key in ('rgb_full', 'ce_full')})
    joint = sum(all(r['effects'][k]['minus_gain'] > r['effects'][k]['repeat_tolerance'] for k in ('rgb_mass', 'ce_mass')) for r in measured)
    required_joint = max(2, math.ceil(SPEC['joint_fraction']*len(measured)))
    clauses = {'coverage': len(measured) >= SPEC['minimum_covered_views'], 'joint_views': joint >= required_joint}
    for key in ('rgb_mass', 'ce_mass'):
        e = summaries[key]
        clauses[key+'_gain'] = bool(e and e['minus_gain'] > e['repeat_tolerance'] and e['minus_relative_gain'] is not None
                                      and e['minus_relative_gain'] >= SPEC['minimum_relative_gain'])
        clauses[key+'_direction'] = bool(e and e['minus_gain']-e['plus_gain'] > e['repeat_tolerance'])
    clauses['full_rgb_no_regression'] = bool(summaries['rgb_full']['minus_gain'] >= -summaries['rgb_full']['repeat_tolerance'])
    clauses['full_semantic_improves'] = bool(summaries['ce_full']['minus_gain'] > summaries['ce_full']['repeat_tolerance'])
    decision = ('inconclusive_coverage' if not clauses['coverage'] else
                'necessary_interference_signal_present' if all(clauses.values()) else 'specified_intervention_not_supported')
    return {'decision': decision, 'clauses': clauses, 'covered_views': [r['name'] for r in measured],
            'joint_improving_views': joint, 'required_joint_views': required_joint, 'effects': summaries,
            'pooled_raw': {case: cm_scores(sum((np.asarray(r['predictions'][i]['cm']) for r in rows), np.zeros((5, 5), np.int64)))
                           for i, case in enumerate(ORDER[:3])},
            'cm_scope': 'First independent prediction per state; repeats used for numerical loss floor, never double-count GT'}


@contextmanager
def restore_scene(scene):
    """All tensors/buffers, training modes and parameter flags restored on exceptions."""
    tensors = {k: v.detach().cpu().clone() for k, v in scene.state_dict().items()}
    flags = {k: p.requires_grad for k, p in scene.named_parameters()}
    modes = {k: m.training for k, m in scene.named_modules()}
    report = {'tensor_hashes_before': {k: tensor_hash(v) for k, v in tensors.items()}}
    try:
        scene.eval().requires_grad_(False)
        yield report
    finally:
        scene.load_state_dict(tensors, strict=True)
        for name, parameter in scene.named_parameters():
            parameter.requires_grad_(flags[name])
        for name, module in scene.named_modules():
            module.training = modes[name]
        report['tensor_hashes_after'] = {k: tensor_hash(v) for k, v in scene.state_dict().items()}
        report['all_tensors_restored_exact'] = all(torch.equal(scene.state_dict()[k].detach().cpu(), v) for k, v in tensors.items())
        report['all_flags_restored'] = all(p.requires_grad == flags[k] for k, p in scene.named_parameters())
        report['all_modes_restored'] = all(m.training == modes[k] for k, m in scene.named_modules())
        require(all(report.values()), 'State restoration failed')


def read_targets(view, complete):
    require(complete, 'GT must not be decoded until all predictions finish')
    import cv2
    rgb = cv2.imread(view['image_path'], cv2.IMREAD_COLOR)
    labels = cv2.imread(view['mask_path'], cv2.IMREAD_UNCHANGED)
    valid = cv2.imread(view['valid_path'], cv2.IMREAD_GRAYSCALE)
    require(rgb is not None and labels is not None and valid is not None
            and rgb.shape == (989, 1320, 3) and labels.shape == valid.shape == rgb.shape[:2], 'TRAIN grid differs')
    require(rgb.dtype == labels.dtype == valid.dtype == np.uint8, 'Expected uint8 targets')
    return np.ascontiguousarray(rgb[..., ::-1]).astype(np.float32)/255, labels, valid > 0


def worker(plan_path):
    started = time.monotonic(); plan = read(plan_path); output = Path(plan['output'])
    audit = {'status': 'running', 'plan_sha256': digest(plan_path), 'scene_calls': 0, 'raster_calls': 0,
             'contribution_calls': 0, 'backwards': 0, 'optimizer_steps': 0, 'restoration': {}, 'rows': [], 'prediction_arrays': []}
    def expired(_signum, _frame):
        raise TimeoutError('Fixed worker deadline, no retries')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(1., float(os.environ['PROBE_SECONDS'])-SPEC['timeout_grace_seconds']))
    try:
        verify(plan, imports=True)
        import gsplat

        from bridge_rgs.train import load_scene
        torch.set_num_threads(8)
        scene, state = load_scene(plan['checkpoint'])
        camera_before = tensor_hash(state['training_cameras'])
        require(camera_before == plan['training_cameras_sha256'], 'Loaded cameras differ')
        ids = torch.as_tensor(np.load(plan['group_ids'], allow_pickle=False), device='cuda')
        before_inputs = {path: digest(path) for path in (plan['checkpoint'], plan['manifest'], plan['init_points'])}
        stored = []; trace = []
        predictions_dir = output/'predictions'; predictions_dir.mkdir()
        audit['prediction_arrays'] = []
        with restore_scene(scene) as restoration, torch.no_grad():
            audit['restoration'] = restoration
            parameter = scene.splats['opacity_logits']; original = parameter.detach().clone()
            audit['interventions'] = {}
            for case in ORDER[:3]:
                changed = shifted_logits(original, ids, case)
                opacity = changed[ids].sigmoid(); delta = opacity-original[ids].sigmoid()
                audit['interventions'][case] = {'mean_opacity': float(opacity.mean()), 'sum_opacity': float(opacity.double().sum()),
                    'min_opacity': float(opacity.min()), 'max_opacity': float(opacity.max()),
                    'zero_or_one_fraction': float(((opacity == 0) | (opacity == 1)).float().mean()),
                    'mean_actual_delta': float(delta.double().mean()), 'max_abs_actual_delta': float(delta.abs().max()),
                    'logits_sha256': tensor_hash(changed)}
            for view in plan['views']:
                predictions = []; mass = None; mass_audit = None
                K = torch.tensor(view['K'], dtype=torch.float32, device='cuda')
                pose = state['training_cameras'][view['camera_index']].cuda()
                for index, case in enumerate(ORDER):
                    parameter.copy_(shifted_logits(original, ids, case))
                    with capture_scene(gsplat) as cap:
                        result = scene.render(K, pose, 1320, 989, degree=3, semantics=True, refine=False, absgrad=False)
                    audit['scene_calls'] += 1; audit['raster_calls'] += cap['calls']
                    require(cap['calls'] == 2 and cap['semantic_calls'] == 1, 'Unexpected scene raster budget')
                    require(not torch.is_grad_enabled() and not result['rgb'].requires_grad and not cap['raw'].requires_grad, 'Unexpected autograd graph')
                    pred = {'rgb': result['rgb'].cpu().numpy().copy(), 'raw': cap['raw'].cpu().numpy().copy(),
                            'alpha': cap['alpha'].cpu().numpy().copy(),
                            'raw_ids': result['p3d'].argmax(-1).to(torch.uint8).cpu().numpy().copy(),
                            'rgb_alpha': result['alpha'][..., 0].cpu().numpy().copy()}
                    rgb_alpha = pred['rgb_alpha']
                    alpha_difference = float(np.max(np.abs(rgb_alpha.astype(np.float64)-pred['alpha'])))
                    require(alpha_difference <= SPEC['mass_atol'], 'inconclusive_render_contract: RGB/semantic alpha mismatch')
                    if index == 0:
                        indicator, alpha, _ = gsplat.rasterization(**contribution_arguments(cap['kwargs'], ids))
                        audit['raster_calls'] += 1; audit['contribution_calls'] += 1
                        mass, mass_audit = verify_mass(indicator[0].cpu().numpy(), alpha[0, ..., 0].cpu().numpy(), pred['alpha'], rgb_alpha)
                        for kind, array in [('mass', mass), ('contribution_total', indicator[0, ..., 1].cpu().numpy()),
                                            ('contribution_alpha', alpha[0, ..., 0].cpu().numpy())]:
                            path = predictions_dir/f"{view['name']}.{kind}.npy"
                            np.save(path, array, allow_pickle=False)
                            audit['prediction_arrays'].append({'path': str(path), 'sha256': digest(path), 'shape': list(array.shape), 'dtype': str(array.dtype)})
                        mass = np.load(predictions_dir/f"{view['name']}.mass.npy", mmap_mode='r', allow_pickle=False)
                    paths = {}
                    for key, array in pred.items():
                        path = predictions_dir/f"{view['name']}.{index}.{case}.{key}.npy"
                        np.save(path, array, allow_pickle=False)
                        audit['prediction_arrays'].append({'path': str(path), 'sha256': digest(path), 'shape': list(array.shape), 'dtype': str(array.dtype)})
                        paths[key] = str(path)
                    predictions.append(paths)
                    del pred
                    trace.append({'name': view['name'], 'case': case, 'repeat': index//3,
                                  'scene_calls': audit['scene_calls'], 'raster_calls': audit['raster_calls'],
                                  'rgb_semantic_alpha_max_error': alpha_difference})
                    del cap, result
                stored.append((view, predictions, mass, mass_audit))
            require((audit['scene_calls'], audit['raster_calls'], audit['contribution_calls']) == (48, 104, 8), 'Final render budget differs')
            audit['predictions_finished_utc'] = datetime.now(UTC).isoformat()
            audit['trace'] = trace
            # No TRAIN RGB/mask/valid decoder has been invoked before this boundary.
            audit['scoring_started_utc'] = datetime.now(UTC).isoformat()
            for view, predictions, mass, mass_audit in stored:
                rgb, labels, valid = read_targets(view, complete=audit['scene_calls'] == 48)
                row = {'name': view['name'], 'mass_contract': mass_audit,
                       'coverage': coverage(mass, valid & (labels < 5)),
                       'predictions': [score_prediction({k: np.load(path, mmap_mode='r', allow_pickle=False) for k, path in p.items()},
                                                       mass, rgb, labels, valid, plan['class_weights']) for p in predictions]}
                audit['rows'].append(row)
            audit['summary'] = summarize(audit['rows'])
            audit['no_parameter_gradients'] = all(p.grad is None for p in scene.parameters())
            require(audit['no_parameter_gradients'], 'Unexpected gradient buffer')
            audit['camera_unchanged'] = tensor_hash(state['training_cameras']) == camera_before
            require(audit['camera_unchanged'], 'Camera tensor changed')
        require(all(digest(p) == sha for p, sha in before_inputs.items()), 'Persistent input bytes changed')
        verify(plan)
        audit['status'] = 'completed'
    except TimeoutError as exc:
        audit.update(status='inconclusive_timeout', error=str(exc))
        raise
    except Exception as exc:
        audit.update(status='inconclusive_render_contract' if 'inconclusive_render_contract' in str(exc) else 'failed', error=repr(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        audit['elapsed_seconds'] = time.monotonic()-started
        write(output/'audit.json', audit)
    return audit


def gpu_idle():
    result = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True, check=True)
    root = ET.fromstring(result.stdout); graphics = []
    require(root.findall('gpu'), 'GPU listing unavailable')
    for gpu in root.findall('gpu'):
        processes = gpu.find('processes')
        require(processes is not None and (processes.text or '').strip() in {'', 'None'}, 'GPU process listing unavailable')
        for process in processes.findall('process_info'):
            require(process.findtext('type') == 'G', 'Compute/unknown GPU process active')
            graphics.append({key: process.findtext(key) for key in ('pid', 'type', 'process_name')})
    return graphics


def run(plan_path):
    start = time.monotonic(); plan_path = Path(plan_path).resolve(); plan = read(plan_path)
    output = Path(plan['output']); receipt_path = output/'execution_receipt.json'
    require(not receipt_path.exists() and not (output/'audit.json').exists(), 'One attempt only, no overwrite or retry')
    receipt = {'status': 'preflight', 'plan_sha256': digest(plan_path), 'started_utc': datetime.now(UTC).isoformat()}
    write(receipt_path, receipt)
    try:
        verify(plan)
        receipt['graphics_residents'] = gpu_idle()
        require(shutil.disk_usage(output).free >= SPEC['minimum_free_bytes'], 'Insufficient prediction disk budget')
        seconds = SPEC['timeout_seconds']-(time.monotonic()-start)
        require(seconds > SPEC['timeout_grace_seconds'], 'Preflight exhausted time budget')
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(plan_path)]
        env = dict(os.environ, PYTHONPATH=plan['source_snapshot'], PROBE_SECONDS=str(seconds), PYTHONDONTWRITEBYTECODE='1')
        receipt.update(status='running', command=command, timeout_seconds=seconds, source_hashes=plan['source_hashes'])
        write(receipt_path, receipt, replace=True)
        with (output/'worker.log').open('x') as stream:
            process = subprocess.run(command, cwd=plan['source_snapshot'], env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=seconds, check=False)
        receipt['exit_code'] = process.returncode
        require(process.returncode == 0, 'Worker failed; no retry')
        audit = read(output/'audit.json')
        require(audit['status'] == 'completed' and all(audit['restoration'].values()), 'Worker not complete/restored')
        receipt.update(status='completed', decision=audit['summary']['decision'])
    except subprocess.TimeoutExpired:
        receipt.update(status='inconclusive_timeout', error='Hard120s budget, child killed; no retry', restoration_verified=False)
        raise
    except Exception as exc:
        child = read(output/'audit.json') if (output/'audit.json').exists() else {}
        status = child.get('status')
        receipt.update(status=status if status in {'inconclusive_timeout', 'inconclusive_render_contract'} else 'failed', error=repr(exc))
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-start, finished_utc=datetime.now(UTC).isoformat())
        if (output/'audit.json').exists():
            receipt['audit_sha256'] = digest(output/'audit.json')
        write(receipt_path, receipt, replace=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(); mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', nargs='?', const=str(DEFAULT_OUTPUT)); mode.add_argument('--run')
    mode.add_argument('--worker'); mode.add_argument('--verify')
    args = parser.parse_args()
    if args.prepare:
        print(prepare(Path.cwd(), args.prepare))
    elif args.verify:
        verify(read(args.verify), imports=True); print('CPU source/input/import verification passed')
    elif args.worker:
        worker(args.worker)
    else:
        print(json.dumps(run(args.run), indent=2))


if __name__ == '__main__':
    main()
