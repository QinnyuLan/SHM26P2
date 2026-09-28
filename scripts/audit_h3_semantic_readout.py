"""Fixed H3 decoder-null readout intervention. Prepare CPU; execute only on authorization."""
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
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

BASE_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
ALGEBRA = Path('/mnt/data/SHM2026/runs/h3_semantic_readout_cpu_audit_v1/report.json')
ALGEBRA_SHA = 'dde4019996075d4db3a81942976b35df746dac866c4173d04cf26d94730fce50'
DOCUMENT = 'h3_semantic_readout_diagnostic.md'
DOC_SHA = '89b87594afb6d758c615cfc20fceeef41bad3b8335fb8420c980c3fad49c9c4c'
FEATURE = 'splats.sem_features'
SPEC = {
    'protocol': 'h3_fixed_decoder_null_readout_four_render_v1',
    'view_names': ['002.png', '205.png'], 'sorted_train_indices': [0, 175],
    'pixel_protocol': 'legacy_mixed_v1', 'width': 1320, 'height': 989, 'sh_degree': 3,
    'states': ['original', 'projected'], 'scene_renders': 4, 'expected_raster_calls': 8,
    'backward_calls': 0, 'optimizer_steps': 0, 'checkpoint_writes': 0,
    'projection': 'FP32(FP64(sem_features) @ fixed CPU-report row_projection)',
    'source_intervention': 'all Gaussian sem_features; recompute ordinary features and cross moments in each render',
    'invariance': {'rgb_depth_alpha_bitwise_equal': True, 'p3d_max_absolute': 1e-6},
    'ce': {'weights': [1., 1., 1., 1., 1.], 'lovasz_weight': 0.,
           'implementation': 'H3 frozen losses.semantic_loss, FP32 probabilities and log clamp_min(1e-7)',
           'reduction': 'sum(valid*class-weighted NLL)/sum(valid*(label!=255)); equal mean of two view CEs',
           'interpretation': 'descriptive unweighted CE, not the original H3 weighted training objective'},
    'pixels': 'only two TRAIN masks/valid, after all four predictions; no real RGB or VAL',
    'deadline_seconds': 105, 'external_timeout_seconds': 120,
    'failed_invariance': 'retain predictions and restore state; no label scoring/dependency interpretation; no retry',
    'interpretation': 'fixed head joint feature/moment dependency only; projection is OOD and removes null mean; not information necessity or a retraining bound',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def fixed_views(manifest):
    require(manifest.get('pixel_protocol', 'legacy_mixed_v1') == 'legacy_mixed_v1', 'Require legacy manifest')
    training = [v for v in manifest['views'] if v['split'] == 'train']
    names = [v['name'] for v in training]
    require(len(names) == len(set(names)) == 350, 'Require 350 distinct TRAIN cameras')
    chosen = [sorted(training, key=lambda v: v['name'])[i] for i in SPEC['sorted_train_indices']]
    require([v['name'] for v in chosen] == SPEC['view_names'], 'Fixed names changed')
    keys = ('name', 'split', 'width', 'height', 'K', 'w2c_original', 'mask_path', 'valid_path')
    rows = [{**{k: v[k] for k in keys}, 'camera_index': names.index(v['name'])} for v in chosen]
    require(all(v['width'] == 1320 and v['height'] == 989 and v['mask_path'] and v['valid_path'] for v in rows),
            'Require native labeled TRAIN views')
    return rows, names


def project_features(features, projection):
    p = torch.tensor(projection, dtype=torch.float64, device='cpu')
    require(p.shape == (16, 16) and features.shape[-1] == 16, 'Fixed projection dimensions')
    require(torch.allclose(p, p.T, atol=1e-14, rtol=0) and torch.allclose(p@p, p, atol=1e-14, rtol=0),
            'Require the fixed orthogonal projector')
    return (features.detach().cpu().double() @ p).float()


def invariance(original, projected):
    equality = {key: tensor_hash(original[key]) == tensor_hash(projected[key]) for key in ('rgb', 'depth', 'alpha')}
    difference = (original['p3d'].double()-projected['p3d'].double()).abs()
    maximum = float(difference.max())
    location = list(np.unravel_index(int(difference.argmax()), tuple(difference.shape)))
    return {'exact': equality, 'p3d_max_absolute': maximum, 'p3d_max_location_y_x_class': [int(x) for x in location],
            'raw_argmax_changes': int((original['p3d'].argmax(-1) != projected['p3d'].argmax(-1)).sum()),
            'passed': bool(all(equality.values()) and maximum <= SPEC['invariance']['p3d_max_absolute'])}


@contextmanager
def preserved_scene(scene, cameras, report):
    before = {k: v.detach().cpu().clone() for k, v in scene.state_dict().items()}
    camera_before = cameras.detach().clone()
    flags = {k: p.requires_grad for k, p in scene.named_parameters()}
    modes = {k: m.training for k, m in scene.named_modules()}
    report['tensor_hashes_before'] = {k: tensor_hash(v) for k, v in before.items()}
    report['camera_hash_before'] = tensor_hash(cameras)
    try:
        scene.eval()
        scene.requires_grad_(False)
        yield before
    finally:
        report['non_feature_tensors_unchanged_before_restore'] = all(
            tensor_hash(v) == report['tensor_hashes_before'][k]
            for k, v in scene.state_dict().items() if k != FEATURE)
        report['cameras_unchanged_before_restore'] = torch.equal(cameras, camera_before)
        with torch.no_grad():
            scene.load_state_dict(before, strict=True)
            cameras.copy_(camera_before)
        for k, p in scene.named_parameters():
            p.requires_grad_(flags[k])
        # Direct assignment preserves heterogeneous child training modes.
        for k, m in scene.named_modules():
            m.training = modes[k]
        after = {k: tensor_hash(v) for k, v in scene.state_dict().items()}
        report['tensor_hashes_after'] = after
        report['all_model_tensors_finally_restored_exact'] = after == report['tensor_hashes_before']
        report['flags_restored_exact'] = flags == {k: p.requires_grad for k, p in scene.named_parameters()}
        report['modes_restored_exact'] = modes == {k: m.training for k, m in scene.named_modules()}
        report['camera_hash_after'] = tensor_hash(cameras)
        report['cameras_restored_exact'] = torch.equal(cameras, camera_before)
        report['no_parameter_gradients'] = all(p.grad is None for p in scene.parameters())


@contextmanager
def pixel_guard(allowed, evidence):
    old = cv2.imread
    allowed = {str(Path(p).resolve()) for p in allowed}
    evidence.update(predictions_completed=False, reads=[])

    def guarded(path, *args, **kwargs):
        path = str(Path(path).resolve())
        require(evidence['predictions_completed'] and path in allowed,
                'Pixel decode before all predictions or outside two TRAIN masks/valid')
        result = old(path, *args, **kwargs)
        evidence['reads'].append(path)
        return result
    cv2.imread = guarded
    try:
        yield
    finally:
        cv2.imread = old


def score_prediction(probabilities, labels, valid):
    from bridge_rgs.losses import confusion_matrix, iou_scores, semantic_loss
    weights = torch.tensor(SPEC['ce']['weights'], dtype=torch.float32)
    cm = confusion_matrix(probabilities, labels, valid)
    return {'ce': float(semantic_loss(probabilities, labels, valid.float(), weights, lovasz_weight=0)),
            'valid_pixels': int((valid & (labels != 255)).sum()),
            'confusion_matrix': cm.tolist(), 'iou': iou_scores(cm)}


def correction_counts(original, projected, labels, valid):
    keep = valid.bool() & (labels < 5)
    raw = original['p3d'].argmax(-1)
    first, second = original['probabilities'].argmax(-1), projected['probabilities'].argmax(-1)
    corrected = keep & (labels == 2) & (raw == 0) & (first == 2)
    return {'original_cable_raw_bg_corrected_by_head': int(corrected.sum()),
            'correction_retained_after_projection': int((corrected & (second == 2)).sum()),
            'correction_lost_after_projection': int((corrected & (second != 2)).sum()),
            'new_cable_corrections': int((keep & (labels == 2) & (first != 2) & (second == 2)).sum()),
            'new_cable_false_positives': int((keep & (labels != 2) & (first != 2) & (second == 2)).sum()),
            'all_class_new_errors': int((keep & (first == labels) & (second != labels)).sum()),
            'all_class_new_corrections': int((keep & (first != labels) & (second == labels)).sum()),
            'final_argmax_changes': int((keep & (first != second)).sum())}


def source_files(snapshot):
    return {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.md'}}


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), 'Preserve existing diagnostic directory')
    require(not torch.cuda.is_initialized(), 'CPU preparation only')
    torch.set_num_threads(8)
    base = root/'runs/h3_moments/02_cross/last.pt'
    origin = base.parent/'source_snapshot'
    receipt_path = base.parent/'experiment_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    require(receipt['status'] == 'completed' and digest(base) == BASE_SHA, 'Require fixed completed H3')
    sources = source_files(origin)
    require(sources == receipt['source_hashes'], 'H3 frozen package differs')
    require(digest(ALGEBRA) == ALGEBRA_SHA and digest(root/'docs'/DOCUMENT) == DOC_SHA, 'CPU evidence/design changed')
    algebra = json.loads(ALGEBRA.read_text())
    manifest = root/'artifacts/prepared/manifest.json'
    require(digest(manifest) == algebra['manifest_sha256'], 'Legacy manifest differs')
    manifest_data = json.loads(manifest.read_text())
    views, names = fixed_views(manifest_data)
    state = torch.load(base, map_location='cpu', mmap=True, weights_only=False)
    cameras = state['training_cameras']
    original = torch.tensor([v['w2c_original'] for v in manifest_data['views'] if v['split'] == 'train'], dtype=torch.float32)
    require(torch.equal(cameras, original), 'H3 cameras differ from original TRAIN metadata')
    require(state['refiner_config']['depth_moments'] == 'cross' and state['feature_dim'] == 16 and state['sh_degree'] == 3,
            'Wrong H3 architecture')
    projected = project_features(state['model'][FEATURE], algebra['row_projection'])
    candidate_hash = tensor_hash(projected)
    require(candidate_hash == algebra['feature_decomposition']['projected_fp32_sha256'], 'Projection differs from prior CPU report')
    paths = [base, manifest, receipt_path, ALGEBRA, root/'docs'/DOCUMENT, root/'uv.lock', root/'pyproject.toml', Path(__file__).resolve()]
    allowed = sorted({v[k] for v in views for k in ('mask_path', 'valid_path')})
    paths += [Path(p) for p in allowed]
    inputs = {str(p): digest(p) for p in paths}
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(origin/'bridge_rgs', snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    shutil.copy2(root/'docs'/DOCUMENT, snapshot/DOCUMENT)
    frozen = source_files(snapshot)
    require(all(frozen[k] == h for k, h in sources.items()), 'Only add diagnostic entry/doc, no H3 changes')
    plan = {'status': 'locked_pending_root_gpu_authorization', 'specification': SPEC,
            'root': str(root), 'output': str(output), 'source_snapshot': str(snapshot),
            'source_hashes': frozen, 'original_h3_source_hashes': sources, 'input_hashes': inputs,
            'base': str(base), 'base_sha256': BASE_SHA, 'manifest': str(manifest),
            'views': views, 'training_camera_names': names, 'training_camera_sha256': tensor_hash(cameras),
            'allowed_pixel_paths': allowed, 'row_projection': algebra['row_projection'],
            'projected_features_sha256': candidate_hash, 'algebra_report_sha256': ALGEBRA_SHA,
            'external_timeout_seconds': 120, 'minimum_free_bytes': 512*1024**2,
            'saved_arrays': 'four compressed NPZ, each p3d and final probabilities; other input/output tensor SHA and exact comparisons in receipt',
            'prepare_cuda_initialized': False, 'prepare_pixel_decodes': 0}
    write(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    require(plan['specification'] == SPEC and plan['status'] == 'locked_pending_root_gpu_authorization', 'Plan differs')
    snapshot = Path(plan['source_snapshot'])
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Use frozen entrypoint')
    require(source_files(snapshot) == plan['source_hashes'], 'Frozen source changed')
    require(all(plan['source_hashes'][k] == h for k, h in plan['original_h3_source_hashes'].items()), 'H3 package changed')
    require(digest(plan['base']) == BASE_SHA == plan['base_sha256'], 'Wrong base')
    algebra = json.loads(ALGEBRA.read_text())
    require(digest(ALGEBRA) == ALGEBRA_SHA and plan['row_projection'] == algebra['row_projection']
            and plan['projected_features_sha256'] == algebra['feature_decomposition']['projected_fp32_sha256'],
            'The unique predeclared projector/candidate changed')
    for path, expected in plan['input_hashes'].items():
        require(digest(path) == expected, f'Input changed: {path}')
    require(fixed_views(json.loads(Path(plan['manifest']).read_text())) == (plan['views'], plan['training_camera_names']),
            'Fixed views changed')
    for name in ('train', 'model', 'losses', 'refinement', 'depth_moments'):
        importlib.import_module('bridge_rgs.'+name)
    loaded = {}
    for name, module in list(sys.modules.items()):
        if name == 'bridge_rgs' or name.startswith('bridge_rgs.'):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(snapshot) and plan['source_hashes'].get(str(path.relative_to(snapshot))) == digest(path),
                    f'Unfrozen package import {name}')
            loaded[name] = {'path': str(path), 'sha256': digest(path)}
    return loaded


def execute(plan_path):
    started = time.monotonic()
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    output = Path(plan['output'])
    # Exclusive durable marker forbids a retry even after abrupt termination.
    with (output/'execution_started.json').open('x') as stream:
        json.dump({'plan_sha256': digest(plan_path), 'pid': os.getpid()}, stream)
    os.chdir(plan['root'])
    torch.set_num_threads(8)
    report = {'status': 'failed', 'plan_sha256': digest(plan_path), 'base_sha256': BASE_SHA,
              'specification': SPEC, 'scene_renders': 0, 'raster_calls': 0,
              'backward_calls': 0, 'optimizer_steps': 0, 'checkpoint_written': False,
              'views': [], 'restoration': {}, 'pixel_scope': {}, 'real_rgb_pixels_decoded': 0, 'val_pixels_decoded': 0,
              'source_hashes': plan['source_hashes'], 'input_hashes': plan['input_hashes']}
    old_handler = signal.getsignal(signal.SIGALRM)

    def deadline(*_):
        raise TimeoutError('Fixed 105s diagnostic deadline; no retry')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(SPEC['deadline_seconds'])
    try:
        report['actual_imports'] = verify(plan)
        require(shutil.disk_usage(output).free >= plan['minimum_free_bytes'], 'Insufficient output disk')
        import gsplat

        from bridge_rgs.train import load_scene
        scene, state = load_scene(plan['base'], 'cuda')
        cameras = state['training_cameras'].clone().cuda()
        require(tensor_hash(cameras) == plan['training_camera_sha256'], 'Camera state changed')
        require(scene.refiner_config['depth_moments'] == 'cross', 'Moments must be recomputed')
        projected = project_features(state['model'][FEATURE], plan['row_projection'])
        require(tensor_hash(projected) == plan['projected_features_sha256'], 'Actual feature candidate differs')
        predictions = {}
        old_raster = gsplat.rasterization

        def raster(*args, **kwargs):
            report['raster_calls'] += 1
            require(report['raster_calls'] <= 8, 'Raster budget exceeded')
            return old_raster(*args, **kwargs)
        gsplat.rasterization = raster
        try:
            with preserved_scene(scene, cameras, report['restoration']) as before, pixel_guard(plan['allowed_pixel_paths'], report['pixel_scope']):
                for condition in SPEC['states']:
                    with torch.no_grad():
                        scene.splats['sem_features'].copy_(before[FEATURE].to('cuda') if condition == 'original' else projected.to('cuda'))
                    require(all(tensor_hash(v) == report['restoration']['tensor_hashes_before'][k]
                                for k, v in scene.state_dict().items() if k != FEATURE), 'Non-feature tensor changed')
                    for view in plan['views']:
                        K = torch.tensor(view['K'], device='cuda', dtype=torch.float32)
                        pose = cameras[view['camera_index']].clone()
                        pose_before, K_before = tensor_hash(pose), tensor_hash(K)
                        report['scene_renders'] += 1
                        with torch.no_grad():
                            result = scene.render(K, pose, view['width'], view['height'], degree=3,
                                                  semantics=True, geometry_grad=False, refine=True, absgrad=False)
                        require(all(bool(torch.isfinite(result[k]).all()) for k in ('rgb', 'depth', 'alpha', 'p3d', 'probabilities', 'residual', 'features', 'depth_moments')),
                                'Nonfinite prediction/evidence')
                        require(tensor_hash(K) == K_before and tensor_hash(pose) == pose_before, 'Render modified camera inputs')
                        arrays = {k: result[k].detach().cpu().contiguous() for k in ('rgb', 'depth', 'alpha', 'p3d', 'probabilities')}
                        hashes = {k: tensor_hash(result[k]) for k in ('rgb', 'depth', 'alpha', 'p3d', 'probabilities', 'residual', 'features', 'depth_moments')}
                        file = output/f'{Path(view["name"]).stem}.{condition}.npz'
                        np.savez_compressed(file, p3d=arrays['p3d'].numpy(), probabilities=arrays['probabilities'].numpy())
                        predictions[(view['name'], condition)] = arrays
                        report.setdefault('predictions', []).append({'name': view['name'], 'condition': condition,
                            'path': str(file), 'sha256': digest(file), 'tensor_hashes': hashes,
                            'shapes': {k: list(v.shape) for k, v in arrays.items()}, 'camera_index': view['camera_index']})
                        del result
                require(report['scene_renders'] == 4 and report['raster_calls'] == 8, 'Incomplete fixed render budget')
                report['pixel_scope']['predictions_completed'] = True
                report['all_predictions_completed_seconds'] = time.monotonic()-started
                for view in plan['views']:
                    name = view['name']
                    report['views'].append({'name': name, 'invariance': invariance(predictions[(name, 'original')], predictions[(name, 'projected')])})
                report['numerical_gate_passed'] = all(row['invariance']['passed'] for row in report['views'])
                if report['numerical_gate_passed']:
                    for view, row in zip(plan['views'], report['views'], strict=True):
                        label_array = cv2.imread(view['mask_path'], cv2.IMREAD_GRAYSCALE)
                        valid_array = cv2.imread(view['valid_path'], cv2.IMREAD_GRAYSCALE)
                        require(label_array is not None and valid_array is not None and label_array.shape == valid_array.shape == (989, 1320), 'Label support mismatch')
                        require(bool(np.isin(label_array, [0, 1, 2, 3, 4, 255]).all()), 'Unexpected labels')
                        labels = torch.from_numpy(label_array.astype(np.int64))
                        valid = torch.from_numpy(valid_array > 0)
                        labels[~valid] = 255
                        row['scores'] = {condition: {key: score_prediction(predictions[(view['name'], condition)][key], labels, valid)
                                                   for key in ('p3d', 'probabilities')} for condition in SPEC['states']}
                        row['correction_counts'] = correction_counts(predictions[(view['name'], 'original')], predictions[(view['name'], 'projected')], labels, valid)
                    from bridge_rgs.losses import iou_scores
                    report['summary'] = {}
                    for condition in SPEC['states']:
                        report['summary'][condition] = {}
                        for key in ('p3d', 'probabilities'):
                            scores = [row['scores'][condition][key] for row in report['views']]
                            cm = sum((torch.tensor(s['confusion_matrix']) for s in scores), torch.zeros(5, 5, dtype=torch.int64))
                            report['summary'][condition][key] = {'equal_view_mean_ce': sum(s['ce'] for s in scores)/2,
                                                               'pooled_confusion_matrix': cm.tolist(), 'pooled_iou': iou_scores(cm)}
                report['status'] = 'completed'
        finally:
            gsplat.rasterization = old_raster
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        report['all_bound_inputs_and_sources_unchanged'] = (
            all(digest(p) == h for p, h in plan['input_hashes'].items()) and source_files(Path(plan['source_snapshot'])) == plan['source_hashes'])
        report['elapsed_seconds'] = time.monotonic()-started
        restoration = report['restoration']
        expected = ('all_model_tensors_finally_restored_exact', 'flags_restored_exact', 'modes_restored_exact',
                    'cameras_restored_exact', 'no_parameter_gradients', 'non_feature_tensors_unchanged_before_restore',
                    'cameras_unchanged_before_restore')
        report['all_invariants_passed'] = bool(report.get('numerical_gate_passed', False)
            and report['all_bound_inputs_and_sources_unchanged'] and all(restoration.get(k, False) for k in expected))
        write(output/'execution_receipt.json', report)
    require(report['all_invariants_passed'], 'Diagnostic numerical/restoration gate failed; retain record, no retry')
    print(json.dumps({'status': report['status'], 'receipt': str(output/'execution_receipt.json'),
                      'sha256': digest(output/'execution_receipt.json'), 'scene_renders': report['scene_renders'],
                      'elapsed_seconds': report['elapsed_seconds']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path)
    group.add_argument('--execute', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.prepare:
        plan = prepare(args.root, args.prepare)
        print(json.dumps({'plan': str(plan), 'sha256': digest(plan), 'runner_sha256': digest(__file__)}))
    else:
        execute(args.execute)


if __name__ == '__main__':
    main()
