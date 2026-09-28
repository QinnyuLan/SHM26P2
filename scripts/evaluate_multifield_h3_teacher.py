"""Fixed four-arm H3/H+ fusion with three actually rendered RGB fields; no training."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
MATCHED = RUNS/'matched_rgb_teacher_adaptation_v1/evaluation_v2'
CACHE = RUNS/'matched_rgb_teacher_adaptation_v1/cache'
TRANSFER = RUNS/'rgb_teacher_transfer_v1'
DOCUMENT = 'multifield_h3_teacher_protocol.md'
HELPER = Path(__file__).with_name('matched_teacher_evaluation_base.py')
if not HELPER.is_file():
    HELPER = MATCHED/'source_snapshot/evaluate_matched_rgb_teacher_adaptation.py'
_spec = importlib.util.spec_from_file_location('matched_teacher_evaluation_base', HELPER)
matched = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(matched)
base = matched.base
read, write, require, sha, utc = base.read, base.write, base.require, base.sha, base.utc
ARMS = ('A', 'B', 'C', 'D', 'E')
TEACHER_ARMS = ARMS[:4]
COMPONENTS = ('selected', 'capacity_1m', 'mcmc')
INFERENCE = dict(base.INFERENCE)
SPEC = {
    'protocol': 'multifield_h3_fixed_half_teacher_v1', 'arms': list(ARMS), 'candidate': 'D',
    'teachers': {'A': 'historical_00f5', 'B': 'historical_00f5', 'C': 'selected_2k_last', 'D': 'composite_2k_last'},
    'h3_rgb_input': 'Own original render tensor; never replaced with the composite RGB',
    'teacher_inputs': {'A': 'direct quantized H3 legacy canvas', 'B': 'composite original RGB through legacy adapter',
                      'C': 'same as B', 'D': 'same as B'},
    'probability_weights': [.5, .5], 'fusion_grid': 'legacy canvas; soft blend then one distortion warp then argmax',
    'rgb_mean': 'float32 equal mean of original-grid uint8 components, np.rint ties-to-even, uint8',
    'inference': INFERENCE, 'scene_renders': 150, 'joint_masks_before_gt': 250,
    'soft_caches': 'HWC float32 NPY: 50 H3 and 200 teacher canvases; no compression',
    'training_steps': 0, 'bootstrap_repeats': 5000, 'bootstrap_seed': 20260926,
    'comparisons': ['D_minus_A', 'D_minus_B', 'D_minus_C'],
    'engineering_control': 'E uses this execution A soft-blend mask and this execution composite RGB; no additional teacher call',
    'engineering_comparisons': ['E_minus_A', 'E_minus_Swin'],
    'system_gate': dict(base.SPEC['gate']), 'domain_matching_gate': dict(matched.SPEC['domain_matching_gate']),
    'internal_seconds': 570, 'external_seconds': 600, 'retries': 0,
    'gpu_process_policy': dict(matched.SPEC['gpu_process_policy']),
    'scope': 'Three-field engineering ensemble, not a single shared geometry or established innovation; reused development set',
    'cost_scope': '150 actual scene calls, teacher inference, model loading, probability/PNG I/O and scoring; desktop-shared GPU, not exclusive FPS',
}


def average_rgb(a, b):
    require(a.dtype == b.dtype == np.uint8 and a.shape == b.shape, 'Require aligned uint8 components')
    return np.rint((a.astype(np.float32)+b.astype(np.float32))*.5).astype(np.uint8)


def blend_soft(h3, teacher):
    require(h3.dtype == teacher.dtype == np.float32 and h3.shape == teacher.shape
            and h3.ndim == 3 and h3.shape[-1] == 5, 'Require aligned HWC FP32 five-class soft probabilities')
    for value in (h3, teacher):
        require(np.isfinite(value).all() and (value >= 0).all()
                and np.allclose(value.sum(-1), 1., atol=2e-5, rtol=0), 'Invalid soft probabilities')
    return .5*h3 + .5*teacher


def prediction_barrier(predictions, views):
    expected = {(arm, v['name']) for arm in ARMS for v in views}
    require(len(views) == 50 and len(expected) == 250 and len(predictions) == 250
            and {(p['arm'], p['name']) for p in predictions} == expected,
            'All 250 unique joint masks must be saved before GT')


def verify_canvas(camera, state, adapter_info, official):
    K, width, height, back = official.distortion_render_grid(camera['K'], camera['distortion'],
        camera['width'], camera['height'], official.pixel_protocol(state))
    require(official.pixel_protocol(state) == base.LEGACY
            and adapter_info['canvas'] == [width, height]
            and np.array_equal(K, np.asarray(adapter_info['canvas_K'], np.float32)),
            'H3 and teacher must share the exact legacy canvas')
    return K, width, height, back


def prepare(output):
    require(not torch.cuda.is_initialized(), 'CPU preparation only')
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite an experiment')
    mp, mr = read(MATCHED/'plan.json'), read(MATCHED/'execution_receipt.json')
    tp, tr = read(TRANSFER/'plan.json'), read(TRANSFER/'execution_receipt.json')
    cp, cr = read(CACHE/'plan.json'), read(CACHE/'execution_receipt.json')
    require(mr['status'] == tr['status'] == cr['status'] == 'completed', 'Prior executions must complete')
    require(mr['plan_sha256'] == sha(MATCHED/'plan.json') and tr['plan_sha256'] == sha(TRANSFER/'plan.json')
            and cr['plan_sha256'] == sha(CACHE/'plan.json'), 'Prior plan binding differs')
    require(mr['bound_inputs_and_sources_unchanged'] and tr['bound_inputs_and_sources_unchanged'], 'Prior input audit failed')
    require(mr['predictions_finished_utc'] < mr['source_scoring_started_utc']
            and tr['predictions_finished_utc'] < tr['source_scoring_started_utc'], 'Prior target order changed')
    views = tp['views']
    require(len(views) == 50 and sum(v['source_annotation_path'] is not None for v in views) == 41, 'Require 50/41')
    require([v['name'] for v in views] == sorted(v['name'] for v in views), 'Fixed sorted camera order required')
    forbidden = {v[k] for v in views for k in ('source_image_path', 'source_annotation_path') if v[k]}
    inputs = {}
    def bind(path, expected=None):
        path = str(Path(path).resolve())
        require(path not in forbidden, 'Preparation cannot read GT payloads')
        if path in inputs:
            require(expected is None or inputs[path] == expected, f'Conflicting input declarations: {path}')
            return
        digest = sha(path)
        require(expected is None or digest == expected, f'Input changed: {path}')
        inputs[path] = digest
    for parent in (MATCHED, TRANSFER, CACHE):
        for name in ('plan.json', 'execution_receipt.json'):
            bind(parent/name)
    for plan in (mp, tp, cp):
        for path, digest in plan['input_hashes'].items():
            bind(path, digest)
        require(base.source_files(Path(plan['source_snapshot'])) == plan['source_hashes'], 'Prior snapshot changed')
    require(sha(HELPER) == mp['source_hashes']['evaluate_matched_rgb_teacher_adaptation.py'], 'Wrong matched helper')
    bind(HELPER)
    bind(matched.HELPER_PATH, mp['source_hashes']['rgb_teacher_transfer_base.py'])
    package = Path(cp['source_snapshot'])/'bridge_rgs'
    # The 36-file renderer package is already verified for all three scene formats.
    # Teacher/model/grid implementations must also equal the successful selected/2k evaluations.
    matched_package = Path(mp['source_snapshot'])/'bridge_rgs'
    for name in ('model.py', 'train.py', 'teacher.py', 'teacher_adapters.py', 'coordinates.py', 'refinement.py', 'depth_moments.py'):
        require(sha(package/name) == sha(matched_package/name), f'Incompatible renderer/teacher source: {name}')
    for path in package.rglob('*.py'):
        bind(path)
    bind(Path(mp['source_snapshot'])/'compare_official_evaluations.py')
    endpoints, training_bindings = matched.training_ready(mp['training_plan'], mp['training_completion'], forbidden)
    for path, digest in training_bindings.items():
        bind(path, digest)
    teachers = {'old': {'path': tp['teacher']['teacher_checkpoint'], 'sha256': tp['teacher']['teacher_checkpoint_sha256']},
                **{key: {'path': e['last_checkpoint'], 'sha256': e['last_checkpoint_sha256']}
                   for key, e in endpoints.items()}}
    require(teachers['old']['sha256'] == base.TEACHER_SHA, 'Historical teacher identity changed')
    for role, record in teachers.items():
        bind(record['path'], record['sha256'])
        state = torch.load(record['path'], map_location='cpu', weights_only=False)
        require(matched.profile_id(state) == base.LEGACY and bool(state['ema_decoder']), 'Need legacy EMA')
        if role != 'old':
            matched.checkpoint_contract(state, endpoints[role], mp['backbone'], mp['training_manifests'][role])
        record['configuration'] = state['configuration']
        record['provenance'] = state['provenance']
        require(state['configuration']['channels'] == 192 and state['configuration'].get('adapter_rank', 0) == 0,
                'Only fixed H+ head-only architecture is permitted')
    selected_receipt = read(Path(cp['components']['selected']['directory'])/'execution_receipt.json')
    component_refs = {}
    for role, component in cp['components'].items():
        receipt_path = Path(component['directory'])/'execution_receipt.json'
        bind(receipt_path)
        receipt = read(receipt_path)
        require(receipt['status'] == 'completed' and receipt['checkpoint_sha256'] == component['checkpoint_sha256'],
                'Incomplete/wrong component')
        require(sorted(receipt['source_records'], key=lambda x: x['name'])
                == sorted(selected_receipt['source_records'], key=lambda x: x['name']), 'Different official cameras/targets')
        bind(component['checkpoint'], component['checkpoint_sha256'])
        component_refs[role] = {r['name']: r for r in receipt['predictions']}
        for row in receipt['predictions']:
            bind(row['rgb'], row['rgb_sha256'])
            if role == 'selected':
                bind(row['mask'], row['mask_sha256'])
    require(all(set(refs) == {v['name'] for v in views} for refs in component_refs.values()), 'Missing component RGB')
    prior_controls = {'B': {r['name']: r for r in tr['predictions'] if r['arm'] == 'composite_rgb_candidate'},
                      **{arm: {r['name']: r for r in mr['predictions'] if r['arm'] == role}
                         for arm, role in [('C', 'selected'), ('D', 'composite')]}}
    for refs in prior_controls.values():
        require(set(refs) == {v['name'] for v in views}, 'Missing pure-teacher control')
        for row in refs.values():
            bind(row['mask'], row['mask_sha256'])
    rows = []
    for v in views:
        composite = v['inputs']['composite_rgb_candidate']
        bind(composite['path'], composite['sha256'])
        rows.append({**base.fingerprint_record(v), 'name': v['name'],
            'source_image_path': v['source_image_path'], 'source_annotation_path': v['source_annotation_path'],
            'component_rgb': {k: {'path': component_refs[k][v['name']]['rgb'],
                                  'sha256': component_refs[k][v['name']]['rgb_sha256']} for k in COMPONENTS},
            'composite_rgb': composite,
            'selected_mask': {'path': component_refs['selected'][v['name']]['mask'],
                              'sha256': component_refs['selected'][v['name']]['mask_sha256']},
            'pure_controls': {a: {'path': p[v['name']]['mask'], 'sha256': p[v['name']]['mask_sha256']}
                              for a, p in prior_controls.items()}})
    for path in (tp['selected_metrics'], tp['composite_metrics'], tp['prior_rgb_gate']):
        bind(path)
    require(sha(tp['selected_metrics']) == selected_receipt['official_metrics_sha256'], 'Unbound selected metrics')
    require(sha(tp['composite_metrics']) == read(RUNS/'rgb_fixed_ensemble_v1/execution_receipt.json')['metrics_sha256'],
            'Unbound composite metrics')
    swin = RUNS/'mask2former_reference_v1/official_evaluation'
    swin_receipt = read(swin/'execution_receipt.json')
    require(swin_receipt['status'] == 'completed'
            and swin_receipt['official_evaluation_fingerprint'] == tp['reference_fingerprint'], 'Swin population differs')
    bind(swin/'execution_receipt.json')
    bind(swin/'official_metrics.json', swin_receipt['official_metrics_sha256'])
    for path in (Path(__file__), ROOT/'docs'/DOCUMENT, ROOT/'tests/test_multifield_h3_teacher.py', ROOT/'uv.lock'):
        bind(path)
    require(not forbidden.intersection(inputs), 'GT was accidentally bound as a payload')
    require(shutil.disk_usage(output.parent).free >= 10*1024**3, 'Need >=10 GiB for 250 FP32 canvases and output')
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(package, snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for src, dst in ((Path(__file__), Path(__file__).name), (HELPER, 'matched_teacher_evaluation_base.py'),
                     (matched.HELPER_PATH, 'rgb_teacher_transfer_base.py'),
                     (Path(mp['source_snapshot'])/'compare_official_evaluations.py', 'compare_official_evaluations.py'),
                     (ROOT/'docs'/DOCUMENT, DOCUMENT), (ROOT/'tests/test_multifield_h3_teacher.py', 'test_multifield_h3_teacher.py')):
        shutil.copy2(src, snapshot/dst)
    plan = {'status': 'locked_pending_root_gpu_handoff', 'specification': SPEC, 'root': str(ROOT), 'output': str(output),
        'source_snapshot': str(snapshot), 'source_hashes': base.source_files(snapshot), 'input_hashes': inputs,
        'components': cp['components'], 'teachers': teachers, 'backbone': mp['backbone'], 'views': rows,
        'selected_metrics': tp['selected_metrics'], 'composite_metrics': tp['composite_metrics'],
        'swin_metrics': str(swin/'official_metrics.json'),
        'prior_rgb_gate': tp['prior_rgb_gate'], 'reference_fingerprint': tp['reference_fingerprint'],
        'scoring_protocol': tp['scoring_protocol'], 'evaluation_family': tp['evaluation_family'],
        'runtime_versions': tp['runtime_versions'], 'prepare_gt_payload_reads': 0, 'prepare_cuda_initialized': False,
        'external_timeout_seconds': 600, 'known_gaussian_count': 1994145,
        'env': {'PYTHONPATH': str(snapshot), 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '8',
                'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'TORCH_CUDA_ARCH_LIST': '12.0', 'MAX_JOBS': '4'}}
    write(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    snapshot = Path(plan['source_snapshot'])
    require(plan['specification'] == SPEC and base.source_files(snapshot) == plan['source_hashes'], 'Source/spec changed')
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Execute frozen runner only')
    require(all(sha(p) == h for p, h in plan['input_hashes'].items()), 'Bound input changed')
    require({k: importlib.metadata.version(k) for k in plan['runtime_versions']} == plan['runtime_versions'], 'Versions changed')


def runtime_imports(plan):
    records = matched.runtime_imports(plan)
    records['matched_teacher_evaluation_base'] = {'path': str(HELPER.resolve()), 'sha256': sha(HELPER)}
    for record in records.values():
        p = Path(record['path'])
        require(p.is_relative_to(plan['source_snapshot'])
                and record['sha256'] == plan['source_hashes'][str(p.relative_to(plan['source_snapshot']))], 'Unfrozen helper')
    return records


def decode(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    require(image is not None and image.dtype == np.uint8 and image.ndim == 3 and image.shape[-1] == 3, 'Invalid RGB PNG')
    return image[..., ::-1].copy()


def save_png(path, value, reference=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), 'Refuse PNG overwrite')
    require(cv2.imwrite(str(path), value), 'PNG write failed')
    digest = sha(path)
    if reference is not None:
        require(digest == reference['sha256'] and Path(path).read_bytes() == Path(reference['path']).read_bytes(),
                'New RGB PNG differs from bound source bytes')
    return {'path': str(path), 'sha256': digest}


def save_soft(path, value):
    require(value.dtype == np.float32 and value.ndim == 3 and value.shape[-1] == 5
            and np.isfinite(value).all(), 'Need finite FP32 HWC soft canvas')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
    return {'path': str(path), 'sha256': sha(path), 'shape': list(value.shape), 'dtype': 'float32'}


def engineering_control(output, name, fresh_a_mask, composite):
    """Use the in-memory newly inferred A result, not any historical mask file."""
    mask = save_png(output/'E/mask'/name, fresh_a_mask)
    rgb = save_png(output/'E/rgb'/name, decode(composite['path'])[..., ::-1], composite)
    return mask, rgb


def render_fields(plan, report, official):
    output = Path(plan['output'])
    h3_cache, rgb_cache = {}, {}
    for role in COMPONENTS:
        component = plan['components'][role]
        scene, state = official._load_scene(component['checkpoint'], 'cuda')
        scene.eval().requires_grad_(False)
        require(official.pixel_protocol(state) == component['profile'], 'Renderer profile changed')
        actual_render = scene.render
        captured = []
        def capture(*args, _render=actual_render, _role=role, _captured=captured, **kwargs):
            value = _render(*args, **kwargs)
            report['scene_renders'] += 1
            if _role == 'selected':
                _captured.append(value)
            return value
        scene.render = capture
        rgb_cache[role] = {}
        try:
            for view in plan['views']:
                camera = view['camera']
                rgb, _, info = official.predict_official_camera(scene, camera, state)
                require(np.array_equal(rgb, decode(view['component_rgb'][role]['path'])), 'Component RGB pixels differ')
                rgb_cache[role][view['name']] = save_png(output/'components'/role/view['name'], rgb[..., ::-1],
                                                        view['component_rgb'][role])
                if role == 'selected':
                    require(len(captured) == 1, 'Exactly one scene render per camera')
                    result = captured.pop()
                    _, _, _, boundary = base.adapter_maps(camera, official.distortion_render_grid)
                    _, width, height, _ = verify_canvas(camera, state, boundary, official)
                    raw_rgb = result['rgb'].clamp(0, 1).cpu().numpy()
                    soft = result['probabilities'].cpu().numpy()
                    require(soft.shape == (height, width, 5), 'H3 canvas shape differs')
                    h3_cache[view['name']] = {'soft': save_soft(output/'soft/h3'/f"{Path(view['name']).stem}.npy", soft),
                        'canvas_rgb': save_png(output/'h3_canvas_rgb'/view['name'],
                            (raw_rgb*255).round().astype(np.uint8)[..., ::-1]),
                        'boundary': boundary, 'official_render_info': info}
                    del result
        finally:
            scene.render = actual_render
        del scene, state, actual_render, capture
        gc.collect()
        torch.cuda.empty_cache()
    require(report['scene_renders'] == 150, 'Fixed three-field render budget differs')
    composite = {}
    for view in plan['views']:
        rgb = average_rgb(*(decode(rgb_cache[role][view['name']]['path']) for role in ('capacity_1m', 'mcmc')))
        composite[view['name']] = save_png(output/'composite_rgb'/view['name'], rgb[..., ::-1], view['composite_rgb'])
    report['h3_canvases'], report['rendered_rgb'], report['composite_rgb'] = h3_cache, rgb_cache, composite
    write(output/'render_receipt.json', {'scene_renders': 150, 'h3_canvases': h3_cache,
        'rendered_rgb': rgb_cache, 'composite_rgb': composite, 'gt_payload_reads': 0})
    return h3_cache, rgb_cache, composite


def predict_arms(plan, report, official, teacher):
    output = Path(plan['output'])
    h3_cache, rgb_cache, composite = render_fields(plan, report, official)
    model, previous_role = None, None
    for arm in TEACHER_ARMS:
        role = {'A': 'old', 'B': 'old', 'C': 'selected', 'D': 'composite'}[arm]
        if role != previous_role:
            state = torch.load(plan['teachers'][role]['path'], map_location='cpu', weights_only=False)
            options = teacher.checkpoint_adapter_options(state['configuration'])
            if model is None:
                model = teacher.load_teacher(plan['backbone']['model_dir'], 5, state['configuration']['channels'],
                                             device='cuda', pixel_profile=base.LEGACY, **options)
                report['teacher_parameters'] = sum(p.numel() for p in model.parameters())
            model.decoder.load_state_dict(state['ema_decoder'], strict=True)
            teacher.load_checkpoint_adapters(model, state)
            model.eval().requires_grad_(False)
            previous_role = role
            del state
        for view in plan['views']:
            name, camera = view['name'], view['camera']
            mx, my, back, boundary = base.adapter_maps(camera, official.distortion_render_grid)
            require(boundary == h3_cache[name]['boundary'], 'Adapter changed between branches')
            canvas = decode(h3_cache[name]['canvas_rgb']['path']) if arm == 'A' else base.input_canvas(
                decode(composite[name]['path']), mx, my)
            probabilities, _ = teacher.predict_image(model, canvas, **INFERENCE)
            teacher_soft = np.ascontiguousarray(probabilities.transpose(1, 2, 0), dtype=np.float32)
            teacher_record = save_soft(output/'soft'/arm/f'{Path(name).stem}.npy', teacher_soft)
            h3_soft = np.load(h3_cache[name]['soft']['path'], allow_pickle=False)
            soft = blend_soft(h3_soft, teacher_soft)
            pure = base.output_probabilities(probabilities, back).argmax(-1).astype(np.uint8)
            joint = base.output_probabilities(soft.transpose(2, 0, 1), back).argmax(-1).astype(np.uint8)
            if arm == 'A':
                require(np.array_equal(joint, cv2.imread(view['selected_mask']['path'], cv2.IMREAD_UNCHANGED)),
                        'A must reproduce the old complete selected mask pixel-exactly')
            else:
                require(np.array_equal(pure, cv2.imread(view['pure_controls'][arm]['path'], cv2.IMREAD_UNCHANGED)),
                        f'{arm}: pure-teacher path must reproduce the old corresponding mask')
            joint_record = save_png(output/arm/'mask'/name, joint, view['selected_mask'] if arm == 'A' else None)
            pure_record = save_png(output/arm/'pure_teacher_mask'/name, pure)
            rgb_record = rgb_cache['selected'][name] if arm == 'A' else composite[name]
            report['predictions'].append({'arm': arm, 'name': name, 'rgb': rgb_record['path'],
                'rgb_sha256': rgb_record['sha256'], 'mask': joint_record['path'], 'mask_sha256': joint_record['sha256'],
                'pure_teacher_mask': pure_record, 'teacher_soft_canvas': teacher_record, 'boundary': boundary,
                'control_pixel_exact': True, 'control_type': 'joint selected' if arm == 'A' else 'pure teacher'})
            if arm == 'A':
                # E is written from this newly computed soft-blend result, never an old mask.
                e_mask, e_rgb = engineering_control(output, name, joint, composite[name])
                report['predictions'].append({'arm': 'E', 'name': name, 'rgb': e_rgb['path'],
                    'rgb_sha256': e_rgb['sha256'], 'mask': e_mask['path'], 'mask_sha256': e_mask['sha256'],
                    'teacher_soft_canvas': teacher_record, 'boundary': boundary,
                    'control_pixel_exact': True, 'control_type': 'this execution A soft-blend; new composite RGB'})
        print({'arm': arm, 'completed_predictions': len(report['predictions'])}, flush=True)
    prediction_barrier(report['predictions'], plan['views'])
    report['predictions_finished_utc'] = utc()
    write(output/'predictions_receipt.json', {'status': 'all_250_joint_masks_before_GT',
        'predictions': report['predictions'], 'h3_canvases': report['h3_canvases'],
        'predictions_finished_utc': report['predictions_finished_utc'], 'annotation_payload_reads': 0})


def score(plan, report, official, compare):
    prediction_barrier(report['predictions'], plan['views'])
    require(report['annotation_payload_reads'] == 0, 'Premature annotation read')
    report['source_scoring_started_utc'] = utc()
    rows = {a: [] for a in ARMS}
    rgb_rows = {a: matched.common_rgb_rows(read(plan['selected_metrics'] if a == 'A' else plan['composite_metrics']),
                                         plan['views']) for a in ARMS}
    predictions = {(r['arm'], r['name']): r for r in report['predictions']}
    for view in plan['views']:
        truth = None
        if view['source_annotation_path'] is not None:
            content = Path(view['source_annotation_path']).read_bytes()
            report['annotation_payload_reads'] += 1
            require(hashlib.sha256(content).hexdigest() == view['source_annotation_sha256'], 'Changed annotation')
            truth = official.rasterize_official_annotation(content, view['camera']['width'], view['camera']['height'])
            require(hashlib.sha256(truth.tobytes()).hexdigest() == view['rasterized_mask_sha256'], 'Rasterizer differs')
        for arm in ARMS:
            prediction = predictions[arm, view['name']]
            require(sha(prediction['rgb']) == prediction['rgb_sha256'] and sha(prediction['mask']) == prediction['mask_sha256'],
                    'Saved prediction changed')
            row = dict(rgb_rows[arm][view['name']])
            if truth is not None:
                mask = cv2.imread(prediction['mask'], cv2.IMREAD_UNCHANGED)
                require(mask.dtype == np.uint8 and mask.shape == truth.shape and (mask < 5).all(), 'Bad delivered mask')
                keep = truth != 255
                cm = np.bincount(5*truth[keep].astype(np.int64)+mask[keep], minlength=25).reshape(5, 5)
                row.update(confusion_matrix=cm.tolist(), semantic_pixels=int(keep.sum()), semantic_ignore_pixels=int((~keep).sum()))
            rows[arm].append(row)
    require(report['annotation_payload_reads'] == 41 and official.official_fingerprint(
        [base.fingerprint_record(v) for v in plan['views']]) == plan['reference_fingerprint'], 'Wrong scoring population')
    metrics = {}
    for arm in ARMS:
        pooled = sum(np.asarray(r['confusion_matrix'], np.int64) for r in rows[arm] if 'confusion_matrix' in r)
        iou = compare._iou(pooled)
        value = dict(evaluation_family=plan['evaluation_family'], official_evaluation_fingerprint=plan['reference_fingerprint'],
            scoring_protocol=plan['scoring_protocol'], inference_protocol=SPEC, validation_views=50,
            semantic_validation_views=41, views=rows[arm], confusion_matrix=pooled.tolist(), iou=iou.tolist(),
            miou_all=float(np.nanmean(iou)), miou_foreground=float(np.nanmean(iou[1:])),
            **{k: float(np.mean([r[k] for r in rows[arm]])) for k in ('psnr', 'ssim', 'lpips')},
            rgb_scoring='Scores reused only after actual renderer PNG identity; no RGB GT reads or LPIPS',
            provenance={'arm': arm, 'components': plan['components'], 'teachers': plan['teachers'],
                        'single_shared_geometry': arm == 'A', 'weights': [.5, .5]})
        compare._validate(value)
        if arm == 'A':
            old = read(plan['selected_metrics'])
            require(value['confusion_matrix'] == old['confusion_matrix'], 'A pooled CM must exactly reproduce selected')
        write(Path(plan['output'])/arm/'official_metrics.json', value)
        metrics[arm] = value
    return metrics


def comparisons(plan, path, metrics, compare):
    output, pairs = Path(plan['output']), {}
    for arm in ('A', 'B', 'C'):
        pair = compare.paired_official_comparison(metrics[arm], metrics['D'], repeats=5000, seed=20260926)
        pair.update(reference_arm=arm, candidate_arm='D', plan_sha256=sha(path),
            reference_metrics_sha256=sha(output/arm/'official_metrics.json'),
            candidate_metrics_sha256=sha(output/'D/official_metrics.json'))
        pp = output/f'paired_D_minus_{arm}.json'
        write(pp, pair)
        pairs[arm] = {'path': str(pp), 'sha256': sha(pp)}
    system = base.adoption_clauses(read(plan['prior_rgb_gate']), read(pairs['A']['path']))
    domain = matched.domain_matching_clauses(read(pairs['C']['path']))
    extra = {}
    for label, ref, ref_path in [('A', metrics['A'], output/'A/official_metrics.json'),
                                 ('Swin', read(plan['swin_metrics']), plan['swin_metrics'])]:
        pair = compare.paired_official_comparison(ref, metrics['E'], repeats=5000, seed=20260926)
        pair.update(candidate_arm='E', reference_role=label, plan_sha256=sha(path),
            reference_metrics_sha256=sha(ref_path), candidate_metrics_sha256=sha(output/'E/official_metrics.json'))
        pp = output/f'paired_E_minus_{label}.json'
        write(pp, pair)
        extra[label] = {'path': str(pp), 'sha256': sha(pp)}
    identity = {**read(plan['prior_rgb_gate'])['clauses'],
        'all50_E_masks_exact_this_execution_A': all(
            (output/'E/mask'/v['name']).read_bytes() == (output/'A/mask'/v['name']).read_bytes() for v in plan['views']),
        'all50_E_rgb_exact_actual_composite': all(
            sha(output/'E/rgb'/v['name']) == v['composite_rgb']['sha256'] for v in plan['views'])}
    gate = {'system_passed': bool(all(system.values())), 'system_clauses': system,
        'domain_matching_evidence_passed': bool(all(domain.values())), 'domain_matching_clauses': domain,
        'passed': bool(all(system.values()) and all(domain.values())), 'candidate': 'D', 'comparisons': pairs,
        'prior_rgb_gate_sha256': sha(plan['prior_rgb_gate']), 'plan_sha256': sha(path),
        'engineering_control_E': {'passed': bool(all(identity.values())), 'clauses': identity, 'comparisons': extra},
        'consequence': 'No automatic adoption; report engineering system and matched-domain evidence separately'}
    write(output/'system_gate.json', gate)
    return pairs, gate


def execute(path):
    path = Path(path).resolve()
    plan, started = read(path), time.monotonic()
    output = Path(plan['output'])
    write(output/'execution_started.json', {'utc': utc(), 'pid': os.getpid(), 'plan_sha256': sha(path)})
    report = {'status': 'failed', 'schema': SPEC['protocol'], 'specification': SPEC, 'plan_sha256': sha(path),
        'started_utc': utc(), 'predictions': [], 'scene_renders': 0, 'optimizer_steps': 0,
        'annotation_payload_reads': 0, 'gt_rgb_payload_reads': 0}
    handler = signal.getsignal(signal.SIGALRM)
    def deadline(*_):
        raise TimeoutError('Fixed 570s deadline; no retry')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(570)
    try:
        os.chdir(plan['root'])
        torch.set_num_threads(8)
        cv2.setNumThreads(8)
        verify(plan)
        sys.path.insert(0, plan['source_snapshot'])
        official = importlib.import_module('bridge_rgs.official_evaluate')
        teacher = importlib.import_module('bridge_rgs.teacher')
        compare = importlib.import_module('compare_official_evaluations')
        report['actual_imports'] = runtime_imports(plan)
        gpu = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True, check=True, timeout=5).stdout
        report['gpu_before'] = matched.gpu_process_records(gpu)
        report['gpu_inventory_sha256'] = hashlib.sha256(gpu.encode()).hexdigest()
        matched.enforce_gpu_process_policy(report['gpu_before'])
        report['nonexclusive_gpu_cost_scope'] = SPEC['gpu_process_policy']['cost_scope']
        with torch.inference_mode():
            predict_arms(plan, report, official, teacher)
        metrics = score(plan, report, official, compare)
        pairs, gate = comparisons(plan, path, metrics, compare)
        report.update(status='completed', finished_utc=utc(), comparisons=pairs, gate_passed=gate['passed'],
            gate_sha256=sha(output/'system_gate.json'), metrics_sha256={a: sha(output/a/'official_metrics.json') for a in ARMS},
            official_evaluation_fingerprint=plan['reference_fingerprint'], actual_imports=runtime_imports(plan))
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, handler)
        report['bound_inputs_and_sources_unchanged'] = base.source_files(Path(plan['source_snapshot'])) == plan['source_hashes'] \
            and all(sha(p) == h for p, h in plan['input_hashes'].items())
        if not report['bound_inputs_and_sources_unchanged']:
            report['status'] = 'failed'
        report['elapsed_seconds_full_multifield_bundle_and_scoring'] = time.monotonic()-started
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        write(output/'execution_receipt.json', report)
    require(report['status'] == 'completed', 'Final invariance check failed')
    print({'status': report['status'], 'receipt': str(output/'execution_receipt.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path)
    mode.add_argument('--execute', type=Path)
    args = parser.parse_args()
    if args.prepare:
        path = prepare(args.prepare)
        print({'plan': str(path), 'plan_sha256': sha(path), 'runner_sha256': sha(__file__)})
    else:
        execute(args.execute)


if __name__ == '__main__':
    main()
