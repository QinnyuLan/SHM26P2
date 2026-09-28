"""Two fixed pure-H+ readouts of existing original-grid RGB PNGs; explicit legacy migration."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
SELECTED = RUNS/'official_selected_ensemble_v1/cross_teacher'
COMPOSITE = RUNS/'rgb_fixed_ensemble_v1'
PACKAGE = RUNS/'official_selected_ensemble_v1/source_snapshot/bridge_rgs'
TEACHER_SHA = '00f5b84ac9a56c39512c5b8e43f70110397923feea1a2b4c78bdffd1f5524bff'
LEGACY = 'legacy_mixed_v1'
DOCUMENT = 'rgb_teacher_transfer_protocol.md'
ARMS = ('selected_rgb_control', 'composite_rgb_candidate')
INFERENCE = {'tile_size': 768, 'stride': 512, 'flip': True, 'context_weight': .25, 'context_short_side': 768}
SPEC = {'protocol': 'original_png_to_legacy_pure_hplus_v1', 'arms': list(ARMS), 'teacher_weight': 1.,
            'inference': INFERENCE, 'views_per_arm': 50, 'annotated_views': 41, 'masks_before_any_gt': 100,
            'input_adapter': 'original K/distortion -> initUndistortRectifyMap to legacy distortion_render_grid canvas; no half-pixel conjugation',
            'border': 'OpenCV INTER_LINEAR, BORDER_CONSTANT zero; no external context or validity exclusion',
            'output_adapter': 'soft HWC probabilities -> legacy distortion_render_grid INTER_LINEAR -> argmax',
            'source_profiles_preserved': True, 'renderer_calls': 0, 'training_steps': 0, 'checkpoint_tag_mutations': 0,
            'bootstrap_repeats': 5000, 'bootstrap_seed': 20260926,
            'gate': {'previous_rgb_four_clauses': True, 'miou_gain_at_least': .002, 'miou_ci_lower_strict': 0., 'cable_gain_at_least': -.001},
            'internal_seconds': 270, 'external_seconds': 300, 'retries': 0,
            'scope': 'Pure 2D H+ readout engineering comparison; candidate retains two RGB fields, no H3 third field or semantic Gaussian readout; reused development views, not blind test or innovation',
            'timing': 'Existing RGB PNG migration, H+ inference and scoring/I/O only; excludes both scene renders; not end-to-end FPS'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def utc():
    return datetime.now(UTC).isoformat()


def source_files(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.md'}}


def input_declaration(role, source):
    require(role in ARMS and source.get('input_grid') == 'original_distorted_uint8_png', 'Explicit original-grid source required')
    profiles = source.get('renderer_profiles')
    expected = [LEGACY] if role == ARMS[0] else ['colmap_corner_v2', 'colmap_corner_v2']
    require(profiles == expected, 'Do not relabel renderer profiles to load the legacy teacher')
    return dict(source, output_teacher_grid=LEGACY, explicit_coordinate_transfer=True, pure_teacher_weight=1.,
                uses_h3_semantic_readout=False, extra_renderer_for_boundary=False)


def adapter_maps(camera, grid_function):
    K, distortion = (np.asarray(camera[k], np.float32) for k in ('K', 'distortion'))
    width, height = camera['width'], camera['height']
    canvas_K, cw, ch, back = grid_function(K, distortion, width, height, pixel_protocol=LEGACY)
    mx, my = cv2.initUndistortRectifyMap(K, distortion, None, canvas_K, (cw, ch), cv2.CV_32FC1)
    inside = ((mx >= 0) & (mx <= width-1) & (my >= 0) & (my <= height-1)).astype(np.float32)
    coverage = inside if back is None else cv2.remap(inside, back[..., 0], back[..., 1], cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    info = {'canvas': [cw, ch], 'canvas_K': canvas_K.tolist(),
                'canvas_incomplete_source_support_pixels': int((inside == 0).sum()),
                'canvas_incomplete_source_support_fraction': float((inside == 0).mean()),
                'original_incomplete_roundtrip_footprint_pixels': int((coverage < 1-1e-6).sum()),
                'original_incomplete_roundtrip_footprint_fraction': float((coverage < 1-1e-6).mean())}
    return mx, my, back, info


def input_canvas(rgb, mx, my):
    require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3, 'Require uint8 RGB')
    return cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def output_probabilities(probabilities, back):
    require(probabilities.ndim == 3 and probabilities.shape[0] == 5 and np.isfinite(probabilities).all()
            and (probabilities >= 0).all(), 'Invalid teacher soft probabilities')
    hwc = np.ascontiguousarray(probabilities.transpose(1, 2, 0), dtype=np.float32)
    mapped = hwc if back is None else cv2.remap(hwc, back[..., 0], back[..., 1], cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    require(np.isfinite(mapped).all() and (mapped.sum(-1) > 0).all(), 'Empty/invalid original-grid probability footprint')
    return mapped


def fingerprint_record(view):
    return {k: view[k] for k in ('camera', 'source_image_sha256', 'source_annotation_sha256', 'rasterized_mask_sha256')}


def adoption_clauses(old_rgb_gate, comparison):
    require(set(old_rgb_gate['clauses']) == {'psnr_gain_at_least_0_15_dB', 'psnr_paired_95_lower_positive',
                                           'ssim_point_not_lower', 'lpips_point_not_higher'}, 'RGB gate schema differs')
    m = comparison['metrics']
    return {**old_rgb_gate['clauses'], 'miou_gain_at_least_0_20pp': m['miou_all']['difference'] >= .002,
            'miou_paired_95_lower_positive': m['miou_all']['paired_view_bootstrap_95_interval'][0] > 0,
            'cable_gain_at_least_minus_0_10pp': m['stay_cable_iou']['difference'] >= -.001}


def prepare(output):
    require(not torch.cuda.is_initialized(), 'Preparation is CPU only')
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite an experiment')
    sr, sm = read(SELECTED/'execution_receipt.json'), read(SELECTED/'official_metrics.json')
    cr, cm, cp, gate = [read(COMPOSITE/f) for f in ('execution_receipt.json', 'rgb_metrics.json', 'plan.json', 'rgb_gate.json')]
    require(sr['status'] == cr['status'] == 'completed' and cr['bound_inputs_and_sources_unchanged'], 'Incomplete RGB sources')
    require(sha(SELECTED/'official_metrics.json') == sr['official_metrics_sha256'] and sha(COMPOSITE/'rgb_metrics.json') == cr['metrics_sha256'], 'Unbound RGB metrics')
    require(sha(COMPOSITE/'plan.json') == cr['plan_sha256'] and sha(COMPOSITE/'rgb_gate.json') == cr['gate_sha256'], 'Composite source/gate differs')
    require(gate['passed'] and all(gate['clauses'].values()) and gate['plan_sha256'] == cr['plan_sha256'], 'Prior RGB gate did not pass')
    require(sm['official_evaluation_fingerprint'] == cm['inherited_common_reference_fingerprint'] == cp['reference_fingerprint'], 'Different camera/GT population')
    require(sr['predictions_finished_utc'] < sr['source_scoring_started_utc'] and cr['predictions_finished_utc'] < cr['source_scoring_started_utc'], 'Old scoring order differs')
    bindings = {}
    def bind(path, expected=None):
        path = str(Path(path).resolve())
        value = sha(path)
        require(expected is None or value == expected, f'Changed input: {path}')
        bindings[path] = value
    for path in [SELECTED/'execution_receipt.json', SELECTED/'official_metrics.json', COMPOSITE/'execution_receipt.json',
                 COMPOSITE/'rgb_metrics.json', COMPOSITE/'plan.json', COMPOSITE/'rgb_gate.json', ROOT/'uv.lock',
                 Path(__file__), ROOT/'docs'/DOCUMENT, ROOT/'tests/test_rgb_teacher_transfer.py', ROOT/'scripts/compare_official_evaluations.py']:
        bind(path)
    bind(COMPOSITE/'paired_minus_reference_500k.json', gate['primary_pair_sha256'])
    sources = sorted(sr['source_records'], key=lambda r: r['name'])
    require(len(sources) == 50 and sum(r['source_annotation_path'] is not None for r in sources) == 41, 'Require 50/41 official population')
    origin = {}
    for role, record in cp['original_evaluations'].items():
        folder = Path(record['directory'])
        bind(folder/'execution_receipt.json', record['receipt_sha256'])
        rr = read(folder/'execution_receipt.json')
        require(sorted(rr['source_records'], key=lambda r: r['name']) == sources, 'Original camera/GT metadata differs')
        bind(record['checkpoint'], record['checkpoint_sha256'])
        origin[role] = record
    bind(sr['checkpoint'], sr['checkpoint_sha256'])
    predictions = [{r['name']: r for r in receipt['predictions']} for receipt in (sr, cr)]
    require(all(sorted(p) == [r['name'] for r in sources] for p in predictions), 'Incomplete prediction set')
    rows = []
    for record in sources:
        images = {}
        for arm, p in zip(ARMS, predictions, strict=True):
            rgb = p[record['name']]
            bind(rgb['rgb'], rgb['rgb_sha256'])
            images[arm] = {'path': rgb['rgb'], 'sha256': rgb['rgb_sha256']}
        rows.append(dict(record, inputs=images))
    teacher = sr['teacher_ensemble']
    require(teacher['teacher_checkpoint_sha256'] == TEACHER_SHA and teacher['pixel_protocol']['id'] == LEGACY
            and teacher['inference'] == INFERENCE, 'Selected teacher/inference changed')
    for path, value in teacher['input_files_sha256'].items():
        bind(path, value)
    for path, value in source_files(PACKAGE).items():
        bind(PACKAGE/path, value)
    forbidden = {r[k] for r in sources for k in ('source_image_path', 'source_annotation_path') if r[k]}
    require(not forbidden.intersection(bindings), 'GT must remain unopened during preparation')
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(PACKAGE, snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in [Path(__file__), ROOT/'docs'/DOCUMENT, ROOT/'tests/test_rgb_teacher_transfer.py', ROOT/'scripts/compare_official_evaluations.py']:
        shutil.copy2(path, snapshot/path.name)
    declarations = {
        ARMS[0]: input_declaration(ARMS[0], {'input_grid': 'original_distorted_uint8_png', 'renderer_profiles': [LEGACY], 'rgb_components': [{'checkpoint': sr['checkpoint'], 'sha256': sr['checkpoint_sha256']}]}),
        ARMS[1]: input_declaration(ARMS[1], {'input_grid': 'original_distorted_uint8_png', 'renderer_profiles': ['colmap_corner_v2']*2,
            'rgb_components': [origin[k] for k in ('capacity_1m', 'mcmc')], 'rgb_weights': [.5, .5], 'composite_plan_sha256': cr['plan_sha256']})}
    plan = {'status': 'locked_pending_root_gpu_handoff', 'specification': SPEC, 'root': str(ROOT), 'output': str(output),
                'source_snapshot': str(snapshot), 'source_hashes': source_files(snapshot), 'input_hashes': bindings, 'views': rows,
                'teacher': teacher, 'input_declarations': declarations, 'original_evaluations': origin,
                'selected_metrics': str(SELECTED/'official_metrics.json'), 'composite_metrics': str(COMPOSITE/'rgb_metrics.json'),
                'prior_rgb_gate': str(COMPOSITE/'rgb_gate.json'), 'reference_fingerprint': sm['official_evaluation_fingerprint'],
                'scoring_protocol': sm['scoring_protocol'], 'evaluation_family': sm['evaluation_family'],
                'runtime_versions': sr['runtime_versions'], 'external_timeout_seconds': 300,
                'cost_evidence': {'receipt': str(SELECTED/'execution_receipt.json'), 'predictions_seconds': (datetime.fromisoformat(sr['predictions_finished_utc'])-datetime.fromisoformat(sr['started_utc'])).total_seconds(),
                    'rationale': 'Prior 50-view full bundle took ~54 s prediction/~68 s complete; two teacher-only arms have fixed 300s cap incl provenance/scoring, no scene renders or LPIPS'},
                'prepare_gt_payload_reads': 0, 'prepare_cuda_initialized': False,
                'env': {'PYTHONPATH': str(snapshot), 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'}}
    write(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    snapshot = Path(plan['source_snapshot'])
    require(plan['specification'] == SPEC and source_files(snapshot) == plan['source_hashes'], 'Source/specification changed')
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Execute frozen runner only')
    require(all(sha(p) == h for p, h in plan['input_hashes'].items()), 'Bound input changed')
    require({n: importlib.metadata.version(n) for n in plan['runtime_versions']} == plan['runtime_versions'], 'Dependency versions changed')
    for arm in ARMS:
        input_declaration(arm, plan['input_declarations'][arm])


def imports(plan):
    snapshot = Path(plan['source_snapshot'])
    records = {}
    for name, mod in list(sys.modules.items()):
        if name == 'bridge_rgs' or name.startswith('bridge_rgs.') or name == 'compare_official_evaluations':
            path = Path(mod.__file__).resolve()
            require(path.is_relative_to(snapshot) and sha(path) == plan['source_hashes'][str(path.relative_to(snapshot))], 'Unfrozen import')
            records[name] = {'path': str(path), 'sha256': sha(path)}
    return records


def execute(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    output = Path(plan['output'])
    started = time.monotonic()
    write(output/'execution_started.json', {'utc': utc(), 'pid': os.getpid(), 'plan_sha256': sha(plan_path)})
    report = {'status': 'failed', 'schema': 'multi_component_rgb_pure_2d_teacher_transfer_v1', 'plan_sha256': sha(plan_path),
                  'specification': SPEC, 'started_utc': utc(), 'predictions': [], 'annotation_payload_reads': 0, 'gt_rgb_payload_reads': 0,
                  'scene_renders': 0, 'optimizer_steps': 0, 'input_declarations': plan['input_declarations'], 'teacher': plan['teacher']}
    old_handler = signal.getsignal(signal.SIGALRM)
    def deadline(*_):
        raise TimeoutError('Fixed 270s internal deadline; no retry')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(270)
    try:
        os.chdir(plan['root'])
        torch.set_num_threads(8)
        cv2.setNumThreads(8)
        verify(plan)
        sys.path.insert(0, plan['source_snapshot'])
        official = importlib.import_module('bridge_rgs.official_evaluate')
        teacher = importlib.import_module('bridge_rgs.teacher')
        compare = importlib.import_module('compare_official_evaluations')
        report['actual_imports'] = imports(plan)
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], check=True, capture_output=True, text=True, timeout=5).stdout.strip()
        require(not gpu, f'GPU not idle: {gpu}')
        report['gpu_before'] = gpu
        checkpoint = torch.load(plan['teacher']['teacher_checkpoint'], map_location='cpu', weights_only=False)
        require(teacher.checkpoint_pixel_protocol(checkpoint) == LEGACY, 'Teacher profile changed')
        provenance = checkpoint['provenance']
        require(provenance['class_names'] == official.CLASS_NAMES, 'Teacher class IDs changed')
        backing = plan['teacher']['teacher_modelscope_backbone']
        require(provenance['model_dir'] == backing['model_dir'] and provenance['model_config_sha256'] == backing['config_sha256']
                and provenance['model_weights_sha256'] == backing['weights_sha256']
                and provenance.get('model_index_sha256', {}) == backing['index_sha256'], 'Backbone provenance differs')
        model = teacher.load_teacher(backing['model_dir'], 5, checkpoint['configuration']['channels'], device='cuda',
                    pixel_profile=LEGACY, **teacher.checkpoint_adapter_options(checkpoint['configuration']))
        model.decoder.load_state_dict(checkpoint['ema_decoder'], strict=True)
        teacher.load_checkpoint_adapters(model, checkpoint)
        model.eval().requires_grad_(False)
        report['teacher_parameters'] = sum(p.numel() for p in model.parameters())
        for arm in ARMS:
            (output/arm/'rgb').mkdir(parents=True)
            (output/arm/'mask').mkdir()
            for view in plan['views']:
                camera, inp = view['camera'], view['inputs'][arm]
                payload = Path(inp['path']).read_bytes()
                require(hashlib.sha256(payload).hexdigest() == inp['sha256'], 'Input RGB changed')
                rgb = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
                require(rgb is not None and rgb.dtype == np.uint8 and rgb.shape == (camera['height'], camera['width'], 3), 'Invalid RGB PNG')
                rgb = rgb[..., ::-1].copy()
                mx, my, back, boundary = adapter_maps(camera, official.distortion_render_grid)
                canvas = input_canvas(rgb, mx, my)
                probabilities, _ = teacher.predict_image(model, canvas, **INFERENCE)
                require(probabilities.shape == (5, *canvas.shape[:2]), 'Teacher canvas mismatch')
                mask = output_probabilities(probabilities, back).argmax(-1).astype(np.uint8)
                require(mask.shape == rgb.shape[:2], 'Output grid mismatch')
                rp, mp = output/arm/'rgb'/view['name'], output/arm/'mask'/view['name']
                with rp.open('xb') as stream:
                    stream.write(payload)
                require(cv2.imwrite(str(mp), mask), 'Mask write failed')
                report['predictions'].append({'arm': arm, 'name': view['name'], 'rgb': str(rp), 'rgb_sha256': sha(rp), 'mask': str(mp), 'mask_sha256': sha(mp), 'boundary': boundary})
        require(len(report['predictions']) == 100 and report['annotation_payload_reads'] == 0, 'Prediction-first contract failed')
        report['predictions_finished_utc'] = utc()
        write(output/'predictions_receipt.json', {'status': 'all_100_masks_complete_before_semantic_gt', 'predictions': report['predictions'],
                    'utc': report['predictions_finished_utc'], 'plan_sha256': sha(plan_path), 'annotation_payload_reads': 0, 'gt_rgb_payload_reads': 0})
        report['source_scoring_started_utc'] = utc()
        sources = []
        rows = {arm: [] for arm in ARMS}
        old_metrics = [read(plan[k]) for k in ('selected_metrics', 'composite_metrics')]
        rgb_rows = {a: {r['name']: r for r in m['views']} for a, m in zip(ARMS, old_metrics, strict=True)}
        pred = {(r['arm'], r['name']): r for r in report['predictions']}
        for view in plan['views']:
            camera = view['camera']
            truth = None
            if view['source_annotation_path'] is not None:
                content = Path(view['source_annotation_path']).read_bytes()
                report['annotation_payload_reads'] += 1
                require(hashlib.sha256(content).hexdigest() == view['source_annotation_sha256'], 'Annotation changed')
                truth = official.rasterize_official_annotation(content, camera['width'], camera['height'])
                require(hashlib.sha256(truth.tobytes(order='C')).hexdigest() == view['rasterized_mask_sha256'], 'GT rasterizer differs')
            sources.append(fingerprint_record(view))
            for arm in ARMS:
                p = pred[arm, view['name']]
                require(sha(p['rgb']) == view['inputs'][arm]['sha256'] and sha(p['mask']) == p['mask_sha256'], 'Prediction changed before scoring')
                old = rgb_rows[arm][view['name']]
                row = {k: old[k] for k in ('name', 'width', 'height', 'rgb_pixels', 'psnr', 'ssim', 'lpips')}
                if truth is not None:
                    mask = cv2.imread(p['mask'], cv2.IMREAD_UNCHANGED)
                    keep = truth != 255
                    cm = np.bincount(5*truth[keep].astype(np.int64)+mask[keep], minlength=25).reshape(5, 5)
                    row.update(confusion_matrix=cm.tolist(), semantic_pixels=int(keep.sum()), semantic_ignore_pixels=int((~keep).sum()))
                rows[arm].append(row)
        require(report['annotation_payload_reads'] == 41 and official.official_fingerprint(sources) == plan['reference_fingerprint'], 'Official support fingerprint differs')
        metrics = {}
        for arm in ARMS:
            pooled = sum(np.asarray(r['confusion_matrix'], np.int64) for r in rows[arm] if 'confusion_matrix' in r)
            iou = compare._iou(pooled)
            m = dict(evaluation_family=plan['evaluation_family'], official_evaluation_fingerprint=plan['reference_fingerprint'],
                scoring_protocol=plan['scoring_protocol'], inference_protocol=SPEC, validation_views=50, semantic_validation_views=41,
                views=rows[arm], confusion_matrix=pooled.tolist(), iou=iou.tolist(), miou_all=float(np.nanmean(iou)),
                miou_foreground=float(np.nanmean(iou[1:])), **{k: float(np.mean([r[k] for r in rows[arm]])) for k in ('psnr', 'ssim', 'lpips')},
                rgb_scoring='Bound previous per-view scores reused only after exact output PNG bytes; no new RGB GT reads or LPIPS',
                provenance=plan['input_declarations'][arm])
            compare._validate(m)
            write(output/arm/'official_metrics.json', m)
            metrics[arm] = m
        comparisons = {}
        for role, reference in [('same_adapter_control', metrics[ARMS[0]]), ('selected_complete_system', old_metrics[0])]:
            pair = compare.paired_official_comparison(reference, metrics[ARMS[1]], repeats=5000, seed=20260926)
            pair.update(reference_role=role, candidate_metrics_sha256=sha(output/ARMS[1]/'official_metrics.json'),
                        reference_metrics_sha256=sha(output/ARMS[0]/'official_metrics.json') if role == 'same_adapter_control' else sha(plan['selected_metrics']),
                        plan_sha256=sha(plan_path))
            path = output/f'paired_minus_{role}.json'
            write(path, pair)
            comparisons[role] = {'path': str(path), 'sha256': sha(path)}
        clauses = adoption_clauses(read(plan['prior_rgb_gate']), read(comparisons['selected_complete_system']['path']))
        gate = {'passed': all(clauses.values()), 'clauses': clauses, 'prior_rgb_gate_sha256': sha(plan['prior_rgb_gate']),
                    'comparison_sha256': comparisons['selected_complete_system']['sha256'], 'plan_sha256': sha(plan_path),
                    'consequence': 'Only proposes a multi-field RGB + pure 2D H+ joint candidate; no automatic adoption/training, no shared 3D semantic geometry claim'}
        write(output/'system_gate.json', gate)
        report.update(status='completed', finished_utc=utc(), comparisons=comparisons, gate_sha256=sha(output/'system_gate.json'),
                    gate_passed=gate['passed'], metrics_sha256={a: sha(output/a/'official_metrics.json') for a in ARMS},
                    official_evaluation_fingerprint=plan['reference_fingerprint'], actual_imports=imports(plan))
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        report['bound_inputs_and_sources_unchanged'] = source_files(Path(plan['source_snapshot'])) == plan['source_hashes'] and all(sha(p) == h for p, h in plan['input_hashes'].items())
        if not report['bound_inputs_and_sources_unchanged']:
            report['status'] = 'failed'
        report['elapsed_seconds_existing_png_teacher_transfer_and_scoring'] = time.monotonic()-started
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        write(output/'execution_receipt.json', report)
    require(report['status'] == 'completed', 'Input/source invariance failed')
    print(json.dumps({'status': report['status'], 'gate_passed': report['gate_passed'], 'receipt': str(output/'execution_receipt.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path)
    group.add_argument('--execute', type=Path)
    args = parser.parse_args()
    if args.prepare:
        path = prepare(args.prepare)
        print(json.dumps({'plan': str(path), 'plan_sha256': sha(path), 'runner_sha256': sha(__file__)}))
    else:
        execute(args.execute)


if __name__ == '__main__':
    main()
