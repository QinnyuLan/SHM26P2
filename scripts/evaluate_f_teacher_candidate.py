"""Evaluate the pre-registered F + selected-2k DINOv3 semantic candidate.

F RGB PNGs and the frozen H3 soft readout are inputs.  The target annotations
are opened only after all 50 candidate masks have been written.  No model or
RGB pixels are optimized in this script.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
F_RUN = RUNS/'ibgs_joint_replacement_v2'
MULTI = RUNS/'multifield_h3_teacher_v1'
MATCHED = RUNS/'matched_rgb_teacher_adaptation_v1'
SNAPSHOT = MATCHED/'evaluation_v2/source_snapshot'
OUT = RUNS/'f_semantic_teacher_candidate_v1'
TEACHER = MATCHED/'selected/last.pt'
MODEL_DIR = ROOT/'models/dinov3-vith16plus'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def import_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iou(matrix):
    matrix = np.asarray(matrix)
    diag = np.diagonal(matrix, axis1=-2, axis2=-1).astype(np.float64)
    union = matrix.sum(-1) + matrix.sum(-2) - diag
    return np.divide(diag, union, out=np.full_like(diag, np.nan, dtype=np.float64), where=union > 0)


def paired(reference, candidate, repeats=5000, seed=20260926):
    names = sorted(reference['views'])
    assert names == sorted(candidate['views'])
    rng = np.random.default_rng(seed)
    sample = rng.integers(len(names), size=(repeats, len(names)))
    result = {'views': names, 'semantic_views': names, 'bootstrap_repeats': repeats,
              'seed': seed, 'difference_direction': 'candidate minus F; IoU/mIoU higher is better',
              'metrics': {}}

    def add(key, ref, cand, diff):
        result['metrics'][key] = {
            'reference': float(ref), 'candidate': float(cand), 'difference': float(cand-ref),
            'paired_view_bootstrap_95_interval': np.quantile(diff, [.025, .975]).tolist(),
            'finite_bootstrap_replicates': int(np.isfinite(diff).sum())}

    # Semantic scores are pooled over each sampled set of annotated views.
    ref_m = np.asarray([reference['views'][n]['confusion_matrix'] for n in names], np.int64)
    cand_m = np.asarray([candidate['views'][n]['confusion_matrix'] for n in names], np.int64)
    ref_i = iou(ref_m.sum(0)); cand_i = iou(cand_m.sum(0))
    ref_s = iou(ref_m[sample].sum(1)); cand_s = iou(cand_m[sample].sum(1))
    add('miou_all', np.nanmean(ref_i), np.nanmean(cand_i), np.nanmean(cand_s, 1)-np.nanmean(ref_s, 1))
    add('miou_foreground', np.nanmean(ref_i[1:]), np.nanmean(cand_i[1:]),
        np.nanmean(cand_s[:, 1:], 1)-np.nanmean(ref_s[:, 1:], 1))
    for index, name in enumerate(('background', 'deck', 'stay_cable', 'tower', 'foundation')):
        add(name+'_iou', ref_i[index], cand_i[index], cand_s[:, index]-ref_s[:, index])
    return result


def main():
    require(not OUT.exists(), 'Refuse to overwrite candidate output')
    OUT.mkdir(parents=True)
    multi_plan = read(MULTI/'plan.json')
    fmetrics = read(F_RUN/'official_metrics.json')
    fviews = {v['name']: v for v in multi_plan['views']}
    names = sorted(fviews)
    require(len(names) == 50 and sum(v['source_annotation_path'] is not None for v in fviews.values()) == 41,
            'Require fixed 50/41 population')
    # Bind all non-label payloads before the GPU stage.
    bindings = {}
    for p in (F_RUN/'plan.json', F_RUN/'render/main_execution_receipt.json',
              F_RUN/'render/rgb', MULTI/'plan.json', TEACHER,
              MODEL_DIR/'config.json', MODEL_DIR/'model.safetensors',
              SNAPSHOT/'rgb_teacher_transfer_base.py', SNAPSHOT/'bridge_rgs/teacher.py'):
        if Path(p).is_file():
            bindings[str(Path(p).resolve())] = sha(p)
    for name in names:
        rgb = F_RUN/'render/rgb'/name
        h3 = MULTI/'soft/A'/f'{Path(name).stem}.npy'
        require(rgb.is_file() and h3.is_file(), f'Missing frozen F/H3 payload {name}')
        bindings[str(rgb.resolve())] = sha(rgb)
        bindings[str(h3.resolve())] = sha(h3)
    snapshot = str(SNAPSHOT)
    base = import_from(SNAPSHOT/'rgb_teacher_transfer_base.py', 'f_semantic_base')
    # Import the frozen package from the matched evaluation snapshot.
    import sys
    sys.path.insert(0, snapshot)
    official = importlib.import_module('bridge_rgs.official_evaluate')
    teacher_mod = importlib.import_module('bridge_rgs.teacher')
    with torch.inference_mode():
        state = torch.load(TEACHER, map_location='cpu', weights_only=False)
        profile = state['pixel_protocol']
        model = teacher_mod.load_teacher(MODEL_DIR, 5, state['configuration']['channels'],
                                         device='cuda', pixel_profile=profile,
                                         **teacher_mod.checkpoint_adapter_options(state['configuration']))
        model.decoder.load_state_dict(state['ema_decoder'], strict=True)
        teacher_mod.load_checkpoint_adapters(model, state)
        model.eval().requires_grad_(False)
        records = []
        # Prediction barrier: no annotation bytes are read in this loop.
        for name in names:
            view = fviews[name]
            rgb_bgr = cv2.imread(str(F_RUN/'render/rgb'/name), cv2.IMREAD_COLOR)
            require(rgb_bgr is not None and rgb_bgr.dtype == np.uint8, 'Invalid F RGB')
            rgb = rgb_bgr[..., ::-1].copy()
            mx, my, back, adapter = base.adapter_maps(view['camera'], official.distortion_render_grid)
            canvas = base.input_canvas(rgb, mx, my)
            prediction, _ = teacher_mod.predict_image(model, canvas, tile_size=768, stride=512,
                flip=True, context_weight=.25, context_short_side=768)
            h3 = np.load(MULTI/'soft/A'/f'{Path(name).stem}.npy', allow_pickle=False)
            teacher_legacy = np.ascontiguousarray(prediction.transpose(1, 2, 0), dtype=np.float32)
            require(h3.dtype == np.float32 and h3.shape == teacher_legacy.shape, 'H3/teacher grid mismatch')
            soft_legacy = np.ascontiguousarray(.5*h3 + .5*teacher_legacy, dtype=np.float32)
            soft = base.output_probabilities(soft_legacy.transpose(2, 0, 1), back)
            mask = soft.argmax(-1).astype(np.uint8)
            path = OUT/'mask'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            require(cv2.imwrite(str(path), mask), 'Mask write failed')
            records.append({'name': name, 'mask': str(path), 'mask_sha256': sha(path),
                            'teacher_soft_sha256': hashlib.sha256(soft.tobytes()).hexdigest(),
                            'teacher_grid': [int(soft.shape[1]), int(soft.shape[0])], 'adapter': adapter})
        require(len(records) == 50, 'Prediction barrier incomplete')
        del model
        torch.cuda.empty_cache()
    write(OUT/'prediction_receipt.json', {'status': 'all_50_candidate_masks_before_gt',
        'records': records, 'annotation_reads': 0, 'F_rgb_source': str(F_RUN/'render/rgb')})

    # Only now read the 41 label payloads.
    per_view = {}
    f_per = {v['name']: v for v in fmetrics['views']}
    for name in names:
        view = fviews[name]
        row = {'name': name, 'width': view['camera']['width'], 'height': view['camera']['height'],
               'rgb_pixels': view['camera']['width']*view['camera']['height']}
        if view['source_annotation_path'] is not None:
            annotation = Path(view['source_annotation_path']).read_bytes()
            require(sha(view['source_annotation_path']) == view['source_annotation_sha256'], 'Annotation changed')
            truth = official.rasterize_official_annotation(annotation, row['width'], row['height'])
            pred = cv2.imread(str(OUT/'mask'/name), cv2.IMREAD_UNCHANGED)
            keep = truth != 255
            matrix = np.bincount(5*truth[keep].astype(np.int64)+pred[keep].astype(np.int64), minlength=25).reshape(5,5)
            row.update(confusion_matrix=matrix.tolist(), semantic_pixels=int(keep.sum()),
                       semantic_ignore_pixels=int((~keep).sum()))
        per_view[name] = row
    annotated = [n for n in names if 'confusion_matrix' in per_view[n]]
    matrices = np.asarray([per_view[n]['confusion_matrix'] for n in annotated], np.int64)
    pooled = matrices.sum(0); values = iou(pooled)
    candidate = {'views': names, 'semantic_views': 41, 'per_view': per_view,
        'confusion_matrix': pooled.tolist(), 'iou': values.tolist(),
        'miou_all': float(np.nanmean(values)), 'miou_foreground': float(np.nanmean(values[1:])),
        'F_reference_metrics_sha256': sha(F_RUN/'official_metrics.json'),
        'teacher_checkpoint_sha256': sha(TEACHER), 'prediction_receipt_sha256': sha(OUT/'prediction_receipt.json')}
    write(OUT/'candidate_metrics.json', candidate)
    reference = {'views': {n: {'confusion_matrix': np.asarray(f_per[n]['confusion_matrix']).tolist()} for n in annotated}}
    cand_for_pair = {'views': {n: per_view[n] for n in annotated}}
    pair = paired(reference, cand_for_pair)
    pair.update(reference='F', candidate='selected_2k_teacher_on_F_RGB', candidate_metrics_sha256=sha(OUT/'candidate_metrics.json'))
    write(OUT/'paired_candidate_minus_F.json', pair)
    gates = pair['metrics']
    gate = {'miou_gain_at_least_0_20pp': gates['miou_all']['difference'] >= .002,
            'miou_ci_lower_positive': gates['miou_all']['paired_view_bootstrap_95_interval'][0] > 0,
            'cable_gain_at_least_minus_0_10pp': gates['stay_cable_iou']['difference'] >= -.001,
            'rgb_unchanged_by_construction': True}
    write(OUT/'candidate_gate.json', {'passed': bool(all(gate.values())), 'clauses': gate,
        'pair_sha256': sha(OUT/'paired_candidate_minus_F.json')})
    write(OUT/'execution_receipt.json', {'status': 'completed', 'natural_completion': True,
        'prediction_count': 50, 'annotation_reads': 41, 'gt_reads_after_prediction_barrier': True,
        'input_bindings': bindings, 'candidate_metrics_sha256': sha(OUT/'candidate_metrics.json'),
        'pair_sha256': sha(OUT/'paired_candidate_minus_F.json'), 'gate_sha256': sha(OUT/'candidate_gate.json')})
    print(json.dumps({'candidate': candidate['miou_all'], 'F': fmetrics['miou_all'], 'gate': gate}, indent=2))


if __name__ == '__main__':
    main()
