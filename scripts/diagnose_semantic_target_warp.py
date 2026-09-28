"""One CPU diagnostic of prepared hard-target / official soft-warp disagreement.

No model is loaded. No candidate is optimized or selected. The fixed cached S
predictions preceded this diagnostic and its TRAIN-only target reads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
PRIOR = Path('/mnt/data/SHM2026/runs/cross_renderer_authority_v1')
MANIFEST = ROOT/'artifacts/prepared/manifest.json'
NAMES = [f'{i:03}.png' for i in (2, 21, 41, 59, 79, 100, 118, 137, 156, 176, 200, 220, 241, 259, 278, 300)]
DOCUMENT = 'semantic_target_warp_protocol.md'
SPEC = {'protocol': 'semantic_target_warp_v1', 'names': NAMES, 'profile': 'legacy_mixed_v1',
        'prediction': 'fixed cached H3 S; exact frozen official bilinear W then argmax',
        'target_roundtrip': 'argmax W(onehot(prepared hard mask)), invalid/unknown source rows zero',
        'support': 'original target known AND every strictly-positive bilinear tap in-bounds and native valid/known',
        'boundary': 'class-specific binary dilation-minus-erosion, Chebyshev radius3, replicate image border; known full support only',
        'oracle': 'Replace current prediction by original GT only where current error AND target_roundtrip differs from original GT',
        'primary': 'Descriptive counts/CM/intersection/oracle on common support, per view and pooled; no adoption gate',
        'gradient_diagnostic': False, 'model_loads': 0, 'renders': 0, 'optimizer_steps': 0, 'gpu_calls': 0,
        'internal_seconds': 180, 'external_seconds': 240, 'retries': 0,
        'limits': 'TRAIN in-fit predictions; intersection does not prove causation; roundtrip target is not a model accuracy ceiling; not official full-image evaluation or innovation'}
VERSIONS = ('numpy', 'opencv-python-headless', 'pillow', 'torch', 'scikit-image')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def files(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def fixed_views(manifest):
    labeled = sorted((v for v in manifest['views'] if v['split'] == 'train' and v.get('mask_path')), key=lambda v: v['name'])
    require(len(labeled) == 259, 'Need the original 259 labeled TRAIN views')
    selected = [labeled[i*258//15] for i in range(16)]
    require([v['name'] for v in selected] == NAMES, 'Fixed TRAIN selection changed')
    return selected


def full_bilinear_support(native_known, back):
    """All nonzero mathematical taps must be present; zero-weight outside taps do not count."""
    native_known = np.asarray(native_known, bool)
    if back is None:
        return native_known.copy()
    require(back.ndim == 3 and back.shape[-1] == 2 and np.isfinite(back).all(), 'Invalid official map')
    x, y = back[..., 0].astype(np.float64), back[..., 1].astype(np.float64)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    dx, dy = x-x0, y-y0
    height, width = native_known.shape
    result = np.ones(x.shape, bool)
    for ox, oy, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                           (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        xx, yy = x0+ox, y0+oy
        inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        good = np.zeros_like(result)
        good[inside] = native_known[yy[inside], xx[inside]]
        result &= (weight == 0) | good
    return result


def warp(values, back):
    values = np.ascontiguousarray(values, dtype=np.float32)
    return values.copy() if back is None else cv2.remap(values, back[..., 0], back[..., 1],
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def class_boundary(target, category):
    binary = (target == category).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    return (cv2.dilate(binary, kernel, borderType=cv2.BORDER_REPLICATE) !=
            cv2.erode(binary, kernel, borderType=cv2.BORDER_REPLICATE))


def confusion(target, pred, keep):
    return np.bincount(target[keep].astype(np.int64)*5+pred[keep], minlength=25).reshape(5, 5)


def iou(matrix):
    m = np.asarray(matrix, np.float64)
    union = m.sum(0)+m.sum(1)-np.diag(m)
    values = np.divide(np.diag(m), union, out=np.zeros(5), where=union > 0)
    present = union > 0
    return {'iou': [float(v) if p else None for v, p in zip(values, present, strict=True)],
            'miou_all': float(values[present].mean()) if present.any() else None,
            'miou_foreground': float(values[1:][present[1:]].mean()) if present[1:].any() else None}


def measures(target, prediction, target_roundtrip, keep):
    mismatch, error = target_roundtrip != target, prediction != target
    intersection = mismatch & error & keep
    oracle = prediction.copy()
    oracle[intersection] = target[intersection]
    counts = {'support_pixels': int(keep.sum()), 'target_disagreement': int((mismatch & keep).sum()),
              'current_errors': int((error & keep).sum()), 'error_intersection': int(intersection.sum()),
              'current_correct_despite_target_disagreement': int((mismatch & ~error & keep).sum())}
    counts['by_gt_class'] = [{key: int((mask & (target == c)).sum()) for key, mask in
            [('support_pixels', keep), ('target_disagreement', mismatch & keep), ('current_errors', error & keep),
             ('error_intersection', intersection)]} for c in range(5)]
    matrices = {'actual': confusion(target, prediction, keep).tolist(),
                'target_roundtrip': confusion(target, target_roundtrip, keep).tolist(),
                'oracle_intersection_only': confusion(target, oracle, keep).tolist()}
    return finish_measures(counts, matrices)


def finish_measures(counts, matrices):
    scores = {key: iou(value) for key, value in matrices.items()}
    actual, oracle = scores['actual']['miou_all'], scores['oracle_intersection_only']['miou_all']
    return {**counts, 'confusion_matrices': matrices, 'scores': scores,
            'target_disagreement_fraction': counts['target_disagreement']/counts['support_pixels'] if counts['support_pixels'] else None,
            'fraction_of_current_errors_in_intersection': counts['error_intersection']/counts['current_errors'] if counts['current_errors'] else None,
            'oracle_miou_all_gain': oracle-actual if actual is not None else None}


def pooled_measures(records):
    scalar_keys = ('support_pixels', 'target_disagreement', 'current_errors', 'error_intersection',
                   'current_correct_despite_target_disagreement')
    counts = {key: sum(r[key] for r in records) for key in scalar_keys}
    counts['by_gt_class'] = [{key: sum(r['by_gt_class'][c][key] for r in records)
                             for key in records[0]['by_gt_class'][c]} for c in range(5)]
    matrices = {key: sum((np.asarray(r['confusion_matrices'][key], np.int64) for r in records),
                         np.zeros((5, 5), np.int64)).tolist() for key in records[0]['confusion_matrices']}
    return finish_measures(counts, matrices)


def view_measurements(probabilities, native_mask, native_valid, original_target, back):
    require(probabilities.dtype == np.float32 and probabilities.shape == (*native_mask.shape, 5), 'Wrong cached probability grid/dtype')
    require(np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1+5e-6).all()
            and np.allclose(probabilities.sum(-1), 1., rtol=0, atol=5e-6), 'Invalid cached S probabilities')
    require(native_valid.shape == native_mask.shape and np.isin(native_mask, [0, 1, 2, 3, 4, 255]).all(), 'Invalid native targets')
    require(np.isin(original_target, [0, 1, 2, 3, 4, 255]).all(), 'Invalid original targets')
    native_known = (native_valid > 0) & (native_mask < 5)
    onehot = np.eye(5, dtype=np.float32)[np.minimum(native_mask, 4)]*native_known[..., None]
    mapped = warp(probabilities, back)
    mapped_target = warp(onehot, back)
    require(mapped.shape[:2] == original_target.shape, 'Original grid differs')
    footprint = full_bilinear_support(native_known, back)
    support = footprint & (original_target < 5)
    require(np.all(mapped.sum(-1)[support] > 0) and np.all(mapped_target.sum(-1)[support] > 0), 'Missing supported probability mass')
    prediction, target_rt = mapped.argmax(-1).astype(np.uint8), mapped_target.argmax(-1).astype(np.uint8)
    regions = {'all': measures(original_target, prediction, target_rt, support)}
    for c in range(5):
        boundary = class_boundary(original_target, c)
        regions[f'class_{c}_boundary3'] = measures(original_target, prediction, target_rt, support & boundary)
        regions[f'class_{c}_nonboundary'] = measures(original_target, prediction, target_rt, support & ~boundary)
    return {'original_pixels': int(original_target.size), 'original_known_pixels': int((original_target < 5).sum()),
            'excluded_incomplete_or_invalid_native_footprint': int(((original_target < 5) & ~footprint).sum()),
            'support_fraction_of_original_known': float(support.sum()/(original_target < 5).sum()) if (original_target < 5).any() else None,
            'native_actual_confusion_matrix': confusion(native_mask, probabilities.argmax(-1), native_known).tolist(),
            'native_actual_scores': iou(confusion(native_mask, probabilities.argmax(-1), native_known)),
            'regions': regions}


def load_official(snapshot):
    require('bridge_rgs.official_evaluate' not in sys.modules, 'Official package already imported')
    sys.path.insert(0, str(snapshot))
    official = importlib.import_module('bridge_rgs.official_evaluate')
    require(not official.torch.cuda.is_initialized(), 'CPU only')
    return official


def prepare(output):
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite a diagnostic')
    prior, receipt, launch = (read(PRIOR/name) for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json'))
    declaration = read(PRIOR/'analysis_input.json')
    require(receipt['status'] == launch['status'] == 'completed' and launch['exit_code'] == 0
            and launch['natural_completion'] and receipt['inputs_and_sources_unchanged'], 'Cached prediction run did not complete naturally')
    require(receipt['plan_sha256'] == sha(PRIOR/'plan.json') == launch['plan_sha256']
            and launch['execution_receipt_sha256'] == sha(PRIOR/'execution_receipt.json')
            and declaration['prediction_receipt']['sha256'] == sha(PRIOR/'execution_receipt.json'), 'Cached provenance differs')
    inputs = {}
    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or expected == digest, f'Changed bound input: {path}')
        inputs[str(path)] = digest
        return {'path': str(path), 'sha256': digest}
    for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'analysis_input.json', 'independent_cpu_review.json'):
        bind(PRIOR/name)
    require(read(PRIOR/'independent_cpu_review.json')['status'] == 'passed', 'Cached numerical audit failed')
    bind(MANIFEST, prior['input_hashes'][str(MANIFEST)])
    bind(ROOT/'uv.lock')
    manifest = read(MANIFEST)
    views = fixed_views(manifest)
    rows_by_name = {v['name']: v for v in prior['views']}
    cached = {v['name']: v for v in declaration['views']}
    predictions = {v['name']: v for v in receipt['predictions']}
    rows = []
    for view in views:
        record = cached[view['name']]
        require(record['split'] == 'train' and record['S'] == predictions[view['name']]['S'], 'Wrong cached S identity')
        camera = rows_by_name[view['name']]['camera']
        require(camera['split'] == 'train' and np.array_equal(camera['w2c'], view['w2c_original']), 'Wrong TRAIN camera')
        require(record['adapter_info']['canvas'] == [view['width'], view['height']]
                and np.array_equal(np.asarray(record['adapter_info']['canvas_K'], np.float32), np.asarray(view['K'], np.float32)), 'Cached canvas calibration differs')
        row = {'name': view['name'], 'split': 'train', 'camera': camera,
               'native_K': view['K'], 'native_size': [view['width'], view['height']],
               'S': bind(record['S']['path'], record['S']['sha256'])}
        for role in ('mask', 'valid'):
            require(Path(record[role]['path']).resolve() == Path(view[role+'_path']).resolve(), 'Target path changed')
            row[role] = bind(record[role]['path'], record[role]['sha256'])
        row['annotation'] = bind(view['source_annotation_path'])
        rows.append(row)
    old_snapshot = Path(prior['source_snapshot'])
    require(files(old_snapshot) == prior['source_hashes'], 'Cached source tree changed')
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(old_snapshot/'bridge_rgs', snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    package_hashes = files(snapshot/'bridge_rgs')
    for relative, digest in package_hashes.items():
        require(digest == prior['source_hashes']['bridge_rgs/'+relative], 'Copied source changed')
    for source in (Path(__file__), ROOT/'tests/test_semantic_target_warp.py', ROOT/'docs'/DOCUMENT):
        shutil.copy2(source, snapshot/source.name)
    plan = {'specification': SPEC, 'output': str(output), 'source_snapshot': str(snapshot),
            'source_hashes': files(snapshot), 'input_hashes': inputs, 'views': rows,
            'prior_source_package': str(old_snapshot/'bridge_rgs'), 'prior_source_package_hashes': package_hashes,
            'runtime_versions': {name: importlib.metadata.version(name) for name in VERSIONS},
            'thread_environment': {'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
            'prepare_label_decodes': 0, 'prepare_probability_array_loads': 0,
            'prepare_target_bytes_hashed': True, 'status': 'locked_pending_root_review',
            'created_utc': datetime.now(UTC).isoformat()}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json'),
                      'sources': len(plan['source_hashes']), 'inputs': len(inputs)}))


def verify(plan):
    require(plan['specification'] == SPEC, 'Specification changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Execute frozen worker')
    require(files(Path(plan['source_snapshot'])) == plan['source_hashes'], 'Source tree changed')
    for name, value in plan['thread_environment'].items():
        require(os.environ.get(name) == value, f'Unexpected CPU threads: {name}')
    for name, value in plan['runtime_versions'].items():
        require(importlib.metadata.version(name) == value, f'Runtime changed: {name}')
    for path, digest in plan['input_hashes'].items():
        require(sha(path) == digest, f'Bound input changed: {path}')


def execute(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    output, snapshot = Path(plan['output']), Path(plan['source_snapshot'])
    require(plan_path == output/'plan.json', 'Wrong plan location')
    require(not (output/'execution_started.json').exists(), 'One execution only; never retry')
    verify(plan)
    official = load_official(snapshot)
    imports = {name: {'path': str(Path(module.__file__).resolve()), 'sha256': sha(module.__file__)}
               for name, module in sys.modules.items() if name.startswith('bridge_rgs') and getattr(module, '__file__', None)}
    for record in imports.values():
        path = Path(record['path'])
        require(path.is_relative_to(snapshot) and plan['source_hashes'][str(path.relative_to(snapshot))] == record['sha256'], 'Non-frozen imported package')
    started = {'plan_sha256': sha(plan_path), 'started_utc': datetime.now(UTC).isoformat(), 'pid': os.getpid()}
    write(output/'execution_started.json', started)
    status = {**started, 'status': 'running', 'actual_imports': imports, 'model_loads': 0, 'renders': 0,
              'gpu_calls': 0, 'optimizer_steps': 0, 'label_image_decodes': 0, 'annotation_rasterizations': 0,
              'val_target_reads': 0, 'cached_probability_arrays_loaded': 0}
    beginning = time.perf_counter()
    failure = None
    def alarm(_signum, _frame):
        raise TimeoutError('Fixed CPU diagnostic deadline exceeded')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(SPEC['internal_seconds'])
    try:
        results = []
        for view in plan['views']:
            require(view['split'] == 'train' and view['name'] in NAMES, 'Non-fixed TRAIN target')
            camera = view['camera']
            K, width, height, back = official.distortion_render_grid(camera['K'], camera['distortion'],
                                                    camera['width'], camera['height'], pixel_protocol=SPEC['profile'])
            require([width, height] == view['native_size'] and np.array_equal(K, np.asarray(view['native_K'], np.float32)),
                    'Official warp canvas is not the cached S canvas; no silent crop/pad')
            probabilities = np.load(view['S']['path'], allow_pickle=False)
            status['cached_probability_arrays_loaded'] += 1
            native = cv2.imread(view['mask']['path'], cv2.IMREAD_UNCHANGED)
            valid = cv2.imread(view['valid']['path'], cv2.IMREAD_UNCHANGED)
            status['label_image_decodes'] += 2
            require(native is not None and valid is not None and native.dtype == valid.dtype == np.uint8,
                    'Invalid prepared target PNG')
            target = official.rasterize_official_annotation(Path(view['annotation']['path']).read_bytes(), camera['width'], camera['height'])
            status['annotation_rasterizations'] += 1
            result = view_measurements(probabilities, native, valid, target, back)
            results.append({'name': view['name'], 'split': 'train', **result})
            print(json.dumps({'name': view['name'], 'support': result['regions']['all']['support_pixels'],
                              'target_disagreement': result['regions']['all']['target_disagreement'],
                              'error_intersection': result['regions']['all']['error_intersection']}), flush=True)
        pooled = {region: pooled_measures([r['regions'][region] for r in results]) for region in results[0]['regions']}
        report = {'plan_sha256': sha(plan_path), 'specification': SPEC, 'views': results, 'pooled': pooled,
                  'per_view_equal_weight': {key: float(np.mean([r['regions']['all'][key] for r in results])) for key in
                       ('target_disagreement_fraction', 'fraction_of_current_errors_in_intersection', 'oracle_miou_all_gain')
                       if all(r['regions']['all'][key] is not None for r in results)},
                  'decision': 'Descriptive necessity evidence only; no model adoption or automatic training permission',
                  'limits': ['Roundtrip label disagreement is not a model accuracy upper bound.',
                             'The intersection-only oracle repairs only recorded current mistakes inside that discrepancy set; it is not a deployable method.',
                             'Prepared TRAIN probabilities were fitted on these views; this does not establish VAL benefit or explain all boundary errors.',
                             'The CE gradient extension was not run; neither target causality nor parameter-update benefit is measured.',
                             'Common-support metrics are not the official full-image scores.']}
        write(output/'analysis.json', report)
        verify(plan)
        require(not official.torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
        status.update(status='completed', inputs_and_sources_unchanged=True, output_files={'analysis.json': sha(output/'analysis.json')},
                      cuda_initialized=False)
    except BaseException as exc:  # noqa: BLE001 - preserve the unique failed attempt
        failure = exc
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.alarm(0)
        status.update(elapsed_seconds=time.perf_counter()-beginning, ended_utc=datetime.now(UTC).isoformat())
        write(output/'execution_receipt.json', status)
    if failure is not None:
        raise failure
    print(json.dumps({'status': status['status'], 'analysis_sha256': status['output_files']['analysis.json'],
                      'elapsed_seconds': status['elapsed_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare')
    choice.add_argument('--execute')
    args = parser.parse_args()
    prepare(args.prepare) if args.prepare else execute(args.execute)


if __name__ == '__main__':
    main()
