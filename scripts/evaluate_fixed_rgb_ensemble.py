"""Fixed 1M/MCMC delivered-PNG RGB average; no rendering, training or semantic scoring."""
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
EVALUATIONS = {
    'capacity_1m': RUNS/'rgb_capacity_1m_reference_v1/evaluation_official',
    'mcmc': RUNS/'rgb_mcmc_reference_500k/evaluation_official',
    'reference_500k': RUNS/'ssim_fixed_corner_v2_rgb_full/evaluation_official',
}
SCORING_SOURCE = RUNS/'ssim_fixed_appearance_replay_v1/source_snapshot/bridge_rgs'
SCORER_SHA = '9b3799367a983495677122bf1bf75b06116342fde703466e43e3d8de092433f0'
CONTROL_SHA = '12bd34d0536da05f2a70a10f3b736312c679989c2a7fab3d42f84cae18d08ea8'
DOCUMENT = 'fixed_rgb_ensemble_protocol.md'
RGB_KEYS = ('psnr', 'ssim', 'lpips')
SPEC = {
    'protocol': 'fixed_1m_mcmc_delivered_uint8_rgb_mean_v1',
    'members': ['capacity_1m', 'mcmc'], 'weights': [.5, .5],
    'arithmetic': 'np.rint((uint8_A.astype(float32)+uint8_B.astype(float32))*.5).astype(uint8)',
    'grid': '50 common original-camera delivered RGB PNGs; not native/pinhole arrays',
    'color': 'quantized encoded RGB average, not linear-light radiance or unquantized renderer averaging',
    'views': 50, 'reference': 'reference_500k', 'bootstrap_repeats': 5000, 'bootstrap_seed': 20260926,
    'gate': {'psnr_gain_at_least': .15, 'psnr_ci_lower_strict': 0., 'ssim_gain_at_least': 0., 'lpips_gain_at_most': 0.},
    'prediction_first': 'Write/hash all 50 predictions before reading/hashing any target RGB bytes',
    'semantic_scoring': False, 'mask_outputs': False, 'training': False, 'scene_renders': 0,
    'internal_seconds': 200, 'external_seconds': 240, 'retry': False,
    'selection': 'Fixed default weights; no TRAIN selection, weight/member search or automatic semantic follow-up',
    'scope': 'Previously used development views; not blind test, joint-system adoption or a shared-geometry innovation',
    'timing_scope': 'Cached-PNG combination plus scoring only; not two-scene end-to-end rendering FPS',
    'known_structure_cost': {'retained_fields': 2, 'gaussians': 1496009, 'new_view_scene_renders': 2},
}


def require(condition, message):
    if not condition:
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


def source_files(snapshot):
    return {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.md'}}


def average_png(first, second):
    require(first.dtype == second.dtype == np.uint8 and first.shape == second.shape
            and first.ndim == 3 and first.shape[-1] == 3, 'Require matching uint8 RGB images')
    return np.rint((first.astype(np.float32)+second.astype(np.float32))*.5).astype(np.uint8)


def rgb_rows(metrics):
    rows = {v['name']: v for v in metrics['views']}
    require(len(rows) == len(metrics['views']) == 50, 'Exactly 50 unique original views required')
    for row in rows.values():
        require(all(np.isfinite(row[k]) for k in RGB_KEYS)
                and row['rgb_pixels'] == row['width']*row['height'], 'Invalid RGB values or image support')
    return rows


def paired_rgb(reference, candidate):
    """Only RGB means; the same sorted-view bootstrap as the existing scorer."""
    a, b = rgb_rows(reference), rgb_rows(candidate)
    require(set(a) == set(b), 'Different paired view names')
    names = sorted(a)
    require(all(all(a[n][k] == b[n][k] for k in ('width', 'height', 'rgb_pixels')) for n in names), 'Different paired grids')
    sample = np.random.default_rng(20260926).integers(50, size=(5000, 50))
    metrics = {}
    for key in RGB_KEYS:
        av, bv = [np.asarray([rows[n][key] for n in names], dtype=np.float64) for rows in (a, b)]
        differences = (bv-av)[sample].mean(1)
        metrics[key] = {'reference': float(av.mean()), 'candidate': float(bv.mean()),
                        'difference': float(bv.mean()-av.mean()),
                        'paired_view_bootstrap_95_interval': np.quantile(differences, [.025, .975]).tolist(),
                        'finite_bootstrap_replicates': 5000}
    return {'views': names, 'bootstrap_repeats': 5000, 'seed': 20260926, 'metrics': metrics,
            'difference_direction': 'candidate minus reference; LPIPS improves when negative',
            'scope': 'Paired reused development views, not selection-adjusted or blind-test evidence'}


def gate_clauses(metrics):
    require(set(metrics) == set(RGB_KEYS), 'RGB-only gate required')
    return {'psnr_gain_at_least_0_15_dB': metrics['psnr']['difference'] >= .15,
            'psnr_paired_95_lower_positive': metrics['psnr']['paired_view_bootstrap_95_interval'][0] > 0,
            'ssim_point_not_lower': metrics['ssim']['difference'] >= 0,
            'lpips_point_not_higher': metrics['lpips']['difference'] <= 0}


def score_rgb_only(scorer, prediction, target, perceptual, device):
    # This internal dummy meets the old API shape contract. No target mask is
    # supplied, so its semantic branch is unreachable; no dummy mask is saved.
    placeholder = np.zeros(prediction.shape[:2], dtype=np.uint8)
    values = scorer(prediction, placeholder, target, None, perceptual, device=device)
    require(set(values) == {*RGB_KEYS, 'rgb_pixels'}, 'Scorer unexpectedly returned semantic fields')
    return values


def completed_sources():
    receipts, datasets, bindings = {}, {}, {}
    require(sha(EVALUATIONS['reference_500k']/'official_metrics.json') == CONTROL_SHA, 'Fixed 500k reference changed')
    for role, folder in EVALUATIONS.items():
        receipt_path, metric_path = folder/'execution_receipt.json', folder/'official_metrics.json'
        receipt, metrics = read(receipt_path), read(metric_path)
        require(receipt['status'] == 'completed' and sha(metric_path) == receipt['official_metrics_sha256'], 'Incomplete/unbound old evaluation')
        require(receipt['predictions_finished_utc'] <= receipt['source_scoring_started_utc'], 'Old scoring order differs')
        require(receipt['official_evaluation_fingerprint'] == metrics['official_evaluation_fingerprint'], 'Old fingerprint differs')
        require(receipt['checkpoint_pixel_protocol']['id'] == 'colmap_corner_v2', 'Expected corner-v2 members/reference')
        rgb_rows(metrics)
        require(sha(receipt['checkpoint']) == receipt['checkpoint_sha256'], 'Completed model bytes changed')
        bindings.update({str(receipt_path): sha(receipt_path), str(metric_path): sha(metric_path),
                         receipt['checkpoint']: receipt['checkpoint_sha256']})
        for source in receipt['loaded_source_modules'].values():
            require(sha(source['path']) == source['sha256'], 'Old actual scoring source changed')
            bindings[source['path']] = source['sha256']
        receipts[role], datasets[role] = receipt, metrics
    reference = receipts['reference_500k']
    source_records = sorted(reference['source_records'], key=lambda row: row['name'])
    names = [row['name'] for row in source_records]
    require(len(names) == len(set(names)) == 50, 'Expected 50 source records')
    for role in EVALUATIONS:
        require(sorted(receipts[role]['source_records'], key=lambda row: row['name']) == source_records,
                'Member/reference camera or expected GT source metadata differs')
        require(datasets[role]['scoring_protocol'] == datasets['reference_500k']['scoring_protocol']
                and receipts[role]['official_evaluation_fingerprint'] == reference['official_evaluation_fingerprint'],
                'Common official protocol differs')
        require(sorted(rgb_rows(datasets[role])) == names, 'RGB metrics camera population differs')
    predictions = {role: {row['name']: row for row in receipts[role]['predictions']} for role in SPEC['members']}
    require(all(sorted(value) == names and len(value) == 50 for value in predictions.values()), 'Incomplete member RGB predictions')
    rows = []
    for source in source_records:
        name = source['name']
        members = {}
        for role in SPEC['members']:
            item = predictions[role][name]
            require(sha(item['rgb']) == item['rgb_sha256'], 'Member PNG changed')
            bindings[item['rgb']] = item['rgb_sha256']
            members[role] = {'path': item['rgb'], 'sha256': item['rgb_sha256']}
        rows.append({'name': name, 'camera': source['camera'], 'members': members,
                     'source_image_path': source['source_image_path'], 'source_image_sha256': source['source_image_sha256']})
    return receipts, datasets, bindings, rows


def prepare(output):
    output = Path(output).resolve()
    require(not output.exists() and not torch.cuda.is_initialized(), 'CPU-only, new output directory required')
    require(sha(SCORING_SOURCE/'official_evaluate.py') == SCORER_SHA, 'Known 500k scorer differs')
    receipts, datasets, bindings, views = completed_sources()
    # Bind cached LPIPS dependencies; never trigger a download during this run.
    lpips_origin = Path(importlib.util.find_spec('lpips').origin).parent
    perceptual_files = [lpips_origin/'weights/v0.1/alex.pth', Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth']
    for path in [*perceptual_files, ROOT/'uv.lock', ROOT/'docs'/DOCUMENT, Path(__file__).resolve(),
                 ROOT/'tests/test_fixed_rgb_ensemble.py']:
        bindings[str(path)] = sha(path)
    # Source RGB/annotation payloads are intentionally absent from this hash pass.
    require(not ({v['source_image_path'] for v in views} & bindings.keys()), 'GT accessed during preparation')
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(SCORING_SOURCE, snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    require(source_files(snapshot/'bridge_rgs') == source_files(SCORING_SOURCE), 'Scoring package copy differs')
    for path in [Path(__file__), ROOT/'docs'/DOCUMENT, ROOT/'tests/test_fixed_rgb_ensemble.py']:
        shutil.copy2(path, snapshot/path.name)
    original_reference = datasets['reference_500k']
    plan = {'status': 'locked_pending_root_gpu_handoff', 'specification': SPEC, 'root': str(ROOT), 'output': str(output),
            'source_snapshot': str(snapshot), 'source_hashes': source_files(snapshot), 'input_hashes': bindings,
            'scoring_source_origin': str(SCORING_SOURCE), 'scorer_sha256': SCORER_SHA, 'views': views,
            'original_evaluations': {role: {'directory': str(EVALUATIONS[role]),
                'receipt_sha256': sha(EVALUATIONS[role]/'execution_receipt.json'),
                'metrics_sha256': sha(EVALUATIONS[role]/'official_metrics.json'),
                'checkpoint': receipt['checkpoint'], 'checkpoint_sha256': receipt['checkpoint_sha256'],
                'checkpoint_pixel_protocol': receipt['checkpoint_pixel_protocol'], 'training_manifest': receipt['training_manifest']}
                for role, receipt in receipts.items()},
            'reference_fingerprint': original_reference['official_evaluation_fingerprint'],
            'original_evaluation_family': original_reference['evaluation_family'],
            'original_scoring_protocol': original_reference['scoring_protocol'],
            'fingerprint_scope': 'Inherited common camera/GT identity from completed receipts; this run re-verifies only source RGB bytes, never annotation/mask bytes',
            'perceptual_weights': [str(p) for p in perceptual_files], 'prepare_gt_payload_reads': 0,
            'prepare_cuda_initialized': False, 'external_timeout_seconds': 240,
            'env': {'PYTHONPATH': str(snapshot), 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '8',
                    'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'},
            'runtime_versions': receipts['reference_500k']['runtime_versions']}
    write(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    snapshot = Path(plan['source_snapshot'])
    require(plan['specification'] == SPEC and plan['status'] == 'locked_pending_root_gpu_handoff', 'Plan differs')
    require(Path(__file__).resolve() == snapshot/Path(__file__).name and source_files(snapshot) == plan['source_hashes'], 'Frozen source differs')
    require(all(sha(path) == value for path, value in plan['input_hashes'].items()), 'Bound source/input bytes differ')
    require({name: importlib.metadata.version(name) for name in plan['runtime_versions']} == plan['runtime_versions'], 'Scoring dependency versions differ')
    module = importlib.import_module('bridge_rgs.official_evaluate')
    require(sha(module.__file__) == SCORER_SHA, 'Actual scorer differs from fixed 500k scorer')
    imports = {}
    for name, value in list(sys.modules.items()):
        if name == 'bridge_rgs' or name.startswith('bridge_rgs.'):
            path = Path(value.__file__).resolve()
            require(path.is_relative_to(snapshot) and sha(path) == plan['source_hashes'][str(path.relative_to(snapshot))], 'Unfrozen scoring import')
            imports[name] = {'path': str(path), 'sha256': sha(path)}
    return module, imports


def decode_prediction(content, width, height):
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
    require(image is not None and image.dtype == np.uint8 and image.shape == (height, width, 3), 'Invalid member/output PNG')
    return image[..., ::-1].copy()


def execute(plan_path):
    started = time.monotonic()
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    output = Path(plan['output'])
    write(output/'execution_started.json', {'plan_sha256': sha(plan_path), 'pid': os.getpid(), 'utc': utc()})
    os.chdir(plan['root'])
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    report = {'status': 'failed', 'plan_sha256': sha(plan_path), 'specification': SPEC, 'started_utc': utc(),
              'gt_rgb_payload_reads': 0, 'annotation_payload_reads': 0, 'semantic_scoring': False,
              'scene_renders': 0, 'optimizer_steps': 0, 'mask_outputs': 0, 'predictions': []}
    old_handler = signal.getsignal(signal.SIGALRM)

    def deadline(*_):
        raise TimeoutError('Fixed 200s internal deadline; retain failure without retry')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(200)
    try:
        scorer, report['actual_imports'] = verify(plan)
        query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, check=True, timeout=5)
        require(not query.stdout.strip(), 'GPU compute occupied or query unavailable')
        report['gpu_before'] = {'compute_processes': [], 'query': query.args}
        rgb_dir = output/'rgb'
        rgb_dir.mkdir()
        for view in plan['views']:
            camera = view['camera']
            images = []
            for role in SPEC['members']:
                source = view['members'][role]
                payload = Path(source['path']).read_bytes()
                require(hashlib.sha256(payload).hexdigest() == source['sha256'], 'Member changed during generation')
                images.append(decode_prediction(payload, camera['width'], camera['height']))
            combined = average_png(*images)
            path = rgb_dir/view['name']
            require(Path(view['name']).name == view['name'], 'Unsafe view filename')
            ok, encoded = cv2.imencode('.png', combined[..., ::-1])
            require(ok, 'Could not encode average')
            with path.open('xb') as stream:
                stream.write(encoded.tobytes())
            report['predictions'].append({'name': view['name'], 'rgb': str(path), 'rgb_sha256': sha(path),
                                          'width': camera['width'], 'height': camera['height']})
        require(len(report['predictions']) == 50 and report['gt_rgb_payload_reads'] == 0, 'Incomplete prediction-first pass')
        report['predictions_finished_utc'] = utc()
        write(output/'predictions_receipt.json', {'status': 'all_50_predictions_complete_before_gt',
            'plan_sha256': sha(plan_path), 'predictions': report['predictions'], 'utc': report['predictions_finished_utc'],
            'gt_payload_reads': 0})
        perceptual = scorer._lpips('cuda')
        report['source_scoring_started_utc'] = utc()
        records = []
        for view, prediction in zip(plan['views'], report['predictions'], strict=True):
            # First target payload read/hash; all fifty output PNGs already durable.
            payload = Path(view['source_image_path']).read_bytes()
            report['gt_rgb_payload_reads'] += 1
            require(hashlib.sha256(payload).hexdigest() == view['source_image_sha256'], 'Target RGB differs from completed common reference')
            camera = view['camera']
            truth = scorer._decode_rgb(payload, camera['width'], camera['height'], 'common target RGB')
            delivered = Path(prediction['rgb']).read_bytes()
            require(hashlib.sha256(delivered).hexdigest() == prediction['rgb_sha256'], 'Candidate PNG changed')
            rgb = decode_prediction(delivered, camera['width'], camera['height'])
            values = score_rgb_only(scorer.score_official_arrays, rgb, truth, perceptual, 'cuda')
            records.append({'name': view['name'], 'width': camera['width'], 'height': camera['height'], **values,
                            'rgb_sha256': prediction['rgb_sha256'], 'source_rgb_sha256': view['source_image_sha256']})
        metrics = {'modality': 'RGB_only', 'evaluation_views': 50, 'views': records,
                   **{key: float(np.mean([row[key] for row in records])) for key in RGB_KEYS},
                   'inherited_common_reference_fingerprint': plan['reference_fingerprint'],
                   'fingerprint_scope': plan['fingerprint_scope'], 'scorer_sha256': SCORER_SHA}
        write(output/'rgb_metrics.json', metrics)
        pairs = {}
        for role, old in plan['original_evaluations'].items():
            reference = read(Path(old['directory'])/'official_metrics.json')
            pair = paired_rgb(reference, metrics)
            pair.update(reference_role=role, reference_metrics_sha256=old['metrics_sha256'],
                        candidate_metrics_sha256=sha(output/'rgb_metrics.json'), plan_sha256=sha(plan_path))
            path = output/f'paired_minus_{role}.json'
            write(path, pair)
            pairs[role] = {'path': str(path), 'sha256': sha(path)}
        primary = read(pairs['reference_500k']['path'])
        clauses = gate_clauses(primary['metrics'])
        gate = {'passed': all(clauses.values()), 'clauses': clauses, 'primary_reference': 'reference_500k',
                'primary_pair_sha256': pairs['reference_500k']['sha256'], 'plan_sha256': sha(plan_path),
                'no_automatic_semantic_training_or_joint_adoption': True}
        write(output/'rgb_gate.json', gate)
        report.update(status='completed', metrics_sha256=sha(output/'rgb_metrics.json'), comparisons=pairs,
                      gate_sha256=sha(output/'rgb_gate.json'), gate_passed=gate['passed'], finished_utc=utc())
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        report['bound_inputs_and_sources_unchanged'] = (source_files(Path(plan['source_snapshot'])) == plan['source_hashes']
            and all(sha(path) == value for path, value in plan['input_hashes'].items()))
        if not report['bound_inputs_and_sources_unchanged']:
            report['status'] = 'failed'
        report['elapsed_seconds_cached_png_combination_and_scoring_only'] = time.monotonic()-started
        report['peak_cuda_allocated_bytes_scoring_only'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        write(output/'execution_receipt.json', report)
    require(report['status'] == 'completed', 'Failed source/input invariance')
    print(json.dumps({'status': report['status'], 'gate_passed': report['gate_passed'],
                      'receipt': str(output/'execution_receipt.json'), 'sha256': sha(output/'execution_receipt.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path)
    group.add_argument('--execute', type=Path)
    args = parser.parse_args()
    if args.prepare:
        plan = prepare(args.prepare)
        print(json.dumps({'plan': str(plan), 'plan_sha256': sha(plan), 'runner_sha256': sha(__file__)}))
    else:
        execute(args.execute)


if __name__ == '__main__':
    main()
