"""Independent CPU audit of four completed warm-IBGS official RGB endpoints.

No producer/scorer imports, Torch, model load, annotation decode or renderer.
OpenCV supplies the same documented camera-inversion/interpolation and PNG
codec operations; the map construction, quantization and statistics are local.
SSIM/LPIPS scores are bound observations, not independently recomputed pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import signal
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

ARMS = ('full', 'no_source')
READOUTS = ('raw', 'fused')
KEYS = ('psnr', 'ssim', 'lpips')
SCORER_SHA = '9b3799367a983495677122bf1bf75b06116342fde703466e43e3d8de092433f0'
GRID_SHA = '5c180581ad680f76a708f73abcc5850dad0ec6f4c3aa25c664bd9d0d217e2b2f'
WRAPPER_SHA = '041a40203529db626a42fca92098fde8b8bf3eb70dafae799b436396104bc36b'
REFERENCES = {
    'aa_1m': '0b4b90fca81ad4f220794b9656ce308cca4dc68a8efe83769efbfde2cd51100e',
    'E_rgb': '40d022bc522315eaa864fa114261f6e66c5ca6991b3de74bf309a3a4e2ad2ef9',
}
WEIGHTS = {
    'alex.pth': 'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0',
    'alexnet-owt-7be5be79.pth': '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
}


def need(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound(path, expected):
    need(sha(path) == expected, 'Changed bound file: '+str(path))


def payload(path, expected):
    value = Path(path).read_bytes()
    need(hashlib.sha256(value).hexdigest() == expected, 'Changed payload: '+str(path))
    return value


def natural(run, stage):
    receipt_path = run/f'{stage}_execution_receipt.json'
    execution, launch = read(receipt_path), read(run/f'{stage}_launch_receipt.json')
    need(execution['status'] == launch['status'] == 'completed'
         and launch['natural_completion'] is True and launch['exit_code'] == 0,
         'Natural completed render and score are required before payload access')
    bound(receipt_path, launch['execution_receipt_sha256'])
    return execution


def index_predictions(records, names):
    expected = {(a, r, n) for a in ARMS for r in READOUTS for n in names}
    indexed = {(x['arm'], x['readout'], x['name']): x for x in records}
    need(len(records) == len(indexed) == 4*len(names) and set(indexed) == expected,
         'Missing, duplicated or mismatched fixed RGB population')
    need(len({x['path'] for x in records}) == len(records), 'Aliased output paths')
    return indexed


def source_indices(camera, rows):
    """Independent FP64 geometric ranking; no source RGB values used."""
    pose = np.asarray(camera['w2c'], dtype=np.float64)
    center = np.linalg.solve(pose[:3, :3], -pose[:3, 3])
    ray = np.linalg.solve(pose[:3, :3], [0., 0., 1.]); ray /= np.linalg.norm(ray)
    eligible = []
    for j, row in enumerate(rows):
        if row['name'] == camera['name']:
            continue
        p = np.asarray(row['w2c_original'], dtype=np.float64)
        c = np.linalg.solve(p[:3, :3], -p[:3, 3])
        d = np.linalg.solve(p[:3, :3], [0., 0., 1.]); d /= np.linalg.norm(d)
        distance = float(np.linalg.norm(c-center))
        angle = float(np.arccos(np.clip(d@ray, -1., 1.))*180/np.pi)
        if .01 < distance < 1.5 and angle < 30.:
            eligible.append((distance, angle, row['name'], j))
    return [x[-1] for x in sorted(eligible)[:4]]


def native_map(camera):
    """Reconstruct the corner-coordinate map and reject unsupported overscan."""
    width, height = camera['width'], camera['height']
    K = np.asarray(camera['K'], np.float32)
    distortion = np.asarray(camera['distortion'], np.float32)
    if distortion.size == 0 or not np.any(distortion):
        return None
    y, x = np.indices((height, width), dtype=np.float32)
    corners = np.stack((x+.5, y+.5), axis=-1)
    uv = cv2.undistortPoints(corners.reshape(-1, 1, 2), K, distortion, P=K).reshape(height, width, 2)
    uv -= np.float32(.5)
    need(np.isfinite(uv).all(), 'Nonfinite corner distortion map')
    lo = np.floor(uv.min(axis=(0, 1))); hi = np.ceil(uv.max(axis=(0, 1)))
    need(np.all(lo >= 0) and np.all(hi <= [width-1, height-1]), 'Unexpected nonnative covering canvas')
    return uv


def delivered_rgb(native, mapping):
    need(native.dtype == np.float32 and native.ndim == 3 and native.shape[2] == 3
         and np.isfinite(native).all(), 'Invalid native FP32 RGB')
    clipped = np.clip(native, np.float32(0), np.float32(1))
    if mapping is not None:
        clipped = cv2.remap(clipped, mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.rint(clipped*np.float32(255)).astype(np.uint8)


def decode_rgb(data, shape):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    need(image is not None and image.dtype == np.uint8 and image.shape == shape, 'Wrong RGB image/grid')
    return image[..., ::-1].copy()


def psnr(prediction, target):
    need(prediction.dtype == target.dtype == np.uint8 and prediction.shape == target.shape, 'Invalid PSNR inputs')
    a = (prediction.astype(np.float32)/np.float32(255)).astype(np.float64)
    b = (target.astype(np.float32)/np.float32(255)).astype(np.float64)
    residual = a-b
    mse = float(np.einsum('hwi,hwi->', residual, residual)/residual.size)
    return float(-10*np.log10(max(mse, 1e-12)))


def bootstrap_counts():
    draws = np.random.default_rng(20260926).integers(0, 50, size=(5000, 50))
    counts = np.zeros((5000, 50), np.int64)
    np.add.at(counts, (np.arange(5000)[:, None], draws), 1)
    return counts


def paired(reference, candidate, counts):
    need(reference.shape == candidate.shape == (50, 3) and np.isfinite(reference).all()
         and np.isfinite(candidate).all(), 'Expected finite 50x3 paired scores')
    delta = candidate-reference
    replicas = counts@delta/50.
    return {key: {'reference': float(reference[:, j].mean()), 'candidate': float(candidate[:, j].mean()),
                  'difference': float(candidate[:, j].mean()-reference[:, j].mean()),
                  'paired_view_bootstrap_95_interval': np.percentile(replicas[:, j], [2.5, 97.5]).tolist(),
                  'finite_bootstrap_replicates': 5000} for j, key in enumerate(KEYS)}


def clauses(metrics):
    return {'psnr_gain_at_least_0_15_dB': metrics['psnr']['difference'] >= .15,
            'psnr_paired_95_lower_positive': metrics['psnr']['paired_view_bootstrap_95_interval'][0] > 0,
            'ssim_point_not_lower': metrics['ssim']['difference'] >= 0,
            'lpips_point_not_higher': metrics['lpips']['difference'] <= 0}


def metric_rows(metrics, names, views):
    rows = {x['name']: x for x in metrics['views']}
    need(len(rows) == len(metrics['views']) == 50 and sorted(rows) == names, 'Metrics camera population differs')
    for name, row in rows.items():
        camera = views[name]['camera']
        need((row['width'], row['height'], row['rgb_pixels']) ==
             (camera['width'], camera['height'], camera['width']*camera['height']), 'Metric grid mismatch')
        if 'source_rgb_sha256' in row:
            need(row['source_rgb_sha256'] == views[name]['source_image_sha256'], 'Metrics use different GT identity')
    values = np.asarray([[rows[n][k] for k in KEYS] for n in names], np.float64)
    need(np.isfinite(values).all(), 'Nonfinite metric')
    for j, key in enumerate(KEYS):
        need(abs(float(values[:, j].mean())-metrics[key]) <= 1e-12, 'Wrong equal-view aggregate')
    return rows, values


def verify_imports(records, snapshot, expected):
    need(expected <= set(records), 'Required actual imports missing')
    for item in records.values():
        path = Path(item['path']).resolve()
        need(path.is_relative_to(snapshot), 'Actual import outside frozen package')
        bound(path, item['sha256'])


def audit(run, expected_plan, report):
    # Completion precedes any native, PNG, GT or endpoint access.
    rendered, scored = natural(run, 'render'), natural(run, 'score')
    bound(run/'plan.json', expected_plan)
    plan = read(run/'plan.json'); snapshot = Path(plan['source_snapshot'])
    need(rendered['plan_sha256'] == scored['plan_sha256'] == expected_plan, 'Stage plan mismatch')
    spec = plan['specification']
    need(spec['protocol'] == 'ibgs_warm_evaluation_v1' and spec['arms'] == list(ARMS)
         and spec['readouts'] == list(READOUTS) and spec['primary_candidate'] == 'full/fused'
         and spec['bootstrap_repeats'] == 5000 and spec['bootstrap_seed'] == 20260926
         and spec['rgb_gate'] == {'psnr_gain_db': .15, 'psnr_ci_lower_strict': 0., 'ssim_min': 0., 'lpips_max': 0.},
         'Fixed evaluation contract differs')
    sources = {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob('*'))
               if p.is_file() and p.suffix in {'.py', '.md'}}
    need(sources == plan['source_hashes'], 'Frozen sources changed')
    for path, expected in plan['input_hashes'].items():
        bound(path, expected)
    for path, expected in plan['runtime_sources'].items():
        need(plan['input_hashes'].get(path) == expected, 'Unbound renderer runtime source')
    bound(snapshot/'scoring/bridge_rgs/official_evaluate.py', SCORER_SHA)
    bound(snapshot/'scoring/bridge_rgs/evaluate.py', GRID_SHA)
    bound(snapshot/'rgb_scoring_helpers.py', WRAPPER_SHA)
    need({Path(p).name: plan['input_hashes'][p] for p in plan['perceptual_weights']} == WEIGHTS,
         'Perceptual weights differ')
    verify_imports(rendered['actual_imports'], snapshot/'render',
                   {'gaussian_renderer', 'bridge_rgs.ibgs_adapter', 'bridge_rgs.ibgs_warm_training'})
    verify_imports(scored['actual_imports'], snapshot/'scoring',
                   {'bridge_rgs.official_evaluate', 'bridge_rgs.evaluate'})
    need(rendered['actual_numerics'] == {'cudnn_tf32': False, 'matmul_tf32': False,
                                       'benchmark': False, 'matmul_precision': 'highest'}
         and scored['actual_numerics'] == plan['scoring_numerics'], 'Stage numerical policy differs')
    for receipt in (rendered, scored):
        need(receipt['sources_and_inputs_unchanged'] and all(receipt[k] == 0 for k in
             ('annotation_reads', 'teacher_calls', 'optimizer_steps')), 'Non-RGB evaluation operation')
    need((rendered['source_depth_calls'], rendered['target_calls'], rendered['VAL_rgb_reads']) == (700, 100, 0)
         and scored['VAL_rgb_reads'] == 50, 'Render/score counts differ')
    need(plan['prepare_VAL_payload_reads'] == 0 and not plan['prepare_cuda_initialized'], 'Preparation scope differs')
    training = Path(plan['training']); parent = read(training/'execution_receipt.json')
    bound(training/'plan.json', plan['training_plan_sha256'])
    parent_launch = read(training/'launch_receipt.json')
    need(parent['status'] == parent_launch['status'] == 'completed' and parent_launch['natural_completion']
         and parent_launch['exit_code'] == 0, 'Training did not naturally complete')
    bound(training/'execution_receipt.json', parent_launch['execution_receipt_sha256'])
    need({a['arm'] for a in parent['arms']} == set(ARMS), 'Training arm mismatch')
    for arm in parent['arms']:
        need(arm['status'] == 'completed' and arm['steps'] == 6000
             and plan['endpoints'][arm['arm']] == {'path': arm['last_checkpoint'], 'sha256': arm['checkpoint_sha256']},
             'Fixed trained endpoint not preserved')
    contract = read(plan['data_contract']['path']); bound(plan['data_contract']['path'], plan['data_contract']['sha256'])
    train = contract['train_rows']
    need(len(train) == len({v['name'] for v in train}) == 350 and all(v['split'] == 'train' for v in train),
         'Source bank must be 350 unique TRAIN views')
    need(all(plan['input_hashes'].get(p) == h for p, h in contract['pixel_hashes'].items()), 'Unbound source RGB/valid')
    views = {v['camera']['name']: v for v in plan['views']}; names = sorted(views)
    need(len(views) == len(plan['views']) == 50 and not set(names)&{v['name'] for v in train}, 'Camera split differs')
    old_path = [p for p in plan['input_hashes'] if p.endswith('rgb_capacity_1m_reference_v1/evaluation_official/execution_receipt.json')]
    need(len(old_path) == 1, 'Missing old official source records')
    old = {v['camera']['name']: v for v in read(old_path[0])['source_records']}
    for name in names:
        view = views[name]
        need(view['camera'] == old[name]['camera'] and view['source_image_path'] == old[name]['source_image_path']
             and view['source_image_sha256'] == old[name]['source_image_sha256'], 'Reference camera/GT identity differs')
        indices = source_indices(view['camera'], train)
        need(view['source_indices'] == indices and view['source_names'] == [train[i]['name'] for i in indices],
             'Source selection differs from fixed geometric rule')
        need(np.array_equal(view['camera']['K'], contract['common_camera']['K'])
             and np.array_equal(np.asarray(view['render_K'], np.float32), np.asarray(contract['common_camera']['K'], np.float32)),
             'Shared source/target camera intrinsics differ')
    native_path = rendered['native_predictions']['path']; bound(native_path, rendered['native_predictions']['sha256'])
    native_receipt, png_receipt = read(native_path), read(run/'predictions_receipt.json')
    need(native_receipt['status'] == 'completed' and png_receipt['status'] == 'all_200_pngs_before_GT'
         and native_receipt['VAL_payload_reads'] == png_receipt['VAL_payload_reads'] == 0, 'Prediction barrier differs')
    need(native_receipt['finished_utc'] <= rendered['finished_utc'] <= scored['started_utc']
         and png_receipt['finished_utc'] <= scored['predictions_finished_utc'] <= scored['scoring_started_utc'],
         'Recorded prediction-before-GT ordering differs')
    native = index_predictions(native_receipt['records'], names)
    pngs = index_predictions(png_receipt['records'], names)
    need(set(scored['metrics']) == {f'{a}/{r}' for a in ARMS for r in READOUTS}, 'Wrong four endpoint metrics')
    datasets, matrices, rowsets = {}, {}, {}
    for key, desc in scored['metrics'].items():
        bound(desc['path'], desc['sha256']); data = read(desc['path'])
        need(data['inherited_common_reference_fingerprint'] == plan['reference_fingerprint']
             and data['scoring_protocol'] == plan['scoring_protocol'], 'Scoring identity differs')
        rowsets[key], matrices[key] = metric_rows(data, names, views); datasets[key] = data
    max_psnr = 0.; image_checks = 0
    cv2.setNumThreads(8)
    for name in names:
        view = views[name]; camera = view['camera']; shape = (camera['height'], camera['width'], 3)
        mapping = native_map(camera)
        target = decode_rgb(payload(view['source_image_path'], view['source_image_sha256']), shape)
        for a in ARMS:
            for r in READOUTS:
                n, p = native[a, r, name], pngs[a, r, name]
                need(n['shape'] == list(shape) and n['dtype'] == 'float32', 'Native metadata differs')
                array = np.load(io.BytesIO(payload(n['path'], n['sha256'])), allow_pickle=False)
                need(array.shape == shape, 'Wrong native dimensions')
                prediction = decode_rgb(payload(p['path'], p['sha256']), shape)
                need(np.array_equal(delivered_rgb(array, mapping), prediction), 'Native-to-PNG warp/quantization differs')
                row = rowsets[f'{a}/{r}'][name]
                need(row['rgb_sha256'] == p['sha256'], 'Metrics not bound to this PNG')
                error = abs(psnr(prediction, target)-row['psnr']); max_psnr = max(max_psnr, error)
                need(error <= 1e-10, 'Independent PNG PSNR differs')
                image_checks += 1
    for key, desc in plan['reference_metrics'].items():
        need(desc['sha256'] == REFERENCES[key], 'Reference score identity changed')
        bound(desc['path'], desc['sha256'])
        _, matrices[key] = metric_rows(read(desc['path']), names, views)
    expected_pairs = {f'{a}_{r}_minus_{ref}': (f'{a}/{r}', ref)
                      for a in ARMS for r in READOUTS for ref in REFERENCES}
    expected_pairs.update({f'full_minus_no_source_{r}': (f'full/{r}', f'no_source/{r}') for r in READOUTS})
    need(set(scored['comparisons']) == set(expected_pairs), 'Wrong fixed ten comparisons')
    counts = bootstrap_counts(); summary = {}; max_stat = 0.
    for key, (candidate, reference) in expected_pairs.items():
        desc = scored['comparisons'][key]; bound(desc['path'], desc['sha256']); saved = read(desc['path'])
        need(saved['views'] == names and saved['bootstrap_repeats'] == 5000 and saved['seed'] == 20260926,
             'Paired cohort or resampling changed')
        result = paired(matrices[reference], matrices[candidate], counts)
        for metric in KEYS:
            for field in ('reference', 'candidate', 'difference', 'paired_view_bootstrap_95_interval'):
                error = float(np.max(np.abs(np.asarray(result[metric][field])-saved['metrics'][metric][field])))
                max_stat = max(max_stat, error); need(error <= 1e-11, 'Independent paired statistics differ')
            need(saved['metrics'][metric]['finite_bootstrap_replicates'] == 5000, 'Incomplete bootstrap')
        gates = clauses(result)
        need(gates == saved['gate_clauses'] == desc['gates'], 'Engineering clauses differ')
        summary[key] = {'metrics': result, 'gates': gates, 'passed_clauses': sum(gates.values())}
    # Final identities; no re-rendering or new semantic information is inferred.
    need(sources == {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob('*'))
                     if p.is_file() and p.suffix in {'.py', '.md'}}, 'Sources changed during audit')
    report.update(status='passed', plan_sha256=expected_plan,
                  render_execution_sha256=sha(run/'render_execution_receipt.json'),
                  score_execution_sha256=sha(run/'score_execution_receipt.json'),
                  source_files=len(sources), bound_inputs=len(plan['input_hashes']),
                  native_png_exact=image_checks, GT_rgb_decodes=50, semantic_decodes=0, GPU_calls=0,
                  independent_psnr_max_abs_difference=max_psnr, paired_stat_max_abs_difference=max_stat,
                  endpoints={k: {m: v[m] for m in KEYS} for k, v in datasets.items()}, comparisons=summary,
                  primary_candidate='full/fused', automatic_adoption=False,
                  limits=['SSIM/LPIPS pixels not independently re-scored; bound fixed scorer/weights and saved values only.',
                          'Camera/source identities and prediction-before-GT order use bound metadata, source review and recorded timestamps; no independent runtime I/O tracing.',
                          'No independent GPU/render/VJP or model-state reconstruction; endpoints are SHA-bound.',
                          'Development RGB evaluation only; existing E semantic values are not re-evaluated.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); output = args.output or args.run/'independent_cpu_review.json'
    need(not output.exists(), 'Never overwrite an audit')
    started = time.monotonic()
    report = {'status': 'failed', 'audit_source_sha256': sha(__file__), 'started_utc': datetime.now(UTC).isoformat()}

    def deadline(*_):
        raise TimeoutError('Independent CPU audit exceeded 180 seconds')

    signal.signal(signal.SIGALRM, deadline); signal.alarm(180)
    try:
        audit(args.run, args.expected_plan_sha256, report)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        data = json.dumps(report, indent=2, allow_nan=False)+'\n'
        with output.open('x') as stream:
            stream.write(data)
        signal.alarm(0)
    print(json.dumps({'status': report['status'], 'output': str(output), 'sha256': sha(output)}), flush=True)


if __name__ == '__main__':
    main()
