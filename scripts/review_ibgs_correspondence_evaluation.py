"""Independent CPU audit for the fixed correspondence RGB evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def review(run):
    start = time.perf_counter(); run = Path(run); plan = read(run/'plan.json')
    render, score = read(run/'render_execution_receipt.json'), read(run/'score_execution_receipt.json')
    launch_r, launch_s = read(run/'render_launch_receipt.json'), read(run/'score_launch_receipt.json')
    assert render['status'] == launch_r['status'] == score['status'] == launch_s['status'] == 'completed'
    assert launch_r['exit_code'] == launch_s['exit_code'] == 0 and launch_r['natural_completion'] and launch_s['natural_completion']
    assert launch_r['execution_receipt_sha256'] == sha(run/'render_execution_receipt.json')
    assert launch_s['execution_receipt_sha256'] == sha(run/'score_execution_receipt.json')
    assert launch_r['plan_sha256'] == launch_s['plan_sha256'] == sha(run/'plan.json')
    barrier = read(run/'predictions_receipt.json'); assert barrier['status'] == 'all_50_pngs_before_GT'
    assert barrier.get('VAL_payload_reads', 0) == 0 and score['predictions_finished_utc'] <= score['scoring_started_utc']
    assert len(barrier['records']) == 50 and all(item['arm'] == 'top4_normalized_permuted' for item in barrier['records'])
    metrics = read(run/'metrics_top4_normalized_permuted.json'); rows = {r['name']: r for r in metrics['views']}
    assert len(rows) == 50 and score['VAL_rgb_reads'] == score['lpips_calls'] == 50 and score['annotation_reads'] == 0
    errors = []
    for view in plan['views']:
        name = view['camera']['name']; row = rows[name]
        pred = next(r for r in barrier['records'] if r['name'] == name)
        assert sha(pred['path']) == pred['sha256'] == row['rgb_sha256']
        assert sha(view['source_image_path']) == view['source_image_sha256'] == row['source_rgb_sha256']
        with Image.open(pred['path']) as image: actual = np.asarray(image.convert('RGB'))
        with Image.open(view['source_image_path']) as image: target = np.asarray(image.convert('RGB'))
        a = (actual.astype(np.float32)/255).astype(np.float64); b = (target.astype(np.float32)/255).astype(np.float64)
        psnr = -10*np.log10(max(np.square(a-b).sum()/a.size, 1e-12)); errors.append(abs(float(psnr)-row['psnr']))
    names = sorted(rows); rng = np.random.default_rng(20260926); sample = rng.integers(0, 50, (5000, 50))
    for key in ('psnr', 'ssim', 'lpips'):
        errors.append(abs(float(np.mean([rows[n][key] for n in names]))-metrics[key]))
    for filename in ('paired_correct_minus_permuted.json', 'paired_top4_normalized_permuted_minus_top4_normalized.json',
                     'paired_top4_normalized_permuted_minus_E_rgb.json', 'paired_top4_normalized_permuted_minus_F_rgb.json'):
        pair = read(run/filename); reference = pair['metrics']
        # Reconstruct means and bootstrap intervals from the bound candidate/reference metric rows.
        if filename == 'paired_correct_minus_permuted.json':
            other = read(plan['reference_metrics']['top4_normalized']['path'])
        elif filename.endswith('E_rgb.json'):
            other = read('/mnt/data/SHM2026/runs/rgb_fixed_ensemble_v1/rgb_metrics.json')
        elif filename.endswith('F_rgb.json'):
            other = read('/mnt/data/SHM2026/runs/ibgs_top4_mcmc_replacement_v1/rgb_metrics.json')
        else:
            other = read('/mnt/data/SHM2026/runs/ibgs_layer_heads_evaluation_v1/metrics_top4_normalized.json')
        ref = {r['name']: r for r in other['views']}
        candidate_minus_reference = filename != 'paired_correct_minus_permuted.json'
        for key in ('psnr', 'ssim', 'lpips'):
            delta = np.asarray([rows[n][key]-ref[n][key] for n in names]) if candidate_minus_reference else \
                np.asarray([ref[n][key]-rows[n][key] for n in names])
            errors.extend([abs(float(delta.mean())-reference[key]['difference']),
                           *np.abs(np.quantile(delta[sample].mean(1), [.025, .975])-reference[key]['paired_view_bootstrap_95_interval'])])
    assert max(errors) < 1e-12
    result = {'status':'passed','plan_sha256':sha(run/'plan.json'),'worker_sha256':sha(__file__),
        'render_receipt_sha256':sha(run/'render_execution_receipt.json'),'score_receipt_sha256':sha(run/'score_execution_receipt.json'),
        'predictions_verified':50,'RGB_GT_reads':50,'annotation_reads':0,'gpu_calls':0,'max_absolute_error':float(max(errors)),
        'elapsed_seconds':time.perf_counter()-start,'scope':'Independent uint8/GT PSNR and saved means/paired intervals; no independent SSIM/LPIPS pixel rescoring'}
    with (run/'independent_cpu_review.json').open('x') as stream: json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True); review(parser.parse_args().run)
