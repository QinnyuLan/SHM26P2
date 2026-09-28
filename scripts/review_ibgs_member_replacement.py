"""Independent CPU byte, PSNR and paired-statistics review of fixed replacement."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path('/mnt/data/SHM2026/runs/ibgs_top4_mcmc_replacement_v1')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    start = time.perf_counter(); plan = read(RUN/'plan.json'); receipt = read(RUN/'execution_receipt.json')
    launch = read(RUN/'launch_receipt.json'); barrier = read(RUN/'predictions_receipt.json')
    assert receipt['status'] == launch['status'] == 'completed' and launch['exit_code'] == 0
    assert launch['natural_completion'] and launch['execution_receipt_sha256'] == sha(RUN/'execution_receipt.json')
    assert receipt['plan_sha256'] == launch['plan_sha256'] == sha(RUN/'plan.json')
    assert barrier['status'] == 'all_50_predictions_complete_before_gt' and barrier['gt_payload_reads'] == 0
    assert barrier['predictions'] == receipt['predictions'] and barrier['utc'] <= receipt['source_scoring_started_utc']
    predictions = {v['name']: v for v in receipt['predictions']}; names = sorted(predictions)
    assert len(names) == len(receipt['predictions']) == len(plan['views']) == 50
    assert sorted(v['name'] for v in plan['views']) == names
    assert all(sha(p) == h for p, h in plan['input_hashes'].items())
    # Reverify the complete existing output barrier before this review reads GT.
    assert all(sha(v['rgb']) == v['rgb_sha256'] for v in predictions.values())
    metrics = read(RUN/'rgb_metrics.json'); assert sha(RUN/'rgb_metrics.json') == receipt['metrics_sha256']
    rows = {v['name']: v for v in metrics['views']}; pixel_errors = []
    for view in plan['views']:
        row = rows[view['name']]; pred = predictions[view['name']]
        with Image.open(pred['rgb']) as image:
            actual = np.asarray(image.convert('RGB'))
        members = []
        for role in ('top4_normalized', 'mcmc'):
            item = view['members'][role]; assert sha(item['path']) == item['sha256']
            with Image.open(item['path']) as image:
                members.append(np.asarray(image.convert('RGB')))
        # Integer ties-to-even independently, without calling the blend helper.
        total = members[0].astype(np.uint16)+members[1].astype(np.uint16)
        floor = total//2
        expected = (floor+((total % 2 == 1) & (floor % 2 == 1))).astype(np.uint8)
        assert np.array_equal(expected, actual)
        assert sha(view['source_image_path']) == view['source_image_sha256'] == row['source_rgb_sha256']
        with Image.open(view['source_image_path']) as image:
            target = np.asarray(image.convert('RGB'))
        a = (actual.astype(np.float32)/255).astype(np.float64)
        b = (target.astype(np.float32)/255).astype(np.float64)
        psnr = -10*np.log10(max(np.square(a-b).sum()/a.size, 1e-12))
        pixel_errors.append(abs(float(psnr)-row['psnr']))
    draws = np.random.default_rng(20260926).integers(0, 50, (5000, 50))
    counts = np.stack([np.bincount(d, minlength=50)/50 for d in draws]); statistic_errors = []
    for key in ('psnr', 'ssim', 'lpips'):
        values = np.asarray([rows[n][key] for n in names])
        statistic_errors.append(abs(float(values.mean())-metrics[key]))
    for role, descriptor in plan['reference_metrics'].items():
        assert sha(descriptor['path']) == descriptor['sha256']
        reference = {v['name']: v for v in read(descriptor['path'])['views']}
        pair = read(RUN/f'paired_minus_{role}.json')
        for key, stated in pair['metrics'].items():
            delta = np.asarray([rows[n][key]-reference[n][key] for n in names])
            statistic_errors.extend([abs(float(delta.mean())-stated['difference']),
                *np.abs(np.quantile(counts@delta, [.025, .975])-stated['paired_view_bootstrap_95_interval'])])
    pair = read(RUN/'paired_minus_E_rgb.json')['metrics']
    clauses = {'psnr_gain_at_least_0_15_dB': pair['psnr']['difference'] >= .15,
               'psnr_paired_95_lower_positive': pair['psnr']['paired_view_bootstrap_95_interval'][0] > 0,
               'ssim_point_not_lower': pair['ssim']['difference'] >= 0,
               'lpips_point_not_higher': pair['lpips']['difference'] <= 0}
    gate = read(RUN/'rgb_gate.json'); assert clauses == gate['clauses'] and all(clauses.values()) == gate['passed']
    assert max(pixel_errors) < 1e-10 and max(statistic_errors) < 1e-12
    result = {'status': 'passed', 'plan_sha256': sha(RUN/'plan.json'),
              'execution_receipt_sha256': sha(RUN/'execution_receipt.json'), 'worker_sha256': sha(__file__),
              'exact_delivered_pngs': 50, 'member_pngs_verified': 100, 'GT_RGB_reads': 50,
              'PSNR_max_absolute_error': max(pixel_errors), 'summary_max_absolute_error': max(statistic_errors),
              'independent_gate': clauses, 'gpu_calls': 0, 'semantic_reads': 0,
              'scope': 'independent integer blend, delivered PNG/GT PSNR and saved SSIM/LPIPS statistics; no perceptual pixel rescoring',
              'elapsed_seconds': time.perf_counter()-start}
    with (RUN/'independent_cpu_review.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
