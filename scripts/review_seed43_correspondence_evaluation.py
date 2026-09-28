"""Independent CPU review of the two seed-43 RGB endpoint evaluations."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/mnt/data/SHM2026/runs/ibgs_correspondence_seed43_evaluation_v1')


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def main():
    plan = json.loads((ROOT/'plan.json').read_text())
    pred = json.loads((ROOT/'predictions_receipt.json').read_text())
    names = [v['camera']['name'] for v in plan['views']]
    arms = ['top4_normalized_correct_seed43', 'top4_normalized_permuted_seed43']
    checks = {'prediction_barrier_100': len(pred['records']) == 100,
              'unique_arm_name_pairs': len({(r['arm'], r['name']) for r in pred['records']}) == 100,
              'no_gpu': True}
    recomputed = {}
    for arm in arms:
        metrics = json.loads((ROOT/f'metrics_{arm}.json').read_text())
        rows = []
        for name in names:
            target = cv2.imread(next(v['source_image_path'] for v in plan['views'] if v['camera']['name'] == name), cv2.IMREAD_COLOR)
            image = cv2.imread(str(ROOT/'predictions'/arm/name), cv2.IMREAD_COLOR)
            require = target is not None and image is not None and target.shape == image.shape
            if not require:
                raise ValueError(f'Missing RGB pair {arm}/{name}')
            mse = np.mean((image.astype(np.float64)-target.astype(np.float64))**2)/(255.0**2)
            rows.append(-10.0*math.log10(mse))
        observed = [v['psnr'] for v in metrics['views']]
        checks[f'{arm}_psnr_exact'] = bool(np.max(np.abs(np.asarray(rows)-observed)) <= 1e-6)
        checks[f'{arm}_png_sha_bound'] = all(sha(ROOT/'predictions'/arm/name) == next(r['sha256'] for r in pred['records'] if r['arm'] == arm and r['name'] == name) for name in names)
        recomputed[arm] = {'psnr_mean': float(np.mean(rows)), 'max_abs_psnr_error': float(np.max(np.abs(np.asarray(rows)-observed)))}
    # Pair report is recomputed from the frozen metric files by the project helper.
    import importlib.util
    source = Path('/mnt/data/SHM2026/runs/ibgs_correspondence_evaluation_v1/source_snapshot/ibgs_warm_evaluation_base.py')
    spec = importlib.util.spec_from_file_location('review_legacy', source)
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    helper = legacy.scoring_modules('/mnt/data/SHM2026/runs/ibgs_correspondence_evaluation_v1/source_snapshot')[1]
    correct = json.loads((ROOT/'metrics_top4_normalized_correct_seed43.json').read_text())
    permuted = json.loads((ROOT/'metrics_top4_normalized_permuted_seed43.json').read_text())
    pair = helper.paired_rgb(permuted, correct)
    observed_pair = json.loads((ROOT/'paired_correct_minus_permuted.json').read_text())
    checks['mechanism_pair_exact'] = pair['metrics'] == observed_pair['metrics']
    report = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
              'recomputed': recomputed, 'metric_pair': pair['metrics'],
              'metric_files': {arm: sha(ROOT/f'metrics_{arm}.json') for arm in arms},
              'prediction_receipt_sha256': sha(ROOT/'predictions_receipt.json')}
    (ROOT/'independent_cpu_review.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if report['status'] != 'passed':
        raise SystemExit('independent review failed')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
