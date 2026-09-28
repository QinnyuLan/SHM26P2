"""Evaluate a fixed weighted top4/MCMC RGB export without model fitting.

The weight is an explicit deployment hyperparameter.  All 50 predictions are
materialized and hashed before any target RGB bytes are opened.  The script is
for a development candidate; it does not alter the default F bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
F_RUN = RUNS/'ibgs_joint_replacement_v2'
SNAPSHOT = RUNS/'multifield_h3_teacher_v1/source_snapshot'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def paired(reference, candidate, repeats=5000, seed=20260928):
    names = sorted(reference['views'])
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(names), size=(repeats, len(names)))
    result = {'views': names, 'bootstrap_repeats': repeats, 'seed': seed,
              'difference_direction': 'candidate minus F; LPIPS improves when negative', 'metrics': {}}
    for key in ('psnr', 'ssim', 'lpips'):
        ref = np.asarray([reference['views'][n][key] for n in names], np.float64)
        cand = np.asarray([candidate['views'][n][key] for n in names], np.float64)
        diff = cand-ref
        result['metrics'][key] = {'reference': float(ref.mean()), 'candidate': float(cand.mean()),
            'difference': float(diff.mean()),
            'paired_view_bootstrap_95_interval': np.quantile(diff[indices].mean(1), [.025, .975]).tolist(),
            'finite_bootstrap_replicates': repeats}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-top4', type=float, default=.6)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(0 <= args.weight_top4 <= 1, 'Weight must be in [0,1]')
    output = args.output.resolve(); require(not output.exists(), 'Refuse nonempty output'); output.mkdir(parents=True)
    fmetrics = read(F_RUN/'official_metrics.json'); names = sorted(v['name'] for v in fmetrics['views'])
    require(len(names) == 50, 'Fixed 50-view RGB population required')
    bindings = {str(p.resolve()): sha(p) for p in (F_RUN/'plan.json', F_RUN/'render/main_execution_receipt.json', F_RUN/'official_metrics.json')}
    prediction_hashes = {}; records = []
    # Prediction barrier: only component predictions are opened here.
    for name in names:
        first = F_RUN/'render/components/top4_normalized'/name; second = F_RUN/'render/components/mcmc'/name
        require(first.is_file() and second.is_file(), f'Missing F component: {name}')
        bindings[str(first.resolve())] = sha(first); bindings[str(second.resolve())] = sha(second)
        a, b = cv2.imread(str(first), cv2.IMREAD_COLOR), cv2.imread(str(second), cv2.IMREAD_COLOR)
        require(a is not None and b is not None and a.shape == b.shape == (989, 1320, 3), 'Invalid component grid')
        prediction = np.rint(args.weight_top4*a.astype(np.float32)+(1-args.weight_top4)*b.astype(np.float32)).astype(np.uint8)
        path = output/'rgb'/name; path.parent.mkdir(parents=True, exist_ok=True); require(cv2.imwrite(str(path), prediction), 'RGB write failed')
        digest = sha(path); prediction_hashes[name] = digest; records.append({'name': name, 'path': str(path), 'sha256': digest})
    write(output/'prediction_receipt.json', {'status':'all_50_predictions_before_gt','weight_top4':args.weight_top4,'records':records,'annotation_reads':0,'target_rgb_reads':0})
    # Only now decode the 50 source RGB targets.
    sys.path.insert(0, str(SNAPSHOT)); official = importlib.import_module('bridge_rgs.official_evaluate'); perceptual = official._lpips('cuda'); rows=[]
    source_by_name = {v['name']: v for v in read(ROOT/'artifacts/prepared/manifest.json')['views'] if v['split']=='val'}
    for name in names:
        target_path = Path(source_by_name[name]['source_image_path']); require(target_path.is_file(), 'Missing target RGB')
        prediction = cv2.cvtColor(cv2.imread(str(output/'rgb'/name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB); target = cv2.cvtColor(cv2.imread(str(target_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        row = official.score_official_arrays(prediction, np.zeros(prediction.shape[:2], np.uint8), target, None, perceptual, device='cuda'); rows.append({'name': name, **row, 'rgb_sha256': prediction_hashes[name], 'source_rgb_sha256': sha(target_path)})
    metrics={'views':rows,'weight_top4':args.weight_top4,'psnr':float(np.mean([r['psnr'] for r in rows])),'ssim':float(np.mean([r['ssim'] for r in rows])),'lpips':float(np.mean([r['lpips'] for r in rows]))}; write(output/'official_rgb_metrics.json',metrics)
    reference={'views':{v['name']:v for v in fmetrics['views']}}; candidate={'views':{v['name']:v for v in rows}}; pair=paired(reference,candidate); write(output/'paired_minus_F.json',pair)
    m=pair['metrics']; clauses={'psnr_gain_at_least_0_15_dB':m['psnr']['difference']>=.15,'psnr_ci_lower_positive':m['psnr']['paired_view_bootstrap_95_interval'][0]>0,'ssim_point_not_lower':m['ssim']['difference']>=0,'lpips_point_not_higher':m['lpips']['difference']<=0}; write(output/'candidate_gate.json',{'passed':bool(all(clauses.values())),'clauses':clauses,'research_gate_scope':'F engineering gate; candidate not default until blind/held-out confirmation'})
    write(output/'execution_receipt.json',{'status':'completed','natural_completion':True,'weight_top4':args.weight_top4,'prediction_count':50,'target_rgb_reads':50,'prediction_barrier':True,'input_bindings':bindings,'official_metrics_sha256':sha(output/'official_rgb_metrics.json'),'pair_sha256':sha(output/'paired_minus_F.json')})
    print(json.dumps({'weight_top4':args.weight_top4,'psnr':metrics['psnr'],'ssim':metrics['ssim'],'lpips':metrics['lpips'],'gate':clauses},indent=2))


if __name__ == '__main__':
    main()
