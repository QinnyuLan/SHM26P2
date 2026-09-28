"""Independent CPU RGB check for the completed 1M-versus-500k official evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
RUN = Path('/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1')
CONTROL = Path('/mnt/data/SHM2026/runs/ssim_fixed_corner_v2_rgb_full/evaluation_official/official_metrics.json')
TRAIN_PLAN_SHA = '40191687627d54da91b78c8b819e83bc04b04b986ae6568efe042c6dbcd447f3'
CONTROL_SHA = '12bd34d0536da05f2a70a10f3b736312c679989c2a7fab3d42f84cae18d08ea8'
HELPER = ROOT/'scripts/audit_mcmc_official_results.py'
HELPER_SHA = 'bc46370fb50ec7854e95eda0dfc14ee1e0d0344e626ac101ac9078cce85aef07'
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise ValueError('Previously tested independent CPU helper changed')
SPEC = importlib.util.spec_from_file_location('fixed_independent_rgb_audit', HELPER)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
read, digest, check = helper.read, helper.digest, helper.check


def png_psnr(rgb, truth):
    check(rgb.dtype == truth.dtype == np.uint8 and rgb.shape == truth.shape
          and rgb.ndim == 3 and rgb.shape[-1] == 3, 'Invalid delivered/target RGB arrays')
    error = (rgb.astype(np.float32)/255).astype(np.float64) - (truth.astype(np.float32)/255).astype(np.float64)
    return float(-10*np.log10(max(float(np.sum(error*error, dtype=np.float64)/error.size), 1e-12)))


def rgb_predictions(data, receipt):
    check(receipt['predictions_finished_utc'] <= receipt['source_scoring_started_utc'], 'Scoring started before all predictions')
    views, predictions, sources = [{v['name']: v for v in records} for records in
                                   (data['views'], receipt['predictions'], receipt['source_records'])]
    check(len(views) == len(predictions) == len(sources) == 50 and set(views) == set(predictions) == set(sources), 'Incomplete RGB/source records')
    records = []
    for name in sorted(views):
        v, p, s = views[name], predictions[name], sources[name]
        check(digest(p['rgb']) == p['rgb_sha256'] and digest(s['source_image_path']) == s['source_image_sha256'], 'RGB or source bytes changed')
        rgb, truth = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in (p['rgb'], s['source_image_path'])]
        check(rgb is not None and truth is not None and rgb.shape == (v['height'], v['width'], 3), 'Wrong RGB dimensions')
        psnr = png_psnr(rgb, truth)  # Same BGR permutation leaves MSE unchanged.
        error = abs(psnr-v['psnr'])
        check(error < 1e-10, 'PNG PSNR differs from the official value')
        records.append({'name': name, 'psnr': psnr, 'absolute_difference': error, 'rgb_sha256': p['rgb_sha256'],
                        'source_rgb_sha256': s['source_image_sha256']})
    fingerprint_records = [{key: s[key] for key in ('camera', 'source_image_sha256', 'source_annotation_sha256', 'rasterized_mask_sha256')}
                           for s in sources.values()]
    payload = {'evaluation_family': data['evaluation_family'], 'scoring_protocol': data['scoring_protocol'],
               'views': sorted(fingerprint_records, key=lambda value: value['camera']['name'])}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    check(fingerprint == data['official_evaluation_fingerprint'], 'Recorded common protocol fingerprint differs')
    return records, {name: value['camera'] for name, value in sources.items()}


def gate_clauses(metrics):
    check(set(metrics) == set(helper.RGB_KEYS), 'No semantic ranking allowed')
    return {'psnr_gain_at_least_0_15_dB': metrics['psnr']['difference'] >= .15,
            'psnr_paired_95_lower_positive': metrics['psnr']['paired_view_bootstrap_95_interval'][0] > 0,
            'ssim_point_not_lower': metrics['ssim']['difference'] >= 0,
            'lpips_point_not_higher': metrics['lpips']['difference'] <= 0}


def main(expected_plan_sha):
    output = RUN/'independent_official_cpu_audit.json'
    check(not output.exists(), 'Do not overwrite an audit')
    # No unfinished checkpoint/prediction is accessed before the completion guards.
    comparison, launch, endpoint = [read(RUN/name) for name in
                                  ('official_comparison_receipt.json', 'official_launch_receipt.json', 'cpu_endpoint_audit.json')]
    check(comparison['status'] == launch['status'] == 'completed' and launch['exit_code'] == 0
          and endpoint['status'] == 'passed', 'Require natural completed official evaluation and passed endpoint')
    plan_path = RUN/'official_preparation/plan.json'
    plan = read(plan_path)
    check(digest(plan_path) == expected_plan_sha == launch['plan_sha256'] == comparison['plan_sha256'], 'Official plan chain differs')
    check(digest(RUN/'plan.json') == TRAIN_PLAN_SHA == plan['training_plan_sha256'] == endpoint['plan_sha256'], 'Training plan chain differs')
    check(plan['bootstrap_repeats'] == 5000 and plan['bootstrap_seed'] == 20260926
          and plan['reference'] == str(CONTROL) and digest(CONTROL) == CONTROL_SHA, 'Fixed comparison changed')
    check(plan['continue_to_semantic_gate'] == {'psnr_min_gain_db': .15, 'psnr_paired_95_lower_strict': 0.,
                                              'ssim_min_gain': 0., 'lpips_max_gain': 0.}, 'Fixed gate changed')
    for path, sha in plan['bound_inputs'].items():
        check(digest(path) == sha, 'Bound input/source changed: '+path)
    snapshot = Path(plan['source_snapshot'])
    check({str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')} == plan['source_hashes'], 'Scoring snapshot changed')
    paths = {'reference': CONTROL, 'candidate': RUN/'evaluation_official/official_metrics.json'}
    datasets, receipts, png_records, cameras = {}, {}, {}, {}
    cv2.setNumThreads(8)
    for role, path in paths.items():
        data, receipt = helper.bound_metrics(path)
        for record in [receipt['entrypoint_source'], *receipt['loaded_source_modules'].values()]:
            check(digest(record['path']) == record['sha256'], 'Actual scoring source changed')
        png_records[role], cameras[role] = rgb_predictions(data, receipt)
        datasets[role], receipts[role] = data, receipt
    candidate_receipt = receipts['candidate']
    check(digest(candidate_receipt['checkpoint']) == candidate_receipt['checkpoint_sha256']
          == launch['checkpoint_sha256'] == endpoint['checkpoint_sha256'], 'Wrong field endpoint')
    check(cameras['reference'] == cameras['candidate'] and datasets['reference']['official_evaluation_fingerprint']
          == datasets['candidate']['official_evaluation_fingerprint'], 'Camera population/common scoring protocol differs')
    pair_path, gate_path = RUN/'paired_official_rgb_minus_500k.json', RUN/'semantic_investment_gate.json'
    check(digest(pair_path) == comparison['paired_report_sha256'] and digest(gate_path) == comparison['gate_sha256'], 'Comparison/gate hash differs')
    pair, gate = read(pair_path), read(gate_path)
    check(pair['seed'] == 20260926 and pair['bootstrap_repeats'] == 5000
          and set(pair['metrics']) == set(helper.RGB_KEYS), 'Wrong pairing or semantic contamination')
    for role, path in paths.items():
        binding = pair[role+'_source']
        check(binding['path'] == str(path) and binding['sha256'] == digest(path)
              and binding['receipt_sha256'] == digest(path.parent/'execution_receipt.json')
              and binding['checkpoint_sha256'] == receipts[role]['checkpoint_sha256'], 'Paired endpoint provenance differs')
    metrics = helper.independent_bootstrap(datasets['reference'], datasets['candidate'])
    errors = [float(np.max(np.abs(np.asarray(value)-np.asarray(pair['metrics'][metric][field]))))
              for metric in helper.RGB_KEYS for field, value in metrics[metric].items()]
    check(max(errors) < 1e-12 and gate['metrics'] == pair['metrics'], 'Independent RGB bootstrap or gate metrics differ')
    clauses = gate_clauses(metrics)
    check(gate['clauses'] == clauses and gate['passed'] == all(clauses.values())
          and gate['evaluation_plan_sha256'] == expected_plan_sha
          and gate['training_plan_sha256'] == TRAIN_PLAN_SHA
          and gate['candidate_metrics_sha256'] == digest(paths['candidate']), 'Independent four-gate result differs')
    report = {'status': 'passed', 'cpu_only': True, 'audit_script_sha256': digest(__file__), 'independent_helper_sha256': HELPER_SHA,
              'official_evaluation_plan_sha256': expected_plan_sha, 'training_plan_sha256': TRAIN_PLAN_SHA,
              'checkpoint_sha256': endpoint['checkpoint_sha256'], 'official_comparison_receipt_sha256': digest(RUN/'official_comparison_receipt.json'),
              'metrics_sources': {role: {'path': str(path), 'sha256': digest(path), 'receipt_sha256': digest(path.parent/'execution_receipt.json')}
                                  for role, path in paths.items()},
              'png_rgb_count': {'candidate': 50, 'reference': 50}, 'per_view_psnr_recomputed': png_records,
              'psnr_max_absolute_discrepancy': max(v['absolute_difference'] for rows in png_records.values() for v in rows),
              'source_and_bound_inputs_exact': True, 'cameras_exact': True,
              'paired_max_discrepancy': max(errors), 'paired_rgb': metrics, 'investment_gate': {'passed': all(clauses.values()), 'clauses': clauses},
              'limits': 'Only delivered RGB PNG and source RGB bytes/PSNR are rescored. SSIM/LPIPS remain bound to the completed official scorer. No masks decoded or evaluated; fingerprint retains bound annotation/raster hashes. Fixed-view single-run RGB capacity comparison, not equal compute, selection-adjusted generalization, joint-system adoption, or innovation.'}
    for path, sha in plan['bound_inputs'].items():
        check(digest(path) == sha, 'Bound input/source changed during CPU audit: '+path)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'audit_sha256': digest(output), 'paired_max_discrepancy': max(errors),
                      'psnr_max_absolute_discrepancy': report['psnr_max_absolute_discrepancy'], 'gate': report['investment_gate']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-plan-sha256', required=True)
    main(parser.parse_args().expected_plan_sha256)
