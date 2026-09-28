"""Independent CPU-only delivered-PNG/paired-RGB audit; does not import the scoring or paired helper."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

RUN = Path('/mnt/data/SHM2026/runs/rgb140_mip_filter_500k')
METRICS_SHA = 'c6a85cdeda7848084c3434762775183794a493cad519e4855d1eb05928dc0085'
PLAN_SHA = 'e61536f9275b982248b67f993de2b17b49b7fa1b8a4fd9cd2c6288f9016958e5'
RGB_KEYS = ('psnr', 'ssim', 'lpips')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def bound_metrics(path):
    path = Path(path)
    data, receipt = read(path), read(path.parent / 'execution_receipt.json')
    check(receipt['status'] == 'completed' and digest(path) == receipt['official_metrics_sha256'],
          'Incomplete or unbound scoring result')
    for key in ('evaluation_family', 'scoring_protocol', 'official_evaluation_fingerprint', 'inference_protocol'):
        check(data[key] == receipt[key], 'Metrics/receipt protocol differs')
    check(data['evaluation_family'] == 'official_original_pixel_grid'
          and data['scoring_protocol']['id'] == 'official_original_grid_v1', 'Not official original-grid scores')
    check(data['validation_views'] == len(data['views']) == 50, 'Not all 50 views')
    check(len({v['name'] for v in data['views']}) == 50, 'Duplicate view')
    for view in data['views']:
        check(view['rgb_pixels'] == view['width'] * view['height']
              and all(np.isfinite(view[k]) for k in RGB_KEYS), 'Invalid full-image RGB score')
    for key in RGB_KEYS:
        check(abs(np.mean([v[key] for v in data['views']]) - data[key]) < 1e-12, 'Aggregate is not view mean')
    return data, receipt


def independent_bootstrap(reference, candidate):
    """Multinomial view-count weighting, independently implemented from index-array means."""
    av, bv = [{r['name']: r for r in data['views']} for data in (reference, candidate)]
    check(set(av) == set(bv), 'Different view population')
    names = sorted(av)
    for name in names:
        check(all(av[name][k] == bv[name][k] for k in ('width', 'height', 'rgb_pixels')), 'Different grid/support')
    rng = np.random.default_rng(20260926)
    counts = np.empty((5000, 50), np.int64)
    for i in range(5000):
        counts[i] = np.bincount(rng.integers(0, 50, 50), minlength=50)
    result = {}
    for metric in RGB_KEYS:
        old, new = [np.array([views[name][metric] for name in names], np.float64) for views in (av, bv)]
        differences = counts @ (new - old) / 50
        result[metric] = {'reference': float(old.mean()), 'candidate': float(new.mean()),
                          'difference': float(new.mean() - old.mean()),
                          'paired_view_bootstrap_95_interval': np.percentile(differences, [2.5, 97.5], method='linear').tolist(),
                          'finite_bootstrap_replicates': int(np.isfinite(differences).sum())}
    return result


def main():
    output = RUN / 'official_result_cpu_audit.json'
    check(not output.exists(), 'Do not overwrite a result audit')
    cv2.setNumThreads(8)
    plan_path = RUN / 'official_evaluation_plan.json'
    plan, launch, comparison = (read(plan_path), read(RUN / 'official_launch_receipt.json'),
                                read(RUN / 'official_comparison_receipt.json'))
    check(digest(plan_path) == PLAN_SHA == launch['plan_sha256'] == comparison['plan_sha256'], 'Plan chain differs')
    check(launch['status'] == comparison['status'] == 'completed' and launch['exit_code'] == 0, 'Not completed exit0')
    check(plan['bootstrap_repeats'] == 5000 and plan['bootstrap_seed'] == 20260926, 'Bootstrap protocol changed')
    for path, sha in plan['bound_inputs'].items():
        check(digest(path) == sha, 'Bound evaluation input changed: ' + path)
    actual_source = {str(p.relative_to(plan['source_snapshot'])): digest(p)
                     for p in Path(plan['source_snapshot']).rglob('*.py')}
    check(actual_source == plan['source_hashes'], 'Evaluation snapshot changed')
    metrics_path = RUN / 'evaluation_official/official_metrics.json'
    check(digest(metrics_path) == METRICS_SHA, 'Unexpected candidate score bytes')
    data, receipt = bound_metrics(metrics_path)
    check(digest(receipt['checkpoint']) == receipt['checkpoint_sha256'] == launch['checkpoint_sha256'], 'Wrong field endpoint')
    for record in [receipt['entrypoint_source'], *receipt['loaded_source_modules'].values()]:
        check(digest(record['path']) == record['sha256'], 'Loaded source changed')
    check(receipt['predictions_finished_utc'] <= receipt['source_scoring_started_utc'], 'Prediction/scoring phase order differs')
    predictions = {v['name']: v for v in receipt['predictions']}
    sources = {v['name']: v for v in receipt['source_records']}
    views = {v['name']: v for v in data['views']}
    check(len(predictions) == len(sources) == 50 and set(predictions) == set(sources) == set(views), 'Incomplete predictions/GT binding')
    psnr_records, fingerprint_records = [], []
    for name in sorted(views):
        v, p, s = views[name], predictions[name], sources[name]
        for kind in ('rgb', 'mask'):
            check(digest(p[kind]) == p[kind + '_sha256'], 'Prediction PNG hash mismatch')
        rgb = cv2.imread(p['rgb'], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(p['mask'], cv2.IMREAD_UNCHANGED)
        check(rgb is not None and rgb.dtype == np.uint8 and rgb.shape == (v['height'], v['width'], 3), 'RGB PNG dimensions/type differ')
        check(mask is not None and mask.dtype == np.uint8 and mask.shape == rgb.shape[:2]
              and np.isin(mask, [0, 1, 2, 3, 4]).all(), 'Mask PNG dimensions/type differ')
        check(digest(s['source_image_path']) == s['source_image_sha256'], 'Original RGB changed')
        if s['source_annotation_path'] is not None:
            check(digest(s['source_annotation_path']) == s['source_annotation_sha256'], 'Original annotation changed')
        truth = cv2.imread(s['source_image_path'], cv2.IMREAD_UNCHANGED)
        check(truth is not None and truth.dtype == np.uint8 and truth.shape == rgb.shape, 'Original target dimensions differ')
        # Both BGR arrays have the same channel permutation, so MSE is invariant.
        # Match declared delivered-PNG contract: normalize float32, accumulate float64.
        error = (rgb.astype(np.float32) / 255).astype(np.float64) - (truth.astype(np.float32) / 255).astype(np.float64)
        mse = float(np.sum(error * error, dtype=np.float64) / error.size)
        psnr = float(-10 * np.log10(max(mse, 1e-12)))
        discrepancy = abs(psnr - v['psnr'])
        check(discrepancy < 1e-10, 'Independently recomputed PNG PSNR differs')
        psnr_records.append({'name': name, 'psnr': psnr, 'absolute_difference': discrepancy,
                             'rgb_sha256': p['rgb_sha256'], 'mask_sha256': p['mask_sha256'],
                             'source_rgb_sha256': s['source_image_sha256'], 'width': v['width'], 'height': v['height']})
        fingerprint_records.append({k: s[k] for k in ('camera', 'source_image_sha256', 'source_annotation_sha256', 'rasterized_mask_sha256')})
    payload = {'evaluation_family': data['evaluation_family'], 'scoring_protocol': data['scoring_protocol'],
               'views': sorted(fingerprint_records, key=lambda x: x['camera']['name'])}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    check(fingerprint == data['official_evaluation_fingerprint'], 'Canonical common fingerprint differs')
    gate = read(RUN / 'semantic_investment_gate.json')
    check(digest(RUN / 'semantic_investment_gate.json') == comparison['gate_sha256'], 'Gate receipt SHA differs')
    comparisons, paired_errors, reference_hashes = {}, {}, {}
    for name, path in plan['references'].items():
        ref, ref_receipt = bound_metrics(path)
        check(ref['official_evaluation_fingerprint'] == fingerprint and ref['scoring_protocol'] == data['scoring_protocol'], 'Mixed scoring protocols')
        paired_path = RUN / f'paired_official_rgb_minus_{name}.json'
        check(digest(paired_path) == comparison['paired_reports'][name], 'Paired report receipt SHA differs')
        pair = read(paired_path)
        for role, metric_file, execution_receipt in [('reference', Path(path), ref_receipt), ('candidate', metrics_path, receipt)]:
            binding = pair[role + '_source']
            check(binding['path'] == str(metric_file) and binding['sha256'] == digest(metric_file)
                  and binding['receipt_sha256'] == digest(metric_file.parent / 'execution_receipt.json')
                  and binding['checkpoint_sha256'] == execution_receipt['checkpoint_sha256'], 'Paired source binding differs')
        comparisons[name] = independent_bootstrap(ref, data)
        errors = []
        for key in RGB_KEYS:
            for field, value in comparisons[name][key].items():
                difference = np.max(np.abs(np.asarray(value) - np.asarray(pair['metrics'][key][field])))
                check(difference < 1e-12, 'Independent paired RGB bootstrap differs')
                errors.append(float(difference))
            check(gate['comparisons'][name][key] == pair['metrics'][key], 'Gate used different pair')
        paired_errors[name] = max(errors)
        reference_hashes[name] = {'path': path, 'sha256': digest(path), 'receipt_sha256': digest(Path(path).parent / 'execution_receipt.json')}
    threshold = plan['continue_to_semantic_gate']
    check(threshold == read(RUN / 'plan.json')['continue_to_semantic_gate'], 'Adoption gate changed after training')
    strong = comparisons['ssim_fixed']
    clauses = {'psnr_gain_at_least_0_15_dB': strong['psnr']['difference'] >= threshold['psnr_mean_improvement_at_least_dB'],
               'psnr_paired_95_lower_positive': strong['psnr']['paired_view_bootstrap_95_interval'][0] > threshold['paired_psnr_95CI_lower_strictly_above'],
               'ssim_point_not_lower': strong['ssim']['difference'] >= 0,
               'lpips_point_not_higher': strong['lpips']['difference'] <= 0}
    check(clauses == gate['clauses'] and all(clauses.values()) == gate['passed'], 'Independent gate differs')
    report = {'status': 'passed', 'cpu_only': True, 'audit_script_sha256': digest(__file__),
              'official_metrics_sha256': METRICS_SHA, 'official_evaluation_plan_sha256': PLAN_SHA,
              'official_execution_receipt_sha256': digest(metrics_path.parent / 'execution_receipt.json'),
              'official_comparison_receipt_sha256': digest(RUN / 'official_comparison_receipt.json'),
              'official_launch_receipt_sha256': digest(RUN / 'official_launch_receipt.json'),
              'endpoint_audit_sha256': digest(RUN / 'cpu_endpoint_audit.json'),
              'source_and_bound_inputs_endpoint_hashes_exact': True,
              'candidate_rgb_metrics': {k: data[k] for k in RGB_KEYS},
              'png_count': {'rgb': 50, 'mask': 50}, 'png_dimensions_and_hashes_exact': True,
              'per_view_psnr_recomputed': psnr_records,
              'psnr_max_absolute_discrepancy': max(x['absolute_difference'] for x in psnr_records),
              'common_fingerprint_independently_serialized': fingerprint,
              'fingerprint_limit': 'Source RGB and annotation bytes rehashed; recorded rasterized-mask hash retained, annotation rasterization not rerun.',
              'scoring_limit': 'PSNR recomputed from PNG/GT only. SSIM and LPIPS preserve frozen scoring source and hash-bound per-view values; no costly rescoring.',
              'bootstrap_method': 'Independent NumPy implementation: 5000 draws of 50 view IDs, seed20260926, bincount-weighted paired deltas, percentile linear95CI.',
              'paired_numeric_tolerance': 1e-12, 'paired_max_discrepancy': paired_errors,
              'reference_sources': reference_hashes, 'paired_rgb': comparisons,
              'investment_gate': {'passed': all(clauses.values()), 'clauses': clauses},
              'scope': 'Fixed development-view RGB only; single-run conditional bootstrap, not multiple seeds, selection-adjusted uncertainty, blind test or innovation evidence. Untrained semantic scores excluded.'}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'audit_sha256': digest(output),
                      'psnr_max_discrepancy': report['psnr_max_absolute_discrepancy'],
                      'bootstrap_max_discrepancy': paired_errors, 'gate': report['investment_gate']}, indent=2))


if __name__ == '__main__':
    main()
