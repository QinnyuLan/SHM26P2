"""CPU aggregation of the fixed sparse-front pair; no threshold selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def summarize(root):
    plan_path = root / 'diagnostic_plan/plan.json'
    plan = json.loads(plan_path.read_text())
    prefixes = [Path(row['name']).stem for row in plan['views']]
    with np.load(root / 'diagnostic_plan/targets.npz', allow_pickle=False) as targets:
        reference_weights = np.concatenate([targets[p+'_confidence'] for p in prefixes])
        reference_covered = np.concatenate([targets[p+'_covered'] for p in prefixes])
    assert digest(root / 'diagnostic_plan/targets.npz') == plan['targets_sha256']
    audit_path = root / 'pair_invariance_audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'passed' and audit['paired_sampler_state_identical']
    summary = {'plan_sha256': digest(plan_path), 'audit_sha256': digest(audit_path),
               'script_sha256': digest(__file__), 'targets': len(reference_weights),
               'covered': int(reference_covered.sum()), 'train_views': [r['name'] for r in plan['views']],
               'aggregation': 'Float64 confidence-weighted pooled target means; quantiles are unweighted [min,p10,p50,p90,max]. Same fixed targets, weights and coverage in all arms.',
               'diagnostics': {}, 'evaluation': audit}
    combined = {}
    for name in ('before', '00', '01'):
        directory = root / ('diagnostics_'+name)
        report = json.loads((directory / 'report.json').read_text())
        assert report['complete'] and report['scene_bitwise_unchanged']
        assert not report['validation_or_rgb_pixels_read'] and report['optimizer_steps'] == 0
        assert report['plan_sha256'] == summary['plan_sha256']
        assert digest(directory / 'mass_results.npz') == report['mass_results_sha256']
        if name != 'before':
            arm = '00_rgb_ed' if name == '00' else '01_rgb_ed_front'
            receipt = json.loads((root / arm / 'experiment_receipt.json').read_text())
            assert receipt['status'] == 'completed'
            assert report['measured_checkpoint_sha256'] == audit['arms'][arm]['checkpoint_sha256'] == receipt['checkpoint_sha256']
        with np.load(directory / 'mass_results.npz', allow_pickle=False) as source:
            arrays = {key: np.concatenate([source[p+'_'+key] for p in prefixes]) for key in
                      ('confidence', 'covered', 'alpha', 'low16', 'high16', 'low32', 'high32',
                       'near_band_lower16', 'near_band_upper16', 'near_band_lower32', 'near_band_upper32')}
        assert np.array_equal(arrays['confidence'], reference_weights)
        assert np.array_equal(arrays['covered'], reference_covered)
        keep = reference_covered
        weights = reference_weights[keep].astype(np.float64)
        stats = {'checkpoint_sha256': report['measured_checkpoint_sha256'],
                 'report_sha256': digest(directory / 'report.json'),
                 'mass_results_sha256': report['mass_results_sha256'], 'pooled': {},
                 'alpha_below_0.5_count': int((arrays['alpha'] < .5).sum()),
                 'alpha_below_0.1_count': int((arrays['alpha'] < .1).sum())}
        for bins in (16, 32):
            arrays[f'interval{bins}'] = arrays[f'high{bins}'].astype(np.float64)-arrays[f'low{bins}']
        for key, values in arrays.items():
            if key in ('confidence', 'covered'):
                continue
            assert np.isfinite(values).all()
            values = values[keep].astype(np.float64)
            stats['pooled'][key] = {'weighted_mean': float(np.average(values, weights=weights)),
                                    'quantiles': np.quantile(values, [0, .1, .5, .9, 1]).tolist()}
        summary['diagnostics'][name] = stats
        combined[name] = arrays
    for left, right in [('00', 'before'), ('01', 'before'), ('01', '00')]:
        name = left+'_minus_'+right
        summary[name] = {}
        weights = reference_weights[reference_covered].astype(np.float64)
        for key in ('alpha', 'low32', 'high32', 'interval32', 'near_band_lower32', 'near_band_upper32'):
            delta = combined[left][key][reference_covered].astype(np.float64)-combined[right][key][reference_covered]
            summary[name][key] = {'weighted_mean_delta': float(np.average(delta, weights=weights)),
                                  'quantiles': np.quantile(delta, [0, .1, .5, .9, 1]).tolist(),
                                  'fraction_decreased': float((delta < -1e-6).mean())}
    a, b = audit['arms']['01_rgb_ed_front'], audit['arms']['00_rgb_ed']
    summary['evaluation_01_minus_00'] = {k: a['metrics'][k]-b['metrics'][k]
                                        for k in ('psnr', 'ssim', 'lpips', 'miou_all', 'miou_foreground')}
    summary['evaluation_01_minus_00']['iou'] = (np.array(a['metrics']['iou'])-b['metrics']['iou']).tolist()
    summary['evaluation_01_minus_00']['raw3d_iou'] = (np.array(a['raw3d']['iou'])-b['raw3d']['iou']).tolist()
    summary['limitations'] = [
        'Fixed single-scene engineering control, not a novel loss or cross-scene result.',
        'Bounds describe this rasterizer center-depth mass, not physical surface termination or calibrated 3D truth.',
        'The 32-bin intervals remain broad; lower-loss reduction does not establish exact free-space reduction.',
        'Sparse TRAIN anchors cannot rule out local holes or degradation outside their support.',
        'New CDF corner-coordinate sampling differs from existing ED; whole preprocessing convention is not yet unified.'
    ]
    (root / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for name, item in summary['diagnostics'].items():
        print(name, {key: round(item['pooled'][key]['weighted_mean'], 7) for key in
                     ('alpha', 'low32', 'high32', 'interval32', 'near_band_lower32', 'near_band_upper32')},
              'alpha<.5', item['alpha_below_0.5_count'],
              'interval_median', item['pooled']['interval32']['quantiles'][2])
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/sparse_front_pair'))
    summarize(parser.parse_args().root)
