"""Small CPU contracts for the independent RGB-only endpoint auditor."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('capacity_rgb_audit', ROOT/'scripts/audit_rgb_capacity_official.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_png_psnr_delivered_float32_normalization_and_channel_permutation():
    rgb = np.arange(90, dtype=np.uint8).reshape(5, 6, 3)
    truth = rgb+3
    error = rgb.astype(np.float32)/255-truth.astype(np.float32)/255
    # Cast before subtraction, matching official PNG precision; not FP32 difference.
    precise = (rgb.astype(np.float32)/255).astype(np.float64)-(truth.astype(np.float32)/255).astype(np.float64)
    expected = -10*np.log10(np.square(precise).mean())
    assert np.isfinite(error).all()
    assert audit.png_psnr(rgb, truth) == expected
    assert audit.png_psnr(rgb[..., ::-1], truth[..., ::-1]) == pytest.approx(expected, abs=1e-12)
    assert audit.png_psnr(rgb, rgb) == 120
    with pytest.raises(ValueError):
        audit.png_psnr(rgb.astype(float), truth)


def test_independent_count_bootstrap_matches_direct_index_and_direction():
    old = {'views': [{'name': f'{i:03}', 'width': 12, 'height': 13, 'rgb_pixels': 156,
                      'psnr': 25+i/100, 'ssim': .8+i/1000, 'lpips': .3-i/1000} for i in range(50)]}
    new = copy.deepcopy(old)
    for i, view in enumerate(new['views']):
        view['psnr'] += (i-20)/100
        view['ssim'] += (i-20)/10000
        view['lpips'] -= (i-20)/10000
    result = audit.helper.independent_bootstrap(old, new)
    sample = np.random.default_rng(20260926).integers(50, size=(5000, 50))
    for key in audit.helper.RGB_KEYS:
        delta = np.array([b[key]-a[key] for a, b in zip(old['views'], new['views'], strict=True)])
        assert np.allclose(result[key]['paired_view_bootstrap_95_interval'],
                           np.quantile(delta[sample].mean(1), [.025, .975]), rtol=0, atol=1e-12)
    assert result['psnr']['difference'] > 0 and result['lpips']['difference'] < 0


def test_incomplete_scoring_never_hashes_or_reads_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, 'RUN', tmp_path)
    values = {'official_comparison_receipt.json': {'status': 'running'},
              'official_launch_receipt.json': {'status': 'completed', 'exit_code': 0},
              'cpu_endpoint_audit.json': {'status': 'passed'}}
    monkeypatch.setattr(audit, 'read', lambda path: values[Path(path).name])
    def forbidden(*args):
        raise AssertionError('No result or checkpoint may be hashed before completion')
    monkeypatch.setattr(audit, 'digest', forbidden)
    with pytest.raises(ValueError, match='completed'):
        audit.main('pending')


def test_gate_is_rgb_only_and_requires_psnr_effect_size_and_positive_ci():
    metrics = {'psnr': {'difference': .15, 'paired_view_bootstrap_95_interval': [.01, .3]},
               'ssim': {'difference': 0}, 'lpips': {'difference': 0}}
    assert all(audit.gate_clauses(metrics).values())
    metrics['psnr']['difference'] = .149
    assert not all(audit.gate_clauses(metrics).values())
    metrics['psnr']['difference'] = .15
    metrics['psnr']['paired_view_bootstrap_95_interval'][0] = 0
    assert not all(audit.gate_clauses(metrics).values())
    with pytest.raises(ValueError, match='semantic'):
        audit.gate_clauses(dict(metrics, miou_all={'difference': 1}))
