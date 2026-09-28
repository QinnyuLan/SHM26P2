"""Small CPU contracts for the single fixed RGB combination."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

WORKER = Path(__file__).with_name('evaluate_fixed_rgb_ensemble.py')
if not WORKER.exists():
    WORKER = Path(__file__).resolve().parents[1]/'scripts/evaluate_fixed_rgb_ensemble.py'
spec = importlib.util.spec_from_file_location('fixed_rgb_ensemble_test', WORKER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def dataset(offset=0.):
    return {'views': [{'name': f'{i:03d}.png', 'width': 20, 'height': 30, 'rgb_pixels': 600,
                       'psnr': 25+i*.01+offset, 'ssim': .8+i*.0001+offset*.01,
                       'lpips': .3-i*.0001-offset*.01} for i in range(50)]}


def test_uint8_mean_no_overflow_and_bankers_rounding():
    first = np.array([[[0, 1, 2], [254, 255, 255]]], np.uint8)
    second = np.array([[[1, 2, 3], [255, 254, 255]]], np.uint8)
    actual = module.average_png(first, second)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, [[[0, 2, 2], [254, 254, 255]]])
    np.testing.assert_array_equal(module.average_png(first, first), first)
    with pytest.raises(ValueError):
        module.average_png(first.astype(np.float32), second)


def test_rgb_only_api_placeholder_cannot_escape_as_semantic_score():
    image = np.zeros((12, 13, 3), np.uint8)
    def scorer(prediction, placeholder, target, target_mask, perceptual, device):
        assert target_mask is None and placeholder.shape == image.shape[:2]
        assert placeholder.dtype == np.uint8 and not placeholder.any()
        return {'psnr': 120., 'ssim': 1., 'lpips': 0., 'rgb_pixels': 156}
    result = module.score_rgb_only(scorer, image, image, None, 'cpu')
    assert set(result) == {'psnr', 'ssim', 'lpips', 'rgb_pixels'}
    with pytest.raises(ValueError, match='semantic'):
        module.score_rgb_only(lambda *a, **k: {**result, 'confusion_matrix': [[0]]}, image, image, None, 'cpu')


def test_rgb_bootstrap_same_sorted_pairs_and_independent_count_reference():
    reference, candidate = dataset(), dataset(.2)
    for i, row in enumerate(candidate['views']):
        row['psnr'] += .1*np.sin(i)
    pair = module.paired_rgb(reference, candidate)
    rng = np.random.default_rng(20260926)
    counts = np.stack([np.bincount(rng.integers(0, 50, 50), minlength=50) for _ in range(5000)])
    for key in module.RGB_KEYS:
        a, b = [np.array([v[key] for v in d['views']]) for d in (reference, candidate)]
        interval = np.quantile(counts@(b-a)/50, [.025, .975])
        np.testing.assert_allclose(interval, pair['metrics'][key]['paired_view_bootstrap_95_interval'], atol=1e-14, rtol=0)
        assert pair['metrics'][key]['difference'] == float(b.mean()-a.mean())
    assert module.paired_rgb({'views': reference['views'][::-1]}, candidate) == pair
    assert set(pair['metrics']) == set(module.RGB_KEYS)
    with pytest.raises(ValueError, match='unique'):
        module.paired_rgb({'views': reference['views'][:-1]}, candidate)


def test_four_gate_conjunction_and_no_semantic_keys():
    pair = module.paired_rgb(dataset(), dataset(.2))
    assert all(module.gate_clauses(pair['metrics']).values())
    pair['metrics']['psnr']['difference'] = .1
    pair['metrics']['psnr']['paired_view_bootstrap_95_interval'][0] = -.01
    clauses = module.gate_clauses(pair['metrics'])
    assert not clauses['psnr_gain_at_least_0_15_dB'] and not clauses['psnr_paired_95_lower_positive']
    assert clauses['ssim_point_not_lower'] and clauses['lpips_point_not_higher']
    with pytest.raises(ValueError, match='RGB-only'):
        module.gate_clauses({**pair['metrics'], 'miou': {}})
