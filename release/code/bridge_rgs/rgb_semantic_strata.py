"""Descriptive RGB errors on fixed GT classes, not new semantic predictions."""
from __future__ import annotations

import numpy as np

LABELS = (0, 1, 2, 3, 4, 255)
NAMES = ('background', 'deck', 'stay_cable', 'tower', 'foundation', 'ignore')


def class_error_sums(prediction, target, mask):
    """Return pixel counts and RGB-channel SSE; retain ignore for reconciliation.

    Match the official RGB conversion (uint8 -> float32/255 -> float64
    subtraction), without recomputing SSIM/LPIPS or changing the scorer.
    """
    if (prediction.dtype != np.uint8 or target.dtype != np.uint8
            or prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3
            or mask.dtype != np.uint8 or mask.shape != target.shape[:2]
            or not np.isin(mask, LABELS).all()):
        raise ValueError('Matching uint8 RGB and official GT class grid required')
    error = ((prediction.astype(np.float32)/255).astype(np.float64)
             -(target.astype(np.float32)/255).astype(np.float64))**2
    channel_sse = error.sum(-1)
    counts = np.asarray([np.count_nonzero(mask == label) for label in LABELS], np.int64)
    sums = np.asarray([channel_sse[mask == label].sum(dtype=np.float64) for label in LABELS])
    if counts.sum() != mask.size or not np.isfinite(sums).all():
        raise ValueError('Invalid class partition')
    return counts, sums


def class_summary(counts, sums):
    counts, sums = np.asarray(counts), np.asarray(sums)
    if (counts.ndim != 2 or counts.shape[1] != 6 or sums.shape != counts.shape
            or not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any()
            or not np.isfinite(sums).all() or (sums < 0).any() or (sums[counts == 0] != 0).any()):
        raise ValueError('Invalid per-view class sufficient statistics')
    result = {}
    for k, name in enumerate(NAMES):
        pixels = int(counts[:, k].sum()); total = float(sums[:, k].sum())
        mse = total/(3*pixels) if pixels else None
        result[name] = {'pixels': pixels, 'views_present': int(np.count_nonzero(counts[:, k])),
                        'sse_rgb_channels': total, 'pooled_mse': mse,
                        'pooled_psnr': -10*float(np.log10(max(mse, 1e-12))) if pixels else None,
                        'semantic_evaluation_class': k < 5}
    return result


def paired_class_errors(counts, reference, candidate, *, repeats=5000, seed=20260926):
    """Bootstrap whole views, with the identical resampled GT denominator.

    Class PSNR is computed from pooled class MSE, not mean per-view PSNR.
    Completely absent classes remain missing, never zero-error successes.
    Ignore is descriptive reconciliation only, excluded from paired reports.
    """
    counts = np.asarray(counts); reference = np.asarray(reference); candidate = np.asarray(candidate)
    a, b = class_summary(counts, reference), class_summary(counts, candidate)
    if len(counts) == 0 or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('Positive view population and bootstrap count required')
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(counts), size=(repeats, len(counts)))
    hist = np.asarray([np.bincount(x, minlength=len(counts)) for x in drawn], np.int64)
    denominator = 3*(hist@counts)
    result = {}
    for k, name in enumerate(NAMES[:5]):
        present = denominator[:, k] > 0
        if a[name]['pooled_mse'] is None:
            result[name] = {'reference': a[name], 'candidate': b[name], 'difference_mse': None,
                            'difference_psnr': None, 'mse_95_interval': None,
                            'psnr_95_interval': None, 'finite_replicates': 0}
            continue
        ma = (hist@reference[:, k])[present]/denominator[present, k]
        mb = (hist@candidate[:, k])[present]/denominator[present, k]
        dp = -10*np.log10(np.maximum(mb, 1e-12))+10*np.log10(np.maximum(ma, 1e-12))
        result[name] = {'reference': a[name], 'candidate': b[name],
                        'difference_mse': b[name]['pooled_mse']-a[name]['pooled_mse'],
                        'difference_psnr': b[name]['pooled_psnr']-a[name]['pooled_psnr'],
                        'mse_95_interval': np.quantile(mb-ma, [.025, .975]).tolist(),
                        'psnr_95_interval': np.quantile(dp, [.025, .975]).tolist(),
                        'finite_replicates': int(present.sum())}
    return {'classes': result, 'bootstrap_repeats': repeats, 'seed': seed,
            'direction': 'candidate minus reference; smaller MSE and larger PSNR improve',
            'scope': 'descriptive reused development views; no adoption gate, semantic score or multiplicity correction'}
