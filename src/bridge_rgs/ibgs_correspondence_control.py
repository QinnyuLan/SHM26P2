"""Matched normalized-layer control with sparse, read-only intervention diagnostics."""
from __future__ import annotations

import torch

from .ibgs_layer_controls import permute_supported_layer_features


@torch.no_grad()
def permute_with_diagnostics(features, weights, support, *, diagnose=False, chunk_pixels=65536):
    if chunk_pixels < 1:
        raise ValueError('Positive diagnostic chunk size required')
    before = [(id(t), t._version) for t in (features, weights, support)]
    permuted = permute_supported_layer_features(features, weights, support)
    if before != [(id(t), t._version) for t in (features, weights, support)]:
        raise ValueError('Control modified original evidence or coefficients')
    if not diagnose:
        return permuted, {}
    totals = {'pixel_source_groups': 0, 'supported_groups': 0, 'multilayer_groups': 0,
              'active_feature_vectors': 0, 'changed_active_feature_vectors': 0,
              'changed_multilayer_groups': 0, 'equal_coefficient_multilayer_groups': 0,
              'identical_feature_multilayer_groups': 0, 'weighted_raw_sum_changed_groups': 0}
    largest = 0.
    for start in range(0, len(features), chunk_pixels):
        f, p = features[start:start+chunk_pixels], permuted[start:start+chunk_pixels]
        coefficient = weights[start:start+chunk_pixels, :, None]*support[start:start+chunk_pixels]
        active = coefficient > 0; count = active.sum(1); multi = count >= 2
        changed = (f != p).any(-1) & active
        equal = (coefficient.amax(1) == coefficient.masked_fill(~active, float('inf')).amin(1)) & multi
        # FP64 diagnostics only. These values never replace FP32 head inputs.
        delta = (p.double()-f.double()).masked_fill(~active[..., None], 0)
        difference = (delta*coefficient.double()[..., None]).sum(1).abs().amax(-1)
        if not torch.isfinite(difference).all():
            raise ValueError('Nonfinite supported feature diagnostic')
        totals['pixel_source_groups'] += count.numel()
        totals['supported_groups'] += int((count > 0).sum())
        totals['multilayer_groups'] += int(multi.sum())
        totals['active_feature_vectors'] += int(active.sum())
        totals['changed_active_feature_vectors'] += int(changed.sum())
        totals['changed_multilayer_groups'] += int((multi & changed.any(1)).sum())
        totals['equal_coefficient_multilayer_groups'] += int(equal.sum())
        totals['identical_feature_multilayer_groups'] += int((multi & ~changed.any(1)).sum())
        totals['weighted_raw_sum_changed_groups'] += int((multi & ~equal & (difference > 1e-12)).sum())
        largest = max(largest, float(difference.max()))
    stats = {**totals, 'weighted_raw_sum_max_absolute_change': largest,
             'weighted_raw_sum_change_threshold': 1e-12,
             'multilayer_fraction_all_groups': totals['multilayer_groups']/max(totals['pixel_source_groups'], 1),
             'changed_fraction_active_vectors': totals['changed_active_feature_vectors']/max(totals['active_feature_vectors'], 1),
             'diagnostic_scope': 'Before MLP; does not measure the learned pooled activation or causal attribution'}
    return permuted, stats
