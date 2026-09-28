import numpy as np
import pytest
import torch

from bridge_rgs.projection_stress import (
    consensus_proxy,
    score_predictions,
    select_groups,
    zero_mean_jitter,
)
from bridge_rgs.reliability import ellipse_sample_probabilities, fuse_multiview_evidence


def test_groups_use_only_training_camera_metadata():
    views = []
    for i in range(24):
        pose = np.eye(4)
        pose[0, 3] = -i
        views.append({"name": f"{i:03}", "split": "val" if i % 5 == 0 else "train",
                      "mask_path": "must_not_be_opened", "w2c": pose.tolist()})
    groups = select_groups(views, anchors=8, group_size=4)
    names = {v["name"] for v in views if v["split"] == "train"}
    assert len(groups) == 8
    assert len({g["anchor"] for g in groups}) == 8
    assert all(len(g["views"]) == len(set(g["views"])) == 4 for g in groups)
    assert all(set(g["views"]) <= names for g in groups)
    assert groups == select_groups(views[::-1], anchors=8, group_size=4)


def test_proxy_needs_distinct_original_views_and_sufficient_purity():
    labels = np.array([[2, 1, 1, -1], [2, -1, 1, -1], [-1, -1, 2, -1], [-1, -1, 1, -1]])
    proxy, votes, purity = consensus_proxy(labels, ["a", "b", "c", "d"])
    np.testing.assert_array_equal(proxy, [2, -1, -1, -1])
    np.testing.assert_array_equal(votes, [2, 1, 4, 0])
    assert purity[2] == .75
    with pytest.raises(ValueError, match="distinct"):
        consensus_proxy(labels, ["a", "a", "c", "d"])


def test_fixed_coverage_does_not_fill_with_rejected_predictions():
    proxy = np.zeros(8, int)
    prediction = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    full = score_predictions(proxy, prediction, np.arange(8, 0, -1), coverages=(.25, .5, 1))
    rejected = score_predictions(proxy, prediction, [8, 7, 0, 0, 0, 0, 0, 0], coverages=(.25, .5, 1))
    assert full["error_rate"] == .75
    assert rejected["error_rate"] == 0
    assert rejected["coverage"] == .25
    assert full["fixed_coverage_curve"][0] == rejected["fixed_coverage_curve"][0]
    assert not rejected["fixed_coverage_curve"][1]["reachable"]
    assert rejected["fixed_coverage_curve"][1]["error_rate"] is None


def test_noise_is_reproducible_centered_and_same_across_methods():
    first = zero_mean_jitter(4096, 43)
    np.testing.assert_array_equal(first, zero_mean_jitter(4096, 43))
    np.testing.assert_allclose(first.mean(0), 0, atol=1e-7)
    assert not np.array_equal(first, zero_mean_jitter(4096, 44))
    # Finite Gaussian draws fluctuate; this is not an empirical whitening step.
    np.testing.assert_allclose(np.cov(first.T), np.eye(2), atol=.1)


def test_sigma_zero_identity_and_scoring_cannot_change_evidence():
    p = torch.zeros(5, 12, 12)
    p[0, :, :6] = 1
    p[2, :, 6:] = 1
    uv = torch.tensor([[4.9, 5.], [6., 5.]])
    covariance = torch.eye(2)[None].repeat(2, 1, 1) * .25
    a = ellipse_sample_probabilities(p, uv, covariance, confidence_threshold=.65)
    b = ellipse_sample_probabilities(p, uv + 0 * torch.tensor(zero_mean_jitter(2, 43)),
                                    covariance + 0**2 * torch.eye(2), confidence_threshold=.65)
    assert torch.equal(a.probabilities, b.probabilities)
    assert torch.equal(a.weights, b.weights)
    fused = fuse_multiview_evidence(torch.stack([a.probabilities, b.probabilities]),
                                   torch.stack([a.weights, b.weights]))
    original_weight = fused.weights.clone()
    first = score_predictions(np.array([0, 2]), fused.probabilities.argmax(-1).numpy(), fused.weights.numpy())
    second = score_predictions(np.array([2, 0]), fused.probabilities.argmax(-1).numpy(), fused.weights.numpy())
    assert first["error_rate"] != second["error_rate"]
    assert torch.equal(fused.weights, original_weight)


def test_empty_proxy_reports_missing_error_and_zero_population():
    result = score_predictions(np.array([-1, -1]), np.array([0, 2]), np.array([.8, .9]))
    assert result["proxy_count"] == result["accepted"] == 0
    assert result["error_rate"] is result["coverage"] is None
