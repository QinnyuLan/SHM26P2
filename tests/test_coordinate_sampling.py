"""Corner-aware sparse supervision and historical front-gate compatibility."""
import numpy as np
import pytest
import torch

from bridge_rgs.coordinates import CORNER, LEGACY
from bridge_rgs.ray_support import (
    SparseDepthSupport,
    SparseDepthTargets,
    _sample,
    sparse_depth_loss,
)
from bridge_rgs.ray_termination import SparseFrontSupport, sample_corner_pixel_features
from bridge_rgs.reliability import _sample_map, ellipse_sample_probabilities


def linear_field(height=5, width=6):
    y, x = torch.meshgrid(torch.arange(height)+.5, torch.arange(width)+.5, indexing="ij")
    return 1 + x + 2*y


def test_sparse_ellipse_and_front_share_corner_linear_field_oracle():
    plane = linear_field()
    uv = torch.tensor([[.5, .5], [2.3, 1.9], [5.5, 4.5]])
    expected = 1 + uv[:, 0] + 2*uv[:, 1]
    torch.testing.assert_close(_sample(plane, uv, CORNER), expected)
    torch.testing.assert_close(_sample_map(plane, uv[:, None], CORNER)[:, 0, 0], expected)
    torch.testing.assert_close(sample_corner_pixel_features(plane[..., None], uv)[:, 0], expected)
    # Retain the old +.5 on each axis rather than silently changing old losses.
    torch.testing.assert_close(_sample(plane, uv[:2]), expected[:2] + 1.5)
    torch.testing.assert_close(_sample_map(plane, uv[:2, None])[:, 0, 0], expected[:2] + 1.5)
    targets = SparseDepthTargets(uv[:2], expected[:2], torch.ones(2), torch.arange(2), {}, CORNER)
    loss, _ = sparse_depth_loss(plane, torch.ones_like(plane), targets)
    assert loss < 1e-12
    targets.pixel_protocol = LEGACY
    old_loss, _ = sparse_depth_loss(plane, torch.ones_like(plane), targets)
    assert old_loss > .001


def test_zero_covariance_ellipse_samples_all_evidence_at_same_corner():
    depth = linear_field()
    y, x = torch.meshgrid(torch.arange(5)+.5, torch.arange(6)+.5, indexing="ij")
    p0 = .1 + x/10
    probabilities = torch.stack([p0, 1-p0])
    confidence = .2 + y/10
    uv = torch.tensor([[.5, .5], [2.3, 1.9], [5.5, 4.5]])
    options = {"depths": 1+uv[:, 0]+2*uv[:, 1], "depth_map": depth,
               "alpha_map": torch.ones_like(depth), "teacher_confidence_map": confidence,
               "depth_relative_tolerance": 0, "depth_absolute_tolerance": 1e-4}
    result = ellipse_sample_probabilities(probabilities, uv, torch.zeros(3, 2, 2),
                                          pixel_protocol=CORNER, **options)
    torch.testing.assert_close(result.probabilities[:, 0], .1+uv[:, 0]/10)
    torch.testing.assert_close(result.weights, .2+uv[:, 1]/10)
    torch.testing.assert_close(result.valid_fraction, torch.ones(3))
    old = ellipse_sample_probabilities(probabilities, uv, torch.zeros(3, 2, 2), **options)
    assert torch.count_nonzero(old.weights) == 0


def make_support(uv, protocol, border=0, observed=False):
    uv = np.asarray(uv, np.float64)
    depth = np.arange(len(uv)) + 2.
    points = np.concatenate((uv * depth[:, None], depth[:, None]), 1)
    n = len(points)
    arrays = {"points": points, "covariances": np.tile(np.eye(3)*1e-6, (n, 1, 1)),
              "reprojection_error": np.full(n, .1), "num_observations": np.full(n, 3),
              "observation_image_ids": np.tile([1, 2, 3], n),
              "observation_offsets": np.arange(n+1)*3, "track_ids": np.arange(n)}
    views = [{"image_id": i, "name": str(i), "split": "train", "width": 6, "height": 5,
              "pixel_protocol": protocol} for i in (1, 2, 3)]
    rays = {view["image_id"]: dict(enumerate(uv)) for view in views} if observed else None
    return SparseDepthSupport(arrays, views,
                               {"uv_mode": "observed" if observed else "projected", "border_px": border},
                               observed_rays=rays)


def test_target_bounds_and_validity_gate_move_together():
    uv = [[.5, .5], [5.5, 4.5], [0., 0.], [5.6, 4.5], [2.5, 2.5]]
    K, pose, valid = torch.eye(3), torch.eye(4), torch.ones(5, 6)
    new = make_support(uv, CORNER)
    targets = new.targets_for_view(1, K, pose, 6, 5, valid)
    assert targets.point_indices.tolist() == [0, 1, 4]
    assert targets.pixel_protocol == CORNER
    old = make_support(uv, LEGACY).targets_for_view(1, K, pose, 6, 5, valid)
    assert old.point_indices.tolist() == [0, 2, 4]
    valid[2, 2] = 0
    targets = new.targets_for_view(1, K, pose, 6, 5, valid)
    assert targets.point_indices.tolist() == [0, 1]


def test_legacy_front_keeps_correct_sampler_but_historical_ed_pregate():
    K, pose, valid = torch.eye(3), torch.eye(4), torch.ones(5, 6)
    valid[2, 2] = 0
    uv = [[1.5, 1.5], [.5, .5]]
    legacy = SparseFrontSupport(make_support(uv, LEGACY, observed=True))
    corner = SparseFrontSupport(make_support(uv, CORNER, observed=True))
    a = legacy.targets(1, K, pose, 6, 5, valid)
    b = corner.targets(1, K, pose, 6, 5, valid)
    assert a["pixels"].tolist() == [[.5, .5]]
    assert b["pixels"].tolist() == uv
    assert legacy.provenance["pixel_protocol"] == LEGACY
    assert corner.provenance["pixel_protocol"] == CORNER
    # The front sampler itself was already correct and remains unversioned.
    probe = sample_corner_pixel_features(valid[..., None], torch.tensor(uv))[:, 0]
    torch.testing.assert_close(probe, torch.ones(2))


def test_fusion_rejects_cross_protocol_before_reading_pseudo_or_using_cuda():
    from types import SimpleNamespace

    from bridge_rgs.fusion import fuse_scene_evidence
    scene = SimpleNamespace(pixel_protocol=LEGACY)
    with pytest.raises(ValueError, match="protocol mismatch"):
        fuse_scene_evidence(scene, [{"pixel_protocol": CORNER}], None, "/not/read")
    with pytest.raises(ValueError, match="mix views"):
        fuse_scene_evidence(scene, [{"pixel_protocol": CORNER}, {}], None, "/not/read")
