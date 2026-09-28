import numpy as np
import pytest
import torch

from bridge_rgs.ray_support import SparseDepthSupport
from bridge_rgs.ray_termination import SparseFrontSupport
from bridge_rgs.train import optional_sparse_front_loss


def support():
    arrays = {"points": np.array([[0., 0., 2.], [.1, 0., 3.]]),
              "covariances": np.tile(np.eye(3)*1e-5, (2, 1, 1)),
              "reprojection_error": np.array([.1, .5]), "num_observations": np.array([3, 3]),
              "observation_image_ids": np.array([1, 2, 3, 1, 2, 3]),
              "observation_offsets": np.array([0, 3, 6]), "track_ids": np.array([10, 11])}
    K = torch.tensor([[20., 0, 8], [0, 20, 8], [0, 0, 1]])
    pose = torch.eye(4)
    views = [{"image_id": i, "split": "train", "width": 17, "height": 17} for i in (1, 2, 3)]
    rays = {i: {0: np.array([0., 0.]), 1: np.array([.1/3, 0.])} for i in (1, 2, 3)}
    return SparseFrontSupport(SparseDepthSupport(arrays, views, observed_rays=rays)), K, pose


def test_off_switch_never_accesses_support_data_or_scene():
    class Forbidden:
        def loss(self, *args):
            raise AssertionError("Disabled front loss attempted an extra render")
    assert optional_sparse_front_loss(Forbidden(), None, None, None, None, 0) == (None, {})
    with pytest.raises(ValueError, match="initialized support"):
        optional_sparse_front_loss(None, None, None, None, None, .05)


def test_front_targets_are_train_only_cached_and_use_observed_rays():
    front, K, pose = support()
    target = front.targets(1, K, pose, 17, 17, torch.ones(17, 17))
    assert len(target["depth"]) == 2
    assert torch.allclose(target["pixels"][1], torch.tensor([8+2/3, 8.]))
    assert not target["confidence"].requires_grad
    assert target is front.targets(1, K, pose, 17, 17, torch.ones(17, 17))
    changed = pose.clone()
    changed[0, 3] = .01
    with pytest.raises(ValueError, match="fixed cameras"):
        front.targets(1, K, changed, 17, 17, torch.ones(17, 17))
    with pytest.raises(ValueError, match="not a TRAIN"):
        front.targets(4, K, pose, 17, 17, torch.ones(17, 17))


def test_low_alpha_is_retained_and_mass_is_not_alpha_normalized(monkeypatch):
    front, K, pose = support()
    logit = torch.tensor(0., requires_grad=True)

    def fake_render(scene, intrinsic, extrinsic, width, height, edges):
        alpha = .2*logit.sigmoid()
        return (.1*logit.sigmoid()).expand(height, width, len(edges)), alpha.expand(height, width, 1)

    monkeypatch.setattr("bridge_rgs.ray_termination.render_front_mass", fake_render)
    loss, stats = front.loss(None, K, pose, 17, 17, torch.ones(17, 17), 1)
    assert stats["sparse_front_low_alpha_count"] == stats["sparse_front_covered"] == 2
    assert float(loss.detach()) == pytest.approx(.05)
    loss.backward()
    assert float(logit.grad) == pytest.approx(.025)


def test_no_supported_targets_skips_renderer(monkeypatch):
    front, K, pose = support()
    def forbidden(*args):
        raise AssertionError("No-target view should not render front features")
    monkeypatch.setattr("bridge_rgs.ray_termination.render_front_mass", forbidden)
    loss, stats = front.loss(None, K, pose, 17, 17, torch.zeros(17, 17), 1)
    assert loss is None and stats["sparse_front_covered"] == 0
    assert stats["sparse_front_lower_loss"] is None and stats["sparse_front_total_alpha"] is None
    assert stats["sparse_front_view_id"] == 1
