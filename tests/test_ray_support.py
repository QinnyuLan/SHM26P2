"""CPU regression checks for sparse evidence isolation and depth gradients."""
import copy
import json

import numpy as np
import pytest
import torch

from bridge_rgs.ray_support import (
    RaySupportConfig,
    SparseDepthSupport,
    SparseDepthTargets,
    sparse_depth_loss,
)


def fixture(points=None):
    points = np.array([[0., 0., 2.]] if points is None else points, dtype=np.float64)
    n = len(points)
    arrays = {"points": points, "covariances": np.tile(np.eye(3) * 1e-5, (n, 1, 1)),
              "reprojection_error": np.full(n, .2), "num_observations": np.full(n, 3),
              "observation_image_ids": np.tile([1, 2, 3], n),
              "observation_offsets": np.arange(n + 1) * 3, "track_ids": np.arange(n) + 10}
    K = torch.tensor([[20., 0, 8], [0, 20, 8], [0, 0, 1]])
    pose = torch.eye(4)
    views = [{"image_id": i, "name": f"{i:03d}.png", "camera_id": 1,
              "split": "train", "width": 17, "height": 17,
              "K": K.tolist(), "w2c_original": pose.tolist()} for i in [1, 2, 3]]
    return arrays, views, K, pose


def projected_support(arrays, views, **kwargs):
    return SparseDepthSupport(arrays, views, {"uv_mode": "projected", **kwargs})


def test_train_membership_fail_closed_and_no_duplicate_view_counts():
    arrays, views, K, pose = fixture()
    support = projected_support(arrays, views)
    with pytest.raises(ValueError, match="not a TRAIN"):
        support.targets_for_view(4, K, pose, 17, 17, torch.ones(17, 17))
    bad = copy.deepcopy(arrays)
    bad["observation_image_ids"][-1] = 4
    with pytest.raises(ValueError, match="held-out"):
        projected_support(bad, views)
    bad["observation_image_ids"][-1] = 1
    with pytest.raises(ValueError, match="Repeated image ID"):
        projected_support(bad, views)
    views[0]["split"] = "val"
    with pytest.raises(ValueError, match="TRAIN views only"):
        projected_support(arrays, views)


def test_quality_positive_depth_covariance_and_valid_edge_gates():
    points = [[0, 0, 2], [.1, 0, 2], [0, .1, 2], [0, 0, -2], [-.7, 0, 2], [.2, 0, 2]]
    arrays, views, K, pose = fixture(points)
    arrays["reprojection_error"][1] = 1.01
    arrays["covariances"][2] = np.eye(3) * 4  # Uncertain point position, not shape.
    support = projected_support(arrays, views)
    targets = support.targets_for_view(1, K, pose, 17, 17, torch.ones(17, 17))
    assert targets.point_indices.tolist() == [0, 5]
    # The invalid cell is within the radius-2 support of point 5 only.
    valid = torch.ones(17, 17)
    valid[8, 12] = 0
    targets = support.targets_for_view(1, K, pose, 17, 17, valid)
    assert targets.point_indices.tolist() == [0]
    arrays["covariances"][0, 2, 2] = -1  # Invalid covariance cannot gain confidence.
    support = projected_support(arrays, views)
    targets = support.targets_for_view(1, K, pose, 17, 17, valid)
    assert not len(targets.depth)


def test_original_observed_uv_used_with_bilinear_sampling_and_local_gate():
    arrays, views, K, pose = fixture()
    rays = {i: {0: np.array([.025, 0])} for i in [1, 2, 3]}
    support = SparseDepthSupport(arrays, views, observed_rays=rays)
    depth = torch.full((17, 17), 2.)
    depth[8, 9] = 4
    targets = support.targets_for_view(1, K, pose, 17, 17, torch.ones(17, 17))
    assert targets.pixels.tolist() == [[8.5, 8.0]]
    loss, stats = support.loss(depth, torch.ones_like(depth), torch.ones_like(depth), K, pose, 1)
    assert loss > 0
    assert stats["sparse_depth_median_relative_error"] == pytest.approx(.5)
    rays[1][0] = np.array([.2, 0])  # 4 px local residual exceeds the fixed gate.
    targets = support.targets_for_view(1, K, pose, 17, 17, torch.ones(17, 17))
    assert not len(targets.depth)


@pytest.mark.parametrize("kind", ["log_huber", "relative_huber"])
def test_depth_loss_is_scene_scale_invariant(kind):
    arrays, views, K, pose = fixture([[0, 0, 2], [.2, 0, 4]])
    valid = torch.ones(17, 17)
    support = projected_support(arrays, views, loss_kind=kind)
    depth = torch.full((17, 17), 3.)
    before, stats1 = support.loss(depth, valid, valid, K, pose, 1)
    target1 = support.targets_for_view(1, K, pose, 17, 17, valid)
    scaled = copy.deepcopy(arrays)
    scaled["points"] *= 31
    scaled["covariances"] *= 31**2
    support2 = projected_support(scaled, views, loss_kind=kind)
    after, stats2 = support2.loss(depth * 31, valid, valid, K, pose, 1)
    target2 = support2.targets_for_view(1, K, pose, 17, 17, valid)
    torch.testing.assert_close(before, after)
    torch.testing.assert_close(target1.confidence, target2.confidence)
    torch.testing.assert_close(target1.pixels, target2.pixels)
    assert stats1["sparse_depth_median_relative_error"] == pytest.approx(
        stats2["sparse_depth_median_relative_error"])


def test_targets_camera_confidence_alpha_detached_but_rendered_depth_has_gradient():
    arrays, views, K, pose = fixture()
    K.requires_grad_()
    pose.requires_grad_()
    support = projected_support(arrays, views)
    valid = torch.ones(17, 17, requires_grad=True)
    targets = support.targets_for_view(1, K, pose, 17, 17, valid)
    assert all(not x.requires_grad for x in (targets.pixels, targets.depth, targets.confidence))
    # An expected-depth toy renderer allows front opacity to receive depth gradients.
    front_opacity = torch.tensor(.5, requires_grad=True)
    depth = torch.ones(17, 17) * (front_opacity * .5 + (1 - front_opacity) * 2.)
    alpha = torch.ones(17, 17, requires_grad=True)
    loss, _ = sparse_depth_loss(depth, alpha, targets)
    loss.backward()
    assert front_opacity.grad > 0  # Gradient descent reduces this too-near contribution.
    assert alpha.grad is None  # No direct reward for suppressing the loss weight.
    assert K.grad is None and pose.grad is None and valid.grad is None
    # Even externally supplied target tensors cannot receive a gradient.
    targets.depth.requires_grad_()
    targets.confidence.requires_grad_()
    targets.pixels.requires_grad_()
    depth = torch.ones(17, 17, requires_grad=True)
    loss, _ = sparse_depth_loss(depth, alpha, targets)
    loss.backward()
    assert depth.grad.abs().sum() > 0
    assert all(x.grad is None for x in (targets.depth, targets.confidence, targets.pixels))


def test_no_support_returns_none_and_nonfinite_depth_is_excluded():
    arrays, views, K, pose = fixture()
    support = projected_support(arrays, views)
    valid = torch.ones(17, 17)
    depth = torch.full((17, 17), float("nan"), requires_grad=True)
    loss, stats = support.loss(depth, valid, valid, K, pose, 1)
    assert loss is None and stats["sparse_depth_valid"] == 0
    loss, stats = support.loss(torch.ones_like(valid), valid * .1, valid, K, pose, 1)
    assert loss is None and stats["sparse_depth_valid"] == 0
    loss, stats = support.loss(torch.ones_like(valid), valid, valid * 0, K, pose, 1)
    assert loss is None and stats["sparse_depth_targets"] == 0


def test_source_parser_skips_val_values_and_matches_triangulation_last_observation(tmp_path):
    arrays, views, K, pose = fixture()
    # Nonexistent original RGB/mask paths must never be opened.
    for view in views:
        view.update(image_path="/not/an/image", mask_path="/not/a/label")
    val_view = dict(views[0], image_id=4, name="004.png", split="val")
    np.savez(tmp_path / "init_points.npz", **arrays)
    camera_dir = tmp_path / "camera_parameters"
    camera_dir.mkdir()
    lines = []
    for view in views:
        lines.extend([f"{view['image_id']} 1 0 0 0 0 0 0 1 {view['name']}",
                      "8 8 10 8.5 8 10"])
    lines.extend(["4 1 0 0 0 0 0 0 1 004.png", "VAL VALUES MUST NOT BE PARSED"])
    (camera_dir / "images.txt").write_text("\n".join(lines) + "\n")
    manifest = {"views": views + [val_view], "dataset_root": str(tmp_path),
                "init_points_path": "init_points.npz", "source_cameras": {
                    "1": {"K": K.tolist(), "opencv_distortion": [0, 0, 0, 0, 0]}}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    support = SparseDepthSupport.from_manifest(manifest_path)
    assert support.provenance["observation_recovery"]["duplicate_observations"] == 3
    targets = support.targets_for_view(1, K, pose, 17, 17, torch.ones(17, 17))
    assert targets.pixels.tolist() == [[8.5, 8.0]]
    with pytest.raises(ValueError, match="not a TRAIN"):
        support.targets_for_view(4, K, pose, 17, 17, torch.ones(17, 17))


def test_expected_depth_can_hide_front_back_compensation():
    # This limitation is intentional: it prevents calling this a full free-space loss.
    targets = SparseDepthTargets(torch.tensor([[8., 8.]]), torch.tensor([2.]),
                                 torch.ones(1), torch.tensor([0]), {})
    expected_depth = torch.full((17, 17), .5 * 1. + .5 * 3.)
    loss, stats = sparse_depth_loss(expected_depth, torch.ones_like(expected_depth), targets)
    assert loss == 0
    assert stats["sparse_depth_median_relative_error"] == 0


def test_strict_config_schema_and_explicit_projected_fallback(tmp_path):
    with pytest.raises(TypeError):
        RaySupportConfig(unrecognized_threshold=1)
    with pytest.raises(ValueError):
        RaySupportConfig(border_px=1.5)
    arrays, views, _, _ = fixture()
    np.savez(tmp_path / "init.npz", **arrays)
    manifest = {"views": views, "dataset_root": str(tmp_path),
                "init_points_path": str(tmp_path / "init.npz")}
    with pytest.raises(FileNotFoundError, match="explicitly set"):
        SparseDepthSupport.from_manifest(manifest)
    support = SparseDepthSupport.from_manifest(manifest, config={"uv_mode": "projected"})
    assert "Approximation" in support.provenance["observation_source"]
