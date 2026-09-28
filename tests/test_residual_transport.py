"""Synthetic CPU-only transport contracts; never open scene/image/GT files."""
import json

import numpy as np
import pytest

from bridge_rgs.residual_transport import (
    apply_shrinkage,
    fit_shrinkage,
    project_world,
    shrink_statistics,
    transport_residual,
    unproject_depth,
)


def source(height=3, width=5):
    x = np.arange(width, dtype=np.float64)[None, :]
    residual = np.repeat(np.broadcast_to(x[..., None]/10, (height, width, 3)), 1, 2).copy()
    return {"K": np.eye(3), "w2c": np.eye(4), "depth": np.full((height, width), 2.),
            "alpha": np.ones((height, width)), "valid": np.ones((height, width), bool),
            "residual": residual}


def transport(sources, **kw):
    s = sources[0]
    kwargs = {"target_depth": s["depth"], "target_alpha": s["alpha"],
              "target_valid": np.ones(s["depth"].shape, bool),
              "target_K": s["K"], "target_w2c": s["w2c"], "sources": sources}
    kwargs.update(kw)
    return transport_residual(**kwargs)


def test_corner_centers_depth_is_z_and_roundtrip():
    K = np.array([[4., 0, 1.7], [0, 5., .9], [0, 0, 1.]])
    depth = np.full((3, 5), 2.)
    world = unproject_depth(depth, K, np.eye(4))
    np.testing.assert_allclose(world[0, 0], [2*(.5-1.7)/4, 2*(.5-.9)/5, 2])
    assert np.linalg.norm(world[0, 0]) > depth[0, 0]
    projection = project_world(world, K, np.eye(4))
    y, x = np.indices(depth.shape)
    np.testing.assert_allclose(projection["array_xy"], np.stack([x, y], -1), atol=1e-14)
    np.testing.assert_allclose(projection["depth"], depth)
    assert projection["finite_positive"].all()


def test_pose_roundtrip_and_second_camera_translation():
    theta = .3
    pose = np.eye(4)
    pose[:3, :3] = np.asarray([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                              [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)
    pose[:3, 3] = [1, -2, 3]
    K = np.array([[5., 0, 2.5], [0, 5., 1.5], [0, 0, 1.]])
    world = unproject_depth(np.full((3, 5), 2.), K, pose)
    projected = project_world(world, K, pose)
    y, x = np.indices((3, 5))
    np.testing.assert_allclose(projected["array_xy"], np.stack([x, y], -1), atol=2e-14)
    other = pose.copy()
    other[0, 3] += .4
    shifted = project_world(world, K, other)
    np.testing.assert_allclose(shifted["array_xy"][..., 0], x+1, atol=2e-14)


def test_identity_true_and_content_reflection_same_support():
    s = source()
    out = transport([s])
    np.testing.assert_array_equal(out["true_residual"], s["residual"])
    np.testing.assert_array_equal(out["wrong_residual"], s["residual"][:, ::-1])
    assert out["valid"].all()
    np.testing.assert_array_equal(out["source_count"], 1)
    assert s["w2c"].tolist() == np.eye(4).tolist()
    json.dumps(out["source_summaries"], allow_nan=False)


def test_wrong_does_not_reflect_depth_but_rgb_valid_is_shared():
    s = source()
    s["depth"][:, 0] = 3  # Only the true geometric first column is occluded.
    s["valid"][:, 1] = False
    out = transport([s], target_depth=np.full((3, 5), 2.))
    # Last column survives despite the mirrored location's mismatching depth.
    assert out["valid"][:, -1].all()
    assert not out["valid"][:, 0].any()
    assert not out["valid"][:, 1].any()
    assert not out["valid"][:, 3].any()  # Mirrored invalid RGB column.
    np.testing.assert_array_equal(out["true_residual"][~out["valid"]], 0)
    np.testing.assert_array_equal(out["wrong_residual"][~out["valid"]], 0)


def test_fixed_thresholds_alpha_depth_and_zero_fallback():
    s = source()
    alpha = np.ones((3, 5))
    alpha[0] = .949
    s["alpha"][1] = .949
    s["depth"][2] = 2.1
    out = transport([s], target_depth=np.full((3, 5), 2.), target_alpha=alpha)
    assert not out["valid"].any()
    np.testing.assert_array_equal(out["true_residual"], 0)
    np.testing.assert_array_equal(out["wrong_residual"], 0)
    s["depth"][:] = 2.02  # Difference/max = .00990099, inside fixed .01.
    s["alpha"][:] = .95
    out = transport([s], target_depth=np.full((3, 5), 2.), target_alpha=np.full((3, 5), .95))
    assert out["valid"].all()


def test_bilinear_valid_support_rejects_any_positive_invalid_tap():
    s = source()
    s["w2c"][0, 3] = 1.  # Depth2, focal1 -> half pixel offset.
    s["valid"][:, 1] = False
    out = transport([s], target_w2c=np.eye(4))
    assert not out["valid"][:, 0].any()  # .5 samples columns0/1.
    assert not out["valid"][:, 1].any()  # 1.5 samples columns1/2.
    assert not out["valid"][:, -1].any()  # Out of source image.
    # Source RGB valid is checked at both reflected and true coordinates.
    assert not out["valid"][:, 2].any()


def test_uniform_source_aggregation_and_common_counts():
    a, b = source(), source()
    a["residual"][:] = .2
    b["residual"][:] = .6
    b["alpha"][0] = 0
    out = transport([a, b])
    np.testing.assert_allclose(out["true_residual"][0], .2)
    np.testing.assert_allclose(out["true_residual"][1:], .4)
    np.testing.assert_array_equal(out["source_count"][0], 1)
    np.testing.assert_array_equal(out["source_count"][1:], 2)
    np.testing.assert_array_equal(out["true_residual"], out["wrong_residual"])


def test_invalid_projection_is_masked_before_sampling_and_no_coordinate_clamp():
    s = source()
    s["w2c"][2, 3] = -4
    out = transport([s], target_w2c=np.eye(4), target_depth=np.full((3, 5), 2.))
    assert not out["valid"].any()
    projection = project_world(np.array([[1e300, 0, 1e-300]]), np.eye(3), np.eye(4))
    assert not projection["finite_positive"].any()
    np.testing.assert_array_equal(projection["array_xy"], 0)


def test_shrink_equal_view_not_pixel_weighted_and_closed_form():
    small = shrink_statistics(np.zeros((1, 1, 3)), np.full((1, 1, 3), .2),
                              np.ones((1, 1, 3)), np.ones((1, 1), bool))
    large = shrink_statistics(np.zeros((4, 8, 3)), np.full((4, 8, 3), .8),
                              np.ones((4, 8, 3)), np.ones((4, 8), bool))
    fitted = fit_shrinkage([small, large])
    assert fitted["coefficient"] == pytest.approx(.5)
    assert fitted["coefficient"] != pytest.approx((.2+32*.8)/33)
    for step in [-.01, .01]:
        loss = np.mean([(.2-(.5+step))**2, (.8-(.5+step))**2])
        assert loss > np.mean([(.2-.5)**2, (.8-.5)**2])
    json.dumps(fitted, allow_nan=False)


def test_fit_uses_whole_valid_view_and_zero_energy_has_zero_coefficient():
    r = np.zeros((1, 4, 3))
    r[:, 0] = 1
    stats = shrink_statistics(np.zeros_like(r), r*.6, r, np.ones((1, 4), bool))
    assert stats["denominator"] == .25
    assert fit_shrinkage([stats])["coefficient"] == pytest.approx(.6)
    zero = shrink_statistics(r, r+1, r*0, np.ones((1, 4), bool))
    assert fit_shrinkage([zero])["coefficient"] == 0
    assert fit_shrinkage([zero])["zero_residual_energy"] is True


def test_shrink_boundary_and_no_implicit_rgb_clipping():
    for covariance, expected in [(-.1, 0), (2., 1)]:
        assert fit_shrinkage([{"numerator": covariance, "denominator": 1.,
                               "zero_mse": 4., "valid_pixels": 1}])["coefficient"] == expected
    result = apply_shrinkage(np.full((1, 1, 3), .9), np.ones((1, 1, 3)), .5)
    np.testing.assert_allclose(result, 1.4)


def test_bad_shapes_nonfinite_empty_support_and_bad_cameras_fail():
    s = source()
    with pytest.raises(ValueError, match="boolean"):
        transport([s], target_valid=np.ones((3, 5)))
    s["residual"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        transport([s])
    with pytest.raises(ValueError, match="valid RGB"):
        shrink_statistics(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)),
                          np.zeros((1, 1, 3)), np.zeros((1, 1), bool))
    pose = np.eye(4)
    pose[0, 0] = 2
    with pytest.raises(ValueError, match="orthonormal"):
        unproject_depth(np.ones((1, 1)), np.eye(3), pose)
    with pytest.raises(ValueError, match="Zero residual energy"):
        fit_shrinkage([{"numerator": 1., "denominator": 0., "zero_mse": 1., "valid_pixels": 1}])
    with pytest.raises(ValueError, match="overflow"):
        fit_shrinkage([{"numerator": 1e308, "denominator": 1e308,
                        "zero_mse": 1e308, "valid_pixels": 1}]*2)
