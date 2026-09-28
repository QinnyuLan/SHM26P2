"""Synthetic-only contracts: no dataset, labels, camera files or Torch/GPU."""
import numpy as np
import pytest

from bridge_rgs.structure_axes import (
    best_constant_direction,
    bootstrap_axis,
    canonical_axis,
    feature_centers,
    line_samples,
    local_line_evidence,
    orientation_blocks,
    principal_axis,
    project_direction,
    reflect_intrinsics,
    spatial_blocks,
    unoriented_degrees,
)


def test_double_tower_local_dyads_recover_vertical_when_global_pca_is_horizontal():
    rng = np.random.default_rng(42)
    z = np.linspace(-3, 3, 128)
    points = np.concatenate([
        np.column_stack([np.full_like(z, x), np.zeros_like(z), z])
        for x in (-20., 20.)
    ])
    points[:, :2] += rng.normal(0, .001, (len(points), 2))
    _, vectors = np.linalg.eigh(np.cov(points.T))
    assert unoriented_degrees(vectors[:, -1], [1, 0, 0]) < .01
    evidence = local_line_evidence(points)  # Fixed k=16, includes each point itself.
    assert evidence["neighbor_indices"].shape == (256, 16)
    assert evidence["usable"].all()
    assert all(i in row for i, row in enumerate(evidence["neighbor_indices"]))
    blocks, _ = spatial_blocks(points)
    tensors, _ = orientation_blocks(evidence["directions"], evidence["linearity"], blocks)
    result = principal_axis(tensors)
    assert unoriented_degrees(result["axis"], [0, 0, 1]) < .1
    assert result["relative_eigengap"] > .99


def test_coincident_points_are_unusable_not_a_spurious_axis():
    result = local_line_evidence(np.ones((16, 3)))
    assert not result["usable"].any()
    np.testing.assert_array_equal(result["linearity"], 0)
    np.testing.assert_array_equal(result["eigenvalues"], 0)


def test_blocks_equalize_density_and_keep_local_linearity_weights():
    directions = np.array([[1., 0, 0], [0., 1, 0], [0., 1, 0]])
    weights = np.array([.8, .2, 1.])
    blocks = np.array([10, 10, 20])
    tensors, keys = orientation_blocks(directions, weights, blocks)
    np.testing.assert_array_equal(keys, [10, 20])
    np.testing.assert_allclose(tensors[0], np.diag([.8, .2, 0]))
    np.testing.assert_allclose(tensors[1], np.diag([0, 1, 0]))
    np.testing.assert_allclose(np.trace(tensors, axis1=1, axis2=2), 1)
    many = np.concatenate([np.tile([0, 1], 100), [2]])
    duplicated, _ = orientation_blocks(directions[many], weights[many], blocks[many])
    np.testing.assert_allclose(duplicated, tensors, atol=1e-14)
    np.testing.assert_allclose(principal_axis(tensors)["axis"], [0, 1, 0])
    negated, _ = orientation_blocks(-directions, weights, blocks)
    np.testing.assert_array_equal(negated, tensors)


def test_spatial_blocks_keep_tails_and_handle_zero_spans():
    points = np.column_stack([np.r_[-1e4, np.arange(98), 1e4], np.ones(100), np.ones(100)])
    inverse, meta = spatial_blocks(points)
    cells = np.asarray(meta["cells"])
    assert meta["bins"] == 4 and meta["quantiles"] == [.02, .98]
    np.testing.assert_allclose(meta["lower"], np.quantile(points, .02, axis=0))
    np.testing.assert_allclose(meta["upper"], np.quantile(points, .98, axis=0))
    np.testing.assert_array_equal(cells[inverse[0]], [0, 0, 0])
    np.testing.assert_array_equal(cells[inverse[-1]], [3, 0, 0])
    np.testing.assert_array_equal(cells[:, 1:], 0)


def test_bootstrap_uses_fixed_256_block_draws_and_no_global_rng():
    theta = np.deg2rad([-20., -10., 0., 10., 20.])
    axes = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(5)])
    tensors = axes[:, :, None] * axes[:, None, :]
    before = np.random.get_state()
    result = bootstrap_axis(tensors)
    after = np.random.get_state()
    assert result["repeats"] == 256 and result["seed"] == 20260927
    assert result["axes"].shape == (256, 3)
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    index = np.random.default_rng(20260927).integers(0, 5, size=(256, 5))
    expected = np.stack([principal_axis(tensors[i])["axis"] for i in index])
    np.testing.assert_array_equal(result["axes"], expected)
    np.testing.assert_array_equal(bootstrap_axis(tensors)["axes"], expected)
    assert result["angle_95_percentile_degrees"] == np.quantile(result["angles_degrees"], .95)


def test_axis_sign_ties_and_extreme_finite_norms():
    np.testing.assert_allclose(canonical_axis([-1, 1, 0]), [1/np.sqrt(2), -1/np.sqrt(2), 0])
    for scale in (1e-300, 1e300):
        np.testing.assert_allclose(canonical_axis(np.array([-3., 4., 0])*scale), [-.6, .8, 0])
    assert unoriented_degrees([1, 0, 0], [-1, 0, 0]) == 0
    assert unoriented_degrees([1, 0, 0], [0, 1, 0]) == 90
    tied = principal_axis(np.eye(3)[None])
    assert tied["relative_eigengap"] == 0  # Arbitrary eigendirection is not confidence.


def camera():
    return np.array([[950., 0, 662.3], [0, 960., 493.7], [0, 0, 1.]])


def test_vanishing_point_at_infinity_and_axis_sign():
    centers = feature_centers(1320, 989)
    forward, valid, sine = project_direction(camera(), np.eye(3), [1., 0, 0], centers)
    backward, other, _ = project_direction(camera(), np.eye(3), [-1., 0, 0], centers)
    assert valid.all() and (sine > 0).all()
    np.testing.assert_array_equal(forward, backward)
    np.testing.assert_array_equal(valid, other)
    np.testing.assert_array_equal(forward[..., 0], 1)
    np.testing.assert_array_equal(forward[..., 1], 0)


def test_ray_parallel_axis_is_invalid_and_cannot_count_repeated_center_samples():
    K = camera()
    p = K[:2, 2][None]
    unit, valid, sine = project_direction(K, np.eye(3), [0, 0, 1], p)
    assert not valid[0] and sine[0] < 1e-15
    np.testing.assert_array_equal(unit, 0)
    points, support = line_samples(p, unit, [-16., 0, 16.], 1320, 989)
    np.testing.assert_array_equal(points, np.broadcast_to(p[:, None], points.shape))
    assert not support.any()
    # A nearby, nonzero direction still fails the prescribed foreshortening gate.
    _, valid, _ = project_direction(K, np.eye(3), [0, 0, 1], p + [1e-3, 0])
    assert not valid[0]


def test_projective_direction_matches_two_actual_3d_points_under_camera_rotation():
    rng = np.random.default_rng(812)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R[:, 0] *= np.linalg.det(R)
    translation = np.array([.2, -.3, 1.])
    axis = canonical_axis([.4, -.5, .7])
    Xcamera = np.array([[.3, .7, 5.], [-1., .1, 8.], [.5, -.8, 12.]])
    Xworld = (Xcamera - translation) @ R
    K = camera()

    def project(points):
        projected = (points @ R.T + translation) @ K.T
        return projected[:, :2] / projected[:, 2:]

    start, end = project(Xworld), project(Xworld + .03*axis)
    unit, valid, _ = project_direction(K, R, axis, start)
    assert valid.all()
    assert np.max(unoriented_degrees(unit, end-start)) < 2e-6
    # Depth/translation change the point but not the field at a fixed pixel.
    scaled_camera = Xcamera*3
    distant = (scaled_camera-translation) @ R
    distant_unit, _, _ = project_direction(K, R, axis, project(distant))
    np.testing.assert_allclose(unit, distant_unit, atol=1e-15)


def test_stride_centers_and_feature_array_conversion_do_not_use_resize_ratio():
    centers = feature_centers(1320, 989)
    assert centers.shape == (62, 83, 2)
    np.testing.assert_array_equal(centers[0, 0], [.5, .5])
    np.testing.assert_array_equal(centers[-1, -1], [1312.5, 976.5])
    np.testing.assert_array_equal((centers[23, 17]-.5)/16, [17., 23.])
    wrong_resize_center = (17.+.5)*1320/83
    assert abs(centers[23, 17, 0]-wrong_resize_center) > 1


def test_mirror_recomputes_the_field_at_actual_full_image_reflected_centers():
    W, H = 1320, 989
    K, axis = camera(), np.array([.4, .9, .1])
    centers = feature_centers(W, H)
    original_locations = centers.copy()
    original_locations[..., 0] = W-centers[..., 0]
    original, valid, _ = project_direction(K, np.eye(3), axis, original_locations)
    mirrored, mirrored_valid, _ = project_direction(reflect_intrinsics(K, W), np.eye(3), axis, centers)
    np.testing.assert_allclose(mirrored, original*[-1, 1], atol=3e-16)
    np.testing.assert_array_equal(valid, mirrored_valid)
    np.testing.assert_array_equal(reflect_intrinsics(reflect_intrinsics(K, W), W), K)
    # Reversing the coarse grid alone is seven native pixels off for this width.
    naive_locations = centers[:, ::-1]
    np.testing.assert_array_equal(original_locations[..., 0]-naive_locations[..., 0], 7.)
    naive, _, _ = project_direction(K, np.eye(3), axis, naive_locations)
    assert np.max(np.abs(mirrored-naive*[-1, 1])) > 1e-4


def test_line_samples_are_supported_by_feature_lattice_not_native_image_extent():
    centers = np.array([[.5, .5], [1312.5, 976.5]])
    points, valid = line_samples(centers, np.array([[1., 0], [1., 0]]), [-16, 0, 7], 1320, 989)
    np.testing.assert_array_equal(valid, [[False, True, True], [True, True, False]])
    assert points[-1, -1, 0] == 1319.5  # Native center but outside all-four-tap lattice support.


def test_best_constant_is_dyad_optimum_and_reflects_covariantly():
    theta = np.deg2rad([5, 15, 45, 50, 60])
    directions = np.column_stack([np.cos(theta), np.sin(theta)])
    axis, gap = best_constant_direction(directions)
    theta_dense = np.linspace(0, np.pi, 10001)
    alternatives = np.column_stack([np.cos(theta_dense), np.sin(theta_dense)])
    loss = np.mean(1-(directions @ axis)**2)
    assert loss <= np.min(np.mean(1-(directions @ alternatives.T)**2, axis=0)) + 1e-14
    reflected, reflected_gap = best_constant_direction(directions*[-1, 1])
    assert unoriented_degrees(reflected, axis*[-1, 1]) < 2e-6
    assert gap == pytest.approx(reflected_gap)
    with_invalid = np.concatenate([directions, np.zeros((1, 2))])
    selected, _ = best_constant_direction(with_invalid, np.array([True]*5+[False]))
    np.testing.assert_array_equal(axis, selected)


@pytest.mark.parametrize("run", [
    lambda: canonical_axis([0, 0, 0]),
    lambda: canonical_axis([np.nan, 1]),
    lambda: unoriented_degrees([1, 0], [0, 0]),
    lambda: local_line_evidence(np.ones((16, 3)), neighbors=2.5),
    lambda: local_line_evidence(np.ones((16, 3)), minimum_linearity=np.nan),
    lambda: spatial_blocks(np.ones((16, 3)), bins=True),
    lambda: orientation_blocks([[0, 0, 0]], [1.], [0]),
    lambda: orientation_blocks([[1, 0, 0]], [1.], [[0]]),
    lambda: orientation_blocks([[1, 0, 0]], [1.], [np.nan]),
    lambda: principal_axis(np.zeros((0, 3, 3))),
    lambda: principal_axis(np.diag([1., 0, -1])[None]),
    lambda: bootstrap_axis(np.eye(3)[None], repeats=0),
    lambda: bootstrap_axis(np.eye(3)[None], seed=-1),
    lambda: feature_centers(1320.5, 989),
    lambda: project_direction(np.zeros((3, 3)), np.eye(3), [1, 0, 0], [1, 2]),
    lambda: project_direction(camera(), np.diag([-1., 1, 1]), [1, 0, 0], [1, 2]),
    lambda: project_direction(camera(), np.eye(3), [1, 0], [1, 2]),
    lambda: project_direction(camera(), np.eye(3), [1, 0, 0], [1, 2], minimum_sine=2),
    lambda: best_constant_direction([[0., 0]]),
    lambda: best_constant_direction([[1., 0]], [1]),
    lambda: reflect_intrinsics(camera(), 0),
    lambda: line_samples([[1, 2]], [[1, 0]], [np.inf], 1320, 989),
    lambda: line_samples([[1, 2]], [[1, 0]], [], 1320, 989),
])
def test_invalid_inputs_fail_closed(run):
    with pytest.raises(ValueError):
        run()
