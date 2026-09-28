"""Synthetic CPU camera contracts. No dataset/camera/image/VAL payloads or CUDA."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from bridge_rgs.ibgs_camera_contract import (
    array_intrinsics,
    array_to_grid,
    make_ibgs_camera,
    ndc_to_array,
    pixel_rays,
    project_world_ibgs,
    reference_to_source,
    require_official_shared_centered,
    resize_corner_intrinsics,
    strided_array_intrinsics,
    texture_coordinates,
)


def intrinsic(width=9, height=7):
    return np.array([[10., 0, width/2], [0, 11., height/2], [0, 0, 1.]])


def test_ndc_projection_ray_and_corner_array_are_one_contract():
    K = intrinsic()
    camera = make_ibgs_camera(K, np.eye(4), 9, 7)
    rays = pixel_rays(K, 9, 7)
    depth = np.arange(63).reshape(7, 9)/20+2
    xy = project_world_ibgs(rays*depth[..., None], camera)
    y, x = np.indices((7, 9))
    np.testing.assert_allclose(xy, np.stack([x, y], -1), atol=2e-15)
    np.testing.assert_allclose(rays[..., 0], (x-camera['Cx'])/camera['Fx'])
    np.testing.assert_allclose(rays[..., 1], (y-camera['Cy'])/camera['Fy'])
    np.testing.assert_array_equal(ndc_to_array(np.array([[0., 0.]]), 1320, 989), [[659.5, 494.]])
    np.testing.assert_array_equal(texture_coordinates(xy)[3, 4], [4.5, 3.5])


def test_unpatched_center_disagrees_with_footprint_and_identity_warp_hides_it():
    K = intrinsic()
    pixel = np.array([3., 2.])
    depth = 5.
    old_ray = np.r_[(pixel-K[:2, 2])/np.diag(K)[:2], 1.]
    # Same-camera incorrect inverse/forward cancel, so identity warp alone misses this.
    np.testing.assert_array_equal(old_ray[:2]*np.diag(K)[:2]+K[:2, 2], pixel)
    old_point = old_ray*depth
    camera = make_ibgs_camera(K, np.eye(4), 9, 7)
    np.testing.assert_allclose(project_world_ibgs(old_point[None], camera)[0], pixel-.5)
    # Different source camera z makes the two half-pixel errors stop cancelling.
    source_pose = np.eye(4); source_pose[2, 3] = 2.
    correct_point = np.r_[(pixel+.5-K[:2, 2])/np.diag(K)[:2], 1.]*depth
    true_source = project_world_ibgs(correct_point[None], make_ibgs_camera(K, source_pose, 9, 7))[0]
    old_source_point = old_point+source_pose[:3, 3]
    old_source = old_source_point[:2]/old_source_point[2]*np.diag(K)[:2]+K[:2, 2]
    np.testing.assert_allclose(old_source-true_source, .5*(1-depth/(depth+2)), atol=1e-14)


def test_rotation_argument_transpose_glm_matrix_and_reference_transform():
    theta = .37
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    pose = np.eye(4); pose[:3, :3] = R; pose[:3, 3] = [1, -2, .4]
    camera = make_ibgs_camera(intrinsic(), pose, 9, 7)
    np.testing.assert_array_equal(camera['R_argument'], R.T)
    np.testing.assert_array_equal(camera['world_view_transform'], pose.T)
    center_h = np.r_[camera['camera_center'], 1]
    np.testing.assert_allclose(center_h@camera['world_view_transform'], [0, 0, 0, 1], atol=1e-15)
    reference = np.eye(4); reference[:3, 3] = [-.2, .6, 1]
    ref_point = np.array([.1, .3, 4, 1])
    transform = reference_to_source(reference, pose)
    np.testing.assert_allclose(transform@ref_point, pose@np.linalg.inv(reference)@ref_point)
    world = np.linalg.inv(pose)@np.array([.1, -.3, 4, 1])
    expected = np.array([10*.1/4+4, 11*(-.3)/4+3])
    np.testing.assert_allclose(project_world_ibgs(world[None, :3], camera)[0], expected, atol=1e-14)


def test_noncenter_projection_algebra_but_minimal_official_port_rejects_it():
    K = intrinsic(); K[:2, 2] += [.23, -.17]
    camera = make_ibgs_camera(K, np.eye(4), 9, 7)
    rays = pixel_rays(K, 9, 7)
    y, x = np.indices((7, 9))
    np.testing.assert_allclose(project_world_ibgs(rays*3, camera), np.stack([x, y], -1), atol=2e-15)
    with pytest.raises(ValueError, match='centered'):
        require_official_shared_centered([camera])
    a = make_ibgs_camera(intrinsic(), np.eye(4), 9, 7)
    bK = intrinsic(); bK[0, 0] += .1
    with pytest.raises(ValueError, match='identical'):
        require_official_shared_centered([a, make_ibgs_camera(bK, np.eye(4), 9, 7)])
    assert require_official_shared_centered([a, a]) is True


def test_grid_sample_integer_array_contract_both_align_corners_choices():
    y, x = np.indices((7, 9))
    image = torch.from_numpy(np.stack([x, y], 0).astype(np.float64))[None]
    xy = np.stack([x, y], -1).astype(np.float64)
    for align in (True, False):
        grid = torch.from_numpy(array_to_grid(xy, 9, 7, align_corners=align))[None]
        sampled = F.grid_sample(image, grid, align_corners=align, mode='bilinear')
        torch.testing.assert_close(sampled, image, atol=2e-14, rtol=0)
    # W-1 normalization and default align_corners=False are not interchangeable.
    wrong = torch.from_numpy(array_to_grid(xy, 9, 7, align_corners=True))[None]
    sampled = F.grid_sample(image, wrong, align_corners=False)
    assert (sampled-image).abs().max().item() > 1
    assert not torch.cuda.is_initialized()


def test_actual_size_resize_and_integer_stride_have_different_principal_points():
    K = intrinsic(1320, 989)
    resized = resize_corner_intrinsics(K, (1320, 989), (660, 494))
    assert resized[1, 1] == pytest.approx(11*494/989)
    np.testing.assert_array_equal(resized[:2, 2], [330, 247])
    resized_array = array_intrinsics(resized)
    strided = strided_array_intrinsics(K, 2, 0)
    assert strided[0, 2]-resized_array[0, 2] == .25
    full_rays = pixel_rays(K, 1320, 989)
    selected = full_rays[0:988:2, 0:1320:2]
    y, x = np.indices((494, 660))
    np.testing.assert_allclose(selected[..., 0], (x-strided[0, 2])/strided[0, 0], atol=1e-14)
    np.testing.assert_allclose(selected[..., 1], (y-strided[1, 2])/strided[1, 1], atol=1e-14)


def test_official_python_projection_functions_cpu_match_contract():
    path = Path('/mnt/data/SHM2026/third_party/ibgs/utils/graphics_utils.py')
    spec = importlib.util.spec_from_file_location('ibgs_graphics_cpu', path)
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    K = intrinsic(); K[:2, 2] += [.2, -.3]
    pose = np.eye(4); pose[:3, 3] = [.2, -.4, .6]
    camera = make_ibgs_camera(K, pose, 9, 7)
    actual = official.getProjectionMatrixCenterShift(.01, 100., K[0, 2], K[1, 2], K[0, 0], K[1, 1], 9, 7)
    np.testing.assert_allclose(actual.numpy(), camera['projection_matrix'].T, rtol=1e-7, atol=1e-8)
    np.testing.assert_array_equal(official.getWorld2View2(camera['R_argument'], camera['T_argument']), pose.astype(np.float32))
    # Official 1e-7 homogeneous epsilon is a small separate bias, not half a pixel.
    p = np.array([[.2, -.3, 4.]])
    unshifted = make_ibgs_camera(intrinsic(), np.eye(4), 9, 7)
    exact = project_world_ibgs(p, unshifted)
    eps = project_world_ibgs(p, unshifted, homogeneous_epsilon=1e-7)
    assert 0 < np.max(abs(eps-exact)) < 1e-6
    assert not torch.cuda.is_initialized()


def test_invalid_camera_domains_are_explicit():
    K = intrinsic(); K[0, 1] = 1
    with pytest.raises(ValueError, match='zero-skew'):
        make_ibgs_camera(K, np.eye(4), 9, 7)
    with pytest.raises(ValueError, match='znear'):
        make_ibgs_camera(intrinsic(), np.eye(4), 9, 7, znear=100)
    pose = np.eye(4); pose[0, 0] = -1
    with pytest.raises(ValueError, match='proper'):
        make_ibgs_camera(intrinsic(), pose, 9, 7)
