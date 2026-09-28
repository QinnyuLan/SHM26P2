"""Independent CPU oracles for the versioned corner/array interfaces."""
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from bridge_rgs.coordinates import (
    CORNER,
    LEGACY,
    annotate_manifest,
    corner_to_array_K,
    inside_bilinear_centers,
    nearest_center_indices,
    pixel_protocol,
    protocol_metadata,
    resize_discrete_numpy,
    resize_discrete_tensor,
)
from bridge_rgs.data import CLASS_NAMES, ColmapCamera, undistortion_maps
from bridge_rgs.evaluate import evaluation_fingerprint


def test_profiles_are_explicit_fail_closed_and_never_relabel_old_sources():
    assert pixel_protocol({}) == LEGACY
    assert pixel_protocol(protocol_metadata(CORNER)) == CORNER
    for bad in ("automatic", {"schema_version": 2}, {"schema_version": 2, "pixel_protocol": None},
                {"id": CORNER}, {"id": CORNER, "version": 1},
                {"id": CORNER, "version": 2, "prepare_warp": "legacy"}):
        with pytest.raises(ValueError):
            pixel_protocol(bad)
    old = {"views": [{"name": "one"}]}
    resolved = annotate_manifest(old)
    assert resolved["views"][0]["pixel_protocol"] == LEGACY
    assert "pixel_protocol" not in old["views"][0]
    with pytest.raises(ValueError, match="disagrees"):
        annotate_manifest({"pixel_protocol": protocol_metadata(CORNER),
                           "views": [{"pixel_protocol": LEGACY}]})


@pytest.mark.parametrize("k", [0., .13, -.12])
@pytest.mark.parametrize("max_width", [None, 40])
def test_prepare_corner_map_matches_independent_radial_equation(k, max_width):
    camera = ColmapCamera(1, "SIMPLE_RADIAL", 80, 61, np.array([53., 39.2, 30.7, k]))
    mx, my, K, valid = undistortion_maps(camera, max_width, CORNER)
    height, width = mx.shape
    y, x = np.mgrid[:height, :width]
    nx, ny = (x + .5 - K[0, 2]) / K[0, 0], (y + .5 - K[1, 2]) / K[1, 1]
    radial = 1 + k * (nx**2 + ny**2)
    expected_x = camera.K[0, 0] * nx * radial + camera.K[0, 2] - .5
    expected_y = camera.K[1, 1] * ny * radial + camera.K[1, 2] - .5
    np.testing.assert_allclose(mx, expected_x, atol=5e-6, rtol=0)
    np.testing.assert_allclose(my, expected_y, atol=5e-6, rtol=0)
    np.testing.assert_allclose(K[0], camera.K[0] * width / camera.width)
    np.testing.assert_allclose(K[1], camera.K[1] * height / camera.height)
    if k == 0 and max_width is None:
        assert (valid == 255).all()
        np.testing.assert_array_equal(mx, x)
        np.testing.assert_array_equal(my, y)


def test_legacy_prepare_maps_are_exact_old_opencv_call_and_no_global_k_shift():
    camera = ColmapCamera(1, "SIMPLE_RADIAL", 80, 61, np.array([53., 39.2, 30.7, .13]))
    mx, my, K, _ = undistortion_maps(camera, 40)
    ex, ey = cv2.initUndistortRectifyMap(camera.K, camera.distortion, None, K,
                                        (40, 30), cv2.CV_32FC1)
    np.testing.assert_array_equal(mx, ex)
    np.testing.assert_array_equal(my, ey)
    before = camera.K.copy()
    array_K = corner_to_array_K(before)
    np.testing.assert_array_equal(before, camera.K)
    np.testing.assert_allclose(array_K[:2, 2], before[:2, 2] - .5)


@pytest.mark.parametrize("source,target", [(4, 2), (5, 2), (989, 494), (989, 360), (7, 11)])
def test_center_resize_has_exact_numpy_torch_phase_and_preserves_discrete_values(source, target):
    values = np.arange(source, dtype=np.float32)[:, None]
    expected = ((2 * np.arange(target) + 1) * source) // (2 * target)
    np.testing.assert_array_equal(nearest_center_indices(source, target), expected)
    resized = resize_discrete_numpy(values, (1, target), CORNER)
    tensor = resize_discrete_tensor(torch.from_numpy(values), (target, 1), CORNER)
    np.testing.assert_array_equal(resized[:, 0], expected)
    np.testing.assert_array_equal(tensor.numpy(), resized)
    np.testing.assert_array_equal(resize_discrete_numpy(values, (1, target)),
                                  cv2.resize(values, (1, target), interpolation=cv2.INTER_NEAREST))
    np.testing.assert_array_equal(resize_discrete_numpy(values, (1, source), CORNER), values)


def test_v2_bounds_describe_full_bilinear_centers_and_border_layers():
    uv = torch.tensor([[.5, .5], [4.5, 3.5], [0., 0.], [4.6, 3.5], [1.5, 1.5]])
    assert inside_bilinear_centers(uv, 5, 4, CORNER).tolist() == [True, True, False, False, True]
    assert inside_bilinear_centers(uv, 5, 4, CORNER, 1).tolist() == [False, False, False, False, True]


@pytest.mark.parametrize("protocol", [LEGACY, CORNER])
def test_both_manifest_loaders_propagate_version_without_mutating_source(tmp_path, protocol):
    from bridge_rgs.data import load_manifest as data_load
    from bridge_rgs.io import load_manifest as io_load
    manifest = {"schema_version": 1, "class_names": CLASS_NAMES, "views": [{"name": "001.png"}]}
    if protocol == CORNER:
        manifest.update(schema_version=2, pixel_protocol=protocol_metadata(CORNER))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    before = path.read_bytes()
    for loader in (data_load, io_load):
        value = loader(path)
        assert value["views"][0]["pixel_protocol"] == protocol
        assert value["_manifest_sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before


def test_legacy_fingerprint_is_identical_and_new_protocol_changes_it():
    views = [{"name": "one", "width": 5, "height": 4, "K": np.eye(3).tolist()}]
    payload = {"scale": .5, "views": [{key: v.get(key) for key in (
        "name", "image_path", "mask_path", "valid_path", "K", "w2c_original",
        "w2c", "width", "height", "split")} for v in views]}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert evaluation_fingerprint(views, .5) == expected
    old = annotate_manifest({"views": views})["views"]
    assert evaluation_fingerprint(old, .5) == expected
    new = annotate_manifest({"pixel_protocol": protocol_metadata(CORNER), "views": views})["views"]
    assert evaluation_fingerprint(new, .5, manifest_sha256="a") != expected
    assert evaluation_fingerprint(new, .5, manifest_sha256="a") != evaluation_fingerprint(
        new, .5, manifest_sha256="b")
