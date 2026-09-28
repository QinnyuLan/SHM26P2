"""Numerical contracts that protect pose correctness and held-out isolation."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from bridge_rgs.data import (
    ColmapCamera,
    ColmapImage,
    camera_based_split,
    rasterize_labelme,
    read_colmap_images,
    undistortion_maps,
)
from bridge_rgs.geometry import (
    bounded_bundle_adjustment,
    camera_uncertainty,
    collect_train_tracks,
    triangulate_tracks,
)
from bridge_rgs.prepare import prepare_dataset


def synthetic_scene():
    camera = ColmapCamera(1, "SIMPLE_RADIAL", 160, 120, np.array([110., 80., 60., .05]))
    points = np.array([[-.2, -.15, 4.], [.3, .2, 5.], [.1, -.2, 3.], [.5, .1, 6.]])
    images = {}
    for i, x in enumerate([-.8, -.3, .25, .7, 1.1], 1):
        w2c = np.eye(4)
        w2c[0, 3] = -x
        projected, _ = cv2.projectPoints(points, np.zeros(3), w2c[:3, 3], camera.K, camera.distortion)
        images[i] = ColmapImage(i, f"{i:03d}.png", 1, w2c, projected[:, 0], np.arange(len(points)))
    return {1: camera}, images, points


def test_colmap_pair_parser_retains_empty_observation_line(tmp_path):
    path = tmp_path / "images.txt"
    path.write_text("# poses\n1 1 0 0 0 1 2 3 1 empty.png\n\n"
                    "2 0.7071067811865476 0 0 0.7071067811865476 0 0 0 1 name with spaces.png\n10 20 3\n")
    images = read_colmap_images(path)
    assert images[1].xy.shape == (0, 2)
    np.testing.assert_allclose(images[1].center, [-1, -2, -3])
    np.testing.assert_allclose(images[2].w2c[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert images[2].name == "name with spaces.png"
    assert images[2].point3d_ids.tolist() == [3]


def test_train_only_triangulation_is_immune_to_validation_pixels():
    cameras, images, points = synthetic_scene()
    splits = {i: "val" if i == 3 else "train" for i in images}
    tracks = collect_train_tracks(images, cameras, splits)
    result = triangulate_tracks(tracks, images, cameras, max_points=100, min_parallax_degrees=.1)
    np.testing.assert_allclose(result.points, points[result.track_ids], atol=1e-7)
    assert all(o.image_id != 3 for obs in result.observations for o in obs)
    images[3].xy[:] = np.nan
    images[3].point3d_ids[:] = 987654
    poisoned = triangulate_tracks(collect_train_tracks(images, cameras, splits), images, cameras,
                                  max_points=100, min_parallax_degrees=.1)
    np.testing.assert_array_equal(result.points, poisoned.points)
    assert 987654 not in poisoned.track_ids
    assert np.all(np.linalg.eigvalsh(result.covariances) > 0)
    pose_cov, audit = camera_uncertainty(result, images, cameras, splits)
    np.testing.assert_allclose(np.diag(pose_cov[3]), np.asarray(audit["prior_std"])**2)
    assert audit["train_observation_counts"]["3"] == 0


def test_robust_triangulation_rejects_bad_training_correspondence():
    cameras, images, points = synthetic_scene()
    images[2].xy[0] += [50., -35.]
    splits = {i: "train" for i in images}
    result = triangulate_tracks(collect_train_tracks(images, cameras, splits), images, cameras,
                                 min_track_length=3, min_parallax_degrees=.1, seed=5)
    index = result.track_ids.tolist().index(0)
    np.testing.assert_allclose(result.points[index], points[0], atol=1e-6)
    assert 2 not in [o.image_id for o in result.observations[index]]


def test_split_uses_camera_poses_only_and_full_fit_is_explicit():
    _, images, _ = synthetic_scene()
    expected = camera_based_split(images, val_every=3, seed=42)
    assert set(expected.values()) == {"train", "val"}
    for view in images.values():
        view.xy[:] = np.nan
    assert expected == camera_based_split(dict(reversed(list(images.items()))), val_every=3, seed=42)
    assert set(camera_based_split(images, val_every=0).values()) == {"train"}
    with pytest.raises(ValueError):
        camera_based_split(images, val_every=1)


def test_pinhole_maps_preserve_geometry_and_mask_order(tmp_path):
    camera = ColmapCamera(1, "SIMPLE_RADIAL", 160, 120, np.array([110., 80., 60., .05]))
    mx, my, K, valid = undistortion_maps(camera, max_width=80)
    np.testing.assert_allclose(K, [[55., 0, 40.], [0, 55., 30.], [0, 0, 1.]])
    # Destination center maps to original principal point exactly.
    np.testing.assert_allclose([mx[30, 40], my[30, 40]], [80, 60], atol=1e-5)
    assert valid[30, 40] == 255 and valid[0, 0] == 0
    annotation = {"imageWidth": 16, "imageHeight": 12, "shapes": [
        {"label": "deck", "shape_type": "polygon", "points": [[1, 1], [12, 1], [12, 10], [1, 10]]},
        {"label": "stay_cable", "shape_type": "polygon", "points": [[4, 4], [8, 4], [8, 8], [4, 8]]}]}
    path = tmp_path / "label.json"
    path.write_text(json.dumps(annotation))
    mask = rasterize_labelme(path)
    assert mask[0, 0] == 0 and mask[2, 2] == 1 and mask[5, 5] == 2


def test_bounded_ba_fixes_gauge_and_excludes_heldout():
    cameras, images, _ = synthetic_scene()
    splits = {i: "val" if i == 3 else "train" for i in images}
    cloud = triangulate_tracks(collect_train_tracks(images, cameras, splits), images, cameras,
                                min_parallax_degrees=.1)
    overrides, audit = bounded_bundle_adjustment(cloud, images, cameras, splits,
                                                 max_points=4, max_nfev=3)
    assert 3 not in overrides
    assert 3 not in audit["fixed_anchor_image_ids"]
    assert len(audit["fixed_anchor_image_ids"]) == 2
    for image_id in audit["fixed_anchor_image_ids"]:
        assert image_id not in overrides
    assert audit["rms_native_px_after"] <= audit["rms_native_px_before"] + 1e-8


def test_preparation_exports_train_provenance_and_null_unlabeled(tmp_path):
    _cameras, images, _ = synthetic_scene()
    dataset, output = tmp_path / "Dataset", tmp_path / "prepared"
    for name in ["camera_parameters", "images", "unlabeled_Images", "json"]:
        (dataset / name).mkdir(parents=True)
    (dataset / "camera_parameters/cameras.txt").write_text("1 SIMPLE_RADIAL 160 120 110 80 60 0.05\n")
    lines = []
    for i, view in images.items():
        lines.append(f"{i} 1 0 0 0 {-view.center[0]} 0 0 1 {view.name}\n")
        lines.append(" ".join(f"{xy[0]} {xy[1]} {j}" for j, xy in enumerate(view.xy)) + "\n")
        folder = "unlabeled_Images" if i == 5 else "images"
        Image.new("RGB", (160, 120), (80+i, 120, 160)).save(dataset / folder / view.name)
        if i != 5:
            (dataset / "json" / f"{i:03d}.json").write_text(json.dumps({
                "imageWidth": 160, "imageHeight": 120, "shapes": [
                    {"label": "deck", "shape_type": "polygon", "points": [[0, 0], [159, 0], [159, 119], [0, 119]]}]}))
    (dataset / "camera_parameters/images.txt").write_text("".join(lines))
    manifest_path = prepare_dataset(dataset, output, max_width=80, val_every=3,
                                    max_points=100, min_track_length=2, workers=1)
    manifest = json.loads(manifest_path.read_text())
    cloud = np.load(output / "init_points.npz")
    by_id = {v["image_id"]: v for v in manifest["views"]}
    assert by_id[5]["mask_path"] is None
    assert all(by_id[int(i)]["split"] == "train" for i in cloud["observation_image_ids"])
    assert cloud["semantic_counts"].shape == (4, 5)
    assert cloud["colors"].min() >= 0 and cloud["colors"].max() <= 1
    for view in manifest["views"]:
        np.testing.assert_array_equal(view["w2c"], view["w2c_original"])
        assert Path(view["image_path"]).is_absolute()
        if view["mask_path"]:
            mask = np.asarray(Image.open(view["mask_path"]))
            valid = np.asarray(Image.open(view["valid_path"])) > 0
            assert np.all(mask[~valid] == 255)
