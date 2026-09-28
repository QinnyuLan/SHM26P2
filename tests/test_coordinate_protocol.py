"""Preparation and camera-only export compatibility without a CUDA model."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from bridge_rgs.coordinates import CORNER, LEGACY, protocol_metadata
from bridge_rgs.evaluate import distortion_render_grid, evaluate_scene, render_cameras
from bridge_rgs.prepare import _cloud_appearance, prepare_dataset


def test_v2_preparation_writes_new_manifest_and_npz_provenance(tmp_path):
    dataset = tmp_path / "Dataset"
    for directory in ("camera_parameters", "images"):
        (dataset / directory).mkdir(parents=True)
    (dataset / "camera_parameters/cameras.txt").write_text(
        "1 SIMPLE_RADIAL 80 61 55 40 30.5 0.03\n")
    points = np.array([[-.2, 0., 4.], [.2, .1, 4.5], [0., -.2, 3.5]])
    K = np.array([[55., 0, 40], [0, 55., 30.5], [0, 0, 1]])
    lines = []
    for i, center in enumerate((-.5, 0., .5), 1):
        xy, _ = cv2.projectPoints(points, np.zeros(3), np.array([-center, 0., 0.]),
                                   K, np.array([.03, 0, 0, 0, 0]))
        lines += [f"{i} 1 0 0 0 {-center} 0 0 1 {i:03d}.png",
                  " ".join(f"{p[0]} {p[1]} {j}" for j, p in enumerate(xy[:, 0]))]
        Image.new("RGB", (80, 61), (100, 120, 140)).save(dataset / "images" / f"{i:03d}.png")
    (dataset / "camera_parameters/images.txt").write_text("\n".join(lines) + "\n")
    manifest_path = prepare_dataset(dataset, tmp_path / "new_prepared", max_width=40,
                                     val_every=0, min_track_length=2, workers=1,
                                     pixel_protocol=CORNER)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["pixel_protocol"] == protocol_metadata(CORNER)
    with np.load(manifest["init_points_path"], allow_pickle=False) as arrays:
        assert arrays["pixel_protocol"].item() == CORNER
        assert arrays["points"].shape == (3, 3)
        legacy_arrays = {key: arrays[key] for key in arrays.files if key != "pixel_protocol"}
    from bridge_rgs.ray_support import SparseDepthSupport
    support = SparseDepthSupport.from_manifest(manifest_path)
    assert support.pixel_protocol == CORNER
    np.savez(tmp_path / "legacy_points.npz", **legacy_arrays)
    with pytest.raises(ValueError, match="protocol mismatch"):
        SparseDepthSupport.from_manifest(manifest_path, init_points_path=tmp_path / "legacy_points.npz")


def test_new_prepare_refuses_any_existing_nonempty_output_before_dataset_io(tmp_path):
    output = tmp_path / "old_prepared"
    output.mkdir()
    sentinel = output / "manifest.json"
    sentinel.write_text('{"schema_version":1}')
    with pytest.raises(FileExistsError, match="different pixel protocol"):
        prepare_dataset(tmp_path / "missing_dataset", output, pixel_protocol=CORNER)
    assert sentinel.read_text() == '{"schema_version":1}'
    sentinel.write_text(json.dumps({"schema_version": 2, "pixel_protocol": protocol_metadata(CORNER)}))
    before = sentinel.read_bytes()
    with pytest.raises(FileExistsError, match="different pixel protocol"):
        prepare_dataset(tmp_path / "missing_dataset", output, pixel_protocol=LEGACY)
    with pytest.raises(FileExistsError, match="new or empty"):
        prepare_dataset(tmp_path / "missing_dataset", output, pixel_protocol=CORNER)
    assert sentinel.read_bytes() == before


def test_prior_reads_corner_nearest_and_preserves_legacy_rint(tmp_path):
    rgb = np.tile((np.arange(5, dtype=np.uint8) * 40)[None, :, None], (3, 1, 3))
    mask = np.tile(np.arange(5, dtype=np.uint8), (3, 1))
    Image.fromarray(rgb).save(tmp_path / "rgb.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    Image.fromarray(np.full((3, 5), 255, np.uint8)).save(tmp_path / "valid.png")
    uv = [[.5, .5], [1.5, .5], [2., .5], [4.5, .5], [-.1, .5]]
    cloud = SimpleNamespace(points=np.zeros((len(uv), 3)), observations=[
        [SimpleNamespace(image_id=1, xy_normalized=np.array(p))] for p in uv])
    views = [{"image_id": 1, "split": "train", "K": np.eye(3), "width": 5, "height": 3,
              "image_path": str(tmp_path / "rgb.png"), "mask_path": str(tmp_path / "mask.png"),
              "valid_path": str(tmp_path / "valid.png")}]
    color, count = _cloud_appearance(cloud, views, pixel_protocol=CORNER)
    assert count[:-1].argmax(1).tolist() == [0, 1, 2, 4]
    assert count[-1].sum() == 0
    np.testing.assert_allclose(color[:-1, 0], np.array([0, 1, 2, 4]) * 40 / 255)
    _, legacy_count = _cloud_appearance(cloud, views)
    assert legacy_count.argmax(1).tolist() == [0, 2, 2, 4, 0]
    assert legacy_count[-1].sum() == 1


def test_load_view_routes_mask_and_valid_phase_without_shifting_rgb_or_k(tmp_path):
    from bridge_rgs.io import load_view
    y, x = np.mgrid[:21, :19]
    rgb = np.stack((x*8, y*8, (x+y)*4), -1).astype(np.uint8)
    mask = ((x+y) % 5).astype(np.uint8)
    valid = np.full(mask.shape, 255, np.uint8)
    valid[0] = 0
    valid[:, -1] = 0
    for name, value in (("rgb", rgb), ("mask", mask), ("valid", valid)):
        Image.fromarray(value).save(tmp_path / f"{name}.png")
    K = np.array([[13., 0, 9.2], [0, 14., 10.4], [0, 0, 1]])
    view = {"image_path": str(tmp_path / "rgb.png"), "mask_path": str(tmp_path / "mask.png"),
            "valid_path": str(tmp_path / "valid.png"), "K": K.tolist(), "w2c": np.eye(4).tolist()}
    old = load_view(view, device="cpu", scale=.5)
    new = load_view(dict(view, pixel_protocol=CORNER), device="cpu", scale=.5)
    torch.testing.assert_close(old["rgb"], new["rgb"], rtol=0, atol=0)
    torch.testing.assert_close(old["K"], new["K"], rtol=0, atol=0)
    np.testing.assert_allclose(new["K"].numpy()[:2], K[:2] * np.array([10/19, 10/21])[:, None])
    ix = ((2*np.arange(10)+1)*19)//20
    iy = ((2*np.arange(10)+1)*21)//20
    expected_valid = valid[iy[:, None], ix[None, :]] > 0
    expected_mask = mask[iy[:, None], ix[None, :]].copy()
    expected_mask[~expected_valid] = 255
    np.testing.assert_array_equal(new["valid"].numpy(), expected_valid)
    np.testing.assert_array_equal(new["mask"].numpy(), expected_mask)
    assert not torch.equal(old["valid"], new["valid"])


@pytest.mark.parametrize("k", [.12, -.12])
def test_v2_export_map_uses_output_corner_centers_and_integer_overscan(k):
    K = np.array([[80., 0, 37.2], [0, 79., 29.3], [0, 0, 1]], np.float32)
    render_K, w, h, source = distortion_render_grid(K, [k, 0, 0, 0, 0], 80, 60, CORNER)
    yy, xx = np.mgrid[:60, :80]
    corner = np.stack([xx+.5, yy+.5], -1).astype(np.float32)
    rays = cv2.undistortPoints(corner.reshape(-1, 1, 2), K, np.array([k, 0, 0, 0, 0]))
    expected = rays[:, 0].reshape(60, 80, 2)
    sampled_rx = (source[..., 0] + .5 - render_K[0, 2]) / render_K[0, 0]
    sampled_ry = (source[..., 1] + .5 - render_K[1, 2]) / render_K[1, 1]
    np.testing.assert_allclose(np.stack([sampled_rx, sampled_ry], -1), expected, atol=2e-7)
    assert (source >= 0).all()
    assert source[..., 0].max() <= w-1 and source[..., 1].max() <= h-1
    assert np.all((K[:2, 2] - render_K[:2, 2]) % 1 == 0)


def test_zero_distortion_export_remains_direct_in_both_protocols():
    K = np.array([[80., 0, 37.2], [0, 79., 29.3], [0, 0, 1]], np.float32)
    for protocol in (LEGACY, CORNER):
        render_K, width, height, source = distortion_render_grid(K, [0]*5, 80, 60, protocol)
        np.testing.assert_array_equal(render_K, K)
        assert source is None and (width, height) == (80, 60)


class CornerRayScene(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.splats = {"means": torch.zeros(1, 3)}

    def render(self, K, pose, width, height, **kwargs):
        y, x = torch.meshgrid(torch.arange(height)+.5, torch.arange(width)+.5, indexing="ij")
        rgb = torch.stack(((x-K[0, 2])/K[0, 0]*.3+.5,
                           (y-K[1, 2])/K[1, 1]*.3+.5, torch.full_like(x, .4)), -1)
        p = torch.nn.functional.one_hot((x >= K[0, 2]).long(), 5).float()
        return {"rgb": rgb, "probabilities": p}


def test_export_uses_checkpoint_protocol_even_with_new_camera_metadata(tmp_path, monkeypatch):
    import bridge_rgs.train
    K = [[30., 0, 20], [0, 30., 15], [0, 0, 1]]
    camera = {"name": "001.png", "K": K, "w2c": np.eye(4).tolist(), "width": 40,
              "height": 30, "distortion": [.2, 0, 0, 0, 0],
              "image_path": "/must_not_open.png", "mask_path": "/must_not_open_mask.png"}
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({"schema_version": 2, "pixel_protocol": protocol_metadata(CORNER),
                                "views": [camera]}))
    calls = []
    import bridge_rgs.evaluate as module
    original = distortion_render_grid
    def capture(*args):
        calls.append(args[-1])
        return original(*args)
    monkeypatch.setattr(module, "distortion_render_grid", capture)
    for state, expected in (({}, LEGACY), ({"pixel_protocol": protocol_metadata(CORNER)}, CORNER)):
        monkeypatch.setattr(bridge_rgs.train, "load_scene", lambda *a, state=state, **k: (CornerRayScene(), state))
        receipt = render_cameras("unused.pt", path, tmp_path / expected)
        assert calls[-1] == expected
        assert receipt["pixel_protocol"]["id"] == expected
    output = cv2.imread(str(tmp_path / CORNER / "rgb/001.png"))[..., ::-1] / 255
    yy, xx = np.mgrid[:30, :40]
    pixels = np.stack([xx+.5, yy+.5], -1).astype(np.float32)
    rays = cv2.undistortPoints(pixels.reshape(-1, 1, 2), np.array(K), np.array(camera["distortion"]))
    np.testing.assert_allclose(output[..., :2], rays[:, 0].reshape(30, 40, 2)*.3+.5, atol=2/255)


def test_evaluation_rejects_cross_protocol_before_opening_images(tmp_path):
    manifest = {"schema_version": 2, "pixel_protocol": protocol_metadata(CORNER), "views": []}
    with pytest.raises(ValueError, match="protocol mismatch"):
        evaluate_scene(CornerRayScene(), manifest, tmp_path / "never_created")
    assert not (tmp_path / "never_created").exists()


def test_legacy_evaluation_cannot_skip_a_known_checkpoint_manifest_sha(tmp_path):
    from bridge_rgs.io import load_manifest
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"views": []}))
    manifest = load_manifest(path)
    scene = CornerRayScene()
    scene.manifest_sha256 = "0"*64
    with pytest.raises(ValueError, match="manifest SHA"):
        evaluate_scene(scene, manifest, tmp_path / "never_created")
    assert not (tmp_path / "never_created").exists()
    # Absent SHA still supports real old checkpoints; reaching view validation
    # proves no invented source hash was required or silently stamped on them.
    del scene.manifest_sha256
    with pytest.raises(ValueError, match="No held-out views"):
        evaluate_scene(scene, manifest, tmp_path / "old_checkpoint")
