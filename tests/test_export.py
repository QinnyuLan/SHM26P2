"""Submission grid, camera-only inference and final-fit pipeline contracts."""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch import nn

from bridge_rgs.evaluate import distortion_render_grid, read_render_cameras, render_cameras
from bridge_rgs.export import export_ply


class RayScene(nn.Module):
    def __init__(self):
        super().__init__()
        self.splats = {"means": torch.zeros(4, 3)}
        self.last_pose = None

    def render(self, K, w2c, width, height, **kwargs):
        self.last_pose = w2c.clone()
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        rx = (x - K[0, 2]) / K[0, 0]
        ry = (y - K[1, 2]) / K[1, 1]
        rgb = torch.stack([rx * .3 + .5, ry * .3 + .5, torch.ones_like(rx)*.4], -1)
        classes = torch.where(rx >= 0, 1, 2)
        probabilities = torch.nn.functional.one_hot(classes, 5).float()
        return {"rgb": rgb, "probabilities": probabilities, "p3d": probabilities}


def test_prepared_manifest_official_grid_restores_original_camera(tmp_path):
    K = [[120, 0, 80], [0, 120, 60], [0, 0, 1]]
    view = {"name": "one.png", "camera_id": 1, "width": 80, "height": 60,
            "K": [[60, 0, 40], [0, 60, 30], [0, 0, 1]], "w2c": np.eye(4).tolist(),
            "image_path": "/intentionally/missing.png", "mask_path": "/also/missing.png"}
    manifest = {"views": [view], "source_cameras": {"1": {
        "K": K, "width": 160, "height": 120, "opencv_distortion": [.1, 0, 0, 0, 0]}}}
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(manifest))
    official = read_render_cameras(path, "official")[0]
    pinhole = read_render_cameras(path, "pinhole")[0]
    assert (official["width"], official["height"]) == (160, 120)
    assert official["distortion"][0] == .1
    assert pinhole["K"] == view["K"]
    assert (pinhole["width"], pinhole["height"]) == (80, 60)


def test_negative_distortion_canvas_covers_all_output_rays():
    K = np.array([[100., 0, 80], [0, 100., 60], [0, 0, 1]], dtype=np.float32)
    render_K, width, height, source = distortion_render_grid(K, [-.12, 0, 0, 0, 0], 160, 120)
    assert width > 160 and height > 120
    assert source.min() >= 0
    assert source[..., 0].max() <= width-1 and source[..., 1].max() <= height-1
    center = source[60, 80]
    np.testing.assert_allclose(center, render_K[:2, 2], atol=1e-5)


def test_camera_only_export_uses_rays_and_emits_official_ids(tmp_path, monkeypatch):
    import bridge_rgs.train
    scene = RayScene()
    monkeypatch.setattr(bridge_rgs.train, "load_scene", lambda *args, **kwargs: (scene, {}))
    K = np.array([[80., 0, 40], [0, 80., 30], [0, 0, 1]], dtype=np.float32)
    original_pose = np.eye(4)
    original_pose[0, 3] = .2
    camera = {"name": "007.png", "image_id": 17, "camera_id": 1,
              "K": K.tolist(), "w2c": np.eye(4).tolist(), "w2c_original": original_pose.tolist(),
              "distortion": [-.12, 0, 0, 0, 0], "width": 80, "height": 60,
              "image_path": "/must/not/read/rgb.png", "mask_path": "/must/not/read/mask.png"}
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps([camera]))
    output = tmp_path / "submission"
    receipt = render_cameras("unused-checkpoint.pt", path, output, grid="official")
    rgb = cv2.imread(str(output / "rgb/007.png"))[..., ::-1] / 255.
    mask = cv2.imread(str(output / "mask/007.png"), 0)
    xx, yy = np.meshgrid(np.arange(80), np.arange(60))
    pixels = np.stack([xx, yy], -1).astype(np.float32)
    rays = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, np.array(camera["distortion"]))[:, 0].reshape(60, 80, 2)
    expected = rays * .3 + .5
    np.testing.assert_allclose(rgb[..., :2], expected, atol=2/255)
    assert set(np.unique(mask)) == {1, 2}
    np.testing.assert_allclose(scene.last_pose, original_pose)
    assert receipt["records"][0]["image_id"] == 17
    assert receipt["records"][0]["pinhole_canvas"][0] > 80


def test_semantic_ply_contains_normalized_rotations_and_probabilities(tmp_path, monkeypatch):
    import bridge_rgs.train
    from bridge_rgs.model import GaussianScene
    points = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 2]], dtype=np.float32)
    scene = GaussianScene(points, np.full((4, 3), .5, dtype=np.float32), feature_dim=8, sh_degree=1)
    with torch.no_grad():
        scene.splats["quats"] *= 3
    monkeypatch.setattr(bridge_rgs.train, "load_scene", lambda *args, **kwargs: (scene, {}))
    path = export_ply("unused.pt", tmp_path / "model.ply")
    properties = []
    with path.open("rb") as stream:
        while True:
            line = stream.readline().decode().strip()
            if line.startswith("property"):
                _, typ, name = line.split()
                properties.append((name, "<f4" if typ == "float" else "u1"))
            if line == "end_header":
                break
        vertices = np.fromfile(stream, dtype=np.dtype(properties))
    assert len(vertices) == 4
    np.testing.assert_array_equal(vertices["x"], points[:, 0])
    quaternions = np.stack([vertices[f"rot_{i}"] for i in range(4)], -1)
    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=-1), 1.)
    probabilities = np.stack([vertices[f"semantic_p_{i}"] for i in range(5)], -1)
    np.testing.assert_allclose(probabilities.sum(-1), 1., atol=1e-6)
    np.testing.assert_array_equal(vertices["semantic_id"], probabilities.argmax(-1))


def _pipeline_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_data_pipeline_isolates_artifacts_and_never_scores_validation(tmp_path, monkeypatch):
    pipeline = _pipeline_module()
    config = {"manifest": "artifacts/custom/manifest.json", "output": "runs/custom",
              "pseudo_dir": None, "init_points": "artifacts/custom/init_points.npz", "eval_every": 1000}
    derived = pipeline.pipeline_configuration(config, True)
    assert config["manifest"] == "artifacts/custom/manifest.json"
    assert derived["manifest"] == "artifacts/custom_all/manifest.json"
    assert derived["init_points"] == "artifacts/custom_all/init_points.npz"
    assert derived["eval_every"] == 0
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    prepared = tmp_path / derived["manifest"]
    prepared.parent.mkdir(parents=True)
    prepared.write_text(json.dumps({"split": {"val_every": 0}, "views": [
        {"split": "train", "mask_path": "label.png"}, {"split": "train", "mask_path": None}]}))
    commands = []
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "run", lambda *args: commands.append(args))
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--all-data", "--resume", "--gaussian-config", "config.yaml"])
    pipeline.main()
    assert [c[2] for c in commands] == ["train", "export-ply"]
    receipt = json.loads((tmp_path / "runs/custom_all/final_fit_protocol.json").read_text())
    assert receipt["rgb_training_views"] == 2
    assert receipt["semantic_training_views"] == 1
    assert receipt["reported_validation_score"] is None


def test_resume_pipeline_uses_configured_paths_and_checkpoint(tmp_path, monkeypatch):
    pipeline = _pipeline_module()
    config = {"manifest": "my_data/manifest.json", "output": "my_run", "pseudo_dir": None,
              "eval_every": 50, "steps": 100}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    prepared = tmp_path / config["manifest"]
    prepared.parent.mkdir()
    prepared.write_text(json.dumps({"split": {"val_every": 8}, "views": [
        {"split": "train", "mask_path": "train.png"},
        {"split": "val", "mask_path": "val.png"}]}))
    checkpoint = tmp_path / "my_run/last.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    commands = []
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "run", lambda *args: commands.append(args))
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--resume", "--gaussian-config", "config.yaml"])
    pipeline.main()
    assert [c[2] for c in commands] == ["train", "evaluate", "export-ply"]
    assert commands[0][-2:] == ("--resume", checkpoint)
    evaluation = commands[1]
    assert evaluation[evaluation.index("--manifest")+1] == prepared
    saved = yaml.safe_load((tmp_path / "my_run/pipeline_config.yaml").read_text())
    assert saved["manifest"] == config["manifest"]
    assert saved["output"] == config["output"]
