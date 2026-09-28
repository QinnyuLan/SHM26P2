import json

import cv2
import numpy as np
import pytest
import torch

import bridge_rgs.render_ensemble as module
from bridge_rgs.coordinates import CORNER, protocol_metadata
from bridge_rgs.evaluate import render_cameras


def test_fixed_blend_uses_quantized_rendered_rgb_and_exact_prediction_protocol(monkeypatch):
    rgb = np.array([[[.123, .5, 1.2], [-.1, .002, .75]]], np.float32)
    student = np.full((1, 2, 5), .2, np.float32)
    teacher = np.zeros((5, 1, 2), np.float32)
    teacher[3] = 1
    calls = []

    def predict(model, image, **kwargs):
        calls.append((image.copy(), kwargs))
        return teacher, None

    monkeypatch.setattr(module, "predict_image", predict)
    predictor = object.__new__(module.FixedRenderedTeacher)
    predictor.model = None
    result = predictor.blend(rgb, student)
    np.testing.assert_array_equal(calls[0][0], (rgb.clip(0, 1)*255).round().astype(np.uint8))
    assert calls[0][1] == {"tile_size": 768, "stride": 512, "flip": True,
                           "context_weight": .25, "context_short_side": 768}
    np.testing.assert_array_equal(result, .5*student+.5*teacher.transpose(1, 2, 0))
    np.testing.assert_array_equal(result.argmax(-1), [[3, 3]])


def test_teacher_grid_mismatch_fails_before_loading_backbone(tmp_path):
    path = tmp_path / "teacher.pt"
    torch.save({"provenance": {}}, path)
    with pytest.raises(ValueError, match="protocol mismatch"):
        module.FixedRenderedTeacher(path, "not-opened.pt", {"pixel_protocol": protocol_metadata(CORNER)})


def test_sharded_teacher_requires_bound_index_before_loading_backbone(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"dummy")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text("{}")
    source = {"class_names": module.CLASS_NAMES, "model_dir": str(tmp_path),
              "manifest_sha256": "m", "model_config_sha256": module.file_sha256(tmp_path / "config.json"),
              "model_weights_sha256": {"model-00001-of-00001.safetensors": module.file_sha256(tmp_path / "model-00001-of-00001.safetensors")}}
    path = tmp_path / "teacher.pt"
    torch.save({"provenance": source}, path)
    with pytest.raises(ValueError, match="index"):
        module.FixedRenderedTeacher(path, "not-opened.pt", {"manifest_sha256": "m"})


def test_teacher_dependencies_checked_again_after_inference(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"original")
    predictor = object.__new__(module.FixedRenderedTeacher)
    predictor.input_files = {str(path): module.file_sha256(path)}
    predictor.verify_inputs()
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed during inference"):
        predictor.verify_inputs()


def test_ensemble_export_uses_overscan_render_before_warp_and_does_not_read_photos(tmp_path, monkeypatch):
    import bridge_rgs.train

    class Scene(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.splats = {"means": torch.zeros(1, 3)}

        def render(self, K, pose, width, height, **kwargs):
            rgb = torch.full((height, width, 3), .3)
            probabilities = torch.zeros(height, width, 5)
            probabilities[..., 0] = 1
            return {"rgb": rgb, "probabilities": probabilities}

    shapes = []

    class Predictor:
        def __init__(self, *args):
            self.receipt = {"teacher_weight": .5, "input": "render only"}

        def blend(self, rgb, probabilities):
            shapes.append(rgb.shape)
            np.testing.assert_allclose(rgb, .3)
            probabilities = np.zeros_like(probabilities)
            probabilities[..., 4] = 1
            return probabilities

    monkeypatch.setattr(bridge_rgs.train, "load_scene", lambda *args: (Scene(), {}))
    monkeypatch.setattr(module, "FixedRenderedTeacher", Predictor)
    camera = {"name": "a.png", "width": 60, "height": 40,
              "K": [[50, 0, 30], [0, 50, 20], [0, 0, 1]], "w2c": np.eye(4).tolist(),
              "distortion": [-.2, 0, 0, 0, 0], "image_path": "/must/not/read.png",
              "mask_path": "/must/not/read/mask.png"}
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps([camera]))
    output = tmp_path / "output"
    receipt = render_cameras("unused.pt", path, output, teacher_checkpoint="unused-teacher.pt")
    assert shapes[0][0] > 40 and shapes[0][1] > 60
    mask = cv2.imread(str(output / "mask/a.png"), 0)
    assert mask.shape == (40, 60) and np.all(mask == 4)
    assert receipt["semantic_ensemble"]["teacher_weight"] == .5
