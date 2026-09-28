"""CPU-only original-grid contracts; no real renderer, LPIPS download or GPU."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from skimage.metrics import structural_similarity

from bridge_rgs import official_evaluate as official
from bridge_rgs.coordinates import CORNER, LEGACY, protocol_metadata
from bridge_rgs.data import CLASS_NAMES, rasterize_labelme


class FlatScene(torch.nn.Module):
    def __init__(self, events=None):
        super().__init__()
        self.splats = {"means": torch.zeros(4, 3)}
        self.events = events if events is not None else []

    def render(self, K, w2c, width, height, **kwargs):
        self.events.append("prediction")
        p = torch.zeros(height, width, 5)
        p[..., 1] = 1
        return {"rgb": torch.full((height, width, 3), .5003), "probabilities": p}


class CapturePerceptual:
    def __init__(self):
        self.inputs = []

    def __call__(self, first, second):
        self.inputs.append((first.clone(), second.clone()))
        return (first-second).square().mean().reshape(1, 1, 1, 1)


@pytest.fixture
def reference_fixture(tmp_path, monkeypatch):
    K = [[20., 0, 8.], [0, 20., 8.], [0, 0, 1.]]
    views = []
    for i in range(2):
        name = f"{i:03d}.png"
        image = tmp_path / name
        cv2.imwrite(str(image), np.full((16, 16, 3), 128, np.uint8))
        annotation = None
        if i == 0:
            annotation = tmp_path / "000.json"
            annotation.write_text(json.dumps({"imageWidth": 16, "imageHeight": 16, "shapes": [
                {"label": "deck", "points": [[0, 0], [15, 0], [15, 15], [0, 15]], "shape_type": "polygon"}]}))
        pose = np.eye(4)
        pose[0, 3] = i
        views.append({"name": name, "image_id": i+1, "camera_id": 1, "split": "val",
                      "K": [[10, 0, 4], [0, 10, 4], [0, 0, 1]], "width": 8, "height": 8,
                      "w2c": np.eye(4).tolist(), "w2c_original": pose.tolist(),
                      "image_path": "/must/not/read/prepared.png", "valid_path": "/must/not/read/valid.png",
                      "source_image_path": str(image), "source_annotation_path": str(annotation) if annotation else None})
    manifest = {"schema_version": 1, "class_names": CLASS_NAMES, "ignore_label": 255, "views": views,
                "source_cameras": {"1": {"K": K, "width": 16, "height": 16, "opencv_distortion": [0]*5}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(official, "REFERENCE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(official, "VAL_COUNT", 2)
    monkeypatch.setattr(official, "ANNOTATION_COUNT", 1)
    return path, manifest


def test_reference_uses_original_camera_and_plain_whitelist(reference_fixture):
    path, manifest = reference_fixture
    result = official.read_official_reference(path)
    assert len(result["cameras"]) == 2
    assert set(result["cameras"][0]) == official.CAMERA_KEYS
    assert result["cameras"][0]["width"] == 16
    assert result["cameras"][0]["K"] == manifest["source_cameras"]["1"]["K"]
    assert result["cameras"][1]["w2c"] == manifest["views"][1]["w2c_original"]


def test_reference_rejects_hash_count_missing_source_and_duplicates(reference_fixture, monkeypatch):
    path, manifest = reference_fixture
    monkeypatch.setattr(official, "REFERENCE_SHA256", "f"*64)
    with pytest.raises(ValueError, match="frozen reference"):
        official.read_official_reference(path)
    monkeypatch.setattr(official, "REFERENCE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(official, "VAL_COUNT", 50)
    with pytest.raises(ValueError, match="exactly"):
        official.read_official_reference(path)
    monkeypatch.setattr(official, "VAL_COUNT", 2)
    Path(manifest["views"][0]["source_image_path"]).unlink()
    with pytest.raises(FileNotFoundError):
        official.read_official_reference(path)


@pytest.mark.parametrize("profile", [LEGACY, CORNER])
def test_analytic_gsplat_center_rays_follow_checkpoint_profile(reference_fixture, profile):
    path, _ = reference_fixture
    camera = official.read_official_reference(path)["cameras"][0]
    camera["distortion"] = [-.2, 0, 0, 0, 0]

    class RayScene(FlatScene):
        def render(self, K, w2c, width, height, **kwargs):
            y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
            rays = torch.stack(((x+.5-K[0, 2])/K[0, 0], (y+.5-K[1, 2])/K[1, 1]), -1)
            rgb = torch.cat((.5+.3*rays, torch.full((height, width, 1), .4)), -1)
            p = torch.zeros(height, width, 5)
            p[..., 1] = 1
            return {"rgb": rgb, "probabilities": p}

    state = {} if profile == LEGACY else {"pixel_protocol": protocol_metadata(CORNER)}
    rgb, mask, info = official.predict_official_camera(RayScene(), camera, state)
    y, x = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    pixels = np.stack((x, y), -1).astype(np.float32)
    K = np.array(camera["K"], np.float32)
    if profile == CORNER:
        pixels += .5
    uv = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, np.array(camera["distortion"]), P=K).reshape(16, 16, 2)
    # Actual Gaussian pixel centers are .5 even when the legacy warp is unshifted.
    if profile == LEGACY:
        uv += .5
    expected = .5 + .3*(uv-K[:2, 2])/np.array([K[0, 0], K[1, 1]])
    np.testing.assert_allclose(rgb[..., :2]/255, expected, atol=1/255)
    assert (mask == 1).all() and info["checkpoint_pixel_protocol"] == profile
    assert info["pinhole_canvas"][0] > 16


def test_camera_metadata_cannot_upgrade_checkpoint(reference_fixture):
    path, _ = reference_fixture
    camera = official.read_official_reference(path)["cameras"][0]
    camera["pixel_protocol"] = protocol_metadata(CORNER)
    with pytest.raises(ValueError, match="whitelist"):
        official.predict_official_camera(FlatScene(), camera, {})


def test_soft_probabilities_are_warped_before_argmax(reference_fixture, monkeypatch):
    path, _ = reference_fixture
    camera = official.read_official_reference(path)["cameras"][0]
    source = np.zeros((16, 16, 2), np.float32)
    source[..., 0], source[..., 1] = .75, 5
    monkeypatch.setattr(official, "distortion_render_grid", lambda *args: (np.eye(3, dtype=np.float32), 16, 16, source))

    class SoftScene(FlatScene):
        def render(self, K, w2c, width, height, **kwargs):
            result = super().render(K, w2c, width, height, **kwargs)
            result["probabilities"][:, 1, 1] = .49
            result["probabilities"][:, 1, 2] = .51
            return result

    _, mask, _ = official.predict_official_camera(SoftScene(), camera, {})
    assert (mask == 1).all()  # Nearest sampling of precomputed IDs would instead give class2.


def test_known_rasterization_identical_and_unknown_ignore_draw_order(tmp_path):
    annotation = {"imageWidth": 16, "imageHeight": 16, "shapes": [
        {"label": " deck ", "points": [[.2, .4], [12.7, .4], [12.7, 12.4], [.2, 12.4]]},
        {"label": "tower", "points": [[6, 6], [9, 6], [9, 9], [6, 9]]}]}
    path = tmp_path / "known.json"
    path.write_text(json.dumps(annotation))
    actual = official.rasterize_official_annotation(path.read_bytes(), 16, 16)
    np.testing.assert_array_equal(actual, rasterize_labelme(path))
    annotation["shapes"].insert(1, {"label": "unknown-object", "points": [[2, 2], [10, 2], [10, 10], [2, 10]]})
    mask = official.rasterize_official_annotation(json.dumps(annotation), 16, 16)
    assert mask[3, 3] == 255 and mask[7, 7] == 3 and mask[1, 1] == 1 and mask[15, 15] == 0
    with pytest.raises(ValueError, match="shape mismatch"):
        official.rasterize_official_annotation(json.dumps(annotation), 15, 16)


def test_scoring_delivered_bytes_full_lpips_complete_ssim_and_ignore():
    pred, truth = np.zeros((16, 16, 3), np.uint8), np.zeros((16, 16, 3), np.uint8)
    pred[0, 0] = 255
    target_mask = np.zeros((16, 16), np.uint8)
    target_mask[0, 0], target_mask[2, 2] = 255, 1
    predicted_mask = np.zeros_like(target_mask)
    predicted_mask[2, 2] = 2
    perceptual = CapturePerceptual()
    metrics = official.score_official_arrays(pred, predicted_mask, truth, target_mask, perceptual)
    assert metrics["psnr"] == pytest.approx(10*np.log10(256))
    expected = structural_similarity(truth.astype(np.float32)/255, pred.astype(np.float32)/255,
                                    data_range=1, channel_axis=-1, win_size=11,
                                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    assert metrics["ssim"] == pytest.approx(expected)
    first, second = perceptual.inputs[0]
    assert first.shape == (1, 3, 16, 16)
    assert first[0, 0, 0, 0] == 1 and second[0, 0, 0, 0] == -1  # No GT replacement at image edge.
    matrix = np.array(metrics["confusion_matrix"])
    assert matrix.sum() == 255 and matrix[1, 2] == 1 and matrix[0, 0] == 254


def test_fingerprint_is_gt_sensitive_and_separate_from_native(reference_fixture):
    path, _ = reference_fixture
    camera = official.read_official_reference(path)["cameras"][0]
    record = {"camera": camera, "source_image_sha256": "a", "source_annotation_sha256": "b", "rasterized_mask_sha256": "c"}
    initial = official.official_fingerprint([record])
    changed = dict(record, rasterized_mask_sha256="d")
    assert official.official_fingerprint([changed]) != initial
    with pytest.raises(ValueError, match="no model/profile"):
        official.official_fingerprint([dict(record, checkpoint_pixel_protocol=LEGACY)])
    metric = {"evaluation_family": official.FAMILY, "official_evaluation_fingerprint": initial}
    official.require_same_official_protocol(metric, dict(metric))
    with pytest.raises(ValueError, match="native-grid"):
        official.require_same_official_protocol(metric, {"evaluation_fingerprint": initial})


def test_lineage_rejects_all_data_known_wrong_hash_and_unstamped_corner(reference_fixture, tmp_path):
    path, manifest = reference_fixture
    reference = official.read_official_reference(path)
    state = {"config": {"manifest": str(path)}}
    assert official.validate_training_lineage(state, reference)["checkpoint_declared_sha256"] is None
    with pytest.raises(ValueError, match="SHA differs"):
        official.validate_training_lineage(dict(state, manifest_sha256="a"*64), reference)
    with pytest.raises(ValueError, match="requires training manifest SHA"):
        official.validate_training_lineage(dict(state, pixel_protocol=protocol_metadata(CORNER)), reference)
    all_data = copy.deepcopy(manifest)
    all_data["views"][0]["split"] = "train"
    all_path = tmp_path / "all_data.json"
    all_path.write_text(json.dumps(all_data))
    with pytest.raises(ValueError, match="not held out"):
        official.validate_training_lineage({"config": {"manifest": str(all_path)}}, reference)


@pytest.mark.parametrize("use_teacher", [False, True])
def test_complete_predict_pass_precedes_gt_reads_and_profiles_share_fingerprint(reference_fixture, tmp_path, monkeypatch, use_teacher):
    path, manifest = reference_fixture
    real_read = Path.read_bytes
    source_paths = {Path(v["source_image_path"]) for v in manifest["views"]}
    source_paths |= {Path(v["source_annotation_path"]) for v in manifest["views"] if v.get("source_annotation_path")}
    events = []

    def read_bytes(file):
        if file in source_paths:
            assert events.count("prediction") == 2
            if use_teacher:
                assert events.count("teacher_prediction") == 2
            events.append("source_payload_read")
        return real_read(file)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(official, "_lpips", lambda device: CapturePerceptual())
    if use_teacher:
        import bridge_rgs.render_ensemble

        class Teacher:
            def __init__(self, *args, **kwargs):
                self.receipt = {"teacher_weight": .5, "input": "rendered RGB only"}

            def blend(self, rgb, p):
                events.append("teacher_prediction")
                np.testing.assert_allclose(rgb, .5003)
                return p

            def verify_inputs(self):
                events.append("teacher_inputs_verified")

        monkeypatch.setattr(bridge_rgs.render_ensemble, "FixedRenderedTeacher", Teacher)
    fingerprints = []
    for profile in (LEGACY, CORNER):
        events.clear()
        training_manifest = copy.deepcopy(manifest)
        state = {"config": {"manifest": str(path)}}
        if profile == CORNER:
            training_manifest.update(schema_version=2, pixel_protocol=protocol_metadata(CORNER))
            corner_path = tmp_path / "corner_manifest.json"
            corner_path.write_text(json.dumps(training_manifest))
            state.update(pixel_protocol=protocol_metadata(CORNER), manifest_sha256=hashlib.sha256(real_read(corner_path)).hexdigest(),
                         config={"manifest": str(corner_path)})
        checkpoint = tmp_path / f"{profile}.pt"
        torch.save(state, checkpoint)
        monkeypatch.setattr(official, "_load_scene", lambda *args, state=state: (FlatScene(events), state))
        out = tmp_path / profile
        metric = official.evaluate_official(checkpoint, out, path, device="cpu",
                                           teacher_checkpoint="unused.pt" if use_teacher else None)
        expected_events = ["prediction", "teacher_prediction"] * 2 if use_teacher else ["prediction"] * 2
        assert events[:len(expected_events)] == expected_events
        assert metric["psnr"] == 120 and metric["miou_all"] == 1
        assert metric["validation_views"] == 2 and metric["semantic_validation_views"] == 1
        assert "evaluation_fingerprint" not in metric
        receipt = json.loads((out / "execution_receipt.json").read_text())
        assert receipt["status"] == "completed"
        if use_teacher:
            assert receipt["teacher_ensemble"] == metric["teacher_ensemble"] == {
                "teacher_weight": .5, "input": "rendered RGB only"}
            assert metric["inference_protocol"] == official.TEACHER_INFERENCE
            assert events[-1] == "teacher_inputs_verified"
        else:
            assert "teacher_ensemble" not in metric
            assert metric["inference_protocol"] == official.PLAIN_INFERENCE
        assert receipt["checkpoint_pixel_protocol"]["id"] == profile
        assert receipt["source_records"][0]["rasterized_mask_sha256"]
        assert receipt["official_metrics_sha256"] == hashlib.sha256((out / "official_metrics.json").read_bytes()).hexdigest()
        assert receipt["loaded_source_modules"]["bridge_rgs.official_evaluate"]["path"] == str(Path(official.__file__).resolve())
        fingerprints.append(metric["official_evaluation_fingerprint"])
    assert fingerprints[0] == fingerprints[1]


def test_official_teacher_blends_on_covering_canvas_before_soft_warp(reference_fixture, monkeypatch):
    path, _ = reference_fixture
    camera = official.read_official_reference(path)["cameras"][0]
    source = np.zeros((16, 16, 2), np.float32)
    source[..., 0], source[..., 1] = .75, 5
    monkeypatch.setattr(official, "distortion_render_grid", lambda *args: (np.eye(3, dtype=np.float32), 20, 18, source))

    class Teacher:
        def blend(self, rgb, p):
            assert rgb.shape == (18, 20, 3)
            q = p.copy()
            q[:, 1, 1], q[:, 1, 2] = .49, .51
            return q

    rgb, mask, _ = official.predict_official_camera(FlatScene(), camera, {}, teacher=Teacher())
    assert np.all(mask == 1)  # Blending IDs after warp would instead return 2.
    assert np.all(rgb == 128)


def test_official_rejects_invalid_teacher_output(reference_fixture):
    path, _ = reference_fixture
    camera = official.read_official_reference(path)["cameras"][0]

    class Teacher:
        def blend(self, rgb, p):
            return np.full_like(p, np.nan)

    with pytest.raises(ValueError, match="Teacher mixture"):
        official.predict_official_camera(FlatScene(), camera, {}, teacher=Teacher())


def test_original_rgb_dimension_mismatch_does_not_resize():
    _, content = cv2.imencode(".png", np.zeros((16, 17, 3), np.uint8))
    with pytest.raises(ValueError, match="shape mismatch"):
        official._decode_rgb(content.tobytes(), 16, 16, "source")


def test_snapshot_module_resolves_relative_data_from_explicit_workspace(reference_fixture, tmp_path, monkeypatch):
    path, manifest = reference_fixture
    for view in manifest["views"]:
        for key in ("source_image_path", "source_annotation_path"):
            if view.get(key):
                view[key] = Path(view[key]).name
    path.write_text(json.dumps(manifest))
    snapshot = tmp_path / "runs/example/source_snapshot/bridge_rgs/official_evaluate.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(Path(official.__file__).read_bytes())
    spec = importlib.util.spec_from_file_location("bridge_rgs.official_evaluate_snapshot_cpu", snapshot)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REFERENCE_SHA256 = hashlib.sha256(path.read_bytes()).hexdigest()
    module.VAL_COUNT, module.ANNOTATION_COUNT = 2, 1
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    reference = module.read_official_reference("manifest.json", workspace_root=tmp_path)
    lineage = module.validate_training_lineage({"config": {"manifest": "manifest.json"}}, reference)
    assert lineage["path"] == str(path)
    assert reference["targets"][0]["source_image_path"] == str(tmp_path / "000.png")
    assert module._loaded_sources()["bridge_rgs.official_evaluate"]["path"] == str(snapshot)
