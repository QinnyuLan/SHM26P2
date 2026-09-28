"""CPU tests cover semantic alignment, ignored pixels, frozen weights and split isolation."""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from bridge_rgs import teacher
from bridge_rgs.coordinates import CORNER, LEGACY, protocol_metadata
from bridge_rgs.teacher_domains import domain_schedule, select_image_source, validate_image_sources


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="dinov3_vit", patch_size=16, num_hidden_layers=4, hidden_size=16
        )
        self.proj = nn.Conv2d(3, 16, 16, stride=16)

    def forward(self, pixel_values, output_hidden_states=True):
        patches = self.proj(pixel_values).flatten(2).transpose(1, 2)
        registers = patches.new_full((patches.shape[0], 5, 16), 1000)
        tokens = torch.cat((registers, patches), 1)
        return SimpleNamespace(hidden_states=tuple(tokens * (i + 1) for i in range(5)))


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model():
    return teacher.DINOv3Teacher(TinyBackbone(), num_classes=3, channels=16)


def tiny_loader(*args, pixel_profile=LEGACY, **kwargs):
    model = tiny_model()
    model.pixel_protocol = pixel_profile
    return model


def test_frozen_backbone_and_decoder_gradients():
    model = tiny_model().train()
    assert not model.backbone.training
    image = torch.rand(1, 3, 64, 64)
    features = model.extract_features(image)
    assert len(features) == 4
    assert all(feature.shape == (1, 16, 4, 4) for feature in features)
    assert all(not feature.requires_grad for feature in features)
    output = model(image)
    assert output["logits"].shape == (1, 3, 64, 64)
    loss, _ = teacher.supervised_loss(output, torch.randint(0, 3, (1, 64, 64)))
    loss.backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.decoder.parameters()
    )


def test_ignored_pixels_have_no_loss_gradient():
    logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    boundary = torch.randn(1, 1, 8, 8, requires_grad=True)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :4] = 255
    loss, _ = teacher.supervised_loss({"logits": logits, "boundary_logits": boundary}, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(logits.grad[:, :, :4]) == 0
    assert torch.count_nonzero(boundary.grad[:, :, :4]) == 0
    all_ignore = torch.full_like(target, 255)
    empty_loss, _ = teacher.supervised_loss(
        {"logits": logits, "boundary_logits": boundary}, all_ignore
    )
    assert empty_loss.item() == 0


def test_crop_preserves_label_valid_alignment_and_padding():
    mask = np.zeros((24, 30), np.uint8)
    mask[:, 15:] = 1
    image = np.repeat((mask * 255)[:, :, None], 3, 2)
    valid = np.ones_like(mask, dtype=bool)
    valid[:3] = False
    mask[~valid] = 255
    crop, labels, crop_valid = teacher.aligned_crop(
        image, mask, valid, 32, np.random.default_rng(7), augment=False
    )
    assert crop.shape == (3, 32, 32)
    assert torch.all(labels[~crop_valid] == 255)
    assert torch.equal((crop[0] > 0.5)[crop_valid], labels[crop_valid] == 1)


def test_consistency_rejects_uncertain_or_invalid_targets():
    calibration = teacher.ClassConfidence(3, "cpu")
    logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    uniform = torch.ones_like(logits) / 3
    valid = torch.ones((1, 8, 8), dtype=torch.bool)
    loss, fraction = calibration.loss(logits, uniform, valid)
    assert fraction == 0 and loss.item() == 0
    confident = F.one_hot(torch.ones((1, 8, 8), dtype=torch.long), 3).permute(0, 3, 1, 2).float()
    loss, fraction = calibration.loss(logits, confident, valid)
    assert fraction == 1 and loss.item() > 0
    loss.backward()
    assert logits.grad.abs().sum() > 0
    loss, fraction = calibration.loss(logits, confident, ~valid)
    assert loss.item() == 0 and fraction == 0


def test_context_frame_preserves_full_extent_alignment_and_ignored_padding():
    mask = np.zeros((40, 62), np.uint8)
    mask[:, 31:] = 1
    image = np.repeat((mask * 255)[:, :, None], 3, 2)
    valid = np.ones_like(mask, dtype=bool)
    valid[:3] = False
    frame, labels, keep = teacher.aligned_context_frame(
        image, mask, valid, 40, np.random.default_rng(7), augment=False
    )
    assert frame.shape == (3, 48, 64)
    assert torch.all(labels[~keep] == 255)
    assert not keep[40:].any() and not keep[:, 62:].any()
    assert torch.equal((frame[0] > 0.5)[keep], labels[keep] == 1)


def test_tiled_prediction_returns_native_grid_and_normalized_probabilities():
    model = tiny_model()
    image = np.random.default_rng(2).integers(0, 256, (71, 93, 3), dtype=np.uint8)
    probabilities, confidence = teacher.predict_image(
        model, image, tile_size=64, stride=48, flip=True
    )
    assert probabilities.shape == (3, 71, 93)
    assert confidence.shape == (71, 93)
    np.testing.assert_allclose(probabilities.sum(0), 1, atol=1e-6)
    assert np.isfinite(probabilities).all()
    assert ((confidence >= 0) & (confidence <= 1)).all()
    with pytest.raises(ValueError, match="stride"):
        teacher.predict_image(model, image, tile_size=64, stride=80)
    context = teacher.predict_context_image(model, image, short_side=64, flip=True)
    blended, confidence = teacher.predict_image(
        model, image, tile_size=64, stride=48, flip=False,
        context_weight=0.5, context_short_side=64,
    )
    assert context.shape == blended.shape == (3, 71, 93)
    np.testing.assert_allclose(context.sum(0), 1, atol=1e-6)
    np.testing.assert_allclose(blended.sum(0), 1, atol=1e-6)
    assert np.isfinite(confidence).all()


def test_training_split_excludes_validation_and_duplicate_names():
    manifest = {
        "views": [
            {"name": "train", "split": "train", "mask_path": "label.png"},
            {"name": "unlabeled", "split": "train", "mask_path": None},
            {"name": "validation", "split": "val", "mask_path": "secret.png"},
        ]
    }
    labeled, unlabeled = teacher.training_views(manifest)
    assert [v["name"] for v in labeled] == ["train"]
    assert [v["name"] for v in unlabeled] == ["unlabeled"]
    manifest["views"].append(manifest["views"][0])
    with pytest.raises(ValueError, match="unique"):
        teacher.training_views(manifest)


def test_confidence_calibration_uses_predicted_class_precision_and_target_coverage():
    total = np.array([[90, 10], [5, 95]])
    accepted = np.array([[80, 4], [2, 70]])
    summary = teacher.confidence_calibration_summary(
        total, {0.8: accepted}, np.array([120, 200]), {0.8: np.array([110, 150])}
    )["threshold_metrics"]["0.8"]
    assert summary["labeled_train_prediction_error_rate"][0] == pytest.approx(2 / 82)
    assert summary["labeled_train_predicted_class_coverage"][0] == pytest.approx(82 / 95)
    assert summary["labeled_train_target_class_coverage"][0] == pytest.approx(0.84)
    assert summary["labeled_train_correct_target_class_coverage"][0] == pytest.approx(0.8)
    assert summary["all_train_predicted_class_coverage"][0] == pytest.approx(110 / 120)


def test_same_camera_real_and_rendered_cache_entries_are_distinct(tmp_path):
    view = {"name": "same.png", "split": "train", "height": 32, "width": 32,
            "mask_path": None, "valid_path": None, "image_path_sources": {}}
    for domain, value in (("real", 25), ("rendered", 200)):
        path = tmp_path / (domain + ".png")
        cv2.imwrite(str(path), np.full((32, 32, 3), value, np.uint8))
        view["image_path_sources"][domain] = {"path": str(path)}
    reader = teacher.ViewReader(3)
    assert reader.read(select_image_source(view, False))[0].mean() == 25
    assert reader.read(select_image_source(view, True))[0].mean() == 200
    assert reader.read(select_image_source(view, False))[0].mean() == 25
    schedule = domain_schedule(2000, 0.5, 12)
    assert schedule.sum() == 1000
    np.testing.assert_array_equal(schedule, domain_schedule(2000, 0.5, 12))
    with pytest.raises(ValueError, match="train"):
        select_image_source({**view, "split": "val"}, True)


@pytest.mark.parametrize("profile", [LEGACY, CORNER])
def test_derived_domain_warmstart_preserves_split_and_never_reads_real_val(tmp_path, monkeypatch, profile):
    from copy import deepcopy

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    monkeypatch.setattr(teacher, "load_teacher", tiny_loader)
    original = {"class_names": ["background", "deck", "cable"], "views": []}
    if profile == CORNER:
        original.update(schema_version=2, pixel_protocol=protocol_metadata(profile))
    for split in ("train", "val"):
        real = tmp_path / f"{split}_real.png"
        rendered = tmp_path / f"{split}_rendered.png"
        label = tmp_path / f"{split}_mask.png"
        # There is deliberately no real validation photograph on disk.
        if split == "train":
            cv2.imwrite(str(real), np.full((64, 64, 3), 60, np.uint8))
        cv2.imwrite(str(rendered), np.full((64, 64, 3), 180, np.uint8))
        cv2.imwrite(str(label), np.zeros((64, 64), np.uint8))
        original["views"].append({"name": split + ".png", "split": split,
                                  "image_path": str(real), "mask_path": str(label),
                                  "valid_path": None, "width": 64, "height": 64})
    original_path = tmp_path / "original.json"
    original_path.write_text(json.dumps(original))
    renderer = tmp_path / "renderer.pt"
    torch.save({"pixel_protocol": protocol_metadata(profile),
                "manifest_sha256": teacher.file_sha256(original_path)}, renderer)
    derived = deepcopy(original)
    protocol = {"original_manifest": str(original_path),
                "pixel_protocol": protocol_metadata(profile),
                "original_manifest_sha256": teacher.file_sha256(original_path),
                "renderer_checkpoint": str(renderer),
                "renderer_checkpoint_sha256": teacher.file_sha256(renderer)}
    for view in derived["views"]:
        split = view["split"]
        path = str(tmp_path / f"{split}_rendered.png")
        row = {"name": view["name"], "image_path": path, "sha256": teacher.file_sha256(path),
               "width": 64, "height": 64}
        receipt = tmp_path / f"{split}_receipt.json"
        receipt.write_text(json.dumps({"status": "completed", "grid": "pinhole",
                                       "pixel_protocol": protocol_metadata(profile),
                                       "checkpoint_sha256": teacher.file_sha256(renderer),
                                       "source_manifest_sha256": teacher.file_sha256(original_path),
                                       "records": [row]}))
        protocol[f"{split}_render_receipt"] = {"path": str(receipt), "sha256": teacher.file_sha256(receipt)}
        view["image_path_sources"] = {"rendered": {"path": path, "sha256": row["sha256"]}}
        view["image_domain"] = "mixed_real_rendered" if split == "train" else "rendered_rgb"
        if split == "train":
            view["image_path_sources"]["real"] = {"path": view["image_path"], "sha256": teacher.file_sha256(view["image_path"])}
        else:
            view["image_path"] = path
    derived["image_source_protocol"] = protocol
    assert validate_image_sources(derived) == protocol
    from bridge_rgs import teacher_domains

    original_sha = teacher_domains._sha256
    def no_val_bytes(path):
        if str(path) == str(tmp_path / "val_rendered.png"):
            raise AssertionError("VAL cache pixel payload read")
        return original_sha(path)
    with monkeypatch.context() as patch:
        patch.setattr(teacher_domains, "_sha256", no_val_bytes)
        assert validate_image_sources(derived, verify_validation_files=False) == protocol
        with pytest.raises(AssertionError, match="VAL cache"):
            validate_image_sources(derived)  # The historical default remains unchanged.
    bad = deepcopy(derived)
    bad["views"][0]["mask_path"] = "generated_mask.png"
    with pytest.raises(ValueError, match="Non-image fields"):
        validate_image_sources(bad)
    bad = deepcopy(derived)
    bad["views"][0]["split"] = "val"
    with pytest.raises(ValueError, match="Non-image fields"):
        validate_image_sources(bad)
    manifest_path = tmp_path / "derived.json"
    manifest_path.write_text(json.dumps(derived))
    checkpoint = tmp_path / "original_teacher.pt"
    torch.save({"configuration": {"adapter_rank": 0}, "step": 1,
                "provenance": {"manifest_sha256": teacher.file_sha256(original_path),
                               "pixel_protocol": protocol_metadata(profile),
                               "model_config_sha256": teacher.file_sha256(model_dir / "config.json"),
                               "model_weights_sha256": {}},
                "ema_decoder": tiny_model().decoder.state_dict(),
                "class_thresholds": torch.full((3,), 0.8)}, checkpoint)
    config = teacher.TeacherConfig(
        steps=2, crop_size=64, channels=16, eval_every=2, log_every=1,
        device="cpu", val_stride=48, consistency_start=100,
        render_mix_probability=0.5, eval_flip=True, eval_context_weight=0.25,
        eval_context_short_side=64, eval_all_validation_views=True,
        warmstart_checkpoint=str(checkpoint),
    )
    result = teacher.train_teacher(manifest_path, model_dir, tmp_path / "adapt", config)
    assert result["domain_counts"] == {"real": 1, "rendered": 1}
    assert result["validation"]["image_domains"] == ["rendered_rgb"]
    assert result["validation"]["flip_tta"]
    assert result["validation"]["context_weight"] == 0.25
    assert not (tmp_path / "val_real.png").exists()
    saved = torch.load(tmp_path / "adapt/last.pt", weights_only=False)
    assert teacher.checkpoint_pixel_protocol(saved) == profile
    assert saved["provenance"]["warmstart"]["pixel_protocol"] == protocol_metadata(profile)


@pytest.mark.parametrize("profile", [LEGACY, CORNER])
def test_train_resume_and_export_preserve_split_provenance(tmp_path, monkeypatch, profile):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    monkeypatch.setattr(teacher, "load_teacher", tiny_loader)
    manifest = {"class_names": ["background", "deck", "cable"], "views": []}
    if profile == CORNER:
        manifest.update(schema_version=2, pixel_protocol=protocol_metadata(profile))
    for name, split, labeled in [
        ("train.png", "train", True),
        ("unlabeled.png", "train", False),
        ("val.png", "val", True),
    ]:
        image_path, mask_path = tmp_path / name, tmp_path / (name + ".mask.png")
        cv2.imwrite(str(image_path), np.full((64, 64, 3), 120, np.uint8))
        mask = np.zeros((64, 64), np.uint8)
        mask[:, 32:] = 1
        cv2.imwrite(str(mask_path), mask)
        manifest["views"].append(
            {
                "name": name,
                "split": split,
                "image_path": str(image_path),
                "mask_path": str(mask_path) if labeled else None,
                "valid_path": None,
                "width": 64,
                "height": 64,
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = teacher.TeacherConfig(
        steps=1,
        crop_size=64,
        channels=16,
        consistency_start=1,
        eval_every=1,
        log_every=1,
        device="cpu",
        val_stride=48,
    )
    output = tmp_path / "train"
    first = teacher.train_teacher(manifest_path, model_dir, output, config)
    assert first["step"] == 1
    assert first["best_validation_miou"] is not None
    checkpoint = torch.load(output / "last.pt", weights_only=False)
    assert checkpoint["provenance"]["train_labeled_views"] == ["train.png"]
    assert checkpoint["provenance"]["train_unlabeled_views"] == ["unlabeled.png"]
    assert checkpoint["validation"]["views"] == ["val.png"]
    assert checkpoint["pixel_protocol"] == checkpoint["provenance"]["pixel_protocol"] == protocol_metadata(profile)
    config.steps = 2
    second = teacher.train_teacher(
        manifest_path, model_dir, output, config, resume=output / "last.pt"
    )
    assert second["step"] == 2
    warm_config = teacher.TeacherConfig(
        steps=1, crop_size=64, channels=16, eval_every=1, log_every=1,
        device="cpu", val_stride=48, warmstart_checkpoint=str(output / "last.pt"),
    )
    warm_output = tmp_path / "warm"
    teacher.train_teacher(manifest_path, model_dir, warm_output, warm_config)
    warm_config.steps = 2
    resumed_warm = teacher.train_teacher(
        manifest_path, model_dir, warm_output, warm_config, resume=warm_output / "last.pt"
    )
    assert resumed_warm["step"] == 2
    export = tmp_path / "pseudo"
    teacher.predict_teacher(
        manifest_path, output / "last.pt", export, device="cpu", tile_size=64, stride=48, flip=False
    )
    assert sorted(p.name for p in export.glob("*.npz")) == ["train.npz", "unlabeled.npz"]
    provenance = json.loads((export / "provenance.json").read_text())
    assert provenance["complete"]
    assert provenance["manifest_sha256"] == teacher.file_sha256(manifest_path)
    assert provenance["pixel_protocol"] == protocol_metadata(profile)
    assert provenance["teacher_provenance"]["pixel_protocol"] == protocol_metadata(profile)
    assert teacher.verify_pseudo_provenance(manifest_path, export)["complete"]
    calibration = json.loads((export / "train_confidence_calibration.json").read_text())
    assert calibration["labeled_views"] == ["train.png"]
    assert calibration["prediction_views"] == ["train.png", "unlabeled.png"]
    assert set(calibration["threshold_metrics"]) == {"0.65", "0.8", "0.9"}
    assert calibration["pixel_protocol"] == protocol_metadata(profile)
    evaluation = teacher.evaluate_checkpoint(
        manifest_path,
        output / "last.pt",
        tmp_path / "validation",
        device="cpu",
        tile_size=64,
        stride=48,
        previews=0,
    )
    assert evaluation["views"] == ["val.png"]
    assert evaluation["fixed_checkpoint_evaluation"]
    assert evaluation["pixel_protocol"] == protocol_metadata(profile)
    assert not list((tmp_path / "validation").glob("*.npz"))
    with np.load(export / "unlabeled.npz") as data:
        assert data["probs"].shape == (3, 64, 64)
        assert data["split"].item() == "train"
    manifest["views"][0]["split"] = "val"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest differs"):
        teacher.predict_teacher(manifest_path, output / "last.pt", tmp_path / "bad", device="cpu")


def test_checkpoint_protocol_only_uses_stored_declarations(tmp_path):
    manifest = tmp_path / "today.json"
    manifest.write_text(json.dumps({"schema_version": 2, "pixel_protocol": protocol_metadata(CORNER)}))
    historical = {"provenance": {"manifest": str(manifest),
                                 "manifest_sha256": teacher.file_sha256(manifest)}}
    assert teacher.checkpoint_pixel_protocol(historical) == LEGACY
    for checkpoint in ({"pixel_protocol": CORNER},
                       {"provenance": {"pixel_protocol": protocol_metadata(CORNER)}},
                       {"pixel_protocol": CORNER, "provenance": {"pixel_protocol": CORNER}}):
        assert teacher.checkpoint_pixel_protocol(checkpoint) == CORNER
    with pytest.raises(ValueError, match="disagree"):
        teacher.checkpoint_pixel_protocol({"pixel_protocol": CORNER,
                                           "provenance": {"pixel_protocol": LEGACY}})
    with pytest.raises(ValueError, match="Unknown"):
        teacher.checkpoint_pixel_protocol({"provenance": {"pixel_protocol": "auto"}})


def test_backbone_index_binding_is_required_only_for_sharded_models(tmp_path):
    assert teacher.backbone_index_hashes(tmp_path) == {}
    teacher.require_backbone_index({}, {}, "historical")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text('{"weight_map": {"x": "model-1.safetensors"}}')
    hashes = teacher.backbone_index_hashes(tmp_path)
    with pytest.raises(ValueError, match="index"):
        teacher.require_backbone_index({}, hashes, "sharded")
    teacher.require_backbone_index({"model_index_sha256": hashes}, hashes, "new")
    index.write_text('{"weight_map": {"x": "model-2.safetensors"}}')
    with pytest.raises(ValueError, match="changed"):
        teacher.require_backbone_index({"model_index_sha256": hashes}, teacher.backbone_index_hashes(tmp_path), "changed")


def test_periodic_recovery_saves_do_not_evaluate_and_augmentation_rng_ignores_model_size(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    image, mask = tmp_path / "image.png", tmp_path / "mask.png"
    cv2.imwrite(str(image), np.full((64, 64, 3), 120, np.uint8))
    cv2.imwrite(str(mask), np.zeros((64, 64), np.uint8))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"class_names": ["background", "deck", "cable"], "views": [{
        "name": "train.png", "split": "train", "image_path": str(image), "mask_path": str(mask),
        "valid_path": None, "width": 64, "height": 64}]}))
    config = teacher.TeacherConfig(steps=3, crop_size=64, channels=16, consistency_start=100,
                                   eval_every=3, checkpoint_every=1, independent_augmentation_rng=True,
                                   log_every=3, device="cpu", cpu_threads=1)
    original_save = torch.save
    saves, streams = [], []

    def save(value, path, *args, **kwargs):
        saves.append((value["step"], value["validation"], value["provenance"]["model_index_sha256"]))
        return original_save(value, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", save)
    for count in (1, 173):
        stream = []

        def loader(*args, count=count, **kwargs):
            torch.rand(count)  # Model construction may consume a different RNG prefix.
            return tiny_loader(*args, **kwargs)

        def augment(value, strength, stream=stream):
            stream.append(torch.rand(4).tolist())
            return value

        monkeypatch.setattr(teacher, "load_teacher", loader)
        monkeypatch.setattr(teacher, "photometric_augment", augment)
        teacher.train_teacher(manifest, model_dir, tmp_path / f"run{count}", config)
        streams.append(stream)
    assert streams[0] == streams[1] and len(streams[0]) == 3
    assert saves == [(1, None, {}), (2, None, {}), (3, None, {})] * 2
    config.independent_augmentation_rng = False
    config.steps = 4
    with pytest.raises(ValueError, match="augmentation RNG"):
        teacher.train_teacher(manifest, model_dir, tmp_path / "run173", config,
                              resume=tmp_path / "run173/last.pt")


@pytest.mark.parametrize("profile", [LEGACY, CORNER])
def test_head_only_weight_loading_rejects_mixed_profiles(profile):
    model = tiny_loader(pixel_profile=profile)
    assert not model.train_backbone_adapters
    teacher.load_checkpoint_adapters(model, {"provenance": {"pixel_protocol": profile}})
    wrong = CORNER if profile == LEGACY else LEGACY
    with pytest.raises(ValueError, match="Pixel protocol mismatch"):
        teacher.load_checkpoint_adapters(model, {"provenance": {"pixel_protocol": wrong}})


@pytest.mark.parametrize("operation", ["predict", "evaluate", "warmstart", "resume"])
@pytest.mark.parametrize("profile", [LEGACY, CORNER])
def test_teacher_entrypoints_reject_mixed_checkpoint_even_with_matching_manifest_sha(
    tmp_path, monkeypatch, operation, profile
):
    manifest = {"pixel_protocol": protocol_metadata(profile), "class_names": ["background", "deck", "cable"],
                "views": [{"name": "train.png", "split": "train", "mask_path": "not-opened.png"}]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    checkpoint = tmp_path / "wrong.pt"
    source = {"manifest_sha256": teacher.file_sha256(manifest_path)}
    # A missing declaration represents a genuine historical legacy checkpoint.
    if profile == LEGACY:
        source["pixel_protocol"] = protocol_metadata(CORNER)
    torch.save({"provenance": source}, checkpoint)
    monkeypatch.setattr(teacher, "load_teacher", tiny_loader)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="Pixel protocol mismatch"):
        if operation == "predict":
            teacher.predict_teacher(manifest_path, checkpoint, tmp_path / "out", device="cpu")
        elif operation == "evaluate":
            teacher.evaluate_checkpoint(manifest_path, checkpoint, tmp_path / "out", device="cpu")
        else:
            config = teacher.TeacherConfig(steps=2, crop_size=64, channels=16, device="cpu",
                                            warmstart_checkpoint=str(checkpoint) if operation == "warmstart" else "")
            teacher.train_teacher(manifest_path, model_dir, tmp_path / "out", config,
                                  resume=checkpoint if operation == "resume" else None)


def pseudo_fixture(tmp_path, profile, *, explicit=True):
    manifest = {"class_names": ["background", "deck"],
                "views": [{"name": "train.png", "split": "train", "width": 2, "height": 1,
                           "mask_path": "not-opened.png"}]}
    if explicit:
        manifest["pixel_protocol"] = protocol_metadata(profile)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    digest = teacher.file_sha256(manifest_path)
    directory = tmp_path / "pseudo"
    directory.mkdir()
    path = directory / "train.npz"
    np.savez_compressed(path, probs=np.zeros((2, 1, 2), np.float16),
                        confidence=np.zeros((1, 2), np.float16), valid=np.ones((1, 2), bool),
                        split="train", view_name="train.png", manifest_sha256=digest, checkpoint_sha256="teacher_sha")
    source = {"manifest_sha256": digest, "train_labeled_views": ["train.png"], "train_unlabeled_views": []}
    provenance = {"complete": True, "kind": "train_only_teacher_soft_probabilities", "split": "train",
                  "class_names": manifest["class_names"], "manifest_sha256": digest,
                  "teacher_provenance": source, "checkpoint_sha256": "teacher_sha",
                  "views": [{"name": "train.png", "split": "train", "width": 2, "height": 1,
                             "sha256": teacher.file_sha256(path)}]}
    if explicit:
        provenance["pixel_protocol"] = source["pixel_protocol"] = protocol_metadata(profile)
    (directory / "provenance.json").write_text(json.dumps(provenance))
    return manifest_path, directory, provenance


@pytest.mark.parametrize("profile,explicit", [(LEGACY, False), (LEGACY, True), (CORNER, True)])
def test_pseudo_directory_profile_preserves_historical_npz_format(tmp_path, profile, explicit):
    manifest, directory, _ = pseudo_fixture(tmp_path, profile, explicit=explicit)
    assert teacher.verify_pseudo_provenance(manifest, directory)["complete"]
    with np.load(directory / "train.npz") as value:
        assert "pixel_protocol" not in value.files


@pytest.mark.parametrize("field", ["directory", "teacher", "calibration", "embedded"])
def test_pseudo_profiles_reject_mixing_despite_matching_sha_lineage(tmp_path, field):
    manifest, directory, provenance = pseudo_fixture(tmp_path, CORNER)
    if field == "directory":
        provenance.pop("pixel_protocol")
    elif field == "teacher":
        provenance["teacher_provenance"].pop("pixel_protocol")
    elif field == "calibration":
        path = directory / "train_confidence_calibration.json"
        path.write_text(json.dumps({"pixel_protocol": protocol_metadata(LEGACY)}))
        provenance["train_confidence_calibration"] = {"sha256": teacher.file_sha256(path)}
    else:
        path = directory / "train.npz"
        with np.load(path) as values:
            tensors = dict(values)
        np.savez_compressed(path, **tensors, pixel_protocol=LEGACY)
        provenance["views"][0]["sha256"] = teacher.file_sha256(path)
    (directory / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="Pixel protocol mismatch"):
        teacher.verify_pseudo_provenance(manifest, directory)


def test_pseudo_still_rejects_changed_manifest_with_same_profile(tmp_path):
    manifest, directory, _ = pseudo_fixture(tmp_path, CORNER)
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="Pseudo manifest differs"):
        teacher.verify_pseudo_provenance(manifest, directory, verify_files=False)


@pytest.mark.parametrize("mismatch", ["source", "train", "val", "renderer", "renderer_hash"])
def test_render_teacher_rejects_mixed_grid_or_renderer_manifest(tmp_path, mismatch):
    manifest = {"pixel_protocol": protocol_metadata(CORNER), "views": []}
    original = tmp_path / "original.json"
    original.write_text(json.dumps(manifest))
    protocol = {"pixel_protocol": protocol_metadata(CORNER), "original_manifest": str(original),
                "original_manifest_sha256": teacher.file_sha256(original)}
    for split in ("train", "val"):
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps({"pixel_protocol": protocol_metadata(CORNER if mismatch != split else LEGACY)}))
        protocol[f"{split}_render_receipt"] = {"path": str(path)}
    renderer = tmp_path / "renderer.pt"
    state = {"pixel_protocol": protocol_metadata(CORNER if mismatch != "renderer" else LEGACY),
             "manifest_sha256": "wrong" if mismatch == "renderer_hash" else teacher.file_sha256(original)}
    torch.save(state, renderer)
    protocol["renderer_checkpoint"] = str(renderer)
    if mismatch == "source":
        protocol.pop("pixel_protocol")
    with pytest.raises(ValueError, match="Pixel protocol mismatch|manifest SHA differs"):
        teacher.verify_teacher_render_protocol(manifest, protocol)


def test_legacy_renderer_stays_legacy_and_modern_renderer_requires_manifest_sha():
    assert teacher.require_teacher_renderer_protocol({}, {"views": []}, "current") == LEGACY
    modern = {"pixel_protocol": protocol_metadata(CORNER)}
    with pytest.raises(ValueError, match="manifest SHA differs"):
        teacher.require_teacher_renderer_protocol(modern, {**modern, "views": []}, "current")


def test_validation_can_be_disabled_without_reading_pixels_or_selecting_best(tmp_path, monkeypatch):
    model_dir = tmp_path/'model'
    model_dir.mkdir()
    (model_dir/'config.json').write_text('{}')
    monkeypatch.setattr(teacher, 'load_teacher', tiny_loader)
    image, mask = tmp_path/'rgb.png', tmp_path/'mask.png'
    cv2.imwrite(str(image), np.full((64, 64, 3), 120, np.uint8))
    cv2.imwrite(str(mask), np.zeros((64, 64), np.uint8))
    views = [{'name': 'train.png', 'split': 'train', 'image_path': str(image), 'mask_path': str(mask),
              'valid_path': None, 'width': 64, 'height': 64},
             {'name': 'val.png', 'split': 'val', 'image_path': 'MISSING_VAL_RGB', 'mask_path': 'MISSING_VAL_GT',
              'valid_path': None, 'width': 64, 'height': 64}]
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'class_names': ['background', 'deck', 'cable'], 'views': views}))
    def no_evaluation(*args, **kwargs):
        raise AssertionError('Evaluation must not be called')
    monkeypatch.setattr(teacher, 'evaluate_teacher', no_evaluation)
    config = teacher.TeacherConfig(steps=2, crop_size=64, channels=16, consistency_start=100,
        eval_every=1, checkpoint_every=1, evaluate_validation=False, device='cpu', cpu_threads=1)
    result = teacher.train_teacher(manifest, model_dir, tmp_path/'out', config)
    assert result['validation'] is None and result['best_checkpoint'] is None
    state = torch.load(tmp_path/'out/last.pt', weights_only=False)
    assert state['step'] == 2 and state['validation'] is None and not state['configuration']['evaluate_validation']
    assert len(json.loads(manifest.read_text())['views']) == 2
    config.steps, config.evaluate_validation = 3, True
    with pytest.raises(ValueError, match='validation policy'):
        teacher.train_teacher(manifest, model_dir, tmp_path/'out', config, resume=tmp_path/'out/last.pt')
