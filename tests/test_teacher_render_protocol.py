"""CPU metadata/receipt contracts for teacher rendering-domain export scripts."""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from bridge_rgs.coordinates import CORNER, LEGACY, protocol_metadata
from bridge_rgs.teacher import file_sha256, verify_teacher_render_protocol
from bridge_rgs.teacher_domains import validate_image_sources


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def cameras(profile, *, native=True):
    width, height = (1320, 989) if native else (4, 3)
    manifest = {"class_names": ["background", "deck", "cable"], "views": []}
    if profile == CORNER:
        manifest.update(schema_version=2, pixel_protocol=protocol_metadata(CORNER))
    for index in range(400):
        split = "train" if index < 350 else "val"
        manifest["views"].append({"name": f"{index:03d}.png", "split": split,
                                  "image_id": index, "camera_id": 1, "width": width, "height": height,
                                  "K": np.eye(3).tolist(), "w2c": np.eye(4).tolist(),
                                  "w2c_original": np.eye(4).tolist(),
                                  "image_path": f"{split}_real.png",
                                  "mask_path": "unopened_gt.png" if index < 259 else None,
                                  "valid_path": None})
    return manifest


@pytest.mark.parametrize("profile", [LEGACY, CORNER])
def test_train_camera_whitelist_preserves_grid_and_excludes_pixel_sources(profile):
    module = script("render_teacher_training_views")
    manifest = cameras(profile)
    result = module.training_cameras(manifest)
    assert len(result) == 350
    assert {row["name"] for row in result} == {row["name"] for row in manifest["views"][:350]}
    assert all(set(row) == {"name", "image_id", "camera_id", "width", "height", "K", "w2c", "w2c_original"}
               for row in result)
    manifest["views"][0]["pixel_protocol"] = CORNER if profile == LEGACY else LEGACY
    with pytest.raises(ValueError, match="View pixel protocol disagrees"):
        module.training_cameras(manifest)


@pytest.mark.parametrize("profile", [LEGACY, CORNER])
def test_render_domain_script_propagates_profile_without_real_validation_input(tmp_path, monkeypatch, profile):
    module = script("prepare_teacher_render_domain")
    manifest = cameras(profile, native=False)
    real, rendered_train = tmp_path / "real.png", tmp_path / "rendered_train.png"
    cv2.imwrite(str(real), np.full((3, 4, 3), 50, np.uint8))
    cv2.imwrite(str(rendered_train), np.full((3, 4, 3), 150, np.uint8))
    for view in manifest["views"]:
        view["image_path"] = str(real if view["split"] == "train" else tmp_path / "val_photo_does_not_exist.png")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    checkpoint = tmp_path / "renderer.pt"
    # The legacy branch deliberately models an old checkpoint without a profile.
    state = {"manifest_sha256": file_sha256(manifest_path)}
    if profile == CORNER:
        state["pixel_protocol"] = protocol_metadata(CORNER)
    torch.save(state, checkpoint)
    receipt = {"status": "completed", "grid": "pinhole", "checkpoint": str(checkpoint),
               "checkpoint_sha256": file_sha256(checkpoint),
               "source_manifest_sha256": file_sha256(manifest_path),
               "records": [{"name": view["name"], "image_path": str(rendered_train),
                            "sha256": file_sha256(rendered_train), "width": 4, "height": 3}
                           for view in manifest["views"] if view["split"] == "train"]}
    if profile == CORNER:
        receipt["pixel_protocol"] = protocol_metadata(CORNER)
    receipt_path = tmp_path / "train_receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    calls = []

    def fake_render(checkpoint_path, camera_path, output, *, grid):
        # Mock only GPU rasterization; execute all real lineage and file audits.
        document = json.loads(camera_path.read_text())
        assert document["pixel_protocol"] == protocol_metadata(profile)
        assert document["source_manifest_sha256"] == file_sha256(manifest_path)
        assert checkpoint_path == checkpoint and grid == "pinhole"
        assert len(document["views"]) == 50
        assert all(not any(key in row for key in ("image_path", "mask_path", "valid_path"))
                   for row in document["views"])
        (output / "rgb").mkdir()
        for row in document["views"]:
            cv2.imwrite(str(output / "rgb" / row["name"]), np.full((3, 4, 3), 150, np.uint8))
        calls.append(document)
        return {"views": 50, "pixel_protocol": protocol_metadata(profile)}

    monkeypatch.setattr(module, "render_cameras", fake_render)
    output = tmp_path / "derived"
    monkeypatch.setattr(sys, "argv", ["prepare_teacher_render_domain", "--manifest", str(manifest_path),
                                      "--train-receipt", str(receipt_path), "--output", str(output)])
    module.main()
    assert len(calls) == 1 and not (tmp_path / "val_photo_does_not_exist.png").exists()
    derived = json.loads((output / "manifest.json").read_text())
    protocol = validate_image_sources(derived)
    verify_teacher_render_protocol(derived, protocol)
    assert protocol["pixel_protocol"] == protocol_metadata(profile)
    val_receipt = json.loads(Path(protocol["val_render_receipt"]["path"]).read_text())
    assert val_receipt["pixel_protocol"] == protocol_metadata(profile)
    assert [view["mask_path"] for view in derived["views"]] == [view["mask_path"] for view in manifest["views"]]

    # Same manifest SHA is insufficient when the receipt silently drops v2.
    if profile == CORNER:
        receipt.pop("pixel_protocol")
        receipt_path.write_text(json.dumps(receipt))
        monkeypatch.setattr(sys, "argv", ["prepare_teacher_render_domain", "--manifest", str(manifest_path),
                                          "--train-receipt", str(receipt_path), "--output", str(tmp_path / "bad")])
        with pytest.raises(ValueError, match="Pixel protocol mismatch"):
            module.main()
        assert len(calls) == 1


def test_context_script_rejects_legacy_teacher_on_v2_before_loading_backbone(tmp_path, monkeypatch):
    module = script("evaluate_teacher_context")
    manifest = {"pixel_protocol": protocol_metadata(CORNER), "class_names": ["background"], "views": []}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"provenance": {"manifest_sha256": file_sha256(manifest_path)}}, checkpoint)

    def forbidden(*args, **kwargs):
        pytest.fail("A rejected grid must not load the backbone")

    monkeypatch.setattr(module, "load_teacher", forbidden)
    monkeypatch.setattr(sys, "argv", ["evaluate_teacher_context", "--manifest", str(manifest_path),
                                      "--checkpoint", str(checkpoint), "--output", str(tmp_path / "out.json")])
    with pytest.raises(ValueError, match="Pixel protocol mismatch"):
        module.main()


def test_render_adaptation_comparison_rejects_different_profiles(tmp_path, monkeypatch):
    module = script("compare_teacher_render_adaptation")
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text("{}")  # Historical missing declaration is legacy.
    after.write_text(json.dumps({"pixel_protocol": protocol_metadata(CORNER)}))
    monkeypatch.setattr(sys, "argv", ["compare_teacher_render_adaptation", "--before", str(before),
                                      "--after", str(after), "--output", str(tmp_path / "comparison.json")])
    with pytest.raises(ValueError, match="Pixel protocol mismatch"):
        module.main()
