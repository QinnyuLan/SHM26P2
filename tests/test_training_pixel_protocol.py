import copy
import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from bridge_rgs.coordinates import CORNER, LEGACY, protocol_metadata, resize_discrete_numpy
from bridge_rgs.teacher import aligned_context_frame
from bridge_rgs.train import bind_training_pixel_protocol


def write_manifest(tmp_path, protocol=None):
    manifest = {"views": []}
    if protocol:
        manifest["pixel_protocol"] = protocol_metadata(protocol)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return manifest, path


def test_absent_metadata_binds_legacy_without_changing_input(tmp_path):
    manifest, path = write_manifest(tmp_path)
    before = path.read_bytes()
    scene, config = SimpleNamespace(), {}
    bind_training_pixel_protocol(scene, manifest, path, config, initial={})
    assert scene.pixel_protocol == config["pixel_protocol"] == LEGACY
    assert scene.manifest_sha256 == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before


def test_new_profile_cannot_be_silently_attached_to_old_field(tmp_path):
    manifest, path = write_manifest(tmp_path, CORNER)
    with pytest.raises(ValueError, match="protocol mismatch"):
        bind_training_pixel_protocol(SimpleNamespace(), manifest, path, {}, initial={})
    with pytest.raises(ValueError, match="protocol mismatch"):
        bind_training_pixel_protocol(SimpleNamespace(), manifest, path, {"pixel_protocol": LEGACY})
    scene, config = SimpleNamespace(), {}
    bind_training_pixel_protocol(scene, manifest, path, config)
    initial = {"pixel_protocol": protocol_metadata(CORNER), "manifest_sha256": scene.manifest_sha256}
    bind_training_pixel_protocol(scene, manifest, path, config, initial)
    changed = copy.deepcopy(manifest)
    changed["note"] = "input changed in place"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="manifest SHA"):
        bind_training_pixel_protocol(scene, changed, path, config, initial)


def test_teacher_context_categorical_grid_is_versioned():
    image = np.arange(4*8*3, dtype=np.uint8).reshape(4, 8, 3)
    mask = (np.arange(32).reshape(4, 8) % 5).astype(np.uint8)
    valid = np.ones((4, 8), dtype=bool)
    outputs = {}
    for protocol in (LEGACY, CORNER):
        rgb, target, keep = aligned_context_frame(image, mask, valid, 2,
            np.random.default_rng(42), augment=False, patch_size=1, pixel_protocol=protocol)
        np.testing.assert_array_equal(target.numpy(), resize_discrete_numpy(mask, (4, 2), protocol))
        assert keep.all()
        outputs[protocol] = rgb
    np.testing.assert_array_equal(outputs[LEGACY].numpy(), outputs[CORNER].numpy())
    _, legacy_target, _ = aligned_context_frame(image, mask, valid, 2,
        np.random.default_rng(42), augment=False, patch_size=1)
    np.testing.assert_array_equal(legacy_target.numpy(), cv2.resize(mask, (4, 2), interpolation=cv2.INTER_NEAREST))


@pytest.mark.parametrize("protocol", [LEGACY, CORNER])
def test_checkpoint_roundtrip_keeps_declared_pixel_grid_and_manifest_hash(tmp_path, monkeypatch, protocol):
    from bridge_rgs.model import GaussianScene
    from bridge_rgs.train import checkpoint, load_scene

    points = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 2]], np.float32)
    scene = GaussianScene(points, np.full((4, 3), .5, np.float32), feature_dim=8, sh_degree=1)
    manifest, manifest_path = write_manifest(tmp_path, protocol)
    config = {}
    bind_training_pixel_protocol(scene, manifest, manifest_path, config)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.tensor([0], dtype=torch.uint8))
    state = checkpoint(scene, {}, config, 0, torch.eye(4)[None], {})
    path = tmp_path / "new.pt"
    torch.save(state, path)
    loaded, reloaded = load_scene(path, device="cpu")
    assert loaded.pixel_protocol == protocol
    assert loaded.manifest_sha256 == scene.manifest_sha256
    assert reloaded["pixel_protocol"] == protocol_metadata(protocol)
    for key, value in scene.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])

    state.pop("pixel_protocol")
    state.pop("manifest_sha256")
    torch.save(state, path)
    historical, _ = load_scene(path, device="cpu")
    assert historical.pixel_protocol == LEGACY and historical.manifest_sha256 is None
