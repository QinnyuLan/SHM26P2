"""CPU contracts for the bounded capacity measurement; never loads real weights."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bridge_rgs import teacher
from bridge_rgs.coordinates import CORNER

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/preflight_dinov3_capacity.py"
spec = importlib.util.spec_from_file_location("dinov3_capacity_preflight", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def model_files(tmp_path, arm):
    directory = tmp_path / arm
    directory.mkdir()
    config = {"model_type": "dinov3_vit", "patch_size": 16}
    (directory / "config.json").write_text(json.dumps(config))
    if arm == "hplus":
        (directory / "model.safetensors").write_bytes(b"fake single weights")
    else:
        (directory / "one.safetensors").write_bytes(b"fake first shard")
        (directory / "two.safetensors").write_bytes(b"fake second shard")
        (directory / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}))
    hashes = {p.name: preflight.digest(p) for p in directory.iterdir()}
    receipt = {"provider": "ModelScope", **preflight.MODELS[arm], "files_sha256": hashes}
    if arm == "7b":
        source = {**preflight.MODELS[arm], "files": {
            name: {"sha256": sha, "size": (directory / name).stat().st_size}
            for name, sha in hashes.items()}}
        source_path = directory / "download_source_manifest.json"
        source_path.write_text(json.dumps(source))
        receipt.update(status="completed", model_dir=str(directory.resolve()),
                       source_manifest_sha256=preflight.digest(source_path))
    (directory / "download_provenance.json").write_text(json.dumps(receipt))
    return directory, receipt


def save_receipt(directory, receipt):
    (directory / "download_provenance.json").write_text(json.dumps(receipt))


def test_historical_hplus_receipt_remains_supported(tmp_path):
    directory, receipt = model_files(tmp_path, "hplus")
    assert "status" not in receipt
    bound = preflight.bind_model(directory, "hplus")
    assert len(bound["files_sha256"]) == 3
    assert bound["configuration"]["patch_size"] == 16


def test_completed_sharded_receipt_binds_index_source_and_all_shards(tmp_path):
    directory, _ = model_files(tmp_path, "7b")
    bound = preflight.bind_model(directory, "7b")
    assert {Path(p).name for p in bound["files_sha256"]} == {
        "config.json", "one.safetensors", "two.safetensors", "model.safetensors.index.json",
        "download_source_manifest.json", "download_provenance.json"}


@pytest.mark.parametrize("change", [{"status": "running"}, {"model_dir": "/elsewhere"},
                                     {"provider": "unknown"}, {"revision": "wrong"}])
def test_unfinished_or_wrong_source_rejected_before_any_weight_hash(tmp_path, monkeypatch, change):
    directory, receipt = model_files(tmp_path, "7b")
    receipt.update(change)
    save_receipt(directory, receipt)
    monkeypatch.setattr(preflight, "digest", lambda _: pytest.fail("Must not hash unfinished files"))
    with pytest.raises(ValueError):
        preflight.bind_model(directory, "7b")


@pytest.mark.parametrize("filename", ["../weights", "/tmp/weights", "x.part", "x.tmp", ".", ""])
def test_unsafe_or_partial_model_filename_rejected(tmp_path, filename):
    with pytest.raises(ValueError):
        preflight.safe_model_files(tmp_path, {filename: "0" * 64})


def test_mutated_model_file_or_source_manifest_rejected(tmp_path):
    directory, _ = model_files(tmp_path, "7b")
    (directory / "one.safetensors").write_bytes(b"changed firstone")
    with pytest.raises(ValueError):
        preflight.bind_model(directory, "7b")
    (directory / "download_source_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="source manifest"):
        preflight.bind_model(directory, "7b")


def test_index_cannot_reference_unbound_shard(tmp_path):
    directory, receipt = model_files(tmp_path, "7b")
    index_path = directory / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {"a": "one.safetensors", "b": "other.safetensors"}}))
    receipt["files_sha256"][index_path.name] = preflight.digest(index_path)
    source_path = directory / "download_source_manifest.json"
    source = json.loads(source_path.read_text())
    source["files"][index_path.name] = {"size": index_path.stat().st_size, "sha256": preflight.digest(index_path)}
    source_path.write_text(json.dumps(source))
    receipt["source_manifest_sha256"] = preflight.digest(source_path)
    save_receipt(directory, receipt)
    with pytest.raises(ValueError, match="undeclared"):
        preflight.bind_model(directory, "7b")


def manifest():
    return {"pixel_protocol": CORNER, "views": [{"name": "002.png", "split": "train",
             "mask_path": "mask.png", "width": 1320, "height": 989}]}


@pytest.mark.parametrize("field,value", [("split", "val"), ("mask_path", None), ("width", 1319)])
def test_selection_rejects_nonfixed_train_view(field, value):
    source = manifest()
    source["views"][0][field] = value
    with pytest.raises(ValueError):
        preflight.selected_view(source)


def test_selection_rejects_legacy_or_duplicate_view():
    source = manifest()
    assert preflight.selected_view(source)["name"] == "002.png"
    source.pop("pixel_protocol")
    with pytest.raises(ValueError, match="corner_v2"):
        preflight.selected_view(source)
    source = manifest()
    source["views"] += copy.deepcopy(source["views"])
    with pytest.raises(ValueError):
        preflight.selected_view(source)


def test_crop_context_keep_fixed_coordinates_labels_and_ignored_padding(monkeypatch):
    y, x = np.indices((989, 1320))
    image = np.stack((x % 256, y % 256, (x+y) % 256), -1).astype(np.uint8)
    mask = ((x // 150) % 5).astype(np.uint8)
    valid = np.ones_like(mask, dtype=bool)
    valid[:4] = False
    mask[500:510, 600:610] = 255  # Valid image support may itself lack a class label.
    original_mask = mask.copy()
    monkeypatch.setattr(teacher, "ViewReader", lambda _: SimpleNamespace(read=lambda v: (image, mask, valid)))
    actual_image, (crop, context), metadata = preflight.data_for_steps({})
    assert actual_image is image and metadata["crop_xy"] == [276, 110]
    torch.testing.assert_close(crop[0][:, 0, 0], torch.tensor(image[110, 276] / 255, dtype=torch.float32))
    assert crop[0].shape == (3, 768, 768) and context[0].shape == (3, 768, 1040)
    assert not context[2][:, 1025:].any()
    assert torch.all(context[1][:, 1025:] == 255)
    assert set(context[1].unique().tolist()) == {0, 1, 2, 3, 4, 255}
    assert np.array_equal(mask, original_mask)


def test_hash_handles_bfloat16_and_scalar_buffers_without_mutation():
    model = torch.nn.Linear(3, 2).to(torch.bfloat16)
    model.register_buffer("counter", torch.tensor(4))
    original = {k: v.clone() for k, v in model.state_dict().items()}
    before = preflight.tensor_state_hash(model)
    assert preflight.tensor_state_hash(model) == before
    assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert preflight.tensor_state_hash(model) != before


def test_fixed_two_steps_and_complete_fourteen_forward_protocol():
    rows = len(teacher._tile_starts(989, 768, 512))
    columns = len(teacher._tile_starts(1320, 768, 512))
    assert 2 * rows * columns + 2 == preflight.SPEC["inference"]["expected_complete_forward_calls"] == 14
    assert preflight.SPEC["supervised_steps"] == 2
    assert preflight.SPEC["optional_native_inference"]["forward_calls"] == 1
    assert preflight.SPEC["per_arm_process_seconds"] == 600
    assert not preflight.SPEC["checkpoint_or_probability_cache_written"]
    with pytest.raises(ValueError, match="locked"):
        preflight.verify({"status": "pending_completed_7b_download", "specification": preflight.SPEC})


def test_saved_forward_metadata_survives_native_hook_list_reuse():
    captured = [{"last_shape": [1, 2309, 1280], "dtype": "torch.bfloat16"} for _ in range(14)]
    recorded = preflight.snapshot_forward_calls(captured)
    captured[0]["last_shape"][1] = 5151
    captured.clear()
    captured.append({"last_shape": [1, 5151, 1280]})
    assert recorded["forward_calls"] == len(recorded["calls"]) == 14
    assert all(call["last_shape"] == [1, 2309, 1280] for call in recorded["calls"])


@pytest.mark.parametrize("processes", ["N/A", "Not Supported", "<process_info><type>C</type></process_info>"])
def test_gpu_idle_check_fails_closed(monkeypatch, processes):
    xml = f"<nvidia_smi_log><gpu><processes>{processes}</processes></gpu></nvidia_smi_log>"
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=xml))
    with pytest.raises(ValueError):
        preflight.gpu_idle()


def test_gpu_idle_allows_desktop_graphics_only(monkeypatch):
    xml = "<nvidia_smi_log><gpu><processes><process_info><type>G</type></process_info></processes></gpu></nvidia_smi_log>"
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=xml))
    preflight.gpu_idle()
