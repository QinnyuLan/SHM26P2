"""CPU preflight contracts for the fixed appearance experiment."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location("appearance_runner", Path(__file__).parents[1] / "scripts/run_raw_grid_appearance.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def manifest():
    return {"pixel_protocol": "colmap_corner_v2", "views": [
        {"name": f"{i:03}.png", "split": "train", "w2c": np.eye(4).tolist(),
         "w2c_original": np.eye(4).tolist(), "image_path": "/never/decode/rgb.png",
         "source_image_path": "/never/decode/raw.jpg", "valid_path": "/never/decode/valid.png",
         "mask_path": "/never/decode/label.png"} for i in range(350)]}


def test_view_population_is_metadata_only_and_all_train(monkeypatch):
    value = manifest()
    value["views"].append({"name": "val.png", "split": "val"})
    monkeypatch.setattr(runner.cv2, "imread", lambda *a, **k: pytest.fail("Must not decode images/GT"))
    assert len(runner.fixed_training_views(value)) == 350


@pytest.mark.parametrize("mutation", ["camera", "duplicate", "count", "profile", "missing_rgb"])
def test_rejects_camera_population_or_profile_change(mutation):
    value = manifest()
    if mutation == "camera":
        value["views"][0]["w2c"][0][3] = .1
    elif mutation == "duplicate":
        value["views"][0]["name"] = value["views"][1]["name"]
    elif mutation == "count":
        value["views"].pop()
    elif mutation == "profile":
        value["pixel_protocol"] = "legacy_mixed_v1"
    else:
        value["views"][0]["source_image_path"] = None
    with pytest.raises(ValueError):
        runner.fixed_training_views(value)


def test_shuffle_is_fixed_balanced_and_independent_of_global_rng():
    first = runner.training_order(350, 3000, 42)
    runner.random.seed(123)
    for _ in range(200):
        runner.random.random()
    assert first == runner.training_order(350, 3000, 42)
    assert first != runner.training_order(350, 3000, 43)
    assert len(first) == 3000
    for start in range(0, 2800, 350):
        assert sorted(first[start:start+350]) == list(range(350))


def test_checkpoint_camera_order_excludes_val_and_preserves_manifest_order():
    value = manifest()
    value["views"].reverse()
    for i, view in enumerate(value["views"]):
        view["w2c_original"][0][3] = i
    value["views"] += [{"split": "val", "name": f"val{i}", "w2c_original": np.eye(4).tolist()}
                       for i in range(50)]
    poses = runner.original_training_poses(value)
    assert poses.shape == (350, 4, 4)
    assert torch.equal(poses[:, 0, 3], torch.arange(350, dtype=torch.float32))


def test_hash_binding_detects_input_and_frozen_source_changes(tmp_path):
    source = tmp_path / "source.py"
    data = tmp_path / "image.dat"
    source.write_text("source")
    data.write_text("data")
    plan = {"snapshot": str(tmp_path), "input_hashes": {str(data): runner.digest(data)},
            "source_hashes": {source.name: runner.digest(source)}}
    runner.bind_inputs(plan)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Frozen source"):
        runner.bind_inputs(plan)
    source.write_text("source")
    data.write_text("changed")
    with pytest.raises(ValueError, match="Input changed"):
        runner.bind_inputs(plan)


def test_tensor_hash_covers_dtype_shape_and_values():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert runner.tensor_hash(a) == runner.tensor_hash(a.clone())
    assert runner.tensor_hash(a) != runner.tensor_hash(a.reshape(3, 2))
    assert runner.tensor_hash(a) != runner.tensor_hash(a.double())
    assert runner.tensor_hash(a) != runner.tensor_hash(a+1)


def test_only_explicit_graphics_clients_allowed(monkeypatch):
    class Result:
        stdout = "<nvidia_smi_log><gpu><processes><process_info><pid>4</pid><type>G</type></process_info></processes></gpu></nvidia_smi_log>"
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: Result())
    assert runner.gpu_idle()["compute_clients"] == 0
    original = Result.stdout
    for client in ("C", "C+G", "unknown"):
        Result.stdout = original.replace("<type>G</type>", f"<type>{client}</type>")
        with pytest.raises(ValueError, match="occupied"):
            runner.gpu_idle()


def test_fixed_plan_is_json_safe_and_only_appearance_trainable():
    value = copy.deepcopy(runner.SPEC)
    assert json.loads(json.dumps(value, allow_nan=False)) == value
    assert set(value["lr"]) == set(value["trainable"]) == set(runner.ALLOWED)
    assert value["steps"] == 3000 and value["checkpoint_optimizer_saved"] is False


def test_unknown_gpu_process_inventory_rejected(monkeypatch):
    class Result:
        stdout = "<nvidia_smi_log><gpu><processes>N/A</processes></gpu></nvidia_smi_log>"
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(ValueError, match="determine"):
        runner.gpu_idle()
