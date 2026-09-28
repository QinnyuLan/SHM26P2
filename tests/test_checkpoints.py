"""CPU contracts for the three-tensor appearance storage format."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bridge_rgs import checkpoints as cp
from bridge_rgs.coordinates import CORNER, LEGACY, protocol_metadata


@pytest.fixture
def example(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pixel_protocol": protocol_metadata(CORNER), "views": []}))
    base = {"format_version": 1, "config": {"manifest": str(manifest), "seed": 42,
                                          "pixel_protocol": CORNER},
            "step": 8000, "scene_scale": 2., "feature_dim": 16, "sh_degree": 1,
            "refiner_config": {"type": "legacy"}, "training_cameras": torch.eye(4)[None],
            "pixel_protocol": protocol_metadata(CORNER), "manifest_sha256": cp.file_sha256(manifest),
            "model": {"splats.means": torch.arange(12, dtype=torch.float32).reshape(4, 3),
                      "splats.sh0": torch.zeros(4, 1, 3), "splats.sh_rest": torch.ones(4, 3, 3),
                      "background_logits": torch.ones(3), "splats.opacity_logits": torch.ones(4),
                      "refiner.test.weight": torch.eye(3)},
            "optimizers": {"stale": 1}, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.zeros(3), "numpy_rng": np.random.get_state(),
            "density_state": {"stale": 2}, "stats": {"stale": 3},
            "arbitrary_future_resume_state": {"stale": 4}}
    source = tmp_path / "base.pt"
    torch.save(base, source)
    config = dict(base["config"], parameter_scope="appearance_only", output="new-run")
    return {"path": tmp_path / "delta.pt", "base_checkpoint": source,
            "base_sha256": cp.file_sha256(source), "manifest_path": manifest,
            "model": {k: base["model"][k] + .25 for k in cp.APPEARANCE_KEYS},
            "experiment": "appearance_control", "config": config, "step": 2000}


def rewrite_delta(example, mutate):
    cp.save_appearance_delta(**example)
    value = torch.load(example["path"], weights_only=False)
    mutate(value)
    torch.save(value, example["path"])


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_full_checkpoint_exact_historical_compatibility(example):
    reference = torch.load(example["base_checkpoint"], weights_only=False)
    assert_nested_equal(cp.load_checkpoint(example["base_checkpoint"]), reference)
    # Full historical checkpoints need not contain today's manifest/profile tags.
    del reference["pixel_protocol"], reference["manifest_sha256"]
    torch.save(reference, example["base_checkpoint"])
    assert_nested_equal(cp.load_checkpoint(example["base_checkpoint"]), reference)


def test_only_three_tensors_saved_and_all_other_scene_fields_inherited(example):
    cp.save_appearance_delta(**example)
    stored = torch.load(example["path"], weights_only=False)
    assert set(stored["model"]) == cp.APPEARANCE_KEYS
    assert "training_cameras" not in stored and "optimizers" not in stored
    assert stored["base_checkpoint"]["path"] == str(example["base_checkpoint"].resolve())
    loaded = cp.load_checkpoint(example["path"])
    base = torch.load(example["base_checkpoint"], weights_only=False)
    for key in base["model"]:
        expected = example["model"][key] if key in cp.APPEARANCE_KEYS else base["model"][key]
        assert torch.equal(loaded["model"][key], expected)
    assert_nested_equal(loaded["training_cameras"], base["training_cameras"])
    assert loaded["config"] == base["config"] and loaded["step"] == 8000
    assert loaded["appearance_delta"]["config"] == example["config"]
    assert loaded["appearance_delta"]["step"] == 2000
    assert loaded["checkpoint_kind"] == cp.DELTA_CHECKPOINT_KIND
    assert loaded["dependency_provenance"]["delta"]["sha256"] == cp.file_sha256(example["path"])
    for stale in ("optimizers", "torch_rng", "cuda_rng", "numpy_rng", "density_state", "stats",
                  "arbitrary_future_resume_state"):
        assert stale not in loaded


def test_new_trainer_state_is_separate_and_never_resumable(example):
    trainer = {"optimizers": {"sh0": {"state": {0: {"exp_avg": torch.ones(4, 1, 3)}}}},
               "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state()}
    cp.save_appearance_delta(**example, trainer_state=trainer)
    state = cp.load_checkpoint(example["path"])
    assert_nested_equal(state["appearance_delta"]["trainer_state"], trainer)
    assert state["appearance_delta"]["ordinary_resume_allowed"] is False
    assert "optimizers" not in state and "torch_rng" not in state


@pytest.mark.parametrize("change", ["extra", "missing", "shape", "dtype", "nan", "inf", "object"])
def test_invalid_model_rejected_on_save_and_load(example, change):
    def corrupt(value):
        model = value["model"]
        if change == "extra":
            model["splats.opacity_logits"] = torch.ones(4)
        elif change == "missing":
            del model["splats.sh0"]
        elif change == "shape":
            model["splats.sh0"] = torch.zeros(5, 1, 3)
        elif change == "dtype":
            model["splats.sh0"] = model["splats.sh0"].double()
        elif change == "object":
            model["splats.sh0"] = [1, 2, 3]
        else:
            model["splats.sh0"][0, 0, 0] = float(change)
    bad = copy.deepcopy(example)
    corrupt(bad)
    with pytest.raises(ValueError):
        cp.save_appearance_delta(**bad)
    assert not example["path"].exists()
    rewrite_delta(example, corrupt)
    with pytest.raises(ValueError):
        cp.load_checkpoint(example["path"])


@pytest.mark.parametrize("field,new", [
    ("pixel_protocol", protocol_metadata(LEGACY)),
    ("pixel_protocol", None),
    ("checkpoint_format", "unknown"), ("step", -1), ("step", True), ("experiment", ""),
    ("trainer_state", {"model": {}}), ("extra_key", True),
])
def test_corrupt_delta_metadata_is_rejected(example, field, new):
    rewrite_delta(example, lambda v: v.update({field: new}))
    with pytest.raises(ValueError):
        cp.load_checkpoint(example["path"])


@pytest.mark.parametrize("field,new", [
    ("pixel_protocol", LEGACY), ("parameter_scope", "all"), ("manifest", "other.json"),
    ("lr", float("nan")),
])
def test_delta_config_must_preserve_manifest_profile_and_appearance_scope(example, field, new):
    example["config"][field] = new
    with pytest.raises(ValueError):
        cp.save_appearance_delta(**example)


def test_missing_base_profile_means_legacy_not_manifest_upgrade(example):
    base = torch.load(example["base_checkpoint"], weights_only=False)
    del base["pixel_protocol"], base["config"]["pixel_protocol"]
    torch.save(base, example["base_checkpoint"])
    example["base_sha256"] = cp.file_sha256(example["base_checkpoint"])
    with pytest.raises(ValueError, match="protocol mismatch"):
        cp.save_appearance_delta(**example)


@pytest.mark.parametrize("which", ["base_checkpoint", "manifest_path"])
def test_changed_dependency_rejected(example, which):
    cp.save_appearance_delta(**example)
    with example[which].open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="SHA256"):
        cp.load_checkpoint(example["path"])


def test_wrong_selected_sha_and_relative_stored_path_rejected(example):
    wrong = dict(example, base_sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        cp.save_appearance_delta(**wrong)
    rewrite_delta(example, lambda v: v["base_checkpoint"].update(path="base.pt"))
    with pytest.raises(ValueError, match="absolute"):
        cp.load_checkpoint(example["path"])


def test_delta_chain_and_materialized_delta_as_base_forbidden(example):
    cp.save_appearance_delta(**example)
    next_args = dict(example, path=example["path"].with_name("next.pt"),
                     base_checkpoint=example["path"], base_sha256=cp.file_sha256(example["path"]))
    with pytest.raises(ValueError, match="chains forbidden"):
        cp.save_appearance_delta(**next_args)
    state = cp.load_checkpoint(example["path"])
    materialized = example["path"].with_name("materialized.pt")
    torch.save(state, materialized)
    next_args.update(base_checkpoint=materialized, base_sha256=cp.file_sha256(materialized))
    with pytest.raises(ValueError, match="Materialized"):
        cp.save_appearance_delta(**next_args)


def test_no_overwrite_even_concurrent_publish_and_temporary_cleanup(example, monkeypatch):
    original = cp.os.link
    def racing_link(src, dst):
        Path(dst).write_bytes(b"concurrent result")
        original(src, dst)
    monkeypatch.setattr(cp.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        cp.save_appearance_delta(**example)
    assert example["path"].read_bytes() == b"concurrent result"
    assert not list(example["path"].parent.glob(".*.tmp"))
    with pytest.raises(FileExistsError):
        cp.save_appearance_delta(**example)


def test_detect_base_mutation_during_load(example, monkeypatch):
    cp.save_appearance_delta(**example)
    original = cp.torch.load
    def mutate_after_load(path, **kwargs):
        value = original(path, **kwargs)
        if Path(path) == example["base_checkpoint"]:
            with Path(path).open("ab") as handle:
                handle.write(b"changed")
        return value
    monkeypatch.setattr(cp.torch, "load", mutate_after_load)
    with pytest.raises(ValueError, match="Base checkpoint SHA256"):
        cp.load_checkpoint(example["path"])


def test_failed_serialization_never_publishes_or_leaves_temporary(example, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(cp.torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        cp.save_appearance_delta(**example)
    assert not example["path"].exists()
    assert not list(example["path"].parent.glob(".*.tmp"))


def test_tensor_views_do_not_serialize_unrelated_backing_storage(example):
    oversized = torch.zeros(1_000_000)
    example["model"]["background_logits"] = oversized[:3]
    cp.save_appearance_delta(**example)
    assert example["path"].stat().st_size < 20_000
    assert torch.equal(cp.load_checkpoint(example["path"])["model"]["background_logits"],
                       torch.zeros(3))


def test_legacy_full_base_can_bind_explicit_legacy_delta_without_upgrading(example):
    base = torch.load(example["base_checkpoint"], weights_only=False)
    del base["pixel_protocol"], base["config"]["pixel_protocol"]
    example["manifest_path"].write_text(json.dumps({"views": []}))
    base["manifest_sha256"] = cp.file_sha256(example["manifest_path"])
    torch.save(base, example["base_checkpoint"])
    example["base_sha256"] = cp.file_sha256(example["base_checkpoint"])
    example["config"]["pixel_protocol"] = LEGACY
    cp.save_appearance_delta(**example)
    value = cp.load_checkpoint(example["path"])
    assert "pixel_protocol" not in value
    assert value["dependency_provenance"]["pixel_protocol"] == protocol_metadata(LEGACY)


def test_missing_base_manifest_sha_rejected_for_new_delta(example):
    base = torch.load(example["base_checkpoint"], weights_only=False)
    del base["manifest_sha256"]
    torch.save(base, example["base_checkpoint"])
    example["base_sha256"] = cp.file_sha256(example["base_checkpoint"])
    with pytest.raises(ValueError, match="provenance schema"):
        cp.save_appearance_delta(**example)


def test_materialized_state_loads_exactly_into_scene_on_cpu(example):
    from bridge_rgs.model import GaussianScene
    scene = GaussianScene(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
                          np.full((4, 3), .5), feature_dim=16, sh_degree=1)
    base = torch.load(example["base_checkpoint"], weights_only=False)
    base["model"] = scene.state_dict()
    base["refiner_config"] = scene.refiner_config
    torch.save(base, example["base_checkpoint"])
    example["base_sha256"] = cp.file_sha256(example["base_checkpoint"])
    example["model"] = {k: base["model"][k] + .25 for k in cp.APPEARANCE_KEYS}
    cp.save_appearance_delta(**example)
    state = cp.load_checkpoint(example["path"])
    scene.load_state_dict(state["model"], strict=True)
    for name, tensor in scene.state_dict().items():
        assert torch.equal(tensor, state["model"][name])
