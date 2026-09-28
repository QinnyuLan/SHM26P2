"""CPU contracts for the training adapter, not CUDA MCMC mathematics."""
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bridge_rgs import mcmc_reference, train
from bridge_rgs.model import GaussianScene


def recipe():
    path = Path(__file__).resolve().parents[1] / "configs/rgb_mcmc_reference_500k.yaml"
    return yaml.safe_load(path.read_text())


def scene():
    points = np.array([[-.2, -.2, 3], [.2, -.2, 3], [-.2, .2, 3], [.2, .2, 3]], np.float32)
    return GaussianScene(points, np.full((4, 3), .8, np.float32))


def cpu_checkpoint(monkeypatch):
    model = scene()
    optimizers = model.optimizers()
    policy = mcmc_reference.validate_reference_training(recipe())
    runtime = mcmc_reference.initialize_reference(model, optimizers, policy)
    model.mcmc_reference_config = asdict(policy)
    model.mcmc_reference_state = runtime.state_dict()
    model.mcmc_reference_runtime = runtime
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.zeros(0, dtype=torch.uint8))
    value = train.checkpoint(model, optimizers, recipe(), 0, torch.eye(4)[None], {})
    return model, value


@pytest.mark.parametrize("change", [
    {"densification": "none"}, {"steps": 31000}, {"seed": 5}, {"view_sampler_seed": 5},
    {"view_sampling": "random"}, {"independent_view_rng": False},
    {"progressive_resolution": True}, {"train_scale": .75}, {"sh_interval": 500},
    {"initial_opacity": .1}, {"initial_scale_multiplier": 1.}, {"background_points": 1000},
    {"opacity_lr": .025}, {"init_points": "other.npz"},
])
def test_strict_resume_rejects_mode_rng_schedule_and_initialization_changes(change):
    old = recipe()
    with pytest.raises(ValueError, match="MCMC"):
        train.validate_mcmc_resume(old, {**old, **change})
    if "densification" in change:
        with pytest.raises(ValueError, match="MCMC"):
            train.validate_mcmc_resume({**old, **change}, old)


def test_resume_allows_only_nonoptimization_output_changes_and_legacy_is_untouched():
    config = recipe()
    train.validate_mcmc_resume(config, {**config, "output": "elsewhere", "log_every": 50})
    train.validate_mcmc_resume({"densification": "hybrid"}, {"densification": "none"})
    mcmc_reference.validate_reference_training(config)


def test_checkpoint_runtime_is_current_and_frozen_load_never_runs_strategy(tmp_path, monkeypatch):
    model, value = cpu_checkpoint(monkeypatch)
    assert value["mcmc_reference_state"]["counters"]["last_step"] == 0
    source = tmp_path / "rgb.pt"
    torch.save(value, source)
    fail = lambda *args, **kwargs: pytest.fail("Inference/frozen inheritance ran the strategy")
    monkeypatch.setattr(mcmc_reference, "initialize_reference", fail)
    monkeypatch.setattr(mcmc_reference, "reference_post_step", fail)
    loaded, _ = train.load_scene(source, device="cpu")
    assert not hasattr(loaded, "mcmc_reference_runtime")
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])
    assert loaded.mcmc_reference_config == value["mcmc_reference_config"]
    frozen_config = {**recipe(), "densification": "none", "freeze_geometry": True, "freeze_rgb": True}
    inherited = train.checkpoint(loaded, loaded.optimizers(), frozen_config, 7,
                                torch.eye(4)[None], {})
    # Stage 7 is not the old strategy's step 0; frozen metadata must stay unchanged.
    assert inherited["mcmc_reference_state"]["counters"]["last_step"] == 0
    destination = tmp_path / "semantic.pt"
    torch.save(inherited, destination)
    final, _ = train.load_scene(destination, device="cpu")
    assert final.mcmc_reference_state["counters"] == loaded.mcmc_reference_state["counters"]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("mutation", ["missing", "policy", "step", "count", "scale", "binoms"])
def test_corrupt_strategy_metadata_is_rejected_before_inference(monkeypatch, mutation):
    model, state = cpu_checkpoint(monkeypatch)
    if mutation == "missing":
        state.pop("mcmc_reference_state")
    elif mutation == "policy":
        state["config"]["mcmc_reference"]["noise_lr"] = 1.
    elif mutation == "step":
        state["step"] = 9
    elif mutation == "count":
        state["mcmc_reference_state"]["counters"]["current_gaussians"] = 5
    elif mutation == "scale":
        state["mcmc_reference_state"]["scene_scale"] = 8.
    else:
        state["mcmc_reference_state"]["binoms"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="MCMC"):
        train.restore_mcmc_metadata(model, state)


def test_old_checkpoint_has_no_new_metadata_or_strategy_side_effect(monkeypatch):
    model = scene()
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.zeros(0, dtype=torch.uint8))
    value = train.checkpoint(model, model.optimizers(), {"densification": "rgb140_mixed"},
                             0, torch.eye(4)[None], {})
    assert "mcmc_reference_config" not in value and "mcmc_reference_state" not in value
    train.restore_mcmc_metadata(model, value)
    assert not hasattr(model, "mcmc_reference_config")


@pytest.mark.parametrize("ignore_masks", [True, False])
def test_rgb_reference_cache_strips_mask_without_mutating_manifest(monkeypatch, ignore_masks):
    original = {"name": "002.JPG", "mask_path": "official-mask.png", "image_path": "rgb.png"}
    before = deepcopy(original)
    received = []
    monkeypatch.setattr(train, "load_view", lambda view, **kwargs: received.append(dict(view)) or {})
    train.ImageCache(ignore_masks=ignore_masks).get(original, 1.)
    assert received[0]["mask_path"] == (None if ignore_masks else "official-mask.png")
    assert original == before


def test_actual_regularization_log_and_counters_are_strict_json_serializable(monkeypatch):
    model, state = cpu_checkpoint(monkeypatch)
    _, stats = mcmc_reference.reference_regularization(model, mcmc_reference.MCMCReferenceConfig())
    stats.update({f"mcmc_{key}": value for key, value
                  in state["mcmc_reference_state"]["counters"].items()})
    stats.update(mcmc_topology_changed=False, mcmc_training_view="002.JPG")
    with pytest.raises(TypeError):
        json.dumps(stats, allow_nan=False)
    train.mcmc_log_scalars(stats)
    assert json.loads(json.dumps(stats, allow_nan=False)) == stats


def test_nonfinite_gradient_or_parameter_fails_without_silent_correction():
    model = scene()
    parameter = model.splats["sh0"]
    parameter.grad = torch.zeros_like(parameter)
    train.check_mcmc_finite(model, 1, gradients=True)
    parameter.grad[0, 0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="gradient.*sh0"):
        train.check_mcmc_finite(model, 100, gradients=True)
    with torch.no_grad():
        parameter[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="parameter.*sh0"):
        train.check_mcmc_finite(model, 100)
