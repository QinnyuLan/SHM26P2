"""Late-iterate semantic parameter averaging on exactly unchanged geometry.

This is an engineering control based on parameter averaging, not a new method.
It produces one model for camera-only inference, without an inference ensemble.
Optimizer moments from an individual iterate cannot resume the averaged model.
"""
from __future__ import annotations

import copy

import torch


def is_semantic_parameter(name):
    return (name == "splats.sem_features"
            or name.startswith(("semantic_decoder.", "refiner.")))


def average_semantic_states(states):
    if len(states) < 2:
        raise ValueError("At least two checkpoints are required")
    latest = states[-1]
    if len({state["step"] for state in states}) != len(states):
        raise ValueError("Checkpoint steps must be distinct")
    if [state["step"] for state in states] != sorted(state["step"] for state in states):
        raise ValueError("Checkpoints must be ordered by training step")
    metadata = ("format_version", "feature_dim", "sh_degree", "scene_scale", "refiner_config",
                "pixel_protocol", "manifest_sha256")
    reference = states[0]
    for state in states:
        if any(state.get(key) != reference.get(key) for key in metadata):
            raise ValueError("Checkpoint architecture/scene metadata differ")
        if state["config"] != reference["config"]:
            raise ValueError("Averaging requires checkpoints from the same training stage")
        if not torch.equal(state["training_cameras"], reference["training_cameras"]):
            raise ValueError("Training cameras changed between checkpoints")
        if state["model"].keys() != reference["model"].keys():
            raise ValueError("Checkpoint model schemas differ")
    model = {}
    for key, first in reference["model"].items():
        values = [state["model"][key].detach().cpu() for state in states]
        if any(value.shape != first.shape or value.dtype != first.dtype for value in values):
            raise ValueError(f"Parameter shape/dtype changed: {key}")
        if any(not torch.isfinite(value).all() for value in values):
            raise ValueError(f"Nonfinite parameter: {key}")
        if is_semantic_parameter(key):
            if not first.is_floating_point():
                raise ValueError(f"Cannot average nonfloating semantic parameter: {key}")
            result = torch.zeros_like(first, device="cpu", dtype=torch.float64)
            for value in values:
                result.add_(value.double(), alpha=1 / len(values))
            model[key] = result.to(first.dtype)
        else:
            if any(not torch.equal(value, values[0]) for value in values[1:]):
                raise ValueError(f"Geometry/RGB/semantic evidence changed: {key}")
            model[key] = values[0].clone()
    result = {key: copy.deepcopy(latest[key]) for key in metadata if key in latest}
    result.update(model=model, config=copy.deepcopy(latest["config"]), step=latest["step"],
                  training_cameras=latest["training_cameras"].detach().cpu().clone(),
                  checkpoint_kind="averaged_semantics_inference_or_warmstart",
                  averaging={"steps": [state["step"] for state in states],
                             "weights": [1 / len(states)] * len(states),
                             "geometry_rgb_unchanged": True,
                             "optimizer_state_available": False})
    return result
