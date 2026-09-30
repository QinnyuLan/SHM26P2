"""Strict, non-resumable appearance deltas over one SHA-bound full scene.

Full checkpoints retain their historical torch.load behavior.  This storage
format changes exactly three appearance tensors; it is not a general patcher.
The base must remain available.  New-stage metadata/optional trainer state are
namespaced, never mistaken for the base stage's optimizer or RNG state.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

from .coordinates import pixel_protocol, protocol_metadata, require_matching_protocol

APPEARANCE_DELTA_FORMAT = "bridge_rgs_appearance_delta_v1"
APPEARANCE_KEYS = frozenset({"splats.sh0", "splats.sh_rest", "background_logits"})
DELTA_CHECKPOINT_KIND = "appearance_delta_inference_or_warmstart"
_SCENE_FIELDS = {
    "format_version", "model", "config", "step", "scene_scale", "feature_dim",
    "sh_degree", "refiner_config", "training_cameras", "pixel_protocol", "manifest_sha256",
}
_DELTA_FIELDS = {
    "checkpoint_format", "base_checkpoint", "manifest", "pixel_protocol", "model",
    "experiment", "config", "step", "trainer_state",
}
_TRAINER_FIELDS = {"optimizers", "torch_rng", "cuda_rng", "numpy_rng", "sampler_state", "stats"}


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid SHA256 for {name}")
    return value


def _reference(value, name):
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"Invalid {name} dependency schema")
    if not isinstance(value["path"], str) or not Path(value["path"]).is_absolute():
        raise ValueError(f"{name} dependency path must be absolute")
    _sha(value["sha256"], name)
    return Path(value["path"])


def _unchanged(path, expected, name):
    if file_sha256(path) != expected:
        raise ValueError(f"{name} SHA256 mismatch or file changed during operation")


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _full_base(path, expected, map_location):
    _unchanged(path, expected, "Base checkpoint")
    base = torch.load(path, map_location=map_location, weights_only=False)
    _unchanged(path, expected, "Base checkpoint")
    if not isinstance(base, dict) or "checkpoint_format" in base:
        raise ValueError("Appearance delta base must be a full checkpoint; delta chains forbidden")
    if base.get("checkpoint_kind") == DELTA_CHECKPOINT_KIND or "appearance_delta" in base:
        raise ValueError("Materialized appearance deltas cannot be used as a base")
    required = {"model", "config", "step", "scene_scale", "feature_dim", "sh_degree",
                "training_cameras", "manifest_sha256"}
    if base.get("format_version") != 1 or not required <= base.keys():
        raise ValueError("Base checkpoint is missing the full scene/provenance schema")
    if not isinstance(base["model"], Mapping) or not isinstance(base["config"], dict):
        raise TypeError("Base checkpoint model/config must be dictionaries")
    _sha(base["manifest_sha256"], "base manifest")
    if "pixel_protocol" in base["config"]:
        require_matching_protocol(base["config"], base, "base configuration")
    return base


def _validate_model(model, base):
    if not isinstance(model, Mapping) or set(model) != APPEARANCE_KEYS:
        raise ValueError("Appearance model must contain exactly the three allowed appearance keys")
    n = base["model"].get("splats.means")
    if not isinstance(n, torch.Tensor) or n.ndim != 2 or n.shape[1] != 3:
        raise ValueError("Base checkpoint has invalid Gaussian means")
    degree = base["sh_degree"]
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise ValueError("Base checkpoint has invalid SH degree")
    shapes = {"splats.sh0": (len(n), 1, 3),
              "splats.sh_rest": (len(n), (degree + 1) ** 2 - 1, 3),
              "background_logits": (3,)}
    for name in APPEARANCE_KEYS:
        value, original = model[name], base["model"].get(name)
        if any(not isinstance(x, torch.Tensor) or not x.is_floating_point()
               or x.layout != torch.strided for x in (value, original)):
            raise ValueError(f"Appearance tensor must be a dense floating tensor: {name}")
        if (tuple(value.shape) != shapes[name] or value.shape != original.shape
                or value.dtype != original.dtype):
            raise ValueError(f"Appearance shape/dtype mismatch: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite appearance tensor: {name}")


def _validate_delta(delta, base):
    if set(delta) - _DELTA_FIELDS or _DELTA_FIELDS - {"trainer_state"} - set(delta):
        raise ValueError("Appearance delta schema has unexpected or missing fields")
    if delta["checkpoint_format"] != APPEARANCE_DELTA_FORMAT:
        raise ValueError("Unsupported checkpoint_format")
    if delta["pixel_protocol"] is None:
        raise ValueError("Delta requires an explicit pixel_protocol")
    _reference(delta["base_checkpoint"], "base checkpoint")
    manifest_path = _reference(delta["manifest"], "manifest")
    expected = delta["manifest"]["sha256"]
    if expected != base["manifest_sha256"]:
        raise ValueError("Delta manifest SHA256 differs from base")
    _unchanged(manifest_path, expected, "Manifest")
    manifest = json.loads(manifest_path.read_text())
    _unchanged(manifest_path, expected, "Manifest")
    require_matching_protocol(delta, base, "delta and base")
    require_matching_protocol(delta, manifest, "delta and manifest")
    if not isinstance(delta["experiment"], str) or not delta["experiment"].strip():
        raise ValueError("Delta experiment must be a nonempty string")
    if isinstance(delta["step"], bool) or not isinstance(delta["step"], int) or delta["step"] < 0:
        raise ValueError("Delta step must be a nonnegative integer")
    config = delta["config"]
    if not isinstance(config, dict) or config.get("parameter_scope") != "appearance_only":
        raise ValueError("Delta config must explicitly use appearance_only parameter_scope")
    if not config.get("manifest") or config["manifest"] != base["config"].get("manifest"):
        raise ValueError("Delta config manifest must match the base config manifest")
    if config.get("pixel_protocol") is None:
        raise ValueError("Delta config requires an explicit pixel_protocol")
    require_matching_protocol(config, delta, "delta configuration")
    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Delta config must be finite JSON metadata") from error
    trainer = delta.get("trainer_state")
    if trainer is not None and (not isinstance(trainer, dict) or set(trainer) - _TRAINER_FIELDS):
        raise ValueError("Independent trainer_state has unsupported fields")
    _validate_model(delta["model"], base)


def save_appearance_delta(path, *, base_checkpoint, base_sha256, model, manifest_path,
                          experiment, config, step, trainer_state=None):
    """Save a new file without overwrite, atomically, returning its dependencies.

    ``model`` must have exactly APPEARANCE_KEYS (not a full scene state_dict).
    ``base_sha256`` is mandatory: the caller binds its selected immutable base.
    The new stage's config/step/trainer_state are kept separate from base state.
    This function never reads model tensors from another delta or resumes Adam.
    """
    target = Path(path).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    base_path = Path(base_checkpoint).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True)
    expected = _sha(base_sha256, "base checkpoint")
    base = _full_base(base_path, expected, "cpu")
    _validate_model(model, base)
    if Path(base["config"].get("manifest", "")).resolve() != manifest_path:
        raise ValueError("Selected manifest path differs from the base configured manifest")
    delta = {"checkpoint_format": APPEARANCE_DELTA_FORMAT,
             "base_checkpoint": {"path": str(base_path), "sha256": expected},
             "manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
             "pixel_protocol": protocol_metadata(pixel_protocol(base)), "model": _cpu_copy(model),
             "experiment": experiment, "config": copy.deepcopy(config), "step": step}
    if trainer_state is not None:
        delta["trainer_state"] = _cpu_copy(trainer_state)
    _validate_delta(delta, base)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(delta, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _unchanged(base_path, expected, "Base checkpoint")
        _unchanged(manifest_path, delta["manifest"]["sha256"], "Manifest")
        # A hard-link publish is atomic and fails if a concurrent writer won.
        # os.replace would silently overwrite the completed result and is avoided.
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(target), "sha256": file_sha256(target),
            "base_checkpoint": delta["base_checkpoint"], "manifest": delta["manifest"]}


def load_checkpoint(path, map_location="cpu"):
    """Load a historical full checkpoint unchanged, or materialize a strict delta.

    Delta output has base config/step and camera/field tensors unchanged except
    the three appearance keys.  The new stage lives under ``appearance_delta``.
    No base optimizers/RNG/stats/density state survive; even optional delta trainer
    state is namespaced and cannot be consumed by the ordinary resume path.
    """
    path = Path(path)
    before = file_sha256(path)
    value = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(value, dict) or "checkpoint_format" not in value:
        return value
    if value["checkpoint_format"] != APPEARANCE_DELTA_FORMAT:
        raise ValueError("Unsupported checkpoint_format")
    _unchanged(path, before, "Delta checkpoint")
    base_path = _reference(value.get("base_checkpoint"), "base checkpoint")
    base = _full_base(base_path, value["base_checkpoint"]["sha256"], map_location)
    _validate_delta(value, base)
    result = {key: base[key] for key in _SCENE_FIELDS if key in base}
    result["model"] = dict(base["model"])
    result["model"].update(value["model"])
    result["checkpoint_kind"] = DELTA_CHECKPOINT_KIND
    result["appearance_delta"] = {key: value[key] for key in
                                  ("experiment", "config", "step", "trainer_state") if key in value}
    result["appearance_delta"]["ordinary_resume_allowed"] = False
    result["dependency_provenance"] = {
        "checkpoint_format": APPEARANCE_DELTA_FORMAT,
        "delta": {"path": str(path.resolve()), "sha256": before},
        "base_checkpoint": dict(value["base_checkpoint"]), "manifest": dict(value["manifest"]),
        "pixel_protocol": protocol_metadata(value), "appearance_keys": sorted(APPEARANCE_KEYS),
    }
    _unchanged(base_path, value["base_checkpoint"]["sha256"], "Base checkpoint")
    _unchanged(Path(value["manifest"]["path"]), value["manifest"]["sha256"], "Manifest")
    _unchanged(path, before, "Delta checkpoint")
    return result
