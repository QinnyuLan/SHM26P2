"""Isolated deck-context experiment; no GaussianScene/refinement source changes.

Attach after loading a normal scene, call ``adapter.set_step(step)`` before each
render, and reattach explicitly when loading an experimental checkpoint. Only
methods are patched: parameter names, identities and state_dict keys stay intact.
The adapter is single-camera/single-thread, and does not serialize into state_dict.
"""
from __future__ import annotations

import inspect
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F

from .structure_axes import (
    best_constant_direction,
    feature_centers,
    line_samples,
    project_direction,
)

AXIS = np.array([.39810321798792697, -.012510913979839057, .917255310619157],
                dtype=np.float64)
OFFSETS = np.linspace(-256., 256., 17)
STRIDE = 16
BLEND_STEPS = 500
MODES = ("original", "per_camera", "projective", "wrong")
ROLL_90 = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
_ABSENT = object()


def _cpu64(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu", dtype=torch.float64).numpy()
    return np.array(value, dtype=np.float64, copy=True)


def _integer(value, name, minimum=1):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def prepare_sampling_geometry(K, rotation, width, height):
    """FP64, detached geometry; output FP32 grids use align_corners=True.

    All sampled arms share the per-query/per-offset validity intersection.
    Coordinates outside this intersection are still finite but never counted.
    This support itself contains geometry, so controls are conditional on it.
    """
    width, height = _integer(width, "width"), _integer(height, "height")
    K, rotation = _cpu64(K), _cpu64(rotation)
    centers = feature_centers(width, height, STRIDE)
    true, true_valid, _ = project_direction(K, rotation, AXIS, centers)
    if true_valid.any():
        constant, gap = best_constant_direction(true, true_valid)
        constant_field = np.broadcast_to(constant, true.shape)
        constant_valid = np.ones_like(true_valid)
    else:
        constant, gap = None, None
        constant_field, constant_valid = np.zeros_like(true), np.zeros_like(true_valid)
    wrong, wrong_valid, _ = project_direction(K, ROLL_90 @ rotation, AXIS, centers)
    fields = {"per_camera": (constant_field, constant_valid),
              "projective": (true, true_valid), "wrong": (wrong, wrong_valid)}
    fh, fw = centers.shape[:2]
    grids, masks = {}, {}
    for mode, (direction, defined) in fields.items():
        points, supported = line_samples(centers, direction, OFFSETS, width, height, STRIDE)
        xy = (points-.5)/STRIDE
        normalized = np.empty_like(xy)
        normalized[..., 0] = 2*xy[..., 0]/(fw-1)-1 if fw > 1 else 0.
        normalized[..., 1] = 2*xy[..., 1]/(fh-1)-1 if fh > 1 else 0.
        grids[mode] = normalized.astype(np.float32)
        masks[mode] = supported & defined[..., None]
    common = masks["per_camera"] & masks["projective"] & masks["wrong"]
    count = common.sum(-1)
    direction_comparable = true_valid & wrong_valid
    angles = np.rad2deg(np.arccos(np.clip(
        np.abs((true[direction_comparable]*wrong[direction_comparable]).sum(-1)), 0., 1.)))
    diagnostics = {
        "feature_shape": [fh, fw], "offset_count": len(OFFSETS),
        "align_corners": True, "per_camera_constant": None if constant is None else constant.tolist(),
        "per_camera_dyad_gap": gap,
        "common_valid_fraction": float(common.mean()),
        "total_queries": int(count.size), "nonempty_queries": int((count > 0).sum()),
        "total_offset_slots": int(common.size), "common_valid_samples": int(common.sum()),
        "empty_fraction": float((count == 0).mean()),
        "only_center_fraction": float(((count == 1) & common[..., len(OFFSETS)//2]).mean()),
        "mean_valid_samples": float(count.mean()), "minimum_valid_samples": int(count.min()),
        "maximum_valid_samples": int(count.max()),
        "own_valid_fraction": {mode: float(mask.mean()) for mode, mask in masks.items()},
        "direction_valid_fraction": {mode: float(defined.mean())
                                     for mode, (_, defined) in fields.items()},
        "true_wrong_comparable_queries": int(direction_comparable.sum()),
        "true_wrong_unoriented_degrees_median": float(np.median(angles)) if angles.size else None,
        "true_wrong_unoriented_degrees_p95": float(np.quantile(angles, .95)) if angles.size else None,
    }
    return {"grids": grids, "common_valid": common, "own_valid": masks,
            "diagnostics": diagnostics, "width": width, "height": height}


def sampled_average(x, grid, common_valid):
    """BCHW feature mean on common offsets, with old row mean at empty queries."""
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("Floating BCHW features required")
    _, _, height, width = x.shape
    if (grid.shape != (height, width, len(OFFSETS), 2)
            or common_valid.shape != (height, width, len(OFFSETS))
            or common_valid.dtype != torch.bool):
        raise ValueError("Sampling grid/support does not match the feature lattice")
    if grid.requires_grad or common_valid.requires_grad:
        raise ValueError("Geometry/support must be detached")
    grid = grid.to(device=x.device, dtype=x.dtype)
    common_valid = common_valid.to(device=x.device)
    sample_grid = grid.reshape(1, height, width*len(OFFSETS), 2).expand(x.shape[0], -1, -1, -1)
    sampled = F.grid_sample(x, sample_grid, mode="bilinear", padding_mode="zeros",
                            align_corners=True)
    sampled = sampled.reshape(*x.shape, len(OFFSETS))
    count = common_valid.sum(-1)
    pooled = (sampled * common_valid[None, None]).sum(-1) / count.clamp_min(1)[None, None]
    empty = count == 0
    old_mean = x.mean(-1, keepdim=True).expand_as(x)
    pooled = torch.where(empty[None, None], old_mean, pooled)
    return pooled, empty


class ProjectiveContextAdapter:
    """Instance-local horizontal-branch adapter; no trainable state of its own."""

    def __init__(self, scene, config):
        config = dict(config)
        self.mode = config.pop("mode", "projective")
        if self.mode not in MODES or config:
            raise ValueError(f"Expected mode in {MODES} and no other experimental options")
        if getattr(scene, "_projective_context_adapter", None) is not None:
            raise ValueError("Scene already has a projective-context adapter")
        panorama = getattr(getattr(scene, "refiner", None), "panorama", None)
        if (panorama is None or getattr(panorama, "mode", None) != "pyramid_strip"
                or len(panorama.strips) != 2 or len(panorama.pooled) != len(panorama.bin_sizes)):
            raise ValueError("Expected the existing multiscale pyramid_strip refiner")
        self.scene, self.panorama = scene, panorama
        self.step = 0
        self.diagnostics = {"mode": self.mode, "step": 0, "blend": 0., "sampling_applied": False}
        self._camera, self._cache_key, self._geometry, self._device_cache = None, None, None, None
        self._render_override = scene.__dict__.get("render", _ABSENT)
        self._forward_override = panorama.__dict__.get("forward", _ABSENT)
        self._original_render, self._original_forward = scene.render, panorama.forward
        self._render_signature = inspect.signature(self._original_render)
        self._attached = True

        def wrapped_render(_scene, *args, **kwargs):
            bound = self._render_signature.bind(*args, **kwargs)
            for name in ("K", "w2c", "width", "height"):
                if name not in bound.arguments:
                    raise ValueError(f"render call lacks required camera argument {name}")
            previous = self._camera
            self.set_camera(bound.arguments["K"], bound.arguments["w2c"],
                            bound.arguments["width"], bound.arguments["height"])
            try:
                return self._original_render(*args, **kwargs)
            finally:
                self._camera = previous

        def wrapped_forward(_panorama, x):
            return self.forward(x)

        scene.render = MethodType(wrapped_render, scene)
        panorama.forward = MethodType(wrapped_forward, panorama)
        scene._projective_context_adapter = self

    @property
    def configuration(self):
        return {"mode": self.mode, "axis": AXIS.tolist(), "offsets_native_px": OFFSETS.tolist(),
                "stride": STRIDE, "blend_steps": BLEND_STEPS, "align_corners": True,
                "wrong_rotation": "Rz90 @ R", "common_support": "three sampled arms per query and offset",
                "empty_input": "old row mean", "empty_branch_output": "original horizontal branch",
                "interpolation": "after full strip projection/GroupNorm/SiLU"}

    def set_step(self, step):
        self.step = _integer(step, "step", minimum=0)

    def set_camera(self, K, w2c_or_rotation, width, height):
        """For direct refiner calls; render wrapper sets/restores this automatically."""
        K, pose = _cpu64(K), _cpu64(w2c_or_rotation)
        if pose.shape == (4, 4):
            if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-6):
                raise ValueError("Finite homogeneous w2c required")
            rotation = pose[:3, :3].copy()
        elif pose.shape == (3, 3):
            rotation = pose
        else:
            raise ValueError("Expected 4 by 4 w2c or 3 by 3 rotation")
        self._camera = (K, rotation, _integer(width, "width"), _integer(height, "height"))

    def _sampling_tensors(self, x):
        if self._camera is None:
            raise RuntimeError("Set the actual camera before a sampled refiner call")
        K, R, width, height = self._camera
        key = (K.tobytes(), R.tobytes(), width, height)
        if key != self._cache_key:
            self._geometry = prepare_sampling_geometry(K, R, width, height)
            self._cache_key, self._device_cache = key, None
        geometry = self._geometry
        if tuple(geometry["diagnostics"]["feature_shape"]) != tuple(x.shape[-2:]):
            raise ValueError("Camera and stride16 features disagree; crops/flips require explicit camera transforms")
        device_key = (x.device, x.dtype, self.mode)
        if self._device_cache is None or self._device_cache[0] != device_key:
            grid = torch.from_numpy(geometry["grids"][self.mode]).to(device=x.device, dtype=x.dtype)
            valid = torch.from_numpy(geometry["common_valid"]).to(device=x.device)
            self._device_cache = (device_key, grid, valid)
        return self._device_cache[1], self._device_cache[2]

    def forward(self, x):
        blend = min(self.step/BLEND_STEPS, 1.)
        self.diagnostics = {"mode": self.mode, "step": self.step, "blend": blend,
                            "sampling_applied": self.mode != "original" and blend > 0}
        if self.mode == "original" or blend == 0:
            return self._original_forward(x)
        grid, common = self._sampling_tensors(x)
        self.diagnostics.update(self._geometry["diagnostics"])
        module, size = self.panorama, x.shape[-2:]
        branches = [x]
        for bins, project in zip(module.bin_sizes, module.pooled):
            pooled = F.adaptive_avg_pool2d(x, (min(bins, size[0]), min(bins, size[1])))
            branches.append(F.interpolate(project(pooled), size, mode="bilinear", align_corners=False))
        old = module.strips[0](x.mean(-1, keepdim=True)).expand(-1, -1, -1, size[1])
        averaged, empty = sampled_average(x, grid, common)
        sampled = module.strips[0](averaged)
        sampled = torch.where(empty[None, None], old, sampled)
        horizontal = sampled if blend == 1 else (1-blend)*old + blend*sampled
        vertical = module.strips[1](x.mean(-2, keepdim=True)).expand(-1, -1, size[0], -1)
        branches.extend([horizontal, vertical])
        return module.merge(torch.cat(branches, dim=1))

    def detach(self):
        """Restore exactly the prior instance method overrides (if any)."""
        if not self._attached:
            return
        for obj, name, previous in ((self.scene, "render", self._render_override),
                                    (self.panorama, "forward", self._forward_override)):
            if previous is _ABSENT:
                delattr(obj, name)
            else:
                setattr(obj, name, previous)
        delattr(self.scene, "_projective_context_adapter")
        self._attached = False


def attach_projective_context(scene, config):
    return ProjectiveContextAdapter(scene, config)
