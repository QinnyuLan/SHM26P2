"""Versioned pixel interfaces; absent provenance always means historical behavior.

Both profiles keep camera intrinsics and projected UV in COLMAP corner coordinates.
The legacy profile intentionally preserves the old, mixed sampling interfaces.
"""
from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

LEGACY = "legacy_mixed_v1"
CORNER = "colmap_corner_v2"
_COMMON = {"intrinsics_and_projected_uv": "corner_origin; first_center=(0.5,0.5)",
           "array_coordinates": "integer_element_centers; first_center=(0,0)"}
_PROFILES = {
    LEGACY: dict(_COMMON, id=LEGACY, version=1, prepare_warp="legacy_unconjugated",
                 categorical_resize="legacy_nearest", appearance_prior_nearest="rint_uv",
                 sparse_sampling="legacy_array_interpretation",
                 official_export="legacy_unconjugated",
                 front_sampling="corner_with_legacy_depth_gate"),
    CORNER: dict(_COMMON, id=CORNER, version=2, prepare_warp="corner_conjugate",
                 categorical_resize="center_nearest_half_up_v1",
                 appearance_prior_nearest="floor_corner_uv", sparse_sampling="corner_bilinear",
                 official_export="corner_conjugate", front_sampling="corner_bilinear"),
}


def pixel_protocol(value=None):
    """Resolve an ID, protocol object, manifest/view, or checkpoint without I/O."""
    if value is None:
        return LEGACY
    if isinstance(value, Mapping):
        if "pixel_protocol" in value:
            if value["pixel_protocol"] is None and value.get("schema_version", 1) >= 2:
                raise ValueError("Versioned manifest requires explicit pixel_protocol")
            return pixel_protocol(value["pixel_protocol"])
        if "id" in value:
            name = value["id"]
            if not isinstance(name, str):
                raise ValueError("Pixel protocol ID must be a string")
            expected_version = {LEGACY: 1, CORNER: 2}.get(name)
            if expected_version is None or value.get("version") != expected_version:
                raise ValueError("Unknown or incomplete pixel protocol object")
            if any(key not in _PROFILES[name] or _PROFILES[name][key] != item
                   for key, item in value.items()):
                raise ValueError("Pixel protocol fields disagree with its fixed profile")
            return name
        if value.get("schema_version", 1) >= 2:
            raise ValueError("Versioned manifest requires explicit pixel_protocol")
        return LEGACY
    if not isinstance(value, str) or value not in {LEGACY, CORNER}:
        raise ValueError(f"Unknown pixel protocol: {value!r}")
    return value


def protocol_metadata(value=None):
    return dict(_PROFILES[pixel_protocol(value)])


def annotate_manifest(manifest):
    """Return in-memory views carrying their source protocol; never rewrite files."""
    name = pixel_protocol(manifest)
    result = dict(manifest)
    result["views"] = []
    for original in manifest["views"]:
        if "pixel_protocol" in original and pixel_protocol(original) != name:
            raise ValueError("View pixel protocol disagrees with manifest")
        view = dict(original)
        view["pixel_protocol"] = name
        result["views"].append(view)
    return result


def require_matching_protocol(first, second, context):
    if pixel_protocol(first) != pixel_protocol(second):
        raise ValueError(f"Pixel protocol mismatch for {context}; use a separate migration experiment")


def corner_to_array_K(K):
    """Only for raster APIs. Scale the corner K before applying this conversion."""
    result = np.array(K, copy=True)
    if result.shape != (3, 3):
        raise ValueError("Expected 3x3 intrinsics")
    result = result.astype(np.result_type(result.dtype, np.float32), copy=False)
    result[:2, 2] -= .5
    return result


def nearest_center_indices(source_size, target_size):
    """Pixel-center nearest resize, with exact rational arithmetic and ties up."""
    if source_size < 1 or target_size < 1:
        raise ValueError("Resize dimensions must be positive")
    return ((2 * np.arange(target_size, dtype=np.int64) + 1) * source_size) // (2 * target_size)


def resize_discrete_numpy(values, size, protocol=LEGACY):
    """Resize HW[...], using OpenCV's (width,height) size convention."""
    if pixel_protocol(protocol) == LEGACY:
        return cv2.resize(values, size, interpolation=cv2.INTER_NEAREST)
    x = nearest_center_indices(values.shape[1], int(size[0]))
    y = nearest_center_indices(values.shape[0], int(size[1]))
    return values[y[:, None], x[None, :]]


def resize_discrete_tensor(values, size, protocol=LEGACY):
    """Resize [...,H,W]; size is (height,width), matching torch.interpolate."""
    import torch
    from torch.nn import functional as F
    if pixel_protocol(protocol) == LEGACY:
        original_shape = values.shape
        resized = F.interpolate(values.reshape(1, -1, *original_shape[-2:]), size,
                                mode="nearest")
        return resized.reshape(*original_shape[:-2], *size)
    y = torch.as_tensor(nearest_center_indices(values.shape[-2], int(size[0])),
                        device=values.device)
    x = torch.as_tensor(nearest_center_indices(values.shape[-1], int(size[1])),
                        device=values.device)
    return values.index_select(-2, y).index_select(-1, x)


def sampling_grid(uv, width, height, protocol=LEGACY):
    """Grid for align_corners=False; legacy treats incoming UV as array indices."""
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    location = uv if pixel_protocol(protocol) == CORNER else uv + .5
    return 2 * location / uv.new_tensor([width, height]) - 1


def inside_bilinear_centers(uv, width, height, protocol=LEGACY, border=0):
    offset = .5 if pixel_protocol(protocol) == CORNER else 0.
    return ((uv[..., 0] >= border + offset)
            & (uv[..., 0] <= width - 1 - border + offset)
            & (uv[..., 1] >= border + offset)
            & (uv[..., 1] <= height - 1 - border + offset))
