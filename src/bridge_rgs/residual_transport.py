"""CPU-only expected-depth residual transport; no scene, labels or optimizer.

Coordinates use gsplat/COLMAP corner intrinsics: pixel centers are j+.5,i+.5;
OpenCV array sampling therefore uses projected corner coordinates minus .5.
OpenCV INTER_LINEAR uses its fixed interpolation table, not exact real bilinear.
ED is camera-z expectation, not ray length or a physical surface guarantee.
"""
from __future__ import annotations

import cv2
import numpy as np

ALPHA_MINIMUM = .95
DEPTH_RELATIVE_TOLERANCE = .01


def _finite(value, name, shape=None):
    out = np.asarray(value, dtype=np.float64)
    if shape is not None and out.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name}: finite values required")
    return out


def _camera(K, w2c):
    K = _finite(K, "K", (3, 3))
    w2c = _finite(w2c, "w2c", (4, 4))
    if not np.allclose(K[2], [0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError("K must use conventional homogeneous pinhole coordinates")
    if not np.allclose(w2c[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError("w2c must be homogeneous")
    R = w2c[:3, :3]
    if (not np.allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-5)
            or abs(np.linalg.det(R)-1) > 1e-5):
        raise ValueError("w2c must have a proper orthonormal rotation")
    try:
        inverse = np.linalg.inv(K)
    except np.linalg.LinAlgError as exc:
        raise ValueError("K must be invertible") from exc
    if not np.isfinite(inverse).all():
        raise ValueError("Nonfinite inverse K")
    return K, inverse, R, w2c[:3, 3]


def _mask(value, shape, name):
    out = np.asarray(value)
    if out.shape != shape or out.dtype != np.bool_:
        raise ValueError(f"{name}: boolean array of shape {shape} required")
    return out


def _depth_alpha(depth, alpha):
    depth = _finite(depth, "depth")
    if depth.ndim != 2 or not all(depth.shape):
        raise ValueError("Depth must be a nonempty HW array")
    alpha = _finite(alpha, "alpha", depth.shape)
    if np.any(depth < 0) or np.any(alpha < 0) or np.any(alpha > 1+1e-6):
        raise ValueError("Nonnegative depth and alpha in [0,1] required")
    return depth, alpha


def unproject_depth(depth, K, w2c):
    """Camera-z depth to world positions. Zero depth remains invalid to callers."""
    depth = _finite(depth, "depth")
    if depth.ndim != 2 or not all(depth.shape) or np.any(depth < 0):
        raise ValueError("Nonempty nonnegative HW depth required")
    _, inverse, R, t = _camera(K, w2c)
    y, x = np.indices(depth.shape, dtype=np.float64)
    rays = np.stack([x+.5, y+.5, np.ones_like(x)], axis=-1) @ inverse.T
    with np.errstate(over="ignore", invalid="ignore"):
        # Invert the stored matrix, not an assumed exact orthogonal transpose:
        # an FP32 rotation promoted to FP64 has a small orthogonality residual.
        world = (depth[..., None]*rays-t) @ np.linalg.inv(R).T
    if not np.isfinite(world).all():
        raise ValueError("Depth unprojection overflow")
    return world


def project_world(world, K, w2c):
    """Return array_xy, camera-z and positive finite projection validity.

    Invalid image coordinates are zero placeholders and must use the returned
    mask. They are never cast as huge offscreen values into OpenCV.
    """
    world = _finite(world, "world")
    if world.ndim < 2 or world.shape[-1] != 3:
        raise ValueError("World positions must have trailing dimension 3")
    K, _, R, t = _camera(K, w2c)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        camera = world @ R.T+t
        homogeneous = camera @ K.T
        valid = np.isfinite(homogeneous).all(-1) & (camera[..., 2] > 0)
        xy = np.zeros(world.shape[:-1]+(2,), dtype=np.float64)
        np.divide(homogeneous[..., :2], homogeneous[..., 2, None],
                  out=xy, where=valid[..., None])
        xy -= .5
        valid &= np.isfinite(xy).all(-1)
    xy[~valid] = 0
    return {"array_xy": xy, "depth": camera[..., 2], "finite_positive": valid}


def _sample(array, xy, inside):
    # Replace invalid coordinates before FP32 conversion, avoiding overflow/NaN.
    safe = np.where(inside[..., None], xy, 0).astype(np.float32)
    return cv2.remap(np.ascontiguousarray(array), safe[..., 0], safe[..., 1],
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0)


def transport_residual(target_depth, target_alpha, target_valid, target_K,
                       target_w2c, sources):
    """Same-support true/wrong residuals from a fixed list of source dictionaries.

    Each source has residual(HWC3), depth(HW), alpha(HW), valid(HW bool), K, w2c.
    Wrong correspondence reflects only residual and RGB-valid in array x. Source
    depth/alpha stay at the true geometric projection. Both arms share every
    accepted source and use its uniform weight. Thresholds are fixed heuristics.
    """
    target_depth, target_alpha = _depth_alpha(target_depth, target_alpha)
    shape = target_depth.shape
    target_valid = _mask(target_valid, shape, "target_valid")
    if not isinstance(sources, (list, tuple)) or not sources:
        raise ValueError("A nonempty fixed source list is required")
    world = unproject_depth(target_depth, target_K, target_w2c)
    target_ok = target_valid & (target_depth > 0) & (target_alpha >= ALPHA_MINIMUM)
    true_sum = np.zeros(shape+(3,), dtype=np.float64)
    wrong_sum = np.zeros_like(true_sum)
    counts = np.zeros(shape, dtype=np.int32)
    summaries = []
    for source in sources:
        depth, alpha = _depth_alpha(source["depth"], source["alpha"])
        height, width = depth.shape
        if max(height, width, *shape) >= 32767:
            raise ValueError("OpenCV remap image dimensions must be below 32767")
        valid = _mask(source["valid"], depth.shape, "source valid")
        residual = _finite(source["residual"], "source residual", depth.shape+(3,))
        projection = project_world(world, source["K"], source["w2c"])
        xy = projection["array_xy"]
        inside = (projection["finite_positive"] & (xy[..., 0] >= 0)
                  & (xy[..., 0] <= width-1) & (xy[..., 1] >= 0)
                  & (xy[..., 1] <= height-1))
        mirrored = xy.copy()
        mirrored[..., 0] = width-1-xy[..., 0]
        normal_valid = _sample(valid.astype(np.float32), xy, inside) == 1
        mirror_valid = _sample(valid.astype(np.float32), mirrored, inside) == 1
        source_depth = _sample(depth, xy, inside)
        source_alpha = _sample(alpha, xy, inside)
        z = projection["depth"]
        relative = np.full(shape, np.inf)
        positive = inside & (source_depth > 0)
        np.divide(np.abs(z-source_depth), np.maximum(z, source_depth),
                  out=relative, where=positive)
        geometric = (target_ok & inside & positive & (source_alpha >= ALPHA_MINIMUM)
                     & (relative <= DEPTH_RELATIVE_TOLERANCE))
        common = geometric & normal_valid & mirror_valid
        true_samples = _sample(residual, xy, inside)
        wrong_samples = _sample(residual, mirrored, inside)
        true_sum[common] += true_samples[common]
        wrong_sum[common] += wrong_samples[common]
        counts += common
        summaries.append({"geometric_pixels": int(geometric.sum()),
                          "normal_rgb_valid_pixels": int((geometric & normal_valid).sum()),
                          "mirrored_rgb_valid_pixels": int((geometric & mirror_valid).sum()),
                          "common_pixels": int(common.sum()),
                          "source_shape": [height, width]})
    usable = counts > 0
    np.divide(true_sum, counts[..., None], out=true_sum, where=usable[..., None])
    np.divide(wrong_sum, counts[..., None], out=wrong_sum, where=usable[..., None])
    if not np.isfinite(true_sum).all() or not np.isfinite(wrong_sum).all():
        raise ValueError("Residual aggregation overflow")
    return {"true_residual": true_sum, "wrong_residual": wrong_sum,
            "valid": usable, "source_count": counts, "source_summaries": summaries}


def shrink_statistics(base, target, residual, valid):
    """Sufficient statistics for one full target view, with unsupported r=0.

    valid is the target RGB-valid mask, not transport coverage. Statistics average
    all 3 channels/pixels; each view has one vote in the subsequent fit.
    """
    base = _finite(base, "base")
    if base.ndim != 3 or base.shape[-1] != 3:
        raise ValueError("HWC3 base required")
    target = _finite(target, "target", base.shape)
    residual = _finite(residual, "residual", base.shape)
    valid = _mask(valid, base.shape[:2], "valid")
    count = int(valid.sum())
    if not count:
        raise ValueError("A target view needs valid RGB pixels; do not drop it silently")
    error = target[valid]-base[valid]
    r = residual[valid]
    with np.errstate(over="ignore", invalid="ignore"):
        numerator = float(np.mean(r*error, dtype=np.float64))
        denominator = float(np.mean(r*r, dtype=np.float64))
        zero_mse = float(np.mean(error*error, dtype=np.float64))
    if not np.isfinite([numerator, denominator, zero_mse]).all():
        raise ValueError("Shrink statistics overflow")
    return {"numerator": numerator, "denominator": denominator,
            "zero_mse": zero_mse, "valid_pixels": count}


def fit_shrinkage(statistics):
    """Fit one scalar in [0,1] by equal-view un-clipped MSE, no target selection."""
    if not statistics:
        raise ValueError("At least one fitting view required")
    numerator, denominator = [], []
    for row in statistics:
        values = _finite([row["numerator"], row["denominator"], row["zero_mse"]], "stats")
        if values[1] < 0 or values[2] < 0 or row["valid_pixels"] <= 0:
            raise ValueError("Invalid shrink sufficient statistics")
        if values[1] == 0 and values[0] != 0:
            raise ValueError("Zero residual energy cannot have nonzero covariance")
        numerator.append(values[0])
        denominator.append(values[1])
    with np.errstate(over="ignore", invalid="ignore"):
        numerator, denominator = float(np.mean(numerator)), float(np.mean(denominator))
    if not np.isfinite([numerator, denominator]).all():
        raise ValueError("Aggregated shrink statistics overflow")
    coefficient = 0. if denominator == 0 else float(np.clip(numerator/denominator, 0, 1))
    return {"coefficient": coefficient, "numerator": numerator, "denominator": denominator,
            "views": len(statistics), "zero_residual_energy": denominator == 0,
            "objective": "equal_view_unclipped_mse"}


def apply_shrinkage(base, residual, coefficient):
    """Return un-clipped correction; clipping belongs to explicit output scoring."""
    base = _finite(base, "base")
    residual = _finite(residual, "residual", base.shape)
    if not np.isscalar(coefficient) or not np.isfinite(coefficient) or not 0 <= coefficient <= 1:
        raise ValueError("Shrink coefficient must be a finite scalar in [0,1]")
    result = base+coefficient*residual
    if not np.isfinite(result).all():
        raise ValueError("Residual correction overflow")
    return result
