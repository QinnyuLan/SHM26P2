"""CPU geometry for a prospective projective-context experiment.

These helpers do not change the scene or refiner. Axes are unoriented; the
estimator describes local line evidence, not physical surface truth.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _unit_vectors(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] == 0 or not np.isfinite(value).all():
        raise ValueError("Finite nonzero vectors required")
    # Scaling first also handles finite vectors whose naive norm over/underflows.
    scale = np.max(np.abs(value), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Finite nonzero vectors required")
    scaled = value / scale
    return scaled / np.linalg.norm(scaled, axis=-1, keepdims=True)


def _intrinsics(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("Finite 3 by 3 intrinsics required")
    try:
        inverse = np.linalg.inv(value)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Invertible intrinsics required") from exc
    if not np.isfinite(inverse).all():
        raise ValueError("Finite inverse intrinsics required")
    return value, inverse


def canonical_axis(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1:
        raise ValueError("A finite nonzero vector is required")
    value = _unit_vectors(value)
    return value * (1 if value[np.argmax(abs(value))] >= 0 else -1)


def unoriented_degrees(first, second):
    first, second = _unit_vectors(first), _unit_vectors(second)
    if first.shape[-1] != second.shape[-1]:
        raise ValueError("Vector dimensions differ")
    return np.degrees(np.arccos(np.clip(abs(np.sum(first * second, axis=-1)), 0, 1)))


def local_line_evidence(points, neighbors=16, minimum_linearity=.65):
    """Same local PCA/linearity/support rule as estimate_local_structure, FP64 CPU."""
    neighbors = _positive_integer(neighbors, "neighbors")
    if (not np.isscalar(minimum_linearity) or not np.isfinite(minimum_linearity)
            or not 0 <= minimum_linearity <= 1):
        raise ValueError("minimum_linearity must be finite in [0,1]")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < neighbors:
        raise ValueError("Need N by 3 points and the complete neighbor count")
    if not np.isfinite(points).all() or neighbors < 3:
        raise ValueError("Finite points and at least three neighbors required")
    _, indices = cKDTree(points).query(points, k=neighbors, workers=1)
    local = points[indices]
    center = local.mean(1)
    centered = local - center[:, None]
    covariance = np.einsum("nki,nkj->nij", centered, centered) / (neighbors-1)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 0)
    directions = vectors[..., -1]
    largest = np.maximum(values[:, -1], 1e-12)
    linearity = (values[:, -1]-values[:, -2]) / largest
    radius = np.sqrt(values.sum(-1)).clip(1e-6)
    usable = ((values[:, -1] > 1e-12) & (linearity >= minimum_linearity)
              & (np.linalg.norm(points-center, axis=-1) <= 2.5*radius))
    return {"directions": directions, "linearity": linearity, "usable": usable,
            "eigenvalues": values, "neighbor_indices": indices}


def spatial_blocks(points, bins=4):
    """Fixed world-axis blocks with 2/98% robust bounds; tails stay in edge blocks.

    This partition is conditional on the supplied coordinate frame. It is not
    claimed to be rotation invariant or a calibrated uncertainty model.
    """
    bins = _positive_integer(bins, "bins")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or bins < 1:
        raise ValueError("Nonempty 3D points and positive bin count required")
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite block coordinate")
    lower, upper = np.quantile(points, [.02, .98], axis=0)
    span = upper-lower
    unit = np.divide(points-lower, span, out=np.zeros_like(points), where=span > 1e-12)
    cells = np.floor(unit.clip(0, 1)*bins).astype(np.int64).clip(0, bins-1)
    keys, inverse = np.unique(cells, axis=0, return_inverse=True)
    return inverse, {"cells": keys.tolist(), "lower": lower.tolist(), "upper": upper.tolist(),
                     "bins": bins, "quantiles": [.02, .98]}


def orientation_blocks(directions, linearity, block_ids):
    """Each occupied block contributes unit total weight after local weighting."""
    directions = np.asarray(directions, dtype=np.float64)
    linearity = np.asarray(linearity, dtype=np.float64)
    block_ids = np.asarray(block_ids)
    if (linearity.ndim != 1 or block_ids.ndim != 1
            or directions.shape != (len(linearity), 3) or len(block_ids) != len(linearity)
            or not len(linearity) or not np.isfinite(directions).all()
            or not np.isfinite(linearity).all() or np.any(linearity <= 0)
            or np.any(linearity > 1) or block_ids.dtype.kind not in "iu"):
        raise ValueError("Positive line evidence and aligned block IDs required")
    directions = _unit_vectors(directions)
    tensors = []
    keys = np.unique(block_ids)
    for key in keys:
        use = block_ids == key
        weights = linearity[use] / linearity[use].sum()
        tensors.append(np.einsum("n,ni,nj->ij", weights, directions[use], directions[use]))
    return np.stack(tensors), keys


def principal_axis(tensors):
    tensors = np.asarray(tensors, dtype=np.float64)
    if (tensors.ndim != 3 or tensors.shape[1:] != (3, 3) or not len(tensors)
            or not np.isfinite(tensors).all()):
        raise ValueError("Finite 3 by 3 orientation tensors required")
    if not np.allclose(tensors, tensors.swapaxes(-1, -2), atol=1e-12, rtol=1e-12):
        raise ValueError("Symmetric orientation tensors required")
    if np.any(np.linalg.eigvalsh(tensors) < -1e-12):
        raise ValueError("Positive semidefinite orientation tensors required")
    matrix = tensors.mean(0)
    values, vectors = np.linalg.eigh((matrix+matrix.T)*.5)
    if values[-1] <= 0:
        raise ValueError("Orientation evidence is empty")
    return {"axis": canonical_axis(vectors[:, -1]), "eigenvalues": values,
            "relative_eigengap": float((values[-1]-values[-2])/values[-1])}


def bootstrap_axis(tensors, repeats=256, seed=20260927):
    """Conditional block bootstrap: neighborhood tangents are held fixed."""
    repeats = _positive_integer(repeats, "repeats")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    tensors = np.asarray(tensors, dtype=np.float64)
    original = principal_axis(tensors)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(tensors), size=(repeats, len(tensors)))
    axes = np.stack([principal_axis(tensors[index])["axis"] for index in sampled])
    angles = unoriented_degrees(axes, original["axis"])
    return {"reference": original, "axes": axes, "angles_degrees": angles,
            "angle_95_percentile_degrees": float(np.quantile(angles, .95)),
            "repeats": repeats, "seed": seed}


def feature_centers(width, height, stride=16):
    """Four k3/s2/p1 convolutions: renderer centers at .5 + stride*j."""
    width = _positive_integer(width, "width")
    height = _positive_integer(height, "height")
    stride = _positive_integer(stride, "stride")
    ys, xs = np.meshgrid(np.arange((height+stride-1)//stride)*stride+.5,
                         np.arange((width+stride-1)//stride)*stride+.5, indexing="ij")
    return np.stack([xs, ys], axis=-1)


def project_direction(K, rotation, world_axis, centers, minimum_sine=1e-3):
    """Homogeneous vanishing-line field, including a vanishing point at infinity.

    A reflected K is accepted for image augmentation. Camera translation and
    depth do not enter the direction of a projected fixed world axis.
    """
    K, inverse_K = _intrinsics(K)
    rotation = np.asarray(rotation, np.float64)
    centers = np.asarray(centers, np.float64)
    if rotation.shape != (3, 3) or centers.ndim < 1 or centers.shape[-1] != 2:
        raise ValueError("K/R must be 3 by 3 and centers end in 2 coordinates")
    if not all(np.isfinite(x).all() for x in (K, rotation, centers)):
        raise ValueError("Finite camera and grid required")
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-5)):
        raise ValueError("R must be a proper camera rotation; reflect K for augmentation")
    if (not np.isscalar(minimum_sine) or not np.isfinite(minimum_sine)
            or not 0 <= minimum_sine <= 1):
        raise ValueError("minimum_sine must be finite in [0,1]")
    world_axis = canonical_axis(world_axis)
    if world_axis.shape != (3,):
        raise ValueError("A three dimensional world axis is required")
    camera_axis = rotation @ world_axis
    camera_axis /= np.linalg.norm(camera_axis)
    homogeneous = K @ camera_axis
    direction = homogeneous[:2] - centers*homogeneous[2]
    rays = np.concatenate([centers, np.ones((*centers.shape[:-1], 1))], axis=-1)
    rays = rays @ inverse_K.T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    sine = np.linalg.norm(np.cross(rays, camera_axis), axis=-1)
    norm = np.linalg.norm(direction, axis=-1)
    valid = (sine >= minimum_sine) & (norm > 1e-12)
    unit = np.divide(direction, norm[..., None], out=np.zeros_like(direction),
                     where=valid[..., None])
    return unit, valid, sine


def best_constant_direction(directions, valid=None):
    """Minimize mean 1-(e dot v)^2 (squared chord distance between dyads / 2)."""
    directions = np.asarray(directions, np.float64)
    if directions.ndim < 1 or directions.shape[-1] != 2 or not np.isfinite(directions).all():
        raise ValueError("Finite direction grid ending in two coordinates required")
    if valid is not None:
        valid = np.asarray(valid)
        if valid.shape != directions.shape[:-1] or valid.dtype.kind != "b":
            raise ValueError("Boolean validity must match direction grid")
        directions = directions[valid]
    directions = directions.reshape(-1, 2)
    if not len(directions):
        raise ValueError("No valid directions for a constant control")
    directions = _unit_vectors(directions)
    values, vectors = np.linalg.eigh(directions.T @ directions / len(directions))
    return canonical_axis(vectors[:, -1]), float(values[-1]-values[0])


def reflect_intrinsics(K, width):
    width = _positive_integer(width, "width")
    K, _ = _intrinsics(K)
    return np.array([[-1., 0., width], [0., 1., 0.], [0., 0., 1.]]) @ K


def line_samples(centers, directions, offsets, width, height, stride=16):
    """Pixel-space samples; retain only points supported by the feature lattice."""
    width = _positive_integer(width, "width")
    height = _positive_integer(height, "height")
    stride = _positive_integer(stride, "stride")
    centers, directions = np.asarray(centers, np.float64), np.asarray(directions, np.float64)
    offsets = np.asarray(offsets, np.float64)
    if (centers.ndim < 1 or centers.shape[-1] != 2 or directions.shape != centers.shape
            or offsets.ndim != 1 or not len(offsets)
            or not all(np.isfinite(x).all() for x in (centers, directions, offsets))):
        raise ValueError("Aligned finite 2D centers/directions and a nonempty offset vector required")
    points = centers[..., None, :] + directions[..., None, :]*offsets[:, None]
    fh, fw = (height+stride-1)//stride, (width+stride-1)//stride
    feature_xy = (points-.5)/stride
    valid = ((feature_xy[..., 0] >= 0) & (feature_xy[..., 0] <= fw-1)
             & (feature_xy[..., 1] >= 0) & (feature_xy[..., 1] <= fh-1)
             & (np.linalg.norm(directions, axis=-1)[..., None] > 1e-12))
    return points, valid
