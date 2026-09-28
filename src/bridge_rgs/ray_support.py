"""Conservative TRAIN-only SfM expected-depth supervision, not a novel loss.

Positions/covariances are the offline triangulation estimates, never Gaussian
shape covariances. Depth uncertainty is conditional on the supplied camera.
Expected depth alone does not constrain the complete ray termination distribution:
front/back opacity can compensate, so this is not a full free-space constraint.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .coordinates import (
    CORNER,
    LEGACY,
    annotate_manifest,
    inside_bilinear_centers,
    require_matching_protocol,
    sampling_grid,
)
from .coordinates import pixel_protocol as resolve_protocol


@dataclass(frozen=True)
class RaySupportConfig:
    min_observations: int = 3
    max_reprojection_error_px: float = 1.0
    max_current_reprojection_error_px: float = 2.0
    max_relative_depth_std: float = 0.10
    relative_depth_std_scale: float = 0.02
    observation_saturation: int = 8
    min_confidence: float = 0.05
    border_px: int = 2
    min_alpha: float = 0.5
    max_targets_per_view: int = 4096
    uv_mode: str = "observed"
    loss_kind: str = "log_huber"
    huber_delta: float = 0.1
    near_error_fraction: float = 0.1

    def __post_init__(self):
        integers = (self.min_observations, self.observation_saturation,
                    self.border_px, self.max_targets_per_view)
        if any(isinstance(x, bool) or not isinstance(x, int) for x in integers):
            raise ValueError("Observation, border, and target counts must be integers")
        if self.min_observations < 2 or self.observation_saturation < self.min_observations:
            raise ValueError("Need >=2 observations and saturation >= min_observations")
        positive = (self.max_reprojection_error_px, self.max_current_reprojection_error_px,
                    self.max_relative_depth_std, self.relative_depth_std_scale, self.huber_delta)
        if not all(np.isfinite(x) and x > 0 for x in positive):
            raise ValueError("Quality/loss scales must be finite and positive")
        if not 0 <= self.min_alpha <= 1 or not 0 <= self.min_confidence <= 1:
            raise ValueError("Alpha/confidence thresholds must be in [0,1]")
        if self.border_px < 0 or self.max_targets_per_view < 1:
            raise ValueError("Invalid border or target budget")
        if not 0 < self.near_error_fraction < 1:
            raise ValueError("near_error_fraction must be in (0,1)")
        if self.uv_mode not in {"observed", "projected"}:
            raise ValueError("uv_mode must be observed or explicitly projected")
        if self.loss_kind not in {"log_huber", "relative_huber"}:
            raise ValueError("Unknown sparse depth loss_kind")


@dataclass
class SparseDepthTargets:
    pixels: torch.Tensor
    depth: torch.Tensor
    confidence: torch.Tensor
    point_indices: torch.Tensor
    stats: dict[str, Any]
    pixel_protocol: str = LEGACY


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plane(value, name):
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"{name} must be HxW or HxWx1")
    return value


def _sample(plane, pixels, pixel_protocol=LEGACY):
    height, width = plane.shape
    if resolve_protocol(pixel_protocol) == CORNER:
        grid = sampling_grid(pixels, width, height, CORNER)[None, None]
        return F.grid_sample(plane[None, None], grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)[0, 0, 0]
    scale = pixels.new_tensor([width - 1, height - 1])
    grid = (2 * pixels / scale - 1)[None, None]
    return F.grid_sample(plane[None, None], grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)[0, 0, 0]


def _observed_rays(path, views, point_indices, track_ids, source_cameras):
    """Parse only selected TRAIN observation lines; discard all other lines."""
    rays = {}
    duplicate_count = 0
    with Path(path).open() as handle:
        while line := handle.readline():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.strip().split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError("Malformed COLMAP image header")
            image_id = int(fields[0])
            observations = handle.readline()
            if observations == "":
                raise ValueError("Missing COLMAP observation line")
            if image_id not in views:
                continue  # Held-out observation values are not parsed or used.
            view = views[image_id]
            if fields[9] != view["name"] or int(fields[8]) != view["camera_id"]:
                raise ValueError(f"COLMAP/manifest identity mismatch for image {image_id}")
            values = np.fromstring(observations, dtype=np.float64, sep=" ")
            if len(values) % 3:
                raise ValueError(f"Malformed TRAIN observations for image {image_id}")
            values = values.reshape(-1, 3)
            indices = point_indices[image_id]
            lookup = {int(track_ids[i]): i for i in indices}
            # Match geometry.triangulate_tracks' explicit last-keypoint-per-image
            # policy; this dataset contains duplicate track IDs in some images.
            selected_by_track = {}
            for j, track in enumerate(values[:, 2]):
                track = int(track)
                if track in lookup:
                    duplicate_count += int(track in selected_by_track)
                    selected_by_track[track] = j
            selected = list(selected_by_track.values())
            ids = values[selected, 2].astype(np.int64)
            camera = source_cameras[str(view["camera_id"])]
            xy = values[selected, :2]
            normalized = cv2.undistortPoints(
                xy[:, None], np.asarray(camera["K"], np.float64),
                np.asarray(camera["opencv_distortion"], np.float64)
            )[:, 0] if len(xy) else np.empty((0, 2))
            rays[image_id] = {lookup[int(track)]: uv for track, uv in zip(ids, normalized)}
    missing = set(views) - set(rays)
    if missing:
        raise ValueError(f"Missing TRAIN camera observation lines: {sorted(missing)[:8]}")
    return rays, {"duplicate_observations": duplicate_count,
                  "duplicate_policy": "last keypoint per image/track, matching triangulation"}


class SparseDepthSupport:
    """Cache TRAIN track membership once, then build detached per-view targets.

    Prefer ``from_manifest`` for original observed UVs. The direct constructor
    accepts synthetic normalized observation rays for CPU tests and other importers.
    All NPZ observation IDs must belong to TRAIN, or construction fails closed.
    """

    def __init__(self, arrays, train_views, config=None, observed_rays=None,
                 pixel_protocol=None):
        self.config = (config if isinstance(config, RaySupportConfig)
                       else RaySupportConfig(**(config or {})))
        if any(view["split"] != "train" for view in train_views):
            raise ValueError("Sparse depth support accepts TRAIN views only")
        protocols = {resolve_protocol(view) for view in train_views}
        if len(protocols) > 1:
            raise ValueError("Sparse support cannot mix pixel protocols")
        self.pixel_protocol = (next(iter(protocols), LEGACY) if pixel_protocol is None
                               else resolve_protocol(pixel_protocol))
        if protocols and protocols != {self.pixel_protocol}:
            raise ValueError("Sparse support views disagree with pixel protocol")
        self.views = {int(view["image_id"]): dict(view) for view in train_views}
        if len(self.views) != len(train_views):
            raise ValueError("Duplicate TRAIN image IDs")
        self.points = np.array(arrays["points"], dtype=np.float64, copy=True)
        self.covariances = np.array(arrays["covariances"], dtype=np.float64, copy=True)
        self.reprojection = np.array(arrays["reprojection_error"], dtype=np.float64, copy=True)
        self.observations = np.array(arrays["num_observations"], dtype=np.int64, copy=True)
        offsets = np.asarray(arrays["observation_offsets"], dtype=np.int64)
        image_ids = np.asarray(arrays["observation_image_ids"], dtype=np.int64)
        n = len(self.points)
        self.track_ids = np.array(arrays.get("track_ids", np.arange(n)), dtype=np.int64)
        if (self.points.shape != (n, 3) or self.covariances.shape != (n, 3, 3)
                or self.reprojection.shape != (n,) or self.observations.shape != (n,)
                or offsets.shape != (n + 1,) or self.track_ids.shape != (n,)):
            raise ValueError("Invalid SfM support array shapes")
        if (offsets[0] != 0 or offsets[-1] != len(image_ids)
                or np.any(np.diff(offsets) < 0)
                or not np.array_equal(np.diff(offsets), self.observations)):
            raise ValueError("Observation offsets/counts disagree")
        if not set(image_ids.tolist()).issubset(self.views):
            raise ValueError("SfM evidence contains held-out or unknown observation IDs")
        if len(np.unique(self.track_ids)) != n:
            raise ValueError("Duplicate SfM track IDs")
        by_view = {image_id: [] for image_id in self.views}
        for point_id in range(n):
            ids = image_ids[offsets[point_id]:offsets[point_id + 1]]
            if len(np.unique(ids)) != len(ids):
                raise ValueError("Repeated image ID must not inflate multi-view support")
            for image_id in ids:
                by_view[int(image_id)].append(point_id)
        self.by_view = {key: np.asarray(value, np.int64) for key, value in by_view.items()}
        finite_cov = np.isfinite(self.covariances).all(axis=(1, 2))
        symmetric = (self.covariances + self.covariances.transpose(0, 2, 1)) / 2
        eig = np.linalg.eigvalsh(np.where(finite_cov[:, None, None], symmetric, 0))
        tolerance = 1e-6 * np.maximum(np.abs(eig).max(1), np.finfo(np.float64).tiny)
        covariance_ok = finite_cov & (eig.min(1) >= -tolerance)
        self.covariances = symmetric
        self.quality = (np.isfinite(self.points).all(1) & covariance_ok
                        & np.isfinite(self.reprojection) & (self.reprojection >= 0)
                        & (self.reprojection <= self.config.max_reprojection_error_px)
                        & (self.observations >= self.config.min_observations))
        self.observed_rays = observed_rays
        self.provenance = {
            "uv_mode": self.config.uv_mode, "config": asdict(self.config),
            "covariance_source": "offline SfM point-position covariance conditional on camera",
            "point_count": n, "quality_point_count": int(self.quality.sum()),
            "train_view_count": len(self.views),
            "pixel_protocol": self.pixel_protocol,
        }

    @classmethod
    def from_manifest(cls, manifest, config=None, init_points_path=None, colmap_images_path=None):
        if not isinstance(manifest, dict):
            manifest_path = Path(manifest).resolve()
            manifest = json.loads(manifest_path.read_text())
        else:
            manifest_path = None
        manifest = annotate_manifest(manifest)
        init_path = Path(init_points_path or manifest["init_points_path"])
        if not init_path.is_absolute() and manifest_path is not None:
            init_path = manifest_path.parent / init_path
        with np.load(init_path, allow_pickle=False) as data:
            prior_protocol = str(data["pixel_protocol"].item()) if "pixel_protocol" in data else None
            require_matching_protocol(manifest, prior_protocol, "SfM support source")
            keys = ("points", "covariances", "reprojection_error", "num_observations",
                    "observation_image_ids", "observation_offsets", "track_ids")
            arrays = {key: data[key] for key in keys}
        views = [view for view in manifest["views"] if view["split"] == "train"]
        support = cls(arrays, views, config, pixel_protocol=resolve_protocol(manifest))
        support.provenance.update(init_points_path=str(init_path.resolve()),
                                  init_points_sha256=_sha256(init_path))
        if support.config.uv_mode == "observed":
            path = (Path(colmap_images_path) if colmap_images_path is not None else
                    Path(manifest["dataset_root"]) / "camera_parameters" / "images.txt")
            if not path.is_file():
                raise FileNotFoundError(
                    f"Original TRAIN observations missing: {path}; explicitly set "
                    "uv_mode='projected' to use the documented approximation")
            support.observed_rays, recovery = _observed_rays(
                path, support.views, support.by_view, support.track_ids,
                manifest["source_cameras"])
            support.provenance["observation_source"] = str(path.resolve())
            support.provenance["observation_source_sha256"] = _sha256(path)
            support.provenance["observation_recovery"] = recovery
        else:
            support.provenance["observation_source"] = (
                "Approximation: current-pose projection of triangulated point; not measured UV")
        return support

    @torch.no_grad()
    def targets_for_view(self, view_id, K, w2c, width, height, valid):
        view_id = int(view_id)
        if view_id not in self.views:
            raise ValueError(f"Image {view_id} is not a TRAIN support view")
        if width < 2 or height < 2:
            raise ValueError("Sparse sampling needs width/height >= 2")
        config = self.config
        index = self.by_view[view_id]
        stats = {"sparse_depth_observed": len(index),
                 "sparse_depth_uv_mode": config.uv_mode,
                 "sparse_depth_pixel_protocol": self.pixel_protocol}
        index = index[self.quality[index]]
        stats["sparse_depth_quality_candidates"] = len(index)
        if config.uv_mode == "observed":
            if self.observed_rays is None or view_id not in self.observed_rays:
                raise ValueError("Observed UV mode requires original TRAIN keypoints")
            rays = self.observed_rays[view_id]
            keep = np.array([i in rays for i in index], dtype=bool)
            stats["sparse_depth_missing_observations"] = int((~keep).sum())
            index = index[keep]
            normalized = np.asarray([rays[i] for i in index], np.float64).reshape(-1, 2)
        pose, intrinsic = _array(w2c).astype(np.float64), _array(K).astype(np.float64)
        if (pose.shape != (4, 4) or intrinsic.shape != (3, 3)
                or not np.isfinite(pose).all() or not np.isfinite(intrinsic).all()):
            raise ValueError("Invalid support camera")
        camera_points = self.points[index] @ pose[:3, :3].T + pose[:3, 3]
        depth = camera_points[:, 2]
        positive = depth > 0
        denominator = np.where(positive, depth, 1)
        projected = camera_points @ intrinsic.T
        projected = projected[:, :2] / denominator[:, None]
        if config.uv_mode == "observed":
            rays_h = np.concatenate([normalized, np.ones((len(index), 1))], axis=1)
            pixels_h = rays_h @ intrinsic.T
            pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
            # Local error uses manifest-native pixels, independent of render scale.
            sx = self.views[view_id]["width"] / width
            sy = self.views[view_id]["height"] / height
            local_error = np.linalg.norm((pixels - projected) * [sx, sy], axis=1)
        else:
            pixels = projected
            local_error = np.zeros(len(index))
        # Var[z | fixed camera] = R_z Sigma_position R_z^T. Not splat shape.
        variance = np.einsum("i,nij,j->n", pose[2, :3], self.covariances[index], pose[2, :3])
        relative_std = np.sqrt(np.maximum(variance, 0)) / denominator
        confidence = np.minimum(self.observations[index] / config.observation_saturation, 1)
        confidence *= 1 / (1 + (self.reprojection[index] / config.max_reprojection_error_px)**2)
        confidence *= 1 / (1 + (relative_std / config.relative_depth_std_scale)**2)
        confidence *= 1 / (1 + (local_error / config.max_current_reprojection_error_px)**2)
        accepted = (positive & np.isfinite(pixels).all(1)
                    & (relative_std <= config.max_relative_depth_std)
                    & (local_error <= config.max_current_reprojection_error_px)
                    & (confidence >= config.min_confidence)
                    & inside_bilinear_centers(pixels, width, height, self.pixel_protocol,
                                              config.border_px))
        pixels, depth = pixels[accepted], depth[accepted]
        confidence, index = confidence[accepted], index[accepted]
        valid = _plane(valid.detach(), "valid")
        if valid.shape != (height, width):
            raise ValueError("Validity grid differs from rendered grid")
        device, dtype = valid.device, torch.float32
        tensor_pixels = torch.as_tensor(pixels, device=device, dtype=dtype)
        if len(index):
            invalid = (~(torch.isfinite(valid) & (valid > 0))).float()[None, None]
            radius = config.border_px
            if radius:
                invalid = F.max_pool2d(F.pad(invalid, (radius,) * 4, value=1),
                                       2 * radius + 1, stride=1)
            support = 1 - invalid[0, 0]
            # Bilinear footprint must lie entirely within eroded valid support.
            mask = (_sample(support, tensor_pixels, self.pixel_protocol) >= 1 - 1e-6).cpu().numpy()
            pixels, depth, confidence, index = (
                array[mask] for array in (pixels, depth, confidence, index))
        stats["sparse_depth_targets_before_cap"] = len(index)
        if len(index) > config.max_targets_per_view:
            chosen = np.linspace(0, len(index) - 1, config.max_targets_per_view).astype(np.int64)
            pixels, depth, confidence, index = (
                array[chosen] for array in (pixels, depth, confidence, index))
        stats["sparse_depth_targets"] = len(index)
        return SparseDepthTargets(
            torch.as_tensor(pixels.copy(), device=device, dtype=dtype),
            torch.as_tensor(depth.copy(), device=device, dtype=dtype),
            torch.as_tensor(confidence.copy(), device=device, dtype=dtype),
            torch.as_tensor(index.copy(), device=device), stats, self.pixel_protocol)

    def loss(self, depth, alpha, valid, K, w2c, view_id):
        """Return (scalar loss or None, detached Python stats); never read GT/RGB.

        The caller sets trainable scene parameters and adds its own loss weight.
        A None result means there is no supported target; do not add a zero loss
        merely to construct optimizer gradients for an otherwise inactive branch.
        """
        plane = _plane(depth, "depth")
        targets = self.targets_for_view(view_id, K, w2c, plane.shape[1], plane.shape[0], valid)
        return sparse_depth_loss(depth, alpha, targets, self.config)


def sparse_depth_loss(depth, alpha, targets, config=None):
    """Robust scale-invariant residual with detached evidence/alpha weighting.

    Alpha is gated/weighted without gradient, preventing a direct incentive to
    lower opacity simply to reduce its loss weight. Rendered expected depth still
    carries gradients to the opacity/geometry allowed by the caller's renderer.
    """
    config = (config if isinstance(config, RaySupportConfig)
              else RaySupportConfig(**(config or {})))
    depth, alpha = _plane(depth, "depth"), _plane(alpha, "alpha")
    if depth.shape != alpha.shape:
        raise ValueError("Depth and alpha grids differ")
    stats = dict(targets.stats)
    stats.update(sparse_depth_valid=0, sparse_depth_loss=0.0,
                 sparse_depth_median_relative_error=None, sparse_depth_too_near_fraction=None,
                 sparse_depth_mean_alpha=None, sparse_depth_confidence_sum=0.0)
    if not len(targets.depth):
        return None, stats
    pixels = targets.pixels.detach().to(device=depth.device, dtype=depth.dtype)
    reference = targets.depth.detach().to(device=depth.device, dtype=depth.dtype)
    confidence = targets.confidence.detach().to(device=depth.device, dtype=depth.dtype)
    predicted = _sample(depth, pixels, targets.pixel_protocol)
    opacity = _sample(alpha.detach().to(depth), pixels, targets.pixel_protocol).detach()
    keep = (torch.isfinite(predicted.detach()) & (predicted.detach() > 0)
            & torch.isfinite(opacity) & (opacity > 0) & (opacity >= config.min_alpha)
            & torch.isfinite(reference) & (reference > 0)
            & torch.isfinite(confidence) & (confidence > 0))
    if not keep.any():
        return None, stats
    ratio = predicted[keep] / reference[keep]
    residual = ratio.log() if config.loss_kind == "log_huber" else ratio - 1
    penalty = F.huber_loss(residual, torch.zeros_like(residual),
                           delta=config.huber_delta, reduction="none")
    weights = (confidence[keep] * opacity[keep].clamp(0, 1)).detach()
    loss = (weights * penalty).sum() / weights.sum().clamp_min(torch.finfo(depth.dtype).tiny)
    error = (ratio.detach() - 1).abs()
    stats.update(sparse_depth_valid=int(keep.sum()), sparse_depth_loss=float(loss.detach()),
                 sparse_depth_median_relative_error=float(error.median()),
                 sparse_depth_too_near_fraction=float(
                     (ratio.detach() < 1 - config.near_error_fraction).float().mean()),
                 sparse_depth_mean_alpha=float(opacity[keep].mean()),
                 sparse_depth_confidence_sum=float(weights.sum()))
    return loss, stats
