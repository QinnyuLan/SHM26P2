"""Prepare undistorted bridge views and a robust, train-only point seed.

Run ``python -m bridge_rgs.prepare --help`` for the independent data pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .coordinates import CORNER, LEGACY, protocol_metadata
from .coordinates import pixel_protocol as resolve_protocol
from .data import (
    CLASS_NAMES,
    ColmapImage,
    camera_based_split,
    rasterize_labelme,
    read_colmap_cameras,
    read_colmap_images,
    undistortion_maps,
)
from .geometry import (
    TriangulatedCloud,
    bounded_bundle_adjustment,
    camera_uncertainty,
    collect_train_tracks,
    triangulate_tracks,
)


def _find_images(dataset_root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for dirname in ("images", "unlabeled_Images", "unlabeled_images"):
        folder = dataset_root / dirname
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            if path.name in images:
                raise ValueError(f"Duplicate image name in dataset: {path.name}")
            images[path.name] = path.resolve()
    return images


def _cloud_appearance(cloud: TriangulatedCloud, views: list[dict[str, Any]],
                      max_samples: int = 8, pixel_protocol=LEGACY) -> tuple[np.ndarray, np.ndarray]:
    """Median color and class evidence sampled solely from training views."""
    samples = np.full((len(cloud.points), max_samples, 3), np.nan, dtype=np.float32)
    protocol = resolve_protocol(pixel_protocol)
    class_counts = np.zeros((len(cloud.points), len(CLASS_NAMES)), dtype=np.float32)
    by_view: dict[int, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
    view_lookup = {v["image_id"]: v for v in views}
    for point_index, observations in enumerate(cloud.observations):
        choose = np.linspace(0, len(observations)-1, min(len(observations), max_samples)).astype(int)
        for slot, oi in enumerate(choose):
            obs = observations[oi]
            if view_lookup[obs.image_id]["split"] != "train":
                raise AssertionError("Held-out view in point appearance sampling")
            by_view[obs.image_id].append((point_index, slot, obs.xy_normalized))
    for image_id, requests in by_view.items():
        view = view_lookup[image_id]
        K = np.asarray(view["K"])
        image = np.asarray(Image.open(view["image_path"]).convert("RGB"))
        valid = np.asarray(Image.open(view["valid_path"])) > 0
        normalized = np.stack([x[2] for x in requests])
        pixels = normalized * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
        if protocol == CORNER:
            inside = (np.isfinite(pixels).all(1)
                      & (pixels[:, 0] >= 0) & (pixels[:, 0] < view["width"])
                      & (pixels[:, 1] >= 0) & (pixels[:, 1] < view["height"]))
            pixels = np.floor(np.where(inside[:, None], pixels, 0)).astype(int)
        else:
            pixels = np.rint(pixels).astype(int)
            inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < view["width"]) &
                      (pixels[:, 1] >= 0) & (pixels[:, 1] < view["height"]))
        slots = np.array([r[1] for r in requests])[inside]
        point_indices = np.array([r[0] for r in requests])[inside]
        pixels = pixels[inside]
        keep = valid[pixels[:, 1], pixels[:, 0]]
        point_indices, slots, pixels = point_indices[keep], slots[keep], pixels[keep]
        samples[point_indices, slots] = image[pixels[:, 1], pixels[:, 0]] / 255.
        if view["mask_path"]:
            mask = np.asarray(Image.open(view["mask_path"]))
            labels = mask[pixels[:, 1], pixels[:, 0]]
            np.add.at(class_counts, (point_indices, labels), 1.)
    missing = np.isnan(samples).all(axis=(1, 2))
    samples[missing, 0] = .5
    colors = np.nanmedian(samples, axis=1)
    return colors.astype(np.float32), class_counts


def prepare_dataset(dataset_root: str | Path, output_dir: str | Path,
                    max_width: int | None = 1320, val_every: int = 8, seed: int = 42,
                    min_track_length: int = 3, max_points: int = 100000,
                    max_reprojection_error: float = 3.,
                    min_parallax_degrees: float = .5,
                    bundle_adjustment: bool = False, ba_max_points: int = 500,
                    ba_max_nfev: int = 20, workers: int = 8,
                    pixel_protocol=LEGACY) -> Path:
    """Create ``manifest.json``, train-only ``init_points.npz`` and audit JSON.

    Returns the absolute manifest path. Split decisions precede geometry and
    pixel reading. Validation files are converted to the same pinhole grid for
    later evaluation, but never consumed by geometry or appearance estimation.
    """
    if max_width is not None and max_width <= 0:
        raise ValueError("max_width must be positive or None")
    if min_track_length < 2 or max_points < 1:
        raise ValueError("min_track_length >= 2 and max_points >= 1 are required")
    dataset_root, output_dir = Path(dataset_root).resolve(), Path(output_dir).resolve()
    protocol = resolve_protocol(pixel_protocol)
    previous_manifest = output_dir / "manifest.json"
    if previous_manifest.is_file():
        previous_protocol = resolve_protocol(json.loads(previous_manifest.read_text()))
        if previous_protocol != protocol:
            raise FileExistsError("Cannot overwrite a prepared directory with a different pixel protocol")
    if protocol == CORNER and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("corner_v2 preparation requires a new or empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("images", "masks", "valid"):
        (output_dir / folder).mkdir(exist_ok=True)
    cameras = read_colmap_cameras(dataset_root / "camera_parameters/cameras.txt")
    images = read_colmap_images(dataset_root / "camera_parameters/images.txt")
    image_files = _find_images(dataset_root)
    missing = sorted(v.name for v in images.values() if v.name not in image_files)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} posed images, first: {missing[:5]}")
    splits = camera_based_split(images, val_every=val_every, seed=seed)
    split_payload = {v.name: splits[i] for i, v in sorted(images.items(), key=lambda kv: kv[1].name)}
    split_hash = hashlib.sha256(json.dumps(split_payload, sort_keys=True).encode()).hexdigest()
    print(f"Preparing {len(images)} views: {dict(Counter(splits.values()))}, split {split_hash[:12]}", flush=True)
    maps = {cid: undistortion_maps(camera, max_width, protocol) for cid, camera in cameras.items()}
    for cid, (_, _, _, valid) in maps.items():
        if not cv2.imwrite(str(output_dir / "valid" / f"camera_{cid}.png"), valid):
            raise OSError("Failed to write validity map")

    def prepare_view(view: ColmapImage) -> dict[str, Any]:
        camera = cameras[view.camera_id]
        mx, my, K, valid = maps[view.camera_id]
        source = image_files[view.name]
        rgb = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if rgb is None or rgb.shape[:2] != (camera.height, camera.width):
            raise ValueError(f"Invalid image or camera shape mismatch: {source}")
        undistorted = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out_name = f"{Path(view.name).stem}.png"
        image_path = output_dir / "images" / out_name
        if not cv2.imwrite(str(image_path), undistorted, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
            raise OSError(f"Failed to write {image_path}")
        annotation = dataset_root / "json" / f"{Path(view.name).stem}.json"
        mask_path = None
        if annotation.exists():
            raw_mask = rasterize_labelme(annotation, camera.width, camera.height)
            mask = cv2.remap(raw_mask, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            mask[valid == 0] = 255
            mask_path = output_dir / "masks" / out_name
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError(f"Failed to write {mask_path}")
        return {"name": view.name, "image_id": view.image_id, "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path else None,
                "valid_path": str(output_dir / "valid" / f"camera_{view.camera_id}.png"),
                "width": valid.shape[1], "height": valid.shape[0], "K": K.tolist(),
                "w2c": view.w2c.tolist(), "w2c_original": view.w2c.tolist(),
                "split": splits[view.image_id], "camera_id": view.camera_id,
                "source_image_path": str(source),
                "source_annotation_path": str(annotation) if annotation.exists() else None}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        views = list(pool.map(prepare_view, sorted(images.values(), key=lambda v: v.name)))
    tracks = collect_train_tracks(images, cameras, splits)
    print(f"Triangulating {len(tracks)} tracks from training observations only", flush=True)
    cloud = triangulate_tracks(tracks, images, cameras, min_track_length=min_track_length,
                               max_points=max_points, max_reprojection_error=max_reprojection_error,
                               min_parallax_degrees=min_parallax_degrees, seed=seed)
    if not len(cloud.points):
        raise RuntimeError("No valid train-only triangulation; inspect camera conventions or thresholds")
    pose_overrides: dict[int, np.ndarray] = {}
    ba_audit: dict[str, Any] = {"enabled": False}
    if bundle_adjustment:
        print(f"Bounded train-only bundle adjustment ({ba_max_points} points maximum)", flush=True)
        pose_overrides, ba_audit = bounded_bundle_adjustment(cloud, images, cameras, splits,
                                                            max_points=ba_max_points,
                                                            max_nfev=ba_max_nfev, seed=seed)
        if pose_overrides:
            cloud = triangulate_tracks(tracks, images, cameras, min_track_length=min_track_length,
                                       max_points=max_points, max_reprojection_error=max_reprojection_error,
                                       min_parallax_degrees=min_parallax_degrees, seed=seed,
                                       pose_overrides=pose_overrides)
            if not len(cloud.points):
                raise RuntimeError("Camera refinement invalidated all points")
    covariance, uncertainty_audit = camera_uncertainty(cloud, images, cameras, splits, pose_overrides)
    for view in views:
        image_id = view["image_id"]
        view["w2c"] = pose_overrides.get(image_id, images[image_id].w2c).tolist()
        # The differentiable projection/fusion API uses translation first.
        covariance_order = [3, 4, 5, 0, 1, 2]
        view["pose_covariance"] = covariance[image_id][np.ix_(covariance_order, covariance_order)].tolist()
        view["pose_covariance_order"] = ["tx", "ty", "tz", "rx", "ry", "rz"]
        view["pose_uncertainty_source"] = "train_tracks_conditional" if splits[image_id] == "train" else "prior_only"
    colors, semantic_counts = _cloud_appearance(cloud, views, pixel_protocol=protocol)
    observation_ids = np.array([o.image_id for observations in cloud.observations for o in observations], dtype=np.int64)
    if any(splits[int(i)] != "train" for i in observation_ids):
        raise AssertionError("Validation leakage into exported point cloud")
    extra_arrays = {"pixel_protocol": np.asarray(protocol)} if protocol == CORNER else {}
    np.savez_compressed(output_dir / "init_points.npz", points=cloud.points.astype(np.float32),
                        colors=colors, covariances=cloud.covariances.astype(np.float32),
                        track_ids=cloud.track_ids, reprojection_error=cloud.reprojection_error.astype(np.float32),
                        num_observations=np.array([len(o) for o in cloud.observations], dtype=np.int32),
                        observation_image_ids=observation_ids,
                        observation_offsets=np.concatenate([[0], np.cumsum([len(o) for o in cloud.observations])]),
                        semantic_counts=semantic_counts,
                        scene_radius=np.array(uncertainty_audit["scene_radius"], dtype=np.float32),
                        **extra_arrays)
    counts = Counter((v["split"], bool(v["mask_path"])) for v in views)
    audit = {"dataset_root": str(dataset_root), "view_count": len(views),
             "split_counts": {f"{split}_{'labeled' if labeled else 'unlabeled'}": count
                              for (split, labeled), count in sorted(counts.items())},
             "split_sha256": split_hash, "triangulation": cloud.audit, "bundle_adjustment": ba_audit,
             "uncertainty": uncertainty_audit,
             "point_reprojection_native_px": {"mean": float(cloud.reprojection_error.mean()),
                                               "median": float(np.median(cloud.reprojection_error)),
                                               "p95": float(np.percentile(cloud.reprojection_error, 95))},
             "validation_used_in_triangulation": False,
             "validation_used_in_bundle_adjustment": False,
             "validation_used_in_point_appearance": False,
             "upstream_camera_caveat": "Supplied COLMAP poses, intrinsics and track IDs were estimated upstream using all released views. Evaluation is held-out image supervision conditional on that supplied calibration, not a fully inductive SfM benchmark."}
    manifest = {"schema_version": 1, "class_names": CLASS_NAMES, "ignore_label": 255,
                "dataset_root": str(dataset_root), "views": views,
                "init_points_path": str(output_dir / "init_points.npz"),
                "geometry_audit_path": str(output_dir / "geometry_audit.json"),
                "coordinate_convention": "COLMAP/OpenCV world-to-camera; +x right, +y down, +z forward",
                "pixel_grid": "native PINHOLE K (optionally scaled), same grid for RGB/mask/validity",
                "scene_radius": uncertainty_audit["scene_radius"],
                "pose_covariance_order": ["tx", "ty", "tz", "rx", "ry", "rz"],
                "split": {"method": "camera_center_pca_interleaved" if val_every else "all_view_final_fit",
                          "val_every": val_every, "seed": seed, "sha256": split_hash,
                          "interpretation": "interpolation validation" if val_every else "training-view fitting only"},
                "source_cameras": {str(cid): {"model": c.model, "width": c.width, "height": c.height,
                                               "params": c.params.tolist(), "K": c.K.tolist(),
                                               "opencv_distortion": c.distortion.tolist()}
                                   for cid, c in cameras.items()},
                "preparation": {"max_width": max_width, "min_track_length": min_track_length,
                                "max_points": max_points, "max_reprojection_error": max_reprojection_error,
                                "min_parallax_degrees": min_parallax_degrees,
                                "bundle_adjustment": bundle_adjustment, "ba_max_points": ba_max_points,
                                "ba_max_nfev": ba_max_nfev},
                "protocol_caveat": audit["upstream_camera_caveat"]}
    if protocol == CORNER:
        manifest.update(schema_version=2, pixel_protocol=protocol_metadata(protocol))
        audit["pixel_protocol"] = protocol_metadata(protocol)
    (output_dir / "geometry_audit.json").write_text(json.dumps(audit, indent=2))
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = output_dir / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2))
    temporary_manifest.replace(manifest_path)
    print(f"Prepared {len(cloud.points)} points, median error {np.median(cloud.reprojection_error):.3f}px: {manifest_path}", flush=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", "--dataset", default="Dataset")
    parser.add_argument("--output-dir", "--output", default="outputs/prepared")
    parser.add_argument("--max-width", type=int, default=1320)
    parser.add_argument("--val-every", type=int, default=8, help="0 explicitly fits all views")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-track-length", type=int, default=3)
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--max-reprojection-error", type=float, default=3.)
    parser.add_argument("--min-parallax-degrees", type=float, default=.5)
    parser.add_argument("--bundle-adjustment", action="store_true")
    parser.add_argument("--ba-max-points", type=int, default=500)
    parser.add_argument("--ba-max-nfev", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare_dataset(**vars(args))


if __name__ == "__main__":
    main()
