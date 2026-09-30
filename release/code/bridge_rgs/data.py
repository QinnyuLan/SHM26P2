"""COLMAP and LabelMe input, deterministic view splits, prepared data loading.

Coordinates follow COLMAP/OpenCV: ``w2c`` maps world points to a camera whose
axes are right, down, forward. Semantic IDs are fixed by the competition.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = ["background", "deck", "stay_cable", "tower", "foundation"]
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def K(self) -> np.ndarray:
        if self.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            f, cx, cy = self.params[:3]
            fx, fy = f, f
        elif self.model in {"PINHOLE", "OPENCV"}:
            fx, fy, cx, cy = self.params[:4]
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {self.model}")
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    @property
    def distortion(self) -> np.ndarray:
        d = np.zeros(5, dtype=np.float64)
        if self.model == "SIMPLE_RADIAL":
            d[0] = self.params[3]
        elif self.model == "RADIAL":
            d[:2] = self.params[3:5]
        elif self.model == "OPENCV":
            d[:4] = self.params[4:8]
        elif self.model not in {"SIMPLE_PINHOLE", "PINHOLE"}:
            raise ValueError(f"Unsupported COLMAP camera model: {self.model}")
        return d


@dataclass
class ColmapImage:
    image_id: int
    name: str
    camera_id: int
    w2c: np.ndarray
    xy: np.ndarray
    point3d_ids: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return -self.w2c[:3, :3].T @ self.w2c[:3, 3]


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert COLMAP's scalar-first unit quaternion to a rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Zero COLMAP quaternion")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


def read_colmap_cameras(path: str | Path) -> dict[int, ColmapCamera]:
    result = {}
    expected = {"SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4,
                "RADIAL": 5, "OPENCV": 8}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        cid, model, width, height = int(fields[0]), fields[1], int(fields[2]), int(fields[3])
        if model not in expected:
            raise ValueError(f"Unsupported camera model {model}; do not silently discard distortion")
        params = np.asarray(fields[4:], dtype=np.float64)
        if len(params) != expected[model]:
            raise ValueError(f"Invalid parameter count for camera {cid}: {model}")
        result[cid] = ColmapCamera(cid, model, width, height, params)
    return result


def read_colmap_images(path: str | Path) -> dict[int, ColmapImage]:
    """Read paired pose/track lines, retaining empty observation lines."""
    result: dict[int, ColmapImage] = {}
    with Path(path).open() as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.strip().split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError(f"Invalid COLMAP pose line: {line[:120]}")
            image_id = int(fields[0])
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = quaternion_to_rotation(np.asarray(fields[1:5], dtype=np.float64))
            pose[:3, 3] = np.asarray(fields[5:8], dtype=np.float64)
            observations = stream.readline()
            if observations == "":
                raise ValueError(f"Missing observation line after image {image_id}")
            values = np.fromstring(observations.strip(), dtype=np.float64, sep=" ")
            if len(values) % 3:
                raise ValueError(f"Malformed observations for image {image_id}")
            values = values.reshape(-1, 3)
            if image_id in result:
                raise ValueError(f"Duplicate COLMAP image ID {image_id}")
            result[image_id] = ColmapImage(image_id, fields[9], int(fields[8]), pose,
                                         values[:, :2], values[:, 2].astype(np.int64))
    return result


def camera_based_split(images: dict[int, ColmapImage], val_every: int = 8,
                       seed: int = 42) -> dict[int, str]:
    """Hold out whole views, interleaved along the dominant camera-center axis.

    Uses only camera poses and names, never RGB or annotations. Interleaving is
    an interpolation validation protocol; it does not establish spatial OOD
    generalization. ``val_every=0`` explicitly requests all-view final fitting.
    """
    ordered = sorted(images.values(), key=lambda x: x.name)
    if val_every == 0:
        return {v.image_id: "train" for v in ordered}
    if val_every < 2:
        raise ValueError("val_every must be 0 (full fit) or >= 2")
    if len(ordered) < 3:
        raise ValueError("At least three cameras are required for a held-out split")
    centers = np.stack([v.center for v in ordered])
    centered = centers - np.mean(centers, axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    axis *= 1 if axis[np.argmax(np.abs(axis))] >= 0 else -1
    along = centered @ axis
    indices = np.lexsort((np.arange(len(ordered)), along))
    offset = seed % val_every
    val_ids = {ordered[indices[i]].image_id for i in range(offset, len(ordered), val_every)}
    if not val_ids:
        val_ids.add(ordered[indices[len(ordered)//2]].image_id)
    return {v.image_id: "val" if v.image_id in val_ids else "train" for v in ordered}


def rasterize_labelme(path: str | Path, width: int | None = None,
                      height: int | None = None) -> np.ndarray:
    """Rasterize in LabelMe draw order; later polygons take overlap precedence."""
    annotation = json.loads(Path(path).read_text())
    width = int(annotation["imageWidth"]) if width is None else width
    height = int(annotation["imageHeight"]) if height is None else height
    if (width, height) != (annotation["imageWidth"], annotation["imageHeight"]):
        raise ValueError(f"Image/annotation shape mismatch: {path}")
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for shape in annotation["shapes"]:
        label = shape["label"].strip()
        if label not in CLASS_IDS:
            raise ValueError(f"Unknown annotation label {label!r} in {path}")
        if shape.get("shape_type", "polygon") != "polygon":
            raise ValueError(f"Unsupported annotation shape in {path}: {shape.get('shape_type')}")
        if len(shape["points"]) < 3:
            raise ValueError(f"Polygon has fewer than three points: {path}")
        draw.polygon([tuple(p) for p in shape["points"]], fill=CLASS_IDS[label])
    return np.asarray(mask, dtype=np.uint8)


def undistortion_maps(camera: ColmapCamera, max_width: int | None = None,
                     pixel_protocol="legacy_mixed_v1"
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preserve native pinhole K/FOV, optionally downscale; no automatic crop."""
    from .coordinates import CORNER, corner_to_array_K
    from .coordinates import pixel_protocol as resolve_protocol
    protocol = resolve_protocol(pixel_protocol)
    scale = min(1., max_width / camera.width) if max_width else 1.
    width, height = round(camera.width * scale), round(camera.height * scale)
    K = camera.K.copy()
    K[0] *= width / camera.width
    K[1] *= height / camera.height
    if protocol == CORNER:
        if not np.any(camera.distortion) and (width, height) == (camera.width, camera.height):
            # Avoid negative floating-point tails at the exact identity border.
            map_x, map_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                      np.arange(height, dtype=np.float32))
        else:
            map_x, map_y = cv2.initUndistortRectifyMap(
                corner_to_array_K(camera.K), camera.distortion, None, corner_to_array_K(K),
                (width, height), cv2.CV_32FC1)
    else:
        map_x, map_y = cv2.initUndistortRectifyMap(camera.K, camera.distortion, None, K,
                                                 (width, height), cv2.CV_32FC1)
    valid = ((map_x >= 0) & (map_x <= camera.width - 1) &
             (map_y >= 0) & (map_y <= camera.height - 1)).astype(np.uint8) * 255
    return map_x, map_y, K, valid


def load_manifest(path: str | Path) -> dict[str, Any]:
    import hashlib

    from .coordinates import annotate_manifest
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    content = path.read_bytes()
    manifest = json.loads(content)
    if manifest["class_names"] != CLASS_NAMES:
        raise ValueError("Manifest semantic class ordering differs from competition mapping")
    for view in manifest["views"]:
        for key in ("image_path", "mask_path", "valid_path"):
            if view.get(key) and not Path(view[key]).is_absolute():
                view[key] = str((path.parent / view[key]).resolve())
    manifest = annotate_manifest(manifest)
    manifest["_manifest_sha256"] = hashlib.sha256(content).hexdigest()
    return manifest


class BridgeDataset:
    """Minimal lazy torch dataset. Missing labels are -1, never background.

    The list of views is split before any image is read. Callers should build
    training teachers using ``split='train'`` and optionally ``labeled_only``.
    """
    def __init__(self, manifest_path: str | Path, split: str = "train",
                 labeled_only: bool = False):
        self.manifest = load_manifest(manifest_path)
        self.views = [v for v in self.manifest["views"] if v["split"] == split
                      and (not labeled_only or v.get("mask_path"))]

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        view = self.views[index]
        rgb = np.array(Image.open(view["image_path"]).convert("RGB"), dtype=np.float32) / 255.
        valid = np.array(Image.open(view["valid_path"])) > 0
        mask = np.array(Image.open(view["mask_path"]), dtype=np.int64) if view.get("mask_path") else np.full(valid.shape, -1, dtype=np.int64)
        mask[~valid] = -1
        return {"image": torch.from_numpy(rgb).permute(2, 0, 1),
                "mask": torch.from_numpy(mask), "valid": torch.from_numpy(valid),
                "K": torch.tensor(view["K"], dtype=torch.float32),
                "w2c": torch.tensor(view["w2c"], dtype=torch.float32),
                "name": view["name"], "index": index, "has_mask": bool(view.get("mask_path"))}
