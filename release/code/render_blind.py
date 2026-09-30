#!/usr/bin/env python3
"""Render RGB and semantic-ID images from camera parameters only.

This wrapper deliberately does not import dataset loaders or evaluation code.
It validates the external camera contract, calls the audited renderer, and
checks the written files before returning success.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RELEASE_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_CODE))
sys.path.insert(1, str(ROOT / "src"))

from bridge_rgs.evaluate import render_cameras


def _finite_array(value, shape, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return array


def validate_cameras(path: Path, grid: str) -> list[dict]:
    """Validate camera metadata without opening any target image or label."""
    payload = json.loads(path.read_text())
    records = payload.get("views") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("Camera JSON must be a non-empty list or an object with a views list")
    normalized = []
    names = set()
    for index, original in enumerate(records):
        if not isinstance(original, dict):
            raise TypeError(f"camera {index} is not an object")
        required = {"name", "K", "w2c", "width", "height"}
        missing = sorted(required - original.keys())
        if missing:
            raise ValueError(f"camera {index} is missing {missing}")
        name = str(original["name"])
        if not name or Path(name).name != name or Path(name).stem in names:
            raise ValueError(f"camera {index} has a duplicate or unsafe name: {name!r}")
        names.add(Path(name).stem)
        K = _finite_array(original["K"], (3, 3), f"camera {name} K")
        pose = _finite_array(original["w2c"], (4, 4), f"camera {name} w2c")
        if not np.allclose(K[2], [0.0, 0.0, 1.0]) or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError(f"camera {name} has invalid pinhole intrinsics")
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError(f"camera {name} has an invalid homogeneous pose")
        width, height = int(original["width"]), int(original["height"])
        if width != original["width"] or height != original["height"] or min(width, height) < 11:
            raise ValueError(f"camera {name} has invalid dimensions")
        distortion = original.get("distortion", [])
        if grid == "official":
            distortion = _finite_array(distortion, (5,), f"camera {name} distortion").tolist()
        else:
            distortion = []
        normalized.append(
            {
                "name": name,
                "K": K.tolist(),
                "w2c": pose.tolist(),
                "width": width,
                "height": height,
                "distortion": distortion,
            }
        )
    return normalized


def verify_outputs(output: Path, records: list[dict]) -> None:
    """Check output dimensions and enforce the five-class ID contract."""
    for record in records:
        stem = Path(record["name"]).stem
        rgb_path, mask_path = output / "rgb" / f"{stem}.png", output / "mask" / f"{stem}.png"
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or mask is None or rgb.shape[:2] != mask.shape[:2]:
            raise RuntimeError(f"missing or mismatched output for {stem}")
        if rgb.shape[1] != record["width"] or rgb.shape[0] != record["height"]:
            raise RuntimeError(f"unexpected RGB dimensions for {stem}")
        labels = np.unique(mask)
        if mask.ndim != 2 or np.any(~np.isin(labels, np.arange(5, dtype=np.uint8))):
            raise RuntimeError(f"semantic IDs outside 0..4 for {stem}: {labels.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", choices=("official", "pinhole"), default="official")
    parser.add_argument("--max-views", type=int)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.cameras.is_file():
        raise FileNotFoundError(args.cameras)
    if args.max_views is not None and args.max_views < 1:
        raise ValueError("--max-views must be positive")
    records = validate_cameras(args.cameras, args.grid)
    if args.max_views:
        records = records[: args.max_views]
    args.output.mkdir(parents=True, exist_ok=True)
    # A temporary normalized JSON prevents auxiliary paths in a user-supplied
    # object from being interpreted as image or annotation inputs.
    with tempfile.TemporaryDirectory(prefix="bridge_rgs_cameras_") as folder:
        camera_file = Path(folder) / "cameras.json"
        camera_file.write_text(json.dumps({"views": records}, indent=2))
        render_cameras(args.checkpoint, camera_file, args.output, args.grid, len(records))
    verify_outputs(args.output, records)
    receipt = json.loads((args.output / "render_receipt.json").read_text())
    receipt.update(
        {
            "release_entrypoint": str(Path(__file__).resolve()),
            "target_rgb_or_labels_read": False,
            "verified_semantic_id_range": [0, 4],
            "verified_views": len(records),
        }
    )
    (args.output / "render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "views": len(records),
                "output": str(args.output.resolve()),
                "rgb_dir": str((args.output / "rgb").resolve()),
                "mask_dir": str((args.output / "mask").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
