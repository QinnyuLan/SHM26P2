"""Export the fixed 350 TRAIN cameras for teacher rendering-domain adaptation.

Only manifest camera metadata and a Gaussian checkpoint are inputs to rendering.
Original RGB, label, and validity files are never opened. Generated semantic masks
are incidental renderer outputs and must never be used as teacher ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def training_cameras(manifest: dict) -> list[dict]:
    """Whitelist camera fields, with strict current-dataset split/grid checks."""
    from bridge_rgs.coordinates import annotate_manifest

    annotate_manifest(manifest)
    views = manifest["views"]
    all_names = [view["name"] for view in views]
    if len(set(all_names)) != len(all_names):
        raise ValueError("Manifest contains duplicate view names")
    train = [view for view in views if view["split"] == "train"]
    if len(train) != 350:
        raise ValueError(f"Expected exactly 350 TRAIN cameras, got {len(train)}")
    names = [view["name"] for view in train]
    if any(Path(name).name != name for name in names):
        raise ValueError("Camera names must be basenames")
    if len({Path(name).stem for name in names}) != len(names):
        raise ValueError("Output filename stems collide")
    cameras = []
    for view in train:
        if (view["width"], view["height"]) != (1320, 989):
            raise ValueError(f"Expected native 1320x989 grid: {view['name']}")
        # Do not copy any image/mask/valid/source paths or the whole view record.
        camera = {key: view[key] for key in (
            "name", "image_id", "camera_id", "width", "height", "K", "w2c_original"
        )}
        camera["w2c"] = camera["w2c_original"]
        cameras.append(camera)
    return cameras


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/prepared/manifest.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/strong_rgb/last.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/strong_rgb/train_render"))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)

    import cv2
    import torch
    from PIL import Image

    import bridge_rgs.evaluate as evaluation
    from bridge_rgs.coordinates import protocol_metadata, require_matching_protocol
    from bridge_rgs.teacher import require_teacher_renderer_protocol

    cv2.setNumThreads(args.threads)
    torch.set_num_threads(args.threads)
    manifest_path = args.manifest.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    manifest_hash, checkpoint_hash = sha256(manifest_path), sha256(checkpoint)
    manifest = json.loads(manifest_path.read_text())
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    profile = require_teacher_renderer_protocol(state, manifest, manifest_hash)
    del state
    cameras = training_cameras(manifest)
    train_names = [camera["name"] for camera in cameras]
    train_set = set(train_names)
    excluded_names = {view["name"] for view in manifest["views"] if view["split"] != "train"}
    if train_set & excluded_names:
        raise ValueError("Non-training camera leakage")
    output = args.output.resolve()
    # Do not overwrite an earlier export or permit stale validation outputs.
    for path in (output / "rgb", output / "mask"):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Refusing existing image outputs: {path}")
    for filename in ("source_receipt.json", "camera_only.json", "render_receipt.json"):
        if (output / filename).exists():
            raise FileExistsError(output / filename)
    output.mkdir(parents=True, exist_ok=True)
    camera_path = output / "camera_only.json"
    write_json(camera_path, {"pixel_protocol": protocol_metadata(profile),
                             "source_manifest_sha256": manifest_hash, "views": cameras})
    package_path = Path(evaluation.__file__).resolve().parent
    source_files = [package_path / name for name in (
        "evaluate.py", "model.py", "refinement.py", "train.py", "io.py", "data.py",
        "coordinates.py", "teacher.py"
    )]
    source_hashes = {path.name: sha256(path) for path in source_files}
    receipt = {
        "status": "running", "started_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(), "threads": args.threads,
        "source_manifest": str(manifest_path), "source_manifest_sha256": manifest_hash,
        "pixel_protocol": protocol_metadata(profile),
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "camera_only_input": str(camera_path), "camera_only_input_sha256": sha256(camera_path),
        "camera_only_fields": sorted(cameras[0]), "grid": "pinhole",
        "train_view_names": train_names, "train_view_count": len(cameras),
        "expected_size": [1320, 989], "source_package": str(package_path),
        "source_hashes": source_hashes, "script_sha256": sha256(Path(__file__).resolve()),
        "pixel_input_policy": "No original RGB, GT mask, or validity pixels are read.",
        "generated_mask_policy": "Incidental output only; forbidden as teacher labels.",
    }
    receipt_path = output / "source_receipt.json"
    write_json(receipt_path, receipt)
    print(json.dumps({"status": "rendering", "pid": os.getpid(),
                      "views": len(cameras), "output": str(output)}), flush=True)
    try:
        rendered = evaluation.render_cameras(checkpoint, camera_path, output, grid="pinhole")
        require_matching_protocol(rendered, profile, "teacher training render result")
        if rendered["views"] != 350 or {r["name"] for r in rendered["records"]} != train_set:
            raise AssertionError("Renderer receipt is not the strict training camera set")
        expected_files = {f"{Path(name).stem}.png" for name in train_names}
        if {path.name for path in (output / "rgb").iterdir()} != expected_files:
            raise AssertionError("Rendered RGB directory contains missing or unexpected files")
        records = []
        for camera in cameras:
            rgb_path = output / "rgb" / f"{Path(camera['name']).stem}.png"
            # Only generated image headers are inspected; never load original pixels.
            with Image.open(rgb_path) as image:
                if image.size != (1320, 989) or image.mode != "RGB":
                    raise AssertionError(f"Unexpected rendered RGB grid/mode: {rgb_path}")
            records.append({"name": camera["name"], "image_path": str(rgb_path),
                            "sha256": sha256(rgb_path), "width": 1320, "height": 989})
        if sha256(manifest_path) != manifest_hash or sha256(checkpoint) != checkpoint_hash:
            raise AssertionError("Manifest or checkpoint changed during export")
        if source_hashes != {path.name: sha256(path) for path in source_files}:
            raise AssertionError("Rendering source changed during export")
        receipt.update(status="completed", records=records, strict_train_set_verified=True,
                       output_dimensions_verified=True,
                       render_receipt_sha256=sha256(output / "render_receipt.json"))
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["finished_utc"] = datetime.now(UTC).isoformat()
        write_json(receipt_path, receipt)
    print(json.dumps({"status": "completed", "views": len(records),
                      "source_receipt": str(receipt_path)}), flush=True)


if __name__ == "__main__":
    main()
