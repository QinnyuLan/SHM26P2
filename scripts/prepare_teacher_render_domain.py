"""Create an image-only manifest for supervised real/rendered teacher adaptation.

Validation RGB is freshly generated from camera metadata only, on all 50 held-out
cameras. The original validation photographs and all label pixels stay unopened.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
from PIL import Image

from bridge_rgs.coordinates import protocol_metadata, require_matching_protocol
from bridge_rgs.evaluate import render_cameras
from bridge_rgs.teacher import (
    file_sha256,
    require_teacher_renderer_protocol,
    verify_teacher_render_protocol,
)
from bridge_rgs.teacher_domains import validate_image_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/prepared/manifest.json"))
    parser.add_argument("--train-receipt", type=Path, default=Path("runs/strong_rgb/train_render/source_receipt.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/teacher_render_adapt_v1"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    original_path = args.manifest.resolve()
    original = json.loads(original_path.read_text())
    receipt_path = args.train_receipt.resolve()
    train_receipt = json.loads(receipt_path.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Derived manifest already exists; keep its lineage immutable")
    train = [v for v in original["views"] if v["split"] == "train"]
    val = [v for v in original["views"] if v["split"] == "val"]
    if len(train) != 350 or len(val) != 50 or sum(bool(v.get("mask_path")) for v in train) != 259:
        raise ValueError("Expected fixed 350/50 split and 259 train annotations")
    if train_receipt["source_manifest_sha256"] != file_sha256(original_path):
        raise ValueError("Training renderer used a different manifest")
    require_matching_protocol(train_receipt, original, "teacher training render receipt")
    checkpoint = Path(train_receipt["checkpoint"])
    if train_receipt["checkpoint_sha256"] != file_sha256(checkpoint):
        raise ValueError("Renderer checkpoint changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    profile = require_teacher_renderer_protocol(state, original, file_sha256(original_path))
    del state
    cameras = [{key: view[key] for key in (
        "name", "image_id", "camera_id", "width", "height", "K", "w2c", "w2c_original"
    )} for view in val]
    val_output = output / "validation_render"
    val_output.mkdir(exist_ok=True)
    camera_path = val_output / "camera_only.json"
    camera_path.write_text(json.dumps({"pixel_protocol": protocol_metadata(profile),
                                       "source_manifest_sha256": file_sha256(original_path),
                                       "views": cameras}, indent=2) + "\n")
    rendered = render_cameras(checkpoint, camera_path, val_output, grid="pinhole")
    require_matching_protocol(rendered, profile, "teacher validation render result")
    if rendered["views"] != 50:
        raise AssertionError("Missing validation camera renderings")
    records = []
    for view in val:
        image_path = val_output / "rgb" / (Path(view["name"]).stem + ".png")
        with Image.open(image_path) as image:
            if image.size != (view["width"], view["height"]) or image.mode != "RGB":
                raise AssertionError("Incorrect rendered validation RGB grid")
        records.append({"name": view["name"], "image_path": str(image_path),
                        "sha256": file_sha256(image_path),
                        "width": view["width"], "height": view["height"]})
    val_receipt = {
        "status": "completed", "grid": "pinhole",
        "pixel_protocol": protocol_metadata(profile),
        "source_manifest": str(original_path), "source_manifest_sha256": file_sha256(original_path),
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "camera_only_input": str(camera_path), "camera_only_input_sha256": file_sha256(camera_path),
        "records": records, "script_sha256": file_sha256(Path(__file__)),
        "input_policy": "Camera metadata only; no original RGB, annotation or validity pixels read",
        "generated_mask_policy": "Generated masks are incidental and forbidden as teacher labels",
    }
    val_receipt_path = val_output / "source_receipt.json"
    val_receipt_path.write_text(json.dumps(val_receipt, indent=2) + "\n")
    by_name = {r["name"]: r for r in train_receipt["records"] + records}
    derived = deepcopy(original)
    for view in derived["views"]:
        row = by_name[view["name"]]
        view["image_path_sources"] = {"rendered": {"path": row["image_path"], "sha256": row["sha256"]}}
        if view["split"] == "train":
            view["image_path_sources"]["real"] = {
                "path": view["image_path"], "sha256": file_sha256(view["image_path"])
            }
            view["image_domain"] = "mixed_real_rendered"
        else:
            view["image_path"] = row["image_path"]
            view["image_domain"] = "rendered_rgb"
    derived["image_source_protocol"] = {
        "pixel_protocol": protocol_metadata(profile),
        "original_manifest": str(original_path), "original_manifest_sha256": file_sha256(original_path),
        "renderer_checkpoint": str(checkpoint), "renderer_checkpoint_sha256": file_sha256(checkpoint),
        "train_render_receipt": {"path": str(receipt_path), "sha256": file_sha256(receipt_path)},
        "val_render_receipt": {"path": str(val_receipt_path), "sha256": file_sha256(val_receipt_path)},
        "policy": "Only image inputs change; exact original names/splits/labels/validity/cameras preserved; validation input is rendered RGB only",
    }
    validate_image_sources(derived)
    verify_teacher_render_protocol(derived, derived["image_source_protocol"])
    destination = output / "manifest.json"
    destination.write_text(json.dumps(derived, indent=2) + "\n")
    print(json.dumps({"manifest": str(destination), "sha256": file_sha256(destination),
                      "train": len(train), "validation": len(val),
                      "validation_input": "camera-rendered RGB only"}), flush=True)


if __name__ == "__main__":
    main()
