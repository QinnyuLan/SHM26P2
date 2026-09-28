"""Evaluate fixed local/global blending weights on the existing development split.

The outputs are development metrics, never a new test set. Each view is encoded
once per inference mode and evaluated at every predeclared probability weight.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bridge_rgs.coordinates import protocol_metadata
from bridge_rgs.evaluate import boundary_counts, boundary_scores
from bridge_rgs.teacher import (
    IGNORE_LABEL,
    ViewReader,
    checkpoint_adapter_options,
    confusion_metrics,
    file_sha256,
    load_checkpoint_adapters,
    load_teacher,
    predict_context_image,
    predict_image,
    require_teacher_manifest_protocol,
    require_teacher_renderer_protocol,
    verify_teacher_render_protocol,
)
from bridge_rgs.teacher_domains import validate_image_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--context-short-side", type=int, default=768)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--renderer-checkpoint", type=Path)
    parser.add_argument("--all-validation-views", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    if any(not 0 <= weight <= 1 for weight in args.weights):
        raise ValueError("Weights must be between zero and one")
    torch.set_num_threads(8)
    manifest = json.loads(args.manifest.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source = checkpoint["provenance"]
    profile = require_teacher_manifest_protocol(checkpoint, manifest, "teacher context evaluation")
    source_protocol = validate_image_sources(manifest)
    verify_teacher_render_protocol(manifest, source_protocol)
    compatible_hashes = {file_sha256(args.manifest)}
    if source_protocol:
        compatible_hashes.add(source_protocol["original_manifest_sha256"])
    if source["manifest_sha256"] not in compatible_hashes:
        raise ValueError("Checkpoint and manifest differ")
    if args.render_dir:
        if not args.renderer_checkpoint:
            raise ValueError("Rendered RGB diagnostics require the rendering checkpoint provenance")
        renderer = torch.load(args.renderer_checkpoint, map_location="cpu", weights_only=False)
        require_teacher_renderer_protocol(renderer, manifest,
                                          (source_protocol or {}).get("original_manifest_sha256", file_sha256(args.manifest)))
        del renderer
    model_dir = Path(source["model_dir"])
    if file_sha256(model_dir / "config.json") != source["model_config_sha256"]:
        raise ValueError("Backbone configuration differs")
    weights = {path.name: file_sha256(path) for path in sorted(model_dir.glob("*.safetensors"))}
    if weights != source["model_weights_sha256"]:
        raise ValueError("Backbone weights differ")
    classes = len(manifest["class_names"])
    model = load_teacher(
        model_dir, classes, checkpoint["configuration"]["channels"],
        pixel_profile=profile,
        **checkpoint_adapter_options(checkpoint["configuration"]),
    )
    model.decoder.load_state_dict(checkpoint["ema_decoder"])
    load_checkpoint_adapters(model, checkpoint)
    views = [view for view in manifest["views"] if view["split"] == "val" and (
        args.all_validation_views or view.get("mask_path")
    )]
    if args.render_dir:
        # Replace the image path BEFORE ViewReader sees a view. Real validation
        # photographs are never opened by the rendered-domain inference branch.
        views = [
            {**view, "image_path": str(args.render_dir.resolve() / (Path(view["name"]).stem + "_rgb.png"))}
            for view in views
        ]
    reader = ViewReader(classes)
    confusion = {weight: np.zeros((classes, classes), np.int64) for weight in args.weights}
    boundaries = {weight: np.zeros((classes, 4), np.int64) for weight in args.weights}
    per_view = {weight: [] for weight in args.weights}
    if args.save_predictions:
        for weight in args.weights:
            (args.output.parent / (args.output.stem + "_predictions") / str(weight)).mkdir(parents=True, exist_ok=True)
    for index, view in enumerate(views):
        image, target, valid = reader.read(view)
        local, _ = predict_image(
            model, image, tile_size=args.tile_size, stride=args.stride, flip=args.flip
        )
        context = predict_context_image(
            model, image, short_side=args.context_short_side, flip=args.flip
        )
        keep = valid & (target != IGNORE_LABEL)
        for weight in args.weights:
            prediction = ((1 - weight) * local + weight * context).argmax(0)
            encoded = target[keep].astype(np.int64) * classes + prediction[keep]
            local_confusion = np.bincount(encoded, minlength=classes**2).reshape(classes, classes)
            confusion[weight] += local_confusion
            boundaries[weight] += boundary_counts(prediction, target, keep, classes=classes)
            per_view[weight].append({"name": view["name"], **confusion_metrics(local_confusion)})
            if args.save_predictions:
                cv2.imwrite(str(args.output.parent / (args.output.stem + "_predictions") / str(weight) / view["name"]), prediction.astype(np.uint8))
        print(f"context comparison {index + 1}/{len(views)}: {view['name']}", flush=True)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_sha256": file_sha256(args.manifest),
        "pixel_protocol": protocol_metadata(profile),
        "teacher_training_manifest_sha256": source["manifest_sha256"],
        "step": checkpoint["step"],
        "views": [view["name"] for view in views],
        "labeled_views": [view["name"] for view in views if view.get("mask_path")],
        "protocol": "development validation; comparing predeclared context probability weights",
        "image_domain": "rendered_rgb" if args.render_dir or source_protocol else "real_rgb",
        "image_source_protocol": source_protocol,
        "renderer_checkpoint": str(args.renderer_checkpoint.resolve()) if args.renderer_checkpoint else (source_protocol or {}).get("renderer_checkpoint"),
        "renderer_checkpoint_sha256": file_sha256(args.renderer_checkpoint) if args.renderer_checkpoint else (source_protocol or {}).get("renderer_checkpoint_sha256"),
        "input_image_sha256": {view["name"]: file_sha256(view["image_path"]) for view in views},
        "inference_source_sha256": file_sha256(Path(__file__)),
        "flip": args.flip,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "context_short_side": args.context_short_side,
        "metrics": {
            str(weight): {
                **confusion_metrics(value),
                "miou_foreground": float(np.mean(confusion_metrics(value)["per_class_iou"][1:])),
                "boundary_f1_2px": boundary_scores(boundaries[weight]),
                "boundary_counts": boundaries[weight].tolist(),
                "per_view": per_view[weight],
            }
            for weight, value in confusion.items()
        },
        "boundary_protocol": "bridge_rgs.evaluate: inner 1px, Chebyshev tolerance 2, valid erosion 7x7",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
