"""Compare pre/post adaptation under identical camera-rendered RGB inputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from bridge_rgs.coordinates import protocol_metadata, require_matching_protocol
from bridge_rgs.teacher import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-weight", default="0.25")
    args = parser.parse_args()
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    require_matching_protocol(before, after, "teacher rendered-domain comparison")
    keys = ("manifest_sha256", "views", "labeled_views", "image_domain",
            "renderer_checkpoint_sha256", "input_image_sha256", "flip",
            "tile_size", "stride", "context_short_side", "boundary_protocol")
    for key in keys:
        if before[key] != after[key]:
            raise ValueError(f"Evaluation protocol differs: {key}")
    if before["image_domain"] != "rendered_rgb":
        raise ValueError("This comparison requires camera-rendered RGB inputs")
    a, b = (value["metrics"][args.context_weight] for value in (before, after))
    for row_a, row_b in zip(a["per_view"], b["per_view"], strict=True):
        if row_a["name"] != row_b["name"] or not np.array_equal(
            np.array(row_a["confusion"]).sum(1), np.array(row_b["confusion"]).sum(1)
        ):
            raise ValueError("Scored annotation pixels differ")
    result = {
        "role": "DINOv3 post-render semantic segmentation domain-adaptation control",
        "pixel_protocol": protocol_metadata(before),
        "before": str(args.before.resolve()), "before_sha256": file_sha256(args.before),
        "after": str(args.after.resolve()), "after_sha256": file_sha256(args.after),
        "before_teacher_checkpoint_sha256": before["checkpoint_sha256"],
        "after_teacher_checkpoint_sha256": after["checkpoint_sha256"],
        "renderer_checkpoint_sha256": before["renderer_checkpoint_sha256"],
        "identical_protocol_keys": list(keys),
        "native_grid": [1320, 989], "inferred_camera_views": len(before["views"]),
        "scored_labeled_views": len(before["labeled_views"]),
        "context_weight": float(args.context_weight),
        "miou_before": a["miou"], "miou_after": b["miou"],
        "miou_delta_percentage_points": 100 * (b["miou"] - a["miou"]),
        "foreground_miou_before": a["miou_foreground"],
        "foreground_miou_after": b["miou_foreground"],
        "per_class_iou_before": a["per_class_iou"],
        "per_class_iou_after": b["per_class_iou"],
        "per_class_iou_delta_percentage_points": (100 * (
            np.array(b["per_class_iou"]) - np.array(a["per_class_iou"])
        )).tolist(),
        "boundary_f1_2px_before": a["boundary_f1_2px"],
        "boundary_f1_2px_after": b["boundary_f1_2px"],
        "scope": "Fixed interpolation development split, including checkpoint selection. RGB reconstruction is unchanged. The difference includes extra optimization; without a matched extra-budget real-only arm it does not isolate a causal domain-adaptation benefit. This engineering baseline is not proof of novel 3D semantics or independent-test generalization.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
