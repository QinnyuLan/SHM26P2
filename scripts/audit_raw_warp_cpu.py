"""Synthetic CPU check on a fixed TRAIN camera map; no image/label pixels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from bridge_rgs.evaluate import distortion_render_grid
from bridge_rgs.raw_grid import build_raw_grid

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/raw_warp_cpu_contract.json"


def difference(first, second):
    delta = np.abs(first.astype(np.float64) - second.astype(np.float64))
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "p99_abs": float(np.quantile(delta, .99)),
            "rounded_uint8_unequal_elements": int(np.count_nonzero(np.rint(first*255) != np.rint(second*255)))}


def main():
    if OUTPUT.exists():
        raise FileExistsError("Refuse to overwrite CPU warp evidence")
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    plan_path = ROOT / "artifacts/raw_grid_training_diagnostic_plan.json"
    plan = json.loads(plan_path.read_text())
    view = next(v for v in plan["views"] if v["name"] == "002.png")
    assert view["split"] == "train"
    camera = plan["source_cameras"][str(view["camera_id"])]
    layout = build_raw_grid(camera["K"], camera["opencv_distortion"], camera["width"], camera["height"])
    _, _, _, mapping = distortion_render_grid(camera["K"], camera["opencv_distortion"],
                                               camera["width"], camera["height"], "colmap_corner_v2")
    assert mapping is not None and layout.warp.weight_policy == "continuous_float32"
    mx, my = mapping[..., 0], mapping[..., 1]
    base, fractions = cv2.convertMaps(mx, my, cv2.CV_16SC2, nninterpolation=False)
    shape = (layout.render_height, layout.render_width)
    yy, xx = np.indices(shape, dtype=np.float32)
    rng = np.random.default_rng(42)
    patterns = {"linear_ramp": np.stack((xx/shape[1], yy/shape[0], (xx+yy)/sum(shape)), -1),
                "checkerboard": np.repeat(((xx+yy) % 2)[..., None], 3, -1),
                "uniform_random_seed42": rng.random((*shape, 3), dtype=np.float32)}
    grid = torch.from_numpy(mapping.copy())
    grid = (grid + .5) * (2 / torch.tensor([layout.render_width, layout.render_height])) - 1
    results = {}
    for name, image in patterns.items():
        actual_cv = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        fixed_cv = cv2.remap(image, base, fractions, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        values = torch.tensor(image, requires_grad=True)
        gather = layout.warp(values)
        sampled = F.grid_sample(values.permute(2, 0, 1)[None], grid[None], mode="bilinear",
                                padding_mode="zeros", align_corners=False)[0].permute(1, 2, 0)
        comparisons = {"torch_gather_vs_official_floatmap": difference(gather.detach().numpy(), actual_cv),
                       "torch_gridsample_vs_official_floatmap": difference(sampled.detach().numpy(), actual_cv),
                       "opencv_fixed5bit_vs_official_floatmap": difference(fixed_cv, actual_cv)}
        assert comparisons["torch_gather_vs_official_floatmap"]["max_abs"] <= 2e-6
        gather.mean().backward()
        assert values.grad is not None and torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0
        results[name] = {**comparisons, "image_gradient_finite": True,
                         "image_gradient_sum": float(values.grad.sum())}
    source = ROOT / "src/bridge_rgs/raw_grid.py"
    report = {"status": "passed", "cpu_only": True, "camera": "002.png", "camera_split": "train",
              "input_pixels": "synthetic patterns only; no actual RGB/mask/annotation decoded",
              "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              "source": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "opencv": cv2.__version__, "torch": torch.__version__, "warp": layout.warp.receipt(),
              "acceptance": {"float_forward_max_abs_tolerance": 2e-6,
                             "scope": "CPU only; CUDA renderer/crop/gradient smoke remains separate"},
              "patterns": results,
              "prior_diagnostic_erratum": "The original roundtrip plan assumed a 1/32 OpenCV table, but actual installed 5.0 float-map remap is continuous. The roundtrip script called real cv2, so its measured numbers remain unchanged. Original plan and report are preserved.",
              "official_reference": "https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration"}
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
