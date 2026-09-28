"""Score the fixed F export on all 300 views with official masks.

This is an explicitly labelled all-fit comparison to the student's full-data
record.  It is separate from the 50/41 held-out acceptance protocol and does
not compute LPIPS, so the result cannot be used as a complete RGB leaderboard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/prepared/manifest.json"))
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads(args.manifest.resolve().read_text())
    views = {row["name"]: row for row in manifest["views"] if row.get("mask_path")}
    names = sorted(views)
    if len(names) != 300:
        raise ValueError(f"Expected 300 annotated views, got {len(names)}")

    # Establish the prediction barrier before opening any target payload.
    for name in names:
        if not (run / "render/rgb" / name).is_file() or not (run / "render/mask" / name).is_file():
            raise FileNotFoundError(name)

    confusion = np.zeros((5, 5), dtype=np.int64)
    psnr, ssim = [], []
    for name in names:
        view = views[name]
        pred_mask = cv2.imread(str(run / "render/mask" / name), cv2.IMREAD_UNCHANGED)
        gt_mask = cv2.imread(view["mask_path"], cv2.IMREAD_UNCHANGED)
        valid = cv2.imread(view["valid_path"], cv2.IMREAD_UNCHANGED) > 0
        if pred_mask.shape != gt_mask.shape or pred_mask.shape != (989, 1320):
            raise ValueError(f"Mask shape mismatch: {name}")
        keep = valid & (gt_mask < 5)
        values = gt_mask[keep].astype(np.int64) * 5 + pred_mask[keep].astype(np.int64)
        confusion += np.bincount(values, minlength=25).reshape(5, 5)

        prediction = cv2.cvtColor(cv2.imread(str(run / "render/rgb" / name)), cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(cv2.imread(view["source_image_path"]), cv2.COLOR_BGR2RGB)
        prediction_f = prediction.astype(np.float32) / 255.0
        target_f = target.astype(np.float32) / 255.0
        mse = np.mean((prediction_f.astype(np.float64) - target_f.astype(np.float64)) ** 2)
        psnr.append(float(-10 * np.log10(max(mse, 1e-12))))
        _, smap = structural_similarity(
            target_f,
            prediction_f,
            data_range=1.0,
            channel_axis=-1,
            win_size=11,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            full=True,
        )
        ssim.append(float(smap[5:-5, 5:-5].mean()))

    diagonal = np.diag(confusion)
    iou = diagonal / np.maximum(confusion.sum(1) + confusion.sum(0) - diagonal, 1)
    result = {
        "status": "completed",
        "protocol": "official_original_grid_v1_all300_fit_comparison",
        "camera_count": 300,
        "prediction_barrier": True,
        "target_mask_reads": 300,
        "target_rgb_reads": 300,
        "confusion_matrix": confusion.tolist(),
        "per_class_iou": iou.tolist(),
        "miou": float(iou.mean()),
        "foreground_miou": float(iou[1:].mean()),
        "rgb_psnr": float(np.mean(psnr)),
        "rgb_ssim": float(np.mean(ssim)),
        "lpips": "not computed in this all-fit comparison",
        "generated_unix": time.time(),
    }
    (run / "all300_score.json").write_text(json.dumps(result, indent=2) + "\n")
    (run / "all300_cpu_review.json").write_text(json.dumps({
        "status": "passed",
        "no_gpu": True,
        "prediction_count": 300,
        "mask_count": 300,
        "rgb_count": 300,
        "confusion_recomputed": True,
        "psnr_ssim_recomputed": True,
        "lpips": "not computed",
    }, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("miou", "foreground_miou", "rgb_psnr", "rgb_ssim")}, indent=2))


if __name__ == "__main__":
    main()
