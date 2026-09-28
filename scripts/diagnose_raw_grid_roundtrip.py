"""Fixed TRAIN-only interpolation diagnostic; plan before decoding any RGB pixels."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/raw_grid_training_diagnostic_plan.json"
REPORT = ROOT / "artifacts/raw_grid_training_diagnostic.json"
SNAPSHOT = ROOT / "runs/corner_v2_preparation/source_snapshot"
MANIFEST = ROOT / "artifacts/prepared_corner_v2/manifest.json"
ORIGINAL_PLAN = ROOT / "runs/teacher_signal_audit16/plan.json"
PROTOCOL = "fixed16_train_raw_prepared_raw_v1"
VARIANTS = ("float_roundtrip", "prepared_uint8_roundtrip_float", "prepared_uint8_roundtrip_uint8")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def frozen_modules():
    sys.path.insert(0, str(SNAPSHOT))
    modules = {name: importlib.import_module("bridge_rgs." + name)
               for name in ("data", "coordinates", "evaluate")}
    for name, module in modules.items():
        if Path(module.__file__).resolve() != SNAPSHOT / "bridge_rgs" / (name + ".py"):
            raise ValueError("Module did not load from the fixed v2 training source")
    return modules


def complete_bilinear_support(valid, mx, my):
    """Conservative support: every floor/ceil tap is in bounds and valid.

    Exact integer coordinates need only that center. Fractional taps are retained
    conservatively even when OpenCV's 1/32 interpolation table rounds a weight to 0.
    """
    height, width = valid.shape
    inside = (np.isfinite(mx) & np.isfinite(my) & (mx >= 0) & (my >= 0)
              & (mx <= width - 1) & (my <= height - 1))
    x = np.where(inside, mx, 0)
    y = np.where(inside, my, 0)
    x0, x1 = np.floor(x).astype(int), np.ceil(x).astype(int)
    y0, y1 = np.floor(y).astype(int), np.ceil(y).astype(int)
    return inside & valid[y0, x0] & valid[y0, x1] & valid[y1, x0] & valid[y1, x1]


def complete_windows(valid, size):
    if size < 1 or size % 2 != 1:
        raise ValueError("Window must be positive and odd")
    return cv2.erode(valid.astype(np.uint8), np.ones((size, size), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def ssim_map(reference, predicted, gaussian):
    """Population covariance; constants .01/.03, range 1, average RGB channels."""
    reference, predicted = reference.astype(np.float64), predicted.astype(np.float64)
    if gaussian:
        def smooth(values):
            return cv2.GaussianBlur(values, (11, 11), 1.5, borderType=cv2.BORDER_CONSTANT)
    else:
        def smooth(values):
            return cv2.boxFilter(values, cv2.CV_64F, (7, 7), normalize=True,
                                 borderType=cv2.BORDER_CONSTANT)
    mx, my = smooth(reference), smooth(predicted)
    vx, vy = smooth(reference * reference) - mx * mx, smooth(predicted * predicted) - my * my
    covariance = smooth(reference * predicted) - mx * my
    result = ((2 * mx * my + .01 ** 2) * (2 * covariance + .03 ** 2)
              / ((mx * mx + my * my + .01 ** 2) * (vx + vy + .03 ** 2)))
    return result.mean(-1)


def edge_statistics(values, keep):
    # Fixed RGB Rec.709 luma on encoded [0,1] values; no linear-light conversion.
    luma = values.astype(np.float64) @ np.array([.2126, .7152, .0722])
    gx = cv2.Sobel(luma, cv2.CV_64F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(luma, cv2.CV_64F, 0, 1, ksize=3, scale=1 / 8)
    squared = gx * gx + gy * gy
    return {"mean_squared_gradient": float(squared[keep].mean()),
            "mean_gradient_magnitude": float(np.sqrt(squared[keep]).mean())}


def score(reference, predicted, pixel_keep, window_keep):
    if not window_keep.any() or np.any(window_keep & ~pixel_keep):
        raise ValueError("Need nonempty complete-window support contained in pixel support")
    error = np.square(reference.astype(np.float64) - predicted.astype(np.float64)).mean(-1)
    result = {}
    for name, keep in (("pixel_common", pixel_keep), ("common_complete_11x11", window_keep)):
        mse = float(error[keep].mean())
        result[name] = {"pixels": int(keep.sum()), "mse": mse,
                        "psnr_db": float(-10 * np.log10(max(mse, 1e-12)))}
    common = result["common_complete_11x11"]
    common["ssim_7box"] = float(ssim_map(reference, predicted, False)[window_keep].mean())
    common["ssim_11gaussian_sigma1.5"] = float(ssim_map(reference, predicted, True)[window_keep].mean())
    before, after = edge_statistics(reference, window_keep), edge_statistics(predicted, window_keep)
    common["edge_reference"] = before
    common["edge_roundtrip"] = after
    common["squared_gradient_ratio"] = after["mean_squared_gradient"] / max(before["mean_squared_gradient"], 1e-30)
    common["gradient_magnitude_ratio"] = after["mean_gradient_magnitude"] / max(before["mean_gradient_magnitude"], 1e-30)
    return result


def create_plan():
    if PLAN.exists() or REPORT.exists():
        raise FileExistsError("Refuse to replace diagnostic plan or report")
    modules = frozen_modules()
    manifest = json.loads(MANIFEST.read_text())
    if modules["coordinates"].pixel_protocol(manifest) != "colmap_corner_v2":
        raise ValueError("Expected v2 prepared data")
    population = sorted([v for v in manifest["views"] if v["split"] == "train" and v.get("mask_path")],
                        key=lambda v: v["name"])
    assert len(population) == 259
    indices = [i * 258 // 15 for i in range(16)]
    selected = [population[i] for i in indices]
    original = json.loads(ORIGINAL_PLAN.read_text())
    assert indices == original["sample_indices"] and [v["name"] for v in selected] == original["sample_names"]
    inputs = [record(MANIFEST), record(ORIGINAL_PLAN), record(ROOT / "uv.lock")]
    names = set()
    views = []
    for view in selected:
        assert view["split"] == "train" and view["mask_path"]
        for key in ("source_image_path", "image_path", "valid_path"):
            if view[key] not in names:
                inputs.append(record(view[key]))
                names.add(view[key])
        # Whitelist geometry and RGB paths; annotation/mask payloads are never used.
        views.append({key: view[key] for key in ("name", "image_id", "camera_id", "split", "width", "height",
                                                 "K", "source_image_path", "image_path", "valid_path")})
    plan = {"protocol": PROTOCOL, "created_utc": datetime.now(UTC).isoformat(),
            "sample_rule": "same sorted labeled TRAIN population; i*258//15 for i=0..15",
            "sample_indices": indices, "sample_names": [v["name"] for v in views], "views": views,
            "source_cameras": manifest["source_cameras"], "input_hashes": inputs,
            "runner": record(__file__), "source_snapshot": str(SNAPSHOT),
            "loaded_source": {name: record(module.__file__) for name, module in modules.items()},
            "variants": list(VARIANTS),
            "definitions": {
                "float_roundtrip": "raw uint8 /255 -> float32 bilinear undistort -> float32 bilinear raw rewarp",
                "prepared_uint8_roundtrip_float": "actual prepared PNG /255 -> float32 bilinear raw rewarp",
                "prepared_uint8_roundtrip_uint8": "previous variant rounded with np.rint(*255) to uint8, then /255",
                "support": "raw centers whose every floor/ceil pinhole sampling tap is in bounds and prepared-valid; conservative before OpenCV 1/32 table rounding",
                "main_metrics_support": "same raw mask eroded by an 11x11 all-ones window, with zero outside image; PSNR, SSIM7 and SSIM11 share exactly these centers",
                "ssim": "RGB channel mean; population covariance; range1; C1=.01^2 C2=.03^2; 7x7 normalized box versus 11x11 Gaussian sigma1.5",
                "edge_energy": "same complete11 support; Rec709 encoded RGB luma; 3x3 Sobel /8; mean squared norm and mean norm",
                "aggregation": "arithmetic mean per view metrics; also pooled squared error PSNR and summed-support edge energy ratio",
                "bounds": "no image filling, no extrapolation, no GT replacement; excluded border never called full image; no LPIPS or model inference",
                "scope": "TRAIN-only engineering interpolation diagnosis; camera/split metadata may be read, but no VAL image/mask/annotation payloads and no pose optimization; no model causal attribution"},
            "runtime": {"opencv": cv2.__version__, "numpy": np.__version__},
            "output": str(REPORT), "max_output_budget_bytes": 1 << 20}
    PLAN.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "planned_before_pixel_decode", "plan": str(PLAN), "sha256": digest(PLAN),
                      "sample_names": plan["sample_names"]}))


def run():
    if REPORT.exists():
        raise FileExistsError("Refuse to replace diagnostic report")
    plan = json.loads(PLAN.read_text())
    assert plan["protocol"] == PROTOCOL and plan["runner"]["sha256"] == digest(__file__)
    modules = frozen_modules()
    for item in [*plan["input_hashes"], *plan["loaded_source"].values()]:
        assert digest(item["path"]) == item["sha256"], item["path"]
    assert list(plan["variants"]) == list(VARIANTS)
    cv2.setNumThreads(8)
    records, maps = [], {}
    for view in plan["views"]:
        assert view["split"] == "train"
        cid = str(view["camera_id"])
        if cid not in maps:
            source = plan["source_cameras"][cid]
            camera = modules["data"].ColmapCamera(int(cid), source["model"], source["width"], source["height"],
                                                 np.array(source["params"]))
            mx, my, K, valid = modules["data"].undistortion_maps(camera, 1320, "colmap_corner_v2")
            assert np.array_equal(K, np.array(view["K"]))
            render_K, cw, ch, reverse = modules["evaluate"].distortion_render_grid(
                source["K"], source["opencv_distortion"], source["width"], source["height"], "colmap_corner_v2")
            assert np.array_equal(np.array(source["K"]), K)  # This diagnostic has no resize or K change.
            if reverse is None:
                rx, ry = np.meshgrid(np.arange(source["width"], dtype=np.float32),
                                      np.arange(source["height"], dtype=np.float32))
            else:
                reverse = reverse.copy()
                # Undo integer overscan canvas shift; never synthesize its missing RGB.
                reverse -= (render_K[:2, 2] - np.array(source["K"], np.float32)[:2, 2])
                rx, ry = reverse[..., 0], reverse[..., 1]
            support = complete_bilinear_support(valid > 0, rx, ry)
            maps[cid] = (mx, my, rx, ry, valid, support, complete_windows(support, 11), [cw, ch])
        mx, my, rx, ry, valid, support, common, canvas = maps[cid]
        raw_u8 = cv2.imread(view["source_image_path"], cv2.IMREAD_COLOR)[..., ::-1].copy()
        prepared_u8 = cv2.imread(view["image_path"], cv2.IMREAD_COLOR)[..., ::-1].copy()
        stored_valid = cv2.imread(view["valid_path"], cv2.IMREAD_GRAYSCALE)
        assert raw_u8.shape == prepared_u8.shape == (view["height"], view["width"], 3)
        assert np.array_equal(stored_valid, valid)
        reproduced = cv2.remap(raw_u8, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        assert np.array_equal(reproduced, prepared_u8), "Prepared PNG was not reproduced exactly"
        raw, prepared = raw_u8.astype(np.float32) / 255, prepared_u8.astype(np.float32) / 255
        float_pinhole = cv2.remap(raw, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        float_raw = cv2.remap(float_pinhole, rx, ry, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        prepared_raw = cv2.remap(prepared, rx, ry, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        delivered = np.rint(prepared_raw * 255).clip(0, 255).astype(np.uint8).astype(np.float32) / 255
        result = {"name": view["name"], "split": "train", "prepared_png_reproduced_exact": True,
                  "pixels": int(support.size), "common_supported_pixels": int(support.sum()),
                  "common_supported_fraction": float(support.mean()), "complete11_pixels": int(common.sum()),
                  "complete11_fraction": float(common.mean()), "complete7_pixels": int(complete_windows(support, 7).sum()),
                  "official_pinhole_canvas": canvas,
                  "variants": {name: score(raw, values, support, common)
                               for name, values in zip(VARIANTS, [float_raw, prepared_raw, delivered], strict=True)}}
        records.append(result)
        print(json.dumps({"view": view["name"], "completed": len(records), "of": 16}), flush=True)
    aggregate = {}
    for variant in VARIANTS:
        aggregate[variant] = {}
        for scope in ("pixel_common", "common_complete_11x11"):
            values = [r["variants"][variant][scope] for r in records]
            pixels = sum(v["pixels"] for v in values)
            mse = sum(v["mse"] * v["pixels"] for v in values) / pixels
            item = {"pixels": pixels, "mean_view_psnr_db": float(np.mean([v["psnr_db"] for v in values])),
                    "pooled_mse": mse, "pooled_psnr_db": float(-10 * np.log10(max(mse, 1e-12)))}
            if scope == "common_complete_11x11":
                for key in ("ssim_7box", "ssim_11gaussian_sigma1.5", "squared_gradient_ratio", "gradient_magnitude_ratio"):
                    item["mean_view_" + key] = float(np.mean([v[key] for v in values]))
                for field in ("mean_squared_gradient", "mean_gradient_magnitude"):
                    before = sum(v["edge_reference"][field] * v["pixels"] for v in values) / pixels
                    after = sum(v["edge_roundtrip"][field] * v["pixels"] for v in values) / pixels
                    item["pooled_" + field] = {"reference": before, "roundtrip": after, "ratio": after / before}
            aggregate[variant][scope] = item
    for item in [*plan["input_hashes"], *plan["loaded_source"].values()]:
        assert digest(item["path"]) == item["sha256"]
    report = {"status": "completed", "protocol": PROTOCOL, "plan": record(PLAN), "runner": record(__file__),
              "finished_utc": datetime.now(UTC).isoformat(), "sample_names": plan["sample_names"],
              "inputs_unchanged_after_run": True, "cpu_only": True, "mask_annotation_payloads_read": False,
              "validation_payloads_read": False, "no_ground_truth_border_fill": True,
              "definitions": plan["definitions"], "aggregate": aggregate, "views": records}
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    assert REPORT.stat().st_size < plan["max_output_budget_bytes"]
    print(json.dumps({"status": "completed", "report": str(REPORT), "sha256": digest(REPORT)}))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"))
    args = parser.parse_args()
    create_plan() if args.mode == "plan" else run()
