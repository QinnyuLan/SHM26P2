"""Camera-only prediction and evaluation; export official IDs as uint8 PNG."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

from .coordinates import CORNER, LEGACY, protocol_metadata, require_matching_protocol
from .coordinates import pixel_protocol as resolve_protocol
from .io import load_manifest, load_view
from .losses import confusion_matrix, iou_scores

PALETTE = np.array([[35, 42, 52], [230, 160, 35], [225, 55, 65],
                    [70, 130, 225], [110, 200, 125]], dtype=np.uint8)


def evaluation_fingerprint(views, scale, pixel_protocol=None, manifest_sha256=None):
    """Identify the scored cameras/grid and split without using model predictions."""
    payload = {"scale": scale, "views": [{key: view.get(key) for key in (
        "name", "image_path", "mask_path", "valid_path", "K", "w2c_original",
        "w2c", "width", "height", "split")} for view in views]}
    protocols = {resolve_protocol(view) for view in views}
    if len(protocols) > 1:
        raise ValueError("Cannot evaluate mixed pixel protocols")
    protocol = (next(iter(protocols), LEGACY) if pixel_protocol is None
                else resolve_protocol(pixel_protocol))
    if protocols and protocols != {protocol}:
        raise ValueError("Evaluation views disagree with requested pixel protocol")
    if protocol == CORNER:
        payload.update(pixel_protocol=protocol_metadata(protocol), manifest_sha256=manifest_sha256)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def boundary_counts(prediction, target, valid, tolerance=2, classes=5):
    """Per-class matched/total boundary counts, excluding invalid-border artifacts.

    Boundaries are inner one-pixel outlines; precision and recall use a square
    tolerance neighbourhood. These counts support aggregation over images.
    """
    kernel = np.ones((3, 3), np.uint8)
    support = cv2.erode(valid.astype(np.uint8), np.ones((2*tolerance+3,)*2, np.uint8),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    neighbourhood = np.ones((2*tolerance+1,)*2, np.uint8)
    result = []
    for category in range(classes):
        p, y = (prediction == category).astype(np.uint8), (target == category).astype(np.uint8)
        p = (p != cv2.erode(p, kernel)) & support
        y = (y != cv2.erode(y, kernel)) & support
        p_match = p & cv2.dilate(y.astype(np.uint8), neighbourhood).astype(bool)
        y_match = y & cv2.dilate(p.astype(np.uint8), neighbourhood).astype(bool)
        result.append([int(p_match.sum()), int(p.sum()), int(y_match.sum()), int(y.sum())])
    return np.array(result)


def boundary_scores(counts):
    scores = []
    for p_match, p_total, y_match, y_total in counts:
        if p_total + y_total == 0:
            scores.append(None)
            continue
        precision, recall = p_match / max(p_total, 1), y_match / max(y_total, 1)
        scores.append(float(2 * precision * recall / max(precision + recall, 1e-12)))
    return scores


@torch.no_grad()
def evaluate_scene(scene, manifest, output, max_views=None, scale=1.0, lpips_metric=False,
                   checkpoint_pixel_protocol=None, refiner_flip_tta=False):
    if scale <= 0 or (max_views is not None and max_views < 1):
        raise ValueError("scale and max_views must be positive")
    from .coordinates import annotate_manifest
    manifest = annotate_manifest(manifest)
    protocol = resolve_protocol(manifest)
    checkpoint_protocol = (getattr(scene, "pixel_protocol", None) if checkpoint_pixel_protocol is None
                           else checkpoint_pixel_protocol)
    require_matching_protocol(manifest, checkpoint_protocol, "checkpoint evaluation grid")
    expected_hash = getattr(scene, "manifest_sha256", None)
    if isinstance(checkpoint_pixel_protocol, dict):
        expected_hash = checkpoint_pixel_protocol.get("manifest_sha256", expected_hash)
    actual_hash = manifest.get("_manifest_sha256")
    if (expected_hash is not None and (actual_hash is not None or protocol == CORNER)
            and expected_hash != actual_hash):
        raise ValueError("Evaluation manifest SHA differs from checkpoint provenance")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    views = [v for v in manifest["views"] if v["split"] == "val"]
    if max_views:
        views = views[:max_views]
    if not views:
        raise ValueError("No held-out views: all-data fits cannot be reported as validation")
    was_training = scene.training
    scene.eval()
    device = scene.splats["means"].device
    matrix = torch.zeros(5, 5, device=device, dtype=torch.long)
    matrix3d = torch.zeros_like(matrix)
    boundaries = np.zeros((5, 4), dtype=np.int64)
    records = []
    perceptual = None
    if lpips_metric:
        import lpips
        perceptual = lpips.LPIPS(net="alex").to(device).eval()
    for view in views:
        # Data can be opened for scoring only after the untouched camera is selected.
        data = load_view(view, device=device, scale=scale, original_pose=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tic = time.perf_counter()
        rendered = scene.render(data["K"], data["w2c"], data["width"], data["height"], absgrad=False)
        if refiner_flip_tta:
            from .refiner_tta import horizontal_flip_average
            rendered["probabilities"] = horizontal_flip_average(scene.refiner, rendered)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - tic
        image = rendered["rgb"].clamp(0, 1)
        valid = data["valid"].bool()
        if not valid.any():
            raise ValueError(f"No valid evaluation pixels: {view['name']}")
        mse = (image[valid] - data["rgb"][valid]).square().mean()
        pred, target = image.cpu().numpy(), data["rgb"].cpu().numpy()
        _, smap = structural_similarity(target, pred, data_range=1, channel_axis=-1,
                                       gaussian_weights=True, sigma=1.5, use_sample_covariance=False, full=True)
        # Erode masks so SSIM windows do not include invalid undistorted borders.
        ss_valid = cv2.erode(valid.cpu().numpy().astype(np.uint8), np.ones((11, 11), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        if not ss_valid.any():
            raise ValueError(f"No complete 11x11 SSIM windows: {view['name']}")
        record = {"name": view["name"], "psnr": float(-10 * mse.clamp_min(1e-12).log10()),
                  "ssim": float(smap[ss_valid].mean()), "render_seconds": seconds,
                  "width": data["width"], "height": data["height"]}
        if perceptual is not None:
            # Invalid pixels equal GT in both LPIPS inputs, with policy recorded below.
            masked_pred = torch.where(valid[..., None], image, data["rgb"])
            record["lpips"] = float(perceptual(masked_pred.permute(2, 0, 1)[None] * 2 - 1,
                                               data["rgb"].permute(2, 0, 1)[None] * 2 - 1))
        if data["mask"] is not None:
            m = confusion_matrix(rendered["probabilities"], data["mask"], valid)
            matrix += m
            m3d = confusion_matrix(rendered["p3d"], data["mask"], valid)
            matrix3d += m3d
            bc = boundary_counts(rendered["probabilities"].argmax(-1).cpu().numpy(),
                                 data["mask"].cpu().numpy(),
                                 (valid & (data["mask"] < 5)).cpu().numpy())
            boundaries += bc
            record.update(confusion_matrix=m.cpu().tolist(),
                          confusion_matrix_3d=m3d.cpu().tolist(), boundary_counts=bc.tolist(),
                          boundary_f1_2px=boundary_scores(bc))
            record.update(iou_scores(m))
        name = Path(view["name"]).stem
        mask = rendered["probabilities"].argmax(-1).byte().cpu().numpy()
        cv2.imwrite(str(output / f"{name}_rgb.png"), (pred[..., ::-1] * 255).round().astype(np.uint8))
        cv2.imwrite(str(output / f"{name}_mask.png"), mask)
        if len(records) < 8:
            panels = [target, pred, PALETTE[mask] / 255.]
            if data["mask"] is not None:
                panels.append(PALETTE[data["mask"].clamp(0, 4).cpu().numpy()] / 255.)
            strip = (np.concatenate(panels, 1)[..., ::-1] * 255).round().astype(np.uint8)
            cv2.imwrite(str(output / f"{name}_comparison.jpg"), strip)
        records.append(record)
    summary = {"protocol": "heldout pixels; original unoptimized camera; undistorted pinhole grid",
               "evaluation_fingerprint": evaluation_fingerprint(
                   views, scale, protocol, manifest.get("_manifest_sha256")),
               "validation_views": len(records),
               "semantic_validation_views": sum(v.get("mask_path") is not None for v in views),
               "scale": scale, "gaussians": len(scene.splats["means"]),
               "psnr": float(np.mean([r["psnr"] for r in records])),
               "ssim": float(np.mean([r["ssim"] for r in records])),
               "lpips": float(np.mean([r["lpips"] for r in records])) if perceptual else None,
               "lpips_validity_policy": "invalid pixels replaced with target in prediction" if perceptual else "not computed",
               "mean_render_seconds": float(np.mean([r["render_seconds"] for r in records])),
               **iou_scores(matrix), "semantic_3d": iou_scores(matrix3d),
               "confusion_matrix": matrix.cpu().tolist(), "confusion_matrix_3d": matrix3d.cpu().tolist(),
               "boundary_f1_2px": boundary_scores(boundaries),
               "boundary_policy": "inner 1px; Chebyshev tolerance 2px; valid support eroded 7px",
               "views": records}
    if protocol == CORNER:
        summary["pixel_protocol"] = protocol_metadata(protocol)
    if refiner_flip_tta:
        summary["inference_protocol"] = "one_3d_render_two_refiner_passes_horizontal_flip_probability_mean_0.5"
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    scene.train(was_training)
    return summary


def evaluate(checkpoint_path, manifest_path, output, **kwargs):
    from .train import load_scene
    scene, state = load_scene(checkpoint_path)
    return evaluate_scene(scene, load_manifest(manifest_path), output,
                          checkpoint_pixel_protocol=state, **kwargs)


def read_render_cameras(cameras_path, grid="official"):
    """Read only camera metadata, including source distortion of prepared data."""
    from .data import read_colmap_cameras, read_colmap_images
    if grid not in {"official", "pinhole"}:
        raise ValueError("grid must be official or pinhole")
    path = Path(cameras_path)
    records = []
    if path.is_dir():
        cameras = read_colmap_cameras(path / "cameras.txt")
        images = read_colmap_images(path / "images.txt")
        for image in sorted(images.values(), key=lambda image: image.name):
            camera = cameras[image.camera_id]
            records.append({"name": image.name, "image_id": image.image_id,
                            "camera_id": image.camera_id, "K": camera.K.tolist(),
                            "w2c": image.w2c.tolist(), "width": camera.width,
                            "height": camera.height, "distortion": camera.distortion.tolist()})
    else:
        values = json.loads(path.read_text())
        raw_records = values["views"] if isinstance(values, dict) else values
        sources = values.get("source_cameras", {}) if isinstance(values, dict) else {}
        for original in raw_records:
            record = dict(original)
            source = sources.get(str(record.get("camera_id")))
            if grid == "official" and source:
                record.update(K=source["K"], width=source["width"], height=source["height"],
                              distortion=source["opencv_distortion"])
            records.append(record)
    names = [Path(str(record["name"])).stem for record in records]
    if len(set(names)) != len(names):
        raise ValueError("Camera output filename stems must be unique")
    return records


def distortion_render_grid(K, distortion, width, height, pixel_protocol=LEGACY):
    """Return a covering pinhole canvas and map into the distorted output grid.

    For negative distortion some official pixels lie outside the native pinhole
    canvas. Overscan renders these rays, avoiding fabricated replicated borders.
    """
    K = np.asarray(K, dtype=np.float32)
    protocol = resolve_protocol(pixel_protocol)
    distortion = np.asarray(distortion, dtype=np.float32)
    if not distortion.size or not np.any(distortion):
        return K.copy(), int(width), int(height), None
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack([xx, yy], -1).astype(np.float32)
    if protocol == CORNER:
        pixels += .5
    source = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, distortion, P=K).reshape(height, width, 2)
    if protocol == CORNER:
        source -= .5
    if not np.isfinite(source).all():
        raise ValueError("Distortion inversion produced nonfinite coordinates")
    minimum = np.minimum(np.floor(source.reshape(-1, 2).min(0)), [0, 0]).astype(int)
    maximum = np.maximum(np.ceil(source.reshape(-1, 2).max(0)), [width-1, height-1]).astype(int)
    canvas_width, canvas_height = (maximum - minimum + 1).tolist()
    if canvas_width * canvas_height > width * height * 16:
        raise ValueError("Distortion requires more than 16x overscan; verify calibration")
    render_K = K.copy()
    render_K[:2, 2] -= minimum
    source -= minimum.astype(np.float32)
    return render_K, canvas_width, canvas_height, source


@torch.no_grad()
def render_cameras(checkpoint_path, cameras_path, output, grid="official", max_views=None,
                   *, teacher_checkpoint=None, refiner_flip_tta=False):
    """Accept camera JSON or a COLMAP text directory; never read RGB/labels."""
    from .train import load_scene
    if max_views is not None and max_views < 1:
        raise ValueError("max_views must be positive")
    if teacher_checkpoint is not None and refiner_flip_tta:
        raise ValueError("Teacher mixture and flip TTA are separate inference controls")
    records = read_render_cameras(cameras_path, grid)
    if max_views:
        records = records[:max_views]
    if not records:
        raise ValueError("No cameras to render")
    scene, state = load_scene(checkpoint_path)
    # The model's historical export protocol wins over camera-file metadata.
    protocol = resolve_protocol(state)
    scene.eval()
    device = scene.splats["means"].device
    teacher = None
    if teacher_checkpoint is not None:
        from .render_ensemble import FixedRenderedTeacher
        teacher = FixedRenderedTeacher(teacher_checkpoint, checkpoint_path, state, device)
    output = Path(output)
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    (output / "mask").mkdir(exist_ok=True)
    exported = []
    for camera in records:
        K = np.array(camera["K"], dtype=np.float32)
        pose = torch.tensor(camera.get("w2c_original", camera["w2c"]), device=device).float()
        width, height = int(camera["width"]), int(camera["height"])
        distortion = camera.get("distortion", []) if grid == "official" else []
        render_K, canvas_width, canvas_height, source = distortion_render_grid(
            K, distortion, width, height, protocol)
        result = scene.render(torch.tensor(render_K, device=device), pose, canvas_width, canvas_height, absgrad=False)
        if refiner_flip_tta:
            from .refiner_tta import horizontal_flip_average
            result["probabilities"] = horizontal_flip_average(scene.refiner, result)
        rgb = result["rgb"].clamp(0, 1).cpu().numpy()
        probabilities = result["probabilities"].cpu().numpy()
        if teacher is not None:
            probabilities = teacher.blend(rgb, probabilities)
        if source is not None:
            rgb = cv2.remap(rgb, source[..., 0], source[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            probabilities = cv2.remap(probabilities, source[..., 0], source[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        name = Path(camera["name"]).stem
        rgb_path, mask_path = output / "rgb" / f"{name}.png", output / "mask" / f"{name}.png"
        if not cv2.imwrite(str(rgb_path), (rgb[..., ::-1] * 255).round().astype(np.uint8)):
            raise OSError(f"Could not write {rgb_path}")
        if not cv2.imwrite(str(mask_path), probabilities.argmax(-1).astype(np.uint8)):
            raise OSError(f"Could not write {mask_path}")
        exported.append({"name": camera["name"], "image_id": camera.get("image_id"),
                         "camera_id": camera.get("camera_id"), "width": width, "height": height,
                         "distortion": list(distortion), "pinhole_canvas": [canvas_width, canvas_height],
                         "rgb": str(rgb_path.resolve()), "mask": str(mask_path.resolve())})
    receipt = {"views": len(records), "grid": grid, "checkpoint": str(Path(checkpoint_path).resolve()),
               "input": "camera parameters only", "cameras": str(Path(cameras_path).resolve()),
               "records": exported,
               "class_ids": {"background": 0, "deck": 1, "stay_cable": 2, "tower": 3, "foundation": 4}}
    receipt["pixel_protocol"] = protocol_metadata(protocol)
    if "dependency_provenance" in state:
        receipt["checkpoint_dependencies"] = state["dependency_provenance"]
    if teacher is not None:
        receipt["semantic_ensemble"] = teacher.receipt
    if refiner_flip_tta:
        receipt["inference_protocol"] = "one_3d_render_two_refiner_passes_horizontal_flip_probability_mean_0.5"
    (output / "render_receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt
