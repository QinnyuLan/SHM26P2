"""Independent original-grid evaluation, never a prepared-grid guard override."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from .coordinates import (
    CORNER,
    annotate_manifest,
    pixel_protocol,
    protocol_metadata,
    require_matching_protocol,
)
from .data import CLASS_IDS, CLASS_NAMES, rasterize_labelme
from .evaluate import distortion_render_grid
from .losses import iou_scores

DEFAULT_REFERENCE = Path("artifacts/prepared/manifest.json")
REFERENCE_SHA256 = "551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa"
VAL_COUNT, ANNOTATION_COUNT = 50, 41
FAMILY = "official_original_pixel_grid"
CAMERA_KEYS = {"name", "image_id", "camera_id", "split", "K", "w2c", "width", "height", "distortion"}
SCORING_PROTOCOL = {
    "id": "official_original_grid_v1",
    "version": 1,
    "ground_truth_grid": "original distorted image; no undistortion or validity mask",
    "rgb_prediction": "clamp float RGB before warp; OpenCV INTER_LINEAR; numpy round-to-even uint8 PNG",
    "semantic_prediction": "OpenCV INTER_LINEAR on soft probabilities before argmax; lowest ID breaks ties",
    "rgb_scoring": "decode delivered uint8 PNG to float32 RGB/255; full image",
    "psnr": "-10log10(max(full-image RGB MSE,1e-12)); mean over views",
    "ssim": {"window": 11, "gaussian_weights": True, "sigma": 1.5,
             "use_sample_covariance": False, "data_range": 1.0,
             "support": "all complete windows, excluding 5 pixels on each image edge",
             "aggregation": "mean RGB SSIM map over complete centers, then mean over views"},
    "lpips": {"net": "alex", "version": "0.1", "input_range": [-1, 1],
              "support": "full image; no GT replacement, masks, crops or resizing", "aggregation": "mean over views"},
    "rasterizer": "labelme_pillow_polygon_draw_order_v1_with_unknown_ignore_overlay_v1",
    "class_names": CLASS_NAMES,
    "ignore_label": 255,
    "semantic_aggregation": "pooled 5x5 confusion; target255 ignored; present-union all/foreground IoU",
}
PLAIN_INFERENCE = "single checkpoint; one renderer/refiner pass per covering canvas; no teacher or TTA"
TEACHER_INFERENCE = "fixed rendered DINOv3/scene probability mixture 0.5; tile768 stride512 flip context0.25; before original-grid warp"


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _utc():
    return datetime.now(UTC).isoformat()


def _loaded_sources():
    """Bind actual loaded modules, including immutable source-snapshot paths."""
    names = ("official_evaluate", "evaluate", "coordinates", "data", "losses", "train", "model", "refinement", "checkpoints",
             "render_ensemble", "teacher", "teacher_adapters", "teacher_domains")
    records = {}
    for name in names:
        module = sys.modules.get(f"bridge_rgs.{name}")
        path = Path(__file__) if name == "official_evaluate" else Path(module.__file__) if module is not None else None
        if path is not None:
            records[f"bridge_rgs.{name}"] = {"path": str(path.resolve()), "sha256": _file_hash(path)}
    return records


def _resolve(path, workspace_root=None):
    """Relative dataset/config paths refer to the workspace, never package __file__."""
    path = Path(path).expanduser()
    root = Path.cwd() if workspace_root is None else Path(workspace_root).expanduser().resolve()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _camera(view, sources):
    if str(view.get("camera_id")) not in sources:
        raise ValueError(f"Missing source camera for {view.get('name')}")
    source = sources[str(view["camera_id"])]
    # Require the stored original pose; never silently use an optimized pose.
    camera = {key: view[key] for key in ("name", "image_id", "camera_id", "split")}
    camera.update(K=source["K"], w2c=view["w2c_original"], width=source["width"],
                  height=source["height"], distortion=source["opencv_distortion"])
    K, pose, distortion = (np.asarray(camera[key], dtype=np.float64) for key in ("K", "w2c", "distortion"))
    if (K.shape != (3, 3) or pose.shape != (4, 4) or distortion.ndim != 1
            or not all(np.isfinite(x).all() for x in (K, pose, distortion))):
        raise ValueError("Invalid original camera arrays")
    if (K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1])
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or int(camera["width"]) != camera["width"] or int(camera["height"]) != camera["height"]
            or min(camera["width"], camera["height"]) < 11):
        raise ValueError("Invalid original camera dimensions/intrinsics/pose")
    if Path(camera["name"]).name != camera["name"]:
        raise ValueError("Camera names must be plain filenames")
    return copy.deepcopy(camera)


def read_official_reference(path=DEFAULT_REFERENCE, workspace_root=None):
    """Read metadata/existence only; image and annotation bytes stay unopened."""
    workspace_root = str(_resolve(".", workspace_root))
    path = _resolve(path, workspace_root)
    data = path.read_bytes()
    if _hash(data) != REFERENCE_SHA256:
        raise ValueError("Official v1 requires the frozen reference manifest SHA")
    manifest = json.loads(data)
    if manifest["class_names"] != CLASS_NAMES or manifest.get("ignore_label", 255) != 255:
        raise ValueError("Official class IDs/ignore value differ")
    selected = sorted((v for v in manifest["views"] if v["split"] == "val"), key=lambda v: v["name"])
    if len(selected) != VAL_COUNT or sum(bool(v.get("source_annotation_path")) for v in selected) != ANNOTATION_COUNT:
        raise ValueError("Official v1 requires exactly the fixed 50 RGB / 41 annotated views")
    names = [Path(v["name"]).stem for v in selected]
    if len(set(names)) != len(names) or len({v["image_id"] for v in selected}) != len(selected):
        raise ValueError("Duplicate official name/image ID")
    cameras, targets = [], []
    for view in selected:
        cameras.append(_camera(view, manifest["source_cameras"]))
        paths = {"source_image_path": _resolve(view["source_image_path"], workspace_root),
                 "source_annotation_path": _resolve(view["source_annotation_path"], workspace_root) if view.get("source_annotation_path") else None}
        for value in paths.values():
            if value is not None and not value.is_file():
                raise FileNotFoundError(value)
        targets.append({"name": view["name"], **{k: str(v) if v is not None else None for k, v in paths.items()}})
    return {"path": str(path), "sha256": _hash(data), "workspace_root": workspace_root,
            "cameras": cameras, "targets": targets}


def validate_training_lineage(state, reference):
    """A new scoring grid must not turn an all-data fit into held-out validation."""
    manifest_path = _resolve(state["config"]["manifest"], reference["workspace_root"])
    data = manifest_path.read_bytes()
    manifest, actual_hash = annotate_manifest(json.loads(data)), _hash(data)
    expected = state.get("manifest_sha256")
    if pixel_protocol(state) == CORNER and expected is None:
        raise ValueError("Corner checkpoint requires training manifest SHA provenance")
    if expected is not None and expected != actual_hash:
        raise ValueError("Checkpoint training manifest SHA differs")
    require_matching_protocol(state, manifest, "official checkpoint training lineage")
    if manifest["class_names"] != CLASS_NAMES:
        raise ValueError("Checkpoint class IDs differ")
    views = {v["name"]: v for v in manifest["views"]}
    if len(views) != len(manifest["views"]):
        raise ValueError("Duplicate checkpoint manifest view")
    for camera, target in zip(reference["cameras"], reference["targets"], strict=True):
        view = views.get(camera["name"])
        if view is None or view["split"] != "val":
            raise ValueError("Official reference view was not held out in checkpoint training")
        if _camera(view, manifest["source_cameras"]) != camera:
            raise ValueError("Checkpoint original camera differs from official reference")
        for key in ("source_image_path", "source_annotation_path"):
            source = str(_resolve(view[key], reference["workspace_root"])) if view.get(key) else None
            if source != target[key]:
                raise ValueError("Checkpoint original data sources differ from official reference")
    return {"path": str(manifest_path), "observed_sha256": actual_hash,
            "checkpoint_declared_sha256": expected, "pixel_protocol": protocol_metadata(state)}


@torch.inference_mode()
def predict_official_camera(scene, camera, checkpoint_state, *, teacher=None):
    """Camera whitelist only; no RGB, annotation, validity or profile metadata."""
    if set(camera) != CAMERA_KEYS:
        raise ValueError("Prediction accepts only the official camera whitelist")
    protocol = pixel_protocol(checkpoint_state)
    device = scene.splats["means"].device
    K, width, height, source = distortion_render_grid(camera["K"], camera["distortion"],
                                                     camera["width"], camera["height"], protocol)
    result = scene.render(torch.tensor(K, device=device),
                          torch.tensor(camera["w2c"], device=device).float(), width, height, absgrad=False)
    rgb = result["rgb"].clamp(0, 1).cpu().numpy()
    probabilities = result["probabilities"].cpu().numpy()
    if rgb.shape != (height, width, 3) or probabilities.shape != (height, width, len(CLASS_NAMES)):
        raise ValueError("Renderer returned the wrong canvas/class shape")
    if not np.isfinite(rgb).all() or not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Renderer returned invalid RGB/probabilities")
    if teacher is not None:
        probabilities = teacher.blend(rgb, probabilities)
        if (probabilities.shape != (height, width, len(CLASS_NAMES))
                or not np.isfinite(probabilities).all() or (probabilities < 0).any()):
            raise ValueError("Teacher mixture returned invalid probabilities")
    if source is not None:
        rgb = cv2.remap(rgb, source[..., 0], source[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        probabilities = cv2.remap(probabilities, source[..., 0], source[..., 1], cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
    if (probabilities.sum(-1) <= 0).any():
        raise ValueError("Official output includes uncovered semantic rays")
    return ((rgb * 255).round().astype(np.uint8), probabilities.argmax(-1).astype(np.uint8),
            {"pinhole_canvas": [width, height], "checkpoint_pixel_protocol": protocol})


def rasterize_official_annotation(content, width, height):
    """Use the existing known-label rasterizer; unknown areas retain draw order."""
    annotation = json.loads(content)
    if (annotation["imageWidth"], annotation["imageHeight"]) != (width, height):
        raise ValueError("Original annotation/image shape mismatch")
    sanitized = copy.deepcopy(annotation)
    # imageData is not part of polygon rasterization, and need not be copied to disk.
    sanitized.pop("imageData", None)
    unknown = []
    for shape in sanitized["shapes"]:
        label = shape["label"].strip()
        unknown.append(label not in CLASS_IDS)
        if unknown[-1]:
            shape["label"] = "background"
    with TemporaryDirectory(prefix="bridge-official-labelme-") as folder:
        path = Path(folder) / "annotation.json"
        path.write_text(json.dumps(sanitized))
        mask = rasterize_labelme(path, width, height).copy()
    if any(unknown):
        ignore = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(ignore)
        for shape, is_unknown in zip(sanitized["shapes"], unknown, strict=True):
            draw.polygon([tuple(p) for p in shape["points"]], fill=255 if is_unknown else 0)
        mask[np.asarray(ignore) > 0] = 255
    return mask


def _decode_rgb(content, width, height, context):
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {context}")
    if image.shape != (height, width, 3):
        raise ValueError(f"Original/prediction RGB shape mismatch: {context}")
    return image[..., ::-1].copy()


def score_official_arrays(predicted_rgb, predicted_mask, target_rgb, target_mask, perceptual, device="cpu"):
    """Score delivered uint8 images, including full-image LPIPS inputs."""
    if (predicted_rgb.dtype != np.uint8 or target_rgb.dtype != np.uint8
            or predicted_rgb.shape != target_rgb.shape or predicted_rgb.ndim != 3
            or predicted_rgb.shape[-1] != 3 or min(predicted_rgb.shape[:2]) < 11):
        raise ValueError("Scoring requires matching uint8 RGB images of at least 11x11")
    if (predicted_mask is None or predicted_mask.dtype != np.uint8
            or predicted_mask.shape != target_rgb.shape[:2]
            or not np.isin(predicted_mask, np.arange(5)).all()):
        raise ValueError("Prediction mask must contain exactly official class IDs on the original grid")
    pred, target = predicted_rgb.astype(np.float32) / 255, target_rgb.astype(np.float32) / 255
    mse = float(np.square(pred.astype(np.float64) - target.astype(np.float64)).mean())
    _, smap = structural_similarity(target, pred, data_range=1., channel_axis=-1, win_size=11,
                                   gaussian_weights=True, sigma=1.5, use_sample_covariance=False, full=True)
    with torch.inference_mode():
        p = torch.tensor(pred, device=device).permute(2, 0, 1)[None] * 2 - 1
        t = torch.tensor(target, device=device).permute(2, 0, 1)[None] * 2 - 1
        perceptual_value = float(perceptual(p, t).reshape(()))
    result = {"psnr": float(-10 * np.log10(max(mse, 1e-12))), "ssim": float(smap[5:-5, 5:-5].mean()),
              "lpips": perceptual_value, "rgb_pixels": pred.shape[0] * pred.shape[1]}
    if not all(np.isfinite(result[k]) for k in ("psnr", "ssim", "lpips")):
        raise ValueError("Nonfinite official score")
    if target_mask is not None:
        if target_mask.shape != predicted_mask.shape or not np.isin(target_mask, [0, 1, 2, 3, 4, 255]).all():
            raise ValueError("Invalid original rasterized semantic mask")
        valid = target_mask != 255
        matrix = np.bincount(target_mask[valid].astype(np.int64) * 5 + predicted_mask[valid], minlength=25).reshape(5, 5)
        result.update(confusion_matrix=matrix.tolist(), semantic_pixels=int(valid.sum()),
                      semantic_ignore_pixels=int((~valid).sum()), **iou_scores(torch.tensor(matrix)))
    return result


def official_fingerprint(records):
    """Domain-separated common GT protocol; model/profile provenance excluded."""
    allowed = {"camera", "source_image_sha256", "source_annotation_sha256", "rasterized_mask_sha256"}
    if any(set(record) != allowed for record in records):
        raise ValueError("Unexpected fingerprint fields; no model/profile metadata is allowed")
    payload = {"evaluation_family": FAMILY, "scoring_protocol": SCORING_PROTOCOL,
               "views": sorted(records, key=lambda r: r["camera"]["name"])}
    return _hash(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def require_same_official_protocol(first, second):
    key = "official_evaluation_fingerprint"
    if (first.get("evaluation_family") != FAMILY or second.get("evaluation_family") != FAMILY
            or not first.get(key) or first[key] != second.get(key)):
        raise ValueError("Cannot compare different official protocols or native-grid metrics")


def _load_scene(path, device):
    from .train import load_scene
    return load_scene(path, device=device)


def _lpips(device):
    import lpips
    return lpips.LPIPS(net="alex", version="0.1").to(device).eval()


def evaluate_official(checkpoint_path, output, reference_manifest=DEFAULT_REFERENCE, device="cuda", *,
                      workspace_root=None, entrypoint=None, teacher_checkpoint=None):
    """Predict every fixed camera first, then open source scoring files."""
    output, checkpoint_path = _resolve(output, workspace_root), _resolve(checkpoint_path, workspace_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Official evaluation refuses a nonempty output directory")
    reference = read_official_reference(reference_manifest, workspace_root)
    checkpoint_sha = _file_hash(checkpoint_path)
    from .checkpoints import load_checkpoint
    state = load_checkpoint(checkpoint_path, map_location="cpu")
    lineage = validate_training_lineage(state, reference)
    model_protocol = pixel_protocol(state)
    inference_protocol = TEACHER_INFERENCE if teacher_checkpoint is not None else PLAIN_INFERENCE
    receipt = {"status": "predicting", "evaluation_family": FAMILY, "started_utc": _utc(),
               "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
               "workspace_root": reference["workspace_root"],
               "entrypoint_source": ({"path": str(Path(entrypoint).resolve()), "sha256": _file_hash(entrypoint)}
                                     if entrypoint is not None else None),
               "checkpoint_pixel_protocol": protocol_metadata(model_protocol), "training_manifest": lineage,
               "reference_manifest": {"path": reference["path"], "sha256": reference["sha256"]},
               "scoring_protocol": SCORING_PROTOCOL, "inference_protocol": inference_protocol,
               "source_read_policy": "All predictions complete before source RGB/annotation bytes are hashed or decoded",
               "runtime_versions": {name: importlib.metadata.version(name) for name in
                                    ("numpy", "pillow", "opencv-python-headless", "scikit-image", "torch", "lpips")}}
    if "dependency_provenance" in state:
        receipt["checkpoint_dependencies"] = state["dependency_provenance"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "rgb").mkdir()
    (output / "mask").mkdir()
    receipt_path = output / "execution_receipt.json"
    _json_write(receipt_path, receipt)
    try:
        scene, loaded_state = _load_scene(checkpoint_path, device)
        teacher = None
        if teacher_checkpoint is not None:
            from .render_ensemble import FixedRenderedTeacher
            teacher = FixedRenderedTeacher(_resolve(teacher_checkpoint, workspace_root), checkpoint_path,
                                           loaded_state, device=device)
            receipt["teacher_ensemble"] = teacher.receipt
        receipt["loaded_source_modules"] = _loaded_sources()
        if pixel_protocol(loaded_state) != model_protocol:
            raise ValueError("Loaded model profile differs from audited checkpoint")
        if state.get("dependency_provenance") != loaded_state.get("dependency_provenance"):
            raise ValueError("Loaded checkpoint dependency provenance changed")
        scene.eval()
        predictions = []
        for camera in reference["cameras"]:
            if teacher is None:
                rgb, mask, info = predict_official_camera(scene, camera, state)
            else:
                rgb, mask, info = predict_official_camera(scene, camera, state, teacher=teacher)
            stem = Path(camera["name"]).stem
            rgb_path, mask_path = output / "rgb" / f"{stem}.png", output / "mask" / f"{stem}.png"
            if not cv2.imwrite(str(rgb_path), rgb[..., ::-1]) or not cv2.imwrite(str(mask_path), mask):
                raise OSError("Could not write official prediction PNG")
            predictions.append({"name": camera["name"], "rgb": str(rgb_path), "mask": str(mask_path),
                                "rgb_sha256": _file_hash(rgb_path), "mask_sha256": _file_hash(mask_path), **info})
        if _file_hash(checkpoint_path) != checkpoint_sha:
            raise ValueError("Checkpoint changed during prediction")
        receipt.update(status="scoring", predictions_finished_utc=_utc(), predictions=predictions)
        _json_write(receipt_path, receipt)
        perceptual = _lpips(device)
        records, fingerprint_records = [], []
        matrix = np.zeros((5, 5), np.int64)
        receipt["source_scoring_started_utc"] = _utc()
        for camera, target, prediction in zip(reference["cameras"], reference["targets"], predictions, strict=True):
            # First source payload access occurs here, after the complete prediction pass.
            image_bytes = Path(target["source_image_path"]).read_bytes()
            truth = _decode_rgb(image_bytes, camera["width"], camera["height"], "source image")
            annotation_bytes = (Path(target["source_annotation_path"]).read_bytes()
                                if target["source_annotation_path"] else None)
            mask = (rasterize_official_annotation(annotation_bytes, camera["width"], camera["height"])
                    if annotation_bytes is not None else None)
            rgb_bytes, mask_bytes = Path(prediction["rgb"]).read_bytes(), Path(prediction["mask"]).read_bytes()
            if _hash(rgb_bytes) != prediction["rgb_sha256"] or _hash(mask_bytes) != prediction["mask_sha256"]:
                raise ValueError("Prediction files changed before scoring")
            rgb = _decode_rgb(rgb_bytes, camera["width"], camera["height"], "prediction")
            pred_mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            values = score_official_arrays(rgb, pred_mask, truth, mask, perceptual, device=device)
            if "confusion_matrix" in values:
                matrix += np.asarray(values["confusion_matrix"], np.int64)
            records.append({"name": camera["name"], "width": camera["width"], "height": camera["height"], **values})
            fingerprint_records.append({"camera": camera, "source_image_sha256": _hash(image_bytes),
                                        "source_annotation_sha256": _hash(annotation_bytes) if annotation_bytes is not None else None,
                                        "rasterized_mask_sha256": _hash(mask.tobytes(order="C")) if mask is not None else None})
        fingerprint = official_fingerprint(fingerprint_records)
        metrics = {"evaluation_family": FAMILY, "official_evaluation_fingerprint": fingerprint,
                   "scoring_protocol": SCORING_PROTOCOL, "inference_protocol": inference_protocol,
                   "validation_views": len(records),
                   "semantic_validation_views": sum("confusion_matrix" in r for r in records),
                   **{key: float(np.mean([r[key] for r in records])) for key in ("psnr", "ssim", "lpips")},
                   "confusion_matrix": matrix.tolist(), **iou_scores(torch.tensor(matrix)), "views": records}
        if teacher is not None:
            teacher.verify_inputs()
            metrics["teacher_ensemble"] = teacher.receipt
        for dependency in ("base_checkpoint", "manifest"):
            value = state.get("dependency_provenance", {}).get(dependency)
            if value is not None and _file_hash(value["path"]) != value["sha256"]:
                raise ValueError(f"Checkpoint {dependency} dependency changed during evaluation")
        _json_write(output / "official_metrics.json", metrics)
        receipt.update(status="completed", finished_utc=_utc(), official_evaluation_fingerprint=fingerprint,
                       official_metrics_sha256=_file_hash(output / "official_metrics.json"),
                       source_records=[dict(target, **record) for target, record in
                                       zip(reference["targets"], fingerprint_records, strict=True)])
        _json_write(receipt_path, receipt)
        return metrics
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_utc=_utc())
        _json_write(receipt_path, receipt)
        raise
