"""Bounded TRAIN-only refiner intervention audit. No optimization or teacher.

Use a locked plan and its explicit source PYTHONPATH; draft plans cannot run.
All ten predictions of a view finish before its GT/valid images are decoded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROTOCOL = "paired_render_student_sensitivity_corner_v2_v1"
PROFILE = "colmap_corner_v2"
CLASSES = ("background", "deck", "stay_cable", "tower", "foundation")
METHODS = ("raw_render", "paired_025", "paired_050", "real_rgb",
           "roll_plus_025", "roll_plus_050", "roll_minus_025", "roll_minus_050",
           "next_camera_050", "affine_rgb")
EPSILON = 1e-7


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def fixed_views(manifest):
    from bridge_rgs.coordinates import annotate_manifest, pixel_protocol
    manifest = annotate_manifest(manifest)
    if pixel_protocol(manifest) != PROFILE or tuple(manifest["class_names"]) != CLASSES:
        raise ValueError("Require the five-class colmap_corner_v2 manifest")
    views = manifest["views"]
    if len({v["name"] for v in views}) != len(views):
        raise ValueError("Duplicate manifest view names")
    population = sorted((v for v in views if v["split"] == "train" and v.get("mask_path")),
                        key=lambda v: v["name"])
    if len(population) != 259:
        raise ValueError("Require exactly 259 labeled TRAIN views")
    indices = [i * 258 // 15 for i in range(16)]
    selected = [population[i] for i in indices]
    if len({(v["height"], v["width"]) for v in selected}) != 1:
        raise ValueError("Camera-mismatch control requires identical native dimensions")
    return indices, selected


def protocol_spec():
    return {
        "methods": list(METHODS), "pixel_profile": PROFILE, "probability_log_epsilon": EPSILON,
        "refiner_config": {"type": "multiscale", "channels": 64, "residual_bound": 6.0, "context": "pyramid_strip"},
        "sample_rule": "259 labeled TRAIN names sorted, floor(i*258/15), i=0..15",
        "clipping": {"raw_render": "unchanged raw float, never clamp or quantize",
                     "paired_025_050": "(1-t)*raw_R+t*real_I, no clipping",
                     "real_rgb": "prepared TRAIN uint8 decoded RGB /255, no other transform",
                     "roll_next_affine": "clip candidate to [0,1]; preserve and report preclip statistics"},
        "roll": "plus/minus floor(H/3),floor(W/3); same shift for all three channels",
        "next_camera": "cyclic next of fixed16; per-channel RMS scaled to target residual; zero source RMS gives zero",
        "affine": "per-channel least-squares gain/bias on fixed grid [::16,::16], no GT; constant input falls back to mean offset",
        "local_statistics": "32x32 non-overlapping blocks including incomplete edge blocks",
        "roll_seam_region": "union of 5-pixel-wide bands centered at +/-roll wrap rows and columns, clipped to image bounds",
        "primary": "per-view equal-present-GT-class mean margin/NLL, then equal-camera mean",
        "primary_contrasts": ["paired_050-minus-roll_plus_050", "paired_050-minus-roll_minus_050",
                              "paired_050-minus-next_camera_050", "paired_050-minus-affine_rgb"],
        "bootstrap": {"unit": "camera", "repeats": 5000, "seed": 20260926, "interval": "percentile95"},
        "max_cameras": 16, "max_wall_seconds": 300, "torch_threads": 8,
        "deadline_enforcement": "cooperative checks before render/head and after each scored view; an individual operation is not preempted",
        "budget": "16 evidence renders +16 native verification renders; 160 intervention heads +16 native heads",
        "scoring": "Decode GT/valid only after all ten predictions for that camera; earlier byte hashes are provenance only",
        "scope": "TRAIN real RGB is a diagnostic intervention, not camera-only performance; no teacher/optimization/cache export",
    }


def validate_rgb(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected finite HWC float RGB")
    return values


def affine_candidate(raw, real):
    """Six scalar coefficients fitted on a fixed pixel lattice, never on GT."""
    raw, real = validate_rgb(raw), validate_rgb(real)
    if raw.shape != real.shape:
        raise ValueError("RGB grids differ")
    x, y = raw[::16, ::16].reshape(-1, 3).astype(np.float64), real[::16, ::16].reshape(-1, 3).astype(np.float64)
    centered_x, centered_y = x - x.mean(0), y - y.mean(0)
    denominator = (centered_x ** 2).sum(0)
    gain = np.divide((centered_x * centered_y).sum(0), denominator,
                     out=np.ones(3), where=denominator > 1e-12)
    bias = y.mean(0) - gain * x.mean(0)
    return (raw * gain.astype(np.float32) + bias.astype(np.float32)), {"gain": gain.tolist(), "bias": bias.tolist()}


def interventions(raw, real, next_residual):
    """Yield exactly ten inputs, including a byte-preserving raw baseline."""
    raw, real, next_residual = map(validate_rgb, (raw, real, next_residual))
    if raw.shape != real.shape or raw.shape != next_residual.shape:
        raise ValueError("All intervention grids must match")
    if (real < 0).any() or (real > 1).any():
        raise ValueError("Decoded real RGB must be within [0,1]")
    delta = real - raw
    yield "raw_render", raw, raw, {}
    for t in (.25, .5):
        candidate = (1 - t) * raw + t * real
        yield f"paired_{int(t*100):03d}", candidate, candidate, {"t": t}
    yield "real_rgb", real, real, {}
    shift = (raw.shape[0] // 3, raw.shape[1] // 3)
    for sign, name in ((1, "plus"), (-1, "minus")):
        rolled = np.roll(delta, tuple(sign * value for value in shift), axis=(0, 1))
        for t in (.25, .5):
            before = raw + t * rolled
            yield f"roll_{name}_{int(t*100):03d}", np.clip(before, 0, 1), before, {"t": t, "shift": [sign * x for x in shift]}
    target_rms = np.sqrt(np.mean(delta.astype(np.float64) ** 2, axis=(0, 1)))
    source_rms = np.sqrt(np.mean(next_residual.astype(np.float64) ** 2, axis=(0, 1)))
    scale = np.divide(target_rms, source_rms, out=np.zeros(3), where=source_rms > 0)
    scaled_next = (next_residual.astype(np.float64) * scale).astype(np.float32)
    before = raw + .5 * scaled_next
    yield "next_camera_050", np.clip(before, 0, 1), before, {
        "per_channel_scale": scale.tolist(), "source_zero_rms": (source_rms == 0).tolist(),
        "source_rms": source_rms.tolist(), "target_rms": target_rms.tolist()}
    before, coefficients = affine_candidate(raw, real)
    yield "affine_rgb", np.clip(before, 0, 1), before, coefficients


def quantiles(values):
    return {str(q): float(x) for q, x in zip((0, .1, .5, .9, .99, 1),
               np.quantile(values, (0, .1, .5, .9, .99, 1)), strict=True)}


def input_statistics(raw, candidate, preclip):
    delta = candidate - raw
    energy = np.mean(delta.astype(np.float64) ** 2, axis=-1)
    pre_energy = np.mean((preclip.astype(np.float64) - raw) ** 2, axis=-1)
    clipped = np.any(candidate != preclip, axis=-1)
    blocks = []
    for y in range(0, raw.shape[0], 32):
        for x in range(0, raw.shape[1], 32):
            region = np.s_[y:y+32, x:x+32]
            blocks.append({"y": y, "x": x, "rms": float(np.sqrt(energy[region].mean())),
                           "preclip_rms": float(np.sqrt(pre_energy[region].mean())),
                           "clip_pixel_fraction": float(clipped[region].mean())})
    return {"raw_out_of_range_channel_fraction": float(((raw < 0) | (raw > 1)).mean()),
            "raw_out_of_range_pixel_fraction": float(np.any((raw < 0) | (raw > 1), -1).mean()),
            "candidate_out_of_range_channel_fraction": float(((candidate < 0) | (candidate > 1)).mean()),
            "preclip_out_of_range_channel_fraction": float(((preclip < 0) | (preclip > 1)).mean()),
            "clip_channel_fraction": float((candidate != preclip).mean()),
            "clip_pixel_fraction": float(clipped.mean()), "raw_min": float(raw.min()), "raw_max": float(raw.max()),
            "actual_rms": float(np.sqrt(energy.mean())), "preclip_rms": float(np.sqrt(pre_energy.mean())),
            "actual_per_channel_rms": np.sqrt(np.mean(delta.astype(np.float64) ** 2, (0, 1))).tolist(),
            "pixel_rms_quantiles": quantiles(np.sqrt(energy)), "blocks32": blocks}, energy.astype(np.float32)


def probabilities(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 5 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite five-class nonnegative HWC probabilities")
    mass = values.sum(-1, keepdims=True)
    if (mass <= 0).any():
        raise ValueError("Zero probability mass")
    return values / mass


def boundary_band(target, keep):
    trusted = cv2.erode(keep.astype(np.uint8), np.ones((7, 7), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    edge = np.zeros_like(keep)
    for category in range(5):
        binary = ((target == category) & keep).astype(np.uint8)
        edge |= (binary - cv2.erode(binary, np.ones((3, 3), np.uint8),
                                   borderType=cv2.BORDER_CONSTANT, borderValue=0)) > 0
    return cv2.dilate((edge & trusted).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & trusted


def iou_from_cm(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    intersection = matrix.diagonal()
    union = matrix.sum(0) + matrix.sum(1) - intersection
    values = np.divide(intersection, union, out=np.full(5, np.nan), where=union > 0)
    return {"per_class": [float(v) if np.isfinite(v) else None for v in values],
            "all5": float(np.nanmean(values)) if (union > 0).any() else None}


def summarize_predictions(predictions, energies, target, valid):
    """Post-prediction scoring, preserving pixel sums and class/camera estimands."""
    if tuple(predictions) != METHODS or tuple(energies) != METHODS:
        raise ValueError("Exactly ten ordered interventions are required before scoring")
    target, valid = np.asarray(target), np.asarray(valid, dtype=bool)
    if target.shape != valid.shape or target.ndim != 2:
        raise ValueError("Invalid GT/valid grid")
    keep = valid & (target >= 0) & (target < 5)
    if not keep.any():
        raise ValueError("No scored TRAIN pixels")
    safe_target = np.where(keep, target, 0).astype(np.int64)
    base = probabilities(predictions["raw_render"])
    if base.shape[:2] != target.shape:
        raise ValueError("Prediction and label grids differ")
    base_log = np.log(np.maximum(base, EPSILON))
    base_pred = base.argmax(-1)
    correct_base = base_pred == target
    edge = boundary_band(target, keep)
    regions = {"all": keep, "boundary2": edge, "nonboundary": keep & ~edge,
               "baseline_correct": keep & correct_base, "baseline_wrong": keep & ~correct_base}
    seam = np.zeros_like(keep)
    h, w = keep.shape
    for y in (h//3, (-(h//3)) % h):
        seam[max(0, y-2):min(h, y+3)] = True
    for x in (w//3, (-(w//3)) % w):
        seam[:, max(0, x-2):min(w, x+3)] = True
    regions["roll_wrap_seam_union2"] = keep & seam
    base_target = np.take_along_axis(base_log, safe_target[..., None], -1)[..., 0]
    wrong_base = base_log.copy()
    np.put_along_axis(wrong_base, safe_target[..., None], -np.inf, -1)
    base_margin = base_target - wrong_base.max(-1)
    result = {}
    for name in METHODS:
        p = probabilities(predictions[name])
        if p.shape != base.shape or energies[name].shape != target.shape:
            raise ValueError("Mismatched intervention grids")
        log_p = np.log(np.maximum(p, EPSILON))
        log_target = np.take_along_axis(log_p, safe_target[..., None], -1)[..., 0]
        wrong = log_p.copy()
        np.put_along_axis(wrong, safe_target[..., None], -np.inf, -1)
        margin = log_target - wrong.max(-1)
        mixture_log = np.log(np.maximum((base + p) * .5, EPSILON))
        js = np.maximum(0, .5 * (base * (base_log - mixture_log) + p * (log_p - mixture_log)).sum(-1))
        centered_delta = (log_p - log_p.mean(-1, keepdims=True)) - (base_log - base_log.mean(-1, keepdims=True))
        l2 = (centered_delta ** 2).mean(-1)
        pred = p.argmax(-1)
        values = {"nll_sum": -log_target, "margin_sum": margin, "margin_delta_sum": margin-base_margin,
                  "nll_gain_sum": log_target-base_target, "js_sum": js,
                  "centered_log_probability_mse_sum": l2, "rgb_energy_sum": energies[name]}

        scored_regions = {}
        for region_name, selection in regions.items():
            labels = target[selection].astype(np.int64)
            predicted = pred[selection]
            original_correct = correct_base[selection]
            cm = np.bincount(labels * 5 + predicted, minlength=25).reshape(5, 5)
            counts_by_class = {"pixels": np.bincount(labels, minlength=5),
                               "corrections": np.bincount(labels, weights=(predicted == labels) & ~original_correct, minlength=5),
                               "harms": np.bincount(labels, weights=(predicted != labels) & original_correct, minlength=5),
                               "wrong": np.bincount(labels, weights=predicted != labels, minlength=5)}
            sums_by_class = {key: np.bincount(labels, weights=value[selection], minlength=5) for key, value in values.items()}
            per_class = {class_name: {**{key: int(value[category]) for key, value in counts_by_class.items()},
                                      **{key: float(value[category]) for key, value in sums_by_class.items()}}
                         for category, class_name in enumerate(CLASSES)}
            present = [value for value in per_class.values() if value["pixels"]]
            macro = {key.removesuffix("_sum"): float(np.mean([value[key] / value["pixels"] for value in present]))
                     if present else None for key in values}
            scored_regions[region_name] = {"counts": add_counts(list(per_class.values())), "by_gt_class": per_class,
                                           "class_equal": macro, "classes_present": len(present),
                                           "confusion_matrix": cm.tolist(), "iou": iou_from_cm(cm)}
        result[name] = {"regions": scored_regions}
    return result


def add_counts(rows):
    return {key: sum(row[key] for row in rows) for key in rows[0]}


def rates(counts):
    n = counts["pixels"]
    return {**counts, **{key.removesuffix("_sum") + "_per_pixel": value / n if n else None
                        for key, value in counts.items() if key.endswith("_sum")},
            "js_per_rgb_energy": counts["js_sum"] / (counts["rgb_energy_sum"] + 1e-8 * n) if n else None}


def interval(values, indices):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Paired scalar interval needs finite camera values")
    sampled = values[indices].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(sampled, (.025, .975)).tolist(),
            "per_camera": values.tolist()}


def aggregate_views(rows, repeats=5000, seed=20260926):
    if not rows:
        raise ValueError("No scored views")
    rng = np.random.default_rng(seed)
    boot = rng.integers(len(rows), size=(repeats, len(rows)))
    summary = {}
    for name in METHODS:
        regions = {}
        for region in rows[0]["scores"][name]["regions"]:
            values = [row["scores"][name]["regions"][region] for row in rows]
            cm = np.sum([value["confusion_matrix"] for value in values], axis=0)
            camera_macro = {key: [value["class_equal"][key] for value in values if value["class_equal"][key] is not None]
                            for key in values[0]["class_equal"]}
            regions[region] = {"pooled_counts": rates(add_counts([value["counts"] for value in values])),
                               "by_gt_class": {c: rates(add_counts([value["by_gt_class"][c] for value in values])) for c in CLASSES},
                               "equal_camera_equal_present_class": {key: float(np.mean(v)) if v else None for key, v in camera_macro.items()},
                               "nonempty_cameras": sum(value["counts"]["pixels"] > 0 for value in values),
                               "confusion_matrix": cm.tolist(), "iou": iou_from_cm(cm)}
        summary[name] = regions
    comparisons = [(name, "raw_render") for name in METHODS if name != "raw_render"]
    comparisons += [("paired_050", name) for name in ("roll_plus_050", "roll_minus_050", "next_camera_050", "affine_rgb")]
    paired = {}
    for candidate, reference in comparisons:
        item = {}
        for key in ("margin", "nll", "margin_delta", "nll_gain", "js", "centered_log_probability_mse"):
            values = [row["scores"][candidate]["regions"]["all"]["class_equal"][key]
                      - row["scores"][reference]["regions"]["all"]["class_equal"][key] for row in rows]
            item[key + "_candidate_minus_reference"] = interval(values, boot)
        a = np.asarray([row["scores"][candidate]["regions"]["all"]["confusion_matrix"] for row in rows])
        b = np.asarray([row["scores"][reference]["regions"]["all"]["confusion_matrix"] for row in rows])
        deltas = [100 * (iou_from_cm(a[index].sum(0))["all5"] - iou_from_cm(b[index].sum(0))["all5"]) for index in boot]
        item["secondary_all5_pp"] = {"point": 100 * (iou_from_cm(a.sum(0))["all5"] - iou_from_cm(b.sum(0))["all5"]),
                                     "ci95": np.quantile(deltas, (.025, .975)).tolist()}
        paired[f"{candidate}-minus-{reference}"] = item
    explanation = {}
    for key in ("margin_delta", "nll_gain"):
        a = summary["paired_050"]["all"]["equal_camera_equal_present_class"][key]
        b = summary["affine_rgb"]["all"]["equal_camera_equal_present_class"][key]
        explanation[key] = {"paired": a, "affine": b, "paired_minus_affine": a-b,
                            "affine_over_paired_descriptive_ratio": b/a if abs(a) > 1e-8 else None}
    return {"aggregate": summary, "paired": paired, "affine_explanation": explanation,
            "bootstrap": {"repeats": repeats, "seed": seed, "unit": "camera; no independent-scene inference"}}


@torch.inference_mode()
def head_prediction(refiner, rendered, rgb):
    kwargs = {"depth_moments": rendered["depth_moments"]} if "depth_moments" in rendered else {}
    prior = rendered["refinement_prior"]
    residual = refiner(rendered["features"], rgb, rendered["depth"], rendered["alpha"], p3d=prior, **kwargs)
    return (prior.log() + residual).softmax(-1)


def check_deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("Fixed 300-second audit budget exhausted; no automatic retry")


def response_statistics(base, candidate, blocks):
    """Label-free output response, measured on the same fixed spatial blocks."""
    base, candidate = probabilities(base), probabilities(candidate)
    log_base, log_candidate = np.log(np.maximum(base, EPSILON)), np.log(np.maximum(candidate, EPSILON))
    mixture_log = np.log(np.maximum(.5*(base+candidate), EPSILON))
    js = np.maximum(0, .5*(base*(log_base-mixture_log) + candidate*(log_candidate-mixture_log)).sum(-1))
    difference = (log_candidate-log_candidate.mean(-1, keepdims=True)) - (log_base-log_base.mean(-1, keepdims=True))
    mse = (difference**2).mean(-1)
    for block in blocks:
        y, x = block["y"], block["x"]
        block["output_js_mean"] = float(js[y:y+32, x:x+32].mean())
        block["centered_log_probability_mse_mean"] = float(mse[y:y+32, x:x+32].mean())
    return {"js_quantiles_all_pixels": quantiles(js),
            "centered_log_probability_rms_quantiles_all_pixels": quantiles(np.sqrt(mse))}


@torch.inference_mode()
def predict_then_score(refiner, rendered, real, next_residual, label_reader, deadline=float("inf")):
    raw = rendered["rgb"].detach().cpu().numpy().copy()
    predictions, energies, diagnostics = {}, {}, {}
    frozen_keys = ("features", "rgb", "depth", "alpha", "refinement_prior", "p3d")
    before = {key: rendered[key].clone() for key in frozen_keys}
    if "depth_moments" in rendered:
        before["depth_moments"] = rendered["depth_moments"].clone()
    for name, candidate, preclip, metadata in interventions(raw, real, next_residual):
        check_deadline(deadline)
        candidate, preclip = validate_rgb(candidate), validate_rgb(preclip)
        rgb = rendered["rgb"] if name == "raw_render" else torch.from_numpy(candidate).to(rendered["rgb"].device)
        predictions[name] = head_prediction(refiner, rendered, rgb).cpu().numpy().copy()
        statistics, energy = input_statistics(raw, candidate, preclip)
        statistics["label_free_output_response"] = response_statistics(
            predictions["raw_render"], predictions[name], statistics["blocks32"])
        diagnostics[name], energies[name] = {**statistics, "intervention_metadata": metadata}, energy
    if tuple(predictions) != METHODS:
        raise ValueError("Intervention ordering changed")
    if not all(torch.equal(rendered[key], value) for key, value in before.items()):
        raise ValueError("Intervention modified frozen evidence")
    # This call is the first GT/valid decode in the prediction/scoring pipeline.
    target, valid = label_reader()
    scores = summarize_predictions(predictions, energies, target, valid)
    check_deadline(deadline)
    return scores, diagnostics, predictions["raw_render"]


def verify_plan(plan, runner_path):
    if plan.get("status") != "locked" or plan.get("protocol") != PROTOCOL:
        raise ValueError("Only a completed-input locked plan may execute; pending draft is not runnable")
    if plan["specification"] != protocol_spec() or digest(runner_path) != plan["runner_sha256"]:
        raise ValueError("Runner or fixed protocol changed")
    checkpoint = plan["input_hashes"]["student_checkpoint"]
    if (plan["checkpoint"] != checkpoint["path"] or plan["checkpoint_sha256"] != checkpoint["sha256"]
            or plan["manifest"] != plan["input_hashes"]["manifest"]["path"]):
        raise ValueError("Plan paths/SHA are not bound to the locked input records")
    for value in plan["input_hashes"].values():
        if digest(value["path"]) != value["sha256"]:
            raise ValueError(f"Input changed: {value['path']}")
    snapshot = Path(plan["source_snapshot"])
    current_hashes = {str(path.relative_to(snapshot)): digest(path) for path in sorted(snapshot.rglob("*.py"))}
    if current_hashes != plan["source_hashes"]:
        raise ValueError("Frozen package inventory/content changed")


def verify_imports(plan):
    snapshot = Path(plan["source_snapshot"]).resolve()
    result = {}
    for name in ("bridge_rgs", "bridge_rgs.train", "bridge_rgs.model", "bridge_rgs.refinement",
                 "bridge_rgs.coordinates", "bridge_rgs.io"):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to(snapshot):
            raise ValueError(f"Actual import is outside frozen snapshot: {name}")
        relative = str(path.relative_to(snapshot))
        if digest(path) != plan["source_hashes"].get(relative):
            raise ValueError(f"Imported source hash mismatch: {name}")
        result[name] = record(path)
    return result


def decode_rgb(view):
    image = cv2.imread(view["image_path"], cv2.IMREAD_COLOR)
    if image is None or image.shape != (view["height"], view["width"], 3):
        raise ValueError("Prepared TRAIN RGB grid differs")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def label_reader(view):
    def read():
        target = cv2.imread(view["mask_path"], cv2.IMREAD_UNCHANGED)
        valid = cv2.imread(view["valid_path"], 0) if view.get("valid_path") else None
        if target is None or target.shape != (view["height"], view["width"]):
            raise ValueError("TRAIN GT grid differs")
        if view.get("valid_path") and (valid is None or valid.shape != target.shape):
            raise ValueError("TRAIN valid grid differs")
        return target, valid > 0 if valid is not None else np.ones(target.shape, bool)
    return read


@torch.inference_mode()
def execute(plan):
    from bridge_rgs.coordinates import pixel_protocol
    from bridge_rgs.train import load_scene
    manifest = json.loads(Path(plan["manifest"]).read_text())
    indices, views = fixed_views(manifest)
    if indices != plan["sample_indices"] or [v["name"] for v in views] != plan["sample_names"]:
        raise ValueError("Locked TRAIN sample changed")
    started = time.monotonic()
    deadline = started + 300
    scene, state = load_scene(plan["checkpoint"])
    if pixel_protocol(state) != PROFILE or state.get("manifest_sha256") != digest(plan["manifest"]):
        raise ValueError("Checkpoint profile/manifest binding differs")
    scene.eval().requires_grad_(False)
    before = {key: value.detach().cpu().clone() for key, value in scene.state_dict().items()}
    metadata_keys = ("feature_dim", "sh_degree", "scene_scale", "refiner_config", "pixel_protocol", "manifest_sha256")
    metadata_before = json.dumps({key: getattr(scene, key) for key in metadata_keys}, sort_keys=True)
    rows = []
    torch.cuda.reset_peak_memory_stats()

    def render(view, refine):
        check_deadline(deadline)
        K = torch.tensor(view["K"], dtype=torch.float32, device="cuda")
        pose = torch.tensor(view["w2c_original"], dtype=torch.float32, device="cuda")
        K_before, pose_before = K.clone(), pose.clone()
        result = scene.render(K, pose, view["width"], view["height"], refine=refine, absgrad=False)
        if not torch.equal(K, K_before) or not torch.equal(pose, pose_before):
            raise ValueError("Renderer mutated camera inputs")
        return result

    current = render(views[0], False)
    real = decode_rgb(views[0])
    first_delta = real - current["rgb"].cpu().numpy()
    for i, view in enumerate(views):
        following_view = views[(i + 1) % 16]
        if i < 15:
            following = render(following_view, False)
            following_real = decode_rgb(following_view)
            next_delta = following_real - following["rgb"].cpu().numpy()
        else:
            following, following_real, next_delta = None, None, first_delta
        native = render(view, True)
        for key in ("features", "rgb", "depth", "alpha", "p3d", "refinement_prior"):
            if not torch.equal(native[key], current[key]):
                raise ValueError(f"Independent native/evidence render differs: {view['name']} {key}")
        native_probability = native["probabilities"].cpu().numpy().copy()
        del native
        scores, inputs, baseline = predict_then_score(scene.refiner, current, real, next_delta,
                                                      label_reader(view), deadline)
        if not np.array_equal(baseline, native_probability):
            raise ValueError(f"Raw baseline does not exactly reproduce native inference: {view['name']}")
        rows.append({"name": view["name"], "next_camera_name": following_view["name"],
                     "native_probability_exact": True, "frozen_evidence_exact": True,
                     "all_ten_predictions_before_gt_decode": True, "scores": scores,
                     "input_diagnostics": inputs})
        current, real = following, following_real
        print(f"paired sensitivity {i+1}/16 {view['name']} elapsed={time.monotonic()-started:.2f}s", flush=True)
    if not all(torch.equal(value.detach().cpu(), before[key]) for key, value in scene.state_dict().items()):
        raise ValueError("Frozen scene model tensor changed")
    if metadata_before != json.dumps({key: getattr(scene, key) for key in metadata_keys}, sort_keys=True):
        raise ValueError("Frozen scene metadata changed")
    torch.cuda.synchronize()
    inference_scoring_seconds = time.monotonic() - started
    check_deadline(deadline)
    # Scalar/CM bootstrap is CPU-only; no further rendering occurs here.
    scene = None
    del current, state, before
    result = aggregate_views(rows)
    return {"status": "completed", "protocol": PROTOCOL, "specification": protocol_spec(),
            "sample_names": plan["sample_names"], "sample_indices": indices, "per_view": rows,
            **result, "model_tensors_unchanged": True, "scene_metadata_unchanged": True,
            "camera_inputs_unchanged": True, "native_baseline_exact_all16": True,
            "render_calls": 32, "intervention_head_calls": 160, "native_verification_head_calls": 16,
            "gpu_inference_and_cpu_scoring_elapsed_seconds": inference_scoring_seconds,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "limits": ["TRAIN interventions use real RGB; this is not camera-only inference performance.",
                       "These TRAIN labels/views were used by the warmstart; saturated in-sample evidence is not validation.",
                       "Raw/paired inputs are unclipped; roll/next/affine clip. Matching preclip norm/spectrum does not match postclip local difficulty.",
                       "Aligned real detail restoration and off-manifold mismatch remain alternative explanations even for positive half-path effects.",
                       "No teacher, KD, optimization, unlabeled/VAL scoring, or new probability/image cache."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    verify_plan(plan, __file__)
    imports = verify_imports(plan)
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.plan, output / "plan.json")
    shutil.copy2(__file__, output / Path(__file__).name)
    receipt = {"status": "running", "pid": os.getpid(), "plan_sha256": digest(args.plan),
               "input_hashes": plan["input_hashes"], "source_hashes": plan["source_hashes"],
               "actual_imports": imports, "runner_sha256": digest(__file__)}
    receipt_path = output / "execution_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        result = execute(plan)
        verify_plan(plan, __file__)
        path = output / "report.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        receipt.update(status="completed", report=record(path))
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
