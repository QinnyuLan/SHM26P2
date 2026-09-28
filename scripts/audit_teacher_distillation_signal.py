"""Read-only, fixed 16 TRAIN-view teacher/student signal audit; no DINO forward."""

import argparse
import hashlib
import importlib
import json
import os
import shutil
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import torch

from bridge_rgs.coordinates import LEGACY, pixel_protocol
from bridge_rgs.teacher import verify_pseudo_provenance
from bridge_rgs.train import load_scene

CONFIDENCE_EDGES = (0., .2, .4, .6, .8, .9, .95, 1.)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fixed_views(manifest):
    population = sorted([view for view in manifest["views"]
                         if view["split"] == "train" and view.get("mask_path")], key=lambda view: view["name"])
    if len(population) != 259 or len({view["name"] for view in population}) != 259:
        raise ValueError("Expected the original 259 unique labeled TRAIN views")
    indices = [index * (len(population) - 1) // 15 for index in range(16)]
    return indices, [population[index] for index in indices]


def normalize_probabilities(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 5 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite nonnegative five-class CHW probabilities")
    mass = values.sum(0, keepdims=True)
    if (mass <= 0).any():
        raise ValueError("Probability mass is zero")
    return values / mass


def soft_kl(teacher, student):
    """Temperature-one KL(T||S), FP32 renormalization and log epsilon 1e-8."""
    teacher, student = normalize_probabilities(teacher), normalize_probabilities(student)
    if teacher.shape != student.shape:
        raise ValueError("Teacher/student probability grids differ")
    return np.maximum(0., (teacher * (np.log(np.maximum(teacher, 1e-8))
                                     - np.log(np.maximum(student, 1e-8)))).sum(0))


def boundary_band(target, keep):
    """Union of inner GT class edges, dilated by 2 pixels; 7x7 trusted support."""
    trusted = cv2.erode(keep.astype(np.uint8), np.ones((7, 7), np.uint8),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    edge = np.zeros_like(keep, bool)
    for category in range(5):
        binary = ((target == category) & keep).astype(np.uint8)
        edge |= (binary - cv2.erode(binary, np.ones((3, 3), np.uint8),
                                   borderType=cv2.BORDER_CONSTANT, borderValue=0)) > 0
    return cv2.dilate((edge & trusted).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & trusted


def pixel_counts(target, tpred, spred, confidence, kl, selected):
    tc, sc = tpred == target, spred == target
    count = int(np.count_nonzero(selected))
    return {"pixels": count,
            "teacher_correct_student_wrong": int(np.count_nonzero(selected & tc & ~sc)),
            "teacher_wrong_student_correct": int(np.count_nonzero(selected & ~tc & sc)),
            "both_correct": int(np.count_nonzero(selected & tc & sc)),
            "both_wrong": int(np.count_nonzero(selected & ~tc & ~sc)),
            "teacher_wrong": int(np.count_nonzero(selected & ~tc)),
            "student_wrong": int(np.count_nonzero(selected & ~sc)),
            "prediction_disagreement": int(np.count_nonzero(selected & (tpred != spred))),
            "confidence_sum": float(confidence[selected].sum(dtype=np.float64)),
            "soft_kl_teacher_to_student_sum": float(kl[selected].sum(dtype=np.float64))}


def summarize_view(target, keep, teacher, student, confidence, classes):
    if target.shape != keep.shape or teacher.shape[1:] != target.shape or confidence.shape != target.shape:
        raise ValueError("Scored native grids differ")
    if not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
        raise ValueError("Invalid confidence values")
    teacher, student = normalize_probabilities(teacher), normalize_probabilities(student)
    tpred, spred = teacher.argmax(0), student.argmax(0)
    kl = soft_kl(teacher, student)
    keep = keep & (target < 5)
    edge = boundary_band(target, keep)
    regions = {"all": keep, "gt_boundary_band2": edge,
               "gt_cable": keep & (target == 2), "gt_foundation": keep & (target == 4),
               "gt_cable_boundary_band2": edge & (target == 2),
               "gt_foundation_boundary_band2": edge & (target == 4)}
    accepted = confidence >= .8

    def statistics(selection):
        return pixel_counts(target, tpred, spred, confidence, kl, selection)

    result = {"regions": {}, "by_gt_class": {}, "by_teacher_predicted_class": {}}
    for name, region in regions.items():
        bins = {}
        for index, (low, high) in enumerate(pairwise(CONFIDENCE_EDGES)):
            in_bin = (confidence >= low) & ((confidence <= high) if index == len(CONFIDENCE_EDGES) - 2 else (confidence < high))
            bins[f"[{low:g},{high:g}{']' if high == 1 else ')'}"] = statistics(region & in_bin)
        result["regions"][name] = {"all": statistics(region), "accepted_ge_0.8": statistics(region & accepted),
                                    "rejected_lt_0.8": statistics(region & ~accepted), "confidence_bins": bins}
    for category, name in enumerate(classes):
        for key, labels in (("by_gt_class", target), ("by_teacher_predicted_class", tpred)):
            selected = keep & (labels == category)
            result[key][name] = {"all": statistics(selected), "accepted_ge_0.8": statistics(selected & accepted)}
    return result


def merge_counts(first, second):
    if first is None:
        return second
    if isinstance(first, dict):
        if first.keys() != second.keys():
            raise ValueError("Mismatched count keys")
        return {key: merge_counts(first[key], second[key]) for key in first}
    return first + second


def rates(value):
    """Derive ratios only after adding pixel counts, never average per-view rates."""
    result = {key: rates(item) if isinstance(item, dict) else item for key, item in value.items()}
    if "pixels" in value:
        count = value["pixels"]
        for field in ("teacher_wrong", "student_wrong", "prediction_disagreement", "teacher_correct_student_wrong",
                      "teacher_wrong_student_correct", "confidence_sum", "soft_kl_teacher_to_student_sum"):
            result[field + "_per_pixel"] = value[field] / count if count else None
        result["net_teacher_correct_pixel_advantage"] = value["teacher_correct_student_wrong"] - value["teacher_wrong_student_correct"]
    if "all" in value and "accepted_ge_0.8" in value:
        total, accepted = value["all"], value["accepted_ge_0.8"]
        result["accepted_coverage"] = accepted["pixels"] / total["pixels"] if total["pixels"] else None
        corrections = total["teacher_correct_student_wrong"]
        result["teacher_correction_retention_at_0.8"] = accepted["teacher_correct_student_wrong"] / corrections if corrections else None
    return result


def verify_inputs(plan):
    for record in plan["input_hashes"].values():
        if digest(record["path"]) != record["sha256"]:
            raise ValueError(f"Input changed: {record['path']}")
    for path, sha in plan["pseudo_files"].items():
        if digest(path) != sha:
            raise ValueError(f"Pseudo changed: {path}")
    for key, sha in plan["source_hashes"].items():
        if digest(Path(plan["source_snapshot"]) / key) != sha:
            raise ValueError(f"Snapshot changed: {key}")


@torch.inference_mode()
def execute(plan, output):
    # Full directory verification includes all 350 files, even though only 16
    # predeclared labeled TRAIN views enter the actual signal audit.
    provenance = verify_pseudo_provenance(plan["manifest"], plan["pseudo_dir"], verify_files=True)
    manifest = json.loads(Path(plan["manifest"]).read_text())
    indices, views = fixed_views(manifest)
    assert indices == plan["sample_indices"] and [view["name"] for view in views] == plan["sample_names"]
    assert len(provenance["views"]) == 350 and pixel_protocol(manifest) == LEGACY
    scene, state = load_scene(plan["checkpoint"])
    assert pixel_protocol(state) == LEGACY
    scene.eval().requires_grad_(False)
    before = {key: value.detach().cpu().clone() for key, value in scene.state_dict().items()}
    accumulated, per_view = None, []
    classes = manifest["class_names"]
    assert classes == ["background", "deck", "stay_cable", "tower", "foundation"]
    for view in views:
        assert view["split"] == "train" and view["mask_path"]
        K = torch.tensor(view["K"], dtype=torch.float32, device="cuda")
        pose = torch.tensor(view["w2c_original"], dtype=torch.float32, device="cuda")
        rendered = scene.render(K, pose, view["width"], view["height"], absgrad=False)
        student = rendered["probabilities"].permute(2, 0, 1).cpu().numpy()
        del rendered
        with np.load(Path(plan["pseudo_dir"]) / (Path(view["name"]).stem + ".npz"), allow_pickle=False) as values:
            teacher = values["probs"].astype(np.float32)
            confidence = values["confidence"].astype(np.float32)
            pseudo_valid = values["valid"].astype(bool)
        # Both predictions are complete here. Only now open TRAIN labels/valid.
        target = cv2.imread(view["mask_path"], cv2.IMREAD_UNCHANGED)
        assert target is not None and target.shape == (view["height"], view["width"])
        valid = cv2.imread(view["valid_path"], 0) > 0 if view.get("valid_path") else np.ones(target.shape, bool)
        assert np.array_equal(valid, pseudo_valid)
        counts = summarize_view(target, valid, teacher, student, confidence, classes)
        accumulated = merge_counts(accumulated, counts)
        per_view.append({"name": view["name"], "split": "train", "predictions_completed_before_label_read": True,
                         "scores": rates(counts)})
        print(f"signal audit {len(per_view)}/16 {view['name']}", flush=True)
    assert all(torch.equal(value.detach().cpu(), before[key]) for key, value in scene.state_dict().items())
    return {"status": "completed", "protocol": plan["protocol"], "sample_names": plan["sample_names"],
            "sample_indices": indices, "full_pseudo_provenance_verified": True,
            "model_tensors_unchanged": True, "no_teacher_forward_or_optimization": True,
            "confidence_definition": provenance["confidence_definition"],
            "confidence_bins": list(CONFIDENCE_EDGES), "threshold": .8,
            "soft_kl": "KL(teacher||student), temperature 1, both FP32-renormalized over classes, log epsilon 1e-8, mean over each region's labeled valid pixels",
            "gt_boundary": "Union of 3x3 inner GT class edges, 5x5 dilation (2-pixel band), 7x7 valid/known-label eroded support",
            "aggregate": rates(accumulated), "per_view": per_view,
            "limits": ["Both teacher and student optimized on these TRAIN labels; correctness is optimistic in-sample evidence.",
                       "The fixed 16-view sample does not establish quality on 91 unlabeled views, VAL, other seeds, or other bridges.",
                       "Teacher probabilities came from real TRAIN RGB; current student predictions use only camera metadata and the frozen scene.",
                       "This is a descriptive audit to assess a possible classical final-only soft-KD compression control, not a performance or novelty claim."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/generated_teacher_signal_audit/plan.json"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    plan = json.loads(args.plan.read_text())
    assert plan["protocol"] == "fixed_16_labeled_train_teacher_signal_v1"
    assert len(plan["sample_names"]) == 16
    assert digest(__file__) == plan["runner_sha256"]
    snapshot = Path(plan["source_snapshot"]).resolve()
    imports = {}
    for name in ("bridge_rgs.teacher", "bridge_rgs.model", "bridge_rgs.train", "bridge_rgs.coordinates", "bridge_rgs.refinement"):
        path = Path(importlib.import_module(name).__file__).resolve()
        assert path.is_relative_to(snapshot)
        imports[name] = {"path": str(path), "sha256": digest(path)}
    verify_inputs(plan)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "execution_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Preserve the already executed signal audit")
    shutil.copy2(__file__, output / "audit_teacher_distillation_signal_executed.py")
    shutil.copy2(args.plan, output / "plan.json")
    receipt = {"status": "running", "pid": os.getpid(), "plan_sha256": digest(args.plan),
               "input_hashes": plan["input_hashes"], "actual_imports": imports,
               "source_hashes": plan["source_hashes"], "runner_sha256": digest(__file__)}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        result = execute(plan, output)
        verify_inputs(plan)
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        receipt.update(status="completed", report_sha256=digest(output / "report.json"))
        print(json.dumps(result["aggregate"]["regions"]["all"], indent=2), flush=True)
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
