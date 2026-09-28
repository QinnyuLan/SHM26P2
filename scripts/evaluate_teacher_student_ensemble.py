"""One predeclared 0.5/0.5 camera-only teacher/student probability ensemble.

This is an expensive engineering upper-bound control. It performs no weight
search, changes no parameters, and never opens real validation photographs.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from bridge_rgs.evaluate import boundary_counts, boundary_scores
from bridge_rgs.teacher import (
    IGNORE_LABEL,
    checkpoint_adapter_options,
    confusion_metrics,
    file_sha256,
    load_checkpoint_adapters,
    load_teacher,
    predict_image,
)
from bridge_rgs.train import load_scene


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/prepared/manifest.json"))
    parser.add_argument("--student", type=Path, default=Path("runs/strong_semantic_averaged/last.pt"))
    parser.add_argument("--teacher", type=Path, default=Path("runs/teacher_render_adapt_v1/best.pt"))
    parser.add_argument("--teacher-evaluation", type=Path, default=Path("runs/teacher_render_adapt_v1/adapted_strongrgb.json"))
    parser.add_argument("--student-evaluation", type=Path, default=Path("runs/strong_semantic_averaged/evaluation_native/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/teacher_student_ensemble"))
    parser.add_argument("--execution-note", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.json").exists():
        raise FileExistsError("Refusing to overwrite a completed fixed-weight ensemble")
    manifest = json.loads(args.manifest.read_text())
    views = [v for v in manifest["views"] if v["split"] == "val"]
    if len(views) != 50 or sum(bool(v.get("mask_path")) for v in views) != 41:
        raise ValueError("Expected fixed 50 camera / 41 label validation split")
    previous_teacher = json.loads(args.teacher_evaluation.read_text())
    previous_student = json.loads(args.student_evaluation.read_text())
    if previous_teacher["checkpoint_sha256"] != file_sha256(args.teacher):
        raise ValueError("Teacher evaluation checkpoint differs")
    if (not previous_teacher["flip"] or previous_teacher["context_short_side"] != 768
            or previous_teacher["tile_size"] != 768 or previous_teacher["stride"] != 512
            or "0.25" not in previous_teacher["metrics"]):
        raise ValueError("Teacher reference protocol differs")
    teacher_checkpoint = torch.load(args.teacher, map_location="cpu", weights_only=False)
    source = teacher_checkpoint["provenance"]
    protocol = source["image_source_protocol"]
    if protocol["original_manifest_sha256"] != file_sha256(args.manifest):
        raise ValueError("Original manifest differs from teacher adaptation")
    if file_sha256(protocol["val_render_receipt"]["path"]) != protocol["val_render_receipt"]["sha256"]:
        raise ValueError("Validation render receipt differs")
    renderer_checkpoint = torch.load(protocol["renderer_checkpoint"], map_location="cpu", weights_only=False)
    if file_sha256(protocol["renderer_checkpoint"]) != protocol["renderer_checkpoint_sha256"]:
        raise ValueError("Renderer checkpoint changed")
    model_dir = Path(source["model_dir"])
    if file_sha256(model_dir / "config.json") != source["model_config_sha256"] or {
        p.name: file_sha256(p) for p in sorted(model_dir.glob("*.safetensors"))
    } != source["model_weights_sha256"]:
        raise ValueError("Frozen DINOv3 backbone differs")
    student, saved_student = load_scene(args.student)
    student.eval()
    if saved_student["config"]["manifest"] != str(args.manifest):
        raise ValueError("Student training manifest differs")
    frozen = [key for key in renderer_checkpoint["model"] if key.startswith("splats.")
              and key != "splats.sem_features"] + ["background_logits"]
    for key in frozen:
        if not torch.equal(renderer_checkpoint["model"][key], student.state_dict()[key].cpu()):
            raise ValueError(f"Student and teacher RGB rendering geometry differ: {key}")
    teacher = load_teacher(model_dir, len(manifest["class_names"]),
                           teacher_checkpoint["configuration"]["channels"],
                           **checkpoint_adapter_options(teacher_checkpoint["configuration"]))
    teacher.decoder.load_state_dict(teacher_checkpoint["ema_decoder"])
    load_checkpoint_adapters(teacher, teacher_checkpoint)
    teacher.eval()
    domains = ("student", "teacher", "ensemble_0.5")
    cm = {key: np.zeros((5, 5), dtype=np.int64) for key in domains}
    boundaries = {key: np.zeros((5, 4), dtype=np.int64) for key in domains}
    for key in domains:
        (args.output / key).mkdir(exist_ok=True)
    input_rows = {row["name"]: row for row in json.loads(
        Path(protocol["val_render_receipt"]["path"]).read_text())["records"]}
    per_view = []
    torch.cuda.reset_peak_memory_stats()
    for index, view in enumerate(views):
        # The only inputs to prediction are camera metadata and the shared scene.
        K = torch.tensor(view["K"], device="cuda", dtype=torch.float32)
        pose = torch.tensor(view["w2c_original"], device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        start = time.perf_counter()
        rendered = student.render(K, pose, view["width"], view["height"], absgrad=False)
        torch.cuda.synchronize()
        render_time = time.perf_counter() - start
        image = (rendered["rgb"].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        student_probs = rendered["probabilities"].permute(2, 0, 1).cpu().numpy()
        source_image = input_rows[view["name"]]
        if file_sha256(source_image["image_path"]) != source_image["sha256"]:
            raise ValueError("Teacher reference rendered RGB changed")
        reference_rgb = cv2.cvtColor(cv2.imread(source_image["image_path"]), cv2.COLOR_BGR2RGB)
        if not np.array_equal(image, reference_rgb):
            raise ValueError(f"Shared-scene rendered pixels differ: {view['name']}")
        torch.cuda.synchronize()
        teacher_start = time.perf_counter()
        teacher_probs, _ = predict_image(teacher, image, tile_size=768, stride=512,
                                         flip=True, context_weight=0.25, context_short_side=768)
        torch.cuda.synchronize()
        teacher_time = time.perf_counter() - teacher_start
        predictions = {"student": student_probs.argmax(0), "teacher": teacher_probs.argmax(0),
                       "ensemble_0.5": (0.5 * student_probs + 0.5 * teacher_probs).argmax(0)}
        elapsed = time.perf_counter() - start
        # GT/validity is opened only after all three predictions are complete.
        valid = cv2.imread(view["valid_path"], 0) > 0 if view.get("valid_path") else np.ones(image.shape[:2], dtype=bool)
        target = cv2.imread(view["mask_path"], 0) if view.get("mask_path") else np.full(image.shape[:2], IGNORE_LABEL, dtype=np.uint8)
        keep = valid & (target != IGNORE_LABEL)
        row = {"name": view["name"], "rendered_rgb_sha256": source_image["sha256"],
               "student_full_render_seconds": render_time, "teacher_postprocess_seconds": teacher_time,
               "combined_inference_and_rgb_audit_seconds": elapsed, "metrics": {}}
        for key, prediction in predictions.items():
            encoded = target[keep].astype(np.int64) * 5 + prediction[keep]
            local = np.bincount(encoded, minlength=25).reshape(5, 5)
            cm[key] += local
            boundaries[key] += boundary_counts(prediction, target, keep)
            row["metrics"][key] = confusion_metrics(local)
            cv2.imwrite(str(args.output / key / view["name"]), prediction.astype(np.uint8))
        per_view.append(row)
        print(f"fixed ensemble {index + 1}/50 {view['name']}", flush=True)
    if cm["teacher"].tolist() != previous_teacher["metrics"]["0.25"]["confusion"]:
        raise AssertionError("Teacher confusion does not reproduce the fixed baseline")
    if cm["student"].tolist() != previous_student["confusion_matrix"]:
        raise AssertionError("Student confusion does not reproduce the fixed baseline")
    teacher_parameters = sum(p.numel() for p in teacher.parameters())
    student_parameters = sum(p.numel() for p in student.parameters())
    result = {
        "protocol": "camera-only shared geometry RGB -> fixed DINOv3+student probabilities, one weight0.5; no val photographs or weight sweep",
        "student_checkpoint": str(args.student.resolve()), "student_checkpoint_sha256": file_sha256(args.student),
        "teacher_checkpoint": str(args.teacher.resolve()), "teacher_checkpoint_sha256": file_sha256(args.teacher),
        "manifest_sha256": file_sha256(args.manifest), "image_source_protocol": protocol,
        "source_sha256": file_sha256(Path(__file__)), "execution_note": args.execution_note,
        "validation_views": 50, "semantic_validation_views": 41,
        "teacher_reference_cm_exact": True, "student_reference_cm_exact": True,
        "shared_rgb_pixels_exact": True, "fixed_teacher_weight": 0.5,
        "class_names": manifest["class_names"],
        "parameters": {"student_scene": student_parameters, "teacher_total": teacher_parameters,
                       "teacher_frozen_backbone": sum(p.numel() for p in teacher.backbone.parameters()),
                       "teacher_decoder": sum(p.numel() for p in teacher.decoder.parameters()),
                       "combined_loaded": student_parameters + teacher_parameters},
        "mean_seconds": {key: float(np.mean([row[key] for row in per_view])) for key in (
            "student_full_render_seconds", "teacher_postprocess_seconds", "combined_inference_and_rgb_audit_seconds")},
        "timing_scope": "Concurrent workload latency, not isolated FPS. Combined time includes source-RGB audit I/O, excludes GT loading/scoring and model loading.",
        "peak_torch_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "metrics": {key: {**confusion_metrics(cm[key]),
                          "miou_foreground": float(np.mean(confusion_metrics(cm[key])["per_class_iou"][1:])),
                          "boundary_f1_2px": boundary_scores(boundaries[key])} for key in domains},
        "per_view": per_view,
        "scope": "Heavy engineering ensemble control, not a new mechanism or the lightweight shared-field method; development evaluation only",
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value["miou"] for key, value in result["metrics"].items()}))


if __name__ == "__main__":
    main()
