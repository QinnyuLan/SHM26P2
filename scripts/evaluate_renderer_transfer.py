"""Explicit old-renderer teacher transfer to one new shared scene, fixed 0.5 mix.

Costly engineering control. No model updates, probability archives, weight
search, real validation photographs, or changes to previous ensemble receipts.
"""

import argparse
import importlib
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from bridge_rgs.evaluate import boundary_counts, boundary_scores
from bridge_rgs.refinement import MultiScaleRefinementHead
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


def verify_file(path, expected):
    if file_sha256(path) != expected:
        raise ValueError(f"Input changed: {path}")


@torch.inference_mode()
def execute(plan, output, receipt):
    manifest = json.loads(Path(plan["manifest"]).read_text())
    views = [view for view in manifest["views"] if view["split"] == "val"]
    assert len(views) == 50 and sum(bool(view.get("mask_path")) for view in views) == 41
    assert len({view["name"] for view in views}) == 50
    assert plan["inference"] == {"tile_size": 768, "stride": 512, "flip": True,
                                  "context_weight": .25, "context_short_side": 768,
                                  "teacher_weight": .5}
    teacher_state = torch.load(plan["teacher_checkpoint"], map_location="cpu", weights_only=False)
    source = teacher_state["provenance"]
    training_domain = source["image_source_protocol"]
    assert training_domain["original_manifest_sha256"] == plan["manifest_sha256"]
    assert training_domain["renderer_checkpoint_sha256"] == plan["teacher_training_renderer_sha256"]
    assert training_domain["renderer_checkpoint_sha256"] != plan["inference_renderer_sha256"]
    for key in ("train_render_receipt", "val_render_receipt"):
        item = training_domain[key]
        verify_file(item["path"], item["sha256"])
        receipt["input_hashes"][f"teacher_training_{key}"] = item
    model_dir = Path(source["model_dir"])
    verify_file(model_dir/"config.json", source["model_config_sha256"])
    weights = {path.name: file_sha256(path) for path in sorted(model_dir.glob("*.safetensors"))}
    assert weights == source["model_weights_sha256"]
    receipt["teacher_modelscope_backbone"] = {"model_dir": str(model_dir),
                                              "config_sha256": source["model_config_sha256"],
                                              "weights_sha256": weights}
    scene, base = load_scene(plan["inference_renderer"])
    scene.eval().requires_grad_(False)
    assert base["config"]["manifest"] == plan["manifest"]
    frozen_keys = [key for key in base["model"] if not key.startswith("refiner.")]
    common_parameters = sum(parameter.numel() for name, parameter in scene.named_parameters()
                            if not name.startswith("refiner."))
    heads, configs, references, per_head_parameters = {}, {}, {}, {}
    for name, item in plan["students"].items():
        state = torch.load(item["checkpoint"], map_location="cpu", weights_only=False)
        assert all(torch.equal(base["model"][key], state["model"][key]) for key in frozen_keys)
        assert torch.equal(base["training_cameras"], state["training_cameras"])
        assert all(state[key] == base[key] for key in ("scene_scale", "feature_dim", "sh_degree"))
        assert state["config"]["manifest"] == plan["manifest"]
        config = state["refiner_config"]
        head = MultiScaleRefinementHead(state["feature_dim"],
                                       **{key: value for key, value in config.items() if key != "type"}).cuda()
        head.load_state_dict({key.removeprefix("refiner."): value for key, value in state["model"].items()
                              if key.startswith("refiner.")})
        heads[name], configs[name] = head.eval().requires_grad_(False), config
        per_head_parameters[name] = sum(parameter.numel() for parameter in head.parameters())
        references[name] = json.loads(Path(item["evaluation"]).read_text())
        assert references[name]["validation_views"] == 50 and references[name]["semantic_validation_views"] == 41
        del state
    del base
    teacher = load_teacher(model_dir, len(manifest["class_names"]), teacher_state["configuration"]["channels"],
                           **checkpoint_adapter_options(teacher_state["configuration"]))
    teacher.decoder.load_state_dict(teacher_state["ema_decoder"])
    load_checkpoint_adapters(teacher, teacher_state)
    teacher.eval().requires_grad_(False)
    domains = ["teacher", *heads, *(f"ensemble_{name}_0.5" for name in heads)]
    confusion = {key: np.zeros((5, 5), np.int64) for key in domains}
    boundaries = {key: np.zeros((5, 4), np.int64) for key in domains}
    for key in domains:
        (output/key).mkdir()
    reference_views = {name: {row["name"]: row for row in value["views"]} for name, value in references.items()}
    per_view = []
    torch.cuda.reset_peak_memory_stats()
    for index, view in enumerate(views):
        K = torch.tensor(view["K"], dtype=torch.float32, device="cuda")
        pose = torch.tensor(view["w2c_original"], dtype=torch.float32, device="cuda")
        probabilities, times, rgb_hashes = {}, {}, {}
        image = None
        torch.cuda.synchronize()
        start = time.perf_counter()
        for name, head in heads.items():
            scene.refiner, scene.refiner_config = head, configs[name]
            torch.cuda.synchronize()
            then = time.perf_counter()
            rendered = scene.render(K, pose, view["width"], view["height"], absgrad=False,
                                    refinement_grad_to_field=False)
            torch.cuda.synchronize()
            times[name] = time.perf_counter()-then
            current = (rendered["rgb"].clamp(0, 1).cpu().numpy()*255).round().astype(np.uint8)
            if image is None:
                image = current
            else:
                assert np.array_equal(current, image)
            path = Path(plan["students"][name]["evaluation"]).parent/(Path(view["name"]).stem+"_rgb.png")
            reference_rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
            assert reference_rgb is not None and np.array_equal(current, reference_rgb[..., ::-1])
            rgb_hashes[name] = file_sha256(path)
            probabilities[name] = rendered["probabilities"].permute(2, 0, 1).cpu().numpy()
            del rendered
        assert len(set(rgb_hashes.values())) == 1
        torch.cuda.synchronize()
        then = time.perf_counter()
        teacher_probabilities, _ = predict_image(teacher, image, tile_size=768, stride=512,
                                                 flip=True, context_weight=.25, context_short_side=768)
        torch.cuda.synchronize()
        teacher_time = time.perf_counter()-then
        predictions = {name: value.argmax(0) for name, value in probabilities.items()}
        predictions["teacher"] = teacher_probabilities.argmax(0)
        for name, value in probabilities.items():
            predictions[f"ensemble_{name}_0.5"] = (.5*value+.5*teacher_probabilities).argmax(0)
        elapsed = time.perf_counter()-start
        # All nine predictions are complete before either validity or GT is opened.
        valid = cv2.imread(view["valid_path"], 0) > 0 if view.get("valid_path") else np.ones(image.shape[:2], bool)
        target = cv2.imread(view["mask_path"], 0) if view.get("mask_path") else np.full(image.shape[:2], IGNORE_LABEL, np.uint8)
        assert target is not None and target.shape == valid.shape == image.shape[:2]
        keep = valid & (target != IGNORE_LABEL)
        row = {"name": view["name"], "rendered_rgb_sha256": rgb_hashes,
               "student_render_seconds": times, "one_teacher_protocol_seconds": teacher_time,
               "four_students_one_teacher_inference_and_rgb_audit_seconds": elapsed,
               "predictions_finished_before_gt_read": True, "metrics": {}}
        for name, prediction in predictions.items():
            local = np.bincount(target[keep].astype(np.int64)*5+prediction[keep], minlength=25).reshape(5, 5)
            confusion[name] += local
            boundaries[name] += boundary_counts(prediction, target, keep)
            row["metrics"][name] = confusion_metrics(local)
            if name in heads and view.get("mask_path"):
                assert local.tolist() == reference_views[name][view["name"]]["confusion_matrix"]
            assert cv2.imwrite(str(output/name/view["name"]), prediction.astype(np.uint8))
        per_view.append(row)
        print(f"renderer transfer {index+1}/50 {view['name']}", flush=True)
        del probabilities, teacher_probabilities, predictions
    assert all(confusion[name].tolist() == references[name]["confusion_matrix"] for name in heads)
    teacher_parameters = sum(parameter.numel() for parameter in teacher.parameters())
    results = {name: {**confusion_metrics(value),
                       "miou_foreground": float(np.mean(confusion_metrics(value)["per_class_iou"][1:])),
                       "boundary_f1_2px": boundary_scores(boundaries[name])} for name, value in confusion.items()}
    return {"protocol": "renderer_transfer_v1", "plan": plan,
            "teacher_training_image_source_protocol": training_domain,
            "renderer_transfer_explicit": True, "teacher_updated": False,
            "all_student_nonrefiner_tensors_and_cameras_exact": True,
            "all_student_scene_scale_feature_dim_sh_degree_exact": True,
            "all_50_student_rgb_pngs_and_fresh_renders_exact": True,
            "all_student_global_and_per_view_confusions_reproduced": True,
            "teacher_protocol_invocations": 50, "validation_views": 50, "semantic_validation_views": 41,
            "common_renderer_rgb_metrics": {key: references["base"][key] for key in ("psnr", "ssim", "lpips")},
            "teacher_training_old_renderer_metrics_not_reused": True,
            "metrics": results, "per_view": per_view,
            "parameters": {"shared_field": common_parameters, "student_heads": per_head_parameters,
                           "teacher_total": teacher_parameters,
                           "teacher_backbone_including_any_loaded_adapters": sum(p.numel() for p in teacher.backbone.parameters()),
                           "teacher_decoder": sum(p.numel() for p in teacher.decoder.parameters()),
                           "loaded_bundle": common_parameters+sum(per_head_parameters.values())+teacher_parameters,
                           "each_student_and_teacher": {name: common_parameters+count+teacher_parameters for name, count in per_head_parameters.items()}},
            "mean_seconds": {"students": {name: float(np.mean([row["student_render_seconds"][name] for row in per_view])) for name in heads},
                             "teacher_protocol": float(np.mean([row["one_teacher_protocol_seconds"] for row in per_view])),
                             "whole_four_student_bundle_including_rgb_audit": float(np.mean([row["four_students_one_teacher_inference_and_rgb_audit_seconds"] for row in per_view]))},
            "peak_torch_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
            "timing_scope": "Execution under the receipt workload; bundle includes RGB audit I/O, excludes labels/scoring/loading. Not an isolated single-model FPS benchmark.",
            "scope": "Fixed costly engineering transfer/ensemble control, not a new mechanism or an upper bound; fixed development split."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/generated_renderer_transfer/support_plan.json"))
    parser.add_argument("--execution-note", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    plan = json.loads(args.plan.read_text())
    assert json.loads(Path(plan["h3_report"]).read_text())["status"] == "completed"
    verify_file(plan["h3_report"], plan["h3_report_sha256"])
    snapshot, output = Path(plan["source_snapshot"]).resolve(), Path(plan["output"])
    for key, value in plan["source_hashes"].items():
        verify_file(snapshot/key, value)
    imports = {}
    for name in ("bridge_rgs.model", "bridge_rgs.train", "bridge_rgs.refinement", "bridge_rgs.depth_moments", "bridge_rgs.teacher"):
        path = Path(importlib.import_module(name).__file__).resolve()
        assert path.is_relative_to(snapshot)
        imports[name] = {"path": str(path), "sha256": file_sha256(path)}
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output/"execution_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Refusing to overwrite renderer-transfer execution")
    shutil.copytree(snapshot, output/"source_snapshot", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, output/"evaluate_renderer_transfer_executed.py")
    shutil.copy2(args.plan, output/"plan.json")
    inputs = {key: {"path": str(Path(plan[key]).resolve()), "sha256": plan[f"{key}_sha256"]}
              for key in ("manifest", "teacher_checkpoint", "teacher_training_renderer", "inference_renderer", "h3_report")}
    for name, item in plan["students"].items():
        for key in ("checkpoint", "evaluation"):
            inputs[f"student_{name}_{key}"] = {"path": str(Path(item[key]).resolve()), "sha256": item[f"{key}_sha256"]}
    receipt = {"status": "running", "protocol": "renderer_transfer_v1", "execution_note": args.execution_note,
               "plan_sha256": file_sha256(args.plan), "script_sha256": file_sha256(__file__),
               "input_hashes": inputs, "source_hashes": plan["source_hashes"], "actual_imports": imports}
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    try:
        for item in inputs.values():
            verify_file(item["path"], item["sha256"])
        result = execute(plan, output, receipt)
        for item in receipt["input_hashes"].values():
            verify_file(item["path"], item["sha256"])
        (output/"metrics.json").write_text(json.dumps(result, indent=2)+'\n')
        receipt.update(status="completed", metrics_sha256=file_sha256(output/"metrics.json"))
        print(json.dumps({key: value["miou"] for key, value in result["metrics"].items()}))
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == "__main__":
    main()
