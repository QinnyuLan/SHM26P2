"""CPU audit of completed fixed appearance arms; never starts or resumes training."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/raw_grid_appearance_v1"
PLAN_SHA = "a8f889c3c9d7678cb0a1376dfd6bb1f06423e1e470f3f3853991259832869a1d"
ALLOWED = {"splats.sh0", "splats.sh_rest", "background_logits"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def engineering_gate(specification, original_minus_native, original_minus_base):
    gate = specification["engineering_gate"]
    checks = {}
    for name, comparison in (("native", original_minus_native), ("base", original_minus_base)):
        values = comparison["metrics"]
        checks[f"psnr_delta_vs_{name}_at_least_minimum"] = values["psnr"]["difference"] >= gate["original_minus_native_and_base_psnr_db_min"]
        if gate["both_psnr_paired_interval_lower_positive"]:
            checks[f"psnr_interval_vs_{name}_lower_positive"] = values["psnr"]["paired_view_bootstrap_95_interval"][0] > 0
        if gate["ssim_point_non_decreasing"]:
            checks[f"ssim_vs_{name}_non_decreasing"] = values["ssim"]["difference"] >= 0
        if gate["lpips_point_non_increasing"]:
            checks[f"lpips_vs_{name}_non_increasing"] = values["lpips"]["difference"] <= 0
    maximum_drop = gate["all5_and_cable_max_drop_from_base_pp"] / 100
    for key in ("miou_all", "stay_cable_iou"):
        checks[f"{key}_drop_from_base_within_limit"] = original_minus_base["metrics"][key]["difference"] >= -maximum_drop
    return {"passed": all(checks.values()), "checks": checks, "locked_thresholds": gate,
            "interpretation": "SSIM/LPIPS point directions required against both native and base; semantic safeguard against base only."}


def audit():
    result_path = OUTPUT / "result_audit.json"
    if result_path.exists():
        raise FileExistsError("Refuse to overwrite result audit")
    # Gate on metadata before hashing/loading any possibly active endpoint.
    execution = read(OUTPUT / "execution_receipt.json")
    if execution["status"] != "completed":
        raise RuntimeError(f"Experiment not completed ({execution['status']}); no endpoint audit or restart")
    plan = read(OUTPUT / "plan.json")
    assert sha(OUTPUT / "plan.json") == execution["plan_sha256"] == PLAN_SHA
    assert execution["all_bound_inputs_unchanged"] is True
    snapshot = Path(plan["snapshot"])
    for relative, expected in plan["source_hashes"].items():
        assert sha(snapshot / relative) == expected
    for path, expected in plan["input_hashes"].items():
        assert sha(path) == expected
    sys.path.insert(0, str(snapshot))
    checkpoints = importlib.import_module("bridge_rgs.checkpoints")
    official = importlib.import_module("bridge_rgs.official_evaluate")
    comparison_module = importlib.import_module("compare_official_evaluations")
    assert Path(checkpoints.__file__).resolve() == snapshot / "bridge_rgs/checkpoints.py"
    assert Path(comparison_module.__file__).resolve() == snapshot / "compare_official_evaluations.py"
    torch.set_num_threads(8)
    base = checkpoints.load_checkpoint(plan["base"], map_location="cpu")
    order, generator = [], random.Random(plan["specification"]["view_seed"])
    while len(order) < 3000:
        indices = list(range(350))
        generator.shuffle(indices)
        order.extend(indices)
    assert order[:3000] == plan["training_order"]
    names = [plan["view_names"][i] for i in plan["training_order"]]
    sequence_sha = hashlib.sha256(json.dumps(names).encode()).hexdigest()
    summaries, metrics, provenance = {}, {}, {}
    metrics["base"], provenance["base"] = comparison_module.read_completed(plan["base_metrics"])
    comparison_module._validate(metrics["base"])
    frozen = {key: tensor_hash(value) for key, value in base["model"].items() if key not in ALLOWED}
    for arm in plan["arms"]:
        folder = OUTPUT / arm
        receipt = read(folder / "training_receipt.json")
        endpoint = next(item for item in execution["arms"] if item["arm"] == arm)
        assert sha(folder / "training_receipt.json") == endpoint["training_receipt_sha256"]
        assert receipt["status"] == "completed" and receipt["steps"] == 3000
        assert receipt["plan_sha256"] == PLAN_SHA and receipt["specification"] == plan["specification"]
        assert receipt["base_checkpoint_sha256"] == plan["input_hashes"][plan["base"]]
        assert receipt["manifest_sha256"] == plan["input_hashes"][plan["manifest"]]
        assert receipt["frozen_tensor_hashes"] == frozen
        assert receipt["view_sequence_sha256"] == sequence_sha
        assert all(receipt[key] is True for key in ("frozen_tensors_exact", "reconstructed_model_exact", "original_training_cameras_exact"))
        assert receipt["labels_decoded_during_training"] == receipt["val_pixels_read_during_training"] == 0
        checkpoint = folder / "appearance.pt"
        assert sha(checkpoint) == receipt["checkpoint_sha256"] == endpoint["checkpoint_sha256"]
        compact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert set(compact["model"]) == ALLOWED and compact["step"] == 3000 and "trainer_state" not in compact
        assert compact["base_checkpoint"] == {"path": plan["base"], "sha256": receipt["base_checkpoint_sha256"]}
        restored = checkpoints.load_checkpoint(checkpoint, map_location="cpu")
        assert restored["appearance_delta"]["ordinary_resume_allowed"] is False
        assert restored["appearance_delta"]["step"] == 3000
        assert not any(k in restored for k in ("optimizers", "torch_rng", "cuda_rng", "density_state"))
        assert set(restored["model"]) == set(base["model"])
        for key, value in restored["model"].items():
            assert value.dtype == base["model"][key].dtype and value.shape == base["model"][key].shape
            if key not in ALLOWED:
                assert torch.equal(value, base["model"][key])
            else:
                assert torch.equal(value, compact["model"][key]) and torch.isfinite(value).all()
                assert tensor_hash(base["model"][key]) == receipt["appearance_before"][key]
                assert tensor_hash(value) == receipt["appearance_after"][key]
                assert receipt["appearance_before"][key] != receipt["appearance_after"][key]
        assert torch.equal(restored["training_cameras"], base["training_cameras"])
        assert all(restored[k] == base[k] for k in ("scene_scale", "feature_dim", "sh_degree", "refiner_config", "manifest_sha256", "pixel_protocol"))
        log = [json.loads(line) for line in (folder / "train.jsonl").read_text().splitlines()]
        assert [v["step"] for v in log] == [1, *range(100, 3001, 100)]
        assert all(v["view"] == names[v["step"]-1] for v in log)
        path = folder / "evaluation_official/official_metrics.json"
        metrics[arm], provenance[arm] = comparison_module.read_completed(path)
        comparison_module._validate(metrics[arm])
        assert sha(path) == endpoint["metrics_sha256"]
        official.require_same_official_protocol(metrics["base"], metrics[arm])
        eval_receipt = read(path.parent / "execution_receipt.json")
        assert eval_receipt["checkpoint_sha256"] == receipt["checkpoint_sha256"]
        assert eval_receipt["inference_protocol"] == official.PLAIN_INFERENCE
        assert eval_receipt["predictions_finished_utc"] <= eval_receipt["source_scoring_started_utc"]
        fields = ("camera", "source_image_sha256", "source_annotation_sha256", "rasterized_mask_sha256")
        assert official.official_fingerprint([{k: item[k] for k in fields} for item in eval_receipt["source_records"]]) == metrics[arm]["official_evaluation_fingerprint"]
        assert len(eval_receipt["predictions"]) == 50
        for prediction in eval_receipt["predictions"]:
            for kind in ("rgb", "mask"):
                assert sha(prediction[kind]) == prediction[kind + "_sha256"]
        for item in eval_receipt["loaded_source_modules"].values():
            relative = str(Path(item["path"]).relative_to(snapshot))
            assert sha(item["path"]) == item["sha256"] == plan["source_hashes"][relative]
        summaries[arm] = {"training_receipt": binding(folder / "training_receipt.json"),
                          "checkpoint": binding(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
                          "training_seconds": receipt["training_seconds"], "peak_allocated_gpu_bytes": receipt["peak_allocated_gpu_bytes"],
                          "smoke": receipt["smoke"], "frozen_tensor_count": len(frozen),
                          "all_frozen_tensors_cameras_metadata_exact": True, "only_three_appearance_tensors_changed": True,
                          "delta_overlay_exact": True, "sequence_sha256": sequence_sha,
                          "logged_steps_correct": len(log), "full_step_order_evidence": "runner full actual_names digest; explicit logs step1/every100",
                          "rgb_pixels_per_step": sorted({v["rgb_pixels"] for v in log}),
                          "ssim7_centers_per_step": sorted({v["ssim7_centers"] for v in log})}
        del restored, compact
    pairs = {}
    for reference, candidate in (("base", "00_native"), ("base", "01_original"), ("00_native", "01_original")):
        filename = f"paired_{candidate}_minus_{reference}.json"
        existing = read(OUTPUT / filename)
        assert sha(OUTPUT / filename) == execution["comparisons"][filename]
        recomputed = comparison_module.paired_official_comparison(metrics[reference], metrics[candidate])
        assert {k: v for k, v in existing.items() if k != "inputs"} == recomputed
        assert existing["inputs"] == {"reference": provenance[reference], "candidate": provenance[candidate]}
        pairs[f"{candidate}_minus_{reference}"] = {"source": binding(OUTPUT / filename), "metrics": existing["metrics"]}
    gate = engineering_gate(plan["specification"], pairs["01_original_minus_00_native"], pairs["01_original_minus_base"])
    # Explicitly posthoc, same official fingerprint and completed receipt; no native metrics.
    legacy_path = ROOT / "runs/official_common_grid_v1/legacy_support/official_metrics.json"
    legacy, legacy_source = comparison_module.read_completed(legacy_path)
    legacy_receipt = read(Path(legacy_source["receipt"]))
    assert sha(legacy_source["checkpoint"]) == legacy_source["checkpoint_sha256"]
    for item in legacy_receipt["loaded_source_modules"].values():
        assert sha(item["path"]) == item["sha256"]
    posthoc = comparison_module.paired_official_comparison(legacy, metrics["01_original"])
    posthoc.update(analysis_status="posthoc engineering contextual comparison; not part of locked gate",
                   reference_source=legacy_source, candidate_source=provenance["01_original"])
    posthoc_path = OUTPUT / "posthoc_paired_01_original_minus_legacy_support.json"
    if posthoc_path.exists():
        raise FileExistsError("Refuse to overwrite posthoc result")
    posthoc_path.write_text(json.dumps(posthoc, indent=2, allow_nan=False) + "\n")
    for path, expected in plan["input_hashes"].items():
        assert sha(path) == expected
    assert sha(OUTPUT / "plan.json") == PLAN_SHA
    report = {"status": "passed", "cpu_only": True, "plan": binding(OUTPUT / "plan.json"),
              "execution_receipt": binding(OUTPUT / "execution_receipt.json"), "auditor": binding(__file__),
              "input_file_count_rehashed": len(plan["input_hashes"]), "source_file_count_rehashed": len(plan["source_hashes"]),
              "official_evaluation_fingerprint": metrics["base"]["official_evaluation_fingerprint"],
              "arms": summaries, "three_locked_pairs_exactly_recomputed": True,
              "locked_comparisons": pairs, "engineering_gate": gate,
              "posthoc_original_minus_legacy_support": {"source": binding(posthoc_path), "metrics": posthoc["metrics"]},
              "metrics": {key: {k: m[k] for k in ("psnr", "ssim", "lpips", "miou_all", "miou_foreground", "iou")}
                          for key, m in {"legacy_support": legacy, **metrics}.items()},
              "limitations": "Fixed development bridge/views and one seed; objective/support comparison; no native grid mixed; no GPU or training in this audit."}
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "engineering_gate": gate, "result": str(result_path)}))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    audit()
