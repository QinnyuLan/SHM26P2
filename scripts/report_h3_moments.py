"""Fixed-endpoint H3 audit: frozen field, matched RNG and three information pairs."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from compare_evaluations import paired_comparison

from bridge_rgs.refinement import MultiScaleRefinementHead
from bridge_rgs.teacher import file_sha256


def exact_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact_state(a[key], b[key]) for key in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact_state(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def rng_digest(state):
    h = hashlib.sha256()
    for key in ("torch_rng", "cuda_rng"):
        h.update(state[key].cpu().numpy().tobytes())
    h.update(state["numpy_rng"][0].encode())
    h.update(state["numpy_rng"][1].tobytes())
    h.update(repr(state["numpy_rng"][2:]).encode())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/h3_moments"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    arms = ["00_zero", "01_variance", "02_cross"]
    receipts = [json.loads((args.root/arm/"experiment_receipt.json").read_text()) for arm in arms]
    assert all(r["status"] == "completed" for r in receipts)
    assert all(r["source_hashes"] == receipts[0]["source_hashes"] and r["input_hashes"] == receipts[0]["input_hashes"] for r in receipts)
    configs = []
    for arm, receipt in zip(arms, receipts, strict=True):
        config = copy.deepcopy(receipt["config"])
        assert Path(config.pop("output")).name == arm
        assert config["refiner"].pop("depth_moments") == arm.split("_", 1)[1]
        assert config["steps"] == 3000 and config["save_every"] == config["eval_every"] == 0
        assert "semantic_lr_schedule" not in config and config["parameter_scope"] == "refiner_only"
        configs.append(config)
    assert configs[0] == configs[1] == configs[2]
    for value in receipts[0]["input_hashes"].values():
        assert file_sha256(value["path"]) == value["sha256"]
    initial = torch.load(receipts[0]["input_hashes"]["initial_checkpoint"]["path"], map_location="cpu", weights_only=False)
    baseline_folder = Path(receipts[0]["input_hashes"]["initial_checkpoint"]["path"]).parent/"evaluation_native"
    baseline = json.loads((baseline_folder/"metrics.json").read_text())
    manifest = json.loads(Path(receipts[0]["input_hashes"]["manifest"]["path"]).read_text())
    views = [view for view in manifest["views"] if view["split"] == "train"]
    population = [index for index, view in enumerate(views) if view.get("mask_path")]
    assert len(population) == 259
    generator, sequence = np.random.default_rng(42), []
    while len(sequence) < 3000:
        order = generator.permutation(population).tolist()
        cursor = min(len(order), 3000-len(sequence))
        sequence.extend(order[:cursor])
    rows, metrics, rng_states = [], [], []
    for arm, receipt in zip(arms, receipts, strict=True):
        folder = args.root/arm
        assert all(file_sha256(folder/"source_snapshot"/key) == value for key, value in receipt["source_hashes"].items())
        checkpoint = torch.load(folder/"last.pt", map_location="cpu", weights_only=False)
        assert checkpoint["step"] == 3000 and file_sha256(folder/"last.pt") == receipt["checkpoint_sha256"]
        assert not list(folder.glob("step_*.pt")) and not list(folder.glob("metrics_*.json"))
        assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())
        fixed = [key for key in initial["model"] if not key.startswith("refiner.")]
        assert all(torch.equal(initial["model"][key], checkpoint["model"][key]) for key in fixed)
        assert torch.equal(initial["training_cameras"], checkpoint["training_cameras"])
        changes = [key for key in initial["model"] if not torch.equal(initial["model"][key], checkpoint["model"][key])]
        assert changes and all(key.startswith("refiner.") for key in changes)
        for name, optimizer in checkpoint["optimizers"].items():
            if name != "heads":
                assert not optimizer["state"]
            else:
                assert set(optimizer["state"]).issubset(set(optimizer["param_groups"][1]["params"]))
                assert optimizer["param_groups"][1]["lr"] == .0003
        extra = checkpoint["density_state"]["extras"]
        assert extra["sampler_order"] == order and extra["sampler_cursor"] == cursor
        assert extra["sampler_rng_state"] == generator.bit_generator.state
        logs = [json.loads(line) for line in (folder/"train.jsonl").read_text().splitlines()]
        for line in logs:
            assert line["refiner_crop"] is None and "semantic_lr_multiplier" not in line
            assert line["refiner_crop_view"] == views[sequence[line["step"]-1]]["name"]
        rng_states.append({key: checkpoint[key] for key in ("torch_rng", "cuda_rng", "numpy_rng")})
        head = MultiScaleRefinementHead(checkpoint["feature_dim"], **{key: value for key, value in checkpoint["refiner_config"].items() if key != "type"})
        head.load_state_dict({key.removeprefix("refiner."): value for key, value in checkpoint["model"].items() if key.startswith("refiner.")})
        projection_columns = {name: value.detach().abs().sum(dim=(0, 2, 3)).tolist()
                              for name, value in head.named_parameters() if name.startswith("moment_")}
        if arm == "00_zero":
            assert all(not any(columns) for columns in projection_columns.values())
        elif arm == "01_variance":
            assert all(columns[0] > 0 and not any(columns[1:]) for columns in projection_columns.values())
        else:
            assert all(columns[0] > 0 and any(columns[1:]) for columns in projection_columns.values())
        path = folder/"evaluation_native/metrics.json"
        assert file_sha256(path) == receipt["evaluation_sha256"]
        result = json.loads(path.read_text())
        assert result["confusion_matrix_3d"] == baseline["confusion_matrix_3d"]
        old_views = {view["name"]: view for view in baseline["views"]}
        assert len(result["views"]) == len(old_views) == 50
        for view in result["views"]:
            old = old_views[view["name"]]
            assert view.get("confusion_matrix_3d") == old.get("confusion_matrix_3d")
            name = Path(view["name"]).stem+"_rgb.png"
            assert file_sha256(folder/"evaluation_native"/name) == file_sha256(baseline_folder/name)
        metrics.append(result)
        rows.append({"arm": arm, "miou_all": result["miou_all"], "miou_foreground": result["miou_foreground"],
                     "iou": result["iou"], "raw_3d": result["semantic_3d"],
                     "cable_boundary_f1": result["boundary_f1_2px"][2],
                     "psnr": result["psnr"], "ssim": result["ssim"], "lpips": result["lpips"],
                     "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
                     "new_projection_parameters": sum(parameter.numel() for name, parameter in head.named_parameters() if name.startswith("moment_")),
                     "projection_column_l1_diagnostic_not_feature_importance": projection_columns,
                     "old_refiner_tensor_changes": len(changes), "rng_sha256": rng_digest(checkpoint),
                     "checkpoint_sha256": receipt["checkpoint_sha256"],
                     "elapsed_through_last_training_step": logs[-1]["elapsed_seconds"],
                     "peak_training_allocated_gpu_gb": logs[-1]["peak_gpu_gb"]})
        del checkpoint, head
    assert exact_state(rng_states[0], rng_states[1]) and exact_state(rng_states[0], rng_states[2])
    assert len({row["head_parameters"] for row in rows}) == 1
    pairs = {}
    for a, b, key in [(0, 1, "variance_minus_zero"), (0, 2, "cross_minus_zero"), (1, 2, "cross_minus_variance")]:
        comparison = paired_comparison(metrics[a], metrics[b])
        comparison.update(reference_arm=arms[a], candidate_arm=arms[b])
        (args.root/f"paired_{key}.json").write_text(json.dumps(comparison, indent=2)+'\n')
        pairs[key] = comparison["metrics"]
    report = {"status": "completed", "endpoint": 3000, "primary_pair": "cross_minus_variance",
              "same_source_and_inputs": True, "only_config_differences": ["output", "refiner.depth_moments"],
              "all_nonrefiner_tensors_and_cameras_exact": True, "frozen_optimizer_states_empty": True,
              "all_50_rgb_pngs_exact_to_warmstart": True, "all_raw_confusion_matrices_exact_to_warmstart": True,
              "torch_cuda_numpy_rng_exact_between_arms": True, "view_sampler_order_cursor_rng_exact": True,
              "same_head_parameter_count": True, "no_intermediate_validation_or_checkpoint_selection": True,
              "labeled_train_view_count": 259, "view_visits_min_max": [11, 12],
              "warmstart_miou_all": baseline["miou_all"], "runs": rows, "paired": pairs,
              "scope": "Fixed one-scene development split, one seed, existing moment-rendering framework; information utility hypothesis only. Matched stored parameter counts do not imply equal active input dimensions."}
    (args.root/"h3_report.json").write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
