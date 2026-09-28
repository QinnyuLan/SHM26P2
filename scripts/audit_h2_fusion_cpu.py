"""Read-only H2 target/loss audit; never loads a renderer or uses CUDA."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("02_teacher_fixed_sampling", "03_teacher_projection_sampling")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def quantiles(value):
    value = torch.as_tensor(value).detach().double().flatten()
    if not len(value):
        return None
    return dict(zip(("min", "p25", "median", "p75", "p95", "max"),
                    value.quantile(torch.tensor([0., .25, .5, .75, .95, 1.], dtype=torch.float64)).tolist()))


def balance(mass):
    mass = torch.as_tensor(mass).float()
    mean = mass.sum() / (mass > 0).sum().clamp_min(1)
    factors = (mean / mass.clamp_min(1e-8)).clamp(max=3)
    balanced = mass * factors
    denominator = balanced.sum().clamp_min(1)
    return factors, balanced, denominator


def analyze_target(checkpoint, losses):
    ids, target, before = checkpoint["density_state"]["fused_target"]
    model = checkpoint["model"]
    n = len(model["splats.means"])
    assert len(ids) == len(torch.unique(ids)) and int(ids.min()) >= 0 and int(ids.max()) < n
    after, conflicts = losses.supervised_teacher_weights(target, before, model["semantic_prior_counts"][ids])
    category = target.argmax(-1)
    mass = torch.zeros(5).scatter_add_(0, category, after)
    factors, balanced, denominator = balance(mass)
    effective = after * factors[category]
    coefficient = checkpoint["config"]["multiview_weight"]
    coefficient_per_point = coefficient * effective / denominator
    features = model["splats.sem_features"][ids].detach().clone().requires_grad_(True)
    classifier = torch.nn.Linear(features.shape[1], 5)
    classifier.load_state_dict({"weight": model["semantic_decoder.weight"], "bias": model["semantic_decoder.bias"]})
    value = coefficient * losses.balanced_local_distillation(features, classifier, target, after)
    logits = torch.nn.functional.linear(features, classifier.weight.detach(), classifier.bias.detach())
    kl = (target * (target.clamp_min(1e-7).log() - logits.log_softmax(-1))).sum(-1)
    reconstructed = (coefficient_per_point * kl).sum()
    torch.testing.assert_close(value, reconstructed, atol=1e-8, rtol=1e-5)
    grad = torch.autograd.grad(value, features)[0]
    assert classifier.weight.grad is None and classifier.bias.grad is None
    logits_gradient = coefficient_per_point[:, None] * (logits.softmax(-1).detach() * target.sum(-1, keepdim=True) - target)
    analytic = logits_gradient @ classifier.weight.detach()
    torch.testing.assert_close(grad, analytic, atol=1e-8, rtol=1e-4)
    accepted = after > 0
    classes = []
    for c in range(5):
        selected = accepted & (category == c)
        classes.append({
            "class_id": c,
            "teacher_count": int(((before > 0) & (category == c)).sum()),
            "accepted_count": int(selected.sum()),
            "rejected_gt_conflict_count": int(((before > 0) & conflicts & (category == c)).sum()),
            "accepted_weight_sum": float(mass[c]),
            "class_balance_factor": float(factors[c]),
            "normalized_balanced_weight_share": float(balanced[c] / denominator),
            "loss_coefficient_sum": float(coefficient_per_point[selected].sum()),
            "loss_coefficient_per_point": quantiles(coefficient_per_point[selected]),
            "weighted_loss_contribution_at_saved_features": float((coefficient_per_point[selected] * kl[selected]).sum().detach()),
            "feature_gradient_l2": float(grad[selected].norm()),
            "feature_gradient_l2_per_point": quantiles(grad[selected].norm(dim=-1)),
        })
    result = {
        "saved_step": checkpoint["step"],
        "target_generation_step": 3000 if checkpoint["step"] == 3000 else 1400,
        "target_used_before_save": checkpoint["step"] == 1500,
        "target_total_scheduled_training_steps": 0 if checkpoint["step"] == 3000 else 200,
        "interpretation": "Current-state CPU gradient proxy, not a recovered historical applied gradient",
        "scene_gaussians": n,
        "candidate_count": len(ids),
        "candidate_unique_count": len(torch.unique(ids)),
        "teacher_positive_count": int((before > 0).sum()),
        "accepted_unique_count": int(accepted.sum()),
        "accepted_scene_fraction": float(accepted.sum()) / n,
        "accepted_candidate_fraction": float(accepted.float().mean()),
        "accepted_weight_sum": float(after.sum()),
        "accepted_weight_quantiles": quantiles(after[accepted]),
        "balanced_denominator": float(denominator),
        "denominator_clamp_active": bool(balanced.sum() < 1),
        "loss_coefficient": coefficient,
        "effective_soft_target_mass_by_class": ((effective / denominator)[:, None] * target).sum(0).tolist(),
        "logits_gradient_l1_by_class": logits_gradient.abs().sum(0).tolist(),
        "logits_gradient_signed_sum_by_class": logits_gradient.sum(0).tolist(),
        "weighted_loss_at_saved_features": float(value.detach()),
        "feature_gradient_l2": float(grad.norm()),
        "feature_gradient_nonzero_rows": int((grad.norm(dim=-1) > 0).sum()),
        "feature_gradient_max_abs": float(grad.abs().max()),
        "classifier_gradient_detached": True,
        "sem_features_optimizer": {k: v for k, v in checkpoint["optimizers"]["sem_features"]["param_groups"][0].items() if k != "params"},
        "classes": classes,
        "candidate_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        "stored_stats_fusion": {k: v for k, v in checkpoint["stats"].items() if "fusion" in k or "fused" in k},
    }
    return result, ids, ids[accepted]


def analyze_logs(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    used = [r for r in rows if r["step"] in range(300, 3000, 200)]
    assert len(used) == 14
    summaries = []
    for r in used:
        factors, balanced, denom = balance(r["fusion_accepted_weight_by_class"])
        summaries.append({
            "log_step": r["step"], "target_generation_step": r["step"] - 100,
            "scheduled_use_steps": 200,
            "accepted_count_by_class": r["fusion_accepted_by_class"],
            "teacher_count_by_class": r["fusion_teacher_by_class"],
            "accepted_weight_by_class": r["fusion_accepted_weight_by_class"],
            "class_balance_factors": factors.tolist(),
            "normalized_balanced_weight_share": (balanced / denom).tolist(),
            "loss_coefficient_sum_by_class": (.05 * balanced / denom).tolist(),
            "weighted_local_kd_at_log": r["fusion_weighted_kd"],
            "semantic_loss_at_log": r["semantic_loss"],
        })
    share = np.array([r["normalized_balanced_weight_share"] for r in summaries])
    counts = np.array([r["accepted_count_by_class"] for r in summaries])
    all_logged_losses = [r["fusion_weighted_kd"] for r in rows if r["step"] >= 300]
    return {
        "policy": "One row per used target set, generation+100; repeated adjacent logs are not independent evidence",
        "used_batch_count": 14,
        "fusion_active_training_steps": 2800,
        "unique_history_ids_recoverable": False,
        "accepted_count_by_class_mean": counts.mean(0).tolist(),
        "class_present_batches": (counts > 0).sum(0).tolist(),
        "mean_balanced_weight_share": share.mean(0).tolist(),
        "mean_class_coefficient": (.05 * share.mean(0)).tolist(),
        "cable_share_when_present": quantiles(share[counts[:, 2] > 0, 2]),
        "weighted_local_kd_selected_14_rows": quantiles([r["weighted_local_kd_at_log"] for r in summaries]),
        "weighted_local_kd_selected_14_mean": float(np.mean([r["weighted_local_kd_at_log"] for r in summaries])),
        "weighted_local_kd_all_28_logged_rows": quantiles(all_logged_losses),
        "rows": summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/diagnostics/h2_fusion_cpu.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    report = {"kind": "read_only_cpu_h2_fusion_audit", "gpu_used": False,
              "class_names": ["background", "deck", "cable", "tower", "foundation"], "arms": {}}
    arrays = {}
    for arm in ARMS:
        run = ROOT / "runs/h2_multiscale" / arm
        receipt = json.loads((run / "experiment_receipt.json").read_text())
        assert receipt["status"] == "completed"
        for rel, expected in receipt["source_hashes"].items():
            assert sha(run / "source_snapshot" / rel) == expected, rel
        loss_path = run / "source_snapshot/bridge_rgs/losses.py"
        spec = importlib.util.spec_from_file_location("audited_h2_losses", loss_path)
        losses = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(losses)
        entry = {"receipt_sha256": sha(run / "experiment_receipt.json"),
                 "source_hashes_verified": receipt["source_hashes"],
                 "input_lineage_recorded_in_receipt": receipt["input_hashes"],
                 "config": receipt["config"], "checkpoints": {},
                 "history_logs": analyze_logs(run / "train.jsonl")}
        for name in ("last.pt", "step_001500.pt"):
            path = run / name
            actual_sha = sha(path)
            if name == "last.pt":
                assert actual_sha == receipt["checkpoint_sha256"]
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            assert all(not v.is_cuda for v in checkpoint["model"].values())
            result, ids, accepted = analyze_target(checkpoint, losses)
            result.update(path=str(path), sha256=actual_sha)
            entry["checkpoints"][name] = result
            arrays[(arm, name)] = (ids, accepted)
        report["arms"][arm] = entry
    report["paired_candidates"] = {}
    for name in ("last.pt", "step_001500.pt"):
        a, b = (arrays[(arm, name)] for arm in ARMS)
        aset, bset = set(a[1].tolist()), set(b[1].tolist())
        report["paired_candidates"][name] = {
            "candidate_ids_identical": torch.equal(a[0], b[0]),
            "accepted_intersection": len(aset & bset),
            "accepted_union": len(aset | bset),
            "accepted_jaccard": len(aset & bset) / max(len(aset | bset), 1),
        }
    report["execution_script_sha256"] = sha(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
