"""Fixed TRAIN-only UV-jitter stress check; no fitting, VAL scoring or threshold search."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import bridge_rgs.projection_stress as scoring
from bridge_rgs.losses import supervised_teacher_weights
from bridge_rgs.reliability import (
    _sample_map,
    ellipse_sample_probabilities,
    fuse_multiview_evidence,
    projection_jacobians,
)
from bridge_rgs.teacher import verify_pseudo_provenance
from bridge_rgs.train import load_scene


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return np.quantile(values, [0, .1, .5, .9, 1]).tolist() if len(values) else None


def center_visible(uv, z, depth, alpha):
    height, width = depth.shape
    inside = (torch.isfinite(uv).all(-1) & torch.isfinite(z) & (z > 0)
              & (uv[:, 0] >= 0) & (uv[:, 0] <= width-1)
              & (uv[:, 1] >= 0) & (uv[:, 1] <= height-1))
    sampled_z = _sample_map(depth, uv[:, None])[..., 0, 0]
    sampled_alpha = _sample_map(alpha, uv[:, None])[..., 0, 0]
    return (inside & torch.isfinite(sampled_z) & (sampled_z > 0)
            & ((sampled_z-z).abs() <= .01+.05*z.abs()) & (sampled_alpha >= .3))


def read_scoring_gt(view, uv_native, original_visible):
    """Called after all predictions for this view; return no values to evidence code."""
    count = len(uv_native)
    labels, near_boundary = np.full(count, -1, np.int16), np.zeros(count, bool)
    hashes = {}
    if not view.get("mask_path"):
        return labels, near_boundary, hashes
    if view["split"] != "train":
        raise ValueError("GT scoring may only open TRAIN labels")
    mask = cv2.imread(view["mask_path"], cv2.IMREAD_GRAYSCALE)
    valid = cv2.imread(view["valid_path"], cv2.IMREAD_GRAYSCALE) > 0
    if mask.shape != (view["height"], view["width"]):
        raise ValueError("GT scoring grid mismatch")
    hashes = {key: {"path": view[key], "sha256": digest(view[key])}
              for key in ("mask_path", "valid_path")}
    uv = np.rint(np.nan_to_num(uv_native, nan=-1e8, posinf=-1e8, neginf=-1e8)).astype(np.int64)
    height, width = mask.shape
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < width)
              & (uv[:, 1] >= 0) & (uv[:, 1] < height) & original_visible)
    ids = np.flatnonzero(inside)
    ids = ids[valid[uv[ids, 1], uv[ids, 0]] & (mask[uv[ids, 1], uv[ids, 0]] < 5)]
    labels[ids] = mask[uv[ids, 1], uv[ids, 0]]
    boundary = np.zeros_like(valid)
    different = (mask[1:] != mask[:-1]) & valid[1:] & valid[:-1]
    boundary[1:] |= different
    boundary[:-1] |= different
    different = (mask[:, 1:] != mask[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    boundary[:, 1:] |= different
    boundary[:, :-1] |= different
    distance = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    near_boundary[ids] = distance[uv[ids, 1], uv[ids, 0]] <= 3.
    return labels, near_boundary, hashes


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/refiner_multiscale_v1/last.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/prepared/manifest.json"))
    parser.add_argument("--pseudo", type=Path, default=Path("artifacts/pseudo_strong_v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/projection_jitter_stress"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a preregistered diagnostic")
    args.output.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text())
    provenance = verify_pseudo_provenance(args.manifest, args.pseudo, verify_files=False)
    metadata = {v["name"]: v for v in provenance["views"]}
    views = [v for v in manifest["views"] if v["split"] == "train"]
    by_name = {v["name"]: v for v in views}
    groups = scoring.select_groups(views, anchors=16, group_size=4)
    selected_names = sorted({name for group in groups for name in group["views"]})
    inputs = {"checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": digest(args.checkpoint)},
              "manifest": {"path": str(args.manifest.resolve()), "sha256": digest(args.manifest)},
              "pseudo_provenance": {"path": str((args.pseudo / "provenance.json").resolve()),
                                    "sha256": digest(args.pseudo / "provenance.json")}}
    scene, state = load_scene(args.checkpoint)
    scene.eval()
    cameras = state["training_cameras"].cuda()
    original_cameras = torch.tensor([v.get("w2c_original", v["w2c"]) for v in views],
                                   device="cuda", dtype=torch.float32)
    if not torch.equal(cameras, original_cameras):
        raise ValueError("This fixed control requires unchanged TRAIN cameras")
    if Path(state["config"]["manifest"]).resolve() != args.manifest.resolve():
        raise ValueError("Checkpoint manifest path mismatch")
    receipt = json.loads((args.checkpoint.parent / "experiment_receipt.json").read_text())
    if receipt["input_hashes"]["manifest"]["sha256"] != inputs["manifest"]["sha256"]:
        raise ValueError("Checkpoint manifest SHA mismatch")
    count = 4096
    ids = np.sort(np.random.default_rng(42).choice(len(scene.splats["means"]), count, replace=False))
    xyz = scene.splats["means"][torch.tensor(ids, device="cuda")].detach()
    priors = scene.semantic_prior_counts[torch.tensor(ids, device="cuda")].detach().cpu()
    noise = {name: scoring.zero_mean_jitter(count, 43+i) for i, name in enumerate(selected_names)}
    plan = {"kind": "train_only_projection_jitter_synthetic_oracle_check", "inputs": inputs,
            "groups": groups, "selected_names": selected_names, "gaussian_ids": ids.tolist(),
            "gaussian_count": len(scene.splats["means"]), "seed_ids": 42,
            "seed_jitter": "43 + index in sorted selected_names; empirically centered Gaussian",
            "jitter_empirical_covariances": {n: np.cov(noise[n].T).tolist() for n in selected_names},
            "jitter_units": "480-wide fusion grid pixels", "sigma_levels": [0, 1, 2],
            "covariance_methods": {"fixed": ".25 I", "oracle": "(.25 + sigma^2) I"},
            "grid": "width=min(480,native width), height=round(native height*width/native width)",
            "interpolation": "fusion.py exact: bilinear align_corners=False probs/confidence; nearest valid",
            "evidence_thresholds": {"confidence": .65, "max_projection_std": 6,
                                    "alpha": .3, "depth_relative": .05, "depth_absolute": .01,
                                    "min_valid_fraction": .5, "min_views": 2},
            "proxy": {"min_distinct_original_visible_gt_views": 2, "purity": .8,
                      "labels": "nearest native original projection; visibility on original 480 grid",
                      "boundary_stratum": "within 3 native pixels of any visible label boundary"},
            "coverage_curve": [.1, .25, .5, .75, 1.],
            "gt_gate": False, "prior_conflict_rule": "secondary only; min mass 2, purity .8",
            "limitations": ["Not 3D GT", "Known UV noise covariance is not estimated SE3 calibration",
                            "Teacher and priors trained on overlapping TRAIN labels",
                            "Repeated points and overlapping view groups are correlated"]}
    package = Path(scoring.__file__).resolve().parent
    shutil.copytree(package, args.output / "source_snapshot" / "bridge_rgs",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, args.output / "source_snapshot" / Path(__file__).name)
    plan["source_hashes"] = {str(p.relative_to(args.output / "source_snapshot")): digest(p)
                             for p in sorted((args.output / "source_snapshot").rglob("*.py"))}
    # This is committed to disk before opening any GT mask or computing predictions.
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    np.savez_compressed(args.output / "fixed_design.npz", ids=ids,
                        names=np.array(selected_names), jitter=np.stack([noise[n] for n in selected_names]))
    records, gt_labels, gt_boundaries, gt_hashes, pseudo_hashes = {}, {}, {}, {}, {}
    original_visibility, noisy_visibility = {}, {}
    tic = time.perf_counter()
    for index, name in enumerate(selected_names):
        view = by_name[name]
        if view["split"] != "train":
            raise ValueError("Unexpected non-TRAIN view")
        path = args.pseudo / (Path(name).stem + ".npz")
        checksum = digest(path)
        if checksum != metadata[name]["sha256"]:
            raise ValueError(f"Pseudo SHA changed: {name}")
        pseudo_hashes[name] = checksum
        with np.load(path, allow_pickle=False) as stored:
            if (str(stored["view_name"].item()) != name or str(stored["split"].item()) != "train"
                    or str(stored["manifest_sha256"].item()) != inputs["manifest"]["sha256"]
                    or str(stored["checkpoint_sha256"].item()) != provenance["checkpoint_sha256"]):
                raise ValueError("Embedded pseudo provenance mismatch")
            p = torch.tensor(stored["probs"].astype(np.float32), device="cuda")
            valid = torch.tensor(stored["valid"].astype(np.float32), device="cuda")
            teacher_confidence = torch.tensor(stored["confidence"].astype(np.float32), device="cuda")
        full_h, full_w = p.shape[-2:]
        width, height = min(480, full_w), round(full_h * min(480, full_w) / full_w)
        p = F.interpolate(p[None], (height, width), mode="bilinear", align_corners=False)[0]
        valid = F.interpolate(valid[None, None], (height, width), mode="nearest")[0, 0]
        teacher_confidence = F.interpolate(teacher_confidence[None, None], (height, width),
                                          mode="bilinear", align_corners=False)[0, 0]
        p = p / p.sum(0, keepdim=True).clamp_min(1e-7)
        K = torch.tensor(view["K"], device="cuda").float()
        K[0] *= width / view["width"]
        K[1] *= height / view["height"]
        pose = cameras[views.index(view)]
        rendered = scene.render(K, pose, width, height, semantics=False, absgrad=False)
        projection = projection_jacobians(xyz, pose, K)
        depth, alpha = rendered["depth"][..., 0], rendered["alpha"][..., 0] * valid
        original_visibility[name] = center_visible(projection.uv, projection.depth, depth, alpha).cpu().numpy()
        per_view = {}
        for sigma in (0, 1, 2):
            uv = projection.uv + sigma * torch.tensor(noise[name], device="cuda")
            noisy_visibility[name, sigma] = center_visible(uv, projection.depth, depth, alpha).cpu().numpy()
            for method in ("fixed", "oracle"):
                covariance = torch.eye(2, device="cuda")[None].repeat(count, 1, 1)
                covariance *= .25 + (sigma**2 if method == "oracle" else 0)
                evidence = ellipse_sample_probabilities(
                    p, uv, covariance, depths=projection.depth, depth_map=depth, alpha_map=alpha,
                    teacher_confidence_map=teacher_confidence, confidence_threshold=.65,
                    max_projection_std=6)
                per_view[sigma, method] = {key: getattr(evidence, key).cpu()
                                          for key in ("probabilities", "weights", "stability", "valid_fraction")}
        for key in per_view[0, "fixed"]:
            if not torch.equal(per_view[0, "fixed"][key], per_view[0, "oracle"][key]):
                raise AssertionError("Sigma-zero methods must match bitwise")
        records[name] = per_view
        uv_native = projection.uv.cpu().numpy() / np.array([width/view["width"], height/view["height"]])
        labels, boundary, hashes = read_scoring_gt(view, uv_native, original_visibility[name])
        gt_labels[name], gt_boundaries[name], gt_hashes[name] = labels, boundary, hashes
        print(json.dumps({"view": name, "completed": index+1, "total": len(selected_names),
                          "original_visible": int(original_visibility[name].sum()),
                          "gt_scoring_labels": int((labels >= 0).sum())}), flush=True)
    all_proxy, all_boundary, group_proxy_stats = [], [], []
    result_arrays = {}
    for group in groups:
        names = group["views"]
        proxy, votes, purity = scoring.consensus_proxy(np.stack([gt_labels[n] for n in names]), names)
        boundary = np.stack([gt_boundaries[n] for n in names]).any(0)
        all_proxy.append(proxy)
        all_boundary.append(boundary)
        group_proxy_stats.append({"anchor": group["anchor"], "proxy_count": int((proxy >= 0).sum()),
                                  "classes": np.bincount(proxy[proxy >= 0], minlength=5).tolist(),
                                  "votes_quantiles": quantiles(votes[proxy >= 0]),
                                  "purity_quantiles": quantiles(purity[proxy >= 0])})
    proxy, boundary = np.concatenate(all_proxy), np.concatenate(all_boundary)
    result_arrays.update(proxy=proxy, boundary=boundary, ids=np.tile(ids, len(groups)),
                         group_index=np.repeat(np.arange(len(groups)), count))
    scores = {}
    for sigma in (0, 1, 2):
        for method in ("fixed", "oracle"):
            prediction, weights_before, weights_after, conflicts = [], [], [], []
            for group in groups:
                evidence = [records[n][sigma, method] for n in group["views"]]
                fused = fuse_multiview_evidence(torch.stack([r["probabilities"] for r in evidence]),
                                               torch.stack([r["weights"] for r in evidence]), min_views=2)
                after, conflict = supervised_teacher_weights(fused.probabilities, fused.weights, priors)
                prediction.append(fused.probabilities.argmax(-1).numpy())
                weights_before.append(fused.weights.numpy())
                weights_after.append(after.numpy())
                conflicts.append(conflict.numpy())
            pred, before, after, conflict = map(np.concatenate,
                                               (prediction, weights_before, weights_after, conflicts))
            key = f"sigma{sigma}_{method}"
            result_arrays.update({key+"_prediction": pred, key+"_weight_before": before,
                                  key+"_weight_after": after, key+"_prior_conflict": conflict})
            scores[key] = {"before_prior": scoring.score_predictions(proxy, pred, before),
                           "after_prior": scoring.score_predictions(proxy, pred, after),
                           "boundary_before_prior": scoring.score_predictions(np.where(boundary, proxy, -1), pred, before),
                           "interior_before_prior": scoring.score_predictions(np.where(~boundary, proxy, -1), pred, before),
                           "reliability_quantiles_all_positive": quantiles(before[before > 0]),
                           "reliability_quantiles_proxy_correct": quantiles(before[(proxy >= 0) & (proxy == pred) & (before > 0)]),
                           "reliability_quantiles_proxy_wrong": quantiles(before[(proxy >= 0) & (proxy != pred) & (before > 0)]),
                           "prior_conflicts_with_accepted_proxy": int(((proxy >= 0) & (before > 0) & conflict).sum()),
                           "per_group": [scoring.score_predictions(p, y, w) for p, y, w
                                         in zip(all_proxy, prediction, weights_before, strict=True)]}
    visibility = {}
    for sigma in (0, 1, 2):
        original = np.stack([original_visibility[n] for n in selected_names])
        noisy = np.stack([noisy_visibility[n, sigma] for n in selected_names])
        visibility[str(sigma)] = {"unique_view_point_pairs": int(original.size),
                                  "original_visible": int(original.sum()), "noisy_visible": int(noisy.sum()),
                                  "lost": int((original & ~noisy).sum()), "gained": int((~original & noisy).sum())}
        result_arrays[f"visibility_original_sigma{sigma}"] = original
        result_arrays[f"visibility_noisy_sigma{sigma}"] = noisy
        for method in ("fixed", "oracle"):
            visibility[str(sigma)][method] = {
                "nine_point_valid_fraction_quantiles": quantiles(np.concatenate([records[n][sigma, method]["valid_fraction"].numpy() for n in selected_names])),
                "positive_view_weights": sum(int((records[n][sigma, method]["weights"] > 0).sum()) for n in selected_names)}
    for name in selected_names:
        for sigma in (0, 1, 2):
            for method in ("fixed", "oracle"):
                for key, value in records[name][sigma, method].items():
                    result_arrays[f"view_{Path(name).stem}_sigma{sigma}_{method}_{key}"] = value.numpy()
    np.savez_compressed(args.output / "results.npz", **result_arrays)
    for item in inputs.values():
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Source input changed during diagnostic")
    result = {"complete": True, "plan_sha256": digest(args.output / "plan.json"),
              "results_sha256": digest(args.output / "results.npz"), "seconds": time.perf_counter()-tic,
              "unique_train_views": len(selected_names), "unique_scored_gaussian_ids": len(set(ids[np.any(np.stack(all_proxy) >= 0, axis=0)])),
              "proxy_group_point_pairs": int((proxy >= 0).sum()), "groups": group_proxy_stats,
              "visibility": visibility, "scores": scores, "pseudo_hashes": pseudo_hashes,
              "scoring_gt_hashes": gt_hashes, "sigma_zero_bitwise_equal": True,
              "validation_pixels_opened": False, "image_rgb_pixels_opened": False}
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": True, "output": str(args.output), "proxy_pairs": result["proxy_group_point_pairs"]}), flush=True)


if __name__ == "__main__":
    main()
