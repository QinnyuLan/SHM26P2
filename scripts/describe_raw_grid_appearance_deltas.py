"""Posthoc CPU descriptions of completed appearance changes; no causal claim."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/raw_grid_appearance_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(array.size), "mean": float(array.mean()), "rms": float(np.sqrt(np.square(array).mean())),
            "mean_abs": float(np.abs(array).mean()), "max_abs": float(np.abs(array).max()),
            "quantiles_0_25_50_75_90_99_100": np.quantile(array, [0, .25, .5, .75, .9, .99, 1]).tolist()}


def dc_proxy(sh0):
    colors = sh0[:, 0].double().numpy() * .28209479177387814 + .5
    return {"definition": "SH0-only color proxy; not view-dependent rendered RGB or actual clamp saturation",
            "min": float(colors.min()), "max": float(colors.max()),
            "fraction_channels_below0": float((colors < 0).mean()),
            "fraction_channels_above1": float((colors > 1).mean()),
            "fraction_gaussians_any_channel_outside01": float(((colors < 0) | (colors > 1)).any(-1).mean()),
            "fraction_channels_at_or_below.01": float((colors <= .01).mean()),
            "fraction_channels_at_or_above.99": float((colors >= .99).mean())}


def main():
    output = OUT / "posthoc_delta_diagnostics.json"
    if output.exists():
        raise FileExistsError("Refuse to overwrite posthoc diagnostics")
    audit_path = OUT / "result_audit.json"
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "passed"
    plan = json.loads((OUT / "plan.json").read_text())
    assert sha(OUT / "plan.json") == audit["plan"]["sha256"]
    assert sha(plan["base"]) == plan["input_hashes"][plan["base"]]
    torch.set_num_threads(8)
    base = torch.load(plan["base"], map_location="cpu", weights_only=False)
    before = base["model"]
    sigma = before["splats.log_scales"].double().exp().numpy()
    max_sigma = sigma.max(-1)
    aspect = max_sigma / sigma.min(-1)
    thresholds = np.quantile(max_sigma, [.5, .9, .99])
    boundaries = [-np.inf, *thresholds, np.inf]
    bins = [(f"scale_quantile_{a}_{b}", (max_sigma > boundaries[i]) & (max_sigma <= boundaries[i+1]))
            for i, (a, b) in enumerate(((0, 50), (50, 90), (90, 99), (99, 100)))]
    prior = before["semantic_prior_counts"].numpy()
    no_votes = prior.sum(-1) == 0
    bg_votes = ~no_votes & (prior.argmax(-1) == 0)
    fg_votes = ~no_votes & (prior.argmax(-1) != 0)
    prior_bins = [("no_human_prior_votes", no_votes), ("background_dominant_prior", bg_votes),
                  ("foreground_dominant_prior", fg_votes)]
    base_metrics = json.loads(Path(plan["base_metrics"]).read_text())
    base_views = {v["name"]: v for v in base_metrics["views"]}
    result = {}
    states = {}
    for arm in plan["arms"]:
        checkpoint = OUT / arm / "appearance.pt"
        assert sha(checkpoint) == audit["arms"][arm]["checkpoint"]["sha256"]
        model = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        states[arm] = model
        tensors = {}
        for key, value in model.items():
            delta = value.double().numpy() - before[key].double().numpy()
            tensors[key] = {"shape": list(value.shape), "delta": summary(delta),
                            "before": summary(before[key].numpy()), "after": summary(value.numpy())}
        delta0 = model["splats.sh0"].double().numpy() - before["splats.sh0"].double().numpy()
        delta_rest = model["splats.sh_rest"].double().numpy() - before["splats.sh_rest"].double().numpy()
        energy = np.square(delta0).sum((1, 2)) + np.square(delta_rest).sum((1, 2))
        gaussian_rms = np.sqrt(energy / 48)
        groups = {}
        for name, keep in bins + prior_bins:
            if not keep.any():
                groups[name] = {"gaussians": 0}
                continue
            groups[name] = {"gaussians": int(keep.sum()), "fraction_gaussians": float(keep.mean()),
                            "delta_coefficient_rms_per_gaussian": summary(gaussian_rms[keep]),
                            "share_total_squared_coefficient_change": float(energy[keep].sum() / energy.sum()),
                            "max_scale_median": float(np.median(max_sigma[keep])),
                            "aspect_median": float(np.median(aspect[keep])),
                            "no_human_prior_fraction": float(no_votes[keep].mean()),
                            "background_dominant_prior_fraction": float(bg_votes[keep].mean())}
        after_metrics = json.loads((OUT / arm / "evaluation_official/official_metrics.json").read_text())
        assert after_metrics["official_evaluation_fingerprint"] == base_metrics["official_evaluation_fingerprint"]
        views = [{"name": v["name"], **{key + "_difference": v[key] - base_views[v["name"]][key]
                                        for key in ("psnr", "ssim", "lpips")}}
                 for v in after_metrics["views"]]
        sorted_views = sorted(views, key=lambda v: v["psnr_difference"])
        psnr = np.array([v["psnr_difference"] for v in views])
        total_drop = np.maximum(-psnr, 0).sum()
        result[arm] = {"tensors": tensors, "dc_proxy_before": dc_proxy(before["splats.sh0"]),
                       "dc_proxy_after": dc_proxy(model["splats.sh0"]),
                       "background_sigmoid_before": before["background_logits"].sigmoid().tolist(),
                       "background_sigmoid_after": model["background_logits"].sigmoid().tolist(),
                       "background_near_saturation_after": ((model["background_logits"].sigmoid() <= .01)
                                                             | (model["background_logits"].sigmoid() >= .99)).tolist(),
                       "fixed_posthoc_groups": groups, "view_differences": sorted_views,
                       "psnr_difference_summary": summary(psnr), "psnr_worse_view_count": int((psnr < 0).sum()),
                       "psnr_drop_gt1db_count": int((psnr < -1).sum()),
                       "psnr_drop_gt2db_count": int((psnr < -2).sum()),
                       "psnr_drop_gt3db_count": int((psnr < -3).sum()),
                       "worst5_share_total_psnr_drop": float(sum(max(-v["psnr_difference"], 0) for v in sorted_views[:5]) / total_drop)}
    between = {key: summary(states["01_original"][key].double().numpy() - states["00_native"][key].double().numpy())
               for key in states["00_native"]}
    report = {"status": "completed", "analysis_status": "posthoc descriptive only; no causal or selected-Gaussian claim",
              "cpu_only": True, "audited_result": {"path": str(audit_path), "sha256": sha(audit_path)},
              "script_sha256": sha(__file__), "base_checkpoint_sha256": sha(plan["base"]),
              "scale_group_definition": "base max(exp(log_scales)) world-unit quantiles 0-50/50-90/90-99/99-100; no VAL-derived threshold",
              "scale_thresholds_50_90_99": thresholds.tolist(), "scene_scale": base["scene_scale"],
              "prior_group_caveat": "Voting groups are inherited 2D-label evidence; zero votes does not identify a physical background or artificial Gaussian origin.",
              "saturation_caveat": "Only background sigmoid and SH0 DC proxy checked; actual directional RGB/clamp saturation needs rendering and is not inferred here.",
              "arms": result, "raw_minus_native_appearance_tensor_differences": between}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "completed", "output": str(output),
                      "arms": {a: {k: v[k] for k in ("psnr_worse_view_count", "psnr_difference_summary", "worst5_share_total_psnr_drop")}
                               for a, v in result.items()}}))


if __name__ == "__main__":
    main()
