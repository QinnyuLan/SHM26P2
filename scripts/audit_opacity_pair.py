"""Verify an opacity-only matched pair and compare all native held-out views."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from compare_evaluations import paired_comparison


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def opacity_stats(checkpoint):
    p = checkpoint["model"]["splats.opacity_logits"].double().sigmoid()
    entropy = -(torch.special.xlogy(p, p) + torch.special.xlogy(1-p, 1-p))
    return {"mean": float(p.mean()), "median": float(p.median()),
            "below_001_fraction": float((p < .01).double().mean()),
            "above_099_fraction": float((p > .99).double().mean()),
            "mean_bernoulli_entropy": float(entropy.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("--treatment-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)
    snapshots, receipts, metrics, checks = {}, {}, {}, {}
    for name, directory in (("control", args.control), ("treatment", args.treatment)):
        receipt = json.loads((directory / "experiment_receipt.json").read_text())
        checkpoint_path, metrics_path = directory / "last.pt", directory / "evaluation_native/metrics.json"
        assert receipt["status"] == "completed", f"{name}: incomplete experiment"
        assert sha256(checkpoint_path) == receipt["checkpoint_sha256"], f"{name}: checkpoint changed"
        assert sha256(metrics_path) == receipt["evaluation_sha256"], f"{name}: evaluation changed"
        assert receipt["input_hashes"]["initial_checkpoint"]["sha256"] == sha256(args.baseline)
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert saved["config"]["parameter_scope"] == "opacity_only"
        assert baseline["model"].keys() == saved["model"].keys()
        changed = [key for key in baseline["model"] if not equal(baseline["model"][key], saved["model"][key])]
        assert changed == ["splats.opacity_logits"], f"{name}: non-opacity model state changed: {changed}"
        assert equal(baseline["training_cameras"], saved["training_cameras"]), f"{name}: camera changed"
        snapshots[name], receipts[name] = saved, receipt
        metrics[name] = json.loads(metrics_path.read_text())
        assert metrics[name]["validation_views"] == 50 and metrics[name]["semantic_validation_views"] == 41
        assert metrics[name]["scale"] == 1 and metrics[name]["lpips"] is not None
        checks[name] = {"changed_model_keys": changed, "cameras_bitwise_unchanged": True,
                        "step": saved["step"], "opacity_statistics": opacity_stats(saved)}
    a, b = snapshots["control"], snapshots["treatment"]
    assert a["step"] == b["step"]
    differences = sorted(key for key in a["config"].keys() | b["config"].keys()
                         if a["config"].get(key) != b["config"].get(key))
    assert differences == sorted(["output", args.treatment_key]), differences
    assert receipts["control"]["source_hashes"] == receipts["treatment"]["source_hashes"]
    assert receipts["control"]["input_hashes"] == receipts["treatment"]["input_hashes"]
    for key in ("torch_rng", "cuda_rng", "numpy_rng"):
        assert equal(a[key], b[key]), f"Different final RNG: {key}"
    for key in ("sampler_order", "sampler_cursor", "sampler_rng_state"):
        assert equal(a["density_state"]["extras"][key], b["density_state"]["extras"][key]), key
    report = {"baseline": str(args.baseline.resolve()), "baseline_sha256": sha256(args.baseline),
              "control": str(args.control.resolve()), "treatment": str(args.treatment.resolve()),
              "baseline_opacity": opacity_stats(baseline), "invariants": checks,
              "source_and_input_hashes_identical": True, "sampler_and_rng_identical": True,
              "configuration_difference": differences,
              "comparison": paired_comparison(metrics["control"], metrics["treatment"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"invariants": checks, "metrics": report["comparison"]["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
