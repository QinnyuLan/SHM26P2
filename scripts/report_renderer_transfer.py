"""Report all predeclared fixed-weight renderer-transfer controls without selection."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from compare_evaluations import paired_comparison


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/teacher_renderer_transfer_support"))
    args = parser.parse_args()
    receipt = json.loads((args.output / "execution_receipt.json").read_text())
    assert receipt["status"] == "completed"
    metrics_path = args.output / "metrics.json"
    assert sha256(metrics_path) == receipt["metrics_sha256"]
    metrics = json.loads(metrics_path.read_text())
    plan = metrics["plan"]
    assert sha256(args.output / "plan.json") == receipt["plan_sha256"]
    assert plan["inference"]["teacher_weight"] == .5
    reference_path = Path(plan["students"]["base"]["evaluation"])
    assert sha256(reference_path) == plan["students"]["base"]["evaluation_sha256"]
    native = json.loads(reference_path.read_text())
    transfer_views = {view["name"]: view for view in metrics["per_view"]}
    assert len(transfer_views) == 50
    adapted = {}
    for domain in metrics["metrics"]:
        document = {key: copy.deepcopy(native[key]) for key in
                    ("protocol", "evaluation_fingerprint", "scale", "lpips_validity_policy")}
        document["views"] = []
        for row in native["views"]:
            item = {key: row[key] for key in ("name", "psnr", "ssim", "lpips") if key in row}
            if "confusion_matrix" in row:
                item["confusion_matrix"] = transfer_views[row["name"]]["metrics"][domain]["confusion"]
            document["views"].append(item)
        adapted[domain] = document
    pairs = [(student, f"ensemble_{student}_0.5") for student in plan["students"]]
    pairs += [("teacher", f"ensemble_{student}_0.5") for student in plan["students"]]
    pairs += [("ensemble_01_variance_0.5", "ensemble_02_cross_0.5"),
              ("ensemble_base_0.5", "ensemble_02_cross_0.5")]
    results = {}
    for reference, candidate in pairs:
        key = f"{candidate}_minus_{reference}"
        comparison = paired_comparison(adapted[reference], adapted[candidate])
        comparison.update(reference_domain=reference, candidate_domain=candidate,
                          source_metrics_path=str(metrics_path.resolve()),
                          source_metrics_sha256=receipt["metrics_sha256"],
                          all_prediction_pairs_reported_without_weight_search=True)
        (args.output / f"paired_{key}.json").write_text(json.dumps(comparison, indent=2) + "\n")
        results[key] = comparison["metrics"]
    report = {"status": "completed", "protocol": "renderer_transfer_v1",
              "input_metrics_sha256": receipt["metrics_sha256"],
              "plan_sha256": receipt["plan_sha256"],
              "all_predeclared_domains": list(metrics["metrics"]),
              "no_additional_model_updates_or_inference": True,
              "metrics": metrics["metrics"], "paired": results,
              "scope": "One scene and development split; view bootstrap does not measure cross-seed, cross-scene, or blind-test uncertainty. All four fixed combinations are reported, not selected by a weight sweep. No extra novelty claim."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: {metric: value[metric] for metric in ("miou_all", "stay_cable_iou")}
                      for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
