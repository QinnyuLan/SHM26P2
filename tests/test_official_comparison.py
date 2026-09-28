import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bridge_rgs.official_evaluate import FAMILY, PLAIN_INFERENCE, SCORING_PROTOCOL

spec = importlib.util.spec_from_file_location(
    "official_comparison", Path(__file__).parents[1] / "scripts/compare_official_evaluations.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def refresh_semantic_summary(metrics):
    matrix = np.asarray([v["confusion_matrix"] for v in metrics["views"]
                         if "confusion_matrix" in v]).sum(0)
    union = matrix.sum(0) + matrix.sum(1) - matrix.diagonal()
    scores = [float(matrix[i, i] / u) if u else None for i, u in enumerate(union)]
    metrics.update(confusion_matrix=matrix.tolist(), iou=scores,
                   miou_all=float(np.mean([s for s in scores if s is not None])),
                   miou_foreground=float(np.mean([s for s in scores[1:] if s is not None])))
    return metrics


def fixture_metrics():
    views = []
    matrix = np.diag([29, 23, 23, 23, 23]).tolist()
    for i in range(50):
        view = {"name": f"{i:03d}.png", "width": 11, "height": 11, "rgb_pixels": 121,
                "psnr": 20. + i / 50, "ssim": .8, "lpips": .2}
        if i < 41:
            view.update(confusion_matrix=copy.deepcopy(matrix), semantic_pixels=121, semantic_ignore_pixels=0)
        views.append(view)
    return refresh_semantic_summary({"evaluation_family": FAMILY, "official_evaluation_fingerprint": "fixed_fixture",
            "scoring_protocol": copy.deepcopy(SCORING_PROTOCOL), "validation_views": 50,
            "inference_protocol": PLAIN_INFERENCE,
            "semantic_validation_views": 41, "views": views,
            "confusion_matrix": (np.asarray(matrix) * 41).tolist(),
            **{key: float(np.mean([v[key] for v in views])) for key in ("psnr", "ssim", "lpips")}})


def test_identical_predictions_zero_and_reordering_invariant():
    a = fixture_metrics()
    b = copy.deepcopy(a)
    b["views"].reverse()
    result = module.paired_official_comparison(a, b, repeats=100)
    assert all(v["difference"] == 0 and v["paired_view_bootstrap_95_interval"] == [0, 0]
               for v in result["metrics"].values())


def test_paired_constant_rgb_gain_and_pooled_semantic_direction():
    a, b = fixture_metrics(), fixture_metrics()
    for view in b["views"]:
        view["psnr"] += 2
        view["lpips"] -= .05
        if "confusion_matrix" in view:
            view["confusion_matrix"][2][2] -= 3
            view["confusion_matrix"][2][0] += 3
    b["psnr"] += 2
    b["lpips"] -= .05
    refresh_semantic_summary(b)
    result = module.paired_official_comparison(a, b, repeats=100)["metrics"]
    assert result["psnr"]["paired_view_bootstrap_95_interval"] == pytest.approx([2, 2])
    assert result["lpips"]["difference"] == pytest.approx(-.05)
    assert result["stay_cable_iou"]["difference"] == pytest.approx(-3 / 23)
    assert result["miou_all"]["difference"] < 0


def test_semantic_bootstrap_pools_heterogeneous_region_counts_before_iou():
    a, b = fixture_metrics(), fixture_metrics()
    for metrics, recovered in ((a, 0), (b, 10)):
        for view in metrics["views"][:40]:
            view["confusion_matrix"] = np.diag([117, 1, 1, 1, 1]).tolist()
        last = np.diag([18, 1, recovered, 1, 1])
        last[2, 0] = 100 - recovered
        metrics["views"][40]["confusion_matrix"] = last.tolist()
        refresh_semantic_summary(metrics)
    result = module.paired_official_comparison(a, b, repeats=1000, seed=123)["metrics"]["stay_cable_iou"]
    assert result["reference"] == pytest.approx(40 / 140)
    assert result["candidate"] == pytest.approx(50 / 140)
    assert result["difference"] == pytest.approx(10 / 140)
    assert result["difference"] != pytest.approx((10 / 100) / 41)
    rng = np.random.default_rng(123)
    rng.integers(50, size=(1000, 50))  # The independent RGB resampling consumes this part of the stream.
    sampled = rng.integers(41, size=(1000, 41))
    large_view_count = (sampled == 40).sum(1)
    expected_delta = 10 * large_view_count / (41 + 99 * large_view_count)
    assert result["paired_view_bootstrap_95_interval"] == pytest.approx(np.quantile(expected_delta, [.025, .975]))


def test_nine_unannotated_views_still_contribute_to_rgb_scores():
    a, b = fixture_metrics(), fixture_metrics()
    for view in b["views"][41:]:
        view["psnr"] += 1
    b["psnr"] = float(np.mean([v["psnr"] for v in b["views"]]))
    result = module.paired_official_comparison(a, b, repeats=1000, seed=12)
    assert len(result["views"]) == 50 and len(result["semantic_views"]) == 41
    assert result["metrics"]["psnr"]["difference"] == pytest.approx(9 / 50)
    sample = np.random.default_rng(12).integers(50, size=(1000, 50))
    assert result["metrics"]["psnr"]["paired_view_bootstrap_95_interval"] == pytest.approx(
        np.quantile((sample >= 41).mean(1), [.025, .975]))
    assert result["metrics"]["miou_all"]["difference"] == 0


def test_absent_class_remains_undefined_and_report_serializes_without_nan():
    a = fixture_metrics()
    for view in a["views"][:41]:
        matrix = view["confusion_matrix"]
        matrix[0][0] += matrix[4][4]
        matrix[4][4] = 0
    refresh_semantic_summary(a)
    result = module.paired_official_comparison(a, copy.deepcopy(a), repeats=100)
    foundation = result["metrics"]["foundation_iou"]
    assert foundation == {"reference": None, "candidate": None, "difference": None,
                          "paired_view_bootstrap_95_interval": None, "finite_bootstrap_replicates": 0}
    assert result["metrics"]["miou_foreground"]["difference"] == 0
    assert result["metrics"]["miou_foreground"]["finite_bootstrap_replicates"] == 100
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mutation", ["native", "fingerprint", "subset", "duplicate", "support", "pooled", "scoring"])
def test_reject_incompatible_or_incomplete_comparisons(mutation):
    a, b = fixture_metrics(), fixture_metrics()
    if mutation == "native":
        b["evaluation_family"] = "native"
    elif mutation == "fingerprint":
        b["official_evaluation_fingerprint"] = "different"
    elif mutation == "subset":
        b["views"].pop()
    elif mutation == "duplicate":
        b["views"][1]["name"] = b["views"][0]["name"]
    elif mutation == "support":
        b["views"][0]["rgb_pixels"] = 120
    elif mutation == "pooled":
        b["confusion_matrix"][0][0] += 1
    else:
        b["scoring_protocol"]["id"] = "other"
    with pytest.raises(ValueError):
        module.paired_official_comparison(a, b, repeats=100)


@pytest.mark.parametrize("key", ["iou", "miou_all", "miou_foreground"])
def test_reject_semantic_summary_inconsistent_with_pooled_counts(key):
    a, b = fixture_metrics(), fixture_metrics()
    if key == "iou":
        b[key][2] = .5
    else:
        b[key] = .5
    with pytest.raises(ValueError):
        module.paired_official_comparison(a, b, repeats=100)


def test_reject_negative_ignore_count_even_if_sum_covers_image():
    a, b = fixture_metrics(), fixture_metrics()
    b["views"][0]["confusion_matrix"][0][0] += 1
    b["views"][0].update(semantic_pixels=122, semantic_ignore_pixels=-1)
    refresh_semantic_summary(b)
    with pytest.raises(ValueError):
        module.paired_official_comparison(a, b, repeats=100)


def fixture_receipt(tmp_path):
    metrics = fixture_metrics()
    path = tmp_path / "official_metrics.json"
    path.write_text(json.dumps(metrics))
    receipt = {
        "status": "completed", "evaluation_family": FAMILY,
        "official_evaluation_fingerprint": metrics["official_evaluation_fingerprint"],
        "official_metrics_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint": "/model.pt", "checkpoint_sha256": "checkpoint-fixture",
        "checkpoint_pixel_protocol": {"id": "colmap_corner_v2"}, "training_manifest": {"path": "/manifest.json"},
        "scoring_protocol": copy.deepcopy(SCORING_PROTOCOL),
        "inference_protocol": PLAIN_INFERENCE,
    }
    receipt_path = tmp_path / "execution_receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    return metrics, path, receipt, receipt_path


def test_completed_receipt_binds_exact_metrics_and_profile_provenance(tmp_path):
    metrics, path, receipt, receipt_path = fixture_receipt(tmp_path)
    loaded, provenance = module.read_completed(path)
    assert loaded == metrics
    assert provenance["checkpoint_pixel_protocol"] == receipt["checkpoint_pixel_protocol"]
    receipt["status"] = "failed"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="completed"):
        module.read_completed(path)
    receipt["status"] = "completed"
    receipt_path.write_text(json.dumps(receipt))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="exact metric bytes"):
        module.read_completed(path)


@pytest.mark.parametrize("mutation", ["scoring", "inference", "native"])
def test_reject_receipt_protocol_mismatch(tmp_path, mutation):
    _, path, receipt, receipt_path = fixture_receipt(tmp_path)
    if mutation == "scoring":
        receipt["scoring_protocol"]["rgb_scoring"] = "different support"
    elif mutation == "inference":
        receipt["inference_protocol"] = "different inference than bound metrics"
    else:
        receipt["evaluation_family"] = "native"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        module.read_completed(path)
