"""CPU contracts for the fixed TRAIN refiner sensitivity audit; no model fit."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = load_script("audit_paired_render_sensitivity")
maker = load_script("make_paired_render_sensitivity_plan")


def manifest():
    views = [{"name": f"{i:03d}.png", "split": "train", "mask_path": "mask", "image_path": "rgb",
              "valid_path": "valid", "width": 12, "height": 9} for i in range(259)]
    views.extend([{"name": "990.png", "split": "val", "mask_path": "mask", "width": 12, "height": 9},
                  {"name": "991.png", "split": "train", "mask_path": None, "width": 12, "height": 9}])
    return {"pixel_protocol": audit.PROFILE, "class_names": list(audit.CLASSES), "views": list(reversed(views))}


def test_fixed_16_is_sorted_train_labeled_and_rejects_legacy_duplicates():
    data = manifest()
    indices, views = audit.fixed_views(data)
    assert indices == [i*258//15 for i in range(16)]
    assert [v["name"] for v in views] == [f"{i:03d}.png" for i in indices]
    data.pop("pixel_protocol")
    with pytest.raises(ValueError, match="corner_v2"):
        audit.fixed_views(data)
    data = manifest()
    data["views"].append(data["views"][-1])
    with pytest.raises(ValueError, match="Duplicate"):
        audit.fixed_views(data)
    data = manifest()
    data["views"][-1]["pixel_protocol"] = "legacy_mixed_v1"
    with pytest.raises(ValueError):
        audit.fixed_views(data)


def test_raw_baseline_never_clamps_and_half_paths_keep_continuous_raw_range():
    raw = np.full((9, 12, 3), 2., np.float32)
    raw[0, 0] = [-.3, .2, 1.3]
    real = np.full_like(raw, .5)
    values = {name: (x, before, info) for name, x, before, info in audit.interventions(raw, real, real-raw)}
    assert tuple(values) == audit.METHODS
    np.testing.assert_array_equal(values["raw_render"][0], raw)
    np.testing.assert_array_equal(values["paired_025"][0], .75*raw+.25*real)
    np.testing.assert_array_equal(values["paired_050"][0], .5*raw+.5*real)
    assert values["paired_050"][0].max() > 1
    for name in audit.METHODS[4:]:
        assert 0 <= values[name][0].min() <= values[name][0].max() <= 1
    np.testing.assert_array_equal(values["real_rgb"][0], real)


def test_roll_preserves_preclip_spectrum_and_next_residual_scales_channels():
    rng = np.random.default_rng(12)
    raw = rng.uniform(.1, .9, (9, 12, 3)).astype(np.float32)
    real = rng.uniform(.1, .9, raw.shape).astype(np.float32)
    delta = real-raw
    next_delta = delta * np.array([2., 0, .5], np.float32)
    values = {name: (x, pre, info) for name, x, pre, info in audit.interventions(raw, real, next_delta)}
    expected = np.roll(delta, (3, 4), (0, 1))
    reconstructed = (values["roll_plus_050"][1]-raw)*2
    np.testing.assert_allclose(reconstructed, expected, atol=1e-7)
    np.testing.assert_allclose(np.abs(np.fft.fft2(reconstructed, axes=(0, 1))),
                               np.abs(np.fft.fft2(delta, axes=(0, 1))), atol=2e-6)
    scaled = (values["next_camera_050"][1]-raw)*2
    np.testing.assert_allclose(scaled[..., (0, 2)], delta[..., (0, 2)], atol=1e-7)
    np.testing.assert_array_equal(scaled[..., 1], 0)
    assert values["next_camera_050"][2]["source_zero_rms"] == [False, True, False]


def test_affine_fixed_lattice_and_constant_channel_fallback():
    rng = np.random.default_rng(1)
    raw = rng.uniform(.2, .7, (64, 80, 3)).astype(np.float32)
    raw[..., 2] = .3
    real = raw * np.array([1.1, .8, 1.], np.float32) + np.array([.1, .05, .2], np.float32)
    candidate, info = audit.affine_candidate(raw, real)
    np.testing.assert_allclose(candidate, real, atol=1e-7)
    assert info["gain"][2] == 1.
    assert info["bias"][2] == pytest.approx(.2)
    # Off-lattice changes do not change the fit.
    real[1::16, 1::16] = 0
    assert audit.affine_candidate(raw, real)[1] == info


def test_input_statistics_reports_actual_clipping_including_partial_blocks():
    raw = np.zeros((33, 34, 3), np.float32)
    raw[0, 0] = [-.5, 0, 1.5]
    before = raw + .5
    x = np.clip(before, 0, 1)
    stats, energy = audit.input_statistics(raw, x, before)
    assert len(stats["blocks32"]) == 4
    assert stats["raw_out_of_range_pixel_fraction"] == pytest.approx(1/(33*34))
    assert stats["clip_channel_fraction"] == pytest.approx(1/(33*34*3))
    assert stats["clip_pixel_fraction"] == pytest.approx(1/(33*34))
    assert stats["actual_rms"] == pytest.approx(np.sqrt(energy.astype(np.float64).mean()))
    assert stats["preclip_rms"] == .5


def prediction_inputs(target, favored):
    base = np.full((*target.shape, 5), .2, np.float32)
    candidate = np.full_like(base, .05)
    np.put_along_axis(candidate, favored[..., None], .8, -1)
    predictions = {name: base.copy() if name == "raw_render" else candidate.copy() for name in audit.METHODS}
    energies = {name: np.zeros(target.shape, np.float32) if name == "raw_render"
                else np.ones(target.shape, np.float32) for name in audit.METHODS}
    return predictions, energies


def test_scores_ignore_invalid_pixels_and_track_opposing_flips():
    target = np.array([[0, 1, 2, 4, 255, 3]], np.uint8)
    favored = np.array([[1, 1, 0, 4, 2, 3]], np.int64)
    preds, energies = prediction_inputs(target, favored)
    valid = np.ones(target.shape, bool)
    valid[0, -1] = False
    result = audit.summarize_predictions(preds, energies, target, valid)
    all_scores = result["paired_050"]["regions"]["all"]
    counts = all_scores["counts"]
    assert counts["pixels"] == 4 and counts["corrections"] == 2 and counts["harms"] == 1
    assert all_scores["classes_present"] == 4
    assert all_scores["by_gt_class"]["tower"]["pixels"] == 0
    assert np.sum(all_scores["confusion_matrix"]) == 4
    assert result["raw_render"]["regions"]["all"]["counts"]["js_sum"] == 0
    assert result["raw_render"]["regions"]["all"]["counts"]["margin_delta_sum"] == 0
    assert counts["centered_log_probability_mse_sum"] > 0


def test_primary_is_equal_class_then_equal_camera_not_pixel_pooling():
    target = np.array([[0, 0, 0, 1]], np.uint8)
    favored = np.array([[0, 0, 0, 0]], np.int64)
    preds, energies = prediction_inputs(target, favored)
    scores = audit.summarize_predictions(preds, energies, target, np.ones_like(target, bool))
    assert scores["paired_050"]["regions"]["all"]["class_equal"]["margin"] == pytest.approx(0, abs=1e-6)
    assert scores["paired_050"]["regions"]["all"]["counts"]["margin_sum"] > 0
    y2 = np.zeros((1, 1), np.uint8)
    p2, e2 = prediction_inputs(y2, y2)
    scores2 = audit.summarize_predictions(p2, e2, y2, np.ones_like(y2, bool))
    report = audit.aggregate_views([{"scores": scores}, {"scores": scores2}], repeats=32)
    assert report["aggregate"]["paired_050"]["all"]["equal_camera_equal_present_class"]["margin"] == pytest.approx(np.log(16)/2, abs=1e-6)
    assert report["paired"]["paired_050-minus-affine_rgb"]["margin_candidate_minus_reference"]["ci95"] == [0., 0.]
    json.dumps(report, allow_nan=False)


def test_boundary_band_excludes_unknown_neighborhood_and_margin_sign():
    target = np.zeros((17, 17), np.uint8)
    target[:, 8:] = 2
    keep = np.ones_like(target, bool)
    edge = audit.boundary_band(target, keep)
    assert edge[8, 7] and edge[8, 8] and not edge[0].any()
    keep[8, 8] = False
    assert not audit.boundary_band(target, keep)[6:11, 6:11].any()
    p, e = prediction_inputs(target, target.astype(np.int64))
    report = audit.summarize_predictions(p, e, target, keep)
    value = report["paired_050"]["regions"]["all"]["class_equal"]
    assert value["nll_gain"] > 0 and value["margin_delta"] > 0


def evidence():
    p = torch.full((9, 12, 5), .2)
    return {"features": torch.zeros(9, 12, 16), "rgb": torch.linspace(-.2, 1.2, 9*12*3).reshape(9, 12, 3),
            "depth": torch.ones(9, 12, 1), "alpha": torch.ones(9, 12, 1),
            "p3d": p, "refinement_prior": p}


class EvidenceSpy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from bridge_rgs.refinement import MultiScaleRefinementHead
        self.head = MultiScaleRefinementHead()
        self.inputs = []

    def forward(self, features, rgb, depth, alpha, p3d):
        self.inputs.append(self.head.evidence_tensor(features, rgb, depth, alpha, p3d).clone())
        residual = rgb.new_zeros(*rgb.shape[:2], 5)
        residual[..., 0] = rgb[..., 0]
        return residual


def test_all_ten_predictions_precede_gt_and_rgb_gradient_is_recomputed():
    rendered = evidence()
    spy = EvidenceSpy()
    real = np.zeros((9, 12, 3), np.float32)

    def reader():
        assert len(spy.inputs) == 10
        return np.zeros((9, 12), np.uint8), np.ones((9, 12), bool)

    scores, diagnostics, baseline = audit.predict_then_score(spy, rendered, real, real, reader)
    assert tuple(scores) == audit.METHODS and tuple(diagnostics) == audit.METHODS
    expected = audit.head_prediction(spy, rendered, rendered["rgb"]).numpy()
    np.testing.assert_array_equal(baseline, expected)
    assert torch.count_nonzero(spy.inputs[0][:, -2] != spy.inputs[3][:, -2]) > 0
    assert torch.equal(spy.inputs[0][:, :16], spy.inputs[3][:, :16])
    assert torch.equal(spy.inputs[0][:, 19:-2], spy.inputs[3][:, 19:-2])
    assert torch.equal(spy.inputs[0][:, -1], spy.inputs[3][:, -1])


def test_evidence_mutation_aborts_before_gt_and_timeout_aborts():
    class Mutating(torch.nn.Module):
        def forward(self, features, rgb, depth, alpha, p3d):
            features.add_(1)
            return torch.zeros_like(p3d)

    def reader():
        pytest.fail("GT must not be opened after evidence-contract failure")

    with pytest.raises(ValueError, match="frozen evidence"):
        audit.predict_then_score(Mutating(), evidence(), np.zeros((9, 12, 3), np.float32),
                                 np.zeros((9, 12, 3), np.float32), reader)
    with pytest.raises(TimeoutError):
        audit.predict_then_score(EvidenceSpy(), evidence(), np.zeros((9, 12, 3), np.float32),
                                 np.zeros((9, 12, 3), np.float32), reader, deadline=0)


def test_pending_plan_cannot_execute_or_read_changing_checkpoint(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="pending draft"):
        audit.verify_plan({"status": "pending_completed_checkpoint_and_official_evaluation"}, "unused")
    a, b = tmp_path / "train.json", tmp_path / "official.json"
    a.write_text(json.dumps({"status": "running"}))
    b.write_text(json.dumps({"status": "completed"}))
    monkeypatch.setattr(maker, "digest", lambda path: pytest.fail("Must not read checkpoint before completion"))
    with pytest.raises(ValueError, match="both be completed"):
        maker.validate_completed_inputs({"pending_checkpoint_path": "changing_last.pt"}, a, b)


def test_draft_records_static_train_inputs_without_checkpoint_access(tmp_path, monkeypatch):
    data = manifest()
    for field in ("image_path", "mask_path", "valid_path"):
        path = tmp_path / field
        path.write_bytes(b"test-static-bytes")
        for view in data["views"]:
            if view.get(field):
                view[field] = str(path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(maker, "ROOT", tmp_path)
    (tmp_path / "uv.lock").write_bytes(b"uv")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/audit_paired_render_sensitivity.py").write_bytes(b"runner")
    pending = tmp_path / "no_checkpoint_exists.pt"
    draft = maker.prepare_draft(path, pending, tmp_path / "new_output")
    assert not pending.exists() and draft["checkpoint"] is None and draft["checkpoint_sha256"] is None
    assert draft["checkpoint_bytes_read_at_draft"] is False
    assert len(draft["input_hashes"]) == 2 + 16*3


def test_invalid_input_rejected_before_prediction():
    raw = np.zeros((9, 12, 3), np.float32)
    invalid = raw.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        list(audit.interventions(invalid, raw, raw))
    with pytest.raises(ValueError, match="grids"):
        list(audit.interventions(raw, raw[:1], raw))
    with pytest.raises(ValueError, match="Zero probability"):
        audit.probabilities(np.zeros((1, 1, 5), np.float32))


def test_completed_receipts_bind_actual_corner_checkpoint_and_fixed_endpoint(tmp_path):
    from bridge_rgs.official_evaluate import FAMILY, PLAIN_INFERENCE, SCORING_PROTOCOL
    checkpoint = tmp_path / "last.pt"
    state = {"step": 8000, "manifest_sha256": "abc", "pixel_protocol": audit.PROFILE,
             "refiner_config": audit.protocol_spec()["refiner_config"]}
    torch.save(state, checkpoint)
    (tmp_path / "evaluation_native").mkdir()
    native = tmp_path / "evaluation_native/metrics.json"
    native.write_text("{}")
    (tmp_path / "official").mkdir()
    official_metrics = tmp_path / "official/official_metrics.json"
    official_metrics.write_text("{}")
    train_path, official_path = tmp_path / "experiment_receipt.json", tmp_path / "official/execution_receipt.json"
    train = {"status": "completed", "config": {"steps": 8000}, "checkpoint_sha256": audit.digest(checkpoint),
             "evaluation_sha256": audit.digest(native), "input_hashes": {"manifest": {"sha256": "abc"}}}
    official = {"status": "completed", "checkpoint": str(checkpoint), "checkpoint_sha256": audit.digest(checkpoint),
                "checkpoint_pixel_protocol": audit.PROFILE, "evaluation_family": FAMILY,
                "inference_protocol": PLAIN_INFERENCE, "scoring_protocol": SCORING_PROTOCOL,
                "training_manifest": {"observed_sha256": "abc"},
                "official_metrics_sha256": audit.digest(official_metrics)}
    train_path.write_text(json.dumps(train))
    official_path.write_text(json.dumps(official))
    draft = {"pending_checkpoint_path": str(checkpoint), "input_hashes": {"manifest": {"sha256": "abc"}}}
    records = maker.validate_completed_inputs(draft, train_path, official_path)
    assert records["student_checkpoint"]["sha256"] == audit.digest(checkpoint)
    state["pixel_protocol"] = "legacy_mixed_v1"
    torch.save(state, checkpoint)
    train["checkpoint_sha256"] = official["checkpoint_sha256"] = audit.digest(checkpoint)
    train_path.write_text(json.dumps(train))
    official_path.write_text(json.dumps(official))
    with pytest.raises(ValueError, match="Actual checkpoint profile"):
        maker.validate_completed_inputs(draft, train_path, official_path)


def test_locked_path_binding_and_source_inventory_are_fail_closed(tmp_path):
    runner = tmp_path / "runner.py"
    runner.write_text("pass")
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"frozen")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    source = tmp_path / "snapshot"
    source.mkdir()
    module = source / "model.py"
    module.write_text("pass")
    plan = {"status": "locked", "protocol": audit.PROTOCOL, "specification": audit.protocol_spec(),
            "runner_sha256": audit.digest(runner), "checkpoint": str(checkpoint),
            "checkpoint_sha256": audit.digest(checkpoint), "manifest": str(manifest_path),
            "input_hashes": {"student_checkpoint": audit.record(checkpoint), "manifest": audit.record(manifest_path)},
            "source_snapshot": str(source), "source_hashes": {"model.py": audit.digest(module)}}
    audit.verify_plan(plan, runner)
    plan["checkpoint"] = str(tmp_path / "another.pt")
    with pytest.raises(ValueError, match="bound"):
        audit.verify_plan(plan, runner)
    plan["checkpoint"] = str(checkpoint)
    (source / "unexpected.py").write_text("pass")
    with pytest.raises(ValueError, match="inventory"):
        audit.verify_plan(plan, runner)
