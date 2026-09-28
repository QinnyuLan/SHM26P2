"""CPU contracts for the single bounded H1 recovery diagnostic."""
import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bridge_rgs import reliability as api

spec = importlib.util.spec_from_file_location(
    "camera_recovery", Path(__file__).parents[1] / "scripts/audit_camera_attribution_recovery.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture_metadata():
    root = Path(__file__).parents[1]
    return (json.loads((root / "artifacts/prepared/manifest.json").read_text()),
            json.loads((root / "artifacts/pose_stress_mild/manifest.json").read_text()))


def test_fixed_eight_historical_train_only_and_no_label_paths():
    # Metadata only; no checkpoint load, image/label decoder, or GPU call.
    manifest, stress = fixture_metadata()
    views = audit.selected_views(manifest, stress)
    assert [v["name"] for v in views] == list(audit.NAMES)
    assert all(v["split"] == "train" and set(v) == set(audit.VIEW_KEYS) | {"half_twist"} for v in views)
    for view in views:
        original = np.array(stress["pose_stress"]["injected_left_twists_translation_first"][view["name"]])
        np.testing.assert_array_equal(view["half_twist"], original * .5)
    audit.validate_twists(views, 16.261470794677734)


@pytest.mark.parametrize("change", ["profile", "population", "injection"])
def test_sampling_contract_cannot_change(change):
    manifest, stress = fixture_metadata()
    if change == "profile":
        manifest["pixel_protocol"] = "colmap_corner_v2"
    elif change == "population":
        manifest["views"].pop()
    else:
        stress["pose_stress"]["injected_left_twists_translation_first"].pop("003.png")
    with pytest.raises(ValueError):
        audit.selected_views(manifest, stress)


def test_trust_uses_actual_model_scale_not_manifest_radius():
    pose = torch.eye(4, dtype=torch.double)
    options = audit.production_options(16.261470794677734, pose)
    assert options["max_translation"] == pytest.approx(.016261470794677734)
    assert options["max_rotation"] == .002
    assert options["prior_precision"][0, 0] == pytest.approx(1 / (.005 * 16.261470794677734) ** 2)
    with pytest.raises(ValueError, match="bounds"):
        audit.validate_twists([{"half_twist": [.02, 0, 0, 0, 0, 0]}], 16.26147)


def analytic_renderer():
    basis = torch.randn(8, 8, 3, 6, generator=torch.Generator().manual_seed(47), dtype=torch.double) * 100
    calls = []

    def render(pose):
        calls.append(pose.clone())
        rotation = pose[:3, :3]
        vector = torch.cat((pose[:3, 3], torch.stack((rotation[2, 1] - rotation[1, 2],
                           rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1])) / 2))
        return .5 + torch.einsum("...i,i->...", basis, vector)

    return render, calls


def test_cached_solve_exactly_matches_production_verified_diagnostic():
    render, _ = analytic_renderer()
    pose = api.se3_exp(torch.tensor([.0002, -.0001, 0, 0, .0003, -.0002], dtype=torch.double))
    target = render(torch.eye(4, dtype=torch.double))
    weight = torch.ones(8, 8, 1, dtype=torch.double)
    options = audit.production_options(2., pose)
    base, jacobian = api.finite_difference_camera_jacobian(render, pose, 2e-5, 1e-4)
    _, result = audit.solve_verified(api, render, pose, base, jacobian, target, weight, options)
    expected = api.camera_compensated_residual(render, pose, target, weights=weight,
                                              translation_eps=2e-5, rotation_eps=1e-4, **options)
    torch.testing.assert_close(torch.tensor(result["delta"], dtype=torch.double), expected.delta, atol=0, rtol=0)
    assert result["accepted"] == expected.accepted
    assert result["after_mse"] == float(expected.after_energy)


def test_full_cpu_view_has_two_jacobians_four_verifications_and_exact_zero_real_increment():
    render, calls = analytic_renderer()
    pose = torch.eye(4, dtype=torch.double)
    target = render(pose)
    calls.clear()
    x, y = torch.meshgrid(torch.linspace(-.2, .2, 10, dtype=torch.double),
                          torch.linspace(-.2, .2, 10, dtype=torch.double), indexing="ij")
    points = torch.stack((x.flatten(), y.flatten(), torch.ones(100, dtype=torch.double) * 3), -1)
    K = torch.tensor([[20., 0, 4], [0, 20, 4], [0, 0, 1]], dtype=torch.double)
    twist = torch.tensor([.0002, -.0001, .0001, .0001, .0003, -.0002], dtype=torch.double)
    result = audit.audit_view(api, render, pose, twist, target, torch.ones(8, 8, dtype=torch.double), points, K, 2.)
    assert len(calls) == 30
    assert result["common_points"] == 100
    assert result["recovery_ratios"]["self_production"] < .02
    assert result["recovery_ratios"]["real_increment"] == result["recovery_ratios"]["self_production"]
    assert result["recovery_ratios"]["self_no_prior"] < .02
    assert len(result["information_base"]["linear_mode_retention"]) == 6


def test_real_increment_uses_composition_not_delta_subtraction():
    pose = api.se3_exp(torch.tensor([.2, -.1, .4, .1, .2, -.2], dtype=torch.double))
    E = api.se3_exp(torch.tensor([.001, .002, 0., .01, -.02, .01], dtype=torch.double))
    D0 = api.se3_exp(torch.tensor([.001, 0., -.003, .04, .02, -.01], dtype=torch.double))
    De = D0 @ torch.linalg.inv(E)
    metrics = audit.pose_separation(De @ E @ pose, D0 @ pose, 2.)
    assert metrics["rotation_radians"] < 1e-14
    assert metrics["translation_scene_scale"] < 1e-14


def test_failed_nonlinear_candidate_keeps_original_pose_and_reports_proposal():
    pose = torch.eye(4, dtype=torch.double)

    def render(candidate):
        x = candidate[0, 3]
        return (x + 1000 * x.square()).expand(1, 1, 3)

    base, jacobian = api.finite_difference_camera_jacobian(render, pose)
    result_pose, result = audit.solve_verified(api, render, pose, base, jacobian,
        torch.full_like(base, -.02), torch.ones(1, 1, 1),
        {"damping": 1e-8, "max_translation": .1, "max_rotation": .1, "prior_precision": None})
    assert not result["accepted"] and result["explained_fraction"] == 0
    assert result["delta"] == [0.] * 6 and result["proposed_delta"][0] != 0
    assert torch.equal(result_pose, pose)


def test_projection_points_are_not_dropped_after_bad_correction():
    points = torch.tensor([[0., 0, 3.]] * 32)
    first, second = torch.eye(4), torch.eye(4)
    second[2, 3] = -4
    assert audit.projection_distance(api, points, first, second, torch.eye(3)) is None


def records():
    return [{"name": name, "recovery_ratios": {k: .2 for k in ("self_production", "real_increment", "self_no_prior")}}
            for name in audit.NAMES]


def test_gate_requires_all_eight_fixed_records_no_partial_success():
    values = records()
    with pytest.raises(ValueError, match="Incomplete"):
        audit.summarize(values[:-1])
    assert audit.summarize(values)["real_increment"]["necessary_gate_passed"]
    for value in values[:3]:
        value["recovery_ratios"]["real_increment"] = None
    assert not audit.summarize(values)["real_increment"]["necessary_gate_passed"]


def test_bound_inputs_detect_changes(tmp_path):
    path = tmp_path / "input"
    path.write_text("stable")
    plan = {"input_hashes": {str(path): audit.digest(path)}, "source_hashes": {}}
    audit.bind(plan)
    path.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        audit.bind(plan)


def test_outer_hard_timeout_preserves_draft_and_forbids_retry(tmp_path, monkeypatch):
    snapshot = tmp_path / "source_snapshot"
    snapshot.mkdir()
    script = snapshot / "audit_camera_attribution_recovery.py"
    script.write_text("frozen")
    plan = {"status": "cpu_registered_draft_pending_gpu_handoff", "protocol": audit.PROTOCOL,
            "specification": copy.deepcopy(audit.SPEC), "source_snapshot": str(snapshot),
            "output": str(tmp_path), "workspace_root": str(tmp_path), "source_hashes": {script.name: audit.digest(script)},
            "input_hashes": {}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    before = audit.digest(plan_path)
    monkeypatch.setattr(audit, "__file__", str(script))
    monkeypatch.setattr(audit, "gpu_idle", list)

    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 120
        assert kwargs["env"]["PYTHONPATH"] == str(snapshot)
        raise subprocess.TimeoutExpired(command, 120)

    monkeypatch.setattr(audit.subprocess, "run", timeout)
    result = audit.execute(plan_path)
    assert result["status"] == "inconclusive_timeout"
    assert audit.digest(plan_path) == before
    with pytest.raises(ValueError, match="rerun"):
        audit.execute(plan_path)


def test_gpu_handoff_rejects_compute_or_unknown_processes(monkeypatch):
    result = SimpleNamespace(stdout="<nvidia_smi_log><gpu><processes><process_info><type>G</type></process_info></processes></gpu></nvidia_smi_log>")
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **kw: result)
    assert len(audit.gpu_idle()) == 1
    result.stdout = result.stdout.replace("<type>G</type>", "<type>C+G</type>")
    with pytest.raises(ValueError, match="compute"):
        audit.gpu_idle()
