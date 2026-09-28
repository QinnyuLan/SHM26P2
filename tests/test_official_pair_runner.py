import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "official_pair_runner", Path(__file__).parents[1] / "scripts/run_official_pair.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def make_plan(root):
    bundle = root / runner.BUNDLE
    plan = {"workspace_root": str(root), "source_snapshot": str(bundle / "source_snapshot"),
            "inputs": [{"arm": arm, "output": str(bundle / arm), "checkpoint": str(root / folder / "last.pt"),
                        "training_receipt": str(root / folder / "experiment_receipt.json")}
                       for arm, folder in [("legacy_support", "runs/support_split_semantic_coupled"),
                                           ("corner_v2", "runs/corner_v2_semantic_coupled")]],
            "comparison_output": str(bundle / "comparison.json")}
    write(bundle / "execution_plan.json", plan)
    return plan


def complete_receipt():
    return {"status": "completed", "finished_utc": "2026-09-26T20:00:00+00:00",
            "checkpoint_sha256": "checkpoint", "evaluation_sha256": "metrics"}


def test_running_stage_rejected_before_any_checkpoint_hash_or_gpu_query(tmp_path, monkeypatch):
    plan = make_plan(tmp_path)
    write(Path(plan["inputs"][0]["training_receipt"]), complete_receipt())
    write(Path(plan["inputs"][1]["training_receipt"]), {"status": "running"})

    def forbidden(*args, **kwargs):
        pytest.fail("Checkpoint hashing/GPU queries must not precede completion metadata")

    monkeypatch.setattr(runner, "digest", forbidden)
    monkeypatch.setattr(runner, "check_gpu_idle", forbidden)
    with pytest.raises(ValueError, match="naturally completed"):
        runner.run_pair(tmp_path, 0, "synthetic CPU test")
    receipt = runner.read_json(tmp_path / runner.BUNDLE / "pair_execution_receipt.json")
    assert receipt["status"] == "failed" and receipt["commands"] == []
    assert not Path(plan["inputs"][0]["output"]).exists()
    with pytest.raises(FileExistsError):
        runner.run_pair(tmp_path, 0, "must preserve failed attempt")


def test_both_v2_stage_audits_required_and_semantic_freeze_contract(tmp_path):
    plan = make_plan(tmp_path)
    for arm in plan["inputs"]:
        write(Path(arm["training_receipt"]), complete_receipt())
    write(tmp_path / "runs/corner_v2_rgb_full/experiment_receipt.json", complete_receipt())
    for stage, folder, step in [("rgb", "corner_v2_rgb_full", 30000), ("semantic", "corner_v2_semantic_coupled", 8000)]:
        write(tmp_path / "runs" / folder / "stage_audit.json",
              {"status": "passed", "stage": stage, "step": step,
               "semantic_rgb_all50_png_and_metrics_exact_to_source": stage == "semantic",
               "semantic_frozen_rgb_geometry_keys": ["splats.means"] if stage == "semantic" else None})
    assert len(runner.completed_endpoints(tmp_path, plan)) == 5
    path = tmp_path / "runs/corner_v2_semantic_coupled/stage_audit.json"
    audit = runner.read_json(path)
    audit["semantic_rgb_all50_png_and_metrics_exact_to_source"] = False
    write(path, audit)
    with pytest.raises(ValueError, match="frozen-geometry"):
        runner.completed_endpoints(tmp_path, plan)


@pytest.mark.parametrize("kind", ["C", "C+G", "", "unavailable"])
def test_gpu_exclusivity_rejects_compute_or_unknown_listing(monkeypatch, kind):
    process = ("N/A" if kind == "unavailable" else
               f"<process_info><pid>123</pid><type>{kind}</type><process_name>job</process_name></process_info>")
    xml = f"<nvidia_smi_log><gpu><processes>{process}</processes></gpu></nvidia_smi_log>"
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=xml))
    with pytest.raises(ValueError, match="compute or unclassified|unavailable"):
        runner.check_gpu_idle()


def test_graphics_only_desktop_clients_are_recorded_without_blocking_compute_handoff(monkeypatch):
    xml = ("<nvidia_smi_log><gpu><processes><process_info><pid>123</pid><type>G</type>"
           "<process_name>Xorg</process_name></process_info></processes></gpu></nvidia_smi_log>")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=xml))
    result = runner.check_gpu_idle()
    assert result["compute_process_count"] == 0 and result["process_count"] == 1
    assert result["graphics_only_processes"] == [{"pid": "123", "name": "Xorg", "type": "G"}]


def test_idle_gpu_and_disk_reservation_math(monkeypatch, tmp_path):
    xml = "<nvidia_smi_log><gpu><uuid>GPU-test</uuid><product_name>fake</product_name><processes/></gpu></nvidia_smi_log>"
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=xml))
    assert runner.check_gpu_idle()["process_count"] == 0
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=2 << 30))
    assert runner.check_space(tmp_path, 1 << 30)["available_after_reservations"] == 1 << 30
    with pytest.raises(ValueError, match="Insufficient"):
        runner.check_space(tmp_path, (1 << 30) + 1)
    assert runner.SECOND_ARM_MINIMUM == 556 << 20
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=400 << 20))
    with pytest.raises(ValueError, match="Insufficient"):
        runner.check_space(tmp_path, 0, minimum=runner.SECOND_ARM_MINIMUM)


def test_snapshot_entrypoint_and_source_bytes_are_bound(tmp_path):
    snapshot = tmp_path / "source_snapshot"
    file = snapshot / "bridge_rgs/a.py"
    file.parent.mkdir(parents=True)
    file.write_text("x = 1\n")
    entry = tmp_path / "scripts/evaluate_official.py"
    entry.parent.mkdir()
    entry.write_text("pass\n")
    hashes = {"bridge_rgs/a.py": runner.digest(file)}
    tree = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt = write(tmp_path / "source_snapshot_receipt.json", {
        "source_hashes": hashes, "source_tree_sha256": tree,
        "runner_hashes": {"scripts/evaluate_official.py": runner.digest(entry)}, "test_hashes": {}})
    plan = {"source_snapshot": str(snapshot), "source_snapshot_receipt": str(receipt),
            "source_snapshot_receipt_sha256": runner.digest(receipt), "source_tree_sha256": tree}
    registry = {}
    runner.check_sources(plan, registry)
    assert str(entry) in registry and str(file) in registry
    entry.write_text("changed\n")
    with pytest.raises(ValueError, match="SHA mismatch"):
        runner.check_sources(plan, {})


def test_nonempty_evaluation_outputs_are_never_overwritten(tmp_path):
    plan = make_plan(tmp_path)
    runner.require_empty_outputs(plan)
    existing = Path(plan["inputs"][0]["output"]) / "execution_receipt.json"
    write(existing, {"status": "failed"})
    before = existing.read_bytes()
    with pytest.raises(FileExistsError):
        runner.require_empty_outputs(plan)
    assert existing.read_bytes() == before


def test_post_run_byte_binding_detects_input_mutation(tmp_path):
    file = tmp_path / "model.pt"
    file.write_bytes(b"immutable checkpoint fixture")
    registry = {}
    original = runner.bind(registry, file)
    file.write_bytes(b"changed after renderer")
    with pytest.raises(ValueError, match="SHA mismatch"):
        runner.bind(registry, file, original)


def test_lpips_requires_both_existing_local_weights_and_hashes_them(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
    package = tmp_path / "lpips"
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(package / "__init__.py")))
    with pytest.raises(ValueError, match="local LPIPS weight missing"):
        runner.lpips_weights({})
    paths = [tmp_path / "hub/checkpoints/alexnet-owt-7be5be79.pth", package / "weights/v0.1/alex.pth"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    result = runner.lpips_weights({})
    assert {x["path"] for x in result.values()} == {str(p) for p in paths}
    assert all(x["sha256"] == runner.digest(x["path"]) for x in result.values())


def output_fixture(root):
    import numpy as np
    from PIL import Image

    from bridge_rgs import official_evaluate as official
    spec = importlib.util.spec_from_file_location(
        "pair_comparator_fixture", Path(__file__).parents[1] / "scripts/compare_official_evaluations.py")
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    bundle, arm = root / "pair", "legacy_support"
    output = bundle / arm
    (output / "rgb").mkdir(parents=True)
    (output / "mask").mkdir()
    reference = {"cameras": [], "targets": []}
    predictions, records, views = [], [], []
    for i in range(50):
        name = f"{i:03d}.png"
        camera = {"name": name, "width": 11, "height": 11}
        image = root / "original" / name
        image.parent.mkdir(exist_ok=True)
        image.write_bytes(b"source RGB bytes used only after completed predictions")
        annotation = write(image.with_suffix(".json"), {"shapes": []}) if i < 41 else None
        target = {"name": name, "source_image_path": str(image),
                  "source_annotation_path": str(annotation) if annotation else None}
        reference["cameras"].append(camera)
        reference["targets"].append(target)
        prediction = {"name": name}
        for kind, channels in (("rgb", 3), ("mask", None)):
            path = output / kind / name
            Image.fromarray(np.zeros((11, 11, channels) if channels else (11, 11), dtype=np.uint8)).save(path)
            prediction.update({kind: str(path), kind + "_sha256": runner.digest(path)})
        predictions.append(prediction)
        records.append(dict(target, camera=camera, source_image_sha256=runner.digest(image),
                            source_annotation_sha256=runner.digest(annotation) if annotation else None,
                            rasterized_mask_sha256="synthetic_mask_hash" if annotation else None))
        view = dict(camera, rgb_pixels=121, psnr=20., ssim=.8, lpips=.2)
        if annotation:
            view.update(confusion_matrix=np.diag([29, 23, 23, 23, 23]).tolist(),
                        semantic_pixels=121, semantic_ignore_pixels=0)
        views.append(view)
    fields = ("camera", "source_image_sha256", "source_annotation_sha256", "rasterized_mask_sha256")
    fingerprint = official.official_fingerprint([{key: r[key] for key in fields} for r in records])
    metrics = {"evaluation_family": official.FAMILY, "official_evaluation_fingerprint": fingerprint,
               "scoring_protocol": official.SCORING_PROTOCOL, "inference_protocol": official.PLAIN_INFERENCE,
               "validation_views": 50, "semantic_validation_views": 41, "views": views,
               "psnr": 20., "ssim": .8, "lpips": .2, "iou": [1.] * 5, "miou_all": 1., "miou_foreground": 1.,
               "confusion_matrix": (np.diag([29, 23, 23, 23, 23]) * 41).tolist()}
    metric_path = write(output / "official_metrics.json", metrics)
    frozen = {"source_hashes": {f"bridge_rgs/{name}.py": f"hash-{name}" for name in runner.CORE},
              "runner_hashes": {"scripts/evaluate_official.py": "entrypoint-hash"}}
    receipt = {"status": "completed", "evaluation_family": official.FAMILY,
               "official_evaluation_fingerprint": fingerprint, "official_metrics_sha256": runner.digest(metric_path),
               "scoring_protocol": official.SCORING_PROTOCOL, "inference_protocol": official.PLAIN_INFERENCE,
               "checkpoint": "/synthetic/model.pt", "checkpoint_sha256": "model-sha",
               "checkpoint_pixel_protocol": {"id": "legacy_mixed_v1"}, "training_manifest": {},
               "entrypoint_source": {"path": str(bundle / "scripts/evaluate_official.py"), "sha256": "entrypoint-hash"},
               "loaded_source_modules": {f"bridge_rgs.{name}": {
                   "path": str(bundle / "source_snapshot/bridge_rgs" / f"{name}.py"), "sha256": f"hash-{name}"}
                   for name in runner.CORE}, "predictions_finished_utc": "2026-09-26T20:00:00+00:00",
               "source_scoring_started_utc": "2026-09-26T20:00:01+00:00", "predictions": predictions,
               "source_records": records, "runtime_versions": {"torch": "synthetic"}}
    path = write(output / "execution_receipt.json", receipt)
    arm = {"output": str(output), "actual_checkpoint_sha256": "model-sha", "pixel_protocol": "legacy_mixed_v1"}
    return arm, reference, frozen, official, comparison, receipt, path


def test_completed_output_audit_checks_all_pngs_and_binds_original_bytes(tmp_path):
    arm, reference, frozen, official, comparison, _, _ = output_fixture(tmp_path)
    registry = {}
    result = runner.audit_output(arm, reference, frozen, official, comparison, registry)
    assert result["png_files_checked"] == 100
    assert len(registry) == 93  # 50 original images, 41 annotations, metrics, receipt.


@pytest.mark.parametrize("mutation", ["png", "source_module", "metric_hash", "camera", "gt_before_prediction"])
def test_output_audit_rejects_broken_provenance_or_delivery(tmp_path, mutation):
    arm, reference, frozen, official, comparison, receipt, path = output_fixture(tmp_path)
    if mutation == "png":
        Path(receipt["predictions"][0]["rgb"]).write_bytes(b"modified PNG")
    elif mutation == "source_module":
        receipt["loaded_source_modules"]["bridge_rgs.model"]["path"] = "/unfrozen/model.py"
    elif mutation == "metric_hash":
        receipt["official_metrics_sha256"] = "changed"
    elif mutation == "camera":
        receipt["source_records"][0]["camera"] = dict(receipt["source_records"][0]["camera"], width=12)
    else:
        receipt["source_scoring_started_utc"] = "2026-09-26T19:59:59+00:00"
    write(path, receipt)
    with pytest.raises(ValueError):
        runner.audit_output(arm, reference, frozen, official, comparison, {})
