"""Prepare CPU-only, non-launching H+/7B capacity drafts from fixed old protocols."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from bridge_rgs.coordinates import pixel_protocol
from bridge_rgs.teacher import TeacherConfig, verify_teacher_render_protocol
from bridge_rgs.teacher_domains import domain_schedule, validate_image_sources

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/mnt/data/SHM2026/runs/teacher_capacity_v1")
DRAFT = ROOT / "configs/generated_teacher_capacity_v1"
FINGERPRINT = "21a2f19c5d4d703403a0107402d5dd5e006dcd98e5f8e023820c388c3e69a85d"
REFERENCE = Path("/mnt/data/SHM2026/runs/official_selected_ensemble_v1/cross_teacher")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def make_configs(root: Path = ROOT, output: Path = OUTPUT) -> dict:
    """Four explicit effective configs; the two capacities differ only in identity."""
    templates = {
        "real": json.loads((root / "configs/teacher_strong_v1.json").read_text()),
        "render_adapt": json.loads((root / "configs/teacher_render_adapt_v1.json").read_text()),
    }
    models = {"hplus": root / "models/dinov3-vith16plus",
              "vit7b": Path("/mnt/data/SHM2026/models/dinov3-vit7b16")}
    result = {}
    for capacity, model_dir in models.items():
        for stage, template in templates.items():
            config = asdict(TeacherConfig(**template["config"]))
            config.update(independent_augmentation_rng=True, checkpoint_every=1000,
                          eval_every=config["steps"], adapter_rank=0)
            config["warmstart_checkpoint"] = (
                str(output / capacity / "real/last.pt") if stage == "render_adapt" else ""
            )
            result[f"{capacity}_{stage}"] = {
                "manifest": str((root / template["manifest"]).resolve()),
                "model_dir": str(model_dir.resolve()),
                "output_dir": str((output / capacity / stage).resolve()),
                "config": config,
            }
    return result


def adoption_gate(pair_7b_minus_hplus: dict, pair_7b_minus_selected: dict,
                  rgb_png_exact: bool) -> dict:
    """Pairs use fractions, not percentage points; no gate-driven retraining."""
    checks = {"all_50_rgb_png_exact": bool(rgb_png_exact)}
    for name, pair in (("matched_hplus", pair_7b_minus_hplus),
                       ("selected_legacy", pair_7b_minus_selected)):
        metrics = pair["metrics"]
        all5 = metrics["miou_all"]
        checks[f"{name}_all5_gain_at_least_0.30pp"] = all5["difference"] >= .003
        checks[f"{name}_all5_ci_lower_positive"] = all5["paired_view_bootstrap_95_interval"][0] > 0
        checks[f"{name}_foreground_non_decreasing"] = metrics["miou_foreground"]["difference"] >= 0
        checks[f"{name}_cable_non_decreasing"] = metrics["stay_cable_iou"]["difference"] >= 0
    return {"passed": all(checks.values()), "checks": checks}


def bind(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "size": path.stat().st_size}


def main() -> None:
    if DRAFT.exists():
        raise FileExistsError(f"Refusing to overwrite reviewed drafts: {DRAFT}")
    configs = make_configs()
    original_path = ROOT / "artifacts/prepared/manifest.json"
    derived_path = ROOT / "runs/teacher_render_adapt_v1/manifest.json"
    original, derived = (json.loads(p.read_text()) for p in (original_path, derived_path))
    assert pixel_protocol(original) == pixel_protocol(derived) == "legacy_mixed_v1"
    protocol = validate_image_sources(derived, verify_files=True)
    verify_teacher_render_protocol(derived, protocol)
    train = [v for v in original["views"] if v["split"] == "train"]
    val = [v for v in original["views"] if v["split"] == "val"]
    assert (len(train), len(val), sum(bool(v.get("mask_path")) for v in train),
            sum(bool(v.get("mask_path")) for v in val)) == (350, 50, 259, 41)
    train_support_paths = sorted({Path(v[key]) for v in train for key in ("mask_path", "valid_path")
                                  if v.get(key)})
    train_support = [bind(path) for path in train_support_paths]
    selected_metrics_path = REFERENCE / "official_metrics.json"
    selected_receipt_path = REFERENCE / "execution_receipt.json"
    metrics = json.loads(selected_metrics_path.read_text())
    receipt = json.loads(selected_receipt_path.read_text())
    assert receipt["status"] == "completed"
    assert receipt["official_metrics_sha256"] == sha256(selected_metrics_path)
    assert metrics["official_evaluation_fingerprint"] == FINGERPRINT
    scene = Path(receipt["checkpoint"])
    assert sha256(scene) == receipt["checkpoint_sha256"]
    backbones = {}
    for capacity in ("hplus", "vit7b"):
        model_dir = Path(configs[f"{capacity}_real"]["model_dir"])
        download_path = model_dir / "download_provenance.json"
        download = json.loads(download_path.read_text())
        if capacity == "vit7b":
            assert download["status"] == "completed"
            assert download["revision"] == "5251e00b307184bb247d076713375235d554fefb"
        for name in download["files_sha256"]:
            assert (model_dir / name).is_file()
        backbones[capacity] = {
            "model_dir": str(model_dir), "download_receipt": bind(download_path),
            "model_id": download["model_id"], "revision": download["revision"],
            "download_verified_file_sha256": download["files_sha256"],
            "verification_scope": "Download completed provenance bound here; fresh full-file SHA required again at preflight/launch",
            "config": bind(model_dir / "config.json"),
        }
    source_files = sorted((ROOT / "src/bridge_rgs").glob("*.py"))
    source_review = {str(p.relative_to(ROOT)): sha256(p) for p in source_files}
    domain = domain_schedule(2000, .5, 20260926)
    preflight_root = Path("/mnt/data/SHM2026/preflight/dinov3_capacity_v1")
    preflight = {}
    for capacity, directory in (("hplus", "hplus"), ("vit7b", "7b")):
        path = preflight_root / directory / "worker_receipt.json"
        value = json.loads(path.read_text())
        assert value["status"] == "completed" and value["backbone_tensors_unchanged"]
        preflight[capacity] = {"receipt": bind(path), "steps": value["steps"],
                               "full_protocol_seconds": value["inference"]["seconds_complete_protocol"],
                               "limits": value["specification"]["limits"]}
    plan = {
        "status": "reviewable_draft_no_training_authorized_by_this_file",
        "created_utc": datetime.now(UTC).isoformat(), "workspace_root": str(ROOT),
        "output_root": str(OUTPUT), "generator": bind(Path(__file__)),
        "source_review_sha256": source_review, "source_snapshot": None,
        "source_policy": "Freeze one complete reviewed package/runner for both capacities after actual preflight, before any formal stage",
        "actual_capacity_preflight": preflight,
        "configurations": {}, "inputs": {
            "real_manifest": bind(original_path), "render_adapt_manifest": bind(derived_path),
            "image_source_protocol": protocol, "image_source_live_sha_audit": "passed",
            "train_labels_and_validity": train_support, "backbones": backbones,
            "inference_scene": bind(scene), "selected_reference_metrics": bind(selected_metrics_path),
            "selected_reference_receipt": bind(selected_receipt_path),
        },
        "split": {"train_views": [v["name"] for v in train],
                  "train_labeled_views": [v["name"] for v in train if v.get("mask_path")],
                  "train_unlabeled_views": [v["name"] for v in train if not v.get("mask_path")],
                  "validation_views": [v["name"] for v in val],
                  "semantic_validation_views": [v["name"] for v in val if v.get("mask_path")]},
        "stage2_domain_schedule": {"real_steps": int((domain == 0).sum()),
                                   "rendered_steps": int(domain.sum()),
                                   "sha256": hashlib.sha256(domain.tobytes()).hexdigest(),
                                   "seed": 20260926},
        "checkpoint_selection": {
            "real_endpoint": "last.pt at exactly 6000; use its ema_decoder for stage2",
            "render_adapt_endpoint": "last.pt at exactly 2000; use its ema_decoder for final inference",
            "recovery": "rolling last every1000; fixed total step schedule on resume",
            "best_alias": "Current code may write best.pt only at the single terminal evaluation; never consume that filename or choose by its metric",
            "no_search": ["seed", "layer selection", "LoRA", "fusion weight", "validation step"],
        },
        "randomness": {
            "numpy": "Both capacities start each stage with same default_rng(seed); identical masks/grids and branch schedule control all view/crop/flip draws",
            "torch": "independent_augmentation_rng=True resets CPU/CUDA torch RNG to seed+1 after architecture construction/warmstart; no model stochastic layers or adapters",
            "head_initialization": "Fresh head for each capacity in stage1; shared seed does not imply identical differently-shaped tensors",
            "audit_required": ["same ordered supervised/unlabeled view/context/crop RNG-transition digest", "same image-domain schedule", "same final NumPy and torch/CUDA RNG states", "same photometric random draw sequence"],
            "audit_limit": "Current every100-step metrics alone are not proof of every training sample; formal runner must capture full-order digest",
        },
        "final_evaluation": {
            "family": "official_original_grid_v1", "expected_fingerprint": FINGERPRINT,
            "reference_manifest": str(original_path), "camera_count": 50, "semantic_count": 41,
            "scene": str(scene), "teacher_weight": .5, "student_weight": .5,
            "teacher_inference": {"tile_size": 768, "stride": 512, "flip": True,
                                  "context_weight": .25, "context_short_side": 768},
            "prediction": "Camera-only scene render -> quantized RGB teacher -> fixed .5 probabilities on pinhole overscan -> soft probability official warp -> argmax; source GT only after predictions",
            "output_paths": {capacity: str(OUTPUT / capacity / "official_final") for capacity in backbones},
            "paired_comparisons": ["vit7b minus matched_hplus", "vit7b minus selected_legacy", "matched_hplus minus selected_legacy"],
            "bootstrap_repeats": 5000, "bootstrap_seed": 20260926,
            "rgb_invariance": "All50 RGB PNG must be byte-identical across both capacities and selected same-scene reference",
            "selected_reference": {"miou_all": metrics["miou_all"],
                                   "miou_foreground": metrics["miou_foreground"],
                                   "stay_cable_iou": metrics["iou"][2]},
        },
        "engineering_adoption_gate": {
            "references": ["matched_hplus", "selected_legacy"],
            "miou_all_min_delta_fraction_each": .003,
            "miou_all_paired_95_lower_strictly_positive_each": True,
            "miou_foreground_point_non_decreasing_each": True,
            "stay_cable_point_non_decreasing_each": True,
            "all50_rgb_png_exact": True,
            "failure_policy": "Report all fixed outcomes; retain current selected reference; no weight/step/seed rescue search",
            "scope": "Engineering adoption only, not academic novelty or superiority to peer Dev30",
        },
        "budget_estimates_not_measurements": {
            "hplus_total_minutes": [35, 55], "vit7b_total_hours": [1.5, 3.5],
            "peak_planning_gib": {"hplus": [3, 5], "vit7b": [18, 22]},
            "new_storage_reserve_gib": 5,
            "checkpoints": "No backbone copies; four stages each rollinglast+terminalbest, plus atomic temp; roughly1GiB head states total, budget5GiB with two official PNG sets/logs/snapshots",
            "gpu_schedule": "Sequential exclusive per stage; actual preflight results supersede estimates",
        },
        "launch_requirements": ["Root review of these drafts", "Actual H+/7B crop/context/gradient/memory preflight passed", "One immutable source snapshot and entrypoint bound for allfour stages", "All dataset/model/scene inputs rechecked and output directories unused", "Full-order RNG audit available; fixed endpoint/receipt/natural exit checks", "No active GPU task conflict; free data-volume space>=5GiB"],
        "limits": ["One bridge, one fixed seed, repeatedly examined development validation", "Training rendered RGB from old strong scene differs from H3cross inference scene equally for both arms", "Matched H+ is freshly initialized under this fixed protocol, not a replay of historically selected H+ plus continuation/best selection", "Capacity changes feature width/depth and small DPT input-projection parameter count; gain is not a novel3D mechanism"],
    }
    DRAFT.mkdir(parents=True)
    for name, config in configs.items():
        path = DRAFT / f"{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        plan["configurations"][name] = bind(path)
    (DRAFT / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": plan["status"], "plan": str(DRAFT / "plan.json"),
                      "sha256": sha256(DRAFT / "plan.json"), "gpu_used": False}, indent=2))


if __name__ == "__main__":
    main()
