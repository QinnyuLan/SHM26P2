"""Generate matched H2 semantic-continuation configs; never launch training."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shlex
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from bridge_rgs.refinement import normalize_refiner_config

ROOT = Path(__file__).resolve().parents[1]


def workspace_path(path: str | Path) -> Path:
    path = Path(path)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def config_path(path: str | Path) -> str:
    resolved = workspace_path(path)
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)


def inherited_config(checkpoint: dict, base: dict | None = None) -> dict:
    """Retain the selected checkpoint's routing, schema and split explicitly."""
    saved = checkpoint.get("config")
    if not isinstance(saved, dict) or "manifest" not in saved:
        raise ValueError("Selected checkpoint must contain its training config and manifest")
    config = copy.deepcopy(saved)
    if base is not None:
        if not isinstance(base, dict):
            raise TypeError("Base configuration must be a mapping")
        config.update(copy.deepcopy(base))
    saved_route = saved.get("refiner_field_grad", True)
    route = config.get("refiner_field_grad", saved_route)
    if type(route) is not bool or route != saved_route:
        raise ValueError("H2 cannot change the selected checkpoint's refiner_field_grad routing")
    if workspace_path(config["manifest"]) != workspace_path(saved["manifest"]):
        raise ValueError("H2 base config and selected checkpoint must use the same manifest")
    schema = normalize_refiner_config(checkpoint.get("refiner_config", saved.get("refiner")))
    if normalize_refiner_config(config.get("refiner", schema)) != schema:
        raise ValueError("H2 cannot replace the selected checkpoint's refiner architecture")
    if config.get("semantic_weight", .2) <= 0:
        raise ValueError("Select a trained semantic stage with positive semantic_weight")
    config["refiner"] = schema
    config["refiner_field_grad"] = route
    return config


def make_variants(base: dict, warmstart: str, pseudo_dir: str, runs_root: str,
                  *, include_gt: bool = True, fusion_weight: float = .05,
                  pseudo_threshold: float = .8) -> dict[str, dict]:
    """Change only the declared continuation controls; preserve the trained head."""
    if not math.isfinite(fusion_weight) or fusion_weight <= 0:
        raise ValueError("fusion_weight must be finite and positive")
    if not math.isfinite(pseudo_threshold) or not 0 <= pseudo_threshold <= 1:
        raise ValueError("pseudo_threshold must lie in [0,1]")
    common = copy.deepcopy(base)
    common.update(
        warmstart=warmstart, warmstart_reset_refiner=False,
        steps=3000, train_scale=1.0, progressive_resolution=False,
        train_labeled_only=False, view_sampling="shuffle", independent_view_rng=True,
        view_sampler_seed=base.get("view_sampler_seed", base.get("seed", 42)),
        semantic_start=1, refine_start=1, freeze_geometry=True, freeze_rgb=True,
        optimize_cameras=False, densification="none", semantic_geometry_weight=0.,
        pseudo_dir=pseudo_dir, pseudo_weight=0., pseudo_refiner_weight=.1,
        pseudo_threshold=pseudo_threshold, pseudo_class_reweight_cap=3.,
        multiview_fusion=False, multiview_weight=0., projection_uncertainty=False,
        fusion_every=200, fusion_points=4096,
        local_teacher_kd=True, supervised_teacher_priority=True,
        log_every=100, save_every=1500, eval_every=1500, eval_scale=1.,
    )
    common.pop("eval_max_views", None)  # All held-out views, regardless of pilot settings.
    variants = {}
    if include_gt:
        variants["00_gt_continuation"] = {"pseudo_dir": None, "pseudo_refiner_weight": 0.}
    variants.update({
        "01_teacher_no_fusion": {},
        "02_teacher_fixed_sampling": {"multiview_fusion": True,
                                      "multiview_weight": fusion_weight},
        "03_teacher_projection_sampling": {"multiview_fusion": True,
                                           "multiview_weight": fusion_weight,
                                           "projection_uncertainty": True},
    })
    return {name: {**copy.deepcopy(common), **overrides, "output": str(Path(runs_root) / name)}
            for name, overrides in variants.items()}


def continuation_budget(manifest: dict, seed: int, steps: int = 3000) -> dict:
    """Predict the actual labeled exposure of the existing independent sampler."""
    if type(seed) is not int or seed < 0:
        raise ValueError("H2 requires an explicit nonnegative integer view sampler seed")
    views = [view for view in manifest["views"] if view["split"] == "train"]
    if len(views) != 350:
        raise ValueError("This bridge H2 protocol requires exactly 350 training views")
    names = [str(view["name"]) for view in views]
    if len(set(names)) != len(names):
        raise ValueError("Training view names must be unique")
    generator = np.random.default_rng(seed)
    sequence = []
    while len(sequence) < steps:
        sequence.extend(generator.permutation(len(views)).tolist())
    sequence = sequence[:steps]
    exposure = Counter(sequence)
    labeled_steps = sum(bool(views[index].get("mask_path")) for index in sequence)
    return {
        "steps_per_arm": steps, "train_views": len(views),
        "labeled_views": sum(bool(view.get("mask_path")) for view in views),
        "labeled_steps": labeled_steps, "unlabeled_steps": steps - labeled_steps,
        "min_view_visits": min(exposure.values()), "max_view_visits": max(exposure.values()),
        "view_sampler_seed": seed,
        "fusion_refreshes_in_fusion_arms": steps // 200,
        "fusion_targets_used_by_loss": (steps - 1) // 200,
        "final_refresh_used_by_training": False,
        "acceptance_log_steps": list(range(300, steps, 200)),
    }


def generate(warmstart: str | Path, pseudo_dir: str | Path, output_dir: str | Path,
             runs_root: str | Path, *, base_config: str | Path | None = None,
             include_gt: bool = True, fusion_weight: float = .05,
             pseudo_threshold: float = .8, allow_pending_pseudo: bool = False) -> dict:
    """Read metadata and write new YAMLs/plan only; never alter selected inputs."""
    warmstart_path = workspace_path(warmstart)
    if not warmstart_path.is_file():
        raise FileNotFoundError(warmstart_path)
    # mmap avoids materializing large optimizer tensors; all storages stay on CPU.
    checkpoint = torch.load(warmstart_path, map_location="cpu", mmap=True, weights_only=False)
    explicit = None if base_config is None else yaml.safe_load(workspace_path(base_config).read_text())
    base = inherited_config(checkpoint, explicit)
    manifest_path = workspace_path(base["manifest"])
    manifest = json.loads(manifest_path.read_text())
    sampler_seed = base.get("view_sampler_seed", base.get("seed", 42))
    budget = continuation_budget(manifest, sampler_seed)
    pseudo_path = workspace_path(pseudo_dir)
    status = "pending_explicitly_allowed"
    if (pseudo_path / "provenance.json").is_file():
        from bridge_rgs.teacher import verify_pseudo_provenance
        provenance = verify_pseudo_provenance(manifest_path, pseudo_path, verify_files=False)
        expected = {str(view["name"]) for view in manifest["views"] if view["split"] == "train"}
        if {str(view["name"]) for view in provenance["views"]} != expected:
            raise ValueError("H2 pseudo export must include every one of the 350 training views")
        status = "metadata_verified_full_file_verification_deferred_to_training"
    elif not allow_pending_pseudo:
        raise FileNotFoundError("Completed pseudo provenance is missing; use --allow-pending-pseudo "
                                "only to prepare configs before export")
    variants = make_variants(base, config_path(warmstart), config_path(pseudo_dir),
                             config_path(runs_root), include_gt=include_gt,
                             fusion_weight=fusion_weight, pseudo_threshold=pseudo_threshold)
    destination = workspace_path(output_dir)
    paths = {name: destination / f"{name}.yaml" for name in variants}
    plan_path = destination / "h2_plan.json"
    occupied = [path for path in [*paths.values(), plan_path] if path.exists()]
    if occupied:
        raise FileExistsError(f"Refusing to overwrite generated H2 files: {occupied}")
    for config in variants.values():
        if workspace_path(config["output"]).exists():
            raise FileExistsError(f"Experiment output already exists: {config['output']}")
    first_name = next(iter(variants))
    snapshot = str(Path(variants[first_name]["output"]) / "source_snapshot")
    commands = {}
    for name, path in paths.items():
        command = ["uv", "run", "python", "scripts/run_experiment.py", config_path(path)]
        if name != first_name:
            command += ["--source-snapshot", snapshot]
        commands[name] = command
    plan = {
        "kind": "h2_configuration_plan_not_execution_receipt",
        "warmstart": config_path(warmstart),
        "base_config": None if base_config is None else config_path(base_config),
        "pseudo_dir": config_path(pseudo_dir), "pseudo_status": status,
        "inherited_refiner_field_grad": base["refiner_field_grad"],
        "refiner": base["refiner"], "preserve_existing_refiner": True,
        "budget": budget, "configs": {name: config_path(path) for name, path in paths.items()},
        "run_commands": commands,
        "provenance_owner": "scripts/run_experiment.py and verified teacher export provenance",
        "threshold_policy": "Lock thresholds using training-only diagnostics before scoring arms",
        "training_started": False,
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        path.write_text(yaml.safe_dump(variants[name], sort_keys=False))
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmstart", type=Path, required=True,
                        help="Selected semantic checkpoint, read only; preserves its routing/head")
    parser.add_argument("--pseudo-dir", type=Path, required=True,
                        help="One fixed train-only export shared by all teacher arms")
    parser.add_argument("--output-dir", type=Path, required=True, help="New YAML/plan directory")
    parser.add_argument("--runs-root", type=Path, required=True, help="New experiment output root")
    parser.add_argument("--base-config", type=Path,
                        help="Optional inherited settings; routing/schema/manifest must match checkpoint")
    parser.add_argument("--without-gt-control", action="store_true", help="Write three arms instead of four")
    parser.add_argument("--fusion-weight", type=float, default=.05)
    parser.add_argument("--pseudo-threshold", type=float, default=.8,
                        help="Predeclare from training diagnostics, never tune on validation")
    parser.add_argument("--allow-pending-pseudo", action="store_true",
                        help="Permit a future export path; training still requires full verification")
    args = parser.parse_args()
    plan = generate(args.warmstart, args.pseudo_dir, args.output_dir, args.runs_root,
                    base_config=args.base_config, include_gt=not args.without_gt_control,
                    fusion_weight=args.fusion_weight, pseudo_threshold=args.pseudo_threshold,
                    allow_pending_pseudo=args.allow_pending_pseudo)
    print(json.dumps({key: plan[key] for key in ("configs", "budget", "training_started")}, indent=2))
    for command in plan["run_commands"].values():
        print("NOT RUN:", shlex.join(command))


if __name__ == "__main__":
    main()
