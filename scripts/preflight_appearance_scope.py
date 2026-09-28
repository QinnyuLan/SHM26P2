"""Run exactly two real RGB updates and prove all non-appearance state is unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/support15k_appearance_polish.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/appearance_scope_preflight"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a preflight")
    args.output.mkdir(parents=True)
    snapshot = args.output / "source_snapshot"
    shutil.copytree(Path(__file__).resolve().parents[1] / "src/bridge_rgs", snapshot / "bridge_rgs",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    sys.path.insert(0, str(snapshot.resolve()))
    from bridge_rgs import train as training

    config = yaml.safe_load(args.config.read_text())
    config.update(output=str(args.output), steps=2, log_every=1, save_every=2, eval_every=0)
    (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    initial = torch.load(config["warmstart"], map_location="cpu", mmap=True, weights_only=False)
    inputs = {"checkpoint": {"path": config["warmstart"], "sha256": digest(config["warmstart"])},
              "manifest": {"path": config["manifest"], "sha256": digest(config["manifest"])}}
    source_hashes = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))}
    allowed = {"splats.sh0", "splats.sh_rest", "background_logits"}
    report = {"inputs": inputs, "source_hashes": source_hashes,
              "script_sha256": digest(__file__), "gradients": {}, "status": "running"}
    original_apply = training.apply_parameter_scope
    original_degree = training.rendering_sh_degree

    def capture(gradient, name):
        record = {"norm": float(gradient.norm()), "nonzero": int(gradient.count_nonzero()),
                  "finite": bool(torch.isfinite(gradient).all())}
        report["gradients"].setdefault(name, []).append(record)
        if not record["finite"]:
            raise AssertionError(f"Non-finite gradient: {name}")

    def apply_and_record(model, options):
        original_apply(model, options)
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        if trainable != allowed:
            raise AssertionError(f"Unexpected trainable parameters: {trainable}")
        report["trainable"] = sorted(trainable)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.register_hook(lambda grad, name=name: capture(grad, name))

    def full_degree(step, maximum, options):
        degree = original_degree(step, maximum, options)
        if degree != 3 or maximum != 3:
            raise AssertionError("Real preflight requires full SH3 from step one")
        return degree

    training.apply_parameter_scope = apply_and_record
    training.rendering_sh_degree = full_degree
    try:
        checkpoint = training.train(config)
    finally:
        training.apply_parameter_scope = original_apply
        training.rendering_sh_degree = original_degree
    final = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if set(final["model"]) != set(initial["model"]):
        raise AssertionError("Model schema changed")
    changed = {key for key, value in final["model"].items()
               if not torch.equal(value, initial["model"][key])}
    if changed != allowed:
        raise AssertionError(f"Changed keys must be precisely SH/background: {changed}")
    if not torch.equal(final["training_cameras"], initial["training_cameras"]):
        raise AssertionError("Cameras changed")
    if any(final[key] != initial[key] for key in ("sh_degree", "feature_dim", "refiner_config", "scene_scale")):
        raise AssertionError("Architecture/scene metadata changed")
    if not final["config"]["freeze_geometry"] or final["config"]["freeze_rgb"]:
        raise AssertionError("Saved freeze flags contradict actual scope")
    if any(len(report["gradients"].get(name, [])) != 2 for name in allowed):
        raise AssertionError("Each appearance parameter must receive two RGB gradients")
    if any(final["optimizers"][name]["state"] for name in ("means", "quats", "log_scales", "opacity_logits", "sem_features")):
        raise AssertionError("Frozen parameters unexpectedly accumulated Adam state")
    for item in inputs.values():
        if digest(item["path"]) != item["sha256"]:
            raise AssertionError("Input changed during preflight")
    report.update(status="passed", changed_model_keys=sorted(changed),
                  protected_model_keys=sorted(set(final["model"]) - allowed),
                  cameras_bitwise_unchanged=True, architecture_metadata_unchanged=True,
                  saved_freeze_rgb=final["config"]["freeze_rgb"],
                  saved_freeze_geometry=final["config"]["freeze_geometry"],
                  sh_degree_from_step_one=3, step=final["step"],
                  checkpoint_sha256=digest(checkpoint),
                  sampling_population="all 350 TRAIN; independent seeded shuffle",
                  adam_lr={key: [g["lr"] for g in value["param_groups"]]
                           for key, value in final["optimizers"].items()})
    (args.output / "preflight_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "changed_model_keys": report["changed_model_keys"]}))


if __name__ == "__main__":
    main()
