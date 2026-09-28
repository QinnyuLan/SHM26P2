"""Execute preparation → DINOv3 → soft labels → 3D training and export.

The Gaussian config supplies manifest, pseudo and output paths. --all-data
creates separate *_all artifacts/runs and disables held-out scoring explicitly.
Existing stage receipts/checkpoints are used only with explicit --resume.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    command = [sys.executable, *map(str, args)]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _config_path(path):
    path = _absolute(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _all_directory(path):
    path = Path(path)
    return path if path.name.endswith("_all") else path.with_name(path.name + "_all")


def pipeline_configuration(config, all_data=False):
    """Derive isolated final-fit paths without mutating the caller's config."""
    config = dict(config)
    if all_data:
        manifest = Path(config["manifest"])
        config["manifest"] = str(_all_directory(manifest.parent) / manifest.name)
        config["output"] = str(_all_directory(config["output"]))
        if config.get("pseudo_dir"):
            config["pseudo_dir"] = str(_all_directory(config["pseudo_dir"]))
        if config.get("init_points"):
            config["init_points"] = str(Path(config["manifest"]).with_name("init_points.npz"))
        config["teacher_output"] = str(_all_directory(config.get("teacher_output", "runs/teacher")))
        config["eval_every"] = 0
        config["evaluation_protocol"] = "all400_rgb_all300_semantic_final_fit_no_validation"
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--all-data", action="store_true", help="Separate all400/300 final fit; no validation score")
    parser.add_argument("--dataset", default="Dataset")
    parser.add_argument("--teacher-steps", type=int, default=6000)
    parser.add_argument("--gaussian-config", default="configs/bridge_rgs.yaml")
    args = parser.parse_args()
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("MAX_JOBS", "4")
    from bridge_rgs.coordinates import LEGACY, pixel_protocol, require_matching_protocol
    from bridge_rgs.io import read_config
    config = pipeline_configuration(read_config(_absolute(args.gaussian_config)), args.all_data)
    # Missing configuration metadata keeps historical pipeline reproduction explicit.
    # New recipes opt into the corrected protocol and separate artifact paths.
    requested_pixel_protocol = pixel_protocol(config.get("pixel_protocol", LEGACY))
    prepared = _absolute(config["manifest"])
    expected_val_every = 0 if args.all_data else int(config.get("val_every", 8))
    if args.resume and prepared.exists():
        previous = json.loads(prepared.read_text())
        if previous["split"]["val_every"] != expected_val_every:
            raise ValueError("Prepared split differs from requested protocol; use a separate config/output")
        require_matching_protocol(previous, requested_pixel_protocol, "pipeline resume")
    else:
        run("-m", "bridge_rgs.cli", "prepare", "--dataset", args.dataset,
            "--output", prepared.parent,
            "--max-width", str(config.get("prepare_max_width", 1320)),
            "--max-points", str(config.get("prepare_max_points", 60000)),
            "--seed", str(config.get("seed", 42)), "--val-every", str(expected_val_every),
            "--pixel-protocol", requested_pixel_protocol)
    manifest = json.loads(prepared.read_text())
    has_validation = any(v["split"] == "val" for v in manifest["views"])
    if not has_validation:
        config["eval_every"] = 0

    if config.get("pseudo_dir"):
        model = _absolute(config.get("dinov3_model_dir", "models/dinov3-vith16plus"))
        if not (args.resume and (model / "download_provenance.json").exists()):
            run("scripts/download_dinov3.py", "--output", model)
        teacher = _absolute(config.get("teacher_output", "runs/teacher"))
        teacher_command = ["-m", "bridge_rgs.teacher", "train", "--manifest", prepared,
                           "--model-dir", model, "--output", teacher, "--steps", str(args.teacher_steps)]
        if args.resume and (teacher / "last.pt").exists():
            import torch

            from bridge_rgs.teacher import file_sha256
            previous = torch.load(teacher / "last.pt", map_location="cpu", weights_only=False)
            if previous["provenance"]["manifest_sha256"] != file_sha256(prepared):
                raise ValueError("Teacher checkpoint was trained on a different prepared manifest")
            if previous["step"] < args.teacher_steps:
                run(*teacher_command, "--resume", teacher / "last.pt")
            elif previous["step"] > args.teacher_steps:
                raise ValueError("Existing teacher exceeds requested steps; use a separate output for a shorter run")
        else:
            run(*teacher_command)
        teacher_checkpoint = teacher / "best.pt" if has_validation and (teacher / "best.pt").exists() else teacher / "last.pt"
        pseudo = _absolute(config["pseudo_dir"])
        valid_pseudo = False
        if args.resume and (pseudo / "provenance.json").exists():
            provenance = json.loads((pseudo / "provenance.json").read_text())
            from bridge_rgs.teacher import file_sha256, verify_pseudo_provenance
            if provenance.get("complete") and provenance.get("checkpoint_sha256") == file_sha256(teacher_checkpoint):
                verify_pseudo_provenance(prepared, pseudo)
                valid_pseudo = True
        if not valid_pseudo:
            run("-m", "bridge_rgs.teacher", "predict", "--manifest", prepared,
                "--checkpoint", teacher_checkpoint, "--model-dir", model, "--output", pseudo)

    output = _absolute(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    import yaml
    effective = output / "pipeline_config.yaml"
    effective.write_text(yaml.safe_dump(config, sort_keys=False))
    gaussian_last = output / "last.pt"
    command = ["-m", "bridge_rgs.cli", "train", "--config", _config_path(effective)]
    if args.resume and gaussian_last.exists():
        command.extend(["--resume", gaussian_last])
    run(*command)
    if has_validation:
        run("-m", "bridge_rgs.cli", "evaluate", "--checkpoint", gaussian_last,
            "--manifest", prepared, "--output", output / "evaluation", "--lpips")
    else:
        (output / "final_fit_protocol.json").write_text(json.dumps({
            "protocol": "all-view fitting; validation intentionally absent",
            "rgb_training_views": len(manifest["views"]),
            "semantic_training_views": sum(bool(v.get("mask_path")) for v in manifest["views"]),
            "reported_validation_score": None}, indent=2))
    run("-m", "bridge_rgs.cli", "export-ply", "--checkpoint", gaussian_last,
        "--output", output / "bridge_semantic.ply")


if __name__ == "__main__":
    main()
