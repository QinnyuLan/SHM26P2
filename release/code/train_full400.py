#!/usr/bin/env python3
"""Reproduce the three-stage full-view training recipe.

The script creates isolated paths, prepares all 400 views, and then runs the
RGB, semantic, and depth-feature cross-moment stages. A dry run is available
for submission review; use a fresh run directory for a clean reproduction.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RELEASE_CODE = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = RELEASE_CODE / "configs"
DEFAULT_DATASET = ROOT / "release_assets/SHM2026/dataset"


def command(args: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("$", " ".join(args), flush=True)
    if not dry_run:
        subprocess.run(args, cwd=ROOT, env=env, check=True)


def stage_config(template: str, manifest: Path, output: Path, overrides: dict) -> Path:
    config = yaml.safe_load((TEMPLATE_DIR / template).read_text())
    config.update({"manifest": str(manifest), "output": str(output), **overrides})
    output.mkdir(parents=True, exist_ok=True)
    path = output / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts/release_full400")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/release_full400")
    parser.add_argument("--max-points", type=int, default=60000)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    artifact_root, run_root = args.artifact_root.resolve(), args.run_root.resolve()
    manifest = artifact_root / "manifest.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(RELEASE_CODE), str(ROOT / "src"), env.get("PYTHONPATH", ""))
    )
    uv = str(ROOT / ".venv/bin/uv") if (ROOT / ".venv/bin/uv").is_file() else "uv"
    if not args.skip_prepare:
        command(
            [
                uv,
                "run",
                "python",
                "-m",
                "bridge_rgs.cli",
                "prepare",
                "--dataset",
                str(dataset),
                "--output",
                str(artifact_root),
                "--max-width",
                "1320",
                "--max-points",
                str(args.max_points),
                "--val-every",
                "0",
                "--seed",
                "42",
                "--workers",
                "8",
                "--pixel-protocol",
                "colmap_corner_v2",
            ],
            env,
            args.dry_run,
        )
    rgb_config = stage_config("full400_rgb.yaml", manifest, run_root / "rgb", {})
    semantic_config = stage_config(
        "full400_semantic.yaml",
        manifest,
        run_root / "semantic",
        {"warmstart": str(run_root / "rgb" / "last.pt")},
    )
    cross_config = stage_config(
        "full400_semantic_moments_cross.yaml",
        manifest,
        run_root / "semantic_cross",
        {"warmstart": str(run_root / "semantic" / "last.pt")},
    )
    for config in (rgb_config, semantic_config, cross_config):
        command(
            [uv, "run", "python", "-m", "bridge_rgs.cli", "train", "--config", str(config)],
            env,
            args.dry_run,
        )
    receipt = {
        "manifest": str(manifest),
        "stages": [str(rgb_config), str(semantic_config), str(cross_config)],
        "validation_views": 0,
        "dry_run": args.dry_run,
    }
    (run_root / "release_training_receipt.json").parent.mkdir(parents=True, exist_ok=True)
    (run_root / "release_training_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
