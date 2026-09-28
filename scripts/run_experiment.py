"""Train and evaluate using an immutable source snapshot for each experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--source-snapshot", type=Path,
                        help="Reuse a verified run's source_snapshot directory")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "source_snapshot"
    receipt_path = output / "experiment_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"Refusing to overwrite experiment receipt: {receipt_path}")
    source_package = ((args.source_snapshot / "bridge_rgs") if args.source_snapshot
                      else ROOT / "src/bridge_rgs")
    shutil.copytree(source_package, snapshot / "bridge_rgs",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    effective = output / "experiment_config.yaml"
    effective.write_text(yaml.safe_dump(config, sort_keys=False))
    inputs = {"manifest": ROOT / config["manifest"], "uv_lock": ROOT / "uv.lock"}
    manifest = json.loads(inputs["manifest"].read_text())
    init_points = config.get("init_points", manifest.get("init_points_path"))
    if init_points:
        inputs["init_points"] = ROOT / init_points
    if config.get("pseudo_dir"):
        inputs["pseudo_provenance"] = ROOT / config["pseudo_dir"] / "provenance.json"
    if config.get("sparse_depth_weight", 0) > 0 or config.get("sparse_front_weight", 0) > 0:
        track_path = config.get("sparse_depth", {}).get("colmap_images_path")
        if track_path is None:
            track_path = Path(manifest["dataset_root"]) / "camera_parameters/images.txt"
        if config.get("sparse_depth", {}).get("uv_mode", "observed") == "observed":
            inputs["sparse_depth_tracks"] = ROOT / track_path
    if args.resume or config.get("warmstart"):
        inputs["initial_checkpoint"] = ROOT / (args.resume or config["warmstart"])
    receipt = {"started_utc": datetime.now(UTC).isoformat(),
               "config": config, "resume": str(args.resume) if args.resume else None,
               "source_origin": str(source_package.resolve()),
               "source_hashes": {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))},
               "input_hashes": {name: {"path": str(path), "sha256": digest(path)} for name, path in inputs.items()},
               "status": "running"}
    receipt_path.write_text(json.dumps(receipt, indent=2))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(snapshot)
    environment["OMP_NUM_THREADS"] = "8"
    environment["OPENBLAS_NUM_THREADS"] = "8"
    train_command = [sys.executable, "-m", "bridge_rgs.cli", "train", "--config", str(effective)]
    if args.resume:
        train_command.extend(["--resume", str(args.resume)])
    try:
        with (output / "process.log").open("w") as log:
            subprocess.run(train_command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, "-m", "bridge_rgs.cli", "evaluate", "--checkpoint", str(output / "last.pt"),
                            "--manifest", config["manifest"], "--output", str(output / "evaluation_native"), "--lpips"],
                           cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        receipt["status"] = "completed"
        receipt["checkpoint_sha256"] = digest(output / "last.pt")
        receipt["evaluation_sha256"] = digest(output / "evaluation_native/metrics.json")
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        raise
    finally:
        receipt["finished_utc"] = datetime.now(UTC).isoformat()
        receipt_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"output": str(output), "status": receipt["status"]}))


if __name__ == "__main__":
    main()
