"""Create one fixed 1NN semantic warmstart; never train or alter target RGB geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

import bridge_rgs.semantic_transfer as implementation


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verified_manifest(state, checkpoint):
    path = Path(state["config"]["manifest"]).resolve(strict=True)
    current = digest(path)
    receipt_path = checkpoint.parent / "experiment_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    original = receipt["input_hashes"]["manifest"]
    if Path(original["path"]).resolve() != path or original["sha256"] != current:
        raise ValueError(f"Manifest differs from original experiment receipt: {checkpoint}")
    manifest = json.loads(path.read_text())
    return {"path": str(path), "sha256": current,
            "train_names": [v["name"] for v in manifest["views"] if v["split"] == "train"],
            "validation_names": [v["name"] for v in manifest["views"] if v["split"] == "val"],
            "experiment_receipt_sha256": digest(receipt_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/refiner_multiscale_v1/last.pt"))
    parser.add_argument("--target", type=Path, default=Path("runs/strong_rgb/last.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/semantic_transfer_1nn/last.pt"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    source_path, target_path = args.source.resolve(strict=True), args.target.resolve(strict=True)
    output = args.output.resolve()
    provenance_path = output.with_suffix(".provenance.json")
    snapshot = output.parent / "source_snapshot"
    if output.exists() or provenance_path.exists() or snapshot.exists():
        raise FileExistsError("Refusing to overwrite semantic transfer output/provenance")
    input_hashes = {"source": {"path": str(source_path), "sha256": digest(source_path)},
                    "target": {"path": str(target_path), "sha256": digest(target_path)}}
    source, target = [torch.load(path, map_location="cpu", mmap=True, weights_only=False)
                      for path in (source_path, target_path)]
    source_manifest = verified_manifest(source, source_path)
    target_manifest = verified_manifest(target, target_path)
    result = implementation.transfer_semantic_1nn(
        source, target, source_manifest_sha256=source_manifest["sha256"],
        target_manifest_sha256=target_manifest["sha256"])
    for item in input_hashes.values():
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Input checkpoint changed during transfer")
    result["semantic_transfer"].update(inputs=input_hashes, source_manifest=source_manifest,
                                     target_manifest=target_manifest,
                                     script_sha256=digest(Path(__file__).resolve()))
    output.parent.mkdir(parents=True, exist_ok=True)
    package = Path(implementation.__file__).resolve().parent
    shutil.copytree(package, snapshot / "bridge_rgs",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    result["semantic_transfer"]["source_snapshot_hashes"] = {
        str(path.relative_to(snapshot)): digest(path) for path in sorted(snapshot.rglob("*.py"))}
    torch.save(result, output)
    provenance = dict(result["semantic_transfer"], checkpoint=str(output),
                      checkpoint_sha256=digest(output))
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(output), "provenance": str(provenance_path),
                      "source_points": provenance["source_points"],
                      "target_points": provenance["target_points"],
                      "within_source_3sigma_shape":
                          provenance["within_nearest_source_3sigma_shape_fraction"]}))


if __name__ == "__main__":
    main()
