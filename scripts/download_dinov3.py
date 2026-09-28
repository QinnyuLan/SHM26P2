#!/usr/bin/env python3
"""Download the public ModelScope DINOv3 ViT-H+/16 snapshot and verify its weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

MODEL_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
REVISION = "98b7096b2938406ade58801a0bf75eca3198b5bd"
EXPECTED_SHA256 = "3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="models/dinov3-vith16plus")
    args = parser.parse_args()
    from modelscope.hub.snapshot_download import snapshot_download

    output = Path(args.output).resolve()
    path = Path(snapshot_download(MODEL_ID, revision=REVISION, local_dir=str(output)))
    actual = sha256_file(path / "model.safetensors")
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"Weight SHA256 mismatch: {actual}; expected {EXPECTED_SHA256}")
    config = json.loads((path / "config.json").read_text())
    if (config.get("hidden_size"), config.get("num_hidden_layers")) != (1280, 32):
        raise RuntimeError("Downloaded snapshot is not the expected ViT-H+/16 architecture")
    provenance = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "provider": "ModelScope",
        "source_url": f"https://modelscope.cn/models/{MODEL_ID}",
        "downloaded_at": datetime.now(UTC).isoformat(),
        "architecture": "DINOv3ViTModel",
        "parameters": 840592640,
        "files_sha256": {
            p.name: sha256_file(p)
            for p in path.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.name != "download_provenance.json"
        },
    }
    (path / "download_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {"model_dir": str(path), "revision": REVISION, "weights_sha256": actual}, indent=2
        )
    )


if __name__ == "__main__":
    main()
