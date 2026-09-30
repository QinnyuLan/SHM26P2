#!/usr/bin/env python3
"""Run deterministic pre-submission checks for the release package.

The checksum pass is optional because ModelScope assets may be downloaded after
cloning. When an asset root is supplied, every published checkpoint is checked
for both byte count and SHA-256 before the script returns success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "release"
MANIFEST = RELEASE / "checkpoints/checkpoint_manifest.json"
CONTRACT = RELEASE / "configs/render_contract.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, help="ModelScope checkout root to checksum")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    contract = json.loads(CONTRACT.read_text())
    assert contract["pixel_protocol"] == "colmap_corner_v2"
    assert sorted(contract["class_ids"].values()) == [0, 1, 2, 3, 4]
    assert contract["official_sensor"]["width"] == 1320
    assert contract["official_sensor"]["height"] == 989
    assert len(manifest["files"]) == 3
    for config in (
        "full400_rgb.yaml",
        "full400_semantic.yaml",
        "full400_semantic_moments_cross.yaml",
    ):
        yaml.safe_load((RELEASE / "configs" / config).read_text())
    for source in ("render_blind.py", "train_full400.py", "verify_release.py"):
        compile((RELEASE / "code" / source).read_text(), source, "exec")

    checked = []
    missing = []
    if args.assets_root:
        assets_root = args.assets_root.resolve()
        for item in manifest["files"]:
            path = assets_root / item["path"]
            if not path.is_file():
                missing.append(str(path))
                continue
            actual_bytes = path.stat().st_size
            actual_sha = sha256(path)
            if actual_bytes != item["bytes"] or actual_sha != item["sha256"]:
                raise SystemExit(
                    f"checksum mismatch for {path}: {actual_bytes}/{actual_sha} "
                    f"(expected {item['bytes']}/{item['sha256']})"
                )
            checked.append(str(path))
    print(
        json.dumps(
            {
                "contract": "ok",
                "configs": "ok",
                "python": "ok",
                "checkpoints_checked": len(checked),
                "missing": missing,
            },
            indent=2,
        )
    )
    if missing:
        raise SystemExit("missing published assets; download ModelScope assets before deployment")


if __name__ == "__main__":
    main()
