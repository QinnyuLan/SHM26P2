#!/usr/bin/env python3
"""Download and verify the public SHM2026 ModelScope asset bundle."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from modelscope import snapshot_download

DATASET_ID = "sky931/SHM2026"


def verify_manifest(root: Path) -> tuple[int, list[str]]:
    manifest = root / "manifest.sha256"
    if not manifest.is_file():
        return 0, [f"missing manifest: {manifest}"]
    failures: list[str] = []
    checked = 0
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        checked += 1
        if digest.hexdigest() != expected:
            failures.append(f"sha256 mismatch: {relative}")
    return checked, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("release_assets/SHM2026"))
    parser.add_argument("--revision", default="master")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.output),
        max_workers=args.workers,
    )
    if args.skip_verify:
        return
    checked, failures = verify_manifest(args.output)
    if failures:
        raise SystemExit("asset verification failed:\n" + "\n".join(failures))
    print(f"verified {checked} files from {DATASET_ID}@{args.revision}")


if __name__ == "__main__":
    main()
