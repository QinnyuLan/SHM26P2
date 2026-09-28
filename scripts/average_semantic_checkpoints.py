"""Create one semantic-averaged checkpoint; preserve geometry/RGB exactly."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from bridge_rgs.averaging import average_semantic_states


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance_path = args.output.with_suffix(".provenance.json")
    if args.output.exists() or provenance_path.exists():
        raise FileExistsError("Refusing to overwrite an averaging result")
    torch.set_num_threads(8)
    states = [torch.load(path, map_location="cpu", mmap=True, weights_only=False)
              for path in args.checkpoints]
    result = average_semantic_states(states)
    result["averaging"]["sources"] = [
        {"path": str(path.resolve()), "sha256": digest(path)} for path in args.checkpoints]
    manifest = Path(result["config"]["manifest"])
    result["averaging"]["manifest_sha256"] = digest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    provenance = dict(result["averaging"], checkpoint_sha256=digest(args.output))
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(args.output), **provenance}, indent=2))


if __name__ == "__main__":
    main()
