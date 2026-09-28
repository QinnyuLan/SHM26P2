"""Read-only checkpoint reference inventory from local run receipts/provenance.

Absence from this inventory is NEVER permission to delete a file. Planned work,
open processes and references outside the scanned documents are not covered.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
SHA = re.compile(r"[0-9a-fA-F]{64}\Z")


def checkpoint_string(value):
    return (isinstance(value, str) and "\n" not in value and "://" not in value
            and Path(value).suffix.lower() in SUFFIXES)


def resolve_reference(value, document, model_dir=None):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(), "absolute"
    if model_dir is not None:
        base = Path(model_dir)
        return ((ROOT / base if not base.is_absolute() else base) / path).resolve(), "model_dir"
    if path.parts[0] in {"runs", "artifacts", "models", "Dataset"}:
        return (ROOT / path).resolve(), "workspace-relative"
    # Bare checkpoint names in run metadata conventionally live beside it.
    return (document.parent / path).resolve(), "document-relative"


def references(document, data):
    result = []

    def append(value, field, expected=None, model_dir=None, role="reference", needs_model_dir=False):
        raw = Path(value).expanduser()
        explicit_path = raw.is_absolute() or raw.parts[0] in {"runs", "artifacts", "models", "Dataset"}
        unresolved = needs_model_dir and model_dir is None and not explicit_path
        if unresolved:
            path, resolution = None, "unresolved-no-model-dir"
        else:
            path, resolution = resolve_reference(value, document, model_dir)
        result.append({"path": str(path) if path is not None else None, "raw_path": value,
                       "document": str(document.relative_to(ROOT)), "field": field,
                       "expected_sha256": expected.lower() if isinstance(expected, str) and SHA.fullmatch(expected) else None,
                       "resolution": resolution, "role": role})
        if unresolved:
            result[-1].update(verification="unresolved",
                              reason="Relative backbone weight filename has no enclosing model_dir; no path inferred")

    def walk(value, field="$", parent=None, key=None, model_dir=None):
        if isinstance(value, dict):
            # The nearest declaration wins; nested receipt blocks must not use
            # an unrelated top-level directory or a sibling block's directory.
            if "model_dir" in value:
                declared = value["model_dir"]
                model_dir = declared if isinstance(declared, str) and declared.strip() else None
            for name, item in value.items():
                pointer = f"{field}.{name}"
                if checkpoint_string(name) and isinstance(item, str) and SHA.fullmatch(item):
                    backbone_files = ((isinstance(key, str) and key.endswith("weights_sha256")) or
                                      (key == "files_sha256" and ".backbone_source." in pointer))
                    append(name, pointer, item, model_dir if backbone_files else None,
                           "hashed_input", needs_model_dir=backbone_files)
                else:
                    walk(item, pointer, value, name, model_dir)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{field}[{index}]", parent, key, model_dir)
        elif checkpoint_string(value):
            parent = parent or {}
            expected = parent.get("sha256") if key == "path" else parent.get(f"{key}_sha256")
            role = "reference"
            if key == "checkpoint" and document.name.endswith(".provenance.json"):
                role = "derived_output"
            append(value, field, expected, role=role)

    walk(data)
    # Link duplicated config/command pointers to an explicit hash in THIS
    # document only; do not borrow a current producer hash as historical proof.
    document_hashes = defaultdict(set)
    for item in result:
        if item["path"] is not None and item["expected_sha256"]:
            document_hashes[item["path"]].add(item["expected_sha256"])
    for item in result:
        if item["path"] is None:
            continue
        hashes = document_hashes[item["path"]]
        if not item["expected_sha256"] and len(hashes) == 1:
            item["expected_sha256"] = next(iter(hashes))
            item["hash_link"] = "another explicit reference in the same document"
    return result


def main():
    documents = [path for path in (ROOT / "runs").rglob("*.json")
                 if "source_snapshot" not in path.parts and
                 any(token in path.name for token in ("receipt", "provenance", "lineage"))]
    edges, errors, scanned = [], [], []
    for path in sorted(documents):
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                continue
            edges.extend(references(path, data))
            scanned.append(str(path.relative_to(ROOT)))
        except (OSError, ValueError) as error:
            errors.append({"document": str(path.relative_to(ROOT)), "error": str(error)})
    grouped = defaultdict(list)
    unresolved = []
    for edge in edges:
        if edge["path"] is None:
            unresolved.append(edge)
        else:
            grouped[edge["path"]].append(edge)
    artifacts = []
    for name, incoming in sorted(grouped.items()):
        path = Path(name)
        exists = path.is_file()
        record = {"path": name, "exists": exists, "references": incoming}
        if exists:
            before = path.stat()
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            after = path.stat()
            stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            record.update(bytes=after.st_size, sha256=digest, stable_during_hash=stable)
        for edge in incoming:
            if not exists:
                edge["verification"] = "missing"
            elif not record["stable_during_hash"]:
                edge["verification"] = "changed_during_inventory"
            elif edge["expected_sha256"] is None:
                edge["verification"] = "exists_without_recorded_input_hash"
            else:
                edge["verification"] = "match" if edge["expected_sha256"] == record["sha256"] else "mismatch"
        artifacts.append(record)
    known_missing = str(ROOT / "runs/support_split_rgb_full_resumed2/step_020000.pt")
    report = {
        "generated_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "workspace": str(ROOT), "mode": "read-only; no artifact mutated or deleted",
        "scope": "Checkpoint paths in runs/**/*receipt*.json, *provenance*.json and *lineage*.json; immutable source copies excluded",
        "limitations": ["Not a deletion allow-list. Missing references do not prove a checkpoint is unneeded.",
                       "Planned configs, live process file descriptors and informal dependencies are outside this scan.",
                       "Hashes are current observations; mutable last/best paths may legitimately differ from an earlier recorded input.",
                       "Unhashed references remain unverified even when a file exists.",
                       "Unanchored relative backbone weight names are unresolved, not missing; no current path is guessed."],
        "summary": {"documents": len(scanned), "checkpoint_paths": len(artifacts),
                    "references": len(edges), "missing_paths": sum(not item["exists"] for item in artifacts),
                    "unresolved_references": len(unresolved),
                    "mismatched_references": sum(edge["verification"] == "mismatch" for edge in edges)},
        "known_retention_limitation": {"path": known_missing,
            "expected_sha256": "f730749483f199d8ea004ecbec71e024e470fc19c9c99ecd9f56cc672212c21c",
            "missing": not Path(known_missing).is_file(),
            "note": "Original 20k input was removed before its appearance-stage dependency was communicated; derived last and completed audits remain. No regenerated substitute."},
        "last_checkpoint_dependencies": [{"path": item["path"], "exists": item["exists"],
                                           "reference_documents": sorted({edge["document"] for edge in item["references"]})}
                                          for item in artifacts if Path(item["path"]).name == "last.pt"],
        "artifacts": artifacts, "unresolved_references": unresolved,
        "scanned_documents": scanned, "read_errors": errors,
    }
    output = ROOT / "artifacts/checkpoint_dependencies.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), **report["summary"], "read_errors": len(errors)}))


if __name__ == "__main__":
    main()
