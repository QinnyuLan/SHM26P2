"""Fetch one pinned ModelScope 7B snapshot, with bounded Range resume and SHA256.

No model import, GPU access, shared cache, or duplicate full-weight staging copy.
All partials, locks, progress and receipts are in the chosen /mnt/data directory.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

MODEL_ID = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
REVISION = "5251e00b307184bb247d076713375235d554fefb"
API = f"https://modelscope.cn/api/v1/models/{MODEL_ID}/repo"
DEFAULT_OUTPUT = Path("/mnt/data/SHM2026/models/dinov3-vit7b16")
PINNED = {
    "config.json": (746, "b317786dd342ad8f51ce8246f39754ba648c7d375ad75d7b415507fe58d74ce6"),
    "configuration.json": (74, "2e298ee07ca76e112e8cb5524061000871a464b0088d14becb7d0564215a9380"),
    "LICENSE.md": (7503, "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e"),
    "README.md": (14468, "05d39e90f6e87aa1bc10b9f4b2386f43e24db4eacf237b59ba1ee2c8feee743c"),
    "preprocessor_config.json": (585, "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7"),
    "model.safetensors.index.json": (48723, "ae26856def93bcf537202109c60ef76ca22d1f373dc85d2b68aeb6fe940c85fd"),
    "model-00001-of-00006.safetensors": (4980241600, "7132627f25459ee8797cb2965d3427706a87119ccfcdfef1bd7977dd7580821f"),
    "model-00002-of-00006.safetensors": (4967510232, "a7b17660c408adf235c318328010ecded0ee24181b97b806c1e45b42efc5ff4b"),
    "model-00003-of-00006.safetensors": (4967510568, "b5937a7a7051239798a6984d07e2d68fb1f8f93d0947c63be9ffe1bbbbe8dab7"),
    "model-00004-of-00006.safetensors": (4967543448, "569c817cc4424410c1d49df48e053ba4206eb7a1bd2c53381b7f7dfb32c4d57e"),
    "model-00005-of-00006.safetensors": (4967543320, "51ab5686ebe67cb48b738caee366d6a0bd0fb19b3f0d37dc37b8e82bf34c0e66"),
    "model-00006-of-00006.safetensors": (2013860920, "d3f77e2cbd0f9a349eeaf2559213f2495e43e4229a02a33f083f48ab5539ecbb"),
}
CHUNK_BYTES = 4 * 1024 * 1024


class ContractError(RuntimeError):
    """Non-transient failure; preserve bytes and stop without automatic retry."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_listing(listing: dict, expected: dict = PINNED) -> dict:
    if listing.get("Code") != 200 or not listing.get("Success"):
        raise ContractError("ModelScope listing unsuccessful")
    rows = {item["Path"]: item for item in listing["Data"]["Files"]}
    result = {}
    for name, (size, digest) in expected.items():
        row = rows.get(name, {})
        if row.get("Size") != size or row.get("Sha256") != digest:
            raise ContractError(f"Server declaration differs from pinned manifest: {name}")
        result[name] = {"size": size, "sha256": digest}
    return result


def verify_file(path: Path, size: int, digest: str) -> str:
    if path.stat().st_size != size:
        raise ContractError(f"Size mismatch; file preserved: {path}")
    actual = sha256(path)
    if actual != digest:
        raise ContractError(f"SHA256 mismatch; file preserved: {path}")
    return actual


def validate_response(response, offset: int, expected_size: int) -> None:
    response.raise_for_status()
    if response.status_code == 206:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if not match or tuple(map(int, match.groups())) != (
            offset, expected_size - 1, expected_size
        ):
            raise ContractError("Invalid Content-Range; partial file preserved")
    elif response.status_code == 200 and offset == 0:
        pass
    else:
        raise ContractError("Server ignored Range resume; partial file preserved")
    length = response.headers.get("Content-Length")
    if length is not None and int(length) != expected_size - offset:
        raise ContractError("Response Content-Length differs from pinned size")
    if response.headers.get("Content-Encoding", "identity") != "identity":
        raise ContractError("Encoded download response is not byte-resumable")


def download_file(output: Path, name: str, size: int, digest: str, *, attempts: int = 3,
                  get=requests.get, pause=time.sleep, progress=lambda *args: None,
                  stop: threading.Event | None = None) -> dict:
    """Resume only an exact byte range. Hash after transfer, rename only when valid."""
    if Path(name).name != name or attempts < 1:
        raise ValueError("Invalid file name or retry budget")
    target, partial = output / name, output / (name + ".part")
    if target.exists():
        actual = verify_file(target, size, digest)
        progress(name, size, "verified_existing")
        return {"status": "verified_existing", "size": size, "sha256": actual, "attempts": 0}
    if partial.exists() and partial.stat().st_size > size:
        raise ContractError(f"Oversized partial preserved: {partial}")
    initial = partial.stat().st_size if partial.exists() else 0
    used_attempts = 0
    for attempt in range(1, attempts + 1):
        if stop is not None and stop.is_set():
            raise InterruptedError("Download interrupted; partial preserved")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            break
        used_attempts = attempt
        progress(name, offset, f"downloading_attempt_{attempt}")
        try:
            with get(API, params={"Revision": REVISION, "FilePath": name},
                     headers={"Range": f"bytes={offset}-", "Accept-Encoding": "identity"},
                     stream=True, timeout=(15, 60)) as response:
                validate_response(response, offset, size)
                with partial.open("ab") as stream:
                    received = offset
                    for block in response.iter_content(chunk_size=CHUNK_BYTES):
                        if stop is not None and stop.is_set():
                            raise InterruptedError("Download interrupted; partial preserved")
                        if not block:
                            continue
                        if received + len(block) > size:
                            raise ContractError("Download exceeded pinned size; partial preserved")
                        stream.write(block)
                        received += len(block)
                        progress(name, received, "downloading")
                    stream.flush()
                    os.fsync(stream.fileno())
                if received != size:
                    raise requests.ConnectionError("Incomplete response; retain partial for resume")
            break
        except requests.RequestException:
            if attempt == attempts:
                raise
            pause(min(2 ** attempt, 8))
    progress(name, size, "hashing")
    actual = verify_file(partial, size, digest)
    partial.replace(target)
    progress(name, size, "verified")
    return {"status": "verified", "size": size, "sha256": actual,
            "resumed_from_bytes": initial, "attempts": used_attempts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--attempts", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path("/mnt/data")):
        raise ValueError("All output/partials must remain on /mnt/data")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".download.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, output)


def run(args, output: Path) -> None:
    # One bounded metadata request. Rerunning is explicit if this fails.
    listing_response = requests.get(API + "/files", params={"Revision": REVISION,
                                    "Recursive": "true"}, timeout=(15, 60))
    listing_response.raise_for_status()
    listing = listing_response.json()
    declarations = validate_listing(listing)
    source = {"model_id": MODEL_ID, "revision": REVISION, "files": declarations}
    source_path = output / "download_source_manifest.json"
    if source_path.exists() and json.loads(source_path.read_text()) != source:
        raise ContractError("Existing directory is bound to a different source manifest")
    atomic_json(source_path, source)
    if args.verify_only:
        for name, (size, digest) in PINNED.items():
            verify_file(output / name, size, digest)
        print(json.dumps({"status": "verified", "files": len(PINNED)}), flush=True)
        return
    remaining = sum(max(0, size - max(
        (output / name).stat().st_size if (output / name).exists() else 0,
        (output / (name + ".part")).stat().st_size
        if (output / (name + ".part")).exists() else 0
    )) for name, (size, _) in PINNED.items())
    if shutil.disk_usage(output).free < remaining + 2 * 1024**3:
        raise ContractError("Insufficient data-volume space plus 2 GiB reserve")
    started = time.monotonic()
    state_lock, stop = threading.Lock(), threading.Event()
    state = {"status": "running", **source, "started_at": datetime.now(UTC).isoformat(),
             "pid": os.getpid(), "workers": args.workers, "attempts_per_file_max": args.attempts,
             "script_sha256": sha256(Path(__file__)), "progress": {}, "results": {},
             "temporary_files_policy": "same-directory .part; no shared cache or full duplicate"}
    initial_bytes = sum((output / (name + ".part")).stat().st_size
                        if (output / (name + ".part")).exists() else
                        (output / name).stat().st_size if (output / name).exists() else 0
                        for name in PINNED)

    def update(name, count, status):
        with state_lock:
            state["progress"][name] = {"bytes": count, "status": status}

    def save_progress():
        with state_lock:
            state["elapsed_seconds"] = time.monotonic() - started
            atomic_json(output / "download_state.json", state)
            count = sum(item["bytes"] for item in state["progress"].values())
            # Until each worker has reported, this can undercount queued partials.
            rate = max(0, count - initial_bytes) / max(1, state["elapsed_seconds"])
            total = sum(size for size, _ in PINNED.values())
            print(json.dumps({"status": state["status"], "reported_bytes": count,
                              "total_bytes": total, "mean_mib_per_s": rate / 1024**2,
                              "eta_seconds": (total - count) / rate if rate > 0 else None,
                              "files_verified": len(state["results"])}), flush=True)

    def interrupt(signum, frame):
        stop.set()
        raise KeyboardInterrupt(f"Signal {signum}; retaining partial files")

    signal.signal(signal.SIGTERM, interrupt)
    failures = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(download_file, output, name, size, digest,
                       attempts=args.attempts, progress=update, stop=stop): name
                       for name, (size, digest) in PINNED.items()}
            try:
                while pending:
                    done, _ = concurrent.futures.wait(pending, timeout=10,
                               return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        name = pending.pop(future)
                        try:
                            state["results"][name] = future.result()
                        except (OSError, RuntimeError, requests.RequestException, ValueError) as error:
                            failures[name] = f"{type(error).__name__}: {error}"
                    save_progress()
            finally:
                stop.set()
        if failures:
            raise RuntimeError(f"Bounded download failed; rerun to resume: {failures}")
        index = json.loads((output / "model.safetensors.index.json").read_text())
        if index["metadata"] != {"total_parameters": 6716035072, "total_size": 26864140288}:
            raise ContractError("Index total differs")
        expected_shards = {name for name in PINNED if name.endswith(".safetensors")}
        if set(index["weight_map"].values()) != expected_shards:
            raise ContractError("Index does not reference exactly six pinned shards")
        receipt = {"status": "completed", "model_id": MODEL_ID, "revision": REVISION,
                   "provider": "ModelScope", "source_url": f"https://modelscope.cn/models/{MODEL_ID}",
                   "downloaded_at": datetime.now(UTC).isoformat(), "model_dir": str(output),
                   "architecture": "DINOv3ViTModel", "parameters": 6716035072,
                   "source_manifest_sha256": sha256(source_path),
                   "downloader_sha256": state["script_sha256"],
                   "files_sha256": {name: row["sha256"] for name, row in state["results"].items()},
                   "files": {name: {"server_declared": declarations[name], "actual": row}
                             for name, row in state["results"].items()}}
        atomic_json(output / "download_provenance.json", receipt)
        state["status"] = "completed"
    except BaseException as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save_progress()


if __name__ == "__main__":
    main()
