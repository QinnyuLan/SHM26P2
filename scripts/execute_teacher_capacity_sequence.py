"""Execute the four authorized fixed teacher stages sequentially; never retry failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def execute(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "locked_authorized":
        raise ValueError("Launch manifest is not locked and authorized")
    for path, expected in manifest["locked_files_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"Locked source/config changed: {path}")
    output = Path(manifest["output_root"])
    receipt_path = output / "execution_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Sequence already has an execution receipt; no automatic restart")
    env = dict(os.environ)
    env.update(manifest["environment"])
    receipt = {"status": "running", "launch_manifest_sha256": sha(manifest_path),
               "pid": os.getpid(), "started_utc": datetime.now(UTC).isoformat(), "stages": []}
    write(receipt_path, receipt)
    try:
        for stage in manifest["stages"]:
            started = time.monotonic()
            item = {"key": stage["key"], "status": "running", "command": stage["command"],
                    "log_path": stage["log_path"], "started_utc": datetime.now(UTC).isoformat()}
            receipt["stages"].append(item)
            with Path(stage["log_path"]).open("x") as log:
                process = subprocess.Popen(stage["command"], cwd=manifest["workspace_root"],
                                           env=env, stdout=log, stderr=subprocess.STDOUT)
                item["pid"] = process.pid
                write(receipt_path, receipt)
                print(f"Started {stage['key']} PID {process.pid}", flush=True)
                code = process.wait()
            item.update(exit_code=code, elapsed_seconds=time.monotonic()-started,
                        finished_utc=datetime.now(UTC).isoformat())
            if code:
                item["status"] = "failed"
                raise RuntimeError(f"{stage['key']} exited {code}; no retry")
            endpoint = Path(stage["stage_receipt"])
            state = json.loads(endpoint.read_text())
            if state["status"] != "completed" or state["steps"] != stage["steps"]:
                raise ValueError("Stage did not complete its fixed endpoint")
            item.update(status="completed", stage_receipt_sha256=sha(endpoint))
            write(receipt_path, receipt)
            print(f"Completed {stage['key']} in {item['elapsed_seconds']:.2f}s", flush=True)
        receipt.update(status="completed", finished_utc=datetime.now(UTC).isoformat())
        write(receipt_path, receipt)
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}",
                       finished_utc=datetime.now(UTC).isoformat())
        write(receipt_path, receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute(args.launch_manifest.resolve())
    else:
        print(json.dumps({"status": "inspection_only_no_training",
                          "launch_manifest_sha256": sha(args.launch_manifest)}))


if __name__ == "__main__":
    main()
