"""Measure camera-only native RGB+semantic rendering, excluding file I/O."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from bridge_rgs.evaluate import evaluation_fingerprint
from bridge_rgs.io import load_manifest
from bridge_rgs.train import load_scene


def gpu_clients():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=True)
    return sorted({int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--refiner-flip-tta", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true",
                        help="Diagnostic timing only; disclose other GPU client PIDs")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("repeats and warmup must be positive")
    torch.set_num_threads(8)
    manifest = load_manifest(args.manifest)
    views = [v for v in manifest["views"] if v["split"] == "val"]
    if not views:
        raise ValueError("A fixed validation-camera set is required for comparable timing")
    scene, checkpoint = load_scene(args.checkpoint)
    from bridge_rgs.coordinates import pixel_protocol, require_matching_protocol
    require_matching_protocol(checkpoint, manifest, "benchmark grid")
    if checkpoint.get("manifest_sha256") and checkpoint["manifest_sha256"] != manifest["_manifest_sha256"]:
        raise ValueError("Benchmark manifest SHA differs from checkpoint")
    scene.eval()
    cameras = [(torch.tensor(v["K"], device="cuda", dtype=torch.float32),
                torch.tensor(v.get("w2c_original", v["w2c"]), device="cuda", dtype=torch.float32),
                int(v["width"]), int(v["height"])) for v in views]
    before = [pid for pid in gpu_clients() if pid != os.getpid()]
    if before and not args.allow_shared_gpu:
        raise RuntimeError(f"Other GPU clients are active: {before}; benchmark after training completes")
    times = [[] for _ in views]

    def prediction(camera):
        result = scene.render(*camera, semantics=True, refine=True, absgrad=False)
        if args.refiner_flip_tta:
            from bridge_rgs.refiner_tta import horizontal_flip_average
            result["probabilities"] = horizontal_flip_average(scene.refiner, result)
        result["probabilities"].argmax(-1)

    with torch.no_grad():
        for _ in range(args.warmup):
            prediction(cameras[0])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            for i, camera in enumerate(cameras):
                torch.cuda.synchronize()
                start = time.perf_counter()
                # Include final official class selection; exclude PNG encoding and device-to-host transfer.
                prediction(camera)
                torch.cuda.synchronize()
                times[i].append(time.perf_counter() - start)
    after = [pid for pid in gpu_clients() if pid != os.getpid()]
    if after and not args.allow_shared_gpu:
        raise RuntimeError(f"Other GPU clients appeared during benchmark: {after}; discard this timing")
    values = np.array(times)
    sha = hashlib.sha256()
    with args.checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
    report = {
        "protocol": "camera-only; native undistorted grid; RGB + refined semantic argmax; batch 1",
        "excludes": "model load, JIT compilation, image/mask input, PNG output, host transfer, distortion remapping",
        "timing": "synchronized wall time after warmup; all validation cameras in fixed order",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha.hexdigest(),
        "training_step": checkpoint["step"], "evaluation_fingerprint": evaluation_fingerprint(
            views, 1.0, pixel_protocol(manifest), manifest["_manifest_sha256"]),
        "manifest_sha256": manifest["_manifest_sha256"], "refiner_flip_tta": args.refiner_flip_tta,
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "actual_imports": {name: {"path": str(Path(importlib.import_module(name).__file__).resolve()),
                                  "sha256": hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()}
                           for name in ("bridge_rgs.train", "bridge_rgs.model", "bridge_rgs.refinement",
                                        "bridge_rgs.refiner_tta", "bridge_rgs.coordinates")},
        "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "other_gpu_client_pids_before": before, "other_gpu_client_pids_after": after,
        "shared_gpu_diagnostic": bool(before or after or args.allow_shared_gpu),
        "gaussians": len(scene.splats["means"]), "parameters": sum(p.numel() for p in scene.parameters()),
        "refiner_parameters": sum(p.numel() for p in scene.refiner.parameters()),
        "repeats": args.repeats, "warmup": args.warmup,
        "mean_seconds": float(values.mean()), "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.quantile(values, .95)), "fps_from_mean_latency": float(1 / values.mean()),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "views": [{"name": v["name"], "width": v["width"], "height": v["height"], "seconds": t}
                  for v, t in zip(views, times)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("mean_seconds", "fps_from_mean_latency", "peak_gpu_allocated_gib")}, indent=2))


if __name__ == "__main__":
    main()
