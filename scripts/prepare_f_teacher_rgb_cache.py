"""Render the F (top4-normalized + MCMC) RGB domain for teacher adaptation.

This preparation is camera-only: labels, target RGB and validity payloads are never
opened.  The result deliberately follows the existing matched-teacher cache schema
so that the already audited DINOv3 adaptation launcher can be reused unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/sky/workspace/SHM2026")
RUNS = Path("/mnt/data/SHM2026/runs")
F_RUN = RUNS / "ibgs_joint_replacement_v2"
F_BUNDLE = F_RUN / "deployment/bundle.json"
F_WORKER = F_RUN / "deployment/render_ibgs_joint_bundle.py"
MCMC = RUNS / "rgb_mcmc_reference_500k/last.pt"
MANIFEST = ROOT / "artifacts/prepared/manifest.json"
OUT = RUNS / "f_teacher_rgb_cache_v1"
ADAPTER_SOURCE = RUNS / "matched_rgb_teacher_adaptation_v1/cache/source_snapshot/evaluate_rgb_teacher_transfer.py"
SCHEMA_ID = "original_png_multi_component_teacher_domain_v1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text())


def write(path: Path, value, replace=False):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def train_cameras(manifest):
    train = sorted((v for v in manifest["views"] if v["split"] == "train"), key=lambda v: v["name"])
    require(len(train) == 350 and sum(bool(v.get("mask_path")) for v in train) == 259,
            "Expected fixed 350 TRAIN / 259 annotated split")
    out = []
    sensor_distortion = read(F_BUNDLE)["sensor"]["distortion"]
    for v in train:
        row = {k: v[k] for k in ("name", "image_id", "camera_id", "split", "K", "w2c", "width", "height")}
        row["distortion"] = sensor_distortion
        out.append(row)
    return out


def average(a, b):
    require(a.dtype == b.dtype == np.uint8 and a.shape == b.shape, "RGB component mismatch")
    return np.rint((a.astype(np.float32) + b.astype(np.float32)) * .5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--stage", choices=("render",), default="render")
    args = ap.parse_args()
    output = args.output.resolve()
    manifest = read(MANIFEST)
    cameras = train_cameras(manifest)
    if output.exists():
        require((output / "plan.json").is_file() and (output / "cameras.json").is_file(),
                "Existing cache is not a resumable F plan")
    else:
        output.mkdir(parents=True)
        plan = {"status": "prepared", "id": SCHEMA_ID, "manifest": str(MANIFEST), "manifest_sha256": sha(MANIFEST),
                "bundle": str(F_BUNDLE), "bundle_sha256": sha(F_BUNDLE), "worker": str(F_WORKER), "worker_sha256": sha(F_WORKER),
                "mcmc": str(MCMC), "mcmc_sha256": sha(MCMC), "cameras": cameras,
                "domains": ["top4_normalized", "f_composite"], "gt_payload_reads": 0}
        write(output / "plan.json", plan)
        write(output / "cameras.json", cameras)
    # A separate IBGS uv child keeps the custom extension isolated from gsplat.
    ibgs_dir = output / "ibgs"
    if not ibgs_dir.exists():
        ibgs_dir.mkdir()
        cmd = ["uv", "run", "--no-project", "--python", "/mnt/data/SHM2026/third_party/ibgs/.venv/bin/python",
               "python", "-B", str(F_WORKER), "--bundle", str(F_BUNDLE), "--cameras", str(output / "cameras.json"),
               "--output", str(ibgs_dir), "--stage", "ibgs"]
        completed = subprocess.run(cmd, cwd=ROOT, check=False)
        require(completed.returncode == 0, "IBGS camera-only render failed")
    ibgs_receipt = read(ibgs_dir / "ibgs_execution_receipt.json")
    require(ibgs_receipt["status"] == "completed" and ibgs_receipt["selector_calls"] == 350,
            "Incomplete IBGS render")
    # Convert native top4 arrays to the exact F original-grid PNG protocol.
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("f_bundle_worker", F_WORKER)
    worker = module_from_spec(spec)
    spec.loader.exec_module(worker)
    bundle = read(F_BUNDLE)
    # Use the exact frozen scoring-side distortion helper for the PNG barrier.
    sys.path.insert(0, bundle["main"]["source_snapshot"])
    from bridge_rgs import official_evaluate as frozen_official
    top_dir = output / "components/top4_normalized"; top_dir.mkdir(parents=True, exist_ok=True)
    native = read(ibgs_receipt["native_predictions"]["path"])["records"]
    for row in native:
        value = np.load(row["path"], allow_pickle=False)
        require(value.dtype == np.float32 and value.shape == (989, 1320, 3), "Bad native top4 array")
        value = np.clip(value, 0., 1.)
        camera = next(c for c in cameras if c["name"] == row["name"])
        _, _, _, back = frozen_official.distortion_render_grid(np.asarray(camera["K"], np.float64),
            np.asarray(camera["distortion"], np.float64), camera["width"], camera["height"], "colmap_corner_v2")
        if back is not None:
            value = cv2.remap(value, back[..., 0], back[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        path = top_dir / row["name"]
        require(cv2.imwrite(str(path), np.rint(value * 255).astype(np.uint8)[..., ::-1]), "Top4 PNG write failed")
    # Render MCMC RGB in the normal gsplat environment.
    import torch
    sys.path.insert(0, str(ROOT))
    from bridge_rgs import official_evaluate as official
    m_dir = output / "components/mcmc"; m_dir.mkdir(parents=True, exist_ok=True)
    scene, state = official._load_scene(MCMC, "cuda")
    scene.eval().requires_grad_(False)
    with torch.inference_mode():
        for camera in cameras:
            rgb, _, _ = official.predict_official_camera(scene, camera, state)
            require(cv2.imwrite(str(m_dir / camera["name"]), rgb[..., ::-1]), "MCMC PNG write failed")
    del scene, state; torch.cuda.empty_cache()
    # Composite and migrate both domains to the audited legacy teacher canvas.
    ad_spec = spec_from_file_location("f_teacher_adapter", ADAPTER_SOURCE)
    adapter = module_from_spec(ad_spec); ad_spec.loader.exec_module(adapter)
    original_root = output / "original_rgb"; legacy_root = output / "legacy_rgb"
    records = {"train_selected": [], "train_composite": [], "val_composite": []}
    for camera in cameras:
        name = camera["name"]
        top = cv2.cvtColor(cv2.imread(str(top_dir / name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mcmc = cv2.cvtColor(cv2.imread(str(m_dir / name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        composite = average(top, mcmc)
        row = {"name": name, "camera": camera}
        for domain, rgb in (("top4_normalized", top), ("f_composite", composite)):
            op = original_root / domain / name; op.parent.mkdir(parents=True, exist_ok=True)
            lp = legacy_root / domain / name; lp.parent.mkdir(parents=True, exist_ok=True)
            require(cv2.imwrite(str(op), rgb[..., ::-1]), "Original RGB write failed")
            mx, my, _, info = adapter.adapter_maps(camera, frozen_official.distortion_render_grid)
            canvas = adapter.input_canvas(rgb, mx, my)
            require(cv2.imwrite(str(lp), canvas[..., ::-1]), "Legacy RGB write failed")
            rec = {"name": name, "image_path": str(lp), "sha256": sha(lp), "original_rgb": {"image_path": str(op), "sha256": sha(op)},
                   "camera": camera, "adapter_info": info}
            records["train_selected" if domain == "top4_normalized" else "train_composite"].append(rec)
    receipt = {"status": "completed", "id": SCHEMA_ID, "original_manifest_sha256": sha(MANIFEST),
        "components": {"top4_normalized": {"checkpoint": bundle["ibgs"]["checkpoint"]["path"], "checkpoint_sha256": bundle["ibgs"]["checkpoint"]["sha256"], "profile": "colmap_corner_v2"},
                        "mcmc": {"checkpoint": str(MCMC), "checkpoint_sha256": sha(MCMC), "profile": "colmap_corner_v2"}},
        "records": records, "scene_renders": 350 + 350, "gt_payload_reads": 0,
        "plan_sha256": sha(output / "plan.json"), "ibgs_receipt_sha256": sha(ibgs_dir / "ibgs_execution_receipt.json"),
        "adapter": "original_png_to_legacy_pure_hplus_v1", "train_count": 350,
        "created_utc": datetime.now(UTC).isoformat()}
    write(output / "execution_receipt.json", receipt)
    write(output / "launch_receipt.json", {"status": "completed", "exit_code": 0, "natural_completion": True,
        "execution_receipt_sha256": sha(output / "execution_receipt.json"), "root_observed_live_exit": True})
    print(json.dumps({"status": "completed", "output": str(output), "train": 350, "gt_payload_reads": 0}))


if __name__ == "__main__":
    main()
