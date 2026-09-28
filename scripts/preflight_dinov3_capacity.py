"""Two-step TRAIN-only DINOv3 capacity preflight; never a teacher training run."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch

REVISION_7B = "5251e00b307184bb247d076713375235d554fefb"
REVISION_HPLUS = "98b7096b2938406ade58801a0bf75eca3198b5bd"
MODELS = {
    "hplus": {"model_id": "facebook/dinov3-vith16plus-pretrain-lvd1689m", "revision": REVISION_HPLUS},
    "7b": {"model_id": "facebook/dinov3-vit7b16-pretrain-lvd1689m", "revision": REVISION_7B},
}
SPEC = {
    "protocol": "dinov3_frozen_backbone_capacity_preflight_v1", "arms": ["hplus", "7b"],
    "view": "002.png", "split": "train", "pixel_protocol": "colmap_corner_v2",
    "seed": 20260926, "decoder_channels": 192, "classes": 5,
    "backbone_dtype": "bfloat16", "decoder_parameters_dtype": "float32", "amp": "cuda_bfloat16",
    "backbone_frozen": True, "adapters": False, "attention_implementation": "sdpa",
    "supervised_steps": 2, "input_shapes_hw": [[768, 768], [768, 1040]],
    "crop": "fixed center crop of prepared TRAIN002, no resize/augmentation",
    "context": "existing aligned_context_frame(short_side=768,augment=False,corner_v2); right padding ignored",
    "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": .01, "gradient_clip": 1.},
    "loss": "existing supervised_loss: edge-weighted CE + .5 Dice + .1 boundary BCE",
    "ema_decoder": "allocated throughout training; update min(.995,1-1/(step+1))",
    "inference": {"tile_size": 768, "stride": 512, "flip": True, "context_weight": .25,
                  "context_short_side": 768, "expected_complete_forward_calls": 14,
                  "decoder": "the same online decoder after two steps; no accuracy scoring"},
    "optional_native_inference": {"padded_hw": [992, 1328], "forward_calls": 1,
                                   "failure_policy": "report CUDA OOM as optional failure; no resize or retry"},
    "per_arm_process_seconds": 600, "nvidia_sampling_interval_seconds": .5,
    "minimum_output_free_bytes": 256 << 20, "checkpoint_or_probability_cache_written": False,
    "limits": "One TRAIN view/two decoder steps; capacity and timing only, not accuracy, convergence, or 6k training peak with consistency/LoRA.",
}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value, *, replace=False):
    path = Path(path)
    if replace:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temp.replace(path)
    else:
        with path.open("x") as handle:
            handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def safe_model_files(directory, declared):
    if not isinstance(declared, dict) or not declared:
        raise ValueError("Missing model file hashes")
    result = {}
    for filename, sha in declared.items():
        if (not isinstance(filename, str) or not filename or Path(filename).name != filename
                or filename in {".", ".."} or filename.endswith((".part", ".tmp"))
                or not isinstance(sha, str) or len(sha) != 64
                or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError("Unsafe filename or invalid SHA in model receipt")
        result[str((Path(directory) / filename).resolve())] = sha
    return result


def bind_model(directory, arm):
    """Never read weights until the completed ModelScope receipt is present."""
    directory = Path(directory).resolve()
    receipt_path = directory / "download_provenance.json"
    receipt = json.loads(receipt_path.read_text())
    expected = MODELS[arm]
    if (receipt.get("provider") != "ModelScope" or receipt.get("model_id") != expected["model_id"]
            or receipt.get("revision") != expected["revision"]):
        raise ValueError("ModelScope revision/provider differs")
    if arm == "7b" and (receipt.get("status") != "completed" or receipt.get("model_dir") != str(directory)):
        raise ValueError("7B download is not completed in the expected directory")
    declared = receipt.get("files_sha256")
    files = safe_model_files(directory, declared)
    if "config.json" not in declared:
        raise ValueError("Model receipt does not bind config")
    if arm == "7b":
        source_path = directory / "download_source_manifest.json"
        if digest(source_path) != receipt.get("source_manifest_sha256"):
            raise ValueError("7B source manifest is not SHA-bound")
        source = json.loads(source_path.read_text())
        if (source.get("model_id") != expected["model_id"] or source.get("revision") != expected["revision"]
                or set(source.get("files", {})) != set(declared)):
            raise ValueError("7B source manifest/revision/file set differs")
        for filename, sha in declared.items():
            entry = source["files"][filename]
            if entry.get("sha256") != sha or entry.get("size") != (directory / filename).stat().st_size:
                raise ValueError("7B source declaration/size differs")
        files[str(source_path)] = digest(source_path)
    for path, sha in files.items():
        if digest(path) != sha:
            raise ValueError(f"Model SHA mismatch: {path}")
    config = json.loads((directory / "config.json").read_text())
    if config.get("model_type") != "dinov3_vit" or config.get("patch_size") != 16:
        raise ValueError("Expected HF-format DINOv3 ViT/16 configuration")
    if arm == "7b":
        if "model.safetensors.index.json" not in declared:
            raise ValueError("7B shard index is not SHA-bound")
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        shards = set(index.get("weight_map", {}).values())
        if len(shards) < 2 or any(s not in declared or not s.endswith(".safetensors") for s in shards):
            raise ValueError("Index references undeclared or missing shards")
    elif "model.safetensors" not in declared:
        raise ValueError("H+ full weights are not SHA-bound")
    files[str(receipt_path)] = digest(receipt_path)
    return {"model_dir": str(directory), **expected, "files_sha256": files,
            "configuration": config, "download_receipt": str(receipt_path)}


def selected_view(manifest):
    from bridge_rgs.coordinates import CORNER, pixel_protocol
    if pixel_protocol(manifest) != CORNER:
        raise ValueError("Capacity preflight requires corner_v2 TRAIN data")
    selected = [v for v in manifest["views"] if v["name"] == SPEC["view"]]
    if (len(selected) != 1 or selected[0]["split"] != "train" or not selected[0].get("mask_path")
            or (selected[0]["height"], selected[0]["width"]) != (989, 1320)):
        raise ValueError("Require fixed labeled native TRAIN002")
    return selected[0]


def prepare(root, output, model_7b):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve existing preflight")
    manifest_path = root / "artifacts/prepared_corner_v2/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    view = selected_view(manifest)
    inputs = {str(p): digest(p) for p in (manifest_path, root / "uv.lock", Path(__file__).resolve())}
    for key in ("image_path", "mask_path", "valid_path"):
        path = (root / view[key]).resolve()
        inputs[str(path)] = digest(path)
    output.mkdir(parents=True)
    draft = {"status": "pending_completed_7b_download", "root": str(root), "output": str(output),
             "specification": SPEC, "manifest": str(manifest_path), "view": view,
             "models": {"hplus": {"model_dir": str(root / "models/dinov3-vith16plus"), **MODELS["hplus"]},
                        "7b": {"model_dir": str(Path(model_7b).resolve()), **MODELS["7b"]}},
             "input_hashes": inputs, "runner_sha256": digest(__file__),
             "pending_weights_opened": False, "execution_authorization": "Root review after completed-download lock; no GPU at prepare/lock"}
    write_json(output / "draft_plan.json", draft)
    return output / "draft_plan.json"


def lock(draft_path):
    draft_path = Path(draft_path).resolve()
    draft = json.loads(draft_path.read_text())
    output, root = Path(draft["output"]), Path(draft["root"])
    if draft["status"] != "pending_completed_7b_download" or draft["specification"] != SPEC:
        raise ValueError("Unrecognized pending preflight")
    if (output / "plan.json").exists() or (output / "source_snapshot").exists():
        raise FileExistsError("Preserve locked preflight")
    for path, sha in draft["input_hashes"].items():
        if digest(path) != sha:
            raise ValueError(f"Draft input/source changed: {path}")
    models = {arm: bind_model(draft["models"][arm]["model_dir"], arm) for arm in SPEC["arms"]}
    snapshot = output / "source_snapshot"
    shutil.copytree(root / "src/bridge_rgs", snapshot / "bridge_rgs",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    hashes = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))}
    for path, sha in hashes.items():
        source = root / "src" / path if path.startswith("bridge_rgs/") else Path(__file__)
        if digest(source) != sha:
            raise ValueError("Main source changed while locking")
    plan = {**draft, "status": "locked", "models": models, "source_snapshot": str(snapshot),
            "source_hashes": hashes, "draft_sha256": digest(draft_path)}
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def verify(plan, arm=None):
    if plan["status"] != "locked" or plan["specification"] != SPEC:
        raise ValueError("Only a locked fixed preflight may execute")
    snapshot = Path(plan["source_snapshot"])
    actual = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))}
    if actual != plan["source_hashes"] or Path(__file__).resolve() != snapshot / Path(__file__).name:
        raise ValueError("Use the frozen runner and package")
    for name in ("teacher", "coordinates", "teacher_adapters"):
        path = Path(importlib.import_module(f"bridge_rgs.{name}").__file__).resolve()
        if path != snapshot / "bridge_rgs" / f"{name}.py":
            raise ValueError(f"Wrong actual module import: {name}")
    for path, sha in plan["input_hashes"].items():
        if digest(path) != sha:
            raise ValueError(f"Input/source SHA mismatch: {path}")
    if arm is not None:
        for path, sha in plan["models"][arm]["files_sha256"].items():
            if digest(path) != sha:
                raise ValueError(f"Weight/config/index/source SHA mismatch: {path}")


def data_for_steps(view):
    """Fixed center crop and existing context transform; never any VAL row."""
    from bridge_rgs.teacher import ViewReader, aligned_context_frame
    image, mask, valid = ViewReader(5).read(view)
    h, w = image.shape[:2]
    y, x = (h - 768) // 2, (w - 768) // 2
    target = mask[y:y+768, x:x+768].copy()
    keep = valid[y:y+768, x:x+768].copy()
    target[~keep] = 255
    crop = (torch.from_numpy(image[y:y+768, x:x+768].copy()).permute(2, 0, 1).float() / 255,
            torch.from_numpy(target).long(), torch.from_numpy(keep))
    context = aligned_context_frame(image, mask, valid, 768, np.random.default_rng(SPEC["seed"]),
                                    augment=False, pixel_protocol="colmap_corner_v2")
    if [list(crop[0].shape[-2:]), list(context[0].shape[-2:])] != SPEC["input_shapes_hw"]:
        raise ValueError("Fixed input shapes changed")
    if any(bool((label[~keep] != 255).any()) for _, label, keep in (crop, context)):
        raise ValueError("Supervision validity and ignored padding differ")
    return image, (crop, context), {"crop_xy": [x, y], "crop_hw": [768, 768],
                                  "context_resized_hw": [768, 1025], "context_padded_hw": [768, 1040]}


def tensor_state_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        value = value.detach().cpu().contiguous()
        h.update(name.encode() + str(value.dtype).encode() + str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def cuda_memory():
    return {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved()}


def snapshot_forward_calls(captured):
    """Keep completed-phase metadata independent of the hook's reusable list."""
    return {"forward_calls": len(captured), "calls": copy.deepcopy(captured)}


def worker(plan_path, arm):
    started = time.monotonic()
    plan = json.loads(Path(plan_path).read_text())
    output = Path(plan["output"]) / arm
    receipt = {"status": "running", "arm": arm, "pid": os.getpid(), "plan_sha256": digest(plan_path),
               "specification": SPEC, "steps": [], "validation_accuracy_computed": False}
    def phase(name):
        write_json(output / "progress.json", {"phase": name, "elapsed_seconds": time.monotonic()-started}, replace=True)
    try:
        phase("verify_before_loading_including_all_weight_shards")
        verify(plan, arm)
        torch.set_num_threads(8)
        cv2.setNumThreads(8)
        torch.manual_seed(SPEC["seed"])
        np.random.seed(SPEC["seed"])
        from bridge_rgs.teacher import load_teacher, predict_image, supervised_loss, update_ema
        image, inputs, transforms = data_for_steps(plan["view"])
        receipt["data_transforms"] = transforms
        phase("load_bf16_backbone_and_fp32_decoder")
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        model = load_teacher(plan["models"][arm]["model_dir"], 5, 192, "cuda", pixel_profile="colmap_corner_v2")
        torch.cuda.synchronize()
        receipt["load_seconds"] = time.monotonic() - tick
        receipt["load_memory"] = cuda_memory()
        if (any(p.requires_grad or p.dtype != torch.bfloat16 for p in model.backbone.parameters())
                or any(p.dtype != torch.float32 for p in model.decoder.parameters())
                or model.backbone.config._attn_implementation != "sdpa"):
            raise ValueError("Backbone/decoder dtype, freeze or SDPA configuration differs")
        receipt["architecture"] = {"backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
                                   "decoder_parameters": sum(p.numel() for p in model.decoder.parameters()),
                                   "hidden_size": model.backbone.config.hidden_size,
                                   "layers": model.backbone.config.num_hidden_layers,
                                   "feature_layers": model.feature_layers,
                                   "sdpa_config": model.backbone.config._attn_implementation}
        phase("hash_backbone_state_before")
        backbone_hash = tensor_state_hash(model.backbone)
        ema = copy.deepcopy(model.decoder).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-4, weight_decay=.01)
        captured = []
        def hidden_hook(_module, _args, out):
            hidden = out.hidden_states
            captured.append({"count": len(hidden), "last_shape": list(hidden[-1].shape),
                             "dtype": str(hidden[-1].dtype),
                             "hidden_bytes_returned": sum(v.numel()*v.element_size() for v in hidden)})
        hook = model.backbone.register_forward_hook(hidden_hook)
        for step, (pixels, labels, valid) in enumerate(inputs, 1):
            phase(f"supervised_step_{step}")
            model.train()
            pixels, labels = pixels[None].cuda(), labels[None].cuda()
            old_decoder = tensor_state_hash(model.decoder)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            tick = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(pixels)
                loss, components = supervised_loss(prediction, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite supervised loss")
            loss.backward()
            if any(p.grad is not None for p in model.backbone.parameters()):
                raise ValueError("Frozen backbone received gradients")
            grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads) or not any(bool(g.abs().max() > 0) for g in grads):
                raise ValueError("Decoder gradient ineffective or nonfinite")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.)
            if not torch.isfinite(grad_norm):
                raise ValueError("Nonfinite decoder gradient norm")
            optimizer.step()
            update_ema(ema, model.decoder, min(.995, 1-1/(step+1)))
            torch.cuda.synchronize()
            seconds = time.monotonic()-tick
            memory = cuda_memory()
            if any(not torch.isfinite(p).all() for p in model.decoder.parameters()):
                raise ValueError("Updated decoder has nonfinite parameters")
            if old_decoder == tensor_state_hash(model.decoder):
                raise ValueError("Decoder did not update")
            receipt["steps"].append({"step": step, "input_shape": list(pixels.shape),
                "logit_shape": list(prediction["logits"].shape), "logit_dtype": str(prediction["logits"].dtype),
                "loss": float(loss.detach()), "components": components, "gradient_norm": float(grad_norm),
                "decoder_updated": True, "backbone_grad_none": True, "support_pixels": int(valid.sum()),
                "supervised_pixels": int((labels != 255).sum()),
                "seconds": seconds, "memory": memory, "hidden_states": captured[-1]})
            del prediction, loss, pixels, labels
        phase("complete_14_forward_inference")
        captured.clear()
        optimizer.zero_grad(set_to_none=True)
        # Retain EMA/Adam allocations: honest cost of evaluation inside this training process.
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.monotonic()
        probabilities, confidence = predict_image(model, image, tile_size=768, stride=512, flip=True,
                                                   context_weight=.25, context_short_side=768)
        torch.cuda.synchronize()
        inference_seconds = time.monotonic()-tick
        if len(captured) != 14 or probabilities.shape != (5, 989, 1320) or confidence.shape != (989, 1320):
            raise ValueError("Inference protocol/shape changed")
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(0), 1, atol=2e-5):
            raise ValueError("Invalid probability normalization")
        receipt["inference"] = {"seconds_complete_protocol": inference_seconds, **snapshot_forward_calls(captured),
                                 "memory": cuda_memory(), "shape": list(probabilities.shape),
                                 "probabilities_written": False, "accuracy_scored": False,
                                 "timing_includes_probability_stitching_and_CPU_transfer": True,
                                 "ema_optimizer_memory_retained": True}
        del probabilities, confidence
        phase("optional_one_native_padded_inference")
        captured.clear()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick = time.monotonic()
        native_input = native_output = None
        optional = {"input_shape": [1, 3, 992, 1328], "accuracy_scored": False, "retry": False}
        try:
            native = np.pad(image, ((0, 3), (0, 8), (0, 0)), mode="edge")
            native_input = torch.from_numpy(native).permute(2, 0, 1)[None].cuda().float() / 255
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                native_output = model(native_input)
            torch.cuda.synchronize()
            if len(captured) != 1 or native_output["logits"].shape != (1, 5, 992, 1328):
                raise ValueError("Optional native shape/forward count differs")
            optional.update(status="completed", logit_shape=list(native_output["logits"].shape),
                            hidden_states=captured[-1])
        except torch.cuda.OutOfMemoryError as error:
            optional.update(status="cuda_out_of_memory", error=str(error), forward_attempts=1)
        finally:
            optional.update(seconds=time.monotonic()-tick, memory=cuda_memory())
            del native_input, native_output
        receipt["optional_native_inference"] = optional
        if optional["status"] == "cuda_out_of_memory":
            torch.cuda.empty_cache()
        hook.remove()
        phase("verify_after_including_all_weight_shards")
        if tensor_state_hash(model.backbone) != backbone_hash:
            raise ValueError("Backbone tensor bytes changed")
        verify(plan, arm)
        if digest(plan_path) != receipt["plan_sha256"]:
            raise ValueError("Locked plan changed")
        receipt.update(status="completed", total_seconds=time.monotonic()-started,
                       backbone_state_sha256=backbone_hash, backbone_tensors_unchanged=True,
                       model_files_and_source_unchanged=True,
                       versions={"torch": torch.__version__, "cuda": torch.version.cuda,
                                 "transformers": importlib.import_module("transformers").__version__,
                                 "opencv": cv2.__version__})
    except Exception as error:
        receipt.update(status="failed", total_seconds=time.monotonic()-started,
                       error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "worker_receipt.json", receipt)


def gpu_idle():
    xml = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, check=True, timeout=10).stdout
    root = ET.fromstring(xml)
    for gpu in root.findall("gpu"):
        processes = gpu.find("processes")
        if processes is None or (processes.text or "").strip() in {"N/A", "Not Supported"}:
            raise ValueError("Cannot verify GPU process state")
    if not root.findall("gpu") or any(p.findtext("type") != "G" for p in root.findall(".//process_info")):
        raise ValueError("Require an idle GPU except desktop graphics")


def nvidia_sample(pid):
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    own = [row for row in processes.splitlines() if row.split(",")[0].strip() == str(pid)]
    return {"gpu_memory_used_total_utilization_csv": gpu, "worker_compute_memory_csv": own}


def execute(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    verify(plan)
    output = Path(plan["output"])
    if (output / "execution_receipt.json").exists():
        raise FileExistsError("No preflight rerun")
    if shutil.disk_usage(output).free < SPEC["minimum_output_free_bytes"]:
        raise ValueError("Insufficient result filesystem reserve")
    overall = {"status": "running", "plan_sha256": digest(plan_path), "arms": []}
    write_json(output / "execution_receipt.json", overall)
    try:
        for arm in SPEC["arms"]:
            gpu_idle()
            arm_output = output / arm
            arm_output.mkdir()
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=plan["source_snapshot"],
                       OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1")
            started = time.monotonic()
            samples = []
            timed_out = False
            with (arm_output / "process.log").open("x") as stream:
                process = subprocess.Popen([sys.executable, __file__, "--worker", arm, "--plan", str(plan_path)],
                                           stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=plan["root"])
                try:
                    while process.poll() is None:
                        if time.monotonic()-started >= SPEC["per_arm_process_seconds"]:
                            timed_out = True
                            process.kill()
                            break
                        sample = nvidia_sample(process.pid)
                        sample["seconds"] = time.monotonic()-started
                        progress = arm_output / "progress.json"
                        sample["phase"] = json.loads(progress.read_text())["phase"] if progress.exists() else "starting"
                        samples.append(sample)
                        time.sleep(SPEC["nvidia_sampling_interval_seconds"])
                finally:
                    if process.poll() is None:
                        process.kill()
                    exit_code = process.wait()
            write_json(arm_output / "nvidia_samples.json", {"interval_seconds": .5, "samples": samples,
                       "limits": "Sampled process/device memory, not a continuous physical peak; torch allocator peaks reported separately."})
            record = {"arm": arm, "pid": process.pid, "exit_code": exit_code, "timed_out": timed_out,
                      "subprocess_wall_seconds": time.monotonic()-started}
            worker_receipt = arm_output / "worker_receipt.json"
            if worker_receipt.exists():
                record["worker_receipt_sha256"] = digest(worker_receipt)
                record["worker_status"] = json.loads(worker_receipt.read_text())["status"]
            overall["arms"].append(record)
            if timed_out or exit_code != 0 or record.get("worker_status") != "completed":
                raise RuntimeError(f"Preflight arm failed without retry: {arm}")
            gpu_idle()
        overall["status"] = "completed"
    except Exception as error:
        overall.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "execution_receipt.json", overall, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--lock", type=Path)
    action.add_argument("--run", type=Path)
    action.add_argument("--worker", choices=SPEC["arms"])
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("/mnt/data/SHM2026/preflight/dinov3_capacity_v1"))
    parser.add_argument("--model-7b", type=Path, default=Path("/mnt/data/SHM2026/models/dinov3-vit7b16"))
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.root, args.output, args.model_7b))
    elif args.lock:
        print(lock(args.lock))
    elif args.run:
        execute(args.run)
    else:
        if args.plan is None:
            parser.error("--worker requires --plan")
        worker(args.plan, args.worker)


if __name__ == "__main__":
    main()
