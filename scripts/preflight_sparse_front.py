"""Two real updates per arm: default-off render check and opacity-only invariants."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=Path("configs/generated_sparse_front_pair"))
    parser.add_argument("--output", type=Path, default=Path("runs/sparse_front_preflight"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a preflight")
    torch.set_num_threads(8)
    args.output.mkdir(parents=True)
    snapshot = args.output / "source_snapshot"
    shutil.copytree(Path(__file__).resolve().parents[1] / "src/bridge_rgs", snapshot / "bridge_rgs",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    sys.path.insert(0, str(snapshot.resolve()))
    from bridge_rgs import ray_termination
    from bridge_rgs import train as training

    originals = training.apply_parameter_scope, ray_termination.SparseFrontSupport.loss, ray_termination.render_front_mass
    report = {"source_hashes": {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.py"))},
              "script_sha256": digest(__file__), "arms": {}, "status": "running"}
    states = []
    for name in ("00_rgb_ed", "01_rgb_ed_front"):
        config = yaml.safe_load((args.configs / (name+".yaml")).read_text())
        config.update(output=str(args.output/name), steps=2, log_every=1, save_every=2, eval_every=0)
        initial = torch.load(config["warmstart"], map_location="cpu", mmap=True, weights_only=False)
        record = {"front_render_calls": 0, "front_gradients": [],
                  "initial_checkpoint_sha256": digest(config["warmstart"]),
                  "manifest_sha256": digest(config["manifest"])}
        report["arms"][name] = record

        def apply_and_record(scene, options, record=record):
            originals[0](scene, options)
            record["trainable"] = [key for key, value in scene.named_parameters() if value.requires_grad]
            if record["trainable"] != ["splats.opacity_logits"]:
                raise AssertionError("Preflight permits only opacity updates")

        def gradient_probe(self, scene, *params, record=record):
            loss, stats = originals[1](self, scene, *params)
            if loss is not None:
                gradient = torch.autograd.grad(loss, scene.splats["opacity_logits"], retain_graph=True)[0]
                record["front_gradients"].append({"norm": float(gradient.norm()),
                                                   "nonzero": int(gradient.count_nonzero()),
                                                   "finite": bool(torch.isfinite(gradient).all()),
                                                   "stats": stats})
                if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
                    raise AssertionError("Front component needs finite nonzero opacity gradient")
            return loss, stats

        def counted_render(*params, record=record, **kwargs):
            record["front_render_calls"] += 1
            return originals[2](*params, **kwargs)

        training.apply_parameter_scope = apply_and_record
        ray_termination.SparseFrontSupport.loss = gradient_probe
        ray_termination.render_front_mass = counted_render
        try:
            result = training.train(config)
        finally:
            training.apply_parameter_scope, ray_termination.SparseFrontSupport.loss, ray_termination.render_front_mass = originals
        state = torch.load(result, map_location="cpu", mmap=True, weights_only=False)
        states.append(state)
        if set(state["model"]) != set(initial["model"]):
            raise AssertionError("Model schema changed")
        changed = [key for key in state["model"] if not torch.equal(state["model"][key], initial["model"][key])]
        if changed != ["splats.opacity_logits"]:
            raise AssertionError(f"Unexpected changed model state: {changed}")
        if not torch.equal(state["training_cameras"], initial["training_cameras"]):
            raise AssertionError("Camera parameters changed")
        if any(state[key] != initial[key] for key in ("feature_dim", "sh_degree", "refiner_config", "scene_scale")):
            raise AssertionError("Architecture metadata changed")
        if any(state["optimizers"][key]["state"] for key in state["optimizers"] if key != "opacity_logits"):
            raise AssertionError("Frozen parameters accumulated optimizer state")
        if state["stats"]["sh_degree"] != 3:
            raise AssertionError("Opacity polish must retain full SH3")
        expected = 2 if config["sparse_front_weight"] > 0 else 0
        if record["front_render_calls"] != expected or len(record["front_gradients"]) != expected:
            raise AssertionError("Off must render zero times, on exactly once per step")
        record.update(status="passed", changed_model_keys=changed, cameras_bitwise_unchanged=True,
                      metadata_bitwise_unchanged=True, step=state["step"], checkpoint_sha256=digest(result))
    a, b = [state["density_state"]["extras"] for state in states]
    if any(a[key] != b[key] for key in ("sampler_order", "sampler_cursor", "sampler_rng_state")):
        raise AssertionError("Paired two-step view sampler states differ")
    if torch.equal(states[0]["model"]["splats.opacity_logits"], states[1]["model"]["splats.opacity_logits"]):
        raise AssertionError("Positive front loss must change the actual update")
    report.update(status="passed", paired_sampler_state_identical=True,
                  default_off_extra_render_calls=0, positive_front_changes_opacity_update=True)
    (args.output/"preflight_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": "passed", "arms": list(report["arms"])}))


if __name__ == "__main__":
    main()
