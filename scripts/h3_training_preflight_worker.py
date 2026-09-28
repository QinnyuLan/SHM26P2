"""Run the real trainer with a read-only check at optimizer construction."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import torch
import yaml


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    modules = {}
    for name in ("bridge_rgs", "bridge_rgs.model", "bridge_rgs.train",
                 "bridge_rgs.refinement", "bridge_rgs.depth_moments"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(args.snapshot.resolve()), (name, str(path))
        modules[name] = {"path": str(path), "sha256": sha(path)}
    from bridge_rgs.model import GaussianScene
    from bridge_rgs.train import train

    config = yaml.safe_load(args.config.read_text())
    assert config["steps"] == 2 and config["parameter_scope"] == "refiner_only"
    initial = torch.load(config["warmstart"], map_location="cpu", weights_only=False)
    original = GaussianScene.optimizers
    hook_calls = []

    def audited_optimizers(scene, *arguments, **keywords):
        state = scene.state_dict()
        assert set(state) - set(initial["model"]) == {
            "refiner.moment_detail.weight", "refiner.moment_half.weight"}
        assert all(torch.equal(value, state[key].detach().cpu()) for key, value in initial["model"].items())
        assert all(torch.count_nonzero(state[key]) == 0
                   for key in ("refiner.moment_detail.weight", "refiner.moment_half.weight"))
        assert all(parameter.requires_grad == name.startswith("refiner.")
                   for name, parameter in scene.named_parameters())
        optimizers = original(scene, *arguments, **keywords)
        assert all(not optimizer.state for optimizer in optimizers.values())
        assert optimizers["heads"].param_groups[1]["lr"] == .0003
        hook_calls.append(1)
        record = {"status": "passed", "observation_point": "Real trainer optimizer construction, after warmstart and parameter scope, before first forward/update",
                  "all_old_model_tensors_exact": True, "all_new_projection_weights_zero": True,
                  "only_refiner_requires_grad": True, "fresh_optimizer_states_empty": True,
                  "actual_imports": modules, "old_model_tensor_count": len(initial["model"]),
                  "new_projection_tensor_count": 2, "config_sha256": sha(args.config),
                  "initial_checkpoint_sha256": sha(config["warmstart"])}
        path = Path(config["output"])/"warmstart_initialization_audit.json"
        path.write_text(json.dumps(record, indent=2)+'\n')
        return optimizers

    GaussianScene.optimizers = audited_optimizers
    try:
        train(config)
        assert len(hook_calls) == 1
    finally:
        GaussianScene.optimizers = original


if __name__ == "__main__":
    main()
