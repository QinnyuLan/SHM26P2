"""Matched experiment plans preserve the selected model and training permissions."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

from bridge_rgs.train import (
    initialize_density_extras,
    initialize_view_sampler,
    sample_training_view,
)


def generator_module():
    path = Path(__file__).resolve().parents[1] / "scripts/make_h2_configs.py"
    spec = importlib.util.spec_from_file_location("make_h2_configs_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("routing", [True, False])
def test_h2_arms_preserve_routing_and_only_change_declared_factors(routing):
    module = generator_module()
    base = {"manifest": "manifest.json", "refiner_field_grad": routing,
            "refiner": {"type": "multiscale", "channels": 64}, "semantic_weight": 1.,
            "class_weight_power": .25, "raw_class_weight_power": .5, "eval_max_views": 8}
    inherited = module.inherited_config({"config": base})
    arms = module.make_variants(inherited, "chosen.pt", "fixed_pseudo", "runs/h2")
    for config in arms.values():
        assert config["refiner_field_grad"] == routing
        assert config["raw_class_weight_power"] == .5
        assert config["warmstart_reset_refiner"] is False
        assert config["steps"] == 3000 and config["pseudo_weight"] == 0
        assert config["local_teacher_kd"] and config["supervised_teacher_priority"]
        assert config["train_labeled_only"] is False and config["independent_view_rng"]
        assert "eval_max_views" not in config
    gt, no_fusion, fixed, projection = arms.values()
    differences = lambda a, b: {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
    assert differences(gt, no_fusion) == {"output", "pseudo_dir", "pseudo_refiner_weight"}
    assert differences(no_fusion, fixed) == {"output", "multiview_fusion", "multiview_weight"}
    assert differences(fixed, projection) == {"output", "projection_uncertainty"}
    assert all(config["pseudo_refiner_weight"] == .1 for config in (no_fusion, fixed, projection))
    assert "eval_max_views" in base  # The inherited source dictionary was not mutated.
    assert len(module.make_variants(inherited, "x.pt", "p", "r", include_gt=False)) == 3


def test_generation_rejects_hidden_routing_schema_and_split_changes():
    module = generator_module()
    saved = {"config": {"manifest": "one.json", "refiner_field_grad": False,
                        "refiner": {"type": "multiscale", "channels": 64}, "semantic_weight": 1}}
    for override in ({"refiner_field_grad": True}, {"manifest": "different.json"},
                     {"refiner": {"type": "legacy"}}):
        with pytest.raises(ValueError):
            module.inherited_config(saved, override)


def test_planned_budget_matches_real_independent_view_sampler():
    module = generator_module()
    views = [{"name": str(i), "split": "train", "mask_path": "label" if i < 259 else None}
             for i in range(350)]
    plan = module.continuation_budget({"views": views}, 42)
    extras = initialize_density_extras(1, 350, "cpu")
    generator = initialize_view_sampler({"independent_view_rng": True, "seed": 42}, extras)
    sequence = [sample_training_view(list(range(350)), "shuffle", extras, generator)
                for _ in range(3000)]
    assert plan["labeled_steps"] == sum(index < 259 for index in sequence)
    assert plan["unlabeled_steps"] == sum(index >= 259 for index in sequence)
    assert plan["min_view_visits"] == 8 and plan["max_view_visits"] == 9
    assert plan["fusion_refreshes_in_fusion_arms"] == 15
    assert plan["fusion_targets_used_by_loss"] == len(plan["acceptance_log_steps"]) == 14


def test_generator_writes_only_new_configs_and_never_modifies_selected_checkpoint(tmp_path, monkeypatch):
    module = generator_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    manifest = {"views": [{"name": str(i), "split": "train",
                           "mask_path": "label" if i < 259 else None} for i in range(350)]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    source = tmp_path / "chosen.pt"
    torch.save({"config": {"manifest": "manifest.json", "semantic_weight": 1.,
                           "refiner_field_grad": False, "refiner": {"type": "multiscale"}},
                "model": {"untouched": torch.ones(3)}}, source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(FileNotFoundError, match="provenance"):
        module.generate(source, "future_pseudo", "generated", "runs/h2")
    assert not (tmp_path / "generated").exists()
    plan = module.generate(source, "future_pseudo", "generated", "runs/h2",
                           allow_pending_pseudo=True)
    assert plan["training_started"] is False
    assert not (tmp_path / "runs").exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert len(list((tmp_path / "generated").glob("*.yaml"))) == 4
    config = yaml.safe_load((tmp_path / plan["configs"]["01_teacher_no_fusion"]).read_text())
    assert config["refiner_field_grad"] is False
    assert "--source-snapshot" in plan["run_commands"]["03_teacher_projection_sampling"]
    with pytest.raises(FileExistsError, match="overwrite"):
        module.generate(source, "future_pseudo", "generated", "runs/h2", allow_pending_pseudo=True)
