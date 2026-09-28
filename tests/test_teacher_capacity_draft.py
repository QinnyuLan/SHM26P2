"""CPU-only checks for capacity isolation, stage endpoints and fixed adoption gate."""
import copy
import importlib.util
from pathlib import Path

import numpy as np

from bridge_rgs.teacher_domains import domain_schedule

spec = importlib.util.spec_from_file_location(
    "teacher_capacity_draft", Path(__file__).parents[1] / "scripts/make_teacher_capacity_draft.py"
)
draft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(draft)


def pair(all5=.003, lower=.0001, foreground=0., cable=0.):
    return {"metrics": {
        "miou_all": {"difference": all5, "paired_view_bootstrap_95_interval": [lower, .009]},
        "miou_foreground": {"difference": foreground}, "stay_cable_iou": {"difference": cable},
    }}


def test_capacities_have_identical_effective_hyperparameters():
    configs = draft.make_configs()
    for stage in ("real", "render_adapt"):
        hplus, vit7b = (copy.deepcopy(configs[f"{capacity}_{stage}"])
                        for capacity in ("hplus", "vit7b"))
        assert hplus["manifest"] == vit7b["manifest"]
        assert hplus.pop("model_dir") != vit7b.pop("model_dir")
        assert hplus.pop("output_dir") != vit7b.pop("output_dir")
        if stage == "render_adapt":
            assert hplus["config"].pop("warmstart_checkpoint") != vit7b["config"].pop("warmstart_checkpoint")
        assert hplus == vit7b


def test_fixed_endpoints_and_absolute_stage_lineage():
    configs = draft.make_configs()
    for capacity in ("hplus", "vit7b"):
        real, render = (configs[f"{capacity}_{stage}"] for stage in ("real", "render_adapt"))
        for value in (real, render):
            assert all(Path(value[key]).is_absolute() for key in ("manifest", "model_dir", "output_dir"))
            config = value["config"]
            assert config["independent_augmentation_rng"]
            assert config["checkpoint_every"] == 1000
            assert config["eval_every"] == config["steps"]
            assert config["adapter_rank"] == 0
            assert config["seed"] == 20260926
            assert config["channels"] == 192 and config["crop_size"] == 768
        assert real["config"]["steps"] == 6000
        assert real["config"]["warmstart_checkpoint"] == ""
        assert real["config"]["consistency_start"] == 1000
        assert real["config"]["consistency_weight"] == .5
        assert render["config"]["steps"] == 2000
        assert render["config"]["consistency_start"] > 2000
        assert render["config"]["render_mix_probability"] == .5
        assert render["config"]["warmstart_checkpoint"] == str(Path(real["output_dir"]) / "last.pt")
        assert "best.pt" not in str(real) + str(render)


def test_domain_schedule_exact_and_does_not_change_numpy_view_rng():
    a, b = np.random.default_rng(20260926), np.random.default_rng(20260926)
    schedule = domain_schedule(2000, .5, 20260926)
    assert int(schedule.sum()) == 1000
    np.testing.assert_array_equal(a.integers(259, size=2000), b.integers(259, size=2000))
    np.testing.assert_array_equal(schedule, domain_schedule(2000, .5, 20260926))


def test_adoption_gate_accepts_exact_030pp_boundary_and_requires_both_references():
    assert draft.adoption_gate(pair(), pair(), True)["passed"]
    assert not draft.adoption_gate(pair(all5=.002999), pair(), True)["passed"]
    assert not draft.adoption_gate(pair(), pair(all5=.002999), True)["passed"]
    assert not draft.adoption_gate(pair(lower=0), pair(), True)["passed"]
    assert not draft.adoption_gate(pair(), pair(lower=0), True)["passed"]


def test_adoption_gate_preserves_foreground_cable_and_all_rgb():
    assert not draft.adoption_gate(pair(), pair(), False)["passed"]
    for key in ("foreground", "cable"):
        bad = pair(**{key: -1e-12})
        assert not draft.adoption_gate(bad, pair(), True)["passed"]
        assert not draft.adoption_gate(pair(), bad, True)["passed"]
