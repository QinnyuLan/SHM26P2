import copy

import pytest
import torch

from bridge_rgs.semantic_schedule import (
    SemanticLRSchedule,
    normalize_semantic_schedule,
    semantic_multiplier,
    validate_semantic_schedule_resume,
)


def make_optimizers():
    parameters = [torch.nn.Parameter(torch.tensor([float(i + 1)])) for i in range(5)]
    optimizers = {
        "sem_features": torch.optim.Adam([parameters[0]], lr=.01),
        "heads": torch.optim.Adam([
            {"params": [parameters[1]], "lr": .001},
            {"params": [parameters[2]], "lr": .0003},
            {"params": [parameters[3]], "lr": .123},
        ]),
        "sh0": torch.optim.Adam([parameters[4]], lr=.0025),
    }
    return parameters, optimizers


def config(kind="cosine", steps=7, final=.1):
    return {"steps": steps, "semantic_lr_schedule": {"type": kind, "final_multiplier": final}}


def update(parameters, optimizers, schedule, step):
    schedule.apply(step)
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    sum((parameter.square() * (step + 1)).sum() for parameter in parameters).backward()
    for optimizer in optimizers.values():
        optimizer.step()


def test_default_does_not_modify_groups_or_log():
    _, optimizers = make_optimizers()
    before = [{k: v for k, v in group.items() if k != "params"}
              for optimizer in optimizers.values() for group in optimizer.param_groups]
    schedule = SemanticLRSchedule(optimizers, {})
    schedule.initialize()
    assert schedule.apply(999) == {}
    after = [{k: v for k, v in group.items() if k != "params"}
             for optimizer in optimizers.values() for group in optimizer.param_groups]
    assert before == after


def test_cosine_endpoints_monotonic_and_only_semantic_groups():
    _, optimizers = make_optimizers()
    schedule = SemanticLRSchedule(optimizers, config())
    schedule.initialize()
    logs = [schedule.apply(step) for step in range(1, 8)]
    multipliers = [log["semantic_lr_multiplier"] for log in logs]
    assert multipliers[0] == 1. and multipliers[-1] == .1
    assert multipliers == sorted(multipliers, reverse=True)
    for log in logs:
        for name, base in [("features", .01), ("classifier", .001), ("refiner", .0003)]:
            assert log[f"semantic_lr_{name}"] == base * log["semantic_lr_multiplier"]
        assert log["semantic_lr_background"] == .123
    assert "semantic_base_lr" not in optimizers["heads"].param_groups[2]
    assert optimizers["sh0"].param_groups[0]["lr"] == .0025


def test_constant_is_exact_default_optimization():
    a, oa = make_optimizers()
    b, ob = make_optimizers()
    sa, sb = SemanticLRSchedule(oa, {}), SemanticLRSchedule(ob, config("constant"))
    sa.initialize()
    sb.initialize()
    for step in range(1, 8):
        update(a, oa, sa, step)
        update(b, ob, sb, step)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_resume_reproduces_uninterrupted_adam_and_background():
    a, oa = make_optimizers()
    sa = SemanticLRSchedule(oa, config())
    sa.initialize()
    for step in range(1, 4):
        update(a, oa, sa, step)
    b, ob = make_optimizers()
    sb = SemanticLRSchedule(ob, config())  # Capture bases before loading decayed rates.
    with torch.no_grad():
        for x, y in zip(a, b, strict=True):
            y.copy_(x)
    for name in oa:
        ob[name].load_state_dict(copy.deepcopy(oa[name].state_dict()))
    sb.initialize(resume_step=3)
    for step in range(4, 8):
        update(a, oa, sa, step)
        update(b, ob, sb, step)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    for name in oa:
        for index, state in oa[name].state_dict()["state"].items():
            for key, value in state.items():
                assert torch.equal(value, ob[name].state_dict()["state"][index][key])


@pytest.mark.parametrize("changed", [{}, config("constant"), config(steps=8), config(final=.2)])
def test_strict_resume_rejects_schedule_or_horizon_changes(changed):
    with pytest.raises(ValueError, match="Strict resume"):
        validate_semantic_schedule_resume(config(), changed)
    validate_semantic_schedule_resume(config(), config())
    validate_semantic_schedule_resume({"steps": 5}, {"steps": 100})  # Legacy stages unchanged.


@pytest.mark.parametrize("value", [-.1, 0, 1.1, float("nan"), True, "0.1"])
def test_reject_bad_final_multiplier(value):
    with pytest.raises(ValueError, match="final_multiplier"):
        normalize_semantic_schedule(config(final=value))


def test_reject_bad_spec_and_steps():
    for cfg in [config("linear"), config(steps=1), config(steps=3.5), config(steps=True),
                {"semantic_lr_schedule": "cosine"}, {"semantic_lr_schedule": {"unknown": 3}}]:
        with pytest.raises(ValueError):
            normalize_semantic_schedule(cfg)
    for step in [0, 8, 2.5, True]:
        with pytest.raises(ValueError):
            semantic_multiplier(normalize_semantic_schedule(config()), step)


def test_corrupt_saved_rate_is_rejected_before_mutation():
    _, optimizers = make_optimizers()
    schedule = SemanticLRSchedule(optimizers, config())
    schedule.initialize()
    schedule.apply(3)
    optimizers["heads"].param_groups[1]["lr"] *= .5
    before = [group["lr"] for group in schedule.groups]
    with pytest.raises(ValueError, match="does not match"):
        schedule.initialize(resume_step=3)
    assert before == [group["lr"] for group in schedule.groups]


def test_background_update_is_identical_under_cosine():
    a, oa = make_optimizers()
    b, ob = make_optimizers()
    sa, sb = SemanticLRSchedule(oa, config("constant")), SemanticLRSchedule(ob, config())
    sa.initialize()
    sb.initialize()
    for step in range(1, 8):
        update(a, oa, sa, step)
        update(b, ob, sb, step)
    assert all(torch.equal(a[index], b[index]) for index in (3, 4))
    assert not torch.equal(a[0], b[0])
