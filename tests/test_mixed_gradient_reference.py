"""CPU contracts for the isolated, capped RGB140-inspired reference."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from bridge_rgs.densification import apply_densification, quaternion_to_matrix
from bridge_rgs.mixed_gradient_reference import (
    MixedGradientConfig,
    MixedGradientState,
    mixed_gradient_densify,
    reference_post_step,
    standard_opacity_reset,
    validate_reference_training,
)
from bridge_rgs.train import ImageCache, density_checkpoint_state


def field(n=4):
    return nn.ParameterDict({
        "means": nn.Parameter(torch.arange(n * 3).reshape(n, 3).float()),
        "quats": nn.Parameter(torch.tensor([1., 0., 0., 0.]).repeat(n, 1)),
        "log_scales": nn.Parameter(torch.full((n, 3), .005).log()),
        "opacity_logits": nn.Parameter(torch.zeros(n)),
        "sh0": nn.Parameter(torch.arange(n * 3).reshape(n, 1, 3).float()),
    })


def window(n, cap=20):
    state = MixedGradientState(n, "cpu", MixedGradientConfig(max_gaussians=cap))
    state.observations.fill_(1)
    return state


def test_real_signed_backward_and_absolute_statistics_use_separate_visible_means():
    state = window(3)
    state.observations.zero_()
    means2d = torch.zeros(1, 3, 2, requires_grad=True) * 1
    info = {"means2d": means2d, "width": 100, "height": 200,
            "radii": torch.tensor([[[1, 2], [2, 3], [1, 0]]]), "n_cameras": 1}
    state.retain_grad(info, 1)
    # Two opposing per-pixel contributions cancel in the signed derivative.
    ((means2d * .002).sum() + (means2d * -.0018).sum()).backward()
    means2d.absgrad = torch.full_like(means2d, .0038)
    state.accumulate(info, 1)
    expected_signed = torch.tensor([.01, .02]).norm()
    expected_absolute = torch.tensor([.19, .38]).norm()
    torch.testing.assert_close(state.signed_sum, torch.tensor([expected_signed, expected_signed, 0.]))
    torch.testing.assert_close(state.absolute_sum, torch.tensor([expected_absolute, expected_absolute, 0.]))
    assert state.observations.tolist() == [1, 1, 0]
    state.accumulate(info, 2)
    torch.testing.assert_close(state.signed_sum[:2] / state.observations[:2], expected_signed.expand(2))


def test_mixed_gradient_dispatch_does_not_clone_small_abs_only_parent():
    params, state = field(), window(4)
    with torch.no_grad():
        params["log_scales"][2:] = torch.tensor(.02).log()
    state.signed_sum[:] = torch.tensor([.0001, .0003, .0001, .0005])
    state.absolute_sum[:] = torch.tensor([.004, .004, .0005, .0003])
    result, diagnostics = mixed_gradient_densify(params, state, 1., 600)
    assert result.duplicate_count == result.split_count == 1
    assert result.source_indices[result.reset_moments].tolist() == [1, 2, 2]
    assert diagnostics["density_clone_candidates"] == diagnostics["density_split_candidates"] == 1
    # Equality is not above threshold, and no observations must not activate.
    state.signed_sum.fill_(.0002)
    state.absolute_sum.fill_(.0004)
    result, _ = mixed_gradient_densify(params, state, 1., 600)
    assert result.duplicate_count == result.split_count == 0


def test_capacity_ranks_by_each_threshold_and_breaks_ties_by_original_id():
    params, state = field(), window(4, cap=6)
    with torch.no_grad():
        params["log_scales"][1] = torch.tensor(.02).log()
    state.signed_sum[:] = torch.tensor([.0008, 0, .0006, .0006])
    state.absolute_sum[1] = .0012  # score 3, tying small IDs 2/3, despite larger raw gradient.
    result, diagnostics = mixed_gradient_densify(params, state, 1., 600)
    assert result.source_indices[result.reset_moments].tolist() == [0, 1, 1]
    assert len(result.source_indices) == 6
    assert diagnostics["density_cap_rejected"] == 2


def test_actual_500k_cap_does_not_grow_or_recycle_when_full():
    params = field(500_000)
    state = window(500_000, cap=500_000)
    state.signed_sum.fill_(.001)
    result, stats = mixed_gradient_densify(params, state, 1., 600)
    assert len(result.source_indices) == 500_000
    assert result.duplicate_count == result.split_count == result.pruned_count == 0
    assert stats["density_cap_rejected"] == 500_000
    with pytest.raises(ValueError, match="exceeds"):
        MixedGradientState(500_001, "cpu")


def test_split_uses_rotated_gaussian_noise_all_axis_shrink_and_preserved_opacity():
    params, state = field(2), window(2)
    with torch.no_grad():
        params["log_scales"][1] = torch.tensor([.02, .03, .04]).log()
        params["quats"][1] = torch.tensor([2**-.5, 0., 0., 2**-.5])
    state.absolute_sum[1] = .001
    generator = torch.Generator().manual_seed(71)
    noise = torch.randn(2, 1, 3, generator=generator)
    expected = params["means"][1] + torch.einsum(
        "ij,bnj->bni", quaternion_to_matrix(params["quats"][1]),
        noise * params["log_scales"][1].exp())[:, 0]
    generator.manual_seed(71)
    result, _ = mixed_gradient_densify(params, state, 1., 600, generator)
    torch.testing.assert_close(result.tensors["means"][-2:], expected)
    torch.testing.assert_close(result.tensors["log_scales"][-2:].exp(),
                               (params["log_scales"][1].exp() / 1.6).expand(2, 3))
    torch.testing.assert_close(result.tensors["opacity_logits"][-2:], params["opacity_logits"][1].expand(2))


def test_growth_precedes_prune_and_never_borrows_future_pruned_capacity():
    params, state = field(4), window(4, cap=5)
    state.signed_sum.fill_(.001)
    with torch.no_grad():
        params["opacity_logits"][3] = -10
    result, _ = mixed_gradient_densify(params, state, 1., 600)
    assert result.duplicate_count == 1
    assert result.pruned_count == 1
    assert len(result.source_indices) == 4  # did not pre-borrow the later prune's slot.
    assert 3 not in result.source_indices


def test_scale_pruning_after_first_reset_applies_to_children_after_shrink():
    params, state = field(3), window(3)
    with torch.no_grad():
        params["log_scales"][1:] = torch.tensor(.12).log()
    state.absolute_sum[1] = .001
    result, _ = mixed_gradient_densify(params, state, 1., 3000)
    assert result.pruned_count == 0
    result, _ = mixed_gradient_densify(params, state, 1., 3100)
    assert result.pruned_count == 1
    assert result.source_indices.tolist() == [0, 1, 1]


def test_grow_reset_boundary_schedule_is_explicit_and_not_bitwise_comparison():
    policy = MixedGradientConfig()
    grows = [step for step in range(1, 30001) if policy.grows_at(step)]
    resets = [step for step in range(1, 30001) if policy.resets_at(step)]
    assert grows == list(range(600, 15000, 100))
    assert resets == [3000, 6000, 9000, 12000]
    assert not policy.grows_at(500) and not policy.grows_at(15000)
    assert not policy.resets_at(15000)


def test_adam_migration_preserves_clone_parent_resets_new_rows_and_can_step():
    params, state = field(), window(4)
    with torch.no_grad():
        params["log_scales"][1] = torch.tensor(.02).log()
    opt = torch.optim.Adam(params.parameters(), lr=1e-4, amsgrad=True)
    sum(value.square().sum() for value in params.values()).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    old = {key: value for key, value in params.items()}
    old_states = {key: {name: value.clone() for name, value in opt.state[param].items()}
                  for key, param in old.items()}
    state.signed_sum[0] = .001
    state.absolute_sum[1] = .001
    result, _ = mixed_gradient_densify(params, state, 1., 600)
    apply_densification(params, result, opt)
    for key, param in params.items():
        assert old[key] not in opt.state
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            expected = old_states[key][name][result.source_indices].clone()
            expected[result.reset_moments] = 0
            torch.testing.assert_close(opt.state[param][name], expected, rtol=0, atol=0)
        torch.testing.assert_close(opt.state[param]["step"], old_states[key]["step"], rtol=0, atol=0)
    sum(value.square().sum() for value in params.values()).backward()
    opt.step()
    assert all(torch.isfinite(value).all() for value in params.values())


def test_reset_clears_even_unchanged_opacity_moments_preserving_step_and_parameter():
    logits = nn.Parameter(torch.tensor([-7., 0., 2.]))
    opt = torch.optim.Adam([logits], lr=.025, amsgrad=True)
    logits.sum().backward()
    opt.step()
    step = opt.state[logits]["step"].clone()
    lowest = logits[0].detach().clone()
    assert standard_opacity_reset(logits, opt) == 2
    assert opt.param_groups[0]["params"][0] is logits
    torch.testing.assert_close(logits[0], lowest, rtol=0, atol=0)
    assert (logits.sigmoid() <= .010001).all()
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        assert not opt.state[logits][key].any()
    torch.testing.assert_close(opt.state[logits]["step"], step, rtol=0, atol=0)


def test_checkpoint_dual_window_roundtrip_and_old_window_schema_unchanged():
    state = window(3)
    state.signed_sum[:] = torch.tensor([.01, .02, .03])
    state.absolute_sum[:] = .1
    zeros = torch.zeros(3)
    old = density_checkpoint_state(zeros, zeros.long(), zeros.long(), {}, None)
    assert "mixed_gradient" not in old
    saved = density_checkpoint_state(zeros, zeros.long(), zeros.long(), {}, None, mixed_gradient=state)
    restored = MixedGradientState(3, "cpu", state.policy)
    restored.load_state_dict(saved["mixed_gradient"])
    torch.testing.assert_close(restored.signed_sum, state.signed_sum, rtol=0, atol=0)
    torch.testing.assert_close(restored.absolute_sum, state.absolute_sum, rtol=0, atol=0)
    torch.testing.assert_close(restored.observations, state.observations, rtol=0, atol=0)
    with pytest.raises(ValueError, match="policy changed"):
        MixedGradientState(3, "cpu", replace(state.policy, signed_clone_threshold=.003)).load_state_dict(saved["mixed_gradient"])
    with pytest.raises(ValueError, match="shape"):
        MixedGradientState(2, "cpu", state.policy).load_state_dict(saved["mixed_gradient"])


def test_post_step_grows_before_reset_and_dual_window_resets_without_pause():
    params, state = field(2), window(2)
    state.signed_sum.fill_(.001)
    optimizers = {key: torch.optim.Adam([value], lr=.025) for key, value in params.items()}
    sum(value.square().sum() for value in params.values()).backward()
    for opt in optimizers.values():
        opt.step()
    new_state, result, stats = reference_post_step(params, optimizers, state, 1., 3000)
    assert len(params["means"]) == 4 and result.duplicate_count == 2
    assert stats["opacity_reset_count"] == 4  # New rows exist before opacity reset.
    assert stats["opacity_reset_step"] == 3000
    assert not new_state.signed_sum.any() and not new_state.absolute_sum.any()
    assert not new_state.observations.any()
    assert not optimizers["opacity_logits"].state[params["opacity_logits"]]["exp_avg"].any()
    assert new_state.policy.grows_at(3100)
    same_state, result, stats = reference_post_step(params, optimizers, new_state, 1., 15000)
    assert same_state is new_state and result is None and stats == {}


def test_restored_window_and_torch_rng_reproduce_next_stochastic_action_on_cpu():
    params, state = field(3), window(3)
    with torch.no_grad():
        params["log_scales"][1] = torch.tensor(.02).log()
    state.absolute_sum[1] = .001
    state.signed_sum[2] = .001
    saved = state.state_dict()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(717)
        rng = torch.get_rng_state()
        first, _ = mixed_gradient_densify(params, state, 1., 600)
        restored = MixedGradientState(3, "cpu", state.policy)
        restored.load_state_dict(saved)
        torch.set_rng_state(rng)
        second, _ = mixed_gradient_densify(params, restored, 1., 600)
    assert torch.equal(first.source_indices, second.source_indices)
    for key in first.tensors:
        torch.testing.assert_close(first.tensors[key], second.tensors[key], rtol=0, atol=0)


def test_missing_absgrad_fails_instead_of_silent_signed_fallback_and_stop_skips():
    state = window(2)
    fake = SimpleNamespace(grad=torch.zeros(1, 2, 2))
    with pytest.raises(RuntimeError, match="absgrad"):
        state.accumulate({"means2d": fake}, 100)
    state.accumulate({}, 15000)
    state.retain_grad({}, 15000)


def test_reference_config_and_cache_exclude_semantic_supervision(monkeypatch):
    config = yaml.safe_load(Path("configs/rgb140_inspired_mixed_500k.yaml").read_text())
    policy = validate_reference_training(config)
    assert asdict(policy) == asdict(MixedGradientConfig())
    assert config["opacity_lr"] == .025
    with pytest.raises(ValueError, match="region_rgb_weight"):
        validate_reference_training(dict(config, region_rgb_weight=.15))
    with pytest.raises(ValueError, match="semantics"):
        validate_reference_training(dict(config, semantic_start=1))
    with pytest.raises(ValueError, match="opacity_reset_every"):
        validate_reference_training(dict(config, opacity_reset_every=6000))
    seen = []
    monkeypatch.setattr("bridge_rgs.train.load_view", lambda view, **_: seen.append(view) or {"mask": None})
    source = {"name": "train.png", "mask_path": "must_not_read.png"}
    ImageCache(ignore_masks=True).get(source, 1.)
    assert seen[0]["mask_path"] is None
    assert source["mask_path"] == "must_not_read.png"


def test_empty_field_pruning_is_explicit_failure():
    params, state = field(2), window(2)
    with torch.no_grad():
        params["opacity_logits"].fill_(-10)
    with pytest.raises(RuntimeError, match="whole field"):
        mixed_gradient_densify(params, state, 1., 600)
