import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from bridge_rgs import mcmc_reference as m


def field(n=40, scale=7.):
    scene = SimpleNamespace(scene_scale=scale, mip_filter_config=None)
    scene.splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.arange(n * 3).reshape(n, 3).float() / 100),
        "quats": torch.nn.Parameter(torch.tensor([1., 0, 0, 0]).repeat(n, 1)),
        "log_scales": torch.nn.Parameter(torch.full((n, 3), -2.)),
        "opacity_logits": torch.nn.Parameter(torch.zeros(n)),
        "sh0": torch.nn.Parameter(torch.arange(n * 3).reshape(n, 1, 3).float()),
        "sh_rest": torch.nn.Parameter(torch.zeros(n, 3, 3)),
        "sem_features": torch.nn.Parameter(torch.arange(n * 4).reshape(n, 4).float()),
    })
    scene.semantic_prior_counts = torch.zeros(n, 5)
    opts = {k: torch.optim.Adam([p], lr=.01, eps=1e-15) for k, p in scene.splats.items()}
    # Multi-group heads must not be passed to gsplat's splat optimizer check.
    scene.head = torch.nn.Parameter(torch.ones(2))
    opts["heads"] = torch.optim.Adam([{"params": [scene.head]}, {"params": [torch.nn.Parameter(torch.zeros(1))]}])
    return scene, opts


def populate_adam(scene, opts):
    for key, p in scene.splats.items():
        p.grad = torch.ones_like(p)
        opts[key].step()
        opts[key].zero_grad(set_to_none=True)
        moments = opts[key].state[p]
        row = torch.arange(1, len(p) + 1).float().reshape((len(p),) + (1,) * (p.ndim - 1))
        moments["exp_avg"].copy_(row.expand_as(p))
        moments["exp_avg_sq"].copy_((row * 10).expand_as(p))


@pytest.fixture
def cpu_ops(monkeypatch):
    """Use installed topology/Adam code, replacing only CUDA math and RNG draws."""
    import gsplat.strategy.mcmc as installed
    from gsplat.strategy import ops
    records = []

    def relocate_math(opacities, scales, ratios, binoms):
        assert opacities.min() > 0 and opacities.max() < 1
        assert (scales > 0).all() and binoms.shape == (51, 51)
        records.append((opacities.clone(), scales.clone(), ratios.clone()))
        return opacities / ratios, scales * .9

    def noise(params, optimizers, state, scaler):
        records.append(float(scaler))
        params["means"].add_(.001)

    monkeypatch.setattr(ops, "compute_relocation", relocate_math)
    monkeypatch.setattr(ops, "_multinomial_sample", lambda weights, n, replacement=True:
                        torch.zeros(n, dtype=torch.long, device=weights.device))
    monkeypatch.setattr(installed, "inject_noise_to_position", noise)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return records


def policy(cap=42):
    return m.MCMCReferenceConfig(cap_max=cap, refine_start_iter=1,
                                 refine_stop_iter=4, refine_every=2)


def test_production_recipe_is_fixed_and_rejects_legacy_masks_or_auxiliary_paths():
    config = {"densification": m.MODE, "steps": 30000, "semantic_start": 1000000,
              "refine_start": 1000000, "semantic_initialization": False,
              "view_sampling": "shuffle", "independent_view_rng": True}
    assert m.validate_reference_training(config) == m.MCMCReferenceConfig()
    assert asdict(m.validate_reference_training(config))["noise_lr"] == 5e5
    for key, value in [("semantic_initialization", True), ("warmstart", "x.pt"),
                       ("opacity_reset_every", 0), ("mip_filter", {"enabled": True}),
                       ("opacity_lr", .025), ("train_labeled_only", True),
                       ("initial_opacity", True), ("initial_scale_multiplier", .2),
                       ("train_scale", .5), ("progressive_resolution", True),
                       ("independent_view_rng", False), ("view_sampling", "random"),
                       ("mcmc_reference", {"noise_lr": 1.})]:
        with pytest.raises(ValueError):
            m.validate_reference_training(dict(config, **{key: value}))


def test_coordinate_covariance_noise_mapping_uses_inverse_squared_scene_scale():
    dtype = torch.float64
    L, lr, noise_lr = 13.7, .002, 5e5
    A = torch.tensor([[.3, .2, -.1], [.4, .1, .2], [0., -.3, .5]], dtype=dtype)
    covariance = A @ A.T
    z = torch.tensor([.2, -.7, .8], dtype=dtype)
    opacity_gate = .37
    world = covariance @ z * opacity_gate * noise_lr * m.canonical_noise_lr(lr, L)
    canonical = (covariance / L**2) @ z * opacity_gate * noise_lr * (lr / L)
    torch.testing.assert_close(world, L * canonical, atol=1e-14, rtol=1e-14)
    assert m.canonical_noise_lr(0., L) == 0
    for invalid in (0., -1., float("nan")):
        with pytest.raises(ValueError):
            m.canonical_noise_lr(lr, invalid)


def test_regularization_matches_canonical_units_and_has_only_expected_gradients():
    scene, _ = field()
    p = m.MCMCReferenceConfig()
    loss, values = m.reference_regularization(scene, p)
    expected = .01 * scene.splats["opacity_logits"].sigmoid().mean()
    expected += .01 * scene.splats["log_scales"].exp().mean() / scene.scene_scale
    assert torch.equal(loss, expected)
    assert all(not value.requires_grad for value in values.values())
    loss.backward()
    torch.testing.assert_close(scene.splats["opacity_logits"].grad, torch.full((40,), .01 * .25 / 40))
    torch.testing.assert_close(scene.splats["log_scales"].grad,
                               torch.full((40, 3), .01 * torch.exp(torch.tensor(-2.)).item() / (120 * 7)))
    assert scene.splats["means"].grad is None


def test_initialize_preserves_parameters_adam_and_rng_with_correct_aliases():
    scene, opts = field()
    populate_adam(scene, opts)
    old = {k: p.clone() for k, p in scene.splats.items()}
    moments = {k: opts[k].state[p]["exp_avg"].clone() for k, p in scene.splats.items()}
    rng = torch.get_rng_state().clone()
    state = m.initialize_reference(scene, opts, policy())
    assert state.counters["initial_gaussians"] == 40
    assert torch.equal(rng, torch.get_rng_state())
    for key, param in scene.splats.items():
        assert torch.equal(param, old[key])
        assert torch.equal(opts[key].state[param]["exp_avg"], moments[key])
    params, optimizers = m._aliases(scene, opts)
    assert params["scales"] is scene.splats["log_scales"]
    assert params["opacities"] is scene.splats["opacity_logits"]
    assert "heads" not in optimizers


def test_installed_relocation_and_birth_preserve_declared_adam_semantics_and_all_payloads(cpu_ops):
    scene, opts = field()
    populate_adam(scene, opts)
    with torch.no_grad():
        scene.splats["opacity_logits"][0] = -12
    old = dict(scene.splats.items())
    donor_sem = old["sem_features"][1].detach().clone()
    head_id = id(scene.head)
    state = m.initialize_reference(scene, opts, policy())
    assert not m.reference_post_step(scene, opts, state, 1, .002)["mcmc_topology_changed"]
    out = m.reference_post_step(scene, opts, state, 2, .002)
    assert out["mcmc_topology_changed"] and out["mcmc_relocated_this_step"] == 1
    assert out["mcmc_added_this_step"] == 2 and out["mcmc_current_gaussians"] == 42
    assert out["mcmc_noise_gaussian_steps"] == 40 + 42
    assert out["mcmc_noise_scaler"] == .002 / 49 * 5e5
    assert len(cpu_ops) == 4  # noise; relocate math, birth math, noise
    assert id(scene.head) == head_id
    assert scene.semantic_prior_counts.shape == (42, 5)
    assert not torch.count_nonzero(scene.semantic_prior_counts)
    for key, new in scene.splats.items():
        assert new is not old[key] and new.shape[0] == 42
        assert opts[key].param_groups[0]["params"][0] is new
        assert old[key] not in opts[key].state
        moments = opts[key].state[new]
        assert moments["step"].item() == 1
        # Dead row retains its old moments; relocation donor row is zeroed.
        assert torch.all(moments["exp_avg"][0] == 1)
        assert torch.all(moments["exp_avg"][1] == 0)
        assert torch.all(moments["exp_avg"][-2:] == 0)
        assert torch.all(moments["exp_avg_sq"][0] == 10)
    assert torch.equal(scene.splats["sem_features"][0], donor_sem)
    assert torch.equal(scene.splats["sem_features"][-1], donor_sem)


def test_relocation_at_cap_still_replaces_parameter_and_noise_after_stop(cpu_ops):
    scene, opts = field()
    with torch.no_grad():
        scene.splats["opacity_logits"][0] = -12
    state = m.initialize_reference(scene, opts, policy(cap=40))
    m.reference_post_step(scene, opts, state, 1, .002)
    old = scene.splats["means"]
    out = m.reference_post_step(scene, opts, state, 2, .002)
    assert scene.splats["means"] is not old and out["mcmc_topology_changed"]
    assert out["mcmc_added_this_step"] == 0 and len(scene.splats["means"]) == 40
    m.reference_post_step(scene, opts, state, 3, .002)
    out = m.reference_post_step(scene, opts, state, 4, .002)
    assert not out["mcmc_topology_changed"] and out["mcmc_noise_steps"] == 4
    assert out["mcmc_refinement_events"] == 1
    assert not m.MCMCReferenceConfig().refines_at(25000)
    assert sum(m.MCMCReferenceConfig().refines_at(s) for s in range(1, 30001)) == 244


def test_reject_all_dead_nonzero_prior_cap_and_optimizer_alias_mismatch(cpu_ops):
    scene, opts = field()
    state = m.initialize_reference(scene, opts, policy())
    m.reference_post_step(scene, opts, state, 1, .002)
    with torch.no_grad():
        scene.splats["opacity_logits"].fill_(-12)
    old = scene.splats["means"].clone()
    with pytest.raises(ValueError, match="All Gaussians are dead"):
        m.reference_post_step(scene, opts, state, 2, .002)
    assert torch.equal(old, scene.splats["means"]) and state.counters["last_step"] == 1
    scene.semantic_prior_counts[0, 2] = 1
    with pytest.raises(ValueError, match="all-zero"):
        m.initialize_reference(scene, opts, policy())
    scene.semantic_prior_counts.zero_()
    with pytest.raises(ValueError, match="exceeds"):
        m.initialize_reference(scene, opts, policy(cap=39))
    opts["means"].param_groups[0]["params"] = [torch.nn.Parameter(old)]
    with pytest.raises(ValueError, match="identity"):
        m.initialize_reference(scene, opts, policy())


def test_state_resume_rebinds_without_reset_and_rejects_policy_scale_table_or_step(cpu_ops):
    scene, opts = field()
    populate_adam(scene, opts)
    state = m.initialize_reference(scene, opts, policy())
    m.reference_post_step(scene, opts, state, 1, .002)
    m.reference_post_step(scene, opts, state, 2, .002)
    saved = state.state_dict()
    rng = torch.get_rng_state().clone()
    resumed = m.initialize_reference(scene, opts, policy())
    resumed.load_state_dict(saved, scene, opts, expected_step=2)
    assert torch.equal(rng, torch.get_rng_state())
    assert resumed.counters == state.counters
    assert torch.equal(resumed.strategy_state["binoms"], state.strategy_state["binoms"])
    assert resumed.strategy_state["binoms"].data_ptr() != saved["binoms"].data_ptr()
    next_stats = m.reference_post_step(scene, opts, resumed, 3, .002)
    assert next_stats["mcmc_noise_gaussian_steps"] == 40 + 42 + 42
    with pytest.raises(ValueError, match="consecutive"):
        m.reference_post_step(scene, opts, resumed, 3, .002)
    for key, value in [("scene_scale", 8.), ("policy", asdict(m.MCMCReferenceConfig()))]:
        bad = copy.deepcopy(saved); bad[key] = value
        with pytest.raises(ValueError, match="changed"):
            resumed.load_state_dict(bad, scene, opts, expected_step=2)
    with pytest.raises(ValueError, match="step"):
        resumed.load_state_dict(saved, scene, opts, expected_step=1)
    bad = copy.deepcopy(saved); bad["binoms"][0, 0] = 0
    with pytest.raises(ValueError, match="binomial"):
        resumed.load_state_dict(bad, scene, opts, expected_step=2)


def test_cap_is_checked_after_installed_strategy_even_if_upstream_contract_changes(monkeypatch):
    scene, opts = field()
    state = m.initialize_reference(scene, opts, policy())

    def bad_strategy(params, *args, **kwargs):
        # Deliberately return an impossible field; adapter must not silently accept it.
        for key, p in list(params.items()):
            params[key] = torch.nn.Parameter(torch.cat([p, p[:3]]))

    monkeypatch.setattr(state.strategy, "step_post_backward", bad_strategy)
    with pytest.raises(ValueError, match="cap/schedule"):
        m.reference_post_step(scene, opts, state, 1, .002)
    assert state.counters["last_step"] == 0


def test_nondefault_helpers_are_only_for_explicit_synthetic_contracts():
    for kwargs in ({"cap_max": True}, {"noise_lr": float("nan")},
                   {"min_opacity": 1.}, {"refine_stop_iter": 100}):
        with pytest.raises(ValueError):
            m.MCMCReferenceConfig(**kwargs)
