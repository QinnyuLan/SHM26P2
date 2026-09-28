"""Meaningful invariants for deterministic geometry allocation and Adam state."""

import pytest
import torch
from torch import nn

from bridge_rgs.densification import (
    adaptive_gradient_threshold,
    apply_densification,
    density_priority,
    estimate_local_structure,
    hybrid_density_priority,
    normalized_absgrad,
    quaternion_to_matrix,
    reset_opacity_with_optimizer,
    structure_guided_densify,
    support_constrained_split_geometry,
)


def field(count=10):
    means = torch.stack((torch.linspace(-1, 1, count), torch.zeros(count), torch.zeros(count)), -1)
    return nn.ParameterDict({
        "means": nn.Parameter(means),
        "quats": nn.Parameter(torch.tensor([1.0, 0, 0, 0]).repeat(count, 1)),
        "log_scales": nn.Parameter(torch.full((count, 3), -2.0)),
        "opacity_logits": nn.Parameter(torch.zeros(count)),
        "sh0": nn.Parameter(torch.zeros(count, 1, 3)),
        "sh_rest": nn.Parameter(torch.zeros(count, 3, 3)),
        "sem_features": nn.Parameter(torch.zeros(count, 16)),
    })


def test_semantic_error_alone_does_not_trigger_geometry_priority():
    priority = density_priority(
        torch.tensor([0.0, 0.3, 0.3]), torch.ones(3), torch.tensor([5, 1, 3]),
        semantic_boundary=torch.ones(3), geometry_gate=torch.ones(3),
    )
    assert priority[0] == 0
    assert priority[1] == 0
    assert priority[2] > 0


def test_local_pca_identifies_line_and_surface_without_class_names():
    line = torch.stack((torch.linspace(-2, 2, 40), torch.zeros(40), torch.zeros(40)), -1)
    line_structure = estimate_local_structure(line, torch.zeros(1, 3))
    assert line_structure.kind.item() == 1
    torch.testing.assert_close(line_structure.direction[0], torch.tensor([1.0, 0, 0]))
    x, y = torch.meshgrid(torch.linspace(-1, 1, 7), torch.linspace(-1, 1, 7), indexing="ij")
    plane = torch.stack((x.flatten(), y.flatten(), torch.zeros(49)), -1)
    plane_structure = estimate_local_structure(plane, torch.zeros(1, 3), neighbors=49)
    assert plane_structure.kind.item() == 2
    assert plane_structure.direction[0, 2].abs() < 1e-6


def test_fixed_budget_and_persistent_view_gate():
    parameters = field(10)
    result = structure_guided_densify(
        parameters, torch.arange(10).float(), torch.tensor([3] * 9 + [1]),
        max_gaussians=12, max_splits=4,
    )
    assert len(result.tensors["means"]) == 12
    assert result.split_count == 2
    parents = result.source_indices[result.reset_moments]
    assert set(parents.tolist()) == {7, 8}
    assert result.line_splits == 2
    assert (result.tensors["means"][:, 1:] == 0).all()


def test_splitting_preserves_combined_transmittance_and_feature_identity():
    parameters = field(6)
    with torch.no_grad():
        parameters["sem_features"][:, 0] = torch.arange(6)
    result = structure_guided_densify(parameters, torch.arange(6).float(), torch.full((6,), 3), 7, max_splits=1)
    children = result.tensors["opacity_logits"][result.reset_moments].sigmoid()
    torch.testing.assert_close((1 - children).prod(), 1 - parameters["opacity_logits"][5].sigmoid())
    torch.testing.assert_close(result.tensors["sem_features"][-2:, 0], torch.tensor([5.0, 5]))
    torch.testing.assert_close(result.tensors["means"][-2:].mean(0), parameters["means"][5])
    assert (result.tensors["log_scales"][-2:, 1:] == -2).all()


def test_allocation_is_deterministic_and_can_recycle_at_full_budget():
    parameters = field(10)
    arguments = {
        "parameters": parameters, "scores": torch.arange(10).float(),
        "residual_view_counts": torch.full((10,), 3), "max_gaussians": 10,
        "max_splits": 2, "recycle_count": 2, "contribution": torch.arange(10).float(),
    }
    first = structure_guided_densify(**arguments)
    second = structure_guided_densify(**arguments)
    assert first.split_count == first.pruned_count == 2
    assert len(first.tensors["means"]) == 10
    assert 0 not in first.source_indices and 1 not in first.source_indices
    for name in first.tensors:
        torch.testing.assert_close(first.tensors[name], second.tensors[name], atol=0, rtol=0)


def test_excess_initial_count_is_pruned_to_hard_budget():
    result = structure_guided_densify(field(10), torch.ones(10), torch.full((10,), 3), max_gaussians=4)
    assert len(result.tensors["means"]) == 4
    assert result.pruned_count == 6


@pytest.mark.parametrize("rule", ["legacy", "support_constrained"])
def test_optimizer_moments_migrate_and_training_can_continue(rule):
    parameters = field(8)
    optimizer = torch.optim.Adam(parameters.parameters(), lr=0.01)
    sum(value.square().sum() for value in parameters.values()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    old_parameter = parameters["means"]
    old_moments = optimizer.state[old_parameter]["exp_avg"].clone()
    old_step = optimizer.state[old_parameter]["step"].clone()
    result = structure_guided_densify(parameters, torch.arange(8).float(), torch.full((8,), 3), 10, max_splits=2,
                                      structured_split_rule=rule)
    apply_densification(parameters, result, optimizer)
    assert old_parameter not in optimizer.state
    state = optimizer.state[parameters["means"]]
    retained = ~result.reset_moments
    torch.testing.assert_close(state["exp_avg"][retained], old_moments[result.source_indices[retained]])
    assert state["exp_avg"][result.reset_moments].abs().sum() == 0
    torch.testing.assert_close(state["step"], old_step)
    sum(value.square().sum() for value in parameters.values()).backward()
    optimizer.step()
    assert all(torch.isfinite(value).all() for value in parameters.values())


def test_structure_ablation_uses_gaussian_major_axis():
    parameters = field(8)
    with torch.no_grad():
        parameters["log_scales"][:, 2] = 0
    result = structure_guided_densify(
        parameters, torch.arange(8).float(), torch.full((8,), 3), 9,
        max_splits=1, structure_guided=False,
    )
    assert result.line_splits == result.plane_splits == 0
    children = result.tensors["means"][-2:]
    assert children[0, 2] < 0 < children[1, 2]
    torch.testing.assert_close(children[:, 0], parameters["means"][7, 0].expand(2))


def test_far_background_does_not_inherit_bridge_tangent():
    line = torch.stack((torch.linspace(-1, 1, 40), torch.zeros(40), torch.zeros(40)), -1)
    structure = estimate_local_structure(line, torch.tensor([[0.0, 30, 0]]))
    assert structure.kind.item() == 0
    assert structure.confidence.item() == 0


def test_normalized_absgrad_matches_gsplat_resolution_convention():
    gradient = torch.tensor([[.002, .004], [.001, .003]])
    first = normalized_absgrad(gradient, 800, 400)
    second = normalized_absgrad(gradient / 2, 1600, 800)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[0], torch.tensor(2 ** .5 * .8))


def test_hybrid_keeps_multiview_gradient_candidates_without_diagnostics():
    score, support = hybrid_density_priority(
        torch.tensor([.002, .003, 0]), torch.ones(3), torch.tensor([2, 1, 0]),
        torch.tensor([0., 0, .1]), torch.tensor([0, 0, 2]), torch.tensor([0, 0, 2]),
    )
    assert score[0] > 0 and score[1] == 0 and score[2] > 0
    assert support.tolist() == [2, 0, 2]


def test_hybrid_rank_fusion_is_invariant_to_units_with_scaled_thresholds():
    arguments = (torch.tensor([.003, .002]), torch.ones(2), torch.full((2,), 3),
                 torch.tensor([.2, .3]), torch.ones(2), torch.full((2,), 3))
    first, _ = hybrid_density_priority(*arguments)
    scaled = (arguments[0] * 100, arguments[1], arguments[2], arguments[3] / 10,
              arguments[4], arguments[5])
    second, _ = hybrid_density_priority(*scaled, gradient_threshold=.08, residual_threshold=.003)
    torch.testing.assert_close(first, second)


def test_small_gaussians_duplicate_without_erasing_parent_moments_or_scales():
    parameters = field(6)
    result = structure_guided_densify(parameters, torch.arange(6).float(), torch.full((6,), 3),
                                      max_gaussians=7, max_splits=1, clone_scale_threshold=.2)
    assert result.duplicate_count == 1 and result.split_count == 0
    assert result.source_indices.tolist() == [0, 1, 2, 3, 4, 5, 5]
    assert not result.reset_moments[5] and result.reset_moments[6]
    torch.testing.assert_close(result.tensors["log_scales"][-1], parameters["log_scales"][5])


def test_screen_footprint_can_force_split_and_preserve_opacity():
    parameters = field(6)
    result = structure_guided_densify(parameters, torch.arange(6).float(), torch.full((6,), 3),
                                      max_gaussians=7, max_splits=1, clone_scale_threshold=.2,
                                      screen_radii=torch.full((6,), .1), split_screen_radius=.05,
                                      structure_guided=False, shrink_unstructured_all_axes=True,
                                      split_opacity_mode="preserve")
    assert result.split_count == 1 and result.duplicate_count == 0
    torch.testing.assert_close(result.tensors["opacity_logits"][-2:], parameters["opacity_logits"][5].expand(2))
    assert (result.tensors["log_scales"][-2:] < parameters["log_scales"][5]).all()


def test_group_quota_preserves_scene_coverage_near_hard_budget():
    parameters = field(6)
    groups = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
    result = structure_guided_densify(parameters, torch.arange(6).float() + 1, torch.full((6,), 3),
                                      max_gaussians=8, max_splits=4, allocation_groups=groups,
                                      foreground_fraction=.5)
    parents = result.source_indices[result.reset_moments][:2]
    assert groups[parents].sum() == 1
    assert len(result.tensors["means"]) == 8


def test_opacity_reset_only_clears_changed_rows_and_preserves_step():
    parameter = nn.Parameter(torch.tensor([-4., 0., 2.]))
    optimizer = torch.optim.Adam([parameter], lr=.01)
    parameter.square().sum().backward()
    optimizer.step()
    untouched = optimizer.state[parameter]["exp_avg"][0].clone()
    old_step = optimizer.state[parameter]["step"].clone()
    assert reset_opacity_with_optimizer(parameter, optimizer, .1) == 2
    assert parameter.sigmoid().max() <= .100001
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg"][0], untouched)
    assert optimizer.state[parameter]["exp_avg"][1:].abs().sum() == 0
    torch.testing.assert_close(optimizer.state[parameter]["step"], old_step)


def test_adaptive_threshold_uses_only_multiview_support_and_honors_floor():
    values = torch.tensor([.0001, .0002, .1])
    views = torch.tensor([2, 2, 1])
    threshold = adaptive_gradient_threshold(values, views, .0008, quantile=.5, floor=.00005)
    assert abs(threshold - .00015) < 1e-9
    assert adaptive_gradient_threshold(values / 100, views, .0008, quantile=.5, floor=.00005) == .00005


def test_supported_split_offsets_respect_ellipsoid_under_rotation_and_extreme_anisotropy():
    generator = torch.Generator().manual_seed(472)
    scales = torch.exp(torch.rand((512, 3), generator=generator, dtype=torch.float64) * -14)
    rotation = quaternion_to_matrix(torch.randn((512, 4), generator=generator, dtype=torch.float64))
    directions = torch.randn((512, 3), generator=generator, dtype=torch.float64)
    offset, axis = support_constrained_split_geometry(scales, rotation, directions)
    local_offset = (rotation.transpose(-1, -2) @ offset[..., None]).squeeze(-1)
    distance = (local_offset / scales).norm(dim=-1)
    assert (distance <= .5 + 1e-9).all()
    torch.testing.assert_close(distance, torch.full_like(distance, .5), atol=1e-9, rtol=0)
    selected_scale = scales.gather(-1, axis[:, None]).squeeze(-1)
    assert (selected_scale >= .25 * scales.max(-1).values).all()


def test_supported_split_discards_unsupported_thin_direction_and_falls_back_to_major_axis():
    scales = torch.tensor([[1., 1e-6, .1], [1., 1e-6, .1]], dtype=torch.float64)
    rotation = torch.eye(3, dtype=torch.float64).expand(2, 3, 3)
    direction = torch.tensor([[.6, .8, 0], [0, 1., 0]], dtype=torch.float64)
    offset, axis = support_constrained_split_geometry(scales, rotation, direction)
    torch.testing.assert_close(offset, torch.tensor([[.5, 0, 0], [.5, 0, 0]], dtype=torch.float64))
    assert axis.tolist() == [0, 0]


def test_supported_split_selects_variance_contribution_not_largest_direction_cosine():
    scales = torch.tensor([[1., .3, .01]], dtype=torch.float64)
    direction = torch.tensor([[.6, .8, 0]], dtype=torch.float64)
    offset, axis = support_constrained_split_geometry(scales, torch.eye(3, dtype=torch.float64)[None], direction)
    assert axis.item() == 0
    assert (offset / scales).norm() <= .5 + 1e-12


def test_supported_split_isotropic_and_axis_aligned_offsets_match_legacy_formula():
    scales = torch.tensor([[2., 2, 2], [2, .6, .01]], dtype=torch.float64)
    direction = torch.tensor([[.6, .8, 0], [0, 1., 0]], dtype=torch.float64)
    offset, axis = support_constrained_split_geometry(scales, torch.eye(3, dtype=torch.float64).expand(2, 3, 3), direction)
    expected = .5 * (direction.square() * scales.square()).sum(-1).sqrt()[:, None] * direction
    torch.testing.assert_close(offset, expected)
    assert axis.tolist() == [1, 1]


def test_supported_split_preserves_unsupported_thin_scale_and_parent_mean(monkeypatch):
    import bridge_rgs.densification as module
    parameters = field(6).double()
    with torch.no_grad():
        parameters['log_scales'][:] = torch.tensor([1., 1e-6, .1], dtype=torch.float64).log()
    def fake_structure(reference, query, **kwargs):
        n=len(query)
        return module.LocalStructure(query.new_tensor([.6, .8, 0]).expand(n, 3),
                                     torch.ones(n, dtype=torch.long), query.new_ones(n, 3), query.new_ones(n))
    monkeypatch.setattr(module, 'estimate_local_structure', fake_structure)
    result = structure_guided_densify(parameters, torch.arange(6).double(), torch.full((6,), 3),
                                      max_gaussians=7, max_splits=1, split_opacity_mode='preserve',
                                      structured_split_rule='support_constrained')
    assert result.split_count == 1 and len(result.source_indices) == 7
    torch.testing.assert_close(result.tensors['means'][-2:].mean(0), parameters['means'][5])
    torch.testing.assert_close(result.tensors['log_scales'][-2:, 1:], parameters['log_scales'][5, 1:].expand(2, 2))
    assert (result.tensors['log_scales'][-2:, 0] < parameters['log_scales'][5, 0]).all()
    torch.testing.assert_close(result.tensors['opacity_logits'][-2:], parameters['opacity_logits'][5].expand(2))


def test_legacy_default_remains_bitwise_identical_and_rejects_unknown_rule():
    parameters = field(8)
    args = (parameters, torch.arange(8).float(), torch.full((8,), 3), 10)
    implicit = structure_guided_densify(*args)
    explicit = structure_guided_densify(*args, structured_split_rule='legacy')
    for name in implicit.tensors:
        assert torch.equal(implicit.tensors[name], explicit.tensors[name])
    with pytest.raises(ValueError, match='structured_split_rule'):
        structure_guided_densify(*args, structured_split_rule='unknown')


def test_support_rule_leaves_unstructured_splits_and_clones_bitwise_unchanged():
    parameters = field(8)
    with torch.no_grad():
        parameters['log_scales'][:, 2] += 2
    arguments = {'parameters': parameters, 'scores': torch.arange(8).float(),
                 'residual_view_counts': torch.full((8,), 3), 'max_gaussians': 10,
                 'structure_guided': False, 'shrink_unstructured_all_axes': True,
                 'split_opacity_mode': 'preserve'}
    for clone_threshold in (None, 2.):
        legacy = structure_guided_densify(**arguments, clone_scale_threshold=clone_threshold)
        safe = structure_guided_densify(**arguments, clone_scale_threshold=clone_threshold,
                                       structured_split_rule='support_constrained')
        for name in legacy.tensors:
            assert torch.equal(legacy.tensors[name], safe.tensors[name])
