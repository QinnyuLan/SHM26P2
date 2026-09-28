"""Numerical checks of camera conventions, attribution and evidence rejection."""

import torch

from bridge_rgs.reliability import (
    camera_compensated_residual,
    camera_residual_attribution,
    ellipse_sample_probabilities,
    fuse_multiview_evidence,
    geometry_supervision_gate,
    projection_jacobians,
    propagate_projection_covariance,
    se3_exp,
)


def test_se3_zero_has_finite_correct_gradient():
    twist = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(se3_exp, (twist,), eps=1e-6, atol=1e-6)
    torch.testing.assert_close(se3_exp(twist), torch.eye(4, dtype=twist.dtype))


def test_se3_nonzero_and_inverse():
    twist = torch.tensor([0.2, -0.1, 0.4, 0.3, 0.1, -0.2], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(se3_exp, (twist,), eps=1e-6, atol=1e-6)
    torch.testing.assert_close(se3_exp(twist) @ se3_exp(-twist), torch.eye(4, dtype=twist.dtype))


def test_projection_jacobians_match_finite_differences():
    points = torch.tensor([[0.2, -0.3, 3.0], [-0.5, 0.1, 5.0]], dtype=torch.float64)
    pose = se3_exp(torch.tensor([0.1, 0.2, 0.1, 0.03, -0.02, 0.01], dtype=torch.float64))
    K = torch.tensor([[500.0, 2, 100], [0, 450, 80], [0, 0, 1]], dtype=torch.float64)
    result = projection_jacobians(points, pose, K)
    eps = 1e-6
    for dimension in range(6):
        perturbation = torch.zeros(6, dtype=torch.float64)
        perturbation[dimension] = eps
        plus = projection_jacobians(points, se3_exp(perturbation) @ pose, K).uv
        minus = projection_jacobians(points, se3_exp(-perturbation) @ pose, K).uv
        torch.testing.assert_close((plus - minus) / (2 * eps), result.camera_jacobian[..., dimension], atol=1e-6, rtol=1e-6)
    for dimension in range(3):
        perturbation = torch.zeros(3, dtype=torch.float64)
        perturbation[dimension] = eps
        plus = projection_jacobians(points + perturbation, pose, K).uv
        minus = projection_jacobians(points - perturbation, pose, K).uv
        torch.testing.assert_close((plus - minus) / (2 * eps), result.point_jacobian[..., dimension], atol=1e-6, rtol=1e-6)


def test_covariance_propagation_agrees_with_linear_monte_carlo():
    generator = torch.Generator().manual_seed(12)
    camera_J = torch.randn(1, 2, 6, generator=generator, dtype=torch.float64)
    point_J = torch.randn(1, 2, 3, generator=generator, dtype=torch.float64)
    # Correlated joint covariance tests the camera-point cross terms.
    factor = torch.randn(9, 9, generator=generator, dtype=torch.float64) * 0.01
    joint_covariance = factor @ factor.T
    predicted = propagate_projection_covariance(
        camera_J, point_J, joint_covariance[:6, :6], joint_covariance[6:, 6:],
        observation_std=0, camera_point_cross_covariance=joint_covariance[:6, 6:],
    )[0]
    samples = torch.randn(100_000, 9, generator=generator, dtype=torch.float64) @ factor.T
    displacement = samples @ torch.cat((camera_J[0], point_J[0]), -1).T
    empirical = torch.cov(displacement.T)
    torch.testing.assert_close(predicted, empirical, atol=1e-4, rtol=0.02)


def test_attribution_retains_orthogonal_scene_error():
    generator = torch.Generator().manual_seed(7)
    basis, _ = torch.linalg.qr(torch.randn(80, 7, generator=generator, dtype=torch.float64))
    jacobian = basis[:, :6]
    camera_error = jacobian @ torch.tensor([0.1, -0.2, 0.2, 0.1, 0, -0.1], dtype=torch.float64)
    scene_error = 0.4 * basis[:, 6]
    result = camera_residual_attribution(camera_error + scene_error, jacobian, damping=1e-8)
    torch.testing.assert_close(result.remaining, scene_error, atol=1e-8, rtol=1e-6)
    expected = camera_error.square().sum() / (camera_error + scene_error).square().sum()
    torch.testing.assert_close(result.explained_fraction, expected)


def test_attribution_mask_and_degenerate_information_are_safe():
    residual = torch.ones(4, 5, 3)
    jacobian = torch.zeros(4, 5, 3, 6)
    result = camera_residual_attribution(residual, jacobian, weights=torch.zeros(4, 5))
    assert torch.isfinite(result.delta).all()
    assert result.delta.abs().sum() == 0
    assert result.explained_fraction == 0
    torch.testing.assert_close(result.remaining, residual)


def test_compensated_residual_verifies_actual_render_and_bounds_pose():
    pose = torch.eye(4, dtype=torch.float64)

    def render(candidate):
        return candidate[:3, 3].expand(4, 3)

    target = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64).expand(4, 3)
    result = camera_compensated_residual(render, pose, target, max_translation=0.02)
    assert result.accepted
    assert result.delta[:3].norm() <= 0.02000001
    assert result.after_energy < result.before_energy
    torch.testing.assert_close(pose, torch.eye(4, dtype=pose.dtype))


def test_non_linear_diagnostic_rejects_worsening_step():
    pose = torch.eye(4, dtype=torch.float64)

    def render(candidate):
        x = candidate[0, 3]
        return (x + 1000 * x.square()).reshape(1)

    result = camera_compensated_residual(
        render, pose, torch.tensor([-0.02], dtype=torch.float64),
        max_translation=0.1, damping=1e-8,
    )
    assert not result.accepted
    assert result.delta.abs().sum() == 0
    assert result.explained_fraction == 0


def test_ellipse_rejects_out_of_bounds_occlusion_and_extreme_uncertainty():
    probabilities = torch.zeros(2, 20, 20)
    probabilities[0] = 1
    uv = torch.tensor([[10.0, 10], [-20, 3], [10, 10], [10, 10]])
    covariance = torch.eye(2).repeat(4, 1, 1) * 0.1
    covariance[3] *= 10_000
    evidence = ellipse_sample_probabilities(
        probabilities, uv, covariance,
        depths=torch.tensor([3.0, 3, 5, 3]), depth_map=torch.full((20, 20), 3.0),
    )
    assert evidence.weights[0] > 0.9
    torch.testing.assert_close(evidence.weights[1:], torch.zeros(3))
    assert torch.isfinite(evidence.probabilities).all()


def test_ellipse_boundary_stability_changes_even_when_confidence_is_similar():
    probabilities = torch.zeros(2, 30, 30)
    probabilities[0, :, :15] = 1
    probabilities[1, :, 15:] = 1
    covariance = torch.eye(2).repeat(2, 1, 1) * 4
    evidence = ellipse_sample_probabilities(probabilities, torch.tensor([[6.0, 15], [14, 15]]), covariance)
    assert evidence.stability[0] > evidence.stability[1]
    assert evidence.weights[0] > evidence.weights[1]


def test_alpha_gate_and_behind_camera_reject_evidence():
    probability = torch.full((2, 8, 8), 0.5)
    uv = torch.tensor([[3.0, 3], [3, 3]])
    covariance = torch.eye(2).repeat(2, 1, 1)
    evidence = ellipse_sample_probabilities(
        probability, uv, covariance, depths=torch.tensor([2.0, -1.0]), alpha_map=torch.zeros(8, 8)
    )
    assert evidence.weights.sum() == 0


def test_fusion_requires_distinct_views_and_rejects_conflicting_votes():
    probability = torch.tensor([
        [[0.99, 0.01], [0.99, 0.01], [0.99, 0.01]],
        [[0.99, 0.01], [0.01, 0.99], [0.99, 0.01]],
    ])
    weights = torch.tensor([[1.0, 1, 1], [1.0, 1, 0]])
    result = fuse_multiview_evidence(probability, weights)
    assert result.weights[0] > 0.99
    assert result.weights[1] < 0.01
    assert result.weights[2] == 0
    assert result.view_count.tolist() == [2, 2, 1]


def test_empty_fusion_is_uniform_and_finite():
    result = fuse_multiview_evidence(torch.zeros(2, 4, 3), torch.zeros(2, 4))
    torch.testing.assert_close(result.probabilities, torch.full((4, 3), 1 / 3))
    assert result.weights.sum() == 0


def test_invalid_residual_rows_are_ignored_without_poisoning_energy():
    result = camera_residual_attribution(
        torch.tensor([float("inf"), float("nan"), 0.5]), torch.ones(3, 6)
    )
    assert torch.isfinite(result.before_energy)
    assert torch.isfinite(result.after_energy)
    assert torch.isfinite(result.delta).all()


def test_geometry_gate_requires_rgb_structure_and_detaches_reliability():
    projection = torch.ones(4, requires_grad=True)
    gate = geometry_supervision_gate(
        projection, torch.tensor([1.0, 1.0, 0.5, 1.0]),
        torch.tensor([3, 3, 3, 1]), torch.tensor([1.0, 0.0, 1.0, 1.0]),
    )
    torch.testing.assert_close(gate, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert not gate.requires_grad


def test_ellipse_uses_supplied_teacher_quality_beyond_max_probability():
    probabilities = torch.zeros(2, 8, 8)
    probabilities[0] = 1
    uv = torch.tensor([[3.0, 3]])
    covariance = torch.eye(2)[None] * .1
    high = ellipse_sample_probabilities(
        probabilities, uv, covariance, teacher_confidence_map=torch.full((8, 8), .9)
    )
    low = ellipse_sample_probabilities(
        probabilities, uv, covariance, teacher_confidence_map=torch.full((8, 8), .1)
    )
    torch.testing.assert_close(high.weights, 9 * low.weights)
