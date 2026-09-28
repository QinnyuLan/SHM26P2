"""Pure synthetic optimizer contracts; no scene, files, renderer or CUDA."""
import copy
import json

import pytest
import torch

from bridge_rgs.continuous_geometry import (
    MeansTransaction,
    cap_displacement,
    combine_gradients,
    fixed_metric,
)


def transaction(means=None, scene_scale=1.):
    p = torch.nn.Parameter(torch.zeros(1, 3) if means is None else means.clone())
    q = torch.tensor([[1., 0, 0, 0]]).repeat(len(p), 1)
    return MeansTransaction(p, q, torch.zeros(len(p), 3), scene_scale)


def same_state(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right, strict=True):
            same_state(x, y)
    else:
        assert left == right


def test_symmetric_pcgrad_uses_original_other_not_mutated_projection():
    r = torch.tensor([[1., 0, 0]], dtype=torch.float64)
    s = torch.tensor([[-1., 1., 0]], dtype=torch.float64)/.03
    actual, stats = combine_gradients(r, s, 'pcgrad')
    # Original vectors (1,0),(-1,1): projections (.5,.5),(0,1).
    torch.testing.assert_close(actual, torch.tensor([[.5, 1.5, 0]], dtype=torch.float64))
    assert stats['conflicting_original_gradients']
    assert not torch.allclose(actual, torch.tensor([[-.5, 1.5, 0]], dtype=torch.float64))


def test_pcgrad_global_dot_not_per_gaussian_conflict():
    r = torch.tensor([[1., 0, 0], [1., 0, 0]], dtype=torch.float64)
    s = torch.tensor([[-2., 0, 0], [3., 0, 0]], dtype=torch.float64)/.03
    pc, stats = combine_gradients(r, s, 'pcgrad')
    joint, _ = combine_gradients(r, s, 'joint')
    torch.testing.assert_close(pc, torch.tensor([[-1., 0, 0], [4., 0, 0]], dtype=torch.float64))
    assert torch.equal(pc, joint) and not stats['conflicting_original_gradients']
    semantic, _ = combine_gradients(r, s, 'semantic')
    assert torch.equal(semantic, s)


def test_two_dimensional_pcgrad_then_adam_can_increase_rgb_linear_objective():
    r = torch.tensor([[1., 10., 0]], dtype=torch.float64)
    s = torch.tensor([[100., -20., 0]], dtype=torch.float64)/.03
    combined, _ = combine_gradients(r, s, 'pcgrad')
    assert float((r*combined).sum()) > 0  # -combined itself is RGB descent.
    tx = transaction()
    proposal, report = tx.propose(r, s, 'pcgrad')
    assert report['rgb_dot_actual_delta'] > 0  # Adam first step rescales the coordinates.
    assert proposal[0, 0] < 0 and proposal[0, 1] > 0 and proposal[0, 2] == 0
    assert report['scaled_point_count'] == 0 and not report['first_order_RGB_guarantee']
    assert report['rgb_dot_actual_delta'] == pytest.approx(float((r*proposal.double()).sum()))
    json.dumps(report, allow_nan=False)
    tx.resolve(False)


def test_nonidentity_quaternion_metric_and_axis_cap():
    # wxyz(.5,.5,.5,.5) cycles local axes exactly: local z is world x.
    q = torch.tensor([[.5, .5, .5, .5]], dtype=torch.float64)
    rotation, _ = fixed_metric(q, torch.zeros(1, 3, dtype=torch.float64))
    inverse = torch.tensor([[.5, .25, .125]], dtype=torch.float64)
    before = torch.zeros(1, 3)
    after, report = cap_displacement(before, torch.tensor([[1., 0, 0]]), rotation, inverse)
    assert torch.equal(after, torch.tensor([[.125, 0, 0]]))
    covariance = rotation[0] @ torch.diag(torch.tensor([4., 16., 64.], dtype=torch.float64)) @ rotation[0].T
    delta = after[0].double()
    radius = (delta @ torch.linalg.solve(covariance, delta)).sqrt()
    assert float(radius) == 1/64 == report['actual_mahalanobis_max']
    assert report['scaled_point_count'] == 1 and report['actual_nonzero_point_count'] == 1


def test_fp32_unrepresentable_capped_step_restores_point_without_tolerance_rescue():
    before = torch.tensor([[1000., 0, 0], [0., 0, 0]])
    proposed = before+torch.tensor([[.001, 0, 0], [.001, 0, 0]])
    rotation = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    inverse = torch.full((2, 3), 500., dtype=torch.float64)
    result, report = cap_displacement(before, proposed, rotation, inverse)
    assert torch.equal(result[0], before[0])  # Quantized first coordinate would exceed h.
    assert report['fp32_overcap_restored_point_count'] >= 1
    assert report['actual_mahalanobis_max'] <= 1/64


def test_cap_rounding_to_before_is_counted_separately_from_overcap_fallback():
    before = torch.tensor([[1000., 0, 0], [1000., 0, 0], [0., 0, 0], [0., 0, 0]])
    proposed = before+torch.tensor([[.001, 0, 0], [.001, 0, 0], [.001, 0, 0], [0., 0, 0]])
    rotation = torch.eye(3, dtype=torch.float64).repeat(4, 1, 1)
    inverse = torch.tensor([[1000.]*3, [500.]*3, [1.]*3, [1.]*3], dtype=torch.float64)
    result, report = cap_displacement(before, proposed, rotation, inverse)
    # First cap is below half an FP32 ULP at 1000; second rounds beyond its cap.
    assert torch.equal(result[:2], before[:2])
    assert torch.equal(result[2], proposed[2]) and torch.equal(result[3], before[3])
    assert report['adam_proposed_nonzero_point_count'] == 3
    assert report['scaled_point_count'] == 2
    assert report['cap_rounded_to_before_point_count'] == 1
    assert report['fp32_overcap_restored_point_count'] == 1
    assert report['actual_nonzero_point_count'] == 1


def test_fresh_adam_rate_and_reject_restore_all_moments_and_step():
    tx = transaction(scene_scale=3.)
    r = torch.tensor([[2., -1., 4.]])
    s = torch.tensor([[1., 3., -1.]])
    proposal, report = tx.propose(r, s, 'joint')
    assert report['learning_rate'] == pytest.approx(4.8e-6)
    assert report['adam_eps'] == 1e-15 and report['optimizer_proposed_step'] == 1
    tx.resolve(True)
    accepted, state = proposal.clone(), copy.deepcopy(tx.optimizer.state_dict())
    tx.propose(-r*7, s*5, 'pcgrad')
    tx.resolve(False)
    assert torch.equal(tx.means, accepted) and tx.pending is None and tx.means.grad is None
    same_state(tx.optimizer.state_dict(), state)
    _, next_report = tx.propose(r, s, 'joint')
    assert next_report['optimizer_proposed_step'] == 2
    tx.resolve(False)


def test_first_rejection_restores_empty_adam_state_and_requires_resolve():
    tx = transaction(); zero = tx.means.detach().clone()
    tx.propose(torch.ones_like(zero), torch.ones_like(zero), 'joint')
    with pytest.raises(ValueError, match='pending'):
        tx.propose(zero, zero, 'joint')
    tx.resolve(False)
    assert torch.equal(tx.means, zero) and tx.optimizer.state_dict()['state'] == {}
    with pytest.raises(ValueError, match='No pending'):
        tx.resolve(False)


def test_zero_and_opposing_gradients_are_finite_noops_with_fresh_adam():
    zero = torch.zeros(1, 3)
    tx = transaction()
    proposal, report = tx.propose(zero, zero, 'pcgrad')
    assert torch.equal(proposal, zero) and report['actual_nonzero_point_count'] == 0
    json.dumps(report, allow_nan=False); tx.resolve(False)
    r = torch.tensor([[1., 0, 0]], dtype=torch.float64)
    combined, _ = combine_gradients(r, -r/.03, 'pcgrad')
    assert torch.equal(combined, torch.zeros_like(combined))
    combined, _ = combine_gradients(zero, r, 'pcgrad')
    torch.testing.assert_close(combined, .03*r)


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_gradients_fail_before_mutation(value):
    tx = transaction(); before = tx.means.detach().clone()
    bad = torch.zeros(1, 3); bad[0, 0] = value
    with pytest.raises(ValueError, match='finite'):
        tx.propose(bad, torch.zeros_like(bad), 'joint')
    assert torch.equal(tx.means, before) and tx.pending is None and tx.optimizer.state_dict()['state'] == {}


def test_nonfinite_adam_state_exception_rolls_back_proposal():
    tx = transaction(); before = tx.means.detach().clone()
    # Finite FP32 input whose square overflows FP32 Adam's second moment.
    with pytest.raises(ValueError, match='Nonfinite Adam state'):
        tx.propose(torch.full((1, 3), 1e30), torch.zeros(1, 3), 'joint')
    assert torch.equal(tx.means, before) and tx.pending is None and tx.means.grad is None
    assert tx.optimizer.state_dict()['state'] == {}


def test_fixed_metric_copies_inputs_and_rejects_invalid_geometry():
    q = torch.tensor([[1., 0, 0, 0]])
    scales = torch.zeros(1, 3)
    rotation, inverse = fixed_metric(q, scales)
    q.zero_(); scales.add_(10)
    assert torch.equal(rotation, torch.eye(3, dtype=torch.float64)[None])
    assert torch.equal(inverse, torch.ones(1, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match='quaternion'):
        fixed_metric(q, scales)
    with pytest.raises(ValueError, match='scene scale'):
        transaction(scene_scale=float('nan'))
