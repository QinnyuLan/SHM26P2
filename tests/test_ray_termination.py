import pytest
import torch

from bridge_rgs.ray_termination import (
    alpha_termination_weights,
    conservative_depth_bins,
    front_mass_bounds,
    reference_front_mass,
    sample_corner_pixel_features,
)


def test_alpha_mass_is_unnormalized_and_keeps_occlusion():
    alpha = torch.tensor([.2, .5, .8], dtype=torch.float64)
    weights = alpha_termination_weights(alpha)
    torch.testing.assert_close(weights, torch.tensor([.2, .4, .32], dtype=torch.float64))
    mass = reference_front_mass(alpha, torch.tensor([1., 2., 4.]), torch.tensor([1., 2., 3., 5.]))
    torch.testing.assert_close(mass, torch.tensor([0., .2, .6, .92], dtype=torch.float64))
    assert mass[-1] < 1  # No alpha normalization: remaining .08 transmits to background.


def test_expected_depth_can_be_exact_despite_front_termination_mass():
    alpha = torch.tensor([.25, 2/3, 1.], dtype=torch.float64)
    z = torch.tensor([1., 5., 9.], dtype=torch.float64)
    weight = alpha_termination_weights(alpha)
    expected = (weight*z).sum()/weight.sum()
    assert expected == 5.
    mass = reference_front_mass(alpha, z, torch.tensor([4.9]))
    torch.testing.assert_close(mass, torch.tensor([.25], dtype=torch.float64))


@pytest.mark.parametrize("channels", [16, 32])
def test_floor_edges_are_conservative_and_bound_exact_discrete_mass(channels):
    depth = torch.logspace(-1, 2, 100, dtype=torch.float64, requires_grad=True)
    std = (depth.detach() * .01).requires_grad_()
    bins = conservative_depth_bins(depth, std, channels=channels)
    assert bins.covered.all()
    assert not bins.edges.requires_grad and not bins.cutoff.requires_grad
    assert (bins.edges[bins.lower_index] <= bins.cutoff).all()
    assert (bins.gap >= 0).all()
    z = torch.linspace(.01, 110, 500, dtype=torch.float64)
    alpha = torch.full((500,), .01, dtype=torch.float64)
    samples = reference_front_mass(alpha, z, bins.edges)
    exact = reference_front_mass(alpha, z, bins.cutoff)
    lower = samples[bins.lower_index]
    upper = samples[(bins.lower_index+1).clamp_max(len(samples)-1)]
    assert (lower <= exact+1e-14).all()
    assert (exact <= upper+1e-14).all()


def test_margin_uses_position_std_and_marks_bad_targets_uncovered():
    depth = torch.tensor([10., 10., .001, float("nan")])
    std = torch.tensor([.01, 1., .0, .1])
    bins = conservative_depth_bins(depth, std)
    torch.testing.assert_close(bins.cutoff[:2], torch.tensor([9.8, 7.]))
    assert bins.covered.tolist() == [True, True, False, False]
    assert bins.lower_index[2:].tolist() == [-1, -1]
    assert torch.isnan(bins.gap[2:]).all()


def test_only_opacity_has_gradients_and_front_loss_reaches_live_alpha():
    logits = torch.tensor([-.5, .2, 1.], requires_grad=True)
    z = torch.tensor([1., 3., 5.], requires_grad=True)
    edges = torch.tensor([2., 4.], requires_grad=True)
    reference_front_mass(logits.sigmoid(), z, edges).sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert (logits.grad[:2] > 0).all()
    assert logits.grad[-1] == 0
    assert z.grad is None and edges.grad is None


def test_degenerate_or_empty_depth_range_does_not_invent_edges():
    bins = conservative_depth_bins(torch.tensor([5., 5.]), torch.zeros(2))
    assert bins.edges.shape == (1,) and bins.covered.all()
    empty = conservative_depth_bins(torch.tensor([float("nan")]), torch.zeros(1))
    assert len(empty.edges) == 0 and not empty.covered.any()


def test_unsorted_ray_cannot_silently_claim_correct_transmittance():
    with pytest.raises(ValueError, match="increasing depth"):
        reference_front_mass(torch.tensor([.2, .3]), torch.tensor([2., 1.]), torch.tensor([3.]))


def test_nested_16_32_edges_tighten_lower_bound_without_comparing_labels():
    depth = torch.linspace(.1, 100, 100, dtype=torch.float64)
    std = depth * .01
    coarse, fine = [conservative_depth_bins(depth, std, channels=k) for k in (16, 32)]
    assert all(bool((edge == fine.edges).any()) for edge in coarse.edges)
    z = torch.linspace(.01, 110, 500, dtype=torch.float64)
    alpha = torch.full((500,), .01, dtype=torch.float64)
    low = reference_front_mass(alpha, z, coarse.edges)[coarse.lower_index]
    high = reference_front_mass(alpha, z, fine.edges)[fine.lower_index]
    assert (high >= low).all()


def test_bracketing_outside_edges_and_equality_never_interpolates_upward():
    mass = torch.tensor([[.1, .6], [.1, .6], [.1, .6], [.1, .6]])
    lower, upper = front_mass_bounds(mass, torch.full((4,), .8),
                                    torch.tensor([2., 4.]), torch.tensor([1., 2., 3., 5.]))
    torch.testing.assert_close(lower, torch.tensor([0., .1, .1, .6]))
    torch.testing.assert_close(upper, torch.tensor([.1, .1, .6, .8]))


def test_colmap_first_pixel_center_is_half_half_without_half_pixel_shift():
    grid = torch.tensor([[[1.], [3.]], [[5.], [7.]]], requires_grad=True)
    pixels = torch.tensor([[.5, .5], [1.5, 1.5], [1., 1.]], requires_grad=True)
    sampled = sample_corner_pixel_features(grid, pixels)
    torch.testing.assert_close(sampled[:, 0], torch.tensor([1., 7., 4.]))
    sampled.sum().backward()
    assert grid.grad is not None and pixels.grad is None
