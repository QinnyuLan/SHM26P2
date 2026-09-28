"""CPU checks of real point/attribute ownership and finite support changes."""
import math

import pytest
import torch

from bridge_rgs.finite_structure import split_exchange


def field():
    return {
        'means': torch.arange(18, dtype=torch.float64).reshape(6, 3),
        'quats': torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64).repeat(6, 1),
        'log_scales': torch.tensor([[2., .4, .02]], dtype=torch.float64).log().repeat(6, 1),
        'opacity_logits': torch.linspace(-2, 2, 6, dtype=torch.float64),
        'sh': torch.arange(36, dtype=torch.float64).reshape(6, 2, 3),
        'q': torch.eye(5, dtype=torch.float64)[torch.arange(6) % 5],
    }


def test_same_count_explicit_rows_no_implicit_pruning_or_source_mutation():
    p = field(); before = {k: v.clone() for k, v in p.items()}
    p['means'].requires_grad_()
    out = split_exchange(p, torch.tensor([3, 1]), torch.tensor([4, 0]))
    assert out.source_indices.tolist() == [2, 5, 3, 1, 3, 1]
    assert out.retained_count == 2
    for name, value in out.tensors.items():
        assert len(value) == 6 and not value.requires_grad
        assert torch.equal(value[:2], before[name][[2, 5]])
        assert torch.equal(p[name], before[name])
        if name not in ('means', 'log_scales', 'opacity_logits'):
            assert torch.equal(value, before[name][out.source_indices])
    out.tensors['q'].zero_()
    assert torch.equal(p['q'], before['q'])


def test_support_displacement_scale_axis_and_existing_opacity_initialization():
    p = field(); parent = torch.tensor([2]); donor = torch.tensor([5])
    out = split_exchange(p, parent, donor, torch.tensor([[1., 4., 100.]], dtype=torch.float64))
    # Axes below 1/4 of the major scale are excluded from the requested direction.
    assert out.shrink_axes.tolist() == [0]
    assert torch.allclose(out.offsets, torch.tensor([[1., 0., 0.]], dtype=torch.float64))
    assert torch.allclose(((out.offsets / p['log_scales'][parent].exp())**2).sum(-1), torch.tensor([.25], dtype=torch.float64))
    assert torch.equal(out.tensors['means'][-2:], torch.cat((p['means'][parent]-out.offsets, p['means'][parent]+out.offsets)))
    expected_scales = p['log_scales'][parent].clone(); expected_scales[:, 0] -= math.log(1.6)
    assert torch.allclose(out.tensors['log_scales'][-2:], expected_scales.repeat(2, 1))
    expected_alpha = 1-(1-p['opacity_logits'][parent].sigmoid()).sqrt()
    assert torch.allclose(out.tensors['opacity_logits'][-2:].sigmoid(), expected_alpha.repeat(2))


def test_empty_keep_and_column_opacity_supported():
    p = field(); p['opacity_logits'] = p['opacity_logits'][:, None]
    empty = torch.empty(0, dtype=torch.int64)
    keep = split_exchange(p, empty, empty)
    assert keep.retained_count == 6 and keep.source_indices.tolist() == list(range(6))
    assert all(torch.equal(p[k], keep.tensors[k]) for k in p)
    out = split_exchange(p, torch.tensor([1]), torch.tensor([5]))
    assert out.tensors['opacity_logits'].shape == (6, 1)


@pytest.mark.parametrize('parents,donors', [([1, 1], [4, 5]), ([1], [1]), ([1, 2], [5]), ([-1], [5]), ([1], [6])])
def test_rejects_invalid_point_ownership(parents, donors):
    with pytest.raises(ValueError):
        split_exchange(field(), torch.tensor(parents), torch.tensor(donors))


def test_rejects_invalid_geometry_instead_of_silently_repairing():
    p = field(); p['quats'][0] = 0
    with pytest.raises(ValueError, match='Quaternion'):
        split_exchange(p, torch.tensor([1]), torch.tensor([5]))
    p = field(); p['log_scales'][1, 0] = 1000
    with pytest.raises(ValueError, match='scales'):
        split_exchange(p, torch.tensor([1]), torch.tensor([5]))


def test_rotated_major_axis_and_quaternion_gauge_preserve_world_ownership():
    p = field()
    # A +90 degree rotation about z maps local major axis x to world +y.
    root_half = math.sqrt(.5)
    quat = torch.tensor([root_half, 0., 0., root_half], dtype=torch.float64)
    p['quats'][1] = quat
    p['quats'][3] = -7*quat  # identical normalized orientation, distinct parent row
    out = split_exchange(p, torch.tensor([3, 1]), torch.tensor([4, 0]))
    expected = torch.tensor([[0., 1., 0.], [0., 1., 0.]], dtype=torch.float64)
    torch.testing.assert_close(out.offsets, expected, atol=1e-15, rtol=0)
    assert out.shrink_axes.tolist() == [0, 0]
    assert out.source_indices.tolist() == [2, 5, 3, 1, 3, 1]
    torch.testing.assert_close(out.tensors['means'][2:4], p['means'][[3, 1]]-expected)
    torch.testing.assert_close(out.tensors['means'][4:], p['means'][[3, 1]]+expected)
    assert torch.equal(out.tensors['quats'][2:], p['quats'][[3, 1, 3, 1]])
    assert torch.equal(out.tensors['q'][2:], p['q'][[3, 1, 3, 1]])


def test_oblique_world_direction_is_filtered_in_local_ellipsoid_coordinates():
    p = field(); theta = math.pi/3
    p['quats'][2] = torch.tensor([math.cos(theta/2), 0., 0., math.sin(theta/2)], dtype=torch.float64)
    p['log_scales'][2] = torch.tensor([2., 1., .02], dtype=torch.float64).log()
    # Construct R independently from planar trigonometry, not the helper under test.
    R = torch.tensor([[math.cos(theta), -math.sin(theta), 0.],
                      [math.sin(theta), math.cos(theta), 0.], [0., 0., 1.]], dtype=torch.float64)
    local = torch.tensor([1., 3., 100.], dtype=torch.float64)
    direction = (R@local)[None]
    out = split_exchange(p, torch.tensor([2]), torch.tensor([5]), direction)
    supported = torch.tensor([1., 3., 0.], dtype=torch.float64)
    supported /= supported.norm()
    scales = torch.tensor([2., 1., .02], dtype=torch.float64)
    local_offset = .5*supported/((supported/scales).square().sum().sqrt())
    torch.testing.assert_close(out.offsets[0], R@local_offset)
    pulled = R.T@out.offsets[0]
    torch.testing.assert_close((pulled/scales).square().sum(), torch.tensor(.25, dtype=torch.float64))
    assert abs(float(pulled[2])) < 1e-15
    # Largest directional variance contribution is local y, not world y.
    assert out.shrink_axes.tolist() == [1]
    expected = p['log_scales'][2].clone(); expected[1] -= math.log(1.6)
    torch.testing.assert_close(out.tensors['log_scales'][-2:], expected.repeat(2, 1))


def test_finite_components_with_overflowing_quaternion_norm_are_rejected():
    p = field(); p['quats'][1] = 1e300
    assert torch.isfinite(p['quats']).all()
    with pytest.raises(ValueError, match='Quaternion norms must be finite'):
        split_exchange(p, torch.tensor([1]), torch.tensor([5]))
