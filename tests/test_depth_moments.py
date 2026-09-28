import pytest
import torch

from bridge_rgs.depth_moments import semantic_depth_context


def composite(weights, depths, features):
    weights = weights[:, None]
    depths = depths[:, None]
    return [value.reshape(1, 1, -1) for value in (
        (weights * features).sum(0), (weights * depths).sum(0),
        (weights * depths.square()).sum(0), (weights * depths * features).sum(0),
        weights.sum(0))]


def test_equal_marginal_means_have_opposite_depth_feature_contrast():
    weights = torch.tensor([.5, .5], dtype=torch.float64)
    depths = torch.tensor([1., 3.], dtype=torch.float64)
    features = torch.tensor([[2., -2.], [-2., 2.]], dtype=torch.float64)
    a, b = [composite(weights, depths, f) for f in (features, features.flip(0))]
    for index in (0, 1, 2, 4):
        torch.testing.assert_close(a[index], b[index], rtol=0, atol=0)
    ca, cb = semantic_depth_context(*a), semantic_depth_context(*b)
    torch.testing.assert_close(ca[..., :1], cb[..., :1], rtol=0, atol=0)
    torch.testing.assert_close(ca[..., 1:], -cb[..., 1:], rtol=0, atol=0)
    expected = torch.asinh(torch.tensor([-2., 2.], dtype=torch.float64)/1.000001**.5)
    torch.testing.assert_close(ca[0, 0, 1:], expected)


@pytest.mark.parametrize("opacity", [0., 1e-7, .25, 1.])
def test_single_depth_returns_zero_context(opacity):
    values = composite(torch.tensor([opacity]), torch.tensor([2.]), torch.tensor([[3., -1.]]))
    result = semantic_depth_context(*values)
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result) == 0


@pytest.mark.parametrize("opacity", [.3, .99])
@pytest.mark.parametrize("depth", [10., 20., 1000.])
def test_single_distant_depth_does_not_manufacture_float32_variance(opacity, depth):
    values = composite(torch.tensor([opacity], dtype=torch.float32),
                       torch.tensor([depth], dtype=torch.float32),
                       torch.tensor([[3., -1.]], dtype=torch.float32))
    result = semantic_depth_context(*values)
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result) == 0


def test_resolvable_distant_mixture_retains_feature_contrast():
    values = composite(torch.tensor([.3, .6], dtype=torch.float32),
                       torch.tensor([990., 1010.], dtype=torch.float32),
                       torch.tensor([[3., -1.], [-1., 3.]], dtype=torch.float32))
    result = semantic_depth_context(*values)
    assert torch.isfinite(result).all()
    assert result[0, 0, 0] > 0
    assert result[0, 0, 1] < 0 and result[0, 0, 2] > 0


def test_context_is_invariant_to_uniform_coverage_scaling():
    values = composite(torch.tensor([.3, .7], dtype=torch.float64),
                       torch.tensor([1., 4.], dtype=torch.float64),
                       torch.tensor([[2., -1.], [-3., 4.]], dtype=torch.float64))
    torch.testing.assert_close(semantic_depth_context(*values),
                               semantic_depth_context(*[value*.2 for value in values]))


def test_only_feature_terms_receive_gradients():
    values = [v.requires_grad_() for v in composite(torch.tensor([.3, .7]),
              torch.tensor([1., 4.]), torch.tensor([[2., -1.], [-3., 4.]]))]
    semantic_depth_context(*values).square().sum().backward()
    for index in (0, 3):
        assert values[index].grad is not None
        assert torch.isfinite(values[index].grad).all() and values[index].grad.abs().sum() > 0
    for index in (1, 2, 4):
        assert values[index].grad is None


def test_reject_mismatched_grids_and_invalid_epsilon():
    values = composite(torch.tensor([.3, .7]), torch.tensor([1., 4.]), torch.ones(2, 2))
    with pytest.raises(ValueError, match="epsilon"):
        semantic_depth_context(*values, epsilon=0)
    values[1] = torch.ones(2, 2, 1)
    with pytest.raises(ValueError, match="grid"):
        semantic_depth_context(*values)
