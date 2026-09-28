"""No real data, renderer, or default-pipeline integration."""
import pytest
import torch

from bridge_rgs.ibgs_layer_controls import permute_supported_layer_features


def example(n=1):
    features = torch.arange(n*4*4*7, dtype=torch.float32).reshape(n, 4, 4, 7)
    return features, torch.full((n, 4), .1), torch.zeros(n, 4, 4)


def test_known_two_three_four_layer_cyclic_direction():
    f, w, q = example()
    q[0, [0, 2], 0] = 1
    q[0, [0, 2, 3], 1] = 1
    q[0, :, 2] = 1
    q[0, 1, 3] = 1
    out = permute_supported_layer_features(f, w, q)
    for source, active, next_active in ((0, [0, 2], [2, 0]),
                                        (1, [0, 2, 3], [2, 3, 0]),
                                        (2, [0, 1, 2, 3], [1, 2, 3, 0]),
                                        (3, [1], [1])):
        assert torch.equal(out[0, active, source], f[0, next_active, source])


def test_multiset_coefficients_support_and_mass_unchanged():
    f, w, q = example(2)
    q[0, [0, 2, 3], 0] = torch.tensor([.1, .5, 1.])
    q[1, :, 3] = 1
    old_f, old_w, old_q = (v.clone() for v in (f, w, q))
    before = w[..., None]*q
    out = permute_supported_layer_features(f, w, q)
    for pixel in range(2):
        for source in range(4):
            selected = before[pixel, :, source] > 0
            a = sorted(map(tuple, out[pixel, selected, source].tolist()))
            b = sorted(map(tuple, f[pixel, selected, source].tolist()))
            assert a == b
    assert torch.equal(w, old_w) and torch.equal(q, old_q) and torch.equal(f, old_f)
    assert torch.equal(before, w[..., None]*q)
    assert torch.equal(before.sum((1, 2)), (w[..., None]*q).sum((1, 2)))


def test_zero_singleton_and_inactive_nan_preserve_exact_bits():
    f, w, q = example(2)
    f[0].fill_(float('nan'))
    f[1, :, 0] = -0.
    q[1, 2, 1] = 1
    out = permute_supported_layer_features(f, w, q)
    assert torch.equal(out.view(torch.int32), f.view(torch.int32))
    assert out.data_ptr() != f.data_ptr()
    q[1, [1, 3], 2] = 1
    f[1, 0, 2] = float('nan')
    out = permute_supported_layer_features(f, w, q)
    inactive = w[..., None]*q == 0
    assert torch.equal(out[inactive].view(torch.int32), f[inactive].view(torch.int32))


def test_unequal_coefficients_and_distinct_features_change_pooled_value():
    f, w, q = example()
    f.zero_(); w.zero_()
    f[0, 0, 0, 0] = 1; f[0, 2, 0, 0] = 4
    w[0, 0] = .3; w[0, 2] = .1; q[0, [0, 2], 0] = 1
    out = permute_supported_layer_features(f, w, q)
    mass = w[..., None]*q
    before = (f[..., 0]*mass).sum()/4
    after = (out[..., 0]*mass).sum()/4
    torch.testing.assert_close(before, torch.tensor(.175))
    torch.testing.assert_close(after, torch.tensor(.325))
    assert torch.equal(mass, w[..., None]*q)


def test_equal_composite_coefficients_degenerate_even_for_nonlinear_features():
    f, w, q = example()
    w.zero_(); w[0, 0] = .3; w[0, 2] = .15
    q[0, 0, 0] = .5; q[0, 2, 0] = 1
    out = permute_supported_layer_features(f, w, q)
    mass = (w[..., None]*q)[..., None]
    before = (f.square()*mass).sum((1, 2))
    after = (out.square()*mass).sum((1, 2))
    assert torch.equal(before, after)


def test_identical_features_degenerate_with_unequal_coefficients():
    f, w, q = example()
    f.fill_(2); q[0, :, 0] = torch.tensor([.1, .2, .4, .8])
    assert torch.equal(permute_supported_layer_features(f, w, q), f)


def test_detached_and_actual_dtype_underflow_matches_fusion_active_mask():
    f, w, q = example()
    q[0, 0, 0] = 1
    w[0, 1] = torch.finfo(torch.float32).tiny
    q[0, 1, 0] = torch.finfo(torch.float32).tiny
    for value in (f, w, q):
        value.requires_grad_()
    out = permute_supported_layer_features(f, w, q)
    assert torch.equal(out, f) and not out.requires_grad
    assert all(v.grad is None for v in (f, w, q))


@pytest.mark.parametrize('bad', ['shape', 'nan', 'negative', 'above_one'])
def test_invalid_contract_rejected(bad):
    f, w, q = example()
    if bad == 'shape':
        f = f[..., :6]
    elif bad == 'nan':
        q[0, 0, 0] = float('nan')
    elif bad == 'negative':
        w[0, 0] = -.1
    else:
        q[0, 0, 0] = 1.1
    with pytest.raises(ValueError):
        permute_supported_layer_features(f, w, q)
