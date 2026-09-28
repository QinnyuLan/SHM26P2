"""Mass semantics and gradient isolation, without real data or a GPU."""
import pytest
import torch
from torch import nn

from bridge_rgs.ibgs_layer_fusion import MassPreservingLayerFusion, aggregate_layer_evidence


def inputs(n=4):
    return torch.ones(n, 4, 4, 7), torch.zeros(n, 4), torch.zeros(n, 4, 4)


def test_small_mass_is_not_normalized_or_averaged_over_valid_sources():
    mlp = nn.Linear(7, 2)
    with torch.no_grad():
        mlp.weight.zero_(); mlp.bias.fill_(2)
    x, w, s = inputs(1); w[0, 0] = .01; s[0, 0, 0] = 1
    out, active, mass = aggregate_layer_evidence(mlp, x, w, s)
    torch.testing.assert_close(out, torch.full((1, 2), .005))
    torch.testing.assert_close(mass, torch.tensor([.0025]))
    assert active.item()
    s[0, 0, :] = 1
    out, _, _ = aggregate_layer_evidence(mlp, x, w, s)
    torch.testing.assert_close(out, torch.full((1, 2), .02))


def test_invalid_nan_evidence_cannot_leak_through_mlp_bias():
    mlp = nn.Linear(7, 2); x, w, s = inputs(1); x.fill_(float('nan'))
    out, active, mass = aggregate_layer_evidence(mlp, x, w, s)
    assert torch.equal(out, torch.zeros_like(out)) and not active.any() and not mass.any()
    w[0, 0] = .1; s[0, 0, 0] = 1
    with pytest.raises(ValueError, match='Nonfinite active'):
        aggregate_layer_evidence(mlp, x, w, s)


def test_normalized_control_changes_only_evidence_scale_and_keeps_empty_safe():
    mlp = nn.Linear(7, 2)
    with torch.no_grad():
        mlp.weight.zero_(); mlp.bias.fill_(2)
    x, w, s = inputs(2); w[0, 0] = .01; s[0, 0, 0] = .1
    x[1].fill_(float('nan'))
    raw, active, mass = aggregate_layer_evidence(mlp, x, w, s)
    norm, active_n, mass_n = aggregate_layer_evidence(mlp, x, w, s, reduction='normalized')
    torch.testing.assert_close(norm[0], torch.full((2,), 2.))
    torch.testing.assert_close(raw[0], norm[0]*mass[0])
    assert torch.equal(norm[1], torch.zeros(2))
    assert torch.equal(active, active_n) and torch.equal(mass, mass_n)
    assert torch.isfinite(norm).all()


def test_split_duplicate_layer_preserves_evidence_and_permutation_invariance():
    torch.manual_seed(12); mlp = nn.Sequential(nn.Linear(7, 5), nn.ReLU())
    x, w, s = inputs(1); w[0, 0] = .3; s[0, 0] = torch.tensor([1., .2, 0., .7])
    before = aggregate_layer_evidence(mlp, x, w, s)[0]
    w[0, 0] = .1; w[0, 1] = .2; s[0, 1] = s[0, 0]
    after = aggregate_layer_evidence(mlp, x, w, s)[0]
    torch.testing.assert_close(before, after)
    permutation = torch.tensor([3, 1, 0, 2])
    permuted = aggregate_layer_evidence(mlp, x[:, permutation][:, :, permutation],
                                        w[:, permutation], s[:, permutation][:, :, permutation])[0]
    torch.testing.assert_close(before, permuted)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.height, self.width, self.per_view_feat_dim = 2, 2, 2
        self.per_view_mlp = nn.Sequential(nn.Linear(7, 2), nn.ReLU())
        self.conv_decoder = nn.Conv2d(8, 3, 3, padding=1)
        nn.init.constant_(self.conv_decoder.bias, 1)


def test_zero_support_pixel_base_fallback_despite_cnn_neighbors_and_bias():
    backbone = TinyBackbone(); wrapped = MassPreservingLayerFusion(backbone)
    assert sum(p.numel() for p in wrapped.parameters()) == sum(p.numel() for p in backbone.parameters())
    x, w, s = inputs(); w[0, 0] = .1; s[0, 0] = 1
    ray, base = torch.zeros(4, 3), torch.rand(4, 3)
    result = wrapped(x, w, s, ray, base)
    assert torch.equal(result['image_pred'][1:], base[1:])
    assert torch.equal(result['residual'][1:], torch.zeros(3, 3))


def test_only_backbone_receives_gradients():
    backbone = TinyBackbone(); wrapped = MassPreservingLayerFusion(backbone)
    x, w, s = inputs(); w[:, 0] = .2; s[:, 0] = 1
    ray, base = torch.rand(4, 3), torch.rand(4, 3)
    for value in (x, w, s, ray, base):
        value.requires_grad_()
    wrapped(x, w, s, ray, base)['image_pred'].square().sum().backward()
    assert all(value.grad is None for value in (x, w, s, ray, base))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in backbone.parameters())
    assert sum(p.grad.abs().sum() for p in backbone.parameters()) > 0


@pytest.mark.parametrize('bad', ['negative_mass', 'mass_above_one', 'support_above_one'])
def test_invalid_mass_or_support_rejected(bad):
    x, w, s = inputs(1)
    if bad == 'negative_mass': w[0, 0] = -.1
    if bad == 'mass_above_one': w.fill_(.3)
    if bad == 'support_above_one': s.fill_(1.1)
    with pytest.raises(ValueError):
        aggregate_layer_evidence(nn.Linear(7, 2), x, w, s)
