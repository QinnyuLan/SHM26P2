"""Known interventions, degeneracy and immutable evidence; no image or CUDA inputs."""
import torch

from bridge_rgs.ibgs_correspondence_control import permute_with_diagnostics
from bridge_rgs.ibgs_layer_controls import permute_supported_layer_features


def inputs():
    f = torch.zeros(3, 4, 4, 7)
    w = torch.full((3, 4), .1)
    q = torch.zeros(3, 4, 4)
    # Unequal coefficients / unequal features; then equal coefficients; then equal features.
    q[:, :2, 0] = 1
    w[0, 0] = .3
    f[0:2, 0, 0, 0] = 1; f[0:2, 1, 0, 0] = 4
    w[2, 0] = .3; f[2, :2, 0] = 2
    f[:, 3, 3] = float('nan')
    return f, w, q


def test_known_counts_and_diagnostic_degeneracy():
    f, w, q = inputs(); original = [v.clone() for v in (f, w, q)]
    actual, stats = permute_with_diagnostics(f, w, q, diagnose=True, chunk_pixels=1)
    assert stats['pixel_source_groups'] == 12 and stats['supported_groups'] == 3
    assert stats['multilayer_groups'] == 3 and stats['active_feature_vectors'] == 6
    assert stats['changed_active_feature_vectors'] == 4 and stats['changed_multilayer_groups'] == 2
    assert stats['equal_coefficient_multilayer_groups'] == 1
    assert stats['identical_feature_multilayer_groups'] == 1
    assert stats['weighted_raw_sum_changed_groups'] == 1
    assert abs(stats['weighted_raw_sum_max_absolute_change']-.6) < 1e-7
    for a, b in zip((f, w, q), original, strict=True):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    expected = permute_supported_layer_features(f, w, q)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_diagnostics_do_not_change_permutation_or_attach_graph():
    tensors = inputs()
    for t in tensors:
        t.requires_grad_()
    plain, empty = permute_with_diagnostics(*tensors)
    measured, _ = permute_with_diagnostics(*tensors, diagnose=True)
    assert empty == {} and not plain.requires_grad and not measured.requires_grad
    assert torch.equal(plain.view(torch.int32), measured.view(torch.int32))


def test_no_support_reports_zero_without_masking_or_changing_nan_bits():
    f, w, q = inputs(); q.zero_()
    out, stats = permute_with_diagnostics(f, w, q, diagnose=True)
    assert stats['multilayer_groups'] == stats['weighted_raw_sum_changed_groups'] == 0
    assert stats['weighted_raw_sum_max_absolute_change'] == 0
    assert torch.equal(out.view(torch.int32), f.view(torch.int32))
