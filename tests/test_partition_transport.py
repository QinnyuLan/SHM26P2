"""Synthetic CPU contracts for an inactive numerical adapter alternative."""
import numpy as np
import pytest
import torch
from scipy.linalg import sqrtm

from bridge_rgs.partition_transport import prepare_transport, symmetric_ot_2x2

DT = torch.float64


def rotation(angle):
    return torch.tensor([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=DT)


def fixture(condition=100., angle=.3, perturbation=1e-4):
    R = rotation(angle)
    P = R@torch.diag(torch.tensor([np.sqrt(condition-.3), np.sqrt(.7), .4], dtype=DT))[:2]
    C0 = P@P.T+.3*torch.eye(2, dtype=DT)
    Q = rotation(perturbation)
    Cr = Q@C0@Q.T
    return P, C0, (Cr+Cr.T)*.5


def test_matches_independent_scipy_symmetric_ot_at_moderate_condition():
    _, C0, Cr = fixture(100.)
    S = sqrtm(C0.numpy())
    expected = np.linalg.solve(S, sqrtm(S@Cr.numpy()@S))@np.linalg.solve(S, np.eye(2))
    T = symmetric_ot_2x2(C0, Cr)
    np.testing.assert_allclose(T.numpy(), expected, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(T, T.T, atol=0, rtol=0)
    assert torch.linalg.eigvalsh(T).min() > 0


@pytest.mark.parametrize("condition", [1., 100., 1e4, 1e6])
def test_high_anisotropy_footprint_psd_conditional_and_rotation(condition):
    P, _, Cr = fixture(condition)
    got = prepare_transport(P, Cr)
    Q = rotation(np.pi/2)
    rotated = prepare_transport(Q@P, Q@Cr@Q.T)
    torch.testing.assert_close(rotated.transport, Q@got.transport@Q.T, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(rotated.A@Q, got.A, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(rotated.V, got.V, atol=1e-8, rtol=1e-8)
    assert torch.linalg.eigvalsh(got.transport).min() > 0
    assert torch.linalg.eigvalsh(got.effective_noise).min() > 0
    assert torch.linalg.eigvalsh(got.V).min() > -128*torch.finfo(DT).eps
    assert got.diagnostics['footprint_whitened_max_residual'] < 1e-8
    assert got.diagnostics['law_total_covariance_max_residual'] < 1e-8
    assert got.diagnostics['eigenvalue_clipping'] is False


def test_common_covariance_unit_scaling_and_batch_fp32_promotion():
    _, C0, Cr = fixture()
    scale = torch.tensor([1e-12, 1., 1e12], dtype=DT)[:, None, None]
    batch = symmetric_ot_2x2(C0*scale, Cr*scale)
    torch.testing.assert_close(batch, batch[1].expand_as(batch), atol=2e-13, rtol=2e-13)
    got = symmetric_ot_2x2(C0.float(), Cr.float())
    assert got.dtype == DT and got.device.type == 'cpu' and not got.requires_grad


def test_identity_target_and_degenerate_projection_are_valid():
    P = torch.zeros(2, 2, 3, dtype=DT)
    P[1, 0, 0] = 3.
    C0 = P@P.mT+.3*torch.eye(2, dtype=DT)
    result = prepare_transport(P, C0)
    torch.testing.assert_close(result.transport, torch.eye(2, dtype=DT).expand_as(C0), atol=2e-15, rtol=2e-15)
    assert result.A[0].count_nonzero() == 0
    torch.testing.assert_close(result.V[0], torch.eye(3, dtype=DT), atol=0, rtol=0)


def test_local_gauge_and_class_mass_invariants():
    P, _, Cr = fixture(1e4)
    U = torch.tensor([[0., 0, 1.], [1., 0, 0], [0, 1., 0]], dtype=DT)
    original = prepare_transport(P, Cr)
    gauged = prepare_transport(P@U, Cr)
    torch.testing.assert_close(gauged.A, U.T@original.A, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(gauged.V, U.T@original.V@U, atol=2e-12, rtol=2e-12)
    # Total variance establishes the unchanged unit local prior and therefore
    # the conditional slab's integrated/marginal class-mass identity.
    torch.testing.assert_close(original.A@Cr@original.A.T+original.V, torch.eye(3, dtype=DT), atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("bad", ['indefinite', 'singular', 'asymmetric', 'nan'])
def test_invalid_covariance_fails_without_clipping(bad):
    C0 = torch.eye(2, dtype=DT)
    Cr = C0.clone()
    if bad == 'indefinite':
        Cr[0, 0] = -1
    elif bad == 'singular':
        Cr[0, 0] = 0
    elif bad == 'asymmetric':
        Cr[0, 1] = .1
    else:
        Cr[0, 0] = float('nan')
    with pytest.raises(ValueError):
        symmetric_ot_2x2(C0, Cr)


def test_geometry_gradients_bad_noise_and_shape_rejected():
    P, C0, Cr = fixture()
    with pytest.raises(ValueError, match='frozen'):
        prepare_transport(P.requires_grad_(), Cr)
    with pytest.raises(ValueError, match='noise_variance'):
        prepare_transport(P.detach(), Cr, noise_variance=0.)
    with pytest.raises(ValueError, match='identical shape'):
        symmetric_ot_2x2(C0[None], Cr)


@pytest.mark.parametrize('mode', ['integrated', 'point', 'marginal'])
def test_fp32_shader_coefficients_rotate_with_query(mode):
    # This is only the current CPU coefficient path, not Triton execution.
    # Dtype-default closeness tests ordinary FP32 rounding, not a new scene gate.
    from bridge_rgs.partition_projection import FrozenProjection, prepare_shader_coefficients

    P, _, Cr = fixture(1e4)
    O = torch.tensor([[0., -1.], [1., 0.]], dtype=DT)
    n = torch.tensor([[1/np.sqrt(3)]*3], dtype=torch.float32)

    def coeff(p, c):
        result = prepare_transport(p, c)
        projection = FrozenProjection(result.A[None].float(), result.V[None].float(),
                                      torch.zeros(1, 2), torch.ones(1, dtype=torch.bool), {})
        return prepare_shader_coefficients(projection, n, .5, .75, .3, mode)

    original, rotated = coeff(P, Cr), coeff(O@P, O@Cr@O.T)
    assert original.dtype == rotated.dtype == torch.float32
    torch.testing.assert_close(rotated[:, :2]@O.float(), original[:, :2])
    torch.testing.assert_close(rotated[:, 2:], original[:, 2:])
    delta = torch.tensor([[.5, -.75], [-.5, .75]], dtype=torch.float32)
    original_args = delta@original[0, :2, None]+original[:, 2:]
    rotated_args = (delta@O.float().T)@rotated[0, :2, None]+rotated[:, 2:]
    torch.testing.assert_close(rotated_args, original_args)
