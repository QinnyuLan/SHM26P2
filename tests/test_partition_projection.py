"""CPU-only pinhole/EWA and shader coefficient contracts."""
import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from bridge_rgs.partition_projection import (
    REVISION,
    prepare_projection,
    prepare_shader_coefficients,
)
from bridge_rgs.semantic_partition import evaluate_coefficients, mixprob, prepared_coefficients

DT = torch.float64


def scene(dtype=DT):
    means = torch.tensor([[.3, -.2, 3.], [1.1, .4, 5.], [90., -70., 2.]], dtype=dtype)
    quats = torch.tensor([[1., .2, -.3, .1], [.3, .2, .1, .4], [1., 0, 0, 0]], dtype=dtype)
    logs = torch.tensor([[.12, .2, .04], [.2, .06, .1], [.08, .03, .2]], dtype=dtype).log()
    pose = torch.eye(4, dtype=dtype)
    pose[:3, :3] = torch.tensor(Rotation.from_rotvec([.02, -.03, .07]).as_matrix(), dtype=dtype)
    pose[:3, 3] = torch.tensor([.1, -.2, .3], dtype=dtype)
    K = torch.tensor([[400., 0, 101.], [0, 370., 77.], [0, 0, 1.]], dtype=dtype)
    return means, quats, logs, pose, K


def independent_numpy_geometry(means, quats, logs, pose, K, width=640, height=480):
    # scipy uses xyzw rather than the renderer's wxyz; its own normalization
    # and rotation implementation are independent of the adapter helper.
    scales = np.exp(logs)
    rows = []
    for i in range(len(means)):
        xyz = pose[:3, :3] @ means[i] + pose[:3, 3]
        x, y, z = xyz
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        tx = z * np.clip(x/z, -cx/fx-.15*width/fx, (width-cx)/fx+.15*width/fx)
        ty = z * np.clip(y/z, -cy/fy-.15*height/fy, (height-cy)/fy+.15*height/fy)
        J = np.array([[fx/z, 0, -fx*tx/z**2], [0, fy/z, -fy*ty/z**2]])
        q = quats[i]
        R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        P = J @ pose[:3, :3] @ R @ np.diag(scales[i])
        C = P @ P.T + .3*np.eye(2)
        A = np.linalg.solve(C, P).T
        V = np.eye(3)-A@P
        rows.append((P, C, A, V, [fx*x/z+cx, fy*y/z+cy]))
    return rows


def test_clamped_jacobian_actual_principal_point_unclipped_mean_and_quaternion():
    values = scene()
    projection = prepare_projection(*values, 640, 480)
    expected = independent_numpy_geometry(*(v.numpy() for v in values))
    for i, (_, _, A, V, mean) in enumerate(expected):
        np.testing.assert_allclose(projection.A[i].numpy(), A, atol=2e-14)
        np.testing.assert_allclose(projection.V[i].numpy(), V, atol=2e-14)
        np.testing.assert_allclose(projection.means2d[i].numpy(), mean, atol=2e-11)
    assert projection.means2d[2, 0] > 640  # Mean is not clamped with its Jacobian.
    assert projection.visible.all()
    assert projection.diagnostics["mask_source"] == "depth_candidates_only_not_full_visibility"
    # Noncentral K makes the two clip limits asymmetric, not +/-1.3*fov.
    changed_K = values[-1].clone(); changed_K[0, 2] = 320
    changed = prepare_projection(*values[:-1], changed_K, 640, 480)
    assert not torch.allclose(changed.A[2], projection.A[2])


@pytest.mark.parametrize("mode", ["integrated", "point", "marginal"])
def test_shader_coefficients_match_reference_and_delta_sign(mode):
    values = scene()
    projection = prepare_projection(*values, 640, 480)
    expected = independent_numpy_geometry(*(v.numpy() for v in values))
    P = torch.tensor(np.stack([v[0] for v in expected]), dtype=DT)
    n = torch.tensor([[.6, 0, .8], [0, 1., 0], [.8, .6, 0]], dtype=DT)
    b = torch.tensor([.1, -.2, .3], dtype=DT)
    w, tau = torch.tensor([.4, .5, .6], dtype=DT), torch.tensor([.2, .3, .4], dtype=DT)
    coeff = prepare_shader_coefficients(projection, n, b, w, tau, mode)
    reference = prepared_coefficients(P, .3*torch.eye(2, dtype=DT), n, b, w, tau, mode)
    delta = torch.tensor([[[3., -2.], [-4., 1.], [2., 5.]],
                          [[-1., 3.], [2., -4.], [1., -6.]]], dtype=DT)
    affine = (delta*coeff[None, :, :2]).sum(-1)
    shader = torch.special.ndtr(affine+coeff[:, 2])-torch.special.ndtr(affine+coeff[:, 3])
    torch.testing.assert_close(shader, evaluate_coefficients(delta, reference), atol=1e-13, rtol=1e-12)


def test_supplied_metadata_binding_and_inactive_nan_outputs():
    values = scene()
    expected = independent_numpy_geometry(*(v.numpy() for v in values))
    means = torch.tensor(np.array([v[-1] for v in expected]), dtype=DT)
    inverse = np.array([np.linalg.solve(v[1], np.eye(2)) for v in expected])
    conics = torch.tensor(inverse[:, [0, 0, 1], [0, 1, 1]], dtype=DT)
    radii = torch.tensor([[30, 20], [10, 15], [0, 0]], dtype=torch.int32)
    means[2] = float("nan"); conics[2] = float("nan")
    projection = prepare_projection(*values, 640, 480, radii=radii, means2d=means, conics=conics)
    assert projection.visible.tolist() == [True, True, False]
    assert torch.equal(projection.means2d[:2], means[:2])
    assert projection.A[2].count_nonzero() == projection.V[2].count_nonzero() == 0
    n = torch.tensor([[.6, 0, .8], [0, 1., 0], [float("nan")]*3], dtype=DT, requires_grad=True)
    b = torch.tensor([.1, .2, float("nan")], dtype=DT, requires_grad=True)
    coeff = prepare_shader_coefficients(projection, n, b, .5, .2)
    assert coeff[2].tolist() == [0., 0., 0., 0.]
    coeff.sum().backward()
    assert n.grad[2].tolist() == [0., 0., 0.] and b.grad[2] == 0
    assert torch.isfinite(n.grad).all() and torch.isfinite(b.grad).all()
    bad_conic = conics.clone(); bad_conic[0, 0] = -1
    with pytest.raises(ValueError, match="conics must be finite SPD"):
        prepare_projection(*values, 640, 480, radii=radii, conics=bad_conic)
    bad_mean = means.clone(); bad_mean[0, 0] = float("nan")
    with pytest.raises(ValueError, match="means must be finite"):
        prepare_projection(*values, 640, 480, radii=radii, means2d=bad_mean)


def test_near_far_inclusive_and_invalid_inactive_rows_are_safe():
    means = torch.tensor([[0., 0, 0], [0, 0, .01], [0, 0, 1e6], [0, 0, 1e6+1],
                          [0, 0, -1], [float("nan"), 0, 2]], dtype=DT)
    quats = torch.tensor([[1., 0, 0, 0]], dtype=DT).repeat(6, 1)
    quats[0] = 0
    logs = torch.full((6, 3), math.log(.1), dtype=DT)
    pose = torch.eye(4, dtype=DT)
    K = torch.tensor([[20., 0, 10.], [0, 20., 10.], [0, 0, 1.]], dtype=DT)
    result = prepare_projection(means, quats, logs, pose, K, 20, 20)
    assert result.visible.tolist() == [False, True, True, False, False, False]
    assert torch.isfinite(result.A).all() and torch.isfinite(result.V).all()
    active = torch.ones(6, 2, dtype=torch.int32)
    with pytest.raises(ValueError, match="Active gsplat row"):
        prepare_projection(means, quats, logs, pose, K, 20, 20, radii=active)
    empty = prepare_projection(means, quats, logs, pose, K, 20, 20, radii=active*0)
    n = torch.full((6, 3), float("nan"), dtype=DT)
    assert prepare_shader_coefficients(empty, n, float("nan"), -1., 0.).count_nonzero() == 0


@pytest.mark.parametrize("mode", ["integrated", "point", "marginal"])
def test_learnable_parameters_gradcheck_geometry_stays_frozen(mode):
    projection = prepare_projection(*scene(), 640, 480)
    n = torch.tensor([[.6, 0, .8], [.2, 1., .3], [.8, .6, .1]], dtype=DT, requires_grad=True)
    b = torch.tensor([.1, .2, -.3], dtype=DT, requires_grad=True)
    w = torch.tensor([.4, .6, .3], dtype=DT, requires_grad=True)
    tau = torch.tensor([.2, .3, .4], dtype=DT, requires_grad=True)

    def function(normal, offset, width, smoothing):
        return prepare_shader_coefficients(projection, normal/normal.norm(dim=-1, keepdim=True),
                                            offset, width, smoothing, mode)

    assert torch.autograd.gradcheck(function, (n, b, w, tau), eps=1e-6, atol=2e-6, rtol=1e-4)
    assert not projection.A.requires_grad and not projection.V.requires_grad


def test_fp32_and_shape_domain_failures():
    values = scene(torch.float32)
    p = prepare_projection(*values, 640, 480)
    n = torch.tensor([[1., 0, 0]]).expand(3, 3)
    assert torch.isfinite(prepare_shader_coefficients(p, n, 0., .5, .2)).all()
    with pytest.raises(ValueError, match="explicitly frozen"):
        prepare_projection(values[0].clone().requires_grad_(), *values[1:], 640, 480)
    with pytest.raises(ValueError, match="also requires"):
        prepare_projection(*values, 640, 480, means2d=p.means2d)
    with pytest.raises(ValueError, match="Invalid active slab"):
        prepare_shader_coefficients(p, n*2, 0., .5, .2)
    with pytest.raises(ValueError, match="Invalid active slab"):
        prepare_shader_coefficients(p, n, 0., .5, 0.)


def transported_fixture(dtype=DT, anisotropic=False):
    values = scene(dtype)
    if anisotropic:
        values[2][0] = torch.tensor([80., .002, .04], dtype=dtype).log()
    independent = independent_numpy_geometry(*(v.double().numpy() for v in values))
    C0 = torch.tensor(np.array([v[1] for v in independent]), dtype=DT)
    # Fixed, positive-definite congruence perturbation of the actual footprint.
    transform = torch.tensor([[1.003, .002], [-.001, .998]], dtype=DT)
    Cr = transform @ C0 @ transform.T
    inverse = torch.linalg.solve(Cr, torch.eye(2, dtype=DT).expand(3, 2, 2))
    conics = inverse[:, [0, 0, 1], [0, 1, 1]].to(dtype)
    means = torch.tensor(np.array([v[-1] for v in independent]), dtype=dtype)
    # The center belongs to the actual rasterizer. Near-zero cancellation is
    # diagnostic, not a relative output-value identity test in this revision.
    means[0, 0] = 0
    radii = torch.ones(3, 2, dtype=torch.int32)
    projection = prepare_projection(*values, 640, 480, radii=radii, means2d=means, conics=conics)
    actual_inverse = torch.stack((conics[:, 0], conics[:, 1], conics[:, 1], conics[:, 2]), -1).reshape(3, 2, 2).double()
    Cr = torch.linalg.solve(actual_inverse, torch.eye(2, dtype=DT).expand(3, 2, 2))
    return projection, Cr, values, conics, means, radii


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transport_high_anisotropy_psd_total_covariance_and_actual_footprint(dtype):
    p, Cr, *_ = transported_fixture(dtype, anisotropic=True)
    A, V = p.A.double(), p.V.double()
    assert p.diagnostics["revision"] == REVISION
    assert p.diagnostics["old_conic_gate_exceeded_rows"] > 0
    assert p.diagnostics["old_mean_gate_exceeded_rows"] > 0
    assert p.diagnostics["effective_noise_max_absolute_change_from_ideal"] > 0
    assert p.diagnostics["posterior_joseph_absolute_difference"] < 1e-8
    assert p.diagnostics["footprint_relative_reconstruction_error"] < 1e-8
    assert p.diagnostics["conditional_total_covariance_absolute_residual"] < 1e-8
    assert torch.linalg.eigvalsh(V).min() >= -128*torch.finfo(dtype).eps
    assert torch.linalg.eigvalsh(V).max() <= 1+128*torch.finfo(dtype).eps
    closure = A@Cr@A.mT+V-torch.eye(3, dtype=DT)
    # Coefficient storage remains FP32 for the shader. Its covariance closure
    # error grows with anisotropy, so bound the actual casts algebraically,
    # rather than inventing a data-dependent absolute allclose tolerance.
    eps = torch.finfo(dtype).eps
    product_magnitude = A.abs()@Cr.abs()@A.mT.abs()
    # This fixture obtains Cr with a general solve, whereas production uses
    # two SPD Cholesky solves. Include their FP64 conditioning amplification.
    solve_bound = 16*torch.finfo(DT).eps*torch.linalg.cond(Cr)
    cast_bound = ((2*eps+eps**2)*product_magnitude + eps*V.abs()
                  + 64*torch.finfo(DT).eps*(product_magnitude+V.abs()+1)
                  + solve_bound[:, None, None])
    assert (closure.abs() <= cast_bound).all()
    # P'=Cr A.T and the remaining covariance describe a valid joint Gaussian.
    effective_P = Cr@A.mT
    effective_noise = Cr-effective_P@effective_P.mT
    # Use the FP64 output for strict PSD; FP32 coefficient storage may lose
    # tiny positive eigenvalues after reconstructing large cancelling terms.
    if dtype == DT:
        assert torch.linalg.eigvalsh(effective_noise).min() > 0
    assert p.means2d[0, 0] == 0


@pytest.mark.parametrize("mode", ["integrated", "point", "marginal"])
def test_transported_coefficients_learnable_gradcheck(mode):
    projection, *_ = transported_fixture()
    n = torch.tensor([[.6, 0, .8], [.2, 1., .3], [.8, .6, .1]], dtype=DT, requires_grad=True)
    b = torch.tensor([.1, .2, -.3], dtype=DT, requires_grad=True)
    w = torch.tensor([.4, .6, .3], dtype=DT, requires_grad=True)
    tau = torch.tensor([.2, .3, .4], dtype=DT, requires_grad=True)
    def fn(n, b, w, tau):
        return prepare_shader_coefficients(projection, n/n.norm(dim=-1, keepdim=True), b, w, tau, mode)
    assert torch.autograd.gradcheck(fn, (n, b, w, tau), eps=1e-6, atol=2e-6, rtol=1e-4)


def test_transported_integrated_mass_is_same_prior_marginal_and_simplex():
    p, Cr, *_ = transported_fixture()
    n = torch.tensor([[.6, 0, .8], [0, 1., 0], [.8, .6, 0]], dtype=DT)
    coeff = prepare_shader_coefficients(p, n, .1, .7, .8)
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    grid = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), -1).reshape(-1, 2)*np.sqrt(2)
    weights = torch.tensor((weights[:, None]*weights[None, :]).reshape(-1)/np.pi, dtype=DT)
    delta = torch.einsum("nij,rj->rni", torch.linalg.cholesky(Cr), torch.tensor(grid, dtype=DT))
    affine = (delta*coeff[None, :, :2]).sum(-1)
    gate = torch.special.ndtr(affine+coeff[:, 2])-torch.special.ndtr(affine+coeff[:, 3])
    expected = torch.special.ndtr(torch.tensor((-.1+.7)/np.sqrt(1+.8**2)))-torch.special.ndtr(torch.tensor((-.1-.7)/np.sqrt(1+.8**2)))
    torch.testing.assert_close((weights[:, None]*gate).sum(0), expected.expand(3), atol=2e-12, rtol=2e-12)
    qi = torch.tensor([[.1, .2, .3, .15, .25]], dtype=DT).expand(3, 5)
    qo = torch.tensor([[.6, .1, .1, .1, .1]], dtype=DT).expand(3, 5)
    q = mixprob(gate, qi, qo)
    torch.testing.assert_close(q.sum(-1), torch.ones_like(gate), atol=2e-15, rtol=2e-15)
    W = torch.tensor([.2, .1, .3], dtype=DT)
    rendered = (q*W[None, :, None]).sum(1)
    rendered[:, 0] += 1-W.sum()
    torch.testing.assert_close(rendered.sum(-1), torch.ones(len(grid), dtype=DT), atol=2e-15, rtol=2e-15)
