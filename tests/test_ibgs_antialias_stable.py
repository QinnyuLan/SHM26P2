"""CPU contracts for the factored adapter; no CUDA or model/image loading."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from bridge_rgs.ibgs_antialias_stable import (
    aa_opacity_rasterizer,
    compensate_opacity,
    factor_compensation,
)


def fixture(dtype=torch.float64):
    return (torch.tensor([[.2, -.1, 2.], [1.3, .4, 3.]], dtype=dtype),
            torch.tensor([[.035, .07, .021], [.09, .025, .065]], dtype=dtype),
            torch.nn.functional.normalize(torch.tensor([[1., .3, -.1, .2], [.8, -.3, .4, .1]], dtype=dtype), dim=-1),
            torch.tensor([[.4], [.8]], dtype=dtype), torch.eye(4, dtype=dtype))


def evaluate(values, **kwargs):
    return compensate_opacity(*values, width=80, height=60, tanfovx=.8, tanfovy=.6, **kwargs)


def test_spd_with_independent_scipy_camera_covariance():
    values = list(fixture())
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_rotvec([.1, -.2, .05]).as_matrix()
    camera[:3, 3] = [.04, .02, .3]
    values[-1] = torch.tensor(camera)
    result = evaluate(values, scale_modifier=1.25)
    covariance = []
    for mean, scale, q in zip(*(v.numpy() for v in values[:3]), strict=True):
        rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        cov = camera[:3, :3]@rotation@np.diag((1.25*scale)**2)@rotation.T@camera[:3, :3].T
        x, y, z = camera[:3, :3]@mean+camera[:3, 3]
        jacobian = np.array([[50/z, 0, -50*np.clip(x/z, -1.04, 1.04)/z],
                             [0, 50/z, -50*np.clip(y/z, -.78, .78)/z]])
        covariance.append(jacobian@cov@jacobian.T)
    covariance = np.asarray(covariance)
    # Only well-conditioned cases use direct Gram determinants as a reference.
    det = np.linalg.det(covariance)
    blurred = np.linalg.det(covariance+.3*np.eye(2))
    np.testing.assert_allclose(result.covariance2d, covariance, rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(result.det_original, det, rtol=2e-14)
    np.testing.assert_allclose(result.rho, np.sqrt(det/blurred), rtol=2e-14)


def test_extreme_shear_uses_analytic_minor_not_cancelled_gram_reference():
    factor = torch.tensor([[[1e4, 0., 0.], [1e4, 1e-3, 0.]]], dtype=torch.float64, requires_grad=True)
    rho, determinant, blurred = factor_compensation(factor)
    expected_d = 100.
    expected_trace = 2e8+1e-6
    assert determinant.item() == expected_d
    assert blurred.item() == pytest.approx(expected_d+.3*expected_trace+.09, rel=1e-15)
    assert rho.item() == pytest.approx(np.sqrt(expected_d/(expected_d+.3*expected_trace+.09)), rel=1e-15)
    gradient, = torch.autograd.grad(rho.sum(), factor)
    assert bool(torch.isfinite(gradient).all()) and float(gradient.norm()) > 0


def test_actual_failed_slot_from_bound_cpu_capture():
    path = Path('/mnt/data/SHM2026/runs/ibgs_aa_determinant_diagnostic_v1/bad_slots.npz')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '7929ccb8b6f910efe709dfcb1f7e6c3a61b1590a13c0266244963f941bb0ab37'
    with np.load(path, allow_pickle=False) as data:
        assert data['indices'].tolist() == [740284]
        assert data['det_blur32'][0] < 0
        values = [torch.from_numpy(data[k].copy()) for k in ('means32', 'scales32', 'q32', 'opacity32', 'w2c32')]
        original = [v.clone() for v in values]
        for v in values[:4]:
            v.requires_grad_()
        # Bound camera 004 centered K: fx=fy=925.7016189245708, 1320x989.
        result = compensate_opacity(*values, width=1320, height=989,
                                    tanfovx=1320/(2*925.7016189245708),
                                    tanfovy=989/(2*925.7016189245708))
        np.testing.assert_allclose(result.factor2d.detach(), data['factor64'], rtol=3e-14, atol=2e-11)
        np.testing.assert_allclose(result.det_original.detach(), data['det_factor64'], rtol=3e-12)
        np.testing.assert_allclose(result.det_blurred.detach(), data['det_blur_factor64'], rtol=3e-12)
        np.testing.assert_allclose(result.rho.detach(), data['rho_factor64'], rtol=2e-13)
        assert result.opacity.dtype == torch.float32 and result.rho.dtype == torch.float64
        expected = (original[3].double()*torch.from_numpy(data['rho_factor64'])[:, None]).float()
        assert torch.equal(result.opacity.detach(), expected)
        grads = torch.autograd.grad(result.opacity.sum(), values[:4])
        assert all(bool(torch.isfinite(g).all()) for g in grads)
        assert all(torch.equal(v.detach(), before) for v, before in zip(values, original, strict=True))


def test_rank_one_and_zero_have_explicit_finite_zero_derivative():
    factor = torch.tensor([[[1., 2., 3.], [2., 4., 6.]], [[0., 0., 0.], [0., 0., 0.]]],
                          dtype=torch.float64, requires_grad=True)
    rho, determinant, blurred = factor_compensation(factor)
    assert torch.equal(determinant, torch.zeros(2, dtype=torch.float64))
    assert torch.equal(rho, torch.zeros_like(rho)) and bool((blurred > 0).all())
    grad, = torch.autograd.grad(rho.sum(), factor)
    assert torch.equal(grad, torch.zeros_like(grad))


def test_scale_law_and_isotropic_formula():
    factor = torch.tensor([[[.3, .2, .1], [-.4, .8, .2]]], dtype=torch.float64)
    rho, determinant, _ = factor_compensation(factor)
    for scale in (.001, .25, 4., 1000.):
        scaled_rho, scaled_det, scaled_blur = factor_compensation(factor*scale)
        torch.testing.assert_close(scaled_det, determinant*scale**4, rtol=1e-14, atol=0)
        expected_h = determinant*scale**4+.3*factor.square().sum()*scale**2+.09
        torch.testing.assert_close(scaled_blur, expected_h, rtol=1e-14, atol=0)
        torch.testing.assert_close(scaled_rho, torch.sqrt(determinant*scale**4/expected_h))
        assert bool(scaled_rho > rho) == (scale > 1)
    v = list(fixture()); v[0].zero_(); v[0][:, 2] = 2; v[1].fill_(.02)
    torch.testing.assert_close(evaluate(v).rho, torch.full((2,), .25/(.25+.3), dtype=torch.float64))


def test_positive_factor_gradcheck():
    factor = torch.tensor([[[.3, .2, .1], [-.4, .8, .2]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda b: factor_compensation(b)[0], (factor,), eps=1e-6, atol=1e-8, rtol=1e-5)


def test_activated_geometry_chain_gradcheck():
    means, scales, rotations, opacity, camera = fixture()
    means.requires_grad_()
    log_scale = scales.log().requires_grad_()
    raw_quat = (rotations*1.8).requires_grad_()
    logits = torch.logit(opacity).requires_grad_()
    def output(m, s, q, o):
        return evaluate((m, s.exp(), torch.nn.functional.normalize(q, dim=-1), o.sigmoid(), camera)).opacity
    # These activations belong to the caller; the module never recomputes them.
    assert torch.autograd.gradcheck(output, (means, log_scale, raw_quat, logits), eps=1e-6, atol=2e-8, rtol=2e-5)


def test_near_mask_and_fov_safe_branches_and_fp32_output():
    v = list(fixture(torch.float32))
    v[0] = torch.tensor([[0., 0., .009], [9., -8., 2.]], requires_grad=True)
    v[1].requires_grad_(); v[3].requires_grad_()
    result = evaluate(v)
    assert result.active_depth.tolist() == [False, True] and result.opacity.dtype == torch.float32
    assert torch.equal(result.factor2d[0], torch.zeros(2, 3, dtype=torch.float64))
    grads = torch.autograd.grad(result.opacity.sum(), (v[0], v[1], v[3]))
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    assert all(torch.count_nonzero(g[0]) == 0 for g in grads)
    assert grads[0][1, 0] == grads[0][1, 1] == 0 and grads[0][1, 2] != 0
    v[0] = torch.tensor([[0., 0., .01], [0., 0., 0.]])
    assert evaluate(v).active_depth.tolist() == [True, False]


def test_no_hidden_renormalization_of_activated_fp32_quaternion():
    v = list(fixture(torch.float32))
    v[2] *= 1+2*torch.finfo(torch.float32).eps
    original = v[2].clone()
    result = evaluate(v)
    assert torch.equal(v[2], original)
    v[2] = torch.nn.functional.normalize(v[2].double(), dim=-1).float()
    normalized = evaluate(v)
    assert not torch.equal(result.factor2d, normalized.factor2d)


@pytest.mark.parametrize('factor', [torch.full((1, 2, 3), float('nan'), dtype=torch.float64),
                                    torch.full((1, 2, 3), 1e200, dtype=torch.float64),
                                    torch.tensor([[[1e-100, 0., 0.], [0., 1e-100, 0.]]], dtype=torch.float64)])
def test_unrepresentable_factor_fails_without_floor(factor):
    with pytest.raises(ValueError):
        factor_compensation(factor)


def test_hook_preserves_alias_chain_default_no_graph_and_restoration():
    means, scales, rotations, opacity, camera = fixture(torch.float32)
    means.requires_grad_(); scales.requires_grad_(); opacity.requires_grad_()
    class Raster:
        def __init__(self):
            self.raster_settings = SimpleNamespace(viewmatrix=camera.T, image_width=80, image_height=60,
                                                  tanfovx=.8, tanfovy=.6, scale_modifier=1.)
        def forward(self, **kwargs):
            assert kwargs['opacities'].dtype == torch.float32
            return kwargs['opacities'].sum()
    original = Raster.forward
    with pytest.raises(RuntimeError), aa_opacity_rasterizer(Raster, capture_last=True) as state:
        out = Raster().forward(means3D=means, scales=scales, rotations=rotations, opacities=opacity)
        assert state.calls == 1 and state.last is not None
        assert all(bool(torch.isfinite(g).all()) for g in torch.autograd.grad(out, (means, scales, opacity)))
        with pytest.raises(ValueError, match='Nested'), aa_opacity_rasterizer(Raster):
            pass
        raise RuntimeError('synthetic backend exception')
    assert Raster.forward is original
    with aa_opacity_rasterizer(Raster) as state:
        Raster().forward(means3D=means, scales=scales, rotations=rotations, opacities=opacity)
        assert state.calls == 1 and state.last is None
