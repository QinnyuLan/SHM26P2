from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from bridge_rgs.ibgs_antialias import (
    aa_opacity_rasterizer,
    compensate_opacity,
    determinant_compensation,
)


def fixture(dtype=torch.float64):
    return (torch.tensor([[.2, -.1, 2.], [1.3, .4, 3.]], dtype=dtype),
            torch.tensor([[.035, .07, .021], [.09, .025, .065]], dtype=dtype),
            torch.nn.functional.normalize(torch.tensor([[1., .3, -.1, .2], [.8, -.3, .4, .1]], dtype=dtype), dim=-1),
            torch.tensor([[.4], [.8]], dtype=dtype), torch.eye(4, dtype=dtype))


def evaluate(values):
    return compensate_opacity(*values, width=80, height=60, tanfovx=.8, tanfovy=.6)


def test_independent_scipy_covariance_and_determinant():
    values = list(fixture())
    camera = np.eye(4); camera[:3, :3] = Rotation.from_rotvec([.1, -.2, .05]).as_matrix(); camera[:3, 3] = [.04, .02, .3]
    values[-1] = torch.tensor(camera)
    result = evaluate(values)
    expected = []
    for mean, scale, q in zip(*(v.numpy() for v in values[:3]), strict=True):
        R = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        cov = camera[:3, :3]@R@np.diag(scale**2)@R.T@camera[:3, :3].T
        x, y, z = camera[:3, :3]@mean+camera[:3, 3]
        J = np.array([[50/z, 0, -50*np.clip(x/z, -1.04, 1.04)/z],
                      [0, 50/z, -50*np.clip(y/z, -.78, .78)/z]])
        expected.append(J@cov@J.T)
    expected = np.asarray(expected)
    np.testing.assert_allclose(result.covariance2d.detach(), expected, rtol=2e-14, atol=1e-14)
    rho = np.sqrt(np.linalg.det(expected)/np.linalg.det(expected+.3*np.eye(2)))
    np.testing.assert_allclose(result.rho.detach(), rho, rtol=2e-14)
    np.testing.assert_allclose(result.opacity.detach().numpy()[:, 0], values[3].numpy()[:, 0]*rho)


def test_isotropic_axis_closed_form_and_blur_only_once():
    values = list(fixture()); values[0].zero_(); values[0][:, 2] = 2
    values[1].fill_(.02)
    result = evaluate(values)
    variance = (50*.02/2)**2
    torch.testing.assert_close(result.rho, torch.full((2,), variance/(variance+.3), dtype=torch.float64))
    assert bool((result.opacity <= values[3]).all())


def test_geometry_opacity_and_raw_quaternion_finite_differences():
    means, scales, rotations, opacity, camera = fixture()
    log_scale = scales.log().requires_grad_(); logits = torch.logit(opacity).requires_grad_()
    raw_quat = (rotations*1.8).requires_grad_(); means.requires_grad_()
    weights = torch.tensor([[.7], [-.3]], dtype=torch.float64)
    def loss(m, s, q, o):
        values = (m, s.exp(), torch.nn.functional.normalize(q, dim=-1), o.sigmoid(), camera)
        return (evaluate(values).opacity*weights).sum()
    assert torch.autograd.gradcheck(loss, (means, log_scale, raw_quat, logits), eps=1e-6, atol=2e-8, rtol=2e-5)
    gradients = torch.autograd.grad(loss(means, log_scale, raw_quat, logits), (means, log_scale, raw_quat, logits))
    assert all(bool(torch.isfinite(g).all()) and float(g.norm()) > 0 for g in gradients)


def test_fov_clamp_outside_and_safe_nonpositive_depth():
    values = list(fixture()); values[0] = torch.tensor([[9., -8., 2.], [1., 1., 0.]], dtype=torch.float64, requires_grad=True)
    values[1].requires_grad_(); values[3].requires_grad_()
    result = evaluate(values)
    assert result.rho[1] == 0 and result.opacity[1, 0] == 0
    gradient = torch.autograd.grad(result.opacity.sum(), (values[0], values[1], values[3]))
    assert all(bool(torch.isfinite(g).all()) for g in gradient)
    assert torch.equal(gradient[0][1], torch.zeros(3, dtype=torch.float64))
    assert gradient[0][0, 0] == gradient[0][0, 1] == 0
    assert gradient[0][0, 2] != 0
    assert torch.autograd.gradcheck(lambda m: evaluate([m, *values[1:]]).opacity[0],
                                    (values[0],), eps=1e-6, atol=1e-7)


def test_zero_and_negative_determinant_safe_branch_no_hidden_floor():
    cov = torch.tensor([[[0., 0.], [0., 1.]], [[1., 1.01], [1.01, 1.]]],
                       dtype=torch.float64, requires_grad=True)
    rho, determinant, _ = determinant_compensation(cov)
    assert torch.equal(rho, torch.zeros(2, dtype=torch.float64)) and determinant[1] < 0
    gradient, = torch.autograd.grad(rho.sum(), cov)
    assert torch.equal(gradient, torch.zeros_like(cov))
    with pytest.raises(ValueError, match='blurred determinant'):
        determinant_compensation(torch.tensor([[[-.3, 0.], [0., -.3]]], dtype=torch.float64))


def test_fp32_finite_and_invalid_inputs_fail():
    result = evaluate(fixture(torch.float32))
    assert result.rho.dtype == torch.float32 and bool(torch.isfinite(result.rho).all())
    for index, value in ((0, float('nan')), (1, -1.), (2, 0.), (3, 2.)):
        values = list(fixture()); values[index] = values[index].clone()
        values[index].fill_(value)
        with pytest.raises(ValueError):
            evaluate(values)


def test_hook_keeps_opacity_and_geometry_chain_and_restores_exception():
    means, scales, rotations, opacity, camera = fixture()
    means.requires_grad_(); scales.requires_grad_(); opacity.requires_grad_()
    class Raster:
        def __init__(self):
            self.raster_settings = SimpleNamespace(viewmatrix=camera.T, image_width=80, image_height=60,
                                                  tanfovx=.8, tanfovy=.6, scale_modifier=1.)
        def forward(self, **kwargs):
            return kwargs['opacities'].sum()
    original = Raster.forward; raster = Raster()
    with pytest.raises(RuntimeError), aa_opacity_rasterizer(Raster, capture_last=True) as state:
        output = raster.forward(means3D=means, scales=scales, rotations=rotations, opacities=opacity)
        grads = torch.autograd.grad(output, (means, scales, opacity))
        assert state.calls == 1 and state.last is not None and all(float(g.norm()) > 0 for g in grads)
        with pytest.raises(ValueError, match='Nested'), aa_opacity_rasterizer(Raster):
            pass
        raise RuntimeError('backend failed')
    assert Raster.forward is original and torch.equal(opacity.detach(), fixture()[3])
    with aa_opacity_rasterizer(Raster) as state:
        raster.forward(means3D=means, scales=scales, rotations=rotations, opacities=opacity)
        assert state.last is None


def test_no_covariance_precomputed_or_implicit_positional_branch():
    values = fixture()
    class Raster:
        def forward(self, **kwargs):
            raise AssertionError('must reject before backend')
    with aa_opacity_rasterizer(Raster):
        with pytest.raises(ValueError, match='keyword'):
            Raster().forward(values[0])
        with pytest.raises(ValueError, match='Precomputed'):
            Raster().forward(means3D=values[0], scales=values[1], rotations=values[2],
                             opacities=values[3], cov3D_precomp=torch.ones(2, 6))
