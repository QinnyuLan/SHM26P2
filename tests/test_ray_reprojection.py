"""Physical projection identities and numerical bounds; no bridge data/GPU."""
import pytest
import torch

from bridge_rgs.ray_reprojection import project_inverse_depth, shared_depth_interval


def fixture():
    dtype = torch.float64
    rays = torch.tensor([[.2, -.1, 1.], [-.6, .3, 1.]], dtype=dtype)
    rho = torch.tensor([[.1, .5], [.2, .05]], dtype=dtype)
    matrices = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
    matrices[:, :3, 3] = torch.tensor([[.3, -.1, .2], [-.4, .2, -.3], [0, 0, 0]], dtype=dtype)
    focal = torch.tensor([[800, 810], [900, 850], [700, 700]], dtype=dtype)
    principal = torch.tensor([[659.5, 494], [619.5, 474], [599.5, 449]], dtype=dtype)
    return rays, rho, matrices, focal, principal


def test_matches_independent_world_point_projection_and_gradcheck():
    rays, rho, transforms, focal, principal = fixture()
    result = project_inverse_depth(rays, rho, transforms, focal, principal)
    for n in range(2):
        for layer in range(2):
            point = torch.cat((rays[n]/rho[n, layer], torch.ones(1, dtype=rho.dtype)))
            for s in range(3):
                source = (transforms[s] @ point)[:3]
                assert torch.allclose(result['uv'][n, layer, s], focal[s]*source[:2]/source[2]+principal[s])
                assert result['source_z'][n, layer, s] == pytest.approx(source[2].item())
    assert torch.autograd.gradcheck(lambda x: project_inverse_depth(rays, x, transforms, focal, principal)['uv'],
                                   (rho.requires_grad_(),))
    eps = 1e-6
    difference = (project_inverse_depth(rays, rho+eps, transforms, focal, principal)['uv']
                  - project_inverse_depth(rays, rho-eps, transforms, focal, principal)['uv'])/(2*eps)
    assert torch.allclose(result['duv_drho'], difference, atol=1e-7, rtol=1e-6)


def test_pure_rotation_has_zero_depth_parallax():
    rays, rho, transforms, focal, principal = fixture()
    angle = torch.tensor(.2, dtype=rho.dtype)
    transforms[:, :3, 3] = 0
    transforms[:, :3, :3] = torch.tensor([[angle.cos(), 0, angle.sin()], [0, 1, 0],
                                         [-angle.sin(), 0, angle.cos()]], dtype=rho.dtype)
    result = project_inverse_depth(rays, rho, transforms, focal, principal)
    assert torch.count_nonzero(result['duv_drho']) == 0
    assert torch.allclose(result['uv'][:, 0], result['uv'][:, 1])


def test_exact_multi_source_radius_and_positive_depth_bound():
    rays, rho, transforms, focal, principal = fixture()
    center = project_inverse_depth(rays, rho, transforms, focal, principal)
    bounded = shared_depth_interval(rays, rho, transforms, focal, principal, radius_pixels=2)
    interval = bounded['delta_interval']
    for proportion in torch.linspace(0, 1, 41):
        delta = interval[..., 0]*(1-proportion)+interval[..., 1]*proportion
        shifted = project_inverse_depth(rays, rho+delta, transforms, focal, principal)
        assert shifted['valid'].all()
        assert torch.linalg.vector_norm(shifted['uv']-center['uv'], dim=-1).max() <= 2+1e-10
        assert (delta.abs() <= .25*rho+1e-15).all()
    assert (torch.linalg.vector_norm(project_inverse_depth(rays, rho+interval[..., 1], transforms, focal, principal)['uv']
                                     - center['uv'], dim=-1).amax(-1) > 1.99).all()


def test_invalid_depths_and_no_visible_source_stay_inactive():
    rays, rho, transforms, focal, principal = fixture()
    rho[0] = torch.tensor([0, float('nan')])
    transforms[:, 2, 3] = -100
    result = project_inverse_depth(rays, rho, transforms, focal, principal)
    assert not result['valid'].any()
    assert torch.count_nonzero(result['uv']) == torch.count_nonzero(result['duv_drho']) == 0
    assert torch.isfinite(result['source_z']).all()
    interval = shared_depth_interval(rays, rho, transforms, focal, principal)['delta_interval']
    assert torch.count_nonzero(interval) == 0
