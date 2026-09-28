"""Independent CPU scientific contracts, separate from renderer/state integration."""
import copy
import hashlib
import json
import math

import pytest
import torch

from bridge_rgs.mip_filter import (
    POLICY,
    camera_source,
    compute_rho,
    effective_parameters,
    normalize_config,
)


def _cameras():
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = 1000  # Highest-fx camera need not see these Gaussians.
    views = [
        {'name': f'{i:03}.png', 'split': 'train', 'width': 40, 'height': 30,
         'K': [[focal, 0, 7], [0, 35, 9], [0, 0, 1]],
         'w2c': pose.tolist(), 'w2c_original': pose.tolist()}
        for i, (focal, pose) in enumerate(zip([20., 80.], poses, strict=True))
    ]
    return views, poses


def test_closed_form_scale_and_opacity_gradients_and_detached_frequency():
    scales = torch.tensor([[.01, .4, 1.7], [.8, .02, .07]], dtype=torch.float64)
    log_s = scales.log().requires_grad_()
    logits = torch.tensor([-.4, .7], dtype=torch.float64, requires_grad=True)
    rho = torch.tensor([[.3], [.11]], dtype=torch.float64, requires_grad=True)
    scale_weights = torch.tensor([[.7, -.3, 1.1], [-.2, .4, .9]], dtype=torch.float64)
    opacity_weights = torch.tensor([1.3, -.4], dtype=torch.float64)
    filtered, alpha = effective_parameters(log_s, logits, rho)
    objective = (scale_weights*filtered).sum()+(opacity_weights*alpha).sum()
    gl, go, gr = torch.autograd.grad(objective, (log_s, logits, rho), allow_unused=True)

    # d log(c)/d log(s_j) = rho²/(s_j²+rho²), independently differentiated.
    f = (scales.square()+rho.detach().square()).sqrt()
    a = logits.detach().sigmoid()
    c = (scales/f).prod(-1)
    expected_l = scale_weights*scales.square()/f
    expected_l += (opacity_weights*a*c)[:, None]*rho.detach().square()/f.square()
    expected_o = opacity_weights*c*a*(1-a)
    torch.testing.assert_close(gl, expected_l, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(go, expected_o, rtol=1e-12, atol=1e-14)
    assert gr is None

    # Independent smooth scalar directional finite difference, not merely nonzero.
    dl = torch.tensor([[.3, -.7, .2], [.8, .1, -.4]], dtype=torch.float64)
    do = torch.tensor([-.3, .6], dtype=torch.float64)
    analytic = (gl*dl).sum()+(go*do).sum()
    endpoints = []
    eps = 1e-5
    for sign in (-1, 1):
        sf, af = effective_parameters(log_s.detach()+sign*eps*dl,
                                      logits.detach()+sign*eps*do, rho.detach())
        endpoints.append((scale_weights*sf).sum()+(opacity_weights*af).sum())
    torch.testing.assert_close((endpoints[1]-endpoints[0])/(2*eps), analytic,
                               rtol=1e-8, atol=1e-11)


def test_rotated_covariance_determinant_equals_axis_formula():
    scales = torch.tensor([[.03, .4, 1.7], [.2, .7, 1.1]], dtype=torch.float64)
    rotation, _ = torch.linalg.qr(torch.tensor([[1., 2, -1], [2, 1, .5], [-1, .7, 2]],
                                              dtype=torch.float64))
    covariance = rotation@torch.diag_embed(scales.square())@rotation.T
    rho = torch.tensor([[.3], [.12]], dtype=torch.float64)
    regularized = covariance+rho.square()[:, :, None]*torch.eye(3, dtype=torch.float64)
    opacity = torch.tensor([-.3, .7], dtype=torch.float64)
    filtered, alpha = effective_parameters(scales.log(), opacity, rho)
    determinant_coefficient = (torch.linalg.det(covariance)/torch.linalg.det(regularized)).sqrt()
    torch.testing.assert_close(alpha/opacity.sigmoid(), determinant_coefficient,
                               rtol=1e-12, atol=1e-14)
    reconstructed = rotation@torch.diag_embed(filtered.square())@rotation.T
    torch.testing.assert_close(reconstructed, regularized, rtol=1e-12, atol=1e-14)


def test_axis_ratio_product_survives_fp32_determinant_underflow():
    scales = torch.tensor([[1e-10, 1e-9, 2e-8]], dtype=torch.float32)
    assert scales.square().prod() == 0  # Naive determinant is not a valid oracle here.
    log_s = scales.log().requires_grad_()
    opacity = torch.zeros(1, requires_grad=True)
    rho = torch.tensor([[1e-6]], dtype=torch.float32)
    sf, alpha = effective_parameters(log_s, opacity, rho)
    _, reference = effective_parameters(log_s.detach().double(), opacity.detach().double(), rho.double())
    assert alpha.item() > 0 and bool(torch.isfinite(sf).all())
    torch.testing.assert_close(alpha.double(), reference, rtol=3e-6, atol=0)
    alpha.sum().backward()
    assert bool(torch.isfinite(log_s.grad).all()) and bool((log_s.grad > 0).all())
    assert bool(torch.isfinite(opacity.grad).all()) and opacity.grad.item() > 0


def test_global_focal_even_when_camera_unseen_and_noncenter_principal_point():
    views, poses = _cameras()
    source = camera_source(views, poses, 'a'*64, required_count=2)
    # First point: actual cx -> u=37 inside extended [-6,46]; width/2 -> 50 outside.
    # Third/fourth points unseen; fallback must use max of valid point min-z (=6).
    points = torch.tensor([[3., 0, 2], [0, 0, 6], [1000, 0, 1], [0, 0, .2]],
                          requires_grad=True)
    rho, stats = compute_rho(points, source, point_chunk=1)
    expected = torch.tensor([2., 6., 6., 6.]) / 80 * math.sqrt(.2)
    torch.testing.assert_close(rho[:, 0], expected, rtol=1e-6, atol=0)
    assert stats['unseen_gaussians'] == 2 and stats['global_native_max_fx'] == 80
    assert not rho.requires_grad and points.grad is None
    one_chunk, _ = compute_rho(points, source, point_chunk=100)
    assert torch.equal(rho, one_chunk)
    with pytest.raises(ValueError, match='No Gaussian'):
        compute_rho(points[2:].detach(), source)


def test_source_binds_order_native_grid_original_pose_and_manifest_without_pixel_reads():
    views, poses = _cameras()
    # The sampling interface must not open or infer metadata from image payloads.
    for view in views:
        view.update(image_path='/does/not/exist.png', mask_path='/does/not/exist-mask.png')
    source = camera_source(views, poses, 'a'*64, required_count=2)
    expected = [{'name': view['name'], 'K': k.tolist(), 'w2c': p.tolist(), 'size': s.tolist()}
                for view, k, p, s in zip(views, source['K'], poses, source['sizes'], strict=True)]
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert source['metadata']['camera_metadata_sha256'] == digest
    assert source['metadata']['manifest_sha256'] == 'a'*64
    assert source['metadata']['names'] == ['000.png', '001.png']
    reordered = camera_source(views[::-1], poses.flip(0), 'a'*64, required_count=2)
    assert reordered['metadata']['camera_metadata_sha256'] != digest
    modified = copy.deepcopy(views)
    modified[0]['K'][0][0] *= .5
    assert camera_source(modified, poses, 'a'*64, required_count=2)['metadata'] != source['metadata']
    assert camera_source(views, poses, 'b'*64, required_count=2)['metadata'] != source['metadata']
    for bad in [[], [views[0], views[0]], [dict(views[0], split='val'), views[1]]]:
        with pytest.raises(ValueError):
            camera_source(bad, poses, 'a'*64, required_count=2)
    with pytest.raises(ValueError, match='original'):
        camera_source(views, poses.flip(0), 'a'*64, required_count=2)
    modified = copy.deepcopy(views)
    modified[0]['w2c'][0][3] = 99  # Original pose is authoritative, not working pose.
    assert camera_source(modified, poses, 'a'*64, required_count=2)['metadata'] == source['metadata']


@pytest.mark.parametrize('bad', [
    True, {}, {'enabled': 1}, {'enabled': True, 'variance_factor': .3},
    {'enabled': True, 'refresh_every': 100.}, {'enabled': True, 'depth_min': .1},
    {'enabled': True, 'unknown': 1}, {'enabled': True, 'unseen': 'zero'},
])
def test_fixed_policy_rejects_silent_algorithm_and_type_changes(bad):
    with pytest.raises(ValueError):
        normalize_config(bad)


def test_fixed_policy_returns_independent_copy_and_exact_off():
    for value in (None, False, {'enabled': False}):
        assert normalize_config(value) is None
    for value in ({'enabled': True}, dict(POLICY)):
        result = normalize_config(value)
        assert result == POLICY and result is not POLICY and result is not value
        result['variance_factor'] = 99.
        assert POLICY['variance_factor'] == .2
