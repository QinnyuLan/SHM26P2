"""CPU contracts only: no scene files, image payloads, CUDA or installed renderer."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bridge_rgs.depth_moments import semantic_depth_context
from bridge_rgs.direct_q_geometry_render import render_direct_q_geometry
from bridge_rgs.direct_q_render import render_direct_q
from bridge_rgs.model import GaussianScene
from bridge_rgs.refinement import MultiScaleRefinementHead


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_scene():
    with torch.random.fork_rng():
        torch.manual_seed(731)
        scene = GaussianScene(np.array([[-.3, .2, 1.], [.4, -.1, 2.],
                                        [.1, .5, 3.], [-.4, -.3, 4.]], np.float32),
                              np.full((4, 3), .4, np.float32), feature_dim=4,
                              refiner_config={'type': 'multiscale', 'channels': 16,
                                              'context': 'none', 'depth_moments': 'cross'})
        with torch.no_grad():
            scene.splats['sem_features'].normal_(std=.3)
            scene.refiner.out.weight.normal_(std=.025)
            scene.refiner.moment_detail.weight.normal_(std=.02)
            scene.refiner.moment_half.weight.normal_(std=.02)
        scene.eval().requires_grad_(False)
        scene.splats['means'].requires_grad_(True)
        q = torch.rand(4, 5)
        q = (q / q.sum(-1, keepdim=True)).contiguous()
    return scene, q, torch.eye(3), torch.eye(4)


def composite(weights, values, background):
    value = sum(weights[..., i:i+1] * values[i] for i in range(values.shape[0]))
    return value + (1-weights.sum(-1, keepdim=True)) * background


class Raster:
    """Smooth known operator with shared weights, live depth and RGB evidence."""
    def __init__(self):
        self.calls = []

    def __call__(self, *, means, colors, backgrounds, width, height, **kwargs):
        self.calls.append({'means': means, 'colors': colors, **kwargs})
        x = torch.linspace(-.2, .2, width, device=means.device)[None, :, None]
        y = torch.linspace(-.1, .1, height, device=means.device)[:, None, None]
        weights = .11 + .03*torch.sigmoid(means[:, 0] + x + y + .07*means[:, 1])
        alpha = weights.sum(-1, keepdim=True)
        info = {'weights': weights, 'means2d': means[None, :, :2],
                'conics': torch.ones_like(means)[None],
                'opacities': torch.ones(len(means))[None],
                'isect_offsets': torch.zeros(1, 2, 2, dtype=torch.int32),
                'flatten_ids': torch.arange(len(means), dtype=torch.int32)}
        if kwargs.get('render_mode') == 'RGB+ED':
            rgb_values = .4 + .01*colors[:, 0] + .02*means
            rgb = composite(weights, rgb_values, backgrounds[0])
            depth = (weights*means[:, 2]).sum(-1, keepdim=True)/alpha
            result = torch.cat((rgb, depth), -1)
        else:
            result = composite(weights, colors, backgrounds[0])
        return result[None], alpha[None], info


def old_shader_for(raster):
    def shader(means2d, conics, opacity, offsets, ids, qin, qout, coeff, *, width, height):
        assert qin is qout
        # Reuse the exact weights of the original context's RGB raster.
        means = raster.calls[0]['means']
        x = torch.linspace(-.2, .2, width)[None, :, None]
        y = torch.linspace(-.1, .1, height)[:, None, None]
        weights = .11 + .03*torch.sigmoid(means[:, 0] + x + y + .07*means[:, 1])
        return composite(weights, qin, qin.new_tensor([1., 0, 0, 0, 0])), weights.sum(-1, keepdim=True)
    return shader


def test_forward_same_as_old_direct_readout_on_known_shared_operator(monkeypatch):
    scene, q, K, pose = make_scene()
    old_raster = Raster()
    monkeypatch.setitem(sys.modules, 'gsplat', SimpleNamespace(rasterization=old_raster))
    with torch.no_grad():
        old = render_direct_q(scene, q, K, pose, 19, 17, rasterize=old_shader_for(old_raster))
    raster = Raster()
    new = render_direct_q_geometry(scene, q, K, pose, 19, 17, rasterize=raster)
    restricted = render_direct_q_geometry(scene, q, K, pose, 19, 17,
                                         geometry_grad=False, rasterize=Raster())
    for key in ('rgb', 'depth', 'alpha', 'raw', 'p3d', 'features', 'depth_moments',
                'residual', 'probabilities'):
        assert torch.equal(old[key], new[key]), key
        assert torch.equal(new[key], restricted[key]), key
    assert len(raster.calls) == 2 and all(c['means'] is scene.splats['means'] for c in raster.calls)
    assert all(c['rasterize_mode'] == 'antialiased' and not c['absgrad'] for c in raster.calls)
    assert raster.calls[1]['colors'].shape[-1] == 5 + 4 + 2 + 4
    assert raster.calls[1]['colors'].requires_grad  # z/z²/zf, even though q/features are frozen


def test_frozen_head_means_loss_gradient_matches_independent_cpu_secant():
    scene, q, K, pose = make_scene()
    means = scene.splats['means']
    before = {k: v.clone() for k, v in scene.state_dict().items()}
    def objective():
        result = render_direct_q_geometry(scene, q, K, pose, 19, 17, rasterize=Raster())
        return -result['probabilities'][..., 2].double().log().mean()
    loss = objective()
    gradient, = torch.autograd.grad(loss, means)
    direction = torch.tensor([[.1, -.2, .3], [-.2, .1, -.1], [.3, .2, -.2], [-.1, .1, .2]])
    h = .01
    values, endpoints = [], []
    try:
        for sign in (1, -1):
            with torch.no_grad():
                means.copy_(before['splats.means'] + sign*h*direction)
                endpoints.append(means.clone())
                values.append(float(objective()))
    finally:
        with torch.no_grad():
            means.copy_(before['splats.means'])
    analytic = float((gradient.double() * (endpoints[0].double()-endpoints[1].double())/(2*h)).sum())
    observed = (values[0]-values[1])/(2*h)
    assert abs(analytic) > 1e-5
    assert observed == pytest.approx(analytic, rel=.005, abs=1e-6)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(p.grad is None for p in scene.parameters())  # autograd.grad, no hidden accumulation
    assert all(torch.equal(before[k], v) for k, v in scene.state_dict().items())


def test_auxiliary_head_gradients_explicit_and_state_dict_unchanged():
    scene, _, _, _ = make_scene()
    head = scene.refiner
    g = torch.Generator().manual_seed(81)
    values = [torch.rand(17, 19, c, generator=g).requires_grad_() for c in (4, 3, 1, 1, 5)]
    moments = torch.rand(17, 19, 5, generator=g, requires_grad=True)
    state = {k: v.clone() for k, v in head.state_dict().items()}
    old = head(*values, depth_moments=moments)
    full = head(*values, depth_moments=moments, geometry_grad=True)
    assert torch.equal(old, full)
    a = torch.autograd.grad(old.square().mean(), values, allow_unused=True)
    b = torch.autograd.grad(full.square().mean(), values, allow_unused=True)
    assert all(a[i] is None for i in (1, 2, 3))
    assert all(v is not None and torch.isfinite(v).all() and v.abs().sum() > 0 for v in b)
    assert all(torch.equal(v, state[k]) for k, v in head.state_dict().items())
    restored = MultiScaleRefinementHead(feature_dim=4, channels=16, context='none', depth_moments='cross')
    restored.load_state_dict(state, strict=True)


def test_full_moment_derivatives_pass_double_gradcheck_without_forward_change():
    values = [torch.tensor(v, dtype=torch.float64).reshape(1, 1, -1).requires_grad_() for v in
              ([.4, -.1], [1.7], [4.3], [.2, .6], [.7])]
    assert torch.equal(semantic_depth_context(*values), semantic_depth_context(*values, geometry_grad=True))
    assert torch.autograd.gradcheck(lambda *x: semantic_depth_context(*x, geometry_grad=True),
                                    tuple(values), eps=1e-6, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('alpha,depth', [(0., 0.), (.5, 2.), (.999, 1000.)])
def test_unsupported_moment_sqrt_branch_has_finite_zero_geometry_gradient(alpha, depth):
    values = [torch.tensor(v, dtype=torch.float64).reshape(1, 1, -1).requires_grad_() for v in
              ([alpha*.2, alpha*.3], [alpha*depth], [alpha*depth**2],
               [alpha*depth*.2, alpha*depth*.3], [alpha])]
    old = semantic_depth_context(*values)
    new = semantic_depth_context(*values, geometry_grad=True)
    assert torch.equal(old, new) and not torch.count_nonzero(new)
    gradients = torch.autograd.grad(new.sum(), values)
    assert all(torch.isfinite(v).all() and not torch.count_nonzero(v) for v in gradients)


@pytest.mark.parametrize('kind', ['head', 'q', 'camera', 'mip'])
def test_rejects_unfrozen_other_paths_before_any_render(kind):
    scene, q, K, pose = make_scene()
    if kind == 'head':
        scene.refiner.out.weight.requires_grad_(True)
    elif kind == 'q':
        q.requires_grad_(True)
    elif kind == 'camera':
        pose.requires_grad_(True)
    else:
        scene.mip_filter_config = {'enabled': True}
    raster = Raster()
    with pytest.raises(ValueError):
        render_direct_q_geometry(scene, q, K, pose, 19, 17, rasterize=raster)
    assert not raster.calls


def test_failure_does_not_mutate_parameters_flags_buffers_or_methods():
    scene, q, K, pose = make_scene()
    state = {k: v.clone() for k, v in scene.state_dict().items()}
    flags = [p.requires_grad for p in scene.parameters()]
    method = scene.render
    def broken(**kwargs):
        raise RuntimeError('synthetic raster failure')
    with pytest.raises(RuntimeError, match='synthetic raster failure'):
        render_direct_q_geometry(scene, q, K, pose, 19, 17, rasterize=broken)
    assert flags == [p.requires_grad for p in scene.parameters()]
    assert scene.render == method and not scene.training
    assert all(torch.equal(state[k], v) for k, v in scene.state_dict().items())
