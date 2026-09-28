import ast
import hashlib
from pathlib import Path

import pytest
import torch

from bridge_rgs.direct_q_render import render_direct_q


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(5, dtype=torch.float32)*.01)
        self.calls = []

    def forward(self, features, rgb, depth, alpha, *, p3d, **kwargs):
        self.calls.append((features, rgb, depth, alpha, p3d, kwargs))
        return p3d*self.weight+features[..., :1]*self.weight


class Scene(torch.nn.Module):
    def __init__(self, moments=True):
        super().__init__()
        self.splats = torch.nn.ParameterDict({'means': torch.nn.Parameter(torch.ones(2, 3))})
        self.refiner = Head()
        self.mip_filter_config = None
        self.moments = moments
        self.calls = []

    def render(self, K, w2c, width, height, **kwargs):
        self.calls.append((K, w2c, width, height, kwargs, torch.is_grad_enabled()))
        value = self.splats['means'].sum()
        features = value.expand(height, width, 3)
        p = torch.full((height, width, 5), .2)
        result = {'features': features, 'rgb': features*.01, 'depth': features[..., :1],
                  'alpha': torch.ones(height, width, 1)*.75, 'p3d': p,
                  'probabilities': p, 'info': {
                      'means2d': (self.splats['means'][:, :2]+w2c[0, 3])[None],
                      'conics': torch.ones(1, 2, 3), 'opacities': torch.full((1, 2), .5),
                      'isect_offsets': torch.zeros(1, 1, 1, dtype=torch.int32),
                      'flatten_ids': torch.tensor([0, 1], dtype=torch.int32)}}
        if self.moments:
            result['depth_moments'] = features*2
        self.context = result
        return result


class Shader:
    def __init__(self):
        self.args = None

    def __call__(self, *args, width, height):
        self.args = args
        qi, qo = args[5:7]
        # A fixed visibility mixture plus residual class-0 background.
        probability = .125*qi[0]+.125*qo[0]+.25*qi[1]+.25*qo[1]
        background = qi.new_tensor([.25, 0, 0, 0, 0])
        return (probability+background).expand(height, width, 5), torch.full((height, width, 1), .75)


def inputs():
    return torch.tensor([[.2, .3, 0, .4, .1], [.1, .2, .3, .1, .3]]), torch.eye(3), torch.eye(4)


@pytest.mark.parametrize('moments', [False, True])
def test_original_context_preserved_only_prior_replaced_and_head_gradient_survives(moments):
    scene, shader = Scene(moments), Shader()
    q, K, pose = inputs()
    q_saved = q.clone()
    state = {key: value.clone() for key, value in scene.state_dict().items()}
    flags = [p.requires_grad for p in scene.parameters()]
    result = render_direct_q(scene, q, K, pose, 4, 3, rasterize=shader)
    assert len(scene.calls) == len(scene.refiner.calls) == 1
    call = scene.calls[0]
    assert call[0] is K and call[1] is pose and call[2:4] == (4, 3)
    assert call[4] == {'refine': False, 'absgrad': False} and call[5] is False
    for name in ('features', 'rgb', 'depth', 'alpha', 'info', 'original_p3d'):
        assert result[name] is scene.context['p3d' if name == 'original_p3d' else name]
    head = scene.refiner.calls[0]
    for index, name in enumerate(('features', 'rgb', 'depth', 'alpha')):
        assert head[index] is scene.context[name]
    assert head[4] is result['p3d'] and result['refinement_prior'] is result['p3d']
    assert ('depth_moments' in head[5]) == moments
    if moments:
        assert head[5]['depth_moments'] is scene.context['depth_moments']
    assert shader.args[5] is shader.args[6] is q
    torch.testing.assert_close(shader.args[7], torch.tensor([[0., 0., 1., -1.]]).expand(2, 4), rtol=0, atol=0)
    assert all(not value.requires_grad for value in shader.args[:5])
    loss = -result['probabilities'][..., 2].log().mean()
    loss.backward()
    assert scene.refiner.weight.grad is not None and scene.refiner.weight.grad.norm() > 0
    assert scene.splats['means'].grad is None and q.grad is None
    assert flags == [p.requires_grad for p in scene.parameters()] and scene.training
    for key, value in scene.state_dict().items():
        assert torch.equal(value, state[key])
    assert torch.equal(q, q_saved)


def test_refine_false_skips_head_and_probability_floor_is_only_pixel_stage():
    scene, shader = Scene(), Shader()
    q = torch.tensor([[1., 0, 0, 0, 0], [1., 0, 0, 0, 0]])
    result = render_direct_q(scene, q, torch.eye(3), torch.eye(4), 2, 1, refine=False, rasterize=shader)
    assert not scene.refiner.calls and shader.args[5] is q and (q[:, 1:] == 0).all()
    assert (result['raw'][..., 1:] == 0).all()
    assert (result['p3d'] > 0).all()
    torch.testing.assert_close(result['probabilities'], result['p3d'].log().softmax(-1), rtol=0, atol=0)


def test_new_pose_metadata_is_from_the_same_call_without_id_lookup():
    scene, shader = Scene(), Shader()
    q, K, pose = inputs()
    first = render_direct_q(scene, q, K, pose, 2, 2, rasterize=shader)
    old_means = first['info']['means2d'].clone()
    pose[0, 3] = .1
    second = render_direct_q(scene, q, K, pose, 3, 2, rasterize=shader)
    assert second['raw'].shape == (2, 3, 5)
    assert not torch.equal(old_means, second['info']['means2d'])
    assert torch.equal(shader.args[0], second['info']['means2d'][0])


def test_shader_exception_does_not_change_modes_flags_or_render_method():
    scene = Scene().eval()
    q, K, pose = inputs()
    method = scene.render
    flags = [p.requires_grad for p in scene.parameters()]
    def broken(*args, **kwargs):
        raise RuntimeError('shader failure')
    with pytest.raises(RuntimeError, match='shader failure'):
        render_direct_q(scene, q, K, pose, 2, 2, rasterize=broken)
    assert scene.render == method and not scene.training
    assert flags == [p.requires_grad for p in scene.parameters()]


@pytest.mark.parametrize('kind', ['negative', 'nan', 'float64', 'row_sum', 'shape', 'noncontiguous'])
def test_bad_q_rejected_before_scene_render(kind):
    q, K, pose = inputs()
    if kind == 'negative':
        q[0, 0] = -1e-8
    elif kind == 'nan':
        q[0, 0] = torch.nan
    elif kind == 'float64':
        q = q.double()
    elif kind == 'row_sum':
        q[0, 0] += 2e-6
    elif kind == 'shape':
        q = q[:1]
    else:
        q = q.T.contiguous().T
    scene = Scene()
    with pytest.raises(ValueError):
        render_direct_q(scene, q, K, pose, 2, 2, rasterize=Shader())
    assert not scene.calls


def test_cast_row_error_allowed_without_renormalization():
    scene, shader = Scene(), Shader()
    q, K, pose = inputs()
    q[0, 0] += 2e-7
    saved = q.clone()
    render_direct_q(scene, q, K, pose, 2, 2, rasterize=shader)
    assert shader.args[5] is q and torch.equal(q, saved)


def test_direct_shader_call_matches_frozen_passed_preflight_on_cpu():
    # Extract just the bound pure direct_q function, avoiding preflight imports,
    # data reads and GPU initialization. Compare every actual shader argument.
    path = Path('/mnt/data/SHM2026/runs/simplex_scene_preflight_v2/source_snapshot/preflight_simplex_scene.py')
    if not path.exists():
        pytest.skip('Local frozen preflight source unavailable')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '28461cfbe82d86b2e11d3f9b76e25d15974f78d48bc075dae23a18d8041430c3'
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'direct_q')
    namespace = {'SPEC': {'coefficients': [0., 0., 1., -1.]}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)  # noqa: S102
    scene, first_shader, second_shader = Scene(), Shader(), Shader()
    q, K, pose = inputs()
    result = render_direct_q(scene, q, K, pose, 2, 2, refine=False, rasterize=first_shader)
    raw, alpha = namespace['direct_q'](result['info'], q, second_shader, width=2, height=2)
    for first, second in zip(first_shader.args, second_shader.args, strict=True):
        assert torch.equal(first, second)
    assert first_shader.args[5] is first_shader.args[6] is second_shader.args[5] is second_shader.args[6] is q
    assert torch.equal(result['raw'], raw) and torch.equal(result['direct_q_alpha'], alpha)
