"""CPU mathematical/control contracts; no dataset or CUDA execution."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name('audit_coupled_semantic_gradients.py')
if not WORKER.exists():
    WORKER = ROOT/'scripts/audit_coupled_semantic_gradients.py'
spec = importlib.util.spec_from_file_location('coupled_gradient_worker_test', WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class ToyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.2, dtype=torch.float64))

    def forward(self, feature, rgb, depth, alpha, p3d=None):
        entropy = -(p3d*p3d.log()).sum(-1, keepdim=True)
        return self.scale*feature[..., :5]+p3d.square()+entropy*feature[..., 5:6]


def toy_scene():
    scene = nn.Module()
    scene.refiner = ToyHead()
    scene.splats = nn.ParameterDict({'sem_features': nn.Parameter(torch.linspace(-.8, .9, 32, dtype=torch.float64).reshape(2, 16))})
    scene.other = nn.Parameter(torch.tensor(3., dtype=torch.float64))
    scene.register_buffer('buffer', torch.tensor([4.]))
    scene.requires_grad_(False)
    scene.splats['sem_features'].requires_grad_(True)
    return scene


def toy_render(scene):
    feature = scene.splats['sem_features']
    p = torch.cat([feature[:, :4], feature[:, :1]*0], -1).softmax(-1)
    zeros = torch.zeros_like(feature[:, :1])
    residual = scene.refiner(feature, zeros, zeros, zeros, p3d=p)
    return {'p3d': p, 'residual': residual, 'probabilities': (p.log()+residual).softmax(-1)}


def toy_semantic_loss(p, labels, valid, weights, lovasz_weight=.2):
    ce = -(p.log()[torch.arange(len(labels)), labels]*weights[labels]*valid).sum()/valid.sum()
    return ce+lovasz_weight*p.square().mean()


def test_capture_preserves_graph_forward_and_removes_hook():
    scene = toy_scene()
    result, args = worker.captured_render(scene, lambda: toy_render(scene))
    assert args[0] is scene.splats['sem_features'] and args[0].requires_grad
    assert not scene.refiner._forward_pre_hooks
    auxiliary = worker.feature_readout(scene, result, args)
    assert torch.equal(auxiliary[0], result['probabilities'])
    assert torch.equal(auxiliary[1], result['residual'])
    labels = torch.tensor([1, 3])
    losses = worker.objectives(result, auxiliary, labels, torch.ones(2), torch.arange(1., 6.), toy_semantic_loss)
    f = scene.splats['sem_features']
    gradients = [torch.autograd.grad(losses[key], f, retain_graph=i < 2)[0] for i, key in enumerate(worker.KEYS)]
    p = torch.diag(torch.tensor([1.]*4+[0.]*12, dtype=torch.float64))
    stats = worker.gradient_stats(dict(zip(worker.KEYS, gradients)), p)
    assert stats['algebra_passed']
    assert stats['energies']['feature_final']['null_energy'] > 0
    assert not torch.equal(gradients[1], gradients[2])
    assert all(parameter.grad is None for parameter in scene.parameters())
    json.dumps(stats, allow_nan=False)


def test_hook_cleanup_on_render_exception():
    scene = toy_scene()
    with pytest.raises(RuntimeError):
        worker.captured_render(scene, lambda: (_ for _ in ()).throw(RuntimeError('failure')))
    assert not scene.refiner._forward_pre_hooks


def test_anchored_partial_fd_is_not_endpoint_detach():
    scene = toy_scene()
    base, args = worker.captured_render(scene, lambda: toy_render(scene))
    anchor = base['p3d'].detach().clone()
    with torch.no_grad():
        scene.splats['sem_features'][0, 0].add_(.1)
        endpoint, args = worker.captured_render(scene, lambda: toy_render(scene))
        anchored = worker.feature_readout(scene, endpoint, args, anchor)
        unanchored = worker.feature_readout(scene, endpoint, args)
    assert not torch.equal(anchored[0], unanchored[0])
    expected = (anchor.log()+scene.refiner(*args, p3d=anchor)).softmax(-1)
    assert torch.equal(anchored[0], expected)


def test_original_loss_weight_and_residual_regularizer_are_retained():
    p = torch.tensor([[.1, .2, .3, .25, .15]], requires_grad=True)
    result = {'p3d': p, 'probabilities': p, 'residual': torch.ones_like(p)*3}
    called = []

    def loss(probabilities, labels, valid, weights, lovasz_weight=.2):
        called.append(lovasz_weight)
        return probabilities.sum()*7
    losses = worker.objectives(result, (p, result['residual']), None, None, None, loss)
    assert called == [0, .2, .2]
    assert float(losses['raw'].detach()) == 3.5
    assert float(losses['full_final'].detach()) == pytest.approx(7.009, abs=1e-6)
    assert torch.equal(losses['full_final'], losses['feature_final'])


def test_statistics_distinguish_mean_dot_from_dot_of_means_and_mass_closure():
    p = torch.eye(2, dtype=torch.float64)
    r1, r2 = torch.tensor([[1., 0.]]), torch.tensor([[-1., 0.]])
    f1, f2 = torch.tensor([[2., 0.]]), torch.tensor([[-2., 0.]])
    first = worker.gradient_stats({'raw': r1, 'full_final': f1, 'feature_final': f1}, p)
    second = worker.gradient_stats({'raw': r2, 'full_final': f2, 'feature_final': f2}, p)
    means = worker.gradient_stats({'raw': (r1+r2)/2, 'full_final': (f1+f2)/2, 'feature_final': (f1+f2)/2}, p)
    assert (first['raw_dot_full_final']+second['raw_dot_full_final'])/2 == 2
    assert means['raw_dot_full_final'] == 0
    assert means['cancel_ratio'] is None
    mass = first['per_gaussian_inner_product_mass']['full_final']
    assert mass['positive_sum']-mass['negative_absolute_sum'] == mass['signed_sum']


def test_primary_fd_passes_analytic_linear_and_rejects_wrong_derivative():
    g, d = torch.tensor([2.]), torch.tensor([-.5])
    kw = {'gradient': g, 'repeat_gradient': g, 'actual_direction': d, 'base_loss': .1,
          'repeat_loss': .1, 'plus_loss': .099, 'minus_loss': .101, 'h': .001, 'primary': True}
    report = worker.calibrate_fd(**kw)
    assert report['passed'] and report['analytic_realized'] == -1
    kw['plus_loss'] = .0995
    assert worker.calibrate_fd(**kw)['status'] == 'failed'


def test_ulp_repeat_floor_and_zero_direction_are_not_false_passes():
    report = worker.calibrate_fd(torch.ones(1), torch.ones(1), torch.tensor([1e-8]),
                                 1., 1., 1., 1., .001, primary=True)
    assert report['status'] == 'inconclusive' and not report['passed']
    crossed = worker.calibrate_fd(torch.zeros(1), torch.zeros(1), torch.ones(1),
                                  1., 1., 1., 1., .001, primary=False)
    assert crossed['status'] == 'cross_inconclusive'
    direction, maximum = worker.fixed_direction(torch.zeros(3, 16), torch.eye(16, dtype=torch.float64))
    assert maximum == 0 and torch.equal(direction, torch.zeros_like(direction))


def test_raw_null_control_uses_absolute_zero_not_nonzero_gate():
    g = torch.tensor([[1., 0.]])
    d = torch.tensor([[0., 1.]])
    p = torch.diag(torch.tensor([1., 0.], dtype=torch.float64))
    fd = worker.calibrate_fd(g, g, d, .1, .1, .1, .1, .001, primary=False)
    zero = worker.null_zero_control(fd, g, d, p)
    assert zero['zero_consistent'] and zero['analytic_zero_consistent']
    assert not zero['measurable'] and zero['status'] == 'consistent'
    fd['central_difference'] = 1.
    assert worker.null_zero_control(fd, g, d, p)['status'] == 'not_resolved'


def test_exception_restores_every_tensor_flag_mode_gradient_and_camera():
    scene = toy_scene()
    scene.train()
    scene.refiner.eval()
    scene.other.grad = torch.tensor(9., dtype=torch.float64)
    before = {k: v.clone() for k, v in scene.state_dict().items()}
    cameras = torch.eye(4)[None]
    report = {}
    with pytest.raises(RuntimeError), worker.preserved_scene(scene, cameras, report):
        assert [k for k, p in scene.named_parameters() if p.requires_grad] == ['splats.sem_features']
        with torch.no_grad():
            scene.splats['sem_features'].add_(1)
            scene.buffer.add_(2)
            cameras.add_(1)
        scene.other.grad = None
        raise RuntimeError('failure')
    assert all(torch.equal(value, before[key]) for key, value in scene.state_dict().items())
    assert scene.training and not scene.refiner.training and scene.other.grad == 9
    assert torch.equal(cameras, torch.eye(4)[None])
    assert all(report[k] for k in ('all_tensors_restored_exact', 'flags_restored_exact',
               'modes_restored_exact', 'gradients_restored_exact', 'cameras_restored_exact'))
    assert not report['non_feature_unchanged_before_restore']
    json.dumps(report, allow_nan=False)


def test_numerical_flags_restore_on_exception_without_cuda():
    before = worker.numerical_flags()
    report = {}
    with pytest.raises(RuntimeError), worker.numerical_contract(report):
        assert report['actual'] == worker.SPEC['numerics']
        raise RuntimeError('failure')
    assert worker.numerical_flags() == before and report['restored_exact']
    assert not torch.cuda.is_initialized()


def test_fixed_budget_and_no_gpu_query_unknown_success(monkeypatch):
    assert worker.SPEC['scene_renders'] == 259+2+2*2*3*2 == 285
    assert worker.SPEC['field_vjps'] == (259+2)*3 == 783
    assert worker.SPEC['raster_calls'] == 570
    monkeypatch.setattr(worker.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='N/A\n', args=a))
    with pytest.raises(ValueError, match='unrecognized'):
        worker.gpu_idle()


def test_forward_exact_guard_does_not_silently_accept_changed_head():
    scene = toy_scene()
    result, args = worker.captured_render(scene, lambda: toy_render(scene))
    with torch.no_grad():
        scene.refiner.scale.add_(.1)
    with pytest.raises(ValueError, match='forward exact'):
        worker.feature_readout(scene, result, args)


def test_actual_frozen_refiner_and_loss_three_path_cpu_contract():
    package = WORKER.parent/'bridge_rgs'
    if not package.exists():
        package = ROOT/'runs/support_split_semantic_coupled/source_snapshot/bridge_rgs'
    modules = {}
    for name in ('refinement', 'losses'):
        module_spec = importlib.util.spec_from_file_location('frozen_coupled_test_'+name, package/(name+'.py'))
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        modules[name] = module
    torch.set_num_threads(8)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(617)
        head = modules['refinement'].MultiScaleRefinementHead(16, channels=64, residual_bound=6., context='pyramid_strip')
        with torch.no_grad():
            head.out.weight.normal_(std=.01)
        head.requires_grad_(False)
        scene = SimpleNamespace(refiner=head)
        feature = torch.randn(32, 32, 16, requires_grad=True)
        rgb, depth, alpha = torch.rand(32, 32, 3), torch.rand(32, 32, 1), torch.rand(32, 32, 1)
        p = feature[..., :5].softmax(-1)

        def render():
            residual = head(feature, rgb, depth, alpha, p3d=p)
            return {'p3d': p, 'probabilities': (p.log()+residual).softmax(-1), 'residual': residual}
        result, args = worker.captured_render(scene, render)
        auxiliary = worker.feature_readout(scene, result, args)
        labels = torch.arange(32*32).reshape(32, 32) % 5
        losses = worker.objectives(result, auxiliary, labels, torch.ones(32, 32),
                                  torch.tensor([.45604488, .92276901, .85304344, 1.19359136, 1.57455134]),
                                  modules['losses'].semantic_loss)
        assert torch.equal(losses['full_final'], losses['feature_final'])
        gradients = {key: torch.autograd.grad(losses[key], feature, retain_graph=i < 2)[0]
                     for i, key in enumerate(worker.KEYS)}
        decoder = torch.eye(16, dtype=torch.float64)[:5]
        difference = decoder[1:]-decoder[:1]
        projection = torch.linalg.pinv(difference)@difference
        report = worker.gradient_stats(gradients, projection)
        assert report['algebra_passed'] and report['energies']['feature_final']['null_energy'] > 0
        assert all(parameter.grad is None for parameter in head.parameters())
