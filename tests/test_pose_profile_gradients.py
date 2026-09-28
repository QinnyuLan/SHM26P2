"""CPU-only algebra, gradient and fixed-measurement contracts."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().with_name('audit_pose_profile_gradients.py')
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parents[1]/'scripts/audit_pose_profile_gradients.py'
spec = importlib.util.spec_from_file_location('pose_profile_audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def example(rows=19):
    rng = torch.Generator().manual_seed(7)
    j = torch.randn(rows, 6, dtype=torch.float64, generator=rng)*100
    w = torch.linspace(.1, 1., rows, dtype=torch.float64)
    w[3] = 0
    r = torch.randn(rows, dtype=torch.float64, generator=rng)
    permutation = audit.fixed_permutation(rows-1, 0)
    return j, w, r, audit.whitened_system(j, w, 16.261470794677734, permutation)


def test_regularizer_exact_production_normal_sum_not_mean():
    j, w, _, system = example()
    normal = j.T@(w[:, None]*j)
    prior = torch.tensor([1/(.005*16.261470794677734)**2]*3+[1/.01**2]*3, dtype=torch.float64)
    expected = torch.diag(prior+.001*normal.diagonal().clamp_min(1))
    assert torch.allclose(system['normal'], normal, rtol=1e-13, atol=1e-10)
    assert torch.allclose(system['precision'], expected, rtol=1e-13, atol=1e-10)
    assert system['sum_w'] == w.sum()


def test_rgb_pixel_weights_broadcast_to_three_scalar_rows_with_matching_raw_loss():
    generator = torch.Generator().manual_seed(2)
    j = torch.randn(4, 5, 3, 6, dtype=torch.float64, generator=generator)
    w = torch.rand(4, 5, 1, dtype=torch.float64, generator=generator)
    w[0] = 0
    r = torch.randn(4, 5, 3, dtype=torch.float64, generator=generator)
    system = audit.whitened_system(j, w, 16., audit.fixed_permutation(45, 0))
    assert system['sum_w'] == pytest.approx(float(3*w.sum()))
    assert len(system['index']) == 45
    assert torch.allclose(audit.raw_loss(r, w), audit.profile_losses(r, system)['raw'], rtol=1e-13)


def test_profile_equals_dense_psd_woodbury_and_joint_minimum():
    _, _, residual, system = example()
    rw = residual[system['index']]*system['sqrt_w']
    j, precision = system['j'], system['precision']
    covariance = torch.eye(len(rw), dtype=torch.float64)+j@torch.linalg.solve(precision, j.T)
    inverse = torch.linalg.inv(covariance)
    eigenvalues = torch.linalg.eigvalsh(inverse)
    assert bool((eigenvalues > 0).all() & (eigenvalues <= 1+1e-12).all())
    dense = rw@inverse@rw/(2*system['sum_w'])
    delta = -torch.linalg.solve(j.T@j+precision, j.T@rw)
    joint = ((rw+j@delta).square().sum()+delta@precision@delta)/(2*system['sum_w'])
    losses = audit.profile_losses(residual, system)
    assert torch.allclose(losses['profile'], dense, rtol=1e-11, atol=1e-12)
    assert torch.allclose(losses['profile'], joint, rtol=1e-11, atol=1e-12)
    assert 0 <= losses['profile'] <= losses['raw']


def test_envelope_gradient_matches_fixed_operator_and_central_difference():
    _, _, residual, system = example()
    residual.requires_grad_(True)
    loss = audit.profile_losses(residual, system)['profile']
    gradient, = torch.autograd.grad(loss, residual)
    j = system['j']
    rw = residual.detach()[system['index']]*system['sqrt_w']
    delta = -torch.linalg.solve(system['normal']+system['precision'], j.T@rw)
    expected = torch.zeros_like(residual)
    expected[system['index']] = system['sqrt_w']*(rw+j@delta)/system['sum_w']
    assert torch.allclose(gradient, expected, rtol=1e-11, atol=1e-12)
    for i in (0, 3, 7, 18):
        shift = torch.zeros_like(residual)
        shift[i] = 1e-5
        numeric = (audit.profile_losses(residual.detach()+shift, system)['profile']-
                   audit.profile_losses(residual.detach()-shift, system)['profile'])/(2e-5)
        assert numeric == pytest.approx(float(gradient[i]), rel=1e-7, abs=1e-10)
    assert gradient[3] == 0


def test_no_jacobian_weights_or_regularizer_gradient():
    j, w, r, _ = example()
    j.requires_grad_(True)
    w.requires_grad_(True)
    r.requires_grad_(True)
    system = audit.whitened_system(j, w, 16., audit.fixed_permutation(18, 0))
    for value in system.values():
        assert not value.requires_grad
    gradients = torch.autograd.grad(audit.profile_losses(r, system)['profile'], (r, j, w), allow_unused=True)
    assert gradients[0] is not None and gradients[1:] == (None, None)


def test_whitened_permutation_preserves_normal_and_shrink_spectrum_without_refitting_precision():
    j, w, _, system = example()
    true, perm = system['j'], system['permuted_j']
    assert not torch.equal(true, perm)
    assert torch.allclose(true.T@true, perm.T@perm, atol=1e-9, rtol=1e-12)
    p = system['precision'].diagonal().sqrt()
    actual = torch.linalg.eigvalsh((true.T@true)/p[:, None]/p[None, :])
    control = torch.linalg.eigvalsh((perm.T@perm)/p[:, None]/p[None, :])
    assert torch.allclose(actual, control, rtol=1e-12, atol=1e-10)
    wrong = j[system['index']][audit.fixed_permutation(18, 0)]
    assert not torch.allclose(wrong.T@(w[system['index'], None]*wrong), system['normal'])


def test_permutation_deterministic_and_view_specific():
    assert np.array_equal(audit.fixed_permutation(100, 0), audit.fixed_permutation(100, 0))
    assert not np.array_equal(audit.fixed_permutation(100, 0), audit.fixed_permutation(100, 1))


@pytest.mark.parametrize('arm', audit.ARMS)
def test_shared_means_like_chain_gradient_and_no_other_parameter_path(arm):
    _, _, r, system = example()
    means = torch.tensor([.1, -.2, .3], dtype=torch.float64, requires_grad=True)
    a = torch.arange(len(r)*3, dtype=torch.float64).reshape(-1, 3)/100
    residual = r+a@means
    losses = audit.profile_losses(residual, system)
    gradient, = torch.autograd.grad(losses[arm], means)
    direction = means.new_tensor([.3, .2, -.1])
    def value(x):
        return audit.profile_losses(r+a@x, system)[arm]
    fd = (value(means.detach()+1e-5*direction)-value(means.detach()-1e-5*direction))/(2e-5)
    assert fd == pytest.approx(float(gradient@direction), rel=1e-7, abs=1e-10)


def test_zero_j_is_exact_raw_and_zero_weights_rejected():
    j = torch.zeros(10, 6, dtype=torch.float64)
    weights = torch.ones(10, dtype=torch.float64)
    system = audit.whitened_system(j, weights, 16., np.arange(10))
    losses = audit.profile_losses(torch.linspace(-1., 1., 10).double(), system)
    assert torch.equal(losses['raw'], losses['profile']) and torch.equal(losses['raw'], losses['permuted'])
    with pytest.raises(ValueError, match='weights'):
        audit.whitened_system(j, weights*0, 16., np.arange(10))


@pytest.mark.parametrize('kind', ['negative_weight', 'nan_weight', 'nan_j', 'bad_permutation'])
def test_invalid_inputs_are_not_silently_filtered(kind):
    j, w, _, _ = example()
    permutation = audit.fixed_permutation(18, 0)
    if kind == 'negative_weight':
        w[0] = -1
    elif kind == 'nan_weight':
        w[0] = float('nan')
    elif kind == 'nan_j':
        j[0, 0] = float('nan')
    else:
        permutation[0] = permutation[1]
    with pytest.raises(ValueError):
        audit.whitened_system(j, w, 16., permutation)


def test_no_dense_pixel_matrix_or_large_linear_solve(monkeypatch):
    rows = 10007
    rng = torch.Generator().manual_seed(3)
    j = torch.randn(rows, 6, dtype=torch.float64, generator=rng)
    chol = torch.linalg.cholesky
    observed = []
    def guarded(value, *args, **kwargs):
        observed.append(value.shape)
        assert value.shape == (6, 6)
        return chol(value, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('No dense inverse/identity/solve needed by helper')
    monkeypatch.setattr(torch.linalg, 'cholesky', guarded)
    monkeypatch.setattr(torch.linalg, 'inv', forbidden)
    monkeypatch.setattr(torch.linalg, 'solve', forbidden)
    monkeypatch.setattr(torch, 'eye', forbidden)
    system = audit.whitened_system(j, torch.ones(rows), 16., audit.fixed_permutation(rows, 0))
    losses = audit.profile_losses(torch.randn(rows, generator=rng), system)
    assert observed == [(6, 6)] and all(value.numel() <= rows*6 for value in system.values())
    assert all(torch.isfinite(v) for v in losses.values())


@pytest.mark.parametrize('fail', [False, True])
def test_all_tensor_and_grad_flags_restored_including_exception(fail):
    scene = torch.nn.Linear(2, 3)
    scene.register_buffer('counter', torch.tensor(7))
    scene.weight.grad = torch.ones_like(scene.weight)
    before = {k: v.clone() for k, v in scene.state_dict().items()}
    def mutate():
        with audit.preserved_scene(scene):
            scene.weight.requires_grad_(False)
            with torch.no_grad():
                scene.weight.add_(10)
                scene.counter.add_(5)
            scene.weight.grad = None
            if fail:
                raise RuntimeError('test')
    if fail:
        with pytest.raises(RuntimeError):
            mutate()
    else:
        mutate()
    assert all(torch.equal(v, before[k]) for k, v in scene.state_dict().items())
    assert scene.weight.requires_grad and torch.equal(scene.weight.grad, torch.ones_like(scene.weight))


def synthetic_repeat(raw=1., profile=.4, permuted=1., residual=.01):
    values = {'raw': raw, 'profile': profile, 'permuted': permuted}
    row = {'gradient': {k: torch.tensor([[v, 0., 0.]]) for k, v in values.items()},
           'residual': torch.full((2, 2, 3), residual), 'prediction_rms': .5,
           'loss': {k: residual*residual for k in values}}
    return [row, row]


def classify(raw=1., profile=.4, permuted=1., residual=.01):
    repeated = synthetic_repeat(raw, profile, permuted, residual)
    null = synthetic_repeat(0, 0, 0, 0)
    return audit.condition_summary(repeated, null, torch.full((2, 2, 3), .5), torch.ones(2, 2, 1), torch.tensor([[-1., 0., 0.]]))


def test_pose_gate_uses_means_norm_specificity_not_smaller_energy():
    good = classify()
    assert good['pose_specific_suppression_gate']
    assert not classify(profile=.4, permuted=.5)['pose_specific_suppression_gate']
    assert not classify(profile=.6)['pose_specific_suppression_gate']


def test_geometry_requires_positive_repair_and_half_raw_retention():
    assert not classify(profile=.4)['geometry_repair_retention_gate']
    assert classify(profile=.6)['geometry_repair_retention_gate']
    assert not classify(profile=-.6)['geometry_repair_retention_gate']
    assert not classify(raw=-1., profile=-.6)['geometry_direction_measurable']


def test_fp32_residual_floor_marks_tiny_signals_inconclusive():
    result = classify(residual=1e-8)
    assert result['residual_noise_floor'] == pytest.approx(32*torch.finfo(torch.float32).eps)
    assert not result['raw_signal_measurable']


def test_repeat_or_zero_signal_noise_is_not_ignored():
    rows = synthetic_repeat()
    rows[1] = {**rows[1], 'gradient': {**rows[1]['gradient'], 'raw': torch.tensor([[.9, 0., 0.]])}}
    result = audit.condition_summary(rows, synthetic_repeat(0, 0, 0, 0), torch.full((2, 2, 3), .5),
                                     torch.ones(2, 2, 1), torch.tensor([[-1., 0., 0.]]))
    assert result['gradient_noise_floor'] >= .099999
    assert not result['raw_signal_measurable']


def test_fixed_five_view_geometry_gate_and_others_reported_only():
    records = []
    for name in audit.NAMES:
        geometry = classify(profile=.6) if name in audit.GEOMETRY_NAMES else classify(residual=0.)
        records.append({'name': name, 'pose': classify(), 'geometry': geometry})
    assert audit.summarize(records)['status'] == 'necessary_gradient_gates_passed'
    records[0]['geometry']['geometry_direction_measurable'] = False
    assert audit.summarize(records)['status'] == 'inconclusive_measurability'
    with pytest.raises(ValueError, match='eight'):
        audit.summarize(records[:-1])


def test_local_pca_and_exact_fp32_repair_without_geometry_mutation():
    t = torch.linspace(-.05, .05, 22)
    means = torch.stack([.1*t, .03*torch.sin(t*10), 5+t], -1)
    before = means.clone()
    views = []
    for name in audit.NAMES:
        pose = np.eye(4)
        if name not in audit.GEOMETRY_NAMES:
            pose[2, 3] = -10
        views.append({'name': name, 'w2c': pose.tolist(), 'K': [[100., 0, 50.], [0, 100., 50.], [0, 0, 1.]], 'width': 100, 'height': 100})
    result = audit.local_geometry(means, {'anchor_xyz': [0., 0., 5.], 'radius': .1}, views, 16.)
    assert result['indices'] == list(range(22)) and torch.equal(means, before)
    axis = np.asarray(result['axis'])
    assert axis[np.argmax(np.abs(axis))] > 0
    assert result['amplitude'] == .016
    changed = means.clone()
    changed += torch.tensor(result['amplitude']*axis, dtype=means.dtype)
    assert torch.equal(changed.double()-means.double(), torch.tensor(result['actual_fp32_displacement'], dtype=torch.float64))


def test_fixed_budget_and_no_pixel_rgb_key():
    assert audit.SPEC['total_renders'] == 8*(1+2+2*(13+2)) == 264
    assert audit.SPEC['total_backwards'] == 8*(2+2*2*3) == 112
    assert 'image_path' not in audit.VIEW_KEYS and 'mask_path' not in audit.VIEW_KEYS
    assert audit.SPEC['optimizer_steps'] == 0
