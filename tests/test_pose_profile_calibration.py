"""Small CPU contracts for fixed-interval calibration, not a renderer self-test."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().with_name('calibrate_pose_profile_direction.py')
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parents[1]/'scripts/calibrate_pose_profile_direction.py'
spec = importlib.util.spec_from_file_location('profile_calibration', SCRIPT)
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


def fixture(scale=1.):
    generator = torch.Generator().manual_seed(123)
    camera_j = torch.randn(20, 6, generator=generator, dtype=torch.float64)*10
    weights = torch.linspace(.2, 1., 20, dtype=torch.float64)
    weights[3] = 0
    system = cal.ref.whitened_system(camera_j, weights, 16., cal.ref.fixed_permutation(19, 1))
    geometry_j = torch.randn(20, 3, generator=generator, dtype=torch.float64)
    theta = torch.tensor([.3, -.7, .2], dtype=torch.float64)*scale
    theta.requires_grad_()
    residual = geometry_j@theta
    losses = cal.ref.profile_losses(residual, system)
    gradients = {arm: torch.autograd.grad(loss, theta, retain_graph=True)[0].detach()
                 for arm, loss in losses.items()}
    base = {'gradient': gradients, 'residual': residual.detach(),
            'loss': {arm: float(loss.detach()) for arm, loss in losses.items()}}
    return system, geometry_j, theta.detach(), base


def rows_for_fixture(scale=1., wrong_gradient=False):
    system, geometry_j, theta, base = fixture(scale)
    if wrong_gradient:
        base['gradient'] = {arm: gradient*2 for arm, gradient in base['gradient'].items()}
    rows = []
    for fraction in cal.SPEC['fractions']:
        actual = -theta*fraction
        residual = geometry_j@(theta+actual)
        endpoint = {'residual': residual, 'loss': {arm: float(loss)
                    for arm, loss in cal.ref.profile_losses(residual, system).items()}}
        arms = cal.summarize_step([base, copy.deepcopy(base)], [endpoint, copy.deepcopy(endpoint)],
                                  system, actual, [0., 0.])
        rows.append({'fraction': fraction, 'arms': arms})
    return rows


def test_fixed_budget_original_five_and_no_extra_amplitude():
    assert cal.SPEC['views'] == ['003.png', '058.png', '115.png', '170.png', '344.png']
    assert cal.SPEC['fractions'] == [.125, .0625, .03125]
    assert cal.SPEC['total_renders'] == 5*(1+2+13+2+3*2) == 120
    assert cal.SPEC['total_backwards'] == 5*(2+2*3) == 40
    assert cal.SPEC['worker_wall_seconds'] == 180
    assert not cal.SPEC['absolute_full_image_rms_gate']


def test_actual_float32_steps_stay_in_interval_and_are_not_assumed_ideal():
    original = torch.tensor([[13., -53., 117.], [.125, 36.123, -111.7]], dtype=torch.float32)
    changed = original+torch.tensor([.0041856766, -.0004713535, .0157060623])
    repair = original.double()-changed.double()
    rows = cal.interval_steps(original, changed)
    for row in rows:
        assert torch.equal(row['actual'], row['endpoint'].double()-changed.double())
        assert torch.all(row['endpoint'] >= torch.minimum(original, changed))
        assert torch.all(row['endpoint'] <= torch.maximum(original, changed))
        assert torch.all(row['actual']*repair >= 0)
    assert any(row['relative_quantization_error'] > 0 for row in rows)


def test_steps_reject_fully_rounded_away_signal():
    with pytest.raises(ValueError, match='rounded entirely'):
        cal.interval_steps(torch.zeros(2, 3), torch.zeros(2, 3))


@pytest.mark.parametrize('arm', cal.ref.ARMS)
def test_independent_dense_operator_verifies_image_derivative_and_quadratic_identity(arm):
    system, _, _, base = fixture()
    r0 = base['residual']
    r1 = r0+torch.linspace(-.002, .003, len(r0), dtype=torch.float64)
    b, _, q = cal.image_direction_terms(r0, r1, system, arm)
    rw, dr = cal.whiten(r0, system), cal.whiten(r1-r0, system)
    j = system['j' if arm == 'profile' else 'permuted_j']
    p = torch.eye(len(rw), dtype=torch.float64)
    if arm != 'raw':
        p -= j@torch.linalg.solve(system['normal']+system['precision'], j.T)
    assert b == pytest.approx(float((p@rw)@dr/system['sum_w']), rel=1e-12, abs=1e-15)
    assert q == pytest.approx(float(dr@(p@dr)/(2*system['sum_w'])), rel=1e-12, abs=1e-15)
    delta_loss = cal.ref.profile_losses(r1, system)[arm]-cal.ref.profile_losses(r0, system)[arm]
    assert float(delta_loss) == pytest.approx(b+q, rel=1e-10, abs=1e-15)
    assert not system['j'].requires_grad


def test_correct_chain_and_measured_decrease_pass_at_predeclared_fine_steps():
    rows = rows_for_fixture()
    summary = cal.summarize_view(rows)
    assert summary['direction_reliable'] and summary['profile_repair_retained']
    for row in rows:
        for arm in cal.ref.ARMS:
            assert row['arms'][arm]['quadratic_identity_ok']
            assert row['arms'][arm]['parameter_vs_image_relative_error'] < 1e-12


def test_repeatable_but_wrong_backward_fails_even_with_zero_repeat_noise():
    rows = rows_for_fixture(wrong_gradient=True)
    summary = cal.summarize_view(rows)
    assert summary['changes_measurable']
    assert not summary['direction_reliable']
    assert not summary['profile_repair_retained']


def test_low_global_scale_does_not_fail_an_absolute_intensity_threshold():
    rows = rows_for_fixture(scale=1e-7)
    assert cal.summarize_view(rows)['direction_reliable']


def test_measured_repeat_noise_prevents_claiming_signal():
    system, matrix, theta, base = fixture()
    step = -theta/32
    residual = matrix@(theta+step)
    end = {'residual': residual, 'loss': {a: float(l) for a, l in cal.ref.profile_losses(residual, system).items()}}
    noisy = copy.deepcopy(end)
    noisy['loss'] = {a: l+1 for a, l in noisy['loss'].items()}
    rows = cal.summarize_step([base, copy.deepcopy(base)], [end, noisy], system, step, [0., 0.])
    assert all(not rows[a]['measurable_change'] for a in cal.ref.ARMS)


def test_five_views_required_old_result_and_training_permission_unchanged():
    summary = cal.summarize_view(rows_for_fixture())
    records = [{'name': name, 'summary': copy.deepcopy(summary)} for name in cal.SPEC['views']]
    result = cal.summarize(records)
    assert result['status'] == 'independent_direction_calibration_supported'
    assert result['old_probe_result'] == 'inconclusive_measurability'
    assert result['training_authorized'] is False
    records[1]['summary']['changes_measurable'] = False
    assert cal.summarize(records)['status'] == 'inconclusive_numerical_signal'
    with pytest.raises(ValueError, match='original five'):
        cal.summarize(records[:-1])


def test_largest_step_cannot_replace_a_failed_predeclared_fine_step():
    rows = rows_for_fixture()
    rows[-1]['arms']['raw']['direction_reliable'] = False
    assert not cal.summarize_view(rows)['direction_reliable']
    with pytest.raises(ValueError, match='All fixed steps'):
        cal.summarize_view(rows[:-1])


def test_reference_restoration_on_exception_covers_parameters_buffers_and_flags():
    model = torch.nn.Linear(3, 2)
    model.register_buffer('audit_buffer', torch.arange(3.))
    before = copy.deepcopy(model.state_dict())
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    with pytest.raises(RuntimeError, match='injected'), cal.ref.preserved_scene(model):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(42)
                p.requires_grad_(False)
            model.audit_buffer.add_(11)
        raise RuntimeError('injected')
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())
    assert flags == {name: p.requires_grad for name, p in model.named_parameters()}
    assert not torch.cuda.is_initialized()
