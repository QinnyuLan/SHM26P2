"""Synthetic CPU contracts; no scene, labels, renderer, or training."""
import numpy as np
import pytest
import torch
from scipy.optimize import brentq
from scipy.special import expit

from bridge_rgs.semantic_mass_constraint import mass_constrained_endpoints

DT = torch.float64


def prior():
    return torch.tensor([.125, .25, .375, .25, 0.], dtype=DT)


def independent_endpoints(q, h, a):
    """Independent SciPy root and direct endpoints for moderate test inputs."""
    center = np.log(a)-np.log1p(-a)
    root = brentq(lambda x: q@expit(h+x)-a,
                  center-h.max()-2, center-h.min()+2, xtol=5e-15)
    return q*expit(h+root)/a, q*expit(-h-root)/(1-a), root


@pytest.mark.parametrize('a', [.125, .5, .875])
def test_zero_logits_identity_and_mass(a):
    q = prior()
    result = mass_constrained_endpoints(q, torch.zeros_like(q), a)
    torch.testing.assert_close(result.q_in, q, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(result.q_out, q, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(a*result.q_in+(1-a)*result.q_out, q, atol=1e-15, rtol=1e-15)
    assert result.q_in[-1] == result.q_out[-1] == 0
    assert result.diagnostics['normalization_or_probability_floor_or_clamp'] is False


def test_common_shift_gauge_value_and_root_gradient():
    q = prior()
    h = torch.tensor([-.75, .25, 1.5, -.5, 2.], dtype=DT, requires_grad=True)
    a = torch.tensor(.37, dtype=DT, requires_grad=True)
    original = mass_constrained_endpoints(q, h, a)
    shifted = mass_constrained_endpoints(q, h+16., a)
    torch.testing.assert_close(original.q_in, shifted.q_in, atol=0, rtol=0)
    torch.testing.assert_close(original.q_out, shifted.q_out, atol=0, rtol=0)
    torch.testing.assert_close(shifted.root, original.root-16, atol=0, rtol=0)
    gh, ga = torch.autograd.grad(original.root, (h, a))
    s = q*original.allocation*original.complement
    torch.testing.assert_close(gh, -s/s.sum(), atol=1e-15, rtol=1e-15)
    torch.testing.assert_close(gh.sum(), torch.tensor(-1., dtype=DT), atol=1e-15, rtol=0)
    torch.testing.assert_close(ga, 1/s.sum(), atol=1e-15, rtol=1e-15)


def test_class_permutation_and_broadcast_gradient():
    q = prior()
    h = torch.tensor([-.7, .3, 1.5, -.2, 0.], dtype=DT, requires_grad=True)
    a = torch.tensor([.2, .7], dtype=DT, requires_grad=True)
    order = torch.tensor([2, 4, 0, 3, 1])
    base = mass_constrained_endpoints(q, h, a)
    permuted = mass_constrained_endpoints(q[order], h[order], a)
    torch.testing.assert_close(permuted.q_in, base.q_in[:, order], atol=3e-14, rtol=3e-14)
    torch.testing.assert_close(permuted.q_out, base.q_out[:, order], atol=3e-14, rtol=3e-14)
    loss = (base.q_in*torch.arange(5, dtype=DT)).sum()+base.q_out.square().sum()
    gh, ga = torch.autograd.grad(loss, (h, a))
    assert gh.shape == h.shape and ga.shape == a.shape
    assert torch.isfinite(gh).all() and torch.isfinite(ga).all()


@pytest.mark.parametrize('a', [1e-10, 1-1e-10])
def test_predeclared_near_endpoint_safe_case_and_tiny_prior(a):
    q = torch.tensor([.125, .25, .375, .25, 1e-200], dtype=DT)
    h = torch.tensor([-1., .5, 2., -2., .75], dtype=DT, requires_grad=True)
    area = torch.tensor(a, dtype=DT, requires_grad=True)
    result = mass_constrained_endpoints(q, h, area)
    assert result.q_in[-1] > 0 and result.q_out[-1] > 0
    torch.testing.assert_close(result.q_in.sum(), torch.tensor(1., dtype=DT), atol=6e-14, rtol=0)
    torch.testing.assert_close(result.q_out.sum(), torch.tensor(1., dtype=DT), atol=6e-14, rtol=0)
    recovered = area*result.q_in+(1-area)*result.q_out
    torch.testing.assert_close(recovered/q, torch.ones_like(q), atol=3e-14, rtol=0)
    gh, ga = torch.autograd.grad(result.q_in[0]+2*result.q_out[2], (h, area))
    assert torch.isfinite(gh).all() and torch.isfinite(ga)


def test_sigmoid_tail_uses_negative_branch_not_one_minus_positive():
    q = prior()
    h = torch.tensor([50., -1., 0., 1., 0.], dtype=DT, requires_grad=True)
    result = mass_constrained_endpoints(q, h, .4)
    assert result.allocation[0] == 1
    assert result.complement[0] > 0 and result.q_out[0] > 0
    grad, = torch.autograd.grad(result.q_out[0], h)
    assert grad[0] != 0 and torch.isfinite(grad).all()


def test_half_area_requires_real_root_not_half_allocations():
    q = prior()
    h = torch.tensor([-3., 1., .5, 2., 0.], dtype=DT)
    result = mass_constrained_endpoints(q, h, .5)
    qi, qo, root = independent_endpoints(q.numpy(), h.numpy(), .5)
    assert abs(root) > .1 and result.diagnostics['bisection_iterations_max'] > 1
    assert torch.max(abs(result.allocation-.5)) > .3
    np.testing.assert_allclose(result.q_in.numpy(), qi, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(result.q_out.numpy(), qo, atol=3e-14, rtol=3e-14)


def test_gradcheck_and_independent_h_a_finite_differences():
    q = prior()
    h = torch.tensor([-.7, .3, 1.5, -.2, 0.], dtype=DT, requires_grad=True)
    a = torch.tensor(.37, dtype=DT, requires_grad=True)

    def function(logits, occupancy):
        result = mass_constrained_endpoints(q, logits, occupancy)
        return torch.cat((result.q_in, result.q_out))

    assert torch.autograd.gradcheck(function, (h, a), eps=1e-6, atol=2e-7, rtol=2e-5)
    cin = np.array([.3, -.2, .1, 1., .7])
    cout = np.array([-.4, .5, .9, -.2, 1.1])
    result = mass_constrained_endpoints(q, h, a)
    objective = result.q_in@torch.from_numpy(cin)+result.q_out@torch.from_numpy(cout)
    gh, ga = torch.autograd.grad(objective, (h, a))

    def independent_loss(x, y):
        qi, qo, _ = independent_endpoints(q.numpy(), x, y)
        return qi@cin+qo@cout

    step = 1e-5
    x = h.detach().numpy()
    fd = np.array([(independent_loss(x+np.eye(5)[k]*step,float(a.detach()))
                    -independent_loss(x-np.eye(5)[k]*step,float(a.detach())))/(2*step)
                   for k in range(5)])
    fda = (independent_loss(x,float(a.detach())+step)-independent_loss(x,float(a.detach())-step))/(2*step)
    np.testing.assert_allclose(gh.numpy(), fd, atol=2e-10, rtol=2e-8)
    np.testing.assert_allclose(float(ga), fda, atol=2e-10, rtol=2e-8)


def test_saturated_balanced_pseudoroot_is_rejected_by_derivative_contract():
    q = torch.tensor([.5, .5], dtype=DT)
    h = torch.tensor([-100., 100.], dtype=DT)
    with pytest.raises(ValueError, match='Root derivative'):
        mass_constrained_endpoints(q, h, .5)


@pytest.mark.parametrize('bad', ['negative', 'unnormalized', 'learnable', 'nan', 'a0', 'a1', 'fp32'])
def test_invalid_inputs_are_not_repaired(bad):
    q = prior()
    h = torch.zeros_like(q)
    a = .4
    if bad == 'negative':
        q[0] = -.1
    elif bad == 'unnormalized':
        q = q*.9
    elif bad == 'learnable':
        q.requires_grad_()
    elif bad == 'nan':
        h[0] = float('nan')
    elif bad == 'a0':
        a = 0.
    elif bad == 'a1':
        a = 1.
    else:
        q, h = q.float(), h.float()
    with pytest.raises(ValueError):
        mass_constrained_endpoints(q, h, a)


def test_near_boundary_prior_sum_roundoff_is_not_hidden():
    q = prior()
    q[0] += 2*torch.finfo(DT).eps  # Within input sum tolerance, amplified outside.
    with pytest.raises(ValueError, match='simplex/mass closure'):
        mass_constrained_endpoints(q, torch.zeros_like(q), 1-1e-10)


def test_double_backward_is_explicitly_unsupported():
    h = torch.tensor([-.4, .6, 1., -.2, 0.], dtype=DT, requires_grad=True)
    result = mass_constrained_endpoints(prior(), h, .4)
    with pytest.raises(RuntimeError, match='create_graph=True is forbidden'):
        torch.autograd.grad(result.root, h, create_graph=True)


@pytest.mark.parametrize('output', ['q_in', 'q_out', 'both', 'allocation', 'complement', 'root'])
@pytest.mark.parametrize('variable', ['h', 'a'])
def test_all_output_paths_reject_high_order_graph_at_entry(output, variable):
    h = torch.tensor([-.4, .6, 1., -.2, 0.], dtype=DT, requires_grad=True)
    a = torch.tensor(.37, dtype=DT, requires_grad=True)
    result = mass_constrained_endpoints(prior(), h, a)
    if output == 'both':
        loss = result.q_in.square().sum()+result.q_out.square().sum()
    else:
        loss = getattr(result, output).square().sum()
    target = h if variable == 'h' else a
    # The error occurs while requesting the first derivative graph, before
    # an incomplete second derivative can leak through explicit arithmetic.
    with pytest.raises(RuntimeError, match='create_graph=True is forbidden'):
        torch.autograd.grad(loss, target, create_graph=True)
