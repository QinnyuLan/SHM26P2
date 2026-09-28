import pytest
import torch

from bridge_rgs.partition_field import SemanticPartitionField
from bridge_rgs.partition_projection import FrozenProjection, prepare_shader_coefficients


@pytest.mark.parametrize('mode', ['integrated', 'point', 'marginal'])
def test_initialization_preserves_classifier_and_parameterization(mode):
    logits = torch.linspace(-3, 3, 20, dtype=torch.float64).reshape(4, 5)
    field = SemanticPartitionField(logits, mode)
    qi, qo = field.endpoints()
    assert torch.equal(qi, logits.softmax(-1))
    assert torch.equal(qi, qo)
    normal, b, w, tau = field.slab_parameters()
    assert torch.equal(normal, logits.new_tensor([[1., 0, 0]]).expand(4, 3))
    torch.testing.assert_close(b, torch.zeros_like(b))
    torch.testing.assert_close(w, torch.full_like(w, .75))
    torch.testing.assert_close(tau, torch.full_like(tau, .3))
    assert sum(p.numel() for p in field.parameters()) == 4*15


@pytest.mark.parametrize('mode', ['integrated', 'point'])
def test_shape_gradient_wakes_after_endpoint_separation(mode):
    field = SemanticPartitionField(torch.zeros(2, 5, dtype=torch.float64), mode)
    A = torch.tensor([[[.4, .1], [.1, .3], [.05, .08]]], dtype=torch.float64).expand(2, 3, 2)
    V = .3*torch.eye(3, dtype=torch.float64).expand(2, 3, 3)
    projection = FrozenProjection(A, V, torch.zeros(2, 2), torch.ones(2, dtype=torch.bool), {})
    def response():
        coeff = prepare_shader_coefficients(projection, *field.slab_parameters(), mode=mode)
        m = (coeff[:, :2]*coeff.new_tensor([.5, -.4])).sum(-1)
        gate = .5*(torch.erf((m+coeff[:, 2])/2**.5)-torch.erf((m+coeff[:, 3])/2**.5))
        qi, qo = field.endpoints()
        return (qo+gate[:, None]*(qi-qo))[:, 2].sum()
    response().backward()
    for param in (field.direction, field.offset_raw, field.width_raw):
        assert torch.count_nonzero(param.grad) == 0
    field.zero_grad(set_to_none=True)
    with torch.no_grad():
        field.inside_logits[:, 2] = .2
        field.outside_logits[:, 2] = -.2
    response().backward()
    for param in (field.direction, field.offset_raw, field.width_raw):
        assert torch.isfinite(param.grad).all() and param.grad.norm() > 0
