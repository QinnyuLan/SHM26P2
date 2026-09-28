"""Small CPU-only analytic chains; no real images, checkpoint or renderer."""
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    'gradient_chain', Path(__file__).parents[1]/'scripts/audit_appearance_gradient_chain.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture(autouse=True)
def threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def example():
    y, x = torch.meshgrid(torch.arange(11, dtype=torch.float64),
                          torch.arange(13, dtype=torch.float64), indexing='ij')
    base = torch.stack((.3+x*.007, .4+y*.009, .2+(x+y)*.005), -1)
    target = base + .04
    valid = torch.ones(base.shape[:2], dtype=torch.bool)
    valid[0, :2] = False
    return base, target, valid


def test_correct_linear_render_chain_matches_leaf_and_scalar_fd():
    base, target, valid = example()
    parameters = {name: torch.tensor(.01, dtype=torch.float64, requires_grad=True)
                  for name in audit.PARAMETERS}
    prediction = base + parameters[audit.PARAMETERS[0]] + .5*parameters[audit.PARAMETERS[1]] + .2*parameters[audit.PARAMETERS[2]]
    values, gradients = audit.graph_gradients(prediction, parameters, target, valid)
    direction = gradients['combined']['parameters'][audit.PARAMETERS[0]].sign()
    eps = 1e-5
    plus, minus = prediction.detach()+eps*direction, prediction.detach()-eps*direction
    rows, _ = audit.leaf_analysis(prediction.detach(), plus, minus, target, valid, eps, torch.float64)
    for component in audit.COMPONENTS:
        derivative = float(gradients[component]['parameters'][audit.PARAMETERS[0]]*direction)
        assert rows[component]['image_gradient_dot_Jd'] == pytest.approx(derivative, abs=1e-10)
        assert rows[component]['scalar_central_difference'] == pytest.approx(derivative, abs=2e-7)
    assert values['combined'] == pytest.approx(.8*values['l1']+.2*values['dssim7'], abs=1e-15)
    assert all(row['relative_l2'] < 1e-12 for row in audit.reassembly(gradients).values())
    assert all(p.grad is None for p in parameters.values())


def test_scaled_wrong_renderer_backward_is_separated_from_loss_chain():
    class WrongDerivative(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, gradient):
            return gradient*3

    base, target, valid = example()
    parameters = {name: torch.tensor(.01, dtype=torch.float64, requires_grad=True)
                  for name in audit.PARAMETERS}
    prediction = base + WrongDerivative.apply(parameters[audit.PARAMETERS[0]]) + parameters[audit.PARAMETERS[1]] + parameters[audit.PARAMETERS[2]]
    _, gradients = audit.graph_gradients(prediction, parameters, target, valid)
    direction = gradients['combined']['parameters'][audit.PARAMETERS[0]].sign()
    eps = 1e-5
    rows, _ = audit.leaf_analysis(prediction.detach(), prediction.detach()+eps*direction,
                                 prediction.detach()-eps*direction, target, valid, eps, torch.float64)
    param_dot = float(gradients['combined']['parameters'][audit.PARAMETERS[0]]*direction)
    image_dot = rows['combined']['image_gradient_dot_Jd']
    assert param_dot == pytest.approx(3*image_dot, rel=1e-9)
    assert rows['combined']['scalar_central_difference'] == pytest.approx(image_dot, abs=2e-7)


def test_dtype_and_crop_stride_are_recorded_without_extra_render():
    canvas = torch.linspace(.2, .7, 17*19*3).reshape(17, 19, 3)
    base = canvas[3:14, 2:15]
    target, valid = base.detach()+.02, torch.ones(base.shape[:2], dtype=torch.bool)
    assert not base.is_contiguous()
    assert audit.layout_info(base)['stride'] == [57, 3, 1]
    plus, minus = base+.001, base-.001
    for dtype in (torch.float32, torch.float64):
        rows, _ = audit.leaf_analysis(base, plus, minus, target, valid, .001, dtype)
        assert rows['layouts']['baseline_leaf']['dtype'] == str(dtype)
        assert rows['layouts']['baseline_leaf']['contiguous'] == (dtype == torch.float64)
        assert set(rows['reassembly']['value_residuals']) == {'baseline', 'plus', 'minus'}


def test_realized_parameter_direction_and_other_groups_fixed():
    original = torch.tensor([1e4, .1], dtype=torch.float32)
    requested = torch.tensor([1., .2])
    positive, negative = original+.001*requested, original-.001*requested
    gradients = {component: {'parameters': {name: torch.ones(2) for name in audit.PARAMETERS}}
                 for component in audit.COMPONENTS}
    rows, diagnostics = audit.parameter_directions(gradients, requested, positive, negative, original, .001)
    assert diagnostics['relative_l2_direction_error'] > .01
    assert rows['l1']['splats.sh0']['g_dot_requested_direction'] != rows['l1']['splats.sh0']['g_dot_realized_central_direction']
    for component in audit.COMPONENTS:
        for name in audit.PARAMETERS[1:]:
            assert rows[component][name]['direction_is_zero']
            assert rows[component][name]['g_dot_realized_central_direction'] == 0
