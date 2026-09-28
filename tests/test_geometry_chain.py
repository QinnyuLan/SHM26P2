"""Synthetic CPU contracts; no scene, image payload, gsplat import or CUDA."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# A frozen test and worker are siblings. Never prefer the live source in that case.
_worker = Path(__file__).with_name('diagnose_geometry_chain.py')
if not _worker.exists():
    _worker = Path(__file__).resolve().parents[1]/'scripts/diagnose_geometry_chain.py'
_spec = importlib.util.spec_from_file_location('geometry_chain_contract_worker', _worker)
chain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chain)


def array(tensor):
    return tensor.detach().numpy()


def affine_graph(means):
    matrices = [
        [[.3, -.1], [.2, .4]], [[-.2, .1], [.5, .2]],
        [[.1, .2], [-.1, .3]], [[.4, .1], [.2, -.2]],
    ]
    mids = [means @ torch.tensor(m, dtype=means.dtype) for m in matrices]
    weights = torch.tensor([[.1, .2, -.1], [.2, -.1, .1]], dtype=means.dtype)
    raw = (.45+sum((i+1)*mid @ weights for i, mid in enumerate(mids))).reshape(1, 2, 3)
    return mids, raw


def test_simultaneous_parent_intermediate_and_image_gradients_preserve_full_chain():
    means = torch.tensor([[.1, -.2], [.2, .1]], dtype=torch.float64, requires_grad=True)
    mids, raw = affine_graph(means)
    target = torch.tensor([[[.2, .3, .4], [.3, .4, .2]]], dtype=torch.float64)
    loss = (raw.clamp(0, 1)-target).square().mean()
    gradients = torch.autograd.grad(loss, [means, *mids, raw])
    # Requesting intermediates AND their parent does not cut any parent branch.
    fresh = means.detach().clone().requires_grad_(True)
    _, fresh_raw = affine_graph(fresh)
    expected, = torch.autograd.grad((fresh_raw.clamp(0, 1)-target).square().mean(), fresh)
    torch.testing.assert_close(gradients[0], expected, rtol=0, atol=0)
    assert all(torch.count_nonzero(g) > 0 for g in gradients)
    direction = torch.tensor([[.2, -.3], [-.1, .25]], dtype=torch.float64)
    h = .02
    plus, minus = means.detach()+h*direction, means.detach()-h*direction
    mids_p, raw_p = affine_graph(plus)
    mids_m, raw_m = affine_graph(minus)
    A = chain.dot_secant(array(gradients[0]), array(plus), array(minus), h)['value']
    B = sum(chain.dot_secant(array(g), array(p), array(m), h)['value']
            for g, p, m in zip(gradients[1:-1], mids_p, mids_m, strict=True))
    report = chain.mse_decomposition(array(raw), array(raw_p), array(raw_m), array(target),
                                     np.ones((1, 2), bool), array(gradients[-1]), h)
    assert abs(A) > 1e-3  # This is a nondegenerate chain, not a zero-equals-zero test.
    for value in (B, report['C_raw_image_vjp'], report['D_loss_fd']):
        assert value == pytest.approx(A, rel=1e-12, abs=1e-14)
    assert abs(report['quadratic_remainder']) < 1e-14


def test_nonlinear_chain_separates_image_secant_and_exact_quadratic_remainder():
    x = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    middle = x.square()
    weights = torch.tensor([1., 2., 3.], dtype=torch.float64).reshape(1, 1, 3)
    raw = .3+weights*middle.square()
    target = torch.full_like(raw, .1)
    g_x, g_middle, g_image = torch.autograd.grad((raw-target).square().mean(), [x, middle, raw])
    h = .05
    xp, xm = x.detach()+h, x.detach()-h
    rp, rm = .3+weights*xp**4, .3+weights*xm**4
    A = chain.dot_secant(array(g_x), array(xp), array(xm), h)['value']
    B = chain.dot_secant(array(g_middle), array(xp**2), array(xm**2), h)['value']
    report = chain.mse_decomposition(array(raw), array(rp), array(rm), array(target),
                                     np.ones((1, 1), bool), array(g_image), h)
    assert B == pytest.approx(A, rel=1e-12)
    assert abs(report['C_raw_image_vjp']-B) > 1e-4
    assert abs(report['quadratic_remainder']) > 1e-4
    assert report['D_loss_fd'] == pytest.approx(
        report['C_raw_image_vjp']+report['quadratic_remainder'], abs=1e-14)
    assert abs(report['quadratic_identity_residual']) < 1e-14


def test_clamp_crossing_is_distinct_from_quadratic_loss_remainder():
    raw = torch.tensor([[[-.1, .5, 1.1], [-.2, .5, 1.2]]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[[.2, .3, .8], [.1, .2, .3]]], dtype=torch.float64)
    valid = np.array([[True, False]])
    gradient, = torch.autograd.grad((raw.clamp(0, 1)[valid]-target[valid]).square().mean(), raw)
    plus = np.array([[[.2, .6, .8], [.2, .5, .8]]])
    minus = np.array([[[-.3, .4, 1.3], [.2, .5, .8]]])
    report = chain.mse_decomposition(array(raw), plus, minus, array(target), valid, array(gradient), .1)
    assert report['clamp_crossing_valid_channels_plus'] == 2
    assert report['clamp_crossing_valid_channels_minus'] == 0
    assert abs(report['raw_to_clamped_C_difference']) > .1
    assert abs(report['quadratic_remainder']) > .01
    assert report['D_loss_fd']-report['C_raw_image_vjp'] == pytest.approx(
        report['raw_to_clamped_C_difference']+report['quadratic_remainder'], abs=1e-14)
    assert abs(report['quadratic_identity_residual']) < 1e-14
    assert np.count_nonzero(array(gradient)[~valid]) == 0


def test_invalid_projection_storage_is_not_a_complete_intermediate_secant():
    gradient = np.array([[1., 2.], [3., 4.], [0., 0.]])
    plus = np.array([[.2, .4], [np.nan, np.inf], [np.nan, np.nan]])
    minus = np.array([[-.2, -.4], [np.nan, np.inf], [np.nan, np.nan]])
    baseline = np.array([True, True, False])
    endpoint = np.array([True, False, False])
    report = chain.intermediate_secant(gradient, plus, minus, baseline, endpoint, baseline, .2)
    assert report['value'] is None and not report['complete']
    assert report['common_valid_value'] == 5.
    assert report['nonzero_gradient_invalid_endpoint_rows'] == 1
    assert report['nonzero_gradient_common_valid_rows'] == 1
    gradient[1] = 0
    complete = chain.intermediate_secant(gradient, plus, minus, baseline, endpoint, baseline, .2)
    assert complete['complete'] and complete['value'] == 5.
    # Even common-valid zero-adjoint storage need not be multiplied by NaN.
    all_valid = np.ones(3, bool)
    assert chain.intermediate_secant(gradient, plus, minus, all_valid, all_valid, all_valid, .2)['value'] == 5.
    with pytest.raises(ValueError, match='shape/gradient'):
        chain.intermediate_secant(gradient, plus, minus, baseline[:2], endpoint, baseline, .2)
    gradient[0, 0] = np.nan
    with pytest.raises(ValueError, match='shape/gradient'):
        chain.intermediate_secant(gradient, plus, minus, baseline, endpoint, baseline, .2)


def test_fixed_reference_replay_guard_uses_all_and_only_referenced_rows():
    ids = np.array([0, 0, 2, 2])
    baseline = np.array([True, False, True, True])
    endpoint = np.array([True, True, True, False])
    allowed = chain.reference_validity(ids, baseline, endpoint, baseline)
    assert allowed['referenced_unique_rows'] == 2 and allowed['endpoint_replay_allowed']
    endpoint[2] = False
    denied = chain.reference_validity(ids, baseline, endpoint, baseline)
    assert not denied['endpoint_replay_allowed'] and denied['invalid_plus'] == 1
    with pytest.raises(ValueError, match='Baseline list references invalid'):
        chain.reference_validity([1], baseline, endpoint, baseline)
    for invalid_ids in ([], [-1], [4]):
        with pytest.raises(ValueError, match='Invalid baseline references'):
            chain.reference_validity(invalid_ids, baseline, endpoint, baseline)


def test_capture_preserves_graph_tensor_identity_and_restores_after_errors():
    def raster(means2d, colors, *, gain=2.):
        return means2d*colors*gain, means2d.sigmoid()
    module = SimpleNamespace(rasterize_to_pixels=raster)
    means = torch.tensor([.3], dtype=torch.float64, requires_grad=True)
    colors = means.square()
    with chain.capture_raster(module) as captured:
        output = module.rasterize_to_pixels(means, colors)
        assert captured['inputs']['means2d'] is means
        assert captured['inputs']['colors'] is colors
        assert captured['inputs']['gain'] == 2.
        assert captured['output'] is output
        gradient, = torch.autograd.grad(output[0].sum(), means)
        assert float(gradient) == pytest.approx(6*.3**2)
    assert module.rasterize_to_pixels is raster
    with pytest.raises(RuntimeError, match='body'), chain.capture_raster(module):
        raise RuntimeError('body')
    assert module.rasterize_to_pixels is raster
    with pytest.raises(ValueError, match='exactly one'), chain.capture_raster(module):
        module.rasterize_to_pixels(means, colors)
        module.rasterize_to_pixels(means, colors)
    assert module.rasterize_to_pixels is raster
    def failing_raster(value):
        raise RuntimeError('raster failure')
    module.rasterize_to_pixels = failing_raster
    with pytest.raises(RuntimeError, match='raster failure'), chain.capture_raster(module):
        module.rasterize_to_pixels(means)
    assert module.rasterize_to_pixels is failing_raster


def test_numerical_report_strict_json_and_exclusive_write(tmp_path):
    zeros = np.zeros((1, 1, 3))
    report = {
        'A': chain.dot_secant(np.array([2.]), np.array([.2]), np.array([-.2]), .2),
        'B': chain.intermediate_secant(np.array([1., 1.]), np.array([1., np.nan]),
            np.array([-1., np.nan]), [True, True], [True, False], [True, True], .2),
        'image': chain.mse_decomposition(zeros+.5, zeros+.6, zeros+.4, zeros+.2,
            np.ones((1, 1), bool), zeros+.2, .2),
        'replay': chain.reference_validity([0], [True], [False], [True]),
    }
    decoded = json.loads(json.dumps(report, allow_nan=False))
    assert decoded['B']['value'] is None and decoded['B']['complete'] is False
    assert decoded['replay']['endpoint_replay_allowed'] is False
    path = tmp_path/'report.json'
    chain.write(path, report)
    original = path.read_bytes()
    assert json.loads(original) == decoded
    with pytest.raises(FileExistsError):
        chain.write(path, report)
    assert path.read_bytes() == original
    bad = tmp_path/'must_not_exist.json'
    with pytest.raises(ValueError):
        chain.write(bad, {'bad': float('nan')})
    assert not bad.exists()


def test_secant_rejects_nonfinite_or_misaligned_inputs_without_tolerance_padding():
    for gradient, plus, minus, h in (
        ([1.], [2., 3.], [1.], .1), ([np.nan], [2.], [1.], .1),
        ([1.], [np.inf], [1.], .1), ([1.], [2.], [1.], 0.),
    ):
        with pytest.raises(ValueError, match='Invalid finite secant'):
            chain.dot_secant(gradient, plus, minus, h)
    with pytest.raises(ValueError, match='Bad image grids'):
        chain.mse_decomposition(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)),
            np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), np.zeros((1, 1), bool),
            np.zeros((1, 1, 3)), .1)
