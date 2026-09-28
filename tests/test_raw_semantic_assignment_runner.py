import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def runner():
    path = Path(__file__).resolve().parents[1]/'scripts/audit_raw_semantic_assignment.py'
    spec = importlib.util.spec_from_file_location('assignment_runner_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def toy_raster(**kwargs):
    colors = kwargs['colors']
    W = kwargs['weights']
    raw = W @ colors
    return raw[None, None], W.sum(-1)[None, None, :, None], {}


def test_em_capture_leaf_survives_outer_nograd_and_does_not_update_features():
    r = runner(); module = SimpleNamespace(rasterization=toy_raster)
    feature = torch.randn(3, 5, requires_grad=True)
    geom = torch.tensor([[.1, .2, .3]], requires_grad=True)
    with torch.no_grad(), r.capture_raw(module, detach_colors=True) as capture:
        module.rasterization(colors=feature.softmax(-1), weights=geom)
    assert capture['raw'].requires_grad
    grad = torch.autograd.grad(capture['raw'][0, 0, 2], capture['colors'])[0]
    torch.testing.assert_close(grad[:, 2], geom[0].detach())
    assert feature.grad is None and geom.grad is None
    assert module.rasterization is toy_raster


def test_adam_capture_preserves_feature_path_and_restores_wrapper_on_error():
    r = runner(); module = SimpleNamespace(rasterization=toy_raster)
    feature = torch.randn(3, 5, requires_grad=True)
    W = torch.tensor([[.1, .2, .3]])
    try:
        with r.capture_raw(module, detach_colors=False) as capture:
            module.rasterization(colors=feature.softmax(-1), weights=W)
            loss = r.affine_raw_ce(capture['raw'], torch.tensor([[2]]), torch.ones(1, 1), torch.ones(5))
            loss.backward()
            assert feature.grad is not None and torch.count_nonzero(feature.grad) > 0
            raise RuntimeError('finally')
    except RuntimeError:
        pass
    assert module.rasterization is toy_raster


def test_em_only_maps_active_rows_when_actual_qold_is_not_exact_simplex():
    r = runner(); torch.manual_seed(21)
    f = torch.randn(4, 16); W = torch.randn(5, 16); b = torch.randn(5)
    qold = (f @ W.T+b).softmax(-1).numpy().astype(np.float64)
    qold[1, 0] += 1e-7  # Deliberate real-FP32-like simplex rounding error.
    M = np.zeros((4, 5)); M[0] = [1, 0, 3, 0, 1]
    candidate, target, active, _ = r.em_feature_candidate(f, W, b, M, qold)
    assert torch.equal(candidate[~active], f[~active])
    np.testing.assert_array_equal(target[~active], qold[~active])
    np.testing.assert_allclose((candidate[active] @ W.T+b).softmax(-1), target[active], atol=5e-6, rtol=0)
    allzero, _, zeroactive, _ = r.em_feature_candidate(f, W, b, M*0, qold)
    assert not zeroactive.any() and torch.equal(allzero, f)


def test_adam_probability_audit_allows_zero_outside_em_domain():
    audit = runner().q_audit(np.array([[1., 0, 0, 0, 0]]))
    assert audit['zero_components'] == 4 and audit['components_below_epsilon'] == 4
