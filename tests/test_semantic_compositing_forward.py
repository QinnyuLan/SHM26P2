import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).parents[1]/'scripts'
spec = importlib.util.spec_from_file_location('audit_semantic_compositing_lp', SCRIPTS/'audit_semantic_compositing_lp.py')
core = importlib.util.module_from_spec(spec); sys.modules[spec.name] = core; spec.loader.exec_module(core)
spec = importlib.util.spec_from_file_location('compositing_forward', SCRIPTS/'audit_semantic_compositing_forward.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def test_recovery_is_explicit_common_scale_and_keeps_small_tails():
    w = np.array([.2, .3, 1e-15])
    corrected, audit = M.recover_row(w, .75)
    assert corrected[-1] > 0
    np.testing.assert_allclose(corrected/w, np.repeat(.75/w.sum(), 3), rtol=1e-15)
    assert corrected.sum() == pytest.approx(.75)
    assert audit['original_signed_mass_error'] != 0
    for bad, a in [([-1., 2.], .8), ([0, 0], .5), ([1., 0], 0.)]:
        with pytest.raises(ValueError):
            M.recover_row(bad, a)


def test_interior_q_minimum_norm_features_and_preserved_nullspace():
    rng = np.random.default_rng(9)
    current = rng.normal(size=(20, 16)).astype(np.float32)
    weight = rng.normal(size=(5, 16)).astype(np.float32)
    bias = rng.normal(size=5).astype(np.float32)
    q = np.eye(5)[rng.integers(0, 5, 20)]
    original = current.copy()
    features, interior, audit = M.feature_candidate(current, weight, bias, q)
    np.testing.assert_array_equal(current, original)
    assert interior.min() == pytest.approx(1e-5)
    np.testing.assert_allclose(interior.sum(-1), 1.)
    actual = (features @ torch.tensor(weight).T+torch.tensor(bias)).softmax(-1).numpy()
    np.testing.assert_allclose(actual, interior, atol=5e-6, rtol=0)
    D = weight[1:].astype(float)-weight[0].astype(float)
    _, _, vh = np.linalg.svd(D, full_matrices=True)
    # The FP32 write may introduce ~1e-7 null-space rounding, but the FP64
    # construction is the minimum norm row-space solution.
    np.testing.assert_allclose((features.numpy()-current) @ vh[4:].T, 0, atol=3e-7)
    assert audit['decoder_difference_rank'] == 4
    assert audit['target_interior_min_q'] == pytest.approx(1e-5)
    assert audit['cpu_float32_realized_min_q'] > 0


def test_rank_deficient_decoder_is_rejected():
    with pytest.raises(ValueError):
        M.feature_candidate(np.zeros((2, 16)), np.zeros((5, 16)), np.zeros(5), np.full((2, 5), .2))


def test_actual_renderer_gate_is_required_not_positive_proxy():
    assert M.classify_actual([.001]*16).startswith('better_shared_assignment')
    assert M.classify_actual([.001]*15+[-.1]) == 'unresolved_no_renderer_infeasibility_claim'
    assert M.classify_actual([1e-4]*16) == 'unresolved_no_renderer_infeasibility_claim'
    with pytest.raises(ValueError):
        M.classify_actual([1.]*15)


def test_original_reconstruction_gate_did_not_change():
    assert core.SPEC['mass_absolute_tolerance'] == 5e-6
    assert core.SPEC['reconstruction_absolute_tolerance'] == 5e-6
    assert M.SPEC['mass_absolute_tolerance'] == 5e-6
    assert M.SPEC['reconstruction_absolute_tolerance'] == 5e-6
    assert M.SPEC['max_scene_renders'] == 16
    assert M.SPEC['max_color_backwards'] == 16
