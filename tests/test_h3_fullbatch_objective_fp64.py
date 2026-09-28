import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fp64_scalar_ulp_not_fp32_and_identical_finite_difference_contract():
    new = load('audit_h3_fullbatch_objective_fp64')
    old = load('audit_h3_fullbatch_objective')
    assert new.BASE_SHA == old.BASE_SHA
    for key in ('train_sorted_indices', 'view_names', 'pixel_protocol', 'sh_degree', 'parameters',
                'epsilons', 'solver', 'renders_per_view', 'render_calls', 'backward_calls',
                'fd_max_relative_error', 'warp_max_abs_error', 'step_actual_predicted_ratio_bounds'):
        assert new.SPEC[key] == old.SPEC[key]
    original = torch.tensor([.2], dtype=torch.float32)
    plus, minus = original+.001, original-.001
    g = torch.ones(1)
    row = new.finite_difference_record(float(original), float(plus), float(minus), .001, g, original, plus, minus, g)
    assert 'base_fp64_loss' in row and not any('fp32_loss' in k for k in row)
    expected = (abs(np.spacing(float(plus)))+abs(np.spacing(float(minus))))/.002
    assert row['one_ulp_derivative_resolution'] == expected
    assert row['passed']
    assert not torch.cuda.is_initialized()


def test_runner_precision_intervention_does_not_change_inherited_package():
    source = (ROOT/'scripts/audit_h3_fullbatch_objective_fp64.py').read_text()
    assert 'appearance_rgb_loss_fp64(prediction, target, valid)' in source
    assert "shutil.copytree(old_source, snapshot" in source
    assert 'prior_numerical_failure_preserved' in source
    assert 'trial.accept(' not in source.replace('# Deliberately never call trial.accept:', '')
