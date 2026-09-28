"""Synthetic thin RGB-only independent-audit adaptation contracts."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PARENT = Path('/mnt/data/SHM2026/runs/continuous_geometry_proposals_v1/independent_audit_snapshot')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


PATH = HERE/'audit_rgb_geometry_control.py'
if not PATH.exists():
    PATH = HERE.parent/'scripts/audit_rgb_geometry_control.py'
checker = load('rgb_only_independent_checker', PATH)
old_tests = load('old_independent_checker_contracts', PARENT/'test_continuous_geometry_proposals_audit.py')


def example():
    rows, batches, support = old_tests.example()
    for i, row in enumerate(rows):
        row['accepted'] = True; row['accepted_optimizer_steps'] = i+1
        row['acceptance']['control_finite_commit'] = True
        row['proposal']['mode'] = 'rgb'; row['proposal']['optimizer_proposed_step'] = i+1
    # Copy shared baseline: the parent fixture used the same dict for rejection.
    rows[1]['baseline'] = {**rows[1]['baseline'], 'means_sha256': '1'*64}
    return rows, batches, support


def test_finite_rgb_commits_and_steps_are_independently_checked():
    rows, batches, support = example()
    result = checker.check_trace(rows, batches, 'rgb', '0'*64, support, 2, 1.)
    assert result['accepted'] == 2 and result['rejected'] == 0
    assert result['last_means_sha256'] == '2'*64
    json.dumps(result, allow_nan=False)
    rows[0]['accepted'] = False
    with pytest.raises(ValueError, match='acceptance'):
        checker.check_trace(rows, batches, 'rgb', '0'*64, support, 2, 1.)


def test_wrong_semantic_mode_or_cap_is_rejected():
    rows, batches, support = example()
    rows[0]['proposal']['mode'] = 'semantic'
    with pytest.raises(ValueError, match='step'):
        checker.check_trace(rows, batches, 'rgb', '0'*64, support, 2, 1.)
    rows[0]['proposal']['mode'] = 'rgb'; rows[0]['proposal']['actual_mahalanobis_max'] = .02
    with pytest.raises(ValueError, match='cap'):
        checker.check_trace(rows, batches, 'rgb', '0'*64, support, 2, 1.)


def test_saved_record_math_is_exact_inherited_auditor_not_new_formula():
    def definitions(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef)}
    current = definitions(PATH)
    old = definitions(PARENT/'audit_continuous_geometry_proposals.py')
    for name in ('check_rows', 'check_trace', 'cumulative_displacement', 'schedule'):
        assert current[name] == old[name]
    assert current['audit'].replace('rgb_geometry_control_v1', 'continuous_geometry_proposals_v1') == old['audit']
    assert checker.ARMS == ('rgb',)
    assert checker.EXPECTED_COUNTS == {'batch_attempts': 148, 'training_view_pairs': 2072,
        'description_view_pairs': 518, 'raster': 5180, 'means_vjp': 2072, 'target_decodes': 777}
