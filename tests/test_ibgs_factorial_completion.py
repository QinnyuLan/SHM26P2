"""Exact training-body adaptation and profile checks, without CUDA/GT access."""
import importlib.util
from pathlib import Path
from textwrap import indent

import pytest

HERE = Path(__file__).resolve().parent
ENTRY = HERE/'train_ibgs_factorial_completion.py'
if not ENTRY.exists():
    ENTRY = HERE.parent/'scripts/train_ibgs_factorial_completion.py'
spec = importlib.util.spec_from_file_location('fourth_corner_worker', ENTRY)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def test_training_body_reversibly_changes_only_two_control_checks():
    original = (worker.PARENT/'source_snapshot'/worker.ORIGINAL_NAME).read_text()
    adapted = worker.adapted_source(original)
    compile(adapted, worker.BASE_NAME, 'exec')
    restored = adapted
    for before, after in reversed(worker.ADAPTATIONS):
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored == original
    with pytest.raises(ValueError, match='Exact'):
        worker.adapted_source(original.replace('lr=.001', 'lr=.01'))


def test_profile_preserves_every_numeric_training_setting_and_original_object():
    parent = worker.read(worker.PARENT/'plan.json')['specification']
    frozen = dict(parent); new = worker.profile(parent)
    assert parent == frozen
    changed = {k for k in new if new[k] != parent[k]}
    assert changed == {'protocol', 'arms', 'selection', 'scope'}
    assert new['arms'] == ['median4_normalized'] and new['train_steps'] == 6000
    new['adam_betas'][0] = 0
    assert parent['adam_betas'][0] == .9


def test_added_initialization_check_rejects_cross_run_drift(tmp_path):
    text = worker.ADAPTATIONS[0][1]
    text = text.replace('\n        ', '\n')
    path = tmp_path/'initialization_probe.py'
    path.write_text('def check(require, hashes, initial_hashes, plan):\n'+indent(text, '    ')+'\n')
    check = worker.load(path, 'synthetic_initialization_probe').check
    scope = {'require': worker.require, 'hashes': {'a': 'same'}, 'initial_hashes': {'a': 'same'},
             'plan': {'expected_initial_head_hashes': {'a': 'same'}}}
    check(**scope)
    scope['plan']['expected_initial_head_hashes'] = {'a': 'changed'}
    with pytest.raises(ValueError, match='Original three-arm'):
        check(**scope)
