"""Exact frozen-body intervention and preservation of the matched training settings."""
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ENTRY = HERE/'train_ibgs_correspondence_control.py'
if not ENTRY.exists():
    ENTRY = HERE.parent/'scripts/train_ibgs_correspondence_control.py'
spec = importlib.util.spec_from_file_location('correspondence_training', ENTRY)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def test_training_source_restores_exactly_and_rejects_optimizer_drift():
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


def test_original_numeric_training_settings_are_unchanged():
    parent = worker.read(worker.PARENT/'plan.json')['specification']
    new = worker.profile(parent)
    assert {k for k in new if new[k] != parent.get(k)} == {'protocol', 'arms', 'selection', 'scope', 'control'}
    assert new['train_steps'] == 6000 and new['arms'] == ['top4_normalized_permuted']
    new['adam_betas'][0] = 0
    assert parent['adam_betas'][0] == .9


def test_intervention_precedes_the_single_head_forward_and_reads_no_gt():
    original = (worker.PARENT/'source_snapshot'/worker.ORIGINAL_NAME).read_text()
    adapted = worker.adapted_source(original)
    assert adapted.count('result = net(') == original.count('result = net(') == 1
    assert adapted.index("evidence['features'], permutation_stats =") < adapted.index('result = net(')
    assert adapted.count('target = bank[indices[name]].cuda()') == 1
    assert 'permutation_stats' in adapted and 'diagnose=step <= 2 or step % 200 == 0' in adapted
