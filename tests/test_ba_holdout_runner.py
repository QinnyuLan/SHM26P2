"""Execution provenance checks; synthetic files only."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def runner():
    path = Path(__file__).parents[1]/'scripts/diagnose_ba_holdout.py'
    if not path.exists():
        path = Path(__file__).with_name('diagnose_ba_holdout.py')
    spec = importlib.util.spec_from_file_location('ba_runner_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_hashes_detect_changes_but_not_bytecode(tmp_path, runner):
    (tmp_path/'core.py').write_text('value = 1\n')
    before = runner.files(tmp_path)
    (tmp_path/'__pycache__').mkdir()
    (tmp_path/'__pycache__/core.pyc').write_bytes(b'cache')
    assert before == runner.files(tmp_path)
    (tmp_path/'core.py').write_text('value = 2\n')
    assert before != runner.files(tmp_path)


def test_json_and_npz_cannot_replace_evidence(tmp_path, runner):
    path = tmp_path/'receipt.json'
    runner.write_new(path, {'status': 'completed'})
    with pytest.raises(FileExistsError):
        runner.write_new(path, {'status': 'failed'})
    assert json.loads(path.read_text())['status'] == 'completed'
    npz = tmp_path/'result.npz'
    runner.save_new(npz, {'value': np.array([1.])})
    with pytest.raises(FileExistsError):
        runner.save_new(npz, {'value': np.array([2.])})
    with np.load(npz, allow_pickle=False) as f:
        assert f['value'].item() == 1.


def test_nonfinite_json_and_object_arrays_rejected_before_creation(tmp_path, runner):
    path = tmp_path/'bad.json'
    with pytest.raises(ValueError):
        runner.write_new(path, {'bad': float('nan')})
    assert not path.exists()
    path = tmp_path/'bad.npz'
    with pytest.raises(ValueError, match='non-object'):
        runner.save_new(path, {'bad': np.array([{}], dtype=object)})
    assert not path.exists()


def test_started_attempt_cannot_be_reexecuted(tmp_path, runner):
    path = tmp_path/'plan.json'
    path.write_text(json.dumps({'output': str(tmp_path)}))
    (tmp_path/'execution_started.json').write_text('{}')
    with pytest.raises(ValueError, match='Attempt already started'):
        runner.execute(path)


@pytest.mark.parametrize('changed_baseline', [False, True])
def test_execution_freezes_baseline_before_fit_and_preserves_failure(
        tmp_path, runner, monkeypatch, changed_baseline):
    observations = tmp_path/'source.npz'
    arrays = {key: np.zeros(1) for key in runner.GEOMETRY_KEYS}
    # This member cannot be loaded with allow_pickle=False. Execution must not touch it.
    arrays['primary_label'] = np.array([{}], dtype=object)
    np.savez(observations, **arrays)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'views': [{'split': 'train', 'name': 'a', 'image_id': 1,
                                             'w2c_original': np.eye(4).tolist()}]}))
    plan = {'output': str(tmp_path), 'source_snapshot': str(tmp_path), 'analysis_specification': {},
            'observations': str(observations), 'manifest': str(manifest), 'thread_environment': {}}
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    calls = []

    def prepare(loaded, _manifest):
        assert set(loaded) == set(runner.GEOMETRY_KEYS)
        return SimpleNamespace(layout={'rows': np.array([0])})

    def evaluate(problem, overrides):
        calls.append('evaluate')
        value = int(bool(overrides) and changed_baseline)
        return {'arrays': {'baseline_x': np.array([value])}, 'report': {'count': 1}}

    def fit(problem):
        calls.append('fit')
        assert (tmp_path/'baseline_screen.npz').exists()
        assert (tmp_path/'baseline_screen.json').exists()
        return {1: np.eye(4)}, {'accepted': True}

    core = SimpleNamespace(SPEC={}, prepare_problem=prepare, evaluate=evaluate, run_fit=fit,
                           analyze=lambda *_: {'gate': 'synthetic'})
    monkeypatch.setattr(runner, 'verify', lambda _: None)
    monkeypatch.setattr(runner, 'load_core', lambda _: core)
    if changed_baseline:
        with pytest.raises(ValueError, match='Baseline population or geometry changed'):
            runner.execute(path)
    else:
        runner.execute(path)
    assert calls == ['evaluate', 'fit', 'evaluate']
    receipt = json.loads((tmp_path/'execution_receipt.json').read_text())
    assert receipt['status'] == ('failed' if changed_baseline else 'completed')
    assert receipt['label_array_loads'] == 0
    with pytest.raises(ValueError, match='Attempt already started'):
        runner.execute(path)
