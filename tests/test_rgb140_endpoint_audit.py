"""Small CPU contracts for completed-only access and source/log cost reconstruction."""
import importlib.util
import json
from pathlib import Path

import pytest


def helper(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('rgb140_endpoint_test', scripts/'audit_rgb140_reference_endpoint.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_running_receipt_rejects_before_any_checkpoint_hash_or_load(tmp_path, monkeypatch):
    module = helper(monkeypatch)
    (tmp_path/'experiment_receipt.json').write_text(json.dumps({'status': 'running'}))
    (tmp_path/'last.pt').write_bytes(b'active payload must stay unread')
    monkeypatch.setattr(module, 'digest', lambda *a: pytest.fail('no hashing before completed guard'))
    monkeypatch.setattr(module.torch, 'load', lambda *a, **k: pytest.fail('no load before completed guard'))
    with pytest.raises(ValueError, match='checkpoint unopened'):
        module.bound_run(tmp_path, 'not used', mixed=True)


def test_training_complete_still_requires_natural_process_exit_confirmation(tmp_path, monkeypatch):
    module = helper(monkeypatch)
    (tmp_path/'experiment_receipt.json').write_text(json.dumps({'status': 'completed'}))
    (tmp_path/'launch_receipt.json').write_text(json.dumps({'status': 'running'}))
    with pytest.raises(ValueError, match='natural-exit'):
        module.require_completed(tmp_path)
    (tmp_path/'launch_receipt.json').write_text(json.dumps({'status': 'completed', 'exit_code': 1}))
    with pytest.raises(ValueError, match='naturally succeed'):
        module.require_completed(tmp_path)


def test_post_update_counts_apply_to_next_render_and_diagnostics_precede_grow(monkeypatch):
    module = helper(monkeypatch)
    logs = [{'step': 1, 'gaussians': 10}, {'step': 100, 'gaussians': 20},
            {'step': 200, 'gaussians': 15}, {'step': 300, 'gaussians': 99}]
    result = module.reconstruct_cost(logs, [(1320, 989)], total_steps=300, initial_count=10,
                                     diagnostic_every=100, diagnostic_start=100, diagnostic_stop=301)
    assert result['primary_gaussian_steps'] == 100*(10+20+15)
    assert result['diagnostic_gaussian_render_sum'] == 2*(10+20+15)
    assert result['diagnostic_renders'] == 6
    assert result['final_N'] == 99
    assert result['boundary_totals'][100][0] == 1000


def test_native_progressive_phase_boundaries_and_round_to_even_are_not_approximated(monkeypatch):
    module = helper(monkeypatch)
    logs = [{'step': 1, 'gaussians': 62000}]+[
        {'step': step, 'gaussians': 62000} for step in range(100, 30001, 100)]
    result = module.reconstruct_cost(logs, [(1320, 989)]*350)
    phases = result['resolution_phases']
    assert (phases['0.5']['steps'], phases['0.5']['width'], phases['0.5']['height']) == (4499, 660, 494)
    assert (phases['0.75']['steps'], phases['0.75']['width'], phases['0.75']['height']) == (13500, 990, 742)
    assert phases['1.0']['steps'] == 12001
    assert result['primary_render_pixels'] == 4499*660*494+13500*990*742+12001*1320*989
    assert result['primary_gaussian_steps'] == 62000*30000
    assert result['diagnostic_renders'] == 0


def test_missing_boundary_or_variable_size_refuses_cost_reconstruction(monkeypatch):
    module = helper(monkeypatch)
    logs = [{'step': 1, 'gaussians': 10}, {'step': 100, 'gaussians': 20}]
    with pytest.raises(ValueError, match='boundary log'):
        module.reconstruct_cost(logs, [(1320, 989)], total_steps=300, initial_count=10)
    logs += [{'step': 200, 'gaussians': 20}, {'step': 300, 'gaussians': 20}]
    with pytest.raises(ValueError, match='view-order'):
        module.reconstruct_cost(logs, [(1320, 989), (1319, 989)], total_steps=300, initial_count=10)
