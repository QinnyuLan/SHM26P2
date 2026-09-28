"""CPU contracts for the fixed capacity endpoint auditor; no real model reads."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

scripts = Path(__file__).parents[1]/'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('capacity_audit', scripts/'audit_rgb_capacity_endpoint.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
sys.path.pop(0)


def synthetic_rows():
    rows, n = [{'step': 1, 'gaussians': 62000}], 62000
    for step in range(100, 30001, 100):
        row = {'step': step, 'gaussians': n}
        if 500 <= step < 24000 and not any(r <= step < r+400 for r in (6000, 12000, 18000)):
            cap = round(62000+(step-500)/23500*938000)
            allocation = min(12000, round(.06*n))
            growth = min(allocation, max(0, cap-n))
            n += growth
            row.update(gaussians=n, density_target_budget=cap, density_budget=allocation,
                       splits=growth, duplicates=0, pruned=0)
        rows.append(row)
    return rows


def test_existing_window_can_reach_996009_but_not_one_million():
    result = audit.audit_growth(synthetic_rows())
    assert result['actual_final_N'] == result['final_scheduled_cap'] == 996009
    assert result['last_growth_step'] == 23900 and len(result['events']) == 223
    assert not any(r <= x['step'] < r+400 for r in (6000, 12000, 18000) for x in result['events'])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('change', ['post_window', 'pause', 'wrong_addition', 'wrong_budget'])
def test_growth_tampering_is_rejected(change):
    rows = synthetic_rows()
    target = {'post_window': 24000, 'pause': 6000, 'wrong_addition': 1000, 'wrong_budget': 1000}[change]
    row = next(x for x in rows if x['step'] == target)
    if change == 'wrong_addition':
        row['splits'] += 1
    elif change == 'wrong_budget':
        row['density_target_budget'] += 1
    else:
        row['gaussians'] += 1
    with pytest.raises(ValueError):
        audit.audit_growth(rows)


def test_observer_cost_includes_both_diagnostic_renders():
    rows = [{'step': step, 'gaussians': 62000} for step in [1, *range(100, 30001, 100)]]
    counts, reconstruction = audit.observed_cost(rows, [(1320, 989)]*350)
    assert counts['render_calls'] == 32350
    assert counts['gaussian_render_sum'] == 62000*32350
    assert counts['gaussian_canvas_pixel_sum'] == 62000*counts['canvas_pixel_sum']
    assert reconstruction['diagnostic_renders'] == 2350
    assert reconstruction['resolution_phases']['0.5']['steps'] == 4499
    assert reconstruction['resolution_phases']['0.75']['steps'] == 13500
    assert reconstruction['resolution_phases']['1.0']['steps'] == 12001


def test_observer_uses_pre_growth_field_for_step_end_render():
    rows = synthetic_rows()
    base = copy.deepcopy(rows)
    for row in base:
        row['gaussians'] = 62000
    counts, _ = audit.observed_cost(rows, [(1320, 989)])
    flat, _ = audit.observed_cost(base, [(1320, 989)])
    # Explicit step 600 is rendered with the preceding count. An implementation
    # using post-step N would add the total growth to primary render counts.
    by_step = {x['step']: x['gaussians'] for x in rows}
    n, gaussian_sum = 62000, 0
    for step in range(1, 30001):
        gaussian_sum += n*(3 if 500 <= step < 24000 and step % 20 == 0 else 1)
        n = by_step.get(step, n)
    assert counts['gaussian_render_sum'] == gaussian_sum > flat['gaussian_render_sum']


def test_running_outer_gate_precedes_checkpoint_access(tmp_path, monkeypatch):
    (tmp_path/'launch_receipt.json').write_text(json.dumps({'status': 'running', 'plan_sha256': audit.PLAN_SHA}))
    def forbidden(*args, **kwargs):
        raise AssertionError('Checkpoint must not be read while running')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='checkpoint unopened'):
        audit.audit(tmp_path, tmp_path/'audit.json')
    assert not torch.cuda.is_initialized()


def import_fixture(tmp_path, stage):
    modules = ['cli', 'model', 'io', 'losses', 'train', 'refinement']
    modules += ['densification'] if stage == 'train' else ['evaluate']
    snapshot = tmp_path/'source_snapshot'
    package = snapshot/'bridge_rgs'
    package.mkdir(parents=True)
    sources, imports = {}, {}
    for module in modules:
        path = package/f'{module}.py'
        path.write_text(f'# frozen {module}\n')
        sha = audit.digest(path)
        sources[str(path.relative_to(snapshot))] = sha
        imports['bridge_rgs.'+module] = {'path': str(path), 'sha256': sha}
    return snapshot, sources, {'actual_imports': imports}


@pytest.mark.parametrize('stage', ['train', 'native'])
def test_historical_package_attests_io_without_later_data_module(tmp_path, stage):
    snapshot, sources, record = import_fixture(tmp_path, stage)
    assert 'bridge_rgs.io' in record['actual_imports']
    assert 'bridge_rgs.data' not in record['actual_imports']
    audit.audit_runtime_imports(stage, record, snapshot, sources)


@pytest.mark.parametrize(('stage', 'missing'), [
    ('train', 'losses'), ('train', 'densification'), ('native', 'evaluate'), ('native', 'io'),
])
def test_stage_specific_core_attestation_remains_required(tmp_path, stage, missing):
    snapshot, sources, record = import_fixture(tmp_path, stage)
    del record['actual_imports']['bridge_rgs.'+missing]
    with pytest.raises(ValueError, match='Core runtime imports'):
        audit.audit_runtime_imports(stage, record, snapshot, sources)


@pytest.mark.parametrize('stage', ['train', 'native'])
def test_noncore_attested_imports_still_require_frozen_bytes(tmp_path, stage):
    snapshot, sources, record = import_fixture(tmp_path, stage)
    Path(record['actual_imports']['bridge_rgs.refinement']['path']).write_text('# changed\n')
    with pytest.raises(ValueError, match='Runtime source differs: bridge_rgs.refinement'):
        audit.audit_runtime_imports(stage, record, snapshot, sources)
