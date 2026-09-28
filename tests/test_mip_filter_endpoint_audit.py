"""CPU contracts for completed-only access and independently counted refresh events."""
import difflib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


def helper(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('mip_endpoint_test', scripts / 'audit_mip_filter_reference_endpoint.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refresh_records():
    rows = []
    for i, step in enumerate([1] + list(range(100, 30001, 100)), 1):
        rows.append({'step': step, 'gaussians': 100, 'mip_filter_gaussians': 100,
                     'mip_filter_refresh_count': i, 'mip_filter_last_refresh_step': step // 100 * 100,
                     'mip_filter_last_refresh_cuda_ms': 2., 'mip_filter_total_refresh_cuda_ms': 2. * i,
                     'mip_filter_last_refresh_wall_seconds': .125,
                     'mip_filter_total_refresh_wall_seconds': .125 * i})
    saved = {'refresh_count': 301, 'last_refresh_step': 30000,
             'total_refresh_cuda_ms': 602., 'total_refresh_wall_seconds': 37.625}
    return rows, saved


def test_running_endpoint_never_hashes_or_loads_checkpoint(tmp_path, monkeypatch):
    module = helper(monkeypatch)
    (tmp_path / 'experiment_receipt.json').write_text(json.dumps({'status': 'running'}))
    (tmp_path / 'last.pt').write_bytes(b'active checkpoint')
    monkeypatch.setattr(module, 'digest', lambda *a: pytest.fail('hash before completion guard'))
    monkeypatch.setattr(module.torch, 'load', lambda *a, **k: pytest.fail('load before completion guard'))
    with pytest.raises(ValueError, match='checkpoint unopened'):
        module.audit(tmp_path, tmp_path / 'audit.json')


def test_refresh_cost_is_rebuilt_and_endpoint_is_not_double_counted(monkeypatch):
    module = helper(monkeypatch)
    rows, saved = refresh_records()
    result = module.audit_refreshes(rows, saved)
    assert result['count'] == 301 and result['total_cuda_ms'] == 602.
    assert result['initialization_wall_seconds'] == .125 and result['loop_wall_seconds'] == 37.5
    bad = deepcopy(rows)
    bad[-1]['mip_filter_refresh_count'] += 1
    with pytest.raises(ValueError, match='duplicated'):
        module.audit_refreshes(bad, saved)
    bad = deepcopy(rows)
    bad[10]['mip_filter_last_refresh_cuda_ms'] += 1.
    with pytest.raises(ValueError, match='independent sum'):
        module.audit_refreshes(bad, saved)
    bad = deepcopy(rows)
    bad[20]['mip_filter_gaussians'] -= 1
    with pytest.raises(ValueError, match='topology-sized'):
        module.audit_refreshes(bad, saved)


def test_topology_and_sticky_reset_counters_are_not_overcounted(monkeypatch):
    module = helper(monkeypatch)
    from bridge_rgs.mixed_gradient_reference import MixedGradientConfig
    policy = MixedGradientConfig()
    rows, _ = refresh_records()
    n, last_reset = 100, None
    for row in rows:
        step = row['step']
        if policy.grows_at(step):
            row.update(splits=2, duplicates=1, pruned=1)
            n += 2
        row['gaussians'] = n
        if policy.resets_at(step):
            last_reset = step
        if last_reset:
            row.update(opacity_reset_step=last_reset, opacity_reset_count=3)
    result = module.audit_topology(rows, policy)
    assert len(result['events']) == 144
    assert [v['step'] for v in result['resets']] == [3000, 6000, 9000, 12000]
    rows[-1]['gaussians'] += 1
    with pytest.raises(ValueError, match='not explained'):
        module.audit_topology(rows, policy)


def test_restored_proposal_discloses_transient_edit_and_rejects_nonmatching_restore(tmp_path, monkeypatch):
    module = helper(monkeypatch)
    original, transient = tmp_path / 'proposal.md', tmp_path / 'transient.md'
    original.write_text('registered proposal')
    transient.write_text('temporary status edit')
    expected = module.digest(original)
    incident = {'file': str(original), 'preserved_copy': str(transient),
                'expected_preregistered_sha256': expected, 'restored_sha256': expected,
                'transient_sha256': module.digest(transient)}
    (tmp_path / 'proposal_edit_incident.json').write_text(json.dumps(incident))
    (tmp_path / 'transient_proposal_edit.diff').write_text(''.join(difflib.unified_diff(
        original.read_text().splitlines(True), transient.read_text().splitlines(True),
        fromfile='preregistered-restored', tofile='transient-edit')))
    planned = {str(original): {'sha256': expected}}
    report = module.audit_proposal_incident(tmp_path, planned)
    assert report['preregistered_and_endpoint_document_bytes_match']
    assert report['continuous_document_immutability'] is False
    original.write_text('another version must not be accepted')
    with pytest.raises(ValueError, match='restored exactly'):
        module.audit_proposal_incident(tmp_path, planned)
