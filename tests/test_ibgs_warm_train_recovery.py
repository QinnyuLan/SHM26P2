"""Small synthetic contracts; no renderer, checkpoint or real target access."""
import importlib.util
import json
from pathlib import Path

import pytest

ENTRY = Path(__file__).with_name('check_ibgs_warm_train_recovery.py')
if not ENTRY.exists():
    ENTRY = Path(__file__).resolve().parents[1]/'scripts/check_ibgs_warm_train_recovery.py'
spec = importlib.util.spec_from_file_location('ibgs_train_recovery_contract', ENTRY)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_both_completed_fixed_endpoints_required_before_checkpoint_access():
    rows = [{'arm': arm, 'status': 'completed', 'steps': 6000} for arm in worker.ARMS]
    assert set(worker.endpoint_records({'arms': rows})) == set(worker.ARMS)
    for invalid in (rows[:1], rows[:1]+rows[:1], [rows[0], dict(rows[1], steps=5999)],
                    [rows[0], dict(rows[1], status='failed')]):
        with pytest.raises(ValueError):
            worker.endpoint_records({'arms': invalid})


def test_all_64_predictions_before_cached_targets():
    records = [{'arm': a, 'readout': r, 'name': n, 'path': n, 'sha256': 'a'*64}
               for a in worker.ARMS for r in worker.READOUTS for n in worker.NAMES]
    worker.prediction_barrier(records)
    for invalid in (records[:-1], records[:-1]+[records[0]],
                    records[:-1]+[dict(records[-1], name='001.png')],
                    records[:-1]+[dict(records[-1], sha256='')]):
        with pytest.raises(ValueError, match='64 fixed'):
            worker.prediction_barrier(invalid)


def test_natural_lineage_cannot_use_failed_partial_or_different_parent(tmp_path):
    worker.write(tmp_path/'plan.json', {'test': 1})
    execution = {'status': 'completed', 'plan_sha256': worker.sha(tmp_path/'plan.json')}
    worker.write(tmp_path/'execution_receipt.json', execution)
    launch = {**execution, 'exit_code': 0, 'natural_completion': True,
              'execution_receipt_sha256': worker.sha(tmp_path/'execution_receipt.json')}
    path = tmp_path/'launch_receipt.json'
    path.write_text(json.dumps(launch))
    assert worker.natural(tmp_path) == execution
    for field, value in [('exit_code', 1), ('natural_completion', False),
                         ('plan_sha256', 'f'*64), ('execution_receipt_sha256', 'f'*64)]:
        path.write_text(json.dumps({**launch, field: value}))
        with pytest.raises(ValueError):
            worker.natural(tmp_path)


def test_fixed_render_budget_and_no_adoption_gate():
    assert worker.SPEC['source_depth_calls'] == 350*2
    assert worker.SPEC['target_calls'] == len(worker.NAMES)*2
    assert worker.SPEC['native_arrays'] == len(worker.NAMES)*2*2
    assert worker.SPEC['cached_target_loads'] == len(worker.NAMES)
    assert worker.SPEC['VAL_reads'] == 0
    assert 'No gate' in worker.SPEC['selection']


def test_source_named_inside_cache_directory_is_still_bound(tmp_path):
    folder = tmp_path/'official/scene/__pycache__'
    folder.mkdir(parents=True)
    source = folder/'__init__.py'
    source.write_text('# An explicitly bound source remains a source.\n')
    (folder/'compiled.pyc').write_bytes(b'ignored bytecode')
    assert worker.tree(tmp_path) == {'official/scene/__pycache__/__init__.py': worker.sha(source)}
