"""Small CPU failure contracts for the fixed evaluation orchestration only."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1]/'scripts/execute_teacher_capacity_evaluation.py'
spec = importlib.util.spec_from_file_location('capacity_evaluation_runner', SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def endpoint(stage='real'):
    steps = 6000 if stage == 'real' else 2000
    own = {'last_checkpoint': '/runs/hplus/real/last.pt', 'last_checkpoint_sha256': 'realhash'}
    config = {'steps': steps, 'eval_every': steps, 'independent_augmentation_rng': True,
              'warmstart_checkpoint': own['last_checkpoint'] if stage == 'render_adapt' else ''}
    specification = {'output_dir': '/runs/hplus/'+stage, 'config': config}
    draft = {'source_review_sha256': {'src/bridge_rgs/teacher.py': 'source'}}
    launch = {'draft_plan_sha256': 'plan', 'stage_runner_sha256': 'runner'}
    receipt = {'status': 'completed', 'steps': steps, 'capacity': 'hplus', 'stage': stage,
               'configuration': copy.deepcopy(specification), 'plan_sha256': 'plan',
               'runner_sha256': 'runner', 'source_sha256': draft['source_review_sha256'],
               'last_checkpoint': specification['output_dir']+'/last.pt', 'result': {'step': steps},
               'warmstart': {'path': own['last_checkpoint'], 'sha256': 'realhash', 'step': 6000}
               if stage == 'render_adapt' else None}
    return receipt, specification, 'hplus_'+stage, draft, launch, own


def test_sequence_stops_before_any_checkpoint_or_subprocess(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, 'read', lambda path: {'status': 'running'})
    with pytest.raises(ValueError, match='four training stages'):
        audit.training_ready({'training_launch_manifest': {'sha256': 'launch'}},
                             {'output_root': str(tmp_path)}, object())


@pytest.mark.parametrize('mutation', ['steps', 'best', 'source', 'config'])
def test_endpoint_rejects_nonfixed_or_mismatched_records(mutation):
    receipt, specification, key, draft, launch, own = endpoint()
    if mutation == 'steps':
        receipt['steps'] = 5000
    elif mutation == 'best':
        receipt['last_checkpoint'] = '/runs/hplus/real/best.pt'
    elif mutation == 'source':
        receipt['runner_sha256'] = 'other'
    else:
        receipt['configuration']['config']['steps'] = 1
    with pytest.raises(ValueError):
        audit.endpoint_contract(receipt, specification, key, draft, launch, own)


def test_adapt_own_last_warmstart_required():
    values = endpoint('render_adapt')
    audit.endpoint_contract(*values)
    values[0]['warmstart']['path'] = '/runs/vit7b/real/last.pt'
    with pytest.raises(ValueError, match='own fixed'):
        audit.endpoint_contract(*values)


def test_rgb_reads_actual_all50_bytes_and_rejects_changes(tmp_path):
    rows = []
    for i in range(50):
        path = tmp_path/f'{i:03}.png'
        path.write_bytes(bytes([i]))
        rows.append({'name': path.name, 'rgb': str(path), 'rgb_sha256': audit.sha(path)})
    metrics = {'views': [{'name': row['name']} for row in rows]}
    assert len(audit.verify_rgb_files({'predictions': rows}, metrics)) == 50
    Path(rows[0]['rgb']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='Actual RGB'):
        audit.verify_rgb_files({'predictions': rows}, metrics)
    with pytest.raises(ValueError, match='all 50'):
        audit.verify_rgb_files({'predictions': rows[:-1]}, metrics)


def test_result_fingerprint_must_match_locked_plan():
    compare = SimpleNamespace(read_completed=lambda path: ({'official_evaluation_fingerprint': 'wrong'}, {}),
                              _validate=lambda metrics: None)
    with pytest.raises(ValueError, match='fingerprint'):
        audit.checked_result(Path('unused'), {'official_evaluation_fingerprint': 'fixed'}, compare)


def test_failed_attempt_persists_and_cannot_retry(monkeypatch, tmp_path):
    plan = tmp_path/'plan.json'
    plan.write_text('{}')
    monkeypatch.setattr(audit, 'static_plan', lambda path: (_ for _ in ()).throw(ValueError('bad fixed plan')))
    monkeypatch.setattr(audit.subprocess, 'run', lambda *a, **k: pytest.fail('Must not launch'))
    monkeypatch.setattr(audit.subprocess, 'check_output', lambda *a, **k: pytest.fail('Must not query GPU'))
    with pytest.raises(ValueError, match='bad fixed plan'):
        audit.execute(plan)
    receipt = json.loads((tmp_path/'execution/execution_receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['commands'] == []
    assert receipt['runner_sha256'] == audit.sha(SOURCE)
    with pytest.raises(ValueError, match='no retry'):
        audit.execute(plan)


def test_only_original_plan_bytes_accepted(tmp_path):
    plan = tmp_path/'plan.json'
    plan.write_text('{}')
    with pytest.raises(ValueError, match='immutable evaluation plan'):
        audit.static_plan(plan)
