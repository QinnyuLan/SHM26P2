"""CPU-only official capacity gates; never load a running checkpoint."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('capacity_evaluation', ROOT/'scripts/evaluate_rgb_capacity_reference.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def receipts():
    common = {'status': 'completed', 'plan_sha256': runner.TRAIN_PLAN_SHA}
    return {'plan.json': {'training_output': '/not-readable/training'},
            'launch_receipt.json': dict(common, exit_code=0),
            'train_receipt.json': dict(common), 'native_receipt.json': dict(common),
            'cpu_endpoint_audit.json': dict(common, status='passed')}


@pytest.mark.parametrize('filename,key,value', [
    ('launch_receipt.json', 'status', 'running'),
    ('launch_receipt.json', 'exit_code', 124),
    ('native_receipt.json', 'status', 'failed'),
    ('cpu_endpoint_audit.json', 'status', 'failed'),
    ('cpu_endpoint_audit.json', 'plan_sha256', 'different'),
])
def test_unfinished_or_unbound_endpoint_never_hashes_checkpoint(monkeypatch, filename, key, value):
    values = receipts()
    values[filename][key] = value
    hashed = []

    def guarded_sha(path):
        hashed.append(Path(path).name)
        assert Path(path) == runner.RUN/'plan.json', 'Active checkpoint accessed before endpoint guard'
        return runner.TRAIN_PLAN_SHA

    monkeypatch.setattr(runner, 'read', lambda path: values[Path(path).name])
    monkeypatch.setattr(runner, 'sha', guarded_sha)
    with pytest.raises(ValueError):
        runner.completed_endpoint()
    assert hashed == ['plan.json']


def test_gate_fixed_candidate_minus_reference_boundaries_and_no_semantics():
    metrics = {'psnr': {'difference': .15, 'paired_view_bootstrap_95_interval': [1e-8, .3]},
               'ssim': {'difference': 0}, 'lpips': {'difference': 0}}
    assert all(runner.rgb_gate(metrics).values())
    for key, field, value in [('psnr', 'difference', .14999),
                               ('psnr', 'paired_view_bootstrap_95_interval', [0, .3]),
                               ('ssim', 'difference', -1e-8), ('lpips', 'difference', 1e-8)]:
        changed = copy.deepcopy(metrics)
        changed[key][field] = value
        assert not all(runner.rgb_gate(changed).values())
    with pytest.raises(ValueError, match='Only RGB'):
        runner.rgb_gate(dict(metrics, miou_all={'difference': .2}))


def test_existing_scoring_source_is_separate_and_exact():
    source, entry, compare = runner.scoring_sources()
    assert source != runner.RUN/'source_snapshot'
    assert (source/'bridge_rgs/official_evaluate.py').is_file()
    assert runner.sha(entry) == runner.ENTRY_SHA
    assert runner.sha(compare) == runner.COMPARE_SHA


def test_result_writes_cannot_replace_existing_artifacts(tmp_path):
    path = tmp_path/'result.json'
    runner.write(path, {'status': 'completed'})
    with pytest.raises(FileExistsError):
        runner.write(path, {'status': 'failed'})
