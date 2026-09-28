"""FP64 evidence is distinct; prior FP32 passing-looking fixtures cannot authorize it."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runner = load('fullbatch_fp64', ROOT/'scripts/run_fullbatch_appearance_fp64.py')
v2 = load('fullbatch_v2_contracts', ROOT/'tests/test_fullbatch_appearance_v2_gate.py')
producer = load('fp64_diagnostic', ROOT/'scripts/audit_h3_fullbatch_objective_fp64.py')


def evidence():
    receipt, plan = v2.h3_evidence()
    plan['specification'] = copy.deepcopy(producer.SPEC)
    key = 'bridge_rgs/fullbatch_appearance_fp64.py'
    plan['source_hashes'][key] = runner.FP64_HELPER_SHA
    receipt['source_hashes'][key] = runner.FP64_HELPER_SHA
    receipt['actual_imports']['bridge_rgs.fullbatch_appearance_fp64'] = {
        'path': '/snapshot/'+key, 'sha256': runner.FP64_HELPER_SHA}
    receipt.update(objective_dtype='float64', parameter_renderer_warp_dtype='float32',
                   prior_numerical_gate_failure_preserved=True)
    for view in receipt['views']:
        for group in view['finite_differences'].values():
            for entry in group['epsilons']:
                for direction in ('plus', 'minus'):
                    entry[direction+'_fp64_loss'] = entry.pop(direction+'_fp32_loss')
        for point in ('before', 'after'):
            view['temporary_rms_step'][point+'_fp64_loss'] = view['temporary_rms_step'].pop(point+'_fp32_loss')
    return receipt, plan


def test_numeric_solver_protocol_unchanged_except_explicit_image_loss_dtype():
    current, previous = dict(runner.SPEC), dict(v2.runner.SPEC)
    assert current.pop('objective_dtype') == 'float64'
    assert current.pop('parameter_renderer_warp_dtype') == 'float32'
    assert current.pop('protocol') == 'conditional_fullbatch_appearance_h3_fp64_v1'
    previous.pop('protocol')
    assert current.pop('objective') == previous.pop('objective').replace('mean: .8', 'mean: FP64 .8')
    assert current == previous
    assert runner.sha(ROOT/'src/bridge_rgs/fullbatch_appearance_fp64.py') == runner.FP64_HELPER_SHA
    assert runner.sha(ROOT/'src/bridge_rgs/fullbatch_appearance.py') == runner.HELPER_SHA
    assert runner.sha(ROOT/'src/bridge_rgs/losses.py') == runner.LOSS_SHA
    assert hashlib.sha256(json.dumps(producer.SPEC, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == runner.H3_SPEC_SHA


def test_actual_fp64_producer_schema_is_accepted_but_old_schema_rejected():
    receipt, plan = evidence()
    runner.validate_h3_diagnostic(receipt, plan, 'test-plan')
    old_receipt, old_plan = v2.h3_evidence()
    with pytest.raises(ValueError, match='specification'):
        runner.validate_h3_diagnostic(old_receipt, old_plan, 'test-plan')
    with pytest.raises(ValueError, match='specification'):
        v2.runner.validate_h3_diagnostic(receipt, plan, 'test-plan')


@pytest.mark.parametrize('failure', ['wrong-dtype', 'no-preserved-failure', 'changed-helper',
                                     'unfrozen-import', 'numeric-failure', 'failed-run', 'bad-rms'])
def test_new_precision_gate_is_not_a_dtype_label_bypass(failure):
    receipt, plan = evidence()
    if failure == 'wrong-dtype':
        receipt['objective_dtype'] = 'float32'
    elif failure == 'no-preserved-failure':
        receipt.pop('prior_numerical_gate_failure_preserved')
    elif failure == 'changed-helper':
        plan['source_hashes']['bridge_rgs/fullbatch_appearance_fp64.py'] = 'changed'
    elif failure == 'unfrozen-import':
        receipt['actual_imports']['bridge_rgs.fullbatch_appearance_fp64']['path'] = '/main/helper.py'
    elif failure == 'numeric-failure':
        receipt['views'][0]['finite_differences']['background_logits']['epsilons'][0]['realized_relative_discrepancy'] = .0568
    elif failure == 'failed-run':
        receipt['status'] = 'failed'
    else:
        receipt['views'][1]['temporary_rms_step']['actual_to_predicted_decrease_ratio'] = .8
    with pytest.raises(ValueError):
        runner.validate_h3_diagnostic(receipt, plan, 'test-plan')


def test_same_bounded_launch_and_preparation_only_default():
    command = runner.execution_contract(Path('/root'), Path('/snapshot'), Path('/plan.json'))
    assert command['external_timeout_seconds'] == 600
    assert command['command'][:4] == ['timeout', '--signal=TERM', '--kill-after=5s', '600s']
    assert command['command'][-3:] == ['/snapshot/run_fullbatch_appearance_fp64.py', '--run', '/plan.json']
