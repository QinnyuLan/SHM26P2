"""Camera-order reproducibility and fail-closed preflight completion checks."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from bridge_rgs.ibgs_warm_training import camera_order

spec = importlib.util.spec_from_file_location('layer_head_worker', Path(__file__).parents[1]/'scripts/train_ibgs_layer_heads.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_order_matches_existing_fixed_training_order_regardless_of_input_order():
    names = [f'{i:03d}.png' for i in range(350)]
    assert worker.matched_order(names[::-1], 'train') == camera_order(names)
    assert worker.matched_order(names, 'preflight') == ['002.png', '041.png']
    with pytest.raises(ValueError):
        worker.matched_order(names[:-1]+[names[0]], 'train')


def test_success_receipt_without_bound_natural_exit_is_rejected(tmp_path):
    (tmp_path/'plan.json').write_text('{}')
    plan_sha = worker.sha(tmp_path/'plan.json')
    (tmp_path/'execution_receipt.json').write_text(json.dumps({'status': 'completed', 'plan_sha256': plan_sha}))
    launch = {'status': 'completed', 'exit_code': 124, 'natural_completion': False,
              'plan_sha256': plan_sha, 'execution_receipt_sha256': worker.sha(tmp_path/'execution_receipt.json')}
    (tmp_path/'launch_receipt.json').write_text(json.dumps(launch))
    with pytest.raises(ValueError):
        worker.completed(tmp_path)
    launch.update(exit_code=0, natural_completion=True)
    (tmp_path/'launch_receipt.json').write_text(json.dumps(launch))
    assert worker.completed(tmp_path)['status'] == 'completed'
    (tmp_path/'execution_receipt.json').write_text(json.dumps({'status': 'completed', 'plan_sha256': 'changed'}))
    with pytest.raises(ValueError):
        worker.completed(tmp_path)


def test_interleaved_selector_modes_are_materialized_without_changing_selection():
    ids = torch.arange(2*3*2*4, dtype=torch.int32).reshape(2, 3, 2, 4)
    package = {'top4': {'ids': ids[:, :, 1], 'depth': ids[:, :, 1].float(),
                        'weights': ids[:, :, 1].float()/1000}}
    assert not package['top4']['ids'].is_contiguous()
    result = worker.evidence_slots(package, 'top4')
    assert all(v.is_contiguous() for v in result.values())
    assert all(torch.equal(result[k], v) for k, v in package['top4'].items())
