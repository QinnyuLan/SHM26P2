"""CPU contracts only; no endpoint, target, or GPU access."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ENTRY = HERE/'evaluate_ibgs_warm.py'
if not ENTRY.exists():
    ENTRY = HERE.parent/'scripts/evaluate_ibgs_warm.py'
spec = importlib.util.spec_from_file_location('ibgs_eval_contract', ENTRY)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def camera(name, center):
    pose = np.eye(4)
    pose[:3, 3] = -np.asarray(center)
    return {'name': name, 'w2c': pose.tolist(), 'w2c_original': pose.tolist()}


def test_sources_are_strict_train_geometry_not_name_or_input_order():
    target = camera('same', [0, 0, 0])
    rows = [camera('z', [.2, 0, 0]), camera('same', [.1, 0, 0]),
            camera('a', [-.2, 0, 0]), camera('too_close', [.01, 0, 0]),
            camera('too_far', [1.5, 0, 0]), camera('c', [.3, 0, 0]),
            camera('d', [.4, 0, 0]), camera('e', [.5, 0, 0])]
    assert [rows[i]['name'] for i in worker.neighbors(target, rows)] == ['a', 'z', 'c', 'd']
    rows[0]['w2c_original'][2][:3] = [1, 0, 0]
    rows[0]['w2c_original'][0][:3] = [0, 0, -1]
    assert 'z' not in [rows[i]['name'] for i in worker.neighbors(target, rows)]


def predictions():
    return [{'arm': a, 'readout': r, 'name': f'{i:03}.png'}
            for a in worker.ARMS for r in worker.READOUTS for i in range(50)]


def test_all_four_native_and_png_endpoints_required():
    rows = predictions()
    worker.prediction_barrier(rows)
    with pytest.raises(ValueError):
        worker.prediction_barrier(rows[:-1])
    rows[-1] = dict(rows[-2])
    with pytest.raises(ValueError):
        worker.prediction_barrier(rows)


def test_population_cannot_differ_between_arms():
    rows = predictions()
    rows[-1]['name'] = 'different.png'
    with pytest.raises(ValueError, match='populations'):
        worker.prediction_barrier(rows)


def test_natural_completion_and_receipt_identity_are_mandatory(tmp_path):
    execution = tmp_path/'execution_receipt.json'
    execution.write_text(json.dumps({'status': 'completed'}))
    launch = {'status': 'completed', 'exit_code': 0, 'natural_completion': False,
              'execution_receipt_sha256': worker.sha(execution)}
    (tmp_path/'launch_receipt.json').write_text(json.dumps(launch))
    with pytest.raises(ValueError, match='Natural'):
        worker.natural(tmp_path)
    launch['natural_completion'] = True
    (tmp_path/'launch_receipt.json').write_text(json.dumps(launch))
    assert worker.natural(tmp_path)['status'] == 'completed'
    execution.write_text(json.dumps({'status': 'completed', 'changed': True}))
    with pytest.raises(ValueError, match='Changed bound'):
        worker.natural(tmp_path)


def test_no_overwrite_and_fixed_math_contract(tmp_path):
    path = tmp_path/'report.json'
    worker.write(path, {'valid': True})
    with pytest.raises(FileExistsError):
        worker.write(path, {'valid': False})
    assert worker.SPEC['primary_candidate'] == 'full/fused'
    assert worker.SPEC['source_depth_calls'] == 2*350
    assert worker.SPEC['target_calls'] == 2*50
    assert worker.SPEC['native_arrays'] == 2*2*50
    assert worker.SPEC['bootstrap_repeats'] == 5000
    assert worker.SPEC['bootstrap_seed'] == 20260926
