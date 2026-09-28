"""Small CPU contracts; no real arrays, torch, model or renderer."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/check_ibgs_start_transfer.py'
if not SCRIPT.exists():
    SCRIPT = Path(__file__).with_name('check_ibgs_start_transfer.py')
spec = importlib.util.spec_from_file_location('ibgs_start_transfer', SCRIPT)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_metric_rgb_valid_denominator_and_no_implicit_clip():
    image = np.array([[[2., 1., 0.], [100., 100., 100.]]], np.float32)
    truth = np.zeros_like(image); valid = np.array([[True, False]])
    score = worker.metrics(image, truth, valid)
    assert score['valid_pixels'] == 1
    assert score['mse'] == pytest.approx(5/3)
    assert score['psnr'] == pytest.approx(-10*np.log10(5/3))
    assert worker.metrics(np.clip(image, 0, 1), truth, valid)['mse'] == pytest.approx(2/3)


def test_all_predictions_barrier_rejects_missing_repeated_or_unsaved():
    records = [{'name': name, 'path': name, 'sha256': 'a'*64} for name in worker.NAMES]
    worker.prediction_barrier(records)
    for bad in (records[:-1], records[:-1]+[records[0]], [dict(r, sha256='') for r in records]):
        with pytest.raises(ValueError, match='16 saved'):
            worker.prediction_barrier(bad)


def test_prepare_header_reader_does_not_deserialize_payload(tmp_path, monkeypatch):
    path = tmp_path/'cache.npz'
    np.savez(path, rgb=np.ones((2, 3, 3), np.float32), valid=np.ones((2, 3), bool))
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('Preparation cannot deserialize array payload'))
    assert worker.array_headers(path) == {'rgb': {'shape': [2, 3, 3], 'dtype': 'float32'},
                                          'valid': {'shape': [2, 3], 'dtype': 'bool'}}
