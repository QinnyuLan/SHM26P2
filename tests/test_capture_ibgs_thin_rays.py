"""Capture plumbing only; no renderer/model/pixels/GPU are accessed."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
WORKER = HERE/'capture_ibgs_thin_rays.py'
if not WORKER.exists():
    WORKER = HERE.parent/'scripts/capture_ibgs_thin_rays.py'
spec = importlib.util.spec_from_file_location('thin_capture_test', WORKER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_observer_returns_original_tuple_and_arguments_unchanged_even_on_failure():
    args = tuple(object() for _ in range(29))
    result = tuple(object() for _ in range(13)); seen = []
    def original(*actual):
        assert all(a is b for a, b in zip(actual, args, strict=True))
        return result
    backend = SimpleNamespace(rasterize_gaussians=original)
    with m.observing_call(backend, lambda a, r: seen.append((a, r))):
        assert backend.rasterize_gaussians(*args) is result
    assert backend.rasterize_gaussians is original and seen[0][1] is result
    def bad(*_):
        raise RuntimeError('capture failure')
    with pytest.raises(RuntimeError), m.observing_call(backend, bad):
        backend.rasterize_gaussians(*args)
    assert backend.rasterize_gaussians is original


def test_absolute_address_prefix_abi_and_budget_are_fixed():
    sizes = m.prefix_lengths(2, 2, 16, 16, {'geometry': 3, 'binning': 255, 'image': 17})
    # Last field starts at absolute128 multiples; relative end depends on base.
    assert sizes == {'geometry': 1029, 'binning': 401, 'image': 7279}
    assert m.SPEC['rays'] == 16*512*9
    assert m.SPEC['capture_calls'] == 16 and m.SPEC['source_depth_renders'] == 0
    assert m.SPEC['summary_atol'] == 2e-5 and m.SPEC['summary_rtol'] == 2e-4
    with pytest.raises(ValueError):
        m.prefix_lengths(2, -1, 16, 16, {'geometry': 0, 'binning': 0, 'image': 0})


def test_validity_loader_opens_only_valid_member(monkeypatch):
    valid = np.ones((2, 3), bool); accessed = []
    class Lazy:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def __getitem__(self, key):
            accessed.append(key)
            assert key == 'valid', 'RGB payload must never be loaded'
            return valid
    monkeypatch.setattr(m.np, 'load', lambda *a, **k: Lazy())
    assert m.load_validity('synthetic.npz', (2, 3)) is valid and accessed == ['valid']


def test_natural_failure_rejected_before_endpoint_or_arrays(tmp_path):
    (tmp_path/'execution_receipt.json').write_text(json.dumps({'status': 'failed'}))
    (tmp_path/'launch_receipt.json').write_text(json.dumps({'status': 'failed', 'exit_code': 1}))
    with pytest.raises(ValueError, match='natural producer'):
        m.natural(tmp_path)


def test_ragged_traces_preserve_order_and_nonfinite_is_not_silently_consistent():
    rows = []
    for n in (2, 0, 1):
        row = {k: np.arange(n, dtype=np.float32) for k in ('alpha', 'T', 'w', 'z', 'center_depth')}
        row.update(ids=np.arange(n), order=np.arange(1, n+1, dtype=np.uint32),
                   median_mask=np.ones(n, bool), top4_mask=np.zeros(n, bool),
                   xy=np.array([n, 0], np.int32), median_slots=np.full(4, -1, np.int64))
        rows.append(row)
    saved = m.pack_traces(rows)
    assert saved['offsets'].tolist() == [0, 2, 2, 3] and saved['order'].tolist() == [1, 2, 1]
    assert all(a.dtype != object for a in saved.values())
    assert json.loads(json.dumps(m.safe_json({'status': 'nonfinite_cpu_replay', 'z': np.float32(np.inf)}),
                                allow_nan=False)) == {'status': 'nonfinite_cpu_replay', 'z': None}
