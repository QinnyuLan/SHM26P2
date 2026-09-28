import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
WORKER = HERE/'probe_ibgs_renderer_decomposition.py'
if not WORKER.exists():
    WORKER = HERE.parent/'scripts/probe_ibgs_renderer_decomposition.py'
spec = importlib.util.spec_from_file_location('renderer_decomposition_test_worker', WORKER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def original_kwargs():
    return {'rasterize_mode': 'antialiased', 'near_plane': .01, 'far_plane': 1e6,
            'packed': False, 'absgrad': False, 'render_mode': 'RGB+ED', 'sh_degree': 3,
            'means': object(), 'backgrounds': object()}


def test_only_fixed_renderer_arguments_change():
    original = original_kwargs()
    for key, condition in m.CONDITIONS.items():
        result = m.condition_kwargs(original, key)
        assert result == dict(original, **condition)
        assert result['means'] is original['means']
    assert original['rasterize_mode'] == 'antialiased'
    with pytest.raises(ValueError):
        m.condition_kwargs(dict(original, sh_degree=0), 'A')
    with pytest.raises(ValueError):
        m.condition_kwargs(original, 'I')


def test_hooks_restore_on_exception_and_count_actual_entry():
    calls = []
    low = lambda *a, **kw: calls.append(('low', kw))
    rendering = SimpleNamespace(rasterize_to_pixels=low)
    def raster(*a, **kw):
        calls.append(('high', kw)); rendering.rasterize_to_pixels()
    gsplat = SimpleNamespace(rasterization=raster)
    counts = {'high_raster': 0, 'low_raster': 0}
    with pytest.raises(RuntimeError), m.raster_hooks(gsplat, rendering, 'N', counts):
        gsplat.rasterization(**original_kwargs())
        raise RuntimeError('render failure')
    assert gsplat.rasterization is raster and rendering.rasterize_to_pixels is low
    assert counts == {'high_raster': 1, 'low_raster': 1}
    assert calls[0][1]['near_plane'] == .2 and calls[0][1]['rasterize_mode'] == 'classic'


def test_complete_prediction_barrier_is_fixed_not_merely_counted():
    records = [{'condition': c, 'name': n, 'path': f'{c}_{n}', 'sha256': 'bound'}
               for c in m.CONDITIONS for n in m.NAMES]
    m.prediction_barrier(records)
    for bad in (records[:-1], records[::-1], records[:-1]+[records[0]]):
        with pytest.raises(ValueError):
            m.prediction_barrier(bad)


def test_AA_identity_is_clipped_exact_and_failure_not_tolerated():
    a = np.array([[[-.1, .5, 1.1]]], dtype=np.float32)
    old = np.clip(a, 0, 1)
    assert m.identity(a, old) == {'clipped_array_exact': True, 'clipped_max_abs_difference': 0.}
    changed = a.copy(); changed[0, 0, 1] = np.nextafter(np.float32(.5), np.float32(1.))
    result = m.identity(changed, old)
    assert not result['clipped_array_exact'] and result['clipped_max_abs_difference'] > 0
    json.dumps(result, allow_nan=False)


def test_raw_clip_support_and_telescoping_strict_json(tmp_path):
    target = np.zeros((1, 2, 3), np.float32)
    valid = np.array([[True, False]])
    values = {}
    rows = {}
    for c, x in zip(('A', 'C', 'N', 'I'), (-.2, .3, 1.2, .9), strict=True):
        prediction = np.full_like(target, x); prediction[:, 1] = 10000
        row = m.metrics(prediction, target, valid); rows[c] = row
        values[c] = row['raw']['mse']
    assert rows['A']['clipped']['mse'] == 0 and rows['A']['raw']['mse'] > 0
    assert rows['N']['clipped']['mse'] == 1
    path = m.telescoping(values)
    assert abs(path['closure_residual']) < 1e-15
    m.write(tmp_path/'report.json', {'metrics': rows, 'path': path})
    with pytest.raises(ValueError):
        m.write(tmp_path/'invalid.json', {'bad': float('nan')})
    assert not (tmp_path/'invalid.json').exists()
    with pytest.raises(FileExistsError):
        m.write(tmp_path/'report.json', {})
    with pytest.raises(ValueError):
        m.metrics(target, target, np.zeros_like(valid))


def test_actual_camera_cast_and_fixed_train_population():
    K = np.array([[925.7016189245708, 0, 660], [0, 925.7016189245708, 494.5], [0, 0, 1.]])
    camera = {'name': m.NAMES[0], 'split': 'train', 'K': K.tolist(), 'w2c_original': np.eye(4).tolist()}
    k32, w32 = m.actual_camera(camera)
    assert k32.dtype == w32.dtype == np.float32 and np.array_equal(k32, K.astype(np.float32))
    assert float(k32[0, 0]) != K[0, 0]
    with pytest.raises(ValueError):
        m.actual_camera(dict(camera, split='val'))
