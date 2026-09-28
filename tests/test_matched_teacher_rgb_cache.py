"""CPU-only contracts for fixed TRAIN camera cache generation; no model/GT pixels."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
SCRIPT = HERE.parent / 'prepare_matched_teacher_rgb_cache.py'
if not SCRIPT.exists():
    SCRIPT = HERE.parents[1] / 'scripts/prepare_matched_teacher_rgb_cache.py'
spec = importlib.util.spec_from_file_location('matched_teacher_cache_tested', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def manifest():
    original = np.eye(4).tolist()
    optimized = np.eye(4)
    optimized[0, 3] = 123.
    train = [{'name': f'{i + 1:03d}.png', 'image_id': i + 1, 'camera_id': 1, 'split': 'train',
              'image_path': f'/not-opened/rgb/{i}.png',
              'mask_path': f'/not-opened/mask/{i}.png' if i < 259 else None,
              'valid_path': f'/not-opened/valid/{i}.png',
              'source_annotation_path': f'/not-opened/json/{i}.json',
              'K': np.eye(3).tolist(), 'width': 10, 'height': 10,
              'w2c': optimized.tolist(), 'w2c_original': copy.deepcopy(original)}
             for i in range(350)]
    # A deliberately malformed VAL camera must never enter the TRAIN converter.
    val = [{'name': 'val.png', 'image_id': 999, 'camera_id': 999, 'split': 'val',
            'mask_path': '/not-opened/val-mask', 'w2c_original': 'poison'}]
    source = {'K': [[925., 0., 660.], [0., 925., 494.5], [0., 0., 1.]],
              'width': 1320, 'height': 989, 'opencv_distortion': [.009, 0., 0., 0., 0.]}
    return {'views': list(reversed(train)) + val, 'source_cameras': {'1': source}}


def test_train_filter_original_camera_schema_and_no_pixel_io(monkeypatch):
    from bridge_rgs.official_evaluate import CAMERA_KEYS
    data = manifest()
    pristine = copy.deepcopy(data)

    def forbidden(*args, **kwargs):
        raise AssertionError('Camera metadata conversion must not open any payload')

    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    monkeypatch.setattr(Path, 'read_text', forbidden)
    monkeypatch.setattr(Path, 'open', forbidden)
    got = m.camera_records(data)
    assert len(got) == 350 and [v['name'] for v in got] == [f'{i:03d}.png' for i in range(1, 351)]
    assert all(set(v) == CAMERA_KEYS and v['split'] == 'train' for v in got)
    assert got[0]['K'] == data['source_cameras']['1']['K']
    assert (got[0]['width'], got[0]['height']) == (1320, 989)
    assert got[0]['w2c'] == np.eye(4).tolist()
    assert data == pristine
    got[0]['K'][0][0] = -1.
    assert data == pristine  # Deepcopy, not source-camera aliasing.


@pytest.mark.parametrize('case', ['wrong_count', 'wrong_annotation_count', 'duplicate_name'])
def test_train_population_rejects_incomplete_or_duplicate(case):
    data = manifest()
    train = [v for v in data['views'] if v['split'] == 'train']
    if case == 'wrong_count':
        data['views'].remove(train[0])
    elif case == 'wrong_annotation_count':
        next(v for v in train if v['mask_path'])['mask_path'] = None
    else:
        train[0]['name'] = train[1]['name']
    with pytest.raises(ValueError, match='Expected fixed|Duplicate TRAIN'):
        m.camera_records(data)


def test_all_uint8_pairs_average_equals_integer_half_even():
    x, y = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8))
    first = np.repeat(x[..., None], 3, -1)
    second = np.repeat(y[..., None], 3, -1)
    total = first.astype(np.uint16) + second.astype(np.uint16)
    floor = total // 2
    expected = (floor + ((total % 2 == 1) & (floor % 2 == 1))).astype(np.uint8)
    actual = m.average_rgb(first, second)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError):
        m.average_rgb(first.astype(np.float32), second)
    with pytest.raises(ValueError):
        m.average_rgb(first[:-1], second)


def test_adapter_canvas_rejects_changed_annotation_intrinsics_or_size():
    grid = manifest()['source_cameras']['1']
    info = {'canvas': [grid['width'], grid['height']],
            'canvas_K': np.asarray(grid['K'], np.float32).tolist()}
    m.verify_canvas_grid(info, grid)
    shifted = copy.deepcopy(info)
    shifted['canvas_K'][0][2] += .5
    with pytest.raises(ValueError, match='unchanged legacy annotation'):
        m.verify_canvas_grid(shifted, grid)
    with pytest.raises(ValueError, match='unchanged legacy annotation'):
        m.verify_canvas_grid({**info, 'canvas': [1321, 989]}, grid)


def test_source_inventory_detects_edit_add_delete_and_ignores_bytecode(tmp_path):
    package = tmp_path / 'bridge_rgs'
    package.mkdir()
    path = package / 'model.py'
    path.write_text('MODEL = 1\n')
    baseline = m.source_hashes(tmp_path)
    assert list(baseline) == ['bridge_rgs/model.py']
    (package / 'model.pyc').write_bytes(b'not part of python source inventory')
    assert m.source_hashes(tmp_path) == baseline
    path.write_text('MODEL = 2\n')
    assert m.source_hashes(tmp_path) != baseline
    path.write_text('MODEL = 1\n')
    assert m.source_hashes(tmp_path) == baseline
    extra = package / 'extra.py'
    extra.write_text('EXTRA = 0\n')
    assert m.source_hashes(tmp_path) != baseline
    extra.unlink()
    path.unlink()
    assert m.source_hashes(tmp_path) == {}


def test_execute_rejects_changed_source_before_any_gpu_or_receipt(tmp_path, monkeypatch):
    snapshot = tmp_path / 'source_snapshot'
    snapshot.mkdir()
    source = snapshot / 'probe.py'
    source.write_text('ORIGINAL = True\n')
    hashes = m.source_hashes(snapshot)
    source.write_text('ORIGINAL = False\n')
    plan = {'output': str(tmp_path), 'source_snapshot': str(snapshot),
            'source_hashes': hashes, 'specification': m.SPEC}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))

    def forbidden(*args, **kwargs):
        raise AssertionError('Corrupted source must fail before GPU allocation')

    monkeypatch.setattr(m.torch.cuda, 'init', forbidden)
    monkeypatch.setattr(m.torch.cuda, '_lazy_init', forbidden)
    with pytest.raises(ValueError, match='Changed source/spec'):
        m.execute(path)
    assert not (tmp_path / 'execution_receipt.json').exists()
