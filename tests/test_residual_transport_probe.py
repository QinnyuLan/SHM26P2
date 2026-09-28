"""CPU-only camera selection, payload barrier and cross-fit contracts."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
WORKER = HERE/'probe_residual_transport.py'
if not WORKER.exists():
    WORKER = HERE.parent/'scripts/probe_residual_transport.py'
spec = importlib.util.spec_from_file_location('residual_probe_contract', WORKER)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def camera(name, x, angle=0):
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]]
    pose[:3, 3] = -pose[:3, :3]@np.asarray([x, 0, 0])
    return {'name': name, 'split': 'train', 'w2c_original': pose.tolist(),
            'K': [[123.456789, 0, 2.1], [0, 123.456789, 1.7], [0, 0, 1]],
            'image_path': '/synthetic/'+name, 'valid_path': '/synthetic/valid.png'}


def test_excludes_all_targets_and_next_nearest_deterministically():
    train = [camera(name, i*.01) for i, name in enumerate(probe.NAMES)]
    train += [camera(f'source{i}', i+1) for i in range(7)]
    result = probe.select_sources(train, probe.NAMES, 2.)
    assert result == probe.select_sources(train[::-1], probe.NAMES, 2.)
    assert [r['fold'] for r in result] == ['A', 'B']*8
    for row in result:
        assert row['excluded_nearest']['name'] == 'source0'
        assert [s['name'] for s in row['sources']] == [f'source{i}' for i in range(1, 5)]
        assert not set(probe.NAMES) & {s['name'] for s in row['sources']}


def test_camera_distance_uses_rotation_and_translation_and_ties_use_name():
    target = camera('target', 0)
    assert probe.camera_distance(target, camera('rotated', 0, np.pi/2), 1.) == pytest.approx(.25)
    assert probe.camera_distance(target, camera('translated', .5), 1.) == pytest.approx(.25)
    bank = [camera(n, 1) for n in ('z', 'b', 'a', 'e', 'c')]
    row = probe.select_sources([target]+bank, ['target'], 1.)[0]
    assert row['excluded_nearest']['name'] == 'a'
    assert [v['name'] for v in row['sources']] == ['b', 'c', 'e', 'z']


def test_insufficient_or_nontrain_sources_fail_without_backfill():
    train = [camera('target', 0)]+[camera(str(i), i+1) for i in range(4)]
    with pytest.raises(ValueError, match='Insufficient'):
        probe.select_sources(train, ['target'], 1.)
    train.append(dict(camera('val', 8), split='val'))
    with pytest.raises(ValueError, match='TRAIN only'):
        probe.select_sources(train, ['target'], 1.)


def test_actual_camera_is_renderer_fp32_not_original_double():
    view = camera('test', .123456789, .30123456789)
    K, pose = probe.actual_camera(view)
    assert K.dtype == pose.dtype == np.float32
    assert not np.array_equal(K.astype(np.float64), np.asarray(view['K']))
    np.testing.assert_array_equal(pose.astype(np.float64), np.asarray(view['w2c_original'], np.float32).astype(np.float64))


def test_target_rgb_role_and_barrier_before_even_hashing_payload():
    views = [camera(name, i) for i, name in enumerate(probe.NAMES)]+[camera('source', 0)]
    plan = {'views': views, 'source_names': ['source'],
            'pixel_input_hashes': {v[key]: 'inherited-no-read' for v in views for key in ('image_path', 'valid_path')}}
    records = [{'name': n, 'path': '/not-opened/'+n, 'sha256': 'recorded'} for n in probe.NAMES]
    path = views[0]['image_path']
    with pytest.raises(ValueError, match='Wrong RGB role'):
        probe.validate_pixel_access(path, True, False, plan, [])
    with pytest.raises(ValueError, match='All fixed 16'):
        probe.validate_pixel_access(path, True, True, plan, records[:-1])
    probe.validate_pixel_access(path, True, True, plan, records)
    probe.validate_pixel_access(views[-1]['image_path'], True, False, plan, [])
    probe.validate_pixel_access(views[0]['valid_path'], False, False, plan, [])


def test_cross_fit_uses_opposite_fold_for_both_arms_and_zero_is_fixed():
    rows = [{'name': n, 'fold': 'A' if i % 2 == 0 else 'B',
             'statistics': {'true': .2 if i % 2 == 0 else .8, 'wrong': .3 if i % 2 == 0 else .7}}
            for i, n in enumerate(probe.NAMES)]
    def fit(values):
        assert len(values) == 8
        return {'coefficient': float(np.mean(values))}
    fits = probe.cross_fit(rows, fit)
    for row in rows:
        coefficient, fold = probe.heldout_coefficient(row, 'true', fits)
        assert coefficient == pytest.approx(.8 if row['fold'] == 'A' else .2)
        assert fold != row['fold']
        assert probe.heldout_coefficient(row, 'zero', fits) == (0., None)
    rows[0]['fold'] = 'B'
    with pytest.raises(ValueError, match='alternating folds'):
        probe.cross_fit(rows, fit)


def test_strict_json_writes_atomically_and_refuses_overwrite(tmp_path):
    path = tmp_path/'record.json'
    with pytest.raises(ValueError):
        probe.write(path, {'bad': float('nan')})
    assert not path.exists()
    rows = probe.select_sources([camera('target', 0)]+[camera(str(i), i+1) for i in range(5)], ['target'], 1.)
    probe.write(path, {'selection': rows, 'camera': probe.actual_camera(camera('target', 0))[0].tolist()})
    assert json.loads(path.read_text())['selection'] == rows
    with pytest.raises(FileExistsError):
        probe.write(path, {})
