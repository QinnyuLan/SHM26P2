"""Input identity, supported sensor and source-selection isolation contracts."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('joint_export', ROOT/'scripts/render_ibgs_joint_bundle.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
helper = m.load(ROOT/'scripts/render_multifield_bundle.py', 'old_camera_validation')


def camera():
    return {'name': '001.png', 'width': 1320, 'height': 989,
            'K': [[925., 0, 660], [0, 925., 494.5], [0, 0, 1]],
            'w2c': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            'distortion': [.009, 0, 0, 0, 0]}


@pytest.mark.parametrize('key', ['rgb', 'mask', 'source_image_path', 'source_annotation_path'])
def test_no_target_payload_in_camera(tmp_path, key):
    c = camera(); c[key] = '/not/read'
    path = tmp_path/'c.json'; path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='accepts'):
        m.read_cameras(path, {'sensor': camera()}, helper)


@pytest.mark.parametrize('key', ['K', 'width', 'distortion'])
def test_unsupported_sensor_rejected(tmp_path, key):
    c = camera()
    if key == 'K':
        c[key][0][0] += 1
    elif key == 'distortion':
        c[key][0] += .001
    else:
        c[key] += 1
    path = tmp_path/'c.json'; path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='sensor'):
        m.read_cameras(path, {'sensor': camera()}, helper)


def test_target_name_cannot_exclude_source():
    rows = [{'name': '001.png'}, {'name': '365.png'}]
    def neighbors(c, train):
        return [i for i, row in enumerate(train) if row['name'] != c['name']]
    original = camera(); renamed = {**original, 'name': '365.png'}
    assert m.pose_sources(original, rows, neighbors) == m.pose_sources(renamed, rows, neighbors) == ['001.png', '365.png']


def test_no_geometric_neighbors_fails_explicitly():
    with pytest.raises(ValueError, match='support region'):
        m.pose_sources(camera(), [], lambda *_: [])


def test_native_barrier_checks_identity_not_only_count():
    records = [{'arm': 'top4_normalized', 'name': 'a.png'}, {'arm': 'top4_normalized', 'name': 'b.png'}]
    m.native_barrier(records, ['a.png', 'b.png'])
    bad = copy.deepcopy(records); bad[1]['name'] = 'a.png'
    with pytest.raises(ValueError):
        m.native_barrier(bad, ['a.png', 'b.png'])


def test_exact_legacy_body_only_two_control_flow_changes():
    path = Path('/mnt/data/SHM2026/runs/ibgs_layer_heads_evaluation_v1/source_snapshot/evaluate_ibgs_layer_heads.py')
    original = path.read_text(); adapted = m.adapt_layer_worker(original)
    for old, new in reversed(m.ADAPTATIONS):
        adapted = adapted.replace(new, old)
    assert adapted == original
    with pytest.raises(ValueError):
        m.adapt_layer_worker(original+'\n')
