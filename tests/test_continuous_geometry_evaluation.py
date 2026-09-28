"""Synthetic endpoint, barrier and restoration contracts; no model or real pixels."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

PATH = Path(__file__).with_name('evaluate_continuous_geometry_proposals.py')
if not PATH.exists():
    PATH = Path(__file__).resolve().parents[1]/'scripts/evaluate_continuous_geometry_proposals.py'
spec = importlib.util.spec_from_file_location('geometry_endpoint_evaluation_test', PATH)
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


def endpoint():
    base = np.array([[1., 2., 3.], [4., 5., 6.]], np.float32)
    final = base+np.float32(.0001)
    arrays = {'base_means': base, 'means_final': final,
              'actual_delta': final.astype(np.float64)-base.astype(np.float64)}
    receipt = {'base_means_sha256': r.array_sha(base), 'final_means_sha256': r.array_sha(final)}
    return base, arrays, receipt


def test_endpoint_requires_actual_fp32_delta_and_exact_baseline_row_order():
    base, arrays, receipt = endpoint()
    assert np.array_equal(r.validate_endpoint(arrays, base, receipt), arrays['means_final'])
    with pytest.raises(ValueError, match='exact base'):
        r.validate_endpoint(arrays, base[::-1].copy(), receipt)
    changed = copy.deepcopy(arrays); changed['actual_delta'][0, 0] += 1e-12
    with pytest.raises(ValueError, match='actual displacement'):
        r.validate_endpoint(changed, base, receipt)
    wrong = dict(receipt, final_means_sha256='0'*64)
    with pytest.raises(ValueError, match='tensor hash'):
        r.validate_endpoint(arrays, base, wrong)


def test_endpoint_rejects_nonfinite_precision_change_or_extra_state():
    base, arrays, receipt = endpoint()
    for key, value in [('means_final', arrays['means_final'].astype(np.float64)),
                       ('actual_delta', np.full((2, 3), np.nan))]:
        changed = dict(arrays); changed[key] = value
        with pytest.raises(ValueError, match='dtype/finiteness'):
            r.validate_endpoint(changed, base, receipt)
    with pytest.raises(ValueError, match='Unexpected endpoint'):
        r.validate_endpoint(dict(arrays, extra=base), base, receipt)


def population():
    views = [{'name': f'{i:03}.png'} for i in range(50)]
    records = [{'arm': a, 'name': v['name'], 'masks': {k: {} for k in r.KINDS},
                'soft': {k: {} for k in ('raw', 'scene', 'teacher')}} for a in r.ARMS for v in views]
    identity = [{'name': v['name'], 'canvas_rgb_exact': True, 'original_rgb_exact': True,
                 'teacher_soft_exact': True, 'mask_exact': dict.fromkeys(r.KINDS, True)} for v in views]
    return records, views, identity


def test_prediction_barrier_rejects_missing_duplicate_or_failed_baseline_before_GT():
    records, views, identity = population(); r.prediction_barrier(records, views, identity)
    for wrong in (records[:-1], records[:-1]+[records[0]]):
        with pytest.raises(ValueError, match='population'):
            r.prediction_barrier(wrong, views, identity)
    changed = copy.deepcopy(records); del changed[-1]['soft']['teacher']
    with pytest.raises(ValueError, match='soft evidence'):
        r.prediction_barrier(changed, views, identity)
    changed = copy.deepcopy(identity); changed[0]['mask_exact']['joint'] = False
    with pytest.raises(ValueError, match='Baseline identity'):
        r.prediction_barrier(records, views, changed)
    changed = copy.deepcopy(identity); changed[0]['teacher_soft_exact'] = False
    with pytest.raises(ValueError, match='Baseline identity'):
        r.prediction_barrier(records, views, changed)


def test_RGB_teacher_quantization_and_official_float_warp_are_distinct():
    canvas = np.repeat(np.array([[[.0019], [.0039]], [[.0019], [.0039]]], np.float32), 3, axis=-1)
    grid = np.array([[[.5, .5]]], np.float32)
    teacher, delivered = r.rendered_rgb_images(canvas, grid)
    expected = cv2.remap(canvas, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR)
    assert np.array_equal(delivered, (expected[..., ::-1]*255).round().astype(np.uint8))
    wrong = cv2.remap(teacher, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR)
    assert not np.array_equal(wrong, delivered)
    t, d = r.rendered_rgb_images(canvas, None)
    assert np.array_equal(t[..., ::-1], d) and np.array_equal(t, teacher)


def test_counted_calls_keep_argument_identity_and_restore_every_patch_on_exception():
    token = object()
    def first(value):
        assert value is token
        return value
    def second(value):
        raise RuntimeError('expected')
    a, b = SimpleNamespace(f=first), SimpleNamespace(f=second)
    counts = {'a': 0, 'b': 0}
    with pytest.raises(RuntimeError, match='expected'), r.counted_calls([(a, 'f', 'a'), (b, 'f', 'b')], counts):
        assert a.f(token) is token
        b.f(token)
    assert a.f is first and b.f is second and counts == {'a': 1, 'b': 1}


def test_fixed_protocol_and_strict_JSON_do_not_add_candidate_selection(tmp_path):
    assert r.ARMS == ('baseline', 'joint', 'pcgrad', 'trust')
    assert r.NUMERICS == {'cudnn_allow_tf32': True, 'matmul_allow_tf32': False,
                          'matmul_precision': 'highest', 'cudnn_benchmark': False}
    assert r.SPEC['adoption_gate'] is None and r.SPEC['head_adaptation_steps'] == 0
    assert r.SPEC['baseline_joint_miou'] == .9509720470648008
    path = tmp_path/'report.json'; r.write(path, {'specification': r.SPEC})
    assert json.loads(path.read_text())['specification'] == r.SPEC
    with pytest.raises(FileExistsError):
        r.write(path, {})
    bad = tmp_path/'bad.json'
    with pytest.raises(ValueError):
        r.write(bad, {'bad': float('nan')})
    assert not bad.exists()
