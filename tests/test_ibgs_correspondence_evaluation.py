"""Inference intervention identity, complete population and primary comparison sign."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
ENTRY = HERE/'evaluate_ibgs_correspondence_control.py'
if not ENTRY.exists():
    ENTRY = HERE.parent/'scripts/evaluate_ibgs_correspondence_control.py'
spec = importlib.util.spec_from_file_location('correspondence_evaluation', ENTRY)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def base(tmp_path):
    original = (worker.PRIOR/'source_snapshot/evaluate_ibgs_layer_heads.py').read_text()
    adapted = worker.adapted_source(original); restored = adapted
    for before, after in reversed(worker.ADAPTATIONS):
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored == original
    (tmp_path/worker.ORIGINAL_NAME).write_text(original)
    (tmp_path/worker.BASE_NAME).write_text(adapted)
    return worker.profiled_base(tmp_path)


def test_single_arm_counts_and_barrier(tmp_path):
    m = base(tmp_path); names = [f'{i:03}.png' for i in range(50)]
    records = [{'arm': worker.ARM, 'name': n} for n in names]
    m.prediction_barrier(records, names)
    for bad in (records[:-1], records+[records[0]], records[:-1]+[{'arm': 'top4_normalized', 'name': names[-1]}]):
        with pytest.raises(ValueError):
            m.prediction_barrier(bad, names)
    assert m.SPEC['target_calls'] == m.SPEC['selector_calls'] == m.SPEC['lpips_calls'] == 50
    assert m.SPEC['primary_mechanism_comparison'] == 'top4_normalized_minus_top4_normalized_permuted'


def test_fixed_finite_matched_control_endpoint_required(tmp_path):
    m = base(tmp_path)
    parent = {'specification': {'train_steps': 6000}, 'checkpoint': {'sha256': 'original'}, 'cache_manifest_sha256': 'cache'}
    saved = {'protocol': 'ibgs_normalized_correspondence_control_v1', 'arm': worker.ARM, 'step': 6000,
             'plan_sha256': 'plan', 'specification': parent['specification'], 'field_checkpoint': parent['checkpoint'],
             'cache_manifest_sha256': 'cache', 'head': {'toy': torch.ones(2)}}
    m.validate_endpoint(saved, worker.ARM, parent, 'plan', parameter_count=2)
    for changed in ({'protocol': 'ibgs_fixed_layer_heads_v1'}, {'step': 2}, {'arm': 'top4_normalized'},
                    {'head': {'toy': torch.tensor([1., float('nan')])}}):
        with pytest.raises(ValueError):
            m.validate_endpoint({**saved, **changed}, worker.ARM, parent, 'plan', parameter_count=2)


def test_real_control_precedes_the_only_inference_forward(tmp_path):
    m = base(tmp_path); h, w = 2, 3; n = h*w
    f = torch.zeros(n, 4, 4, 7); f[:, 0, 0, 0] = 1; f[:, 1, 0, 0] = 4
    weights = torch.zeros(n, 4); weights[:, 0] = .3; weights[:, 1] = .1
    support = torch.zeros(n, 4, 4); support[:, :2, 0] = 1
    original = [t.clone() for t in (f, weights, support)]
    package = {'top4': {'ids': torch.zeros(h, w, 4, dtype=torch.int32),
                       'depth': torch.ones(h, w, 4), 'weights': weights.reshape(h, w, 4)},
               'raw': torch.zeros(3, h, w), 'ray': torch.ones(3, h, w),
               'metadata': {'focal': [2., 3.], 'principal': [1., .5]}}
    camera = SimpleNamespace(image_name='target', world_view_transform=torch.eye(4), camera_center=torch.zeros(3))
    calls = []
    def builder(*_args, **_kwargs):
        return {'features': f, 'target_weights': weights, 'support': support}
    def net(features, actual_w, actual_q, _ray, rgb):
        calls.append(1)
        assert (features[:, 0, 0, 0] == 4).all() and (features[:, 1, 0, 0] == 1).all()
        assert actual_w is weights and actual_q is support
        return {'image_pred': rgb+.2, 'active': torch.ones(n, dtype=torch.bool), 'supported_mass': torch.full((n,), .4)}
    result, stats = m.infer_layer_head(package, 'top4', net, camera, {'source': camera}, ['source'],
                                      lambda _: {}, {}, builder)
    assert calls == [1] and result.shape == (3, h, w)
    assert stats['permutation_diagnostics']['weighted_raw_sum_changed_groups'] == n
    for actual, expected in zip((f, weights, support), original, strict=True):
        assert torch.equal(actual, expected)


def test_primary_pair_direction_is_correct_minus_permuted(tmp_path):
    helper = worker.load(worker.PRIOR/'source_snapshot/rgb_scoring_helpers.py', 'fixed_pair_helper')
    def metric(offset):
        return {'views': [{'name': str(i), 'width': 20, 'height': 20, 'rgb_pixels': 400,
                           'psnr': 20.+i+offset, 'ssim': .5+offset*.01, 'lpips': .4-offset*.01} for i in range(50)]}
    descriptors = {}
    for role, offset in [('control', 0), ('correct', 2)]:
        p = tmp_path/f'{role}.json'; p.write_text(json.dumps(metric(offset)))
        descriptors[role] = {'path': str(p), 'sha256': worker.sha(p)}
    mock = SimpleNamespace(legacy=lambda: SimpleNamespace(scoring_modules=lambda _: (None, helper)))
    report = {'metrics': {worker.ARM: descriptors['control']}}
    plan = {'source_snapshot': str(tmp_path), 'output': str(tmp_path),
            'reference_metrics': {'top4_normalized': descriptors['correct']}}
    worker.add_mechanism_result(mock, plan, report)
    pair = worker.read(report['primary_mechanism_comparison']['path'])
    assert pair['metrics']['psnr']['difference'] == 2
    assert pair['metrics']['psnr']['paired_view_bootstrap_95_interval'] == [2., 2.]
    assert pair['metrics']['lpips']['difference'] < 0
