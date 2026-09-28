"""Single-arm reuse and paired factorial arithmetic without images or CUDA."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
ENTRY = HERE/'evaluate_ibgs_factorial_completion.py'
if not ENTRY.exists():
    ENTRY = HERE.parent/'scripts/evaluate_ibgs_factorial_completion.py'
spec = importlib.util.spec_from_file_location('factorial_evaluation', ENTRY)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def base(tmp_path):
    content = (worker.PRIOR/'source_snapshot/evaluate_ibgs_layer_heads.py').read_text()
    adapted = worker.adapted_source(content); restored = adapted
    for before, after in reversed(worker.ADAPTATIONS):
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored == content
    (tmp_path/worker.ORIGINAL_NAME).write_text(content)
    (tmp_path/worker.BASE_NAME).write_text(adapted)
    return worker.profiled_base(tmp_path)


def test_exact_reuse_and_single_arm_prediction_barrier(tmp_path):
    module = base(tmp_path); names = [f'{i:03}.png' for i in range(50)]
    records = [{'arm': worker.ARM, 'name': n} for n in names]
    module.prediction_barrier(records, names)
    for bad in (records[:-1], records+[records[0]], records[:-1]+[{'arm': 'top4_mass', 'name': names[-1]}]):
        with pytest.raises(ValueError):
            module.prediction_barrier(bad, names)
    assert module.SPEC['target_calls'] == module.SPEC['selector_calls'] == module.SPEC['lpips_calls'] == 50
    assert module.SPEC['primary_mechanism_comparison'] == 'top4_normalized_minus_median4_normalized'


def test_only_new_protocol_and_fixed_finite_endpoint_accepted(tmp_path):
    module = base(tmp_path)
    parent = {'specification': {'train_steps': 6000}, 'checkpoint': {'sha256': 'original'}, 'cache_manifest_sha256': 'cache'}
    saved = {'protocol': 'ibgs_fixed_layer_factorial_completion_v1', 'arm': worker.ARM, 'step': 6000,
             'plan_sha256': 'plan', 'specification': parent['specification'], 'field_checkpoint': parent['checkpoint'],
             'cache_manifest_sha256': 'cache', 'head': {'toy': torch.ones(2)}}
    module.validate_endpoint(saved, worker.ARM, parent, 'plan', parameter_count=2)
    for changed in ({'protocol': 'ibgs_fixed_layer_heads_v1'}, {'step': 5999}, {'arm': 'top4_normalized'},
                    {'head': {'toy': torch.tensor([1., float('nan')])}}):
        with pytest.raises(ValueError):
            module.validate_endpoint({**saved, **changed}, worker.ARM, parent, 'plan', parameter_count=2)


def toy_metrics():
    offsets = {'median4_mass': 0, 'top4_mass': 1, worker.ARM: 2, 'top4_normalized': 5}
    return {a: {'inherited_common_reference_fingerprint': 'same', 'scoring_protocol': {'fixed': True},
                'views': [{'name': str(i), 'width': 1320, 'height': 989, 'rgb_pixels': 1320*989,
                           'source_rgb_sha256': f'gt-{i}', **{k: float(i+offset) for k in worker.RGB_KEYS}}
                          for i in range(50)]} for a, offset in offsets.items()}


def test_paired_interaction_cancels_shared_view_shocks_and_has_declared_sign():
    result = worker.factorial_interaction(toy_metrics(), repeats=100)
    for metric in result['metrics'].values():
        assert metric['selection_effect_mass'] == 1
        assert metric['selection_effect_normalized'] == 3
        assert metric['interaction'] == 2 and metric['paired_view_bootstrap_95_interval'] == [2., 2.]


def test_interaction_rejects_different_gt_and_duplicate_view():
    metrics = toy_metrics(); changed = copy.deepcopy(metrics)
    changed['top4_mass']['views'][0]['source_rgb_sha256'] = 'other'
    with pytest.raises(ValueError, match='GT/grid'):
        worker.factorial_interaction(changed, repeats=10)
    changed = copy.deepcopy(metrics); changed['top4_mass']['views'][-1] = changed['top4_mass']['views'][0]
    with pytest.raises(ValueError, match='population'):
        worker.factorial_interaction(changed, repeats=10)
