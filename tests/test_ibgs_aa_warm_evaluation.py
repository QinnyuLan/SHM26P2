"""Reject renderer/profile mismatches without loading a model or CUDA."""
import copy
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ENTRY = HERE/'evaluate_ibgs_aa_warm.py'
if not ENTRY.exists():
    ENTRY = HERE.parent/'scripts/evaluate_ibgs_aa_warm.py'
spec = importlib.util.spec_from_file_location('ibgs_aa_evaluation_contract', ENTRY)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def sample():
    profile = {'id': 'ibgs_centered_corner_v2_aa_factored64_near001_v2', 'near_plane': .01,
               'eps2d': .3, 'capture_last': False}
    plan = {'renderer_profile': profile, 'aa_module_sha256': 'module_digest',
            'binary': {'path': '/isolated/backend.so', 'sha256': 'backend_digest'},
            'aa_provider': worker.AA_PROVIDER, 'precision_policy': {'arithmetic': 'FP64 factor'},
            'specification': {'protocol': 'ibgs_aa_stable_matched_v2', 'steps_per_arm': 6000}}
    saved = {'protocol': 'ibgs_aa_stable_matched_v2', 'renderer_profile': copy.deepcopy(profile),
             'aa_module_sha256': plan['aa_module_sha256'], 'renderer_binary': plan['binary'],
             'specification': plan['specification'], 'aa_provider': plan['aa_provider'],
             'precision_policy': copy.deepcopy(plan['precision_policy'])}
    return plan, saved


def test_renderer_identity_survives_training_to_evaluation_schema():
    plan, saved = sample()
    worker.validate_renderer_endpoint(saved, plan)
    plan['training_specification'] = plan.pop('specification')
    plan['specification'] = worker.SPEC
    worker.validate_renderer_endpoint(saved, plan)


@pytest.mark.parametrize('key,value', [
    ('protocol', 'ibgs_warm_matched_v1'), ('renderer_profile', None),
    ('aa_module_sha256', 'other_module'),
    ('aa_provider', 'bridge_rgs.ibgs_antialias'), ('precision_policy', {'arithmetic': 'FP32 Gram'}),
    ('renderer_binary', {'path': '/legacy/backend.so', 'sha256': 'backend_digest'}),
    ('specification', {'protocol': 'ibgs_aa_stable_matched_v2', 'steps_per_arm': 12000}),
])
def test_wrong_renderer_or_training_recipe_is_rejected(key, value):
    plan, saved = sample(); saved[key] = value
    with pytest.raises(ValueError, match='metadata mismatch'):
        worker.validate_renderer_endpoint(saved, plan)


def test_precision_policy_is_bound_to_source_not_only_matching_endpoint(tmp_path):
    source = tmp_path/'ibgs_antialias_stable.py'
    source.write_text("PRECISION_POLICY = {'arithmetic': 'FP64 factor'}\nraise RuntimeError('must not import')\n")
    policy = worker.precision_policy_from_source(source)
    plan, _ = sample()
    worker.validate_precision_binding(plan, policy)
    plan['precision_policy'] = {'arithmetic': 'FP32 Gram'}
    with pytest.raises(ValueError, match='frozen module'):
        worker.validate_precision_binding(plan, policy)


def test_runtime_recipe_only_allows_locked_metadata_and_time_changes():
    plan, _ = sample()
    original = {'protocol': 'ibgs_warm_matched_v1', 'steps_per_arm': 6000,
                'internal_seconds_per_arm': 1800, 'external_seconds': 3900}
    plan['training_specification'] = dict(original, protocol=worker.SPEC['required_training_protocol'],
        renderer_profile=plan['renderer_profile'], aa_provider=worker.AA_PROVIDER,
        precision_policy=plan['precision_policy'], internal_seconds_per_arm=3600, external_seconds=7500)
    assert worker.runtime_training_specification(original, plan) == plan['training_specification']
    assert original['internal_seconds_per_arm'] == 1800
    plan['training_specification']['steps_per_arm'] = 12000
    with pytest.raises(ValueError, match='recipe change'):
        worker.runtime_training_specification(original, plan)


def test_inherited_matching_metadata_cannot_override_fixed_near():
    plan, saved = sample()
    plan['renderer_profile']['near_plane'] = .2
    saved['renderer_profile'] = plan['renderer_profile']
    with pytest.raises(ValueError, match='profile required'):
        worker.validate_renderer_endpoint(saved, plan)
