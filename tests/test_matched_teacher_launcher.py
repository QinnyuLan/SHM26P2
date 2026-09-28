"""Small contracts for the matched-domain observer, not a second trainer."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
SCRIPTS = HERE.parent if (HERE.parent/'run_matched_rgb_teacher_adaptation.py').exists() else HERE.parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('matched_teacher_tested', SCRIPTS/'run_matched_rgb_teacher_adaptation.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def record():
    return {'step': 1, 'events': [
        {'kind': 'read', 'image_path': '/real/source.png', 'mask_path': '/fixed/mask.png',
         'valid_path': '/fixed/valid.png', 'name': '002.png', 'domain': 'rendered', 'split': 'train'},
        {'kind': 'crop', 'view': '002.png', 'domain': 'rendered', 'output_sha256': ['rgb_a', 'target', 'valid'],
         'numpy_before': 'a', 'numpy_after': 'b'},
        {'kind': 'photometric', 'rng_before': 'c', 'rng_after': 'd'}],
        'optimizer_rng_before': 'e', 'optimizer_rng_after': 'f'}


def test_only_controlled_rgb_difference_is_removed():
    a = record()
    b = copy.deepcopy(a)
    b['events'][0]['image_path'] = '/other/source.png'
    b['events'][1]['output_sha256'][0] = 'rgb_b'
    assert m.normalized_trace(a) == m.normalized_trace(b)
    assert a['events'][1]['output_sha256'][0] == 'rgb_a'
    for event, key, value in [(0, 'name', '003.png'), (0, 'mask_path', '/wrong'), (1, 'numpy_after', 'x'), (2, 'rng_after', 'x')]:
        wrong = copy.deepcopy(b)
        wrong['events'][event][key] = value
        assert m.normalized_trace(a) != m.normalized_trace(wrong)
    b['events'][1]['output_sha256'][1] = 'different-target'
    assert m.normalized_trace(a) != m.normalized_trace(b)


def test_fixed_config_disables_all_validation_and_freezes_endpoint():
    settings = m.fixed_config()
    config = m.checked_config(settings)
    assert config.steps == 2000 and not config.evaluate_validation
    assert config.checkpoint_every == 500 and config.independent_augmentation_rng
    for field, value in [('steps', 2500), ('evaluate_validation', True), ('adapter_rank', 4), ('render_mix_probability', .8), ('consistency_start', 1)]:
        with pytest.raises(ValueError):
            m.checked_config(dict(settings, **{field: value}))


def test_incomplete_or_duplicate_arm_receipts_cannot_compare(tmp_path):
    from run_teacher_capacity_stage import write_json
    paths = [tmp_path/'a.json', tmp_path/'b.json']
    for path in paths:
        write_json(path, {'status': 'completed', 'arm': 'selected'})
    with pytest.raises(ValueError, match='completed arms'):
        m.compare_receipts(*paths)
