"""CPU contracts for fixed cached-teacher endpoint evaluation, without scene rendering."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve()
SCRIPT = HERE.parent / 'evaluate_projective_deck_pooling.py'
if not SCRIPT.exists():
    SCRIPT = HERE.parents[1] / 'scripts/evaluate_projective_deck_pooling.py'
spec = importlib.util.spec_from_file_location('deck_evaluation', SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def pairs_at_threshold():
    return {reference: {'metrics': {
        'miou_all': {'difference': threshold, 'paired_view_bootstrap_95_interval': [1e-7, .01]},
        **{name+'_iou': {'difference': -.001 if name == 'stay_cable' else -.002}
           for name in ('background', 'deck', 'stay_cable', 'tower', 'foundation')},
    }} for reference, threshold in evaluation.THRESHOLDS.items()}


def test_all_four_references_require_gain_ci_and_each_class_guard():
    pairs = pairs_at_threshold()
    assert len(evaluation.adoption_clauses(pairs)) == 28
    assert all(evaluation.adoption_clauses(pairs).values())
    for reference in evaluation.THRESHOLDS:
        bad = copy.deepcopy(pairs)
        bad[reference]['metrics']['miou_all']['paired_view_bootstrap_95_interval'][0] = 0.
        assert not all(evaluation.adoption_clauses(bad).values())
        for name in ('background', 'deck', 'stay_cable', 'tower', 'foundation'):
            bad = copy.deepcopy(pairs)
            bad[reference]['metrics'][name+'_iou']['difference'] -= 1e-8
            assert not all(evaluation.adoption_clauses(bad).values())
    with pytest.raises(ValueError, match='three continuation'):
        evaluation.adoption_clauses({k: v for k, v in pairs.items() if k != 'E'})


def test_prediction_barrier_requires_600_masks_and_200_soft(monkeypatch):
    seen = []
    monkeypatch.setattr(evaluation, 'require_record', lambda r: seen.append(r['path']))
    views = [{'name': f'{i:03}.png'} for i in range(50)]
    records = [{'arm': arm, 'name': v['name'], 'teacher_input_rgb_exact': True,
                'masks': {kind: {'path': f'{arm}/{v["name"]}/{kind}'} for kind in evaluation.KINDS},
                'soft_canvas': {'path': f'{arm}/{v["name"]}/soft'}}
               for arm in evaluation.ARMS for v in views]
    evaluation.prediction_barrier(records, views)
    assert len(seen) == len(set(seen)) == 800
    with pytest.raises(ValueError, match='fifty'):
        evaluation.prediction_barrier(records[:-1], views)
    bad = copy.deepcopy(records)
    del bad[-1]['masks']['raw']
    with pytest.raises(ValueError, match='Incomplete'):
        evaluation.prediction_barrier(bad, views)
    bad = copy.deepcopy(records)
    bad[-1]['teacher_input_rgb_exact'] = False
    with pytest.raises(ValueError, match='identity'):
        evaluation.prediction_barrier(bad, views)


def test_soft_mixture_is_warped_before_argmax_and_never_hard_votes():
    scene = np.zeros((1, 2, 5), np.float32)
    teacher = np.zeros_like(scene)
    scene[0, 0, :2] = [.49, .51]
    scene[0, 1, :2] = [.99, .01]
    teacher[0, :, :2] = [.49, .51]
    grid = np.array([[[.5, 0.]]], np.float32)
    actual = evaluation.warp_probabilities((scene+teacher)*np.float32(.5), grid)
    np.testing.assert_allclose(actual[0, 0, :2], [.615, .385], rtol=0, atol=1e-7)
    assert actual.argmax(-1).item() == 0
    with pytest.raises(ValueError, match='Empty'):
        evaluation.warp_probabilities(scene, np.array([[[-100., -100.]]], np.float32))


def test_artifact_serialization_and_new_rgb_identity(tmp_path):
    pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    old = evaluation.save_png(tmp_path/'old.png', pixels)
    new = evaluation.save_png(tmp_path/'new.png', pixels, old)
    assert old['sha256'] == new['sha256']
    with pytest.raises(ValueError, match='New uint8'):
        evaluation.save_png(tmp_path/'new.png', pixels)
    with pytest.raises(ValueError, match='RGB bytes differ'):
        evaluation.save_png(tmp_path/'changed.png', pixels+np.uint8(1), old)
    p = np.full((3, 4, 5), .2, np.float32)
    record = evaluation.save_soft(tmp_path/'soft.npy', p)
    np.testing.assert_array_equal(np.load(record['path']), p)
    with pytest.raises(FileExistsError):
        evaluation.save_soft(tmp_path/'soft.npy', p)
    evaluation.write(tmp_path/'record.json', {'passed': np.bool_(True), 'value': np.float64(.5)})
    assert evaluation.read(tmp_path/'record.json') == {'passed': True, 'value': .5}


def delta_fixture(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{"views": []}')
    (tmp_path/'plan.json').write_text('{"fixed": true}')
    plan = {'output': str(tmp_path), 'manifest': str(manifest), 'specification': {'fixed': True}}
    state = {'feature_dim': 16, 'sh_degree': 3,
             'refiner_config': {'type': 'multiscale', 'channels': 64, 'residual_bound': 6.,
                                'context': 'pyramid_strip', 'depth_moments': 'cross'},
             'training_cameras': torch.eye(4)[None]}
    delta = {'format': evaluation.FORMAT, 'base_checkpoint_sha256': evaluation.BASE_SHA,
             'mode': 'projective', 'step': 2000, 'plan_sha256': evaluation.sha(tmp_path/'plan.json'),
             'specification': plan['specification'], 'refiner_config': state['refiner_config'],
             'pixel_protocol': 'legacy_mixed_v1', 'manifest_sha256': evaluation.sha(manifest),
             'training_camera_sha256': hashlib.sha256(state['training_cameras'].numpy().tobytes()).hexdigest(),
             'refiner_state': {'weight': torch.ones(2)}}
    return delta, plan, state


@pytest.mark.parametrize('key,value', [
    ('format', 'appearance_delta'), ('base_checkpoint_sha256', 'wrong'), ('mode', 'wrong'),
    ('step', 1999), ('model', {}), ('plan_sha256', 'wrong'), ('specification', {}),
    ('refiner_config', {}), ('pixel_protocol', 'colmap_corner_v2'),
    ('manifest_sha256', 'wrong'), ('training_camera_sha256', 'wrong'),
])
def test_delta_rejects_wrong_provenance_or_architecture(tmp_path, key, value):
    delta, plan, state = delta_fixture(tmp_path)
    evaluation.validate_delta(delta, 'projective', plan, state)
    delta[key] = value
    with pytest.raises(ValueError):
        evaluation.validate_delta(delta, 'projective', plan, state)


def test_delta_rejects_nonfinite_head(tmp_path):
    delta, plan, state = delta_fixture(tmp_path)
    delta['refiner_state']['weight'][0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        evaluation.validate_delta(delta, 'projective', plan, state)
