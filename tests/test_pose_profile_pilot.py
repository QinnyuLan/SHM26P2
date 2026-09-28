"""CPU contracts for the isolated means pilot; no renderer/CUDA execution."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bridge_rgs.model import GaussianScene
from bridge_rgs.train import apply_parameter_scope, load_scene, validate_parameter_scope

SCRIPT = Path(__file__).resolve().with_name('run_pose_profile_pilot.py')
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_pose_profile_pilot.py'
spec = importlib.util.spec_from_file_location('fixed_means_pilot', SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def scene_and_base():
    points = np.array([[0., 0., 2.], [.1, 0., 2.], [0., .1, 2.], [.1, .1, 2.]], np.float32)
    scene = GaussianScene(points, np.full_like(points, .5))
    base = {'format_version': 1, 'model': copy.deepcopy(scene.state_dict()), 'scene_scale': scene.scene_scale,
            'feature_dim': 16, 'sh_degree': 3, 'refiner_config': {'type': 'legacy'}, 'step': 6000,
            'config': {'manifest': 'clean.json', 'steps': 6000},
            'training_cameras': torch.eye(4)[None],
            'optimizers': {'means': {'param_groups': [{'lr': 2.6018353271484376e-5,
                'betas': (.9, .999), 'eps': 1e-15, 'params': [0]}]}},
            'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
            'density_state': {'meaningless_old_window': True}, 'stats': {'old': True}}
    return scene, base


def tiny_plan(tmp_path, settings):
    pose = torch.eye(4)
    pose[0, 3] = .001
    return {'stress_manifest': str(tmp_path/'stress.json'), 'output': str(tmp_path/'out'),
            'checkpoint': str(tmp_path/'base.pt'), 'checkpoint_sha256': 'a'*64,
            'input_hashes': {str(tmp_path/'stress.json'): 'b'*64}, 'source_hashes': {'runner.py': 'c'*64},
            'adam': settings, 'views': [{'name': '002.png', 'w2c': pose.tolist()}]}


def test_independent_shuffle_three_full_cycles_and_global_rng_unaffected():
    before = np.random.get_state()
    values = pilot.schedule()
    after = np.random.get_state()
    assert len(values) == 1050
    assert all(sorted(values[i*350:(i+1)*350]) == list(range(350)) for i in range(3))
    assert values[:350] != values[350:700]
    assert values == pilot.schedule()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert pilot.SPEC['raw_train_renders']+pilot.SPEC['profile_train_renders'] == 15750


def test_fresh_exact_adam_only_means_changes_on_cpu():
    scene, base = scene_and_base()
    before = copy.deepcopy(scene.state_dict())
    settings = pilot.adam_settings(base)
    optimizer = pilot.means_optimizer(scene, settings)
    assert not optimizer.state
    assert optimizer.param_groups[0]['lr'] == settings['lr']
    assert optimizer.param_groups[0]['betas'] == tuple(settings['betas'])
    assert optimizer.param_groups[0]['eps'] == settings['eps']
    # Real GaussianScene tensors, synthetic differentiable data term; no rasterizer.
    loss = (scene.splats['means']-1).square().mean()
    loss.backward()
    assert all(p.grad is None for k, p in scene.named_parameters() if k != pilot.KEY)
    optimizer.step()
    assert not torch.equal(before[pilot.KEY], scene.state_dict()[pilot.KEY])
    assert all(torch.equal(v, scene.state_dict()[k]) for k, v in before.items() if k != pilot.KEY)
    assert int(optimizer.state[scene.splats['means']]['step']) == 1
    assert not torch.cuda.is_initialized()


def test_nondefault_source_optimizer_rejected():
    _, base = scene_and_base()
    base['optimizers']['means']['param_groups'][0]['weight_decay'] = .01
    with pytest.raises(ValueError, match='nondefault'):
        pilot.adam_settings(base)


def test_full_checkpoint_actual_cpu_inference_load_and_fresh_scope_warmstart(tmp_path):
    scene, base = scene_and_base()
    plan = tiny_plan(tmp_path, pilot.adam_settings(base))
    state = pilot.inference_checkpoint(base, scene.state_dict(), plan, 'raw', {'steps': 1050})
    path = tmp_path/'last.pt'
    pilot.save_new_checkpoint(state, path)
    loaded_scene, loaded = load_scene(path, device='cpu')
    assert all(torch.equal(loaded_scene.state_dict()[k], value) for k, value in scene.state_dict().items())
    assert all(k not in loaded for k in ('optimizers', 'numpy_rng', 'torch_rng', 'cuda_rng', 'density_state', 'stats'))
    assert loaded['step'] == 1050 and loaded['pose_profile_pilot']['base_step'] == 6000
    assert loaded['checkpoint_kind'] == 'pose_profile_pilot_inference_or_warmstart'
    assert torch.equal(loaded['training_cameras'], torch.tensor([v['w2c'] for v in plan['views']]))
    assert loaded['config']['manifest'] == plan['stress_manifest']
    with pytest.raises(ValueError, match='parameter_scope'):
        validate_parameter_scope(loaded['config'])
    # Model loading and a new valid configuration work; no claim of GPU warmstart execution.
    fresh = {'parameter_scope': 'all', 'warmstart': str(path), 'manifest': plan['stress_manifest']}
    assert validate_parameter_scope(fresh) == 'all'
    apply_parameter_scope(loaded_scene, fresh)
    assert all(p.requires_grad for p in loaded_scene.parameters())
    with pytest.raises(ValueError, match='existing checkpoint'):
        pilot.save_new_checkpoint(state, path)
    assert not torch.cuda.is_initialized()


def test_actual_main_train_rejects_pilot_config_before_any_cuda_or_pixel_operation(tmp_path, monkeypatch):
    import bridge_rgs.train as train_module
    scene, base = scene_and_base()
    plan = tiny_plan(tmp_path, pilot.adam_settings(base))
    state = pilot.inference_checkpoint(base, scene.state_dict(), plan, 'raw', {'steps': 1050})
    # Only bypass the availability predicate, not any tensor/device operation.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    with pytest.raises(ValueError, match='parameter_scope'):
        train_module.train(state['config'], resume=tmp_path/'last.pt')
    assert not torch.cuda.is_initialized()


def manifests():
    views = []
    for i in range(400):
        views.append({'name': f'{i+1:03d}.png', 'split': 'val' if i % 8 == 0 else 'train',
                      'width': 1320, 'height': 989, 'K': np.eye(3).tolist(),
                      'w2c': np.eye(4).tolist(), 'w2c_original': np.eye(4).tolist(),
                      'image_path': f'{i}.png', 'mask_path': f'{i}.mask', 'valid_path': 'valid.png'})
    return {'views': views}, {'views': copy.deepcopy(views)}


def test_views_keep_train_order_and_strip_all_label_paths():
    clean, stress = manifests()
    stress['views'][2]['w2c'][0][3] = .002
    views = pilot.view_contract(clean, stress)
    assert len(views) == 350 and all(v['split'] == 'train' for v in views)
    assert all(set(v) == set(pilot.CAMERA_KEYS) and 'mask_path' not in v for v in views)
    assert views[1]['w2c'][0][3] == .002
    stress['views'][0]['w2c'][0][3] = .01
    with pytest.raises(ValueError, match='VAL camera'):
        pilot.view_contract(clean, stress)


def test_pixel_allowlist_blocks_annotations_and_val_during_training(tmp_path):
    import cv2
    path = tmp_path/'rgb.png'
    cv2.imwrite(str(path), np.zeros((8, 8, 3), np.uint8))
    original = cv2.imread
    with pilot.guarded_reads([path]) as reads:
        assert cv2.imread(str(path)).shape == (8, 8, 3)
        with pytest.raises(ValueError, match='allowlist'):
            cv2.imread(str(tmp_path/'annotation.png'))
    assert cv2.imread is original and reads == [str(path)]


def test_calibration_gate_blocks_uncompleted_or_old_inconclusive(tmp_path, monkeypatch):
    plan = tmp_path/'plan.json'
    plan.write_text('{}')
    monkeypatch.setattr(pilot, 'CALIBRATION', plan)
    monkeypatch.setattr(pilot, 'CALIBRATION_SHA', pilot.ref.digest(plan))
    with pytest.raises(ValueError, match='not completed'):
        pilot.calibration_gate()
    audit = {'status': 'completed', 'tensor_restoration_exact': True, 'saved_cameras_exact': True,
             'summary': {'status': 'inconclusive_measurability', 'reliable_views': 3, 'retained_views': 3}}
    (tmp_path/'audit.json').write_text(json.dumps(audit))
    receipt = {'status': 'completed', 'plan_sha256': pilot.ref.digest(plan),
               'report_sha256': pilot.ref.digest(tmp_path/'audit.json'), 'summary': audit['summary']}
    (tmp_path/'execution_receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='did not support'):
        pilot.calibration_gate()
    audit['summary'].update(status='independent_direction_calibration_supported', reliable_views=5, retained_views=4)
    (tmp_path/'audit.json').write_text(json.dumps(audit))
    receipt.update(report_sha256=pilot.ref.digest(tmp_path/'audit.json'), summary=audit['summary'])
    (tmp_path/'execution_receipt.json').write_text(json.dumps(receipt))
    assert len(pilot.calibration_gate()) == 3


def test_fixed_total_deadline_has_no_per_phase_restart(monkeypatch):
    assert pilot.SPEC['total_wall_seconds'] == 600
    monkeypatch.setattr(pilot.time, 'monotonic', lambda: 599.)
    assert pilot.remaining_budget(600.) == 1.
    monkeypatch.setattr(pilot.time, 'monotonic', lambda: 600.)
    with pytest.raises(TimeoutError, match='no retry'):
        pilot.remaining_budget(600.)
