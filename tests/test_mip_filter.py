import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bridge_rgs.mip_filter import (
    POLICY,
    bind_filter_source,
    camera_source,
    compute_rho,
    effective_parameters,
    normalize_config,
    refresh_due,
    refresh_filter,
    resolve_config,
)
from bridge_rgs.model import GaussianScene


def views_and_poses():
    views = []
    for index, focal in enumerate((10., 20.)):
        views.append({'name': str(index), 'split': 'train', 'width': 20, 'height': 20,
                      'K': [[focal, 0, 8], [0, focal, 9], [0, 0, 1]],
                      'w2c': torch.eye(4).tolist(), 'w2c_original': torch.eye(4).tolist()})
    return views, torch.eye(4).repeat(2, 1, 1)


def scene(enabled=True):
    points = np.array([[0, 0, 1], [.1, 0, 2], [0, .1, 3], [.2, .1, 4]], np.float32)
    s = GaussianScene(points, np.full_like(points, .5), feature_dim=8, sh_degree=1,
                      mip_filter_config={'enabled': enabled})
    if enabled:
        views, poses = views_and_poses()
        bind_filter_source(s, views, poses, 'a'*64, required_count=2)
    return s


def fake_raster(records):
    def raster(**kwargs):
        records.append(kwargs)
        s, o = kwargs['scales'], kwargs['opacities']
        weights = o*(s.mean(-1)+.2)/len(o)*.1
        alpha = weights.sum()
        colors = kwargs['colors']
        if kwargs.get('render_mode') == 'RGB+ED':
            colors = colors.mean(1)
            value = weights @ colors+(1-alpha)*kwargs['backgrounds'][0]
            value = torch.cat([value, kwargs['means'][:, 2].mean()[None]])
        else:
            value = weights @ colors+(1-alpha)*kwargs.get('backgrounds', torch.zeros(1, colors.shape[1]))[0]
        h, w = kwargs['height'], kwargs['width']
        return value[None, None, None].expand(1, h, w, -1), alpha.expand(1, h, w, 1), {}
    return raster


def test_author_sampling_actual_principal_point_depth_frustum_and_unseen_fallback():
    views, poses = views_and_poses(); src = camera_source(views, poses, 'a'*64, required_count=2)
    points = torch.tensor([[0., 0, 1], [0, 0, 4], [100, 0, 1], [0, 0, .2], [-1.1, 0, 1]])
    rho, stats = compute_rho(points, src, point_chunk=2)
    # Last point reaches -3px boundary in first camera with true cx=8, so supported.
    torch.testing.assert_close(rho[:, 0], torch.tensor([1., 4, 4, 4, 1])/20*(.2**.5))
    assert stats['unseen_gaussians'] == 2 and stats['global_native_max_fx'] == 20
    same, _ = compute_rho(points, src, point_chunk=100)
    assert torch.equal(same, rho) and not rho.requires_grad
    with pytest.raises(ValueError, match='No Gaussian'):
        compute_rho(torch.tensor([[0., 0, .1]]), src)


def test_train_population_original_poses_and_native_metadata_are_bound():
    views, poses = views_and_poses()
    with pytest.raises(ValueError, match='exactly'):
        camera_source(views, poses, 'a'*64)
    bad = copy.deepcopy(views); bad[1]['split'] = 'val'
    with pytest.raises(ValueError, match='VAL'):
        camera_source(bad, poses, 'a'*64, required_count=2)
    changed = poses.clone(); changed[0, 0, 3] = .01
    with pytest.raises(ValueError, match='original'):
        camera_source(views, changed, 'a'*64, required_count=2)
    src = camera_source(views, poses, 'a'*64, required_count=2)
    changed = copy.deepcopy(views); changed[0]['K'][0][0] *= .5
    other = camera_source(changed, poses, 'a'*64, required_count=2)
    assert src['metadata'] != other['metadata']


def test_effective_transform_volume_compensation_and_scale_gradient_contract():
    log_s = torch.tensor([[-2., -1, 0.]], dtype=torch.float64, requires_grad=True)
    logits = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
    rho = torch.tensor([[.3]], dtype=torch.float64, requires_grad=True)
    scales, opacity = effective_parameters(log_s, logits, rho)
    expected = .5*torch.sqrt(log_s.exp().square().prod(-1)/(log_s.exp().square()+.09).prod(-1))
    torch.testing.assert_close(opacity, expected)
    (scales.sum()+opacity.sum()).backward()
    assert log_s.grad.abs().sum() > 0 and logits.grad.abs().sum() > 0 and rho.grad is None


def test_rgb_semantic_evidence_share_effective_values_and_detach_geometry(monkeypatch):
    from bridge_rgs import model
    monkeypatch.setattr(model, 'configure_cuda', lambda: None)
    records = []; monkeypatch.setitem(sys.modules, 'gsplat', SimpleNamespace(rasterization=fake_raster(records)))
    s = scene(); K = torch.eye(3); pose = torch.eye(4)
    out = s.render(K, pose, 2, 2, refine=False, geometry_grad=False)
    for key in ('scales', 'opacities'):
        assert torch.equal(records[0][key], records[1][key])
        assert records[0][key].requires_grad and not records[1][key].requires_grad
    loss = -out['p3d'][..., 2].log().mean(); loss.backward()
    assert s.splats['log_scales'].grad is None and s.splats['opacity_logits'].grad is None
    assert s.splats['sem_features'].grad.abs().sum() > 0
    s.render_evidence_gate(torch.tensor([0]), torch.tensor([.7]), K, pose, 2, 2)
    for key in ('scales', 'opacities'):
        assert torch.equal(records[0][key], records[2][key])
    s.zero_grad(set_to_none=True); records.clear()
    out = s.render(K, pose, 2, 2, refine=False, geometry_grad=True)
    (-out['p3d'][..., 2].log().mean()).backward()
    assert s.splats['log_scales'].grad.abs().sum() > 0 and s.splats['opacity_logits'].grad.abs().sum() > 0


def test_refresh_schedule_deduplicates_endpoint_and_same_count_topology():
    assert not refresh_due(99, 30000)
    assert refresh_due(99, 30000, True)
    assert refresh_due(100, 30000, True) is True
    assert refresh_due(30000, 30000, True) is True
    assert refresh_due(259, 259)
    s = scene(); views, poses = views_and_poses()
    src = camera_source(views, poses, 'a'*64, required_count=2)
    old = s.mip_filter_rho.clone()
    # Same N with new centers must refresh: caller's explicit topology event drives it.
    s.splats['means'] = torch.nn.Parameter(s.splats['means'].detach()+torch.tensor([0, 0, 1.]))
    if refresh_due(100, 30000, True):
        refresh_filter(s, src, 100)
    assert s.mip_filter_state['refresh_count'] == 2 and not torch.equal(old, s.mip_filter_rho)
    # N change is repaired by fresh recomputation, with no intermediate render.
    s.splats['means'] = torch.nn.Parameter(s.splats['means'].detach()[:3])
    refresh_filter(s, src, 101)
    assert s.mip_filter_rho.shape == (3, 1)
    assert s.mip_filter_state['last_refresh_cuda_ms'] == 0


def test_config_inheritance_explicit_migration_rejection_and_legacy_off():
    config = {}; assert resolve_config(config, {}) is None and config == {}
    state = {'mip_filter_config': POLICY}; config = {}
    assert resolve_config(config, state) == POLICY and config['mip_filter'] == POLICY
    for old, requested in [({}, {'enabled': True}), (state, {'enabled': False})]:
        with pytest.raises(ValueError, match='cannot change'):
            resolve_config({'mip_filter': requested}, old)
    with pytest.raises(ValueError, match='fixed'):
        normalize_config({'enabled': True, 'refresh_every': 200})


def test_checkpoint_enabled_and_legacy_roundtrip_and_missing_buffer_reject(tmp_path, monkeypatch):
    from bridge_rgs.train import checkpoint, load_scene
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda: torch.zeros(1, dtype=torch.uint8))
    for enabled in (False, True):
        s = scene(enabled); config = {'mip_filter': POLICY} if enabled else {}
        state = checkpoint(s, {}, config, 100, torch.eye(4).repeat(2, 1, 1), {})
        p = tmp_path/f'{enabled}.pt'; torch.save(state, p)
        restored, _ = load_scene(p, device='cpu')
        assert restored.mip_filter_config == s.mip_filter_config
        assert restored.mip_filter_state == s.mip_filter_state
        for key, value in s.state_dict().items():
            assert torch.equal(value, restored.state_dict()[key])
        if enabled:
            mismatch = copy.deepcopy(state); mismatch['manifest_sha256'] = 'b'*64; torch.save(mismatch, p)
            with pytest.raises(ValueError, match='manifest SHA'):
                load_scene(p, device='cpu')
            missing = copy.deepcopy(state); del missing['model']['mip_filter_rho']; torch.save(missing, p)
            with pytest.raises(ValueError, match='schema'):
                load_scene(p, device='cpu')
        else:
            assert 'mip_filter_config' not in state and 'mip_filter_rho' not in state['model']


def test_off_matches_existing_frozen_model_rng_state_forward_and_gradients(monkeypatch):
    path = Path('/mnt/data/SHM2026/runs/rgb140_inspired_mixed_500k/source_snapshot/bridge_rgs/model.py')
    if not path.exists():
        pytest.skip('Existing immutable RGB reference source unavailable')
    spec = importlib.util.spec_from_file_location('bridge_rgs._mip_old_model_contract', path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    import bridge_rgs.model as current
    monkeypatch.setattr(old, 'configure_cuda', lambda: None); monkeypatch.setattr(current, 'configure_cuda', lambda: None)
    monkeypatch.setitem(sys.modules, 'gsplat', SimpleNamespace(rasterization=fake_raster([])))
    points = np.array([[0, 0, 1], [.1, 0, 2], [0, .1, 3], [.2, .1, 4]], np.float32)
    torch.manual_seed(18); legacy = old.GaussianScene(points, np.full_like(points, .5)); old_rng = torch.get_rng_state()
    torch.manual_seed(18); new = current.GaussianScene(points, np.full_like(points, .5)); new_rng = torch.get_rng_state()
    assert torch.equal(old_rng, new_rng) and legacy.state_dict().keys() == new.state_dict().keys()
    for key in legacy.state_dict(): assert torch.equal(legacy.state_dict()[key], new.state_dict()[key])
    a = legacy.render(torch.eye(3), torch.eye(4), 2, 2, refine=False, geometry_grad=True)
    b = new.render(torch.eye(3), torch.eye(4), 2, 2, refine=False, geometry_grad=True)
    for key in ('rgb', 'depth', 'alpha', 'p3d', 'probabilities'):
        assert torch.equal(a[key], b[key])
    for result in (a, b): (result['rgb'].sum()-result['p3d'][..., 2].log().sum()).backward()
    for (_, p), (_, q) in zip(legacy.named_parameters(), new.named_parameters()):
        assert p.grad is None and q.grad is None or p.grad is not None and q.grad is not None and torch.equal(p.grad, q.grad)


def test_frozen_semantic_resume_inherits_rgb_filter_time_without_recomputing(tmp_path, monkeypatch):
    from bridge_rgs.mip_filter import validate_resume_step
    from bridge_rgs.train import checkpoint, load_scene
    s = scene(); s.mip_filter_state['last_refresh_step'] = 30000
    with pytest.raises(ValueError, match='ahead'):
        validate_resume_step(s, 1000)
    s.splats['means'].requires_grad_(False)
    validate_resume_step(s, 1000)
    original_rho = s.mip_filter_rho.clone(); original_meta = copy.deepcopy(s.mip_filter_state)
    views, poses = views_and_poses()
    bind_filter_source(s, views, poses, 'a'*64, required_count=2)
    assert torch.equal(s.mip_filter_rho, original_rho) and s.mip_filter_state == original_meta
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda: torch.zeros(1, dtype=torch.uint8))
    state = checkpoint(s, {}, {'mip_filter': POLICY, 'freeze_geometry': True}, 1000, poses, {})
    path = tmp_path/'frozen_stage.pt'; torch.save(state, path)
    restored, saved = load_scene(path, device='cpu')
    restored.splats['means'].requires_grad_(False)  # Same order as trainer scope application.
    validate_resume_step(restored, saved['step'])
    assert torch.equal(restored.mip_filter_rho, original_rho) and restored.mip_filter_state == original_meta


def test_new_configs_only_add_filter_and_change_output():
    import yaml

    from bridge_rgs.mixed_gradient_reference import validate_reference_training
    root = Path(__file__).resolve().parents[1]
    for suffix in ('500k', 'smoke'):
        old = yaml.safe_load((root/f'configs/rgb140_inspired_mixed_{suffix}.yaml').read_text())
        new = yaml.safe_load((root/f'configs/rgb140_mip_filter_{suffix}.yaml').read_text())
        assert {k for k in set(old)|set(new) if old.get(k) != new.get(k)} == {'output', 'mip_filter'}
        assert validate_reference_training(old) == validate_reference_training(new)
        assert normalize_config(new['mip_filter']) == POLICY
    assert sum(refresh_due(i, 30000) for i in range(1, 30001))+1 == 301
