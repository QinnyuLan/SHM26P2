"""CPU contracts for prepared-only full-batch appearance descent."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bridge_rgs import fullbatch_appearance as fb

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('fullbatch_runner', ROOT/'scripts/run_fullbatch_appearance.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def parameters():
    return {key: torch.nn.Parameter(torch.tensor([.4, -.7])) for key in fb.KEYS}


def grads(params, factor=1.):
    return {key: p.detach().clone()*factor for key, p in params.items()}


def exact(mapping, reference):
    assert mapping.keys() == reference.keys()
    assert all(torch.equal(value, reference[key]) for key, value in mapping.items())


def test_preconditioner_has_negative_dot_and_accepted_step_bias_correction():
    params = parameters()
    opt = fb.RMSArmijo(params)
    g = grads(params)
    proposal = opt.propose(g)
    assert proposal.analytic_slope < 0
    for key in fb.KEYS:
        torch.testing.assert_close(proposal.direction[key], -fb.RATES[key]*g[key]/(g[key].abs()+fb.EPS))
    with opt.candidate(proposal, g, 1.) as trial:
        assert trial.accept(1., .999)
    assert opt.accepted_steps == 1
    g2 = grads(params, 2.)
    next_proposal = opt.propose(g2)
    for key in fb.KEYS:
        value = .999*proposal.second_moment[key]+.001*g2[key].square()
        expected = -fb.RATES[key]*g2[key]/((value/(1-.999**2)).sqrt()+fb.EPS)
        torch.testing.assert_close(next_proposal.direction[key], expected)


def test_all_rejected_alphas_start_at_same_base_and_do_not_commit_moments():
    params = parameters()
    before = copy.deepcopy(params)
    opt = fb.RMSArmijo(params)
    g = grads(params)
    proposal = opt.propose(g)
    for alpha in fb.ALPHAS:
        with opt.candidate(proposal, g, alpha) as trial:
            for key in fb.KEYS:
                assert torch.equal(params[key], before[key]+alpha*proposal.direction[key])
            assert not trial.accept(1., 1.1)
        exact(params, before)
        assert opt.accepted_steps == 0
        assert all(torch.count_nonzero(x) == 0 for x in opt.second_moment.values())


def test_exception_after_accept_rolls_back_values_moments_and_counter():
    params = parameters()
    before = copy.deepcopy(params)
    opt = fb.RMSArmijo(params)
    g = grads(params)
    with pytest.raises(RuntimeError, match='failed scoring'), opt.candidate(opt.propose(g), g, 1.) as trial:
        assert trial.accept(1., .999)
        raise RuntimeError('failed scoring')
    exact(params, before)
    assert opt.accepted_steps == 0
    assert all(torch.count_nonzero(x) == 0 for x in opt.second_moment.values())


def test_stale_proposal_and_unregistered_alpha_rejected():
    params = parameters()
    opt = fb.RMSArmijo(params)
    g = grads(params)
    proposal = opt.propose(g)
    with pytest.raises(ValueError, match='unregistered'), opt.candidate(proposal, g, .125):
        pass
    with opt.candidate(proposal, g, .5) as trial:
        trial.accept(1., .999)
    with pytest.raises(ValueError, match='Stale'), opt.candidate(proposal, g, .25):
        pass


@pytest.mark.parametrize('value', [0., float('nan'), float('inf')])
def test_zero_or_nonfinite_gradients_rejected(value):
    params = parameters()
    opt = fb.RMSArmijo(params)
    with pytest.raises(ValueError):
        opt.propose({key: torch.full_like(p, value) for key, p in params.items()})


def test_actual_slope_accounts_for_fp32_rounding():
    params = {key: torch.nn.Parameter(torch.full((2,), 1e8)) for key in fb.KEYS}
    g = {key: torch.ones_like(p) for key, p in params.items()}
    opt = fb.RMSArmijo(params)
    proposal = opt.propose(g)
    assert proposal.analytic_slope < 0
    with opt.candidate(proposal, g, 1.) as trial:
        assert trial.actual_slope == 0
        assert not trial.accept(1., .9)
    assert opt.accepted_steps == 0


def test_armijo_requires_negative_slope_and_material_decrease():
    assert fb.armijo(1., .999, -.1)
    assert not fb.armijo(1., .9999999, -.1)
    assert not fb.armijo(1., .999, 0.)
    assert not fb.armijo(1., .999, -100.)
    with pytest.raises(ValueError, match='Nonfinite'):
        fb.armijo(1., float('nan'), -1.)


def test_budget_accepts_last_full_pass_on_time_not_after_deadline():
    now = [0.]
    budget = fb.PassBudget(lambda: now[0], max_passes=2, seconds=10.)
    assert budget.can_start()
    now[0] = 4.
    assert budget.completed()
    assert budget.can_start()
    now[0] = 10.
    assert budget.completed()
    assert not budget.can_start()
    with pytest.raises(ValueError, match='exceeded'):
        budget.completed()
    now[0] = 0.
    late = fb.PassBudget(lambda: now[0], seconds=10.)
    now[0] = 11.
    assert not late.completed()
    assert not late.can_start()


def test_streamed_mean_gradient_matches_explicit_batch():
    x = torch.tensor([.3, -.7], requires_grad=True)
    items = [torch.tensor([.1, .2]), torch.tensor([.2, .8]), torch.tensor([-.3, .5])]
    streamed = fb.stream_mean_loss(items, lambda target: (x-target).square().mean(), backward=True)
    reference = torch.tensor([.3, -.7], requires_grad=True)
    loss = torch.stack([(reference-target).square().mean() for target in items]).mean()
    loss.backward()
    assert streamed == pytest.approx(float(loss.detach()), abs=1e-7)
    torch.testing.assert_close(x.grad, reference.grad)


def example_checkpoint():
    return {'format_version': 1, 'config': {'manifest': 'historical.json'}, 'step': 3000,
            'scene_scale': 2., 'feature_dim': 16, 'sh_degree': 3,
            'refiner_config': {'head': 'original'},
            'model': {**{key: value.detach().clone() for key, value in parameters().items()},
                      'splats.means': torch.arange(6, dtype=torch.float32).reshape(2, 3),
                      'refiner.weights': torch.ones(2)},
            'training_cameras': torch.eye(4).unsqueeze(0), 'optimizers': {'stale': 1},
            'torch_rng': torch.get_rng_state(), 'cuda_rng': [torch.ones(2)],
            'numpy_rng': np.random.get_state(), 'density_state': {}, 'stats': {},
            'unknown_future_training_state': {'stale': 2}}


def test_full_checkpoint_preserves_historical_metadata_without_invented_declarations(tmp_path):
    base = example_checkpoint()
    model = copy.deepcopy(base['model'])
    for key in fb.KEYS:
        model[key].add_(.1)
    result = fb.inference_checkpoint(base, model, {'manifest_observed': {'sha256': 'observed'}, 'accepted_steps': 1})
    exact(result['model'], model)
    assert result['config'] == base['config'] and result['step'] == base['step']
    assert torch.equal(result['training_cameras'], base['training_cameras'])
    assert 'manifest_sha256' not in result and 'pixel_protocol' not in result
    for key in ('optimizers', 'torch_rng', 'cuda_rng', 'numpy_rng', 'density_state', 'stats',
                'unknown_future_training_state'):
        assert key not in result
    assert result['appearance_optimization']['ordinary_resume_allowed'] is False
    path = tmp_path/'final.pt'
    fb.save_full_inference(result, path)
    exact(torch.load(path, weights_only=False)['model'], model)
    original_bytes = path.read_bytes()
    with pytest.raises(FileExistsError):
        fb.save_full_inference(result, path)
    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob('*.tmp')) == []
    # Copied inference tensors/cameras cannot mutate the in-memory base.
    result['model']['splats.means'].add_(1)
    result['training_cameras'].zero_()
    assert torch.equal(base['model']['splats.means'], model['splats.means'])
    assert base['training_cameras'].sum() == 4


@pytest.mark.parametrize('change', ['frozen', 'shape', 'dtype', 'nan'])
def test_checkpoint_rejects_frozen_change_and_corrupt_model(change):
    base = example_checkpoint()
    model = copy.deepcopy(base['model'])
    if change == 'frozen':
        model['refiner.weights'].add_(1)
    elif change == 'shape':
        model[fb.KEYS[0]] = torch.ones(3)
    elif change == 'dtype':
        model[fb.KEYS[0]] = model[fb.KEYS[0]].double()
    else:
        model[fb.KEYS[0]][0] = float('nan')
    with pytest.raises(ValueError):
        fb.inference_checkpoint(base, model, {})


def test_fixed_training_camera_mapping_uses_checkpoint_not_nominal_pose(tmp_path):
    # Reverse manifest order deliberately; execution order is sorted by name.
    names = [f'{i:03d}' for i in reversed(range(350))]
    manifest = {'views': [{'name': n, 'split': 'train', 'camera_id': 1, 'K': np.eye(3).tolist(),
                            'width': 20, 'height': 10, 'w2c_original': np.eye(4).tolist(),
                            'source_image_path': n+'.png'} for n in names],
                'source_cameras': {'1': {'K': np.eye(3).tolist(), 'width': 20, 'height': 10}}}
    cameras = torch.eye(4).repeat(350, 1, 1)
    cameras[:, 0, 3] = torch.arange(350)
    base = {'training_cameras': cameras}
    bindings = {str(tmp_path/(n+'.png')): 'unused' for n in names}
    views, mapping, receipt = runner.fixed_views(manifest, base, tmp_path, bindings, names)
    assert [v['name'] for v in views] == sorted(names)
    assert torch.equal(mapping['000'], cameras[349])
    assert mapping['000'][0, 3] == 349
    assert receipt['different_from_original_pose_count'] == 349
    with pytest.raises(ValueError, match='name/index'):
        runner.fixed_views(manifest, base, tmp_path, bindings, sorted(names))


def test_diagnostic_gate_requires_bound_completed_audit_and_explicit_interpretation(tmp_path):
    receipt = tmp_path/'audit.json'
    review = tmp_path/'review.json'
    receipt.write_text(json.dumps({'status': 'completed', 'plan_sha256': runner.DIAGNOSTIC_PLAN_SHA,
                                  'render_calls': 40, 'all_model_tensors_finally_restored_exact': True}))
    review.write_text(json.dumps({'decision': 'allow_fullbatch_appearance',
                                 'diagnostic_receipt_sha256': runner.sha(receipt),
                                 'full_objective_fd_and_single_step_review_passed': True, 'reason': 'reviewed'}))
    plan = {'diagnostic_receipt': {'path': str(receipt), 'sha256': runner.sha(receipt)},
            'diagnostic_review': {'path': str(review), 'sha256': runner.sha(review)}}
    runner.diagnostic_gate(plan)
    value = json.loads(review.read_text())
    value['full_objective_fd_and_single_step_review_passed'] = False
    review.write_text(json.dumps(value))
    plan['diagnostic_review']['sha256'] = runner.sha(review)
    with pytest.raises(ValueError, match='reviewed'):
        runner.diagnostic_gate(plan)
    receipt.write_text(receipt.read_text()+' ')
    with pytest.raises(ValueError, match='Changed bound'):
        runner.diagnostic_gate(plan)


def test_legacy_float_gather_matches_official_map_with_image_gradients():
    import cv2

    from bridge_rgs.evaluate import distortion_render_grid
    from bridge_rgs.raw_grid import build_raw_grid
    k = np.array([[20., 0., 10.], [0., 20., 8.], [0., 0., 1.]], np.float32)
    distortion = [.01, 0., 0., 0., 0.]
    layout = build_raw_grid(k, distortion, 20, 16, protocol='legacy_mixed_v1')
    _, width, height, source = distortion_render_grid(k, distortion, 20, 16, 'legacy_mixed_v1')
    source_image = np.random.default_rng(42).uniform(size=(height, width, 3)).astype(np.float32)
    expected = cv2.remap(source_image, source[..., 0], source[..., 1], cv2.INTER_LINEAR)
    tensor = torch.tensor(source_image, requires_grad=True)
    actual = layout.warp(tensor)
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-7, rtol=0)
    actual.sum().backward()
    assert torch.isfinite(tensor.grad).all() and bool(tensor.grad.any())


def test_partial_pass_deadline_rolls_back_candidate_and_discards_gradient():
    params = parameters()
    before = copy.deepcopy(params)
    opt = fb.RMSArmijo(params)
    g = grads(params)
    clock, seen = [0.], []
    budget = fb.PassBudget(lambda: clock[0], seconds=10.)
    def objective(item):
        seen.append(item)
        clock[0] += 6.
        return sum(p.square().sum() for p in params.values())
    with pytest.raises(fb.PassExpired), opt.candidate(opt.propose(g), g, 1.):
        try:
            fb.stream_mean_loss([0, 1, 2], objective, backward=True, check_time=budget.check_time)
        finally:
            for p in params.values():
                p.grad = None  # Same partial-gradient discard as runner.
    assert seen == [0, 1] and budget.passes == 0
    exact(params, before)
    assert opt.accepted_steps == 0 and all(p.grad is None for p in params.values())
    assert not budget.can_start()
