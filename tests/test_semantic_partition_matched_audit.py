"""Independent-auditor CPU contracts; never read an experiment or real labels."""
import copy
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve()
SCRIPT = HERE.with_name('audit_semantic_partition_matched.py')
if not SCRIPT.exists():
    SCRIPT = HERE.parents[1] / 'scripts/audit_semantic_partition_matched.py'
SPEC = importlib.util.spec_from_file_location('independent_partition_audit', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def toy_metrics():
    reference, candidate = [], []
    for i in range(5):
        a = {'name': f'{i}.png', 'psnr': 20.+i, 'ssim': .8+i/100, 'lpips': .2-i/100}
        b = dict(a, psnr=a['psnr']+(i-2)/10, ssim=a['ssim']+.001*i, lpips=a['lpips']-.001*i)
        if i != 4:
            left = np.diag(np.arange(5)+10+i)
            left[0, 2] = 1+i
            right = left.copy(); right[0, 2] -= 1; right[0, 0] += 1
            a['confusion_matrix'] = left.tolist(); b['confusion_matrix'] = right.tolist()
        reference.append(a); candidate.append(b)
    # Independently supplied orders must not alter the paired sample population.
    return {'views': reference[::-1]}, {'views': candidate[2:]+candidate[:2]}


def test_count_bootstrap_matches_direct_index_resampling_and_signed_difference():
    left, right = toy_metrics(); repeats = 137; seed = 319
    output = audit.independent_bootstrap(left, right, repeats, seed)
    a = sorted(left['views'], key=lambda r: r['name']); b = sorted(right['views'], key=lambda r: r['name'])
    rng = np.random.default_rng(seed)
    rgb_indices = rng.integers(5, size=(repeats, 5))
    for key in ('psnr', 'ssim', 'lpips'):
        differences = np.array([y[key]-x[key] for x, y in zip(a, b, strict=True)])
        np.testing.assert_allclose(output[key]['paired_view_bootstrap_95_interval'],
                                   np.quantile(differences[rgb_indices].mean(1), [.025, .975]), atol=1e-14)
        assert output[key]['difference'] == pytest.approx(differences.mean(), abs=1e-14)
    assert output['lpips']['difference'] < 0
    indices = rng.integers(4, size=(repeats, 4))
    def score(rows, sampled):
        matrices = np.array([r['confusion_matrix'] for r in rows if 'confusion_matrix' in r])
        total = matrices[sampled].sum(1)
        diagonal = np.stack([total[:, c, c] for c in range(5)], axis=-1)
        return diagonal / (total.sum(1)+total.sum(2)-diagonal)
    difference = score(b, indices)-score(a, indices)
    np.testing.assert_allclose(output['miou_all']['paired_view_bootstrap_95_interval'],
                               np.quantile(difference.mean(1), [.025, .975]), atol=1e-14)
    for c, name in enumerate(audit.CLASSES):
        np.testing.assert_allclose(output[name+'_iou']['paired_view_bootstrap_95_interval'],
                                   np.quantile(difference[:, c], [.025, .975]), atol=1e-14)


def test_bootstrap_undefined_class_stays_undefined_instead_of_zero():
    a, b = toy_metrics()
    for result in (a, b):
        for row in result['views']:
            if 'confusion_matrix' in row:
                matrix = np.array(row['confusion_matrix']); matrix[4] = 0; matrix[:, 4] = 0
                row['confusion_matrix'] = matrix.tolist()
    result = audit.independent_bootstrap(a, b, 100, 42)
    assert result['foundation_iou'] == {'reference': None, 'candidate': None, 'difference': None,
                                          'paired_view_bootstrap_95_interval': None, 'finite_bootstrap_replicates': 0}
    json.dumps(result, allow_nan=False)


def test_all_four_gates_strict_ci_and_class_guards_are_not_winner_selection():
    pairs = {}
    for reference, threshold in audit.THRESHOLDS.items():
        pairs[reference] = {'miou_all': {'difference': threshold, 'paired_view_bootstrap_95_interval': [1e-8, .03]}}
        pairs[reference].update({name+'_iou': {'difference': -.001 if name == 'stay_cable' else -.002}
                                for name in audit.CLASSES})
    assert len(audit.independent_gate(pairs)) == 28 and all(audit.independent_gate(pairs).values())
    pairs['E']['miou_all']['paired_view_bootstrap_95_interval'][0] = 0
    pairs['point']['stay_cable_iou']['difference'] = -.001001
    clauses = audit.independent_gate(pairs)
    assert not clauses['E_miou_ci_lower_positive'] and not clauses['point_stay_cable_guard']
    assert sum(not v for v in clauses.values()) == 2
    with pytest.raises(AssertionError, match='four fixed'):
        audit.independent_gate({k: v for k, v in pairs.items() if k != 'marginal'})


def test_legacy_identity_and_soft_warp_blend_before_argmax():
    camera = {'width': 3, 'height': 2, 'K': [[3, 0, 1], [0, 4, .5], [0, 0, 1]], 'distortion': [0, 0, 0, 0]}
    calibration, width, height, grid = audit.original_grid(camera)
    assert (width, height, grid) == (3, 2, None)
    np.testing.assert_array_equal(calibration, np.array(camera['K'], np.float32))
    a = np.full((2, 3, 5), .01, np.float32); a[..., 1] = .96
    b = np.full_like(a, .01); b[..., 3] = .96
    # The same floating-point soft tie keeps the earliest class, never averages IDs.
    assert np.all(audit.mask_from_soft((a+b)*np.float32(.5), grid) == 1)
    grid = np.array([[[.5, .25], [1.25, .5]], [[.75, .5], [1.5, .75]]], np.float32)
    rng = np.random.default_rng(8); probabilities = rng.dirichlet(np.ones(5), size=(2, 3)).astype(np.float32)
    expected = cv2.remap(probabilities, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0).argmax(-1).astype(np.uint8)
    np.testing.assert_array_equal(audit.mask_from_soft(probabilities, grid), expected)


def test_independent_polygon_draw_order_unknown_then_known_and_ignore_cm():
    payload = {'imageWidth': 7, 'imageHeight': 5, 'shapes': [
        {'label': 'unknown', 'points': [[1, 1], [5, 1], [5, 3], [1, 3]]},
        {'label': 'stay_cable', 'points': [[2, 1], [3, 1], [3, 3], [2, 3]]}]}
    truth = audit.rasterize(json.dumps(payload).encode(), 7, 5)
    assert truth[2, 1] == 255 and truth[2, 2] == 2 and truth[0, 0] == 0
    prediction = np.zeros((5, 7), np.uint8)
    matrix = audit.cm(prediction, truth)
    assert matrix[2, 0] == 6 and matrix.sum() == (truth < 5).sum()


def optimizer_fixture():
    head = {'weight': torch.ones(2, 2)}
    field = {'inside_logits': torch.ones(2, 5), 'outside_logits': torch.ones(2, 5),
             'direction': torch.ones(2, 3), 'offset_raw': torch.ones(2), 'width_raw': torch.ones(2)}
    groups = []; states = {}; offset = 0
    for tensors, rate in ((list(head.values()), .0003), (list(field.values())[:2], .01), (list(field.values())[2:], .001)):
        ids = list(range(offset, offset+len(tensors))); offset += len(tensors)
        groups.append({'params': ids, 'lr': rate, 'eps': 1e-8, 'betas': (.9, .999), 'weight_decay': 0})
        for pid, tensor in zip(ids, tensors, strict=True):
            states[pid] = {'step': torch.tensor(2000.), 'exp_avg': torch.zeros_like(tensor), 'exp_avg_sq': torch.ones_like(tensor)}
    return {'param_groups': groups, 'state': states}, head, field


def test_adam_marginal_direction_absence_is_legal_only_in_that_arm():
    optimizer, head, field = optimizer_fixture()
    del optimizer['state'][3]  # The marginal model does not use a slab direction.
    audit.inspect_adam(optimizer, head, field, 'marginal')
    with pytest.raises(AssertionError, match='inactive'):
        audit.inspect_adam(optimizer, head, field, 'point')
    changed = copy.deepcopy(optimizer); changed['state'][0]['step'] = torch.tensor(1999.)
    with pytest.raises(AssertionError, match='step/shape'):
        audit.inspect_adam(changed, head, field, 'marginal')
    changed = copy.deepcopy(optimizer); changed['state'][0]['exp_avg'][0, 0] = float('nan')
    with pytest.raises(AssertionError, match='Nonfinite'):
        audit.inspect_adam(changed, head, field, 'marginal')


def test_sampler_full_epochs_and_no_global_rng_consumption():
    np.random.seed(71); before = np.random.get_state(); order = audit.expected_order(); after = np.random.get_state()
    assert len(order) == 2000 and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert all(sorted(order[start:start+259]) == list(range(259)) for start in range(0, 7*259, 259))
    assert order == audit.expected_order()


def test_failed_execution_rejected_before_any_tensor_or_pixel_read(tmp_path, monkeypatch):
    (tmp_path/'plan.json').write_text('{}')
    (tmp_path/'execution_receipt.json').write_text(json.dumps({'status': 'failed'}))
    (tmp_path/'launch_receipt.json').write_text(json.dumps({'status': 'failed', 'exit_code': 1}))
    def forbidden(*args, **kwargs):
        raise RuntimeError('Must not load any array, tensor, or image')
    monkeypatch.setattr(torch, 'load', forbidden); monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(cv2, 'imread', forbidden)
    with pytest.raises(AssertionError, match='Natural launch'):
        audit.run_audit(tmp_path, audit.sha(tmp_path/'plan.json'), 10)
    result = json.loads((tmp_path/'independent_cpu_review.json').read_text())
    assert result['status'] == 'failed' and result['new_renders'] == 0
    with pytest.raises(AssertionError, match='Never overwrite'):
        audit.run_audit(tmp_path, audit.sha(tmp_path/'plan.json'), 10)


def test_numeric_comparison_rejects_nonfinite_kind_substitution():
    tracker = audit.Audit()
    tracker.close({'a': [np.nan, np.inf, -np.inf], 'b': None}, {'a': [np.nan, np.inf, -np.inf], 'b': None}, 'toy')
    with pytest.raises(AssertionError, match='finiteness'):
        tracker.close([np.nan], [np.inf], 'toy')
    with pytest.raises(AssertionError, match='undefined'):
        tracker.close(None, 0, 'toy')


def test_scoring_ast_check_ignores_unused_entrypoints_but_not_used_math(tmp_path):
    old=tmp_path/'old.py';new=tmp_path/'new.py'
    old.write_text('def scoring(x):\n    return x * .5\n')
    new.write_text('# New unused adapter\ndef auxiliary():\n    return 9\n\ndef scoring(x):\n    return x * .5\n')
    assert audit.function_ast(old,'scoring')==audit.function_ast(new,'scoring')
    new.write_text('def scoring(x):\n    return x * .6\n')
    assert audit.function_ast(old,'scoring')!=audit.function_ast(new,'scoring')
    with pytest.raises(AssertionError,match='Missing'):
        audit.function_ast(new,'absent')
