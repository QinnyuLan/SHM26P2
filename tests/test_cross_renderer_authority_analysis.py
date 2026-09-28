"""Synthetic CPU contracts only: no project labels, models, VAL or CUDA."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/analyze_cross_renderer_authority.py'
if not SCRIPT.exists():
    SCRIPT = Path(__file__).with_name('analyze_cross_renderer_authority.py')
SPEC = importlib.util.spec_from_file_location('authority_analysis_test', SCRIPT)
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def probabilities(shape, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.uniform(.01, 1, size=(*shape, 5)).astype(np.float32)
    return x / x.sum(-1, keepdims=True)


def test_conditioned_features_brier_and_tie_contract():
    s = np.array([[.5, .5, 0, 0, 0], [0, 0, .2, .7, .1]])
    t = np.array([[0, 0, 0, 0, 1], [.3, .2, .1, .2, .2]])
    y = np.array([0, 3])
    d, p = np.array([.1, .2]), np.array([.3, .4])
    x, z = a.features_and_target(s * 1.000001, t, d, p, y)
    assert x.shape == (2, 125)
    assert [len(a.COLS[k]) for k in a.COLS] == [75, 100, 100]
    for row, pair in enumerate([4, 15]):
        assert x[row, 3 * pair] == 1
        assert x[row, 3 * pair + 1] == pytest.approx(a.entropy(s)[row])
        assert x[row, 75 + pair] == d[row]
        assert x[row, 100 + pair] == p[row]
        assert np.count_nonzero(x[row, :75:3]) == 1
    truth = np.eye(5)[y]
    np.testing.assert_allclose(z, ((t - truth) ** 2).sum(1) - ((s - truth) ** 2).sum(1))
    assert (np.abs(z) <= 2).all()


def test_entropy_and_js_fp64_normalization_zeros_and_symmetry():
    p, q = np.eye(5)[[0]], np.eye(5)[[4]]
    assert a.entropy(p)[0] == 0
    assert a.js_divergence(p, q)[0] == pytest.approx(np.log(2))
    assert a.js_divergence(p, p)[0] == 0
    np.testing.assert_allclose(a.js_divergence(p, q), a.js_divergence(q * 1.01, p * .98))
    with pytest.raises(ValueError, match='positive mass'):
        a.normalize_probabilities(np.zeros((1, 5)))


def test_whole_block_permutation_preserves_edges_histogram_and_global_rng():
    x = np.arange(69 * 75).reshape(69, 75)
    np.random.seed(79)
    expected_global = np.random.random()
    np.random.seed(79)
    got, report = a.block_permutation(x, 2)
    assert np.random.random() == expected_global
    np.testing.assert_array_equal(np.sort(x.ravel()), np.sort(got.ravel()))
    np.testing.assert_array_equal(got, a.block_permutation(x, 2)[0])
    assert report['moved_pixels'] > 0
    assert {tuple(g['shape']) for g in report['groups']} == {(32, 32), (5, 32), (32, 11), (5, 11)}
    np.testing.assert_array_equal(got[64:, 64:], x[64:, 64:])
    for group in report['groups']:
        bh, bw = group['shape']
        positions = [(r, c) for r in range(0, 69, 32) for c in range(0, 75, 32)
                     if (min(32, 69-r), min(32, 75-c)) == (bh, bw)]
        for destination, source in enumerate(group['destination_to_source']):
            r, c = positions[destination]
            sr, sc = positions[source]
            np.testing.assert_array_equal(got[r:r+bh, c:c+bw], x[sr:sr+bh, sc:sc+bw])


def test_streaming_ridge_equals_dense_standardized_mean_loss():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(129, 7))
    x[:, 3] = 1
    z = x[:, 0] * .3 + rng.normal(size=129) * .1 + .7
    state = a.empty_moments(7)
    for start in range(0, len(z), 17):
        a.add_moments(state, x[start:start+17], z[start:start+17])
    model = a.fit_ridge(state, np.arange(7))
    mean, std = x.mean(0), x.std(0)
    active = std > 1e-12
    normalized = (x - mean) / np.where(active, std, 1)
    normalized[:, ~active] = 0
    beta = np.linalg.solve(normalized.T @ normalized / len(z) + .001*np.eye(7),
                           normalized.T @ (z-z.mean()) / len(z))
    np.testing.assert_allclose(a.predict(model, x), z.mean() + normalized @ beta, atol=1e-13)
    assert model['active'][3] is False
    doubled = a.sum_moments([state, state])
    np.testing.assert_allclose(a.predict(a.fit_ridge(doubled, np.arange(7)), x),
                               a.predict(model, x), atol=1e-13)


def test_four_folds_do_not_use_heldout_targets_or_scaling():
    rng = np.random.default_rng(7)
    states = []
    for _ in range(16):
        state = a.empty_moments()
        x = rng.normal(size=(5, 125))
        a.add_moments(state, x, rng.normal(size=5))
        states.append(state)
    first = a.fit_fold_models(states)
    altered = copy.deepcopy(states)
    for i in (0, 4, 8, 12):
        altered[i] = a.empty_moments()
        a.add_moments(altered[i], np.ones((5, 125))*10000, np.ones(5)*10000)
    second = a.fit_fold_models(altered)
    assert first[0] == second[0]
    assert first[1]['candidate']['intercept'] != second[1]['candidate']['intercept']
    assert first[0]['candidate']['n_train_pixels'] == 12 * 5


def fixture_view_arrays(shape=(35, 19)):
    s, t = probabilities(shape), probabilities(shape, 3)
    y = np.indices(shape).sum(0).astype(np.uint8) % 5
    valid = np.ones(shape, dtype=bool)
    valid[0, 0] = False
    d = a.js_divergence(s, t)
    perm, _ = a.block_permutation(d, 0)
    return {'S': s, 'Tm': t}, d, perm, y, valid


def test_grid_fit_and_full_eval_use_intended_support_and_routing():
    arrays, d, perm, y, valid = fixture_view_arrays()
    state = a.view_fit_moments(arrays, d, perm, y, valid)
    support = valid[::4, ::4]
    x, z = a.features_and_target(arrays['S'][::4, ::4][support],
                                arrays['Tm'][::4, ::4][support], d[::4, ::4][support],
                                perm[::4, ::4][support], y[::4, ::4][support])
    assert state['n'] == support.sum()
    np.testing.assert_allclose(state['sx'], x.sum(0))
    np.testing.assert_allclose(state['sxz'], x.T @ z, atol=1e-13)
    models = {k: {'columns': v.tolist(), 'raw_coefficient': [0.] * len(v),
                   'intercept': value} for (k, v), value in zip(a.COLS.items(), [0., 1., -1.], strict=True)}
    result = a.evaluate_view(arrays, d, perm, y, valid, models)
    _, zz = a.features_and_target(arrays['S'][valid], arrays['Tm'][valid],
                                  d[valid], perm[valid], y[valid])
    assert result['valid_pixels'] == valid.sum() > state['n']
    assert result['mse']['candidate'] == pytest.approx(np.mean((zz-1)**2))
    assert sum(result['case_counts'].values()) == valid.sum()
    assert result['confusion_matrices']['route_baseline'] == result['confusion_matrices']['Tm']
    assert result['confusion_matrices']['route_candidate'] == result['confusion_matrices']['S']
    for c in range(5):
        assert result['class_mse']['candidate'][c] == pytest.approx(np.mean((zz[y[valid]==c]-1)**2))
    json.dumps(result, allow_nan=False)


def gate_views():
    return [{'mse': {'baseline': 1., 'permuted': 1.1, 'candidate': .9},
             'class_mse': {'baseline': [1., 1., None, None, None],
                           'permuted': [1.1, 1.1, None, None, None],
                           'candidate': [.9, .9, None, None, None]},
             'case_counts': {'S_only_correct': 10, 'Tm_only_correct': 10}}
            for _ in range(16)]


def test_resource_gate_and_seeded_view_bootstrap_no_pixel_weighting():
    views = gate_views()
    result = a.summarize(views)
    assert result['resource_gate_passed']
    assert result['joint_positive_classes'] == [0, 1]
    assert result == a.summarize(views)
    views[0]['valid_pixels'] = 999999
    assert result == a.summarize(views)
    assert result['comparisons']['baseline']['mean_mse_gain'] == pytest.approx(.1)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('failure', ['pixels', 'views', 'relative', 'view_signs', 'classes', 'permutation'])
def test_fixed_failure_conditions(failure):
    views = gate_views()
    if failure == 'pixels':
        for view in views:
            view['case_counts']['S_only_correct'] = 6
    elif failure == 'views':
        for i, view in enumerate(views):
            view['case_counts']['Tm_only_correct'] = 100 if i < 3 else 0
    elif failure == 'relative':
        for view in views:
            view['mse']['candidate'] = .995
    elif failure == 'view_signs':
        for view in views[9:]:
            view['mse']['candidate'] = 1.001
    elif failure == 'classes':
        for view in views:
            view['class_mse']['candidate'][1] = 2
    elif failure == 'permutation':
        for view in views:
            view['mse']['permuted'] = .8
    result = a.summarize(views)
    assert not result['resource_gate_passed']
    assert (result['status'] == 'inconclusive_error_support') == (failure in ('pixels', 'views'))


def synthetic_input(tmp_path):
    receipt = tmp_path / 'collector_receipt.json'
    a.write_json(receipt, {'status': 'completed'})
    receipt_record = {'path': str(receipt), 'sha256': a.sha256(receipt)}
    a.write_json(tmp_path/'launch_receipt.json', {'status': 'completed', 'natural_completion': True,
                  'exit_code': 0, 'execution_receipt_sha256': receipt_record['sha256']})
    views = []
    for i, name in enumerate(a.FIXED_NAMES):
        view = {'name': name, 'split': 'train'}
        for k, key in enumerate(a.SPEC['probability_keys']):
            p = tmp_path / f'{i}.{key}.npy'
            np.save(p, probabilities((9, 11), i*4+k))
            view[key] = {'path': str(p), 'sha256': a.sha256(p),
                         'shape': [9, 11, 5], 'dtype': 'float32'}
        for key, array in [('mask', np.indices((9, 11)).sum(0).astype(np.uint8) % 5),
                           ('valid', np.ones((9, 11), np.uint8)*255)]:
            p = tmp_path/f'{i}.{key}.png'
            Image.fromarray(array).save(p)
            view[key] = {'path': str(p), 'sha256': a.sha256(p)}
        views.append(view)
    return {'status': 'predictions_completed', 'protocol': a.SPEC['protocol'],
            'specification': copy.deepcopy(a.SPEC), 'prediction_receipt': receipt_record, 'views': views}


def test_complete_synthetic_run_json_and_nooverwrite(tmp_path):
    value = synthetic_input(tmp_path)
    p = tmp_path / 'input.json'
    a.write_json(p, value)
    receipt = a.run(p, a.sha256(p), tmp_path/'analysis')
    assert receipt['status'] == 'completed' and receipt['gpu_used'] is False
    result = json.loads((tmp_path/'analysis/analysis.json').read_text())
    assert len(result['views']) == 16 and len(result['fold_models']) == 4
    assert all(len(m['raw_coefficient']) == 100 for f in result['fold_models'].values()
               for k, m in f.items() if k != 'baseline')
    assert not result['summary']['resource_gate_passed']
    with pytest.raises(FileExistsError):
        a.run(p, a.sha256(p), tmp_path/'analysis')


@pytest.mark.parametrize('bad', ['not_complete', 'natural_exit', 'receipt_hash', 'probability_hash',
                                'probability_sum', 'name', 'val', 'specification'])
def test_all_prerequisites_before_any_label_decode(tmp_path, monkeypatch, bad):
    value = synthetic_input(tmp_path)
    if bad == 'not_complete':
        value['status'] = 'running'
    elif bad == 'natural_exit':
        a.write_json(tmp_path/'launch_receipt.json', {'status': 'failed', 'exit_code': 1})
    elif bad == 'receipt_hash':
        value['prediction_receipt']['sha256'] = '0'*64
    elif bad == 'probability_hash':
        value['views'][-1]['Tm']['sha256'] = '0'*64
    elif bad == 'probability_sum':
        record = value['views'][-1]['Tm']
        np.save(record['path'], np.ones((9, 11, 5), dtype=np.float32))
        record['sha256'] = a.sha256(record['path'])
    elif bad == 'name':
        value['views'][0]['name'] = '003.png'
    elif bad == 'val':
        value['views'][-1]['split'] = 'val'
    elif bad == 'specification':
        value['specification']['ridge'] = .002
    monkeypatch.setattr(a.Image, 'open', lambda *args: pytest.fail('GT decoded before prerequisites'))
    with pytest.raises(ValueError):
        a.analyze(value)
