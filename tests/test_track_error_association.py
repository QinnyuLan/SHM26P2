"""Synthetic association contracts: no files, dataset, Torch or GPU access."""
import copy
import json

import numpy as np
import pytest

from bridge_rgs.track_error_association import (
    analyze_track_error_association,
    association_gate,
    joint_track_bootstrap,
)


def analyze(points, images, labels, groups, targets, *, strict=None, probability=None,
            excluded=None, query_ids=None):
    targets = np.asarray(targets, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if probability is None:
        probability = np.full((len(targets), 5), .2, dtype=np.float32)
    return analyze_track_error_association(
        np.asarray(points, dtype=np.int64), np.asarray(images, dtype=np.int64), labels,
        np.isin(labels, np.arange(5)) if strict is None else np.asarray(strict, dtype=bool),
        np.asarray(groups, dtype=np.int64), probability, targets,
        point_count=max(points, default=0)+1,
        query_image_ids=np.asarray([1] if query_ids is None else query_ids, dtype=np.int64),
        duplicate_touched_point_indices=excluded, bootstrap_repeats=100, seed=17)


def test_references_exclude_entire_target_group_and_require_three_strict_images():
    # Removing only target would leak the other group-0 vote. Unknown and a
    # non-strict known observation cannot supply the third reference either.
    result = analyze([0]*7, [1, 2, 3, 4, 5, 6, 7], [2, 2, 0, 2, 0, 255, 2],
                     [0, 0, 1, 1, 2, 2, 3], [0],
                     strict=[1, 1, 1, 1, 1, 0, 0])
    arrays, main = result['query_arrays'], result['report']['main']
    np.testing.assert_array_equal(arrays['reference_counts'], [[2, 0, 1, 0, 0]])
    assert arrays['eligible'].tolist() == [True]
    assert arrays['exposure'][0] == pytest.approx(2/3)
    assert main['cross_group_eligible']['queries'] == 1
    assert main['reference_hard_vote_descriptive']['unique_majority_disagrees_with_target'] == 1
    # No reference purity threshold: 2/3 majority remains eligible.
    result = analyze([0]*4, [1, 2, 3, 4], [2, 0, 0, 2], [0, 0, 1, 2], [0])
    assert result['query_arrays']['reference_count'].tolist() == [2]
    assert not result['query_arrays']['eligible'].any()


def test_queries_preserve_order_actual_probability_values_and_confusion_matrix():
    probability = np.array([[.2, .1, .6, .05, .050001], [.8, .05, .05, .05, .05]], np.float32)
    before = probability.copy()
    result = analyze([0]*4+[1]*4, [1, 2, 3, 4]*2, [0]*4+[2]*4,
                     [0, 1, 1, 2]*2, [4, 0], probability=probability)
    arrays, main = result['query_arrays'], result['report']['main']
    np.testing.assert_array_equal(arrays['target_row_indices'], [4, 0])
    np.testing.assert_array_equal(arrays['query_prob'], before.astype(np.float64))
    np.testing.assert_array_equal(probability, before)
    target = np.eye(5)[[2, 0]]
    np.testing.assert_array_equal(arrays['brier'], ((before.astype(float)-target)**2).sum(1))
    cm = main['cross_group_eligible']['confusion_matrix_rows_target_columns_prediction']
    assert cm[0][0] == cm[2][2] == 1
    assert np.asarray(cm).sum() == 2
    assert result['report']['maximum_query_probability_sum_error'] > 0
    json.dumps(result['report'], allow_nan=False)


def test_side_estimates_are_track_equal_and_same_track_can_occur_on_both_sides():
    # Track0 has two compatible correct queries and one exposed wrong query;
    # track1 has one compatible wrong query. Side errors are .5 and 1, not 1/3 and 1.
    points = [0]*6+[1]*4
    images = [1, 2, 3, 4, 5, 6]+[1, 4, 5, 6]
    labels = [0, 0, 2, 0, 0, 0]+[0]*4
    groups = [0, 0, 0, 1, 1, 2]+[0, 1, 1, 2]
    probs = np.eye(5)[[0, 0, 0, 2]]
    result = analyze(points, images, labels, groups, [0, 1, 2, 6],
                     probability=probs, query_ids=[1, 2, 3])
    association = result['report']['main']['association']
    assert association['tracks_on_both_sides'] == 1
    assert association['sides']['compatible']['track_equal_error'] == .5
    assert association['sides']['exposed']['track_equal_error'] == 1
    assert association['difference_exposed_minus_compatible']['error'] == .5


def test_shared_track_bootstrap_matches_explicit_resampling_and_missing_side_na():
    points = np.array([9, 9, 9, 20, 31])
    sides = np.array([0, 0, 1, 0, 1], bool)
    wrong = np.array([0., 1., 1., 0., 0.])
    brier = np.array([.2, .4, .8, .1, .6])
    actual = joint_track_bootstrap(points, sides, wrong, brier, repeats=97, seed=91)
    rng = np.random.default_rng(91)
    unique = np.unique(points)
    expected = {'error': [], 'brier': []}
    for draw in rng.integers(3, size=(97, 3)):
        for metric, values in [('error', wrong), ('brier', brier)]:
            side_means = []
            for side in (False, True):
                track_means = [values[(points == unique[i]) & (sides == side)].mean()
                               for i in draw if ((points == unique[i]) & (sides == side)).any()]
                side_means.append(np.mean(track_means) if track_means else np.nan)
            expected[metric].append(side_means[1]-side_means[0])
    assert actual['tracks_on_both_sides'] == 1
    for metric, values in expected.items():
        finite = np.asarray(values)[np.isfinite(values)]
        np.testing.assert_allclose(actual['metrics'][metric]['difference_95_interval'],
                                   np.quantile(finite, [.025, .975]), rtol=0, atol=1e-15)
        assert actual['metrics'][metric]['finite_replicates'] == len(finite)
        assert actual['metrics'][metric]['undefined_replicates'] == 97-len(finite)
        assert len(finite) < 97


def test_duplicate_sensitivity_removes_entire_track_and_has_no_extra_gate_or_bootstrap():
    result = analyze([0]*4+[1]*4, [1, 2, 3, 4]*2, [2, 0, 0, 0]+[0]*4,
                     [0, 1, 1, 2]*2, [4, 0], excluded=np.array([0]))
    report, arrays = result['report'], result['query_arrays']
    sensitivity = report['duplicate_exclusion_sensitivity']
    assert sensitivity['excluded_tracks'] == 1
    assert sensitivity['excluded_observations'] == 4
    assert sensitivity['excluded_queries'] == 1
    assert sensitivity['cross_group_eligible']['queries'] == 1
    assert sensitivity['association']['sides']['exposed']['queries'] == 0
    assert sensitivity['gate_evaluated'] is False
    assert 'gate' not in sensitivity and 'bootstrap' not in sensitivity['association']
    assert arrays['duplicate_excluded'].tolist() == [False, True]
    assert arrays['sensitivity_eligible'].tolist() == [True, False]
    assert report['main']['cross_group_eligible']['queries'] == 2
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('targets,labels,strict', [([], [0], [True]), ([0], [255], [False]),
                                                 ([0], [0], [True])])
def test_no_query_no_strict_and_no_reference_have_explicit_na(targets, labels, strict):
    result = analyze([0], [1], labels, [0], targets, strict=strict)
    main = result['report']['main']
    assert main['cross_group_eligible']['queries'] == 0
    assert main['association']['difference_exposed_minus_compatible']['brier'] is None
    boot = main['association']['bootstrap']['metrics']['brier']
    assert boot['difference_95_interval'] is None
    assert boot['finite_replicates'] == 0
    assert boot['undefined_replicates'] == 100
    assert main['gate']['status'] == 'inconclusive_error_coverage'
    json.dumps(result['report'], allow_nan=False)


def gate_fixture():
    comparison = {'sides': {'compatible': {'tracks': 20}, 'exposed': {'tracks': 20}},
                  'difference_exposed_minus_compatible': {'brier': .01}}
    return {'cross_group_eligible': {'queries': 1000, 'tracks': 200, 'wrong_tracks': 20,
                                     'wrong_images': 4},
            'association': {'bootstrap': {'metrics': {'brier': {'difference_95_interval': [.001, .02]}}}},
            'by_target_class': {str(k): copy.deepcopy(comparison) for k in range(5)},
            'by_target_group': {str(k): copy.deepcopy(comparison) for k in range(4)}}


def test_gate_exact_boundaries_and_support_vs_signal():
    report = gate_fixture()
    assert association_gate(report)['status'] == 'conditional_error_association'
    for key in ('queries', 'tracks', 'wrong_tracks', 'wrong_images'):
        changed = copy.deepcopy(report)
        changed['cross_group_eligible'][key] -= 1
        assert association_gate(changed)['status'] == 'inconclusive_error_coverage'
    changed = copy.deepcopy(report)
    changed['association']['bootstrap']['metrics']['brier']['difference_95_interval'][0] = 0
    assert association_gate(changed)['status'] == 'not_supported'
    changed = copy.deepcopy(report)
    changed['by_target_class']['2']['sides']['compatible']['tracks'] = 19
    assert association_gate(changed)['status'] == 'not_supported'
    for group in ('1', '2', '3'):
        report['by_target_group'][group]['sides']['exposed']['tracks'] = 19
    result = association_gate(report)
    assert result['status'] == 'not_supported'
    assert result['supported_target_groups'] == [0]


@pytest.mark.parametrize('kind', ['duplicate_pair', 'image_two_groups', 'unknown_strict',
                                  'bad_sum', 'negative_probability'])
def test_invalid_inputs_fail_closed(kind):
    points, images, labels, groups = [0]*4, [1, 2, 3, 4], [0]*4, [0, 1, 1, 2]
    strict, probability = [True]*4, np.full((1, 5), .2)
    if kind == 'duplicate_pair':
        images[1], groups[1] = 1, 0
    elif kind == 'image_two_groups':
        points[1], images[1] = 1, 1
    elif kind == 'unknown_strict':
        labels[1] = 255
    elif kind == 'bad_sum':
        probability[0, 0] = .21
    else:
        probability[0, :2] = [-.1, .5]
    with pytest.raises(ValueError):
        analyze(points, images, labels, groups, [0], strict=strict, probability=probability)
