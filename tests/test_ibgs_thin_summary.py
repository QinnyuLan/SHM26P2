import copy

import numpy as np
import pytest

from bridge_rgs.ibgs_thin_summary import footprint, summarize_view, summarize_views


def fixture():
    xy = np.array([(x, y) for y in range(3) for x in range(3)], np.int32)
    sample = {'xy': xy, 'cell_id': np.zeros(9, np.int32), 'is_center': np.arange(9) == 4}
    rows = [{'xy': point, 'ids': np.array([0, 1, 2]), 'w': np.array([.05, .8, .02]),
             'z': np.array([1., 2., 3.]), 'median_mask': np.array([False, True, False]),
             'top4_mask': np.array([True, True, True]),
             'summary': {'median_depth': 2.}, 'finite_cpu_arithmetic': True,
             'comparison': {'status': 'summary_consistent_reconstruction'}} for point in xy]
    conic = np.array([[.01, 0., 1., .2], [1., 0., 1., .8], [1., 0., 1., .2]])
    return rows, sample, conic, np.ones(9, bool)


def test_conic_inverse_footprint_and_indefinite_na():
    short, ratio, valid = footprint(np.array([[.01, 0., 1.], [1., 0., -1.], [np.nan, 0., 1.]]))
    assert short[0] == 1. and ratio[0] == 10.
    assert valid.tolist() == [True, False, False]
    assert np.isnan(short[1:]).all()


def test_fixed_thin_cohort_recovery_and_unfiltered_omission_distribution():
    rows, sample, conic, domain = fixture()
    value = summarize_view('a', rows, sample, conic, domain)
    center = value['centers'][0]
    assert value['counts']['all_rays'] == 9 and value['counts']['all_centers'] == 1
    assert center['mass_recovered_narrow_front'] == pytest.approx(.05)
    assert center['mass_recovered_with_neighborhood_support'] == pytest.approx(.05)
    assert center['neighborhood_support_counts'] == [9]
    assert center['nonnegligible_subdominant'] is True
    assert center['omitted_relative_gaps'] == [.5, -.5]  # wide/back omissions stay in distribution
    assert value['exact_cuda_contribution_ledger'] is False


def test_mismatch_and_invalid_domain_never_disappear_from_denominators():
    rows, sample, conic, domain = fixture()
    rows[4]['comparison']['status'] = 'production_summary_mismatch'
    value = summarize_view('a', rows, sample, conic, domain)
    assert value['counts']['all_centers'] == 1 and value['counts']['eligible_centers'] == 0
    assert value['descriptors']['mass_recovered_narrow_front']['n'] == 0
    rows[4]['comparison']['status'] = 'summary_consistent_reconstruction'; domain[4] = False
    value = summarize_view('a', rows, sample, conic, domain)
    assert value['counts']['summary_consistent_centers'] == 1
    assert value['counts']['domain_valid_centers'] == value['counts']['eligible_centers'] == 0


def test_repetition_is_same_id_across_distinct_views_not_duplicate_observations():
    rows, sample, conic, domain = fixture()
    first = summarize_view('a', rows, sample, conic, domain)
    single = summarize_views([first])
    assert single['recovered_mass_with_cross_view_support_centers'] == 0
    second = copy.deepcopy(first); second['name'] = 'b'
    both = summarize_views([first, second])
    assert both['recovered_mass_with_spatial_and_cross_view_support_centers'] == 2
    assert both['id_view_count_histogram'] == {'2': 1}
    with pytest.raises(ValueError, match='Distinct'):
        summarize_views([first, first])


def test_zero_mass_empty_ray_remains_eligible_with_no_positive_reference():
    rows, sample, conic, domain = fixture()
    for row in rows:
        for key in ('ids', 'w', 'z', 'median_mask', 'top4_mask'):
            row[key] = row[key][:0]
        row['summary']['median_depth'] = 0.
    value = summarize_view('empty', rows, sample, conic, domain)
    assert value['counts']['eligible_centers'] == 1
    assert value['counts']['positive_median_reference_centers'] == 0
    assert value['centers'][0]['coverage_median'] is None
    assert value['descriptors']['mass_recovered_narrow_front']['mean'] == 0.


def test_continuous_descriptors_use_only_their_own_validity_conditions():
    rows, sample, conic, domain = fixture()
    conic[0, 0] = -1.
    value = summarize_view('a', rows, sample, conic, domain)
    assert value['centers'][0]['omitted_relative_gaps'] == [.5, -.5]
    assert value['centers'][0]['omitted_short_stds'] == [1.]
    conic[0, 0] = .01
    for row in rows:
        row['summary']['median_depth'] = 0.
    value = summarize_view('a', rows, sample, conic, domain)
    assert value['centers'][0]['omitted_relative_gaps'] == []
    assert value['centers'][0]['omitted_short_stds'] == [1., 1.]
    assert value['centers'][0]['mass_recovered_narrow_front'] == 0.
