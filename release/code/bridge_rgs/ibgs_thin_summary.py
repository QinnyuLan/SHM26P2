"""Descriptive evidence from fixed, production-summary-checked ray reconstructions.

No image/label access, fitting, camera selection or adoption gate. Gaussian
footprints and repeated IDs are model proxies, not verified physical cables.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

POLICY = {
    'relative_front_gap': .01,
    'maximum_short_std_pixels': 2.,
    'minimum_axis_ratio': 4.,
    'nonnegligible_recovered_mass': .01,
    'subdominant_cohort_mass': .5,
    'minimum_neighborhood_rays': 3,
    'minimum_repeated_views': 2,
    'footprint': 'inverse actual blurred conic; positive definite required, no floor',
    'mass': 'original unnormalized RGB alpha*T weights of CPU reconstruction',
    'scope': 'descriptive fixed thresholds, not evidence of true cable or an adoption gate',
}


def _require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def distribution(values):
    value = np.asarray(values, np.float64)
    _require(value.ndim == 1 and np.isfinite(value).all(), 'Finite scalar observations required')
    if not len(value):
        return {'n': 0, 'mean': None, 'quantiles': None}
    return {'n': len(value), 'mean': float(value.mean()),
            'quantiles': dict(zip(('min', 'p10', 'p25', 'p50', 'p75', 'p90', 'max'),
                                  map(float, np.quantile(value, (0, .1, .25, .5, .75, .9, 1))), strict=True))}


def footprint(conic):
    """Small and large covariance std from inverse-conic eigenvalues.

    Conic may have invalid/indefinite slots in a real renderer. Those remain
    NA; they are never floored into valid thin-footprint evidence.
    """
    conic = np.asarray(conic, np.float64)
    _require(conic.ndim == 2 and conic.shape[1] >= 3, 'Nx3 or Nx4 conic required')
    count = len(conic)
    finite = np.isfinite(conic[:, :3]).all(1)
    short = np.full(count, np.nan); ratio = np.full(count, np.nan)
    matrix = np.empty((int(finite.sum()), 2, 2), np.float64)
    matrix[:, 0, 0], matrix[:, 0, 1] = conic[finite, 0], conic[finite, 1]
    matrix[:, 1, 0], matrix[:, 1, 1] = conic[finite, 1], conic[finite, 2]
    eigenvalues = np.linalg.eigvalsh(matrix)
    good = np.isfinite(eigenvalues).all(1) & (eigenvalues[:, 0] > 0)
    positions = np.flatnonzero(finite)[good]
    short[positions] = 1/np.sqrt(eigenvalues[good, 1])
    ratio[positions] = np.sqrt(eigenvalues[good, 1]/eigenvalues[good, 0])
    valid = np.isfinite(short) & np.isfinite(ratio)
    return short, ratio, valid


def _ray_features(row, conic):
    ids = np.asarray(row['ids'], np.int64)
    weight, depth = (np.asarray(row[key], np.float64) for key in ('w', 'z'))
    median, top = (np.asarray(row[key], bool) for key in ('median_mask', 'top4_mask'))
    _require(all(x.shape == ids.shape for x in (weight, depth, median, top))
             and ids.ndim == 1 and (ids >= 0).all() and (ids < len(conic)).all(), 'Ray attribute mismatch')
    _require(np.isfinite(weight).all() and (weight >= 0).all(), 'Finite nonnegative ray weights required')
    short, ratio, valid_footprint = footprint(conic[ids])
    legal = np.isfinite(depth) & (depth > 0)
    z_m = float(row['summary']['median_depth'])
    has_reference = np.isfinite(z_m) and z_m > 0
    front_gap = (z_m-depth)/z_m if has_reference else np.full(len(ids), np.nan)
    narrow = valid_footprint & (short <= POLICY['maximum_short_std_pixels']) & (ratio >= POLICY['minimum_axis_ratio'])
    cohort = legal & narrow & (front_gap >= POLICY['relative_front_gap'])
    omitted = cohort & ~median
    recovered = omitted & top
    gap_omitted = legal & ~median & has_reference
    footprint_omitted = legal & ~median & valid_footprint
    recovered_ids = ids[recovered]
    sum_w = float(weight.sum())
    result = {
        'mass_rgb': sum_w, 'mass_legal_plane': float(weight[legal].sum()),
        'mass_median': float(weight[median].sum()), 'mass_top4': float(weight[top].sum()),
        'coverage_median': float(weight[median].sum()/sum_w) if sum_w > 0 else None,
        'coverage_top4': float(weight[top].sum()/sum_w) if sum_w > 0 else None,
        'mass_narrow_front': float(weight[cohort].sum()),
        'mass_omitted_narrow_front': float(weight[omitted].sum()),
        'mass_recovered_narrow_front': float(weight[recovered].sum()),
        'positive_median_reference': bool(has_reference),
        'invalid_footprint_contributions': int((~valid_footprint).sum()),
        'invalid_plane_contributions': int((~legal).sum()),
        'recovered_ids': recovered_ids.tolist(),
        'recovered_weights': weight[recovered].tolist(),
        'omitted_relative_gaps': front_gap[gap_omitted].tolist(),
        'omitted_short_stds': short[footprint_omitted].tolist(),
        'omitted_axis_ratios': ratio[footprint_omitted].tolist(),
    }
    return result


def summarize_view(name, records, sample, conic, valid_domain):
    """All fixed rays stay in denominators; spatial counts use the original cells.

    `valid_domain` is geometry-domain validity only, not a foreground or semantic
    mask. Only summary-consistent, finite, domain-valid rays inform mechanisms.
    Zero mass or missing positive median reference remain in eligible counts.
    """
    xy = np.asarray(sample['xy'])
    cells, centers = np.asarray(sample['cell_id']), np.asarray(sample['is_center'], bool)
    domain = np.asarray(valid_domain, bool)
    _require(len(records) == len(xy) == len(cells) == len(centers) == len(domain)
             and domain.ndim == 1, 'Fixed sample/domain length mismatch')
    _require(all(np.array_equal(r['xy'], p) for r, p in zip(records, xy, strict=True)), 'Reordered sampled rays')
    center_cells = cells[centers]
    _require(len(np.unique(center_cells)) == len(center_cells), 'One center per cell required')
    status = [r['comparison']['status'] for r in records]
    consistent = np.asarray([s == 'summary_consistent_reconstruction' and r['finite_cpu_arithmetic']
                             for s, r in zip(status, records, strict=True)], bool)
    eligible = domain & consistent
    features = [_ray_features(r, conic) if ok else None for r, ok in zip(records, eligible, strict=True)]
    per_cell = defaultdict(Counter)
    view_ids = set()
    for cell, feat in zip(cells, features, strict=True):
        if feat is not None:
            # A Gaussian appears at most once per ray; count ray support, not weights.
            ids = set(feat['recovered_ids']); per_cell[int(cell)].update(ids); view_ids.update(ids)
    rows = []
    for index in np.flatnonzero(centers):
        feature = features[index]
        base = {'cell': int(cells[index]), 'xy': xy[index].tolist(), 'domain_valid': bool(domain[index]),
                'replay_status': status[index], 'eligible': bool(eligible[index])}
        if feature is not None:
            feature = dict(feature)
            repeated = [per_cell[int(cells[index])][point] >= POLICY['minimum_neighborhood_rays']
                        for point in feature['recovered_ids']]
            feature['mass_recovered_with_neighborhood_support'] = float(sum(
                weight for weight, repeat in zip(feature['recovered_weights'], repeated, strict=True) if repeat))
            feature['neighborhood_support_counts'] = [per_cell[int(cells[index])][point]
                                                       for point in feature['recovered_ids']]
            feature['nonnegligible_subdominant'] = bool(
                feature['mass_recovered_narrow_front'] >= POLICY['nonnegligible_recovered_mass']
                and feature['mass_narrow_front'] < POLICY['subdominant_cohort_mass'])
            base.update(feature)
        rows.append(base)
    scalar_keys = ('mass_rgb', 'mass_legal_plane', 'mass_median', 'mass_top4',
                   'coverage_median', 'coverage_top4', 'mass_narrow_front',
                   'mass_omitted_narrow_front', 'mass_recovered_narrow_front',
                   'mass_recovered_with_neighborhood_support')
    descriptors = {key: distribution([r[key] for r in rows if r['eligible'] and r[key] is not None])
                   for key in scalar_keys}
    for key in ('omitted_relative_gaps', 'omitted_short_stds', 'omitted_axis_ratios'):
        descriptors[key] = distribution([v for r in rows if r['eligible'] for v in r[key]])
    counts = {
        'all_rays': len(records), 'all_centers': int(centers.sum()),
        'replay_status_all': dict(Counter(status)),
        'domain_valid_centers': int((centers & domain).sum()),
        'summary_consistent_centers': int((centers & consistent).sum()),
        'eligible_centers': int((centers & eligible).sum()),
        'nonnegligible_subdominant_centers': sum(r.get('nonnegligible_subdominant', False) for r in rows),
        'positive_median_reference_centers': sum(r.get('positive_median_reference', False) for r in rows),
        'recovered_ids_all_eligible_neighborhoods': len(view_ids),
    }
    return {'name': name, 'policy': dict(POLICY), 'counts': counts, 'descriptors': descriptors,
            'centers': rows, 'recovered_ids_all_eligible_neighborhoods': sorted(view_ids),
            'exact_cuda_contribution_ledger': False,
            'interpretation': 'Gaussian footprint/repetition proxies; top4 mass advantage is definitional'}


def summarize_views(views):
    """Equal-view descriptive means plus ID repetition across the fixed views."""
    _require(views and len({v['name'] for v in views}) == len(views), 'Distinct nonempty fixed views required')
    _require(all(v['policy'] == POLICY for v in views), 'Description policy changed')
    id_views = Counter(point for view in views for point in set(view['recovered_ids_all_eligible_neighborhoods']))
    equal_view = {}
    for key in views[0]['descriptors']:
        means = [v['descriptors'][key]['mean'] for v in views]
        valid = [x for x in means if x is not None]
        equal_view[key] = {'views_with_observations': len(valid),
                          'mean_of_view_means': float(np.mean(valid)) if valid else None}
    repeated_mass, jointly_supported_mass = [], []
    eligible, qualifying, with_cross, with_joint = 0, 0, 0, 0
    for view in views:
        view_mass, joint_mass = [], []
        for row in view['centers']:
            if not row['eligible']:
                continue
            eligible += 1; qualifying += int(row['nonnegligible_subdominant'])
            cross = [id_views[p] >= POLICY['minimum_repeated_views'] for p in row['recovered_ids']]
            mass = sum(w for w, ok in zip(row['recovered_weights'], cross, strict=True) if ok)
            joint = sum(w for w, ok, n in zip(row['recovered_weights'], cross,
                         row['neighborhood_support_counts'], strict=True)
                        if ok and n >= POLICY['minimum_neighborhood_rays'])
            view_mass.append(mass); joint_mass.append(joint)
            if row['nonnegligible_subdominant']:
                with_cross += int(mass >= POLICY['nonnegligible_recovered_mass'])
                with_joint += int(joint >= POLICY['nonnegligible_recovered_mass'])
        repeated_mass.append(float(np.mean(view_mass)) if view_mass else None)
        jointly_supported_mass.append(float(np.mean(joint_mass)) if joint_mass else None)
    def mean_available(items):
        items = [x for x in items if x is not None]
        return {'views_with_observations': len(items), 'mean_of_view_means': float(np.mean(items)) if items else None}
    return {'views': len(views), 'policy': dict(POLICY), 'exact_cuda_contribution_ledger': False,
            'all_centers': sum(v['counts']['all_centers'] for v in views), 'eligible_centers': eligible,
            'nonnegligible_subdominant_centers': qualifying,
            'recovered_mass_with_cross_view_support_centers': with_cross,
            'recovered_mass_with_spatial_and_cross_view_support_centers': with_joint,
            'id_view_count_histogram': {str(k): v for k, v in sorted(Counter(id_views.values()).items())},
            'equal_view_descriptors': equal_view,
            'recovered_mass_with_cross_view_support': mean_available(repeated_mass),
            'recovered_mass_with_spatial_and_cross_view_support': mean_available(jointly_supported_mass),
            'inference_limit': 'Repeated model IDs do not verify real thin foreground, useful source RGB or accuracy gain'}
