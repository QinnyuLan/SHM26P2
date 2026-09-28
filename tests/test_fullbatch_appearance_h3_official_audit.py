"""Independent audit direction, pooled semantics, draw order, and fixed gate."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load('fullbatch_official_independent', 'audit_fullbatch_appearance_h3_official.py')
comparison = load('official_reference_comparison_test', 'compare_official_evaluations.py')


def metrics_pair():
    from bridge_rgs.official_evaluate import FAMILY, SCORING_PROTOCOL
    views = []
    for index in range(50):
        view = {'name': f'{index:03}.png', 'width': 20, 'height': 20, 'rgb_pixels': 400,
                'psnr': 20+index*.05, 'ssim': .8+index*.0001, 'lpips': .3-index*.0002}
        if index < 41:
            matrix = np.eye(5, dtype=np.int64)*(10+index)
            matrix[0, 1] = 2+index%4
            view.update(confusion_matrix=matrix.tolist(), semantic_pixels=int(matrix.sum()),
                        semantic_ignore_pixels=400-int(matrix.sum()))
        views.append(view)
    def finish(rows):
        matrix = sum(np.asarray(v['confusion_matrix']) for v in rows if 'confusion_matrix' in v)
        values = audit.iou(matrix)
        return {'views': rows, 'validation_views': 50, 'semantic_validation_views': 41,
                'evaluation_family': FAMILY, 'scoring_protocol': SCORING_PROTOCOL,
                'official_evaluation_fingerprint': audit.FINGERPRINT,
                'confusion_matrix': matrix.tolist(), 'iou': values.tolist(),
                'miou_all': values.mean(), 'miou_foreground': values[1:].mean(),
                **{key: np.mean([v[key] for v in rows]) for key in audit.RGB_KEYS}}
    candidate = copy.deepcopy(views)
    for i, view in enumerate(candidate):
        view['psnr'] += .2 if i%3 else -.1
        view['ssim'] += .002
        view['lpips'] -= .005
        if 'confusion_matrix' in view:
            view['confusion_matrix'][0][1] -= 1
            view['confusion_matrix'][0][0] += 1
    return finish(views), finish(candidate)


def test_independent_count_weighted_bootstrap_matches_fixed_50_41_protocol():
    reference, candidate = metrics_pair()
    actual = audit.independent_pair(reference, candidate)
    expected = comparison.paired_official_comparison(reference, candidate)['metrics']
    for key, fields in actual.items():
        for field, value in fields.items():
            np.testing.assert_allclose(value, expected[key][field], atol=1e-12, rtol=0)
    assert actual['psnr']['difference'] > 0 and actual['lpips']['difference'] < 0
    assert actual['miou_all']['difference'] > 0


def test_draw_order_unknown_ignore_and_later_known_overwrite():
    data = {'imageWidth': 8, 'imageHeight': 8, 'shapes': [
        {'label': 'deck', 'points': [[0, 0], [7, 0], [7, 7], [0, 7]]},
        {'label': 'unknown', 'points': [[1, 1], [6, 1], [6, 6], [1, 6]]},
        {'label': 'stay_cable', 'points': [[3, 3], [4, 3], [4, 4], [3, 4]]}]}
    result = audit.rasterize(json.dumps(data), 8, 8)
    assert result[0, 0] == 1 and result[2, 2] == 255 and result[3, 3] == 2
    # Independent direct draw matches the established two-overlay policy.
    from bridge_rgs.official_evaluate import rasterize_official_annotation
    np.testing.assert_array_equal(result, rasterize_official_annotation(json.dumps(data).encode(), 8, 8))


def test_adoption_uses_inclusive_point_tolerances_and_strict_psnr_ci():
    values = {'psnr': {'difference': .15, 'paired_view_bootstrap_95_interval': [.001, .2]},
              'ssim': {'difference': 0.}, 'lpips': {'difference': 0.},
              'miou_all': {'difference': -.002}, 'stay_cable_iou': {'difference': -.002}}
    assert all(audit.clauses(values).values())
    values['psnr']['paired_view_bootstrap_95_interval'][0] = 0.
    assert not audit.clauses(values)['psnr_paired_95_lower_positive']
    values['stay_cable_iou']['difference'] = -.0020001
    assert not audit.clauses(values)['cable_drop_at_most_0_20pp']


def test_incomplete_annotation_population_is_rejected():
    reference, candidate = metrics_pair()
    candidate['views'][0].pop('confusion_matrix')
    with pytest.raises(ValueError, match='50/41'):
        audit.independent_pair(reference, candidate)
