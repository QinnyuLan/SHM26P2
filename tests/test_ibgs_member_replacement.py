"""Fixed component and population identity for the post-factorial replacement."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ENTRY = Path(__file__).with_name('evaluate_ibgs_member_replacement.py')
if not ENTRY.exists():
    ENTRY = Path(__file__).resolve().parents[1]/'scripts/evaluate_ibgs_member_replacement.py'
spec = importlib.util.spec_from_file_location('member_replacement_test', ENTRY)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def fixture():
    protocol = {'fixed': 'original'}
    ep = {'reference_fingerprint': 'same', 'original_scoring_protocol': protocol, 'views': []}
    lp = {'reference_fingerprint': 'same', 'scoring_protocol': protocol, 'views': []}
    em = {'inherited_common_reference_fingerprint': 'same', 'scoring_protocol': protocol, 'views': []}
    lm = copy.deepcopy(em)
    del em['scoring_protocol']  # Older E metrics inherit it from their bound plan.
    mm = {'official_evaluation_fingerprint': 'same', 'scoring_protocol': protocol, 'views': []}
    eb, lb = [], []
    for i in range(50):
        name = f'{i:03d}.png'; camera = {'name': name, 'width': 3, 'height': 2, 'w2c': [[i]]}
        v = {'camera': camera, 'source_image_path': '/gt/'+name, 'source_image_sha256': 'gt'+name}
        ep['views'].append(dict(name=name, **copy.deepcopy(v), members={
            'mcmc': {'path': '/mcmc/'+name, 'sha256': 'm'+name},
            'capacity_1m': {'path': '/aa/'+name, 'sha256': 'a'+name}}))
        lp['views'].append(copy.deepcopy(v))
        eb.append({'name': name, 'rgb': '/e/'+name, 'rgb_sha256': 'e'+name})
        lb.append({'name': name, 'arm': 'top4_normalized', 'path': '/top/'+name, 'sha256': 't'+name})
        for m, prefix in ((em, 'e'), (lm, 't'), (mm, 'm')):
            m['views'].append({'name': name, 'width': 3, 'height': 2, 'rgb_pixels': 6,
                                   'source_rgb_sha256': 'gt'+name, 'rgb_sha256': prefix+name})
    return ep, lp, eb, lb, em, lm, mm


def test_only_named_member_replaced_and_mcmc_byte_descriptor_preserved():
    args = fixture(); args[3].append({'arm': 'top4_mass', 'name': 'not-selected.png'})
    views = worker.matching_views(*args)
    assert len(views) == 50
    for old, new in zip(args[0]['views'], views, strict=True):
        assert list(new['members']) == ['top4_normalized', 'mcmc']
        assert new['members']['mcmc'] == old['members']['mcmc']
        assert new['members']['mcmc'] is not old['members']['mcmc']
        assert new['members']['top4_normalized']['path'] == '/top/'+new['name']


@pytest.mark.parametrize('change', ['camera', 'GT', 'mcmc', 'top', 'duplicate', 'grid', 'fingerprint'])
def test_substituted_population_or_member_rejected(change):
    args = fixture()
    if change == 'camera':
        args[1]['views'][0]['camera']['w2c'] = [[-1]]
    elif change == 'GT':
        args[1]['views'][0]['source_image_sha256'] = 'wrong'
    elif change == 'mcmc':
        args[0]['views'][0]['members']['mcmc']['sha256'] = 'other'
    elif change == 'top':
        args[5]['views'][0]['rgb_sha256'] = 'other'
    elif change == 'duplicate':
        args[3][-1] = args[3][0]
    elif change == 'grid':
        args[6]['views'][0]['rgb_pixels'] = 3
    else:
        args[6]['official_evaluation_fingerprint'] = 'other'
    with pytest.raises(ValueError):
        worker.matching_views(*args)


def test_unmodified_frozen_math_and_primary_E_gate():
    base = worker.load_base(); helper = base.helpers()
    assert worker.sha(base.__file__) == worker.BASE_SHA
    assert base.MEMBERS == ('top4_normalized', 'mcmc')
    assert base.SPEC['weights'] == [.5, .5] and base.SPEC['primary_reference'] == 'E_rgb'
    assert base.SPEC['scene_renders'] == base.SPEC['teacher_calls'] == base.SPEC['mask_outputs'] == 0
    a = np.array([[[0, 1, 2], [254, 255, 255]]], np.uint8)
    b = np.array([[[1, 2, 3], [255, 254, 255]]], np.uint8)
    np.testing.assert_array_equal(helper.average_png(a, b), [[[0, 2, 2], [254, 254, 255]]])


def test_old_mcmc_metric_rows_receive_only_bound_receipt_lineage():
    ep, _, _, _, _, _, mm = fixture()
    for row in mm['views']:
        del row['rgb_sha256'], row['source_rgb_sha256']
    receipt = {'status': 'completed', 'predictions_finished_utc': 'a', 'source_scoring_started_utc': 'b',
               'official_evaluation_fingerprint': 'same', 'scoring_protocol': mm['scoring_protocol'],
               'predictions': [], 'source_records': []}
    for view in ep['views']:
        item = view['members']['mcmc']
        receipt['predictions'].append({'name': view['name'], 'rgb': item['path'], 'rgb_sha256': item['sha256']})
        receipt['source_records'].append({k: view[k] for k in ('name', 'camera', 'source_image_path', 'source_image_sha256')})
    result = worker.attach_mcmc_lineage(mm, receipt, ep['views'])
    assert result['views'][0]['rgb_sha256'] == 'm000.png' and 'rgb_sha256' not in mm['views'][0]
    receipt['source_records'][0]['camera'] = {'wrong': True}
    with pytest.raises(ValueError, match='substitution'):
        worker.attach_mcmc_lineage(mm, receipt, ep['views'])
