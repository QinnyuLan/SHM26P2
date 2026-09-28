"""Small CPU contracts for the one fixed E/stable-AA delivered-RGB mean."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

WORKER = Path(__file__).with_name('evaluate_ibgs_e_fixed_blend.py')
if not WORKER.exists():
    WORKER = Path(__file__).resolve().parents[1]/'scripts/evaluate_ibgs_e_fixed_blend.py'
spec = importlib.util.spec_from_file_location('ibgs_e_blend_test', WORKER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    ep = {'reference_fingerprint': 'same', 'original_scoring_protocol': {'id': 'original'}, 'views': []}
    sp = {'reference_fingerprint': 'same', 'scoring_protocol': {'id': 'original'}, 'views': []}
    em = {'inherited_common_reference_fingerprint': 'same', 'views': []}
    sm = {'inherited_common_reference_fingerprint': 'same', 'scoring_protocol': {'id': 'original'}, 'views': []}
    predictions, stable = [], []
    for i in range(50):
        name = f'{i:03d}.png'
        camera = {'name': name, 'width': 3, 'height': 2, 'w2c': [[i]]}
        v = {'camera': camera, 'source_image_path': '/gt/'+name, 'source_image_sha256': 'target'+name}
        ep['views'].append({'name': name, **copy.deepcopy(v)})
        sp['views'].append(copy.deepcopy(v))
        predictions.append({'name': name, 'rgb': '/e/'+name, 'rgb_sha256': 'e'+name})
        stable.append({'name': name, 'arm': 'full', 'readout': 'fused', 'path': '/stable/'+name, 'sha256': 's'+name})
        row = {'name': name, 'width': 3, 'height': 2, 'rgb_pixels': 6,
               'source_rgb_sha256': 'target'+name, 'psnr': 25., 'ssim': .8, 'lpips': .3}
        em['views'].append({**row, 'rgb_sha256': 'e'+name})
        sm['views'].append({**row, 'rgb_sha256': 's'+name})
    return ep, sp, predictions, stable, em, sm


def test_metadata_identity_and_only_full_fused_member():
    values = fixture()
    values[3].append({'name': 'ignored.png', 'arm': 'no_source', 'readout': 'fused'})
    views = m.matching_views(*values)
    assert len(views) == 50 and list(views[0]['members']) == list(m.MEMBERS)
    assert views[0]['members']['stable_full_fused']['path'] == '/stable/000.png'


@pytest.mark.parametrize('corruption', ['camera', 'target', 'png', 'fingerprint', 'duplicate'])
def test_metadata_substitution_fails(corruption):
    values = fixture()
    if corruption == 'camera':
        values[1]['views'][0]['camera']['w2c'] = [[999]]
    elif corruption == 'target':
        values[1]['views'][0]['source_image_sha256'] = 'wrong'
    elif corruption == 'png':
        values[5]['views'][0]['rgb_sha256'] = 'wrong'
    elif corruption == 'fingerprint':
        values[5]['inherited_common_reference_fingerprint'] = 'wrong'
    else:
        values[3][-1] = values[3][0]
    with pytest.raises(ValueError):
        m.matching_views(*values)


def test_completed_status_is_not_natural_exit_evidence():
    receipt = {'status': 'completed', 'plan_sha256': 'plan'}
    launch = {'status': 'completed', 'plan_sha256': 'plan', 'exit_code': 0,
              'natural_completion': True, 'execution_receipt_sha256': 'execution'}
    m.natural(receipt, launch, 'plan', 'execution')
    for changed in ({'exit_code': 1}, {'natural_completion': False}, {'execution_receipt_sha256': 'other'}):
        with pytest.raises(ValueError):
            m.natural(receipt, {**launch, **changed}, 'plan', 'execution')


def test_barrier_requires_all_unique_written_unchanged_images(tmp_path):
    views = [{'name': f'{i:03d}.png', 'camera': {'width': 3, 'height': 2}} for i in range(50)]
    records = []
    for view in views:
        path = tmp_path/view['name']; path.write_bytes(view['name'].encode())
        records.append({'name': view['name'], 'width': 3, 'height': 2, 'rgb': str(path), 'rgb_sha256': m.sha(path)})
    m.prediction_barrier(records, views)
    with pytest.raises(ValueError):
        m.prediction_barrier(records[:-1], views)
    with pytest.raises(ValueError):
        m.prediction_barrier(records[:-1]+[records[0]], views)
    Path(records[0]['rgb']).write_bytes(b'changed')
    with pytest.raises(ValueError):
        m.prediction_barrier(records, views)


def test_fixed_frozen_helper_mean_rgb_only_and_gate_json():
    helper = m.helpers()
    a = np.array([[[0, 1, 2], [254, 255, 255]]], np.uint8)
    b = np.array([[[1, 2, 3], [255, 254, 255]]], np.uint8)
    np.testing.assert_array_equal(helper.average_png(a, b), [[[0, 2, 2], [254, 254, 255]]])
    values = fixture(); reference = values[4]; candidate = copy.deepcopy(reference)
    for row in candidate['views']:
        row.update(psnr=row['psnr']+.2, ssim=row['ssim']+.01, lpips=row['lpips']-.01)
    pair = helper.paired_rgb(reference, candidate)
    assert all(helper.gate_clauses(pair['metrics']).values())
    pair['metrics']['lpips']['difference'] = .001
    assert not all(helper.gate_clauses(pair['metrics']).values())
    json.dumps(helper.gate_clauses(pair['metrics']), allow_nan=False)
    assert m.SPEC['weights'] == [.5, .5] and m.SPEC['primary_reference'] == 'E_rgb'
    assert m.SPEC['scene_renders'] == m.SPEC['mask_outputs'] == 0

