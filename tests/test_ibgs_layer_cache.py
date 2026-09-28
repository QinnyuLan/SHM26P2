"""CPU shape, sentinel, field identity and source isolation contracts."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

WORKER = Path(__file__).with_name('prepare_ibgs_layer_cache.py')
if not WORKER.exists():
    WORKER = Path(__file__).resolve().parents[1]/'scripts/prepare_ibgs_layer_cache.py'
spec = importlib.util.spec_from_file_location('ibgs_layer_cache_test', WORKER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def small_cache():
    ids = np.full((2, 3, 4), -1, np.int32)
    z = np.zeros(ids.shape, np.float32); w = z.copy()
    ids[0, 0, :2] = [0, 9]; z[0, 0, :2] = [2., 3.]; w[0, 0, :2] = [.4, .2]
    return ids, z, w


def test_cache_roundtrip_is_separate_mmap_compatible_arrays(tmp_path):
    values = small_cache()
    result = m.validate_cached_selection(*values, height=2, width=3, point_count=10)
    assert result['present_slots'] == 2 and result['empty_slots'] == 22
    for key, value in zip(('ids', 'depth', 'weights'), values, strict=True):
        item = m.save_array(tmp_path/(key+'.npy'), value)
        actual = np.load(item['path'], mmap_mode='r', allow_pickle=False)
        assert isinstance(actual, np.memmap)
        assert item['sha256'] == m.sha(item['path']) and item['shape'] == [2, 3, 4]
        np.testing.assert_array_equal(actual, value)
        with pytest.raises(FileExistsError):
            m.save_array(tmp_path/(key+'.npy'), value)


@pytest.mark.parametrize('corruption', ['negative_id', 'outside_id', 'duplicate', 'empty_weight',
                                      'nonpositive_depth', 'nonfinite', 'excess_mass', 'dtype'])
def test_invalid_cache_never_becomes_reusable(corruption):
    ids, z, w = small_cache()
    if corruption == 'negative_id': ids[0, 0, 0] = -2
    elif corruption == 'outside_id': ids[0, 0, 0] = 10
    elif corruption == 'duplicate': ids[0, 0, 1] = 0
    elif corruption == 'empty_weight': w[1, 0, 0] = .1
    elif corruption == 'nonpositive_depth': z[0, 0, 0] = 0
    elif corruption == 'nonfinite': z[0, 0, 0] = np.nan
    elif corruption == 'excess_mass': w[0, 0, :2] = [.6, .6]
    else: ids = ids.astype(np.int64)
    with pytest.raises(ValueError):
        m.validate_cached_selection(ids, z, w, height=2, width=3, point_count=10)


def test_reused_field_metadata_is_old_full_only_without_loading_weights():
    saved = {'protocol': 'ibgs_warm_matched_v1', 'arm': 'full', 'step': 6000, 'sh_degree': 3,
             'field': {'_xyz': SimpleNamespace(shape=(996009, 3))}}
    m.validate_field_metadata(saved)
    for change in ({'arm': 'no_source'}, {'step': 5999}, {'protocol': 'ibgs_aa_stable_matched_v2'}):
        with pytest.raises(ValueError):
            m.validate_field_metadata({**saved, **change})


def contract():
    K = [[1., 0., 660.], [0., 1., 494.5], [0., 0., 1.]]
    rows = [{'name': f'{i:03d}.png', 'image_id': i, 'split': 'train', 'width': 1320, 'height': 989,
             'K': K, 'w2c_original': np.eye(4).tolist(), 'image_path': f'/RGB/{i}.png',
             'valid_path': '/valid.png'} for i in range(350)]
    return {'train_rows': rows, 'train_count': 350, 'pixel_protocol': 'colmap_corner_v2',
            'common_camera': {'K': K}, 'pixel_hashes': {'/valid.png': 'bound'},
            'neighbors': {'views': [{'name': v['name'], 'neighbors_4': []} for v in rows]}}


def test_train_metadata_excludes_source_rgb_and_allows_short_source_lists():
    value = contract()
    views, neighbors, valid = m.camera_contract(value)
    assert len(views) == len(neighbors) == 350 and all(not v for v in neighbors.values())
    assert all(set(v) == set(m.CAMERA_KEYS) and 'image_path' not in v for v in views)
    assert valid == {'path': '/valid.png', 'sha256': 'bound'}
    value['train_rows'][0]['split'] = 'val'
    with pytest.raises(ValueError): m.camera_contract(value)
    value = contract(); value['neighbors']['views'][0]['neighbors_4'] = [{'name': '999.png'}]
    with pytest.raises(ValueError): m.camera_contract(value)

