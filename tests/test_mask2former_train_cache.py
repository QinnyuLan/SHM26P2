import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

SPEC = importlib.util.spec_from_file_location('cache_script', Path(__file__).parents[1]/'scripts/render_mask2former_train_cache.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def contract():
    views = [{'name': f'{i:03}.png', 'split': 'train' if i < 350 else 'val',
              'mask_path': '/never/open/label.png' if i < 259 or i >= 350 else None,
              'image_path': '/never/open/rgb.png', 'valid_path': '/never/open/valid.png',
              'width': 1320, 'height': 989, 'K': np.eye(3).tolist(),
              'w2c': np.eye(4).tolist()} for i in range(400)]
    manifest = {'views': list(reversed(views))}
    state = {'config': {'manifest': '/metadata/manifest.json', 'optimize_cameras': False},
             'sh_degree': 3, 'training_cameras': torch.eye(4).repeat(350, 1, 1)}
    return manifest, state


def test_metadata_only_exact_train_selection_and_sorted_camera_indices():
    manifest, state = contract()
    cameras = MODULE.camera_contract(manifest, state, '/metadata/manifest.json')
    assert len(cameras) == 259
    assert [v['camera_index'] for v in cameras] == list(range(259))
    assert cameras[0]['name'] == '000.png'
    assert all(not any('path' in key for key in v) for v in cameras)
    assert all(int(v['name'][:3]) < 259 for v in cameras)


@pytest.mark.parametrize('mutation', ['duplicate', 'camera', 'profile', 'count', 'grid', 'manifest'])
def test_reject_ambiguous_or_wrong_contract(mutation):
    manifest, state = contract()
    manifest = copy.deepcopy(manifest)
    if mutation == 'duplicate':
        manifest['views'][1]['name'] = manifest['views'][0]['name']
    elif mutation == 'camera':
        state['training_cameras'][0, 0, 3] = 1
    elif mutation == 'profile':
        state['pixel_protocol'] = 'colmap_corner_v2'
    elif mutation == 'count':
        manifest['views'][-1]['mask_path'] = None
    elif mutation == 'grid':
        manifest['views'][-1]['width'] = 10
    else:
        state['config']['manifest'] = '/other/manifest.json'
    with pytest.raises(ValueError):
        MODULE.camera_contract(manifest, state, '/metadata/manifest.json')


def test_rgb_quantization_matches_renderer_and_no_channel_reorder():
    rgb = torch.tensor([[[-.01, .5, 1.1], [1/255, 2/255, 3/255]]])
    expected = np.array([[[0, 128, 255], [1, 2, 3]]], np.uint8)
    assert np.array_equal(MODULE.rgb_uint8(rgb), expected)
    with pytest.raises(ValueError):
        MODULE.rgb_uint8(torch.full((1, 1, 3), float('nan')))
