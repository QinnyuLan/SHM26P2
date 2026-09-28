"""CPU contracts for explicit image migration; no validation pixel payload needed."""
import copy
import json

import pytest
import torch

from bridge_rgs.coordinates import CORNER, LEGACY
from bridge_rgs.teacher_domains import (
    MULTI_COMPONENT_ADAPTER,
    MULTI_COMPONENT_DOMAIN,
    _sha256,
    validate_image_sources,
    verify_common_original_warmstart,
    verify_multi_component_renderers,
)


def emit(path, value):
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': _sha256(path)}


@pytest.fixture
def domain(tmp_path):
    real, render = tmp_path/'real.png', tmp_path/'render.png'
    real.write_bytes(b'real'); render.write_bytes(b'render')
    original = {'class_names': ['background', 'deck'], 'views': []}
    K = [[10., 0., 16.], [0., 10., 16.], [0., 0., 1.]]
    for split in ('train', 'val'):
        original['views'].append({'name': split+'.png', 'split': split, 'image_path': str(real) if split == 'train' else 'NEVER_OPEN_REAL_VAL',
            'mask_path': split+'_MASK_NEVER_OPENED', 'valid_path': split+'_VALID_NEVER_OPENED', 'K': K, 'width': 32, 'height': 32})
    root = emit(tmp_path/'original.json', original)
    components = {}
    for name, profile in [('selected', LEGACY), ('capacity_1m', CORNER), ('mcmc', CORNER)]:
        p = tmp_path/(name+'.pt')
        torch.save({'pixel_protocol': profile, 'manifest_sha256': 'own_manifest'}, p)
        components[name] = {'checkpoint': str(p), 'checkpoint_sha256': _sha256(p), 'pixel_protocol': profile,
                            'source_manifest_sha256': 'own_manifest'}
    records = {}
    for key in ('train_selected', 'train_composite', 'val_composite'):
        split = key.split('_')[0]
        records[key] = [{'name': split+'.png', 'image_path': str(render) if split == 'train' else 'NEVER_OPEN_VAL_CACHE',
            'sha256': _sha256(render), 'width': 32, 'height': 32, 'adapter_info': {'canvas': [32, 32], 'canvas_K': K}}]
    receipt = {'status': 'completed', 'id': MULTI_COMPONENT_DOMAIN, 'pixel_protocol': LEGACY,
               'adapter': copy.deepcopy(MULTI_COMPONENT_ADAPTER), 'original_manifest_sha256': root['sha256'],
               'components': components, 'records': records}
    bound = emit(tmp_path/'cache.json', receipt)
    protocol = {'id': MULTI_COMPONENT_DOMAIN, 'pixel_protocol': LEGACY, 'original_manifest': root['path'],
                'original_manifest_sha256': root['sha256'], 'cache_receipt': bound, 'train_domain': 'composite'}
    manifest = copy.deepcopy(original)
    manifest['image_source_protocol'] = protocol
    for view in manifest['views']:
        train = view['split'] == 'train'
        r = records['train_composite' if train else 'val_composite'][0]
        view['image_path_sources'] = {'rendered': {'path': r['image_path'], 'sha256': r['sha256']}}
        if train:
            view['image_path_sources']['real'] = {'path': str(real), 'sha256': _sha256(real)}
        else:
            view['image_path'] = r['image_path']
        view['image_domain'] = 'mixed_real_rendered' if train else 'rendered_rgb'
    return manifest, receipt, tmp_path


def test_explicit_target_legacy_accepts_true_corner_components_without_val_reads(domain):
    manifest, _, _ = domain
    assert validate_image_sources(manifest) == manifest['image_source_protocol']
    verify_multi_component_renderers(manifest['image_source_protocol'])


@pytest.mark.parametrize('field', ['mask_path', 'valid_path', 'split', 'K'])
def test_only_image_sources_may_change(domain, field):
    manifest, _, _ = domain
    manifest['views'][0][field] = 'wrong'
    with pytest.raises(ValueError, match='Non-image'):
        validate_image_sources(manifest)


@pytest.mark.parametrize('change', ['profile', 'camera', 'half_pixel', 'population', 'own_manifest'])
def test_grid_lineage_and_domain_population_fail_closed(domain, change):
    manifest, receipt, folder = domain
    if change == 'profile':
        receipt['components']['mcmc']['pixel_protocol'] = LEGACY
    elif change == 'camera':
        receipt['records']['train_composite'][0]['adapter_info']['canvas_K'][0][2] += 1
    elif change == 'half_pixel':
        receipt['adapter']['half_pixel_conjugation'] = True
    elif change == 'population':
        receipt['records']['train_selected'][0]['name'] = 'val.png'
    else:
        receipt['components']['mcmc']['source_manifest_sha256'] = 'wrong'
    manifest['image_source_protocol']['cache_receipt'] = emit(folder/'cache.json', receipt)
    with pytest.raises(ValueError):
        validate_image_sources(manifest)
        verify_multi_component_renderers(manifest['image_source_protocol'])


def test_common_original_requires_actual_bound_historical_manifest(domain):
    manifest, receipt, folder = domain
    # A completed explicit prior domain is also an admissible common-original chain.
    prior_path = folder/'prior.json'
    prior = copy.deepcopy(manifest)
    emit(prior_path, prior)
    initial = {'provenance': {'manifest': str(prior_path), 'manifest_sha256': _sha256(prior_path),
                             'image_source_protocol': prior['image_source_protocol']}}
    result = verify_common_original_warmstart(initial, manifest['image_source_protocol'])
    assert result['common_original_manifest_sha256'] == receipt['original_manifest_sha256']
    initial['provenance']['image_source_protocol'] = dict(prior['image_source_protocol'], original_manifest_sha256='different')
    with pytest.raises(ValueError, match='provenance'):
        verify_common_original_warmstart(initial, manifest['image_source_protocol'])
    initial['provenance']['image_source_protocol'] = prior['image_source_protocol']
    prior['views'][0]['mask_path'] = 'forged_label'
    emit(prior_path, prior)
    initial['provenance']['manifest_sha256'] = _sha256(prior_path)
    with pytest.raises(ValueError, match='Non-image'):
        verify_common_original_warmstart(initial, manifest['image_source_protocol'])
