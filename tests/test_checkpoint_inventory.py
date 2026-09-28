"""Receipt dependency resolution must distinguish missing and unanchored inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/inventory_checkpoint_dependencies.py'
    spec = importlib.util.spec_from_file_location('checkpoint_inventory', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    return module


def test_nearest_model_directory_is_inherited_and_siblings_do_not_leak(inventory, tmp_path):
    data = {'model_dir': 'models/top', 'blocks': [
        {'teacher_modelscope_backbone': {'model_dir': 'models/nested',
                                        'weights_sha256': {'model.safetensors': 'a'*64}}},
        {'model_weights_sha256': {'model.safetensors': 'b'*64}},
        {'backbone_source': {'model_dir': '/explicit/backbone',
                             'files_sha256': {'model.safetensors': 'c'*64}}},
    ]}
    rows = inventory.references(tmp_path/'runs/receipt.json', data)
    assert [row['path'] for row in rows] == [str(tmp_path/'models/nested/model.safetensors'),
                                           str(tmp_path/'models/top/model.safetensors'),
                                           '/explicit/backbone/model.safetensors']
    assert all(row['resolution'] == 'model_dir' for row in rows)


def test_weight_dictionary_without_anchor_is_unresolved_even_if_bare_file_exists(inventory, tmp_path):
    document = tmp_path/'runs/example/render_receipt.json'
    document.parent.mkdir(parents=True)
    (document.parent/'model.safetensors').write_bytes(b'unrelated file: do not guess this path')
    data = {'semantic_ensemble': {'teacher_backbone_weights_sha256': {'model.safetensors': 'a'*64}}}
    row, = inventory.references(document, data)
    assert row['path'] is None and row['verification'] == 'unresolved'
    assert row['resolution'] == 'unresolved-no-model-dir'
    assert row['field'] == '$.semantic_ensemble.teacher_backbone_weights_sha256.model.safetensors'
    assert row['expected_sha256'] == 'a'*64


def test_null_nested_anchor_does_not_fall_back_to_wrong_ancestor(inventory, tmp_path):
    data = {'model_dir': 'models/top', 'child': {'model_dir': None,
            'weights_sha256': {'model.safetensors': 'a'*64}}}
    row, = inventory.references(tmp_path/'runs/receipt.json', data)
    assert row['path'] is None


def test_explicit_weight_path_and_regular_checkpoint_keep_unambiguous_resolution(inventory, tmp_path):
    data = {'weights_sha256': {str(tmp_path/'backbone/model.safetensors'): 'a'*64,
                              'models/other/model.safetensors': 'b'*64},
            'checkpoint': 'last.pt', 'checkpoint_sha256': 'c'*64,
            'nested': {'path': 'runs/source/last.pt', 'sha256': 'd'*64}}
    rows = inventory.references(tmp_path/'runs/run/receipt.json', data)
    assert [row['resolution'] for row in rows] == ['absolute', 'workspace-relative',
                                                  'document-relative', 'workspace-relative']
    assert rows[2]['path'] == str(tmp_path/'runs/run/last.pt')
    assert rows[3]['path'] == str(tmp_path/'runs/source/last.pt')
    assert all(row['expected_sha256'] for row in rows)


def test_inventory_counts_unresolved_separately_and_keeps_real_missing(inventory, tmp_path):
    run = tmp_path/'runs/example'
    run.mkdir(parents=True)
    (tmp_path/'artifacts').mkdir()
    model = tmp_path/'models/frozen'
    model.mkdir(parents=True)
    weight = model/'model.safetensors'
    content = b'actual frozen weights'
    weight.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    receipt = {'teacher_modelscope_backbone': {'model_dir': 'models/frozen',
                'weights_sha256': {'model.safetensors': sha}},
               'semantic_ensemble': {'teacher_backbone_weights_sha256': {'model.safetensors': sha}},
               'warmstart': 'runs/support_split_rgb_full_resumed2/step_020000.pt'}
    (run/'receipt.json').write_text(json.dumps(receipt))
    before = (run/'receipt.json').read_bytes()
    inventory.main()
    report = json.loads((tmp_path/'artifacts/checkpoint_dependencies.json').read_text())
    assert report['summary']['missing_paths'] == 1
    assert report['summary']['unresolved_references'] == 1
    assert report['summary']['checkpoint_paths'] == 2 and report['summary']['references'] == 3
    assert report['summary']['mismatched_references'] == 0
    assert len(report['unresolved_references']) == 1
    assert report['known_retention_limitation']['missing'] is True
    assert 'Not a deletion allow-list' in report['limitations'][0]
    assert weight.read_bytes() == content and (run/'receipt.json').read_bytes() == before
    actual, = [record for record in report['artifacts'] if record['exists']]
    assert actual['references'][0]['verification'] == 'match'
