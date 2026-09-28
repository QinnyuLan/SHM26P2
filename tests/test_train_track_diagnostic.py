"""Root orchestration edge cases: NA support and immutable protocol selection."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT/'scripts/diagnose_train_track_semantics.py'
if not RUNNER.exists():
    RUNNER = Path(__file__).with_name('diagnose_train_track_semantics.py')
spec = importlib.util.spec_from_file_location('track_diagnostic', RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_coordinate_sensitivity_excludes_na_and_ignore():
    arrays = {'primary_label': np.array([-1, 255, 0, 2, 4, 1, 3]),
              'legacy_label': np.array([0, 2, -1, 0, 4, 255, 3]),
              'reprojection_error': np.array([0., 0., 0., 1., 1.01, 0., 0.]),
              'primary_boundary_distance': np.array([np.nan, np.nan, 12., 11., 12., 12., 10.])}
    result = runner.coordinate_sensitivity(arrays)
    assert result['both_known'] == 3
    assert result['primary_only_known'] == result['legacy_only_known'] == 2
    assert result['disagreement'] == 1
    assert result['primary_strict_both_known'] == result['primary_strict_disagreement'] == 1
    cm = np.asarray(result['cm_rows_primary_columns_legacy'])
    assert cm.sum() == 3 and cm[2, 0] == cm[4, 4] == cm[3, 3] == 1


def test_metadata_population_includes_unlabeled_and_rejects_duplicates():
    views = [{'name': f'{i:03d}.png', 'image_id': i, 'split': 'train',
              'mask_path': 'placeholder' if i < 259 else None} for i in range(350)]
    assert len(runner.population({'views': views[::-1]})) == 350
    views[1]['image_id'] = views[0]['image_id']
    with pytest.raises(ValueError, match='unique'):
        runner.population({'views': views})


def test_prepare_and_receipt_never_overwrite(tmp_path):
    with pytest.raises(ValueError, match='overwrite'):
        runner.prepare(tmp_path)
    path = tmp_path/'receipt.json'
    runner.write_new(path, {'status': 'first'})
    with pytest.raises(FileExistsError):
        runner.write_new(path, {'status': 'second'})
    assert runner.read(path) == {'status': 'first'}


def test_source_manifest_ignores_only_runtime_bytecode(tmp_path):
    (tmp_path/'code.py').write_text('pass\n')
    (tmp_path/'__pycache__').mkdir()
    (tmp_path/'__pycache__/code.pyc').write_bytes(b'cache')
    assert set(runner.files(tmp_path)) == {'code.py'}
    first = runner.files(tmp_path)
    (tmp_path/'code.py').write_text('changed\n')
    assert runner.files(tmp_path) != first
