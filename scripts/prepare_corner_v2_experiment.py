"""Lock a budget/source/config and prepare a separate corrected-coordinate dataset.

No training is launched, and no existing dataset/checkpoint is removed or rewritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'runs/corner_v2_preparation'
DATASET = ROOT/'artifacts/prepared_corner_v2'
LEGACY = ROOT/'artifacts/prepared'
MIB = 2**20


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*MIB), b''):
            h.update(block)
    return h.hexdigest()


def hashes(folder, pattern='*'):
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob(pattern)) if p.is_file()}


def main():
    if OUTPUT.exists() or DATASET.exists():
        raise FileExistsError('Fresh preparation audit and data directories are required')
    budget_mib = {'prepared_dataset': 497, 'rgb_last_plus_atomic_temporary': 894,
                  'future_semantic_checkpoint': 241, 'two_final_evaluations': 150,
                  'source_and_logs': 5, 'remaining_reserve': 256}
    free = shutil.disk_usage(ROOT).free
    required = sum(budget_mib.values())*MIB
    if free < required:
        raise OSError(f'Insufficient free budget: {free} < {required}')
    for path in ('configs/corner_v2_rgb_full.yaml', 'configs/corner_v2_semantic_coupled.yaml'):
        if (ROOT/path).exists():
            raise FileExistsError(path)
    OUTPUT.mkdir(parents=True)
    budget = {'locked_utc': datetime.now(UTC).isoformat(), 'free_bytes': free,
              'reserved_mib': budget_mib, 'required_bytes': required,
              'extra_beyond_reserve_bytes': free-required, 'deletion_performed': False,
              'scope': 'Estimated peak includes RGB atomic replacement and a future semantic last; no extra step checkpoints.'}
    (OUTPUT/'budget.json').write_text(json.dumps(budget, indent=2)+'\n')
    snapshot = OUTPUT/'source_snapshot'
    original_hashes = hashes(ROOT/'src/bridge_rgs', '*.py')
    shutil.copytree(ROOT/'src/bridge_rgs', snapshot/'bridge_rgs',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source_hashes = hashes(snapshot, '*.py')
    assert {key.removeprefix('bridge_rgs/'): sha for key, sha in source_hashes.items()} == original_hashes
    assert hashes(ROOT/'src/bridge_rgs', '*.py') == original_hashes
    for baseline, destination, updates in (
        ('configs/support_split_rgb_full.yaml', 'configs/corner_v2_rgb_full.yaml',
         {'manifest': 'artifacts/prepared_corner_v2/manifest.json', 'output': 'runs/corner_v2_rgb_full',
          'pixel_protocol': 'colmap_corner_v2', 'eval_every': 0, 'save_every': 5000}),
        ('configs/support_split_semantic_coupled.yaml', 'configs/corner_v2_semantic_coupled.yaml',
         {'manifest': 'artifacts/prepared_corner_v2/manifest.json', 'output': 'runs/corner_v2_semantic_coupled',
          'warmstart': 'runs/corner_v2_rgb_full/last.pt', 'pixel_protocol': 'colmap_corner_v2',
          'eval_every': 0, 'save_every': 2000}),
    ):
        config = yaml.safe_load((ROOT/baseline).read_text())
        config.update(updates)
        (ROOT/destination).write_text(yaml.safe_dump(config, sort_keys=False))
    input_paths = [ROOT/'Dataset/camera_parameters/cameras.txt', ROOT/'Dataset/camera_parameters/images.txt', ROOT/'uv.lock']
    for name in ('images', 'unlabeled_Images', 'json'):
        input_paths.extend(p for p in (ROOT/'Dataset'/name).iterdir() if p.is_file())
    input_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(input_paths)}
    legacy_hashes = hashes(LEGACY)
    command = [sys.executable, '-m', 'bridge_rgs.cli', 'prepare', '--dataset', 'Dataset',
               '--output', 'artifacts/prepared_corner_v2', '--max-width', '1320',
               '--max-points', '60000', '--val-every', '8', '--seed', '42', '--workers', '8',
               '--pixel-protocol', 'colmap_corner_v2']
    receipt = {'status': 'preparing', 'started_utc': datetime.now(UTC).isoformat(),
               'source_snapshot': str(snapshot), 'source_hashes': source_hashes,
               'source_tree_sha256': hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
               'input_hashes': input_hashes, 'legacy_before_hashes': legacy_hashes,
               'runner_sha256': digest(__file__), 'command': command,
               'preparation': {'pixel_protocol': 'colmap_corner_v2', 'max_width': 1320, 'max_points': 60000,
                               'seed': 42, 'val_every': 8, 'bundle_adjustment': False},
               'config_hashes': {name: digest(ROOT/name) for name in
                                 ('configs/corner_v2_rgb_full.yaml', 'configs/corner_v2_semantic_coupled.yaml')},
               'baseline_config_hashes': {name: digest(ROOT/name) for name in
                                          ('configs/support_split_rgb_full.yaml', 'configs/support_split_semantic_coupled.yaml')}}
    receipt_path = OUTPUT/'preparation_receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    environment = dict(os.environ, PYTHONPATH=str(snapshot), CUDA_VISIBLE_DEVICES='',
                       OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    try:
        with (OUTPUT/'prepare.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert hashes(LEGACY) == legacy_hashes
        assert hashes(snapshot, '*.py') == source_hashes
        assert all(digest(ROOT/path) == sha for path, sha in input_hashes.items())
        old = json.loads((LEGACY/'manifest.json').read_text())
        new = json.loads((DATASET/'manifest.json').read_text())
        assert new['schema_version'] == 2 and new['pixel_protocol']['id'] == 'colmap_corner_v2'
        assert new['split'] == old['split'] and new['source_cameras'] == old['source_cameras']
        assert new['preparation'] == old['preparation']
        keys = ('name', 'image_id', 'camera_id', 'width', 'height', 'K', 'w2c', 'w2c_original',
                'split', 'source_image_path', 'source_annotation_path', 'pose_covariance',
                'pose_covariance_order', 'pose_uncertainty_source')
        assert len(new['views']) == len(old['views']) == 400
        for before, after in zip(old['views'], new['views'], strict=True):
            assert all(before[key] == after[key] for key in keys), before['name']
            assert bool(before['mask_path']) == bool(after['mask_path'])
            assert (after['width'], after['height']) == (1320, 989)
        counts = Counter((v['split'], bool(v['mask_path'])) for v in new['views'])
        assert counts == {('train', True): 259, ('train', False): 91, ('val', True): 41, ('val', False): 9}
        with np.load(LEGACY/'init_points.npz') as a, np.load(DATASET/'init_points.npz') as b:
            assert b['pixel_protocol'].item() == 'colmap_corner_v2'
            assert set(b.files) == set(a.files) | {'pixel_protocol'}
            geometry_keys = [key for key in a.files if key not in ('colors', 'semantic_counts')]
            assert all(a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key]) for key in geometry_keys)
            train_ids = {v['image_id'] for v in new['views'] if v['split'] == 'train'}
            assert set(b['observation_image_ids']).issubset(train_ids)
            counts_changed = (a['semantic_counts'] != b['semantic_counts']).any(axis=1)
            appearance = {'color_rows_changed': int((a['colors'] != b['colors']).any(axis=1).sum()),
                          'semantic_count_rows_changed': int(counts_changed.sum()),
                          'points': len(b['points']), 'observations': len(b['observation_image_ids']),
                          'geometry_arrays_exact': geometry_keys}
        artifact_hashes = hashes(DATASET)
        prepared_bytes = sum(p.stat().st_size for p in DATASET.rglob('*') if p.is_file())
        audit = {'status': 'passed', 'legacy_files_untouched_sha_exact': True,
                 'same_400_view_names_ids_splits_K_original_and_used_poses_and_covariances': True,
                 'same_official_source_cameras': True, 'same_preparation_geometry_options': True,
                 'train_views': 350, 'validation_views': 50, 'train_labeled': 259, 'validation_labeled': 41,
                 'all_point_tracks_geometry_covariance_reprojection_arrays_exact': True,
                 'no_validation_observation_in_seed': True, 'new_npz_profile': 'colmap_corner_v2',
                 'appearance_changes': appearance, 'prepared_bytes': prepared_bytes,
                 'actual_under_prepared_budget': prepared_bytes <= budget_mib['prepared_dataset']*MIB,
                 'remaining_free_bytes': shutil.disk_usage(ROOT).free,
                 'interpretation': 'Grid protocol/data correctness engineering experiment, not novelty. v2 fingerprints must not be merged with or directly paired-bootstrap against legacy.'}
        (OUTPUT/'comparison_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        (OUTPUT/'prepared_hashes.json').write_text(json.dumps(artifact_hashes, indent=2)+'\n')
        receipt.update(status='completed', completed_utc=datetime.now(UTC).isoformat(),
                       manifest_sha256=digest(DATASET/'manifest.json'), init_points_sha256=digest(DATASET/'init_points.npz'),
                       comparison_audit_sha256=digest(OUTPUT/'comparison_audit.json'),
                       prepared_hashes_sha256=digest(OUTPUT/'prepared_hashes.json'), training_started=False)
        print(json.dumps(audit, indent=2))
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
