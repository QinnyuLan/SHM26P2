"""Freeze and execute one TRAIN-only CPU track-label assumption diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
SPEC = {'protocol': 'train_track_semantics_diagnostic_v1', 'primary': 'original_grid_floor',
        'sensitivity': 'legacy_grid_rint', 'bootstrap_repeats': 2000, 'seed': 20260927,
        'internal_timeout_seconds': 900, 'external_timeout_seconds': 960,
        'gpu_calls': 0, 'new_val_pixel_decodes': 0, 'model_adoption': False}
VERSIONS = ('numpy', 'scipy', 'pillow', 'opencv-python-headless')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x') as f:
        f.write(text)


def files(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def population(manifest):
    views = sorted((v for v in manifest['views'] if v['split'] == 'train'), key=lambda v: v['name'])
    require(len(views) == 350 and sum(bool(v.get('mask_path')) for v in views) == 259, 'TRAIN population changed')
    require(len({v['image_id'] for v in views}) == 350 and len({v['name'] for v in views}) == 350,
            'TRAIN image ID/name must be unique')
    return views


def prepare(output):
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite an experiment')
    collector = module(ROOT/'scripts/collect_train_track_semantics.py', 'collector_spec')
    analysis = module(ROOT/'src/bridge_rgs/track_semantic_audit.py', 'analysis_spec')
    manifest_path = ROOT/'artifacts/prepared/manifest.json'
    manifest = read(manifest_path); train = population(manifest)
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(); value = sha(path)
        require(expected is None or value == expected, f'Changed input: {path}')
        inputs[str(path)] = value
        return str(path)

    manifest_path = bind(manifest_path, '551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa')
    init = bind(ROOT/'artifacts/prepared/init_points.npz', 'f407b923d51c733b0253bfd6d7f7c37f573ab3bb1abd1a3fd7d4d5b4125ae911')
    images = bind(Path(manifest['dataset_root'])/'camera_parameters/images.txt')
    cameras = bind(Path(manifest['dataset_root'])/'camera_parameters/cameras.txt')
    for v in train:
        require(v['w2c'] == v['w2c_original'], 'Modified prepared pose')
        require((v['width'], v['height']) == (1320, 989), 'Changed native grid')
        if v.get('mask_path'):
            for key in ('source_annotation_path', 'mask_path', 'valid_path'):
                bind(v[key])
    for p in (ROOT/'uv.lock', ROOT/'artifacts/prepared/geometry_audit.json'):
        bind(p)
    provenance = ROOT/'runs/h3_moments/02_cross/source_snapshot/bridge_rgs'
    for name in ('prepare.py', 'data.py', 'geometry.py'):
        bind(provenance/name)
    snapshot = output/'source_snapshot'
    (snapshot/'bridge_rgs').mkdir(parents=True)
    for name in ('__init__.py', 'track_semantic_audit.py'):
        shutil.copy2(ROOT/'src/bridge_rgs'/name, snapshot/'bridge_rgs'/name)
    for p in (Path(__file__), ROOT/'scripts/collect_train_track_semantics.py',
              ROOT/'tests/test_train_track_semantics.py', ROOT/'tests/test_track_semantic_audit.py',
              ROOT/'tests/test_train_track_diagnostic.py', ROOT/'docs/train_track_semantics_protocol.md'):
        shutil.copy2(p, snapshot/p.name)
    plan = {'specification': SPEC, 'collector_specification': collector.SPEC,
            'analysis_specification': analysis.SPEC, 'output': str(output),
            'source_snapshot': str(snapshot), 'source_hashes': files(snapshot), 'input_hashes': inputs,
            'manifest': manifest_path, 'init_points': init, 'colmap_images': images, 'colmap_cameras': cameras,
            'train_images': [{'image_id': int(v['image_id']), 'name': v['name']} for v in train],
            'runtime_versions': {name: importlib.metadata.version(name) for name in VERSIONS},
            'prepare_label_decodes': 0, 'prepare_label_file_byte_hashing': True,
            'created_utc': datetime.now(UTC).isoformat()}
    write_new(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json'),
                      'source_files': len(plan['source_hashes']), 'bound_inputs': len(inputs)}), flush=True)


def verify(plan):
    require(plan['specification'] == SPEC, 'Runner specification changed')
    snapshot = Path(plan['source_snapshot'])
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Use frozen runner')
    require(files(snapshot) == plan['source_hashes'], 'Source snapshot changed')
    require({name: importlib.metadata.version(name) for name in VERSIONS} == plan['runtime_versions'],
            'Runtime versions changed')
    for path, expected in plan['input_hashes'].items():
        require(sha(path) == expected, f'Input changed: {path}')
    require('torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only execution required')


def coordinate_sensitivity(arrays):
    a, b = arrays['primary_label'], arrays['legacy_label']
    known_a, known_b = (a >= 0) & (a < 5), (b >= 0) & (b < 5)
    both = known_a & known_b
    strict_a = both & (arrays['reprojection_error'] <= 1) & (arrays['primary_boundary_distance'] > 10)
    cm = np.bincount(5*a[both]+b[both], minlength=25).reshape(5, 5)
    return {'both_known': int(both.sum()), 'primary_only_known': int((known_a & ~known_b).sum()),
            'legacy_only_known': int((known_b & ~known_a).sum()),
            'disagreement': int((both & (a != b)).sum()),
            'cm_rows_primary_columns_legacy': cm.tolist(),
            'primary_strict_both_known': int(strict_a.sum()),
            'primary_strict_disagreement': int((strict_a & (a != b)).sum()),
            'interpretation': 'coordinate/raster support sensitivity, not a trained-model comparison'}


def execute(plan_path):
    plan_path = Path(plan_path).resolve(); plan = read(plan_path)
    output = Path(plan['output']); snapshot = Path(plan['source_snapshot'])
    require(plan_path == output/'plan.json', 'Plan must be in output directory')
    require(not (output/'execution_receipt.json').exists(), 'Never rerun an executed plan')
    verify(plan)
    collector = module(snapshot/'collect_train_track_semantics.py', 'frozen_collector')
    analyzer = module(snapshot/'bridge_rgs/track_semantic_audit.py', 'frozen_analyzer')
    require(collector.SPEC == plan['collector_specification'] and analyzer.SPEC == plan['analysis_specification'],
            'Module specifications differ')
    receipt = {'status': 'running', 'plan_sha256': sha(plan_path), 'started_utc': datetime.now(UTC).isoformat(),
               'gpu_calls': 0, 'model_adoption': False, 'new_val_pixel_decodes': 0}
    started = time.perf_counter(); failure = None

    def timeout_handler(_signum, _frame):
        raise TimeoutError('Fixed CPU diagnostic exceeded internal deadline')

    signal.signal(signal.SIGALRM, timeout_handler); signal.alarm(SPEC['internal_timeout_seconds'])
    try:
        print(json.dumps({'stage': 'collect_actual_train_observations'}), flush=True)
        collection = collector.collect(plan)
        receipt['collection_status'] = collection['status']
        require(collection['status'] == 'completed', 'Collection failed')
        with np.load(output/'observations.npz', allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        result = {'plan_sha256': sha(plan_path), 'coordinate_sensitivity': coordinate_sensitivity(arrays)}
        for kind, label, distance in (('primary', 'primary_label', 'primary_boundary_distance'),
                                       ('legacy', 'legacy_label', 'legacy_boundary_distance')):
            print(json.dumps({'stage': 'analyze', 'protocol': kind}), flush=True)
            result[kind] = analyzer.audit_track_labels(arrays['point_index'], arrays['image_id'], arrays[label],
                arrays['reprojection_error'], arrays[distance], arrays['viewing_ray_world'],
                point_count=len(arrays['point_track_ids']), train_images=plan['train_images'],
                bootstrap_repeats=SPEC['bootstrap_repeats'], seed=SPEC['seed'])
        write_new(output/'analysis.json', result)
        verify(plan)
        receipt.update(status='completed', output_files={name: sha(output/name) for name in
                       ('observations.npz', 'collection_receipt.json', 'analysis.json')})
    except BaseException as exc:  # noqa: BLE001 - preserve evidence, then re-raise
        failure = exc; receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.alarm(0)
        receipt['elapsed_seconds'] = time.perf_counter()-started
        receipt['ended_utc'] = datetime.now(UTC).isoformat()
        write_new(output/'execution_receipt.json', receipt)
        print(json.dumps(receipt), flush=True)
    if failure is not None:
        raise failure


def main():
    p = argparse.ArgumentParser(); group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare'); group.add_argument('--execute')
    args = p.parse_args()
    prepare(args.prepare) if args.prepare else execute(args.execute)


if __name__ == '__main__':
    main()
