"""Freeze and execute a CPU-only, disjoint-track check of existing bounded BA."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
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
TRACKS = Path('/mnt/data/SHM2026/runs/train_track_semantics_v1')
GEOMETRY_KEYS = ('point_positions', 'point_track_ids', 'observation_offsets', 'image_id',
                 'raw_xy', 'undistorted_native_xy', 'saved_track_rms')
SPEC = {
    'protocol': 'ba_track_holdout_v1',
    'solver': 'unchanged bridge_rgs.geometry.bounded_bundle_adjustment',
    'max_points': 500, 'max_nfev': 20, 'ba_seed': 42,
    'geometry_only_keys': list(GEOMETRY_KEYS),
    'internal_timeout_seconds': 600, 'external_timeout_seconds': 900,
    'new_pixel_decodes': 0, 'label_array_loads': 0, 'rgb_array_loads': 0,
    'gpu_calls': 0, 'model_renders': 0, 'model_adoption': False,
    'followup': 'Positive geometry screen may motivate a new matched camera-refinement comparison; '
                'does not reopen the stopped profile pilot or certify a new method.',
}
VERSIONS = ('numpy', 'scipy', 'opencv-python-headless', 'pillow')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    content = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x') as stream:
        stream.write(content)


def save_new(path, arrays):
    require(all(isinstance(v, np.ndarray) and v.dtype.kind != 'O' for v in arrays.values()),
            'Only non-object NumPy arrays may be saved')
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def files(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def load_core(snapshot):
    # The process must not have loaded the editable project package before this switch.
    require('bridge_rgs.ba_holdout' not in sys.modules, 'Core already imported')
    sys.path.insert(0, str(snapshot))
    core = importlib.import_module('bridge_rgs.ba_holdout')
    for name in ('ba_holdout', 'geometry', 'data'):
        module = importlib.import_module(f'bridge_rgs.{name}')
        require(Path(module.__file__).resolve() == Path(snapshot)/'bridge_rgs'/f'{name}.py',
                f'Non-frozen module: {name}')
    return core


def prepare(output):
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite a diagnostic run')
    prior = read(TRACKS/'plan.json')
    receipt = read(TRACKS/'execution_receipt.json')
    require(receipt['status'] == 'completed' and receipt['plan_sha256'] == sha(TRACKS/'plan.json'),
            'Incomplete or changed source diagnostic')
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, f'Changed input: {path}')
        inputs[str(path)] = digest
        return str(path)

    manifest = bind(ROOT/'artifacts/prepared/manifest.json',
                    prior['input_hashes'][str(ROOT/'artifacts/prepared/manifest.json')])
    observations = bind(TRACKS/'observations.npz', receipt['output_files']['observations.npz'])
    bind(ROOT/'artifacts/prepared/init_points.npz',
         prior['input_hashes'][str(ROOT/'artifacts/prepared/init_points.npz')])
    for name in ('plan.json', 'execution_receipt.json', 'collection_receipt.json',
                 'independent_collection_review.json', 'independent_statistics_review.json'):
        bind(TRACKS/name)
    for name in ('independent_collection_review.json', 'independent_statistics_review.json'):
        require(read(TRACKS/name)['status'] == 'passed', 'Source independent review did not pass')
    bind(ROOT/'uv.lock')
    snapshot = output/'source_snapshot'
    (snapshot/'bridge_rgs').mkdir(parents=True)
    for name in ('__init__.py', 'ba_holdout.py', 'geometry.py', 'data.py', 'coordinates.py'):
        shutil.copy2(ROOT/'src/bridge_rgs'/name, snapshot/'bridge_rgs'/name)
    for path in (Path(__file__), ROOT/'scripts/audit_ba_holdout.py',
                 ROOT/'tests/test_ba_holdout.py', ROOT/'tests/test_ba_holdout_runner.py',
                 ROOT/'tests/test_ba_holdout_independent.py', ROOT/'docs/ba_holdout_protocol.md',
                 ROOT/'docs/ba_holdout_protocol_review.md'):
        shutil.copy2(path, snapshot/path.name)
    core = load_core(snapshot)
    plan = {
        'specification': SPEC, 'analysis_specification': core.SPEC,
        'output': str(output), 'source_snapshot': str(snapshot),
        'source_hashes': files(snapshot), 'input_hashes': inputs,
        'manifest': manifest, 'observations': observations,
        'runtime_versions': {name: importlib.metadata.version(name) for name in VERSIONS},
        'thread_environment': {name: '1' for name in
                               ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        'created_utc': datetime.now(UTC).isoformat(),
        'prepare_npz_array_loads': 0,
        'caveat': 'Conditional screen of saved inlier tracks from upstream all-view SfM. '
                  'Fit/check tracks and support/score observations are separated here; '
                  'their earlier selection and supplied calibration were not independent.',
    }
    write_new(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json'),
                      'source_count': len(plan['source_hashes']), 'input_count': len(inputs)}),
          flush=True)


def verify(plan):
    require(plan['specification'] == SPEC, 'Changed runner specification')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name,
            'Execute the frozen runner')
    require(files(plan['source_snapshot']) == plan['source_hashes'], 'Changed frozen source')
    for name, version in plan['runtime_versions'].items():
        require(importlib.metadata.version(name) == version, f'Changed runtime: {name}')
    for name, value in plan['thread_environment'].items():
        require(os.environ.get(name) == value, f'Unexpected thread environment: {name}')
    for path, digest in plan['input_hashes'].items():
        require(sha(path) == digest, f'Changed input: {path}')


def execute(path):
    path = Path(path).resolve()
    plan = read(path)
    output = Path(plan['output'])
    require(path == output/'plan.json', 'Unexpected plan location')
    require(not (output/'execution_started.json').exists(), 'Attempt already started; never retry')
    verify(plan)
    core = load_core(Path(plan['source_snapshot']))
    require(core.SPEC == plan['analysis_specification'], 'Changed core specification')
    started = {'plan_sha256': sha(path), 'started_utc': datetime.now(UTC).isoformat(),
               'pid': os.getpid(), 'argv': sys.argv}
    write_new(output/'execution_started.json', started)
    report = {**started, 'status': 'running', 'gpu_calls': 0, 'new_pixel_decodes': 0,
              'label_array_loads': 0, 'rgb_array_loads': 0, 'model_renders': 0,
              'model_adoption': False, 'thread_environment': plan['thread_environment']}
    start = time.perf_counter()
    failure = None

    def alarm(_signum, _frame):
        raise TimeoutError('Fixed CPU diagnostic deadline exceeded')

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(SPEC['internal_timeout_seconds'])
    try:
        with np.load(plan['observations'], allow_pickle=False) as source:
            arrays = {name: source[name] for name in GEOMETRY_KEYS}
        problem = core.prepare_problem(arrays, read(plan['manifest']))
        save_new(output/'geometry_input.npz', {**arrays,
                 **{f'layout_{key}': value for key, value in problem.layout.items()}})
        print('Geometry-only population and disjoint fit/check layout saved.', flush=True)
        baseline = core.evaluate(problem, {})
        save_new(output/'baseline_screen.npz', baseline['arrays'])
        write_new(output/'baseline_screen.json', baseline['report'])
        print('Fixed baseline eligibility saved before BA.', flush=True)
        fit_start = time.perf_counter()
        overrides, fit_audit = core.run_fit(problem)
        write_new(output/'fit_result.json', {'plan_sha256': sha(path), **fit_audit,
                                             'elapsed_seconds': time.perf_counter()-fit_start})
        train = sorted((v for v in read(plan['manifest'])['views'] if v['split'] == 'train'),
                       key=lambda v: v['name'])
        ids = np.asarray([v['image_id'] for v in train], dtype=np.int64)
        original = np.asarray([v['w2c_original'] for v in train], dtype=np.float64)
        candidate = np.asarray([overrides.get(v['image_id'], v['w2c_original']) for v in train],
                               dtype=np.float64)
        save_new(output/'candidate_poses.npz', {'image_id': ids, 'original': original,
                 'candidate': candidate, 'override_image_id': np.asarray(sorted(overrides), np.int64)})
        print(json.dumps({'stage': 'BA completed', 'fit_audit': {key: fit_audit.get(key) for key in
                          ('accepted', 'success', 'nfev', 'selected_points', 'train_observations',
                           'optimized_cameras', 'rms_native_px_before', 'rms_native_px_after')}}),
              flush=True)
        evaluation = core.evaluate(problem, overrides)
        baseline_names = [name for name in evaluation['arrays'] if name.startswith('baseline_')]
        require(bool(baseline_names), 'Baseline arrays missing')
        for name in baseline_names:
            require(np.array_equal(baseline['arrays'][name], evaluation['arrays'][name], equal_nan=True),
                    f'Baseline population or geometry changed after BA: {name}')
        save_new(output/'holdout_arrays.npz', evaluation['arrays'])
        analysis = core.analyze(problem, evaluation, fit_audit)
        write_new(output/'analysis.json', {'plan_sha256': sha(path), **analysis})
        verify(plan)
        report.update(status='completed', inputs_and_sources_unchanged=True,
                      output_files={name: sha(output/name) for name in
                                    ('geometry_input.npz', 'fit_result.json', 'candidate_poses.npz',
                                     'baseline_screen.npz', 'baseline_screen.json',
                                     'holdout_arrays.npz', 'analysis.json')})
    except BaseException as exc:  # noqa: BLE001 - preserve every attempted run
        failure = exc
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.perf_counter()-start
        report['ended_utc'] = datetime.now(UTC).isoformat()
        write_new(output/'execution_receipt.json', report)
        print(json.dumps(report), flush=True)
    if failure is not None:
        raise failure


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare')
    group.add_argument('--execute')
    args = parser.parse_args()
    prepare(args.prepare) if args.prepare else execute(args.execute)


if __name__ == '__main__':
    main()
