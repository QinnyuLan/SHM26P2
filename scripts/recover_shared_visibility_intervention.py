"""CPU-only measurement recovery from a failed run's already saved predictions.

Never imports/calls a renderer, changes the original run, or invents missing
runtime restoration/count/timestamp evidence. Prepare binds bytes only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ORIGINAL = Path('/mnt/data/SHM2026/runs/shared_visibility_intervention_v1')
ORIGINAL_PLAN_SHA = '8aa0538bb24c9f6189462d79fdf2a77cfc9fe734fcd76d00857b82090badb7fd'
ORIGINAL_WORKER_SHA = '11adcf6d165cf1c5f4e857b1b8396577bfd432d63b0bbb66aa1a35f1da1f5447'
PROTOCOL = 'shared_visibility_saved_arrays_cpu_recovery_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def numpy_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported JSON object: {type(value).__name__}')


def write(path, value, replace=False):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False, default=numpy_scalar)+'\n'
    if replace:
        temp = path.with_suffix('.tmp'); temp.write_text(text); temp.replace(path)
    else:
        with path.open('x') as stream:
            stream.write(text)


def expected_arrays(original):
    folder = Path(original['output'])/'predictions'; shape = (989, 1320)
    result = {}
    require(len(original['views']) == 8 and original['specification']['order'] == ['base', 'minus', 'plus']*2, 'Wrong fixed population/order')
    for view in original['views']:
        for index, case in enumerate(original['specification']['order']):
            for key, suffix, dtype in [('rgb', (3,), 'float32'), ('raw', (5,), 'float32'),
                                       ('alpha', (), 'float32'), ('rgb_alpha', (), 'float32'), ('raw_ids', (), 'uint8')]:
                result[str(folder/f"{view['name']}.{index}.{case}.{key}.npy")] = {'shape': list(shape+suffix), 'dtype': dtype}
        for key in ('mass', 'contribution_total', 'contribution_alpha'):
            result[str(folder/f"{view['name']}.{key}.npy")] = {'shape': list(shape), 'dtype': 'float32'}
    require(len(result) == 264, 'Expected264 distinct prediction arrays')
    return result


def bind_original(original):
    require(digest(ORIGINAL/'plan.json') == ORIGINAL_PLAN_SHA, 'Original immutable plan differs')
    for path, sha in original['input_hashes'].items():
        require(digest(path) == sha, 'Original input changed: '+path)
    folder = Path(original['source_snapshot'])
    actual = {str(p.relative_to(folder)): digest(p) for p in folder.rglob('*.py')}
    require(actual == original['source_hashes'], 'Original immutable source differs')


def prepare():
    output = ORIGINAL/'recovery'
    require(not output.exists(), 'Do not overwrite a recovery attempt')
    require(not (ORIGINAL/'audit.json').exists(), 'Expected absent original audit; never replace it')
    original = read(ORIGINAL/'plan.json'); bind_original(original)
    receipt_path = ORIGINAL/'execution_receipt.json'; receipt = read(receipt_path)
    require(receipt['status'] == 'failed' and receipt['exit_code'] == 1
            and receipt['plan_sha256'] == ORIGINAL_PLAN_SHA, 'Original GPU run must remain failed')
    log_path = ORIGINAL/'worker.log'
    require('TypeError: Object of type bool is not JSON serializable' in log_path.read_text(), 'Unexpected original failure')
    worker_path = Path(original['source_snapshot'])/'audit_shared_visibility_intervention.py'
    require(digest(worker_path) == ORIGINAL_WORKER_SHA, 'Original entrypoint differs')
    expected = expected_arrays(original)
    require({str(p) for p in (ORIGINAL/'predictions').glob('*.npy')} == set(expected), 'Missing/extra prediction arrays; no resampling')
    inventory = []
    for path, metadata in expected.items():
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        require(list(array.shape) == metadata['shape'] and str(array.dtype) == metadata['dtype'], 'NPY header differs: '+path)
        inventory.append(dict(path=path, sha256=digest(path), **metadata))
    inputs = dict(original['input_hashes'])
    for path in [ORIGINAL/'plan.json', receipt_path, log_path, *[Path(original['source_snapshot'])/p for p in original['source_hashes']]]:
        inputs[str(path)] = digest(path)
    inputs.update({item['path']: item['sha256'] for item in inventory})
    output.mkdir()
    snapshot = output/'source_snapshot'; snapshot.mkdir()
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    script = snapshot/Path(__file__).name
    plan = {'status': 'cpu_prepared_measurement_recovery_not_executed', 'protocol': PROTOCOL,
            'created_utc': datetime.now(UTC).isoformat(), 'output': str(output),
            'original_plan': str(ORIGINAL/'plan.json'), 'original_plan_sha256': ORIGINAL_PLAN_SHA,
            'original_execution_receipt': str(receipt_path), 'original_worker_log': str(log_path),
            'original_entrypoint': str(worker_path), 'original_entrypoint_sha256': ORIGINAL_WORKER_SHA,
            'source_snapshot': str(snapshot), 'source_hashes': {script.name: digest(script)},
            'input_hashes': inputs, 'prediction_arrays': inventory,
            'original_specification': original['specification'], 'metadata_only_prepare': True,
            'new_renders': 0, 'new_backward': 0, 'new_optimizer_steps': 0,
            'limits': ['Original GPU execution remains failed exit1.',
                       'NPY hashes are first bound after failure, not recovered original runtime hashes.',
                       'No original all-state/camera restoration hashes, actual raster counts or phase timestamps survived.',
                       '264 files match planned predictions; this is file completeness, not restored execution attestation.',
                       'Recompute fixed measurements/gates only. Do not alter original gates, arrays, inputs or records.']}
    write(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    require(plan['protocol'] == PROTOCOL and plan['original_plan_sha256'] == ORIGINAL_PLAN_SHA, 'Wrong recovery protocol')
    snapshot = Path(plan['source_snapshot'])
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Use frozen recovery entrypoint')
    require({str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')} == plan['source_hashes'], 'Recovery source changed')
    for path, sha in plan['input_hashes'].items():
        require(digest(path) == sha, 'Bound recovery input changed: '+path)
    require(not (ORIGINAL/'audit.json').exists(), 'Original audit must stay absent')
    original = read(plan['original_plan']); bind_original(original)
    require(original['specification'] == plan['original_specification'], 'Numeric gates changed')
    require({item['path'] for item in plan['prediction_arrays']} == set(expected_arrays(original)), 'Recovery arrays differ')
    return original


def execute(plan_path):
    # Explicitly prevent a GPU context, including inside the frozen helper import.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    started = time.monotonic(); plan = read(plan_path); original = verify(plan)
    output = Path(plan['output']); receipt_path = output/'execution_receipt.json'
    require(not receipt_path.exists() and not (output/'recovered_measurements.json').exists(), 'No recovery overwrite/retry')
    receipt = {'status': 'running', 'protocol': PROTOCOL, 'plan_sha256': digest(plan_path),
               'original_gpu_status': 'failed', 'new_renders': 0, 'new_backward': 0, 'new_optimizer_steps': 0}
    write(receipt_path, receipt)
    try:
        spec = importlib.util.spec_from_file_location('frozen_visibility_measurements', plan['original_entrypoint'])
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        import torch
        require(not torch.cuda.is_initialized(), 'Recovery must remain CPU-only')
        require(helper.SPEC == original['specification'], 'Frozen helper gates changed')
        predictions = ORIGINAL/'predictions'; rows = []
        for view in original['views']:
            def load(suffix, view=view):
                return np.load(predictions/f"{view['name']}.{suffix}.npy", mmap_mode='r', allow_pickle=False)
            mass = load('mass'); total = load('contribution_total'); alpha = load('contribution_alpha')
            _, mass_audit = helper.verify_mass(np.stack([mass, total], -1), alpha, load('0.base.alpha'), load('0.base.rgb_alpha'))
            rgb, labels, valid = helper.read_targets(view, complete=True)
            scores, alpha_differences = [], []
            for index, case in enumerate(helper.ORDER):
                value = {key: load(f'{index}.{case}.{key}') for key in ('rgb', 'raw', 'alpha', 'rgb_alpha', 'raw_ids')}
                require(all(np.isfinite(v).all() for v in value.values()) and np.isin(value['raw_ids'], np.arange(5)).all(), 'Invalid saved array')
                error = float(np.max(np.abs(value['rgb_alpha'].astype(np.float64)-value['alpha'])))
                require(error <= helper.SPEC['mass_atol'], 'Saved RGB/semantic alpha contract failed')
                alpha_differences.append(error)
                scores.append(helper.score_prediction(value, mass, rgb, labels, valid, original['class_weights']))
            rows.append({'name': view['name'], 'mass_contract': mass_audit,
                         'rgb_semantic_alpha_max_errors': alpha_differences,
                         'coverage': helper.coverage(mass, valid & (labels < 5)), 'predictions': scores})
        summary = helper.summarize(rows)
        require(not torch.cuda.is_initialized(), 'Unexpected CUDA context')
        verify(plan)
        report = {'status': 'recovered_measurements_only', 'protocol': PROTOCOL,
                  'recovery_plan_sha256': digest(plan_path), 'original_plan_sha256': ORIGINAL_PLAN_SHA,
                  'original_execution': {'status': 'failed', 'exit_code': 1, 'receipt': plan['original_execution_receipt'],
                                         'receipt_sha256': plan['input_hashes'][plan['original_execution_receipt']]},
                  'rows': rows, 'summary': summary, 'prediction_array_count': len(plan['prediction_arrays']),
                  'runtime_evidence': {'all_state_restored_exact': None, 'all_state_before_after_hashes': None,
                                       'camera_unchanged': None, 'executed_raster_calls': None,
                                       'prediction_before_gt_runtime_timestamps': None},
                  'planned_calls_not_execution_attestation': {'scene': 48, 'raster': 104},
                  'new_renders': 0, 'new_backward': 0, 'new_optimizer_steps': 0,
                  'cuda_initialized': False, 'original_inputs_and_arrays_unchanged': True,
                  'serialization_recovery': 'Frozen numerical helper unchanged; only np.generic JSON values converted with item(), allow_nan=False.',
                  'limitations': plan['limits'], 'elapsed_seconds': time.monotonic()-started}
        write(output/'recovered_measurements.json', report)
        receipt.update(status='completed_cpu_recovery_only', recovered_measurements_sha256=digest(output/'recovered_measurements.json'),
                       decision=summary['decision'])
    except Exception as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic()-started
        write(receipt_path, receipt, replace=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(); mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--execute'); mode.add_argument('--verify')
    args = parser.parse_args()
    if args.prepare:
        print(prepare())
    elif args.verify:
        verify(read(args.verify)); print('CPU recovery source/input/header bindings verified')
    else:
        print(json.dumps(execute(args.execute), indent=2, allow_nan=False, default=numpy_scalar))


if __name__ == '__main__':
    main()
