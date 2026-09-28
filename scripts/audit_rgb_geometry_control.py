"""Independent RGB-only saved-record/endpoint audit; no producer imports or pixels.

Per-attempt gradients and means are not saved. Their reported directional dots
and caps are checked as records, not independently reconstructed CUDA evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np

ARMS = ('rgb',)
CAP = 1/64
EXPECTED_COUNTS = {'batch_attempts': 148, 'training_view_pairs': 2072,
                   'description_view_pairs': 518, 'raster': 5180,
                   'means_vjp': 2072, 'target_decodes': 777}
MAX_ERROR = 0.


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def ahash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def close(actual, expected, reason, atol=1e-12, rtol=1e-10):
    global MAX_ERROR
    require(np.isfinite(actual) and np.isfinite(expected), reason+' finite')
    error = abs(float(actual)-float(expected))
    require(error <= atol+rtol*abs(float(expected)), reason+' differs')
    MAX_ERROR = max(MAX_ERROR, error)


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)
    elif isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite saved scalar')


def tree(folder):
    folder = Path(folder)
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def schedule(names):
    require(len(names) == len(set(names)) == 259 and names == sorted(names), 'Fixed259 population')
    rng = random.Random(42); result = []
    for epoch in range(4):
        order = list(names); rng.shuffle(order)
        result.extend({'round': epoch, 'names': order[j:j+7]} for j in range(0, 259, 7))
    return result


def check_rows(record, names, support=None, *, cm=False):
    require(record['complete'] is True and record['names'] == names
            and [r['name'] for r in record['rows']] == names, 'Incomplete/misordered population')
    finite(record)
    for row in record['rows']:
        require(row['rgb_mse'] >= 0 and row['raw_ce'] >= 0, 'Negative loss')
        require(type(row['rgb_pixels']) is int and type(row['semantic_pixels']) is int
                and 0 < row['semantic_pixels'] <= row['rgb_pixels'], 'Invalid target support')
        if support is not None:
            for key in ('rgb_pixels', 'semantic_pixels'):
                require(row[key] == support[row['name']][key], 'Changed target support')
        if cm:
            matrix = np.asarray(row['confusion_matrix'])
            require(matrix.shape == (5, 5) and matrix.dtype.kind in 'iu' and (matrix >= 0).all()
                    and int(matrix.sum()) == row['semantic_pixels'], 'Invalid saved per-view confusion')
            if support is not None and 'confusion_matrix' in support[row['name']]:
                require(np.array_equal(matrix.sum(1), np.asarray(support[row['name']]['confusion_matrix']).sum(1)),
                        'Changed GT class support')
    for key in ('rgb_mse', 'raw_ce'):
        close(record[key], math.fsum(r[key] for r in record['rows'])/len(names), 'Equal-view '+key)
    if not cm:
        return None
    total = np.sum([r['confusion_matrix'] for r in record['rows']], axis=0, dtype=np.int64)
    require(np.array_equal(total, np.asarray(record['confusion_matrix'])), 'Pooled CM differs')
    denominator = total.sum(0)+total.sum(1)-np.diag(total)
    ious = [int(total[k, k])/int(denominator[k]) if denominator[k] else None for k in range(5)]
    require(len(record['iou']) == 5, 'Five class IoUs required')
    for actual, expected in zip(record['iou'], ious, strict=True):
        if expected is None:
            require(actual is None, 'Absent class is not zero')
        else:
            close(actual, expected, 'Class IoU')
    mean = math.fsum(x for x in ious if x is not None)/sum(x is not None for x in ious)
    close(record['miou_all'], mean, 'Pooled mIoU')
    return {'rgb_mse': record['rgb_mse'], 'raw_ce': record['raw_ce'],
            'miou_all': mean, 'cable_iou': ious[2], 'iou': ious}


def check_trace(rows, batches, arm, base_hash, support, point_count, scene_scale):
    require(arm in ARMS and len(rows) == len(batches), 'Incomplete arm attempts')
    current = base_hash; accepted = 0; sum_caps = 0.; dots = []
    for index, (row, batch) in enumerate(zip(rows, batches, strict=True), 1):
        finite(row)
        require(row['attempt'] == index and row['round'] == batch['round']
                and row['names'] == batch['names'], 'Attempt order differs')
        before, after = row['baseline'], row['candidate']
        check_rows(before, batch['names'], support); check_rows(after, batch['names'], support)
        require(before['means_sha256'] == current, 'Accepted/rejected means continuity differs')
        tau = max(1e-6*before['rgb_mse'], 1e-12)
        gates = {'semantic_decreases': after['raw_ce'] < before['raw_ce'],
                 'batch_rgb_protected': after['rgb_mse'] <= before['rgb_mse']+tau,
                 'every_view_rgb_protected': all(b['rgb_mse'] <= 1.001*a['rgb_mse']
                    for a, b in zip(before['rows'], after['rows'], strict=True))}
        expected = arm != 'trust' or all(gates.values())
        require(type(row['accepted']) is bool and row['accepted'] == expected
                and row['acceptance']['criteria'] == gates
                and row['acceptance']['control_finite_commit'] == (arm != 'trust'), 'Actual acceptance gate differs')
        close(row['acceptance']['rgb_tolerance'], tau, 'Acceptance RGB tolerance')
        for task, key in (('rgb', 'rgb_mse'), ('semantic', 'raw_ce')):
            close(row['acceptance']['candidate_minus_baseline_'+task], after[key]-before[key], 'Actual task change')
        p = row['proposal']
        require(p['mode'] == ('semantic' if arm == 'trust' else arm)
                and p['optimizer_proposed_step'] == accepted+1, 'Rollback Adam step differs')
        require(p['adam_betas'] == [.9, .999] and p['adam_eps'] == 1e-15
                and p['weight_decay'] == 0 and p['first_order_RGB_guarantee'] is False, 'Optimizer identity changed')
        close(p['learning_rate'], 1.6e-6*scene_scale, 'Means LR')
        require(p['mahalanobis_cap'] == CAP and 0 <= p['actual_mahalanobis_max'] <= CAP, 'Reported per-step cap violated')
        for key in ('adam_proposed_nonzero_point_count', 'scaled_point_count',
                    'fp32_overcap_restored_point_count', 'cap_rounded_to_before_point_count', 'actual_nonzero_point_count'):
            require(type(p[key]) is int and 0 <= p[key] <= point_count, 'Invalid point counter')
        require(type(p['actual_nonzero_coordinate_count']) is int
                and p['actual_nonzero_point_count'] <= p['actual_nonzero_coordinate_count'] <= 3*p['actual_nonzero_point_count'], 'Invalid coordinate counter')
        require((after['means_sha256'] == before['means_sha256']) == (p['actual_nonzero_point_count'] == 0), 'No-op hash mismatch')
        # Saved dots are assertions from the producer; no high-dimensional g was saved.
        dots.append([p['rgb_dot_actual_delta'], p['semantic_dot_actual_delta']])
        require(np.isfinite(dots[-1]).all(), 'Invalid recorded actual-displacement dots')
        if expected:
            current = after['means_sha256']; accepted += 1; sum_caps += p['actual_mahalanobis_max']
        require(row['accepted_optimizer_steps'] == accepted, 'Accepted-step counter differs')
    return {'attempts': len(rows), 'accepted': accepted, 'rejected': len(rows)-accepted,
            'last_means_sha256': current, 'sum_accepted_recorded_max_step': sum_caps,
            'recorded_positive_rgb_dots': sum(r[0] > 0 for r in dots),
            'recorded_positive_semantic_dots': sum(r[1] > 0 for r in dots),
            'actual_dot_scope': 'saved finite scalars only; per-step high-dimensional gradients not saved'}


def cumulative_displacement(base, endpoint, quats, log_scales):
    require(base.dtype == endpoint.dtype == np.float32 and base.shape == endpoint.shape
            and base.ndim == 2 and base.shape[1] == 3 and np.isfinite(endpoint).all(), 'Bad endpoint means')
    q = np.asarray(quats, np.float64); q = q/np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]).transpose(2, 0, 1)
    delta = endpoint.astype(np.float64)-base.astype(np.float64)
    local = np.einsum('nji,nj->ni', rotation, delta)*np.exp(-np.asarray(log_scales, np.float64))
    require(np.isfinite(local).all(), 'Nonfinite cumulative metric')
    return delta, float(np.linalg.norm(local, axis=1).max()), float(np.linalg.norm(delta, axis=1).max())


def audit(plan_path, expected):
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only audit required')
    require(sha(plan_path) == expected, 'Unexpected plan SHA')
    plan = read(plan_path); run = Path(plan['output'])
    # Natural completion is verified before checkpoint, optimizer or endpoint loads.
    launch = read(run/'launch_receipt.json'); receipt = read(run/'execution_receipt.json')
    require(launch['status'] == receipt['status'] == 'completed' and launch['natural_completion'] is True
            and launch['exit_code'] == 0 and launch['plan_sha256'] == receipt['plan_sha256'] == expected
            and launch['execution_receipt_sha256'] == sha(run/'execution_receipt.json'), 'Natural completed execution required')
    produced = {str(run/name): sha(run/name) for name in ('launch_receipt.json', 'execution_receipt.json')}
    spec = plan['specification']
    require(spec['protocol'] == 'rgb_geometry_control_v1' and spec['arms'] == list(ARMS)
            and spec['point_count'] == 498136 and spec['attempts_per_arm'] == 148, 'Unexpected fixed protocol')
    require(plan['interpretation'] == receipt['interpretation'] and receipt['interpretation']['old_fd_gate_passed'] is False
            and receipt['interpretation']['old_fd_passed'] == 3 and receipt['interpretation']['old_fd_count'] == 12
            and receipt['interpretation']['derivative_certified'] is False and receipt['adoption'] == 'none', 'Interpretation changed')
    bound = {**plan['input_hashes'], **plan['installed_sources'],
             plan['expected_gsplat_binary']['path']: plan['expected_gsplat_binary']['sha256']}
    for path, digest in bound.items():
        require(sha(path) == digest, 'Bound input/dependency changed: '+path)
    require(tree(plan['source_snapshot']) == plan['source_hashes'], 'Frozen source changed')
    for record in receipt['actual_imports'].values():
        path = Path(record['path']); rel = str(path.relative_to(plan['source_snapshot']))
        require(sha(path) == record['sha256'] == plan['source_hashes'][rel], 'Actual imported source differs')
    require(receipt['actual_gsplat_binary'] == plan['expected_gsplat_binary'] and receipt['sources_inputs_unchanged']
            and receipt['numerics_actual'] == plan['numerics'] and receipt['numerics_restored']
            and receipt['restoration']['state_exact'] and receipt['restoration']['flags_modes_gradients_restored']
            and receipt['restoration']['state_before'] == receipt['restoration']['state_after'], 'Restoration/runtime attestation differs')
    require(receipt['counts'] == EXPECTED_COUNTS, 'Compute budget differs')
    for key in ('head_calls', 'teacher_calls', 'val_views', 'production_checkpoint_writes'):
        require(receipt[key] == 0, 'Scope violation '+key)
    names = sorted(v['name'] for v in plan['training_views']); batches = schedule(names)
    require(plan['batches'] == batches, 'Fixed shuffle schedule differs')
    base_state = torch.load(plan['base_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    model = base_state['model']; base = model['splats.means'].numpy()
    require(base.shape == (498136, 3) and base.dtype == np.float32 and np.isfinite(base).all(), 'Base means')
    base_hash = ahash(base); require(receipt['base_means_sha256'] == base_hash, 'Original means hash')
    manifest = read(plan['manifest']); train = [v for v in manifest['views'] if v['split'] == 'train']
    require([v['name'] for v in train] == plan['all_training_camera_names']
            and np.array_equal(base_state['training_cameras'].numpy(), np.asarray([v['w2c_original'] for v in train], np.float32)), '350 original camera arrays')
    with np.load(plan['q_delta'], allow_pickle=False) as qfile:
        q = qfile['q_renderer']
        require(q.dtype == np.float32 and q.shape == (len(base), 5)
                and np.array_equal(q, qfile['q_master'].astype(np.float32)), 'q renderer cast identity')
        require(receipt['q_renderer_sha256'] == ahash(q), 'q hash differs')
    fixed = receipt['frozen_state_hashes']
    for key, digest in fixed.items():
        require(key != 'splats.means', 'Means included in frozen non-means scope')
        if key in model:
            require(ahash(model[key].numpy()) == digest, 'Original non-means tensor differs: '+key)
        else:
            require(key == 'semantic_prior_counts', 'Unexpected runtime-only model tensor')
            require(digest == ahash(np.zeros((len(base), 5), np.float32)), 'Runtime-only prior buffer is not original zero initialization')
    require(set(model)-{'splats.means'} <= set(fixed), 'Original non-means tensor omitted')
    require(fixed == {k: v for k, v in receipt['restoration']['state_before'].items() if k != 'splats.means'}, 'Runtime frozen identity')
    baseline_path = Path(receipt['baseline']['path'])
    require(baseline_path == run/'baseline_train.json' and sha(baseline_path) == receipt['baseline']['sha256'], 'Baseline identity')
    produced[str(baseline_path)] = receipt['baseline']['sha256']
    baseline = read(baseline_path); initial = check_rows(baseline, names, cm=True)
    require(baseline['means_sha256'] == base_hash, 'Baseline means')
    support = {row['name']: row for row in baseline['rows']}; results = {}
    require([a['arm'] for a in receipt['arms']] == list(ARMS), 'Arm population differs')
    for entry in receipt['arms']:
        arm = entry['arm']; directory = run/arm; path = directory/'training_receipt.json'
        require(str(path) == entry['path'] and sha(path) == entry['sha256'], 'Arm receipt identity')
        produced[str(path)] = entry['sha256']
        stage = read(path); finite(stage)
        require(stage['status'] == 'completed' and stage['arm'] == arm and stage['plan_sha256'] == expected
                and stage['base_means_sha256'] == base_hash and stage['q_renderer_sha256'] == receipt['q_renderer_sha256']
                and stage['frozen_state_hashes'] == fixed and stage['stop_reason'] == 'fixed_148_attempt_schedule_finished'
                and stage['converged_claimed'] is False and not stage['production_checkpoint']
                and not stage['ordinary_resume_supported'], 'Stage identity/scope')
        for filename, field in (('attempts.jsonl', 'attempts_sha256'), ('means_delta.npz', 'means_delta_sha256'),
                                ('optimizer_state.pt', 'optimizer_state_sha256'), ('endpoint_train.json', 'endpoint_sha256')):
            require(sha(directory/filename) == stage[field], 'Stage artifact changed')
            produced[str(directory/filename)] = stage[field]
        rows = [json.loads(line) for line in (directory/'attempts.jsonl').read_text().splitlines()]
        trace = check_trace(rows, batches, arm, base_hash, support, len(base), base_state['scene_scale'])
        for key in ('attempts', 'accepted', 'rejected'):
            require(stage[key] == trace[key], 'Stage count differs')
        require(stage['final_means_sha256'] == trace['last_means_sha256'], 'Last accepted endpoint hash')
        with np.load(directory/'means_delta.npz', allow_pickle=False) as delta_file:
            require(np.array_equal(delta_file['base_means'], base), 'Arm base differs from original checkpoint')
            endpoint = delta_file['means_final']; require(ahash(endpoint) == stage['final_means_sha256'], 'Endpoint means SHA')
            delta, radius, distance = cumulative_displacement(base, endpoint, model['splats.quats'].numpy(), model['splats.log_scales'].numpy())
            require(np.array_equal(delta, delta_file['actual_delta']), 'Actual FP32 delta differs')
        close(stage['cumulative_mahalanobis_max'], radius, 'Cumulative metric', atol=1e-9)
        close(stage['cumulative_world_l2_max'], distance, 'Cumulative world length')
        require(radius <= trace['sum_accepted_recorded_max_step']+1e-9
                and radius <= trace['accepted']*CAP+1e-9, 'Cumulative step-cap triangle bound violated')
        optimizer = torch.load(directory/'optimizer_state.pt', map_location='cpu', weights_only=True)
        require(len(optimizer['param_groups']) == 1 and len(optimizer['param_groups'][0]['params']) == 1, 'Means-only optimizer')
        group = optimizer['param_groups'][0]
        close(group['lr'], 1.6e-6*base_state['scene_scale'], 'Endpoint Adam LR')
        require(tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-15 and group['weight_decay'] == 0, 'Endpoint Adam config')
        steps = []
        for state in optimizer['state'].values():
            require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Unexpected Adam state')
            for key, value in state.items():
                require(value.device.type == 'cpu' and bool(torch.isfinite(value).all()), 'Nonfinite Adam tensor')
                if key != 'step':
                    require(value.dtype == torch.float32 and value.shape == base.shape, 'Adam moment row shape')
            require(bool((state['exp_avg_sq'] >= 0).all()), 'Negative Adam second moment')
            steps.append(int(state['step']))
        require(steps == stage['optimizer_steps'] and steps == ([trace['accepted']] if trace['accepted'] else []), 'Final Adam committed steps')
        endpoint_record = read(directory/'endpoint_train.json'); summary = check_rows(endpoint_record, names, support, cm=True)
        require(endpoint_record['means_sha256'] == stage['final_means_sha256'], 'Endpoint score identity')
        results[arm] = {**summary, **trace, 'cumulative_mahalanobis_max': radius,
                       'endpoint_mean_sha256': stage['final_means_sha256'],
                       'change_from_initial': {k: summary[k]-initial[k] for k in ('rgb_mse', 'raw_ce', 'miou_all', 'cable_iou')}}
    for path, digest in bound.items():
        require(sha(path) == digest, 'Input changed during independent audit')
    for path, digest in produced.items():
        require(sha(path) == digest, 'Saved producer output changed during audit')
    require(tree(plan['source_snapshot']) == plan['source_hashes'] and not torch.cuda.is_initialized(), 'Source/CUDA audit boundary')
    return {'status': 'passed', 'plan_sha256': expected, 'execution_receipt_sha256': sha(run/'execution_receipt.json'),
            'source_count': len(plan['source_hashes']), 'input_count': len(plan['input_hashes']),
            'counts': receipt['counts'], 'initial': initial, 'arms': results, 'maximum_numeric_difference': MAX_ERROR,
            'pixel_decodes': 0, 'renders': 0, 'new_vjps': 0, 'cuda_initialized': False,
            'before_after_source_input_hashes_exact': True, 'scope': 'saved TRAIN description only; no adoption',
            'limitations': ['No target pixels or CUDA outputs independently regenerated.',
                'Per-step high-dimensional means/gradients/moments are absent: dots and individual caps are recorded assertions.',
                'Saved objective/gate and means-hash continuity, final Adam steps and cumulative displacement independently checked.',
                'Non-means runtime restoration is producer attestation checked against original checkpoint hashes.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    require(not args.output.exists(), 'Do not overwrite an audit attempt')
    started = time.monotonic(); result = {}
    def timeout(*_):
        raise TimeoutError('Independent CPU audit120s budget')
    previous = signal.signal(signal.SIGALRM, timeout); signal.alarm(120)
    try:
        result = audit(args.plan, args.expected_plan_sha256)
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        result.update(elapsed_seconds=time.monotonic()-started, auditor_sha256=sha(__file__))
        payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
        with args.output.open('x') as handle:
            handle.write(payload)


if __name__ == '__main__':
    main()
