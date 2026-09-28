"""Independent saved-record audit of the two scene-gradient routes, CPU only.

Reuses the prior independent auditor's I/O, CM and geometric metric routines.
No producer math, rendering, target decoding, or gradient reconstruction.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np

COMMON_PATH = Path(__file__).with_name('audit_rgb_geometry_control.py')
COMMON_SHA = 'eea30462d43e4ae424932084ce8add40b1a609d7597c33d2b0d712bd9ee77874'
module_spec = importlib.util.spec_from_file_location('scene_routes_independent_common', COMMON_PATH)
common = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(common)
require, read, sha = common.require, common.read, common.sha
ahash, close, finite, tree = common.ahash, common.close, common.finite, common.tree
schedule, cumulative_displacement = common.schedule, common.cumulative_displacement

ARMS = ('full', 'prior_only')
CAP = 1/64
EXPECTED_COUNTS = {'batch_attempts': 296, 'training_view_pairs': 4144,
                   'description_view_pairs': 777, 'adapter_calls': 4921,
                   'standard_gsplat_calls': 9842, 'low_level_raster_calls': 14763,
                   'means_vjp': 6216, 'head_calls': 6993, 'target_decodes': 777}


def check_rows(record, names, support=None, *, cm=False):
    common.check_rows(record, names, support)
    require(all(row['scene_ce'] >= 0 for row in record['rows']), 'Negative scene CE')
    close(record['scene_ce'], math.fsum(row['scene_ce'] for row in record['rows'])/len(names),
          'Equal-view scene CE')
    if not cm:
        return None
    result = {key: record[key] for key in ('rgb_mse', 'raw_ce', 'scene_ce')}
    for kind in ('raw', 'scene'):
        def with_cm(row, kind=kind):
            return {**row, 'confusion_matrix': row[kind+'_confusion_matrix']}
        projected = {**record, **record[kind], 'rows': [with_cm(r) for r in record['rows']]}
        original = {name: with_cm(row) for name, row in support.items()} if support is not None else None
        summary = common.check_rows(projected, names, original, cm=True)
        result[kind] = {key: summary[key] for key in ('miou_all', 'cable_iou', 'iou')}
    for row in record['rows']:
        require(np.array_equal(np.asarray(row['raw_confusion_matrix']).sum(1),
                               np.asarray(row['scene_confusion_matrix']).sum(1)), 'Raw/scene GT row totals differ')
    return result


def check_path_statistics(record, proposal=None, arm=None):
    """Algebra among saved scalars only, not reconstruction of missing gradients."""
    finite(record)
    norms, dots = record['l2'], record['dots']
    require(set(norms) == {'rgb', 'full', 'prior', 'context'} and min(norms.values()) >= 0,
            'Gradient norm scope')
    # g_full = g_prior + g_context, with independent FP64 reductions in the producer.
    close(dots['rgb_context'], dots['rgb_full']-dots['rgb_prior'], 'RGB path dot closure')
    close(norms['full']**2, norms['prior']**2+norms['context']**2+2*dots['prior_context'],
          'Path squared norm closure')
    if proposal is not None:
        actual = record['actual_dots']
        require(set(actual) == set(norms) and record['actual_delta_l2'] >= 0, 'Actual dot scope')
        close(actual['context'], actual['full']-actual['prior'], 'Actual displacement path closure')
        close(actual['rgb'], proposal['rgb_dot_actual_delta'], 'RGB actual proposal dot')
        key = 'full' if arm == 'full' else 'prior'
        close(actual[key], proposal['semantic_dot_actual_delta'], 'Selected semantic proposal dot')
        close(norms['rgb'], proposal['rgb_gradient_l2'], 'Proposal RGB norm')
        close(.03*norms[key], proposal['weighted_semantic_gradient_l2'], 'Proposal weighted route norm')
        close(.03*dots['rgb_'+key], proposal['task_dot_rgb_weighted_semantic'], 'Proposal route task dot')
        require(proposal['conflicting_original_gradients'] == (proposal['task_dot_rgb_weighted_semantic'] < 0),
                'Reported conflict sign')


def check_trace(rows, batches, arm, base_hash, support, point_count, scene_scale):
    require(arm in ARMS and len(rows) == len(batches) == 148, 'Incomplete arm attempts')
    current = base_hash; sum_caps = 0.; positive = dict.fromkeys(('rgb', 'full', 'prior', 'context'), 0)
    changes = dict.fromkeys(('rgb_mse', 'raw_ce', 'scene_ce'), 0)
    for index, (row, batch) in enumerate(zip(rows, batches, strict=True), 1):
        finite(row)
        require(row['attempt'] == row['accepted_optimizer_steps'] == index
                and row['round'] == batch['round'] and row['names'] == batch['names']
                and row['route'] == arm and row['accepted'] is True, 'Fixed accepted attempt/route differs')
        before, after = row['baseline'], row['candidate']
        check_rows(before, batch['names'], support); check_rows(after, batch['names'], support)
        require(before['means_sha256'] == current, 'Means continuity differs')
        require(all(v['prior_forward_exact'] is True for v in before['rows'])
                and all(v['prior_forward_exact'] is None for v in after['rows']), 'Forward route attestation')
        acceptance = row['acceptance']
        require(acceptance['complete_finite_commit'] is True and acceptance['no_descent_guarantee'] is True,
                'Finite commit scope differs')
        for key in changes:
            difference = after[key]-before[key]
            close(acceptance['candidate_minus_baseline'][key], difference, 'Actual '+key+' change')
            changes[key] += difference > 0
        p = row['proposal']
        require(p['mode'] == 'joint' and p['optimizer_proposed_step'] == index
                and p['adam_betas'] == [.9, .999] and p['adam_eps'] == 1e-15
                and p['weight_decay'] == 0 and p['first_order_RGB_guarantee'] is False, 'Optimizer identity')
        close(p['learning_rate'], 1.6e-6*scene_scale, 'Means LR')
        require(p['mahalanobis_cap'] == CAP and 0 <= p['actual_mahalanobis_max'] <= CAP, 'Reported step cap violated')
        for key in ('adam_proposed_nonzero_point_count', 'scaled_point_count',
                    'fp32_overcap_restored_point_count', 'cap_rounded_to_before_point_count', 'actual_nonzero_point_count'):
            require(type(p[key]) is int and 0 <= p[key] <= point_count, 'Invalid point counter')
        require(type(p['actual_nonzero_coordinate_count']) is int
                and p['actual_nonzero_point_count'] <= p['actual_nonzero_coordinate_count'] <= 3*p['actual_nonzero_point_count'],
                'Invalid coordinate counter')
        require((after['means_sha256'] == before['means_sha256']) == (p['actual_nonzero_point_count'] == 0), 'No-op hash mismatch')
        require(before['path_gradients'] == {k: row['path_gradients'][k] for k in ('l2', 'dots', 'aggregation')},
                'Baseline/path summary identity')
        check_path_statistics(row['path_gradients'], p, arm)
        for key in positive:
            positive[key] += row['path_gradients']['actual_dots'][key] > 0
        current = after['means_sha256']; sum_caps += p['actual_mahalanobis_max']
    return {'attempts': len(rows), 'accepted': len(rows), 'rejected': 0, 'last_means_sha256': current,
            'sum_accepted_recorded_max_step': sum_caps, 'recorded_positive_actual_dots': positive,
            'complete_batch_objective_increases': changes,
            'path_scope': 'saved scalar linear identities only; no saved gradient arrays for VJP reconstruction'}


def audit(plan_path, expected):
    import torch
    require(sha(COMMON_PATH) == COMMON_SHA, 'Independent common helper changed')
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
    require(spec['protocol'] == 'scene_geometry_routes_v1' and spec['arms'] == list(ARMS)
            and spec['point_count'] == 498136 and spec['attempts_per_arm'] == 148
            and spec['expected_counts'] == EXPECTED_COUNTS, 'Unexpected fixed protocol')
    require(receipt['interpretation'] == spec['interpretation'] and receipt['adoption'] == 'none', 'Interpretation changed')
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
    for key in ('teacher_calls', 'val_views', 'production_checkpoint_writes'):
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
        require(np.isfinite(q).all() and (q >= 0).all()
                and np.max(np.abs(q.astype(np.float64).sum(1)-1)) <= 1e-6, 'Fixed q finite/simplex')
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
                       'change_from_initial': {**{k: summary[k]-initial[k] for k in ('rgb_mse', 'raw_ce', 'scene_ce')},
                           **{kind: {k: summary[kind][k]-initial[kind][k] for k in ('miou_all', 'cable_iou')}
                              for kind in ('raw', 'scene')}}}
    for path, digest in bound.items():
        require(sha(path) == digest, 'Input changed during independent audit')
    for path, digest in produced.items():
        require(sha(path) == digest, 'Saved producer output changed during audit')
    require(tree(plan['source_snapshot']) == plan['source_hashes'] and not torch.cuda.is_initialized(), 'Source/CUDA audit boundary')
    return {'status': 'passed', 'plan_sha256': expected, 'execution_receipt_sha256': sha(run/'execution_receipt.json'),
            'source_count': len(plan['source_hashes']), 'input_count': len(plan['input_hashes']),
            'counts': receipt['counts'], 'initial': initial, 'arms': results, 'maximum_numeric_difference': common.MAX_ERROR,
            'auditor_helper_sha256': COMMON_SHA,
            'pixel_decodes': 0, 'renders': 0, 'new_vjps': 0, 'cuda_initialized': False,
            'before_after_source_input_hashes_exact': True, 'scope': 'saved TRAIN description only; no adoption',
            'limitations': ['No target/image decoding, render, CUDA execution or new VJP; bound file bytes hashed and original checkpoint/Adam deserialized on CPU.',
                'Per-step high-dimensional means/gradients/moments are absent: dots and individual caps are recorded assertions.',
                'Saved per-view loss means, raw/scene CM, path-dot scalar closure, means continuity, final Adam and cumulative displacement checked.',
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
