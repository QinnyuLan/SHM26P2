"""Bounded fixed-W EM/FW diagnostic; CPU-reviewed candidate, no auto-execution."""
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
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
PARENT = Path('/mnt/data/SHM2026/runs/simplex_direction_diagnostic_v2')
PARENT_HASHES = {
    'plan.json': '0d684781d76582c554f7803f4a42bbca989e0ee742243e8c99e4c2c060108984',
    'execution_receipt.json': 'a7a33d154337c75ff341526adcdc7db26ade1130d340da2b61c14a071e7a952d',
    'launch_receipt.json': 'bcaef417758612bfcad6239af1e2e09a1363054407dab0f8b80a723867c8eb7a',
    'analysis.json': '71c6d5b2b277a757eab13400da75b4ef757c30cc82698a6c7322a24fad54e346',
    'independent_cpu_review.json': 'eefcb75c7ad19e2e24397b97d0e83c04e656e4bdf449add1361c691ee8557e07',
}
SPEC = {
    'protocol': 'raw_simplex_em_fw_v1',
    'start': 'audited simplex_direction_diagnostic_v2 diagnostic_candidate only',
    'views': 259, 'affine_noise': 5e-7,
    'em_blocks': 2, 'em_slots_per_block': 10, 'fw_slots': 2,
    'max_full_passes': 26, 'internal_seconds': 1500, 'external_seconds': 1560,
    'em_rejection': 'skip remaining EM slots in this block; use its single scheduled FW',
    'fw_rejection': 'retain last accepted q and end the entire schedule; no retry or next block',
    'em_acceptance': 'complete actual FP32-rendered full objective strictly decreases',
    'fw_acceptance': 'same direction gate and cache/actual strict-decrease gate as direction v2',
    'absolute_tolerance': 2e-7, 'relative_tolerance': 5e-4, 'signal_multiplier': 10.,
    'bisection_iterations': 64,
    'gap_tolerance': 1e-5,
    'master_dtype': 'float64', 'renderer_dtype': 'float32',
    'gradient': 'per-view FP32 VJP to FP64; sum then divide by259',
    'gap': 'report at every full-gradient point; approximate and not a stopping certificate',
    'score': 'baseline and final raw plus frozen-head TRAIN; no performance adoption gate',
    'val_views': 0, 'teacher_calls': 0, 'production_checkpoints': 0, 'retries': 0,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def q32_sha(value):
    return hashlib.sha256(value.astype(np.float32).tobytes()).hexdigest()


def actual_direction(current, proposal):
    return proposal.astype(np.float32).astype(np.float64)-current.astype(np.float32).astype(np.float64)


@contextmanager
def capture_direct_q(adapter, on_raw):
    """Observe the unchanged raw return; always remove this temporary hook."""
    original = adapter.direct_q
    def wrapped(*args, **kwargs):
        output = original(*args, **kwargs)
        on_raw(output[0])
        return output
    adapter.direct_q = wrapped
    try:
        yield
    finally:
        adapter.direct_q = original


def solve_em_fw(q0, full_pass, fw_proposal, *, release_cache=lambda result: None,
                on_accept=lambda q, row: None):
    """Bounded control flow; callbacks never return a partial pass as complete.

    full_pass(q64, gradient, score, phase, cache_targets) returns complete,
    objective, and (when requested) an FP64 full gradient and target_cache.
    A cache is identified with that exact immutable q32 endpoint. All gradient
    passes provisionally save target raw/labels so an early rejected EM block
    can use the last accepted point without another scene pass. release_cache
    may delete only a callback-owned temporary target cache, never source data.

    fw_proposal(q, current, vertex, vertex_result) returns (proposal_or_None,
    JSONable metadata). It must apply the direction-v2 A/B32/B64 gates and its
    fixed64 scalar search. metadata requires passed and cached_objective when
    passed. Only then does this controller render one actual candidate.

    No renderer q is renormalized into the FP64 master. q*(-g) uses that master
    and a gradient evaluated at its FP32 cast; this is a numerical EM proposal,
    not an exact-arithmetic monotonicity certificate. Actual F is authoritative.
    """
    from bridge_rgs.simplex_em import em_step
    from bridge_rgs.simplex_optimization import frank_wolfe_gap, validate_simplex

    q = validate_simplex(q0).copy()
    count, history, pass_records = 0, [], []
    accepted = {'em': 0, 'fw': 0}

    def run(value, *, gradient, score, phase):
        nonlocal count
        require(count < SPEC['max_full_passes']-(phase != 'final'), 'Final pass must remain reserved')
        result = full_pass(value, gradient=gradient, score=score, phase=phase,
                           cache_targets=phase != 'final')
        require(result.get('complete') is True and np.isfinite(result['objective']),
                'Incomplete/nonfinite pass cannot enter a step decision')
        require(result.get('q32_sha256') == q32_sha(value), 'Pass q32 identity differs from requested q')
        if gradient:
            g = result['gradient']
            require(g.dtype == np.float64 and g.shape == q.shape and np.isfinite(g).all(),
                    'Complete finite FP64 gradient required')
            gap = frank_wolfe_gap(value, g)
            result['gap'] = {'directional': gap.total_gap, 'centered': gap.centered_total_gap,
                             'master_row_sum_error': gap.max_simplex_sum_error,
                             'claim': 'approximate full-gradient diagnostic, not certified bound'}
        if phase != 'final':
            require(result.get('target_cache', {}).get('q32_sha256') == result['q32_sha256'],
                    'Current/vertex target cache must share the pass q32 identity')
        count += 1
        pass_records.append({key: value for key, value in result.items()
                             if key not in ('gradient', 'target_cache')})
        return result

    def accept(proposal, trial, kind, row):
        nonlocal q, current
        release_cache(current)
        q, current = proposal, trial
        accepted[kind] += 1
        on_accept(q, row)

    current = run(q, gradient=True, score=True, phase='baseline')
    baseline = pass_records[-1]
    reason = 'fixed_two_block_schedule_finished'
    for block in range(SPEC['em_blocks']):
        for slot in range(SPEC['em_slots_per_block']):
            step = em_step(q, current['gradient'])
            displacement = actual_direction(q, step.proposal)
            row = {'kind': 'em', 'block': block, 'slot': slot,
                   'base_objective': float(current['objective']), 'accepted': False,
                   'zero_responsibility_rows': int(step.zero_responsibility_rows.sum()),
                   'actual_direction_l2': float(np.linalg.norm(displacement)),
                   'gradient_dot_actual_direction': float(np.sum(current['gradient']*displacement)),
                   'gap_at_current': current['gap']}
            if not np.any(displacement):
                row['reason'] = 'fp32_stagnation_skip_remaining_em_slots'
                history.append(row)
                break
            trial = run(step.proposal, gradient=True, score=False, phase=f'em_{block}_{slot}')
            row.update(trial_objective=float(trial['objective']), gap_at_trial=trial['gap'],
                       accepted=bool(trial['objective'] < current['objective']))
            if row['accepted']:
                row['reason'] = 'actual_full_objective_decreased'
                accept(step.proposal, trial, 'em', row)
            else:
                release_cache(trial)
                row['reason'] = 'nondecrease_skip_remaining_em_slots'
            history.append(row)
            if not row['accepted']:
                break

        require(current['q32_sha256'] == q32_sha(q), 'Current q/gradient/cache association changed')
        vertex = frank_wolfe_gap(q, current['gradient']).vertex
        vertex_result = run(vertex, gradient=False, score=False, phase=f'fw_vertex_{block}')
        proposal, metadata = fw_proposal(q, current, vertex, vertex_result)
        release_cache(vertex_result)
        row = {'kind': 'fw', 'block': block, 'accepted': False,
               'base_objective': float(current['objective']), 'line': metadata,
               'gap_at_current': current['gap']}
        require(type(metadata['passed']) is bool, 'FW direction decision must be a native bool')
        if not metadata['passed']:
            require(proposal is None, 'Failed FW gate must not produce a candidate')
            row['reason'] = 'fixed_direction_gate_not_met'
        else:
            proposal = validate_simplex(proposal)
            require(proposal.shape == q.shape and np.isfinite(metadata['cached_objective']),
                    'Invalid fixed-line proposal')
            if not np.any(actual_direction(q, proposal)):
                row['reason'] = 'fp32_stagnation'
            else:
                trial = run(proposal, gradient=True, score=False, phase=f'fw_candidate_{block}')
                predicted = current['objective']-metadata['cached_objective']
                actual = current['objective']-trial['objective']
                tolerance = SPEC['absolute_tolerance']+SPEC['relative_tolerance']*max(abs(predicted), abs(actual))
                passed = bool(predicted > 0 and actual > 0 and abs(predicted-actual) <= tolerance)
                row.update(accepted=passed, predicted_decrease=float(predicted),
                           actual_decrease=float(actual), agreement_tolerance=float(tolerance),
                           trial_objective=float(trial['objective']), gap_at_trial=trial['gap'],
                           reason='actual_decrease_confirmed' if passed else 'actual_candidate_not_confirmed')
                if passed:
                    accept(proposal, trial, 'fw', row)
                else:
                    release_cache(trial)
        history.append(row)
        if not row['accepted']:
            reason = 'fw_stopped_'+row['reason']
            break
        if max(current['gap']['directional'], current['gap']['centered']) <= SPEC['gap_tolerance']:
            reason = 'approximate_gap_small_after_accepted_fw'
            break
    final = run(q, gradient=True, score=True, phase='final')
    release_cache(current)
    result = {'stop_reason': reason, 'accepted_updates': accepted,
              'complete_passes': count, 'baseline': baseline, 'endpoint': pass_records[-1],
              'final_repeat_objective_difference': float(final['objective']-current['objective']),
              'history': history, 'passes': pass_records, 'convergence_certified': False,
              'performance_adoption': False}
    return q, result


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, replace=False):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    path = Path(path)
    if replace:
        pending = path.with_name('.'+path.name+'.pending')
        with pending.open('x') as handle:
            handle.write(payload)
        pending.replace(path)
    else:
        with path.open('x') as handle:
            handle.write(payload)


def files(folder):
    folder = Path(folder)
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'New data-disk output only')
    lineage = {str(PARENT/name): digest for name, digest in PARENT_HASHES.items()}
    for path, digest in lineage.items():
        require(sha(path) == digest, 'Direction-v2 lineage changed')
    parent, execution, launch, analysis, audit = (read(PARENT/name) for name in PARENT_HASHES)
    require(execution['status'] == launch['status'] == 'completed'
            and launch['natural_completion'] is True and launch['exit_code'] == 0
            and launch['execution_receipt_sha256'] == PARENT_HASHES['execution_receipt.json']
            and execution['analysis_sha256'] == audit['analysis_sha256'] == PARENT_HASHES['analysis.json']
            and execution['plan_sha256'] == launch['plan_sha256'] == audit['plan_sha256'] == PARENT_HASHES['plan.json']
            and audit['status'] == 'passed' and audit['candidate']['confirmed']
            and execution['numerical_status'] == 'direction_consistent_candidate_decreased'
            and execution['candidate_confirmation']['passed'] and execution['direction_gate']['passed']
            and execution['inputs_sources_unchanged'] and execution['state_unchanged_before_restore']
            and execution['restoration']['state_exact']
            and execution['restoration']['flags_modes_gradients_restored'] and execution['numerics_restored'],
            'Passed naturally completed direction candidate and independent audit required')
    endpoint = analysis['diagnostic_candidate']
    lineage[endpoint] = analysis['cache_hashes'][endpoint]
    sources = Path(parent['source_snapshot'])
    require(files(sources) == parent['source_hashes'], 'Parent source changed')
    inputs = {**parent['input_hashes'], **lineage}
    for path, digest in inputs.items():
        require(sha(path) == digest, 'Bound inherited input changed: '+path)
    pixels = sum(v['width']*v['height'] for v in parent['training_views'])
    require(shutil.disk_usage(output.parent).free > pixels*15+2*2**30, 'Insufficient data disk')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')))*1024
    require(available >= 16*2**30, 'Require16GiB available host memory')
    snapshot = output/'source_snapshot'
    shutil.copytree(sources, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in (Path(__file__), ROOT/'tests/test_simplex_em_fw_runner.py', ROOT/'tests/test_simplex_em.py',
                 ROOT/'docs/simplex_em_fw_protocol.md'):
        shutil.copy2(path, snapshot/path.name)
    helper = ROOT/'src/bridge_rgs/simplex_em.py'
    require(not (snapshot/'bridge_rgs/simplex_em.py').exists(), 'New EM helper only')
    shutil.copy2(helper, snapshot/'bridge_rgs/simplex_em.py')
    plan = {key: parent[key] for key in ('base_checkpoint', 'manifest', 'training_views', 'all_training_camera_names',
            'expected_gsplat_binary', 'installed_sources', 'runtime_versions', 'environment', 'numerics', 'class_weights')}
    plan.update(specification=SPEC, output=str(output), source_snapshot=str(snapshot),
                source_hashes=files(snapshot), inherited_source_hashes=parent['source_hashes'],
                input_hashes=inputs, parent_lineage=lineage, endpoint=endpoint,
                endpoint_expected_objective=execution['actual_candidate_objective'],
                status='prepared_pending_root_execution', prepare_checkpoint_loads=0, prepare_pixel_decodes=0,
                temporary_cache_upper_bound_bytes=pixels*15, extra_memory_budget_bytes=16*2**30)
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC and files(plan['source_snapshot']) == plan['source_hashes'], 'Frozen contract changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Frozen entry only')
    for path, digest in {**plan['input_hashes'], **plan['installed_sources']}.items():
        require(sha(path) == digest, 'Bound source/input changed: '+path)
    for key, value in plan['environment'].items():
        require(os.environ.get(key) == value, 'Environment changed: '+key)
    for key, value in plan['runtime_versions'].items():
        require(importlib.metadata.version(key) == value, 'Runtime changed: '+key)
    binary = plan['expected_gsplat_binary']
    require(sha(binary['path']) == binary['sha256'], 'Renderer binary changed')


def run_scene(plan, report, deadline):
    import cv2
    import torch
    require(not any(n.startswith('bridge_rgs') for n in sys.modules), 'No live package imports')
    old = importlib.import_module('optimize_raw_simplex')
    direction = importlib.import_module('diagnose_simplex_direction')
    adapter = importlib.import_module('preflight_simplex_scene')
    from bridge_rgs.train import load_scene
    require(plan['class_weights'] == old.WEIGHTS and SPEC['affine_noise'] == old.SPEC['affine_noise']
            == direction.SPEC['delta'] and SPEC['bisection_iterations'] == direction.SPEC['bisection_iterations']
            and all(SPEC[k] == direction.SPEC[k] for k in ('absolute_tolerance', 'relative_tolerance', 'signal_multiplier')),
            'Inherited objective or fixed line rules differ')
    torch.set_num_threads(4)
    previous_flags = adapter.numerical_flags(plan['numerics'])
    scene = captured = original = None
    original_imread = cv2.imread
    allowed = {v[k] for v in plan['training_views'] for k in ('mask_path', 'valid_path')}
    def guarded_imread(path, *args, **kwargs):
        require(str(path) in allowed, 'Only fixed TRAIN targets/valid may be decoded')
        return original_imread(path, *args, **kwargs)
    cv2.imread = guarded_imread
    output = Path(plan['output']); temporary = output/'temporary_target_cache'; temporary.mkdir()
    report['cache_releases'] = []; cache = {}; counts = report['counts']
    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError('Fixed1500-second deadline; no partial-pass acceptance')
    try:
        scene, state = load_scene(plan['base_checkpoint'])
        require(scene.pixel_protocol == 'legacy_mixed_v1' and scene.sh_degree == 3
                and scene.mip_filter_config is None, 'Wrong fixed H3 field')
        train, _ = old.population(read(plan['manifest']))
        cameras = state['training_cameras'].cpu().numpy()
        require(np.array_equal(cameras, np.asarray([v['w2c_original'] for v in train], np.float32))
                and [v['name'] for v in train] == plan['all_training_camera_names'], 'Cameras changed')
        captured = adapter.capture_scene(scene)
        original = {k: v.detach().cpu().clone() for k, v in scene.state_dict().items()}
        scene.eval().requires_grad_(False)
        q0 = np.load(plan['endpoint'], allow_pickle=False)
        require(q0.shape == (len(scene.splats['means']), 5), 'Endpoint shape differs')
        torch.cuda.reset_peak_memory_stats()

        def full(q, *, gradient, score, phase, cache_targets):
            check_time()
            folder = temporary/phase
            if cache_targets:
                folder.mkdir()
            records = []
            def on_raw(raw):
                if cache_targets:
                    view = plan['training_views'][len(records)]
                    labels, valid = cache[view['name']]
                    keep = valid & (labels < 5)
                    target = labels[keep].astype(np.uint8)
                    keep_gpu = torch.from_numpy(keep).cuda()
                    y_gpu = torch.from_numpy(target.astype(np.int64)).cuda()
                    a = raw.detach()[keep_gpu].gather(1, y_gpu[:, None])[:, 0].cpu().numpy()
                    index = len(records)
                    records.append({'name': view['name'], 'pixels': len(a),
                                    'a': direction.save_array(folder, f'{index:03d}.a.npy', a),
                                    'labels': direction.save_array(folder, f'{index:03d}.y.npy', target)})
            with capture_direct_q(adapter, on_raw):
                result = old.stream_pass(scene, q, plan, adapter, counts, cache, deadline,
                                         gradient=gradient, score=score, phase=phase)
            require(all(p.grad is None for p in scene.parameters()), 'Frozen scene received gradients')
            if cache_targets:
                require(len(records) == 259, 'Missing target cache rows')
                result['target_cache'] = {'q32_sha256': result['q32_sha256'], 'folder': str(folder),
                                          'records': records, 'hashes': {str(p): sha(p) for p in sorted(folder.iterdir())}}
            if phase == 'baseline':
                require(abs(result['objective']-plan['endpoint_expected_objective']) <= 2e-7,
                        'Bound starting candidate objective not reproduced')
            print(json.dumps({'phase': phase, 'full_pass': counts['complete_passes'], 'F': result['objective'],
                              'elapsed_seconds': time.monotonic()-(deadline-SPEC['internal_seconds'])}), flush=True)
            with (output/'passes.jsonl').open('a') as log:
                log.write(json.dumps({k: v for k, v in result.items() if k != 'gradient'}, allow_nan=False)+'\n')
            return result

        def release(result):
            item = result['target_cache']; folder = Path(item['folder'])
            require(folder.parent == temporary and not folder.is_symlink(), 'Only own temporary cache is releasable')
            require({str(p) for p in folder.iterdir()} == set(item['hashes']), 'Unexpected cache content')
            for path, digest in item['hashes'].items():
                require(Path(path).parent == folder and not Path(path).is_symlink() and sha(path) == digest,
                        'Temporary target cache changed')
            report['cache_releases'].append({'folder': str(folder), 'q32_sha256': item['q32_sha256'],
                                            'deleted_files': len(item['hashes']), 'hashes': item['hashes']})
            for path in item['hashes']:
                Path(path).unlink()
            folder.rmdir()
            check_time()

        def fw(q, current, vertex, vertex_result):
            require(current['target_cache']['q32_sha256'] == q32_sha(q)
                    and vertex_result['target_cache']['q32_sha256'] == q32_sha(vertex), 'FW cache/q mismatch')
            records = []
            for a, v in zip(current['target_cache']['records'], vertex_result['target_cache']['records'], strict=True):
                require(a['name'] == v['name'] and a['pixels'] == v['pixels']
                        and current['target_cache']['hashes'][a['labels']]
                        == vertex_result['target_cache']['hashes'][v['labels']], 'FW support differs')
                records.append({**a, 'v': v['a']})
            cw = np.asarray(plan['class_weights'], np.float32).astype(np.float64)
            zero = direction.line_summary(records, cw, 0., bins=True, deadline=deadline)
            require(abs(zero['F']-current['objective']) <= 1e-12, 'Target cache F differs from full pass')
            A = float(np.sum(current['gradient']*actual_direction(q, vertex)))
            gate = direction.direction_gate(A, zero['B64'], zero['B32upstream'])
            metadata = {'passed': gate['passed'], 'direction_gate': gate, 'A': A, 'cache_at_zero': zero}
            if not gate['passed']:
                return None, metadata
            print(json.dumps({'phase': 'fixed_fw_scalar_search', 'elapsed_seconds': time.monotonic()-(deadline-1500)}), flush=True)
            gamma, search = direction.fixed_line_search(lambda t: direction.line_summary(
                records, cw, t, deadline=deadline, with_loss=False, validated=True)['B64'])
            candidate = direction.line_summary(records, cw, gamma, deadline=deadline, validated=True)
            proposal = (1-gamma)*q+gamma*vertex
            metadata.update(gamma=float(gamma), search=search, cached_objective=candidate['F'],
                            cached_candidate=candidate,
                            cast_line_difference_max=float(np.max(np.abs(proposal.astype(np.float32).astype(np.float64)
                                -(q.astype(np.float32).astype(np.float64)+gamma*actual_direction(q, vertex))))))
            return proposal, metadata

        def accepted(q, row):
            path = output/'last_accepted_q_master.npy'; pending = output/'.last_accepted_q_master.pending'
            with pending.open('xb') as handle:
                np.save(handle, q, allow_pickle=False)
            pending.replace(path)
            write(output/'last_accepted_state.json', {'q_master_sha256': sha(path), 'step': row,
                  'ordinary_resume_supported': False, 'production_checkpoint': False}, replace=True)

        q, result = solve_em_fw(q0, full, fw, release_cache=release, on_accept=accepted)
        require(result['complete_passes'] == counts['complete_passes'] <= 26
                and counts['scene'] == 259*counts['complete_passes']
                and counts['gsplat'] == 2*counts['scene'] and counts['shader'] == counts['scene']
                and counts['head'] == 4*259, 'Pass/render/score budget differs')
        path = output/'final_q_delta.npz'
        np.savez(path, q_master=q, q_renderer=q.astype(np.float32))
        result.update(q_delta={'path': str(path), 'sha256': sha(path)}, plan_sha256=report['plan_sha256'],
                      format='raw_simplex_em_fw_q_only_v1', ordinary_resume_supported=False,
                      production_checkpoint=False, val_views=0, teacher_calls=0)
        write(output/'analysis.json', result)
        report.update(analysis_sha256=sha(output/'analysis.json'), stop_reason=result['stop_reason'],
                      accepted_updates=result['accepted_updates'], convergence_certified=False,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        report['actual_imports'] = {}
        for name, module in sys.modules.items():
            if (name.startswith('bridge_rgs') or name in ('optimize_raw_simplex', 'diagnose_simplex_direction',
                                                        'preflight_simplex_scene')) and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve()
                require(path.is_relative_to(plan['source_snapshot'])
                        and plan['source_hashes'][str(path.relative_to(plan['source_snapshot']))] == sha(path),
                        'Nonfrozen imported module')
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}
        from gsplat.cuda._backend import _C
        binary = {'path': str(Path(_C.__file__).resolve()), 'sha256': sha(_C.__file__)}
        require(binary == plan['expected_gsplat_binary'], 'Wrong loaded renderer binary')
        report['actual_gsplat_binary'] = binary
    finally:
        cv2.imread = original_imread
        try:
            if scene is not None and original is not None:
                unchanged = all(adapter.tensor_hash(v) == captured['hashes'][k] for k, v in scene.state_dict().items())
                scene.load_state_dict(original, strict=True)
                report['restoration'] = adapter.restore_scene(scene, captured)
                report['state_unchanged_before_restore'] = unchanged
                require(unchanged and report['restoration']['state_exact']
                        and report['restoration']['flags_modes_gradients_restored'], 'Restoration failed')
        finally:
            adapter.numerical_flags(previous_flags)
            report['numerics_restored'] = adapter.numerical_flags() == previous_flags


def execute(path, expected):
    require(sha(path) == expected, 'Plan SHA differs')
    plan = read(path); output = Path(plan['output'])
    for name in ('execution_started.json', 'execution_receipt.json', 'analysis.json', 'temporary_target_cache',
                 'last_accepted_q_master.npy', 'passes.jsonl', 'final_q_delta.npz'):
        require(not (output/name).exists(), 'Refuse previous attempt/output')
    write(output/'execution_started.json', {'plan_sha256': expected, 'pid': os.getpid(), 'time': time.time()})
    report = {'status': 'running', 'plan_sha256': expected,
              'counts': {k: 0 for k in ('scene', 'gsplat', 'shader', 'vjp', 'head', 'target_decodes',
                                       'complete_passes', 'partial_pass_views')},
              'val_views': 0, 'teacher_calls': 0, 'model_updates': 0, 'production_checkpoint_writes': 0}
    started = time.monotonic()
    def expired(*_):
        raise TimeoutError('Fixed1500-second total budget; no retry')
    prior = signal.signal(signal.SIGALRM, expired); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan)
        sys.path.insert(0, plan['source_snapshot'])
        old = importlib.import_module('optimize_raw_simplex')
        report['gpu_before'] = old.gpu_inventory()
        run_scene(plan, report, started+SPEC['internal_seconds'])
        verify(plan)
        report.update(status='completed', inputs_sources_unchanged=True)
    except BaseException as error:
        report.update(status='inconclusive_timeout' if isinstance(error, TimeoutError) else 'failed',
                      error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, prior)
        report['counts']['total_raster'] = report['counts']['gsplat']+report['counts']['shader']
        report['elapsed_seconds'] = time.monotonic()-started
        write(output/'execution_receipt.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path); action.add_argument('--run', type=Path)
    action.add_argument('--print-spec', action='store_true')
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.print_spec:
        print(json.dumps(SPEC, indent=2)); return
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode generation')
    if args.prepare:
        prepare(args.prepare)
    else:
        execute(args.run, args.expected_plan_sha256)


if __name__ == '__main__':
    main()
