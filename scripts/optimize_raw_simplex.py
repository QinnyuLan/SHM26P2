"""Frozen-W full-TRAIN direct-simplex diagnostic; never a production checkpoint.

Preparation requires a naturally completed numerical preflight. This module is
currently only a CPU-tested candidate; no preparation or GPU execution is
authorized by importing it or by completing its synthetic tests.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
EPS_RUN = Path('/mnt/data/SHM2026/runs/semantic_partition_eps_matched_v1')
PREFLIGHT = Path('/mnt/data/SHM2026/runs/simplex_scene_preflight_v2')
PREFLIGHT_PLAN_SHA = '8b511295e5a9e1522b35d70e9789355fa90d6a4b2cf06f5aeb877b3acf5c971c'
PREFLIGHT_EXEC_SHA = '236d1d8828e3d833d6cafe63a8bf03f8b7a15e3d9ffde9968be225d585f62303'
BASE_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
MANIFEST_SHA = '551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa'
WEIGHTS = [.45604488253593445, .9227690100669861, .8530434370040894,
           1.1935913562774658, 1.5745513439178467]
SPEC = {
    'protocol': 'raw_simplex_fullbatch_v1', 'base_sha256': BASE_SHA,
    'preflight_plan_sha256': PREFLIGHT_PLAN_SHA, 'preflight_execution_sha256': PREFLIGHT_EXEC_SHA,
    'manifest_sha256': MANIFEST_SHA, 'training_views': 259, 'all_training_cameras': 350,
    'class_weights_fp32': WEIGHTS, 'affine_noise': 5e-7,
    'objective': 'equal view mean of valid-known pixel mean weighted affine-noise raw CE',
    'master_dtype': 'float64', 'renderer_q_dtype': 'float32',
    'gradient': 'each view FP32 q VJP cast to FP64, summed then divided by 259',
    'initialization': 'original classifier FP32 q to FP64 and normalize once explicitly',
    'initial_step': '1/max_i(max_c g_ic-min_c g_ic)',
    'later_step': '2*last accepted eta; backtrack *= 0.5',
    'armijo': 1e-4, 'max_backtracks': 8, 'max_accepted': 20, 'max_full_passes': 80,
    'gap_tolerance': 1e-5, 'gap_claim': 'numerical approximate, not certified bound',
    'acceptance': 'actual FP32 displacement dot FP64 gradient < 0 and full F strictly decreases and Armijo',
    'final_pass_reserved': True, 'internal_seconds': 2700, 'external_seconds': 2760,
    'head': 'frozen; endpoint substitutes normalized new p3d and log prior only',
    'val_views': 0, 'teacher_calls': 0, 'feature_mapping': False,
    'production_checkpoint_writes': 0, 'retries': 0,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, replace=False):
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    if replace:
        pending = path.with_name(f'.{path.name}.{os.getpid()}.pending')
        with pending.open('x') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
    else:
        with path.open('x') as stream:
            stream.write(payload)


def files(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def initial_master(q32):
    require(q32.dtype == np.float32 and q32.ndim == 2 and q32.shape[1] == 5,
            'Original classifier q must be FP32 N by 5')
    require(np.isfinite(q32).all() and (q32 >= 0).all(), 'Invalid original q')
    q64 = q32.astype(np.float64)
    sums = q64.sum(axis=1, keepdims=True)
    require((sums > 0).all(), 'Zero original row mass')
    master = q64 / sums
    return master, {
        'original_max_row_sum_error': float(np.max(np.abs(sums - 1))),
        'master_change_max': float(np.max(np.abs(master - q64))),
        'cast_change_max': float(np.max(np.abs(master.astype(np.float32).astype(np.float64) - q64))),
        'base_full_affine_objective_measured': False,
    }


def actual_displacement(current, proposal):
    """Subtract only after both independently rounded FP32 endpoints become FP64."""
    return proposal.astype(np.float32).astype(np.float64) - current.astype(np.float32).astype(np.float64)


def armijo_decision(base_loss, trial_loss, gradient, displacement):
    require(np.isfinite([base_loss, trial_loss]).all(), 'Nonfinite full objective')
    require(gradient.shape == displacement.shape and np.isfinite(gradient).all()
            and np.isfinite(displacement).all(), 'Invalid gradient/displacement')
    dot = float(np.sum(gradient * displacement, dtype=np.float64))
    require(np.isfinite(dot), 'Nonfinite actual gradient dot displacement')
    bound = float(base_loss + SPEC['armijo'] * dot)
    return {'accepted': bool(dot < 0 and trial_loss < base_loss and trial_loss <= bound),
            'actual_gradient_dot_displacement': dot, 'armijo_bound': bound,
            'full_objective_decrease': float(base_loss - trial_loss)}


def solve_projected(q0, full_pass, *, max_accepted=20, max_passes=80,
                    max_backtracks=8, gap_tolerance=1e-5, on_accept=None):
    """Small CPU control loop. Callback returns a complete, fixed-q pass or raises.

    Callback signature is (q64, gradient=bool, score=bool, phase=str). It must
    never return a partial pass. No acceptance occurs before the full return.
    The last complete pass is always reserved for endpoint gradient and scores.
    """
    from bridge_rgs.simplex_optimization import (
        frank_wolfe_gap,
        projected_gradient_step,
        validate_simplex,
    )
    require(max_passes >= 2 and max_accepted >= 0 and max_backtracks >= 1,
            'Invalid bounded solver budget')
    q = validate_simplex(q0).copy()
    count, accepted, last_eta = 0, 0, None
    history = []

    def run(value, *, gradient, score, phase):
        nonlocal count
        require(count < max_passes - (phase != 'final'), 'Final full pass must remain reserved')
        result = full_pass(value, gradient=gradient, score=score, phase=phase)
        require(result.get('complete') is True and np.isfinite(result['objective']),
                'Partial/nonfinite pass cannot enter acceptance')
        if gradient:
            g = result['gradient']
            require(g.dtype == np.float64 and g.shape == q.shape and np.isfinite(g).all(),
                    'Complete FP64 gradient required')
        count += 1
        return result

    current = run(q, gradient=True, score=True, phase='baseline')
    baseline = {k: v for k, v in current.items() if k != 'gradient'}
    reason = 'accepted_budget'
    while accepted < max_accepted:
        g = current['gradient']
        gap = frank_wolfe_gap(q, g)
        gap_record = {'directional': gap.total_gap, 'centered': gap.centered_total_gap,
                      'master_row_sum_error': gap.max_simplex_sum_error}
        with np.errstate(over='ignore'):
            scale = float(np.max(g.max(axis=1) - g.min(axis=1)))
        require(np.isfinite(scale), 'Unrepresentable gradient range')
        if scale == 0:
            reason = 'zero_tangent_gradient'; break
        if max(gap.total_gap, gap.centered_total_gap) <= gap_tolerance:
            reason = 'approximate_gap_small'; break
        if count >= max_passes - 1:
            reason = 'pass_budget'; break
        eta = 1 / scale if last_eta is None else 2 * last_eta
        require(np.isfinite(eta) and eta > 0, 'Unrepresentable positive scalar step')
        accepted_this_round, any_nonzero = False, False
        for attempt in range(max_backtracks):
            if count >= max_passes - 1:
                reason = 'pass_budget'; break
            proposal = projected_gradient_step(q, g, eta).proposal
            displacement = actual_displacement(q, proposal)
            dot = float(np.sum(g * displacement, dtype=np.float64))
            require(np.isfinite(dot), 'Nonfinite actual directional derivative')
            row = {'accepted_updates_before': accepted, 'trial_index': attempt,
                   'eta': eta, 'gap_at_current': gap_record,
                   'actual_displacement_l2': float(np.linalg.norm(displacement)),
                   'master_displacement_l2': float(np.linalg.norm(proposal - q)),
                   'actual_gradient_dot_displacement': dot,
                   'q32_max_row_sum_error': float(np.max(np.abs(proposal.astype(np.float32)
                                                              .astype(np.float64).sum(axis=1) - 1)))}
            any_nonzero |= bool(np.any(displacement != 0))
            if dot >= 0:
                row.update(accepted=False, trial_skipped='nonnegative_actual_direction')
            else:
                trial = run(proposal, gradient=False, score=False, phase='trial')
                row.update(armijo_decision(current['objective'], trial['objective'], g, displacement))
                row['trial_objective'] = trial['objective']
                if row['accepted']:
                    q = proposal  # Keep master, never copy rounded renderer q back to it.
                    accepted += 1; last_eta = eta; accepted_this_round = True
                    if on_accept is not None:
                        on_accept(q, accepted, row)
            row['complete_passes'] = count
            history.append(row)
            if accepted_this_round:
                break
            eta *= .5
        if not accepted_this_round:
            if reason != 'pass_budget':
                reason = 'backtracking_exhausted' if any_nonzero else 'fp32_stagnation'
            break
        if accepted >= max_accepted:
            break
        # One next gradient, at least one trial, and the final pass would be needed.
        if count + 3 > max_passes:
            reason = 'pass_budget'; break
        current = run(q, gradient=True, score=False, phase='gradient')
    final = run(q, gradient=True, score=True, phase='final')
    gap = frank_wolfe_gap(q, final['gradient'])
    endpoint = {k: v for k, v in final.items() if k != 'gradient'}
    endpoint['gap'] = {'directional': gap.total_gap, 'centered': gap.centered_total_gap,
                       'master_row_sum_error': gap.max_simplex_sum_error,
                       'claim': SPEC['gap_claim']}
    return q, {'stop_reason': reason, 'accepted_updates': accepted, 'complete_passes': count,
               'baseline': baseline, 'endpoint': endpoint, 'history': history,
               'convergence_certified': False}


def validate_preflight(plan, receipt, launch, analysis, *, plan_sha, receipt_sha, analysis_sha):
    require(plan['specification']['protocol'] == 'simplex_scene_preflight_v2',
            'Require the serialization-corrected v2; failed v1 is not admissible')
    require(receipt['status'] == 'completed' and receipt['numerical_status'] == 'passed'
            and receipt['plan_sha256'] == plan_sha and receipt['analysis_sha256'] == analysis_sha,
            'Numerically passed preflight required')
    require(analysis['numerical_status'] == 'passed' and analysis['plan_sha256'] == plan_sha,
            'Preflight analysis differs')
    require(launch['status'] == 'completed' and launch['natural_completion'] is True
            and launch['exit_code'] == 0 and launch['execution_receipt_sha256'] == receipt_sha
            and launch['plan_sha256'] == plan_sha, 'Natural completed preflight required')
    require(receipt['inputs_and_sources_unchanged'] and receipt['numerics_restored']
            and receipt['restoration']['state_exact']
            and receipt['restoration']['flags_modes_gradients_restored'], 'Preflight restoration missing')
    require(analysis['specification'] == plan['specification']
            and [r['name'] for r in analysis['records']] == plan['specification']['names']
            and all(r['passed'] and all(r['gates'].values()) for r in analysis['records']),
            'Preflight fixed numerical records do not all pass')
    require(receipt['counts'] == analysis['counts'] == plan['specification']['counts']
            and receipt['numerics_actual'] == plan['specification']['numerics'],
            'Preflight runtime/counts differ')
    require(receipt['actual_gsplat_binary'] == plan['expected_gsplat_binary'], 'Preflight binary differs')


def population(manifest):
    train = [v for v in manifest['views'] if v['split'] == 'train']
    labeled = sorted((v for v in train if v.get('mask_path')), key=lambda v: v['name'])
    require(len(train) == 350 and len(labeled) == 259 and len({v['name'] for v in train}) == 350,
            'Fixed TRAIN population changed')
    return train, labeled


def prepare(output, preflight):
    output, preflight = Path(output).resolve(), Path(preflight).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Use a new data-disk output')
    require(preflight == PREFLIGHT and sha(preflight/'plan.json') == PREFLIGHT_PLAN_SHA
            and sha(preflight/'execution_receipt.json') == PREFLIGHT_EXEC_SHA,
            'Only the predetermined naturally completed v2 preflight is admissible')
    pp, receipt, launch, analysis = (read(preflight / n) for n in
        ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'analysis.json'))
    validate_preflight(pp, receipt, launch, analysis, plan_sha=sha(preflight/'plan.json'),
                       receipt_sha=sha(preflight/'execution_receipt.json'),
                       analysis_sha=sha(preflight/'analysis.json'))
    source = Path(pp['snapshot'])
    require(files(source) == pp['sources'], 'Preflight source changed')
    require(files(source/'bridge_rgs') == pp['inherited_package_hashes'], 'Preflight package changed')
    eps_plan = read(EPS_RUN/'plan.json')
    require(eps_plan['specification']['class_weights_fp32'] == WEIGHTS, 'Class weights changed')
    base = ROOT/'runs/h3_moments/02_cross/last.pt'
    manifest = ROOT/'artifacts/prepared/manifest.json'
    require(sha(base) == BASE_SHA and sha(manifest) == MANIFEST_SHA, 'H3 source changed')
    inputs = {}
    def bind(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        require(expected is None or digest == expected, f'Changed input {path}')
        inputs[str(path)] = digest
        return str(path)
    for path, digest in pp['inputs'].items():
        bind(path, digest)
    for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'analysis.json'):
        bind(preflight/name)
    bind(base, BASE_SHA); bind(manifest, MANIFEST_SHA)
    bind(ROOT/'uv.lock', sha(source/'uv.lock'))
    bind(EPS_RUN/'plan.json')
    eps_worker = Path(eps_plan['source_snapshot'])/'train_semantic_partition_eps_matched.py'
    bind(eps_worker, eps_plan['source_hashes'][eps_worker.name])
    train, labeled = population(read(manifest))
    training_views = []
    for view in labeled:
        row = {k: view[k] for k in ('name', 'split', 'K', 'w2c_original', 'width', 'height')}
        for key in ('mask_path', 'valid_path'):
            path = str(Path(view[key]).resolve())
            row[key] = bind(path, eps_plan['input_hashes'][path])
        training_views.append(row)
    snapshot = output/'source_snapshot'
    shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    require(files(snapshot) == pp['sources'], 'Inherited preflight snapshot differs')
    helper = ROOT/'src/bridge_rgs/simplex_optimization.py'
    require(not (snapshot/'bridge_rgs/simplex_optimization.py').exists(), 'Unexpected existing helper')
    shutil.copy2(helper, snapshot/'bridge_rgs/simplex_optimization.py')
    for path in (Path(__file__), ROOT/'tests/test_raw_simplex_runner.py',
                 ROOT/'tests/test_simplex_optimization.py', ROOT/'docs/simplex_optimization.md',
                 ROOT/'docs/raw_simplex_optimization_protocol.md'):
        shutil.copy2(path, snapshot/path.name)
    plan = {'specification': SPEC, 'status': 'prepared_pending_root_execution',
            'output': str(output), 'source_snapshot': str(snapshot), 'source_hashes': files(snapshot),
            'inherited_preflight_sources': pp['sources'], 'preflight': str(preflight),
            'preflight_plan_sha256': sha(preflight/'plan.json'), 'input_hashes': inputs,
            'base_checkpoint': str(base), 'manifest': str(manifest), 'training_views': training_views,
            'all_training_camera_names': [v['name'] for v in train],
            'expected_gsplat_binary': pp['expected_gsplat_binary'],
            'installed_sources': pp['installed_sources'], 'runtime_versions': pp['runtime_versions'],
            'environment': pp['environment'], 'numerics': pp['specification']['numerics'],
            'prepare_checkpoint_loads': 0, 'prepare_pixel_decodes': 0}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC, 'Specification changed')
    snapshot = Path(plan['source_snapshot'])
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Execute frozen worker only')
    require(files(snapshot) == plan['source_hashes'], 'Frozen source changed')
    binary = plan['expected_gsplat_binary']
    for path, digest in {**plan['input_hashes'], **plan['installed_sources'],
                         binary['path']: binary['sha256']}.items():
        require(sha(path) == digest, f'Changed input/source: {path}')
    for name, version in plan['runtime_versions'].items():
        require(importlib.metadata.version(name) == version, f'Changed runtime: {name}')
    for name, value in plan['environment'].items():
        require(os.environ.get(name) == value, f'Changed environment: {name}')


def confusion_metrics(cm):
    cm = np.asarray(cm, dtype=np.int64)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    iou = np.divide(np.diag(cm), union, out=np.full(5, np.nan), where=union > 0)
    return {'confusion_matrix': cm.tolist(),
            'iou': [float(v) if np.isfinite(v) else None for v in iou],
            'miou_all': float(iou[np.isfinite(iou)].mean()) if np.isfinite(iou).any() else None}


def semantic_metrics(probability, labels, valid, weights):
    import torch
    keep = valid.bool() & (labels >= 0) & (labels < 5)
    require(bool(keep.any()), 'No eligible TRAIN pixels')
    p = probability[keep].double(); y = labels[keep].long()
    require(bool(torch.isfinite(p).all()) and bool((p >= 0).all()), 'Nonfinite semantic probability')
    pred = p.argmax(-1)
    cm = torch.bincount(y*5 + pred, minlength=25).reshape(5, 5).cpu().numpy()
    # This clamp belongs only to a descriptive production-p3d CE, never F.
    ce = -(weights.double()[y] * p.gather(1, y[:, None])[:, 0].clamp_min(1e-7).log()).mean()
    brier = (p.square().sum(-1) - 2*p.gather(1, y[:, None])[:, 0] + 1).mean()
    return {'weighted_ce': float(ce), 'brier': float(brier), 'pixels': int(keep.sum()),
            **confusion_metrics(cm)}


def aggregate_scores(rows, key):
    cm = np.sum([np.asarray(r[key]['confusion_matrix'], np.int64) for r in rows], axis=0)
    return {**confusion_metrics(cm),
            'weighted_ce_view_mean': float(np.mean([r[key]['weighted_ce'] for r in rows])),
            'brier_view_mean': float(np.mean([r[key]['brier'] for r in rows]))}


def accumulate_view_gradient(accumulator, per_view):
    """Preserve each unscaled FP32 VJP before summing it into FP64."""
    require(accumulator.dtype == np.float64 and per_view.dtype == np.float32
            and accumulator.shape == per_view.shape and np.isfinite(per_view).all(),
            'Expected FP64 accumulator and unscaled finite FP32 per-view VJP')
    accumulator += per_view.astype(np.float64)


def stream_pass(scene, master, plan, adapter, counts, cache, deadline, *, gradient, score, phase):
    import cv2
    import torch

    from bridge_rgs.partition_rasterizer import rasterize_semantic_partition
    from bridge_rgs.semantic_assignment import affine_raw_ce
    from bridge_rgs.simplex_optimization import validate_simplex
    validate_simplex(master)
    started = time.monotonic()
    q32 = master.astype(np.float32)
    q = torch.from_numpy(q32).cuda().requires_grad_(gradient)
    q_digest = hashlib.sha256(q32.tobytes()).hexdigest()
    accumulator = np.zeros_like(master) if gradient else None
    weights = torch.tensor(WEIGHTS, device='cuda', dtype=torch.float32)
    rows = []
    before = dict(counts)
    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError('Partial full-TRAIN pass; cannot accept or retry')
    def head(context, p3d):
        kwargs = {'depth_moments': context['depth_moments']} if 'depth_moments' in context else {}
        residual = scene.refiner(context['features'], context['rgb'], context['depth'],
                                 context['alpha'], p3d=p3d, **kwargs)
        counts['head'] += 1
        return (p3d.log() + residual).softmax(-1)
    try:
        for view in plan['training_views']:
            check_time()
            name = view['name']
            if name not in cache:
                target = cv2.imread(view['mask_path'], cv2.IMREAD_UNCHANGED)
                valid = cv2.imread(view['valid_path'], cv2.IMREAD_UNCHANGED)
                require(target is not None and valid is not None
                        and target.shape == valid.shape == (view['height'], view['width']),
                        'Bad TRAIN label/valid grid')
                require(np.isin(target, [0, 1, 2, 3, 4, 255]).all(), 'Unexpected TRAIN labels')
                cache[name] = (target, valid > 0)
                counts['target_decodes'] += 2
            labels = torch.from_numpy(cache[name][0].astype(np.int64)).cuda()
            valid = torch.from_numpy(cache[name][1]).cuda()
            K = torch.tensor(view['K'], dtype=torch.float32, device='cuda')
            pose = torch.tensor(view['w2c_original'], dtype=torch.float32, device='cuda')
            context, captured = adapter.capture_context(scene, K, pose, view['width'], view['height'], counts)
            raw, _ = adapter.direct_q(context['info'], q, rasterize_semantic_partition,
                                      width=view['width'], height=view['height'])
            counts['shader'] += 1
            loss = affine_raw_ce(raw, labels, valid, weights, SPEC['affine_noise'])
            require(bool(torch.isfinite(loss)), 'Nonfinite full-pass view loss')
            row = {'name': name, 'objective': float(loss.detach())}
            if gradient:
                g, = torch.autograd.grad(loss, (q,))
                require(g.dtype == torch.float32 and bool(torch.isfinite(g).all()), 'Bad per-view VJP')
                accumulate_view_gradient(accumulator, g.detach().cpu().numpy())
                counts['vjp'] += 1
                del g
            if score:
                with torch.no_grad():
                    p3d = raw.detach().clamp_min(1e-7)
                    p3d = p3d / p3d.sum(-1, keepdim=True)
                    row['raw'] = semantic_metrics(p3d, labels, valid, weights)
                    row['scene'] = semantic_metrics(head(context, p3d), labels, valid, weights)
                    row['original_raw'] = semantic_metrics(context['p3d'], labels, valid, weights)
                    row['original_scene'] = semantic_metrics(head(context, context['p3d']), labels, valid, weights)
                    row['original_affine_objective'] = float(affine_raw_ce(
                        captured['raw'], labels, valid, weights, SPEC['affine_noise']))
            rows.append(row)
            del raw, loss, context, captured, labels, valid, K, pose
            check_time()
        require(len(rows) == 259 and adapter.tensor_hash(q) == q_digest, 'Incomplete or changing-q pass')
        counts['complete_passes'] += 1
        result = {'complete': True, 'phase': phase, 'objective': float(np.mean([r['objective'] for r in rows])),
                  'views': 259, 'elapsed_seconds': time.monotonic() - started,
                  'q32_sha256': q_digest,
                  'q32_max_row_sum_error': float(np.max(np.abs(q32.astype(np.float64).sum(axis=1)-1))),
                  'counts': {k: counts[k]-before[k] for k in counts}, 'rows': rows}
        if gradient:
            result['gradient'] = accumulator / 259
        if score:
            result['metrics'] = {key: aggregate_scores(rows, key) for key in
                                 ('raw', 'scene', 'original_raw', 'original_scene')}
            result['original_affine_objective'] = float(np.mean([r['original_affine_objective'] for r in rows]))
        return result
    except BaseException:
        counts['partial_pass_views'] += len(rows)
        raise


def gpu_inventory():
    content = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                       '--format=csv,noheader,nounits'], text=True, timeout=5)
    records = []
    for line in content.splitlines():
        pid, name, memory = line.split(',', 2)
        executable = str(Path(f'/proc/{int(pid)}/exe').resolve())
        require(executable == '/usr/share/rustdesk/rustdesk', 'Unknown compute client; never terminate it')
        records.append({'pid': int(pid), 'name': name.strip(), 'executable': executable,
                        'memory_mib': memory.strip()})
    return records


def run_scene(plan, report, deadline):
    import cv2
    import torch
    require(not any(n == 'bridge_rgs' or n.startswith('bridge_rgs.') for n in sys.modules),
            'Import only the frozen package')
    sys.path.insert(0, plan['source_snapshot'])
    adapter = importlib.import_module('preflight_simplex_scene')
    from bridge_rgs.train import load_scene
    require(str(Path(adapter.__file__).resolve()) == str(Path(plan['source_snapshot'])/'preflight_simplex_scene.py'),
            'Preflight adapter imported from outside snapshot')
    torch.set_num_threads(4)
    previous_flags = adapter.numerical_flags(plan['numerics'])
    scene, captured, original = None, None, None
    old_imread = cv2.imread
    allowed = {str(Path(v[key]).resolve()) for v in plan['training_views'] for key in ('mask_path', 'valid_path')}
    def guarded_imread(path, *args, **kwargs):
        require(str(Path(path).resolve()) in allowed, 'Only bound TRAIN masks/valid may be decoded')
        return old_imread(path, *args, **kwargs)
    cv2.imread = guarded_imread
    try:
        scene, state = load_scene(plan['base_checkpoint'])
        require(scene.pixel_protocol == 'legacy_mixed_v1' and scene.sh_degree == 3
                and scene.mip_filter_config is None, 'Wrong fixed H3 field')
        train, _ = population(read(plan['manifest']))
        cameras = state['training_cameras'].detach().cpu().numpy()
        require(np.array_equal(cameras, np.asarray([v['w2c_original'] for v in train], np.float32))
                and [v['name'] for v in train] == plan['all_training_camera_names'], 'TRAIN cameras changed')
        report['training_camera_sha256'] = hashlib.sha256(cameras.tobytes()).hexdigest()
        captured = adapter.capture_scene(scene)
        original = {k: value.detach().cpu().clone() for k, value in scene.state_dict().items()}
        scene.eval().requires_grad_(False)
        with torch.no_grad():
            original_q = scene.semantic_decoder(scene.splats['sem_features']).softmax(-1).cpu().numpy()
        q0, report['initialization'] = initial_master(original_q)
        counts = report['counts']; cache = {}
        torch.cuda.reset_peak_memory_stats()
        def full(q, **kwargs):
            result = stream_pass(scene, q, plan, adapter, counts, cache, deadline, **kwargs)
            require(all(p.grad is None for p in scene.parameters()), 'Frozen scene received gradients')
            print(json.dumps({'phase': result['phase'], 'full_pass': counts['complete_passes'],
                              'objective': result['objective'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
            with (Path(plan['output'])/'passes.jsonl').open('a') as log:
                log.write(json.dumps({k: v for k, v in result.items() if k != 'gradient'}, allow_nan=False)+'\n')
            return result
        def checkpoint_accept(q, accepted, row):
            # Only independently useful q state, never an ordinary scene or resume checkpoint.
            path = Path(plan['output'])/'last_accepted_q_master.npy'
            with path.with_suffix('.tmp').open('wb') as handle:
                np.save(handle, q, allow_pickle=False)
            path.with_suffix('.tmp').replace(path)
            write(Path(plan['output'])/'last_accepted_state.json', {
                'format': 'raw_simplex_diagnostic_q_only_v1', 'accepted_updates': accepted,
                'q_master_sha256': sha(path), 'row': row, 'ordinary_resume_supported': False,
                'model_checkpoint': False}, replace=True)
        q, result = solve_projected(q0, full, max_accepted=SPEC['max_accepted'],
                                   max_passes=SPEC['max_full_passes'], max_backtracks=SPEC['max_backtracks'],
                                   gap_tolerance=SPEC['gap_tolerance'], on_accept=checkpoint_accept)
        require(result['complete_passes'] == counts['complete_passes'], 'Pass accounting mismatch')
        require(counts['scene'] == 259*counts['complete_passes'] and counts['gsplat'] == 2*counts['scene']
                and counts['shader'] == counts['scene'] and counts['head'] == 4*259,
                'Scene/raster/head budget mismatch')
        report['initialization']['base_full_affine_objective_measured'] = True
        report['initialization']['q0_minus_original_affine_objective'] = (
            result['baseline']['objective'] - result['baseline']['original_affine_objective'])
        final_path = Path(plan['output'])/'final_q_delta.npz'
        np.savez(final_path, q_master=q, q_renderer=q.astype(np.float32))
        result.update(format='raw_simplex_diagnostic_v1', base_checkpoint_sha256=BASE_SHA,
                      manifest_sha256=MANIFEST_SHA, plan_sha256=report['plan_sha256'],
                      q_delta={'path': str(final_path), 'sha256': sha(final_path)},
                      ordinary_resume_supported=False, feature_mapping=False,
                      val_views=0, teacher_calls=0, production_checkpoint=False)
        write(Path(plan['output'])/'analysis.json', result)
        report['analysis_sha256'] = sha(Path(plan['output'])/'analysis.json')
        actual = {name: {'path': str(Path(module.__file__).resolve()), 'sha256': sha(module.__file__)}
                  for name, module in sys.modules.items() if name.startswith('bridge_rgs')
                  and getattr(module, '__file__', None)}
        for record in actual.values():
            path = Path(record['path'])
            require(path.is_relative_to(plan['source_snapshot'])
                    and plan['source_hashes'][str(path.relative_to(plan['source_snapshot']))] == record['sha256'],
                    'Non-frozen package import')
        from gsplat.cuda._backend import _C
        binary = {'path': str(Path(_C.__file__).resolve()), 'sha256': sha(_C.__file__)}
        require(binary == plan['expected_gsplat_binary'], 'Loaded gsplat binary differs from preflight')
        report.update(actual_imports=actual, actual_gsplat_binary=binary,
                      accepted_updates=result['accepted_updates'], stop_reason=result['stop_reason'],
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    finally:
        cv2.imread = old_imread
        try:
            if scene is not None and original is not None:
                report['state_unchanged_before_restore'] = all(
                    adapter.tensor_hash(value) == captured['hashes'][key]
                    for key, value in scene.state_dict().items())
                scene.load_state_dict(original, strict=True)
                report['restoration'] = adapter.restore_scene(scene, captured)
                require(report['restoration']['state_exact'] and report['restoration']['flags_modes_gradients_restored'],
                        'State/flags restoration failed')
                require(report['state_unchanged_before_restore'], 'Frozen state changed during diagnostic')
        finally:
            adapter.numerical_flags(previous_flags)
            report['numerics_restored'] = adapter.numerical_flags() == previous_flags


def claim_attempt(output, expected):
    output = Path(output)
    for name in ('execution_started.json', 'execution_receipt.json', 'analysis.json',
                 'final_q_delta.npz', 'last_accepted_q_master.npy', 'passes.jsonl'):
        require(not (output/name).exists(), f'Previous attempt/output exists: {name}')
    # Exclusive x-mode is the race-safe ownership claim; no GPU work precedes it.
    write(output/'execution_started.json', {'plan_sha256': expected, 'pid': os.getpid(),
                                          'started_unix_seconds': time.time(), 'retries': 0})


def execute(path, expected):
    require(sha(path) == expected, 'Expected plan SHA differs')
    plan = read(path); output = Path(plan['output'])
    claim_attempt(output, expected)
    report = {'status': 'running', 'plan_sha256': expected, 'pid': os.getpid(),
              'counts': {key: 0 for key in ('scene', 'gsplat', 'shader', 'vjp', 'head',
                                           'target_decodes', 'complete_passes', 'partial_pass_views')},
              'val_views': 0, 'teacher_calls': 0, 'rgb_payload_reads': 0,
              'model_optimizer_steps': 0, 'production_checkpoint_writes': 0}
    write(output/'execution_receipt.json', report)
    started = time.monotonic(); deadline = started + SPEC['internal_seconds']
    def expired(*_):
        raise TimeoutError('Fixed wall-time budget exhausted; no retry')
    previous_alarm = signal.signal(signal.SIGALRM, expired)
    signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan)
        report['gpu_before'] = gpu_inventory()
        report['cost_scope'] = 'includes setup, model load, all passes, q I/O and source checks; possible desktop sharing, not FPS'
        run_scene(plan, report, deadline)
        verify(plan)
        report.update(status='completed', inputs_sources_unchanged=True)
    except BaseException as exc:
        report.update(status='inconclusive_timeout' if isinstance(exc, TimeoutError) else 'failed',
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous_alarm)
        report['elapsed_seconds'] = time.monotonic() - started
        report['counts']['total_raster'] = report['counts']['gsplat'] + report['counts']['shader']
        write(output/'execution_receipt.json', report, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--print-spec', action='store_true')
    action.add_argument('--prepare', type=Path)
    action.add_argument('--run', type=Path)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.print_spec:
        print(json.dumps(SPEC, indent=2)); return
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode generation')
    if args.prepare:
        require(args.preflight is not None, 'Completed preflight path required')
        prepare(args.prepare, args.preflight)
    else:
        require(args.expected_plan_sha256 is not None, 'Expected plan SHA required')
        execute(args.run, args.expected_plan_sha256)


if __name__ == '__main__':
    main()
