"""Single fixed-W FW-direction diagnostic at the completed raw-simplex endpoint."""
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
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
PARENT = Path('/mnt/data/SHM2026/runs/raw_simplex_fullbatch_v1')
PARENT_PLAN_SHA = 'c19aa6bde8190e5899475b56d02676ded38e8f16632731553c8ddfbf448972d0'
FAILED_V1 = Path('/mnt/data/SHM2026/runs/simplex_direction_diagnostic_v1')
FAILED_V1_HASHES = {
    'plan.json': '6d822a06d7b1852eec3bacca0f06a9ccdb1f0a766fdd7dcbebebdaaa472696f2',
    'execution_receipt.json': '208e870c6caa6c1470f921c5f88e725e1b456603bd61ba8f1ac3136620360aab',
    'launch_receipt.json': 'e0dd5cb890e2dacc6d87f2567b5915a62a9515ef625027fc077326cddb6b3a26',
}
SPEC = {'protocol': 'simplex_direction_diagnostic_v2', 'views': 259,
        'parent_plan_sha256': PARENT_PLAN_SHA, 'direction': 'one full-gradient LMO vertex; tie first',
        'delta': 5e-7, 'absolute_tolerance': 2e-7, 'relative_tolerance': 5e-4,
        'signal_multiplier': 10., 'bins': [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, .1, None],
        'bin_scope': 'B64 positive/negative/absolute mass only, not per-bin VJP validation',
        'bisection_iterations': 64, 'chunk_pixels': 262144,
        'scene_passes_max': 3, 'vjp_passes': 1, 'head_calls': 0,
        'internal_seconds': 600, 'external_seconds': 660, 'retries': 0,
        'extra_memory_budget_bytes': 16 * 2**30,
        'objective': 'same full259 view-equal weighted affine-noise raw CE as parent',
        'claim': 'standard FW line diagnostic, no innovation/adoption/convergence claim',
        'val_views': 0, 'teacher_calls': 0, 'model_updates': 0}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, replace=False):
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    if replace:
        temporary = path.with_name('.' + path.name + '.pending')
        with temporary.open('x') as handle:
            handle.write(payload)
        temporary.replace(path)
    else:
        with path.open('x') as handle:
            handle.write(payload)


def files(path):
    path = Path(path)
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def pair_check(left, right):
    tolerance = SPEC['absolute_tolerance'] + SPEC['relative_tolerance'] * max(abs(left), abs(right))
    return {'left': float(left), 'right': float(right), 'absolute_error': float(abs(left-right)),
            'tolerance': float(tolerance), 'passed': bool(abs(left-right) <= tolerance)}


def direction_gate(a, b64, b32):
    require(np.isfinite([a, b64, b32]).all(), 'Nonfinite derivative')
    comparisons = {'A_vs_B32upstream': pair_check(a, b32), 'B32upstream_vs_B64': pair_check(b32, b64)}
    measurable = min(abs(a), abs(b64), abs(b32)) > SPEC['signal_multiplier']*SPEC['absolute_tolerance']
    negative = max(a, b64, b32) < 0
    return {'comparisons': comparisons, 'A_minus_B64': float(a-b64),
            'measurable': bool(measurable), 'all_negative': bool(negative),
            'passed': bool(measurable and negative and all(c['passed'] for c in comparisons.values()))}


def cached_terms(raw_current, raw_vertex, labels, class_weights, gamma, *, upstream=False,
                 with_loss=True, validated=False):
    """Independent FP64 affine line. Fixed residual background cancels in delta raw."""
    require(raw_current.dtype == raw_vertex.dtype == np.float32 and raw_current.shape == raw_vertex.shape
            and raw_current.ndim == labels.ndim == 1 and len(labels) > 0,
            'Expected FP32 target-only endpoint arrays and nonempty labels')
    require(0 <= gamma <= 1, 'Invalid gamma')
    if not validated:
        require(np.isfinite(raw_current).all() and np.isfinite(raw_vertex).all()
                and (raw_current >= 0).all() and (raw_vertex >= 0).all()
                and np.isin(labels, np.arange(5)).all(), 'Invalid endpoint probabilities/labels')
    a = (1-SPEC['delta'])*raw_current.astype(np.float64) + SPEC['delta']/5
    d_raw = raw_vertex.astype(np.float64) - raw_current.astype(np.float64)
    b = (1-SPEC['delta'])*d_raw
    p = (1-gamma)*a + gamma*((1-SPEC['delta'])*raw_vertex.astype(np.float64)+SPEC['delta']/5)
    weight = class_weights[labels].astype(np.float64)
    values = {'loss_sum': float(np.sum(-weight*np.log(p))) if with_loss else 0.,
              'derivative_sum': float(np.sum(-weight*b/p))}
    if upstream:
        # The caller handles Nvalid: this helper returns unscaled pixel terms.
        values.update(probability=a, derivative_pixel=-weight*b/a,
                      upstream_numerator=-weight*(1-SPEC['delta'])/a, delta_raw=d_raw)
    return values


def line_summary(records, class_weights, gamma, *, bins=False, deadline=None, with_loss=True,
                 validated=False):
    loss, derivative, b32, rows = 0., 0., 0., []
    bin_edges = np.array([*SPEC['bins'][:-1], np.inf])
    totals = {key: np.zeros(len(bin_edges)-1, np.float64) for key in ('positive', 'negative', 'absolute')}
    counts = np.zeros(len(bin_edges)-1, np.int64)
    for record in records:
        a = np.load(record['a'], mmap_mode='r', allow_pickle=False)
        v = np.load(record['v'], mmap_mode='r', allow_pickle=False)
        labels = np.load(record['labels'], mmap_mode='r', allow_pickle=False)
        require(len(a) == len(v) == len(labels) == record['pixels'], 'Cache support differs')
        n = len(a); view_loss = view_derivative = view_b32 = view_abs = 0.
        for start in range(0, n, SPEC['chunk_pixels']):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Scalar cache analysis exhausted fixed deadline')
            stop = min(start + SPEC['chunk_pixels'], n)
            values = cached_terms(a[start:stop], v[start:stop], labels[start:stop], class_weights, gamma,
                                  upstream=bins, with_loss=with_loss, validated=validated)
            view_loss += values['loss_sum']/n
            view_derivative += values['derivative_sum']/n
            if bins:
                up32 = (values['upstream_numerator']/n).astype(np.float32)
                view_b32 += float(np.sum(up32.astype(np.float64)*values['delta_raw']))
                terms = values['derivative_pixel']/n/len(records)
                view_abs += float(np.sum(np.abs(values['derivative_pixel'])))/n
                index = np.searchsorted(bin_edges, values['probability'], side='right') - 1
                counts += np.bincount(index, minlength=len(counts))
                for key, mass in (('positive', np.maximum(terms, 0)), ('negative', np.minimum(terms, 0)),
                                  ('absolute', np.abs(terms))):
                    totals[key] += np.bincount(index, weights=mass, minlength=len(counts))
        loss += view_loss/len(records); derivative += view_derivative/len(records)
        b32 += view_b32/len(records)
        rows.append({'name': record['name'], 'pixels': n, 'F': view_loss, 'B64': view_derivative,
                     'B32upstream': view_b32 if bins else None,
                     'absolute_B_pixel_mass': view_abs if bins else None})
    result = {'gamma': float(gamma), 'F': float(loss) if with_loss else None, 'B64': float(derivative), 'rows': rows}
    if bins:
        absolute = float(totals['absolute'].sum())
        result.update(B32upstream=float(b32), probability_bins={
            'edges': SPEC['bins'], 'pixel_counts': counts.tolist(),
            **{key: values.tolist() for key, values in totals.items()},
            'scope': SPEC['bin_scope']}, cancellation_ratio=absolute/abs(derivative) if derivative else None)
    return result


def fixed_line_search(derivative):
    """One convex line, not a hyperparameter or direction search."""
    lo, hi = 0., 1.
    left, right = float(derivative(lo)), float(derivative(hi))
    require(np.isfinite([left, right]).all() and left < 0, 'Measurable descent direction required')
    if right <= 0:
        return 1., {'rule': 'endpoint derivative nonpositive', 'iterations': 0,
                    'derivative_at_zero': left, 'derivative_at_one': right}
    for _ in range(SPEC['bisection_iterations']):
        middle = (lo+hi)/2
        value = float(derivative(middle))
        require(np.isfinite(value), 'Nonfinite cached derivative')
        if value <= 0:
            lo = middle
        else:
            hi = middle
    return (lo+hi)/2, {'rule': 'fixed64 convex derivative bisection', 'iterations': SPEC['bisection_iterations'],
                       'bracket': [lo, hi], 'derivative_at_zero': left, 'derivative_at_one': right}


def candidate_gate(base, cached_candidate, actual_candidate):
    predicted = base-cached_candidate; actual = base-actual_candidate
    return {'strict_decrease': bool(actual > 0), 'predicted_decrease': float(predicted),
            'actual_decrease': float(actual), 'prediction_agreement': pair_check(predicted, actual),
            'passed': bool(actual > 0 and predicted > 0 and pair_check(predicted, actual)['passed'])}


def validate_parent(plan, receipt, launch, audit, plan_sha, receipt_sha):
    require(plan_sha == PARENT_PLAN_SHA and plan['specification']['protocol'] == 'raw_simplex_fullbatch_v1',
            'Wrong fixed parent experiment')
    require(receipt['status'] == 'completed' and receipt['plan_sha256'] == plan_sha
            and receipt['state_unchanged_before_restore'] and receipt['inputs_sources_unchanged'], 'Parent incomplete')
    require(launch['status'] == 'completed' and launch['natural_completion'] is True
            and launch['exit_code'] == 0 and launch['execution_receipt_sha256'] == receipt_sha
            and launch['plan_sha256'] == plan_sha, 'Parent natural completion required')
    require(audit['status'] == 'passed' and audit['plan_sha256'] == plan_sha, 'Parent independent audit required')


def workflow_revision():
    inputs = {str(FAILED_V1/name): digest for name, digest in FAILED_V1_HASHES.items()}
    for path, digest in inputs.items():
        require(sha(path) == digest, 'Failed v1 evidence changed')
    old, receipt, launch = (read(FAILED_V1/name) for name in FAILED_V1_HASHES)
    require(old['specification'] == {**SPEC, 'protocol': 'simplex_direction_diagnostic_v1'},
            'Workflow revision must not change numerical specification')
    require(receipt['status'] == launch['status'] == 'failed'
            and receipt['error'] == 'ValueError: Frozen contract changed'
            and receipt['counts'] and all(value == 0 for value in receipt['counts'].values())
            and launch['exit_code'] == 1 and launch['natural_completion'] is False
            and launch['execution_receipt_sha256'] == FAILED_V1_HASHES['execution_receipt.json']
            and receipt['plan_sha256'] == launch['plan_sha256'] == FAILED_V1_HASHES['plan.json'],
            'Only the recorded zero-work source-inventory failure is eligible')
    snapshot = Path(old['source_snapshot']); actual = files(snapshot)
    require(all(actual.get(name) == digest for name, digest in old['source_hashes'].items()),
            'Failed v1 has modified/missing bound source')
    added = {name: digest for name, digest in actual.items() if name not in old['source_hashes']}
    require(len(added) == 4 and all(name.startswith('.pytest_cache/') for name in added),
            'Unexpected failed-v1 source additions')
    runner = snapshot/Path(__file__).name
    inputs[str(runner)] = old['source_hashes'][runner.name]
    inputs.update({str(snapshot/name): digest for name, digest in added.items()})
    return inputs, {'failed_run': str(FAILED_V1), 'failure': receipt['error'],
                    'zero_work_counts': receipt['counts'], 'added_cache_files': added,
                    'modified_or_missing_bound_sources': [], 'numerical_specification_unchanged': True,
                    'source_origin': str(PARENT/'source_snapshot'),
                    'only_revision': 'protocol and failed-source lineage; disable pytest cache generation',
                    'frozen_test_rule': 'pytest -p no:cacheprovider; source hashes exact before and after',
                    'old_run_preserved': True}


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'New data-disk output only')
    failure_inputs, revision = workflow_revision()
    pp, execution, launch, audit = (read(PARENT/name) for name in
        ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json'))
    validate_parent(pp, execution, launch, audit, sha(PARENT/'plan.json'), sha(PARENT/'execution_receipt.json'))
    analysis = read(PARENT/'analysis.json')
    require(sha(PARENT/'analysis.json') == execution['analysis_sha256'], 'Parent analysis changed')
    sources = Path(pp['source_snapshot'])
    require(files(sources) == pp['source_hashes'], 'Parent frozen source changed')
    inputs = {**pp['input_hashes'], **failure_inputs}
    for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json',
                 'analysis.json', 'final_q_delta.npz'):
        inputs[str(PARENT/name)] = sha(PARENT/name)
    require(inputs[str(PARENT/'final_q_delta.npz')] == analysis['q_delta']['sha256'], 'Fixed endpoint changed')
    for path, digest in inputs.items():
        require(sha(path) == digest, f'Parent input changed: {path}')
    pixels = sum(v['width']*v['height'] for v in pp['training_views'])
    n = audit['final_q']['shape'][0]
    caches = pixels*9 + 259*n*5*4
    require(caches + 2*2**30 < SPEC['extra_memory_budget_bytes'], 'Conservative cache/working-set budget exceeded')
    require(shutil.disk_usage(output.parent).free > caches + 2**30, 'Insufficient data-disk space')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    require(available >= SPEC['extra_memory_budget_bytes'], 'Insufficient host memory')
    snapshot = output/'source_snapshot'
    shutil.copytree(sources, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in (Path(__file__), ROOT/'tests/test_simplex_direction.py', ROOT/'docs/simplex_direction_protocol.md'):
        shutil.copy2(path, snapshot/path.name)
    plan = {key: pp[key] for key in ('base_checkpoint', 'manifest', 'training_views', 'all_training_camera_names',
                                    'expected_gsplat_binary', 'installed_sources', 'runtime_versions', 'environment', 'numerics')}
    plan.update(specification=SPEC, output=str(output), status='prepared_pending_root_execution',
                parent=str(PARENT), endpoint=str(PARENT/'final_q_delta.npz'),
                source_snapshot=str(snapshot), source_hashes=files(snapshot), input_hashes=inputs,
                inherited_source_hashes=pp['source_hashes'], class_weights=pp['specification']['class_weights_fp32'],
                workflow_revision=revision,
                cache_upper_bound_bytes=caches, host_memory_available_at_prepare=available,
                prepare_pixel_decodes=0, prepare_checkpoint_loads=0)
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC and files(plan['source_snapshot']) == plan['source_hashes'], 'Frozen contract changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Frozen entry only')
    for path, digest in {**plan['input_hashes'], **plan['installed_sources']}.items():
        require(sha(path) == digest, f'Bound input/source changed: {path}')
    for name, value in plan['environment'].items():
        require(os.environ.get(name) == value, f'Environment changed: {name}')
    for name, version in plan['runtime_versions'].items():
        require(importlib.metadata.version(name) == version, f'Runtime changed: {name}')
    binary = plan['expected_gsplat_binary']
    require(sha(binary['path']) == binary['sha256'], 'Expected binary changed')


def save_array(folder, name, value):
    path = folder/name
    require(not path.exists(), 'Refuse cached-array overwrite')
    with path.open('xb') as handle:
        np.save(handle, value, allow_pickle=False)
    return str(path)


def perform(plan, report, deadline):
    import cv2
    import torch
    require(not any(n.startswith('bridge_rgs') for n in sys.modules), 'No live-package imports')
    sys.path.insert(0, plan['source_snapshot'])
    old = importlib.import_module('optimize_raw_simplex')
    adapter = importlib.import_module('preflight_simplex_scene')
    from bridge_rgs.partition_rasterizer import rasterize_semantic_partition
    from bridge_rgs.semantic_assignment import affine_raw_ce
    from bridge_rgs.simplex_optimization import linear_minimization_oracle, validate_simplex
    from bridge_rgs.train import load_scene
    torch.set_num_threads(4)
    old_flags = adapter.numerical_flags(plan['numerics'])
    scene = captured = original = None
    old_imread = cv2.imread
    allowed = {v[key] for v in plan['training_views'] for key in ('mask_path', 'valid_path')}
    def imread(path, *args, **kwargs):
        require(str(path) in allowed, 'Only fixed TRAIN labels/valid may be decoded')
        return old_imread(path, *args, **kwargs)
    cv2.imread = imread
    cache_dir = Path(plan['output'])/'cache'; cache_dir.mkdir()
    records, cache, counts = [], {}, report['counts']
    weights_np = np.asarray(plan['class_weights'], np.float32).astype(np.float64)
    weights = torch.tensor(plan['class_weights'], device='cuda', dtype=torch.float32)
    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError('Fixed diagnostic deadline exhausted')
    def progress(phase):
        print(json.dumps({'phase': phase, 'elapsed_seconds': time.monotonic()-(deadline-SPEC['internal_seconds']),
                          'counts': dict(counts)}), flush=True)
    try:
        scene, state = load_scene(plan['base_checkpoint'])
        require(scene.pixel_protocol == 'legacy_mixed_v1' and scene.mip_filter_config is None,
                'Wrong fixed field')
        train, _ = old.population(read(plan['manifest']))
        require(np.array_equal(state['training_cameras'].cpu().numpy(),
                               np.asarray([v['w2c_original'] for v in train], np.float32)), 'Cameras changed')
        captured = adapter.capture_scene(scene)
        original = {k: v.detach().cpu().clone() for k, v in scene.state_dict().items()}
        scene.eval().requires_grad_(False)
        with np.load(plan['endpoint'], allow_pickle=False) as saved:
            master = saved['q_master']; renderer = saved['q_renderer']
        validate_simplex(master)
        require(renderer.dtype == np.float32 and np.array_equal(renderer, master.astype(np.float32)), 'Bad endpoint cast')
        require(master.shape == (len(scene.splats['means']), 5), 'Endpoint shape differs from fixed scene')
        q = torch.from_numpy(renderer.copy()).cuda().requires_grad_(True)
        full_gradient = np.zeros_like(master)
        torch.cuda.reset_peak_memory_stats()

        def view_context(view):
            check_time()
            if view['name'] not in cache:
                y = cv2.imread(view['mask_path'], cv2.IMREAD_UNCHANGED)
                valid = cv2.imread(view['valid_path'], cv2.IMREAD_UNCHANGED)
                require(y is not None and valid is not None and y.shape == valid.shape == (view['height'], view['width']), 'Target grid')
                require(np.isin(y, [0, 1, 2, 3, 4, 255]).all(), 'Invalid labels')
                cache[view['name']] = (y, (valid > 0) & (y < 5))
                counts['target_decodes'] += 2
            y, valid = cache[view['name']]
            require(valid.any(), 'Empty view support')
            labels = torch.from_numpy(y.astype(np.int64)).cuda()
            support = torch.from_numpy(valid).cuda()
            K = torch.tensor(view['K'], device='cuda', dtype=torch.float32)
            pose = torch.tensor(view['w2c_original'], device='cuda', dtype=torch.float32)
            context, _ = adapter.capture_context(scene, K, pose, view['width'], view['height'], counts)
            return context, labels, support

        def draw(context, value, view):
            counts['shader'] += 1
            return adapter.direct_q(context['info'], value, rasterize_semantic_partition,
                                     width=view['width'], height=view['height'])[0]

        # Exactly one full-gradient pass; no pre-division of tiny FP32 VJPs.
        for index, view in enumerate(plan['training_views']):
            context, labels, keep = view_context(view)
            raw = draw(context, q, view)
            loss = affine_raw_ce(raw, labels, keep, weights, SPEC['delta'])
            gradient, = torch.autograd.grad(loss, q)
            counts['vjp'] += 1
            g = gradient.detach().cpu().numpy()
            old.accumulate_view_gradient(full_gradient, g)
            target = labels[keep]
            a = raw.detach()[keep].gather(1, target[:, None])[:, 0].cpu().numpy()
            records.append({'name': view['name'], 'pixels': len(a), 'F_current': float(loss),
                            'a': save_array(cache_dir, f'{index:03d}.a.npy', a),
                            'labels': save_array(cache_dir, f'{index:03d}.y.npy', target.cpu().numpy().astype(np.uint8)),
                            'gradient': save_array(cache_dir, f'{index:03d}.g.npy', g)})
            del context, raw, loss, gradient, g, labels, keep, target
        counts['complete_passes'] += 1
        progress('current_gradient_complete')
        full_gradient /= 259
        vertex = linear_minimization_oracle(full_gradient)
        actual_d = vertex.astype(np.float32).astype(np.float64)-renderer.astype(np.float64)
        report['A_from_full_gradient'] = float(np.sum(full_gradient*actual_d))
        qv = torch.from_numpy(vertex.astype(np.float32)).cuda()
        with torch.no_grad():
            for view, record in zip(plan['training_views'], records, strict=True):
                context, labels, keep = view_context(view)
                raw = draw(context, qv, view)
                target = labels[keep]
                v = raw[keep].gather(1, target[:, None])[:, 0].cpu().numpy()
                record['v'] = save_array(cache_dir, f'{view["name"]}.v.npy', v)
                record['F_vertex'] = float(affine_raw_ce(raw, labels, keep, weights, SPEC['delta']))
                g = np.load(record['gradient'], mmap_mode='r', allow_pickle=False)
                record['A'] = float(np.sum(g.astype(np.float64)*actual_d))
                del context, raw, labels, keep, target, g
        counts['complete_passes'] += 1
        progress('vertex_forward_complete')
        fixed_cache_hashes = {str(path): sha(path) for path in sorted(cache_dir.iterdir())}
        cache_zero = line_summary(records, weights_np, 0., bins=True, deadline=deadline)
        a = float(np.mean([r['A'] for r in records]))
        gate = direction_gate(a, cache_zero['B64'], cache_zero['B32upstream'])
        measured_base = float(np.mean([r['F_current'] for r in records]))
        require(abs(cache_zero['F']-measured_base) <= 1e-12, 'CPU target-cache F differs from full objective')
        report.update(direction_gate=gate, derivative_at_zero=cache_zero,
                      A_from_per_view=float(a), full_gradient_vs_perview_dot_error=abs(a-report['A_from_full_gradient']),
                      baseline_objective=measured_base, vertex_objective=float(np.mean([r['F_vertex'] for r in records])))
        for row, source in zip(cache_zero['rows'], records, strict=True):
            row['A'] = source['A']; row['A_minus_B32upstream'] = source['A']-row['B32upstream']
        numerical_status = 'direction_inconsistent_or_unmeasurable'
        if gate['passed']:
            progress('scalar_search_start')
            gamma, search = fixed_line_search(lambda t: line_summary(
                records, weights_np, t, deadline=deadline, with_loss=False, validated=True)['B64'])
            line_candidate = line_summary(records, weights_np, gamma, deadline=deadline, validated=True)
            progress('scalar_search_complete')
            candidate_master = (1-gamma)*master + gamma*vertex
            validate_simplex(candidate_master)
            candidate32 = candidate_master.astype(np.float32)
            quantization = candidate32.astype(np.float64) - (renderer.astype(np.float64)+gamma*actual_d)
            report.update(gamma=float(gamma), line_search=search, cached_candidate=line_candidate,
                          line_input_rounding_max=float(np.max(np.abs(quantization))))
            numerical_status = 'fp32_stagnation'
            if not np.array_equal(candidate32, renderer):
                q_candidate = torch.from_numpy(candidate32).cuda()
                actual_rows = []
                with torch.no_grad():
                    for view in plan['training_views']:
                        context, labels, keep = view_context(view)
                        raw = draw(context, q_candidate, view)
                        actual_rows.append({'name': view['name'],
                                            'F': float(affine_raw_ce(raw, labels, keep, weights, SPEC['delta']))})
                        del context, raw, labels, keep
                counts['complete_passes'] += 1
                progress('candidate_forward_complete')
                measured = float(np.mean([r['F'] for r in actual_rows]))
                confirmation = candidate_gate(measured_base, line_candidate['F'], measured)
                report.update(actual_candidate_rows=actual_rows, actual_candidate_objective=measured,
                              candidate_confirmation=confirmation)
                numerical_status = ('direction_consistent_candidate_decreased' if confirmation['passed']
                                    else 'candidate_not_confirmed')
                report['diagnostic_candidate'] = save_array(cache_dir, 'candidate_q_master.npy', candidate_master)
        report['numerical_status'] = numerical_status
        require(counts['complete_passes'] in (2, 3) and counts['scene'] == 259*counts['complete_passes']
                and counts['gsplat'] == 2*counts['scene'] and counts['shader'] == counts['scene']
                and counts['vjp'] == 259, 'Call budget differs')
        report['cache_records'] = records
        report['cache_hashes'] = {str(path): sha(path) for path in sorted(cache_dir.iterdir())}
        require(all(report['cache_hashes'].get(path) == digest for path, digest in fixed_cache_hashes.items()),
                'Validated fixed cache changed during scalar search')
        report['cache_values_validated_before_search_and_sha_unchanged'] = True
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        from gsplat.cuda._backend import _C
        binary = {'path': str(Path(_C.__file__).resolve()), 'sha256': sha(_C.__file__)}
        require(binary == plan['expected_gsplat_binary'], 'Wrong installed renderer binary')
        report['actual_gsplat_binary'] = binary
        report['actual_imports'] = {}
        for name, module in sys.modules.items():
            if name.startswith('bridge_rgs') and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve()
                require(path.is_relative_to(plan['source_snapshot'])
                        and plan['source_hashes'][str(path.relative_to(plan['source_snapshot']))] == sha(path),
                        'Non-frozen package import')
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}
    finally:
        cv2.imread = old_imread
        try:
            if scene is not None and original is not None:
                unchanged = all(adapter.tensor_hash(value) == captured['hashes'][key]
                                for key, value in scene.state_dict().items())
                scene.load_state_dict(original, strict=True)
                report['restoration'] = adapter.restore_scene(scene, captured)
                report['state_unchanged_before_restore'] = unchanged
                require(unchanged and report['restoration']['state_exact']
                        and report['restoration']['flags_modes_gradients_restored'], 'Restoration failed')
        finally:
            adapter.numerical_flags(old_flags)
            report['numerics_restored'] = adapter.numerical_flags() == old_flags


def execute(path, expected):
    require(sha(path) == expected, 'Plan SHA differs')
    plan = read(path); output = Path(plan['output'])
    for name in ('execution_started.json', 'execution_receipt.json', 'analysis.json', 'cache'):
        require(not (output/name).exists(), 'Refuse previous attempt/outputs')
    write(output/'execution_started.json', {'plan_sha256': expected, 'pid': os.getpid(), 'time': time.time()})
    started = time.monotonic()
    report = {'status': 'running', 'plan_sha256': expected,
              'counts': {key: 0 for key in ('scene', 'gsplat', 'shader', 'vjp', 'target_decodes', 'complete_passes')},
              'val_views': 0, 'teacher_calls': 0, 'model_updates': 0, 'head_calls': 0}
    def timeout(*_):
        raise TimeoutError('Fixed600-second diagnostic deadline; no retries')
    prior = signal.signal(signal.SIGALRM, timeout); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan)
        sys.path.insert(0, plan['source_snapshot'])
        old = importlib.import_module('optimize_raw_simplex')
        report['gpu_before'] = old.gpu_inventory()
        perform(plan, report, started+SPEC['internal_seconds'])
        verify(plan)
        write(output/'analysis.json', {k: v for k, v in report.items() if k != 'status'})
        report.update(status='completed', analysis_sha256=sha(output/'analysis.json'), inputs_sources_unchanged=True)
    except BaseException as error:
        report.update(status='inconclusive_timeout' if isinstance(error, TimeoutError) else 'failed',
                      error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, prior)
        report['elapsed_seconds'] = time.monotonic()-started
        report['counts']['total_raster'] = report['counts']['gsplat']+report['counts']['shader']
        write(output/'execution_receipt.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path); group.add_argument('--run', type=Path)
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode generation')
    if args.prepare:
        prepare(args.prepare)
    else:
        execute(args.run, args.expected_plan_sha256)


if __name__ == '__main__':
    main()
