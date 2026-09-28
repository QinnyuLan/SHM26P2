"""Two matched means-only routes through one frozen scene readout; TRAIN only.

Approximate standard-gsplat VJPs are proposals, not derivative certificates.
Full and prior-only arms have identical forward functions and compute budgets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
PROBE = Path('/mnt/data/SHM2026/runs/readout_geometry_probe_v1')
PARENT = Path('/mnt/data/SHM2026/runs/continuous_geometry_proposals_v1')
ARMS = ('full', 'prior_only')
SPEC = {
    'protocol': 'scene_geometry_routes_v1', 'arms': list(ARMS),
    'point_count': 498136, 'train_views': 259, 'rounds': 4, 'batch_size': 7,
    'attempts_per_arm': 148, 'seed': 42,
    'sampling': 'Python random.Random(42); shuffle a fresh sorted index list each round',
    'semantic_weight': .03, 'affine_noise': 5e-7,
    'semantic_objective': 'known-valid class-weighted affine-noise CE of frozen scene probabilities; no Lovasz/residual loss',
    'prior_route': 'external log p and head p3d/entropy all live; all other head inputs detached',
    'context_route': 'full minus prior means VJP, descriptive; no gradient projection',
    'means_lr_scene_scale_factor': 1.6e-6, 'adam_betas': [.9, .999], 'adam_eps': 1e-15,
    'cap': 1/64, 'cap_scope': 'per-point per-step fixed original covariance; not cumulative',
    'acceptance': 'every complete finite candidate including no-op; no loss-based selection',
    'gradient': 'three per-view FP32 VJPs cast to FP64, sum then divide by seven',
    'description': 'shared initial and both fixed final full259 TRAIN raw/scene metrics',
    'frozen': 'q, head, features, SH, opacity, scale, rotation, cameras and all non-means state',
    'expected_counts': {'batch_attempts': 296, 'training_view_pairs': 4144,
                        'description_view_pairs': 777, 'adapter_calls': 4921,
                        'standard_gsplat_calls': 9842, 'low_level_raster_calls': 14763,
                        'means_vjp': 6216, 'head_calls': 6993, 'target_decodes': 777},
    'teacher_calls': 0, 'val_views': 0, 'production_checkpoint_writes': 0,
    'internal_seconds': 600, 'external_seconds': 660,
    'interpretation': 'path ablation, approximate VJP; no innovation/convergence/adoption or cosine-causality claim',
}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def write(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as handle:
        handle.write(payload)

def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def files(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}

def batch_schedule(names):
    require(len(names) == len(set(names)) == 259 and names == sorted(names), 'Fixed sorted259 TRAIN population')
    generator = random.Random(42); result = []
    for epoch in range(4):
        indices = list(range(259)); generator.shuffle(indices)
        for offset in range(0, 259, 7):
            result.append({'round': epoch, 'names': [names[i] for i in indices[offset:offset+7]]})
    require(len(result) == 148 and all(len(r['names']) == 7 for r in result), 'Fixed148 complete batches')
    return result


def prior_readout(refiner, context):
    """Forward-identical head, with every non-prior evidence route stopped."""
    p = context['p3d']
    residual = refiner(context['features'].detach(), context['rgb'].detach(),
                       context['depth'].detach(), context['alpha'].detach(), p3d=p,
                       depth_moments=context['depth_moments'].detach(), geometry_grad=True)
    return (p.log()+residual).softmax(-1)


def gradient_summary(g_rgb, g_full, g_prior, delta=None):
    """Statistics of averaged means gradients, not mean per-view inner products."""
    import torch
    gradients = {'rgb': g_rgb, 'full': g_full, 'prior': g_prior, 'context': g_full-g_prior}
    require(all(x.dtype == torch.float64 and x.shape == g_rgb.shape
                and bool(torch.isfinite(x).all()) for x in gradients.values()), 'Invalid averaged gradients')
    result = {'l2': {k: float(v.norm()) for k, v in gradients.items()},
              'dots': {f'{a}_{b}': float((gradients[a]*gradients[b]).sum())
                       for a, b in [('prior', 'context'), ('rgb', 'full'), ('rgb', 'prior'), ('rgb', 'context')]},
              'aggregation': 'inner products after equal-view gradient mean; not per-view conflict fraction'}
    if delta is not None:
        require(delta.dtype == torch.float64 and delta.shape == g_rgb.shape
                and bool(torch.isfinite(delta).all()), 'Invalid actual FP32 displacement')
        result['actual_dots'] = {k: float((v*delta).sum()) for k, v in gradients.items()}
        result['actual_delta_l2'] = float(delta.norm())
    return result


def evaluate_view_objectives(names, means, view_objectives, counts, *, gradient=False, descriptive=False):
    """Three VJPs share a real forward graph; retain it until the prior VJP."""
    import torch
    require(len(names) > 0 and len(names) == len(set(names)), 'Distinct view population required')
    rows = []
    accumulators = [torch.zeros_like(means, dtype=torch.float64) for _ in range(3)] if gradient else []
    for name in names:
        with torch.set_grad_enabled(gradient):
            rgb, full, prior, metadata = view_objectives(name)
        values = [float(x.detach()) for x in (rgb, full, prior)]
        require(np.isfinite(values).all() and min(values) >= 0, 'Nonfinite/negative objective')
        require(values[1] == values[2], 'Prior-route loss forward differs from full scene loss')
        rows.append({'name': name, 'rgb_mse': values[0], 'scene_ce': values[1], **metadata})
        if gradient:
            for index, (loss, accumulator) in enumerate(zip((rgb, full, prior), accumulators, strict=True)):
                g, = torch.autograd.grad(loss, means, retain_graph=index < 2)
                require(g.dtype == torch.float32 and bool(torch.isfinite(g).all()), 'Invalid FP32 means VJP')
                accumulator.add_(g.detach().double()); counts['means_vjp'] += 1
        counts['description_view_pairs' if descriptive else 'training_view_pairs'] += 1
    result = {'complete': True, 'names': names, 'rows': rows}
    for key in ('rgb_mse', 'scene_ce', 'raw_ce'):
        result[key] = float(np.mean([row[key] for row in rows]))
    if gradient:
        result.update(dict(zip(('g_rgb', 'g_full', 'g_prior'), [g/len(names) for g in accumulators], strict=True)))
        result['path_gradients'] = gradient_summary(result['g_rgb'], result['g_full'], result['g_prior'])
    if descriptive:
        for kind in ('raw', 'scene'):
            cm = np.sum([row[kind+'_confusion_matrix'] for row in rows], axis=0, dtype=np.int64)
            union = cm.sum(0)+cm.sum(1)-np.diag(cm)
            iou = np.divide(np.diag(cm), union, out=np.full(5, np.nan), where=union > 0)
            result[kind] = {'confusion_matrix': cm.tolist(),
                            'iou': [float(x) if np.isfinite(x) else None for x in iou],
                            'miou_all': float(np.nanmean(iou))}
        result['scope'] = 'TRAIN description, not held-out quality'
    return result


def accept_candidate(arm, baseline, candidate):
    require(arm in ARMS, 'Unknown arm')
    require(baseline['names'] == candidate['names'] and len(baseline['names']) == 7, 'Batch population changed')
    for result in (baseline, candidate):
        require(result.get('complete') is True and [r['name'] for r in result['rows']] == result['names'], 'Partial batch')
        for key in ('rgb_mse', 'raw_ce', 'scene_ce'):
            values = [r[key] for r in result['rows']]
            require(np.isfinite(values).all() and np.min(values) >= 0
                    and result[key] == float(np.mean(values)), 'Invalid objective reduction')
    return True, {'complete_finite_commit': True, 'no_descent_guarantee': True,
                  'candidate_minus_baseline': {k: float(candidate[k]-baseline[k])
                                                for k in ('rgb_mse', 'raw_ce', 'scene_ce')}}


def run_attempts(arm, schedule, transaction, evaluate_batch, emit=lambda row: None):
    """Only the chosen semantic route changes; both arms compute all three VJPs."""
    require(arm in ARMS, 'Unknown arm'); history = []
    for index, batch in enumerate(schedule):
        names = batch['names']; require(len(names) == len(set(names)) == 7, 'Seven distinct batch views')
        baseline = evaluate_batch(names, gradient=True)
        accept_candidate(arm, baseline, baseline)
        before = transaction.means.detach().double().clone()
        route = baseline['g_full' if arm == 'full' else 'g_prior']
        proposed, proposal = transaction.propose(baseline['g_rgb'], route, 'joint')
        accepted = False
        try:
            paths = gradient_summary(baseline['g_rgb'], baseline['g_full'], baseline['g_prior'], proposed.double()-before)
            candidate = evaluate_batch(names, gradient=False)
            accepted, acceptance = accept_candidate(arm, baseline, candidate)
        finally:
            transaction.resolve(accepted)
        row = {'attempt': index+1, 'round': batch['round'], 'names': names, 'accepted': accepted,
               'accepted_optimizer_steps': index+1, 'route': arm, 'proposal': proposal,
               'path_gradients': paths, 'acceptance': acceptance,
               'baseline': {k: v for k, v in baseline.items() if k not in ('g_rgb', 'g_full', 'g_prior')},
               'candidate': candidate}
        emit(row); history.append(row)
    return {'attempts': len(history), 'accepted': len(history), 'rejected': 0, 'history': history,
            'stop_reason': 'fixed_148_attempt_schedule_finished', 'converged_claimed': False}


def validate_probe(probe, execution, launch, review, digests):
    require(digests['plan'] == '499e7b76cab849dc9515438c853fd12d9d4e3aa24456211c06778480b0051f7e'
            and digests['execution'] == 'bd8a14883622e195a06d5aa47d5e53ea5af4f04c7154def4b0b8223ddc6dfd8c', 'Fixed probe identity')
    require(execution['status'] == launch['status'] == 'completed' and launch['natural_completion'] is True
            and launch['exit_code'] == 0 and launch['execution_receipt_sha256'] == digests['execution']
            and execution['plan_sha256'] == launch['plan_sha256'] == digests['plan'], 'Incomplete probe')
    require(review['status'] == 'passed' and review['execution_receipt_sha256'] == digests['execution'], 'Probe audit missing/mismatched')
    require(execution['sources_inputs_unchanged'] and execution['numerics_restored']
            and execution['restoration']['state_exact'] and execution['restoration']['flags_modes_gradients_restored'], 'Probe restoration')
    require([r['name'] for r in execution['views']] == ['002.png', '041.png'], 'Probe view identity')
    for row in execution['views']:
        require(all(row['forward_comparison'][key]['exact'] for key in
                    ('rgb', 'probabilities', 'depth', 'alpha', 'features', 'depth_moments'))
                and row['forward_mask_differences'] == {'raw': 0, 'probabilities': 0}, 'Probe forward compatibility failed')
    require(execution['numerics_actual'] == probe['numerics'], 'Probe numerical flags differ')


def prepare(output, probe=PROBE, parent=PARENT):
    output, probe, parent = (Path(p).resolve() for p in (output, probe, parent))
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    paths = {role: probe/name for role, name in [('plan', 'plan.json'), ('execution', 'execution_receipt.json'),
             ('launch', 'launch_receipt.json'), ('review', 'independent_cpu_review.json')]}
    bundles = {key: read(path) for key, path in paths.items()}; digests = {key: sha(path) for key, path in paths.items()}
    validate_probe(bundles['plan'], bundles['execution'], bundles['launch'], bundles['review'], digests)
    prior = bundles['plan']; train_plan = read(parent/'plan.json')
    require(sha(parent/'plan.json') == '61c66187e63325fa0d23e8ea9b8522165f9335b93424f3daa54b612e4553b588', 'Fixed parent population')
    require(files(prior['source_snapshot']) == prior['source_hashes'], 'Probe package changed')
    for key in ('base_checkpoint', 'manifest', 'q_delta', 'numerics', 'expected_gsplat_binary', 'class_weights'):
        require(prior[key] == train_plan[key], 'Parent/probe identity mismatch: '+key)
    inputs = dict(prior['input_hashes'])
    for path in [*paths.values(), *[parent/name for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json')]]:
        inputs[str(path)] = sha(path)
    views = train_plan['training_views']; names = [v['name'] for v in views]
    for view in views:
        require(view['split'] == 'train', 'TRAIN only')
        for key in ('image_path', 'mask_path', 'valid_path'):
            inputs[view[key]] = train_plan['input_hashes'][view[key]]
    require(all(sha(path) == digest for path, digest in inputs.items()), 'Bound inputs changed')
    snapshot = output/'source_snapshot'
    shutil.copytree(prior['source_snapshot'], snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in (Path(__file__), ROOT/'tests/test_scene_geometry_routes.py', ROOT/'docs/scene_geometry_routes_protocol.md'):
        require(not (snapshot/path.name).exists(), 'Refuse inherited source overwrite')
        shutil.copy2(path, snapshot/path.name)
    plan = {k: prior[k] for k in ('base_checkpoint', 'manifest', 'q_delta', 'expected_gsplat_binary',
            'installed_sources', 'runtime_versions', 'environment', 'numerics', 'class_weights')}
    plan.update(specification=SPEC, output=str(output), source_snapshot=str(snapshot), source_hashes=files(snapshot),
                inherited_source_hashes=prior['source_hashes'], input_hashes=inputs, training_views=views,
                all_training_camera_names=train_plan['all_training_camera_names'], batches=batch_schedule(names),
                probe_paths={k: str(v) for k, v in paths.items()}, probe_sha256=digests,
                status='prepared_pending_root_launch', prepare_pixel_decodes=0, prepare_checkpoint_loads=0)
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC and files(plan['source_snapshot']) == plan['source_hashes'], 'Frozen source/spec changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Frozen worker only')
    require(plan['batches'] == batch_schedule([v['name'] for v in plan['training_views']]), 'Changed batch order')
    binary = plan['expected_gsplat_binary']
    for path, digest in {**plan['input_hashes'], **plan['installed_sources'], binary['path']: binary['sha256']}.items():
        require(sha(path) == digest, 'Bound bytes changed: '+path)
    for key, value in plan['runtime_versions'].items():
        require(importlib.metadata.version(key) == value, 'Runtime changed')
    for key, value in plan['environment'].items():
        require(os.environ.get(key) == value, 'Environment changed')

def perform(plan, report, deadline):
    import cv2
    import torch
    from gsplat import rasterization, rendering

    require(not any(n.startswith('bridge_rgs') for n in sys.modules), 'No live bridge package')
    adapter = importlib.import_module('preflight_simplex_scene')
    from bridge_rgs.continuous_geometry import MeansTransaction
    from bridge_rgs.direct_q_geometry_render import render_direct_q_geometry
    from bridge_rgs.semantic_assignment import affine_raw_ce
    from bridge_rgs.train import load_scene
    torch.set_num_threads(4)
    counts = report['counts']; output = Path(plan['output']); by_name = {v['name']: v for v in plan['training_views']}
    allowed = {v[k] for v in plan['training_views'] for k in ('image_path', 'mask_path', 'valid_path')}
    previous_flags = adapter.numerical_flags(plan['numerics']); previous_imread = cv2.imread
    scene = captured = original = head_handle = None; targets = {}
    previous_low_raster = rendering.rasterize_to_pixels

    def low_raster(*args, **kwargs):
        counts['low_level_raster_calls'] += 1
        return previous_low_raster(*args, **kwargs)

    def standard_raster(*args, **kwargs):
        counts['standard_gsplat_calls'] += 1
        return rasterization(*args, **kwargs)

    def head_count(*_):
        counts['head_calls'] += 1

    rendering.rasterize_to_pixels = low_raster

    def guarded_read(path, *args, **kwargs):
        require(str(path) in allowed, 'Only fixed labeled TRAIN targets may be decoded')
        return previous_imread(path, *args, **kwargs)
    cv2.imread = guarded_read

    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError('Fixed training deadline, no retry or truncated schedule')
        require(counts['low_level_raster_calls'] <= SPEC['expected_counts']['low_level_raster_calls']
                and counts['means_vjp'] <= SPEC['expected_counts']['means_vjp'], 'Fixed compute budget exceeded')

    def target(view):
        name = view['name']
        if name not in targets:
            rgb = cv2.imread(view['image_path'], cv2.IMREAD_COLOR)
            labels = cv2.imread(view['mask_path'], cv2.IMREAD_UNCHANGED)
            valid = cv2.imread(view['valid_path'], cv2.IMREAD_UNCHANGED)
            shape = (view['height'], view['width'])
            require(rgb is not None and labels is not None and valid is not None
                    and rgb.shape == (*shape, 3) and labels.shape == valid.shape == shape
                    and np.isin(labels, [0, 1, 2, 3, 4, 255]).all(), 'Invalid original prepared TRAIN grid')
            valid = valid > 0; require(valid.any() and (valid & (labels < 5)).any(), 'Empty TRAIN support')
            targets[name] = (cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), labels, valid)
            counts['target_decodes'] += 3
        rgb, labels, valid = targets[name]
        return (torch.from_numpy(rgb).to('cuda').double()/255,
                torch.from_numpy(labels.astype(np.int64)).to('cuda'), torch.from_numpy(valid).to('cuda'))

    try:
        report['numerics_actual'] = adapter.numerical_flags(); require(report['numerics_actual'] == plan['numerics'], 'Numerical flags')
        scene, checkpoint = load_scene(plan['base_checkpoint']); captured = adapter.capture_scene(scene)
        original = {k: v.detach().cpu().clone() for k, v in scene.state_dict().items()}
        head_handle = scene.refiner.register_forward_hook(head_count)
        require(scene.sh_degree == 3 and scene.pixel_protocol == 'legacy_mixed_v1' and scene.mip_filter_config is None
                and len(scene.splats['means']) == SPEC['point_count'], 'Wrong H3 field')
        train = [v for v in read(plan['manifest'])['views'] if v['split'] == 'train']
        require([v['name'] for v in train] == plan['all_training_camera_names']
                and np.array_equal(checkpoint['training_cameras'].cpu().numpy(),
                                   np.asarray([v['w2c_original'] for v in train], np.float32)), 'Original TRAIN cameras')
        scene.eval().requires_grad_(False); means = scene.splats['means']; means.requires_grad_(True)
        with np.load(plan['q_delta'], allow_pickle=False) as data:
            q32 = data['q_renderer'].copy()
            require(q32.dtype == np.float32 and np.array_equal(q32, data['q_master'].astype(np.float32)), 'Frozen q cast')
        require(q32.shape == (SPEC['point_count'], 5) and np.isfinite(q32).all() and (q32 >= 0).all()
                and np.max(np.abs(q32.astype(np.float64).sum(1)-1)) <= 1e-6, 'Fixed q simplex/cast')
        q = torch.from_numpy(q32).cuda(); q_hash = adapter.tensor_hash(q)
        base_means = means.detach().cpu().numpy().copy(); base_sha = array_sha(base_means)
        weights = torch.tensor(plan['class_weights'], device='cuda', dtype=torch.float32)
        fixed_hashes = {k: adapter.tensor_hash(v) for k, v in scene.state_dict().items() if k != 'splats.means'}
        report.update(base_means_sha256=base_sha, q_renderer_sha256=q_hash, frozen_state_hashes=fixed_hashes)
        torch.cuda.reset_peak_memory_stats()

        def fixed_state():
            require({k: adapter.tensor_hash(v) for k, v in scene.state_dict().items() if k != 'splats.means'} == fixed_hashes,
                    'Non-means state changed')
            require(adapter.tensor_hash(q) == q_hash and all(p.grad is None for p in scene.parameters()), 'q or accumulated gradient changed')

        def evaluate(names, gradient=False, descriptive=False):
            tick(); before = adapter.tensor_hash(means)
            def view_objectives(name):
                tick(); view = by_name[name]; truth, labels, valid = target(view)
                K = means.new_tensor(view['K']); w2c = means.new_tensor(view['w2c_original'])
                result = render_direct_q_geometry(scene, q, K, w2c, view['width'], view['height'],
                                                  geometry_grad=True, rasterize=standard_raster)
                counts['adapter_calls'] += 1
                rgb_loss = (result['rgb'].clamp(0, 1).double()[valid]-truth[valid]).square().mean()
                full_loss = affine_raw_ce(result['probabilities'], labels, valid, weights, SPEC['affine_noise'])
                if gradient:
                    prior_prob = prior_readout(scene.refiner, result)
                    require(torch.equal(prior_prob, result['probabilities']), 'Prior/full probabilities forward must be exact')
                    prior_loss = affine_raw_ce(prior_prob, labels, valid, weights, SPEC['affine_noise'])
                else:
                    prior_loss = full_loss
                # Raw CE is descriptive only; it never enters the geometry proposal.
                with torch.no_grad():
                    raw_loss = affine_raw_ce(result['raw'], labels, valid, weights, SPEC['affine_noise'])
                metadata = {'rgb_pixels': int(valid.sum()), 'semantic_pixels': int((valid & (labels < 5)).sum()),
                            'raw_ce': float(raw_loss), 'prior_forward_exact': True if gradient else None}
                if descriptive:
                    known = valid & (labels < 5)
                    for kind, tensor in (('raw', result['raw']), ('scene', result['probabilities'])):
                        ids = (5*labels[known]+tensor.detach().argmax(-1)[known]).cpu().numpy()
                        metadata[kind+'_confusion_matrix'] = np.bincount(ids, minlength=25).reshape(5, 5).tolist()
                return rgb_loss, full_loss, prior_loss, metadata
            result = evaluate_view_objectives(names, means, view_objectives, counts,
                                              gradient=gradient, descriptive=descriptive)
            require(adapter.tensor_hash(means) == before, 'Means changed within complete batch/pass')
            result['means_sha256'] = before
            return result

        names = sorted(by_name); baseline = evaluate(names, descriptive=True)
        write(output/'baseline_train.json', baseline); report['baseline'] = {'path': str(output/'baseline_train.json'), 'sha256': sha(output/'baseline_train.json')}
        report['arms'] = []
        for arm in ARMS:
            directory = output/arm; directory.mkdir()
            scene.load_state_dict(original, strict=True); scene.eval().requires_grad_(False); means.requires_grad_(True)
            require(adapter.tensor_hash(means) == base_sha, 'Arm initial means differ'); fixed_state()
            tx = MeansTransaction(means, scene.splats['quats'], scene.splats['log_scales'], scene.scene_scale)
            started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
            def emit(row, directory=directory, arm=arm, started=started):
                counts['batch_attempts'] += 1
                with (directory/'attempts.jsonl').open('a') as handle:
                    handle.write(json.dumps(row, allow_nan=False)+'\n')
                if row['attempt'] % 37 == 0:
                    print(json.dumps({'arm': arm, 'attempt': row['attempt'],
                                      'elapsed_seconds': time.monotonic()-started, 'counts': counts}), flush=True)
            result = run_attempts(arm, plan['batches'], tx, evaluate, emit)
            require(result['attempts'] == 148 and tx.pending is None, 'Incomplete final endpoint')
            optimizer_steps = [int(v['step']) for v in tx.optimizer.state.values()]
            require(optimizer_steps in ([result['accepted']], []) and (optimizer_steps or result['accepted'] == 0), 'Adam accepted-step mismatch')
            fixed_state(); endpoint = evaluate(names, descriptive=True); fixed_state()
            final = means.detach().cpu().numpy().copy(); displacement = final.astype(np.float64)-base_means.astype(np.float64)
            local = torch.einsum('nij,ni->nj', tx.rotation, means.detach().double()-torch.from_numpy(base_means).cuda().double())
            cumulative = (local*tx.inverse_scales).norm(dim=-1)
            np.savez(directory/'means_delta.npz', base_means=base_means, means_final=final, actual_delta=displacement)
            torch.save(tx.optimizer.state_dict(), directory/'optimizer_state.pt')
            write(directory/'endpoint_train.json', endpoint)
            write(directory/'training_receipt.json', {'status': 'completed', 'arm': arm, 'plan_sha256': report['plan_sha256'],
                'attempts': result['attempts'], 'accepted': result['accepted'], 'rejected': result['rejected'],
                'stop_reason': result['stop_reason'], 'converged_claimed': False,
                'base_means_sha256': base_sha, 'final_means_sha256': array_sha(final), 'q_renderer_sha256': q_hash,
                'frozen_state_hashes': fixed_hashes, 'optimizer_steps': optimizer_steps,
                'attempts_sha256': sha(directory/'attempts.jsonl'), 'means_delta_sha256': sha(directory/'means_delta.npz'),
                'optimizer_state_sha256': sha(directory/'optimizer_state.pt'), 'endpoint_sha256': sha(directory/'endpoint_train.json'),
                'cumulative_mahalanobis_max': float(cumulative.max()), 'cumulative_world_l2_max': float(np.linalg.norm(displacement, axis=1).max()),
                'elapsed_seconds': time.monotonic()-started, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                'production_checkpoint': False, 'ordinary_resume_supported': False})
            report['arms'].append({'arm': arm, 'path': str(directory/'training_receipt.json'), 'sha256': sha(directory/'training_receipt.json')})
            print(json.dumps({'arm': arm, 'attempts': result['attempts'], 'accepted': result['accepted'], 'counts': counts}), flush=True)
            del tx, result
        require(all(counts[k] == v for k, v in SPEC['expected_counts'].items()) and counts['target_decodes'] == 777, 'Fixed total counts')
    finally:
        cv2.imread = previous_imread
        rendering.rasterize_to_pixels = previous_low_raster
        if head_handle is not None:
            head_handle.remove()
        try:
            if scene is not None and original is not None:
                scene.load_state_dict(original, strict=True)
                report['restoration'] = adapter.restore_scene(scene, captured)
                require(report['restoration']['state_exact'] and report['restoration']['flags_modes_gradients_restored'], 'Full state restoration failed')
        finally:
            adapter.numerical_flags(previous_flags); report['numerics_restored'] = adapter.numerical_flags() == previous_flags
            require(report['numerics_restored'], 'Numerical flag restoration failed')
        report['actual_imports'] = {}
        for name, module in list(sys.modules.items()):
            if (name.startswith('bridge_rgs') or name in ('preflight_simplex_scene', 'optimize_raw_simplex')) and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve()
                require(path.is_relative_to(plan['source_snapshot']) and plan['source_hashes'].get(str(path.relative_to(plan['source_snapshot']))) == sha(path), 'Nonfrozen actual import')
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}
        if counts['standard_gsplat_calls']:
            from gsplat.cuda._backend import _C
            report['actual_gsplat_binary'] = {'path': str(Path(_C.__file__).resolve()), 'sha256': sha(_C.__file__)}
            require(report['actual_gsplat_binary'] == plan['expected_gsplat_binary'], 'Actual renderer binary differs')

def execute(path, expected):
    require(sha(path) == expected, 'Wrong plan SHA'); plan = read(path); output = Path(plan['output'])
    require(not any((output/n).exists() for n in ('execution_started.json', 'execution_receipt.json', 'baseline_train.json')), 'No rerun/overwrite')
    write(output/'execution_started.json', {'plan_sha256': expected, 'pid': os.getpid(), 'time': time.time()})
    report = {'status': 'running', 'plan_sha256': expected,
              'counts': dict.fromkeys([*SPEC['expected_counts']], 0),
              'teacher_calls': 0, 'val_views': 0, 'production_checkpoint_writes': 0,
              'interpretation': SPEC['interpretation'], 'adoption': 'none'}
    started = time.monotonic()
    def timeout(*_):
        raise TimeoutError('Fixed full-run budget, no rescue')
    old_handler = signal.signal(signal.SIGALRM, timeout); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan); sys.path.insert(0, plan['source_snapshot'])
        old = importlib.import_module('optimize_raw_simplex'); report['gpu_before'] = old.gpu_inventory()
        perform(plan, report, started+SPEC['internal_seconds']); verify(plan)
        report.update(status='completed', sources_inputs_unchanged=True)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
        report['elapsed_seconds'] = time.monotonic()-started; write(output/'execution_receipt.json', report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path); action.add_argument('--execute', type=Path)
    parser.add_argument('--probe', type=Path, default=PROBE); parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--expected-plan-sha256'); args = parser.parse_args()
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode generation')
    if args.prepare:
        prepare(args.prepare, args.probe, args.parent)
    else:
        require(args.expected_plan_sha256 is not None, 'Expected plan SHA required')
        execute(args.execute, args.expected_plan_sha256)
