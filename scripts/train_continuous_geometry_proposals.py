"""Fixed approximate-VJP proposals with actual batch acceptance; explicit new protocol.

Inherits the actual standard-gsplat preflight package. No production checkpoint,
teacher, head fitting, VAL scoring, or hyperparameter selection is implemented.
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
RGB_SOURCE = Path('/mnt/data/SHM2026/runs/matched_rgb_teacher_adaptation_v1/plan.json')
ARMS = ('joint', 'pcgrad', 'trust')
EVIDENCE_SHA = {
    'preflight_plan': '6a79de1060458df432b8df8b37edb0a5c6d2a2f9028cc2d4f37f901303787845',
    'preflight_execution': '90ef418aecdbe909ca802ffecdd1c8bcd136c760af9fb5760ab44a3aef3b908e',
    'preflight_launch': '3a8ac3ce970f4ce7725e08f0e76a4800d63dade5e73b92c1789573bd46685998',
    'preflight_review': '0f51b71d006deba716f5470a0d63f2038c837b9b4972abed5146a7c4348e4718',
    'chain_plan': '059bf2e5c4ca558a6252ec8053f2759a6f7ae49174bedbacaa345973acfff43b',
    'chain_execution': '386831daf98b1b3ebc0b28e5086a75709aaa599e175dd18cb9f880bb0214ff57',
    'chain_launch': '14de26714284ac20700fc0c15d2f31c7978e0399afc40dd7cd5d8d004ea3840d',
    'chain_review': 'aae5f60be6d1b7038c0288201ddc7af307bbe766c0f0690c62f481f2ac15fd6a',
}
SPEC = {
    'protocol': 'continuous_geometry_proposals_v1', 'arms': list(ARMS),
    'gradient_interpretation': 'local approximate VJP proposal; old finite-difference calibration remains failed',
    'adoption': 'none; fixed TRAIN description only, no new formula or generalization claim',
    'point_count': 498136, 'train_views': 259, 'rounds': 4, 'batch_size': 7,
    'attempts_per_arm': 148, 'seed': 42,
    'sampling': 'Python random.Random(42); shuffle a fresh sorted index list each round',
    'semantic_weight': .03, 'affine_noise': 5e-7,
    'means_lr_scene_scale_factor': 1.6e-6, 'adam_betas': [.9, .999], 'adam_eps': 1e-15,
    'cap': 1/64, 'cap_scope': 'per-point per-step original covariance, not cumulative displacement',
    'joint_pcgrad_acceptance': 'every complete finite proposal, including no-op',
    'trust': {'strict_batch_semantic_decrease': True, 'rgb_relative_tolerance': 1e-6,
              'rgb_absolute_tolerance': 1e-12, 'per_view_rgb_factor': 1.001},
    'gradient': 'each view FP32 means VJP cast to FP64, sum then divide by seven',
    'losses': 'valid RGB clamp MSE and known-valid weighted affine raw CE, each view equally weighted',
    'frozen': 'all scene state except means; audited EMFW q_renderer cast unchanged',
    'description': 'one shared initial and three fixed-last full259 TRAIN passes, no generalization claim',
    'expected_counts': {'batch_attempts': 444, 'training_view_pairs': 6216,
                        'description_view_pairs': 1036, 'raster': 14504, 'means_vjp': 6216},
    'head_calls': 0, 'teacher_calls': 0, 'val_views': 0, 'production_checkpoint_writes': 0,
    'internal_seconds': 1800, 'external_seconds': 1860,
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


def accept_candidate(arm, baseline, candidate):
    require(arm in ARMS, 'Unknown arm')
    require(baseline['names'] == candidate['names'] and len(baseline['rows']) == len(candidate['rows']) == 7,
            'Candidate batch population differs')
    for value in (baseline, candidate):
        require(value.get('complete') is True and [r['name'] for r in value['rows']] == value['names'], 'Partial batch')
        values = [value['rgb_mse'], value['raw_ce']]
        values += [r[k] for r in value['rows'] for k in ('rgb_mse', 'raw_ce')]
        require(np.isfinite(values).all() and np.all(np.asarray(values) >= 0), 'Nonfinite/negative batch objective')
        require(value['rgb_mse'] == float(np.mean([r['rgb_mse'] for r in value['rows']]))
                and value['raw_ce'] == float(np.mean([r['raw_ce'] for r in value['rows']])), 'Per-view mean differs')
    tau = max(1e-6*baseline['rgb_mse'], 1e-12)
    criteria = {'semantic_decreases': candidate['raw_ce'] < baseline['raw_ce'],
                'batch_rgb_protected': candidate['rgb_mse'] <= baseline['rgb_mse']+tau,
                'every_view_rgb_protected': all(c['rgb_mse'] <= 1.001*b['rgb_mse']
                    for b, c in zip(baseline['rows'], candidate['rows'], strict=True))}
    accepted = arm != 'trust' or all(criteria.values())
    return bool(accepted), {'criteria': {k: bool(v) for k, v in criteria.items()}, 'rgb_tolerance': float(tau),
                            'candidate_minus_baseline_rgb': float(candidate['rgb_mse']-baseline['rgb_mse']),
                            'candidate_minus_baseline_semantic': float(candidate['raw_ce']-baseline['raw_ce']),
                            'control_finite_commit': arm != 'trust',
                            'scope': 'current batch only, no fullTRAIN or VAL protection guarantee'}


def evaluate_view_objectives(names, means, view_objectives, counts, *, gradient=False, descriptive=False):
    """Production reduction: complete per-view losses, equally weighted views.

    The callback supplies two independent scalar graphs plus support metadata.
    Each FP32 means VJP is converted before FP64 accumulation; valid and known
    pixel counts are descriptive, never weights for the across-view reduction.
    """
    import torch

    require(len(names) > 0 and len(names) == len(set(names)), 'Distinct nonempty view population')
    rows = []
    g_rgb = torch.zeros_like(means, dtype=torch.float64) if gradient else None
    g_semantic = torch.zeros_like(means, dtype=torch.float64) if gradient else None
    for name in names:
        with torch.set_grad_enabled(gradient):
            rgb_loss, semantic_loss, metadata = view_objectives(name)
        values = [float(rgb_loss.detach()), float(semantic_loss.detach())]
        require(np.isfinite(values).all(), 'Nonfinite actual objective')
        row = {'name': name, 'rgb_mse': values[0], 'raw_ce': values[1], **metadata}
        if gradient:
            for loss, accumulator in ((rgb_loss, g_rgb), (semantic_loss, g_semantic)):
                g, = torch.autograd.grad(loss, means)
                require(g.dtype == torch.float32 and bool(torch.isfinite(g).all()), 'Nonfinite/non-FP32 means VJP')
                accumulator.add_(g.detach().double()); counts['means_vjp'] += 1
        rows.append(row)
        counts['description_view_pairs' if descriptive else 'training_view_pairs'] += 1
    result = {'complete': True, 'names': names, 'rows': rows,
              'rgb_mse': float(np.mean([r['rgb_mse'] for r in rows])),
              'raw_ce': float(np.mean([r['raw_ce'] for r in rows]))}
    if gradient:
        result.update(g_rgb=g_rgb/len(names), g_semantic=g_semantic/len(names))
    if descriptive:
        cm = np.sum([r['confusion_matrix'] for r in rows], axis=0, dtype=np.int64)
        union = cm.sum(0)+cm.sum(1)-np.diag(cm)
        iou = np.divide(np.diag(cm), union, out=np.full(5, np.nan), where=union > 0)
        result.update(confusion_matrix=cm.tolist(), iou=[float(x) if np.isfinite(x) else None for x in iou],
                      miou_all=float(np.nanmean(iou)), scope='TRAIN description, not held-out quality')
    return result


def run_attempts(arm, schedule, transaction, evaluate_batch, emit=lambda row: None):
    """A pending proposal is resolved exactly once; exceptions reject completely."""
    require(arm in ARMS, 'Unknown arm'); history = []; accepted_steps = 0
    for index, batch in enumerate(schedule):
        names = batch['names']
        require(len(names) == len(set(names)) == 7, 'Seven distinct views per attempt')
        baseline = evaluate_batch(names, gradient=True)
        # Validate a baseline before mutating any state. Reusing it here does
        # not render another batch or determine the actual acceptance.
        accept_candidate(arm, baseline, baseline)
        proposed, proposal = transaction.propose(baseline['g_rgb'], baseline['g_semantic'],
                                                  'semantic' if arm == 'trust' else arm)
        del proposed
        accepted = False
        try:
            candidate = evaluate_batch(names, gradient=False)
            accepted, acceptance = accept_candidate(arm, baseline, candidate)
        finally:
            transaction.resolve(accepted)
        accepted_steps += int(accepted)
        row = {'attempt': index+1, 'round': batch['round'], 'names': names, 'accepted': accepted,
               'accepted_optimizer_steps': accepted_steps, 'proposal': proposal, 'acceptance': acceptance,
               'baseline': {k: v for k, v in baseline.items() if k not in ('g_rgb', 'g_semantic')},
               'candidate': candidate}
        emit(row); history.append(row)
    return {'attempts': len(history), 'accepted': accepted_steps, 'rejected': len(history)-accepted_steps,
            'history': history, 'stop_reason': 'fixed_148_attempt_schedule_finished', 'converged_claimed': False}


def validate_evidence(preflight, chain, digests):
    """A passed audit verifies records; it never makes the old FD gate pass."""
    require(all(digests.get(k) == v for k, v in EVIDENCE_SHA.items()), 'Wrong fixed diagnostic evidence SHA')
    for name, bundle in (('preflight', preflight), ('chain', chain)):
        receipt, launch, review = (bundle[k] for k in ('execution', 'launch', 'review'))
        require(receipt['status'] == launch['status'] == 'completed'
                and launch['natural_completion'] is True and launch['exit_code'] == 0, 'Incomplete diagnostic execution')
        require(receipt['plan_sha256'] == launch['plan_sha256'] == review['plan_sha256'] == digests[name+'_plan']
                and launch['execution_receipt_sha256'] == digests[name+'_execution'], 'Diagnostic receipt SHA chain')
        require(review['status'] == 'passed' and receipt['sources_inputs_unchanged']
                and receipt['restoration']['state_exact'] and receipt['restoration']['flags_modes_gradients_restored']
                and receipt['numerics_restored'], 'Diagnostic audit or restoration failed')
    old = preflight['execution']; audited_old = preflight['review']; localized = chain['execution']
    require(old['numerical_status'] == audited_old['numerical_status'] == 'not_passed'
            and old['main_count'] == old['main_measurable'] == 12
            and old['main_passed'] == audited_old['main_passed'] == 3,
            'The original 3/12 numerical failure must remain a failure')
    require(old['counts'] == {'raster': 56, 'means_vjp': 4, 'target_decodes': 6}
            and localized['diagnostic_status'] == 'completed_localization_only'
            and localized['counts'] == {'native_rasters': 3, 'replay_rasters': 3,
                                       'backward_sweeps': 1, 'target_decodes': 2}, 'Unexpected diagnostic scope')
    require(localized['replay']['endpoint_replay_allowed']
            and localized['replay']['baseline_four_channel_and_alpha_exact'], 'Chain replay was not established')
    require(chain['review']['execution_receipt_sha256'] == digests['chain_execution']
            and chain['review']['before_after_bound_hashes_exact'] is True
            and chain['review']['original_numerical_gate'] == {'status': 'not_passed', 'passed': 3, 'total': 12},
            'Chain review must retain the original numerical failure')
    for key in ('base_checkpoint', 'manifest', 'q_delta', 'numerics', 'expected_gsplat_binary'):
        require(preflight['plan'][key] == chain['plan'][key], 'Diagnostic geometry/source identity differs: '+key)
    return {'old_preflight_numerical_status': 'not_passed', 'old_fd_gate_passed': False,
            'old_fd_passed': 3, 'old_fd_count': 12, 'independent_record_audits_passed': True,
            'chain_status': 'completed_localization_only', 'derivative_certified': False,
            'interpretation': SPEC['gradient_interpretation']}


def prepare(output, preflight, chain):
    output, preflight, chain = (Path(p).resolve() for p in (output, preflight, chain))
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    bundles = {}; evidence_paths = {}; digests = {}
    for name, directory in (('preflight', preflight), ('chain', chain)):
        bundles[name] = {}
        for role, filename in (('plan', 'plan.json'), ('execution', 'execution_receipt.json'),
                               ('launch', 'launch_receipt.json'), ('review', 'independent_cpu_review.json')):
            path = directory/filename
            bundles[name][role] = read(path); evidence_paths[name+'_'+role] = str(path)
            digests[name+'_'+role] = sha(path)
    interpretation = validate_evidence(bundles['preflight'], bundles['chain'], digests)
    prior = bundles['preflight']['plan']
    require(Path(bundles['chain']['plan']['prior_execution']).resolve() == preflight/'execution_receipt.json', 'Chain prior differs')
    source = Path(prior['source_snapshot']); require(files(source) == prior['source_hashes'], 'Preflight source changed')
    require(len(prior['source_hashes']) == 72, 'Expected actual 72-source preflight package')
    chain_plan = bundles['chain']['plan']
    require(files(chain_plan['source_snapshot']) == chain_plan['source_hashes'], 'Chain source changed')
    manifest = read(prior['manifest']); views = sorted((v for v in manifest['views'] if v['split'] == 'train'
                                                     and v.get('mask_path')), key=lambda v: v['name'])
    names = [v['name'] for v in views]; schedule = batch_schedule(names)
    expected = {**read(Path(prior['q_delta']).parent/'plan.json')['input_hashes'],
                **read(RGB_SOURCE)['input_hashes'], **prior['input_hashes']}
    inputs = dict(prior['input_hashes'])
    for path in [Path(p) for p in evidence_paths.values()]+[RGB_SOURCE]:
        inputs[str(path)] = sha(path)
    for view in views:
        for key in ('image_path', 'mask_path', 'valid_path'):
            require(view[key] in expected, 'Missing audited TRAIN bytes: '+view[key])
            inputs[view[key]] = expected[view[key]]
    for path, digest in inputs.items():
        require(sha(path) == digest, 'Bound input changed: '+path)
    snapshot = output/'source_snapshot'; shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    require(not (snapshot/'bridge_rgs/continuous_geometry.py').exists(), 'Unexpected inherited helper')
    shutil.copy2(ROOT/'src/bridge_rgs/continuous_geometry.py', snapshot/'bridge_rgs/continuous_geometry.py')
    for path in (Path(__file__), ROOT/'scripts/train_continuous_semantic_geometry.py',
                 ROOT/'tests/test_continuous_geometry_proposals.py', ROOT/'tests/test_continuous_geometry_train.py',
                 ROOT/'tests/test_continuous_geometry.py', ROOT/'docs/continuous_geometry_proposals_protocol.md'):
        shutil.copy2(path, snapshot/path.name)
    plan = {k: prior[k] for k in ('base_checkpoint', 'manifest', 'all_training_camera_names', 'q_delta',
            'expected_gsplat_binary', 'installed_sources', 'runtime_versions', 'environment', 'numerics', 'class_weights')}
    plan.update(specification=SPEC, output=str(output), source_snapshot=str(snapshot), source_hashes=files(snapshot),
                inherited_source_hashes=prior['source_hashes'], input_hashes=inputs, training_views=views,
                batches=schedule, preflight=str(preflight), preflight_plan_sha256=sha(preflight/'plan.json'),
                evidence_paths=evidence_paths, evidence_sha256=digests, interpretation=interpretation,
                status='prepared_pending_root_launch', prepare_pixel_decodes=0, prepare_checkpoint_loads=0)
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC and files(plan['source_snapshot']) == plan['source_hashes'], 'Frozen source/spec changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Frozen worker only')
    require(plan['batches'] == batch_schedule([v['name'] for v in plan['training_views']]), 'Changed batch order')
    for path, digest in {**plan['input_hashes'], **plan['installed_sources']}.items():
        require(sha(path) == digest, 'Bound bytes changed: '+path)
    for key, value in plan['runtime_versions'].items():
        require(importlib.metadata.version(key) == value, 'Runtime changed')
    for key, value in plan['environment'].items():
        require(os.environ.get(key) == value, 'Environment changed')


def perform(plan, report, deadline):
    import cv2
    import torch
    from gsplat import rasterization

    require(not any(n.startswith('bridge_rgs') for n in sys.modules), 'No live bridge package')
    standard = importlib.import_module('diagnose_continuous_semantic_geometry')
    adapter = importlib.import_module('preflight_simplex_scene')
    from bridge_rgs.continuous_geometry import MeansTransaction
    from bridge_rgs.semantic_assignment import affine_raw_ce
    from bridge_rgs.train import load_scene
    torch.set_num_threads(4)
    counts = report['counts']; output = Path(plan['output']); by_name = {v['name']: v for v in plan['training_views']}
    allowed = {v[k] for v in plan['training_views'] for k in ('image_path', 'mask_path', 'valid_path')}
    previous_flags = adapter.numerical_flags(plan['numerics']); previous_imread = cv2.imread
    scene = captured = original = None; targets = {}

    def guarded_read(path, *args, **kwargs):
        require(str(path) in allowed, 'Only fixed labeled TRAIN targets may be decoded')
        return previous_imread(path, *args, **kwargs)
    cv2.imread = guarded_read

    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError('Fixed training deadline, no retry or truncated schedule')
        require(counts['raster'] <= SPEC['expected_counts']['raster']
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
                rgb, raw = standard.standard_pair(scene, means, q, view, rasterization, counts)
                rgb_loss = (rgb.clamp(0, 1).double()[valid]-truth[valid]).square().mean()
                semantic_loss = affine_raw_ce(raw, labels, valid, weights, SPEC['affine_noise'])
                metadata = {'rgb_pixels': int(valid.sum()), 'semantic_pixels': int((valid & (labels < 5)).sum())}
                if descriptive:
                    known = valid & (labels < 5)
                    ids = (5*labels[known]+raw.detach().argmax(-1)[known]).cpu().numpy()
                    metadata['confusion_matrix'] = np.bincount(ids, minlength=25).reshape(5, 5).tolist()
                return rgb_loss, semantic_loss, metadata
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
            def emit(row, directory=directory):
                counts['batch_attempts'] += 1
                with (directory/'attempts.jsonl').open('a') as handle:
                    handle.write(json.dumps(row, allow_nan=False)+'\n')
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
            if (name.startswith('bridge_rgs') or name in ('diagnose_continuous_semantic_geometry', 'preflight_simplex_scene')) and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve()
                require(path.is_relative_to(plan['source_snapshot']) and plan['source_hashes'].get(str(path.relative_to(plan['source_snapshot']))) == sha(path), 'Nonfrozen actual import')
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}
        if counts['raster']:
            from gsplat.cuda._backend import _C
            report['actual_gsplat_binary'] = {'path': str(Path(_C.__file__).resolve()), 'sha256': sha(_C.__file__)}
            require(report['actual_gsplat_binary'] == plan['expected_gsplat_binary'], 'Actual renderer binary differs')


def execute(path, expected):
    require(sha(path) == expected, 'Wrong plan SHA'); plan = read(path); output = Path(plan['output'])
    require(not any((output/n).exists() for n in ('execution_started.json', 'execution_receipt.json', 'baseline_train.json')), 'No rerun/overwrite')
    write(output/'execution_started.json', {'plan_sha256': expected, 'pid': os.getpid(), 'time': time.time()})
    report = {'status': 'running', 'plan_sha256': expected,
              'counts': dict.fromkeys([*SPEC['expected_counts'], 'target_decodes'], 0),
              'head_calls': 0, 'teacher_calls': 0, 'val_views': 0, 'production_checkpoint_writes': 0,
              'interpretation': plan['interpretation'], 'adoption': 'none'}
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
    parser = argparse.ArgumentParser(description=__doc__); action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path); action.add_argument('--execute', type=Path)
    parser.add_argument('--preflight', type=Path); parser.add_argument('--chain', type=Path)
    parser.add_argument('--expected-plan-sha256'); args = parser.parse_args()
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode generation')
    if args.prepare:
        require(args.preflight is not None and args.chain is not None, 'Explicit failed preflight and audited chain required')
        prepare(args.prepare, args.preflight, args.chain)
    else:
        require(args.expected_plan_sha256 is not None, 'Expected plan SHA required'); execute(args.execute, args.expected_plan_sha256)
