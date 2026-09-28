"""Two fixed TRAIN views: unchanged forward, complete frozen-head means VJP."""
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
PARENT = Path('/mnt/data/SHM2026/runs/continuous_geometry_proposals_v1')
SPEC = {'protocol': 'readout_geometry_probe_v1', 'names': ['002.png', '041.png'],
        'affine_noise': 5e-7, 'internal_seconds': 120, 'external_seconds': 180,
        'expected_counts': {'old_calls': 2, 'new_calls': 4, 'head_calls': 6,
                            'standard_gsplat_calls': 12, 'custom_shader_calls': 2,
                            'means_vjp': 6, 'target_decodes': 6, 'proposals': 2, 'rollbacks': 2},
        'training_commits': 0, 'teacher_calls': 0, 'VAL': 0,
        'scope': 'wiring and actual single-step TRAIN description; approximate VJP, no adoption gate'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(payload)


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    parent = read(PARENT/'plan.json')
    require(sha(PARENT/'plan.json') == '61c66187e63325fa0d23e8ea9b8522165f9335b93424f3daa54b612e4553b588', 'Parent identity')
    require(read(PARENT/'independent_cpu_review.json')['status'] == 'passed', 'Parent audit')
    require(tree(Path(parent['source_snapshot'])) == parent['source_hashes'], 'Parent sources')
    views = [v for v in parent['training_views'] if v['name'] in SPEC['names']]
    require([v['name'] for v in views] == SPEC['names'], 'Fixed TRAIN views')
    inputs = {str(PARENT/name): sha(PARENT/name) for name in
              ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json')}
    for path in [parent['base_checkpoint'], parent['manifest'], parent['q_delta'], str(ROOT/'uv.lock')]:
        inputs[path] = parent['input_hashes'][path]
    for view in views:
        for key in ('image_path', 'mask_path', 'valid_path'):
            inputs[view[key]] = parent['input_hashes'][view[key]]
    require(all(sha(p) == h for p, h in inputs.items()), 'Inputs changed')
    require(sha(ROOT/'src/bridge_rgs/direct_q_render.py') ==
            'e4519b9ff02466dd11835cc775fdbf428c571c7b63fcb2a01b88193d188db13f',
            'Historical direct-q comparator changed')
    snapshot = output/'source_snapshot'
    shutil.copytree(parent['source_snapshot'], snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    overrides = {}
    for name in ('refinement', 'depth_moments', 'direct_q_geometry_render', 'direct_q_render'):
        target = snapshot/'bridge_rgs'/f'{name}.py'
        overrides[str(target.relative_to(snapshot))] = {'old': sha(target) if target.exists() else None,
                                                       'new': sha(ROOT/'src/bridge_rgs'/f'{name}.py')}
        shutil.copy2(ROOT/'src/bridge_rgs'/f'{name}.py', target)
    for path in (Path(__file__), ROOT/'docs/readout_geometry_probe_protocol.md',
                 ROOT/'tests/test_direct_q_geometry_render.py'):
        shutil.copy2(path, snapshot/path.name)
    plan = {k: parent[k] for k in ('base_checkpoint', 'manifest', 'q_delta', 'expected_gsplat_binary',
            'installed_sources', 'runtime_versions', 'environment', 'numerics', 'class_weights')}
    plan.update(specification=SPEC, output=str(output), source_snapshot=str(snapshot),
                source_hashes=tree(snapshot), overrides=overrides, views=views, input_hashes=inputs,
                prepare_pixel_decodes=0, prepare_checkpoint_loads=0)
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC and tree(Path(plan['source_snapshot'])) == plan['source_hashes'], 'Frozen source/spec')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Frozen entry only')
    binary = plan['expected_gsplat_binary']
    for path, digest in {**plan['input_hashes'], **plan['installed_sources'], binary['path']: binary['sha256']}.items():
        require(sha(path) == digest, 'Input/runtime bytes changed: '+path)
    for key, value in plan['environment'].items():
        require(os.environ.get(key) == value, 'Environment changed: '+key)
    for key, value in plan['runtime_versions'].items():
        require(importlib.metadata.version(key) == value, 'Runtime changed: '+key)


def run(plan, report):
    import cv2
    import gsplat
    import torch
    from gsplat import rendering
    require(not any(n.startswith('bridge_rgs') for n in sys.modules), 'No live bridge package')
    adapter = importlib.import_module('preflight_simplex_scene')
    from bridge_rgs import partition_rasterizer
    from bridge_rgs.continuous_geometry import MeansTransaction
    from bridge_rgs.direct_q_geometry_render import render_direct_q_geometry
    from bridge_rgs.direct_q_render import render_direct_q
    from bridge_rgs.semantic_assignment import affine_raw_ce
    from bridge_rgs.train import load_scene
    torch.set_num_threads(4)
    counts = report['counts']; output = Path(plan['output'])
    previous_flags = adapter.numerical_flags(plan['numerics'])
    hooks = []; scene = captured = tx = None
    old_imread = cv2.imread
    allowed = {v[k] for v in plan['views'] for k in ('image_path', 'mask_path', 'valid_path')}

    def image_read(path, *args, **kwargs):
        require(str(path) in allowed, 'Only two TRAIN target grids')
        counts['target_decodes'] += 1
        return old_imread(path, *args, **kwargs)

    def count_call(obj, name, key):
        old = getattr(obj, name); hooks.append((obj, name, old))
        def wrapped(*args, **kwargs):
            counts[key] += 1
            return old(*args, **kwargs)
        setattr(obj, name, wrapped)

    cv2.imread = image_read
    try:
        report['numerics_actual'] = adapter.numerical_flags()
        torch.cuda.reset_peak_memory_stats()
        scene, _ = load_scene(plan['base_checkpoint']); captured = adapter.capture_scene(scene)
        scene.eval().requires_grad_(False); means = scene.splats['means']; means.requires_grad_(True)
        base = means.detach().clone()
        with np.load(plan['q_delta'], allow_pickle=False) as data:
            q = torch.from_numpy(data['q_renderer'].copy()).cuda()
            require(np.array_equal(data['q_renderer'], data['q_master'].astype(np.float32)), 'q cast identity')
        q_hash = adapter.tensor_hash(q)
        weights = torch.tensor(plan['class_weights'], dtype=torch.float32, device='cuda')
        count_call(gsplat, 'rasterization', 'standard_gsplat_calls')
        count_call(rendering, 'rasterize_to_pixels', 'low_level_raster_calls')
        count_call(partition_rasterizer, 'rasterize_semantic_partition', 'custom_shader_calls')
        count_call(scene.refiner, 'forward', 'head_calls')
        report['views'] = []
        for view in plan['views']:
            K = torch.tensor(view['K'], device='cuda', dtype=torch.float32)
            w2c = torch.tensor(view['w2c_original'], device='cuda', dtype=torch.float32)
            rgb = cv2.imread(view['image_path'], cv2.IMREAD_COLOR)
            labels_np = cv2.imread(view['mask_path'], cv2.IMREAD_UNCHANGED)
            valid_np = cv2.imread(view['valid_path'], cv2.IMREAD_UNCHANGED)
            require(rgb is not None and labels_np is not None and valid_np is not None, 'Missing fixed targets')
            target = torch.from_numpy(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)).cuda().double()/255
            labels = torch.from_numpy(labels_np.astype(np.int64)).cuda(); valid = torch.from_numpy(valid_np > 0).cuda()
            def losses(result, valid=valid, target=target, labels=labels):
                return ((result['rgb'].clamp(0, 1).double()[valid]-target[valid]).square().mean(),
                        affine_raw_ce(result['raw'], labels, valid, weights, SPEC['affine_noise']),
                        affine_raw_ce(result['probabilities'], labels, valid, weights, SPEC['affine_noise']))
            def render_new(K=K, w2c=w2c, view=view):
                counts['new_calls'] += 1
                return render_direct_q_geometry(scene, q, K, w2c, view['width'], view['height'], geometry_grad=True)
            with torch.no_grad():
                old = render_direct_q(scene, q, K, w2c, view['width'], view['height'])
            counts['old_calls'] += 1
            fresh = render_new(); terms = losses(fresh)
            arrays = {'base_means': base.cpu().numpy(), 'target_rgb': target.cpu().numpy(),
                      'labels': labels_np, 'valid': valid_np > 0}
            row = {'name': view['name'], 'old_losses': [float(v) for v in losses(old)],
                   'baseline_losses': [float(v.detach()) for v in terms], 'forward_comparison': {}}
            for key in ('rgb', 'raw', 'probabilities', 'depth', 'alpha', 'features', 'depth_moments'):
                a, b = old[key].detach(), fresh[key].detach()
                require(torch.isfinite(a).all() and torch.isfinite(b).all(), 'Nonfinite forward')
                row['forward_comparison'][key] = {'exact': torch.equal(a, b), 'max_abs': float((a-b).abs().max())}
                if key in ('rgb', 'raw', 'probabilities'):
                    arrays['old_'+key] = a.cpu().numpy(); arrays['baseline_'+key] = b.cpu().numpy()
            row['forward_mask_differences'] = {k: int((old[k].argmax(-1) != fresh[k].argmax(-1)).sum())
                                               for k in ('raw', 'probabilities')}
            gradients = []
            for index, (name, loss) in enumerate(zip(('rgb', 'raw', 'scene'), terms, strict=True)):
                g, = torch.autograd.grad(loss, means, retain_graph=index < 2)
                counts['means_vjp'] += 1
                require(g.dtype == torch.float32 and torch.isfinite(g).all() and torch.count_nonzero(g) > 0, 'Broken/nonfinite means VJP')
                gradients.append(g.detach().double()); arrays['gradient_'+name] = g.detach().cpu().numpy()
            del terms, fresh, old
            tx = MeansTransaction(means, scene.splats['quats'], scene.splats['log_scales'], scene.scene_scale)
            _, proposal = tx.propose(gradients[0], gradients[2], 'semantic'); counts['proposals'] += 1
            with torch.no_grad():
                candidate = render_new(); candidate_losses = [float(v) for v in losses(candidate)]
            delta = means.detach().double()-base.double()
            row.update(proposal=proposal, candidate_losses=candidate_losses,
                       gradient_actual_dots=[float((g*delta).sum()) for g in gradients],
                       actual_loss_changes=[a-b for a, b in zip(candidate_losses, row['baseline_losses'], strict=True)])
            arrays['candidate_means'] = means.detach().cpu().numpy().copy()
            for key in ('rgb', 'raw', 'probabilities'):
                arrays['candidate_'+key] = candidate[key].detach().cpu().numpy()
            del candidate, gradients
            tx.resolve(False); counts['rollbacks'] += 1
            require(torch.equal(means.detach(), base) and not tx.optimizer.state, 'Fresh transaction rollback')
            tx = None
            require(all(p.grad is None for p in scene.parameters()) and adapter.tensor_hash(q) == q_hash, 'Frozen gradients/q changed')
            path = output/(Path(view['name']).stem+'_evidence.npz'); np.savez(path, **arrays)
            row['evidence'] = {'path': str(path), 'sha256': sha(path)}; report['views'].append(row)
        require(all(counts[k] == v for k, v in SPEC['expected_counts'].items()), 'Fixed compute count')
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    finally:
        if tx is not None and tx.pending is not None:
            tx.resolve(False)
        for obj, name, old in reversed(hooks):
            setattr(obj, name, old)
        cv2.imread = old_imread
        try:
            if scene is not None and captured is not None:
                report['restoration'] = adapter.restore_scene(scene, captured)
                require(report['restoration']['state_exact'] and report['restoration']['flags_modes_gradients_restored'], 'Scene not restored')
        finally:
            adapter.numerical_flags(previous_flags)
            report['numerics_restored'] = adapter.numerical_flags() == previous_flags
        report['actual_imports'] = {}
        for name, module in list(sys.modules.items()):
            if name.startswith('bridge_rgs') and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve(); require(path.is_relative_to(plan['source_snapshot']), 'Nonfrozen package')
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}


def execute(path, expected):
    require(sha(path) == expected, 'Wrong plan SHA'); plan = read(path); output = Path(plan['output'])
    write(output/'execution_started.json', {'pid': os.getpid(), 'plan_sha256': expected})
    report = {'status': 'running', 'plan_sha256': expected,
              'counts': dict.fromkeys([*SPEC['expected_counts'], 'low_level_raster_calls'], 0)}
    started = time.monotonic()
    def expired(*_):
        raise TimeoutError('Fixed probe budget')
    previous = signal.signal(signal.SIGALRM, expired); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan); sys.path.insert(0, plan['source_snapshot'])
        old = importlib.import_module('optimize_raw_simplex'); report['gpu_before'] = old.gpu_inventory()
        run(plan, report); verify(plan)
        report.update(status='completed', sources_inputs_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}'); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        report['elapsed_seconds'] = time.monotonic()-started; write(output/'execution_receipt.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path); action.add_argument('--execute', type=Path)
    parser.add_argument('--expected-plan-sha256'); args = parser.parse_args()
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode')
    if args.prepare:
        prepare(args.prepare)
    else:
        execute(args.execute, args.expected_plan_sha256)
