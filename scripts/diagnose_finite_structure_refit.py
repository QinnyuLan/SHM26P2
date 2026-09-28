"""Fixed TRAIN A/B finite structure experiment. Prepare only until root launch."""
from __future__ import annotations

import argparse
import copy
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
PARENT = Path('/mnt/data/SHM2026/runs/raw_simplex_em_fw_v1')
RGB_SOURCE = Path('/mnt/data/SHM2026/runs/matched_rgb_teacher_adaptation_v1/plan.json')
COVERAGE = Path('/mnt/data/SHM2026/runs/finite_structure_coverage_v2')
COVERAGE_HASHES = {
    'plan.json': '78326e41191a48f764436f3435f67f28b9c91cd33b741fdcbfbf51b27f80311a',
    'execution_receipt.json': 'db59f2f6cf5ba78252622c0a2652fe7e473cb8da185bd40ff9276a9a8a22833b',
    'launch_receipt.json': '3bd90748f10ea80e731a26436ad797a597fec78611c794889768d404fa88dddf',
    'library.json': '06599d87236a693138f0bd72f61956f21446149ee853e5049903a808c9be7b33',
}
PARENT_HASHES = {
    'plan.json': '29b066604846598a679a7fb0f1d25d88fce452cc25019adb3b552ad5a760fdcc',
    'execution_receipt.json': 'b129e790bb514dd1180e41978264af8b8ed6ee2ce9f297f24e2f17e749b9e7f0',
    'launch_receipt.json': '049643d84509a633b9cf42344100d56e779fead4596b070ae05725f072181422',
    'independent_cpu_review.json': '9c2dcfc8daf108545f06c4c9fda5ccd93ddca007af667f7752b37f01fe67f743',
    'final_q_delta.npz': 'fd735da6ee25b5e5ee4343f6dc48331cbedc7d717b75c9e86fa66285f64a63d5',
}
A_NAMES = [f'{i:03d}.png' for i in (2, 41, 79, 118, 156, 200, 241, 278)]
B_NAMES = [f'{i:03d}.png' for i in (21, 59, 100, 137, 176, 220, 259, 300)]
SPEC = {
    'protocol': 'finite_structure_refit_v1', 'A': A_NAMES, 'B': B_NAMES,
    'point_count': 498136, 'parents': 256, 'actions': 4, 'K': 64, 'editable_rows': 2048,
    'opacity_range': [.005, .95], 'screen_std_range': [1., 32.],
    'minimum_mixed_A_views': 2, 'minimum_frustum_B_views': 2,
    'split_scale_shrink': 1.6, 'split_mahalanobis_offset': .5,
    'affine_noise': 5e-7, 'em_blocks': 2, 'em_slots_per_block': 10,
    'passes_per_state': 26, 'sh_rates': [.0025, .000125], 'sh_betas': [.9, .999], 'sh_eps': 1e-15,
    'absolute_tolerance': 2e-7, 'relative_tolerance': 5e-4, 'signal_multiplier': 10.,
    'bisection_iterations': 64, 'internal_seconds': 600, 'external_seconds': 660,
    'max_scene': 1200, 'max_raster': 3600, 'max_q_vjp': 960, 'max_rgb_vjp': 840,
    'semantics': False, 'head_calls': 0, 'teacher_calls': 0, 'val_views': 0,
    'fw_non_decrease': 'reject q; preserve its gradient/cache; continue fixed slots',
    'partial_or_numerical_failure': 'stop; no remaining slots or B',
    'B_access': 'only metadata/expected inherited SHA until immutable A choices; shared camera valid excepted',
    'claim': 'finite matching attempts, neither conditional optimum nor performance adoption',
}
SH_KEYS = ('sh0', 'sh_rest')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as handle:
        handle.write(payload)


def files(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def q_sha(q):
    return array_sha(q.astype(np.float32))


def sh_sha(sh):
    return {k: array_sha(v) for k, v in zip(SH_KEYS, sh, strict=True)}


class SHTransaction:
    """Only local SH has Adam state. A rejected proposal rolls back state and value."""
    def __init__(self, values, *, device='cpu'):
        import torch
        self.params = [torch.nn.Parameter(torch.as_tensor(v, device=device).clone()) for v in values]
        self.optimizer = torch.optim.Adam([
            {'params': [p], 'lr': lr} for p, lr in zip(self.params, SPEC['sh_rates'], strict=True)],
            betas=tuple(SPEC['sh_betas']), eps=SPEC['sh_eps'], weight_decay=0.)
        self.pending = None

    def values(self):
        return tuple(p.detach().cpu().numpy().copy() for p in self.params)

    def propose(self, gradient):
        import torch
        require(self.pending is None, 'Unresolved SH proposal')
        self.pending = (self.values(), copy.deepcopy(self.optimizer.state_dict()))
        for p, g in zip(self.params, gradient, strict=True):
            require(g.shape == tuple(p.shape) and g.dtype == np.float64 and np.isfinite(g).all(), 'Bad SH gradient')
            p.grad = torch.as_tensor(g, dtype=p.dtype, device=p.device).clone()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        result = self.values()
        require(all(np.isfinite(v).all() for v in result), 'Nonfinite SH proposal')
        return result

    def resolve(self, accepted):
        import torch
        require(self.pending is not None, 'No pending SH proposal')
        values, state = self.pending
        if not accepted:
            with torch.no_grad():
                for p, v in zip(self.params, values, strict=True):
                    p.copy_(torch.as_tensor(v, device=p.device))
            self.optimizer.load_state_dict(state)
        self.optimizer.zero_grad(set_to_none=True)
        self.pending = None


def solve_refit(q0, sh0, full_pass, fw_proposal, *, sh_optimizer=None, release=lambda r: None):
    """26 complete passes, independently accepted q and SH transactions.

    Cached raw/g belongs only to q; RGB/g belongs only to SH, since geometry
    stays fixed inside one state. Rejected-component trial gradients must never
    replace the corresponding accepted gradient. A partial pass raises before
    either component is committed. No convergence test changes this schedule.
    """
    from bridge_rgs.simplex_em import em_step
    from bridge_rgs.simplex_optimization import frank_wolfe_gap, validate_simplex
    q = validate_simplex(q0).copy()
    sh = tuple(v.copy() for v in sh0)
    optimizer = sh_optimizer or SHTransaction(sh)
    passes, history = [], []
    accepted = {'q_em': 0, 'q_fw': 0, 'sh': 0}

    def run(qv, sv, gq, gs, phase):
        require(len(passes) < SPEC['passes_per_state'], 'Exceeded fixed pass budget')
        row = full_pass(qv, sv, gradient_q=gq, gradient_sh=gs, phase=phase)
        require(row.get('complete') is True and np.isfinite([row['q_ce'], row['rgb_mse']]).all(), 'Incomplete/nonfinite pass')
        require(row['q32_sha256'] == q_sha(qv) and row['sh_sha256'] == sh_sha(sv), 'Pass parameter identity differs')
        require(row['target_cache']['q32_sha256'] == row['q32_sha256'], 'Target cache/q mismatch')
        if gq:
            g = row['q_gradient']
            require(g.dtype == np.float64 and g.shape == qv.shape and np.isfinite(g).all(), 'Invalid q gradient')
            gap = frank_wolfe_gap(qv, g)
            row['approximate_gap'] = float(gap.total_gap)
        if gs:
            require(len(row['sh_gradient']) == 2 and all(g.dtype == np.float64 and g.shape == v.shape
                    and np.isfinite(g).all() for g, v in zip(row['sh_gradient'], sv, strict=True)), 'Invalid SH gradient')
        passes.append({k: v for k, v in row.items() if k not in ('q_gradient', 'sh_gradient', 'target_cache')})
        return row

    current_q = current_sh = run(q, sh, True, True, 'baseline')
    for block in range(SPEC['em_blocks']):
        for slot in range(SPEC['em_slots_per_block']):
            proposal = em_step(q, current_q['q_gradient']).proposal
            proposed_sh = optimizer.propose(current_sh['sh_gradient'])
            try:
                trial = run(proposal, proposed_sh, True, True, f'em_{block}_{slot}')
            except BaseException:
                optimizer.resolve(False)
                raise
            q_ok = bool(trial['q_ce'] < current_q['q_ce'])
            sh_ok = bool(trial['rgb_mse'] < current_sh['rgb_mse'])
            history.append({'kind': 'em_sh', 'block': block, 'slot': slot,
                            'q_base': current_q['q_ce'], 'q_trial': trial['q_ce'], 'q_accepted': q_ok,
                            'rgb_base': current_sh['rgb_mse'], 'rgb_trial': trial['rgb_mse'], 'sh_accepted': sh_ok,
                            'q_stagnant': bool(q_sha(proposal) == q_sha(q)),
                            'sh_stagnant': bool(sh_sha(proposed_sh) == sh_sha(sh))})
            optimizer.resolve(sh_ok)
            if sh_ok:
                sh, current_sh = proposed_sh, trial
                accepted['sh'] += 1
            if q_ok:
                release(current_q)
                q, current_q = proposal, trial
                accepted['q_em'] += 1
            else:
                release(trial)
        require(current_q['q32_sha256'] == q_sha(q), 'q and retained gradient differ')
        vertex = frank_wolfe_gap(q, current_q['q_gradient']).vertex
        A = float(np.sum(current_q['q_gradient']*(vertex.astype(np.float32).astype(np.float64)
                                                  - q.astype(np.float32).astype(np.float64))))
        no_direction = A >= -SPEC['signal_multiplier']*SPEC['absolute_tolerance']
        vertex_value = q.copy() if no_direction else vertex
        endpoint = run(vertex_value, sh, False, False, f'fw_vertex_{block}')
        try:
            if no_direction:
                proposed_q, metadata = q.copy(), {'passed': False, 'reason': 'no_measurable_negative_LMO', 'A': A}
            else:
                proposed_q, metadata = fw_proposal(q, current_q, vertex, endpoint)
                require(metadata['passed'], 'Measurable FW direction failed numerical alignment')
        finally:
            release(endpoint)
        trial = run(proposed_q, sh, True, False, f'fw_candidate_{block}')
        q_ok = False
        confirmation = None
        if metadata['passed']:
            predicted = current_q['q_ce']-metadata['cached_objective']
            actual = current_q['q_ce']-trial['q_ce']
            tol = SPEC['absolute_tolerance']+SPEC['relative_tolerance']*max(abs(predicted), abs(actual))
            require(abs(predicted-actual) <= tol, 'FW actual/cache confirmation alignment failed')
            q_ok = bool(actual > 0 and predicted > 0)
            confirmation = {'predicted_decrease': float(predicted), 'actual_decrease': float(actual), 'tolerance': float(tol)}
        history.append({'kind': 'fw', 'block': block, 'q_accepted': q_ok,
                        'line': metadata, 'confirmation': confirmation,
                        'reason': 'accepted' if q_ok else 'finite_noop_or_nondecrease'})
        if q_ok:
            release(current_q)
            q, current_q = proposed_q, trial
            accepted['q_fw'] += 1
        else:
            release(trial)
        # SH gradient remains evaluated at its accepted SH, never overwritten by FW.
    final = run(q, sh, True, False, 'final')
    require(len(passes) == SPEC['passes_per_state'], 'Different state fitting budgets')
    release(current_q)
    release(final)
    return q, sh, {'complete_passes': len(passes), 'accepted': accepted, 'history': history, 'passes': passes,
                   'baseline': passes[0], 'final': passes[-1], 'q_final_repeat': float(final['q_ce']-current_q['q_ce']),
                   'rgb_final_repeat': float(final['rgb_mse']-current_sh['rgb_mse']),
                   'conditional_optimum_claimed': False}


def footprint(info):
    """CPU projection metadata, no scene/label access."""
    means = info['means2d'][0].detach().double().cpu().numpy()
    conic = info['conics'][0].detach().double().cpu().numpy()
    radii = info['radii'][0].detach().cpu().numpy()
    radii = np.max(radii, axis=-1) if radii.ndim == 2 else radii
    matrix = np.zeros((len(means), 2, 2), np.float64)
    matrix[:, 0, 0], matrix[:, 0, 1] = conic[:, 0], conic[:, 1]
    matrix[:, 1, 0], matrix[:, 1, 1] = conic[:, 1], conic[:, 2]
    active = radii > 0
    cov = np.zeros_like(matrix)
    require(np.isfinite(means[active]).all() and np.isfinite(matrix[active]).all(), 'Nonfinite active footprint')
    cov[active] = np.linalg.inv(matrix[active])
    require(np.all(np.linalg.eigvalsh(cov[active]) > 0), 'Nonpositive active footprint')
    return {'means': means, 'conics': matrix, 'covariance': cov, 'active': active}


def probe_mixed(fp, labels, valid):
    height, width = labels.shape
    means, cov = fp['means'], fp['covariance']
    values, vectors = np.linalg.eigh(cov)
    axes = vectors*np.sqrt(np.maximum(values, 0))[:, None, :]
    a, b = axes[:, :, 0], axes[:, :, 1]
    probes = means[:, None, :] + np.stack([np.zeros_like(a), a, -a, b, -b, a+b, -a-b, a-b, -a+b], 1)
    # Culled means need not be valid integer coordinates; never cast them blindly.
    safe = np.where(np.isfinite(probes), probes, -1)
    xy = np.floor(safe).astype(np.int64)
    inside = (xy[..., 0] >= 0) & (xy[..., 0] < width) & (xy[..., 1] >= 0) & (xy[..., 1] < height)
    x, y = np.clip(xy[..., 0], 0, width-1), np.clip(xy[..., 1], 0, height-1)
    known = inside & valid[y, x] & (labels[y, x] < 5)
    bg = np.sum(known & (labels[y, x] == 0), axis=1)
    cable = np.sum(known & (labels[y, x] == 2), axis=1)
    centers = np.isfinite(means).all(1) & (means[:, 0] >= 0) & (means[:, 0] < width) & (means[:, 1] >= 0) & (means[:, 1] < height)
    std = np.sqrt(np.maximum(values[:, 1], 0))
    mixed = fp['active'] & centers & (std >= 1) & (std <= 32) & (bg > 0) & (cable > 0)
    area = np.sqrt(np.maximum(np.linalg.det(cov), 0))
    return mixed, np.where(mixed, np.minimum(bg, cable), 0), np.where(fp['active'] & centers, area, 0.)


def frustum_counts(means, views):
    result = np.zeros(len(means), np.int64)
    for view in views:
        pose, K = np.asarray(view['w2c_original'], np.float64), np.asarray(view['K'], np.float64)
        xyz = means @ pose[:3, :3].T+pose[:3, 3]
        z = xyz[:, 2]
        p = xyz @ K.T
        xy = p[:, :2]/np.where(z > 0, z, 1)[:, None]
        result += (z > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < view['width']) & (xy[:, 1] >= 0) & (xy[:, 1] < view['height'])
    return result


def choose_library(means, alpha, projections, targets, B_views):
    from scipy.spatial import cKDTree
    mixed = np.zeros(len(means), np.int64)
    minority = np.zeros_like(mixed)
    area = np.zeros(len(means), np.float64)
    for fp, (labels, valid) in zip(projections, targets, strict=True):
        m, n, a = probe_mixed(fp, labels, valid)
        mixed += m; minority += n; area += a
    B_count = frustum_counts(means, B_views)
    eligible = np.flatnonzero((mixed >= 2) & (B_count >= 2) & (alpha >= .005) & (alpha <= .95))
    ranked = eligible[np.lexsort((eligible, -minority[eligible], -mixed[eligible]))]
    if len(ranked) < SPEC['parents']:
        return {'status': 'inconclusive_coverage', 'eligible_parents': len(ranked)}
    parents = ranked[:SPEC['parents']]
    distance = cKDTree(means[parents]).query(means, workers=1)[0]
    other = np.setdiff1d(np.arange(len(means)), parents)
    near = other[np.lexsort((other, distance[other]))][:SPEC['editable_rows']-len(parents)]
    U = np.sort(np.concatenate((parents, near)))
    donors = near[np.lexsort((near, alpha[near]*area[near]))][:SPEC['K']]
    return {'status': 'ready', 'eligible_parents': len(ranked), 'parents': parents.tolist(),
            'U': U.tolist(), 'donors': donors.tolist(),
            'actions': {f'action{i}': parents[i::4].tolist() for i in range(4)},
            'parent_mixed_views': mixed[parents].tolist(), 'parent_minority_sum': minority[parents].tolist(),
            'parent_B_frustum_views': B_count[parents].tolist(),
            'donor_proxy': (alpha[donors]*area[donors]).tolist(),
            'donor_proxy_scope': 'actual positive-radius projected area, center in image; not alpha contribution'}


def local_support(fp, indices, height, width):
    """Original 3-sigma ellipses at the renderer's pixel centres, never error-selected."""
    support = np.zeros((height, width), bool)
    for i in indices:
        if not fp['active'][i]:
            continue
        mu, C, Q = fp['means'][i], fp['covariance'][i], fp['conics'][i]
        extent = 3*np.sqrt(np.diag(C))
        lo = np.maximum(0, np.ceil(mu-extent-.5).astype(int))
        hi = np.minimum([width-1, height-1], np.floor(mu+extent-.5).astype(int))
        if np.any(lo > hi):
            continue
        y, x = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        delta = np.stack((x+.5-mu[0], y+.5-mu[1]), -1)
        support[lo[1]:hi[1]+1, lo[0]:hi[0]+1] |= np.einsum('...i,ij,...j->...', delta, Q, delta) <= 9
    return support


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Use a new data-disk directory')
    lineage = {str(PARENT/name): digest for name, digest in PARENT_HASHES.items()}
    for path, digest in lineage.items():
        require(sha(path) == digest, 'Completed parent lineage changed: '+path)
    parent, receipt, launch, audit = (read(PARENT/name) for name in
        ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json'))
    require(receipt['status'] == launch['status'] == 'completed' and launch['natural_completion']
            and launch['exit_code'] == 0 and audit['status'] == 'passed'
            and receipt['plan_sha256'] == launch['plan_sha256'] == PARENT_HASHES['plan.json']
            and launch['execution_receipt_sha256'] == PARENT_HASHES['execution_receipt.json'], 'Completed audited parent required')
    coverage_lineage = {str(COVERAGE/k): v for k, v in COVERAGE_HASHES.items()}
    for path, digest in coverage_lineage.items():
        require(sha(path) == digest, 'Coverage lineage changed')
    ce, cl = read(COVERAGE/'execution_receipt.json'), read(COVERAGE/'launch_receipt.json')
    require(ce['status'] == cl['status'] == 'completed' and cl['natural_completion'] and cl['exit_code'] == 0
            and ce['plan_sha256'] == cl['plan_sha256'] == COVERAGE_HASHES['plan.json']
            and cl['execution_receipt_sha256'] == COVERAGE_HASHES['execution_receipt.json']
            and ce['library_sha256'] == COVERAGE_HASHES['library.json'], 'Coverage did not complete naturally')
    sources = Path(parent['source_snapshot'])
    require(files(sources) == parent['source_hashes'], 'Parent source changed')
    manifest = read(parent['manifest'])
    all_train = [v for v in manifest['views'] if v['split'] == 'train']
    by_name = {v['name']: v for v in all_train}
    require(len(all_train) == 350 and len(by_name) == 350, 'Wrong TRAIN camera population')
    selected = [[by_name[name] for name in names] for names in (A_NAMES, B_NAMES)]
    rgb_plan = read(RGB_SOURCE)  # metadata only; never hash deferred B image bytes here
    expected = {**parent['input_hashes'], **rgb_plan['input_hashes']}
    inputs = {**lineage, **coverage_lineage, str(RGB_SOURCE): sha(RGB_SOURCE), parent['base_checkpoint']: sha(parent['base_checkpoint']),
              parent['manifest']: sha(parent['manifest']), str(ROOT/'uv.lock'): sha(ROOT/'uv.lock')}
    # Every image hash comes from completed source metadata, including A. Only
    # the A byte hashes are checked before selection. Shared camera-valid is A.
    deferred = {}
    for side, views in zip(('A', 'B'), selected, strict=True):
        for view in views:
            require(view['mask_path'] and view['valid_path'], 'Expected labeled fixed TRAIN views')
            for key in ('image_path', 'mask_path', 'valid_path'):
                path = view[key]
                require(path in expected, 'Missing inherited prepared pixel SHA: '+path)
                if side == 'A' or key == 'valid_path':
                    inputs[path] = expected[path]
                else:
                    deferred[path] = expected[path]
    require(not (set(inputs) & set(deferred)), 'B-unique inputs leaked into initial hash list')
    for path, digest in inputs.items():
        require(sha(path) == digest, 'Initial source/input changed: '+path)
    snapshot = output/'source_snapshot'
    shutil.copytree(sources, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('finite_structure.py', 'finite_structure_selection.py'):
        require(not (snapshot/'bridge_rgs'/name).exists(), 'Only explicit new helpers may be added')
        shutil.copy2(ROOT/'src/bridge_rgs'/name, snapshot/'bridge_rgs'/name)
    for path in (Path(__file__), ROOT/'tests/test_finite_structure_refit.py',
                 ROOT/'tests/test_finite_structure.py', ROOT/'tests/test_finite_structure_selection.py',
                 ROOT/'docs/finite_structure_refit_diagnostic.md'):
        shutil.copy2(path, snapshot/path.name)
    plan = {key: parent[key] for key in ('base_checkpoint', 'manifest', 'all_training_camera_names',
            'expected_gsplat_binary', 'installed_sources', 'runtime_versions', 'environment', 'numerics', 'class_weights')}
    plan.update(specification=SPEC, output=str(output), source_snapshot=str(snapshot), source_hashes=files(snapshot),
                inherited_source_hashes=parent['source_hashes'], input_hashes=inputs, deferred_B_hashes=deferred,
                parent_lineage=lineage, coverage_lineage=coverage_lineage, coverage_library=str(COVERAGE/'library.json'),
                preparation_cost='shared coverage scan:8 additional A scene/raster calls, no B pixels', q_delta=str(PARENT/'final_q_delta.npz'), A=selected[0], B=selected[1],
                shared_valid_exception=sorted({v['valid_path'] for v in selected[0]} & {v['valid_path'] for v in selected[1]}),
                status='prepared_pending_root_execution', prepare_checkpoint_loads=0, prepare_pixel_decodes=0)
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan, *, after_choice=False):
    require(plan['specification'] == SPEC and files(plan['source_snapshot']) == plan['source_hashes'], 'Frozen contract changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Frozen entry only')
    inputs = {**plan['input_hashes'], **plan['installed_sources']}
    if after_choice:
        inputs.update(plan['deferred_B_hashes'])
    for path, digest in inputs.items():
        require(sha(path) == digest, 'Input/source changed: '+path)
    for key, value in plan['environment'].items():
        require(os.environ.get(key) == value, 'Environment changed: '+key)
    for key, value in plan['runtime_versions'].items():
        require(importlib.metadata.version(key) == value, 'Runtime changed: '+key)
    require(sha(plan['expected_gsplat_binary']['path']) == plan['expected_gsplat_binary']['sha256'], 'Renderer binary changed')


class PixelBarrier:
    """A/shared-valid initially; B-unique bytes and hashes only after write-once choice."""
    def __init__(self, plan):
        self.plan = plan
        self.opened = False
        self.choice = None

    def allow(self, path):
        allowed = {v[k] for v in self.plan['A'] for k in ('image_path', 'mask_path', 'valid_path')}
        if self.opened:
            allowed |= {v[k] for v in self.plan['B'] for k in ('image_path', 'mask_path', 'valid_path')}
        require(str(path) in allowed, 'Pixel read before immutable A selection or outside fixed TRAIN')

    def open_b(self, path, digest):
        require(not self.opened and sha(path) == digest, 'Immutable A choice missing/changed')
        self.choice = (str(path), digest)
        for name, expected in self.plan['deferred_B_hashes'].items():
            require(sha(name) == expected, 'Deferred B input changed: '+name)
        self.opened = True


def perform(plan, report, deadline):
    import cv2
    import torch
    from gsplat import rasterization

    require(not any(n.startswith('bridge_rgs') for n in sys.modules), 'No live bridge package may be imported')
    adapter = importlib.import_module('preflight_simplex_scene')
    direction = importlib.import_module('diagnose_simplex_direction')
    from bridge_rgs.finite_structure import split_exchange
    from bridge_rgs.finite_structure_selection import (
        evaluate_on_b,
        select_on_a,
        validate_fixed_support,
    )
    from bridge_rgs.partition_rasterizer import rasterize_semantic_partition
    from bridge_rgs.semantic_assignment import affine_raw_ce
    from bridge_rgs.simplex_optimization import validate_simplex
    from bridge_rgs.train import load_scene

    require(SPEC['affine_noise'] == direction.SPEC['delta'] and all(SPEC[k] == direction.SPEC[k] for k in
            ('absolute_tolerance', 'relative_tolerance', 'signal_multiplier', 'bisection_iterations')), 'Inherited FW math differs')
    torch.set_num_threads(4)
    previous = adapter.numerical_flags(plan['numerics'])
    report['numerics_actual'] = adapter.numerical_flags()
    require(report['numerics_actual'] == plan['numerics'], 'Numerical flags not applied')
    scene = original = captured = None
    output = Path(plan['output']); temporary = output/'temporary'; temporary.mkdir()
    barrier = PixelBarrier(plan)
    original_imread = cv2.imread
    def guarded_imread(path, *args, **kwargs):
        barrier.allow(path)
        return original_imread(path, *args, **kwargs)
    cv2.imread = guarded_imread
    counts, targets, supports, support_hashes = report['counts'], {}, {}, {}
    weights_np = np.asarray(plan['class_weights'], np.float32).astype(np.float64)
    weights = torch.tensor(plan['class_weights'], device='cuda', dtype=torch.float32)
    report['cache_releases'] = []

    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError('Fixed 600s budget; no partial pass or B rescue')
        require(counts['scene'] <= SPEC['max_scene'] and counts['gsplat']+counts['shader'] <= SPEC['max_raster']
                and counts['q_vjp'] <= SPEC['max_q_vjp'] and counts['rgb_vjp'] <= SPEC['max_rgb_vjp'], 'Fixed call budget exceeded')

    def progress(phase):
        print(json.dumps({'phase': phase, 'elapsed': time.monotonic()-(deadline-SPEC['internal_seconds']),
                          'counts': counts}), flush=True)

    def target(view):
        if view['name'] not in targets:
            y = cv2.imread(view['mask_path'], cv2.IMREAD_UNCHANGED)
            valid = cv2.imread(view['valid_path'], cv2.IMREAD_UNCHANGED)
            rgb = cv2.imread(view['image_path'], cv2.IMREAD_COLOR)
            shape = (view['height'], view['width'])
            require(y is not None and valid is not None and rgb is not None
                    and y.shape == valid.shape == shape and rgb.shape == (*shape, 3), 'Bad prepared TRAIN grid')
            require(np.isin(y, [0, 1, 2, 3, 4, 255]).all(), 'Bad label IDs')
            valid = valid > 0
            require(valid.any() and (valid & (y < 5)).any(), 'Empty TRAIN support')
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            targets[view['name']] = {'labels': y, 'valid': valid, 'known': valid & (y < 5), 'rgb': rgb,
                'rgb_valid_sha256': array_sha(valid), 'rgb_valid_pixels': int(valid.sum())}
            counts['target_decodes'] += 3
        return targets[view['name']]

    def render(view, *, rgb_gradient=False):
        tick()
        K = torch.tensor(view['K'], dtype=torch.float32, device='cuda')
        pose = torch.tensor(view['w2c_original'], dtype=torch.float32, device='cuda')
        with torch.set_grad_enabled(rgb_gradient):
            context = scene.render(K, pose, view['width'], view['height'], semantics=False, refine=False, absgrad=False)
        counts['scene'] += 1; counts['gsplat'] += 1
        return context, K, pose

    def raw(context, q, view):
        result = adapter.direct_q(context['info'], q, rasterize_semantic_partition, width=view['width'], height=view['height'])
        counts['shader'] += 1
        return result

    def measure(view, context, prediction):
        t = target(view)
        label = torch.from_numpy(t['labels'].astype(np.int64)).cuda()
        valid = torch.from_numpy(t['valid']).cuda()
        truth = torch.from_numpy(t['rgb']).cuda().double()/255
        loss = affine_raw_ce(prediction, label, valid, weights, SPEC['affine_noise'])
        rgb_error = (context['rgb'].clamp(0, 1).double()-truth).square()
        rgb_loss = rgb_error[valid].mean()
        support = supports[view['name']]
        require(array_sha(support) == support_hashes[view['name']], 'Fixed local mask changed')
        local = support & t['valid']
        local_gpu = torch.from_numpy(local).cuda()
        predicted = prediction.detach().argmax(-1).cpu().numpy()
        cm = np.bincount((5*t['labels'][t['known']].astype(np.int64)+predicted[t['known']]).ravel(), minlength=25).reshape(5, 5)
        row = {'name': view['name'], 'full_rgb_mse': float(rgb_loss.detach()),
               'local_rgb_mse': float(rgb_error[local_gpu].detach().mean()) if local.any() else None,
               'local_pixels': int(local.sum()), 'local_support_sha256': support_hashes[view['name']],
               'raw_ce': float(loss.detach()), 'confusion_matrix': cm.tolist(),
               'rgb_valid_pixels': t['rgb_valid_pixels'], 'rgb_valid_sha256': t['rgb_valid_sha256']}
        require(np.isfinite([row['raw_ce'], row['full_rgb_mse']]).all(), 'Nonfinite view objective')
        return loss, rgb_loss, row

    def put_support(view, fp, ids):
        require(view['name'] not in supports, 'Support already fixed')
        support = local_support(fp, ids, view['height'], view['width'])
        supports[view['name']] = support
        support_hashes[view['name']] = array_sha(support)
        np.save(output/f"support_{view['name']}.npy", support, allow_pickle=False)

    def release(result):
        item = result['target_cache']; folder = Path(item['folder'])
        require(folder.parent == temporary and not folder.is_symlink(), 'Only own cache may be removed')
        require({str(p) for p in folder.iterdir()} == set(item['hashes']), 'Unexpected temporary files')
        for name, digest in item['hashes'].items():
            require(Path(name).parent == folder and not Path(name).is_symlink() and sha(name) == digest, 'Cache changed')
        report['cache_releases'].append({'folder': str(folder), 'q32_sha256': item['q32_sha256'], 'files': len(item['hashes'])})
        for name in item['hashes']:
            Path(name).unlink()
        folder.rmdir()
        tick()

    def fw(q, current, vertex, endpoint):
        require(current['target_cache']['q32_sha256'] == q_sha(q)
                and endpoint['target_cache']['q32_sha256'] == q_sha(vertex), 'FW cache/q identity mismatch')
        records = []
        for a, v in zip(current['target_cache']['records'], endpoint['target_cache']['records'], strict=True):
            require(a['name'] == v['name'] and a['pixels'] == v['pixels']
                    and current['target_cache']['hashes'][a['labels']] == endpoint['target_cache']['hashes'][v['labels']], 'FW support changed')
            records.append({**a, 'v': v['a']})
        zero = direction.line_summary(records, weights_np, 0., bins=True, deadline=deadline)
        require(abs(zero['F']-current['q_ce']) <= 1e-12, 'Cached objective differs from actual')
        A = float(np.sum(current['q_gradient']*(vertex.astype(np.float32).astype(np.float64)-q.astype(np.float32).astype(np.float64))))
        gate = direction.direction_gate(A, zero['B64'], zero['B32upstream'])
        require(gate['passed'], 'Measurable direction alignment/negative sign failed')
        gamma, search = direction.fixed_line_search(lambda t: direction.line_summary(
            records, weights_np, t, deadline=deadline, with_loss=False, validated=True)['B64'])
        candidate = direction.line_summary(records, weights_np, gamma, deadline=deadline, validated=True)
        return (1-gamma)*q+gamma*vertex, {'passed': True, 'A': A, 'gate': gate, 'zero': zero,
                                      'gamma': float(gamma), 'search': search, 'cached_objective': candidate['F']}

    def attest_runtime():
        report['actual_imports'] = {}
        for name, module in sys.modules.items():
            if (name.startswith('bridge_rgs') or name in ('preflight_simplex_scene', 'diagnose_simplex_direction')) and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve()
                require(path.is_relative_to(plan['source_snapshot'])
                        and plan['source_hashes'][str(path.relative_to(plan['source_snapshot']))] == sha(path), 'Nonfrozen import')
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}
        from gsplat.cuda._backend import _C
        binary = {'path': str(Path(_C.__file__).resolve()), 'sha256': sha(_C.__file__)}
        require(binary == plan['expected_gsplat_binary'], 'Loaded binary differs')
        report['actual_gsplat_binary'] = binary

    try:
        scene, checkpoint = load_scene(plan['base_checkpoint'])
        require(scene.sh_degree == 3 and scene.pixel_protocol == 'legacy_mixed_v1'
                and scene.mip_filter_config is None and len(scene.splats['means']) == SPEC['point_count'], 'Wrong H3 scene')
        train = [v for v in read(plan['manifest'])['views'] if v['split'] == 'train']
        require([v['name'] for v in train] == plan['all_training_camera_names']
                and np.array_equal(checkpoint['training_cameras'].cpu().numpy(),
                                   np.asarray([v['w2c_original'] for v in train], np.float32)), 'TRAIN cameras changed')
        captured = adapter.capture_scene(scene)
        original = {k: v.detach().cpu().clone() for k, v in scene.state_dict().items()}
        scene.eval().requires_grad_(False)
        with np.load(plan['q_delta'], allow_pickle=False) as data:
            qbase = data['q_master'].copy()
            require(np.array_equal(data['q_renderer'], qbase.astype(np.float32)), 'Original q renderer/master differ')
        validate_simplex(qbase)
        require(qbase.shape == (SPEC['point_count'], 5), 'q row order/size differs')
        torch.cuda.reset_peak_memory_stats()
        original_predictions, projections, control_g = {}, [], np.zeros(SPEC['point_count'], np.float64)
        q_control = torch.from_numpy(qbase.astype(np.float32)).cuda()
        # A only, exactly8 scenes. Standard signed projected gradient is a
        # separate semantic rasterization, never a live geometry update.
        for view in plan['A']:
            context, K, pose = render(view)
            fp = footprint(context['info']); projections.append(fp)
            original_predictions[view['name']] = {k: context[k].detach().cpu().numpy() for k in ('rgb', 'alpha')}
            t = target(view)
            means = scene.splats['means'].detach().clone().requires_grad_(True)
            bg = q_control.new_tensor([[1., 0., 0., 0., 0.]])
            standard, _, info = rasterization(means=means, quats=scene.splats['quats'].detach(),
                scales=scene.splats['log_scales'].detach().exp(), opacities=scene.splats['opacity_logits'].detach().sigmoid(),
                colors=q_control, viewmats=pose[None], Ks=K[None], width=view['width'], height=view['height'],
                backgrounds=bg, packed=False, rasterize_mode='antialiased', near_plane=.01, far_plane=1e6, absgrad=False)
            counts['gsplat'] += 1
            projected = info['means2d']; projected.retain_grad()
            with torch.no_grad():
                direct, _ = raw(context, q_control, view)
                difference = float((direct-standard[0]).abs().max())
                require(difference <= 2e-6, 'Standard gradient-control raw differs from validated direct shader')
            loss = affine_raw_ce(standard[0], torch.from_numpy(t['labels'].astype(np.int64)).cuda(),
                                 torch.from_numpy(t['valid']).cuda(), weights, SPEC['affine_noise'])
            gp, = torch.autograd.grad(loss, projected)
            signed = gp[0].detach().double().cpu().numpy()*np.array([view['width']/2, view['height']/2])
            control_g += np.linalg.norm(signed, axis=-1)/len(plan['A'])
            counts['control_vjp'] += 1
            report.setdefault('gradient_control', []).append({'name': view['name'], 'raw_max_difference': difference})
            del context, direct, standard, info, means, loss, gp
            tick()
        means_np = original['splats.means'].double().numpy()
        alpha_np = original['splats.opacity_logits'].sigmoid().double().numpy()
        library = choose_library(means_np, alpha_np, projections,
            [(target(v)['labels'], target(v)['valid']) for v in plan['A']], plan['B'])
        require(library == read(plan['coverage_library']), 'Full-worker library differs from fixed coverage scan; no rescue')
        report['coverage_library_exact'] = True
        write(output/'library.json', library)
        if library['status'] != 'ready':
            report.update(scientific_status='inconclusive_coverage', B_unique_inputs_opened=False)
            return
        local_ids = np.asarray(library['parents']+library['donors'], np.int64)
        for view, fp in zip(plan['A'], projections, strict=True):
            put_support(view, fp, local_ids)
        gradient_scores = {key: float(control_g[ids].sum()) for key, ids in library['actions'].items()}
        report['gradient_scores'] = gradient_scores
        U_original = np.asarray(library['U'], np.int64)
        final_deltas, results, state_metadata = {}, {}, {}

        def install_state(name):
            scene.load_state_dict(original, strict=True)
            scene.requires_grad_(False)
            source = torch.arange(SPEC['point_count'], device='cuda')
            action = None
            if name != 'keep':
                per_point = dict(scene.splats.items())
                per_point.update({k: v for k, v in scene.named_buffers()
                                  if v is not None and v.ndim and len(v) == SPEC['point_count']})
                action = split_exchange(per_point, torch.tensor(library['actions'][name], device='cuda'),
                                       torch.tensor(library['donors'], device='cuda'))
                with torch.no_grad():
                    for k, value in action.tensors.items():
                        destination = scene.splats[k] if k in scene.splats else dict(scene.named_buffers())[k]
                        destination.copy_(value)
                source = action.source_indices
            source_np = source.cpu().numpy()
            editable = np.flatnonzero(np.isin(source_np, U_original))
            require(len(editable) == SPEC['editable_rows'], 'Changed editable-row budget')
            base = qbase[source_np].copy()
            zero_sh = tuple(scene.splats[k].detach()[editable].cpu().numpy().copy() for k in SH_KEYS)
            fixed_hashes = {k: adapter.tensor_hash(v) for k, v in scene.state_dict().items()
                            if k not in ('splats.sh0', 'splats.sh_rest')}
            outside = np.setdiff1d(np.arange(SPEC['point_count']), editable)
            outside_hashes = {k: adapter.tensor_hash(scene.splats[k].detach()[outside]) for k in SH_KEYS}
            if name not in state_metadata:
                meta = {'source_indices_sha256': array_sha(source_np), 'editable_indices_sha256': array_sha(editable),
                        'q_base_sha256': array_sha(base), 'fixed_state_hashes': fixed_hashes,
                        'outside_SH_hashes': outside_hashes, 'point_count': len(base)}
                if action is not None:
                    meta.update(parent_indices=action.parent_indices.cpu().tolist(),
                                donor_indices=action.donor_indices.cpu().tolist(),
                                world_offsets=action.offsets.cpu().tolist(), shrink_axes=action.shrink_axes.cpu().tolist(),
                                children_start=action.retained_count)
                state_metadata[name] = meta
                np.savez(output/f'{name}_mapping.npz', source_indices=source_np, editable_indices=editable)
            else:
                require(state_metadata[name]['fixed_state_hashes'] == fixed_hashes
                        and state_metadata[name]['source_indices_sha256'] == array_sha(source_np), 'State reconstruction changed')
            return base, editable, zero_sh, fixed_hashes, outside_hashes

        def copy_sh(editable, sh):
            with torch.no_grad():
                for key, value in zip(SH_KEYS, sh, strict=True):
                    scene.splats[key][editable] = torch.from_numpy(value).cuda()

        def state_scope(editable, fixed, outside):
            require({k: adapter.tensor_hash(v) for k, v in scene.state_dict().items()
                     if k not in ('splats.sh0', 'splats.sh_rest')} == fixed, 'Frozen state changed within refit')
            other = np.setdiff1d(np.arange(SPEC['point_count']), editable)
            require({k: adapter.tensor_hash(scene.splats[k].detach()[other]) for k in SH_KEYS} == outside,
                    'SH changed outside editable rows')
            require(all(p.grad is None for p in scene.parameters()), 'Unexpected scene accumulated .grad')

        def full_factory(name, base, editable):
            indices = torch.from_numpy(editable).cuda()
            def full(q, sh, *, gradient_q, gradient_sh, phase):
                tick()
                copy_sh(editable, sh)
                for key in SH_KEYS:
                    scene.splats[key].requires_grad_(gradient_sh)
                fullq = base.copy(); fullq[editable] = q
                q32 = torch.from_numpy(fullq.astype(np.float32)).cuda().requires_grad_(gradient_q)
                cache_folder = temporary/f'{name}_{phase}'; cache_folder.mkdir()
                rows, cache_rows = [], []
                gq = np.zeros_like(q) if gradient_q else None
                gsh = [np.zeros(v.shape, np.float64) for v in sh] if gradient_sh else None
                started = time.monotonic()
                try:
                    for view in plan['A']:
                        context, _, _ = render(view, rgb_gradient=gradient_sh)
                        prediction, semantic_alpha = raw(context, q32, view)
                        loss, rgb_loss, row = measure(view, context, prediction)
                        require(float((semantic_alpha-context['alpha'].detach()).abs().max()) <= 2e-6,
                                'RGB and direct-q alpha paths differ')
                        if gradient_q:
                            gradient, = torch.autograd.grad(loss, q32)
                            g = gradient[indices].detach().double().cpu().numpy()
                            require(np.isfinite(g).all(), 'Nonfinite q VJP')
                            gq += g
                            counts['q_vjp'] += 1
                        if gradient_sh:
                            gradients = torch.autograd.grad(rgb_loss, tuple(scene.splats[k] for k in SH_KEYS))
                            for destination, gradient in zip(gsh, gradients, strict=True):
                                value = gradient[indices].detach().double().cpu().numpy()
                                require(np.isfinite(value).all(), 'Nonfinite SH VJP')
                                destination += value
                            counts['rgb_vjp'] += 1
                        t = target(view)
                        keep = torch.from_numpy(t['known']).cuda()
                        labels = t['labels'][t['known']].astype(np.uint8)
                        y = torch.from_numpy(labels.astype(np.int64)).cuda()
                        a = prediction.detach()[keep].gather(1, y[:, None])[:, 0].cpu().numpy()
                        index = len(rows)
                        cache_rows.append({'name': view['name'], 'pixels': len(a),
                                           'a': direction.save_array(cache_folder, f'{index}.a.npy', a),
                                           'labels': direction.save_array(cache_folder, f'{index}.y.npy', labels)})
                        if phase == 'baseline':
                            original_view = original_predictions[view['name']]
                            drgb = context['rgb'].detach().cpu().numpy().astype(np.float64)-original_view['rgb']
                            dalpha = context['alpha'].detach().cpu().numpy().astype(np.float64)-original_view['alpha']
                            row.update(zero_rgb_change_mse=float(np.mean(drgb*drgb)),
                                       zero_alpha_change_max=float(np.max(np.abs(dalpha))),
                                       zero_alpha_change_mean=float(np.mean(np.abs(dalpha))))
                            if name != 'keep':
                                start = state_metadata[name]['children_start']
                                parents = state_metadata[name]['parent_indices']
                                child_xy = context['info']['means2d'][0, start:].detach().double().cpu().numpy()
                                parent_xy = projections[index]['means'][parents]
                                # Only actual active children have defined projection output.
                                radii = context['info']['radii'][0, start:].detach().cpu().numpy()
                                active = np.max(radii, -1) > 0 if radii.ndim == 2 else radii > 0
                                parent_active = projections[index]['active'][parents]
                                active &= np.concatenate((parent_active, parent_active))
                                displacement = child_xy-np.concatenate((parent_xy, parent_xy))
                                row['child_screen_displacement'] = {
                                    'active_children': int(active.sum()), 'values': displacement[active].tolist(),
                                    'child_offsets': np.flatnonzero(active).tolist(),
                                    'scope': 'actual means2d, only original and child active projections'}
                        rows.append(row)
                        del context, prediction, loss, rgb_loss, semantic_alpha
                        tick()
                    require(len(rows) == 8 and adapter.tensor_hash(q32) == array_sha(fullq.astype(np.float32)), 'Incomplete/changing-q pass')
                    counts['complete_fit_passes'] += 1
                    result = {'complete': True, 'phase': phase, 'q_ce': float(np.mean([r['raw_ce'] for r in rows])),
                              'rgb_mse': float(np.mean([r['full_rgb_mse'] for r in rows])), 'rows': rows,
                              'q32_sha256': q_sha(q), 'full_q32_sha256': array_sha(fullq.astype(np.float32)),
                              'sh_sha256': sh_sha(sh), 'elapsed_seconds': time.monotonic()-started,
                              'target_cache': {'q32_sha256': q_sha(q), 'folder': str(cache_folder), 'records': cache_rows,
                                               'hashes': {str(p): sha(p) for p in sorted(cache_folder.iterdir())}}}
                    if gradient_q:
                        result['q_gradient'] = gq/8
                    if gradient_sh:
                        result['sh_gradient'] = tuple(v/8 for v in gsh)
                    with (output/'passes.jsonl').open('a') as handle:
                        handle.write(json.dumps({'state': name, **{k: v for k, v in result.items()
                            if k not in ('q_gradient', 'sh_gradient')}}, allow_nan=False)+'\n')
                    return result
                except BaseException:
                    counts['partial_pass_views'] += len(rows)
                    raise
            return full

        for name in ['keep', *library['actions']]:
            base, editable, zero_sh, fixed, outside = install_state(name)
            optimizer = SHTransaction(zero_sh, device='cuda')
            q, sh, result = solve_refit(base[editable], zero_sh, full_factory(name, base, editable), fw,
                                      sh_optimizer=optimizer, release=release)
            copy_sh(editable, sh)
            state_scope(editable, fixed, outside)
            path = output/f'{name}_delta.npz'
            np.savez(path, q_master=q, q_renderer=q.astype(np.float32), sh0=sh[0], sh_rest=sh[1], editable_indices=editable)
            final_deltas[name] = {'path': str(path), 'sha256': sha(path), 'q32_sha256': q_sha(q), 'sh_sha256': sh_sha(sh)}
            results[name] = result
            write(output/f'{name}_refit.json', result)
            progress('fitted_'+name)
        original_predictions.clear(); projections.clear()

        def readonly_pass(name, *, final, views, phase, create_support=False):
            base, editable, sh, fixed, outside = install_state(name)
            delta = final_deltas[name]
            if final:
                require(sha(delta['path']) == delta['sha256'], 'Endpoint changed')
                with np.load(delta['path'], allow_pickle=False) as data:
                    base[editable] = data['q_master']
                    sh = (data['sh0'], data['sh_rest'])
                copy_sh(editable, sh)
            q = torch.from_numpy(base.astype(np.float32)).cuda()
            rows = []
            for view in views:
                context, _, _ = render(view)
                if create_support:
                    put_support(view, footprint(context['info']), local_ids)
                with torch.no_grad():
                    prediction, alpha = raw(context, q, view)
                    require(float((alpha-context['alpha']).abs().max()) <= 2e-6, 'Read-only alpha paths differ')
                    _, _, row = measure(view, context, prediction)
                rows.append(row)
                tick()
            state_scope(editable, fixed, outside)
            return {'state': name, 'phase': phase, 'rows': rows,
                    'q32_sha256': array_sha(base.astype(np.float32)), 'sh_sha256': sh_sha(sh),
                    'delta_sha256': delta['sha256'] if final else None}

        repeat_A = readonly_pass('keep', final=True, views=plan['A'], phase='fitted_keep_A_repeat')
        require(repeat_A['sh_sha256'] == results['keep']['final']['sh_sha256']
                and repeat_A['q32_sha256'] == results['keep']['final']['full_q32_sha256'], 'Repeat not the same fitted keep')
        A_rows = {name: result['final']['rows'] for name, result in results.items()}
        validate_fixed_support(A_rows['keep'], repeat_A['rows'],
                               *[r['baseline']['rows'] for r in results.values()], *A_rows.values())
        selection = select_on_a(A_rows['keep'], repeat_A['rows'],
                                {k: v for k, v in A_rows.items() if k != 'keep'}, gradient_scores)
        selection_path = output/'immutable_A_selection.json'
        write(selection_path, {'selection': selection, 'state_deltas': final_deltas,
                               'keep_repeat': repeat_A, 'B_unique_inputs_opened': False,
                               'source_plan_sha256': report['plan_sha256']})
        report['A_selection'] = {'path': str(selection_path), 'sha256': sha(selection_path)}
        barrier.open_b(selection_path, report['A_selection']['sha256'])
        report['B_unique_inputs_opened'] = True
        progress('A_choices_locked_B_opened')
        B_results = {}
        for name in ['keep', *library['actions']]:
            zero = readonly_pass(name, final=False, views=plan['B'], phase='B_zero', create_support=name == 'keep')
            final = readonly_pass(name, final=True, views=plan['B'], phase='B_final')
            B_results[name] = {'zero': zero, 'final': final}
        repeat_B = readonly_pass('keep', final=True, views=plan['B'], phase='fitted_keep_B_repeat')
        require(repeat_B['delta_sha256'] == B_results['keep']['final']['delta_sha256']
                and repeat_B['q32_sha256'] == B_results['keep']['final']['q32_sha256']
                and repeat_B['sh_sha256'] == B_results['keep']['final']['sh_sha256'], 'B repeat not the same fitted keep')
        validate_fixed_support(B_results['keep']['final']['rows'], repeat_B['rows'],
            *[endpoint['rows'] for r in B_results.values() for endpoint in r.values()])
        B = evaluate_on_b(selection, B_results['keep']['final']['rows'], repeat_B['rows'],
                          {k: v['final']['rows'] for k, v in B_results.items() if k != 'keep'})
        require(sha(selection_path) == report['A_selection']['sha256'], 'Immutable A choices changed')
        require(counts['scene'] == 1144 and counts['complete_fit_passes'] == 130
                and counts['q_vjp'] == 960 and counts['rgb_vjp'] == 840
                and counts['control_vjp'] == 8 and counts['gsplat'] == 1152 and counts['shader'] == 1144,
                'Actual calls differ from fixed budget')
        require(not list(temporary.iterdir()), 'Temporary target caches not released')
        write(output/'analysis.json', {'A': {'selection': selection, 'repeat': repeat_A},
            'B': B, 'B_states': B_results, 'B_repeat': repeat_B, 'state_metadata': state_metadata,
            'state_deltas': final_deltas, 'counts': counts, 'support_hashes': support_hashes,
            'conditional_optimum_claimed': False, 'VAL_or_performance_adoption': False,
            'shared_valid_exception': plan['shared_valid_exception']})
        report.update(analysis_sha256=sha(output/'analysis.json'), scientific_status=B.get('status', B.get('scientific_status')),
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    finally:
        cv2.imread = original_imread
        try:
            if scene is not None and original is not None:
                scene.load_state_dict(original, strict=True)
                report['restoration'] = adapter.restore_scene(scene, captured)
                require(report['restoration']['state_exact'] and report['restoration']['flags_modes_gradients_restored'], 'Full restoration failed')
        finally:
            adapter.numerical_flags(previous)
            report['numerics_restored'] = adapter.numerical_flags() == previous
            require(report['numerics_restored'], 'Numerical flags failed restoration')
        if counts['scene']:
            attest_runtime()


def execute(path, expected):
    require(sha(path) == expected, 'Wrong plan SHA')
    plan = read(path); output = Path(plan['output'])
    require(not any((output/k).exists() for k in ('execution_started.json', 'execution_receipt.json', 'analysis.json', 'temporary')), 'Refuse prior attempt')
    write(output/'execution_started.json', {'plan_sha256': expected, 'pid': os.getpid(), 'time': time.time()})
    report = {'status': 'running', 'plan_sha256': expected, 'B_unique_inputs_opened': False,
              'counts': {k: 0 for k in ('scene', 'gsplat', 'shader', 'q_vjp', 'rgb_vjp', 'control_vjp',
                                        'target_decodes', 'complete_fit_passes', 'partial_pass_views')},
              'val_views': 0, 'head_calls': 0, 'teacher_calls': 0, 'production_checkpoint_writes': 0}
    started = time.monotonic()
    def expired(*_):
        raise TimeoutError('Fixed600s deadline; no retries')
    previous = signal.signal(signal.SIGALRM, expired); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan)
        sys.path.insert(0, plan['source_snapshot'])
        old = importlib.import_module('optimize_raw_simplex')
        report['gpu_before'] = old.gpu_inventory()
        perform(plan, report, started+SPEC['internal_seconds'])
        verify(plan, after_choice=report['B_unique_inputs_opened'])
        report.update(status='completed', sources_inputs_unchanged=True)
    except BaseException as error:
        report.update(status='inconclusive_timeout' if isinstance(error, TimeoutError) else 'failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
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
