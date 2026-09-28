"""One frozen, TRAIN-only coupled-field gradient diagnostic; no optimizer/update."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

AUDIT = Path('/mnt/data/SHM2026/runs/support_coupled_source_cpu_audit_v1/report.json')
AUDIT_SHA = '8e9aea80c1bf77bc1f77e677268bfa682ed746d7c888dd81f012df770f8396ce'
BASE_SHA = 'a329fc70d582972fc338715765304b99da208c821ee34288809db500d5acd6b9'
DOCUMENT = 'coupled_semantic_gradient_diagnostic.md'
FEATURE = 'splats.sem_features'
KEYS = ('raw', 'full_final', 'feature_final')
SPEC = {
    'protocol': 'support_coupled_terminal_field_gradient_v1',
    'pixel_protocol': 'legacy_mixed_v1', 'main_views': 259,
    'repeat_and_fd_names': ['002.png', '205.png'], 'main_order': 'labeled TRAIN sorted by name',
    'width': 1320, 'height': 989, 'degree': 3, 'geometry_grad': False,
    'raw': '.5 * original weighted CE; original valid-pixel denominator',
    'final': 'original weighted CE + .2 Lovasz + .001 residual.square().mean()',
    'feature_final': 'same final values; detach BOTH head p3d/entropy and log prior; features stay differentiable',
    'fd_feature_final': 'both p3d occurrences anchored to the unchanged baseline prior for every endpoint',
    'trainable_parameter_for_vjp_only': FEATURE,
    'projector': 'fixed audit FP64 class-difference row projection; null=I-row',
    'epsilons': [.001, .0005, .00025],
    'directions': ['negative raw gradient', 'negative feature-final null gradient'],
    'direction_normalization': 'FP64 absolute maximum; endpoints independently rounded into original FP32 F',
    'main_fd_contracts': ['raw/dR', 'full_final/dN', 'feature_final/dN'],
    'cross_fd_contract': 'full_final/dR descriptive, not a nonzero primary gate',
    'null_zero_control': 'raw/dN absolute tolerance; no nonzero or descent requirement',
    'main_fd_contract_count': 18, 'main_fd_relative_tolerance': .05,
    'measurability_multiplier': 10., 'algebra_relative_l2_tolerance': 1e-5,
    'scene_renders': 285, 'raster_calls': 570, 'field_vjps': 783,
    'optimizer_steps': 0, 'checkpoint_writes': 0,
    'internal_seconds': 210, 'external_seconds': 240,
    'numerics': {'cudnn_allow_tf32': False, 'matmul_allow_tf32': False,
                 'float32_matmul_precision': 'highest', 'cudnn_benchmark': False, 'autocast': False},
    'statistics': 'per-view statistics plus equal-view mean of three gradients; repeated views excluded',
    'interpretation': 'terminal conditional gradients only; neither historical training cause nor an adoption gate',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_files(snapshot):
    return {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.md'}}


def fixed_views(manifest, audit):
    require(manifest.get('pixel_protocol', 'legacy_mixed_v1') == 'legacy_mixed_v1', 'Legacy profile required')
    training = [v for v in manifest['views'] if v['split'] == 'train']
    names = [v['name'] for v in training]
    require(names == audit['train_camera_names'] and len(set(names)) == 350, 'TRAIN camera mapping differs')
    labeled = sorted((v for v in training if v.get('mask_path')), key=lambda v: v['name'])
    require([v['name'] for v in labeled] == audit['train_labeled_names'] and len(labeled) == 259,
            'The full 259 labeled TRAIN list differs')
    keys = ('name', 'split', 'width', 'height', 'K', 'w2c_original', 'mask_path', 'valid_path')
    rows = [{**{k: v[k] for k in keys}, 'camera_index': names.index(v['name'])} for v in labeled]
    require(all(v['width'] == 1320 and v['height'] == 989 and v['valid_path'] for v in rows), 'Native support differs')
    require(set(SPEC['repeat_and_fd_names']) <= {v['name'] for v in rows}, 'Missing calibration views')
    return rows, names


def checked_projection(value, device='cpu'):
    p = torch.as_tensor(value, dtype=torch.float64, device=device)
    require(p.shape == (16, 16) and bool(torch.isfinite(p).all()), 'Invalid fixed projection')
    require(torch.allclose(p, p.T, atol=1e-14, rtol=0)
            and torch.allclose(p@p, p, atol=1e-14, rtol=0), 'Nonorthogonal fixed projector')
    return p


def captured_render(scene, render):
    """Keep exact graph tensors from the native head, without modifying its call."""
    captured = []

    def hook(_module, args, kwargs):
        require(len(args) == 4 and set(kwargs) == {'p3d'}, 'Unexpected original refiner signature')
        captured.append((args, kwargs))

    handle = scene.refiner.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        result = render()
    finally:
        handle.remove()
    require(len(captured) == 1, 'Require exactly one native head call')
    args, kwargs = captured[0]
    require(kwargs['p3d'] is result['p3d'], 'Native head prior is not returned p3d')
    return result, args


def feature_readout(scene, result, args, anchor=None):
    """At FD endpoints anchor is BASE p3d, not the newly rendered endpoint prior."""
    prior = (result['p3d'] if anchor is None else anchor).detach()
    residual = scene.refiner(*args, p3d=prior)
    probabilities = (prior.log()+residual).softmax(-1)
    if anchor is None:
        require(torch.equal(residual, result['residual'])
                and torch.equal(probabilities, result['probabilities']), 'Detached re-readout is not forward exact')
    return probabilities, residual


def objectives(result, feature_prediction, labels, valid, weights, semantic_loss):
    probabilities, residual = feature_prediction
    return {
        'raw': .5*semantic_loss(result['p3d'], labels, valid, weights, 0),
        'full_final': semantic_loss(result['probabilities'], labels, valid, weights)
                      + .001*result['residual'].square().mean(),
        'feature_final': semantic_loss(probabilities, labels, valid, weights)+.001*residual.square().mean(),
    }


def dot(a, b):
    return float((a.double()*b.double()).sum())


def norm(a):
    return float(torch.linalg.vector_norm(a.double()))


def vector_stats(g, p):
    g = g.double()
    row = g@p
    null = g-row
    return {'total_energy': dot(g, g), 'row_energy': dot(row, row), 'null_energy': dot(null, null)}


def gradient_stats(gradients, p):
    r, f, a = (gradients[k] for k in KEYS)
    norms = {k: norm(v) for k, v in gradients.items()}
    total = norm(r.double()+f.double())
    pair = dot(r, f)
    r_null = norm(r.double()-r.double()@p)
    bypass = f.double()-a.double()
    bypass_null = norm(bypass-bypass@p)
    denominator = norms['full_final']+norms['feature_final']
    contributions = {}
    for key in ('full_final', 'feature_final'):
        per_gaussian = (r.double()*gradients[key].double()).sum(-1)
        contributions[key] = {'positive_sum': float(per_gaussian.clamp_min(0).sum()),
                              'negative_absolute_sum': float((-per_gaussian.clamp_max(0)).sum()),
                              'signed_sum': float(per_gaussian.sum())}
    return {'energies': {k: vector_stats(v, p) for k, v in gradients.items()},
            'raw_dot_full_final': pair, 'raw_dot_feature_final': dot(r, a),
            'per_gaussian_inner_product_mass': contributions,
            'raw_full_cosine': pair/(norms['raw']*norms['full_final']) if norms['raw']*norms['full_final'] else None,
            'raw_feature_cosine': dot(r, a)/(norms['raw']*norms['feature_final']) if norms['raw']*norms['feature_final'] else None,
            'raw_plus_final_norm': total,
            'cancel_ratio': total/(norms['raw']+norms['full_final']) if norms['raw']+norms['full_final'] else None,
            'raw_direction_change_under_negative_sum': -dot(r, r.double()+f.double()),
            'raw_null_relative_l2': r_null/norms['raw'] if norms['raw'] else None,
            'bypass_null_relative_l2': bypass_null/denominator if denominator else None,
            'zero_gradient': {k: value == 0 for k, value in norms.items()},
            'algebra_passed': bool((not norms['raw'] or r_null <= 1e-5*norms['raw'])
                                   and (not denominator or bypass_null <= 1e-5*denominator))}


def fixed_direction(gradient, projection=None):
    d = -gradient.double()
    if projection is not None:
        d = d-d@projection
    maximum = float(d.abs().max())
    return (d/maximum if maximum else torch.zeros_like(d)), maximum


def fp32_ulp(value):
    return float(abs(np.spacing(np.float32(value))))


def calibrate_fd(gradient, repeat_gradient, actual_direction, base_loss, repeat_loss,
                 plus_loss, minus_loss, h, *, primary):
    g, d = gradient.double(), actual_direction.double()
    analytic = dot(g, d)
    observed = (plus_loss-minus_loss)/(2*h)
    n = g.numel()
    gamma = n*np.finfo(np.float64).eps/(1-n*np.finfo(np.float64).eps)
    floor_terms = {
        'repeat_gradient_dot': abs(dot(g-repeat_gradient.double(), d)),
        'repeat_loss_over_2h': abs(base_loss-repeat_loss)/(2*h),
        'two_fp32_scalar_ulps_over_2h': 2*max(map(fp32_ulp, (base_loss, repeat_loss, plus_loss, minus_loss)))/(2*h),
        'fp64_dot_roundoff_bound': gamma*float((g*d).abs().sum()),
    }
    floor = max(floor_terms.values())
    measurable = abs(analytic) > 10*floor
    relative = abs(observed-analytic)/abs(analytic) if analytic else None
    passed = bool(measurable and relative is not None and relative <= .05)
    return {'analytic_realized': analytic, 'central_difference': observed, 'absolute_error': abs(observed-analytic),
            'relative_error': relative, 'floor_terms': floor_terms, 'floor': floor, 'measurable': bool(measurable),
            'primary': primary, 'passed': passed,
            'status': ('passed' if passed else 'failed' if measurable else 'inconclusive' if primary else 'cross_inconclusive')}


def null_zero_control(calibration, gradient, actual_direction, p):
    g, d = gradient.double(), actual_direction.double()
    gr, dr = g@p, d@p
    gn, dn = g-gr, d-dr
    leakage = norm(gr)*norm(dr)+norm(gn)*norm(dn)
    tolerance = 10*calibration['floor']+leakage
    return {**calibration, 'primary': False, 'row_direction_norm': norm(dr), 'null_direction_norm': norm(dn),
            'actual_direction_row_fraction': norm(dr)/norm(d) if norm(d) else None,
            'first_order_cauchy_leakage_bound': leakage, 'absolute_zero_tolerance': tolerance,
            'analytic_zero_consistent': bool(abs(calibration['analytic_realized']) <= tolerance),
            'zero_consistent': bool(abs(calibration['central_difference']) <= tolerance),
            'status': 'consistent' if abs(calibration['central_difference']) <= tolerance else 'not_resolved',
            'scope': 'first-order FP32 displacement/numerical-gradient leakage allowance, not a nonlinear error theorem'}


@contextmanager
def preserved_scene(scene, cameras, report):
    before = {k: v.detach().cpu().clone() for k, v in scene.state_dict().items()}
    flags = {k: p.requires_grad for k, p in scene.named_parameters()}
    grads = {k: None if p.grad is None else p.grad.detach().clone() for k, p in scene.named_parameters()}
    modes = {k: m.training for k, m in scene.named_modules()}
    original_cameras = cameras.detach().clone()
    report['tensor_hashes_before'] = {k: tensor_hash(v) for k, v in before.items()}
    report['camera_hash_before'] = tensor_hash(cameras)
    try:
        scene.eval()
        scene.requires_grad_(False)
        scene.splats['sem_features'].requires_grad_(True)
        yield before
    finally:
        report['non_feature_unchanged_before_restore'] = all(
            tensor_hash(v) == report['tensor_hashes_before'][k] for k, v in scene.state_dict().items() if k != FEATURE)
        report['cameras_unchanged_before_restore'] = torch.equal(cameras, original_cameras)
        with torch.no_grad():
            scene.load_state_dict(before, strict=True)
            cameras.copy_(original_cameras)
        for k, p in scene.named_parameters():
            p.requires_grad_(flags[k])
            p.grad = grads[k]
        for k, m in scene.named_modules():
            m.training = modes[k]
        after = {k: tensor_hash(v) for k, v in scene.state_dict().items()}
        report.update(tensor_hashes_after=after, all_tensors_restored_exact=after == report['tensor_hashes_before'],
                      flags_restored_exact=flags == {k: p.requires_grad for k, p in scene.named_parameters()},
                      modes_restored_exact=modes == {k: m.training for k, m in scene.named_modules()},
                      gradients_restored_exact=all((p.grad is None if grads[k] is None else torch.equal(p.grad, grads[k]))
                                                  for k, p in scene.named_parameters()),
                      cameras_restored_exact=torch.equal(cameras, original_cameras))


@contextmanager
def pixels_only(allowed, evidence):
    original = cv2.imread
    allowed = {str(Path(p).resolve()) for p in allowed}
    evidence['reads'] = []

    def read(path, *args, **kwargs):
        name = str(Path(path).resolve())
        require(name in allowed, 'Pixel read outside labeled TRAIN masks/valid')
        evidence['reads'].append(name)
        return original(path, *args, **kwargs)
    cv2.imread = read
    try:
        yield
    finally:
        cv2.imread = original


def load_labels(view, device):
    mask = cv2.imread(view['mask_path'], cv2.IMREAD_GRAYSCALE)
    valid = cv2.imread(view['valid_path'], cv2.IMREAD_GRAYSCALE)
    require(mask is not None and valid is not None and mask.shape == valid.shape == (989, 1320), 'Invalid native labels')
    require(bool(np.isin(mask, [0, 1, 2, 3, 4, 255]).all()), 'Invalid classes')
    mask = mask.astype(np.int64)
    mask[valid == 0] = 255
    return torch.from_numpy(mask).to(device), torch.from_numpy((valid > 0).astype(np.float32)).to(device)


def gpu_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True, timeout=5)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    require(not lines, f'GPU compute processes or unrecognized query response: {lines}')
    return {'compute_processes': [], 'query': result.args}


def numerical_flags():
    return {'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark}


@contextmanager
def numerical_contract(report):
    before = numerical_flags()
    report['before'] = before
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        report['actual'] = {**numerical_flags(), 'autocast': False}
        require(report['actual'] == SPEC['numerics'], 'Numerical flags differ')
        with torch.autocast(device_type='cuda', enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(before['float32_matmul_precision'])
        torch.backends.cudnn.allow_tf32 = before['cudnn_allow_tf32']
        torch.backends.cuda.matmul.allow_tf32 = before['matmul_allow_tf32']
        torch.backends.cudnn.benchmark = before['cudnn_benchmark']
        report['after'] = numerical_flags()
        report['restored_exact'] = report['after'] == before


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists() and not torch.cuda.is_initialized(), 'New directory and CPU-only preparation required')
    require(digest(AUDIT) == AUDIT_SHA, 'Audited source/weights changed')
    audit = json.loads(AUDIT.read_text())
    require(audit['status'] == 'completed_cpu_source_rank_and_train_class_weights'
            and audit['checkpoint_step'] == 8000 and audit['raw_weights_equal_final'], 'Audit contract differs')
    base, origin = Path(audit['base']), Path(audit['source_snapshot'])
    require(digest(base) == BASE_SHA and source_files(origin) == audit['source_hashes'], 'Old coupled source/base changed')
    require(len(audit['source_hashes']) == 17 and audit['rank'] == 4, 'Wrong source/projector')
    checked_projection(audit['row_projection'])
    manifest = root/'artifacts/prepared/manifest.json'
    views, names = fixed_views(json.loads(manifest.read_text()), audit)
    inputs = dict(audit['input_hashes'])
    inputs.update({str(p): digest(p) for p in [AUDIT, base, root/'docs'/DOCUMENT,
                   root/'pyproject.toml', Path(__file__).resolve(), root/'tests/test_coupled_semantic_gradients.py']})
    allowed = sorted({str(Path(v[k]).resolve()) for v in views for k in ('mask_path', 'valid_path')})
    inputs.update({p: digest(p) for p in allowed})
    require(all(digest(p) == h for p, h in inputs.items()), 'Bound audit input changed')
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(origin/'bridge_rgs', snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for p in [Path(__file__), root/'docs'/DOCUMENT, root/'tests/test_coupled_semantic_gradients.py']:
        shutil.copy2(p, snapshot/p.name)
    plan = {'status': 'locked_pending_root_gpu_handoff', 'specification': SPEC, 'root': str(root),
            'output': str(output), 'source_snapshot': str(snapshot), 'source_hashes': source_files(snapshot),
            'original_source_hashes': audit['source_hashes'], 'input_hashes': inputs,
            'base': str(base), 'base_sha256': BASE_SHA, 'manifest': str(manifest), 'views': views,
            'training_camera_names': names, 'training_camera_sha256': audit['camera_tensor_sha256'],
            'class_weights': audit['class_weights_fp32'], 'weights_sha256': audit['weights_sha256'],
            'row_projection': audit['row_projection'], 'allowed_pixel_paths': allowed,
            'audit': str(AUDIT), 'audit_sha256': AUDIT_SHA, 'prepare_cuda_initialized': False,
            'env': {'PYTHONPATH': str(snapshot), 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
                    'TORCH_CUDA_ARCH_LIST': '12.0', 'MAX_JOBS': '4'},
            'external_timeout_seconds': 240, 'minimum_free_bytes': 256*1024**2,
            'storage': 'JSON statistics only; three FP64 gradient accumulators and two calibration views in RAM; no 259-gradient archive'}
    require(all(plan['source_hashes'][k] == h for k, h in audit['source_hashes'].items()), 'Copied old source differs')
    write(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    require(plan['specification'] == SPEC and plan['status'] == 'locked_pending_root_gpu_handoff', 'Unexpected plan')
    snapshot = Path(plan['source_snapshot'])
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Use the frozen entrypoint')
    require(source_files(snapshot) == plan['source_hashes'], 'Source tree changed')
    require(all(plan['source_hashes'][k] == h for k, h in plan['original_source_hashes'].items()), 'Old package changed')
    require(all(digest(p) == h for p, h in plan['input_hashes'].items()), 'Bound input changed')
    audit = json.loads(AUDIT.read_text())
    require(digest(AUDIT) == AUDIT_SHA and plan['row_projection'] == audit['row_projection']
            and plan['class_weights'] == audit['class_weights_fp32']
            and plan['base_sha256'] == BASE_SHA == digest(plan['base']), 'Evidence binding differs')
    require(fixed_views(json.loads(Path(plan['manifest']).read_text()), audit)
            == (plan['views'], plan['training_camera_names']), 'TRAIN view mapping changed')
    for name in ('train', 'model', 'refinement', 'losses'):
        importlib.import_module('bridge_rgs.'+name)
    imports = {}
    for name, module in list(sys.modules.items()):
        if name == 'bridge_rgs' or name.startswith('bridge_rgs.'):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(snapshot) and plan['source_hashes'].get(str(path.relative_to(snapshot))) == digest(path),
                    f'Unfrozen import: {name}')
            imports[name] = {'path': str(path), 'sha256': digest(path)}
    return imports


def execute(plan_path):
    started = time.monotonic()
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    output = Path(plan['output'])
    write(output/'execution_started.json', {'pid': os.getpid(), 'plan_sha256': digest(plan_path)})
    os.chdir(plan['root'])
    torch.set_num_threads(8)
    report = {'status': 'failed', 'specification': SPEC, 'plan_sha256': digest(plan_path),
              'base_sha256': BASE_SHA, 'scene_renders': 0, 'raster_calls': 0, 'field_vjps': 0,
              'optimizer_steps': 0, 'checkpoint_writes': 0, 'views': [], 'repeats': [], 'finite_differences': [],
              'restoration': {}, 'pixel_scope': {}, 'numerical_flags': {},
              'real_rgb_pixels_decoded': 0, 'val_pixels_decoded': 0}
    previous_handler = signal.getsignal(signal.SIGALRM)

    def deadline(*_):
        raise TimeoutError('Fixed 210-second deadline; preserve failure and do not retry')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(SPEC['internal_seconds'])
    try:
        report['actual_imports'] = verify(plan)
        report['gpu_before'] = gpu_idle()
        require(shutil.disk_usage(output).free >= plan['minimum_free_bytes'], 'Insufficient free output space')
        import gsplat

        from bridge_rgs.losses import semantic_loss
        from bridge_rgs.train import load_scene
        scene, checkpoint = load_scene(plan['base'], 'cuda')
        cameras = checkpoint['training_cameras'].clone().cuda()
        require(tensor_hash(cameras) == plan['training_camera_sha256'], 'Saved camera mapping differs')
        require(checkpoint['step'] == 8000 and scene.feature_dim == 16
                and scene.refiner_config == {'type': 'multiscale', 'channels': 64, 'residual_bound': 6., 'context': 'pyramid_strip'},
                'Wrong coupled architecture/step')
        weights = torch.tensor(plan['class_weights'], device='cuda', dtype=torch.float32)
        require(tensor_hash(weights) == plan['weights_sha256'], 'Class weights changed')
        p = checked_projection(plan['row_projection'], 'cuda')
        old_raster = gsplat.rasterization

        def raster(*args, **kwargs):
            report['raster_calls'] += 1
            require(report['raster_calls'] <= SPEC['raster_calls'], 'Raster budget exceeded')
            return old_raster(*args, **kwargs)
        gsplat.rasterization = raster
        try:
            with numerical_contract(report['numerical_flags']), preserved_scene(scene, cameras, report['restoration']) as before, pixels_only(plan['allowed_pixel_paths'], report['pixel_scope']):
                feature = scene.splats['sem_features']
                sums = {k: torch.zeros_like(feature, dtype=torch.float64) for k in KEYS}
                calibration = {}

                def render(view):
                    report['scene_renders'] += 1
                    require(report['scene_renders'] <= SPEC['scene_renders'], 'Scene budget exceeded')
                    K = torch.tensor(view['K'], dtype=torch.float32, device='cuda')
                    pose = cameras[view['camera_index']].detach().clone()
                    return captured_render(scene, lambda: scene.render(K, pose, view['width'], view['height'],
                        degree=3, semantics=True, geometry_grad=False, refine=True, absgrad=False,
                        semantic_classifier_grad=True, refinement_grad_to_field=True))

                def gradients_for(view):
                    result, args = render(view)
                    auxiliary = feature_readout(scene, result, args)
                    labels, valid = load_labels(view, 'cuda')
                    losses = objectives(result, auxiliary, labels, valid, weights, semantic_loss)
                    require(all(bool(torch.isfinite(loss)) for loss in losses.values()), 'Nonfinite objective')
                    require(torch.equal(losses['full_final'], losses['feature_final']), 'Baseline objectives differ')
                    gradients = {}
                    for i, key in enumerate(KEYS):
                        gradients[key] = torch.autograd.grad(losses[key], feature, retain_graph=i < 2)[0]
                        report['field_vjps'] += 1
                        require(bool(torch.isfinite(gradients[key]).all()), 'Nonfinite field gradient')
                    require(all(parameter.grad is None for parameter in scene.parameters()), 'VJP accumulated parameter gradients')
                    return gradients, {k: float(v.detach()) for k, v in losses.items()}, result['p3d'].detach(), labels, valid

                for index, view in enumerate(plan['views']):
                    gradients, losses, prior, labels, valid = gradients_for(view)
                    statistics = gradient_stats(gradients, p)
                    report['views'].append({'name': view['name'], 'losses': losses, **statistics})
                    for key in KEYS:
                        sums[key].add_(gradients[key])
                    if view['name'] in SPEC['repeat_and_fd_names']:
                        calibration[view['name']] = {'gradients': {k: v.cpu() for k, v in gradients.items()},
                            'losses': losses, 'prior': prior.cpu(), 'labels': labels.cpu(), 'valid': valid.cpu()}
                    del gradients, losses, prior, labels, valid
                    if (index+1) % 25 == 0:
                        print(json.dumps({'completed_main_views': index+1, 'elapsed_seconds': time.monotonic()-started}), flush=True)
                for key in KEYS:
                    sums[key].div_(259)
                report['aggregate'] = {
                    'equal_view_mean_gradient': gradient_stats(sums, p),
                    'mean_of_per_view_raw_dot_full_final': sum(v['raw_dot_full_final'] for v in report['views'])/259,
                    'mean_of_per_view_raw_dot_feature_final': sum(v['raw_dot_feature_final'] for v in report['views'])/259,
                    'negative_dot_fraction_descriptive_only': sum(v['raw_dot_full_final'] < 0 for v in report['views'])/259,
                    'equal_view_mean_losses': {k: sum(v['losses'][k] for v in report['views'])/259 for k in KEYS},
                }
                del sums
                for name in SPEC['repeat_and_fd_names']:
                    view = next(v for v in plan['views'] if v['name'] == name)
                    gradients, losses, prior, labels, valid = gradients_for(view)
                    base = calibration[name]
                    base['repeat_gradients'] = {k: v.cpu() for k, v in gradients.items()}
                    base['repeat_losses'] = losses
                    report['repeats'].append({'name': name, 'losses': losses,
                        'prior_exact': torch.equal(prior.cpu(), base['prior']),
                        'gradient_l2_differences': {k: norm(base['repeat_gradients'][k]-base['gradients'][k]) for k in KEYS},
                        **gradient_stats(gradients, p)})
                    del gradients, losses, prior, labels, valid
                # The FD points are independent perturbations of the unchanged original FP32 field.
                original = before[FEATURE].to('cuda')
                for name in SPEC['repeat_and_fd_names']:
                    view = next(v for v in plan['views'] if v['name'] == name)
                    base = calibration[name]
                    gradients = {k: v.to('cuda') for k, v in base['gradients'].items()}
                    repeats = {k: v.to('cuda') for k, v in base['repeat_gradients'].items()}
                    anchor, labels, valid = (base[k].to('cuda') for k in ('prior', 'labels', 'valid'))
                    for direction_name, key in (('dR', 'raw'), ('dN', 'feature_final')):
                        direction, absmax = fixed_direction(gradients[key], p if direction_name == 'dN' else None)
                        for h in SPEC['epsilons']:
                            endpoint_losses, endpoints = {}, {}
                            try:
                                for sign in (1, -1):
                                    endpoint = (original.double()+sign*h*direction).float()
                                    endpoints[sign] = endpoint
                                    try:
                                        with torch.no_grad():
                                            feature.copy_(endpoint)
                                            result, args = render(view)
                                            auxiliary = feature_readout(scene, result, args, anchor=anchor)
                                            values = objectives(result, auxiliary, labels, valid, weights, semantic_loss)
                                            require(all(bool(torch.isfinite(v)) for v in values.values()), 'Nonfinite FD objective')
                                            endpoint_losses[sign] = {k: float(v) for k, v in values.items()}
                                        del result, args, auxiliary, values
                                    finally:
                                        with torch.no_grad():
                                            feature.copy_(original)
                            finally:
                                with torch.no_grad():
                                    feature.copy_(original)
                            actual = (endpoints[1].double()-endpoints[-1].double())/(2*h)
                            measured = {}
                            for objective in (KEYS if direction_name == 'dN' else ('raw', 'full_final')):
                                primary = (direction_name == 'dR' and objective == 'raw') or (
                                    direction_name == 'dN' and objective in ('full_final', 'feature_final'))
                                measured[objective] = calibrate_fd(gradients[objective], repeats[objective], actual,
                                    base['losses'][objective], base['repeat_losses'][objective],
                                    endpoint_losses[1][objective], endpoint_losses[-1][objective], h, primary=primary)
                            if direction_name == 'dN':
                                measured['raw'] = null_zero_control(measured['raw'], gradients['raw'], actual, p)
                            report['finite_differences'].append({'name': name, 'direction': direction_name,
                                'epsilon': h, 'unscaled_direction_absmax': absmax,
                                'actual_direction_norm': norm(actual), 'actual_direction_row_norm': norm(actual@p),
                                'endpoint_losses': {'plus': endpoint_losses[1], 'minus': endpoint_losses[-1]},
                                'contracts': measured})
                    del gradients, repeats, anchor, labels, valid
                require(report['scene_renders'] == 285 and report['raster_calls'] == 570
                        and report['field_vjps'] == 783 and len(report['views']) == 259, 'Incomplete fixed budget')
                primary = [v for row in report['finite_differences'] for v in row['contracts'].values() if v['primary']]
                require(len(primary) == 18, 'Incorrect primary calibration count')
                report['primary_calibration_passed'] = all(v['passed'] for v in primary)
                report['algebra_passed'] = all(v['algebra_passed'] for v in report['views']+report['repeats']) \
                    and report['aggregate']['equal_view_mean_gradient']['algebra_passed']
                report['null_zero_controls_consistent'] = all(row['contracts']['raw']['zero_consistent']
                    for row in report['finite_differences'] if row['direction'] == 'dN')
                report['status'] = 'completed'
        finally:
            gsplat.rasterization = old_raster
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        report['source_and_inputs_unchanged'] = (source_files(Path(plan['source_snapshot'])) == plan['source_hashes']
            and all(digest(path) == expected for path, expected in plan['input_hashes'].items()))
        report['elapsed_seconds'] = time.monotonic()-started
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        flags = ('all_tensors_restored_exact', 'flags_restored_exact', 'modes_restored_exact',
                 'gradients_restored_exact', 'cameras_restored_exact', 'non_feature_unchanged_before_restore',
                 'cameras_unchanged_before_restore')
        report['technical_gate_passed'] = bool(report['status'] == 'completed'
            and report.get('primary_calibration_passed', False) and report.get('algebra_passed', False)
            and report['source_and_inputs_unchanged'] and report['numerical_flags'].get('restored_exact', False)
            and all(report['restoration'].get(k, False) for k in flags))
        write(output/'execution_receipt.json', report)
    print(json.dumps({'status': report['status'], 'technical_gate_passed': report['technical_gate_passed'],
                      'receipt': str(output/'execution_receipt.json'), 'sha256': digest(output/'execution_receipt.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path)
    group.add_argument('--execute', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.prepare:
        plan = prepare(args.root, args.prepare)
        print(json.dumps({'plan': str(plan), 'sha256': digest(plan), 'runner_sha256': digest(__file__)}))
    else:
        execute(args.execute)


if __name__ == '__main__':
    main()
