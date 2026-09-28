"""Fixed synthetic AA-opacity RGB derivative check; root alone runs CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
PORT = Path('/mnt/data/SHM2026/third_party/ibgs_aa_port_v1/python')
BINARY = PORT/'diff_plane_rasterization/_C.cpython-311-x86_64-linux-gnu.so'
BINARY_SHA = 'c504257ae703800e90fc79c8f0541f0cd7757cd3e293e078760a5535c2bae35d'
SPEC = {'protocol': 'ibgs_aa_gradient_v1', 'width': 64, 'height': 48, 'focal': 60.,
        'near_plane': .01, 'eps2d': .3, 'primary_h': .002, 'sensitivity_h': [.001, .004],
        'absolute_tolerance': 1e-5, 'relative_tolerance': .01, 'signal_floor': 1e-6,
        'directions': {'means': [.03, -.02, .2], 'log_scales': [1., -.3, .5],
                       'raw_quaternion': [.2, -.1, .15, .3], 'opacity_logits': [1.]},
        'positive_case': {'means': [.05, -.03, 2.], 'scales': [.035, .07, .021],
                          'raw_quaternion': [1., .3, -.1, .2], 'opacity': .4},
        'cap_case': {'means': [1/60, 1/60, 2.], 'scales': [.5, .5, .5],
                     'raw_quaternion': [1., 0., 0., 0.], 'opacity_logit': 6.},
        'colors': [.6, .3, .2], 'background': [.05, .02, .04],
        'loss': 'fixed central pixel y24,x32; FP32 upstream [1,.7,-.2], FP64 scalar reduction',
        'forward_calls': 33, 'backward_calls': 3, 'optimizer_steps': 0, 'data_reads': 0,
        'internal_seconds': 90, 'external_seconds': 120,
        'scope': 'Positive-rho unsaturated RGB chain only; cap-active counterexample and rho-zero control separate; not entire backend certification'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(payload)


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def fd_report(gradient, plus, minus, loss_plus, loss_minus, h):
    arrays = [np.asarray(x, np.float64) for x in (gradient, plus, minus)]
    require(arrays[0].shape == arrays[1].shape == arrays[2].shape
            and all(np.isfinite(x).all() for x in arrays)
            and np.isfinite([loss_plus, loss_minus, h]).all() and h > 0, 'Finite FD inputs required')
    actual_direction = (arrays[1]-arrays[2])/(2*h)
    analytic = float(np.sum(arrays[0]*actual_direction))
    numeric = float((loss_plus-loss_minus)/(2*h))
    tolerance = float(SPEC['absolute_tolerance']+SPEC['relative_tolerance']*abs(analytic))
    measurable = bool(abs(analytic) > 10*SPEC['signal_floor'])
    return {'h': float(h), 'actual_direction': actual_direction.tolist(), 'actual_dot': analytic,
            'finite_difference': numeric, 'absolute_error': float(abs(analytic-numeric)),
            'tolerance': tolerance, 'measurable': measurable,
            'passed': bool(measurable and abs(analytic-numeric) <= tolerance)}


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk run required')
    require(sha(BINARY) == BINARY_SHA, 'Fixed isolated near-.01 binary required')
    snapshot = output/'source_snapshot'; (snapshot/'bridge_rgs').mkdir(parents=True)
    (snapshot/'bridge_rgs/__init__.py').write_text('')
    for source, target in ((ROOT/'src/bridge_rgs/ibgs_antialias.py', snapshot/'bridge_rgs/ibgs_antialias.py'),
                           (Path(__file__), snapshot/Path(__file__).name),
                           (ROOT/'tests/test_ibgs_aa_gradient.py', snapshot/'test_ibgs_aa_gradient.py'),
                           (ROOT/'docs/ibgs_antialias_contract.md', snapshot/'ibgs_antialias_contract.md')):
        shutil.copy2(source, target)
    bindings = {str(BINARY): BINARY_SHA, str(PORT/'diff_plane_rasterization/__init__.py'): sha(PORT/'diff_plane_rasterization/__init__.py')}
    plan = {'specification': SPEC, 'output': str(output), 'source_snapshot': str(snapshot),
            'source_hashes': tree(snapshot), 'backend_python': str(PORT), 'installed_sources': bindings,
            'binary': {'path': str(BINARY), 'sha256': BINARY_SHA},
            'input_hashes': {str(ROOT/'uv.lock'): sha(ROOT/'uv.lock')},
            'prepare_model_loads': 0, 'prepare_data_reads': 0}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC and Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name,
            'Frozen worker/specification required')
    require(tree(plan['source_snapshot']) == plan['source_hashes'] and
            all(sha(p) == h for p, h in {**plan['installed_sources'], **plan['input_hashes']}.items()), 'Bound source/binary changed')


def run(plan, report):
    import torch
    from diff_plane_rasterization import _C, GaussianRasterizationSettings, GaussianRasterizer

    from bridge_rgs.ibgs_antialias import aa_opacity_rasterizer
    require(sha(_C.__file__) == BINARY_SHA and str(Path(_C.__file__).resolve()) == str(BINARY), 'Wrong actual binary')
    report['actual_binary'] = {'path': _C.__file__, 'sha256': sha(_C.__file__)}
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark, torch.get_float32_matmul_precision(), torch.get_num_threads())
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest'); torch.set_num_threads(4)
    original = GaussianRasterizer.forward
    try:
        torch.cuda.reset_peak_memory_stats()
        device = 'cuda'; width, height, focal = 64, 48, 60.
        identity = torch.eye(4, device=device)
        projection = torch.zeros(4, 4, device=device)
        projection[0, 0], projection[1, 1] = 2*focal/width, 2*focal/height
        projection[2, 2], projection[2, 3], projection[3, 2] = 100./99.99, -1./99.99, 1.
        settings = GaussianRasterizationSettings(image_height=height, image_width=width,
            tanfovx=width/(2*focal), tanfovy=height/(2*focal),
            bg=torch.tensor(SPEC['background'], device=device), scale_modifier=1., viewmatrix=identity,
            projmatrix=projection.T.contiguous(), ref_to_src_list=identity[None],
            src_cam_pos=torch.zeros(1, 3, device=device), src_images=torch.zeros(1, 3, height, width, device=device),
            src_rendered_depths=torch.zeros(1, 1, height, width, device=device), nb_src_images=1,
            buffer_length=4, depth_error_threshold=.01, sh_degree=0, campos=torch.zeros(3, device=device),
            prefiltered=False, render_geo=False, render_depth_only=False, debug=False)
        raster = GaussianRasterizer(settings)
        weights = torch.tensor([1., .7, -.2], device=device, dtype=torch.float32)
        arrays = {}; report['positive_case'] = []
        def inputs(case):
            return {'means': torch.tensor([case['means']], device=device),
                    'log_scales': torch.tensor([case['scales']], device=device).log(),
                    'raw_quaternion': torch.tensor([case['raw_quaternion']], device=device),
                    'opacity_logits': (torch.tensor([[case['opacity_logit']]], device=device) if 'opacity_logit' in case else
                                       torch.logit(torch.tensor([[case['opacity']]], device=device)))}
        def forward(values, label, *, zero_scales=False):
            report['forward_calls'] += 1
            means = values['means']
            outputs = raster(means3D=means, means2D=torch.zeros_like(means), means2D_abs=torch.zeros_like(means),
                colors_precomp=torch.tensor([SPEC['colors']], device=device),
                opacities=values['opacity_logits'].sigmoid(),
                scales=(torch.zeros_like(values['log_scales']) if zero_scales else values['log_scales'].exp()),
                rotations=torch.nn.functional.normalize(values['raw_quaternion'], dim=-1),
                all_map=torch.tensor([[0., 0., -1., 1., 2.]], device=device))
            require(bool(torch.isfinite(outputs[0]).all()), 'Nonfinite RGB')
            arrays[label] = outputs[0].detach().cpu().numpy()
            # Cast the already-fixed FP32 upstream to FP64 so its backward cast is unchanged.
            return (outputs[0][:, 24, 32].double()*weights.double()).sum(), outputs
        with aa_opacity_rasterizer(GaussianRasterizer, capture_last=True) as hook:
            positive = {k: v.requires_grad_() for k, v in inputs(SPEC['positive_case']).items()}
            loss, _ = forward(positive, 'positive_base')
            gradients = torch.autograd.grad(loss, tuple(positive.values())); report['backward_calls'] += 1
            require(all(bool(torch.isfinite(g).all()) for g in gradients), 'Nonfinite positive-case gradient')
            report['positive_rho'] = hook.last.rho.detach().cpu().tolist()
            report['positive_effective_opacity'] = hook.last.opacity.detach().cpu().tolist()
            require(float(hook.last.opacity.max()) < .99, 'Unsaturated fixture became cap-active')
            for (name, base), gradient in zip(positive.items(), gradients, strict=True):
                direction = torch.tensor([SPEC['directions'][name]], device=device)
                row = {'parameter': name, 'gradient': gradient.detach().cpu().tolist(), 'finite_differences': []}
                for h in [SPEC['primary_h'], *SPEC['sensitivity_h']]:
                    plus, minus = ({k: v.detach().clone() for k, v in positive.items()} for _ in range(2))
                    plus[name] = base.detach()+h*direction; minus[name] = base.detach()-h*direction
                    with torch.no_grad():
                        lp, _ = forward(plus, f'{name}_{h}_plus'); lm, _ = forward(minus, f'{name}_{h}_minus')
                    row['finite_differences'].append(fd_report(gradient.detach().cpu().numpy(),
                        plus[name].cpu().numpy(), minus[name].cpu().numpy(), float(lp), float(lm), h))
                report['positive_case'].append(row)
            cap = inputs(SPEC['cap_case']); cap['opacity_logits'].requires_grad_()
            loss, _ = forward(cap, 'cap_base')
            cap_gradient, = torch.autograd.grad(loss, cap['opacity_logits']); report['backward_calls'] += 1
            require(bool(torch.isfinite(cap_gradient).all()), 'Nonfinite cap-case gradient')
            report['cap_effective_opacity'] = hook.last.opacity.detach().cpu().tolist()
            require(float(hook.last.opacity.min()) > .99, 'Fixed cap control is not cap-active')
            report['cap_case'] = []
            for h in [SPEC['primary_h'], *SPEC['sensitivity_h']]:
                plus, minus = ({k: v.detach().clone() for k, v in cap.items()} for _ in range(2))
                plus['opacity_logits'] += h; minus['opacity_logits'] -= h
                with torch.no_grad():
                    lp, _ = forward(plus, f'cap_{h}_plus'); lm, _ = forward(minus, f'cap_{h}_minus')
                report['cap_case'].append(fd_report(cap_gradient.detach().cpu().numpy(),
                    plus['opacity_logits'].cpu().numpy(), minus['opacity_logits'].cpu().numpy(), float(lp), float(lm), h))
            zero = inputs(SPEC['positive_case']); zero['opacity_logits'].requires_grad_()
            loss, outputs = forward(zero, 'zero_rho', zero_scales=True)
            gradient, = torch.autograd.grad(loss, zero['opacity_logits']); report['backward_calls'] += 1
            report['zero_control'] = {'rho_zero': bool((hook.last.rho == 0).all()),
                'finite_zero_opacity_gradient': bool(torch.isfinite(gradient).all() and (gradient == 0).all()),
                'background_exact': bool(torch.equal(outputs[0], settings.bg[:, None, None].expand_as(outputs[0])))}
            report['hook_calls'] = hook.calls
        torch.cuda.synchronize()
        path = Path(plan['output'])/'forward_arrays.npz'
        with path.open('xb') as stream:
            np.savez(stream, **arrays)
        report['arrays'] = {'path': str(path), 'sha256': sha(path)}
        report['primary_unsaturated_passed'] = all(r['finite_differences'][0]['passed'] for r in report['positive_case'])
        report['cap_derivative_mismatch_observed'] = bool(any(not row['passed'] for row in report['cap_case']))
        report['numerical_status'] = ('passed' if report['primary_unsaturated_passed']
                                     and all(report['zero_control'].values()) else 'failed')
        require(report['forward_calls'] == report['hook_calls'] == 33 and report['backward_calls'] == 3, 'Fixed counts changed')
        report['peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
    finally:
        report['hook_restored'] = GaussianRasterizer.forward is original
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = previous[:3]
        torch.set_float32_matmul_precision(previous[3]); torch.set_num_threads(previous[4])
        report['numeric_flags_restored'] = previous == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark, torch.get_float32_matmul_precision(), torch.get_num_threads())
        require(report['hook_restored'] and report['numeric_flags_restored'], 'Restoration failed')


def execute(path, expected):
    require(sha(path) == expected, 'Plan changed')
    plan = json.loads(Path(path).read_text()); output = Path(plan['output'])
    require(not any((output/name).exists() for name in ('execution_started.json', 'execution_receipt.json')), 'One attempt only')
    write(output/'execution_started.json', {'plan_sha256': expected})
    report = {'status': 'running', 'plan_sha256': expected, 'forward_calls': 0, 'backward_calls': 0,
              'optimizer_steps': 0, 'data_reads': 0, 'scope': SPEC['scope']}
    def expired(*_):
        raise TimeoutError('Fixed synthetic AA derivative deadline exhausted')
    old_handler = signal.signal(signal.SIGALRM, expired); signal.alarm(SPEC['internal_seconds'])
    started = time.monotonic()
    try:
        verify(plan)
        sys.path.insert(0, plan['source_snapshot']); sys.path.insert(0, plan['backend_python'])
        require(not any(k.startswith(('bridge_rgs', 'diff_plane_rasterization')) for k in sys.modules), 'No preloaded live module')
        run(plan, report); verify(plan)
        report.update(status='completed', sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
        report['elapsed_seconds'] = time.monotonic()-started
        write(output/'execution_receipt.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path); action.add_argument('--execute', type=Path)
    parser.add_argument('--expected-plan-sha256'); args = parser.parse_args()
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode')
    prepare(args.prepare) if args.prepare else execute(args.execute, args.expected_plan_sha256)
