"""Fixed CPU-only bright/smooth SSIM cancellation diagnostic; no image inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from bridge_rgs.raw_grid import appearance_rgb_loss

SPEC = {'protocol': 'fixed_bright_smooth_complete_ssim_cpu_precision_v1', 'shape': [64, 80, 3],
        'grid': 'FP32 linspace[0,1], meshgrid indexing ij',
        'base': 'z=.7+.01*x+.006*y; channels[z,z+.015,z-.01]',
        'target': 'base+.01+.001*sin(9*x)', 'direction': 'constant .01 in all RGB entries',
        'epsilons': [.001, .0005, .00025], 'endpoints': 'always constructed in FP32 then cast to objective dtype',
        'objective': '.8 L1 + .2 complete-7-window (1-SSIM), valid allTrue, no clamp',
        'threads': 8, 'device': 'cpu'}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fixture():
    y, x = torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 80), indexing='ij')
    z = .7+.01*x+.006*y
    base = torch.stack([z, z+.015, z-.01], dim=-1)
    target = (base+.01+.001*torch.sin(x*9)[..., None]).contiguous()
    return base, target, torch.ones_like(base)*.01, torch.ones((64, 80), dtype=torch.bool)


def moments(x):
    x = x.permute(2, 0, 1)[None].contiguous()
    mean = F.avg_pool2d(x, 7, 1, 3)
    return (F.avg_pool2d(x*x, 7, 1, 3)-mean.square())[..., 3:-3, 3:-3]


def audit():
    torch.set_num_threads(8)
    base, target, direction, valid = fixture()
    records = {}
    for dtype in (torch.float32, torch.float64):
        leaf = base.to(dtype).detach().requires_grad_(True)
        loss, parts = appearance_rgb_loss(leaf, target.to(dtype), valid)
        gradient, = torch.autograd.grad(loss, leaf)
        rows = []
        for eps in SPEC['epsilons']:
            positive, negative = base+eps*direction, base-eps*direction
            plus, _ = appearance_rgb_loss(positive.to(dtype), target.to(dtype), valid)
            minus, _ = appearance_rgb_loss(negative.to(dtype), target.to(dtype), valid)
            analytic = float((gradient.double()*(positive.double()-negative.double())/(2*eps)).sum())
            fd = (float(plus)-float(minus))/(2*eps)
            rows.append({'epsilon': eps, 'loss_plus': float(plus), 'loss_minus': float(minus),
                         'realized_analytic': analytic, 'central_difference': fd,
                         'relative_error': abs(fd-analytic)/max(abs(analytic), 1e-30)})
        records[str(dtype)] = {'loss': float(loss.detach()), 'components': {k: float(v) for k, v in parts.items()},
                               'comparisons': rows}
    v32, v64 = moments(base), moments(base.double())
    return {'status': 'completed_cpu_only', 'specification': SPEC, 'results': records,
            'complete_window_variance': {'fp32_min': float(v32.min()), 'fp64_min': float(v64.min()),
             'fp32_negative_entries': int((v32 < 0).sum()), 'fp64_negative_entries': int((v64 < 0).sum()),
             'fp32_fp64_max_abs_error': float((v32.double()-v64).abs().max()),
             'fp64_median': float(v64.median())},
            'cuda_initialized': torch.cuda.is_initialized(), 'real_images_read': 0,
            'interpretation': 'Fixed synthetic evidence about loss precision only; not proof of the cause of H3 per-view FD failure and not a production loss change.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve any prior CPU result')
    result = audit()
    from bridge_rgs import losses, raw_grid
    result['source_hashes'] = {str(Path(p).resolve()): digest(p) for p in [__file__, losses.__file__, raw_grid.__file__]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'output': str(args.output), 'sha256': digest(args.output), 'results': result['results']}))


if __name__ == '__main__':
    main()
