"""Synthetic CPU/CUDA pooling-layout derivative localization; no scene or dataset."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F


def objective(x, target, weights, operator, layout):
    x, target = x.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
    if layout == 'nchw':
        x, target = x.contiguous(), target.contiguous()
    pool = lambda z: F.avg_pool2d(z, 7, stride=1, padding=3)
    mx = pool(x)
    if operator == 'pool':
        return (mx[0].permute(1, 2, 0)*weights).mean()
    my = pool(target)
    vx, vy = pool(x*x)-mx.square(), pool(target*target)-my.square()
    cov = pool(x*target)-mx*my
    ssim = ((2*mx*my+.01**2)*(2*cov+.03**2) /
            ((mx.square()+my.square()+.01**2)*(vx+vy+.03**2)))
    return (1-ssim[0, :, 3:-3, 3:-3]).mean()


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Do not overwrite an existing diagnostic')
    torch.set_num_threads(8)
    started = time.monotonic()
    report = {'status': 'running', 'dataset_pixels': 0, 'scene_renders': 0,
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'dtype': 'float64', 'epsilon': 1e-5, 'rows': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for height, width in ((37, 53), (989, 1320)):
            generator = torch.Generator().manual_seed(20260927+height)
            base = .2+.6*torch.rand((height, width, 3), generator=generator, dtype=torch.float64)
            target = (base+.025*torch.randn(base.shape, generator=generator, dtype=torch.float64)).clamp(.01, .99)
            weights = torch.randn(base.shape, generator=generator, dtype=torch.float64)
            for operator in ('pool', 'ssim'):
                reference_input = base.clone().requires_grad_(True)
                reference_loss = objective(reference_input, target, weights, operator, 'nchw')
                reference_grad = torch.autograd.grad(reference_loss, reference_input)[0].detach()
                direction = reference_grad/reference_grad.abs().max()
                for device, layout in (('cpu', 'nchw'), ('cpu', 'nhwc'), ('cuda', 'nchw'), ('cuda', 'nhwc')):
                    x = base.to(device).detach().clone().requires_grad_(True)
                    y, w, d = target.to(device), weights.to(device), direction.to(device)
                    loss = objective(x, y, w, operator, layout)
                    gradient = torch.autograd.grad(loss, x)[0].detach()
                    with torch.no_grad():
                        plus = objective(x+1e-5*d, y, w, operator, layout)
                        minus = objective(x-1e-5*d, y, w, operator, layout)
                    analytic = float((gradient*d).sum())
                    difference = (float(plus)-float(minus))/(2e-5)
                    gradient_cpu = gradient.cpu()
                    row = {'shape': [height, width, 3], 'operator': operator,
                           'device': device, 'layout': layout, 'loss': float(loss.detach()),
                           'analytic': analytic, 'finite_difference': difference,
                           'relative_fd_discrepancy': abs(analytic-difference)/max(abs(analytic), 1e-30),
                           'gradient_relative_l2_vs_cpu_nchw': float((gradient_cpu-reference_grad).norm()/reference_grad.norm()),
                           'gradient_absmax_error_vs_cpu_nchw': float((gradient_cpu-reference_grad).abs().max())}
                    report['rows'].append(row)
                    print(json.dumps(row), flush=True)
                    del x, loss, gradient, gradient_cpu
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        with output.open('x') as handle:
            json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
