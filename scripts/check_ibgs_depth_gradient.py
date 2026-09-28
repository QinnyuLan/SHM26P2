"""Fixed synthetic median-depth finite differences for the external IBGS port.

Checks that changing the number of valid source images cannot multiply or erase
the gradient of the target median-depth output. No training or real data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(report):
    import torch
    from diff_plane_rasterization import _C, GaussianRasterizationSettings, GaussianRasterizer

    torch.set_num_threads(4)
    torch.manual_seed(20260927)
    torch.backends.cuda.matmul.allow_tf32 = False
    width, height, focal, count = 64, 48, 60., 25
    device = 'cuda'
    yy, xx = torch.meshgrid(torch.linspace(-.4, .4, 5, device=device),
                           torch.linspace(-.5, .5, 5, device=device), indexing='ij')
    means = torch.stack((xx.flatten(), yy.flatten(),
                         2.+.01*xx.flatten()+.02*yy.flatten()), -1)
    scales = torch.full((count, 3), .25, device=device)
    rotations = torch.tensor([1., 0., 0., 0.], device=device).repeat(count, 1)
    colors = torch.tensor([.3, .4, .5], device=device).repeat(count, 1)
    opacity = torch.full((count, 1), .5, device=device)
    plane = torch.tensor([0., 0., -1., 1., 2.], device=device).repeat(count, 1)
    identity = torch.eye(4, device=device)
    projection = torch.zeros((4, 4), device=device)
    projection[0, 0], projection[1, 1] = 2*focal/width, 2*focal/height
    projection[2, 2], projection[2, 3] = 100./99.99, -1./99.99
    projection[3, 2] = 1
    directions = {'all_distances': torch.ones(count, device=device),
                  'alternating_distances': .4+torch.arange(count, device=device).remainder(2)*.6}
    results = []
    report.update(binary_path=_C.__file__, binary_sha256=sha(_C.__file__),
                  forward_calls=0, backward_calls=0, cases=results,
                  fixed_h=[.002, .004, .008], absolute_tolerance=1e-4,
                  relative_tolerance=.005, dataset_reads=0, optimizer_steps=0)
    for valid_sources in (0, 1, 4):
        source_count = max(valid_sources, 1)
        settings = GaussianRasterizationSettings(
            image_height=height, image_width=width, tanfovx=width/(2*focal),
            tanfovy=height/(2*focal), bg=torch.zeros(3, device=device), scale_modifier=1.,
            viewmatrix=identity, projmatrix=projection.T.contiguous(),
            ref_to_src_list=identity[None].repeat(source_count, 1, 1),
            src_cam_pos=torch.zeros(source_count, 3, device=device),
            src_images=torch.full((source_count, 3, height, width), .5, device=device),
            src_rendered_depths=torch.full((source_count, 1, height, width),
                                          2. if valid_sources else 0., device=device),
            nb_src_images=source_count, buffer_length=4, depth_error_threshold=.01,
            sh_degree=0, campos=torch.zeros(3, device=device), prefiltered=False,
            render_geo=True, render_depth_only=False, debug=False)
        raster = GaussianRasterizer(settings)

        def forward(maps, raster=raster):
            report['forward_calls'] += 1
            return raster(means3D=means, means2D=torch.zeros_like(means),
                          means2D_abs=torch.zeros_like(means), colors_precomp=colors,
                          opacities=opacity, scales=scales, rotations=rotations, all_map=maps)

        maps = plane.clone().requires_grad_()
        outputs = forward(maps)
        value = outputs[3][0, 8:-8, 8:-8].mean()
        gradient = torch.autograd.grad(value, maps)[0]
        report['backward_calls'] += 1
        actual_support = (outputs[4].reshape(-1, 4, height, width).sum(1)[:, 8:-8, 8:-8] > 0).sum(0)
        case = {'valid_sources': valid_sources, 'value': value.item(),
                'support_min': actual_support.min().item(),
                'support_max': actual_support.max().item(), 'directions': []}
        results.append(case)
        if not (actual_support == valid_sources).all() or not torch.isfinite(gradient).all():
            raise RuntimeError('Wrong support or nonfinite analytic gradient')
        for name, direction in directions.items():
            analytic = (gradient[:, 4]*direction).sum().item()
            row = {'name': name, 'analytic': analytic, 'finite_differences': []}
            case['directions'].append(row)
            for h in report['fixed_h']:
                with torch.no_grad():
                    plus, minus = plane.clone(), plane.clone()
                    plus[:, 4] += h*direction
                    minus[:, 4] -= h*direction
                    numeric = ((forward(plus)[3][0, 8:-8, 8:-8].mean()
                                - forward(minus)[3][0, 8:-8, 8:-8].mean())/(2*h)).item()
                error = abs(numeric-analytic)
                threshold = 1e-4+.005*abs(numeric)
                row['finite_differences'].append({'h': h, 'numeric': numeric,
                                                  'absolute_error': error, 'passed': error <= threshold})
    torch.cuda.synchronize()
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['all_passed'] = all(fd['passed'] for c in results for r in c['directions']
                               for fd in r['finite_differences'])
    if not report['all_passed']:
        raise RuntimeError('Fixed median-depth finite-difference contract failed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to('/mnt/data'):
        raise ValueError('Fresh output on data disk required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def timeout_handler(*_):
        raise TimeoutError('Synthetic gradient check exceeded 90 seconds')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(90)
    start = time.monotonic()
    result = {'status': 'failed', 'entry_sha256': sha(__file__),
              'scope': 'target median-depth derivatives only; not RGB/texture/entire backend gradient certification'}
    try:
        check(result)
        result['status'] = 'passed'
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['wall_seconds'] = time.monotonic()-start
        with args.output.open('x') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
        signal.alarm(0)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
