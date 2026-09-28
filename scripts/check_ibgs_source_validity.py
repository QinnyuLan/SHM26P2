"""Fixed two-plane IBGS source-support and pixel-center GPU contract.

Synthetic only. Run once with the dedicated IBGS uv environment after the port
is compiled. This checks forward invariance, not the correctness of a VJP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

SPEC = {
    'protocol': 'ibgs_source_validity_v1',
    'width': 64, 'height': 48, 'focal': 60.0,
    'plane_depths': [2.0, 4.0], 'source_translation_x': 0.4,
    'opacity': [0.4, 0.8], 'scales': [[0.5, 0.5, 0.15], [1.0, 1.0, 0.3]],
    'buffer_length': 4, 'source_depth': 3.0, 'depth_error_threshold': 10.0,
    'depth_threshold_scope': 'wide isolated-support test; not a training visibility threshold',
    'invalid_rgb_values': [10000.0, -10000.0],
    'forward_calls': 2, 'backward_calls': 0, 'optimizer_steps': 0,
    'ray_max_absolute_error': 2e-6, 'output_max_absolute_difference': 0.0,
    'internal_seconds': 90, 'external_seconds': 120,
}
NAMES = ('render', 'radii', 'rendered_normal', 'median_intersected_depth',
         'cam_feat', 'warped_image', 'min_depth_diff', 'camera_ray',
         'use_first_src_frame_mask')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def fixture():
    """Independent FP64 pinhole/alpha proof; no extension calls or image reads."""
    h, w, f = SPEC['height'], SPEC['width'], SPEC['focal']
    yy, xx = np.mgrid[:h, :w]
    valid = np.ones((h, w), dtype=bool)
    valid[:2] = valid[-2:] = False
    valid[:, :2] = valid[:, -2:] = False
    valid[20:28, 44:48] = False
    # Exact radius-one square erosion, zero padding; no project helper import.
    padded = np.pad(valid, 1, constant_values=False)
    eroded = np.logical_and.reduce([padded[dy:dy+h, dx:dx+w]
                                   for dy in range(3) for dx in range(3)])
    ray = np.stack(((xx + .5-w/2)/f, (yy + .5-h/2)/f, np.ones_like(xx)), -1)
    uv = np.stack([np.stack((xx + f*.4/z, yy), -1) for z in (2., 4.)])
    # Both projected Gaussian covariances equal diag(225+.3).
    exponent = -.5*((xx-(w-1)/2)**2+(yy-(h-1)/2)**2)/(15.**2+.3)
    alpha = np.asarray(SPEC['opacity'])[:, None, None]*np.exp(exponent)[None]
    weights = np.stack((alpha[0], (1-alpha[0])*alpha[1]))
    median = (weights*np.asarray([2., 4.])[:, None, None]).sum(0)/(weights.sum(0)+1e-8)
    median_uv = np.stack((xx+f*.4/median, yy), -1)

    def all_taps(mask, coordinates):
        x, y = coordinates[..., 0], coordinates[..., 1]
        inside = (x >= 0) & (y >= 0) & (x <= w-1) & (y <= h-1)
        x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        result = inside.copy()
        for dy in (0, 1):
            for dx in (0, 1):
                # Even zero-weight taps are required here: conservative witness.
                result &= mask[np.clip(y0+dy, 0, h-1), np.clip(x0+dx, 0, w-1)]
        return result

    two_depth = ((alpha >= 1/255).all(0) & (1-alpha[0] > .5)
                 & ((1-alpha[0])*(1-alpha[1]) >= .0001))
    safe = all_taps(eroded, uv[1]) & all_taps(valid, uv[1])
    bad_near = ~all_taps(valid, uv[0])
    median_safe = all_taps(eroded, median_uv)
    witness = two_depth & safe & bad_near & median_safe
    require(witness[24, 32], 'The fixed central query does not cross the source hole')
    require(witness[24, 50], 'The fixed boundary query does not cross source padding')
    image = np.stack((.1+.7*xx/w, .1+.7*yy/h, .2+.2*xx*yy/(w*h))).astype(np.float32)
    proof = {
        'candidate_mixed_validity_queries': int(witness.sum()),
        'central_query_xy': [32, 24], 'boundary_query_xy': [50, 24],
        'central_source_xy_by_depth': uv[:, 24, 32].tolist(),
        'central_median_source_xy': median_uv[24, 32].tolist(),
        'central_alpha_weights': weights[:, 24, 32].tolist(),
        'central_buffer_depths': [2.0, 4.0],
        'source_transform_nonidentity': True,
        'scope': 'independent analytical candidate/witness construction, not a saved CUDA buffer ledger',
    }
    return image, valid, eroded, ray, witness, proof


def run(repository, report, arrays):
    import torch
    from diff_plane_rasterization import _C, GaussianRasterizationSettings, GaussianRasterizer

    sys.path.insert(0, str(repository))
    from types import SimpleNamespace

    from color_aggregation_network import ColorFusionResidualNet, fuse_color

    image, valid, eroded, ray, witness, proof = fixture()
    arrays.update(valid=valid, eroded_source_depth_valid=eroded,
                  cpu_witness=witness, cpu_expected_ray=ray, source_valid_rgb=image)
    report['cpu_construction'] = proof
    report['binary'] = {'path': _C.__file__, 'sha256': sha(_C.__file__)}
    report['repository_sources'] = {str(p.relative_to(repository)): sha(p) for p in (
        repository/'submodules/diff-plane-rasterization/cuda_rasterizer/forward.cu',
        repository/'submodules/diff-plane-rasterization/cuda_rasterizer/backward.cu',
        repository/'submodules/diff-plane-rasterization/cuda_rasterizer/auxiliary.h',
        repository/'submodules/diff-plane-rasterization/cuda_rasterizer/rasterizer_impl.cu',
        repository/'color_aggregation_network.py')}
    torch.set_num_threads(4)
    torch.manual_seed(20260927)
    torch.cuda.manual_seed_all(20260927)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    device = 'cuda'
    w, h, f = SPEC['width'], SPEC['height'], SPEC['focal']
    means = torch.tensor([[0., 0., 2.], [0., 0., 4.]], device=device)
    maps = torch.tensor([[0., 0., -1., 1., 2.], [0., 0., -1., 1., 4.]], device=device)
    projection = torch.zeros((4, 4), device=device)
    projection[0, 0], projection[1, 1] = 2*f/w, 2*f/h
    projection[2, 2], projection[2, 3], projection[3, 2] = 100/99.99, -1/99.99, 1.
    transform = torch.eye(4, device=device)
    transform[0, 3] = SPEC['source_translation_x']
    common = {'image_height': h, 'image_width': w, 'tanfovx': w/(2*f), 'tanfovy': h/(2*f),
                  'bg': torch.zeros(3, device=device), 'scale_modifier': 1.,
                  'viewmatrix': torch.eye(4, device=device), 'projmatrix': projection.T.contiguous(),
                  'ref_to_src_list': transform[None].contiguous(),
                  'src_cam_pos': torch.tensor([[-.4, 0., 0.]], device=device),
                  'src_rendered_depths': torch.as_tensor((3.*eroded)[None, None], device=device,
                                                    dtype=torch.float32).contiguous(),
                  'nb_src_images': 1, 'buffer_length': 4, 'depth_error_threshold': 10.,
                  'sh_degree': 0, 'campos': torch.zeros(3, device=device), 'prefiltered': False,
                  'debug': False, 'render_geo': True, 'render_depth_only': False}
    kwargs = {'means3D': means, 'means2D': torch.zeros_like(means), 'means2D_abs': torch.zeros_like(means),
                  'colors_precomp': torch.tensor([[.3, .4, .5], [.6, .2, .1]], device=device),
                  'opacities': torch.tensor(SPEC['opacity'], device=device)[:, None],
                  'scales': torch.tensor(SPEC['scales'], device=device),
                  'rotations': torch.tensor([[1., 0., 0., 0.]]*2, device=device), 'all_map': maps}
    net = ColorFusionResidualNet(height=h, width=w).to(device).eval()
    options = SimpleNamespace(enable_exposure_correction=False, nb_visible_src_frames=1,
                              residual_resolution_scale=1.)
    records = []
    with torch.no_grad():
        for index, invalid_value in enumerate(SPEC['invalid_rgb_values']):
            source = image.copy()
            source[:, ~valid] = invalid_value
            settings = GaussianRasterizationSettings(**common,
                src_images=torch.as_tensor(source[None], device=device).contiguous())
            outputs = GaussianRasterizer(settings)(**kwargs)
            report['forward_calls'] += 1
            rendered = dict(zip(NAMES, outputs, strict=True))
            for name, value in rendered.items():
                arrays[f'case{index}_{name}'] = value.detach().cpu().numpy()
            require(all(torch.isfinite(t).all() for t in outputs), 'Nonfinite raster output')
            require(torch.count_nonzero(rendered['radii']).item() == 2, 'Both planes must be active')
            # Packed features contain camera displacement and a cosine; their sum is
            # not a generic support flag. This positive fixture uses explicit source0 mask.
            support = rendered['use_first_src_frame_mask'].reshape(h, w) > 0
            arrays[f'case{index}_support'] = support.cpu().numpy()
            require(support[24, 32].item() and support[24, 50].item(), 'Fixed mixed-depth witnesses lack support')
            require((support & torch.as_tensor(witness, device=device)).sum().item() >= 16,
                    'Insufficient nonempty mixed-depth/cross-invalid support')
            depth = rendered['median_intersected_depth'].reshape(h, w)
            require(2.05 < depth[24, 32].item() < 3.95, 'No actual two-depth mixture at central witness')
            observed_ray = rendered['camera_ray'].reshape(3, h, w).permute(1, 2, 0)
            expected = torch.as_tensor(ray, device=device, dtype=torch.float64)
            expected = expected/expected.norm(dim=-1, keepdim=True)
            ray_error = (observed_ray.double()-expected).abs()[support].max().item()
            require(ray_error <= SPEC['ray_max_absolute_error'], 'Patched ray disagrees with corner-v2 pixel centers')
            fusion = fuse_color(rendered, net, None, None, None, 0, options)
            require(fusion is not None, 'No source reached the fixed fusion network')
            arrays[f'case{index}_fused'] = fusion['image_pred'].cpu().numpy()
            require(torch.isfinite(fusion['image_pred']).all(), 'Nonfinite fused output')
            records.append({'invalid_rgb': invalid_value, 'support_pixels': int(support.sum().item()),
                            'witness_support_pixels': int((support & torch.as_tensor(witness, device=device)).sum().item()),
                            'central_depth': depth[24, 32].item(), 'ray_max_error': ray_error})
            report['records'] = records
    differences = {}
    for name in (*NAMES, 'support', 'fused'):
        a, b = arrays[f'case0_{name}'], arrays[f'case1_{name}']
        differences[name] = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
    report['max_output_differences'] = differences
    require(all(v == 0 for v in differences.values()), 'Invalid RGB affects a forward/fused output')
    torch.cuda.synchronize()
    report.update(status='passed', peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                  torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    array_path = output.with_suffix('.npz')
    require(output.is_relative_to('/mnt/data'), 'Artifacts belong on the data disk')
    require(not output.exists() and not array_path.exists(), 'Never overwrite a prior attempt')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'failed', 'specification': SPEC, 'entry_sha256': sha(__file__),
                  'pid': os.getpid(), 'forward_calls': 0, 'backward_calls': 0, 'optimizer_steps': 0,
                  'dataset_reads': 0, 'checkpoint_loads': 0, 'scope': 'fixed synthetic forward support; no VJP accuracy claim'}
    arrays = {}
    started = time.monotonic()

    def timed_out(*_):
        raise TimeoutError('Fixed source-validity check exceeded 90 seconds')

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(SPEC['internal_seconds'])
    try:
        run(args.repository.resolve(), report, arrays)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        if arrays:
            with array_path.open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            report['arrays'] = {'path': str(array_path), 'sha256': sha(array_path)}
        payload = json.dumps(report, indent=2, allow_nan=False)+'\n'
        with output.open('x') as stream:
            stream.write(payload)
        signal.alarm(0)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
