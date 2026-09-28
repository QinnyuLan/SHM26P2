"""Synthetic IBGS extension smoke test; no dataset, checkpoint, or optimizer.

Run with the dedicated uv interpreter after compiling the author's extensions.
The observed ray convention is reported, not silently corrected in this test.
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
from types import SimpleNamespace


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(repository, diagnostics, output):
    import numpy as np
    import torch
    from diff_plane_rasterization import (
        _C,
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    from simple_knn import _C as knn_binary
    from simple_knn._C import distCUDA2

    sys.path.insert(0, str(repository))
    from color_aggregation_network import ColorFusionResidualNet, fuse_color

    torch.set_num_threads(4)
    torch.manual_seed(20260927)
    torch.cuda.manual_seed_all(20260927)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = "cuda"
    width, height, focal, count, sources = 64, 48, 60.0, 25, 4
    # Slightly varying positive z avoids degenerate KNN coordinate bounds.
    yy, xx = torch.meshgrid(
        torch.linspace(-0.4, 0.4, 5, device=device),
        torch.linspace(-0.5, 0.5, 5, device=device),
        indexing="ij",
    )
    means = torch.stack(
        (xx.flatten(), yy.flatten(), 2.0 + 0.01 * xx.flatten() + 0.02 * yy.flatten()), -1
    )
    means.requires_grad_(True)
    actual_knn = distCUDA2(means.detach())
    d2 = (means.detach()[:, None] - means.detach()[None]).square().sum(-1)
    d2.fill_diagonal_(float("inf"))
    reference_knn = d2.topk(3, largest=False).values.mean(-1)
    knn_error = (actual_knn - reference_knn).abs().max().item()
    require(
        torch.allclose(actual_knn, reference_knn, atol=1e-6, rtol=1e-5),
        "simple_knn does not match brute-force 3-NN",
    )

    # A front-facing plane at z=2. Shapes follow the public raster API.
    maps = torch.tensor([0.0, 0.0, -1.0, 1.0, 2.0], device=device).repeat(count, 1)
    maps.requires_grad_(True)
    scales = torch.full((count, 3), 0.25, device=device, requires_grad=True)
    rotations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(count, 1)
    rotations.requires_grad_(True)
    colors = torch.tensor([0.3, 0.4, 0.5], device=device).repeat(count, 1)
    colors.requires_grad_(True)
    opacity = torch.full((count, 1), 0.5, device=device, requires_grad=True)
    screen = torch.zeros_like(means, requires_grad=True)
    screen_abs = torch.zeros_like(means, requires_grad=True)
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
    )
    image = torch.stack(
        (0.1 + 0.7 * xs / width, 0.1 + 0.7 * ys / height, 0.2 + 0.2 * xs * ys / (width * height))
    ).float()
    source_rgb = image[None].repeat(sources, 1, 1, 1).contiguous()
    source_depth = torch.full((sources, 1, height, width), 2.0, device=device)
    identity = torch.eye(4, device=device)
    projection = torch.zeros((4, 4), device=device)
    projection[0, 0] = 2 * focal / width
    projection[1, 1] = 2 * focal / height
    projection[2, 2] = 100.0 / (100.0 - 0.01)
    projection[2, 3] = -100.0 * 0.01 / (100.0 - 0.01)
    projection[3, 2] = 1.0
    common = {
        "image_height": height,
        "image_width": width,
        "tanfovx": width / (2 * focal),
        "tanfovy": height / (2 * focal),
        "bg": torch.zeros(3, device=device),
        "scale_modifier": 1.0,
        "viewmatrix": identity,
        "projmatrix": projection.T.contiguous(),
        "ref_to_src_list": identity[None].repeat(sources, 1, 1).contiguous(),
        "src_cam_pos": torch.zeros(sources, 3, device=device),
        "src_images": source_rgb,
        "src_rendered_depths": source_depth,
        "nb_src_images": sources,
        "buffer_length": 4,
        "depth_error_threshold": 0.01,
        "sh_degree": 0,
        "campos": torch.zeros(3, device=device),
        "prefiltered": False,
        "debug": False,
    }
    kwargs = {
        "means3D": means,
        "means2D": screen,
        "means2D_abs": screen_abs,
        "colors_precomp": colors,
        "opacities": opacity,
        "scales": scales,
        "rotations": rotations,
        "all_map": maps,
    }
    torch.cuda.synchronize()
    start = time.monotonic()
    settings = GaussianRasterizationSettings(**common, render_geo=True, render_depth_only=False)
    outputs = GaussianRasterizer(settings)(**kwargs)
    names = (
        "render",
        "radii",
        "rendered_normal",
        "median_intersected_depth",
        "cam_feat",
        "warped_image",
        "min_depth_diff",
        "camera_ray",
        "use_first_src_frame_mask",
    )
    rendered = dict(zip(names, outputs, strict=True))
    require(all(torch.isfinite(value).all() for value in outputs), "Non-finite forward outputs")
    require(rendered["render"].shape == (3, height, width), "Wrong RGB shape")
    require(torch.count_nonzero(rendered["radii"]).item() == count, "Invisible synthetic means")
    support = rendered["cam_feat"].view(-1, 4, height, width)[:sources].sum(1) > 0
    interior = support[:, 8:-8, 8:-8]
    require(interior.float().mean().item() > 0.9, "Insufficient synthetic source support")
    warped = rendered["warped_image"].view(-1, 3, height, width)[:sources]
    warp_error = ((warped - source_rgb).abs() * support[:, None]).max().item()
    array_path = output.parent/'forward.npz'
    with array_path.open('xb') as stream:
        np.savez(stream, warped=warped.detach().cpu().numpy(),
                 source_rgb=source_rgb.cpu().numpy(), support=support.cpu().numpy(),
                 **{name: value.detach().cpu().numpy() for name, value in rendered.items()})
    diagnostics.update(forward_arrays={'path': str(array_path), 'sha256': sha(array_path)},
                       identity_warp_max_error=warp_error,
                       interior_support_fraction=interior.float().mean().item())
    require(warp_error < 1e-4, f"Identity-view source texture transport failed: {warp_error}")
    depth = rendered["median_intersected_depth"]
    depth_error = (depth[0][support.any(0)] - 2.0).abs().max().item()
    require(depth_error < 1e-4, "Ray-plane depth differs from z=2")

    ray = rendered["camera_ray"].reshape(3, height, width)
    ray_errors = {}
    for label, shift in (
        ("upstream_integer_minus_W_over_2", 0.0),
        ("corner_v2_pixel_centers", 0.5),
    ):
        expected_ray = torch.stack(
            (
                (xs + shift - width / 2) / focal,
                (ys + shift - height / 2) / focal,
                torch.ones_like(xs),
            ),
            0,
        ).float()
        expected_ray = torch.nn.functional.normalize(expected_ray, dim=0)
        ray_errors[label] = (ray - expected_ray)[:, support.any(0)].abs().max().item()

    # Exercise the author's fusion network and backward through the extension.
    net = ColorFusionResidualNet(height=height, width=width).to(device)
    options = SimpleNamespace(
        enable_exposure_correction=False, nb_visible_src_frames=3, residual_resolution_scale=1.0
    )
    fusion = fuse_color(rendered, net, None, None, None, 0, options)
    require(fusion is not None, "Fusion network received no valid sources")
    loss = fusion["image_pred"].square().mean() + 0.01 * depth.mean()
    loss.backward()
    gradients = {}
    for name, parameter in {
        "means": means,
        "colors": colors,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
        "plane": maps,
    }.items():
        require(
            parameter.grad is not None and torch.isfinite(parameter.grad).all(),
            "Missing/non-finite gradient: " + name,
        )
        gradients[name] = {
            "l2": parameter.grad.norm().item(),
            "max_abs": parameter.grad.abs().max().item(),
        }
    require(
        all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()),
        "Missing/non-finite network gradient",
    )
    require(
        gradients["means"]["l2"] > 0 and gradients["plane"]["l2"] > 0,
        "Degenerate geometry gradient",
    )
    torch.cuda.synchronize()
    main_seconds = time.monotonic() - start

    with torch.no_grad():
        depth_settings = GaussianRasterizationSettings(
            **common, render_geo=False, render_depth_only=True
        )
        depth_only = GaussianRasterizer(depth_settings)(**kwargs)[3]
        require(torch.isfinite(depth_only).all(), "Non-finite depth-only outputs")
        depth_only_difference = (depth_only - depth).abs().max().item()
        require(depth_only_difference < 1e-4, "Depth-only/full geometry differ on one plane")
    torch.cuda.synchronize()
    return {
        "status": "passed",
        "scope": "synthetic API/finite-gradient check, not gradient correctness or real-scene quality",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "pixel_size": [width, height],
        "gaussians": count,
        "forward_calls": 2,
        "backward_calls": 1,
        "optimizer_steps": 0,
        "dataset_reads": 0,
        "checkpoint_loads": 0,
        "simple_knn_max_error": knn_error,
        "source_support_pixels": support.sum().item(),
        "identity_warp_max_error": warp_error,
        "plane_depth_max_error": depth_error,
        "depth_only_max_difference": depth_only_difference,
        "ray_convention_max_errors": ray_errors,
        "loss": loss.item(),
        "gradients": gradients,
        "output_shapes": {name: list(value.shape) for name, value in rendered.items()},
        "fusion_parameters": sum(p.numel() for p in net.parameters()),
        "main_forward_backward_seconds": main_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "binaries": {str(p): sha(p) for p in (_C.__file__, knn_binary.__file__)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Never overwrite a smoke receipt")
    require(args.output.resolve().is_relative_to("/mnt/data"), "Use data disk")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def timeout_handler(*_):
        raise TimeoutError("Synthetic test exceeded 90 seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(90)
    started = time.monotonic()
    result = {"status": "failed", "entry_sha256": sha(__file__), "pid": os.getpid()}
    try:
        result.update(run(args.repository.resolve(), result, args.output))
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        signal.alarm(0)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
