"""Thin frozen-field adapter for live IBGS buffers and a prebuilt selector.

No JIT/build, source photograph reads, source-depth refresh or field backward.
The caller supplies the separately bound old official renderer/backend imports.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .ibgs_warm_training import FIELD_KEYS, load_saved_field

ENDPOINT_SHA = 'e977dc1c5c676e930c47ed78e56a5f95c25c8127a5c02f23876ddb8b4e1aa3ee'
BACKEND_SHA = '436b2b55df2b37cddda59e606ec0855ad125eb211058b502197674bd2b93adaf'


def _require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_fixed_full(checkpoint, *, device='cuda'):
    """One old full endpoint; no untrusted q/feature conversion or warm reinit."""
    _require(_sha(checkpoint) == ENDPOINT_SHA, 'Fixed old full checkpoint required')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    _require(saved['protocol'] == 'ibgs_warm_matched_v1' and saved['arm'] == 'full'
             and saved['step'] == 6000 and saved['field']['_xyz'].shape == (996009, 3),
             'Old full endpoint identity mismatch')
    field, background = load_saved_field(saved, device=device)
    for key in FIELD_KEYS:
        getattr(field, key).requires_grad_(False)
    return field, background.detach()


def load_selector_extension(path, expected_sha256):
    """Load the completed check's exact JIT .so; never call a build function."""
    path = Path(path).resolve()
    _require(path.is_relative_to('/mnt/data') and path.is_file() and _sha(path) == expected_sha256,
             'Bound data-disk selector binary required')
    name = path.name.split('.')[0]
    _require(name.startswith('ibgs_layer_select_') and path.suffix == '.so', 'Unexpected extension name')
    if name in sys.modules:
        loaded = sys.modules[name]
        _require(Path(loaded.__file__).resolve() == path, 'Conflicting selector module already imported')
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _require(hasattr(module, 'select_layers'), 'Missing selector entry')
    sys.modules[name] = module
    return module


def render_layer_capture(camera, field, background, *, extension, backend=None,
                         render_fn=None, selector=None):
    """One original source-free raster, one selector, no retained geometry graph.

    camera must be a separate BridgeCamera with empty nearest lists, corner-v2
    centered intrinsics. Output raw/ray are CHW; each selection is HW4 with
    keys ids/depth/weights/ordinals. No historical buffer is uploaded or parsed
    with a new address. Returned selections own their CUDA output storage;
    original raster scratch buffers are released on return.

    Optional injected backend/render_fn/selector exist for CPU plumbing tests.
    Production callers bind these modules to the old snapshot/checked binary.
    """
    if backend is None:
        from diff_plane_rasterization import _C
        _require(_sha(_C.__file__) == BACKEND_SHA, 'Original non-AA backend required')
        backend = _C
    if render_fn is None:
        from gaussian_renderer import render
        render_fn = render
    if selector is None:
        from . import ibgs_layer_select
        selector = ibgs_layer_select
    _require(not camera.nearest_id and not camera.nearest_names, 'Separate empty-source camera required')
    w, h = camera.image_width, camera.image_height
    _require(camera.Cx == (w-1)*.5 and camera.Cy == (h-1)*.5, 'Fixed centered array-coordinate principal point required')
    _require(all(not getattr(field, key).requires_grad for key in FIELD_KEYS)
             and not background.requires_grad, 'Every field parameter and background must be frozen')
    original = backend.rasterize_gaussians
    captured = []
    def observe(*args):
        output = original(*args)
        _require(len(args) == 29 and isinstance(output, tuple) and len(output) == 13,
                 'Original IBGS ABI required')
        _require(not captured and args[15] == 1 and args[16] == 4
                 and args[26] is True and args[27] is False, 'One source-free fullgeo raster required')
        _require(all(int(torch.count_nonzero(args[k])) == 0 for k in (11, 12, 13, 14)),
                 'Source-free placeholders must remain zero')
        captured.append((args, output))  # Live storage, never detached host byte copies.
        return output
    backend.rasterize_gaussians = observe
    try:
        with torch.no_grad():
            package = render_fn(camera, field, SimpleNamespace(),
                SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False),
                SimpleNamespace(shuffle_source_frame=False, enable_exposure_correction=False,
                                nb_visible_src_frames=3, residual_resolution_scale=1.),
                background, learnt_normal=True, nb_src_frames=4, buffer_length=4,
                depth_error_threshold=.01, do_find_closest_frame=False, do_render_src_depth=False,
                render_geo=True, return_depth_normal=False)
            _require(len(captured) == 1, 'Renderer did not make exactly one observed raster call')
            args, output = captured[0]
            _require(int(args[20]) == h and int(args[21]) == w, 'Camera dimensions changed')
            views = selector.buffer_views(*output[10:13], point_count=len(args[1]),
                                          num_rendered=int(output[0]), width=w, height=h)
            focal = [float(np.float32(w)/(np.float32(2)*np.float32(args[18]))),
                     float(np.float32(h)/(np.float32(2)*np.float32(args[19])))]
            principal = [(w-1)*.5, (h-1)*.5]
            layers = selector.select_layers(views['means2d'], views['conic_opacity'], args[8],
                        views['ranges'], views['point_list'], width=w, height=h,
                        focal=focal, principal=principal, extension=extension, strict=True)
            _require(package['render'].shape == package['camera_ray'].shape == (3, h, w),
                     'Original CHW raw/ray layout changed')
            metadata = {'width': w, 'height': h, 'point_count': len(args[1]),
                        'num_rendered': int(output[0]), 'focal': focal, 'principal': principal,
                        'world_to_camera': args[9].T.detach().cpu().tolist(),
                        'camera_center': args[24].detach().cpu().tolist(),
                        'buffer_addresses': views['addresses'],
                        'raster_calls': 1, 'selector_calls': 1, 'field_backward': 0,
                        'source_rgb_reads': 0, 'source_depth_renders': 0,
                        'camera_profile': 'colmap_corner_v2; old non-AA near .2'}
            return {'raw': package['render'], 'ray': package['camera_ray'],
                    'median4': layers['median4'], 'top4': layers['top4'],
                    'status': layers['status'], 'metadata': metadata}
    finally:
        backend.rasterize_gaussians = original
