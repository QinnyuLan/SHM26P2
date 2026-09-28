"""Forward-only, frozen-field median4/top4 selection; import never builds CUDA.

The extension independently traverses the existing IBGS tile ledger. Matching
arithmetic/source does not yet certify matching production decisions at FP32
thresholds: a separate GPU replay check is required before use in training.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch

MODES = ('median4', 'top4')
CONTRACT = {
    'tile': [16, 16], 'slots': 4, 'alpha_cap': .99, 'alpha_min': 1/255,
    'stop_T_strict_less': 1e-4, 'median_before_T_strict_greater': .5,
    'plane_epsilon': 1e-8, 'selection_depth': 'finite positive',
    'top_tie': 'earlier traversal ordinal', 'missing': {'id': -1, 'depth': 0, 'weight': 0},
    'weights': 'original unnormalized alpha * incoming T', 'backward': False,
    'status_bits': {1: 'invalid range', 2: 'invalid point ID', 4: 'nonfinite referenced input',
                    8: 'nonfinite power/alpha/T', 16: 'nonfinite accepted plane depth'},
}


def _require(value, message):
    if not bool(value):
        raise ValueError(message)


def load_extension(*, build_directory, verbose=False):
    """Explicit compilation entry; caller controls CUDA_HOME/arch/MAX_JOBS.

    All generated files must reside under /mnt/data. No automatic call from
    select_layers, no installation into the existing IBGS extension/package.
    """
    folder = Path(build_directory).resolve()
    _require(folder.is_relative_to(Path('/mnt/data').resolve()), 'Build directory must be under /mnt/data')
    sources = [Path(__file__).parent/'cuda'/f'ibgs_layer_select.{suffix}' for suffix in ('cpp', 'cu')]
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()[:16]
    from torch.utils.cpp_extension import load
    folder.mkdir(parents=True, exist_ok=True)
    return load(name='ibgs_layer_select_'+digest, sources=list(map(str, sources)),
                build_directory=str(folder), extra_cflags=['-O3'], extra_cuda_cflags=['-O3'],
                with_cuda=True, verbose=verbose)


class _TensorChunk:
    def __init__(self, value):
        _require(isinstance(value, torch.Tensor) and value.dtype == torch.uint8 and value.ndim == 1
                 and value.is_contiguous() and not value.requires_grad, 'Contiguous uint8 actual buffer required')
        self.value, self.address, self.offset = value, value.data_ptr(), 0

    def take(self, dtype, shape):
        self.offset += (-(self.address+self.offset)) % 128
        count = math.prod(shape)
        end = self.offset + count*torch.empty((), dtype=dtype).element_size()
        _require(count >= 0 and end <= self.value.numel(), 'Truncated or incompatible IBGS buffer')
        value = self.value[self.offset:end].view(dtype).reshape(shape)
        self.offset = end
        return value


def buffer_views(geometry, binning, image, *, point_count, num_rendered, width, height):
    """Zero-copy typed views from actual current tensors, using data_ptr alignment.

    Use the original CUDA buffers, not a host snapshot relocated to a different
    address. CPU tensors are accepted only for synthetic ABI tests. all_map is
    an explicit rasterizer input and must be provided separately. Padded image
    ranges after tile_count are uninitialized and are never exposed.
    """
    _require(all(type(v) is int and v > 0 for v in (point_count, width, height))
             and type(num_rendered) is int and 0 <= num_rendered < 2**31, 'Invalid ABI dimensions')
    _require(geometry.device == binning.device == image.device, 'One buffer device required')
    g, b, im = map(_TensorChunk, (geometry, binning, image))
    n, pixels = point_count, width*height
    g.take(torch.float32, (n,)); g.take(torch.uint8, (n, 3)); g.take(torch.int32, (n,))
    means = g.take(torch.float32, (n, 2)); g.take(torch.float32, (n, 6))
    conic = g.take(torch.float32, (n, 4))
    ids = b.take(torch.int32, (num_rendered,))
    im.take(torch.float32, (pixels,)); im.take(torch.int32, (pixels,))
    ranges = im.take(torch.int32, (pixels, 2))[:((width+15)//16)*((height+15)//16)]
    return {'means2d': means, 'conic_opacity': conic, 'ranges': ranges, 'point_list': ids,
            'addresses': {k: v.address for k, v in (('geometry', g), ('binning', b), ('image', im))}}


def _validate_inputs(means, conic, planes, ranges, ids, width, height, focal, principal, *, require_cuda=True):
    _require(type(width) is int and type(height) is int and width > 0 and height > 0
             and width*height < 2**31, 'Positive int32 image dimensions required')
    tensors = (means, conic, planes, ranges, ids)
    _require(all(isinstance(v, torch.Tensor) for v in tensors), 'Tensor inputs required')
    _require(not any(v.requires_grad for v in tensors), 'Frozen geometry only; no backward exists')
    _require(all(v.device == means.device and v.is_contiguous() for v in tensors),
             'Contiguous inputs on one device required')
    _require(not require_cuda or means.is_cuda, 'CUDA views required for execution')
    _require(all(v.dtype == torch.float32 for v in (means, conic, planes))
             and all(v.dtype == torch.int32 for v in (ranges, ids)), 'FP32 attributes and int32 ledger required')
    _require(means.ndim == 2 and means.shape[1] == 2, 'Nx2 means required')
    n = len(means)
    _require(conic.shape == (n, 4) and planes.shape == (n, 5) and ids.ndim == 1
             and ranges.shape == (((width+15)//16)*((height+15)//16), 2), 'Projected buffer shape mismatch')
    _require(n < 2**31 and ids.numel() <= 2**31-1-256, 'Buffer exceeds int32 traversal ABI')
    values = np.asarray([*focal, *principal], np.float64)
    _require(values.shape == (4,) and np.isfinite(values).all() and (values[:2] > 0).all(), 'Finite actual camera required')
    with np.errstate(over='ignore', under='ignore'):
        actual = values.astype(np.float32)
    _require(np.isfinite(actual).all() and (actual[:2] > 0).all(), 'Camera must remain valid after FP32 cast')
    return tuple(map(float, actual))


def select_layers(means2d, conic_opacity, all_map, ranges, point_list, *,
                  width, height, focal, principal, extension, strict=True):
    """Return both fixed four-slot selections without autograd or lazy builds.

    principal is the actual backend ARRAY-coordinate cx/cy, not corner K.
    strict=True rejects invalid buffers/arithmetic, including nonfinite accepted
    plane depths. strict=False exposes partial diagnostic arrays plus status;
    callers must never train on a nonzero status image. Negative/zero finite
    plane depths still attenuate T but cannot enter either selector. Original
    IBGS permits +inf in its median ring; such cases are explicitly rejected
    here, not silently certified as production-equivalent.
    """
    camera = _validate_inputs(means2d, conic_opacity, all_map, ranges, point_list,
                              width, height, focal, principal)
    _require(extension is not None and hasattr(extension, 'select_layers'), 'Explicitly loaded extension required')
    with torch.no_grad():
        outputs = extension.select_layers(means2d, conic_opacity, all_map, ranges, point_list,
                                          width, height, *camera)
    _require(len(outputs) == 8 and all(not v.requires_grad for v in outputs), 'Forward-only output contract violated')
    ids, depth, weight, order, final_T, last, invalid_plane, status = outputs
    if strict:
        _require(not torch.any(status != 0).item(), 'Invalid layer selection status; inspect strict=False diagnostic output')
    result = {mode: {'ids': ids[..., i, :], 'depth': depth[..., i, :],
                     'weights': weight[..., i, :], 'ordinals': order[..., i, :]}
              for i, mode in enumerate(MODES)}
    result.update(final_T=final_T, last_contributor=last, nonfinite_plane_count=invalid_plane,
                  status=status, exact_production_ledger=False)
    return result


def reference_select_ray(ids, means2d, conic_opacity, all_map, xy, *, focal, principal):
    """Small NumPy contract oracle, NOT a claim of exact CUDA exp/FMA replay."""
    ids = np.asarray(ids, np.int32)
    means, conic, planes = (np.asarray(v, np.float32) for v in (means2d, conic_opacity, all_map))
    n = len(means)
    _require(means.shape == (n, 2) and conic.shape == (n, 4) and planes.shape == (n, 5)
             and ids.ndim == 1 and ((ids >= 0) & (ids < n)).all(), 'Reference shape/ID mismatch')
    _require(all(np.isfinite(v[ids]).all() for v in (means, conic, planes)), 'Finite referenced inputs required')
    f = np.asarray(focal, np.float32); p = np.asarray(principal, np.float32); pixel = np.asarray(xy, np.float32)
    _require(f.shape == p.shape == pixel.shape == (2,) and np.isfinite(np.r_[f,p,pixel]).all()
             and (f > 0).all(), 'Invalid reference camera')
    ray = (pixel-p)/f
    selections = {mode: {'ids': np.full(4, -1, np.int32), 'depth': np.zeros(4, np.float32),
                         'weights': np.zeros(4, np.float32), 'ordinals': np.zeros(4, np.int32)} for mode in MODES}
    T = np.float32(1); before = below = last = bad = invalid_plane = 0; candidates = []
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        for ordinal, point in enumerate(ids, 1):
            dx, dy = means[point]-pixel; a,b,c,opacity = conic[point]
            power = np.float32(-.5)*np.float32(a*dx*dx+c*dy*dy)-b*dx*dy
            _require(np.isfinite(power), 'Nonfinite reference power')
            if power > 0: continue
            alpha = np.minimum(np.float32(.99), np.float32(opacity*np.exp(power)))
            if alpha < np.float32(1/255): continue
            test_T = np.float32(T*np.float32(1-alpha))
            if test_T < np.float32(1e-4): break
            weight = np.float32(alpha*T); plane = planes[point]
            z = np.float32(-plane[4]/np.float32(plane[0]*ray[0]+plane[1]*ray[1]+plane[2]+np.float32(1e-8)))
            if not np.isfinite(z): bad |= 16; invalid_plane += 1
            if np.isfinite(z) and z > 0:
                slot = None
                if T > np.float32(.5): slot=before; before=(before+1)%2
                elif below < 2: slot=2+below; below+=1
                if slot is not None:
                    for key,value in zip(('ids','depth','weights','ordinals'), (point,z,weight,ordinal), strict=True):
                        selections['median4'][key][slot]=value
                candidates.append((point,z,weight,ordinal))
            T, last = test_T, ordinal
    for slot, candidate in enumerate(sorted(candidates,key=lambda v:(-float(v[2]),v[3]))[:4]):
        for key,value in zip(('ids','depth','weights','ordinals'),candidate,strict=True):
            selections['top4'][key][slot]=value
    return dict(selections, final_T=float(T), last_contributor=last,
                nonfinite_plane_count=invalid_plane, status=bad)
