"""Read an explicitly bound IBGS buffer ABI and reconstruct sampled RGB rays.

Pure NumPy, no renderer/model/image loading. CUDA __expf/FMA and CPU arithmetic
need not agree: even a summary-consistent reconstruction is not an exact CUDA
per-contribution ledger. Callers must bind the binary/source, scalar settings,
original GPU buffer addresses, background and actual colors/all_map inputs.
"""
from __future__ import annotations

import hashlib

import numpy as np

TARGETS = tuple(f'{i:03d}.png' for i in (2, 21, 41, 59, 79, 100, 118, 137,
                                       156, 176, 200, 220, 241, 259, 278, 300))
ABI = {'alignment': 128, 'bool_bytes': 1, 'int_bytes': 4, 'float_bytes': 4,
       'block': (16, 16), 'channels': 3, 'plane_params': 5, 'buffer_length': 4}


def _require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sample_rays(name, width=1320, height=989):
    """512 label/error-independent stratified centers and their 3x3 neighborhoods."""
    _require(name in TARGETS and type(width) is int and type(height) is int
             and width >= 96 and height >= 48, 'Fixed TRAIN name and >=3-pixel strata required')
    centers, pixels, cells, central = [], [], [], []
    for r in range(16):
        for c in range(32):
            cell = r*32+c
            x0, x1 = c*width//32+1, (c+1)*width//32-1
            y0, y1 = r*height//16+1, (r+1)*height//16-1
            digest = hashlib.sha256(f'ibgs_thin_rays_v1:{name}:{cell}'.encode()).digest()
            x = x0+int.from_bytes(digest[:8], 'big') % (x1-x0)
            y = y0+int.from_bytes(digest[8:16], 'big') % (y1-y0)
            centers.append((x, y))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    pixels.append((x+dx, y+dy)); cells.append(cell); central.append(dx == dy == 0)
    xy = np.asarray(pixels, np.int32)
    _require(len(np.unique(xy, axis=0)) == 4608, 'Neighborhoods must remain disjoint')
    return {'xy': xy, 'center_xy': np.asarray(centers, np.int32),
            'cell_id': np.asarray(cells, np.int32), 'is_center': np.asarray(central, bool)}


class _Chunk:
    def __init__(self, value, address):
        _require(isinstance(value, np.ndarray) and value.dtype == np.uint8
                 and value.ndim == 1 and value.flags.c_contiguous, 'Contiguous uint8 buffer required')
        _require(type(address) is int and address >= 0, 'Original GPU buffer address required')
        self.value, self.address, self.offset = value, address, 0

    def take(self, dtype, shape):
        self.offset += (-(self.address+self.offset)) % 128
        dtype = np.dtype(dtype)
        count = int(np.prod(shape, dtype=np.int64))
        end = self.offset+count*dtype.itemsize
        _require(count >= 0 and end <= self.value.nbytes, 'Truncated or incompatible buffer ABI')
        result = np.frombuffer(self.value, dtype=dtype, count=count, offset=self.offset).reshape(shape)
        result.setflags(write=False)
        self.offset = end
        return result


def parse_buffers(geometry, binning, image, *, point_count, num_rendered,
                  width, height, addresses):
    """Parse only prefixes before CUB workspaces, using original GPU addresses.

    `num_rendered` is the returned number of Gaussian/tile instances, NOT the
    number of Gaussians. Only tile_count entries of the image ranges are valid.
    No finite checks touch culled/uninitialized Gaussian slots.
    """
    _require(all(type(v) is int and v > 0 for v in (point_count, width, height))
             and type(num_rendered) is int and num_rendered >= 0, 'Invalid counts')
    _require(set(addresses) == {'geometry', 'binning', 'image'}, 'All three original addresses required')
    g, b, im = (_Chunk(x, addresses[k]) for x, k in
                ((geometry, 'geometry'), (binning, 'binning'), (image, 'image')))
    n, pixels = point_count, width*height
    result = {'width': width, 'height': height, 'point_count': n, 'num_rendered': num_rendered}
    result['depths'] = g.take('<f4', (n,))
    g.take('u1', (n, 3)); result['radii'] = g.take('<i4', (n,))
    result['means2d'] = g.take('<f4', (n, 2)); g.take('<f4', (n, 6))
    result['conic_opacity'] = g.take('<f4', (n, 4))
    result['rgb'] = g.take('<f4', (n, 3)); g.take('<u4', (n,))
    result['point_list'] = b.take('<u4', (num_rendered,))
    b.take('<u4', (num_rendered,))
    result['point_list_keys'] = b.take('<u8', (num_rendered,))
    b.take('<u8', (num_rendered,))
    for key, dtype, shape in (
        ('final_T', '<f4', (pixels,)), ('n_contrib', '<u4', (pixels,)),
        ('ranges_allocation', '<u4', (pixels, 2)), ('median_weight', '<f4', (pixels,)),
        ('median_low', '<u4', (pixels,)), ('median_high', '<u4', (pixels,)),
    ):
        result[key] = im.take(dtype, shape)
    tiles = ((width+15)//16)*((height+15)//16)
    result['ranges'] = result.pop('ranges_allocation')[:tiles]
    ranges = result['ranges']
    _require(bool((ranges[:, 0] <= ranges[:, 1]).all() and (ranges <= num_rendered).all()),
             'Invalid tile ranges; check ABI or num_rendered')
    _require(bool((result['point_list'] < n).all()), 'Point-list ID outside original row order')
    _require(num_rendered == 0 or int(ranges[-1, 1]) == num_rendered
             or int(ranges[:, 1].max()) == num_rendered, 'Incomplete tile coverage')
    result['prefix_bytes'] = {'geometry': g.offset, 'binning': b.offset, 'image': im.offset}
    return result


def compare_summary(reconstructed, production, *, atol, rtol):
    """Discrete mismatch is never hidden by a floating tolerance."""
    _require(np.isfinite(atol) and np.isfinite(rtol) and atol >= 0 and rtol >= 0,
             'Explicit nonnegative finite comparison tolerances required')
    discrete = {k: reconstructed[k] == production[k] for k in ('n_contrib', 'median_low', 'median_high')}
    differences, continuous = {}, {}
    for key in ('final_T', 'rgb', 'median_weight', 'median_depth'):
        x, y = np.asarray(reconstructed[key], np.float64), np.asarray(production[key], np.float64)
        _require(x.shape == y.shape, 'Production summary shape mismatch')
        finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
        differences[key] = float(np.max(np.abs(x-y))) if finite else None
        continuous[key] = finite and bool((np.abs(x-y) <= atol+rtol*np.abs(y)).all())
    return {'status': 'summary_consistent_reconstruction' if all(discrete.values()) and all(continuous.values())
            else 'production_summary_mismatch', 'discrete': discrete, 'continuous': continuous,
            'absolute_errors': differences, 'atol': float(atol), 'rtol': float(rtol),
            'exact_cuda_contribution_ledger': False}


def _median_and_top(weights, incoming, depths, orders):
    positive = depths > np.float32(0)
    before = np.flatnonzero(positive & (incoming > np.float32(.5)))
    below = np.flatnonzero(positive & (incoming <= np.float32(.5)))[:2]
    slots = np.full(4, -1, np.int64)
    for j in range(max(0, len(before)-2), len(before)):
        slots[j % 2] = before[j]
    slots[2:2+len(below)] = below
    median = np.zeros(len(weights), bool); median[slots[slots >= 0]] = True
    # The production ring uses only z > 0 (including +inf); the diagnostic
    # top-weight comparator independently requires a finite positive depth.
    candidates = np.flatnonzero(positive & np.isfinite(depths))
    top = candidates[np.lexsort((orders[candidates], -weights[candidates]))[:4]]
    top_mask = np.zeros(len(weights), bool); top_mask[top] = True
    total = np.float32(0); weighted_depth = np.float32(0)
    low = high = int(orders[slots[0]]) if slots[0] >= 0 else 0
    for index in slots:
        if index >= 0:
            total = np.float32(total+weights[index])
            weighted_depth = np.float32(weighted_depth+np.float32(weights[index]*depths[index]))
            low, high = min(low, int(orders[index])), max(high, int(orders[index]))
    return {'median_mask': median, 'top4_mask': top_mask, 'median_slots': slots,
            'median_weight': float(total), 'median_depth': float(weighted_depth/np.float32(total+np.float32(1e-8))),
            'median_low': low, 'median_high': high}


def replay_tile(ids, means2d, conic_opacity, colors, planes, center_depth, xy, *,
                focal, principal, background):
    """Vectorize power/alpha/T over one tile chunk; return every accepted RGB contribution.

    Only RGB+geometry mode, buffer_length=4. Depth-only has an additional
    early-stop branch and is intentionally unsupported. Array coordinates are
    integers, principal point is the actual backend array-coordinate cx/cy.
    CPU multiply/add are separately rounded; no claim to reproduce CUDA FMA.
    """
    ids, xy = np.asarray(ids), np.asarray(xy)
    n, rays = len(ids), len(xy)
    _require(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer)
             and (ids >= 0).all() and xy.shape == (rays, 2)
             and np.issubdtype(xy.dtype, np.integer), 'ID/coordinate shape mismatch')
    values = ((means2d, (n, 2)), (conic_opacity, (n, 4)), (colors, (n, 3)),
              (planes, (n, 5)), (center_depth, (n,)))
    _require(all(isinstance(x, np.ndarray) and x.shape == shape and x.dtype == np.float32
                 and np.isfinite(x).all() for x, shape in values), 'Finite actual FP32 referenced attributes required')
    focal, principal, background = (np.asarray(x, np.float32) for x in (focal, principal, background))
    _require(focal.shape == principal.shape == (2,) and background.shape == (3,)
             and np.isfinite(focal).all() and (focal > 0).all()
             and np.isfinite(principal).all() and np.isfinite(background).all(), 'Invalid actual camera/background')
    _require(np.isfinite(xy).all(), 'Finite ray coordinates required')
    pix = xy.astype(np.float32)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        dx = means2d[None, :, 0]-pix[:, None, 0]
        dy = means2d[None, :, 1]-pix[:, None, 1]
        power = np.float32(-.5)*(conic_opacity[None, :, 0]*dx*dx + conic_opacity[None, :, 2]*dy*dy)
        power -= conic_opacity[None, :, 1]*dx*dy
        alpha = np.minimum(np.float32(.99), conic_opacity[None, :, 3]*np.exp(power))
        candidate = (power <= 0) & (alpha >= np.float32(1/255))
        after = np.cumprod(np.float32(1)-np.where(candidate, alpha, np.float32(0)), axis=1, dtype=np.float32)
        incoming = np.concatenate((np.ones((rays, 1), np.float32), after[:, :-1]), axis=1) if n else after.copy()
        ray = (pix-principal)/focal
        z = -planes[None, :, 4]/(planes[None, :, 0]*ray[:, None, 0]
             + planes[None, :, 1]*ray[:, None, 1]+planes[None, :, 2]+np.float32(1e-8))
    records = []
    for i in range(rays):
        stop = np.flatnonzero(candidate[i] & (after[i] < np.float32(1e-4)))
        stop = int(stop[0]) if len(stop) else n
        accepted = np.flatnonzero(candidate[i, :stop])
        ordinals = (accepted+1).astype(np.uint32)
        weights = (alpha[i, accepted]*incoming[i, accepted]).astype(np.float32)
        T = after[i, accepted[-1]] if len(accepted) else np.float32(1)
        rgb = (np.cumsum(colors[accepted]*weights[:, None], axis=0, dtype=np.float32)[-1]
               if len(accepted) else np.zeros(3, np.float32))
        rgb = (rgb+T*background).astype(np.float32)
        median = _median_and_top(weights, incoming[i, accepted], z[i, accepted], ordinals)
        summary = {k: median[k] for k in ('median_weight', 'median_depth', 'median_low', 'median_high')}
        summary.update(final_T=float(T), n_contrib=int(ordinals[-1]) if len(ordinals) else 0, rgb=rgb.tolist())
        # Margins locate possible threshold sensitivity; they are not formal
        # CUDA error bounds and must not be used to relabel a mismatch as exact.
        visited = slice(0, min(stop+1, n))
        margins = {'power_zero': float(np.min(np.abs(power[i, visited]))) if n else None,
                   'alpha_cut': float(np.min(np.abs(alpha[i, visited]-np.float32(1/255)))) if n else None,
                   'T_stop': float(np.min(np.abs(after[i, visited]-np.float32(1e-4)))) if n else None,
                   'T_median': float(np.min(np.abs(incoming[i, accepted]-np.float32(.5)))) if len(accepted) else None}
        numerical = bool(np.isfinite(power[i, visited]).all() and np.isfinite(alpha[i, visited]).all()
                         and np.isfinite(z[i, accepted]).all() and np.isfinite(rgb).all()
                         and np.isfinite(summary['median_weight']) and np.isfinite(summary['median_depth']))
        records.append({'xy': xy[i].astype(np.int32), 'ids': ids[accepted].copy(), 'order': ordinals,
            'alpha': alpha[i, accepted].copy(), 'T': incoming[i, accepted].copy(), 'w': weights,
            'z': z[i, accepted].copy(), 'center_depth': center_depth[accepted].copy(),
            'median_mask': median['median_mask'], 'top4_mask': median['top4_mask'],
            'median_slots': median['median_slots'], 'summary': summary, 'threshold_margins': margins,
            'finite_cpu_arithmetic': numerical, 'stop_ordinal': stop+1 if stop < n else None,
            'mass_closure_error': float(np.sum(weights, dtype=np.float64)+float(T)-1)})
    return records


def replay_rays(parsed, xy, planes, *, focal, principal, background, raw_rgb,
                median_depth, atol, rtol, render_geo, render_depth_only,
                buffer_length, colors_precomp=None, chunk_rays=32):
    """Reuse each tile's captured list; preserve sampled order and every mismatch."""
    w, h, n = parsed['width'], parsed['height'], parsed['point_count']
    _require(render_geo is True and render_depth_only is False and buffer_length == 4,
             'Only actual RGB+geometry / non-depth-only / buffer4 mode is covered')
    xy = np.asarray(xy)
    _require(xy.ndim == 2 and xy.shape[1] == 2 and np.issubdtype(xy.dtype, np.integer)
             and ((xy >= 0) & (xy < [w, h])).all(), 'Integer in-bounds sampled rays required')
    _require(type(chunk_rays) is int and 1 <= chunk_rays <= 256, 'Bounded tile chunk required')
    _require(isinstance(planes, np.ndarray) and planes.dtype == np.float32 and planes.shape == (n, 5), 'Actual all_map Nx5 required')
    colors = parsed['rgb'] if colors_precomp is None else colors_precomp
    _require(isinstance(colors, np.ndarray) and colors.dtype == np.float32 and colors.shape == (n, 3), 'Actual RGB colors required')
    _require(raw_rgb.shape == (h, w, 3) and median_depth.shape == (h, w), 'Production output shape mismatch')
    tile_ids = (xy[:, 1]//16)*((w+15)//16)+xy[:, 0]//16
    result = [None]*len(xy)
    for tile in np.unique(tile_ids):
        selected = np.flatnonzero(tile_ids == tile)
        start, end = map(int, parsed['ranges'][tile])
        ids = parsed['point_list'][start:end].astype(np.int64)
        _require(bool(((parsed['point_list_keys'][start:end] >> np.uint64(32)) == tile).all()),
                 'Sorted key tile identity mismatch')
        attrs = [parsed['means2d'][ids], parsed['conic_opacity'][ids], colors[ids], planes[ids], parsed['depths'][ids]]
        for begin in range(0, len(selected), chunk_rays):
            chunk = selected[begin:begin+chunk_rays]
            records = replay_tile(ids, *attrs, xy[chunk], focal=focal, principal=principal, background=background)
            for index, row in zip(chunk, records, strict=True):
                x, y = map(int, xy[index]); pixel = y*w+x
                production = {k: float(parsed[k][pixel]) for k in ('final_T', 'median_weight')}
                production.update({k: int(parsed[k][pixel]) for k in ('n_contrib', 'median_low', 'median_high')})
                production.update(rgb=raw_rgb[y, x].tolist(), median_depth=float(median_depth[y, x]))
                comparison = compare_summary(row['summary'], production, atol=atol, rtol=rtol)
                if not row['finite_cpu_arithmetic']:
                    comparison['status'] = 'nonfinite_cpu_replay'
                row.update(production=production, comparison=comparison, tile=int(tile))
                result[index] = row
    return result
