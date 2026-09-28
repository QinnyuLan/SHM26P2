"""Frozen-geometry semantic partition rasterization using gsplat's tile ledger.

This is an experimental single-camera, five-class shader. It preserves the
provided EWA footprints, ordering, AA-compensated opacity and alpha thresholds.
Only endpoint probabilities and per-view CDF coefficients are differentiable.
It is not an exact perspective-volume renderer or a new Gaussian-CDF integral.
Conditional clipping/half-Gaussian integrals have precedents in 3D-HGS/XClipGS.
No training or model selection is launched by importing this module.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _interval_probability(upper, lower):
    # Both evaluations avoid subtraction of two numbers close to one.
    pos = .5 * (libdevice.erfc(lower * .7071067811865476)
                - libdevice.erfc(upper * .7071067811865476))
    neg = .5 * (libdevice.erfc(-upper * .7071067811865476)
                - libdevice.erfc(-lower * .7071067811865476))
    center = .5 * (libdevice.erf(upper * .7071067811865476)
                   - libdevice.erf(lower * .7071067811865476))
    value = tl.where(lower >= 0., pos, tl.where(upper <= 0., neg, center))
    return tl.minimum(tl.maximum(value, 0.), 1.)


@triton.jit
def _forward_kernel(
    MEAN, CONIC, OPACITY, OFFSETS, IDS, QIN, QOUT, COEFF, OUT, ALPHA,
    WIDTH: tl.constexpr, HEIGHT: tl.constexpr, TILE_WIDTH: tl.constexpr,
    TILE_COUNT: tl.constexpr, ISECTS: tl.constexpr,
):
    tile = tl.program_id(0)
    lane = tl.arange(0, 256)
    cls = tl.arange(0, 8)
    ix = (tile % TILE_WIDTH) * 16 + lane % 16
    iy = (tile // TILE_WIDTH) * 16 + lane // 16
    inside = (ix < WIDTH) & (iy < HEIGHT)
    px, py = ix.to(tl.float32) + .5, iy.to(tl.float32) + .5
    begin = tl.load(OFFSETS + tile)
    end = tl.load(OFFSETS + tile + 1, mask=tile+1 < TILE_COUNT, other=ISECTS)
    trans = tl.full((256,), 1., tl.float32)
    done = ~inside
    result = tl.full((256, 8), 0., tl.float32)
    j = begin
    while (j < end) & (tl.sum((~done).to(tl.int32), 0) > 0):
        g = tl.load(IDS + j)
        dx = tl.load(MEAN + g*2) - px
        dy = tl.load(MEAN + g*2+1) - py
        a, b, c = tl.load(CONIC+g*3), tl.load(CONIC+g*3+1), tl.load(CONIC+g*3+2)
        sigma = .5 * (a*dx*dx + c*dy*dy) + b*dx*dy
        alpha = tl.minimum(.999, tl.load(OPACITY+g)*tl.exp(-sigma))
        supported = (~done) & (sigma >= 0.) & (alpha >= 1./255.)
        next_trans = trans * (1.-alpha)
        stop = supported & (next_trans <= 1.e-4)
        use = supported & ~stop
        weight = tl.where(use, alpha*trans, 0.)
        # Coefficients use delta = pixel-center minus projected Gaussian mean.
        affine = -tl.load(COEFF+g*4)*dx - tl.load(COEFF+g*4+1)*dy
        gate = _interval_probability(affine+tl.load(COEFF+g*4+2),
                                     affine+tl.load(COEFF+g*4+3))
        qi = tl.load(QIN+g*5+cls, mask=cls < 5, other=0.)
        qo = tl.load(QOUT+g*5+cls, mask=cls < 5, other=0.)
        probability = qo[None, :] + gate[:, None]*(qi-qo)[None, :]
        result += weight[:, None] * probability
        trans = tl.where(use, next_trans, trans)
        done |= stop
        j += 1
    result += trans[:, None] * (cls[None, :] == 0)
    pixel = iy*WIDTH+ix
    tl.store(OUT+pixel[:, None]*5+cls[None, :], result,
             mask=inside[:, None] & (cls[None, :] < 5))
    tl.store(ALPHA+pixel, 1.-trans, mask=inside)


@triton.jit
def _backward_kernel(
    MEAN, CONIC, OPACITY, OFFSETS, IDS, QIN, QOUT, COEFF, GRAD,
    GIN, GOUT, GCOEFF,
    WIDTH: tl.constexpr, HEIGHT: tl.constexpr, TILE_WIDTH: tl.constexpr,
    TILE_COUNT: tl.constexpr, ISECTS: tl.constexpr,
):
    tile = tl.program_id(0)
    lane = tl.arange(0, 256)
    cls = tl.arange(0, 8)
    ix = (tile % TILE_WIDTH)*16 + lane % 16
    iy = (tile // TILE_WIDTH)*16 + lane // 16
    inside = (ix < WIDTH) & (iy < HEIGHT)
    px, py = ix.to(tl.float32)+.5, iy.to(tl.float32)+.5
    pixel = iy*WIDTH+ix
    grad = tl.load(GRAD+pixel[:, None]*5+cls[None, :],
                   mask=inside[:, None] & (cls[None, :] < 5), other=0.)
    begin = tl.load(OFFSETS+tile)
    end = tl.load(OFFSETS+tile+1, mask=tile+1 < TILE_COUNT, other=ISECTS)
    trans = tl.full((256,), 1., tl.float32)
    done = ~inside
    j = begin
    # Recompute forward transmittance rather than reconstructing it from 1-alpha.
    # Geometry is fixed, so later colors have no derivative through this alpha.
    while (j < end) & (tl.sum((~done).to(tl.int32), 0) > 0):
        g = tl.load(IDS+j)
        dx, dy = tl.load(MEAN+g*2)-px, tl.load(MEAN+g*2+1)-py
        a, b, c = tl.load(CONIC+g*3), tl.load(CONIC+g*3+1), tl.load(CONIC+g*3+2)
        sigma = .5*(a*dx*dx+c*dy*dy)+b*dx*dy
        alpha = tl.minimum(.999, tl.load(OPACITY+g)*tl.exp(-sigma))
        supported = (~done) & (sigma >= 0.) & (alpha >= 1./255.)
        next_trans = trans*(1.-alpha)
        stop = supported & (next_trans <= 1.e-4)
        use = supported & ~stop
        weight = tl.where(use, alpha*trans, 0.)
        affine = -tl.load(COEFF+g*4)*dx-tl.load(COEFF+g*4+1)*dy
        upper, lower = affine+tl.load(COEFF+g*4+2), affine+tl.load(COEFF+g*4+3)
        gate = _interval_probability(upper, lower)
        qi = tl.load(QIN+g*5+cls, mask=cls < 5, other=0.)
        qo = tl.load(QOUT+g*5+cls, mask=cls < 5, other=0.)
        weighted_grad = weight[:, None]*grad
        dqin = tl.sum(weighted_grad*gate[:, None], axis=0)
        dqout = tl.sum(weighted_grad*(1.-gate[:, None]), axis=0)
        dg = tl.sum(weighted_grad*(qi-qo)[None, :], axis=1)
        dhi = dg*.3989422804014327*tl.exp(-.5*upper*upper)
        dlo = -dg*.3989422804014327*tl.exp(-.5*lower*lower)
        tl.atomic_add(GIN+g*5+cls, dqin, mask=cls < 5)
        tl.atomic_add(GOUT+g*5+cls, dqout, mask=cls < 5)
        tl.atomic_add(GCOEFF+g*4, tl.sum((dhi+dlo)*(-dx), axis=0))
        tl.atomic_add(GCOEFF+g*4+1, tl.sum((dhi+dlo)*(-dy), axis=0))
        tl.atomic_add(GCOEFF+g*4+2, tl.sum(dhi, axis=0))
        tl.atomic_add(GCOEFF+g*4+3, tl.sum(dlo, axis=0))
        trans = tl.where(use, next_trans, trans)
        done |= stop
        j += 1


class _PartitionRaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, conics, opacity, offsets, ids, qin, qout, coefficients, width, height):
        result = torch.empty((height, width, 5), device=means.device, dtype=torch.float32)
        alpha = torch.empty((height, width, 1), device=means.device, dtype=torch.float32)
        tiles = offsets.numel()
        args = {'WIDTH': width, 'HEIGHT': height, 'TILE_WIDTH': offsets.shape[-1],
                'TILE_COUNT': tiles, 'ISECTS': ids.numel(), 'num_warps': 8,
                'enable_fp_fusion': True}
        _forward_kernel[(tiles,)](means, conics, opacity, offsets, ids, qin, qout,
                                  coefficients, result, alpha, **args)
        ctx.save_for_backward(means, conics, opacity, offsets, ids, qin, qout, coefficients)
        ctx.launch_args = args
        ctx.mark_non_differentiable(alpha)
        return result, alpha

    @staticmethod
    def backward(ctx, grad_probability, _grad_alpha):
        means, conics, opacity, offsets, ids, qin, qout, coefficients = ctx.saved_tensors
        gin, gout, gcoeff = torch.zeros_like(qin), torch.zeros_like(qout), torch.zeros_like(coefficients)
        _backward_kernel[(offsets.numel(),)](
            means, conics, opacity, offsets, ids, qin, qout, coefficients,
            grad_probability.contiguous(), gin, gout, gcoeff, **ctx.launch_args)
        return None, None, None, None, None, gin, gout, gcoeff, None, None


def rasterize_semantic_partition(means2d, conics, opacity, offsets, ids,
                                 q_in, q_out, coefficients, *, width, height, tile_size=16):
    """Render five class probabilities with fixed gsplat single-camera visibility.

    Projected arrays are [N,2], [N,3], [N], integer tile offsets [ceil(H/16),
    ceil(W/16)] and integer Gaussian IDs sorted by tile/depth as supplied by
    gsplat. CDF coefficients [N,4] are kx,ky,h_upper,h_lower, with h_upper>h_lower.
    q_in and q_out are caller-supplied five-class simplex probabilities [N,5].
    All floating inputs are contiguous CUDA FP32. Geometry requiring gradients
    is rejected; callers must deliberately freeze it. Parameter-domain/simplex
    validation belongs to the field constructor; do not accept arbitrary files
    as kernel inputs. Atomic gradient reductions need not be bitwise repeatable.
    """
    if tile_size != 16 or not isinstance(width, int) or not isinstance(height, int) or min(width, height) < 1:
        raise ValueError('Only positive integer dimensions and 16x16 tiles are supported')
    values = (means2d, conics, opacity, offsets, ids, q_in, q_out, coefficients)
    if not all(isinstance(v, torch.Tensor) for v in values):
        raise TypeError('Tensor inputs required')
    if not means2d.is_cuda or any(v.device != means2d.device for v in values):
        raise ValueError('Inputs must share one CUDA device')
    if any(v.dtype != torch.float32 for v in (means2d, conics, opacity, q_in, q_out, coefficients)):
        raise ValueError('CUDA shader requires float32')
    if offsets.dtype != torch.int32 or ids.dtype != torch.int32:
        raise ValueError('Tile offsets and Gaussian IDs must be int32')
    n = len(means2d)
    if (means2d.shape != (n, 2) or conics.shape != (n, 3) or opacity.shape != (n,)
            or q_in.shape != (n, 5) or q_out.shape != (n, 5) or coefficients.shape != (n, 4)
            or offsets.shape != ((height+15)//16, (width+15)//16) or ids.ndim != 1):
        raise ValueError('Incorrect single-camera geometry, field or tile shapes')
    if any(v.requires_grad for v in (means2d, conics, opacity)):
        raise ValueError('This shader supports fixed geometry only')
    if any(not v.is_contiguous() for v in values):
        raise ValueError('Inputs must be contiguous')
    return _PartitionRaster.apply(*values, width, height)
