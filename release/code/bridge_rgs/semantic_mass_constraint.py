"""Fixed categorical marginal, two-column coupling: independent FP64 helper.

This is standard constrained categorical reparameterization, not a new operator
or a demonstrated training improvement. qbar is fixed. Learning qbar as well
would remove the intended fixed-marginal restriction; this API rejects that.
No model, renderer, filesystem, or GPU launch is invoked here.

The unique root satisfies sum(qbar * sigmoid(h + lambda)) = a. A custom
implicit VJP carries *first-order* gradients to h and a; it does not backprop
through bisection decisions. Every differentiable output rejects construction
of a higher-order derivative graph (create_graph=True).
No probability floor, clamp, hidden normalization, or derivative damping.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable

REVISION = "fixed_categorical_two_column_fp64_v1"
MAX_ITERATIONS = 192
EPS = torch.finfo(torch.float64).eps
ROOT_RELATIVE_TOLERANCE = 64 * EPS
SIMPLEX_ABSOLUTE_TOLERANCE = 256 * EPS
MASS_RELATIVE_TOLERANCE = 128 * EPS
MIN_ROOT_DERIVATIVE = 1e-14


def _pair(logits):
    # Unlike 1-sigmoid(z), the negative branch retains a small tail when the
    # positive branch rounds to one. logaddexp has no softplus threshold.
    zero = torch.zeros_like(logits)
    return (torch.exp(-torch.logaddexp(zero, -logits)),
            torch.exp(-torch.logaddexp(zero, logits)))


def _residual(qbar, a, positive, negative):
    total = qbar.sum(-1)
    inside, outside = (qbar*positive).sum(-1), (qbar*negative).sum(-1)
    # These are the same increasing function in exact arithmetic. Using the
    # smaller column avoids subtracting two masses close to one.
    value = torch.where(a <= total*.5, inside-a, (total-a)-outside)
    return value, torch.minimum(a, total-a)


class _ImplicitRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qbar, centered_h, a):
        total = qbar.sum(-1)
        logit = torch.log(a)-torch.log(total-a)
        # A fixed two-logit padding makes endpoint signs robust to roundoff.
        lower = logit-centered_h.amax(-1)-2.
        upper = logit-centered_h.amin(-1)+2.
        if not bool((torch.isfinite(lower) & torch.isfinite(upper) & (lower < upper)).all()):
            raise ValueError("Nonfinite or collapsed root bracket")
        fl, scale = _residual(qbar, a, *_pair(centered_h+lower[..., None]))
        fu, _ = _residual(qbar, a, *_pair(centered_h+upper[..., None]))
        if not bool(((fl <= 0) & (fu >= 0)).all()):
            raise ValueError("Root bracket does not contain the target mass")
        root = lower*.5+upper*.5
        done = torch.zeros_like(a, dtype=torch.bool)
        iterations = torch.zeros_like(a, dtype=torch.int64)
        for iteration in range(1, MAX_ITERATIONS+1):
            positive, negative = _pair(centered_h+root[..., None])
            residual, _ = _residual(qbar, a, positive, negative)
            newly = (~done) & (residual.abs() <= ROOT_RELATIVE_TOLERANCE*scale)
            iterations = torch.where(newly, iteration, iterations)
            done = done | newly
            if bool(done.all()):
                break
            lower = torch.where((~done) & (residual < 0), root, lower)
            upper = torch.where((~done) & (residual >= 0), root, upper)
            root = torch.where(done, root, lower*.5+upper*.5)
        if not bool(done.all()):
            raise ValueError("FP64 bisection did not meet the fixed relative root residual")
        positive, negative = _pair(centered_h+root[..., None])
        sensitivity = qbar*positive*negative
        derivative = sensitivity.sum(-1)
        if not bool((torch.isfinite(derivative) & (derivative > MIN_ROOT_DERIVATIVE)).all()):
            raise ValueError("Root derivative is ill-conditioned; no clamped implicit gradient")
        ctx.save_for_backward(sensitivity/derivative[..., None], derivative)
        ctx.mark_non_differentiable(iterations)
        return root, iterations

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_root, grad_iterations):
        weights, derivative = ctx.saved_tensors
        return None, -grad_root[..., None]*weights, grad_root/derivative


class _FirstOrderOnly(torch.autograd.Function):
    """Reject high-order graph construction even through explicit endpoint paths."""

    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, gradient):
        if torch.is_grad_enabled():
            raise RuntimeError("Mass constraint supports first-order gradients only; "
                               "create_graph=True is forbidden")
        return gradient


@dataclass(frozen=True)
class MassConstrainedEndpoints:
    q_in: torch.Tensor
    q_out: torch.Tensor
    allocation: torch.Tensor
    complement: torch.Tensor
    root: torch.Tensor  # Lambda for the original, uncentered h.
    diagnostics: dict


def mass_constrained_endpoints(qbar, h, a):
    """Return fixed-marginal endpoints, with first-order h/a gradients.

    qbar and h are FP64 tensors ending in K>=2, a is FP64 tensor or Python
    scalar; leading dimensions broadcast. qbar is nonnegative, fixed, finite,
    normalized within 64 eps, and is never changed. a must lie in (0,1) and
    (0,sum(qbar)). Inputs share device. Zeros in qbar remain exact zeros.

    h is centered by its first coordinate before root finding, preserving the
    public-shift gauge without gratuitous h+lambda cancellation. Finite input
    alone is not a numerical guarantee: bracket/residual/derivative/output
    failures raise ValueError for the entire call. Near-boundary normalization
    errors can be amplified and are not repaired. All outputs stay FP64.
    First-order a gradients can lose absolute accuracy near 0/1 because the
    explicit endpoint quotient and implicit-root terms nearly cancel; finite
    derivatives and mass closure do not certify relative derivative accuracy.
    All differentiable outputs reject create_graph=True at gradient entry.
    Diagnostics require scalar synchronization; no performance claim is made.
    """
    if (not isinstance(qbar, torch.Tensor) or not isinstance(h, torch.Tensor)
            or qbar.dtype != torch.float64 or h.dtype != torch.float64
            or qbar.device != h.device or qbar.ndim < 1 or h.ndim < 1
            or qbar.shape[-1] != h.shape[-1] or h.shape[-1] < 2
            or qbar.requires_grad):
        raise ValueError("Require fixed FP64 qbar and matching FP64 h[...,K], K>=2")
    if isinstance(a, bool):
        raise TypeError("a must be FP64 or a real scalar")
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a, dtype=torch.float64, device=h.device)
    if a.dtype != torch.float64 or a.device != h.device:
        raise ValueError("a must share FP64 dtype and device")
    try:
        batch = torch.broadcast_shapes(qbar.shape[:-1], h.shape[:-1], a.shape)
    except RuntimeError as exc:
        raise ValueError("Incompatible coupling batch dimensions") from exc
    qbar = qbar.expand(*batch, qbar.shape[-1])
    h = h.expand(*batch, h.shape[-1])
    a = a.expand(batch)
    if qbar.numel() == 0 or not bool(torch.isfinite(qbar).all() & torch.isfinite(h).all()
                                    & torch.isfinite(a).all() & (qbar >= 0).all()):
        raise ValueError("Inputs must be nonempty, finite, and qbar nonnegative")
    total = qbar.sum(-1)
    if not bool((((total-1).abs() <= 64*EPS) & (a > 0) & (a < 1) & (a < total)).all()):
        raise ValueError("qbar must already be normalized and 0<a<min(1,sum(qbar))")
    anchor = h[..., :1]
    centered_h = h-anchor
    if not bool(torch.isfinite(centered_h).all()):
        raise ValueError("Logit centering overflow; no input clamping")
    centered_root, iterations = _ImplicitRoot.apply(qbar, centered_h, a)
    positive, negative = _pair(centered_h+centered_root[..., None])
    q_in = qbar*positive/a[..., None]
    q_out = qbar*negative/(1-a)[..., None]
    root = centered_root-anchor[..., 0]
    with torch.no_grad():
        residual, scale = _residual(qbar, a, positive, negative)
        derivative = (qbar*positive*negative).sum(-1)
        in_error = (q_in.sum(-1)-1).abs()
        out_error = (q_out.sum(-1)-1).abs()
        reconstructed = a[..., None]*q_in+(1-a)[..., None]*q_out
        mass_error = (reconstructed-qbar).abs()
        positive_q = qbar > 0
        relative_mass_error = torch.zeros_like(qbar)
        relative_mass_error[positive_q] = mass_error[positive_q]/qbar[positive_q]
        valid = (torch.isfinite(q_in).all() & torch.isfinite(q_out).all()
                 & torch.isfinite(root).all() & (q_in >= 0).all() & (q_out >= 0).all()
                 & (q_in <= 1+SIMPLEX_ABSOLUTE_TOLERANCE).all()
                 & (q_out <= 1+SIMPLEX_ABSOLUTE_TOLERANCE).all()
                 & (in_error <= SIMPLEX_ABSOLUTE_TOLERANCE).all()
                 & (out_error <= SIMPLEX_ABSOLUTE_TOLERANCE).all()
                 & (relative_mass_error <= MASS_RELATIVE_TOLERANCE).all()
                 & (mass_error[~positive_q] == 0).all())
        if not bool(valid):
            raise ValueError("Endpoint simplex/mass closure failed; no repair or renormalization")
        diagnostics = {
            "revision": REVISION, "dtype": "float64", "fixed_qbar": True,
            "root_relative_residual_max": float((residual.abs()/scale).max()),
            "root_relative_tolerance": ROOT_RELATIVE_TOLERANCE,
            "root_derivative_min": float(derivative.min()),
            "root_derivative_lower_bound_exclusive": MIN_ROOT_DERIVATIVE,
            "bisection_iterations_max": int(iterations.max()),
            "qbar_sum_absolute_error_max": float((total-1).abs().max()),
            "q_in_sum_absolute_error_max": float(in_error.max()),
            "q_out_sum_absolute_error_max": float(out_error.max()),
            "simplex_absolute_tolerance": SIMPLEX_ABSOLUTE_TOLERANCE,
            "mass_absolute_error_max": float(mass_error.max()),
            "mass_relative_error_max": float(relative_mass_error.max()),
            "mass_relative_tolerance": MASS_RELATIVE_TOLERANCE,
            "allocation_exact_zero_components": int((positive == 0).sum()),
            "complement_exact_zero_components": int((negative == 0).sum()),
            "positive_qbar_endpoint_zero_components": int(((q_in == 0) & positive_q).sum()
                                                          + ((q_out == 0) & positive_q).sum()),
            "input_qbar_min_positive": float(qbar[positive_q].min()),
            "normalization_or_probability_floor_or_clamp": False,
            "gradient_contract": "implicit first-order h/a only; create_graph=True rejected at every differentiable output",
            "near_boundary_gradient_limit": "a derivatives can lose absolute accuracy through cancellation; no uniform relative-error guarantee",
        }
    guarded = (_FirstOrderOnly.apply(value) for value in (q_in, q_out, positive, negative, root))
    return MassConstrainedEndpoints(*guarded, diagnostics)
