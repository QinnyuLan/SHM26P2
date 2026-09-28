"""Standard joint/PCGrad means transactions with a fixed geometric step cap.

PCGrad uses one global inner product over all means coordinates, not per-point
conflicts. Adam and per-point projection can destroy its first-order descent
property; the caller must evaluate the actual rendered objectives before resolve.
No scene, renderer, label, selection, or acceptance gate is implemented here.
"""
from __future__ import annotations

import copy
import math
from numbers import Real

import torch
from torch import Tensor, nn

SPEC = {
    'modes': ['joint', 'pcgrad', 'semantic'], 'semantic_weight': .03,
    'pcgrad': 'symmetric two-task, both projections use the original other task; global flattened dot',
    'adam_betas': [.9, .999], 'adam_eps': 1e-15, 'weight_decay': 0.,
    'means_lr_scene_scale_factor': 1.6e-6, 'per_point_mahalanobis_cap': 1/64,
    'cap_metric': 'fixed normalized-wxyz quaternion and exp(log_scales), FP64 metric',
    'overcap_after_fp32': 'restore that point exactly; never enlarge cap',
}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _finite(tensor, shape=None):
    _require(isinstance(tensor, Tensor) and tensor.dtype in (torch.float32, torch.float64)
             and (shape is None or tuple(tensor.shape) == shape)
             and bool(torch.isfinite(tensor).all()), 'Expected finite FP32/FP64 tensor with matching shape')


@torch.no_grad()
def combine_gradients(g_rgb: Tensor, g_semantic: Tensor, mode: str) -> tuple[Tensor, dict]:
    """Return FP64 combined gradient; semantic-only uses unweighted gS.

    Normalization/weighting of each task's objective is the caller's contract.
    The only weighting applied here is the fixed .03 for joint and PCGrad.
    """
    _require(mode in SPEC['modes'], 'Unknown optimization mode')
    _finite(g_rgb); _finite(g_semantic, tuple(g_rgb.shape))
    _require(g_rgb.ndim == 2 and g_rgb.shape[1] == 3 and len(g_rgb) > 0
             and g_rgb.device == g_semantic.device, 'Expected same-device [N,3] gradients')
    rgb = g_rgb.detach().double()
    sem = g_semantic.detach().double()
    weighted = SPEC['semantic_weight']*sem
    rr, ss, rs = (rgb*rgb).sum(), (weighted*weighted).sum(), (rgb*weighted).sum()
    _finite(torch.stack((rr, ss, rs)))
    conflict = bool(rs < 0)
    if mode == 'pcgrad' and conflict:
        # rs<0 implies both norms are positive. No epsilon changes this projection.
        _require(bool(rr > 0) and bool(ss > 0), 'Degenerate conflicting task norms')
        projected_rgb = rgb-(rs/ss)*weighted
        projected_sem = weighted-(rs/rr)*rgb
        combined = projected_rgb+projected_sem
    else:
        combined = sem.clone() if mode == 'semantic' else rgb+weighted
    _finite(combined)
    stats = {'mode': mode, 'task_dot_rgb_weighted_semantic': float(rs),
             'rgb_gradient_l2': float(rr.sqrt()), 'weighted_semantic_gradient_l2': float(ss.sqrt()),
             'conflicting_original_gradients': conflict,
             'rgb_dot_combined_gradient': float((rgb*combined).sum()),
             'semantic_dot_combined_gradient': float((sem*combined).sum()),
             'combined_gradient_l2': float(combined.norm())}
    return combined, stats


@torch.no_grad()
def fixed_metric(quats: Tensor, log_scales: Tensor) -> tuple[Tensor, Tensor]:
    """Construct detached FP64 R and 1/s; local displacement is R^T delta."""
    _finite(quats); _finite(log_scales)
    _require(quats.ndim == 2 and quats.shape[1] == 4 and len(quats) > 0
             and log_scales.shape == (len(quats), 3) and quats.device == log_scales.device,
             'Expected same-device quaternions [N,4] and log scales [N,3]')
    q = quats.detach().double().clone()
    norm = q.norm(dim=-1, keepdim=True)
    _require(bool((norm > 1e-12).all()) and bool(torch.isfinite(norm).all()), 'Invalid quaternion norm')
    w, x, y, z = (q/norm).unbind(-1)
    rotation = torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)
    inverse_scales = torch.exp(-log_scales.detach().double())
    _finite(rotation); _finite(inverse_scales)
    _require(bool((inverse_scales > 0).all()), 'Degenerate fixed covariance')
    return rotation, inverse_scales


def _metric_norm(delta: Tensor, rotation: Tensor, inverse_scales: Tensor) -> Tensor:
    local = torch.einsum('nij,ni->nj', rotation, delta)
    return (local*inverse_scales).norm(dim=-1)


@torch.no_grad()
def cap_displacement(before: Tensor, proposed: Tensor, rotation: Tensor,
                     inverse_scales: Tensor) -> tuple[Tensor, dict]:
    """Cap an Adam proposal, then check the realized FP32 displacement strictly.

    A point whose rounded step exceeds the cap returns to its exact prior value.
    The cap is per step, not a cumulative trust region about the initial scene.
    """
    _finite(before); _finite(proposed, tuple(before.shape))
    _require(before.dtype == proposed.dtype == torch.float32 and before.ndim == 2
             and before.shape[1] == 3 and len(before) > 0, 'Expected FP32 [N,3] means')
    _finite(rotation, (len(before), 3, 3)); _finite(inverse_scales, (len(before), 3))
    _require(rotation.dtype == inverse_scales.dtype == torch.float64
             and len({x.device for x in (before, proposed, rotation, inverse_scales)}) == 1
             and bool((inverse_scales > 0).all()), 'Invalid metric dtype/device/scales')
    delta = proposed.double()-before.double()
    proposed_nonzero = (delta != 0).any(-1)
    distance = _metric_norm(delta, rotation, inverse_scales)
    _finite(distance)
    cap = SPEC['per_point_mahalanobis_cap']
    scale = torch.ones_like(distance)
    capped = distance > cap
    scale[capped] = cap/distance[capped]
    result = (before.double()+scale[:, None]*delta).float()
    actual = result.double()-before.double()
    # This collapse is distinct from the explicit over-cap fallback below.
    cap_rounded_to_before = proposed_nonzero & ~(actual != 0).any(-1)
    realized = _metric_norm(actual, rotation, inverse_scales)
    _finite(result); _finite(realized)
    fallback = realized > cap
    result[fallback] = before[fallback]
    actual = result.double()-before.double()
    realized = _metric_norm(actual, rotation, inverse_scales)
    _require(bool((realized <= cap).all()), 'Actual FP32 step violates the fixed cap')
    return result, {'mahalanobis_cap': cap, 'adam_proposed_mahalanobis_max': float(distance.max()),
                    'adam_proposed_nonzero_point_count': int(proposed_nonzero.sum()),
                    'scaled_point_count': int(capped.sum()), 'fp32_overcap_restored_point_count': int(fallback.sum()),
                    'cap_rounded_to_before_point_count': int(cap_rounded_to_before.sum()),
                    'actual_mahalanobis_max': float(realized.max()),
                    'actual_nonzero_point_count': int((actual != 0).any(-1).sum()),
                    'actual_nonzero_coordinate_count': int((actual != 0).sum()),
                    'actual_absmax_displacement': float(actual.abs().max())}


class MeansTransaction:
    """Temporarily mutate the existing means Parameter; resolve after actual rendering.

    Each independent arm constructs a fresh transaction. propose returns a detached
    copy and native JSON statistics; the live Parameter already contains that
    proposal. On any proposal exception or resolve(False), means and the entire
    Adam state are restored. A per-point cap/fallback leaves Adam moments intact
    if the whole proposal is accepted (standard projected Adam); only a rejected
    whole proposal rolls them back. Caller must resolve in its own finally block.
    """
    def __init__(self, means: nn.Parameter, quats: Tensor, log_scales: Tensor, scene_scale: float):
        _require(isinstance(means, nn.Parameter) and means.is_leaf and means.dtype == torch.float32
                 and means.ndim == 2 and means.shape[1] == 3 and len(means) > 0, 'Expected original FP32 means Parameter')
        _finite(means)
        _require(means.grad is None, 'Use explicit task gradients; preexisting means.grad is forbidden')
        _require(isinstance(scene_scale, Real) and not isinstance(scene_scale, bool)
                 and math.isfinite(scene_scale) and scene_scale > 0, 'Expected positive finite scene scale')
        self.rotation, self.inverse_scales = fixed_metric(quats, log_scales)
        _require(len(quats) == len(means) and means.device == quats.device, 'Geometry/means mismatch')
        self.means = means
        self.learning_rate = SPEC['means_lr_scene_scale_factor']*float(scene_scale)
        _require(math.isfinite(self.learning_rate) and self.learning_rate > 0, 'Invalid learning rate')
        self.optimizer = torch.optim.Adam([means], lr=self.learning_rate, betas=(.9, .999),
                                          eps=1e-15, weight_decay=0.)
        self.pending = None

    @torch.no_grad()
    def propose(self, g_rgb: Tensor, g_semantic: Tensor, mode: str) -> tuple[Tensor, dict]:
        _require(self.pending is None, 'Resolve the pending proposal first')
        _require(self.means.grad is None, 'Unexpected accumulated means.grad')
        _finite(g_rgb, tuple(self.means.shape)); _finite(g_semantic, tuple(self.means.shape))
        _require(g_rgb.device == g_semantic.device == self.means.device, 'Gradient device mismatch')
        gradient, stats = combine_gradients(g_rgb, g_semantic, mode)
        gradient32 = gradient.float()
        _finite(gradient32)
        self.pending = (self.means.detach().clone(), copy.deepcopy(self.optimizer.state_dict()))
        before = self.pending[0]
        try:
            self.means.grad = gradient32
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            for state in self.optimizer.state.values():
                for value in state.values():
                    if isinstance(value, Tensor):
                        _require(bool(torch.isfinite(value).all()), 'Nonfinite Adam state')
            proposal, cap_stats = cap_displacement(before, self.means, self.rotation, self.inverse_scales)
            self.means.copy_(proposal)
            actual = self.means.double()-before.double()
            stats.update(cap_stats)
            stats.update(learning_rate=self.learning_rate, adam_betas=[.9, .999], adam_eps=1e-15,
                         weight_decay=0., rgb_dot_actual_delta=float((g_rgb.double()*actual).sum()),
                         semantic_dot_actual_delta=float((g_semantic.double()*actual).sum()),
                         task_gradient_cast_absmax=float((gradient-gradient32.double()).abs().max()),
                         optimizer_proposed_step=int(self.optimizer.state[self.means]['step']),
                         first_order_RGB_guarantee=False)
            _require(all(not isinstance(x, float) or math.isfinite(x) for x in stats.values()), 'Nonfinite proposal statistics')
            return self.means.detach().clone(), stats
        except BaseException:
            self.resolve(False)
            raise

    @torch.no_grad()
    def resolve(self, accepted: bool) -> None:
        _require(type(accepted) is bool, 'Acceptance must be a native bool')
        _require(self.pending is not None, 'No pending proposal')
        before, state = self.pending
        if not accepted:
            self.means.copy_(before)
            self.optimizer.load_state_dict(state)
        self.optimizer.zero_grad(set_to_none=True)
        self.pending = None
