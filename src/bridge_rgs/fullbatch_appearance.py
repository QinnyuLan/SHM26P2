"""Transactional full-batch RGB descent primitives, with no renderer or data reads.

Standard beta1=0 Adam-type preconditioning and fixed Armijo backtracking. This
module neither authorizes a run nor makes any accuracy/novelty claim.
"""
from __future__ import annotations

import copy
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

KEYS = ('splats.sh0', 'splats.sh_rest', 'background_logits')
RATES = dict(zip(KEYS, (2.5e-4, 1.25e-5, 1e-4), strict=True))
ALPHAS = (1., .5, .25)
KIND = 'fullbatch_appearance_inference_or_warmstart'
BETA2, EPS = .999, 1e-8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decrease_floor(loss):
    return max(1e-7, 1e-5*abs(loss))


def armijo(base, candidate, actual_slope):
    require(all(math.isfinite(x) for x in (base, candidate, actual_slope)), 'Nonfinite objective or slope')
    return (actual_slope < 0 and candidate <= base+1e-4*actual_slope
            and base-candidate >= decrease_floor(base))


@dataclass
class Proposal:
    second_moment: dict
    direction: dict
    analytic_slope: float
    generation: int


@dataclass
class Trial:
    actual_slope: float
    displacement_absmax: dict
    accepted: bool = False

    def accept(self, baseline, candidate):
        self.accepted = armijo(baseline, candidate, self.actual_slope)
        return self.accepted


class RMSArmijo:
    """Moments/counter change only on a normally completed accepted transaction."""

    def __init__(self, parameters):
        require(set(parameters) == set(KEYS), 'Exactly three appearance parameters required')
        require(all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in parameters.values()),
                'Finite FP32 appearance parameters required')
        self.parameters = dict(parameters)
        self.second_moment = {k: torch.zeros_like(p) for k, p in parameters.items()}
        self.accepted_steps = 0

    def propose(self, gradients):
        require(set(gradients) == set(KEYS), 'Wrong gradient keys')
        moments, directions = {}, {}
        slope = 0.
        for key, parameter in self.parameters.items():
            g = gradients[key].detach()
            require(g.shape == parameter.shape and g.dtype == parameter.dtype and g.device == parameter.device
                    and bool(torch.isfinite(g).all()), 'Gradient shape/dtype/device/finite mismatch')
            value = BETA2*self.second_moment[key]+(1-BETA2)*g.square()
            direction = -RATES[key]*g/((value/(1-BETA2**(self.accepted_steps+1))).sqrt()+EPS)
            require(bool(torch.isfinite(value).all() & torch.isfinite(direction).all()), 'Nonfinite proposal')
            moments[key], directions[key] = value, direction
            slope += float((g.double()*direction.double()).sum())
        require(math.isfinite(slope) and slope < 0, 'A finite nonzero descent direction is required')
        return Proposal(moments, directions, slope, self.accepted_steps)

    @contextmanager
    def candidate(self, proposal, gradients, alpha):
        """Every alpha starts at the same accepted base after previous rollback."""
        require(alpha in ALPHAS and proposal.generation == self.accepted_steps, 'Stale or unregistered proposal')
        original = {k: p.detach().clone() for k, p in self.parameters.items()}
        trial = None
        clean_exit = False
        try:
            slope, maxima = 0., {}
            with torch.no_grad():
                for key, parameter in self.parameters.items():
                    parameter.copy_(original[key]+alpha*proposal.direction[key])
                    require(bool(torch.isfinite(parameter).all()), 'Nonfinite candidate')
                    displacement = parameter.double()-original[key].double()
                    slope += float((gradients[key].double()*displacement).sum())
                    maxima[key] = float(displacement.abs().max())
            trial = Trial(slope, maxima)
            yield trial
            clean_exit = True
        finally:
            if clean_exit and trial is not None and trial.accepted:
                self.second_moment = proposal.second_moment
                self.accepted_steps += 1
            else:
                with torch.no_grad():
                    for key, parameter in self.parameters.items():
                        parameter.copy_(original[key])


class PassExpired(RuntimeError):
    """Discard the current partial pass; never accept its candidate."""


@dataclass
class PassBudget:
    """Check time at view boundaries; acceptance still requires a complete pass."""
    clock: object
    max_passes: int = 40
    seconds: float = 360.
    passes: int = 0

    def __post_init__(self):
        self.started = self.clock()

    def can_start(self):
        return self.passes < self.max_passes and self.clock()-self.started < self.seconds

    def check_time(self):
        if self.clock()-self.started >= self.seconds:
            raise PassExpired('Full-batch wall-clock budget expired')

    def completed(self):
        require(self.passes < self.max_passes, 'Full-pass budget exceeded')
        self.passes += 1
        # The just-finished pass may be #40; it can still be accepted on time.
        return self.clock()-self.started <= self.seconds


def stream_mean_loss(items, loss_fn, *, backward, check_time=None):
    """One graph at a time; ordered Python-float accumulation, FP32 gradients."""
    require(len(items) > 0, 'Empty full-batch objective')
    total = 0.
    for item in items:
        if check_time is not None:
            check_time()
        loss = loss_fn(item)
        require(loss.ndim == 0 and loss.dtype == torch.float32 and bool(torch.isfinite(loss)),
                'Finite scalar FP32 view objective required')
        total += float(loss.detach())
        if backward:
            (loss/len(items)).backward()
        del loss
        if check_time is not None:
            check_time()
    return total/len(items)


def inference_checkpoint(base, model, metadata):
    """Preserve the base's scene metadata; observed provenance stays namespaced.

    Never invent a missing historical manifest_sha256/pixel_protocol declaration.
    The output is a full model, not an appearance delta or ordinary resume state.
    """
    require(set(model) == set(base['model']), 'Full model schema differs')
    values = {}
    for key, reference in base['model'].items():
        value = model[key].detach().cpu()
        require(value.shape == reference.shape and value.dtype == reference.dtype, 'Model shape/dtype differs')
        require(bool(torch.isfinite(value).all()), 'Nonfinite model tensor')
        if key not in KEYS:
            require(torch.equal(value, reference.cpu()), f'Frozen model tensor changed: {key}')
        values[key] = value.clone()
    require(set(KEYS) <= values.keys(), 'Missing appearance tensors')
    fields = ('format_version', 'config', 'step', 'scene_scale', 'feature_dim', 'sh_degree',
              'refiner_config', 'pixel_protocol', 'manifest_sha256')
    result = {key: copy.deepcopy(base[key]) for key in fields if key in base}
    result.update(model=values, training_cameras=base['training_cameras'].detach().cpu().clone(),
                  checkpoint_kind=KIND, appearance_optimization=copy.deepcopy(metadata))
    result['appearance_optimization']['ordinary_resume_allowed'] = False
    result['appearance_optimization']['optimizer_state_available'] = False
    return result


def save_full_inference(state, path):
    """Atomic, no-overwrite publication of one standalone inference checkpoint."""
    path = Path(path)
    require(state.get('checkpoint_kind') == KIND, 'Wrong derived checkpoint kind')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(state, temporary)
        # Link is atomic and raises if the final name exists; never replace it.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
