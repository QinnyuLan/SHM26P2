"""Stateless learning-rate control for the three semantic optimizer groups."""

from __future__ import annotations

import math


def normalize_semantic_schedule(config):
    """Return the explicit stage contract; absent config preserves legacy behavior."""
    value = config.get("semantic_lr_schedule")
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"type", "final_multiplier"}:
        raise ValueError("semantic_lr_schedule requires only type and final_multiplier")
    kind = value.get("type", "constant")
    final = value.get("final_multiplier", .1)
    steps = config.get("steps", 30000)
    if kind not in {"constant", "cosine"}:
        raise ValueError("semantic_lr_schedule.type must be constant or cosine")
    if isinstance(final, bool) or not isinstance(final, (int, float)) or not math.isfinite(final) or not 0 < final <= 1:
        raise ValueError("semantic_lr_schedule.final_multiplier must be finite in (0, 1]")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < (2 if kind == "cosine" else 1):
        raise ValueError("Semantic schedule needs integer steps, at least two for cosine")
    return {"type": kind, "final_multiplier": float(final), "steps": steps}


def validate_semantic_schedule_resume(saved_config, requested_config):
    """Forbid changing a stage's schedule, endpoint, or enabled state on strict resume."""
    if normalize_semantic_schedule(saved_config) != normalize_semantic_schedule(requested_config):
        raise ValueError("Strict resume cannot change semantic LR schedule; use warmstart")


def semantic_multiplier(spec, step):
    if spec is None:
        return 1.
    if isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= spec["steps"]:
        raise ValueError("Semantic LR step must be inside the fixed stage horizon")
    if spec["type"] == "constant":
        return 1.
    progress = (step - 1) / (spec["steps"] - 1)
    return spec["final_multiplier"] + (1 - spec["final_multiplier"]) * (1 + math.cos(math.pi * progress)) / 2


class SemanticLRSchedule:
    """Capture fresh optimizer base rates before loading a resumed optimizer state.

    Only sem_features and heads[0:2] are ever changed. The background head,
    appearance, geometry, optimizer moments and parameter values are untouched.
    Explicit schedules save immutable base rates in those three parameter groups.
    No config means no optimizer mutation and no new diagnostic fields.
    """

    def __init__(self, optimizers, config):
        self.spec = normalize_semantic_schedule(config)
        self.groups = []
        self.base_rates = []
        self.optimizers = optimizers
        if self.spec is None:
            return
        if len(optimizers["sem_features"].param_groups) != 1 or len(optimizers["heads"].param_groups) != 3:
            raise ValueError("Semantic scheduler requires one feature and three head optimizer groups")
        self.groups = [optimizers["sem_features"].param_groups[0], *optimizers["heads"].param_groups[:2]]
        self.base_rates = [float(group["lr"]) for group in self.groups]
        if any(not math.isfinite(rate) or rate <= 0 for rate in self.base_rates):
            raise ValueError("Semantic optimizer base learning rates must be finite and positive")

    def initialize(self, resume_step=None):
        """Run after optimizer load: validate saved rates before rebinding groups."""
        if self.spec is None:
            return
        # Optimizer.load_state_dict replaces the group dictionaries.
        groups = [self.optimizers["sem_features"].param_groups[0], *self.optimizers["heads"].param_groups[:2]]
        if resume_step is not None:
            multiplier = semantic_multiplier(self.spec, resume_step)
            for group, base in zip(groups, self.base_rates, strict=True):
                if group.get("semantic_base_lr") != base or group["lr"] != base * multiplier:
                    raise ValueError("Resumed semantic optimizer LR/base does not match its saved schedule")
        else:
            for group, base in zip(groups, self.base_rates, strict=True):
                group["semantic_base_lr"] = base
        self.groups = groups

    def apply(self, step):
        if self.spec is None:
            return {}
        multiplier = semantic_multiplier(self.spec, step)
        for group, base in zip(self.groups, self.base_rates, strict=True):
            group["lr"] = base * multiplier
        return {"semantic_lr_multiplier": multiplier,
                "semantic_lr_features": self.groups[0]["lr"],
                "semantic_lr_classifier": self.groups[1]["lr"],
                "semantic_lr_refiner": self.groups[2]["lr"],
                "semantic_lr_background": self.optimizers["heads"].param_groups[2]["lr"]}
