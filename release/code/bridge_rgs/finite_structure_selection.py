"""Pure finite-action decisions; A selects once, B only checks the locked choices.

Rows describe the same fixed support for every state. Empty local support is
None, never a zero-valued MSE. This module reads no files and runs no models.
The worker must verify repeats are the identical fitted keep endpoint, and
bind the actual RGB valid support. Those facts cannot be proved by scalar rows.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import numpy as np

SPEC = {
    "protocol": "finite_structure_selection_v1",
    "keep_id": "keep",
    "repeat_multiplier": 10.0,
    "relative_floor": 1e-6,
    "absolute_floor": 1e-12,
    "per_view_rgb_factor": 1.001,
    "per_view_add_tau": False,
    "repeat_state": "identical fitted keep endpoint; worker verifies state SHA",
    "empty_local": "skip this view only; require matching counts and support SHA in all states",
    "ties": "candidate ID lexicographic; refit requires strict CE improvement over keep",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, label):
    _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)), label)
    value = float(value)
    _require(math.isfinite(value) and value >= 0.0, label)
    return value


def _rows(rows):
    _require(isinstance(rows, Sequence) and len(rows) > 0, "Empty view population")
    result = []
    for row in rows:
        _require(isinstance(row, Mapping), "Expected metric row")
        name = row["name"]
        _require(isinstance(name, str) and bool(name), "Invalid view name")
        count = row["local_pixels"]
        _require(isinstance(count, Integral) and not isinstance(count, (bool, np.bool_))
                 and count >= 0, "Invalid fixed support count")
        support_sha = row.get("local_support_sha256")
        _require(isinstance(support_sha, str) and len(support_sha) == 64
                 and all(c in "0123456789abcdef" for c in support_sha),
                 "Missing or invalid fixed support SHA256")
        local = row["local_rgb_mse"]
        if count == 0:
            _require(local is None, "Empty support must use None, not zero MSE")
        else:
            local = _number(local, "Nonfinite or negative local RGB MSE")
        cm = np.asarray(row["confusion_matrix"])
        _require(cm.shape == (5, 5) and cm.dtype.kind in "iu" and np.all(cm >= 0),
                 "Expected nonnegative integer 5x5 confusion matrix")
        _require(sum(int(x) for x in cm.flat) <= np.iinfo(np.int64).max,
                 "Confusion counts exceed int64")
        result.append({"name": name, "local_pixels": int(count), "local_rgb_mse": local,
                       "local_support_sha256": support_sha,
                       "full_rgb_mse": _number(row["full_rgb_mse"], "Invalid full RGB MSE"),
                       "raw_ce": _number(row["raw_ce"], "Invalid raw CE"),
                       "confusion_matrix": cm.astype(np.int64)})
    _require(len({r["name"] for r in result}) == len(result), "Duplicate view name")
    return result


def _aligned(reference, rows):
    _require([(r["name"], r["local_pixels"], r["local_support_sha256"]) for r in reference]
             == [(r["name"], r["local_pixels"], r["local_support_sha256"]) for r in rows],
             "View order, fixed support counts or hashes differ between states")
    _require(all(np.array_equal(ref["confusion_matrix"].sum(axis=1),
                                row["confusion_matrix"].sum(axis=1))
                 for ref, row in zip(reference, rows, strict=True)),
             "GT class support differs between states")


def _support_records(rows):
    return [{key: row[key] for key in ("name", "local_pixels", "local_support_sha256")}
            for row in rows]


def validate_fixed_support(reference_rows, *other_states):
    """Also usable by the worker to check zero-step versus endpoint/repeat rows.

    The caller must hash the actual mask and ensure it is not mutated; equality
    of reported hashes alone cannot verify an unprovided pixel array.
    """
    reference = _rows(reference_rows)
    for state in other_states:
        _aligned(reference, _rows(state))
    return _support_records(reference)


def _mean(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return math.fsum(x / len(values) for x in values) if values else None


def _summarize(rows):
    total = sum(int(x) for row in rows for x in row["confusion_matrix"].flat)
    _require(total <= np.iinfo(np.int64).max, "Pooled confusion counts exceed int64")
    cm = np.sum([r["confusion_matrix"] for r in rows], axis=0, dtype=np.int64)
    tp = int(cm[2, 2])
    fp, fn = int(cm[:, 2].sum()) - tp, int(cm[2].sum()) - tp
    return {"raw_ce": _mean(rows, "raw_ce"),
            "full_rgb_mse": _mean(rows, "full_rgb_mse"),
            "local_rgb_mse": _mean(rows, "local_rgb_mse"),
            "local_views": sum(r["local_pixels"] > 0 for r in rows),
            "confusion_matrix": cm.tolist(), "cable_tp": tp, "cable_fp": fp, "cable_fn": fn,
            "cable_iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "cable_precision": tp / (tp + fp) if tp + fp else None,
            "cable_recall": tp / (tp + fn) if tp + fn else None}


def _rgb_guard(reference, candidate, keep, repeat):
    aggregates = {}
    for key in ("full_rgb_mse", "local_rgb_mse"):
        ref, new = _mean(reference, key), _mean(candidate, key)
        if ref is None:
            aggregates[key] = {"skipped_empty_support": True, "passed": True}
            continue
        noise = abs(_mean(keep, key) - _mean(repeat, key))
        tau = max(SPEC["repeat_multiplier"] * noise, SPEC["relative_floor"] * ref,
                  SPEC["absolute_floor"])
        aggregates[key] = {"reference": ref, "candidate": new, "difference": new-ref,
                           "keep_repeat_difference": noise, "tau": tau,
                           "skipped_empty_support": False, "passed": bool(new-ref <= tau)}
    per_view = [{"name": ref["name"], "reference": ref["full_rgb_mse"],
                 "candidate": new["full_rgb_mse"],
                 "limit": SPEC["per_view_rgb_factor"] * ref["full_rgb_mse"],
                 "passed": bool(new["full_rgb_mse"]
                                <= SPEC["per_view_rgb_factor"] * ref["full_rgb_mse"])}
                for ref, new in zip(reference, candidate, strict=True)]
    return {"aggregate": aggregates, "per_view": per_view,
            "passed": all(r["passed"] for r in aggregates.values())
                      and all(r["passed"] for r in per_view)}


def _inputs(keep_rows, repeat_rows, candidate_rows):
    _require(isinstance(candidate_rows, Mapping), "Expected candidate map")
    _require(all(isinstance(k, str) and k and k != "keep" for k in candidate_rows),
             "Invalid candidate IDs")
    keep, repeat = _rows(keep_rows), _rows(repeat_rows)
    candidates = {key: _rows(candidate_rows[key]) for key in sorted(candidate_rows)}
    for rows in [repeat, *candidates.values()]:
        _aligned(keep, rows)
    return keep, repeat, candidates


def select_on_a(keep_rows, repeat_rows, candidate_rows, gradient_scores):
    """Return JSON-safe fixed A choices, with one common RGB feasible set."""
    keep, repeat, candidates = _inputs(keep_rows, repeat_rows, candidate_rows)
    _require(isinstance(gradient_scores, Mapping) and set(gradient_scores) == set(candidates),
             "Gradient score IDs differ from candidate IDs")
    scores = {key: _number(gradient_scores[key], "Invalid gradient score") for key in candidates}
    guards = {key: _rgb_guard(keep, rows, keep, repeat) for key, rows in candidates.items()}
    feasible = [key for key in candidates if guards[key]["passed"]]
    summaries = {"keep": _summarize(keep),
                 **{key: _summarize(rows) for key, rows in candidates.items()}}
    gradient = min(feasible, key=lambda key: (-scores[key], key)) if feasible else "keep"
    refit = min(feasible, key=lambda key: (summaries[key]["raw_ce"], key)) if feasible else "keep"
    if summaries[refit]["raw_ce"] >= summaries["keep"]["raw_ce"]:
        refit = "keep"
    return {"protocol": SPEC["protocol"], "phase": "A_choices_locked",
            "candidate_ids": list(candidates), "view_names": [r["name"] for r in keep],
            "fixed_support": _support_records(keep),
            "feasible": feasible, "gradient_scores": scores, "rgb_guards": guards,
            "summaries": summaries, "gradient_choice": gradient, "refit_choice": refit}


def evaluate_on_b(selection, keep_rows, repeat_rows, candidate_rows):
    """Evaluate the two A choices verbatim; never rank or replace using B."""
    keep, repeat, candidates = _inputs(keep_rows, repeat_rows, candidate_rows)
    _require(selection.get("protocol") == SPEC["protocol"]
             and selection.get("phase") == "A_choices_locked", "Missing locked A selection")
    _require(selection["candidate_ids"] == list(candidates), "A/B candidate IDs changed")
    _require(not set(selection["view_names"]) & {r["name"] for r in keep},
             "B camera overlaps A")
    states = {"keep": keep, **candidates}
    gradient, refit = selection["gradient_choice"], selection["refit_choice"]
    _require(gradient in states and refit in states, "Unknown locked choice")
    summaries = {key: _summarize(rows) for key, rows in states.items()}
    checks = {}
    for role, reference in (("keep", "keep"), ("gradient", gradient)):
        old, new = summaries[reference], summaries[refit]
        cable = (new["cable_iou"] is not None and old["cable_iou"] is not None
                 and new["cable_iou"] > old["cable_iou"])
        rgb = _rgb_guard(states[reference], states[refit], keep, repeat)
        checks[role] = {"reference_id": reference, "candidate_id": refit,
                        "raw_ce_difference": new["raw_ce"]-old["raw_ce"],
                        "cable_iou_difference": (new["cable_iou"]-old["cable_iou"]
                                                 if new["cable_iou"] is not None
                                                 and old["cable_iou"] is not None else None),
                        "raw_ce_improved": bool(new["raw_ce"] < old["raw_ce"]),
                        "cable_iou_improved": bool(cable), "rgb_guard": rgb}
    distinct = gradient != refit
    passed = distinct and all(row["raw_ce_improved"] and row["cable_iou_improved"]
                              and row["rgb_guard"]["passed"] for row in checks.values())
    status = ("necessary_signal_present" if passed else
              "selection_not_distinguished" if not distinct else "specified_action_not_supported")
    return {"protocol": SPEC["protocol"], "phase": "B_evaluation_no_reselection",
            "gradient_choice": gradient, "refit_choice": refit, "summaries": summaries,
            "fixed_support": _support_records(keep),
            "comparisons": checks, "choices_differ": distinct, "passed": bool(passed),
            "status": status, "scope": "finite-budget local TRAIN transfer, not capacity or VAL"}
