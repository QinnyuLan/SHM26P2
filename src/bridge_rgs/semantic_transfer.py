"""Fixed 1NN semantic warmstart between checkpoints in exactly the same frame.

This is an engineering transfer control. Geometric proximity is not semantic
confidence; protected target geometry, appearance and TRAIN evidence stay intact.
"""
from __future__ import annotations

import copy
import re

import numpy as np
import torch
from scipy.spatial import cKDTree

from .averaging import is_semantic_parameter
from .refinement import normalize_refiner_config


def _quantiles(array):
    return {str(q): float(np.quantile(array, q)) for q in [0, .5, .9, .95, .99, 1]}


def nearest_indices(source_points, target_points):
    """One nearest feature; resolve exact distance ties by original source index.

    The second distance is inspected only to find ties, never averaged or used
    to choose between k hyperparameters. This policy has no validation input.
    """
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if len(source) == 0 or len(target) == 0:
        raise ValueError("Cannot transfer an empty point set")
    tree = cKDTree(source)
    if len(source) == 1:
        distance, index = tree.query(target, k=1, workers=8)
        return distance, index, 0
    distances, neighbors = tree.query(target, k=2, workers=8)
    distance, index = distances[:, 0].copy(), neighbors[:, 0].copy()
    possible_ties = np.flatnonzero(np.isclose(distances[:, 0], distances[:, 1],
                                             rtol=1e-12, atol=1e-12))
    ties = 0
    for i in possible_ties:
        candidates = np.asarray(tree.query_ball_point(
            target[i], distance[i] + 1e-12 * max(distance[i], 1)), dtype=np.int64)
        squared = ((source[candidates] - target[i]) ** 2).sum(1)
        exact = candidates[squared == squared.min()]
        index[i] = exact.min()
        distance[i] = np.sqrt(squared.min())
        ties += int(len(exact) > 1)
    return distance, index, ties


def transfer_semantic_1nn(source, target, *, source_manifest_sha256, target_manifest_sha256):
    """Return a new inference/warmstart-only state; do not mutate either input."""
    from .coordinates import require_matching_protocol
    require_matching_protocol(source, target, "semantic transfer")
    hashes = (source_manifest_sha256, target_manifest_sha256)
    if any(re.fullmatch(r"[0-9a-f]{64}", value or "") is None for value in hashes):
        raise ValueError("Both manifest SHA256 values are required")
    if source_manifest_sha256 != target_manifest_sha256:
        raise ValueError("Source and target manifest SHA256 differ")
    if any(state.get("manifest_sha256") and state["manifest_sha256"] != manifest_hash
           for state, manifest_hash in zip((source, target), hashes)):
        raise ValueError("Stored checkpoint manifest SHA256 differs from transfer provenance")
    if source["config"]["manifest"] != target["config"]["manifest"]:
        raise ValueError("Source and target TRAIN manifest paths differ")
    for key in ("format_version", "feature_dim", "sh_degree", "scene_scale"):
        if source.get(key) != target.get(key):
            raise ValueError(f"Source/target frame or architecture mismatch: {key}")
    if not np.isfinite(target["scene_scale"]) or target["scene_scale"] <= 0:
        raise ValueError("Invalid scene scale")
    if not torch.equal(source["training_cameras"], target["training_cameras"]):
        raise ValueError("Source and target training cameras differ")
    if not torch.isfinite(target["training_cameras"]).all():
        raise ValueError("Nonfinite camera")
    sm, tm = source["model"], target["model"]
    required = {"splats.means", "splats.quats", "splats.log_scales", "splats.opacity_logits",
                "splats.sh0", "splats.sh_rest", "background_logits", "semantic_prior_counts",
                "splats.sem_features", "semantic_decoder.weight", "semantic_decoder.bias"}
    if not required.issubset(sm) or not required.issubset(tm):
        raise ValueError("Incomplete Gaussian/semantic state")
    if {key for key in sm if not is_semantic_parameter(key)} != {
            key for key in tm if not is_semantic_parameter(key)}:
        raise ValueError("Protected source/target model schemas differ")
    for model in (sm, tm):
        for key, value in model.items():
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite or invalid state tensor: {key}")
        n = len(model["splats.means"])
        dimensions = {"splats.means": (n, 3), "splats.quats": (n, 4),
                      "splats.log_scales": (n, 3), "splats.opacity_logits": (n,),
                      "semantic_prior_counts": (n, 5),
                      "splats.sem_features": (n, target["feature_dim"]),
                      "semantic_decoder.weight": (5, target["feature_dim"]),
                      "semantic_decoder.bias": (5,)}
        if any(tuple(model[key].shape) != shape for key, shape in dimensions.items()):
            raise ValueError("Semantic/geometry parameter dimensions do not match metadata")
        if not any(key.startswith("refiner.") for key in model):
            raise ValueError("Missing refiner parameters")
        if (model["splats.quats"].norm(dim=-1) == 0).any():
            raise ValueError("Invalid zero quaternion")
    for key in ("splats.means", "splats.sem_features", "semantic_decoder.weight",
                "semantic_decoder.bias"):
        if sm[key].dtype != tm[key].dtype:
            raise ValueError(f"Source/target dtype mismatch: {key}")
    source_refiner = normalize_refiner_config(source.get("refiner_config"))
    config_refiner = source["config"].get("refiner", source_refiner)
    if normalize_refiner_config(config_refiner) != source_refiner:
        raise ValueError("Source refiner architecture metadata disagree")
    from .model import RefinementHead
    from .refinement import MultiScaleRefinementHead
    with torch.random.fork_rng(devices=[]):
        if source_refiner["type"] == "legacy":
            head = RefinementHead(source["feature_dim"],
                                  residual_bound=source_refiner.get("residual_bound", 1.))
        else:
            head = MultiScaleRefinementHead(source["feature_dim"], **{
                key: value for key, value in source_refiner.items() if key != "type"})
    expected = head.state_dict()
    actual = {key.removeprefix("refiner."): value for key, value in sm.items()
              if key.startswith("refiner.")}
    if (actual.keys() != expected.keys() or any(
            actual[key].shape != expected[key].shape for key in expected)):
        raise ValueError("Source refiner tensor schema disagrees with architecture metadata")
    points = sm["splats.means"].detach().cpu().double().numpy()
    target_points = tm["splats.means"].detach().cpu().double().numpy()
    distance, index, ties = nearest_indices(points, target_points)
    model = {key: value.detach().cpu().clone() for key, value in tm.items()
             if not is_semantic_parameter(key)}
    model["splats.sem_features"] = sm["splats.sem_features"].detach().cpu()[index].clone()
    for key, value in sm.items():
        if key.startswith(("semantic_decoder.", "refiner.")):
            model[key] = value.detach().cpu().clone()
    protected = [key for key in tm if not is_semantic_parameter(key)]
    if any(not torch.equal(tm[key].detach().cpu(), model[key]) for key in protected):
        raise AssertionError("Protected target state changed")
    scales = sm["splats.log_scales"].detach().cpu().double().exp().numpy()[index]
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Invalid source Gaussian shape scale")
    q = sm["splats.quats"].detach().cpu().double().numpy()[index]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    rotation = np.stack([
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=-1).reshape(-1, 3, 3)
    local = np.einsum("nji,nj->ni", rotation, target_points - points[index])
    shape_distance = np.linalg.norm(local / scales, axis=1)
    copied_config = copy.deepcopy(target["config"])
    copied_config["refiner"] = source_refiner
    copied_config["warmstart_reset_refiner"] = False
    # No stale source/target stage optimization permission is implied by a transfer.
    copied_config["refiner_field_grad"] = source["config"].get("refiner_field_grad", True)
    result = {key: copy.deepcopy(target[key]) for key in (
        "format_version", "feature_dim", "sh_degree", "scene_scale")}
    for key in ("pixel_protocol", "manifest_sha256"):
        if key in target:
            result[key] = copy.deepcopy(target[key])
    result.update(model=model, config=copied_config, step=0, refiner_config=source_refiner,
                  training_cameras=target["training_cameras"].detach().cpu().clone(),
                  checkpoint_kind="semantic_transfer_inference_or_warmstart",
                  semantic_transfer={
                      "method": "fixed Euclidean 1NN; exact ties use lowest source index",
                      "source_step": int(source["step"]), "target_step": int(target["step"]),
                      "source_points": len(points), "target_points": len(target_points),
                      "manifest_sha256": target_manifest_sha256,
                      "protected_target_keys": sorted(protected),
                      "protected_target_tensors_bitwise_equal": True,
                      "training_cameras_bitwise_equal": True,
                      "optimizer_state_available": False, "mapped_point_count": len(index),
                      "exact_nearest_ties_resolved": ties,
                      "unique_source_points_used": len(np.unique(index)),
                      "distance_world_quantiles": _quantiles(distance),
                      "distance_over_scene_scale_quantiles": _quantiles(
                          distance / target["scene_scale"]),
                      "nearest_source_shape_distance_quantiles": _quantiles(shape_distance),
                      "within_nearest_source_3sigma_shape_fraction": float(
                          (shape_distance <= 3).mean()),
                      "coverage_caveat": "Shape coverage is not position uncertainty or semantic "
                          "confidence; all points are transferred without a validation-tuned gate."})
    return result
