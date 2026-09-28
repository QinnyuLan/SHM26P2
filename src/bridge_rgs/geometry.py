"""Train-observation-only triangulation and bounded offline camera refinement."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .data import ColmapCamera, ColmapImage


@dataclass
class TrackObservation:
    image_id: int
    xy_normalized: np.ndarray
    xy_original: np.ndarray


@dataclass
class TriangulatedCloud:
    points: np.ndarray
    covariances: np.ndarray
    track_ids: np.ndarray
    reprojection_error: np.ndarray
    observations: list[list[TrackObservation]]
    audit: dict[str, Any]


def collect_train_tracks(images: dict[int, ColmapImage], cameras: dict[int, ColmapCamera],
                         splits: dict[int, str]) -> dict[int, list[TrackObservation]]:
    """The split gate precedes reading/undistorting any held-out observation."""
    tracks: dict[int, list[TrackObservation]] = defaultdict(list)
    for image_id, view in images.items():
        if splits[image_id] != "train":
            continue
        camera = cameras[view.camera_id]
        keep = view.point3d_ids >= 0
        if not np.any(keep):
            continue
        pixels = view.xy[keep]
        normalized = cv2.undistortPoints(pixels[:, None], camera.K, camera.distortion)[:, 0]
        for track_id, xy, original in zip(view.point3d_ids[keep], normalized, pixels):
            tracks[int(track_id)].append(TrackObservation(image_id, xy, original))
    return dict(tracks)


def triangulate_dlt(poses: np.ndarray, xy_normalized: np.ndarray) -> np.ndarray:
    """Normalized multiview homogeneous DLT in the original world frame."""
    rows = np.empty((len(poses) * 2, 4))
    rows[0::2] = xy_normalized[:, 0, None] * poses[:, 2] - poses[:, 0]
    rows[1::2] = xy_normalized[:, 1, None] * poses[:, 2] - poses[:, 1]
    _, _, vt = np.linalg.svd(rows, full_matrices=False)
    if abs(vt[-1, 3]) < 1e-12:
        return np.full(3, np.nan)
    return vt[-1, :3] / vt[-1, 3]


def project_points(points: np.ndarray, w2c: np.ndarray, K: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    pc = np.asarray(points) @ w2c[:3, :3].T + w2c[:3, 3]
    depth = pc[..., 2]
    pixels = pc @ K.T
    pixels = pixels[..., :2] / np.maximum(depth[..., None], 1e-10)
    return pixels, depth


def _reprojection(point: np.ndarray, poses: np.ndarray, xy: np.ndarray,
                  focals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pc = np.einsum("nij,j->ni", poses[:, :3, :3], point) + poses[:, :3, 3]
    z = pc[:, 2]
    residual = (pc[:, :2] / np.maximum(z[:, None], 1e-10) - xy) * focals
    return np.linalg.norm(residual, axis=1), z


def _point_covariance(point: np.ndarray, poses: np.ndarray, focals: np.ndarray,
                      residual_px: np.ndarray) -> np.ndarray:
    pc = np.einsum("nij,j->ni", poses[:, :3, :3], point) + poses[:, :3, 3]
    J = np.zeros((len(poses), 2, 3))
    J[:, 0, 0] = focals[:, 0] / pc[:, 2]
    J[:, 1, 1] = focals[:, 1] / pc[:, 2]
    J[:, 0, 2] = -focals[:, 0] * pc[:, 0] / pc[:, 2]**2
    J[:, 1, 2] = -focals[:, 1] * pc[:, 1] / pc[:, 2]**2
    J = (J @ poses[:, :3, :3]).reshape(-1, 3)
    sigma2 = max(.25, float(np.sum(residual_px**2) / max(2*len(poses)-3, 1)))
    H = J.T @ J
    return np.linalg.pinv(H + np.eye(3) * max(np.trace(H), 1.) * 1e-10) * sigma2


def triangulate_tracks(tracks: dict[int, list[TrackObservation]],
                       images: dict[int, ColmapImage], cameras: dict[int, ColmapCamera],
                       min_track_length: int = 3, max_points: int = 100000,
                       max_reprojection_error: float = 3., min_parallax_degrees: float = .5,
                       max_track_observations: int = 32, seed: int = 42,
                       pose_overrides: dict[int, np.ndarray] | None = None) -> TriangulatedCloud:
    """Robust DLT with positive-depth, parallax and train reprojection tests.

    All error thresholds are in the original image's pixels. Very long tracks
    are deterministically subsampled, so runtime and each track's influence are
    bounded. Random pair hypotheses rescue an initially contaminated fit.
    """
    rng = np.random.default_rng(seed)
    keys = np.array(sorted(k for k, obs in tracks.items()
                           if len({o.image_id for o in obs}) >= min_track_length), dtype=np.int64)
    # Shuffle before capping to avoid favoring low SfM track IDs or one region.
    rng.shuffle(keys)
    points, covariances, ids, errors, kept_obs = [], [], [], [], []
    rejected = defaultdict(int)
    poses_by_id = {i: (pose_overrides or {}).get(i, v.w2c)[:3] for i, v in images.items()}
    for track_id in keys:
        # A point can have at most one observation per image for DLT/BA.
        unique = {o.image_id: o for o in tracks[int(track_id)]}
        obs = sorted(unique.values(), key=lambda o: o.image_id)
        if len(obs) > max_track_observations:
            chosen = np.sort(rng.choice(len(obs), max_track_observations, replace=False))
            obs = [obs[i] for i in chosen]
        poses = np.stack([poses_by_id[o.image_id] for o in obs])
        xy = np.stack([o.xy_normalized for o in obs])
        focals = np.array([[cameras[images[o.image_id].camera_id].K[0, 0],
                            cameras[images[o.image_id].camera_id].K[1, 1]] for o in obs])
        candidate = triangulate_dlt(poses, xy)
        if not np.isfinite(candidate).all():
            rejected["degenerate"] += 1
            continue
        err, depth = _reprojection(candidate, poses, xy, focals)
        inlier = (err <= max_reprojection_error) & (depth > 1e-5)
        if inlier.sum() < max(min_track_length, len(obs)*.65):
            best_count, best_median = int(inlier.sum()), float(np.median(err[inlier])) if inlier.any() else np.inf
            for _ in range(min(16, len(obs)*2)):
                pair = rng.choice(len(obs), 2, replace=False)
                hypothesis = triangulate_dlt(poses[pair], xy[pair])
                if not np.isfinite(hypothesis).all():
                    continue
                e, z = _reprojection(hypothesis, poses, xy, focals)
                good = (e <= max_reprojection_error) & (z > 1e-5)
                count = int(good.sum())
                median = float(np.median(e[good])) if count else np.inf
                if count > best_count or (count == best_count and median < best_median):
                    candidate, inlier, best_count, best_median = hypothesis, good, count, median
        for _ in range(3):
            if inlier.sum() < min_track_length:
                break
            candidate = triangulate_dlt(poses[inlier], xy[inlier])
            err, depth = _reprojection(candidate, poses, xy, focals)
            updated = (err <= max_reprojection_error) & (depth > 1e-5)
            if np.array_equal(updated, inlier):
                break
            inlier = updated
        if inlier.sum() < min_track_length or not np.isfinite(candidate).all():
            rejected["reprojection_or_depth"] += 1
            continue
        # One final solve ensures the reported point uses exactly saved observations.
        candidate = triangulate_dlt(poses[inlier], xy[inlier])
        err, depth = _reprojection(candidate, poses, xy, focals)
        if np.any(err[inlier] > max_reprojection_error) or np.any(depth[inlier] <= 1e-5):
            rejected["unstable_refit"] += 1
            continue
        iposes = poses[inlier]
        centers = -np.einsum("nji,nj->ni", iposes[:, :3, :3], iposes[:, :3, 3])
        rays = candidate[None] - centers
        rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
        cosine = np.clip((rays @ rays.T).min(), -1., 1.)
        parallax = np.degrees(np.arccos(cosine))
        if parallax < min_parallax_degrees:
            rejected["low_parallax"] += 1
            continue
        points.append(candidate)
        covariances.append(_point_covariance(candidate, iposes, focals[inlier], err[inlier]))
        ids.append(track_id)
        errors.append(float(np.sqrt(np.mean(err[inlier]**2))))
        kept_obs.append([o for o, keep in zip(obs, inlier) if keep])
        if max_points and len(points) >= max_points:
            break
    return TriangulatedCloud(
        np.asarray(points, dtype=np.float64).reshape(-1, 3),
        np.asarray(covariances, dtype=np.float64).reshape(-1, 3, 3),
        np.asarray(ids, dtype=np.int64), np.asarray(errors, dtype=np.float64), kept_obs,
        {"candidate_tracks": len(keys), "accepted_points": len(points), "rejected": dict(rejected),
         "max_reprojection_error_native_px": max_reprojection_error,
         "min_parallax_degrees": min_parallax_degrees,
         "min_track_length": min_track_length, "max_track_observations": max_track_observations,
         "point_covariance": "conditional Gauss-Newton at fixed cameras; residual scaled with 0.5px floor; not marginal SfM posterior"})


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def camera_uncertainty(cloud: TriangulatedCloud, images: dict[int, ColmapImage],
                       cameras: dict[int, ColmapCamera], splits: dict[int, str],
                       pose_overrides: dict[int, np.ndarray] | None = None
                       ) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Approximate conditional pose uncertainty from train residuals + priors.

    This intentionally makes no calibrated coverage or full SfM-posterior claim.
    Held-out cameras receive a prior only; their pixels are never consulted.
    """
    train_centers = np.stack([v.center for i, v in images.items() if splits[i] == "train"])
    scene_radius = max(float(np.percentile(np.linalg.norm(train_centers - np.median(train_centers, axis=0), axis=1), 90)), 1e-3)
    prior_std = np.array([.005]*3 + [.005*scene_radius]*3)
    prior_precision = np.diag(1 / prior_std**2)
    hessians = {i: prior_precision.copy() for i in images}
    counts = {i: 0 for i in images}
    sigma_px = max(.5, float(np.median(cloud.reprojection_error)) / np.sqrt(2*np.log(2))) if len(cloud.points) else 1.
    for point, observations in zip(cloud.points, cloud.observations):
        for o in observations:
            if splits[o.image_id] != "train":
                raise AssertionError("Held-out observation in camera uncertainty")
            view = images[o.image_id]
            pose = (pose_overrides or {}).get(o.image_id, view.w2c)
            pc = pose[:3, :3] @ point + pose[:3, 3]
            K = cameras[view.camera_id].K
            x, y, z = pc
            Jp = np.array([[K[0, 0]/z, 0, -K[0, 0]*x/z**2],
                           [0, K[1, 1]/z, -K[1, 1]*y/z**2]])
            J = Jp @ np.concatenate([-_skew(pc), np.eye(3)], axis=1)
            hessians[o.image_id] += J.T @ J / sigma_px**2
            counts[o.image_id] += 1
    covariances = {i: np.linalg.inv(H) for i, H in hessians.items()}
    return covariances, {"method": "conditional fixed-point Gauss-Newton with camera priors",
                         "parameter_order": ["rx", "ry", "rz", "tx", "ty", "tz"],
                         "perturbation": "left camera-frame SE(3)",
                         "residual_sigma_native_px": sigma_px,
                         "prior_std": prior_std.tolist(), "scene_radius": scene_radius,
                         "train_observation_counts": {str(i): n for i, n in counts.items()},
                         "held_out": "prior only; no validation observations",
                         "limitation": "Ignores point-camera correlations, shared intrinsics, SfM selection and systematic error. This is a quality proxy, not a calibrated posterior."}


def bounded_bundle_adjustment(cloud: TriangulatedCloud, images: dict[int, ColmapImage],
                              cameras: dict[int, ColmapCamera], splits: dict[int, str],
                              max_points: int = 500, max_nfev: int = 20,
                              seed: int = 42) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Sparse robust BA with two fixed camera poses fixing the similarity gauge.

    Intrinsics remain fixed. Camera steps and point movement are bounded, with
    conservative pose priors. Only original training tracks are accepted.
    Refinement is accepted only if raw train reprojection RMS does not worsen.
    """
    if not len(cloud.points):
        return {}, {"enabled": True, "accepted": False, "reason": "no points"}
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(cloud.points), min(max_points, len(cloud.points)), replace=False))
    observed_ids = sorted({o.image_id for i in selected for o in cloud.observations[i]})
    if any(splits[i] != "train" for i in observed_ids):
        raise AssertionError("Held-out camera included in bundle adjustment")
    if len(observed_ids) < 3:
        return {}, {"enabled": True, "accepted": False, "reason": "fewer than three observed training cameras"}
    centers = np.stack([images[i].center for i in observed_ids])
    first = int(np.argmax(np.linalg.norm(centers - np.median(centers, axis=0), axis=1)))
    second = int(np.argmax(np.linalg.norm(centers - centers[first], axis=1)))
    anchors = {observed_ids[first], observed_ids[second]}
    optimized = [i for i in observed_ids if i not in anchors]
    camera_index = {i: j for j, i in enumerate(observed_ids)}
    optimized_index = {i: j for j, i in enumerate(optimized)}
    original_poses = np.stack([images[i].w2c for i in observed_ids])
    scale = max(float(np.percentile(np.linalg.norm(centers - np.median(centers, axis=0), axis=1), 90)), 1e-3)
    base_points = cloud.points[selected]
    obs_camera, obs_point, obs_xy, obs_f = [], [], [], []
    for point_index, ci in enumerate(selected):
        for o in cloud.observations[ci]:
            K = cameras[images[o.image_id].camera_id].K
            obs_camera.append(camera_index[o.image_id]); obs_point.append(point_index)
            obs_xy.append(o.xy_normalized); obs_f.append([K[0, 0], K[1, 1]])
    obs_camera, obs_point = np.array(obs_camera), np.array(obs_point)
    obs_xy, obs_f = np.array(obs_xy), np.array(obs_f)
    ncam = len(optimized)
    opt_rows = np.array([camera_index[i] for i in optimized])
    bounds_camera = np.array([.01]*3 + [.01*scale]*3)
    prior_std = bounds_camera / 3.
    nvars = ncam*6 + len(selected)*3

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        poses = original_poses.copy()
        delta = x[:ncam*6].reshape(-1, 6)
        rotations = Rotation.from_rotvec(delta[:, :3]).as_matrix()
        poses[opt_rows, :3, :3] = rotations @ original_poses[opt_rows, :3, :3]
        poses[opt_rows, :3, 3] = np.einsum("nij,nj->ni", rotations, original_poses[opt_rows, :3, 3]) + delta[:, 3:]
        return poses, base_points + x[ncam*6:].reshape(-1, 3)

    def residual(x: np.ndarray, priors: bool = True) -> np.ndarray:
        poses, points = unpack(x)
        P = poses[obs_camera]
        pc = np.einsum("nij,nj->ni", P[:, :3, :3], points[obs_point]) + P[:, :3, 3]
        z = np.maximum(pc[:, 2, None], 1e-5)
        reproj = ((pc[:, :2]/z - obs_xy) * obs_f).ravel()
        if not priors:
            return reproj
        return np.concatenate([reproj, (x[:ncam*6].reshape(-1, 6)/prior_std).ravel()])

    sparsity = lil_matrix((len(obs_camera)*2 + ncam*6, nvars), dtype=np.int8)
    for j, (camera_row, point_row) in enumerate(zip(obs_camera, obs_point)):
        image_id = observed_ids[camera_row]
        if image_id in optimized_index:
            k = optimized_index[image_id]*6
            sparsity[j*2:j*2+2, k:k+6] = 1
        k = ncam*6 + point_row*3
        sparsity[j*2:j*2+2, k:k+3] = 1
    for j in range(ncam*6):
        sparsity[len(obs_camera)*2+j, j] = 1
    bound = np.concatenate([np.tile(bounds_camera, ncam), np.full(len(selected)*3, .05*scale)])
    x0 = np.zeros(nvars)
    before = float(np.sqrt(np.mean(residual(x0, False)**2)))
    result = least_squares(residual, x0, jac_sparsity=sparsity.tocsr(),
                           bounds=(-bound, bound), loss="soft_l1", f_scale=1.,
                           max_nfev=max_nfev, x_scale="jac", ftol=1e-5)
    after = float(np.sqrt(np.mean(residual(result.x, False)**2)))
    accepted = bool(np.isfinite(after) and after <= before and np.isfinite(result.x).all())
    refined, _ = unpack(result.x)
    overrides = {i: refined[camera_index[i]] for i in optimized} if accepted else {}
    return overrides, {"enabled": True, "accepted": accepted, "success": bool(result.success),
                        "solver_message": result.message, "nfev": result.nfev,
                        "selected_points": len(selected), "train_observations": len(obs_camera),
                        "optimized_cameras": len(optimized), "fixed_anchor_image_ids": sorted(anchors),
                        "gauge": "two original camera poses fixed; shared intrinsics fixed",
                        "rms_native_px_before": before, "rms_native_px_after": after,
                        "camera_delta_bounds": bounds_camera.tolist(),
                        "camera_prior_std": prior_std.tolist(),
                        "note": "Acceptance is train reprojection only; no validation pixel or metric used."}
