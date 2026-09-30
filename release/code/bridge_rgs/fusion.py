"""Visibility- and projection-aware fusion of independent DINOv3 evidence."""
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.nn import functional as F

from .coordinates import pixel_protocol, require_matching_protocol, resize_discrete_tensor
from .reliability import (
    ellipse_sample_probabilities,
    fuse_multiview_evidence,
    projection_jacobians,
    propagate_projection_covariance,
)


@torch.no_grad()
def fuse_scene_evidence(scene, views, cameras, pseudo_dir, max_points=4096,
                        uncertainty=True, init_path=None):
    if len({pixel_protocol(view) for view in views}) > 1:
        raise ValueError("Fusion cannot mix views from different pixel protocols")
    protocol = pixel_protocol(views[0]) if views else pixel_protocol()
    require_matching_protocol(getattr(scene, "pixel_protocol", None), protocol, "fusion field and views")
    candidates = [i for i, v in enumerate(views)
                  if (Path(pseudo_dir) / (Path(v["name"]).stem + ".npz")).exists()]
    if len(candidates) < 2:
        return None
    anchor = candidates[int(np.random.randint(len(candidates)))]
    centers = -torch.einsum("nij,nj->ni", cameras[:, :3, :3].transpose(-1, -2), cameras[:, :3, 3])
    distances = (centers - centers[anchor]).norm(dim=-1)
    order = sorted(candidates, key=lambda i: float(distances[i]))[:4]
    ids = torch.randperm(len(scene.splats["means"]), device="cuda")[:max_points]
    xyz = scene.splats["means"][ids].detach()
    # Conditional triangulation uncertainty is a proxy, never splat spatial scale.
    point_cov = torch.eye(3, device="cuda")[None].expand(len(ids), -1, -1) * (scene.scene_scale * .0005) ** 2
    if init_path:
        with np.load(init_path) as cloud:
            prior_protocol = str(cloud["pixel_protocol"].item()) if "pixel_protocol" in cloud else None
            require_matching_protocol(protocol, prior_protocol, "fusion position-covariance source")
            nearest = cKDTree(cloud["points"]).query(xyz.cpu().numpy())[1]
            point_cov = torch.tensor(cloud["covariances"][nearest], device="cuda").float()
    probabilities, weights = [], []
    for i in order:
        view = views[i]
        protocol = pixel_protocol(view)
        with np.load(Path(pseudo_dir) / (Path(view["name"]).stem + ".npz")) as stored:
            p = torch.tensor(stored["probs"].astype(np.float32), device="cuda")
            valid = torch.tensor(stored["valid"].astype(np.float32), device="cuda")
            teacher_confidence = torch.tensor(stored["confidence"].astype(np.float32), device="cuda")
        full_h, full_w = p.shape[-2:]
        width = min(480, full_w)
        height = round(full_h * width / full_w)
        p = F.interpolate(p[None], (height, width), mode="bilinear", align_corners=False)[0]
        valid = resize_discrete_tensor(valid, (height, width), protocol)
        teacher_confidence = F.interpolate(teacher_confidence[None, None], (height, width),
                                          mode="bilinear", align_corners=False)[0, 0]
        p = p / p.sum(0, keepdim=True).clamp_min(1e-7)
        K = torch.tensor(view["K"], device="cuda").float()
        K[0] *= width / view["width"]
        K[1] *= height / view["height"]
        rendered = scene.render(K, cameras[i], width, height, semantics=False, absgrad=False)
        projection = projection_jacobians(xyz, cameras[i], K)
        covariance = torch.eye(2, device="cuda")[None].repeat(len(ids), 1, 1) * .25
        if uncertainty:
            raw_cov = view.get("pose_covariance")
            camera_cov = (torch.tensor(raw_cov, device="cuda").float() if raw_cov is not None
                          else torch.diag(torch.tensor([.001 ** 2] * 3 + [.0005 ** 2] * 3, device="cuda")))
            order_names = view.get("pose_covariance_order", ["tx", "ty", "tz", "rx", "ry", "rz"])
            canonical_order = ["tx", "ty", "tz", "rx", "ry", "rz"]
            if sorted(order_names) != sorted(canonical_order):
                raise ValueError(f"Unknown camera covariance parameter order: {order_names}")
            if order_names != canonical_order:
                permutation = torch.tensor([order_names.index(name) for name in canonical_order], device="cuda")
                camera_cov = camera_cov[permutation][:, permutation]
            covariance = propagate_projection_covariance(projection.camera_jacobian, projection.point_jacobian,
                                                         camera_cov, point_cov, observation_std=.5)
        evidence = ellipse_sample_probabilities(p, projection.uv, covariance,
                    depths=projection.depth, depth_map=rendered["depth"][..., 0],
                    alpha_map=rendered["alpha"][..., 0] * valid,
                    teacher_confidence_map=teacher_confidence,
                    confidence_threshold=.65, max_projection_std=6,
                    pixel_protocol=protocol)
        probabilities.append(evidence.probabilities)
        weights.append(evidence.weights)
    fused = fuse_multiview_evidence(torch.stack(probabilities), torch.stack(weights), min_views=2)
    return ids, fused.probabilities.detach(), fused.weights.detach()
