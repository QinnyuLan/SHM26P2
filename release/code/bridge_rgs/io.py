"""Shared configuration, lazy image loading, and atomic checkpoint utilities."""
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from .coordinates import annotate_manifest, pixel_protocol, resize_discrete_numpy


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_manifest(path):
    content = Path(path).read_bytes()
    manifest = annotate_manifest(json.loads(content))
    manifest["_manifest_sha256"] = hashlib.sha256(content).hexdigest()
    return manifest


def load_view(view, device="cuda", scale=1.0, original_pose=False):
    protocol = pixel_protocol(view)
    image = cv2.imread(view["image_path"])
    if image is None:
        raise FileNotFoundError(view["image_path"])
    h, w = image.shape[:2]
    size = (max(8, round(w * scale)), max(8, round(h * scale)))
    image = cv2.resize(image[..., ::-1], size, interpolation=cv2.INTER_AREA)
    valid = cv2.imread(view["valid_path"], 0) if view.get("valid_path") else np.ones((h, w), np.uint8) * 255
    valid = resize_discrete_numpy(valid, size, protocol) > 0
    mask = cv2.imread(view["mask_path"], 0) if view.get("mask_path") else None
    if mask is not None:
        mask = resize_discrete_numpy(mask, size, protocol).astype(np.int64)
        mask[~valid] = 255
    K = np.array(view["K"], dtype=np.float32)
    K[0] *= size[0] / w
    K[1] *= size[1] / h
    return {"rgb": torch.tensor(image.copy(), device=device).float() / 255,
            "valid": torch.tensor(valid, device=device).float(),
            "mask": torch.tensor(mask, device=device) if mask is not None else None,
            "K": torch.tensor(K, device=device),
            "w2c": torch.tensor(view.get("w2c_original", view["w2c"]) if original_pose else view["w2c"], device=device).float(),
            "width": size[0], "height": size[1], "pixel_protocol": protocol}


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)
