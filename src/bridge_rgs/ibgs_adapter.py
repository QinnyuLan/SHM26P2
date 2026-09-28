"""TRAIN-only, camera-only target adapter for the external IBGS implementation.

The external CUDA backend must have the separately audited corner-v2 patch.
Source RGB and depth stay on the CPU until the official renderer selects views.
This adapter does not import a checkpoint, read semantic labels, or build a VAL bank.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from .ibgs_camera_contract import make_ibgs_camera, require_official_shared_centered
from .ibgs_losses import erode_valid


def _checked_image(path, expected_hash, *, rgb):
    path = Path(path)
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected_hash:
        raise ValueError(f'Bound image changed: {path}')
    array = cv2.imread(str(path), cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise ValueError(f'Image decode failed: {path}')
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB) if rgb else array


class BridgeCamera:
    """Minimal official renderer protocol; creation never reads target pixels."""

    def __init__(self, row, index):
        spec = make_ibgs_camera(row['K'], row['w2c_original'], row['width'], row['height'])
        require_official_shared_centered([spec])
        self.uid = index
        self.colmap_id = row.get('image_id', index)
        self.image_name = row['name']
        self.image_width, self.image_height = row['width'], row['height']
        self.original_image = None
        self.nearest_id = []
        self.nearest_names = []
        self.R = torch.tensor(spec['R_argument'], dtype=torch.float32, device='cuda')
        self.T = torch.tensor(spec['T_argument'], dtype=torch.float32, device='cuda')
        for key in ('world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center'):
            setattr(self, key, torch.tensor(spec[key], dtype=torch.float32, device='cuda'))
        for key in ('Fx', 'Fy', 'Cx', 'Cy', 'FoVx', 'FoVy', 'znear', 'zfar'):
            setattr(self, key, spec[key])
        self._array_K = torch.tensor(spec['K_array'], dtype=torch.float32)

    def get_calib_matrix_nerf(self, scale=1.):
        if scale != 1:
            raise ValueError('IBGS port currently validates native scale=1 only')
        return self._array_K, self.world_view_transform.T.contiguous()

    def get_k(self, scale=1.):
        return self.get_calib_matrix_nerf(scale)[0].cuda()

    def get_inv_k(self, scale=1.):
        return self.get_k(scale).inverse()

    def get_rays(self, scale=1.):
        self.get_calib_matrix_nerf(scale)
        ys, xs = torch.meshgrid(torch.arange(self.image_height, device='cuda'),
                               torch.arange(self.image_width, device='cuda'), indexing='ij')
        return torch.stack(((xs-self.Cx)/self.Fx, (ys-self.Cy)/self.Fy,
                            torch.ones_like(xs)), -1).float()


class _SourceRGBBank:
    def __init__(self, rows, hashes, cache_size=None):
        self.rows, self.hashes = rows, hashes
        self.cache_size = len(rows) if cache_size is None else cache_size
        if self.cache_size < 1:
            raise ValueError('Positive source-cache size required')
        self.cache = OrderedDict()
        self.decode_count = 0
        self.decoded_names = set()

    def one(self, index):
        row = self.rows[int(index)]
        if index not in self.cache:
            image = _checked_image(row['image_path'], self.hashes[row['image_path']], rgb=True)
            if image.shape != (row['height'], row['width'], 3):
                raise ValueError('Source bank image dimensions changed')
            self.cache[index] = torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous()
            self.decode_count += 1
            self.decoded_names.add(row['name'])
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index].float()/255

    def __getitem__(self, indices):
        if isinstance(indices, (int, np.integer)):
            return self.one(int(indices))
        return torch.stack([self.one(int(i)) for i in indices])


class _SourceDepthBank:
    def __init__(self, rows, valid):
        self.rows, self.valid = rows, valid
        self.cache = {}
        self.zero = torch.zeros(1, rows[0]['height'], rows[0]['width'])
        self.updates = 0

    def __getitem__(self, indices):
        if isinstance(indices, (int, np.integer)):
            return self.cache.get(int(indices), self.zero)
        return torch.stack([self[int(i)] for i in indices])

    def __setitem__(self, index, depth):
        # Erosion ensures any positive bilinear depth tap has four valid RGB taps.
        value = depth.detach().to(device='cpu', dtype=torch.float32).clone()
        if value.shape != self.zero.shape or not torch.isfinite(value).all():
            raise ValueError('Source depth must be finite 1xHxW')
        value *= self.valid
        self.cache[int(index)] = value
        self.updates += 1


class TrainScene:
    """Duck-typed official Scene with an explicit, TRAIN-only source bank."""

    def __init__(self, rows, pixel_hashes, neighbors, gaussians, scene_radius):
        if not rows or any(v['split'] != 'train' for v in rows):
            raise ValueError('Every source-bank row must be TRAIN')
        specs = [make_ibgs_camera(v['K'], v['w2c_original'], v['width'], v['height']) for v in rows]
        require_official_shared_centered(specs)
        if len({v['name'] for v in rows}) != len(rows):
            raise ValueError('Duplicate bank camera')
        if len({v['valid_path'] for v in rows}) != 1:
            raise ValueError('This port expects the existing common undistortion valid mask')
        path = rows[0]['valid_path']
        valid = _checked_image(path, pixel_hashes[path], rgb=False) > 0
        if valid.shape != (rows[0]['height'], rows[0]['width']):
            raise ValueError('Valid mask dimensions changed')
        self.valid_cpu = torch.from_numpy(valid.copy())
        self.valid = self.valid_cpu.cuda()
        self.cameras = [BridgeCamera(v, i) for i, v in enumerate(rows)]
        lookup = {v['name']: i for i, v in enumerate(rows)}
        for camera in self.cameras:
            selected = neighbors[camera.image_name]
            if len(set(selected)) != len(selected) or camera.image_name in selected:
                raise ValueError('Sources must be distinct and exclude target identity')
            camera.nearest_id = [lookup[name] for name in selected]
            camera.nearest_names = list(selected)
        self.gaussians = gaussians
        self.cameras_extent = scene_radius
        self.original_image_list = _SourceRGBBank(rows, pixel_hashes)
        self.rendered_depth_list = _SourceDepthBank(rows, erode_valid(self.valid_cpu, 1))
        self.world_view_transforms = torch.stack([c.world_view_transform.T for c in self.cameras])
        self.camera_centers = torch.stack([c.camera_center for c in self.cameras])
        self.center_rays = torch.nn.functional.normalize(self.world_view_transforms[:, 2, :3], dim=-1)

    def getTrainCameras(self):
        return self.cameras

    def getTestCameras(self):
        return []
