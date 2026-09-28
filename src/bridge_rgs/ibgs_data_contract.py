"""CPU metadata port for a TRAIN-only IBGS scene, without constructing its CUDA Scene.

This does not solve the author's half-pixel camera convention or change the
source-selection thresholds. Shared camera-valid files are geometric assets;
TRAIN RGB paths may not alias a held-out RGB path. No image/mask is decoded.
The returned NPZ descriptors do not deserialize point or semantic arrays.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

MAX_ANGLE_DEGREES = 30.
MIN_DISTANCE = .01
MAX_DISTANCE = 1.5


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _path(value, base):
    path = Path(value)
    path = (base/path).resolve() if not path.is_absolute() else path.resolve()
    _require(path.is_file(), 'Missing bound asset: '+str(path))
    return str(path)


def point_cloud_descriptor(path):
    """Read NPY headers and hash bytes, not point/label values or an IBGS PLY."""
    path = Path(path).resolve(); fields = {}
    with zipfile.ZipFile(path) as archive:
        _require(len(set(archive.namelist())) == len(archive.namelist()), 'Duplicate NPZ member')
        for member in archive.namelist():
            _require(member.endswith('.npy') and '/' not in member, 'Simple NPY members required')
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                _require(version in ((1, 0), (2, 0)), 'Unsupported NPY header version')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(stream)
                _require(not dtype.hasobject, 'Object arrays are not a geometric input')
                fields[member[:-4]] = {'shape': list(shape), 'dtype': str(dtype), 'fortran_order': bool(fortran)}
    _require({'points', 'colors'} <= fields.keys(), 'Point coordinates and colors required')
    shape = fields['points']['shape']
    _require(len(shape) == 2 and shape[0] > 0 and shape[1] == 3
             and fields['colors']['shape'] == shape
             and fields['points']['dtype'] == fields['colors']['dtype'] == 'float32', 'Expected FP32 Nx3 point/color arrays')
    return {'path': str(path), 'sha256': _sha(path), 'point_count': shape[0], 'fields': fields,
            'adapter_arrays': ['points', 'colors'],
            'semantic_arrays_loaded': False, 'arrays_deserialized': False,
            'initialization_scope': 'Original sparse TRAIN point cloud, not the trained 1M Gaussian checkpoint; no automatic RMS filter or far-point addition.'}


def neighbor_contract(rows, scene_radius):
    """Default author filters with explicit self-exclusion and deterministic ties.

The distance needed for 4/8 neighbors is descriptive only. Since the author's
upper distance comparison is strict, a new bound would have to EXCEED the
reported k-th distance. This function never uses it to rescue coverage.
"""
    _require(rows and len({r['name'] for r in rows}) == len(rows), 'Unique nonempty TRAIN rows')
    _require(all(r['split'] == 'train' for r in rows), 'Only TRAIN may enter the source bank')
    _require(np.isfinite(scene_radius) and scene_radius > 0, 'Positive scene radius')
    centers = np.asarray([r['camera_center'] for r in rows], np.float64)
    forward = np.asarray([r['forward_world'] for r in rows], np.float64)
    _require(centers.shape == forward.shape == (len(rows), 3)
             and np.isfinite(centers).all() and np.isfinite(forward).all(), 'Finite camera geometry')
    norms = np.linalg.norm(forward, axis=1)
    _require(np.all(norms > 0), 'Nonzero forward axes')
    forward = forward/norms[:, None]
    result = []
    for i, row in enumerate(rows):
        distance = np.linalg.norm(centers-centers[i], axis=1)
        angle = np.degrees(np.arccos(np.clip(forward@forward[i], -1, 1)))
        _require(np.isfinite(distance).all() and np.isfinite(angle).all(), 'Camera-distance overflow')
        eligible = [j for j in range(len(rows)) if j != i and distance[j] > MIN_DISTANCE and angle[j] < MAX_ANGLE_DEGREES]
        eligible.sort(key=lambda j: (float(distance[j]), float(angle[j]), rows[j]['name']))
        default = [j for j in eligible if distance[j] < MAX_DISTANCE]
        record = {'name': row['name'], 'default_candidate_count': len(default),
                  'angle_and_min_distance_candidate_count': len(eligible)}
        for k in (4, 8):
            record[f'neighbors_{k}'] = [{'index': rows[j]['index'], 'name': rows[j]['name'],
                                        'distance_world': float(distance[j]), 'angle_degrees': float(angle[j])} for j in default[:k]]
            kth = float(distance[eligible[k-1]]) if len(eligible) >= k else None
            record[f'threshold_requirement_{k}'] = {
                'strict_upper_distance_must_exceed': kth,
                'distance_over_scene_radius': kth/scene_radius if kth is not None else None,
                'used_to_change_sources': False}
        result.append(record)
    return {'filters': {'angle_degrees_strict_less': MAX_ANGLE_DEGREES,
                        'distance_strict_greater': MIN_DISTANCE, 'distance_strict_less': MAX_DISTANCE,
                        'maximum_distance_over_scene_radius': MAX_DISTANCE/scene_radius,
                        'ranking': 'center distance, then view-axis angle, then name; self excluded',
                        'exposure_correction_reordering': False, 'coverage_rescue': False},
            'coverage': {'cameras': len(rows), 'zero_candidates': sum(r['default_candidate_count'] == 0 for r in result),
                         'at_least_4': sum(r['default_candidate_count'] >= 4 for r in result),
                         'at_least_8': sum(r['default_candidate_count'] >= 8 for r in result)},
            'views': result}


def build_train_contract(manifest_path, *, required_train_count=350):
    """Return JSON-safe metadata/hashes for a custom scene; no files are written.

The manifest may contain held-out metadata. Only its TRAIN RGB/valid assets are
opened (bytes for SHA only). Raw source-image paths, annotations and semantic
masks are deliberately not returned. The original image split is never redrawn.
"""
    _require(type(required_train_count) is int and required_train_count > 0, 'Positive required TRAIN count')
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    _require(manifest['pixel_protocol']['id'] == 'colmap_corner_v2', 'Only corner_v2 prepared data')
    views = manifest['views']
    _require(len({v['name'] for v in views}) == len(views), 'Duplicate manifest camera name')
    train = sorted((v for v in views if v['split'] == 'train'), key=lambda v: v['name'])
    _require(len(train) == required_train_count, 'Unexpected TRAIN population')
    # Resolve withheld path identities without opening/checking any withheld file.
    def resolve_only(value):
        p = Path(value)
        return str((manifest_path.parent/p).resolve() if not p.is_absolute() else p.resolve())
    heldout_rgb = {resolve_only(v['image_path']) for v in views if v['split'] != 'train'}
    rows = []; pixel_hashes = {}; rgb_paths = set(); common = None
    for i, view in enumerate(train):
        K, pose = np.asarray(view['K'], np.float64), np.asarray(view['w2c_original'], np.float64)
        _require(K.shape == (3, 3) and pose.shape == (4, 4) and np.isfinite(K).all() and np.isfinite(pose).all(), 'Finite camera K/pose')
        _require(np.array_equal(K[2], [0, 0, 1]) and K[0, 0] > 0 and K[1, 1] > 0
                 and K[0, 1] == K[1, 0] == 0 and np.array_equal(pose[3], [0, 0, 0, 1]), 'Pinhole/homogeneous camera')
        R, t = pose[:3, :3], pose[:3, 3]
        _require(np.allclose(R@R.T, np.eye(3), rtol=0, atol=1e-5) and abs(np.linalg.det(R)-1) <= 1e-5, 'Proper rotation')
        _require('w2c' not in view or np.array_equal(pose, view['w2c']), 'Current and original pose differ')
        width, height = view['width'], view['height']
        _require(type(width) is type(height) is int and min(width, height) > 0, 'Positive native dimensions')
        this = {'width': width, 'height': height, 'K': K.tolist(), 'K_fp32': K.astype(np.float32).tolist()}
        _require(common is None or common == this, 'Source bank must share native dimensions and K')
        common = this
        rgb = resolve_only(view['image_path'])
        _require(rgb not in heldout_rgb and rgb not in rgb_paths, 'TRAIN RGB aliases a held-out or duplicate image')
        rgb_paths.add(rgb)
        image_path = _path(view['image_path'], manifest_path.parent)
        valid_path = _path(view['valid_path'], manifest_path.parent)
        for path in (image_path, valid_path):
            if path not in pixel_hashes:
                pixel_hashes[path] = _sha(path)
        center = -np.linalg.solve(R, t); direction = np.linalg.solve(R, np.array([0., 0., 1.]))
        direction /= np.linalg.norm(direction)
        rows.append({'index': i, 'name': view['name'], 'image_stem': Path(view['name']).stem,
                     'image_id': int(view['image_id']), 'camera_id': int(view['camera_id']), 'split': 'train',
                     'image_path': image_path, 'valid_path': valid_path, 'width': width, 'height': height,
                     'K': K.tolist(), 'w2c_original': pose.tolist(),
                     'K_fp32': K.astype(np.float32).tolist(), 'w2c_fp32': pose.astype(np.float32).tolist(),
                     'ibgs_R': R.T.tolist(), 'ibgs_T': t.tolist(),
                     'camera_center': center.tolist(), 'forward_world': direction.tolist()})
    radius = float(manifest['scene_radius'])
    init = point_cloud_descriptor(_path(manifest['init_points_path'], manifest_path.parent))
    return {'format': 'ibgs_train_data_contract_v1', 'manifest': str(manifest_path), 'manifest_sha256': _sha(manifest_path),
            'pixel_protocol': 'colmap_corner_v2', 'train_rows': rows, 'train_count': len(rows),
            'heldout_cameras_in_bank': 0, 'heldout_payloads_read': 0, 'image_or_mask_decodes': 0,
            'pixel_hashes': pixel_hashes, 'shared_valid_files': len({r['valid_path'] for r in rows}),
            'common_camera': common, 'manifest_scene_radius': radius, 'initial_point_cloud': init,
            'neighbors': neighbor_contract(rows, radius),
            'limits': ['This port does not instantiate the author CUDA Camera/Scene or load any model.',
                       'Author reader RMS filtering, nearest-eight exposure reordering and half-pixel ray conventions are not silently applied.',
                       'Prepared RGB-valid is supplied separately; the consuming loss/sampler must explicitly honor it.',
                       'No complete data copy, trained Gaussian conversion, inferred normals, semantic arrays or VAL bank.']}
