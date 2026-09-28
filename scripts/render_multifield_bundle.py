"""Render a verified three-field engineering bundle for arbitrary original cameras.

Only camera JSON and model files are inputs. There is no validation-image lookup.
The semantic branch is the original H3/teacher mixture; the delivered RGB is the
fixed 1M/MCMC mean. This is intentionally not a single shared-geometry model.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def camera_record(value, index):
    """Whitelist camera data, rejecting image/label payloads rather than using them."""
    required = {'name', 'K', 'w2c', 'width', 'height', 'distortion'}
    optional = {'image_id', 'camera_id', 'split'}
    require(required <= value.keys() and not set(value) - required - optional,
            'Camera JSON accepts name, K, w2c, width, height, distortion and optional IDs/split only')
    name = value['name']
    require(isinstance(name, str) and name == Path(name).name and name.endswith('.png'),
            'Each camera name must be a plain .png filename')
    width, height = value['width'], value['height']
    require(type(width) is int and type(height) is int and min(width, height) >= 16,
            'Camera width/height must be positive integers at least 16')
    K, pose, distortion = [np.asarray(value[k], np.float64) for k in ('K', 'w2c', 'distortion')]
    require(K.shape == (3, 3) and pose.shape == (4, 4) and distortion.ndim == 1
            and distortion.size in (4, 5, 8, 12, 14)
            and all(np.isfinite(x).all() for x in (K, pose, distortion)), 'Invalid camera arrays')
    require(K[0, 0] > 0 and K[1, 1] > 0 and np.allclose(K[2], [0, 0, 1])
            and np.allclose(pose[3], [0, 0, 0, 1]), 'Invalid intrinsics or homogeneous pose')
    rotation = pose[:3, :3]
    require(np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4)
            and np.isclose(np.linalg.det(rotation), 1, atol=1e-4), 'w2c must contain a proper rotation')
    return {**value, 'K': K.tolist(), 'w2c': pose.tolist(), 'distortion': distortion.tolist(),
            'image_id': value.get('image_id', index), 'camera_id': value.get('camera_id', index),
            'split': value.get('split', 'test')}


def mean_rgb(a, b):
    require(a.dtype == b.dtype == np.uint8 and a.shape == b.shape and a.shape[-1] == 3,
            'RGB components must be aligned uint8 HWC')
    return np.rint((a.astype(np.float32) + b.astype(np.float32)) * .5).astype(np.uint8)


def semantic_mask(rgb, probabilities, teacher, back):
    """Original selected formula: canvas mixture, one soft warp, then argmax."""
    require(rgb.ndim == 3 and rgb.shape[-1] == 3
            and probabilities.shape == (*rgb.shape[:2], 5), 'Canvas shape mismatch')
    require(np.isfinite(rgb).all() and np.isfinite(probabilities).all()
            and (probabilities >= 0).all(), 'Invalid canvas values')
    mixture = teacher.blend(np.clip(rgb, 0, 1), probabilities)
    require(mixture.shape == probabilities.shape and np.isfinite(mixture).all()
            and (mixture >= 0).all(), 'Invalid teacher mixture')
    if back is not None:
        mixture = cv2.remap(mixture, back[..., 0], back[..., 1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    require((mixture.sum(-1) > 0).all(), 'Uncovered semantic rays')
    return mixture.argmax(-1).astype(np.uint8)


def verify_bundle(bundle):
    require(bundle['schema'] == 'camera_only_three_field_rgb_upgrade_v1'
            and bundle['teacher_weight'] == .5 and bundle['rgb_weights'] == [.5, .5],
            'Unsupported bundle contract')
    require(set(bundle['components']) == {'selected', 'capacity_1m', 'mcmc'}, 'Expected three fields')
    require(bundle['components']['selected']['profile'] == 'legacy_mixed_v1'
            and all(bundle['components'][k]['profile'] == 'colmap_corner_v2'
                    for k in ('capacity_1m', 'mcmc')), 'Renderer profiles changed')
    for path, expected in bundle['input_hashes'].items():
        require(sha(path) == expected, f'Changed bundle dependency: {path}')
    snapshot = Path(bundle['source_snapshot'])
    for name, digest in bundle['package_hashes'].items():
        require(sha(snapshot/name) == digest, f'Changed frozen package: {name}')
    for component in bundle['components'].values():
        require(bundle['input_hashes'].get(component['checkpoint']) == component['checkpoint_sha256'],
                'Unbound scene checkpoint')
    require(bundle['input_hashes'].get(bundle['teacher_checkpoint']) == bundle['teacher_checkpoint_sha256'],
            'Unbound teacher checkpoint')
    backing = bundle['backbone']
    for name, digest in {'config.json': backing['config_sha256'], **backing['weights_sha256'],
                         **backing.get('index_sha256', {})}.items():
        require(bundle['input_hashes'].get(str(Path(backing['model_dir'])/name)) == digest,
                'Unbound backbone dependency')
    require({name: importlib.metadata.version(name) for name in bundle['runtime_versions']}
            == bundle['runtime_versions'], 'Bundle runtime versions differ')


def loaded_sources(bundle):
    records = {}
    root = Path(bundle['source_snapshot'])
    for name, module in sys.modules.items():
        if name == 'bridge_rgs' or name.startswith('bridge_rgs.'):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(root), 'Loaded module escaped bundle')
            key = str(path.relative_to(root))
            require(bundle['package_hashes'].get(key) == sha(path), 'Unbound loaded source')
            records[name] = {'path': str(path), 'sha256': sha(path)}
    return records


def render(bundle_path, camera_path, output, device='cuda'):
    started = time.monotonic()
    bundle_path, camera_path, output = (Path(p).resolve() for p in (bundle_path, camera_path, output))
    bundle = json.loads(bundle_path.read_text())
    verify_bundle(bundle)
    os.chdir(bundle['workspace_root'])
    raw = json.loads(camera_path.read_text())
    cameras = [camera_record(v, i) for i, v in enumerate(raw if isinstance(raw, list) else [raw])]
    require(cameras and len({c['name'] for c in cameras}) == len(cameras), 'Empty/duplicate camera list')
    require(not output.exists(), 'Output must be a fresh directory')
    require(not any(k == 'bridge_rgs' or k.startswith('bridge_rgs.') for k in sys.modules),
            'Start a fresh process to load the frozen bundle package')
    sys.path.insert(0, bundle['source_snapshot'])
    official = importlib.import_module('bridge_rgs.official_evaluate')
    ensemble = importlib.import_module('bridge_rgs.render_ensemble')
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    output.mkdir(parents=True)
    for subdir in ('rgb', 'mask', 'components/capacity_1m', 'components/mcmc'):
        (output/subdir).mkdir(parents=True)
    receipt = {'schema': bundle['schema'], 'status': 'running', 'bundle_sha256': sha(bundle_path),
               'camera_json_sha256': sha(camera_path), 'cameras': cameras, 'scene_renders': 0,
               'teacher_calls': 0, 'gt_payload_reads': 0, 'predictions': [],
               'scope': 'Three independent fields; original H3/teacher semantics and separate mean RGB'}
    try:
        for role in ('selected', 'capacity_1m', 'mcmc'):
            component = bundle['components'][role]
            scene, state = official._load_scene(component['checkpoint'], device)
            require(official.pixel_protocol(state) == component['profile'], 'Loaded field profile mismatch')
            scene.eval().requires_grad_(False)
            teacher = (ensemble.FixedRenderedTeacher(bundle['teacher_checkpoint'], component['checkpoint'], state,
                                                      device=device) if role == 'selected' else None)
            with torch.inference_mode():
                for camera in cameras:
                    if role == 'selected':
                        K, width, height, back = official.distortion_render_grid(
                            camera['K'], camera['distortion'], camera['width'], camera['height'], component['profile'])
                        result = scene.render(torch.tensor(K, device=device),
                                              torch.tensor(camera['w2c'], device=device).float(),
                                              width, height, absgrad=False)
                        mask = semantic_mask(result['rgb'].clamp(0, 1).cpu().numpy(),
                                             result['probabilities'].cpu().numpy(), teacher, back)
                        require(cv2.imwrite(str(output/'mask'/camera['name']), mask), 'Mask save failed')
                        receipt['teacher_calls'] += 1
                        del result
                    else:
                        rgb, _, _ = official.predict_official_camera(scene, camera, state)
                        require(cv2.imwrite(str(output/'components'/role/camera['name']), rgb[..., ::-1]),
                                'RGB component save failed')
                    receipt['scene_renders'] += 1
            if teacher is not None:
                receipt['teacher'] = teacher.receipt
            del scene, state, teacher
            gc.collect()
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
        for camera in cameras:
            name = camera['name']
            members = [cv2.imread(str(output/'components'/k/name)) for k in ('capacity_1m', 'mcmc')]
            rgb = mean_rgb(*members)
            require(cv2.imwrite(str(output/'rgb'/name), rgb), 'Mean RGB save failed')
            receipt['predictions'].append({'name': name, 'rgb': str(output/'rgb'/name),
                'rgb_sha256': sha(output/'rgb'/name), 'mask': str(output/'mask'/name),
                'mask_sha256': sha(output/'mask'/name)})
        verify_bundle(bundle)
        receipt['loaded_sources'] = loaded_sources(bundle)
        receipt['status'] = 'completed'
    except BaseException as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        receipt['status'] = 'failed'
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic()-started
        receipt['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        (output/'execution_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': receipt['status'], 'views': len(cameras),
                      'scene_renders': receipt['scene_renders'], 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--cameras', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    render(args.bundle, args.cameras, args.output, args.device)
