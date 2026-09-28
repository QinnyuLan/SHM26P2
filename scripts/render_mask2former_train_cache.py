"""Render exactly 259 labeled TRAIN cameras; never decode dataset pixel files.

GPU execution requires the separately approved experiment launch. This worker does
not prepare a plan or run at import time; use --checkpoint --manifest --output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from bridge_rgs.coordinates import LEGACY, pixel_protocol

BASE_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
MANIFEST_SHA = '551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def tensor_hash(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


def camera_contract(manifest, state, manifest_path):
    """Validate metadata/tensors only and return whitelisted camera-only records."""
    names = [v['name'] for v in manifest['views']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate manifest camera names')
    train = sorted((v for v in manifest['views'] if v['split'] == 'train'), key=lambda v: v['name'])
    labeled = [v for v in train if v.get('mask_path')]
    config_manifest = Path(state['config']['manifest'])
    # Frozen experiment is launched from the workspace; relative config paths
    # are interpreted against cwd, never against a source_snapshot directory.
    if (pixel_protocol(state) != LEGACY or pixel_protocol(manifest) != LEGACY
            or len(train) != 350 or len(labeled) != 259
            or state['config'].get('optimize_cameras', False)
            or config_manifest.resolve() != Path(manifest_path).resolve()
            or state.get('sh_degree') != 3):
        raise ValueError('Expected fixed legacy selected scene, manifest, and TRAIN counts')
    poses = torch.tensor([v['w2c'] for v in train], dtype=torch.float32)
    actual = state['training_cameras']
    if actual.shape != (350, 4, 4) or actual.dtype != torch.float32 or not torch.equal(actual.cpu(), poses):
        raise ValueError('Checkpoint cameras differ from sorted 350 TRAIN poses')
    result = []
    for index, view in enumerate(train):
        if not view.get('mask_path'):
            continue
        name = view['name']
        K = np.asarray(view['K'], dtype=np.float64)
        if (Path(name).name != name or not name.endswith('.png')
                or (view['width'], view['height']) != (1320, 989)
                or K.shape != (3, 3) or not np.isfinite(K).all()
                or not np.isfinite(np.asarray(view['w2c'])).all()):
            raise ValueError('Unsafe name or invalid native camera')
        result.append({'name': name, 'camera_index': index, 'width': 1320, 'height': 989,
                       'K': view['K']})
    if len({Path(v['name']).stem for v in result}) != 259:
        raise ValueError('Output filename collision')
    return result


def rgb_uint8(rgb):
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not bool(torch.isfinite(rgb).all()):
        raise ValueError('Expected finite HWC RGB')
    return np.round(rgb.detach().clamp(0, 1).cpu().numpy()*255).astype(np.uint8)


def source_hashes():
    import bridge_rgs
    package = Path(bridge_rgs.__file__).resolve().parent
    paths = sorted(package.rglob('*.py')) + [Path(__file__).resolve()]
    return {str(path): digest(path) for path in paths}


def write_receipt(path, receipt):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run(checkpoint, manifest_path, output):
    import cv2

    from bridge_rgs.train import load_scene

    checkpoint, manifest_path, output = (Path(p).resolve() for p in (checkpoint, manifest_path, output))
    if output.exists():
        raise FileExistsError('Use a new cache directory; no overwrite or automatic resume')
    inputs = {str(checkpoint): digest(checkpoint), str(manifest_path): digest(manifest_path)}
    if inputs[str(checkpoint)] != BASE_SHA or inputs[str(manifest_path)] != MANIFEST_SHA:
        raise ValueError('Selected H3 checkpoint or legacy manifest changed')
    sources = source_hashes()
    manifest = json.loads(manifest_path.read_text())
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cameras = camera_contract(manifest, state, manifest_path)
    camera_sha = tensor_hash(state['training_cameras'])
    del state
    output.mkdir(parents=True)
    (output/'rgb').mkdir()
    receipt_path = output/'cache_receipt.json'
    receipt = {'status': 'running', 'started_utc': datetime.now(UTC).isoformat(),
               'checkpoint': str(checkpoint), 'checkpoint_sha256': BASE_SHA,
               'manifest': str(manifest_path), 'manifest_sha256': MANIFEST_SHA,
               'pixel_protocol': LEGACY, 'view_names': [v['name'] for v in cameras],
               'camera_tensor_sha256': camera_sha, 'source_hashes': sources,
               'input_hashes': inputs, 'records': [], 'grid': 'native_pinhole',
               'render_options': {'degree': 3, 'semantics': False, 'refine': False, 'absgrad': False},
               'pixel_input_policy': 'No dataset RGB, mask, or validity pixels opened; metadata only.',
               'quantization': 'FP32 render clamp [0,1], NumPy round to uint8 RGB PNG'}
    write_receipt(receipt_path, receipt)
    try:
        cv2.setNumThreads(8)
        torch.set_num_threads(8)
        scene, state = load_scene(checkpoint)
        scene.eval().requires_grad_(False)
        if camera_contract(manifest, state, manifest_path) != cameras:
            raise ValueError('Camera contract changed after loading scene')
        before = {k: tensor_hash(v) for k, v in scene.state_dict().items()}
        with torch.inference_mode():
            for view in cameras:
                result = scene.render(torch.tensor(view['K'], dtype=torch.float32, device='cuda'),
                                      state['training_cameras'][view['camera_index']].cuda(),
                                      1320, 989, degree=3, semantics=False, refine=False, absgrad=False)
                image = rgb_uint8(result['rgb'])
                if image.shape != (989, 1320, 3):
                    raise ValueError('Unexpected rendered image grid')
                image_path = output/'rgb'/view['name']
                if not cv2.imwrite(str(image_path), image[..., ::-1]):
                    raise OSError(f'Cannot save {image_path}')
                receipt['records'].append({'name': view['name'], 'image_path': str(image_path),
                                           'sha256': digest(image_path), 'width': 1320, 'height': 989})
                del result
        if before != {k: tensor_hash(v) for k, v in scene.state_dict().items()}:
            raise ValueError('Scene parameters changed')
        if camera_sha != tensor_hash(state['training_cameras']):
            raise ValueError('Camera tensor changed')
        if inputs != {path: digest(path) for path in inputs} or sources != source_hashes():
            raise ValueError('Source or input changed during rendering')
        if {p.name for p in (output/'rgb').iterdir()} != set(receipt['view_names']):
            raise ValueError('Cache set is not exactly labeled TRAIN259')
        receipt.update(status='completed', frozen_tensors_verified=True,
                       scene_tensor_hashes=before, rendered_views=len(cameras))
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt['finished_utc'] = datetime.now(UTC).isoformat()
        write_receipt(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.checkpoint, args.manifest, args.output)
    print(json.dumps({'status': receipt['status'], 'views': receipt['rendered_views']}), flush=True)


if __name__ == '__main__':
    main()
