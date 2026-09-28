"""Render camera-only TRAIN RGB and explicitly migrate two domains to legacy.

This is data preparation for a matched teacher adaptation experiment. It neither
reads target pixels nor trains a model. Every renderer keeps its own pixel profile.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import shutil
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
TRANSFER = RUNS/'rgb_teacher_transfer_v1'
SOURCE = RUNS/'rgb_capacity_1m_reference_v1/official_preparation/source_snapshot/bridge_rgs'
SPEC = {'train_cameras': 350, 'validation_cached_cameras': 50, 'renderer_camera_checks': 3,
        'training_scene_renders': 1050, 'total_scene_renders': 1053,
        'domains': ['old_selected', 'new_composite'], 'composite_weights': [.5, .5],
        'teacher_pixel_protocol': 'legacy_mixed_v1',
        'adapter': 'original_uint8 -> initUndistortRectifyMap legacy canvas; INTER_LINEAR; black border',
        'blend': 'round-to-nearest-even mean of two original-grid uint8 RGB predictions',
        'labels': 'Never opened; original legacy mask and valid paths retained later',
        'validation': 'Reuses previous camera-generated composite RGB only; no target payload reads',
        'internal_seconds': 570, 'external_seconds': 600}
ADAPTER = {'id': 'original_png_to_legacy_pure_hplus_v1', 'border': 'constant_zero',
           'interpolation': 'INTER_LINEAR', 'half_pixel_conjugation': False,
           'rgb_mean': 'float32_rint_uint8_0.5'}
CACHE_ID = 'original_png_multi_component_teacher_domain_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, replace=False):
    with Path(path).open('w' if replace else 'x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def utc():
    return datetime.now(UTC).isoformat()


def source_hashes(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(Path(directory).rglob('*.py'))}


def average_rgb(a, b):
    require(a.shape == b.shape and a.dtype == b.dtype == np.uint8, 'Invalid component RGB')
    return np.rint((a.astype(np.float32)+b.astype(np.float32))*.5).astype(np.uint8)


def camera_records(manifest):
    from bridge_rgs.official_evaluate import _camera
    train = sorted((v for v in manifest['views'] if v['split'] == 'train'), key=lambda v: v['name'])
    require(len(train) == 350 and sum(bool(v.get('mask_path')) for v in train) == 259,
            'Expected fixed 350 TRAIN / 259 annotations')
    records = [_camera(v, manifest['source_cameras']) for v in train]
    require(len({r['name'] for r in records}) == 350, 'Duplicate TRAIN camera')
    return records


def prepare(output):
    output = Path(output).resolve()
    require(not output.exists() and not torch.cuda.is_initialized(), 'Fresh CPU preparation required')
    previous = read(TRANSFER/'plan.json')
    transfer_receipt = read(TRANSFER/'execution_receipt.json')
    require(transfer_receipt['status'] == 'completed' and
            sha(TRANSFER/'plan.json') == transfer_receipt['plan_sha256'], 'Incomplete source transfer')
    selected_dir = RUNS/'official_selected_ensemble_v1/cross_teacher'
    selected = read(selected_dir/'execution_receipt.json')
    components = {'selected': {'checkpoint': selected['checkpoint'],
        'checkpoint_sha256': selected['checkpoint_sha256'],
        'directory': str(selected_dir), 'profile': 'legacy_mixed_v1'}}
    for name in ('capacity_1m', 'mcmc'):
        old = previous['original_evaluations'][name]
        components[name] = {k: old[k] for k in ('checkpoint', 'checkpoint_sha256', 'directory')}
        components[name]['profile'] = old['checkpoint_pixel_protocol']['id']
    bindings = {}
    def bind(path, expected=None):
        path = str(Path(path).resolve())
        digest = sha(path)
        require(expected is None or digest == expected, f'Changed source: {path}')
        bindings[path] = digest
    manifest_path = ROOT/'artifacts/prepared/manifest.json'
    bind(manifest_path, '551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa')
    manifest = read(manifest_path)
    cameras = camera_records(manifest)
    check_camera = next(v['camera'] for v in previous['views'] if v['name'] == '001.png')
    for component in components.values():
        bind(component['checkpoint'], component['checkpoint_sha256'])
        receipt_path = Path(component['directory'])/'execution_receipt.json'
        bind(receipt_path)
        receipt = read(receipt_path)
        require(receipt['status'] == 'completed' and
                receipt['checkpoint_sha256'] == component['checkpoint_sha256'], 'Incomplete renderer source')
        record = next(r for r in receipt['predictions'] if r['name'] == '001.png')
        bind(record['rgb'], record['rgb_sha256'])
        component['check_prediction'] = {'path': record['rgb'], 'sha256': record['rgb_sha256']}
        lineage = receipt['training_manifest']
        bind(lineage['path'], lineage['observed_sha256'])
        component['source_manifest_sha256'] = lineage.get('checkpoint_declared_sha256')
        for dependency in receipt.get('checkpoint_dependencies', {}).values():
            if isinstance(dependency, dict) and 'path' in dependency and 'sha256' in dependency:
                bind(dependency['path'], dependency['sha256'])
    validation = []
    for row in previous['views']:
        source = row['inputs']['composite_rgb_candidate']
        bind(source['path'], source['sha256'])
        validation.append({'name': row['name'], 'camera': row['camera'], 'original_rgb': source})
    for path in (TRANSFER/'plan.json', TRANSFER/'execution_receipt.json', ROOT/'uv.lock',
                 ROOT/'scripts/evaluate_rgb_teacher_transfer.py', Path(__file__)):
        bind(path)
    for path in SOURCE.rglob('*.py'):
        bind(path)
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(SOURCE, snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in (Path(__file__), ROOT/'scripts/evaluate_rgb_teacher_transfer.py'):
        shutil.copy2(path, snapshot/path.name)
    plan = {'status': 'prepared', 'specification': SPEC, 'output': str(output), 'root': str(ROOT),
            'source_snapshot': str(snapshot), 'source_hashes': source_hashes(snapshot),
            'input_hashes': bindings, 'components': components, 'train_cameras': cameras,
            'validation': validation, 'check_camera': check_camera,
            'original_manifest': str(manifest_path), 'original_manifest_sha256': sha(manifest_path),
            'teacher_grids': {v['name']: {k: v[k] for k in ('K', 'width', 'height')} for v in manifest['views']},
            'gt_payload_reads': 0, 'created_utc': utc()}
    write(output/'plan.json', plan)
    return output/'plan.json'


def decode(path):
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    require(value is not None and value.shape == (989, 1320, 3), 'Unexpected image size')
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def save_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), 'No image overwrite')
    require(cv2.imwrite(str(path), rgb[..., ::-1]), 'Image save failed')
    return {'image_path': str(path), 'sha256': sha(path), 'width': int(rgb.shape[1]), 'height': int(rgb.shape[0])}


def verify_canvas_grid(info, grid):
    require(info['canvas'] == [grid['width'], grid['height']]
            and np.array_equal(np.asarray(info['canvas_K'], np.float32), np.asarray(grid['K'], np.float32)),
            'Adapter canvas must exactly match unchanged legacy annotation coordinates')


def execute(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    output, snapshot = Path(plan['output']), Path(plan['source_snapshot'])
    require(plan['specification'] == SPEC and source_hashes(snapshot) == plan['source_hashes'], 'Changed source/spec')
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Use frozen runner')
    require(all(sha(p) == h for p, h in plan['input_hashes'].items()), 'Changed input')
    os.chdir(plan['root'])
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    official = importlib.import_module('bridge_rgs.official_evaluate')
    adapter = importlib.import_module('evaluate_rgb_teacher_transfer')
    old_handler = signal.getsignal(signal.SIGALRM)
    def deadline(*_):
        raise TimeoutError('Cache generation exceeded fixed internal 570 seconds')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(570)
    started = time.monotonic()
    receipt = {'status': 'running', 'started_utc': utc(), 'plan_sha256': sha(plan_path),
               'source_manifest_sha256': plan['original_manifest_sha256'], 'gt_payload_reads': 0,
               'original_manifest_sha256': plan['original_manifest_sha256'],
               'id': CACHE_ID, 'adapter': ADAPTER,
               'components': {k: {'checkpoint': v['checkpoint'], 'checkpoint_sha256': v['checkpoint_sha256'],
                   'pixel_protocol': v['profile'], 'source_manifest_sha256': v['source_manifest_sha256']}
                   for k, v in plan['components'].items()},
               'scene_renders': 0, 'component_records': {}, 'adapter_records': {}, 'camera_checks': {}}
    receipt_path = output/'execution_receipt.json'
    write(receipt_path, receipt)
    try:
        for name, component in plan['components'].items():
            scene, state = official._load_scene(component['checkpoint'], 'cuda')
            scene.eval().requires_grad_(False)
            require(official.pixel_protocol(state) == component['profile'], 'Renderer profile changed')
            with torch.no_grad():
                rgb, _, _ = official.predict_official_camera(scene, plan['check_camera'], state)
                receipt['scene_renders'] += 1
                expected = decode(component['check_prediction']['path'])
                require(np.array_equal(rgb, expected), f'{name}: known camera RGB differs')
                receipt['camera_checks'][name] = {'name': '001.png', 'pixel_exact': True,
                    'reference': component['check_prediction']}
                records = []
                for camera in plan['train_cameras']:
                    rgb, _, info = official.predict_official_camera(scene, camera, state)
                    receipt['scene_renders'] += 1
                    records.append({'name': camera['name'], **save_rgb(output/'components'/name/camera['name'], rgb),
                                    'camera': camera, **info})
                receipt['component_records'][name] = records
            del scene, state
            gc.collect()
            torch.cuda.empty_cache()
            write(receipt_path, receipt, True)
            print(json.dumps({'component': name, 'train_rgb_count': len(records), 'elapsed_seconds': time.monotonic()-started}), flush=True)
        members = {k: {r['name']: r for r in v} for k, v in receipt['component_records'].items()}
        for domain in SPEC['domains']:
            records = []
            for camera in plan['train_cameras']:
                name = camera['name']
                if domain == 'old_selected':
                    rgb = decode(members['selected'][name]['image_path'])
                    origins = [members['selected'][name]]
                else:
                    origins = [members[k][name] for k in ('capacity_1m', 'mcmc')]
                    rgb = average_rgb(*(decode(r['image_path']) for r in origins))
                original = save_rgb(output/'original_rgb'/domain/name, rgb)
                mx, my, _, info = adapter.adapter_maps(camera, official.distortion_render_grid)
                verify_canvas_grid(info, plan['teacher_grids'][name])
                canvas = adapter.input_canvas(rgb, mx, my)
                records.append({'name': name, **save_rgb(output/'legacy_rgb'/domain/name, canvas),
                                'original_rgb': original, 'component_inputs': origins, 'adapter_info': info})
            receipt['adapter_records'][domain] = records
        validation_records = []
        for record in plan['validation']:
            camera = record['camera']
            rgb = decode(record['original_rgb']['path'])
            mx, my, _, info = adapter.adapter_maps(camera, official.distortion_render_grid)
            verify_canvas_grid(info, plan['teacher_grids'][record['name']])
            validation_records.append({'name': record['name'], **save_rgb(output/'legacy_rgb/validation'/record['name'], adapter.input_canvas(rgb, mx, my)),
                'original_rgb': record['original_rgb'], 'adapter_info': info})
        receipt['validation_records'] = validation_records
        receipt['records'] = {'train_selected': receipt['adapter_records']['old_selected'],
                              'train_composite': receipt['adapter_records']['new_composite'],
                              'val_composite': validation_records}
        require(receipt['scene_renders'] == SPEC['total_scene_renders'], 'Unexpected render count')
        require(all(sha(p) == h for p, h in plan['input_hashes'].items()), 'Changed input at completion')
        require(source_hashes(snapshot) == plan['source_hashes'], 'Changed source at completion')
        imports = {}
        for name, module in list(sys.modules.items()):
            if name == 'bridge_rgs' or name.startswith('bridge_rgs.') or name == 'evaluate_rgb_teacher_transfer':
                path = Path(module.__file__).resolve()
                require(path.is_relative_to(snapshot) and sha(path) == plan['source_hashes'][str(path.relative_to(snapshot))], 'Import escaped frozen source')
                imports[name] = {'path': str(path), 'sha256': sha(path)}
        receipt.update(status='completed', finished_utc=utc(), elapsed_seconds=time.monotonic()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), actual_imports=imports,
            bound_sources_inputs_unchanged=True, pixel_protocol='legacy_mixed_v1')
        write(receipt_path, receipt, True)
        print(json.dumps({k: receipt[k] for k in ('status', 'scene_renders', 'elapsed_seconds', 'peak_cuda_allocated_bytes')}), flush=True)
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}', finished_utc=utc())
        write(receipt_path, receipt, True)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path)
    group.add_argument('--execute', type=Path)
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.prepare))
    else:
        execute(args.execute)
