"""Camera-only top4-normalized/MCMC RGB and H3/DINOv3 mask export.

Two sequential uv children keep the original IBGS extension isolated from gsplat.
The deployment bundle contains only bound runtime inputs, never evaluation PNGs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

SCHEMA = 'camera_only_top4_mcmc_h3_dinov3_v1'
NEUTRAL_NAME = '__camera_pose_only_target__.png'
LAYER_SHA = '1ccdfed8f19c285c0d4b9ac6a6ffa4a5829db475b9803033dcba5e45def43c8c'
ADAPTATIONS = (
    ("row = {'name': source['name'], 'image_id': source['image_id'],",
     "row = {'name': '__camera_pose_only_target__.png', 'image_id': source['image_id'],"),
    ("report['target_calls'] == report['selector_calls'] == 150",
     "report['target_calls'] == report['selector_calls'] == len(plan['views'])"),
)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def utc():
    return datetime.now(UTC).isoformat()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapt_layer_worker(content):
    require(hashlib.sha256(content.encode()).hexdigest() == LAYER_SHA, 'Exact frozen layer evaluator required')
    for old, new in ADAPTATIONS:
        require(content.count(old) == 1, 'Ambiguous camera/count adaptation')
        content = content.replace(old, new)
    return content


def verify_hashes(hashes):
    for path, expected in hashes.items():
        require(sha(path) == expected, 'Changed runtime dependency: '+str(path))


def bundle_helpers(bundle):
    verify_hashes(bundle['helper_hashes'])
    return load(bundle['main_helper'], 'joint_original_export')


def read_cameras(path, bundle, helper):
    raw = read(path)
    records = raw if isinstance(raw, list) else [raw]
    cameras = [helper.camera_record(value, i) for i, value in enumerate(records)]
    require(cameras and len({c['name'] for c in cameras}) == len(cameras), 'Empty or duplicate cameras')
    common = bundle['sensor']
    for camera in cameras:
        require((camera['width'], camera['height']) == (common['width'], common['height'])
                and np.array_equal(camera['K'], common['K'])
                and np.array_equal(camera['distortion'], common['distortion']),
                'This trained source bank supports only the fixed official sensor and intrinsics')
    return cameras


def pose_sources(camera, rows, neighbors):
    require(NEUTRAL_NAME not in {row['name'] for row in rows}, 'Reserved internal name collision')
    # Target display name must not change geometric source selection.
    indices = neighbors({**camera, 'name': NEUTRAL_NAME}, rows)
    require(1 <= len(indices) <= 4, 'Camera has no TRAIN sources within the trained support region')
    return [rows[i]['name'] for i in indices]


def native_barrier(records, names):
    require(len(names) > 0 and len(names) == len(set(names)) == len(records)
            and {(r['arm'], r['name']) for r in records} == {('top4_normalized', n) for n in names},
            'One fresh native top4 prediction per camera required')


def ibgs_child(bundle, cameras, output, report):
    import torch
    config = copy.deepcopy(bundle['ibgs'])
    verify_hashes(config['input_hashes'])
    verify_hashes({str(Path(config['source_snapshot'])/n): h for n, h in config['source_hashes'].items()})
    original = Path(bundle['layer_original'])
    require(Path(bundle['layer_adapted']).read_text() == adapt_layer_worker(original.read_text()),
            'Layer render control flow differs')
    base = load(bundle['layer_adapted'], 'joint_layer_render')
    base.ARMS = {'top4_normalized': ('top4', 'normalized')}
    base.prediction_barrier = native_barrier
    rows = read(config['data_contract']['path'])['train_rows']
    require(len(rows) == 350 and len({r['name'] for r in rows}) == 350
            and all(r['split'] == 'train' for r in rows), 'TRAIN-only source bank required')
    config['views'] = [{'camera': c, 'width': c['width'], 'height': c['height'],
                        'source_names': pose_sources(c, rows, base.legacy().neighbors)} for c in cameras]
    config['output'] = str(output)
    report.update(target_calls=0, selector_calls=0, source_depth_calls=0)
    base.render_stage(config, report)
    require(not torch.is_grad_enabled() or report['field_unchanged'], 'Field mutation')
    verify_hashes(config['input_hashes'])


def main_child(bundle, cameras, output, report, helper):
    import gc
    import importlib
    import importlib.metadata

    import cv2
    import torch
    config = bundle['main']
    verify_hashes(config['input_hashes'])
    verify_hashes({str(Path(config['source_snapshot'])/n): h for n, h in config['package_hashes'].items()})
    require({n: importlib.metadata.version(n) for n in config['runtime_versions']} == config['runtime_versions'],
            'Main uv runtime differs')
    require(not any(n == 'bridge_rgs' or n.startswith('bridge_rgs.') for n in sys.modules), 'Fresh process required')
    sys.path.insert(0, config['source_snapshot'])
    official = importlib.import_module('bridge_rgs.official_evaluate')
    ensemble = importlib.import_module('bridge_rgs.render_ensemble')
    torch.set_num_threads(8); cv2.setNumThreads(8)
    torch.cuda.reset_peak_memory_stats()
    report.update(scene_renders=0, teacher_calls=0, predictions=[])
    native = read(output/'ibgs_execution_receipt.json')
    require(native['status'] == 'completed' and native['bundle_sha256'] == report['bundle_sha256']
            and native['camera_json_sha256'] == report['camera_json_sha256'], 'IBGS stage mismatch')
    bound = native['native_predictions']; require(sha(bound['path']) == bound['sha256'], 'Native receipt changed')
    records = read(bound['path'])['records']; native_barrier(records, [c['name'] for c in cameras])
    for directory in ('rgb', 'mask', 'components/top4_normalized', 'components/mcmc'):
        (output/directory).mkdir(parents=True)
    # Same main-runtime float32 clip -> corner-v2 warp -> ties-even uint8 as scored candidate.
    lookup = {c['name']: c for c in cameras}
    for item in records:
        require(sha(item['path']) == item['sha256'], 'Native RGB changed')
        value = np.load(item['path'], allow_pickle=False); camera = lookup[item['name']]
        require(value.dtype == np.float32 and value.shape == (989, 1320, 3)
                and np.isfinite(value).all(), 'Invalid native RGB')
        _, _, _, back = official.distortion_render_grid(camera['K'], camera['distortion'],
            camera['width'], camera['height'], 'colmap_corner_v2')
        value = np.clip(value, 0., 1.)
        if back is not None:
            value = cv2.remap(value, back[..., 0], back[..., 1], cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        require(cv2.imwrite(str(output/'components/top4_normalized'/item['name']),
                            np.rint(value*255).astype(np.uint8)[..., ::-1]), 'IBGS PNG save failed')
    for role in ('selected', 'mcmc'):
        component = config['components'][role]
        scene, state = official._load_scene(component['checkpoint'], 'cuda')
        require(official.pixel_protocol(state) == component['profile'], 'Scene profile mismatch')
        scene.eval().requires_grad_(False)
        teacher = (ensemble.FixedRenderedTeacher(config['teacher_checkpoint'], component['checkpoint'], state,
                   device='cuda') if role == 'selected' else None)
        with torch.inference_mode():
            for camera in cameras:
                if role == 'selected':
                    K, width, height, back = official.distortion_render_grid(camera['K'], camera['distortion'],
                        camera['width'], camera['height'], component['profile'])
                    result = scene.render(torch.tensor(K, device='cuda'),
                        torch.tensor(camera['w2c'], device='cuda').float(), width, height, absgrad=False)
                    mask = helper.semantic_mask(result['rgb'].clamp(0, 1).cpu().numpy(),
                        result['probabilities'].cpu().numpy(), teacher, back)
                    require(cv2.imwrite(str(output/'mask'/camera['name']), mask), 'Mask save failed')
                    report['teacher_calls'] += 1
                    del result
                else:
                    rgb, _, _ = official.predict_official_camera(scene, camera, state)
                    require(cv2.imwrite(str(output/'components/mcmc'/camera['name']), rgb[..., ::-1]),
                            'MCMC PNG save failed')
                report['scene_renders'] += 1
        if teacher is not None:
            report['teacher'] = teacher.receipt
        del scene, state, teacher
        gc.collect(); torch.cuda.empty_cache()
    for camera in cameras:
        name = camera['name']
        members = [cv2.imread(str(output/'components'/role/name)) for role in ('top4_normalized', 'mcmc')]
        require(cv2.imwrite(str(output/'rgb'/name), helper.mean_rgb(*members)), 'Mean RGB save failed')
        report['predictions'].append({'name': name, 'rgb': str(output/'rgb'/name),
            'rgb_sha256': sha(output/'rgb'/name), 'mask': str(output/'mask'/name),
            'mask_sha256': sha(output/'mask'/name)})
    verify_hashes(config['input_hashes'])
    report['loaded_sources'] = helper.loaded_sources(config)
    report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()


def execute(args):
    started = time.monotonic()
    bundle_path, camera_path, output = [Path(v).resolve() for v in (args.bundle, args.cameras, args.output)]
    bundle = read(bundle_path)
    require(bundle['schema'] == SCHEMA and sha(__file__) == bundle['worker_sha256'], 'Frozen deployment worker required')
    helper = bundle_helpers(bundle)
    cameras = read_cameras(camera_path, bundle, helper)
    os.chdir(bundle['workspace_root'])
    report = {'schema': SCHEMA, 'status': 'running', 'stage': args.stage, 'started_utc': utc(),
        'bundle_sha256': sha(bundle_path), 'worker_sha256': sha(__file__),
        'camera_json_sha256': sha(camera_path), 'cameras': cameras, 'gt_payload_reads': 0,
        'annotation_reads': 0, 'optimizer_steps': 0}
    if args.stage == 'all':
        require(not output.exists(), 'Output directory must be fresh')
        output.mkdir(parents=True)
    else:
        require(output.is_dir() and not (output/f'{args.stage}_execution_receipt.json').exists(),
                'Child requires a fresh stage in an existing run directory')
    try:
        if args.stage == 'ibgs':
            ibgs_child(bundle, cameras, output, report)
        elif args.stage == 'main':
            main_child(bundle, cameras, output, report, helper)
        else:
            report['children'] = []
            for stage in ('ibgs', 'main'):
                command = (['uv', 'run', '--no-project', '--python', bundle['ibgs']['interpreter']]
                           if stage == 'ibgs' else ['uv', 'run', '--project', bundle['workspace_root'], '--no-sync'])
                command += ['python', '-B', str(Path(__file__).resolve()), '--bundle', str(bundle_path),
                    '--cameras', str(camera_path), '--output', str(output), '--stage', stage]
                child_start = time.monotonic()
                with (output/f'{stage}_stdout.log').open('x') as stream:
                    completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False,
                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4',
                             'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
                child = {'stage': stage, 'exit_code': completed.returncode,
                         'outer_elapsed_seconds': time.monotonic()-child_start}
                receipt = output/f'{stage}_execution_receipt.json'
                if receipt.exists():
                    child['execution_receipt_sha256'] = sha(receipt)
                report['children'].append(child)
                require(completed.returncode == 0, f'{stage} child failed; see its stdout log')
                result = read(receipt)
                require(result['status'] == 'completed' and result['bundle_sha256'] == sha(bundle_path)
                        and result['camera_json_sha256'] == sha(camera_path), 'Incomplete or mismatched child')
                print(json.dumps(child), flush=True)
            first, second = [read(output/f'{s}_execution_receipt.json') for s in ('ibgs', 'main')]
            report.update(scene_renders=first['target_calls']+second['scene_renders'],
                selector_calls=first['selector_calls'], teacher_calls=second['teacher_calls'],
                source_rgb_decodes=first['source_rgb_decodes'], predictions=second['predictions'],
                peak_cuda_allocated_bytes=max(first['peak_allocated_bytes'], second['peak_cuda_allocated_bytes']))
            require(report['scene_renders'] == 3*len(cameras)
                    and report['selector_calls'] == report['teacher_calls'] == len(cameras), 'Wrong inference counts')
        require(sha(bundle_path) == report['bundle_sha256'] and sha(camera_path) == report['camera_json_sha256'],
                'Inputs changed during export')
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, finished_utc=utc())
        name = 'execution_receipt.json' if args.stage == 'all' else f'{args.stage}_execution_receipt.json'
        write(output/name, report)
    print(json.dumps({'status': report['status'], 'stage': args.stage, 'views': len(cameras), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--cameras', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stage', choices=['all', 'ibgs', 'main'], default='all', help=argparse.SUPPRESS)
    execute(parser.parse_args())
