"""Collect 64 prepared-TRAIN soft canvases before any label/valid payload is read."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
PRIOR = RUNS/'multifield_h3_teacher_v1'
MATCHED = RUNS/'matched_rgb_teacher_adaptation_v1'
CACHE = MATCHED/'cache'
DOCUMENT = 'cross_renderer_authority_protocol.md'
HELPER = Path(__file__).with_name('multifield_collection_base.py')
if not HELPER.exists():
    HELPER = PRIOR/'source_snapshot/evaluate_multifield_h3_teacher.py'
_spec = importlib.util.spec_from_file_location('multifield_collection_base', HELPER)
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)
base, matched = shared.base, shared.matched
read, sha, write, require, utc = base.read, base.sha, base.write, base.require, base.utc
H3_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
TEACHER_SHA = '8f277864f32bc2bc1a116cf00013594b96f810a469fd9d4012ad21ad51c96b37'
KINDS = ('S', 'T1', 'T2', 'Tm')
SPEC = {'protocol': 'cross_renderer_authority_collection_v1', 'grid': 'legacy_mixed_v1 prepared TRAIN canvas',
    'selection': '259 labeled TRAIN sorted by name; index floor(i*258/15), i=0..15',
    'views': 16, 'scene_renders': 16, 'teacher_predict_image_calls': 48, 'soft_canvases': 64,
    'teacher_checkpoint_sha256': TEACHER_SHA, 'h3_checkpoint_sha256': H3_SHA,
    'teacher_inputs': {'T1': 'capacity_1m original RGB', 'T2': 'mcmc original RGB', 'Tm': 'fixed equal original RGB mean'},
    'teacher_adapter': 'original uint8 -> existing legacy cv2 map/INTER_LINEAR/zero border, no output back-warp',
    'inference': dict(shared.INFERENCE), 'h3_input': 'Own rendered RGB and original complete H3 field/head; no RGB substitution',
    'labels': 'No label/valid/real RGB payload reads or hashes; inherit TRAIN target SHA from completed training plan',
    'optimization_steps': 0, 'internal_seconds': 570, 'external_seconds': 600, 'retries': 0,
    'gpu_process_policy': dict(shared.SPEC['gpu_process_policy']),
    'limits': 'Models fitted these TRAIN cameras; this is not base-model OOF/new-view evidence or a pure appearance intervention'}


def select_views(manifest):
    train = sorted((v for v in manifest['views'] if v['split'] == 'train'), key=lambda v: v['name'])
    labeled = [v for v in train if v.get('mask_path')]
    require(len(train) == 350 and len(labeled) == 259 and len({v['name'] for v in train}) == 350,
            'Require the fixed 350 TRAIN / 259 unique labeled TRAIN population')
    indices = [i*258//15 for i in range(16)]
    return [labeled[i] for i in indices], indices


def target_declarations(view, old_input_hashes):
    # Metadata only: do not open, hash or decode either target here.
    records = {}
    for role, key in [('mask', 'mask_path'), ('valid', 'valid_path')]:
        path = str(Path(view[key]).resolve())
        require(path in old_input_hashes, 'Target lacks a historical training SHA')
        records[role] = {'path': path, 'sha256': old_input_hashes[path]}
    return records


def barrier(records, views):
    require(len(views) == 16 and len(records) == 16
            and [r['name'] for r in records] == [v['name'] for v in views]
            and len({r['name'] for r in records}) == 16
            and all(all(k in r for k in KINDS) for r in records), 'All 64 unique soft canvases must finish before analysis')


def analysis_declaration(report, views, receipt_record, specification=None):
    require(report['status'] == 'completed' and report['inputs_and_sources_unchanged'],
            'Failed/partial collection cannot authorize label analysis')
    barrier(report['predictions'], views)
    targets = {v['name']: v for v in views}
    result = {'status': 'predictions_completed', 'protocol': 'cross_renderer_authority_analysis_v1',
        'prediction_receipt': receipt_record,
        'views': [dict(r, **{k: targets[r['name']][k] for k in ('mask', 'valid')}) for r in report['predictions']]}
    if specification is not None:
        result['specification'] = specification
    return result


def prepare(output):
    require(not torch.cuda.is_initialized(), 'CPU preparation only')
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite a diagnostic')
    prior, prior_receipt = read(PRIOR/'plan.json'), read(PRIOR/'execution_receipt.json')
    cache_plan, cache = read(CACHE/'plan.json'), read(CACHE/'execution_receipt.json')
    train_plan = read(MATCHED/'plan.json')
    training = read(MATCHED/'training_execution_receipt.json')
    require(prior_receipt['status'] == cache['status'] == training['status'] == 'completed', 'Incomplete sources')
    require(prior_receipt['plan_sha256'] == sha(PRIOR/'plan.json')
            and cache['plan_sha256'] == sha(CACHE/'plan.json')
            and training['plan_sha256'] == sha(MATCHED/'plan.json'), 'Source plan binding differs')
    require(prior_receipt['bound_inputs_and_sources_unchanged'] and cache['bound_sources_inputs_unchanged'],
            'Source invariance check failed')
    audit = read(MATCHED/'independent_training_review.json')
    require(audit['status'] == 'passed' and audit['plan_sha256'] == sha(MATCHED/'plan.json'), 'Training audit not passed')
    manifest_path = Path(cache_plan['original_manifest'])
    manifest = read(manifest_path)
    selected, indices = select_views(manifest)
    forbidden = {str(Path(v[k]).resolve()) for v in manifest['views']
                 for k in ('image_path', 'mask_path', 'valid_path', 'source_image_path', 'source_annotation_path') if v.get(k)}
    inputs = {}
    def bind(path, expected=None):
        path = str(Path(path).resolve())
        require(path not in forbidden, 'Collector cannot read real image/label/valid payloads')
        if path in inputs:
            require(expected is None or inputs[path] == expected, 'Conflicting source SHA')
            return
        digest = sha(path)
        require(expected is None or expected == digest, f'Source changed: {path}')
        inputs[path] = digest
    for p in (PRIOR/'plan.json', PRIOR/'execution_receipt.json', CACHE/'plan.json', CACHE/'execution_receipt.json',
              MATCHED/'plan.json', MATCHED/'training_execution_receipt.json', MATCHED/'independent_training_review.json',
              ROOT/'uv.lock', Path(__file__), ROOT/'docs'/DOCUMENT, ROOT/'tests/test_cross_renderer_authority_collection.py'):
        bind(p)
    analysis_path = ROOT/'scripts/analyze_cross_renderer_authority.py'
    analysis_tests = ROOT/'tests/test_cross_renderer_authority_analysis.py'
    bind(analysis_path)
    bind(analysis_tests)
    analysis_spec = importlib.util.spec_from_file_location('authority_analysis_contract', analysis_path)
    analysis_module = importlib.util.module_from_spec(analysis_spec)
    analysis_spec.loader.exec_module(analysis_module)
    require(analysis_module.SPEC['protocol'] == 'cross_renderer_authority_analysis_v1', 'Wrong analysis contract')
    bind(manifest_path, cache_plan['original_manifest_sha256'])
    original_snapshot = Path(prior['source_snapshot'])
    require(base.source_files(original_snapshot) == prior['source_hashes'], 'Successful multifield source changed')
    require(sha(HELPER) == prior['source_hashes']['evaluate_multifield_h3_teacher.py'], 'Wrong frozen helper')
    for rel, digest in prior['source_hashes'].items():
        bind(original_snapshot/rel, digest)
    scene = prior['components']['selected']
    teacher = prior['teachers']['composite']
    require(scene['checkpoint_sha256'] == H3_SHA and teacher['sha256'] == TEACHER_SHA, 'Wrong fixed endpoints')
    bind(scene['checkpoint'], H3_SHA)
    bind(teacher['path'], TEACHER_SHA)
    stage = MATCHED/'composite/stage_receipt.json'
    bind(stage, training['stages']['composite']['receipt_sha256'])
    sr = read(stage)
    require(training['stages']['composite']['exit_code'] == 0 and sr['status'] == 'completed'
            and sr['steps'] == 2000 and sr['last_checkpoint_sha256'] == TEACHER_SHA, 'Need natural fixed 2k last')
    backing = prior['backbone']
    for name, digest in {'config.json': backing['config_sha256'], **backing['weights_sha256'],
                         **backing['index_sha256']}.items():
        bind(Path(backing['model_dir'])/name, digest)
    component_records = {k: {r['name']: r for r in cache['component_records'][k]} for k in ('capacity_1m', 'mcmc')}
    composite_records = {r['name']: r for r in cache['adapter_records']['new_composite']}
    train_cameras = {c['name']: c for c in cache_plan['train_cameras']}
    rows = []
    for view in selected:
        name, camera = view['name'], train_cameras[view['name']]
        require(camera['split'] == 'train' and np.array_equal(np.asarray(camera['w2c']), np.asarray(view['w2c_original'])),
                'Prepared and source camera pose differ')
        composed = composite_records[name]
        rgb = {'T1': component_records['capacity_1m'][name], 'T2': component_records['mcmc'][name],
               'Tm': composed['original_rgb']}
        for record in rgb.values():
            bind(record['image_path'], record['sha256'])
        bind(composed['image_path'], composed['sha256'])
        expected_K = np.asarray(view['K'], np.float32)
        info = composed['adapter_info']
        require(info['canvas'] == [view['width'], view['height']]
                and np.array_equal(expected_K, np.asarray(info['canvas_K'], np.float32)), 'TRAIN legacy canvas mismatch')
        require(all(component_records[k][name]['camera'] == camera for k in component_records), 'Component camera mismatch')
        rows.append({'name': name, 'split': 'train', 'camera': camera, 'K': expected_K.tolist(),
                     'width': view['width'], 'height': view['height'], 'rgb_inputs': rgb,
                     'composite_legacy': {'path': composed['image_path'], 'sha256': composed['sha256']},
                     'adapter_info': info, **target_declarations(view, train_plan['input_hashes'])})
    require(not forbidden.intersection(inputs), 'Target bytes were bound accidentally')
    require(shutil.disk_usage(output.parent).free > 3*1024**3, 'Need 3 GiB for fixed soft canvases')
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(original_snapshot/'bridge_rgs', snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for path, name in [(Path(__file__), Path(__file__).name), (HELPER, 'multifield_collection_base.py'),
        (original_snapshot/'matched_teacher_evaluation_base.py', 'matched_teacher_evaluation_base.py'),
        (original_snapshot/'rgb_teacher_transfer_base.py', 'rgb_teacher_transfer_base.py'),
        (ROOT/'docs'/DOCUMENT, DOCUMENT), (ROOT/'tests/test_cross_renderer_authority_collection.py', 'test_cross_renderer_authority_collection.py')]:
        shutil.copy2(path, snapshot/name)
    shutil.copy2(analysis_path, snapshot/analysis_path.name)
    shutil.copy2(analysis_tests, snapshot/analysis_tests.name)
    plan = {'status': 'locked_pending_root_gpu_handoff', 'specification': SPEC, 'root': str(ROOT), 'output': str(output),
        'source_snapshot': str(snapshot), 'source_hashes': base.source_files(snapshot), 'input_hashes': inputs,
        'views': rows, 'selected_indices': indices, 'scene': scene, 'teacher': teacher, 'backbone': backing,
        'runtime_versions': prior['runtime_versions'], 'external_timeout_seconds': 600,
        'prepare_gt_payload_reads': 0, 'prepare_real_rgb_payload_reads': 0, 'prepare_cuda_initialized': False,
        'target_sha_source': {'path': str(MATCHED/'plan.json'), 'sha256': sha(MATCHED/'plan.json')},
        'analysis_specification': analysis_module.SPEC,
        'env': dict(prior['env'], PYTHONPATH=str(snapshot))}
    write(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    require(plan['specification'] == SPEC and base.source_files(Path(plan['source_snapshot'])) == plan['source_hashes'],
            'Source/specification changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Execute frozen worker only')
    require(all(sha(p) == h for p,h in plan['input_hashes'].items()), 'Bound source/input changed')
    require({k: importlib.metadata.version(k) for k in plan['runtime_versions']} == plan['runtime_versions'], 'Versions changed')


def imports(plan):
    records = shared.runtime_imports(plan)
    records['multifield_collection_base'] = {'path': str(HELPER.resolve()), 'sha256': sha(HELPER)}
    for r in records.values():
        p = Path(r['path'])
        require(p.is_relative_to(plan['source_snapshot'])
                and plan['source_hashes'][str(p.relative_to(plan['source_snapshot']))] == r['sha256'], 'Unfrozen import')
    return records


def save_probability(path, value):
    require(value.dtype == np.float32 and value.ndim == 3 and value.shape[-1] == 5
            and np.isfinite(value).all() and (value >= 0).all()
            and np.allclose(value.sum(-1), 1., atol=2e-5, rtol=0), 'Need valid HWC FP32 probabilities')
    return shared.save_soft(path, value)


def execute(path):
    path = Path(path).resolve()
    plan, started = read(path), time.monotonic()
    output = Path(plan['output'])
    write(output/'execution_started.json', {'utc': utc(), 'pid': os.getpid(), 'plan_sha256': sha(path)})
    report = {'status': 'failed', 'plan_sha256': sha(path), 'specification': SPEC, 'started_utc': utc(),
        'predictions': [], 'scene_renders': 0, 'teacher_predict_image_calls': 0,
        'gt_payload_reads': 0, 'real_rgb_payload_reads': 0, 'optimizer_steps': 0}
    handler = signal.getsignal(signal.SIGALRM)
    def deadline(*_):
        raise TimeoutError('Fixed 570s collection deadline; no retry')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(570)
    try:
        os.chdir(plan['root'])
        torch.set_num_threads(8)
        cv2.setNumThreads(8)
        verify(plan)
        sys.path.insert(0, plan['source_snapshot'])
        official = importlib.import_module('bridge_rgs.official_evaluate')
        teacher_module = importlib.import_module('bridge_rgs.teacher')
        report['actual_imports'] = imports(plan)
        gpu = subprocess.run(['nvidia-smi','-q','-x'],capture_output=True,text=True,check=True,timeout=5).stdout
        report['gpu_before'] = matched.gpu_process_records(gpu)
        matched.enforce_gpu_process_policy(report['gpu_before'])
        report['gpu_inventory_sha256'] = hashlib.sha256(gpu.encode()).hexdigest()
        report['nonexclusive_cost_scope'] = SPEC['gpu_process_policy']['cost_scope']
        scene, state = official._load_scene(plan['scene']['checkpoint'], 'cuda')
        require(official.pixel_protocol(state) == base.LEGACY, 'H3 profile changed')
        scene.eval().requires_grad_(False)
        teacher_state = torch.load(plan['teacher']['path'], map_location='cpu', weights_only=False)
        require(teacher_state['step'] == 2000 and matched.profile_id(teacher_state) == base.LEGACY
                and teacher_state['configuration']['adapter_rank'] == 0, 'Wrong teacher endpoint/profile')
        model = teacher_module.load_teacher(plan['backbone']['model_dir'], 5, teacher_state['configuration']['channels'],
            device='cuda', pixel_profile=base.LEGACY, **teacher_module.checkpoint_adapter_options(teacher_state['configuration']))
        model.decoder.load_state_dict(teacher_state['ema_decoder'], strict=True)
        teacher_module.load_checkpoint_adapters(model, teacher_state)
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            for view in plan['views']:
                camera, name = view['camera'], view['name']
                mx, my, _, info = base.adapter_maps(camera, official.distortion_render_grid)
                require(info == view['adapter_info'], 'Adapter differs from bound TRAIN cache')
                K, width, height, _ = shared.verify_canvas(camera, state, info, official)
                require(np.array_equal(K, np.asarray(view['K'],np.float32))
                        and [width,height] == [view['width'],view['height']], 'Prepared TRAIN grid differs')
                result = scene.render(torch.tensor(K,device='cuda'),torch.tensor(camera['w2c'],device='cuda').float(),
                                      width,height,absgrad=False)
                report['scene_renders'] += 1
                row = {'name': name, 'split': 'train', 'adapter_info': info}
                row['S'] = save_probability(output/'soft/S'/f'{Path(name).stem}.npy', result['probabilities'].cpu().numpy())
                del result
                decoded = {kind: shared.decode(record['image_path']) for kind,record in view['rgb_inputs'].items()}
                require(np.array_equal(shared.average_rgb(decoded['T1'],decoded['T2']),decoded['Tm']), 'Bound mean RGB differs')
                for kind in ('T1','T2','Tm'):
                    canvas = base.input_canvas(decoded[kind],mx,my)
                    if kind == 'Tm':
                        require(np.array_equal(canvas,shared.decode(view['composite_legacy']['path'])), 'Mean legacy adapter differs')
                    probabilities, _ = teacher_module.predict_image(model,canvas,**SPEC['inference'])
                    report['teacher_predict_image_calls'] += 1
                    row[kind] = save_probability(output/'soft'/kind/f'{Path(name).stem}.npy',
                                                 np.ascontiguousarray(probabilities.transpose(1,2,0),dtype=np.float32))
                report['predictions'].append(row)
        barrier(report['predictions'],plan['views'])
        require(report['scene_renders'] == 16 and report['teacher_predict_image_calls'] == 48, 'Fixed call budget changed')
        require(all(p.grad is None and not p.requires_grad for p in list(scene.parameters())+list(model.parameters())),
                'Inference unexpectedly enabled parameter gradients')
        report.update(status='completed',predictions_finished_utc=utc(),actual_imports=imports(plan))
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,handler)
        report['inputs_and_sources_unchanged'] = base.source_files(Path(plan['source_snapshot'])) == plan['source_hashes'] \
            and all(sha(p)==h for p,h in plan['input_hashes'].items())
        if not report['inputs_and_sources_unchanged']:
            report['status'] = 'failed'
        report['elapsed_seconds'] = time.monotonic()-started
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        write(output/'execution_receipt.json',report)
    require(report['status'] == 'completed', 'Collection did not complete')
    analysis = analysis_declaration(report, plan['views'],
        {'path':str(output/'execution_receipt.json'),'sha256':sha(output/'execution_receipt.json')},
        plan.get('analysis_specification'))
    write(output/'analysis_input.json',analysis)
    print({'status':'completed','analysis_input':str(output/'analysis_input.json')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',type=Path)
    mode.add_argument('--execute',type=Path)
    args=parser.parse_args()
    if args.prepare:
        p=prepare(args.prepare)
        print({'plan':str(p),'plan_sha256':sha(p),'runner_sha256':sha(__file__)})
    else:
        execute(args.execute)


if __name__=='__main__':
    main()
