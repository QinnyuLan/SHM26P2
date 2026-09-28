"""Two fixed 6k warm IBGS engineering arms. Prepare on CPU; root owns GPU launch."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/home/sky/workspace/SHM2026')
REPOSITORY = Path('/mnt/data/SHM2026/third_party/ibgs')
CHECKPOINT = Path('/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1/training/last.pt')
CHECKPOINT_SHA = '35b489fe45ad34ad279018609b6d5eaf6493e4c0d82f9c86ec16925268078092'
PREFLIGHT = Path('/mnt/data/SHM2026/runs/ibgs_port_preflight_v1')
CONTRACT = PREFLIGHT/'data_contract.json'
CONTRACT_SHA = '599f6cbf09a16f56bef6f37d0e0f8ce0a3f24449376213963ee033fc2ea9d84b'
PORT = Path('/mnt/data/SHM2026/third_party/ibgs_port_revision1')
SUPPORT_REVISION = Path('/mnt/data/SHM2026/third_party/ibgs_source_support_revision1/receipt.json')
FUSION_BEFORE_SHA = '7139b23d2c9ab8447ed9dba61063570e04d0fd5e33b511db63568cfdcacacaa1'
FUSION_AFTER_SHA = 'd276ff675c8119f64c7b47e32da95f548e9458c7fdc0c2452882a716099ef5b8'
BINARY = REPOSITORY/'.venv/lib/python3.11/site-packages/diff_plane_rasterization/_C.cpython-311-x86_64-linux-gnu.so'
BINARY_SHA = '436b2b55df2b37cddda59e606ec0855ad125eb211058b502197674bd2b93adaf'
PROTOCOL = ROOT/'docs/ibgs_warm_matched_protocol.md'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    data = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(data)


def tree_hashes(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.md'}}


def helper_import(snapshot=None):
    sys.path.insert(0, str(snapshot or ROOT/'src'))
    from bridge_rgs import ibgs_warm_training
    return ibgs_warm_training


def prepare(output):
    h = helper_import()
    output = output.resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    require(sha(CHECKPOINT) == CHECKPOINT_SHA and sha(CONTRACT) == CONTRACT_SHA
            and sha(BINARY) == BINARY_SHA, 'Fixed input/backend changed')
    preplan = json.loads((PREFLIGHT/'plan_v2.json').read_text())
    pre = json.loads((PREFLIGHT/'result.json').read_text())
    require(pre['status'] == 'passed' and pre['plan_sha256'] == sha(PREFLIGHT/'plan_v2.json')
            and pre['optimizer_steps'] == 0 and pre['field_parameters_unchanged'], 'Passed real preflight required')
    support = json.loads(SUPPORT_REVISION.read_text())
    support_inputs = {str(SUPPORT_REVISION): sha(SUPPORT_REVISION)}
    for change in support['changes']:
        require(sha(change['before']) == change['before_sha256']
                and sha(change['after']) == sha(change['path']) == change['after_sha256'],
                'Support correction source changed')
        support_inputs.update({change['before']: change['before_sha256'],
                               change['after']: change['after_sha256']})
    require(sha(support['patch']['path']) == support['patch']['sha256'], 'Support patch changed')
    support_inputs[support['patch']['path']] = support['patch']['sha256']
    for path, expected in preplan['source_hashes'].items():
        if Path(path) == REPOSITORY/'color_aggregation_network.py':
            require(expected == FUSION_BEFORE_SHA and sha(path) == FUSION_AFTER_SHA,
                    'Only the explicit source-support correction is allowed')
        else:
            require(sha(path) == expected, 'Preflight source changed: '+path)
    contract = json.loads(CONTRACT.read_text())
    rows = contract['train_rows']
    require(len(rows) == 350 and all(v['split'] == 'train' for v in rows), 'Fixed TRAIN population')
    names = [v['name'] for v in rows]
    require(names == sorted(names) and len(set(names)) == 350, 'Ordered unique source bank required')
    # Hash only bound TRAIN RGB/valid. No image decode or VAL/semantic payload.
    require(all(sha(p) == v for p, v in contract['pixel_hashes'].items()), 'TRAIN inputs changed')
    output.mkdir()
    snapshot = output/'source_snapshot'
    package = snapshot/'bridge_rgs'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    for name in ('ibgs_warm_training.py', 'ibgs_adapter.py', 'ibgs_losses.py', 'ibgs_camera_contract.py'):
        shutil.copyfile(ROOT/'src/bridge_rgs'/name, package/name)
    official = snapshot/'official'
    for name in ('scene', 'utils', 'gaussian_renderer', 'arguments'):
        for source in sorted((REPOSITORY/name).rglob('*.py')):
            target = official/source.relative_to(REPOSITORY)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    shutil.copyfile(REPOSITORY/'color_aggregation_network.py', official/'color_aggregation_network.py')
    shutil.copyfile(__file__, snapshot/Path(__file__).name)
    shutil.copyfile(PROTOCOL, snapshot/PROTOCOL.name)
    test = ROOT/'tests/test_ibgs_warm_training.py'
    require(test.is_file(), 'CPU contracts must exist before preparation')
    shutil.copyfile(test, snapshot/test.name)
    order = h.camera_order(names)
    write(output/'camera_order.json', {'seed': 42, 'names': order})
    inputs = {str(CHECKPOINT): CHECKPOINT_SHA, str(CONTRACT): CONTRACT_SHA,
              **contract['pixel_hashes'], **support_inputs}
    for path in (PREFLIGHT/'plan_v2.json', PREFLIGHT/'result.json', PORT/'port_receipt.json',
                 PORT/'verification_launch_receipts.json',
                 Path('/mnt/data/SHM2026/third_party/ibgs_build_validation_v1/requirements_frozen.txt')):
        inputs[str(path)] = sha(path)
    runtime = {str(BINARY): BINARY_SHA}
    installed = REPOSITORY/'.venv/lib/python3.11/site-packages'
    for path in (installed/'diff_plane_rasterization/__init__.py',
                 installed/'simple_knn/_C.cpython-311-x86_64-linux-gnu.so'):
        runtime[str(path)] = sha(path)
    # Keep compiler/math source identity explicitly, even though execution loads .so.
    port = json.loads((PORT/'port_receipt.json').read_text())
    for name, value in port['after_source_hashes'].items():
        require(sha(REPOSITORY/name) == value, 'Port source drift')
        runtime[str(REPOSITORY/name)] = value
    plan = {'protocol': h.SPEC['protocol'], 'specification': h.SPEC,
            'output': str(output), 'source_snapshot': str(snapshot),
            'source_hashes': tree_hashes(snapshot), 'input_hashes': inputs,
            'runtime_sources': runtime, 'checkpoint': str(CHECKPOINT),
            'source_support_revision': {'path': str(SUPPORT_REVISION), 'sha256': sha(SUPPORT_REVISION),
                'preflight_fusion_sha256': FUSION_BEFORE_SHA, 'training_fusion_sha256': FUSION_AFTER_SHA,
                'scope': 'Only structural support flag; both arms; old preflight remains unchanged'},
            'data_contract': str(CONTRACT), 'data_contract_sha256': CONTRACT_SHA,
            'camera_order': str(output/'camera_order.json'),
            'camera_order_sha256': sha(output/'camera_order.json'), 'arms': list(h.ARMS),
            'interpreter': str(REPOSITORY/'.venv/bin/python'),
            'coordinate_profile': 'colmap_corner_v2',
            'environment': {'torch': '2.8.0+cu128', 'tf32': False, 'cudnn_benchmark': False,
                            'matmul_precision': 'highest', 'autocast': False,
                            'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1',
                            'MKL_NUM_THREADS': '1', 'MAX_JOBS': '4', 'TORCH_CUDA_ARCH_LIST': '12.0'},
            'prepared_without_model_deserialization_or_pixel_decode': True}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'plan_sha256': sha(output/'plan.json'),
                      'source_files': len(plan['source_hashes']), 'input_files': len(inputs)}), flush=True)


def verify(plan):
    require(tree_hashes(plan['source_snapshot']) == plan['source_hashes'], 'Frozen sources changed')
    for field in ('input_hashes', 'runtime_sources'):
        require(all(sha(p) == value for p, value in plan[field].items()), f'{field} changed')
    require(sha(plan['camera_order']) == plan['camera_order_sha256'], 'Camera order changed')


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def finite_tensors(values, label):
    import torch
    for name, value in values:
        if value is not None:
            require(bool(torch.isfinite(value).all()), f'Nonfinite {label}: {name}')


def optimizer_tensors(optimizer):
    for index, state in optimizer.state.items():
        for name, value in state.items():
            if hasattr(value, 'dtype'):
                yield f'{id(index)}:{name}', value


def save_endpoint(path, *, field, net, field_opt, head_opt, background, arm, step,
                  plan, plan_sha, neighbors, counts, status):
    import torch
    h = helper_import(Path(plan['source_snapshot']))
    require(not path.exists(), 'Do not overwrite endpoints')
    value = {'protocol': h.SPEC['protocol'], 'status': status, 'arm': arm, 'step': step,
             'field': {name: getattr(field, name).detach().cpu() for name in h.FIELD_KEYS},
             'head': cpu_tree(net.inner.state_dict()), 'background': background.detach().cpu(),
             'sh_degree': 3, 'scene_scale': field.spatial_lr_scale,
             'field_optimizer': cpu_tree(field_opt.state_dict()),
             'head_optimizer': cpu_tree(head_opt.state_dict()), 'specification': h.SPEC,
             'source_plan_sha256': plan_sha, 'sample_order_sha256': plan['camera_order_sha256'],
             'data_contract': {'path': plan['data_contract'], 'sha256': plan['data_contract_sha256']},
             'coordinate_profile': plan['coordinate_profile'], 'neighbors': neighbors,
             'counts': counts, 'rng': {'torch_cpu': torch.get_rng_state(),
                                       'torch_cuda': torch.cuda.get_rng_state_all()}}
    torch.save(value, path)


def train_arm(plan, plan_sha, arm, report):
    import cv2
    import numpy as np
    import torch
    h = helper_import(Path(plan['source_snapshot']))
    sys.path.insert(0, str(Path(plan['source_snapshot'])/'official'))
    from color_aggregation_network import ColorFusionResidualNet, fuse_color
    from gaussian_renderer import render, render_depth

    from bridge_rgs.ibgs_adapter import TrainScene
    directory = Path(plan['output'])/arm
    directory.mkdir()
    start = time.monotonic()
    signal.alarm(h.SPEC['internal_seconds_per_arm'])
    field = net = field_opt = head_opt = background = None
    completed = 0
    counts = {'depth_initialization_forwards': 0, 'target_forwards': 0, 'backwards': 0,
              'field_optimizer_steps': 0, 'head_optimizer_steps': 0, 'empty_support': 0,
              'fusion_calls': 0, 'depth_cache_updates': 0, 'low_level_raster_calls': 0}
    contract = json.loads(Path(plan['data_contract']).read_text())
    neighbors = {v['name']: [n['name'] for n in v['neighbors_4']]
                 for v in contract['neighbors']['views']}
    original_imread = cv2.imread
    allowed = set(contract['pixel_hashes'])
    decoded = []
    def checked_imread(path, *args, **kwargs):
        require(str(Path(path).resolve()) in allowed, 'Only bound TRAIN RGB/valid can be decoded')
        decoded.append(str(Path(path).resolve()))
        return original_imread(path, *args, **kwargs)
    cv2.imread = checked_imread
    from diff_plane_rasterization import _C
    original_raster = _C.rasterize_gaussians
    def counted_raster(*args, **kwargs):
        counts['low_level_raster_calls'] += 1
        return original_raster(*args, **kwargs)
    _C.rasterize_gaussians = counted_raster
    report.update(status='failed', arm=arm, steps=0, counts=counts)
    trace = directory/'trace.jsonl'
    try:
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        field, background = h.load_warm_field(plan['checkpoint'])
        scene = TrainScene(contract['train_rows'], contract['pixel_hashes'], neighbors,
                           field, contract['manifest_scene_radius'])
        cameras = {c.image_name: c for c in scene.cameras}
        rows = contract['train_rows']
        raw_net = ColorFusionResidualNet(height=rows[0]['height'], width=rows[0]['width']).cuda()
        h.initialize_fusion(raw_net)
        net = h.SourceAblatedNet(raw_net, arm)
        field_opt, head_opt = h.build_optimizers(field, net, field.spatial_lr_scale)
        report['initial_field_hashes'] = {name: tensor_hash(getattr(field, name)) for name in h.FIELD_KEYS}
        report['initial_head_hashes'] = {name: tensor_hash(v) for name, v in raw_net.state_dict().items()}
        report['background_sha256'] = tensor_hash(background)
        pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
        options = SimpleNamespace(shuffle_source_frame=False, enable_exposure_correction=False,
                                  nb_visible_src_frames=3, residual_resolution_scale=1.)
        order = json.loads(Path(plan['camera_order']).read_text())['names']
        require(order == h.camera_order(list(cameras)), 'Order not the fixed seed42 epoch schedule')
        with torch.no_grad():
            for camera in scene.cameras:
                depth = render_depth(camera, field, scene, pipe, options, background, True, 4, 4, .01)
                scene.rendered_depth_list[camera.uid] = depth
                counts['depth_initialization_forwards'] += 1
                del depth
        with trace.open('x') as events, (directory/'metrics.jsonl').open('x') as metrics:
            for step, name in enumerate(order, 1):
                camera = cameras[name]
                field_opt.zero_grad(set_to_none=True); head_opt.zero_grad(set_to_none=True)
                package = render(camera, field, scene, pipe, options, background,
                    learnt_normal=True, nb_src_frames=4, buffer_length=4, depth_error_threshold=.01,
                    do_find_closest_frame=False, do_render_src_depth=False,
                    render_geo=True, return_depth_normal=True)
                counts['target_forwards'] += 1
                fusion = h.fuse_training(package, net, options, step, fuse_color)
                if fusion is not None:
                    counts['fusion_calls'] += 1
                    counts['empty_support'] += int(fusion['empty_support'])
                target = scene.original_image_list[camera.uid].cuda()
                loss, pieces = h.training_losses(package, fusion, target, scene.valid, step)
                require(bool(torch.isfinite(loss)), 'Nonfinite objective')
                loss.backward(); counts['backwards'] += 1
                finite_tensors(((n, getattr(field, n).grad) for n in h.FIELD_KEYS), 'field gradient')
                finite_tensors(net.named_parameters(), 'head parameter before update')
                finite_tensors(((n, p.grad) for n, p in net.named_parameters()), 'head gradient')
                field_opt.step(); counts['field_optimizer_steps'] += 1
                head_update = fusion is not None and not fusion['empty_support']
                if head_update:
                    require(any(p.grad is not None for p in net.parameters()), 'Missing active head gradient')
                    head_opt.step(); counts['head_optimizer_steps'] += 1
                finite_tensors(((n, getattr(field, n)) for n in h.FIELD_KEYS), 'updated field')
                finite_tensors(net.named_parameters(), 'updated head')
                finite_tensors(optimizer_tensors(field_opt), 'field Adam state')
                finite_tensors(optimizer_tensors(head_opt), 'head Adam state')
                # No additional renderer: this is the pre-update target depth,
                # detached/masked by the existing bank after the update.
                scene.rendered_depth_list[camera.uid] = package['median_intersected_depth']
                counts['depth_cache_updates'] = scene.rendered_depth_list.updates
                completed = step
                event = {'step': step, 'name': name, 'stage': h.stage_for_step(step),
                         'sources': camera.nearest_names, 'head_update': head_update,
                         'finite': True, 'N': len(field._xyz)}
                events.write(json.dumps(event, allow_nan=False)+'\n')
                if step % 200 == 0:
                    torch.cuda.synchronize()
                    row = {**event, 'loss': loss.item(),
                           **{k: (None if v is None else float(v)) for k, v in pieces.items()},
                           'support_fraction': (None if fusion is None else float(fusion['valid_warp_mask'].mean())),
                           'elapsed_seconds': time.monotonic()-start,
                           'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                           'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'counts': dict(counts)}
                    metrics.write(json.dumps(row, allow_nan=False)+'\n'); metrics.flush(); events.flush()
                    print(json.dumps({'arm': arm, **row}, allow_nan=False), flush=True)
                del package, fusion, target, loss, pieces
        torch.cuda.synchronize()
        require(completed == 6000 and len(field._xyz) == 996009, 'Incomplete/flexible population')
        require(counts['target_forwards'] == counts['backwards'] == counts['field_optimizer_steps'] == 6000
                and counts['depth_initialization_forwards'] == 350
                and counts['low_level_raster_calls'] == counts['depth_cache_updates'] == 6350,
                'Unexpected training/render/cache counts')
        final = directory/'last.pt'
        save_endpoint(final, field=field, net=net, field_opt=field_opt, head_opt=head_opt,
                      background=background, arm=arm, step=completed, plan=plan, plan_sha=plan_sha,
                      neighbors=neighbors, counts=counts, status='completed')
        report.update(status='completed', steps=completed, last_checkpoint=str(final),
                      checkpoint_sha256=sha(final), source_plan_sha256=plan_sha,
                      sample_order_sha256=plan['camera_order_sha256'],
                      trace_path=str(trace), trace_sha256=sha(trace),
                      final_field_hashes={name: tensor_hash(getattr(field, name)) for name in h.FIELD_KEYS},
                      final_head_hashes={name: tensor_hash(v) for name, v in raw_net.state_dict().items()},
                      rgb_decodes=scene.original_image_list.decode_count, valid_decodes=1,
                      rgb_names=sorted(scene.original_image_list.decoded_names),
                      actual_image_decodes=len(decoded), VAL_reads=0, semantic_label_decodes=0)
    except BaseException as error:
        report.update(error=repr(error), steps=completed)
        # Failure is preserved as a distinct endpoint, never renamed to last.
        if all(x is not None for x in (field, net, field_opt, head_opt, background)):
            try:
                path = directory/'failure.pt'
                save_endpoint(path, field=field, net=net, field_opt=field_opt, head_opt=head_opt,
                              background=background, arm=arm, step=completed, plan=plan, plan_sha=plan_sha,
                              neighbors=neighbors, counts=counts, status='failed')
                report['failure_checkpoint'] = {'path': str(path), 'sha256': sha(path)}
            except BaseException as save_error:  # noqa: BLE001 - preserve the primary failure
                report['failure_checkpoint_error'] = repr(save_error)
        raise
    finally:
        signal.alarm(0)
        cv2.imread = original_imread; _C.rasterize_gaussians = original_raster
        report['elapsed_seconds'] = time.monotonic()-start
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        write(directory/'training_receipt.json', report)


def execute(plan_path):
    plan = json.loads(plan_path.read_text()); snapshot = Path(plan['source_snapshot'])
    h = helper_import(snapshot)
    require(plan['specification'] == h.SPEC and plan['arms'] == list(h.ARMS), 'Wrong fixed recipe')
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Execute frozen worker only')
    output = Path(plan['output'])
    require(not any((output/n).exists() for n in ('execution_started.json', 'execution_receipt.json', *h.ARMS)),
            'Single attempt only; previous outputs must remain intact')
    require(Path(sys.prefix) == Path(plan['interpreter']).parent.parent, 'Use the isolated IBGS interpreter')
    verify(plan)
    write(output/'execution_started.json', {'plan_sha256': sha(plan_path), 'pid': os.getpid()})
    import torch
    require(torch.__version__ == plan['environment']['torch'], 'Wrong torch runtime')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.set_float32_matmul_precision('highest')
    def timed_out(*_):
        raise TimeoutError('Fixed 1800-second arm budget exhausted')
    prior_handler = signal.signal(signal.SIGALRM, timed_out)
    start = time.monotonic()
    report = {'status': 'failed', 'plan_sha256': sha(plan_path), 'arms': []}
    try:
        for arm in h.ARMS:
            row = {}
            train_arm(plan, sha(plan_path), arm, row)
            row['receipt_path'] = str(output/arm/'training_receipt.json')
            row['receipt_sha256'] = sha(row['receipt_path'])
            report['arms'].append(row)
            gc.collect(); torch.cuda.empty_cache()
        first, second = report['arms']
        require(first['initial_field_hashes'] == second['initial_field_hashes']
                and first['initial_head_hashes'] == second['initial_head_hashes']
                and first['background_sha256'] == second['background_sha256'], 'Initialization mismatch')
        verify(plan)
        actual = {}
        for name, module in sys.modules.copy().items():
            path = getattr(module, '__file__', None)
            if path and name.split('.')[0] in ('bridge_rgs', 'scene', 'utils', 'gaussian_renderer',
                                               'arguments', 'color_aggregation_network'):
                p = Path(path).resolve()
                require(p.is_relative_to(snapshot) and sha(p) == plan['source_hashes'][str(p.relative_to(snapshot))],
                        'Non-frozen runtime source: '+str(p))
                actual[name] = {'path': str(p), 'sha256': sha(p)}
        report.update(status='completed', actual_imports=actual, inputs_and_sources_unchanged=True,
                      optimizer_steps=12000, VAL_reads=0, semantic_label_decodes=0)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        signal.signal(signal.SIGALRM, prior_handler)
        report['elapsed_seconds'] = time.monotonic()-start
        write(output/'execution_receipt.json', report)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path, metavar='NEW_DATA_DIRECTORY')
    mode.add_argument('--run', type=Path, metavar='FROZEN_PLAN')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    else:
        execute(args.run)


if __name__ == '__main__':
    main()
