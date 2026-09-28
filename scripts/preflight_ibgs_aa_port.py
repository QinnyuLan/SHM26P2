"""Fixed two-TRAIN IBGS AA forward/backward wiring check; no optimizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/home/sky/workspace/SHM2026')
PARENT = Path('/mnt/data/SHM2026/runs/ibgs_aa_compatibility_v1')
GRADIENT = Path('/mnt/data/SHM2026/runs/ibgs_aa_gradient_v1')
CONTRACT = Path('/mnt/data/SHM2026/runs/ibgs_port_preflight_v1/data_contract.json')
CONTRACT_SHA = '599f6cbf09a16f56bef6f37d0e0f8ce0a3f24449376213963ee033fc2ea9d84b'
CHECKPOINT_SHA = '35b489fe45ad34ad279018609b6d5eaf6493e4c0d82f9c86ec16925268078092'
TARGETS = ['002.png', '041.png']
SPEC = {
    'protocol': 'ibgs_aa_port_preflight_v1', 'targets': TARGETS,
    'depth_renders': 8, 'target_renders': 2, 'backward': 2,
    'antialias_calls': 10, 'optimizer_steps': 0, 'seed': 20260927,
    'gaussians': 996009, 'sh_degree': 3, 'near': .01, 'eps2d': .3,
    'internal_seconds': 120, 'external_seconds': 180,
    'VAL_reads': 0, 'semantic_label_decodes': 0,
    'objective': '.5*(raw+fused) + .03*normal + .3*photo; warm training_losses(step=6000)',
    'photo_support': 'source_features.abs().sum(0)>0; inherited completed warm-training fix',
    'fusion': 'fresh original ColorFusionResidualNet, no zero-final initialization or optimization',
    'scope': 'Finite forward/backward wiring only; neither finite-difference certification nor trained quality',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x') as stream:
        stream.write(payload)


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and p.suffix != '.pyc' and '__pycache__' not in p.parts}


def validate_compatibility(execution, launch, review, plan_sha, execution_sha):
    require(execution['status'] == launch['status'] == 'completed'
            and launch['natural_completion'] and launch['exit_code'] == 0,
            'Completed natural compatibility check required')
    require(execution['plan_sha256'] == launch['plan_sha256'] == review['plan_sha256'] == plan_sha
            and launch['execution_receipt_sha256'] == review['execution_receipt_sha256'] == execution_sha,
            'Compatibility lineage mismatch')
    require(review['status'] == 'passed' and execution['field_unchanged'] and execution['hook_restored']
            and execution['prediction_barrier_complete'], 'Compatibility integrity required')
    require(execution['render_calls'] == execution['antialias_calls'] == 16
            and execution['backward'] == execution['optimizer_steps'] == 0,
            'Wrong compatibility scope')


def selected_sources(contract):
    rows = {v['name']: v for v in contract['train_rows']}
    require(len(rows) == 350 and all(v['split'] == 'train' for v in rows.values()), '350 TRAIN-only bank required')
    neighbors = {v['name']: [n['name'] for n in v['neighbors_4']] for v in contract['neighbors']['views']}
    sources = set()
    for name in TARGETS:
        require(name in rows and len(neighbors[name]) == 4, 'Fixed target/four-source coverage changed')
        require(len(set(neighbors[name])) == 4 and name not in neighbors[name]
                and all(n in rows for n in neighbors[name]), 'Invalid source bank selection')
        sources.update(neighbors[name])
    require(len(sources) == 8, 'Eight distinct fixed source cameras required')
    return neighbors, sorted(sources)


def prepare(output):
    output = output.resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    parent = read(PARENT/'plan.json')
    validate_compatibility(read(PARENT/'execution_receipt.json'), read(PARENT/'launch_receipt.json'),
                           read(PARENT/'independent_cpu_review.json'), sha(PARENT/'plan.json'),
                           sha(PARENT/'execution_receipt.json'))
    require(tree(parent['source_snapshot']) == parent['source_hashes'], 'Parent source changed')
    require(all(sha(p) == h for p, h in parent['backend_hashes'].items()), 'Isolated backend changed')
    require(sha(CONTRACT) == CONTRACT_SHA and sha(parent['checkpoint']) == CHECKPOINT_SHA, 'Fixed inputs changed')
    gradient_plan, gradient, gradient_launch, gradient_review = [read(GRADIENT/name) for name in
        ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json')]
    require(gradient['status'] == gradient_launch['status'] == 'completed'
            and gradient['numerical_status'] == gradient_review['status'] == 'passed'
            and gradient['primary_unsaturated_passed'] and gradient['hook_restored'] and gradient['numeric_flags_restored']
            and gradient_launch['natural_completion'] and gradient_launch['exit_code'] == 0,
            'Passed local synthetic gradient preflight required')
    require(gradient['plan_sha256'] == gradient_launch['plan_sha256'] == gradient_review['plan_sha256'] == sha(GRADIENT/'plan.json')
            and gradient_launch['execution_receipt_sha256'] == gradient_review['execution_receipt_sha256'] == sha(GRADIENT/'execution_receipt.json'),
            'Gradient check lineage mismatch')
    require(gradient_plan['binary'] == parent['binary']
            and gradient_plan['source_hashes']['bridge_rgs/ibgs_antialias.py'] == parent['source_hashes']['bridge_rgs/ibgs_antialias.py'],
            'Gradient check must use the same AA module and backend')
    contract = read(CONTRACT)
    neighbors, sources = selected_sources(contract)
    inputs = {str(CONTRACT): CONTRACT_SHA, parent['checkpoint']: CHECKPOINT_SHA,
              contract['manifest']: contract['manifest_sha256'], **contract['pixel_hashes']}
    for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json'):
        inputs[str(PARENT/name)] = sha(PARENT/name)
        inputs[str(GRADIENT/name)] = sha(GRADIENT/name)
    inputs[str(ROOT/'scripts/preflight_ibgs_port.py')] = sha(ROOT/'scripts/preflight_ibgs_port.py')
    require(all(sha(p) == h for p, h in inputs.items()), 'Bound inputs changed')
    snapshot = output/'source_snapshot'
    shutil.copytree(parent['source_snapshot'], snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in (Path(__file__), ROOT/'tests/test_ibgs_aa_port_preflight.py', ROOT/'docs/ibgs_aa_port_preflight_protocol.md'):
        shutil.copy2(path, snapshot/path.name)
    plan = {'specification': SPEC, 'output': str(output), 'source_snapshot': str(snapshot),
            'source_hashes': tree(snapshot), 'input_hashes': inputs,
            'backend_hashes': parent['backend_hashes'], 'binary': parent['binary'],
            'backend_python': parent['backend_python'], 'repository': str(snapshot/'ibgs'),
            'checkpoint': parent['checkpoint'], 'checkpoint_sha256': CHECKPOINT_SHA,
            'contract': str(CONTRACT), 'contract_sha256': CONTRACT_SHA,
            'source_names': sources, 'target_neighbors': {n: neighbors[n] for n in TARGETS},
            'aa_module_sha256': sha(snapshot/'bridge_rgs/ibgs_antialias.py'),
            'prepare_checkpoint_loads': 0, 'prepare_image_decodes': 0}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC and Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name,
            'Frozen entry/specification required')
    require(tree(plan['source_snapshot']) == plan['source_hashes'], 'Frozen source changed')
    require(all(sha(p) == h for p, h in {**plan['input_hashes'], **plan['backend_hashes']}.items()), 'Bound input/backend changed')


def validate_result(report):
    require(report['counts'] == {'depth_renders': 8, 'target_renders': 2, 'backward': 2, 'optimizer_steps': 0}
            and report['antialias_calls'] == 10, 'Fixed render/backward/AA counts changed')
    require(report['field_parameters_unchanged'] and report['network_parameters_unchanged']
            and report['hook_restored'] and report['numerics_restored'], 'State restoration failed')
    require([v['name'] for v in report['target_rows']] == TARGETS, 'Fixed target order changed')
    for row in report['target_rows']:
        require(row['fusion_gradients_finite'] and all(v['finite'] for v in row['gradient_summary'].values()),
                'Nonfinite/missing gradient')
    require(report['rgb_decodes'] == 10 and report['source_depth_updates'] == 8 and report['valid_decodes'] == 1,
            'Unexpected decode/depth-update scope')


def run(plan, report):
    sys.path[:0] = [plan['source_snapshot'], plan['backend_python'], plan['repository']]
    import torch
    from color_aggregation_network import ColorFusionResidualNet, fuse_color
    from diff_plane_rasterization import _C, GaussianRasterizer
    from gaussian_renderer import render, render_depth

    from bridge_rgs.ibgs_adapter import TrainScene
    from bridge_rgs.ibgs_antialias import aa_opacity_rasterizer
    from bridge_rgs.ibgs_warm_training import FIELD_KEYS, load_warm_field, training_losses

    require(Path(_C.__file__).resolve() == Path(plan['binary']['path']).resolve()
            and sha(_C.__file__) == plan['binary']['sha256'], 'Wrong isolated backend')
    report['actual_binary'] = {'path': str(Path(_C.__file__).resolve()), 'sha256': sha(_C.__file__)}
    torch.set_num_threads(4)
    torch.manual_seed(SPEC['seed']); torch.cuda.manual_seed_all(SPEC['seed'])
    original_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    original_forward = GaussianRasterizer.forward
    field = net = scene = None
    original = net_original = None
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.benchmark = False
        report['actual_numerics'] = {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                                    'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                                    'cudnn_benchmark': torch.backends.cudnn.benchmark}
        field, background = load_warm_field(plan['checkpoint'])
        original = {k: getattr(field, k).detach().clone() for k in FIELD_KEYS}
        contract = read(plan['contract'])
        neighbors, source_names = selected_sources(contract)
        require(source_names == plan['source_names'], 'Sources changed')
        scene = TrainScene(contract['train_rows'], contract['pixel_hashes'], neighbors, field, contract['manifest_scene_radius'])
        cameras = {c.image_name: c for c in scene.getTrainCameras()}
        pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
        options = SimpleNamespace(shuffle_source_frame=False, enable_exposure_correction=False,
                                  nb_visible_src_frames=3, residual_resolution_scale=1.)
        net = ColorFusionResidualNet(height=cameras[TARGETS[0]].image_height, width=cameras[TARGETS[0]].image_width).cuda()
        net_original = [p.detach().clone() for p in net.parameters()]
        report.update(gaussian_count=len(field._xyz), sh_degree=field.active_sh_degree,
                      checkpoint_loads=1, raw_rgb_stage='AA-compatible warm import; finite wiring, no training')
        require(len(field._xyz) == SPEC['gaussians'] and field.active_sh_degree == 3, 'Field changed')
        with aa_opacity_rasterizer(GaussianRasterizer) as aa_state:
            try:
                source_ids = sorted({i for n in TARGETS for i in cameras[n].nearest_id})
                require(sorted(scene.cameras[i].image_name for i in source_ids) == source_names, 'Actual source cameras changed')
                with torch.no_grad():
                    for index in source_ids:
                        depth = render_depth(scene.cameras[index], field, scene, pipe, options, background, True, 4, 4, .01)
                        scene.rendered_depth_list[index] = depth
                        report['counts']['depth_renders'] += 1
                for name in TARGETS:
                    camera = cameras[name]
                    require(camera.nearest_names == plan['target_neighbors'][name], 'Source order changed')
                    started = time.monotonic()
                    package = render(camera, field, scene, pipe, options, background,
                                     learnt_normal=True, nb_src_frames=4, buffer_length=4,
                                     depth_error_threshold=.01, do_find_closest_frame=False,
                                     do_render_src_depth=False, render_geo=True, return_depth_normal=True)
                    report['counts']['target_renders'] += 1
                    fusion = fuse_color(package, net, None, None, None, 0, options)
                    require(fusion is not None, 'No actual fusion support: '+name)
                    # TRAIN target supervision is intentional, after this forward.
                    target = scene.original_image_list[camera.uid].cuda()
                    loss, components = training_losses(package, fusion, target, scene.valid, 6000)
                    require(bool(torch.isfinite(loss)), 'Nonfinite real-scene loss')
                    loss.backward(); report['counts']['backward'] += 1
                    gradients = {}
                    for key in FIELD_KEYS:
                        grad = getattr(field, key).grad
                        finite = grad is not None and bool(torch.isfinite(grad).all())
                        gradients[key] = {'finite': finite, 'l2': float(grad.norm()) if finite else None}
                    net_finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in net.parameters())
                    torch.cuda.synchronize()
                    row = {'name': name, 'source_names': camera.nearest_names, 'seconds': time.monotonic()-started,
                           'loss': float(loss), 'raw_rgb_loss': float(components['base_rgb']),
                           'fused_loss': float(components['fused_rgb']), 'normal_loss': float(components['normal']),
                           'photometric_loss': float(components['photo']), 'active_photo_sources': components['active_photo_sources'],
                           'fusion_support_fraction': float(fusion['valid_warp_mask'].mean()),
                           'gradient_summary': gradients, 'fusion_gradients_finite': bool(net_finite)}
                    report['target_rows'].append(row); print(json.dumps(row, allow_nan=False), flush=True)
                    require(all(v['finite'] for v in gradients.values()) and net_finite, 'Nonfinite real-scene gradient: '+name)
                    for key in FIELD_KEYS:
                        getattr(field, key).grad = None
                    net.zero_grad(set_to_none=True)
                    del package, fusion, target, loss, components
            finally:
                report['antialias_calls'] = aa_state.calls
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = original_flags
        report['numerics_restored'] = original_flags == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
        report['hook_restored'] = GaussianRasterizer.forward is original_forward
        report['field_parameters_unchanged'] = original is not None and all(torch.equal(getattr(field, k).detach(), v) for k, v in original.items())
        report['network_parameters_unchanged'] = net_original is not None and all(torch.equal(p.detach(), v) for p, v in zip(net.parameters(), net_original, strict=True))
        if scene is not None:
            report.update(rgb_decodes=scene.original_image_list.decode_count, rgb_names=sorted(scene.original_image_list.decoded_names),
                          valid_decodes=1, source_depth_updates=scene.rendered_depth_list.updates)
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['actual_imports'] = {}
        for name, module in list(sys.modules.items()):
            if name.split('.')[0] in ('bridge_rgs', 'scene', 'utils', 'gaussian_renderer', 'color_aggregation_network') and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve(); relative = str(path.relative_to(plan['source_snapshot']))
                require(sha(path) == plan['source_hashes'][relative], 'Unbound imported source: '+name)
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}
    validate_result(report)


def execute(path, expected):
    require(sha(path) == expected, 'Wrong plan SHA')
    plan = read(path); output = Path(plan['output'])
    write(output/'execution_started.json', {'plan_sha256': expected})
    report = {'status': 'running', 'numerical_status': 'not_completed', 'plan_sha256': expected,
              'counts': {'depth_renders': 0, 'target_renders': 0, 'backward': 0, 'optimizer_steps': 0},
              'target_rows': [], 'VAL_reads': 0, 'semantic_label_decodes': 0,
              'aa_module_sha256': plan['aa_module_sha256']}
    started = time.monotonic()
    def expired(*_):
        raise TimeoutError('120-second AA preflight budget')
    old = signal.signal(signal.SIGALRM, expired); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan); run(plan, report); verify(plan)
        report.update(status='completed', numerical_status='passed', bound_inputs_and_sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed', numerical_status='not_passed', error=repr(error))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)
        report['elapsed_seconds'] = time.monotonic()-started
        write(output/'execution_receipt.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path); action.add_argument('--execute', type=Path)
    parser.add_argument('--expected-plan-sha256'); args = parser.parse_args()
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode')
    prepare(args.prepare) if args.prepare else execute(args.execute, args.expected_plan_sha256)
