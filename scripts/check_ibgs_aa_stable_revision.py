"""Thin, separately recorded verification modes for the factored-FP64 AA revision."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
BASE = RUNS/'ibgs_aa_port_preflight_v1'
PARENTS = {'compatibility': RUNS/'ibgs_aa_compatibility_v1',
           'gradient': RUNS/'ibgs_aa_gradient_v1', 'preflight': BASE}
ENTRIES = {'compatibility': 'check_ibgs_aa_compatibility.py',
           'gradient': 'check_ibgs_aa_gradient.py', 'preflight': 'preflight_ibgs_aa_port.py'}
FAILED = RUNS/'ibgs_aa_warm_matched_v1'
DIAGNOSTIC = RUNS/'ibgs_aa_determinant_diagnostic_v1'
STABLE_SHA = 'ea151cc69520261d50337101b219e9561a15b4f94c1cce9197a9415b096bdcd2'
FAILURE_SHA = 'abbe8973ebc7649e99b162f9d878129232fe5475ffe8672f2e4fc9b48c5b5d8f'
START_SHA = '35b489fe45ad34ad279018609b6d5eaf6493e4c0d82f9c86ec16925268078092'
PROVIDER = 'bridge_rgs.ibgs_antialias_stable'
PROFILE = {'id': 'ibgs_centered_corner_v2_aa_factored64_near001_v2',
           'near_plane': .01, 'eps2d': .3, 'capture_last': False,
           'scope': 'Numerical AA adapter revision; backend covariance and approximate VJPs unchanged'}
BUDGETS = {'compatibility': (100, 120), 'gradient': (90, 120),
           'preflight': (120, 180), 'failure': (90, 120)}
RAW_KEYS = ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(payload)


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and p.suffix != '.pyc' and '__pycache__' not in p.parts}


def load_recipe(path, mode):
    name = 'inherited_aa_revision_'+mode
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def precision_policy(path):
    assignments = [node.value for node in ast.parse(Path(path).read_text()).body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PRECISION_POLICY' for t in node.targets)]
    require(len(assignments) == 1, 'One literal precision policy required')
    return ast.literal_eval(assignments[0])


def completed(directory):
    p, e, l = [read(directory/name) for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json')]
    require(e['status'] == l['status'] == 'completed' and l['natural_completion'] and l['exit_code'] == 0
            and e['plan_sha256'] == l['plan_sha256'] == sha(directory/'plan.json')
            and l['execution_receipt_sha256'] == sha(directory/'execution_receipt.json'), 'Completed predecessor required')
    return p


def prepare(output, mode):
    require(mode in BUDGETS, 'Choose a fixed validation mode')
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk directory required')
    parents = {key: completed(directory) for key, directory in PARENTS.items()}
    parent = parents['preflight']
    require(tree(parent['source_snapshot']) == parent['source_hashes'], 'Inherited source changed')
    stable_path = ROOT/'src/bridge_rgs/ibgs_antialias_stable.py'
    require(sha(stable_path) == STABLE_SHA, 'Fixed stable module required')
    policy = precision_policy(stable_path)
    inputs = dict(parent['input_hashes'])
    # The compatibility cache is needed only by its unchanged scoring function.
    if mode == 'compatibility':
        inputs.update(parents[mode]['input_hashes'])
    for directory in PARENTS.values():
        for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json'):
            inputs[str(directory/name)] = sha(directory/name)
    failed, failed_launch, arm = [read(FAILED/name) for name in ('execution_receipt.json', 'launch_receipt.json', 'full/training_receipt.json')]
    require(failed['status'] == failed_launch['status'] == 'failed' and failed_launch['natural_completion']
            and failed_launch['exit_code'] != 0 and failed_launch['execution_receipt_sha256'] == sha(FAILED/'execution_receipt.json')
            and arm['status'] == 'failed' and arm['steps'] == 61, 'Preserved natural failed training required')
    failure = arm['failure_checkpoint']
    require(failure['sha256'] == FAILURE_SHA and sha(failure['path']) == FAILURE_SHA, 'Exact 61-step failure state required')
    diagnostic, diag_launch = read(DIAGNOSTIC/'execution_receipt.json'), read(DIAGNOSTIC/'launch_receipt.json')
    require(diagnostic['status'] == diag_launch['status'] == 'completed' and diag_launch['natural_completion']
            and diag_launch['exit_code'] == 0 and diag_launch['execution_receipt_sha256'] == sha(DIAGNOSTIC/'execution_receipt.json')
            and diagnostic['original_failure_reproduced'] and diagnostic['bad_count'] == 1
            and diagnostic['camera'] == '004.png' and diagnostic['failure_checkpoint_sha256'] == FAILURE_SHA
            and diagnostic['arrays_sha256'] == sha(DIAGNOSTIC/'bad_slots.npz'), 'Exact determinant diagnostic required')
    for directory, names in ((FAILED, ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'full/training_receipt.json')),
                             (DIAGNOSTIC, ('execution_receipt.json', 'launch_receipt.json', 'bad_slots.npz'))):
        for name in names:
            inputs[str(directory/name)] = sha(directory/name)
    inputs[failure['path']] = FAILURE_SHA
    require(sha(parent['checkpoint']) == START_SHA, 'Original starting checkpoint changed')
    require(all(sha(p) == h for p, h in {**inputs, **parent['backend_hashes']}.items()), 'Bound bytes changed')
    snapshot = output/'source_snapshot'
    shutil.copytree(parent['source_snapshot'], snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    gradient_path = Path(parents['gradient']['source_snapshot'])/ENTRIES['gradient']
    require(sha(gradient_path) == parents['gradient']['source_hashes'][ENTRIES['gradient']], 'Frozen synthetic recipe changed')
    shutil.copy2(gradient_path, snapshot/gradient_path.name)
    shutil.copy2(stable_path, snapshot/'bridge_rgs/ibgs_antialias_stable.py')
    for path in (Path(__file__), ROOT/'tests/test_ibgs_aa_stable_revision.py', ROOT/'docs/ibgs_aa_stable_revision_protocol.md'):
        shutil.copy2(path, snapshot/path.name)
    contract = read(parent['contract'])
    failed_plan = read(FAILED/'plan.json')
    order_path = Path(failed_plan['camera_order'])
    require(read(order_path)['names'][61] == '004.png', 'Expected failed view changed')
    inputs[str(order_path)] = sha(order_path)
    plan = dict(parent)
    plan.update(protocol='ibgs_aa_stable_revision_v2', mode=mode, output=str(output),
                source_snapshot=str(snapshot), source_hashes=tree(snapshot), input_hashes=inputs,
                repository=str(snapshot/'ibgs'), aa_provider=PROVIDER, aa_module_sha256=STABLE_SHA,
                precision_policy=policy, renderer_profile=PROFILE,
                internal_seconds=BUDGETS[mode][0], external_seconds=BUDGETS[mode][1],
                diagnostic_failure_checkpoint=dict(failure),
                failure_camera=next(row for row in contract['train_rows'] if row['name'] == '004.png'),
                original_recipe_sha256={key: sha(snapshot/name) for key, name in ENTRIES.items()},
                compatibility_views=parents['compatibility']['views'],
                failure_objective='mean(raw_RGB * [1,.7,-.2]); FP64 scalar on all native pixels',
                prepare_checkpoint_loads=0, prepare_image_decodes=0)
    # Parent specification remains the immutable inherited two-view recipe;
    # the new protocol/mode/profile/precision fields identify this execution.
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json'), 'mode': mode}))


@contextmanager
def inject_provider(legacy, stable, report):
    """Explicit runtime dependency injection; no old source or module identity is replaced."""
    original = legacy.aa_opacity_rasterizer
    require(original is not stable.aa_opacity_rasterizer, 'Provider already replaced')
    legacy.aa_opacity_rasterizer = stable.aa_opacity_rasterizer
    report['provider_restored'] = False
    try:
        yield
    finally:
        legacy.aa_opacity_rasterizer = original
        report['provider_restored'] = legacy.aa_opacity_rasterizer is original


def verify(plan):
    require(plan['protocol'] == 'ibgs_aa_stable_revision_v2' and plan['mode'] in BUDGETS
            and plan['renderer_profile'] == PROFILE and plan['aa_provider'] == PROVIDER
            and plan['aa_module_sha256'] == STABLE_SHA, 'Wrong stable revision identity')
    snapshot = Path(plan['source_snapshot'])
    require(Path(__file__).resolve() == snapshot/Path(__file__).name and tree(snapshot) == plan['source_hashes'], 'Frozen source required')
    require(precision_policy(snapshot/'bridge_rgs/ibgs_antialias_stable.py') == plan['precision_policy'], 'Changed precision policy')
    require((plan['internal_seconds'], plan['external_seconds']) == BUDGETS[plan['mode']], 'Changed mode budget')
    require(all(sha(p) == h for p, h in {**plan['input_hashes'], **plan['backend_hashes']}.items()), 'Bound input/backend changed')


def validate_failure_state(saved):
    require(saved['status'] == 'failed' and saved['step'] == 61 and saved['arm'] == 'full'
            and saved['protocol'] == 'ibgs_aa_warm_matched_v1' and saved['sh_degree'] == 3,
            'Only exact failure state is allowed for the regression, never for training')


def failure_run(plan, report, stable):
    import torch
    from diff_plane_rasterization import GaussianRasterizer
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel

    from bridge_rgs.ibgs_adapter import BridgeCamera
    from bridge_rgs.ibgs_warm_training import FIELD_KEYS

    saved = torch.load(plan['diagnostic_failure_checkpoint']['path'], map_location='cpu', weights_only=False)
    validate_failure_state(saved)
    field = GaussianModel(3)
    for key in FIELD_KEYS:
        setattr(field, key, torch.nn.Parameter(saved['field'][key].cuda().clone()))
    field.active_sh_degree = 3; field.spatial_lr_scale = float(saved['scene_scale'])
    background = saved['background'].cuda().clone()
    original = {key: getattr(field, key).detach().clone() for key in FIELD_KEYS}
    camera = BridgeCamera(plan['failure_camera'], 0)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    options = SimpleNamespace(shuffle_source_frame=False, enable_exposure_correction=False)
    before = GaussianRasterizer.forward
    try:
        with stable.aa_opacity_rasterizer(GaussianRasterizer, capture_last=False) as hook:
            try:
                package = render(camera, field, None, pipe, options, background, learnt_normal=True,
                    nb_src_frames=1, buffer_length=4, depth_error_threshold=.01,
                    render_geo=False, return_depth_normal=False, do_find_closest_frame=False)
                report['raster_calls'] += 1
                rgb = package['render']; require(bool(torch.isfinite(rgb).all()), 'Nonfinite failed-view raw RGB')
                loss = (rgb.double()*rgb.new_tensor([1., .7, -.2]).double()[:, None, None]).mean()
                loss.backward(); report['backward'] += 1
                gradients = {}
                for key in RAW_KEYS:
                    grad = getattr(field, key).grad
                    finite = grad is not None and bool(torch.isfinite(grad).all())
                    gradients[key] = {'finite': finite, 'l2': float(grad.norm()) if finite else None}
                report.update(camera='004.png', checkpoint_step=61, loss=float(loss), gradient_summary=gradients,
                              parameter_scope='six raw RGB parameters; normal/offset are not required by this path')
                require(all(v['finite'] for v in gradients.values()), 'Missing/nonfinite raw parameter gradient')
                import numpy as np
                path = Path(plan['output'])/'004.raw.npy'
                with path.open('xb') as stream:
                    np.save(stream, rgb.detach().permute(1, 2, 0).cpu().numpy(), allow_pickle=False)
                report['prediction'] = {'path': str(path), 'sha256': sha(path)}
            finally:
                report['antialias_calls'] = hook.calls
    finally:
        report['field_parameters_unchanged'] = all(torch.equal(getattr(field, k).detach(), v) for k, v in original.items())
        report['hook_restored'] = GaussianRasterizer.forward is before


def validate_mode(mode, report):
    if mode == 'compatibility':
        require(report['render_calls'] == report['cached_target_loads'] == report['antialias_calls'] == 16
                and report['prediction_barrier_complete'] and report['field_unchanged'] and report['hook_restored'], 'Incomplete compatibility check')
    elif mode == 'gradient':
        require(report['forward_calls'] == report['hook_calls'] == 33 and report['backward_calls'] == 3
                and report['primary_unsaturated_passed'] and report['numerical_status'] == 'passed'
                and all(report['zero_control'].values()) and report['hook_restored'] and report['numeric_flags_restored'], 'Original synthetic primary gate failed')
    elif mode == 'preflight':
        require(report['counts'] == {'depth_renders': 8, 'target_renders': 2, 'backward': 2, 'optimizer_steps': 0}
                and report['antialias_calls'] == 10 and report['field_parameters_unchanged']
                and report['network_parameters_unchanged'] and report['hook_restored'] and report['numerics_restored'], 'Incomplete two-TRAIN preflight')
    else:
        require(report['raster_calls'] == report['backward'] == report['antialias_calls'] == 1
                and report['field_parameters_unchanged'] and report['hook_restored']
                and set(report['gradient_summary']) == set(RAW_KEYS)
                and all(v['finite'] for v in report['gradient_summary'].values()), 'Failed 004 regression')


def execute(path, expected):
    require(sha(path) == expected, 'Wrong plan SHA')
    plan = read(path); output = Path(plan['output']); mode = plan['mode']
    write(output/'execution_started.json', {'plan_sha256': expected})
    report = {'status': 'running', 'numerical_status': 'not_completed', 'plan_sha256': expected,
              'mode': mode, 'renderer_profile': plan['renderer_profile'], 'precision_policy': plan['precision_policy'],
              'aa_module_sha256': STABLE_SHA, 'optimizer_steps': 0,
              'VAL_reads': 0, 'semantic_label_decodes': 0}
    if mode == 'compatibility':
        report.update(predictions=[], render_calls=0, cached_target_loads=0, backward=0,
                      original_image_decodes=0, VAL_decodes=0)
    elif mode == 'gradient':
        report.update(forward_calls=0, backward_calls=0, data_reads=0)
    elif mode == 'preflight':
        report.update(target_rows=[], counts={'depth_renders': 0, 'target_renders': 0, 'backward': 0, 'optimizer_steps': 0})
    else:
        report.update(raster_calls=0, backward=0, data_reads=0, original_image_decodes=0)
    started = time.monotonic(); previous = None
    def expired(*_):
        raise TimeoutError('Fixed stable AA validation budget exhausted')
    old_signal = signal.signal(signal.SIGALRM, expired); signal.alarm(plan['internal_seconds'])
    try:
        verify(plan)
        require(not any(k.startswith(('bridge_rgs', 'diff_plane_rasterization')) for k in sys.modules), 'Fresh process without preloaded render modules required')
        sys.path[:0] = [plan['source_snapshot'], plan['backend_python'], plan['repository']]
        import torch
        from diff_plane_rasterization import _C
        legacy = importlib.import_module('bridge_rgs.ibgs_antialias')
        stable = importlib.import_module(PROVIDER)
        require(Path(_C.__file__).resolve() == Path(plan['binary']['path']).resolve()
                and sha(_C.__file__) == plan['binary']['sha256'], 'Wrong actual binary')
        require(sha(stable.__file__) == STABLE_SHA and stable.PRECISION_POLICY == plan['precision_policy']
                and stable.aa_opacity_rasterizer.__module__ == PROVIDER, 'Wrong actual stable callable')
        report['actual_binary'] = {'path': str(Path(_C.__file__).resolve()), 'sha256': sha(_C.__file__)}
        report['provider_binding'] = {'module': PROVIDER, 'path': str(Path(stable.__file__).resolve()),
            'sha256': sha(stable.__file__), 'callable_module': stable.aa_opacity_rasterizer.__module__,
            'precision_policy': stable.PRECISION_POLICY,
            'injection': 'Only legacy module aa_opacity_rasterizer attribute temporarily points to this callable; both source files retained'}
        previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                    torch.backends.cudnn.benchmark, torch.get_float32_matmul_precision(), torch.get_num_threads())
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.benchmark = False
        with inject_provider(legacy, stable, report):
            if mode == 'failure':
                failure_run(plan, report, stable)
            else:
                recipe = load_recipe(Path(plan['source_snapshot'])/ENTRIES[mode], mode)
                call_plan = dict(plan)
                if mode == 'compatibility':
                    call_plan['views'] = plan['compatibility_views']
                recipe.run(call_plan, report)
            validate_mode(mode, report)
        require(report['provider_restored'], 'AA provider restoration failed')
        report['actual_imports'] = {}
        for name, module in list(sys.modules.items()):
            if (name.split('.')[0] in ('bridge_rgs', 'scene', 'utils', 'gaussian_renderer', 'color_aggregation_network')
                    or name.startswith('inherited_aa_revision_')) and getattr(module, '__file__', None):
                source = Path(module.__file__).resolve(); rel = str(source.relative_to(plan['source_snapshot']))
                require(sha(source) == plan['source_hashes'][rel], 'Unbound actual import')
                report['actual_imports'][name] = {'path': str(source), 'sha256': sha(source)}
        verify(plan)
        report.update(status='completed', numerical_status='passed', bound_inputs_and_sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed', numerical_status='not_passed', error=repr(error))
        raise
    finally:
        if previous is not None:
            torch.set_float32_matmul_precision(previous[3]); torch.set_num_threads(previous[4])
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = previous[:3]
            report['outer_numerics_restored'] = previous == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark, torch.get_float32_matmul_precision(), torch.get_num_threads())
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            if not report['outer_numerics_restored']:
                report.update(status='failed', numerical_status='not_passed', restoration_error='Outer numeric flags changed')
        signal.alarm(0); signal.signal(signal.SIGALRM, old_signal)
        report['elapsed_seconds'] = time.monotonic()-started
        write(output/'execution_receipt.json', report)
        if previous is not None:
            require(report['outer_numerics_restored'], 'Outer numerical flags were not restored')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path); action.add_argument('--execute', type=Path)
    parser.add_argument('--mode', choices=list(BUDGETS)); parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode')
    prepare(args.prepare, args.mode) if args.prepare else execute(args.execute, args.expected_plan_sha256)
