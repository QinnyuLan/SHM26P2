"""Thin AA/near revision of the completed, frozen warm-IBGS trainer."""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/home/sky/workspace/SHM2026')
PARENT = Path('/mnt/data/SHM2026/runs/ibgs_warm_matched_v1')
PARENT_PLAN_SHA = 'c1af18ac17a1ce934208fa91f2b987f7a46c9920b7e35eb3ba9c95dafc056691'
COMPATIBILITY = Path('/mnt/data/SHM2026/runs/ibgs_aa_compatibility_v1')
PORT = Path('/mnt/data/SHM2026/third_party/ibgs_aa_port_v1')
PROTOCOL = 'ibgs_aa_warm_matched_v1'
PROFILE = {
    'id': 'ibgs_centered_corner_v2_aa_near001_v1', 'near_plane': .01,
    'eps2d': .3, 'compensation': 'sqrt(max(det(C)/det(C+.3I),0))',
    'opacity': 'activated_opacity * rho; no floor or normalization',
    'derivative': 'Torch positive-ratio derivative; constant zero for nonpositive ratio',
    'capture_last': False, 'expected_calls_per_arm': 6350,
    'scope': 'Existing IBGS backend approximate VJPs retained; not gsplat backward equivalence',
}


def read(path):
    return json.loads(Path(path).read_text())


def load_worker(path):
    spec = importlib.util.spec_from_file_location('inherited_ibgs_warm_trainer', path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def revised_specification(original):
    result = copy.deepcopy(original)
    result.update(protocol=PROTOCOL, renderer_profile=copy.deepcopy(PROFILE))
    return result


def completed_evidence(directory, old, *, numerical=False):
    """Natural completion is mandatory; a completed failed numerical test is not enough."""
    directory = Path(directory)
    plan = read(directory/'plan.json')
    execution = read(directory/'execution_receipt.json')
    launch = read(directory/'launch_receipt.json')
    old.require(execution['status'] == launch['status'] == 'completed'
                and launch['natural_completion'] is True and launch['exit_code'] == 0
                and launch['execution_receipt_sha256'] == old.sha(directory/'execution_receipt.json')
                and launch['plan_sha256'] == execution['plan_sha256'] == old.sha(directory/'plan.json'),
                'Natural completed evidence required: '+str(directory))
    if numerical:
        old.require(execution.get('numerical_status') == 'passed',
                    'Passed numerical checks required: '+str(directory))
    return plan, execution


def validate_fd_receipt(receipt, old):
    old.require(receipt.get('numerical_status') == 'passed'
                and receipt.get('primary_unsaturated_passed') is True
                and len(receipt.get('positive_case', [])) == 4
                and all(row['finite_differences'][0]['passed'] is True for row in receipt['positive_case'])
                and set(receipt.get('zero_control', {})) == {'rho_zero', 'finite_zero_opacity_gradient', 'background_exact'}
                and all(value is True for value in receipt['zero_control'].values())
                and receipt.get('forward_calls') == receipt.get('hook_calls') == 33
                and receipt.get('backward_calls') == 3 and receipt.get('optimizer_steps') == 0
                and receipt.get('data_reads') == 0 and receipt.get('hook_restored') is True
                and receipt.get('numeric_flags_restored') is True,
                'Incomplete or failed primary AA numerical evidence')


def copy_inherited_sources(parent, snapshot, old):
    # Copy exact bound files, including any legitimate .py inside a directory
    # named __pycache__; ignoring that directory would change the old contract.
    for name, digest in parent['source_hashes'].items():
        source = Path(parent['source_snapshot'])/name
        old.require(old.sha(source) == digest, 'Inherited source changed: '+name)
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    old.require(old.tree_hashes(snapshot) == parent['source_hashes'], 'Inherited source package changed')


def validate_backward_receipt(receipt, old):
    old.require(receipt.get('numerical_status') == 'passed'
                and receipt.get('counts') == {'depth_renders': 8, 'target_renders': 2,
                                               'backward': 2, 'optimizer_steps': 0}
                and receipt.get('antialias_calls') == 10
                and all(receipt.get(key) is True for key in ('field_parameters_unchanged',
                        'network_parameters_unchanged', 'hook_restored', 'numerics_restored'))
                and [row['name'] for row in receipt.get('target_rows', [])] == ['002.png', '041.png']
                and all(row['fusion_gradients_finite'] is True
                        and len(row['gradient_summary']) == 8
                        and all(item['finite'] is True for item in row['gradient_summary'].values())
                        for row in receipt['target_rows'])
                and receipt.get('rgb_decodes') == 10 and receipt.get('valid_decodes') == 1
                and receipt.get('source_depth_updates') == 8,
                'Incomplete or failed real TRAIN backward evidence')


def validate_aa_binding(evidence_plan, module_sha, binary_sha, old):
    sources = evidence_plan.get('source_hashes', {})
    inputs = evidence_plan.get('input_hashes', {})
    matches = [value for name, value in {**sources, **inputs}.items()
               if Path(name).name == 'ibgs_antialias.py']
    old.require(matches and all(value == module_sha for value in matches), 'AA evidence source differs')
    old.require(evidence_plan.get('binary', {}).get('sha256') == binary_sha,
                'AA evidence backend differs')


def prepare(output, fd_run, backward_run):
    old = load_worker(PARENT/'source_snapshot/train_ibgs_warm_matched.py')
    old.require(old.sha(PARENT/'plan.json') == PARENT_PLAN_SHA, 'Wrong completed parent')
    parent, _ = completed_evidence(PARENT, old)
    old.verify(parent)
    old.require(read(PARENT/'training_contract_review.json')['status'] == 'passed', 'Parent contract review required')
    compatibility, _ = completed_evidence(COMPATIBILITY, old)
    fd_plan, fd_receipt = completed_evidence(fd_run, old, numerical=True)
    validate_fd_receipt(fd_receipt, old)
    backward_plan, backward_receipt = completed_evidence(backward_run, old, numerical=True)
    validate_backward_receipt(backward_receipt, old)
    module_path = ROOT/'src/bridge_rgs/ibgs_antialias.py'
    module_sha = old.sha(module_path)
    build, port = read(PORT/'build_receipt.json'), read(PORT/'preparation.json')
    old.require(build['status'] == 'completed' and build['natural_completion']
                and build['exit_code'] == 0, 'Completed isolated near backend required')
    binary = build['binary']
    old.require(old.sha(binary['path']) == binary['sha256'], 'Isolated backend changed')
    for evidence in (compatibility, fd_plan, backward_plan):
        validate_aa_binding(evidence, module_sha, binary['sha256'], old)
    output = Path(output).resolve()
    old.require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    snapshot = output/'source_snapshot'
    copy_inherited_sources(parent, snapshot, old)
    shutil.copyfile(module_path, snapshot/'bridge_rgs/ibgs_antialias.py')
    for source in (Path(__file__), ROOT/'tests/test_ibgs_aa_warm_training.py',
                   ROOT/'docs/ibgs_aa_warm_matched_protocol.md'):
        shutil.copyfile(source, snapshot/source.name)
    shutil.copyfile(parent['camera_order'], output/'camera_order.json')
    inputs = dict(parent['input_hashes'])
    provenance = {}
    for key, directory, files in (
        ('parent', PARENT, ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'training_contract_review.json')),
        ('compatibility', COMPATIBILITY, ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'analysis.json')),
        ('local_fd', Path(fd_run), ('plan.json', 'execution_receipt.json', 'launch_receipt.json')),
        ('real_backward', Path(backward_run), ('plan.json', 'execution_receipt.json', 'launch_receipt.json')),
        ('isolated_port', PORT, ('preparation.json', 'build_receipt.json', 'near_plane.patch')),
    ):
        provenance[key] = {}
        for name in files:
            path = directory/name
            inputs[str(path)] = old.sha(path)
            provenance[key][name] = {'path': str(path), 'sha256': old.sha(path)}
    runtime = dict(parent['runtime_sources'])  # Also certifies original installation remains untouched.
    runtime[binary['path']] = binary['sha256']
    wrapper = PORT/'python/diff_plane_rasterization/__init__.py'
    runtime[str(wrapper)] = old.sha(wrapper)
    for name, digest in port['source_hashes'].items():
        path = PORT/'source'/name
        old.require(old.sha(path) == digest, 'Isolated source changed')
        runtime[str(path)] = digest
    plan = dict(parent)
    plan.update(protocol=PROTOCOL, specification=revised_specification(parent['specification']),
                renderer_profile=PROFILE, output=str(output), source_snapshot=str(snapshot),
                source_hashes=old.tree_hashes(snapshot), input_hashes=inputs, runtime_sources=runtime,
                inherited_sources=parent['source_hashes'], provenance=provenance,
                camera_order=str(output/'camera_order.json'), camera_order_sha256=old.sha(output/'camera_order.json'),
                binary=binary, backend_python=str(PORT/'python'), aa_module_sha256=module_sha,
                preparation_scope='CPU hashes/metadata only; no model/image load; no GPU',
                actual_recipe_changes=['AA effective opacity for every RGB/depth render', 'near .2 to .01'])
    old.require(plan['camera_order_sha256'] == parent['camera_order_sha256'], 'Same 6000-camera order required')
    old.write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'plan_sha256': old.sha(output/'plan.json'),
                      'source_files': len(plan['source_hashes']), 'input_files': len(inputs)}), flush=True)


def renderer_metadata(plan):
    return {'renderer_profile': copy.deepcopy(PROFILE), 'aa_module_sha256': plan['aa_module_sha256'],
            'renderer_binary': plan['binary']}


def save_endpoint(old, h, hook_state, path, *, field, net, field_opt, head_opt, background,
                  arm, step, plan, plan_sha, neighbors, counts, status):
    """Same fixed endpoint payload, with a distinct mandatory renderer identity."""
    import torch
    old.require(not path.exists(), 'Do not overwrite endpoints')
    counts['aa_opacity_calls'] = hook_state.calls
    if status == 'completed':
        old.require(step == 6000 and hook_state.calls == 6350, 'AA did not cover all arm renders')
    value = {'protocol': PROTOCOL, 'status': status, 'arm': arm, 'step': step,
             'field': {name: getattr(field, name).detach().cpu() for name in h.FIELD_KEYS},
             'head': old.cpu_tree(net.inner.state_dict()), 'background': background.detach().cpu(),
             'sh_degree': 3, 'scene_scale': field.spatial_lr_scale,
             'field_optimizer': old.cpu_tree(field_opt.state_dict()), 'head_optimizer': old.cpu_tree(head_opt.state_dict()),
             'specification': plan['specification'], 'source_plan_sha256': plan_sha,
             'sample_order_sha256': plan['camera_order_sha256'],
             'data_contract': {'path': plan['data_contract'], 'sha256': plan['data_contract_sha256']},
             'coordinate_profile': plan['coordinate_profile'], 'neighbors': neighbors, 'counts': counts,
             'rng': {'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all()},
             **renderer_metadata(plan)}
    torch.save(value, path)


def train_arm(old, h, plan, plan_sha, arm, report, rasterizer_class, aa_context):
    """The inherited 6000-step function and all of its loss/optimizer code run unchanged."""
    original_forward, original_save = rasterizer_class.forward, old.save_endpoint
    hook_state = SimpleNamespace(calls=0)
    report.update(renderer_metadata(plan))
    try:
        with aa_context(rasterizer_class, capture_last=False, near_plane=.01) as hook_state:
            def endpoint(path, **kwargs):
                return save_endpoint(old, h, hook_state, path, **kwargs)
            old.save_endpoint = endpoint
            old.train_arm(plan, plan_sha, arm, report)
            old.require(hook_state.calls == 6350, 'Expected 350+6000 AA calls per arm')
    finally:
        old.save_endpoint = original_save
        restored = rasterizer_class.forward is original_forward
        scope = {'arm': arm, 'plan_sha256': plan_sha, 'aa_calls': hook_state.calls,
                 'capture_last': False, 'hook_restored': restored, **renderer_metadata(plan)}
        directory = Path(plan['output'])/arm
        if directory.exists():
            path = directory/'aa_scope_receipt.json'
            old.write(path, scope)
            report['aa_scope_receipt'] = {'path': str(path), 'sha256': old.sha(path)}
        old.require(restored, 'AA hook restoration failed')


def numerical_flags(torch):
    return {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'benchmark': torch.backends.cudnn.benchmark,
            'precision': torch.get_float32_matmul_precision()}


def execute(plan_path, expected_sha):
    plan_path = Path(plan_path)
    plan = read(plan_path); snapshot = Path(plan['source_snapshot'])
    old = load_worker(snapshot/'train_ibgs_warm_matched.py')
    old.require(old.sha(plan_path) == expected_sha, 'Plan SHA mismatch')
    old.require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Execute frozen new worker only')
    old.verify(plan)
    old.require(plan['renderer_profile'] == PROFILE and plan['protocol'] == PROTOCOL, 'Wrong renderer profile')
    old.require(Path(sys.prefix) == Path(plan['interpreter']).parent.parent, 'Use isolated IBGS interpreter')
    output = Path(plan['output'])
    old.require(not any((output/n).exists() for n in ('execution_started.json', 'execution_receipt.json', *plan['arms'])),
                'Single attempt only')
    old.write(output/'execution_started.json', {'plan_sha256': expected_sha, 'pid': os.getpid()})
    sys.path[:0] = [str(snapshot), plan['backend_python'], str(snapshot/'official')]
    import diff_plane_rasterization
    import torch

    from bridge_rgs.ibgs_antialias import aa_opacity_rasterizer
    h = old.helper_import(snapshot)
    old.require(plan['specification'] == revised_specification(h.SPEC) and plan['arms'] == list(h.ARMS),
                'Only the declared recipe/profile revision is allowed')
    original_spec = h.SPEC
    h.SPEC = plan['specification']  # In-memory metadata only; inherited helper/source bytes are unchanged.
    prior_flags = numerical_flags(torch)
    prior_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Fixed 1800-second arm budget exhausted')))
    start = time.monotonic()
    report = {'status': 'failed', 'plan_sha256': expected_sha, 'arms': [], **renderer_metadata(plan)}
    try:
        old.require(torch.__version__ == plan['environment']['torch'], 'Wrong Torch runtime')
        binary = Path(diff_plane_rasterization._C.__file__).resolve()
        old.require(binary == Path(plan['binary']['path']).resolve() and old.sha(binary) == plan['binary']['sha256'],
                    'Wrong actual backend')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.set_float32_matmul_precision('highest')
        report['numerics_actual'] = numerical_flags(torch)
        old.require(report['numerics_actual'] == {'matmul_tf32': False, 'cudnn_tf32': False,
                    'benchmark': False, 'precision': 'highest'}, 'Numerics settings not active')
        report['actual_binary'] = {'path': str(binary), 'sha256': old.sha(binary)}
        for arm in h.ARMS:
            row = {}
            train_arm(old, h, plan, expected_sha, arm, row, diff_plane_rasterization.GaussianRasterizer, aa_opacity_rasterizer)
            row['receipt_path'] = str(output/arm/'training_receipt.json')
            row['receipt_sha256'] = old.sha(row['receipt_path'])
            report['arms'].append(row)
            gc.collect(); torch.cuda.empty_cache()
        first, second = report['arms']
        for key in ('initial_field_hashes', 'initial_head_hashes', 'background_sha256', 'sample_order_sha256'):
            old.require(first[key] == second[key], 'Arms mismatch: '+key)
        old.verify(plan)
        actual = {}
        for name, module in sys.modules.copy().items():
            path = getattr(module, '__file__', None)
            if path and name.split('.')[0] in ('bridge_rgs', 'scene', 'utils', 'gaussian_renderer',
                                               'arguments', 'color_aggregation_network', 'inherited_ibgs_warm_trainer'):
                p = Path(path).resolve()
                old.require(p.is_relative_to(snapshot) and old.sha(p) == plan['source_hashes'][str(p.relative_to(snapshot))],
                            'Non-frozen imported source: '+str(p))
                actual[name] = {'path': str(p), 'sha256': old.sha(p)}
        report.update(status='completed', actual_imports=actual, inputs_and_sources_unchanged=True,
                      optimizer_steps=12000, aa_opacity_calls=12700, VAL_reads=0, semantic_label_decodes=0)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, prior_handler); h.SPEC = original_spec
        torch.set_float32_matmul_precision(prior_flags['precision'])
        torch.backends.cuda.matmul.allow_tf32 = prior_flags['matmul_tf32']
        torch.backends.cudnn.allow_tf32 = prior_flags['cudnn_tf32']
        torch.backends.cudnn.benchmark = prior_flags['benchmark']
        report['numerics_restored'] = numerical_flags(torch) == prior_flags
        report['elapsed_seconds'] = time.monotonic()-start
        if not report['numerics_restored']:
            report.update(status='failed', restoration_error='Numerical flags not restored')
        old.write(output/'execution_receipt.json', report)
        old.require(report['numerics_restored'], 'Numerical flags not restored')


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', type=Path)
    modes.add_argument('--run', type=Path)
    parser.add_argument('--fd-run', type=Path)
    parser.add_argument('--backward-run', type=Path)
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        if not (args.fd_run and args.backward_run):
            parser.error('--fd-run and --backward-run are required')
        prepare(args.prepare, args.fd_run, args.backward_run)
    else:
        if not args.expected_plan_sha256:
            parser.error('--expected-plan-sha256 is required')
        execute(args.run, args.expected_plan_sha256)


if __name__ == '__main__':
    main()
