"""Thin factored-FP64 revision of the frozen warm-IBGS matched trainer."""
from __future__ import annotations

import argparse
import ast
import copy
import gc
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

ROOT = Path('/home/sky/workspace/SHM2026')
PARENT = Path('/mnt/data/SHM2026/runs/ibgs_warm_matched_v1')
FAILED_AA = Path('/mnt/data/SHM2026/runs/ibgs_aa_warm_matched_v1')
PORT = Path('/mnt/data/SHM2026/third_party/ibgs_aa_port_v1')
PARENT_PLAN_SHA = 'c1af18ac17a1ce934208fa91f2b987f7a46c9920b7e35eb3ba9c95dafc056691'
SCOPE_SHA = '7782f75f9a0c400c692fd765a52f7b63b55573c2562864a3d4f30633fa4173db'
INITIAL = '/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1/training/last.pt'
INITIAL_SHA = '35b489fe45ad34ad279018609b6d5eaf6493e4c0d82f9c86ec16925268078092'
BINARY_SHA = 'c504257ae703800e90fc79c8f0541f0cd7757cd3e293e078760a5535c2bae35d'
PROTOCOL = 'ibgs_aa_stable_matched_v2'
PROVIDER = 'bridge_rgs.ibgs_antialias_stable'
MODES = ('compatibility', 'gradient', 'preflight', 'failure')
PROFILE = {
    'id': 'ibgs_centered_corner_v2_aa_factored64_near001_v2', 'near_plane': .01,
    'eps2d': .3, 'compensation': 'sqrt(D/(D+.3*sum(B**2)+.09)); D=sum(three minor squares)',
    'opacity': 'FP64 activated_opacity*rho, cast to input dtype; no floor or normalization',
    'derivative': 'positive-D Torch derivative; explicit zero at computed rank degeneration',
    'capture_last': False, 'expected_calls_per_arm': 6350,
    'scope': 'Existing IBGS backend approximate VJPs retained; not gsplat backward equivalence',
}


def read(path):
    return json.loads(Path(path).read_text())


def load_scope(path):
    spec = importlib.util.spec_from_file_location('inherited_ibgs_aa_scope', path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def precision_policy(path):
    """Read the bound literal without importing Torch or a live project package."""
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PRECISION_POLICY' for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError('Missing literal PRECISION_POLICY')


def revised_specification(original, policy):
    result = copy.deepcopy(original)
    result.update(protocol=PROTOCOL, renderer_profile=copy.deepcopy(PROFILE),
                  aa_provider=PROVIDER, precision_policy=copy.deepcopy(policy),
                  internal_seconds_per_arm=3600, external_seconds=7500)
    return result


def validate_initial_checkpoint(plan, old):
    old.require(plan['checkpoint'] == INITIAL and plan['input_hashes'].get(INITIAL) == INITIAL_SHA,
                'Only the original 35b489 field is allowed; no failure/resume checkpoint')
    old.require(plan['arms'] == ['full', 'no_source'], 'Both fixed arms required')


def validate_revision(mode, plan, receipt, module_sha, policy, old, scope):
    old.require(mode in MODES and plan.get('mode') == mode
                and plan.get('renderer_profile', {}).get('id') == PROFILE['id']
                and plan.get('aa_provider') == PROVIDER and plan.get('aa_module_sha256') == module_sha
                and plan.get('precision_policy') == policy and plan.get('binary', {}).get('sha256') == BINARY_SHA,
                'Wrong stable validation profile/provider/precision/backend')
    matches = [v for k, v in plan.get('source_hashes', {}).items() if Path(k).name == 'ibgs_antialias_stable.py']
    binding = receipt.get('provider_binding', {})
    old.require(matches and all(v == module_sha for v in matches)
                and binding.get('module') == binding.get('callable_module') == PROVIDER
                and binding.get('sha256') == module_sha and binding.get('precision_policy') == policy
                and receipt.get('provider_restored') is True and receipt.get('numerical_status') == 'passed',
                'Stable numerical evidence must use the actual bound provider')
    if mode == 'gradient':
        scope.validate_fd_receipt(receipt, old)
    elif mode == 'preflight':
        scope.validate_backward_receipt(receipt, old)
    elif mode == 'compatibility':
        old.require(receipt.get('render_calls') == receipt.get('antialias_calls') == 16
                    and receipt.get('backward') == receipt.get('optimizer_steps') == 0
                    and len(receipt.get('predictions', [])) == 16
                    and all(receipt.get(k) is True for k in ('field_unchanged', 'hook_restored', 'prediction_barrier_complete')),
                    'Incomplete stable 16-view compatibility check')
    else:
        old.require(receipt.get('raster_calls') == receipt.get('backward') == 1
                    and receipt.get('optimizer_steps') == receipt.get('data_reads') == 0
                    and receipt.get('field_parameters_unchanged') is True
                    and receipt.get('hook_restored') is True and receipt.get('antialias_calls') == 1
                    and receipt.get('camera') == '004.png' and receipt.get('checkpoint_step') == 61
                    and plan.get('diagnostic_failure_checkpoint', {}).get('sha256')
                        == 'abbe8973ebc7649e99b162f9d878129232fe5475ffe8672f2e4fc9b48c5b5d8f'
                    and len(receipt.get('gradient_summary', {})) == 6
                    and all(v['finite'] is True for v in receipt['gradient_summary'].values()),
                    'Failed-state camera 004 raw forward/backward check required; never a training start')


def prepare(output, validation_runs):
    scope_path = FAILED_AA/'source_snapshot/train_ibgs_aa_warm_matched.py'
    scope = load_scope(scope_path)
    old = scope.load_worker(PARENT/'source_snapshot/train_ibgs_warm_matched.py')
    old.require(old.sha(scope_path) == SCOPE_SHA and old.sha(PARENT/'plan.json') == PARENT_PLAN_SHA,
                'Wrong frozen training ancestor')
    parent, _ = scope.completed_evidence(PARENT, old)
    old.verify(parent); validate_initial_checkpoint(parent, old)
    old.require(read(PARENT/'training_contract_review.json')['status'] == 'passed', 'Parent contract audit required')
    module_path = ROOT/'src/bridge_rgs/ibgs_antialias_stable.py'
    module_sha, policy = old.sha(module_path), precision_policy(module_path)
    old.require(set(validation_runs) == set(MODES), 'All four fixed stable checks required')
    for mode in MODES:
        evidence, receipt = scope.completed_evidence(validation_runs[mode], old, numerical=True)
        validate_revision(mode, evidence, receipt, module_sha, policy, old, scope)
        old.require(old.sha(receipt['provider_binding']['path']) == module_sha, 'Actual validation provider changed')
    build, port = read(PORT/'build_receipt.json'), read(PORT/'preparation.json')
    binary = build['binary']
    old.require(build['status'] == 'completed' and build['natural_completion'] is True
                and build['exit_code'] == 0 and binary['sha256'] == old.sha(binary['path']) == BINARY_SHA,
                'Bound completed near-.01 backend required')
    output = Path(output).resolve()
    old.require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    snapshot = output/'source_snapshot'
    scope.copy_inherited_sources(parent, snapshot, old)
    for source, target in ((scope_path, snapshot/scope_path.name),
                           (module_path, snapshot/'bridge_rgs/ibgs_antialias_stable.py'),
                           (Path(__file__), snapshot/Path(__file__).name),
                           (ROOT/'tests/test_ibgs_aa_stable_training.py', snapshot/'test_ibgs_aa_stable_training.py'),
                           (ROOT/'docs/ibgs_aa_stable_matched_protocol.md', snapshot/'ibgs_aa_stable_matched_protocol.md')):
        shutil.copyfile(source, target)
    shutil.copyfile(parent['camera_order'], output/'camera_order.json')
    inputs, provenance = dict(parent['input_hashes']), {}
    groups = [('parent', PARENT, ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'training_contract_review.json')),
              ('failed_aa_v1', FAILED_AA, ('plan.json', 'execution_receipt.json', 'launch_receipt.json')),
              ('isolated_port', PORT, ('preparation.json', 'build_receipt.json', 'near_plane.patch'))]
    groups += [(mode, Path(validation_runs[mode]), ('plan.json', 'execution_receipt.json', 'launch_receipt.json')) for mode in MODES]
    for key, directory, names in groups:
        provenance[key] = {}
        for name in names:
            path = directory/name; digest = old.sha(path)
            inputs[str(path)] = digest
            provenance[key][name] = {'path': str(path), 'sha256': digest}
    runtime = dict(parent['runtime_sources'])
    runtime[binary['path']] = binary['sha256']
    wrapper = PORT/'python/diff_plane_rasterization/__init__.py'
    runtime[str(wrapper)] = old.sha(wrapper)
    for name, digest in port['source_hashes'].items():
        path = PORT/'source'/name
        old.require(old.sha(path) == digest, 'Near backend source changed')
        runtime[str(path)] = digest
    plan = dict(parent)
    plan.update(protocol=PROTOCOL, specification=revised_specification(parent['specification'], policy),
                renderer_profile=copy.deepcopy(PROFILE), aa_provider=PROVIDER, precision_policy=policy,
                output=str(output), source_snapshot=str(snapshot), source_hashes=old.tree_hashes(snapshot),
                input_hashes=inputs, runtime_sources=runtime, inherited_sources=parent['source_hashes'], provenance=provenance,
                inherited_aa_scope_sha256=SCOPE_SHA, binary=binary, backend_python=str(PORT/'python'),
                aa_module_sha256=module_sha, camera_order=str(output/'camera_order.json'),
                camera_order_sha256=old.sha(output/'camera_order.json'),
                preparation_scope='CPU metadata/hash only; original field fresh start; no resume or model/image load',
                actual_recipe_changes=['AA factor evaluated in FP64 from original activated inputs',
                                       'near .2 to .01 versus original warm ancestor',
                                       'timeout 3600 seconds/arm, 7500 outer; fixed iteration budget unchanged'])
    old.require(plan['camera_order_sha256'] == parent['camera_order_sha256'], 'Same 6000-camera order required')
    old.write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'plan_sha256': old.sha(output/'plan.json'),
                      'source_files': len(plan['source_hashes']), 'input_files': len(inputs)}), flush=True)


def renderer_metadata(plan):
    return {'renderer_profile': copy.deepcopy(PROFILE), 'aa_module_sha256': plan['aa_module_sha256'],
            'renderer_binary': plan['binary'], 'aa_provider': PROVIDER,
            'precision_policy': copy.deepcopy(plan['precision_policy'])}


@contextmanager
def inherited_scope(scope):
    """Only metadata dispatch changes; the frozen arm/endpoint implementation is reused."""
    original = scope.PROTOCOL, scope.PROFILE, scope.renderer_metadata
    scope.PROTOCOL, scope.PROFILE, scope.renderer_metadata = PROTOCOL, PROFILE, renderer_metadata
    try:
        yield
    finally:
        scope.PROTOCOL, scope.PROFILE, scope.renderer_metadata = original


def execute(plan_path, expected_sha):
    plan_path = Path(plan_path); plan = read(plan_path); snapshot = Path(plan['source_snapshot'])
    scope = load_scope(snapshot/'train_ibgs_aa_warm_matched.py')
    old = scope.load_worker(snapshot/'train_ibgs_warm_matched.py')
    old.require(old.sha(plan_path) == expected_sha and Path(__file__).resolve() == snapshot/Path(__file__).name,
                'Only the exact frozen plan/worker may execute')
    old.verify(plan); validate_initial_checkpoint(plan, old)
    old.require(old.sha(snapshot/'train_ibgs_aa_warm_matched.py') == SCOPE_SHA
                and plan['renderer_profile'] == PROFILE and plan['protocol'] == PROTOCOL
                and plan['aa_provider'] == PROVIDER and plan['binary']['sha256'] == BINARY_SHA,
                'Wrong inherited scope or stable renderer')
    old.require(Path(sys.prefix) == Path(plan['interpreter']).parent.parent, 'Use isolated IBGS interpreter')
    output = Path(plan['output'])
    old.require(not any((output/n).exists() for n in ('execution_started.json', 'execution_receipt.json', *plan['arms'])),
                'Single attempt only; failure directories must be retained')
    old.write(output/'execution_started.json', {'plan_sha256': expected_sha, 'pid': os.getpid()})
    sys.path[:0] = [str(snapshot), plan['backend_python'], str(snapshot/'official')]
    import diff_plane_rasterization
    import torch
    provider = importlib.import_module(PROVIDER)
    old.require(Path(provider.__file__).resolve() == snapshot/'bridge_rgs/ibgs_antialias_stable.py'
                and old.sha(provider.__file__) == plan['aa_module_sha256']
                and provider.PRECISION_POLICY == plan['precision_policy'], 'Actual stable provider differs')
    h = old.helper_import(snapshot)
    old.require(plan['specification'] == revised_specification(h.SPEC, provider.PRECISION_POLICY)
                and plan['arms'] == list(h.ARMS), 'Undeclared training recipe change')
    original_spec = h.SPEC; h.SPEC = plan['specification']
    prior_flags = scope.numerical_flags(torch)
    prior_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Fixed 3600-second arm budget exhausted')))
    start = time.monotonic()
    report = {'status': 'failed', 'plan_sha256': expected_sha, 'arms': [], **renderer_metadata(plan)}
    try:
        old.require(torch.__version__ == plan['environment']['torch'], 'Wrong Torch runtime')
        binary = Path(diff_plane_rasterization._C.__file__).resolve()
        old.require(binary == Path(plan['binary']['path']).resolve() and old.sha(binary) == BINARY_SHA, 'Wrong actual backend')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.set_float32_matmul_precision('highest')
        report['numerics_actual'] = scope.numerical_flags(torch)
        old.require(report['numerics_actual'] == {'matmul_tf32': False, 'cudnn_tf32': False,
                    'benchmark': False, 'precision': 'highest'}, 'Numerics settings not active')
        report['actual_binary'] = {'path': str(binary), 'sha256': old.sha(binary)}
        with inherited_scope(scope):
            for arm in h.ARMS:
                row = {}
                scope.train_arm(old, h, plan, expected_sha, arm, row,
                                diff_plane_rasterization.GaussianRasterizer, provider.aa_opacity_rasterizer)
                row['receipt_path'] = str(output/arm/'training_receipt.json')
                row['receipt_sha256'] = old.sha(row['receipt_path'])
                report['arms'].append(row)
                gc.collect(); torch.cuda.empty_cache()
        for key in ('initial_field_hashes', 'initial_head_hashes', 'background_sha256', 'sample_order_sha256'):
            old.require(report['arms'][0][key] == report['arms'][1][key], 'Arms mismatch: '+key)
        old.verify(plan)
        actual = {}
        for name, module in sys.modules.copy().items():
            path = getattr(module, '__file__', None)
            if path and name.split('.')[0] in ('bridge_rgs', 'scene', 'utils', 'gaussian_renderer', 'arguments',
                    'color_aggregation_network', 'inherited_ibgs_warm_trainer', 'inherited_ibgs_aa_scope'):
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
        report['numerics_restored'] = scope.numerical_flags(torch) == prior_flags
        report['elapsed_seconds'] = time.monotonic()-start
        if not report['numerics_restored']:
            report.update(status='failed', restoration_error='Numerical flags not restored')
        old.write(output/'execution_receipt.json', report)
        old.require(report['numerics_restored'], 'Numerical flags not restored')


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', type=Path); modes.add_argument('--run', type=Path)
    for mode in MODES:
        parser.add_argument('--'+mode+'-run', type=Path)
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        runs = {mode: getattr(args, mode+'_run') for mode in MODES}
        if not all(runs.values()):
            parser.error('All four --compatibility/gradient/preflight/failure-run paths are required')
        prepare(args.prepare, runs)
    else:
        if not args.expected_plan_sha256:
            parser.error('--expected-plan-sha256 is required')
        execute(args.run, args.expected_plan_sha256)


if __name__ == '__main__':
    main()
