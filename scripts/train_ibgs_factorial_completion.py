"""Complete the missing median4/normalized corner after the fixed three-arm result.

Reuse the exact old training body, adapting only arm counting and adding a
cross-run initialization check. No optimizer, evidence or photometric changes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
PARENT = RUNS/'ibgs_layer_heads_matched_v1'
EVALUATION = RUNS/'ibgs_layer_heads_evaluation_v1'
PARENT_SHA = 'b9a44c95b4e68b955f4fcc27abf27917134b4c5df8f456ed31c04a69b3a41c1a'
EVALUATION_SHA = '5c744495787a55c48a11cce9e0afc27f8a21758a08013b82fe4454f61b955929'
ORIGINAL_SHA = '5dd48b57ce31be61ca3ce4e4b67a61c083d4e6ead16ef3077f10ce9fa4987c43'
ORIGINAL_NAME = 'train_ibgs_layer_heads.py'
BASE_NAME = 'single_head_training_base.py'
ARM = 'median4_normalized'
ADAPTATIONS = (
    ("require(hashes == initial_hashes, 'Matched head initialization differs')",
     ("require(hashes == initial_hashes, 'Matched head initialization differs')\n"
      "        require(hashes == plan['expected_initial_head_hashes'], 'Original three-arm initialization differs')")),
    ('3*len(order) == report', 'len(ARMS)*len(order) == report'),
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


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def adapted_source(content):
    require(hashlib.sha256(content.encode()).hexdigest() == ORIGINAL_SHA, 'Exact old training source required')
    for old, new in ADAPTATIONS:
        require(content.count(old) == 1, 'Unambiguous control-flow adaptation required')
        content = content.replace(old, new)
    return content


def profile(specification):
    result = copy.deepcopy(specification)
    result.update(protocol='ibgs_fixed_layer_factorial_completion_v1', arms=[ARM],
        selection='fixed last 6000; post-three-arm fourth corner; no early/best selection',
        scope='same original field, source evidence, initialization, order and head budget; '
              'post-result development comparison, not new semantic or novelty evidence')
    return result


def profiled_base(snapshot):
    original = snapshot/ORIGINAL_NAME; base_path = snapshot/BASE_NAME
    require(sha(original) == ORIGINAL_SHA
            and base_path.read_text() == adapted_source(original.read_text()), 'Training body adaptation changed')
    base = load(base_path, 'fixed_factorial_training_base')
    base.SPEC = profile(base.SPEC); base.ARMS = {ARM: ('median4', 'normalized')}
    return base


def natural(folder, stage=None):
    prefix = '' if stage is None else stage+'_'
    receipt_path = folder/(prefix+'execution_receipt.json')
    receipt, launch = read(receipt_path), read(folder/(prefix+'launch_receipt.json'))
    require(receipt['status'] == launch['status'] == 'completed' and launch['exit_code'] == 0
            and launch['natural_completion'] is True
            and receipt['plan_sha256'] == launch['plan_sha256'] == sha(folder/'plan.json')
            and launch['execution_receipt_sha256'] == sha(receipt_path), 'Natural completed parent required')
    return receipt


def prepare(args):
    require(sha(PARENT/'plan.json') == PARENT_SHA
            and sha(EVALUATION/'plan.json') == EVALUATION_SHA, 'Fixed prior experiment required')
    completion = natural(PARENT); scored = natural(EVALUATION, 'score')
    parent = read(PARENT/'plan.json')
    require(completion['completed_updates'] == 18000 and completion['field_unchanged']
            and len(completion['arms']) == 3 and scored['VAL_rgb_reads'] == 50,
            'Complete original three-arm experiment required')
    origin = Path(parent['source_snapshot']); require(sha(origin/ORIGINAL_NAME) == ORIGINAL_SHA, 'Original worker changed')
    old = load(origin/ORIGINAL_NAME, 'original_three_head_training')
    require(old.tree(origin) == parent['source_hashes'], 'Original frozen implementation changed')
    initial = completion['arms'][0]['initial_head_hashes']
    require(all(a['initial_head_hashes'] == initial and a['updates'] == 6000
                and a['parameter_count'] == 66095 and a['status'] == 'completed'
                for a in completion['arms']), 'Original matched initialization required')
    contract = read(parent['data_contract']['path']); names = [r['name'] for r in contract['train_rows']]
    require(all(r['split'] == 'train' for r in contract['train_rows']), 'Only original TRAIN sources')
    order = old.matched_order(names, args.phase)
    if args.phase == 'train':
        require(order == read(parent['camera_order'])['names'], 'Original 6000 camera order required')
        require(args.preflight is not None, 'Natural two-step preflight required')
        pre = natural(args.preflight.resolve()); pp = read(args.preflight.resolve()/'plan.json')
        require(pp['phase'] == 'preflight' and pp['specification'] == profile(parent['specification'])
                and pp['parent_plan_sha256'] == PARENT_SHA and pre['completed_updates'] == 2
                and pre['raster_calls'] == pre['selector_calls'] == 2
                and pre['field_unchanged'] and len(pre['arms']) == 1
                and pre['arms'][0]['initial_head_hashes'] == initial
                and pre['arms'][0]['status'] == 'completed', 'Successful matched single-arm preflight required')
    inputs = dict(parent['input_hashes'])
    for folder, files in ((PARENT, ('plan.json', 'execution_receipt.json', 'launch_receipt.json')),
                          (EVALUATION, ('plan.json', 'score_execution_receipt.json', 'score_launch_receipt.json'))):
        inputs.update({str(folder/n): sha(folder/n) for n in files})
    if args.phase == 'train':
        inputs.update({str(args.preflight.resolve()/n): sha(args.preflight.resolve()/n) for n in
                       ('plan.json', 'execution_receipt.json', 'launch_receipt.json')})
    require(all(sha(p) == h for p, h in inputs.items()), 'Parent input bytes changed')
    output = args.output.resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    snapshot = output/'source_snapshot'; snapshot.mkdir(parents=True)
    for relative, expected in parent['source_hashes'].items():
        source, target = origin/relative, snapshot/relative
        require(source.resolve().is_relative_to(origin.resolve()) and target.resolve().is_relative_to(snapshot.resolve())
                and sha(source) == expected, 'Bound original source copy required')
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    with (snapshot/BASE_NAME).open('x') as stream:
        stream.write(adapted_source((origin/ORIGINAL_NAME).read_text()))
    for path in (Path(__file__), ROOT/'tests/test_ibgs_factorial_completion.py',
                 ROOT/'docs/ibgs_factorial_completion_protocol.md'):
        shutil.copy2(path, snapshot/path.name)
    base = profiled_base(snapshot); hashes = base.tree(snapshot)
    if args.phase == 'train':
        require(hashes == pp['source_hashes'], 'Source changed after single-arm preflight')
    write(output/'camera_order.json', {'names': order})
    inputs[str(output/'camera_order.json')] = sha(output/'camera_order.json')
    plan = {k: parent[k] for k in ('cache_manifest', 'cache_manifest_sha256', 'checkpoint', 'data_contract',
            'selector', 'backend', 'runtime_sources', 'interpreter', 'neighbors')}
    plan.update(specification=base.SPEC, phase=args.phase, output=str(output), source_snapshot=str(snapshot),
                source_hashes=hashes, input_hashes=inputs, camera_order=str(output/'camera_order.json'),
                parent_plan_sha256=PARENT_SHA, prior_evaluation_plan_sha256=EVALUATION_SHA,
                expected_initial_head_hashes=initial, original_worker_sha256=ORIGINAL_SHA,
                adapted_base_sha256=sha(snapshot/BASE_NAME),
                code_adaptations=['explicit original initialization check before first update',
                                  'count total updates using number of arms'],
                external_timeout_seconds=360 if args.phase == 'preflight' else 4860,
                preparation_RGB_decodes=0, preparation_CUDA_calls=0)
    write(output/'plan.json', plan)
    print(json.dumps({'plan_sha256': sha(output/'plan.json'), 'output': str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true'); action.add_argument('--run', type=Path)
    parser.add_argument('--phase', choices=('preflight', 'train')); parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight', type=Path); parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        require(args.phase and args.output, 'Preparation requires phase and fresh output')
        prepare(args); return
    require(args.expected_plan_sha256 and sha(args.run) == args.expected_plan_sha256, 'Explicit plan SHA required')
    plan = read(args.run); snapshot = Path(plan['source_snapshot'])
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Execute frozen wrapper')
    base = profiled_base(snapshot)
    require(plan['specification'] == base.SPEC and plan['parent_plan_sha256'] == PARENT_SHA
            and plan['prior_evaluation_plan_sha256'] == EVALUATION_SHA, 'Fixed factorial profile required')
    # Its parser accepts the same --run and --expected-plan-sha256 arguments;
    # its own __file__ is also frozen and checked before the common execute path.
    base.main()


if __name__ == '__main__':
    main()
