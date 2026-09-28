"""Evaluate both seed-43 correspondence arms on the fixed 50-view RGB grid."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
TEMPLATE = RUNS/'ibgs_correspondence_evaluation_v1'
PARENT = RUNS/'ibgs_layer_heads_matched_v1'
CORRECT = RUNS/'ibgs_correspondence_seed43_correct_v1'
PERMUTED = RUNS/'ibgs_correspondence_seed43_permuted_v1'
F_METRICS = RUNS/'ibgs_top4_mcmc_replacement_v1/rgb_metrics.json'
OUT = RUNS/'ibgs_correspondence_seed43_evaluation_v1'
ARMS = {'top4_normalized_correct_seed43': ('top4', 'normalized'),
        'top4_normalized_permuted_seed43': ('top4', 'normalized')}


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_base():
    base = load(TEMPLATE/'source_snapshot/correspondence_evaluation_base.py', 'seed43_evaluation_base')
    base.ARMS = dict(ARMS)
    base.PRIMARY = 'top4_normalized_correct_seed43'
    base.SPEC = copy.deepcopy(base.SPEC)
    base.SPEC.update(protocol='ibgs_normalized_correspondence_seed43_evaluation_v1', arms=list(ARMS),
                     target_calls=100, selector_calls=100, native_arrays=100, delivered_pngs=100,
                     lpips_calls=100, primary_candidate=base.PRIMARY,
                     comparisons=['correct_minus_permuted', 'correct_minus_F'],
                     selection='fixed endpoints from seed-43 matched replication; no search')

    def validate_endpoint(saved, arm, parent, plan_sha, *, parameter_count=66095):
        expected = {sha(CORRECT/'plan.json'), sha(PERMUTED/'plan.json')}
        require(saved['arm'] == arm and saved['step'] == 6000 and saved['plan_sha256'] in expected,
                'Seed-43 endpoint identity mismatch')
        require(saved['field_checkpoint'] == parent['checkpoint']
                and saved['cache_manifest_sha256'] == parent['cache_manifest_sha256'],
                'Seed-43 endpoint field/cache mismatch')
        require(saved['specification']['seed'] == 43 and saved['specification']['protocol'].startswith(
            'ibgs_normalized_correspondence_seed43'), 'Seed-43 specification mismatch')
        import torch
        head = saved['head']
        require(sum(v.numel() for v in head.values()) == parameter_count
                and all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in head.values()),
                'Finite FP32 head schema required')
    base.validate_endpoint = validate_endpoint
    return base


def prepare():
    require(not OUT.exists(), 'Refuse to overwrite evaluation output')
    template = read(TEMPLATE/'plan.json')
    correct_plan = read(CORRECT/'plan.json'); perm_plan = read(PERMUTED/'plan.json')
    # Require both natural training artifacts and explicit endpoint hashes.
    for folder, plan in ((CORRECT, correct_plan), (PERMUTED, perm_plan)):
        rec = read(folder/'execution_receipt.json'); launch = read(folder/'launch_receipt.json')
        require(rec['status'] == launch['status'] == 'completed' and launch['exit_code'] == 0
                and launch['natural_completion'] and rec['completed_updates'] == 6000,
                f'Completed seed-43 training required: {folder}')
    endpoints = {
        'top4_normalized_correct_seed43': correct_plan['output']+'/top4_normalized_correct_seed43/last.pt',
        'top4_normalized_permuted_seed43': perm_plan['output']+'/top4_normalized_permuted_seed43/last.pt'}
    plans = {'top4_normalized_correct_seed43': correct_plan,
             'top4_normalized_permuted_seed43': perm_plan}
    for arm, path in endpoints.items():
        require(Path(path).is_file(), f'Missing endpoint {path}')
        require(sha(path) == plans[arm]['checkpoint_sha256'] if 'checkpoint_sha256' in plans[arm] else True,
                'Endpoint plan identity malformed')
    base = load_base()
    plan = copy.deepcopy(template)
    plan['specification'] = base.SPEC
    plan['output'] = str(OUT)
    plan['endpoints'] = {arm: {'path': path, 'sha256': sha(path), 'training_plan_sha256': sha(CORRECT/'plan.json') if arm.startswith('top4_normalized_correct') else sha(PERMUTED/'plan.json')}
                         for arm, path in endpoints.items()}
    plan['training_specification'] = {arm: plans[arm]['specification'] for arm in ARMS}
    plan['training_plan_sha256'] = 'seed43_matched_two_arm'
    plan['reference_metrics'] = {'F_rgb': {'path': str(F_METRICS), 'sha256': sha(F_METRICS)}}
    plan['input_hashes'] = dict(template['input_hashes'])
    plan['input_hashes'].update({str(CORRECT/'plan.json'): sha(CORRECT/'plan.json'),
                                 str(CORRECT/'execution_receipt.json'): sha(CORRECT/'execution_receipt.json'),
                                 str(CORRECT/'launch_receipt.json'): sha(CORRECT/'launch_receipt.json'),
                                 str(PERMUTED/'plan.json'): sha(PERMUTED/'plan.json'),
                                 str(PERMUTED/'execution_receipt.json'): sha(PERMUTED/'execution_receipt.json'),
                                 str(PERMUTED/'launch_receipt.json'): sha(PERMUTED/'launch_receipt.json'),
                                 str(F_METRICS): sha(F_METRICS), str(ROOT/'scripts/evaluate_ibgs_correspondence_seed43.py'): sha(__file__)})
    OUT.mkdir(parents=True)
    write(OUT/'plan.json', plan)
    print(json.dumps({'plan_sha256': sha(OUT/'plan.json'), 'output': str(OUT)}))


def execute(stage, plan_path):
    plan = read(plan_path); base = load_base(); output = Path(plan['output'])
    require(sha(plan_path) == plan['plan_sha256'] if 'plan_sha256' in plan else True, 'Plan hash field mismatch')
    report = {'status': 'failed', 'stage': stage, 'plan_sha256': sha(plan_path),
              'VAL_rgb_reads': 0, 'lpips_calls': 0, 'target_calls': 0, 'selector_calls': 0,
              'source_rgb_decodes': 0, 'started': time.time()}
    if stage == 'render':
        base.render_stage(plan, report)
        write(output/'render_execution_receipt.json', {**report, 'status': 'completed',
            'natural_completion': True, 'finished': time.time()})
    else:
        base.score_stage(plan, report)
        # Add the fixed primary mechanism comparison after both endpoint scores exist.
        correct = read(output/'metrics_top4_normalized_correct_seed43.json')
        permuted = read(output/'metrics_top4_normalized_permuted_seed43.json')
        helper = base.legacy().scoring_modules(plan['source_snapshot'])[1]
        pair = helper.paired_rgb(permuted, correct)
        pair.update(reference='top4_normalized_permuted_seed43', candidate='top4_normalized_correct_seed43',
                    protocol=plan['specification']['protocol'])
        write(output/'paired_correct_minus_permuted.json', pair)
        write(output/'score_execution_receipt.json', {**report, 'status': 'completed',
            'natural_completion': True, 'finished': time.time(),
            'mechanism_pair_sha256': sha(output/'paired_correct_minus_permuted.json')})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--stage', choices=('render', 'score'))
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(); return
    require(args.stage and args.plan, 'Stage and plan required')
    execute(args.stage, args.plan)


if __name__ == '__main__':
    main()
