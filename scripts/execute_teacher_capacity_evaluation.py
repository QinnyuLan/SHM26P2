"""Execute the existing locked two-endpoint evaluation once; never train or retry.

Default only verifies the static plan/source bindings. Root must launch --execute
once all four stages have completed and the exclusive GPU slot is released.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PLAN = Path('/mnt/data/SHM2026/runs/teacher_capacity_v1/evaluation_preparation/plan.json')
PLAN_SHA = 'be5a47bd4ace31f796c6d69cc9ed0f4c769a52278d3aa103a74e668da6f53f69'
KEYS = ('hplus_real', 'hplus_render_adapt', 'vit7b_real', 'vit7b_render_adapt')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound(item):
    require(sha(item['path']) == item['sha256'], f"SHA changed: {item['path']}")


def write(path, value, *, update=False):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    if update:
        temp = path.with_suffix('.tmp')
        temp.write_text(payload)
        temp.replace(path)
    else:
        with path.open('x') as handle:
            handle.write(payload)


def static_plan(path):
    require(sha(path) == PLAN_SHA, 'Only the existing immutable evaluation plan is supported')
    plan = read(path)
    for key in ('training_launch_manifest', 'training_draft_plan', 'inference_scene',
                'selected_reference_metrics', 'selected_reference_receipt'):
        bound(plan[key])
    snapshot = Path(plan['source_snapshot'])
    for name, expected in plan['source_files_sha256'].items():
        bound({'path': snapshot/name, 'sha256': expected})
    launch = read(plan['training_launch_manifest']['path'])
    require(launch['status'] == 'locked_authorized', 'Training launch was not locked')
    for name, expected in launch['locked_files_sha256'].items():
        bound({'path': name, 'sha256': expected})
    for name in ('uv.lock', 'pyproject.toml'):
        bound({'path': Path(launch['workspace_root'])/name,
               'sha256': plan['source_files_sha256'][name]})
    return plan, launch


def frozen_helpers(plan):
    snapshot = Path(plan['source_snapshot']).resolve()
    sys.path.insert(0, str(snapshot))
    names = ('run_teacher_capacity_stage', 'compare_official_evaluations', 'make_teacher_capacity_draft')
    modules = [importlib.import_module(name) for name in names]
    imports = {}
    for name, module in list(sys.modules.items()):
        if name in names or name.startswith('bridge_rgs.') or name == 'bridge_rgs':
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(snapshot), f'Non-frozen helper import: {name}')
            relative = str(path.relative_to(snapshot))
            require(sha(path) == plan['source_files_sha256'][relative], f'Imported source changed: {name}')
            imports[name] = {'path': str(path), 'sha256': sha(path)}
    return (*modules, imports)


def completed_sequence(sequence, launch, launch_sha):
    require(sequence.get('status') == 'completed', 'All four training stages must finish first')
    require(sequence['launch_manifest_sha256'] == launch_sha, 'Sequence launch SHA changed')
    require([s['key'] for s in sequence['stages']] == list(KEYS), 'Wrong stage sequence')
    require([s['key'] for s in launch['stages']] == list(KEYS), 'Wrong locked stage sequence')
    for actual, fixed in zip(sequence['stages'], launch['stages'], strict=True):
        require(actual['status'] == 'completed' and actual['exit_code'] == 0,
                'Unfinished or failed training stage')
        require(actual['command'] == fixed['command'], 'Stage command differs from launch')
        bound({'path': fixed['stage_receipt'], 'sha256': actual['stage_receipt_sha256']})


def endpoint_contract(receipt, specification, key, draft, launch, peer=None):
    capacity, stage = key.split('_', 1)
    steps = 6000 if stage == 'real' else 2000
    require(receipt.get('status') == 'completed' and receipt['steps'] == steps,
            'Wrong completed fixed stage endpoint')
    require((receipt['capacity'], receipt['stage']) == (capacity, stage), 'Stage identity changed')
    require(receipt['configuration'] == specification, 'Effective stage config changed')
    require(receipt['plan_sha256'] == launch['draft_plan_sha256']
            and receipt['runner_sha256'] == launch['stage_runner_sha256']
            and receipt['source_sha256'] == draft['source_review_sha256'], 'Stage source identity changed')
    expected = str(Path(specification['output_dir'])/'last.pt')
    require(receipt['last_checkpoint'] == expected and receipt['result']['step'] == steps,
            'Endpoint must be fixed last.pt, never best.pt')
    config = specification['config']
    require(config['steps'] == steps and config['eval_every'] == steps
            and config['independent_augmentation_rng'], 'Training protocol changed')
    if stage == 'render_adapt':
        require(peer is not None, 'Missing own real-stage endpoint')
        warm = {'path': peer['last_checkpoint'], 'sha256': peer['last_checkpoint_sha256'], 'step': 6000}
        require(receipt['warmstart'] == warm and config['warmstart_checkpoint'] == warm['path'],
                'Adaptation must use its own fixed real last EMA')
    else:
        require(receipt['warmstart'] is None and not config['warmstart_checkpoint'], 'Real stage must start fresh')


def training_ready(plan, launch, stage_module):
    # Refuse unfinished sequence before reading any potentially active checkpoint.
    sequence_path = Path(launch['output_root'])/'execution_receipt.json'
    sequence = read(sequence_path)
    completed_sequence(sequence, launch, plan['training_launch_manifest']['sha256'])
    draft = read(plan['training_draft_plan']['path'])
    records, inputs = {}, {str(sequence_path): sha(sequence_path)}
    import torch
    for item in launch['stages']:
        key = item['key']
        config_path = Path(launch['source_snapshot'])/'draft'/Path(draft['configurations'][key]['path']).name
        bound({'path': config_path, 'sha256': draft['configurations'][key]['sha256']})
        specification, record = read(config_path), read(item['stage_receipt'])
        endpoint_contract(record, specification, key, draft, launch, records.get(key.split('_', 1)[0]+'_real'))
        expected_identity = {'plan_sha256': launch['draft_plan_sha256'],
                            'config_sha256': stage_module.object_sha(specification),
                            'runner_sha256': launch['stage_runner_sha256'],
                            'source_sha256': stage_module.object_sha(draft['source_review_sha256'])}
        require(record['trace_identity'] == expected_identity, 'Trace config/source binding differs')
        bound({'path': record['last_checkpoint'], 'sha256': record['last_checkpoint_sha256']})
        state = torch.load(record['last_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        require(state['step'] == record['steps'] and state['configuration'] == specification['config']
                and bool(state['ema_decoder']), 'Checkpoint is not the fixed terminal EMA/config')
        if record['warmstart']:
            warm = state['provenance']['warmstart']
            require((warm['checkpoint'], warm['sha256'], warm['step']) ==
                    (record['warmstart']['path'], record['warmstart']['sha256'], 6000),
                    'Checkpoint warmstart provenance differs')
        del state
        inputs.update({item['stage_receipt']: sha(item['stage_receipt']),
                       record['last_checkpoint']: record['last_checkpoint_sha256']})
        records[key] = record
    comparisons = {}
    for stage in ('real', 'render_adapt'):
        paths = [Path(records[f'{capacity}_{stage}']['audit_root'])/'stage_receipt.json'
                 for capacity in ('hplus', 'vit7b')]
        actual = stage_module.compare_stage_receipts(*paths)
        previous = Path(launch['output_root'])/f'{stage}_rng_comparison.json'
        require(actual['status'] == 'passed' and read(previous) == actual, 'Complete capacity trace/RNG comparison differs')
        inputs[str(previous)] = sha(previous)
        comparisons[stage] = actual
    return records, comparisons, inputs


def verify_rgb_files(receipt, metrics):
    rows = receipt['predictions']
    expected = {view['name'] for view in metrics['views']}
    require(len(rows) == 50 and len(expected) == 50 and {r['name'] for r in rows} == expected,
            'Require all 50 actual RGB PNG files without duplicates')
    result = {}
    for row in rows:
        actual = sha(row['rgb'])
        require(actual == row['rgb_sha256'], 'Actual RGB PNG differs from receipt')
        result[row['name']] = actual
    return result


def checked_result(path, plan, compare, *, endpoint=None):
    metrics, source = compare.read_completed(path)
    compare._validate(metrics)  # Existing 50/41, pooled-CM and scoring-support checks.
    require(metrics['official_evaluation_fingerprint'] == plan['official_evaluation_fingerprint'],
            'Official 50/41 fingerprint changed')
    receipt = read(source['receipt'])
    require(source['checkpoint_sha256'] == plan['inference_scene']['sha256'], 'Inference field changed')
    ensemble = source['teacher_ensemble']
    require(ensemble['teacher_weight'] == .5 and ensemble['inference'] ==
            {k: plan['inference'][k] for k in ('tile_size', 'stride', 'flip', 'context_weight', 'context_short_side')},
            'Fixed teacher inference protocol changed')
    if endpoint:
        require(ensemble['teacher_checkpoint'] == endpoint['last_checkpoint']
                and ensemble['teacher_checkpoint_sha256'] == endpoint['last_checkpoint_sha256'],
                'Evaluation used a different teacher endpoint')
        expected_root = Path(plan['source_snapshot'])
        require(receipt['entrypoint_source']['path'] == str(expected_root/'evaluate_official.py'), 'Wrong scoring entrypoint')
        for item in [receipt['entrypoint_source'], *receipt['loaded_source_modules'].values()]:
            path = Path(item['path'])
            require(path.is_relative_to(expected_root), 'Evaluation imported non-frozen source')
            require(item['sha256'] == plan['source_files_sha256'][str(path.relative_to(expected_root))],
                    'Evaluation source SHA differs')
    return metrics, source, verify_rgb_files(receipt, metrics)


def execute(path):
    path = Path(path).resolve()
    output = path.parent/'execution'
    require(not output.exists(), 'Execution already exists; preserve failed artifacts, no retry')
    output.mkdir()
    receipt_path = output/'execution_receipt.json'
    receipt = {'status': 'running', 'started_utc': datetime.now(UTC).isoformat(),
               'plan': str(path), 'plan_sha256': sha(path), 'runner': str(Path(__file__).resolve()),
               'runner_sha256': sha(__file__), 'commands': [], 'scoring_reimplemented': False}
    write(receipt_path, receipt)
    started = time.monotonic()
    try:
        plan, launch = static_plan(path)
        stage, compare, draft_helper, imports = frozen_helpers(plan)
        records, traces, inputs = training_ready(plan, launch, stage)
        receipt.update(helper_imports=imports, training_inputs_sha256=inputs,
                       complete_trace_comparisons=traces,
                       endpoints={k: {'path': r['last_checkpoint'], 'sha256': r['last_checkpoint_sha256'],
                                      'steps': r['steps']} for k, r in records.items()})
        commands = plan['commands']
        for command in commands:
            require(not Path(command[command.index('--output')+1]).exists(), 'Evaluation output exists; no overwrite/retry')
        results = {'selected_reference': checked_result(Path(plan['selected_reference_metrics']['path']), plan, compare)}
        env = {**os.environ, **plan['environment']}
        for capacity, command in zip(('hplus', 'vit7b'), commands, strict=True):
            endpoint = records[f'{capacity}_render_adapt']
            require(command[command.index('--teacher-checkpoint')+1] == endpoint['last_checkpoint'], 'Command endpoint differs')
            xml = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=5)
            stage.verify_gpu_idle(xml)
            entry = {'capacity': capacity, 'command': command, 'status': 'running'}
            receipt['commands'].append(entry)
            write(receipt_path, receipt, update=True)
            with (output/f'{capacity}.log').open('x') as log:
                result = subprocess.run(command, cwd=launch['workspace_root'], env=env, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            entry.update(exit_code=result.returncode, status='completed' if result.returncode == 0 else 'failed')
            require(result.returncode == 0, f'{capacity} evaluation failed; no retry')
            metrics_path = Path(command[command.index('--output')+1])/'official_metrics.json'
            results[capacity] = checked_result(metrics_path, plan, compare, endpoint=endpoint)
        rgb = {key: value[2] for key, value in results.items()}
        require(rgb['hplus'] == rgb['vit7b'] == rgb['selected_reference'], 'Three sets of actual 50 RGB PNG SHA differ')
        pairs = {}
        for name, reference, candidate in (('vit7b_minus_hplus', 'hplus', 'vit7b'),
                                         ('vit7b_minus_selected_reference', 'selected_reference', 'vit7b'),
                                         ('hplus_minus_selected_reference', 'selected_reference', 'hplus')):
            value = compare.paired_official_comparison(results[reference][0], results[candidate][0], **plan['bootstrap'])
            value.update(reference_source=results[reference][1], candidate_source=results[candidate][1])
            write(output/f'{name}.json', value)
            pairs[name] = value
        gate = draft_helper.adoption_gate(pairs['vit7b_minus_hplus'], pairs['vit7b_minus_selected_reference'], True)
        write(output/'adoption_gate.json', gate)
        static_plan(path)
        for filename, expected in inputs.items():
            bound({'path': filename, 'sha256': expected})
        require(sha(__file__) == receipt['runner_sha256'], 'Execution runner changed')
        receipt.update(status='completed', adoption_gate=gate, rgb_files_sha256=rgb,
                       result_sources={key: value[1] for key, value in results.items()},
                       report_files_sha256={p.name: sha(p) for p in output.glob('*.json') if p != receipt_path})
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-started, finished_utc=datetime.now(UTC).isoformat())
        write(receipt_path, receipt, update=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        print(json.dumps(execute(args.plan)))
    else:
        plan, _ = static_plan(args.plan)
        print(json.dumps({'status': 'static_check_only_no_GPU', 'plan_sha256': sha(args.plan),
                          'runner_sha256': sha(__file__), 'commands': plan['commands']}))


if __name__ == '__main__':
    main()
