"""Evaluate one audited H3 appearance endpoint, plain and fixed historical H+.

The teacher branch is the only adoption candidate. Plain is descriptive; no
VAL-driven branch selection, mixing old masks, or retry is implemented here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/home/sky/workspace/SHM2026')
TEACHER = ROOT/'runs/teacher_render_adapt_v1/best.pt'
TEACHER_SHA = '00f5b84ac9a56c39512c5b8e43f70110397923feea1a2b4c78bdffd1f5524bff'
REFERENCE_ROOT = Path('/mnt/data/SHM2026/runs/official_selected_ensemble_v1')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, replace=False):
    with Path(path).open('w' if replace else 'x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def evaluate(plan_path, audit_path):
    plan_path, audit_path = Path(plan_path).resolve(), Path(audit_path).resolve()
    plan, audit = read(plan_path), read(audit_path)
    run = Path(plan['output']).resolve()
    execution = read(run/'execution_receipt.json')
    require(audit['status'] == 'passed' and audit['plan_sha256'] == sha(plan_path)
            and execution['status'] == 'completed' and execution['accepted_steps'] > 0,
            'A naturally completed and independently audited changed endpoint is required')
    checkpoint = run/'final.pt'
    require(sha(checkpoint) == audit['candidate_checkpoint_sha256'] == execution['checkpoint_sha256'],
            'Candidate differs from its completed audit')
    require(sha(TEACHER) == TEACHER_SHA, 'Historical teacher changed')
    snapshot = Path(plan['source_snapshot']).resolve()
    for name, digest in plan['source_hashes'].items():
        require(sha(snapshot/name) == digest, 'Frozen training source changed')
    prep = run/'official_preparation'
    require(not prep.exists(), 'Preserve existing evaluation; no automatic retry')
    prep.mkdir()
    entry, compare = prep/'evaluate_official.py', prep/'compare_official_evaluations.py'
    shutil.copyfile(ROOT/'scripts/evaluate_official.py', entry)
    shutil.copyfile(ROOT/'scripts/compare_official_evaluations.py', compare)
    shutil.copyfile(__file__, prep/Path(__file__).name)
    references = {key: REFERENCE_ROOT/f'cross_{key}'/'official_metrics.json'
                  for key in ('plain', 'teacher')}
    paths = [plan_path, audit_path, run/'execution_receipt.json', checkpoint, TEACHER,
             entry, compare, prep/Path(__file__).name, ROOT/'uv.lock']
    paths += [snapshot/name for name in plan['source_hashes']]
    paths += [p for metric in references.values() for p in (metric, metric.parent/'execution_receipt.json')]
    environment = {'PYTHONPATH': str(snapshot), 'PYTHONDONTWRITEBYTECODE': '1',
                   'TORCH_CUDA_ARCH_LIST': '12.0', 'MAX_JOBS': '4', 'OMP_NUM_THREADS': '8',
                   'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'}
    commands = {}
    for key in references:
        commands[key] = ['uv', 'run', '--no-sync', 'python', str(entry), str(checkpoint),
                         '--output', str(run/f'official_{key}'), '--workspace-root', str(ROOT)]
        if key == 'teacher':
            commands[key] += ['--teacher-checkpoint', str(TEACHER)]
    frozen = {'created_utc': datetime.now(UTC).isoformat(), 'bound_inputs': {str(p): sha(p) for p in paths},
              'commands': commands, 'environment': environment, 'timeouts': {'plain': 180, 'teacher': 300},
              'reference_metrics': {k: str(p) for k, p in references.items()},
              'adoption_branch': 'teacher', 'bootstrap_repeats': 5000, 'bootstrap_seed': 20260926,
              'adoption_gate': {'psnr_min_gain_db': .15, 'psnr_paired_95_lower': '>0',
                                'ssim_min_gain': 0., 'lpips_max_gain': 0.,
                                'miou_all_min_gain': -.002, 'stay_cable_iou_min_gain': -.002},
              'automatic_retry': False,
              'scope': 'One fixed endpoint, fresh plain and historical H+0.5 masks. Same RGB required. '
                       'Fixed development views, not selection-adjusted, multi-seed, blind or exact peer Dev30.'}
    frozen_path = prep/'plan.json'
    write(frozen_path, frozen)
    launch = {'status': 'running', 'plan_sha256': sha(frozen_path), 'stages': {}}
    write(prep/'launch_receipt.json', launch)
    env = os.environ.copy()
    env.update(environment)
    for key, command in commands.items():
        idle = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                               '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
        require(not idle.strip(), 'GPU compute slot is occupied')
        seconds = frozen['timeouts'][key]
        started = time.perf_counter()
        with (prep/f'{key}.log').open('x') as log:
            process = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=10s', f'{seconds}s',
                                        *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            launch['stages'][key] = {'status': 'running', 'timeout_pid': process.pid}
            write(prep/'launch_receipt.json', launch, replace=True)
            code = process.wait()
        launch['stages'][key].update(status='completed' if code == 0 else 'failed', exit_code=code,
                                    elapsed_seconds=time.perf_counter()-started)
        launch['status'] = 'running' if code == 0 else 'failed'
        write(prep/'launch_receipt.json', launch, replace=True)
        require(code == 0, f'{key} evaluation failed; no retry')
    for path, digest in frozen['bound_inputs'].items():
        require(sha(path) == digest, 'Bound evaluation input changed: ' + path)
    sys.path.insert(0, str(snapshot))
    specification = importlib.util.spec_from_file_location('h3_frozen_comparison', compare)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    comparison_runtime_sources = {}
    for name, imported in list(sys.modules.items()):
        if name == 'bridge_rgs' or name.startswith('bridge_rgs.'):
            source = Path(imported.__file__).resolve()
            require(source.is_relative_to(snapshot), 'CPU comparison import escaped its source snapshot')
            digest = sha(source)
            require(plan['source_hashes'].get(str(source.relative_to(snapshot))) == digest,
                    'CPU comparison source changed')
            comparison_runtime_sources[name] = {'path': str(source), 'sha256': digest}
    comparisons = {}
    for key, path in references.items():
        reference, reference_source = module.read_completed(path)
        candidate, candidate_source = module.read_completed(run/f'official_{key}'/'official_metrics.json')
        official_receipt = read(run/f'official_{key}'/'execution_receipt.json')
        for name, item in official_receipt['loaded_source_modules'].items():
            source = snapshot/(name.replace('.', '/')+'.py')
            require(Path(item['path']).resolve() == source and item['sha256'] == sha(source),
                    'Official runtime import escaped the source snapshot: ' + name)
        result = module.paired_official_comparison(reference, candidate, repeats=5000, seed=20260926)
        result.update(reference_source=reference_source, candidate_source=candidate_source,
                      scope=f'Fixed {key} branch; camera-only predictions. Development-view paired bootstrap.')
        write(run/f'paired_official_{key}.json', result)
        comparisons[key] = result['metrics']
    first = {p.name: sha(p) for p in (run/'official_plain'/'rgb').glob('*.png')}
    second = {p.name: sha(p) for p in (run/'official_teacher'/'rgb').glob('*.png')}
    require(len(first) == 50 and first == second, 'Fresh plain/teacher RGB PNGs differ')
    m = comparisons['teacher']
    clauses = {'psnr_gain_at_least_0_15_dB': m['psnr']['difference'] >= .15,
               'psnr_paired_95_lower_positive': m['psnr']['paired_view_bootstrap_95_interval'][0] > 0,
               'ssim_point_not_lower': m['ssim']['difference'] >= 0,
               'lpips_point_not_higher': m['lpips']['difference'] <= 0,
               'miou_all_drop_at_most_0_20pp': m['miou_all']['difference'] >= -.002,
               'cable_drop_at_most_0_20pp': m['stay_cable_iou']['difference'] >= -.002}
    gate = {'passed': all(clauses.values()), 'clauses': clauses, 'metrics': m,
            'evaluation_plan_sha256': sha(frozen_path), 'fifty_rgb_pngs_identical': True,
            'comparison_runtime_sources': comparison_runtime_sources,
            'plain_branch_is_descriptive_only': True,
            'candidate_metrics_sha256': sha(run/'official_teacher'/'official_metrics.json')}
    write(run/'system_adoption_gate.json', gate)
    launch.update(status='completed', finished_utc=datetime.now(UTC).isoformat(),
                  bound_inputs_postchecked=True, system_adoption_gate_sha256=sha(run/'system_adoption_gate.json'))
    write(prep/'launch_receipt.json', launch, replace=True)
    return gate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--endpoint-audit', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = evaluate(args.plan, args.endpoint_audit)
    except BaseException as error:
        receipt_path = Path(read(args.plan)['output'])/'official_preparation'/'launch_receipt.json'
        if receipt_path.exists():
            failed = read(receipt_path)
            failed.update(status='failed', error=f'{type(error).__name__}: {error}',
                          finished_utc=datetime.now(UTC).isoformat())
            write(receipt_path, failed, replace=True)
        raise
    print(json.dumps(result, indent=2, allow_nan=False))
