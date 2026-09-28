"""Freeze and execute one bounded RGB-only MCMC reference (no automatic retries)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml
from run_official_pair import check_gpu_idle, digest, read_json, require, utc, write_receipt

ROOT = Path(__file__).resolve().parents[1]


def freeze(config_path, directory, smoke, smoke_audit):
    directory = directory.resolve()
    require(not directory.exists(), "Refuse existing experiment directory")
    config = yaml.safe_load(config_path.read_text())
    from bridge_rgs.mcmc_reference import validate_reference_training
    validate_reference_training(config)
    cuda_contract_path = Path('/mnt/data/SHM2026/runs/mcmc_cuda_contract_v1.json')
    cuda_contract = read_json(cuda_contract_path)
    require(cuda_contract['status'] == 'passed'
            and cuda_contract['source_sha256'] == digest(ROOT / 'src/bridge_rgs/mcmc_reference.py'),
            'Real CUDA contract does not cover this adapter source')
    if smoke:
        config.update(steps=610, save_every=0, log_every=100)
    else:
        require(smoke_audit is not None, "Full training requires a completed smoke audit")
        audit = read_json(smoke_audit)
        require(audit['status'] == 'passed', 'Smoke audit failed')
    config['output'] = str(directory)
    manifest_path = (ROOT / config['manifest']).resolve()
    manifest = read_json(manifest_path)
    views = [v for v in manifest['views'] if v['split'] == 'train']
    require(len(views) == 350, 'Expected fixed350 TRAIN views')
    directory.mkdir(parents=True)
    snapshot = directory / 'source_snapshot'
    shutil.copytree(ROOT / 'src/bridge_rgs', snapshot / 'bridge_rgs',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(__file__, directory / 'frozen_runner.py')
    shutil.copyfile(ROOT / 'docs/mcmc_reference_protocol.md', directory / 'protocol_at_lock.md')
    config_dest = directory / 'training_config.yaml'
    config_dest.write_text(yaml.safe_dump(config, sort_keys=False))
    sources = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob('*.py'))}
    if not smoke:
        require(sources == audit['source_hashes'], 'Full run sources differ from audited smoke')
    inputs = {manifest_path, (ROOT / manifest['init_points_path']).resolve(), ROOT / 'uv.lock',
              directory / 'protocol_at_lock.md', directory / 'frozen_runner.py',
              ROOT / 'scripts/run_official_pair.py', cuda_contract_path}
    # Bind TRAIN image/validity bytes without opening any semantic masks.
    inputs.update(Path(v[key]).resolve() for v in views for key in ('image_path', 'valid_path'))
    installed = ROOT / '.venv/lib/python3.11/site-packages/gsplat'
    inputs.update(installed / p for p in ('strategy/mcmc.py', 'strategy/ops.py',
                                         'strategy/base.py', 'relocation.py',
                                         'cuda/csrc/RelocationCUDA.cu'))
    if smoke_audit:
        inputs.add(smoke_audit.resolve())
    plan = {"created_utc": utc(), "config": config, "config_sha256": digest(config_dest),
                "source_snapshot": str(snapshot), "source_hashes": sources,
                "inputs": {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size}
                        for p in sorted(inputs)},
                "smoke": smoke, "time_limit_seconds": 180 if smoke else 3600,
                "automatic_retry": False, "validation_during_training": False,
                "mask_supervision": False, "native_evaluation_after_training": not smoke,
                "scope": 'Complete local MCMC engineering recipe; not single-factor or novelty evidence. '
                      'Smoke has a610-step LR horizon, not a prefix of the30k run.',
                "continue_to_semantic_gate": {"reference": 'ssim_fixed_corner_v2_rgb_full',
                    "psnr_mean_improvement_at_least_dB": .15,
                    "paired_psnr_95CI_lower_strictly_above": 0,
                    "ssim_point_estimate_not_lower": True, "lpips_point_estimate_not_higher": True}}
    with (directory / 'plan.json').open('x') as f:
        json.dump(plan, f, indent=2, allow_nan=False)
    print(json.dumps({'directory': str(directory), 'plan_sha256': digest(directory / 'plan.json')}))


def verify(directory, plan):
    require(digest(directory / 'training_config.yaml') == plan['config_sha256'], 'Config changed')
    snapshot = Path(plan['source_snapshot'])
    require({str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*.py')}
            == plan['source_hashes'], 'Source snapshot changed')
    for path, record in plan['inputs'].items():
        require(digest(path) == record['sha256'], 'Bound input changed: ' + path)


def launch(directory):
    directory = directory.resolve()
    receipt_path = directory / 'launch_receipt.json'
    require(not receipt_path.exists(), 'Refuse retry/overwrite')
    plan = read_json(directory / 'plan.json')
    verify(directory, plan)
    gpu = check_gpu_idle()
    command = [sys.executable, str(directory / 'frozen_runner.py'), 'worker', str(directory)]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / 'scripts'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    receipt = {"status": 'running', "started_utc": utc(), "plan_sha256": digest(directory / 'plan.json'),
                   "gpu_before": gpu, "command": command}
    write_receipt(receipt_path, receipt)
    started = time.perf_counter()
    with (directory / 'wrapper.log').open('x') as log:
        process = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=10s',
                                    str(plan['time_limit_seconds']) + 's', *command],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt['timeout_pid'] = process.pid
        write_receipt(receipt_path, receipt)
        code = process.wait()
    receipt.update(status='completed' if code == 0 else 'failed', exit_code=code,
                   finished_utc=utc(), elapsed_seconds=time.perf_counter() - started)
    write_receipt(receipt_path, receipt)
    require(code == 0, 'Bounded run failed; inspect receipt/log before any separate new plan')
    verify(directory, plan)
    print(json.dumps(receipt, indent=2))


def worker(directory):
    # Frozen runner is executed from the run folder; never infer workspace from __file__ here.
    directory = directory.resolve()
    plan = read_json(directory / 'plan.json')
    workspace = Path('/home/sky/workspace/SHM2026')
    env = dict(os.environ, PYTHONPATH=plan['source_snapshot'])
    commands = [[sys.executable, '-m', 'bridge_rgs.cli', 'train', '--config',
                 str(directory / 'training_config.yaml')]]
    if plan['native_evaluation_after_training']:
        commands.append([sys.executable, '-m', 'bridge_rgs.cli', 'evaluate', '--checkpoint',
                         str(directory / 'last.pt'), '--manifest', plan['config']['manifest'],
                         '--output', str(directory / 'evaluation_native'), '--lpips'])
    receipt_path = directory / 'experiment_receipt.json'
    require(not receipt_path.exists(), 'Worker receipt exists')
    receipt = {"status": 'running', "started_utc": utc(), "config": plan['config'], "resume": None,
                   "source_hashes": plan['source_hashes'], "commands": commands, "completed_commands": 0}
    write_receipt(receipt_path, receipt)
    try:
        with (directory / 'process.log').open('x') as log:
            for command in commands:
                subprocess.run(command, cwd=workspace, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
                receipt['completed_commands'] += 1
                write_receipt(receipt_path, receipt)
        receipt.update(status='completed', checkpoint_sha256=digest(directory / 'last.pt'))
        if plan['native_evaluation_after_training']:
            receipt['evaluation_sha256'] = digest(directory / 'evaluation_native/metrics.json')
    except Exception as error:
        receipt.update(status='failed', error=str(error))
        raise
    finally:
        receipt['finished_utc'] = utc()
        write_receipt(receipt_path, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'launch', 'worker'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/rgb_mcmc_reference_500k.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-audit', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.config, args.directory, args.smoke, args.smoke_audit)
    elif args.action == 'launch':
        launch(args.directory)
    else:
        worker(args.directory)


if __name__ == '__main__':
    main()
