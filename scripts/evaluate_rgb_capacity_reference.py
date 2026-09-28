"""One audited 1M endpoint, common original-grid RGB only against its 500k control.

Use the exact successful 500k official scoring package in a separate snapshot;
never add modules to or reinterpret the immutable 30-file training package.
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
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/home/sky/workspace/SHM2026')
RUN = Path('/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1')
TRAIN_PLAN_SHA = '40191687627d54da91b78c8b819e83bc04b04b986ae6568efe042c6dbcd447f3'
CONTROL = Path('/mnt/data/SHM2026/runs/ssim_fixed_corner_v2_rgb_full')
CONTROL_OFFICIAL_PLAN_SHA = '19e6e8825020691b2f478be342a6486c2f6cf4e9238626de277ef3535d597eac'
CONTROL_METRICS_SHA = '12bd34d0536da05f2a70a10f3b736312c679989c2a7fab3d42f84cae18d08ea8'
ENTRY_SHA = '04308dd8534e0150daeca38cc939b040e1de52c68c23c9aea2f71e72e0bc9d50'
COMPARE_SHA = 'bd02f80db9b80ee079d61fd19ae9f4d025bbbc26e650238bf407c13df2c2ccaa'
RGB_KEYS = ('psnr', 'ssim', 'lpips')
GATE = {'psnr_min_gain_db': .15, 'psnr_paired_95_lower_strict': 0., 'ssim_min_gain': 0., 'lpips_max_gain': 0.}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value, replace=False):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('w' if replace else 'x') as stream:
        stream.write(payload)


def completed_endpoint():
    require(sha(RUN/'plan.json') == TRAIN_PLAN_SHA, 'Wrong immutable capacity plan')
    plan, launch, train, native, audit = [read(RUN/name) for name in
                                        ('plan.json', 'launch_receipt.json', 'train_receipt.json',
                                         'native_receipt.json', 'cpu_endpoint_audit.json')]
    require(launch['status'] == train['status'] == native['status'] == 'completed' and launch['exit_code'] == 0
            and audit['status'] == 'passed', 'Natural train/native completion and independent endpoint pass required')
    require(all(value['plan_sha256'] == TRAIN_PLAN_SHA for value in (launch, train, native, audit)), 'Endpoint plan binding differs')
    checkpoint = Path(plan['training_output'])/'last.pt'
    require(sha(checkpoint) == launch['checkpoint_sha256'] == audit['checkpoint_sha256'], 'Completed checkpoint bytes differ')
    require(sha(Path(plan['training_output'])/'evaluation_native/metrics.json') == launch['native_metrics_sha256'], 'Native result changed')
    for name, digest in plan['source_hashes'].items():
        require(sha(Path(plan['source_snapshot'])/name) == digest, 'Training package changed')
    require(not (Path(plan['source_snapshot'])/'bridge_rgs/official_evaluate.py').exists(), 'Training source unexpectedly augmented')
    return plan, checkpoint


def scoring_sources():
    source_plan = CONTROL/'official_evaluation_plan.json'
    require(sha(source_plan) == CONTROL_OFFICIAL_PLAN_SHA, '500k official source plan changed')
    plan = read(source_plan)
    snapshot = Path(plan['source_snapshot'])
    for path, digest in plan['source_hashes'].items():
        require(sha(path) == digest, 'Known official scoring source changed')
    entry, compare = CONTROL/'evaluate_official_entrypoint.py', snapshot/'compare_official_evaluations.py'
    require(sha(entry) == ENTRY_SHA and sha(compare) == COMPARE_SHA
            and sha(CONTROL/'evaluation_official/official_metrics.json') == CONTROL_METRICS_SHA, 'Known official entry/comparison/control changed')
    return snapshot, entry, compare


def rgb_gate(metrics):
    require(set(metrics) == set(RGB_KEYS), 'Only RGB metrics may enter the capacity gate')
    return {'psnr_gain_at_least_0_15_dB': metrics['psnr']['difference'] >= .15,
            'psnr_paired_95_lower_positive': metrics['psnr']['paired_view_bootstrap_95_interval'][0] > 0,
            'ssim_point_not_lower': metrics['ssim']['difference'] >= 0,
            'lpips_point_not_higher': metrics['lpips']['difference'] <= 0}


def gpu_idle():
    value = subprocess.run(['nvidia-smi', '-q', '-x'], check=True, capture_output=True, text=True, timeout=5)
    state = ET.fromstring(value.stdout)
    require(state.findall('gpu'), 'GPU query missing')
    for gpu in state.findall('gpu'):
        listing = gpu.find('processes')
        require(listing is not None and (listing.text or '').strip() not in {'N/A', 'Not Supported'}, 'GPU query unavailable')
    require(all(p.findtext('type') == 'G' for p in state.findall('.//process_info')), 'GPU compute slot occupied')


def main():
    training_plan, checkpoint = completed_endpoint()  # No active checkpoint read before these guards.
    source, entry_source, compare_source = scoring_sources()
    gpu_idle()
    prep = RUN/'official_preparation'
    require(not prep.exists() and not (RUN/'evaluation_official').exists(), 'No official overwrite/retry')
    prep.mkdir()
    snapshot = prep/'source_snapshot'
    shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    entry = prep/'evaluate_official.py'
    shutil.copy2(entry_source, entry)
    shutil.copy2(__file__, prep/Path(__file__).name)
    sources = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*.py')}
    require(sources == {str(p.relative_to(source)): sha(p) for p in source.rglob('*.py')}, 'Scoring snapshot copy differs')
    compare = snapshot/compare_source.name
    reference = CONTROL/'evaluation_official/official_metrics.json'
    output = RUN/'evaluation_official'
    command = ['uv', 'run', '--no-sync', 'python', str(entry), str(checkpoint), '--output', str(output), '--workspace-root', str(ROOT)]
    environment = {'PYTHONPATH': str(snapshot), 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '8',
                   'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'TORCH_CUDA_ARCH_LIST': '12.0', 'MAX_JOBS': '4'}
    paths = [RUN/name for name in ('plan.json', 'launch_receipt.json', 'train_receipt.json', 'native_receipt.json', 'cpu_endpoint_audit.json')]
    paths += [checkpoint, entry, prep/Path(__file__).name, ROOT/'uv.lock', ROOT/'artifacts/prepared/manifest.json',
              reference, reference.parent/'execution_receipt.json', CONTROL/'official_evaluation_plan.json']
    paths += [snapshot/name for name in sources]
    paths += [Path(training_plan['source_snapshot'])/name for name in training_plan['source_hashes']]
    frozen = {'created_utc': datetime.now(UTC).isoformat(), 'training_plan_sha256': TRAIN_PLAN_SHA,
              'source_snapshot': str(snapshot), 'source_hashes': sources,
              'scoring_source_origin': str(source), 'scoring_source_control_plan_sha256': CONTROL_OFFICIAL_PLAN_SHA,
              'training_snapshot_kept_unchanged': training_plan['source_snapshot'],
              'source_difference': 'Separate official package used by the 500k control; shared renderer modules exact. train.load_scene uses full-compatible checkpoint loader.',
              'bound_inputs': {str(path): sha(path) for path in paths}, 'command': command, 'environment': environment,
              'time_limit_seconds': 240, 'automatic_retry': False, 'reference': str(reference),
              'bootstrap_repeats': 5000, 'bootstrap_seed': 20260926, 'continue_to_semantic_gate': GATE,
              'scope': 'RGB-only capacity comparison to fixed 500k control; untrained semantic output excluded. Greater compute, not innovation or a joint-system result.'}
    plan_path, receipt_path = prep/'plan.json', RUN/'official_launch_receipt.json'
    write(plan_path, frozen)
    receipt = {'status': 'running', 'plan_sha256': sha(plan_path), 'checkpoint_sha256': sha(checkpoint),
               'started_utc': datetime.now(UTC).isoformat(), 'scope': 'Subprocess only; comparison completion recorded separately'}
    write(receipt_path, receipt)
    try:
        started = time.monotonic()
        with (prep/'process.log').open('x') as log:
            process = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=10s', '240s', *command],
                                       cwd=ROOT, env=dict(os.environ, **environment), stdout=log, stderr=subprocess.STDOUT)
            receipt['timeout_pid'] = process.pid
            write(receipt_path, receipt, True)
            code = process.wait()
        receipt.update(status='completed' if code == 0 else 'failed', exit_code=code,
                       elapsed_seconds=time.monotonic()-started, finished_utc=datetime.now(UTC).isoformat())
        write(receipt_path, receipt, True)
        require(code == 0, 'Official evaluation failed, no retry')
        for path, digest in frozen['bound_inputs'].items():
            require(sha(path) == digest, 'Bound official input/source changed')
        sys.path.insert(0, str(snapshot))
        spec = importlib.util.spec_from_file_location('fixed_capacity_comparison', compare)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        candidate, candidate_source = module.read_completed(output/'official_metrics.json')
        baseline, reference_source = module.read_completed(reference)
        official = read(output/'execution_receipt.json')
        require(official['checkpoint_sha256'] == receipt['checkpoint_sha256'] and 'teacher_ensemble' not in official, 'Wrong candidate field/inference')
        for name, record in official['loaded_source_modules'].items():
            expected = snapshot/(name.replace('.', '/')+'.py')
            require(Path(record['path']).resolve() == expected and record['sha256'] == sha(expected), 'Official package import escaped snapshot')
        runtime = {}
        for name, value in list(sys.modules.items()):
            if name == 'bridge_rgs' or name.startswith('bridge_rgs.'):
                path = Path(value.__file__).resolve()
                require(path.is_relative_to(snapshot) and sources.get(str(path.relative_to(snapshot))) == sha(path), 'CPU comparison import escaped scoring package')
                runtime[name] = {'path': str(path), 'sha256': sha(path)}
        pair = module.paired_official_comparison(baseline, candidate, repeats=5000, seed=20260926)
        pair['metrics'] = {key: pair['metrics'][key] for key in RGB_KEYS}
        pair.update(reference_source=reference_source, candidate_source=candidate_source,
                    semantic_metrics_excluded='No semantic ranking: capacity-stage heads are untrained.',
                    scope='Fixed50 development-view RGB bootstrap; not equal compute, selection-adjusted, multi-seed or innovation evidence.')
        pair_path = RUN/'paired_official_rgb_minus_500k.json'
        write(pair_path, pair)
        clauses = rgb_gate(pair['metrics'])
        gate = {'passed': all(clauses.values()), 'clauses': clauses, 'metrics': pair['metrics'],
                'evaluation_plan_sha256': sha(plan_path), 'training_plan_sha256': TRAIN_PLAN_SHA,
                'candidate_metrics_sha256': sha(output/'official_metrics.json'), 'comparison_runtime_sources': runtime,
                'scope': 'Investment gate for a separate same-v2 8k semantic follow-up; no automatic follow-up or joint adoption.'}
        gate_path = RUN/'semantic_investment_gate.json'
        write(gate_path, gate)
        for path, digest in frozen['bound_inputs'].items():
            require(sha(path) == digest, 'Bound input/source changed during comparison')
        write(RUN/'official_comparison_receipt.json', {'status': 'completed', 'plan_sha256': sha(plan_path),
              'paired_report_sha256': sha(pair_path), 'gate_sha256': sha(gate_path), 'all_bound_inputs_postchecked': True,
              'finished_utc': datetime.now(UTC).isoformat()})
        print(json.dumps(gate, indent=2, allow_nan=False))
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(receipt_path, receipt, True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True, help='Only after root GPU handoff and passed endpoint audit')
    parser.parse_args()
    main()
