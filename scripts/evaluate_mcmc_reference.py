"""One fixed endpoint evaluation; never reads an unfinished training checkpoint."""
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

RUN = Path('/mnt/data/SHM2026/runs/rgb_mcmc_reference_500k')
ROOT = Path('/home/sky/workspace/SHM2026')
SNAPSHOT = RUN / 'source_snapshot'
REFERENCES = {
    'mixed': Path('/mnt/data/SHM2026/runs/rgb140_inspired_mixed_500k/evaluation_official/official_metrics.json'),
    'ssim_fixed': Path('/mnt/data/SHM2026/runs/ssim_fixed_corner_v2_rgb_full/evaluation_official/official_metrics.json'),
    'selected': Path('/mnt/data/SHM2026/runs/official_selected_ensemble_v1/cross_teacher/official_metrics.json'),
}

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def save(path, value, exclusive=True):
    with path.open('x' if exclusive else 'w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def utc():
    return datetime.now(UTC).isoformat()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    launch = read(RUN / 'launch_receipt.json')
    training = read(RUN / 'experiment_receipt.json')
    audit = read(RUN / 'cpu_endpoint_audit.json')
    require(launch['status'] == 'completed' and launch['exit_code'] == 0
            and training['status'] == 'completed' and audit['status'] == 'passed',
            'Natural completion and passed CPU endpoint audit required')
    checkpoint = RUN / 'last.pt'
    checkpoint_sha = sha(checkpoint)
    require(checkpoint_sha == training['checkpoint_sha256'] == audit['checkpoint_sha256'],
            'Checkpoint differs from completed endpoint')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_official_pair import check_gpu_idle
    gpu = check_gpu_idle()
    plan = read(RUN / 'plan.json')
    require(launch['plan_sha256'] == sha(RUN / 'plan.json') == audit['plan_sha256'], 'Training and audit plan bindings differ')
    for relative, digest in plan['source_hashes'].items():
        require(sha(SNAPSHOT / relative) == digest, 'Frozen package changed: ' + relative)
    entry_source = Path('/mnt/data/SHM2026/runs/ssim_fixed_corner_v2_rgb_full/evaluate_official_entrypoint.py')
    require(sha(entry_source) == '04308dd8534e0150daeca38cc939b040e1de52c68c23c9aea2f71e72e0bc9d50', 'Entrypoint changed')
    entry = RUN / 'evaluate_official_entrypoint.py'
    compare_source = Path('/mnt/data/SHM2026/runs/ssim_fixed_appearance_replay_v1/source_snapshot/compare_official_evaluations.py')
    require(sha(compare_source) == 'bd02f80db9b80ee079d61fd19ae9f4d025bbbc26e650238bf407c13df2c2ccaa', 'Comparison source changed')
    compare = RUN / 'compare_official_evaluations.py'
    for dest, src in ((entry, entry_source), (compare, compare_source)):
        require(not dest.exists(), 'Refuse existing source copy')
        shutil.copyfile(src, dest)
    output = RUN / 'evaluation_official'
    require(not output.exists(), 'Refuse existing evaluation')
    command = ['uv', 'run', '--no-sync', 'python', str(entry), str(checkpoint), '--output', str(output), '--workspace-root', str(ROOT)]
    environment = {'PYTHONPATH': str(SNAPSHOT), 'PYTHONDONTWRITEBYTECODE': '1',
                   'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'}
    bindings = [SNAPSHOT / relative for relative in plan['source_hashes']]
    bindings += [checkpoint, RUN / 'cpu_endpoint_audit.json', RUN / 'plan.json',
                RUN / 'launch_receipt.json', RUN / 'experiment_receipt.json', Path(__file__), entry, compare,
                ROOT / 'scripts/run_official_pair.py', ROOT / 'artifacts/prepared/manifest.json', ROOT / 'uv.lock']
    for p in REFERENCES.values():
        bindings.extend((p, p.parent / 'execution_receipt.json'))
    frozen = {'created_utc': utc(), 'source_snapshot': str(SNAPSHOT),
              'source_hashes': plan['source_hashes'], 'bound_inputs': {str(p): sha(p) for p in bindings},
              'command': command, 'environment': environment, 'gpu_before': gpu,
              'time_limit_seconds': 180, 'automatic_retry': False,
              'references': {k: str(p) for k, p in REFERENCES.items()},
              'continue_to_semantic_gate': plan['continue_to_semantic_gate'],
              'scope': 'Fixed endpoint RGB only, original-grid50. All untrained semantic metrics excluded. Complete MCMC recipe compared to the stronger SSIM-fixed hybrid; mixed and selected are secondary systems. This is not a single-factor comparison. No academic contribution claim.',
              'bootstrap_repeats': 5000, 'bootstrap_seed': 20260926}
    frozen_path = RUN / 'official_evaluation_plan.json'
    save(frozen_path, frozen)
    receipt_path = RUN / 'official_launch_receipt.json'
    receipt = {'status': 'running', 'scope': 'Official evaluation subprocess only; paired comparison and gate completion recorded separately', 'plan_sha256': sha(frozen_path), 'checkpoint_sha256': checkpoint_sha,
               'endpoint_audit_sha256': sha(RUN / 'cpu_endpoint_audit.json'), 'started_utc': utc()}
    save(receipt_path, receipt)
    env = os.environ.copy()
    env.update(environment)
    started = time.perf_counter()
    with (RUN / 'official_process.log').open('x') as log:
        process = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=10s', '180s', *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt['timeout_pid'] = process.pid
        save(receipt_path, receipt, False)
        code = process.wait()
    receipt.update(status='completed' if code == 0 else 'failed', exit_code=code,
                   elapsed_seconds=time.perf_counter() - started, finished_utc=utc())
    save(receipt_path, receipt, False)
    require(code == 0, 'Official evaluation failed, no automatic retry')
    for path, digest in frozen['bound_inputs'].items():
        require(sha(Path(path)) == digest, 'Bound input changed during evaluation')
    sys.path.insert(0, str(SNAPSHOT))
    spec = importlib.util.spec_from_file_location('fixed_compare', compare)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate, candidate_source = module.read_completed(output / 'official_metrics.json')
    official_receipt = read(output / 'execution_receipt.json')
    for name, record in official_receipt['loaded_source_modules'].items():
        expected = SNAPSHOT / (name.replace('.', '/') + '.py')
        require(Path(record['path']).resolve() == expected and record['sha256'] == sha(expected),
                'Runtime module escaped candidate source snapshot: ' + name)
    comparisons = {}
    for label, path in REFERENCES.items():
        reference, reference_source = module.read_completed(path)
        result = module.paired_official_comparison(reference, candidate, repeats=5000, seed=20260926)
        result['metrics'] = {k: result['metrics'][k] for k in ('psnr', 'ssim', 'lpips')}
        result.update(reference_source=reference_source, candidate_source=candidate_source,
                      semantic_metrics_excluded='Candidate semantic heads were not trained; no semantic ranking',
                      scope='RGB-only fixed50 development-view bootstrap, conditional on endpoints; not selection-adjusted, multi-seed or blind test')
        save(RUN / f'paired_official_rgb_minus_{label}.json', result)
        comparisons[label] = result['metrics']
    strong = comparisons['ssim_fixed']
    clauses = {'psnr_gain_at_least_0_15_dB': strong['psnr']['difference'] >= .15,
               'psnr_paired_95_lower_positive': strong['psnr']['paired_view_bootstrap_95_interval'][0] > 0,
               'ssim_point_not_lower': strong['ssim']['difference'] >= 0,
               'lpips_point_not_higher': strong['lpips']['difference'] <= 0}
    gate = {'passed': all(clauses.values()), 'clauses': clauses,
            'frozen_training_plan_sha256': sha(RUN / 'plan.json'), 'comparisons': comparisons,
            'candidate_official_metrics_sha256': sha(output / 'official_metrics.json'),
            'scope': 'Further semantic investment gate only; no academic or combined RGB/semantic performance claim'}
    save(RUN / 'semantic_investment_gate.json', gate)
    save(RUN / 'official_comparison_receipt.json', {'status': 'completed', 'finished_utc': utc(), 'plan_sha256': sha(frozen_path), 'gate_sha256': sha(RUN / 'semantic_investment_gate.json'), 'all_bound_inputs_postchecked': True, 'source_file_count': len(plan['source_hashes']), 'paired_reports': {label: sha(RUN / f'paired_official_rgb_minus_{label}.json') for label in REFERENCES}})
    print(json.dumps(gate, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
