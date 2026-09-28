"""One new 50-view score completing the existing layer-selection factorial."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
TRAINING = RUNS/'ibgs_factorial_completion_v1'
PRIOR = RUNS/'ibgs_layer_heads_evaluation_v1'
TRAINING_SHA = '8455dc63e6680f9c7828bda255c0f0a464cddf74c29d61789db0a501b5af7d19'
PRIOR_SHA = '5c744495787a55c48a11cce9e0afc27f8a21758a08013b82fe4454f61b955929'
ORIGINAL_SHA = '1ccdfed8f19c285c0d4b9ac6a6ffa4a5829db475b9803033dcba5e45def43c8c'
ORIGINAL_NAME = 'three_head_evaluation_original.py'
BASE_NAME = 'single_head_evaluation_base.py'
ARM = 'median4_normalized'
FOUR_ARMS = ('median4_mass', 'top4_mass', ARM, 'top4_normalized')
RGB_KEYS = ('psnr', 'ssim', 'lpips')
ADAPTATIONS = (
    ("saved['protocol'] == 'ibgs_fixed_layer_heads_v1'",
     "saved['protocol'] == 'ibgs_fixed_layer_factorial_completion_v1'"),
    ('len(records) == 150 and', 'len(records) == 50*len(ARMS) and'),
    ('All 150 unique predictions must precede target RGB access',
     'All fixed unique predictions must precede target RGB access'),
    ("report['target_calls'] == report['selector_calls'] == 150",
     "report['target_calls'] == report['selector_calls'] == 50*len(ARMS)"),
    ("'all_150_pngs_before_GT'", "'all_50_pngs_before_GT'"),
    (("references = {'median4_mass': descriptors['median4_mass'], 'top4_normalized': descriptors['top4_normalized'],\n"
      "                      **plan['reference_metrics']}"), "references = plan['reference_metrics']"),
    ("report['lpips_calls'] == 150", "report['lpips_calls'] == 50*len(ARMS)"),
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
    require(hashlib.sha256(content.encode()).hexdigest() == ORIGINAL_SHA, 'Exact completed three-arm evaluator required')
    for old, new in ADAPTATIONS:
        require(content.count(old) == 1, 'Unambiguous protocol/count/reference adaptation required')
        content = content.replace(old, new)
    return content


def profile(original):
    spec = copy.deepcopy(original)
    spec.update(protocol='ibgs_layer_factorial_evaluation_v1', arms=[ARM], primary_candidate=ARM,
                target_calls=50, selector_calls=50, native_arrays=50, delivered_pngs=50, lpips_calls=50,
                render_seconds=300, render_external_seconds=360, score_seconds=200, score_external_seconds=240,
                comparisons=[f'{ARM}_minus_{r}' for r in ('median4_mass', 'top4_normalized', 'old_full_fused', 'E_rgb')],
                primary_mechanism_comparison='top4_normalized_minus_median4_normalized',
                interaction='(top4_normalized - median4_normalized) - (top4_mass - median4_mass)',
                sole_gate='median4_normalized minus E: PSNR >= .15 dB, paired lower > 0, SSIM >= 0, LPIPS <= 0',
                selection='fixed new 6000-step endpoint; bind all original endpoints; no automatic adoption',
                interpretation='post-three-arm completion of a 2x2 experiment on reused development views; '
                               'single seed, no semantic or novelty claim')
    return spec


def profiled_base(snapshot):
    require(sha(snapshot/ORIGINAL_NAME) == ORIGINAL_SHA
            and (snapshot/BASE_NAME).read_text() == adapted_source((snapshot/ORIGINAL_NAME).read_text()),
            'Frozen evaluator adaptation differs')
    base = load(snapshot/BASE_NAME, 'single_head_fixed_evaluation')
    base.SPEC = profile(base.SPEC); base.ARMS = {ARM: ('median4', 'normalized')}
    base.PRIMARY = ARM; base.TRAINING_PLAN_SHA = TRAINING_SHA
    return base


def factorial_interaction(metrics, *, repeats=5000, seed=20260926):
    """Paired four-arm difference of selection effects on the identical views."""
    require(set(metrics) == set(FOUR_ARMS), 'Four fixed cells required')
    rows = {a: {r['name']: r for r in metrics[a]['views']} for a in FOUR_ARMS}
    names = sorted(rows[ARM]); require(len(names) == 50, 'Exactly 50 unique factorial views required')
    reference = metrics[ARM]
    for a in FOUR_ARMS:
        require(len(metrics[a]['views']) == len(rows[a]) == 50 and sorted(rows[a]) == names
                and metrics[a]['inherited_common_reference_fingerprint'] == reference['inherited_common_reference_fingerprint']
                and metrics[a]['scoring_protocol'] == reference['scoring_protocol'], 'Factorial view population differs')
        require(all(all(rows[a][n][k] == rows[ARM][n][k] for k in
                        ('width', 'height', 'rgb_pixels', 'source_rgb_sha256')) for n in names), 'Factorial GT/grid identity differs')
    sample = np.random.default_rng(seed).integers(0, 50, size=(repeats, 50)); results = {}
    for key in RGB_KEYS:
        values = {a: np.asarray([rows[a][n][key] for n in names], np.float64) for a in FOUR_ARMS}
        require(all(np.isfinite(v).all() for v in values.values()), 'Finite factorial scores required')
        mass = values['top4_mass']-values['median4_mass']
        normalized = values['top4_normalized']-values[ARM]
        interaction = normalized-mass
        results[key] = {'selection_effect_mass': float(mass.mean()),
                        'selection_effect_normalized': float(normalized.mean()),
                        'interaction': float(interaction.mean()),
                        'paired_view_bootstrap_95_interval': np.quantile(interaction[sample].mean(1), [.025, .975]).tolist()}
    return {'formula': '(top4_normalized - median4_normalized) - (top4_mass - median4_mass)',
            'views': names, 'bootstrap_repeats': repeats, 'seed': seed, 'metrics': results,
            'scope': 'post-result, one seed, reused development views; interaction alone is not quality or novelty evidence'}


def prepare(args):
    import torch
    require(sha(TRAINING/'plan.json') == TRAINING_SHA and sha(PRIOR/'plan.json') == PRIOR_SHA, 'Fixed training/evaluation lineage required')
    old_path = PRIOR/'source_snapshot/evaluate_ibgs_layer_heads.py'
    require(sha(old_path) == ORIGINAL_SHA, 'Original evaluator changed')
    old = load(old_path, 'original_completed_evaluation')
    trained = old.natural(TRAINING); old.natural(PRIOR, 'render'); prior_score = old.natural(PRIOR, 'score')
    parent, prior = read(TRAINING/'plan.json'), read(PRIOR/'plan.json')
    require(parent['phase'] == 'train' and trained['status'] == 'completed'
            and trained['completed_updates'] == trained['raster_calls'] == trained['selector_calls'] == 6000
            and trained['field_unchanged'] and len(trained['arms']) == 1, 'Complete fixed fourth arm required')
    arm = trained['arms'][0]
    require(arm['arm'] == ARM and arm['status'] == 'completed' and arm['updates'] == 6000
            and arm['parameter_count'] == 66095 and arm['initial_head_hashes'] == parent['expected_initial_head_hashes'],
            'Original initialization and full endpoint required')
    require(old.files(parent['source_snapshot']) == parent['source_hashes']
            and old.files(prior['source_snapshot']) == prior['source_hashes'], 'Frozen source changed')
    for relative, expected in prior['source_hashes'].items():
        if relative.startswith('render/'):
            require(parent['source_hashes'].get(relative.removeprefix('render/')) == expected,
                    'Original renderer/training component changed')
    for key in ('checkpoint', 'data_contract', 'cache_manifest_sha256', 'selector', 'backend', 'interpreter'):
        require(parent[key] == prior[key], 'Original field/cache/backend contract differs')
    endpoint = arm['checkpoint']; require(sha(endpoint['path']) == endpoint['sha256'], 'Endpoint bytes changed')
    contract = read(parent['data_contract']['path']); train_names = {r['name'] for r in contract['train_rows']}
    views = prior['views']; require(len(train_names) == 350 and len(views) == 50
        and not train_names & {v['camera']['name'] for v in views}, 'Same disjoint TRAIN/VAL cameras required')
    inputs = {**prior['input_hashes'], **parent['input_hashes'], **parent['runtime_sources'], endpoint['path']: endpoint['sha256']}
    for folder, items in ((TRAINING, ('plan.json', 'execution_receipt.json', 'launch_receipt.json')),
                          (PRIOR, ('plan.json', 'render_execution_receipt.json', 'render_launch_receipt.json',
                                   'score_execution_receipt.json', 'score_launch_receipt.json'))):
        inputs.update({str(folder/n): sha(folder/n) for n in items})
    references = {'median4_mass': prior_score['metrics']['median4_mass'],
                  'top4_normalized': prior_score['metrics']['top4_normalized'], **prior['reference_metrics']}
    factorial_references = prior_score['metrics']
    for desc in list(references.values())+list(factorial_references.values()):
        require(sha(desc['path']) == desc['sha256'], 'Prior metrics changed')
        require(read(desc['path'])['inherited_common_reference_fingerprint'] == prior['reference_fingerprint'],
                'Reference scoring population differs')
        inputs[desc['path']] = desc['sha256']
    require(not {v['source_image_path'] for v in views} & inputs.keys(), 'No VAL payload access in prepare')
    require(all(sha(p) == h for p, h in inputs.items()), 'Bound input changed')
    output = args.output.resolve(); require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    snapshot = output/'source_snapshot'; snapshot.mkdir(parents=True)
    old.copy_bound_sources(parent['source_snapshot'], snapshot/'render', parent['source_hashes'])
    scoring_hashes = {k.removeprefix('scoring/'): v for k, v in prior['source_hashes'].items() if k.startswith('scoring/')}
    old.copy_bound_sources(Path(prior['source_snapshot'])/'scoring', snapshot/'scoring', scoring_hashes)
    for name in ('rgb_scoring_helpers.py', 'ibgs_warm_evaluation_base.py'):
        shutil.copy2(Path(prior['source_snapshot'])/name, snapshot/name)
    shutil.copy2(old_path, snapshot/ORIGINAL_NAME)
    with (snapshot/BASE_NAME).open('x') as stream:
        stream.write(adapted_source(old_path.read_text()))
    for path in (Path(__file__), ROOT/'tests/test_ibgs_factorial_evaluation.py',
                 ROOT/'docs/ibgs_factorial_evaluation_protocol.md'):
        shutil.copy2(path, snapshot/path.name)
    base = profiled_base(snapshot)
    saved = torch.load(endpoint['path'], map_location='cpu', weights_only=False)
    base.validate_endpoint(saved, ARM, parent, TRAINING_SHA); del saved
    plan = {k: prior[k] for k in ('reference_fingerprint', 'scoring_protocol', 'scoring_numerics', 'main_runtime',
                                 'perceptual_weights', 'views')}
    plan.update({k: parent[k] for k in ('runtime_sources', 'interpreter', 'checkpoint', 'data_contract',
                                       'cache_manifest', 'cache_manifest_sha256', 'selector', 'backend')})
    require(not torch.cuda.is_initialized(), 'CPU-only preparation required')
    plan.update(specification=base.SPEC, output=str(output), source_snapshot=str(snapshot),
                source_hashes=base.files(snapshot), input_hashes=inputs, training=str(TRAINING),
                training_plan_sha256=TRAINING_SHA, training_specification=parent['specification'],
                endpoints={ARM: endpoint}, reference_metrics=references, factorial_reference_metrics=factorial_references,
                prior_evaluation_plan_sha256=PRIOR_SHA, prepare_VAL_payload_reads=0, prepare_CUDA_initialized=False,
                source_adaptations='endpoint protocol, single-arm counts/barrier and bound reference descriptors only')
    write(output/'plan.json', plan)
    print(json.dumps({'plan_sha256': sha(output/'plan.json'), 'output': str(output)}))


def add_factorial_results(base, plan, report):
    legacy = base.legacy(); _, helper = legacy.scoring_modules(plan['source_snapshot'])
    new = report['metrics'][ARM]
    descriptors = {**plan['factorial_reference_metrics'], ARM: new}
    require(all(sha(d['path']) == d['sha256'] for d in descriptors.values()), 'Factorial metric bytes changed')
    metrics = {a: read(d['path']) for a, d in descriptors.items()}
    interaction = factorial_interaction(metrics)
    interaction['metric_descriptors'] = descriptors
    path = Path(plan['output'])/'factorial_interaction.json'; write(path, interaction)
    report['factorial_interaction'] = {'path': str(path), 'sha256': sha(path)}
    pair = helper.paired_rgb(metrics[ARM], metrics['top4_normalized'])
    pair.update(reference_metrics_sha256=new['sha256'], candidate_metrics_sha256=descriptors['top4_normalized']['sha256'])
    path = Path(plan['output'])/'paired_top4_normalized_minus_median4_normalized.json'; write(path, pair)
    report['primary_mechanism_comparison'] = {'path': str(path), 'sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--prepare', action='store_true'); operation.add_argument('--render', action='store_true')
    operation.add_argument('--score', action='store_true'); parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', type=Path); parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        require(args.output, 'Fresh output required'); prepare(args); return
    require(args.expected_plan_sha256 and sha(args.plan) == args.expected_plan_sha256, 'Explicit plan SHA required')
    plan = read(args.plan); snapshot = Path(plan['source_snapshot'])
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Execute frozen evaluator wrapper')
    base = profiled_base(snapshot)
    require(plan['specification'] == base.SPEC and plan['training_plan_sha256'] == TRAINING_SHA
            and plan['prior_evaluation_plan_sha256'] == PRIOR_SHA, 'Fixed factorial evaluation profile required')
    original_score = base.score_stage
    def complete_score(plan, report):
        original_score(plan, report)
        add_factorial_results(base, plan, report)
    base.score_stage = complete_score
    base.main()


if __name__ == '__main__':
    main()
