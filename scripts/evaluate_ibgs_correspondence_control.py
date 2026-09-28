"""Evaluate a matched normalized feature/weight correspondence control on 50 views."""
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
TRAINING = RUNS/'ibgs_correspondence_matched_v1'
PRIOR = RUNS/'ibgs_layer_heads_evaluation_v1'
TRAINING_SHA = '0b200575d6a2389fdd13a83e689b668d1f1e715c75d9379d5132e173fc9288cf'
PRIOR_SHA = '5c744495787a55c48a11cce9e0afc27f8a21758a08013b82fe4454f61b955929'
ORIGINAL_SHA = '1ccdfed8f19c285c0d4b9ac6a6ffa4a5829db475b9803033dcba5e45def43c8c'
ORIGINAL_NAME = 'three_head_evaluation_original.py'
BASE_NAME = 'correspondence_evaluation_base.py'
ARM = 'top4_normalized_permuted'
RGB_KEYS = ('psnr', 'ssim', 'lpips')
ADAPTATIONS = (
    ("saved['protocol'] == 'ibgs_fixed_layer_heads_v1'",
     "saved['protocol'] == 'ibgs_normalized_correspondence_control_v1'"),
    ('len(records) == 150 and', 'len(records) == 50*len(ARMS) and'),
    ('All 150 unique predictions must precede target RGB access',
     'All fixed unique predictions must precede target RGB access'),
    ("report['target_calls'] == report['selector_calls'] == 150",
     "report['target_calls'] == report['selector_calls'] == 50*len(ARMS)"),
    ("'all_150_pngs_before_GT'", "'all_50_pngs_before_GT'"),
    (("references = {'median4_mass': descriptors['median4_mass'], 'top4_normalized': descriptors['top4_normalized'],\n"
      "                      **plan['reference_metrics']}"), "references = plan['reference_metrics']"),
    ("report['lpips_calls'] == 150", "report['lpips_calls'] == 50*len(ARMS)"),
    ("    inputs = (evidence['features'], evidence['target_weights'], evidence['support'],",
     ("    from bridge_rgs.ibgs_correspondence_control import permute_with_diagnostics\n"
      "    evidence['features'], permutation_stats = permute_with_diagnostics(\n"
      "        evidence['features'], evidence['target_weights'], evidence['support'],\n"
      "        diagnose=True, chunk_pixels=chunk_pixels)\n"
      "    inputs = (evidence['features'], evidence['target_weights'], evidence['support'],")),
    ("'source_names': list(names), 'absent_source_slots': 4-len(names), **feature_stats}",
     ("'source_names': list(names), 'absent_source_slots': 4-len(names),\n"
      "        'permutation_diagnostics': permutation_stats, **feature_stats}")),
    ('No semantic output or reassessment; current E remains unchanged',
     'No semantic output or reassessment; deployed F remains unchanged'),
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
    spec.update(protocol='ibgs_normalized_correspondence_evaluation_v1', arms=[ARM], primary_candidate=ARM,
                target_calls=50, selector_calls=50, native_arrays=50, delivered_pngs=50, lpips_calls=50,
                render_seconds=300, render_external_seconds=360, score_seconds=200, score_external_seconds=240,
                comparisons=[f'{ARM}_minus_{r}' for r in ('top4_normalized', 'E_rgb', 'F_rgb')],
                primary_mechanism_comparison='top4_normalized_minus_top4_normalized_permuted',
                sole_gate='Legacy E RGB engineering gate reported for context only; no automatic F replacement',
                selection='fixed 6000-step endpoint; correctly aligned reference reused, no search',
                interpretation='matched training and inference correspondence control after four-corner study; '
                               'one seed, reused development views, no new semantic or novelty claim')
    spec['control'] = {'diagnostic_views': 'all 50', 'same_function_as_training': True,
                      'primary_direction': 'correct correspondence minus matched permuted control',
                      'evidence_rule': 'report each RGB paired interval; no automatic discovery/novelty verdict'}
    return spec


def profiled_base(snapshot):
    require(sha(snapshot/ORIGINAL_NAME) == ORIGINAL_SHA
            and (snapshot/BASE_NAME).read_text() == adapted_source((snapshot/ORIGINAL_NAME).read_text()),
            'Frozen evaluator adaptation differs')
    base = load(snapshot/BASE_NAME, 'normalized_correspondence_evaluation')
    base.SPEC = profile(base.SPEC); base.ARMS = {ARM: ('top4', 'normalized')}
    base.PRIMARY = ARM; base.TRAINING_PLAN_SHA = TRAINING_SHA
    return base


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
            and trained['permutation_calls'] == 6000
            and trained['field_unchanged'] and len(trained['arms']) == 1, 'Complete matched correspondence control required')
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
    adopted = RUNS/'ibgs_joint_replacement_v2/adoption.json'
    adoption = read(adopted)
    require(adoption['status'] == 'adopted_as_engineering_export' and adoption['label'] == 'F'
            and all(sha(p) == h for p, h in adoption['input_hashes'].items()), 'Verified engineering F required')
    f_path = RUNS/'ibgs_top4_mcmc_replacement_v1/rgb_metrics.json'
    require(sha(f_path) == '5a95bc2b1359c3b138e3c35a3311c360767d950378280f2b4088d2260cba65f8',
            'Fixed F RGB reference changed')
    inputs[str(adopted)] = sha(adopted); inputs.update(adoption['input_hashes'])
    references = {'top4_normalized': prior_score['metrics']['top4_normalized'],
                  'E_rgb': prior['reference_metrics']['E_rgb'],
                  'F_rgb': {'path': str(f_path), 'sha256': sha(f_path)}}
    for desc in references.values():
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
    for path in (Path(__file__), ROOT/'tests/test_ibgs_correspondence_evaluation.py',
                 ROOT/'docs/ibgs_correspondence_evaluation_protocol.md'):
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
                endpoints={ARM: endpoint}, reference_metrics=references,
                prior_evaluation_plan_sha256=PRIOR_SHA, prepare_VAL_payload_reads=0, prepare_CUDA_initialized=False,
                source_adaptations='endpoint protocol and counts; same TRAIN control before head; passive diagnostics; reference binding')
    write(output/'plan.json', plan)
    print(json.dumps({'plan_sha256': sha(output/'plan.json'), 'output': str(output)}))


def add_mechanism_result(base, plan, report):
    _, helper = base.legacy().scoring_modules(plan['source_snapshot'])
    control = report['metrics'][ARM]; correct = plan['reference_metrics']['top4_normalized']
    require(sha(control['path']) == control['sha256'] and sha(correct['path']) == correct['sha256'],
            'Primary comparison metric bytes changed')
    pair = helper.paired_rgb(read(control['path']), read(correct['path']))
    pair.update(reference_metrics_sha256=control['sha256'], candidate_metrics_sha256=correct['sha256'],
        hypothesis='Correct feature-to-wq association, both arms trained for 6000 steps',
        limitations='Does not separate q from w, establish novelty, or provide independent-scene/seed evidence')
    path = Path(plan['output'])/'paired_correct_minus_permuted.json'
    write(path, pair)
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
            and plan['prior_evaluation_plan_sha256'] == PRIOR_SHA, 'Fixed correspondence evaluation profile required')
    original_score = base.score_stage
    def complete_score(plan, report):
        original_score(plan, report)
        add_mechanism_result(base, plan, report)
    base.score_stage = complete_score
    base.main()


if __name__ == '__main__':
    main()
