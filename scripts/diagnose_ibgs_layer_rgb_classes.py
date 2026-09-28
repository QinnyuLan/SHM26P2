"""CPU-only class RGB error diagnostic after the complete fixed 50-view score."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
E_RUN = RUNS/'multifield_h3_teacher_v1'
E_PLAN_SHA = '38052b356b3c970f2288cde49f688fcbc917b4a48d8ccdad35fe77aa6b6d7216'
E_EXECUTION_SHA = '83f445f013f27a9c33e42e39390c24b8a543ab48592d910ec5bf758705e42a53'
TRAINING_PLAN_SHA = 'b9a44c95b4e68b955f4fcc27abf27917134b4c5df8f456ed31c04a69b3a41c1a'
EVALUATOR_SHA = '1ccdfed8f19c285c0d4b9ac6a6ffa4a5829db475b9803033dcba5e45def43c8c'
OFFICIAL_SHA = '9b3799367a983495677122bf1bf75b06116342fde703466e43e3d8de092433f0'
FINGERPRINT = '21a2f19c5d4d703403a0107402d5dd5e006dcd98e5f8e023820c388c3e69a85d'
ARMS = ('median4_mass', 'top4_mass', 'top4_normalized', 'E_rgb')
SPEC = {
    'protocol': 'ibgs_layer_rgb_class_diagnostic_v1', 'primary': 'top4_mass',
    'arms': list(ARMS), 'evaluation_views': 50, 'annotated_views': 41,
    'omitted_unannotated_views': 9, 'selected_pngs': 164,
    'bootstrap_repeats': 5000, 'bootstrap_seed': 20260926,
    'class_labels': [0, 1, 2, 3, 4, 255],
    'class_names': ['background', 'deck', 'stay_cable', 'tower', 'foundation', 'ignore'],
    'aggregation': 'RGB-channel SSE pooled over class pixels; PSNR from pooled MSE',
    'uncertainty': 'paired whole-view bootstrap; all-empty class resamples omitted',
    'reconciliation_psnr_absolute_tolerance': 1e-10,
    'adoption_gate': None, 'multiple_comparison_correction': None,
    'semantic_predictions': 0, 'semantic_metrics': 0, 'render_calls': 0, 'teacher_calls': 0,
    'optimizer_steps': 0, 'model_loads': 0, 'lpips_calls': 0,
    'internal_seconds': 300, 'external_seconds': 360,
    'scope': 'descriptive reused development subset; GT cable is not Gaussian narrow-front stratum; '
             'no new semantic score, mechanism proof or replacement of the 50-view adoption gate',
}


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


def utc():
    return datetime.now(UTC).isoformat()


def source_hashes(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.md'}}


def natural(folder, stage):
    receipt_path = folder/f'{stage}_execution_receipt.json'
    receipt = read(receipt_path); launch = read(folder/f'{stage}_launch_receipt.json')
    require(receipt['status'] == launch['status'] == 'completed'
            and launch['exit_code'] == 0 and launch['natural_completion'] is True
            and receipt['plan_sha256'] == launch['plan_sha256'] == sha(folder/'plan.json')
            and launch['execution_receipt_sha256'] == sha(receipt_path),
            'Complete natural parent evaluation required')
    return receipt


def population(evaluation, annotations):
    """Match metadata only; do not treat the nine missing masks as background."""
    names = [v['camera']['name'] for v in evaluation]
    by_name = {v['name']: v for v in annotations}
    require(len(names) == len(set(names)) == len(annotations) == len(by_name) == 50
            and set(names) == set(by_name), 'Same fixed 50-view population required')
    selected, omitted = [], []
    for view in evaluation:
        name = view['camera']['name']; annotation = by_name[name]
        require(view['camera'] == annotation['camera']
                and view['source_image_path'] == annotation['source_image_path']
                and view['source_image_sha256'] == annotation['source_image_sha256'],
                'Camera/GT identity differs')
        fields = [annotation.get(k) for k in
                  ('source_annotation_path', 'source_annotation_sha256', 'rasterized_mask_sha256')]
        if all(x is None for x in fields):
            omitted.append(name)
        else:
            require(all(isinstance(x, str) and x for x in fields), 'Partial annotation identity')
            selected.append(annotation)
    require(len(selected) == 41 and len(omitted) == 9, 'Exactly 41 known and 9 missing masks')
    return selected, omitted


def prediction_barrier(records, names):
    expected = {(a, n) for a in ARMS for n in names}
    require(len(names) == len(set(names)) == 41 and len(records) == len(expected) == 164
            and {(r['arm'], r['name']) for r in records} == expected,
            'Exactly 164 selected RGB predictions before any GT payload')


def prepare(args):
    evaluation = args.evaluation.resolve(); parent = read(evaluation/'plan.json')
    require(parent['training_plan_sha256'] == TRAINING_PLAN_SHA
            and parent['reference_fingerprint'] == FINGERPRINT
            and parent['source_hashes']['evaluate_ibgs_layer_heads.py'] == EVALUATOR_SHA,
            'Fixed evaluator/training/population required')
    natural(evaluation, 'render'); completed = natural(evaluation, 'score')
    require(completed['VAL_rgb_reads'] == 50 and completed['lpips_calls'] == 150
            and completed['annotation_reads'] == 0, 'Complete original RGB scoring required')
    barrier = completed['predictions_receipt']
    require(sha(barrier['path']) == barrier['sha256'], 'Parent prediction receipt changed')
    delivered = read(barrier['path'])
    names = [v['camera']['name'] for v in parent['views']]
    require(delivered['status'] == 'all_150_pngs_before_GT'
            and delivered['plan_sha256'] == sha(evaluation/'plan.json')
            and len(delivered['records']) == 150
            and {(r['arm'], r['name']) for r in delivered['records']}
                == {(a, n) for a in ARMS[:3] for n in names}, 'Complete parent prediction barrier required')
    require(sha(E_RUN/'plan.json') == E_PLAN_SHA
            and sha(E_RUN/'execution_receipt.json') == E_EXECUTION_SHA, 'E lineage changed')
    e_launch = read(E_RUN/'launch_receipt.json')
    require(e_launch['status'] == 'completed' and e_launch['exit_code'] == 0
            and e_launch['natural_completion'] is True
            and e_launch['execution_receipt_sha256'] == E_EXECUTION_SHA, 'E completion required')
    views, omitted = population(parent['views'], read(E_RUN/'plan.json')['views'])
    chosen = {v['name'] for v in views}
    records = [r for r in delivered['records'] if r['name'] in chosen]
    records += [{'arm': 'E_rgb', 'name': v['name'], **v['composite_rgb']} for v in views]
    prediction_barrier(records, list(chosen))
    metrics = {**completed['metrics'], 'E_rgb': parent['reference_metrics']['E_rgb']}
    inputs = {str(evaluation/n): sha(evaluation/n) for n in
              ('plan.json', 'render_execution_receipt.json', 'render_launch_receipt.json',
               'score_execution_receipt.json', 'score_launch_receipt.json')}
    inputs.update({str(E_RUN/n): sha(E_RUN/n) for n in
                   ('plan.json', 'execution_receipt.json', 'launch_receipt.json')})
    inputs[barrier['path']] = barrier['sha256']
    for descriptor in metrics.values():
        require(sha(descriptor['path']) == descriptor['sha256'], 'Bound metrics changed')
        inputs[descriptor['path']] = descriptor['sha256']
    scoring = Path(parent['source_snapshot'])/'scoring'
    hashes = {k.removeprefix('scoring/'): v for k, v in parent['source_hashes'].items()
              if k.startswith('scoring/') and Path(k).suffix in {'.py', '.md'}}
    require(source_hashes(scoring) == hashes
            and sha(scoring/'bridge_rgs/official_evaluate.py') == OFFICIAL_SHA, 'Frozen scorer changed')
    output = args.output.resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    snapshot = output/'source_snapshot'; snapshot.mkdir(parents=True)
    for relative, expected in hashes.items():
        source = scoring/relative; destination = snapshot/'scoring'/relative
        require(source.resolve().is_relative_to(scoring.resolve()) and sha(source) == expected,
                'Invalid bound scorer path')
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
    for path in (Path(__file__), ROOT/'src/bridge_rgs/rgb_semantic_strata.py',
                 ROOT/'tests/test_rgb_semantic_strata.py', ROOT/'tests/test_ibgs_rgb_class_diagnostic.py',
                 ROOT/'docs/ibgs_rgb_class_diagnostic_protocol.md'):
        shutil.copy2(path, snapshot/path.name)
    write(output/'plan.json', {'specification': SPEC, 'source_snapshot': str(snapshot),
        'source_hashes': source_hashes(snapshot), 'output': str(output), 'input_hashes': inputs,
        'evaluation': str(evaluation), 'views': views, 'omitted_unannotated_names': omitted,
        'predictions': records, 'metrics': metrics, 'reference_fingerprint': FINGERPRINT,
        'prepared_utc': utc(), 'GT_payload_reads_during_preparation': 0})
    print(json.dumps({'plan_sha256': sha(output/'plan.json'), 'output': str(output)}))


def payload(descriptor):
    value = Path(descriptor['path']).read_bytes()
    require(hashlib.sha256(value).hexdigest() == descriptor['sha256'], 'Bound payload changed')
    return value


def execute(plan, report):
    snapshot = Path(plan['source_snapshot'])
    sys.path.insert(0, str(snapshot/'scoring'))
    from bridge_rgs import official_evaluate as official
    require(Path(official.__file__).resolve() == snapshot/'scoring/bridge_rgs/official_evaluate.py'
            and sha(official.__file__) == OFFICIAL_SHA, 'Actual frozen scorer import required')
    module_spec = importlib.util.spec_from_file_location('fixed_rgb_strata', snapshot/'rgb_semantic_strata.py')
    strata = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(strata)
    # Saved predictions are loaded and verified before any GT bytes are read.
    prediction_barrier(plan['predictions'], [v['name'] for v in plan['views']])
    pngs = {(r['arm'], r['name']): payload(r) for r in plan['predictions']}
    report.update(prediction_payload_reads=len(pngs), predictions_verified_utc=utc())
    write(Path(plan['output'])/'predictions_verified.json', {
        'plan_sha256': report['plan_sha256'], 'predictions': plan['predictions'],
        'verified_utc': report['predictions_verified_utc'], 'GT_payload_reads': 0})
    metrics = {a: {r['name']: r for r in read(d['path'])['views']} for a, d in plan['metrics'].items()}
    records, counts, sums = [], [], {a: [] for a in ARMS}
    max_error = 0.
    for view in plan['views']:
        name = view['name']; w, h = view['camera']['width'], view['camera']['height']
        target = official._decode_rgb(payload({'path': view['source_image_path'],
            'sha256': view['source_image_sha256']}), w, h, 'GT'); report['GT_rgb_reads'] += 1
        content = payload({'path': view['source_annotation_path'], 'sha256': view['source_annotation_sha256']})
        report['annotation_reads'] += 1
        mask = official.rasterize_official_annotation(content, w, h)
        require(hashlib.sha256(mask.tobytes(order='C')).hexdigest() == view['rasterized_mask_sha256'],
                'Original annotation raster differs')
        item = {'name': name, 'arms': {}}
        expected_count = None
        for arm in ARMS:
            encoded = pngs[arm, name]; old = metrics[arm][name]
            require(old['rgb_sha256'] == hashlib.sha256(encoded).hexdigest()
                    and old['source_rgb_sha256'] == view['source_image_sha256'], 'Metric pixel lineage differs')
            prediction = official._decode_rgb(encoded, w, h, arm); report['prediction_decodes'] += 1
            count, sse = strata.class_error_sums(prediction, target, mask)
            require(expected_count is None or np.array_equal(count, expected_count), 'Class population differs')
            expected_count = count
            full_psnr = -10*float(np.log10(max(float(sse.sum()/(3*w*h)), 1e-12)))
            difference = abs(full_psnr-old['psnr']); max_error = max(max_error, difference)
            require(difference <= SPEC['reconciliation_psnr_absolute_tolerance'], 'Class SSE does not reconcile')
            sums[arm].append(sse); item['arms'][arm] = {'sse_rgb_channels': sse.tolist(), 'full_psnr': full_psnr}
        counts.append(expected_count); item['class_pixel_counts'] = expected_count.tolist(); records.append(item)
    require(report['GT_rgb_reads'] == report['annotation_reads'] == 41
            and report['prediction_decodes'] == 164, 'Unexpected decode population')
    counts = np.asarray(counts); sums = {a: np.asarray(v) for a, v in sums.items()}
    result = {'specification': SPEC, 'plan_sha256': report['plan_sha256'], 'views': records,
        'omitted_unannotated_names': plan['omitted_unannotated_names'],
        'summary': {a: strata.class_summary(counts, v) for a, v in sums.items()},
        'comparisons': {f'top4_mass_minus_{a}': strata.paired_class_errors(counts, sums[a], sums['top4_mass'],
            repeats=SPEC['bootstrap_repeats'], seed=SPEC['bootstrap_seed']) for a in ARMS if a != 'top4_mass'},
        'maximum_full_psnr_reconciliation_error': max_error}
    path = Path(plan['output'])/'class_rgb_errors.json'; write(path, result)
    report.update(result={'path': str(path), 'sha256': sha(path)},
                  maximum_full_psnr_reconciliation_error=max_error,
                  actual_scoring_imports=official._loaded_sources())
    require(not official.torch.cuda.is_initialized(), 'CPU-only diagnostic required')


def verify(plan):
    require(plan['specification'] == SPEC and source_hashes(Path(plan['source_snapshot'])) == plan['source_hashes']
            and all(sha(p) == h for p, h in plan['input_hashes'].items()), 'Frozen metadata/source changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--evaluation', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', type=Path); parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        require(args.evaluation and args.output, 'Completed evaluation and fresh output required')
        prepare(args); return
    require(args.plan and args.expected_plan_sha256 and sha(args.plan) == args.expected_plan_sha256, 'Plan SHA required')
    plan = read(args.plan); output = Path(plan['output'])
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Execute frozen worker')
    write(output/'execution_started.json', {'plan_sha256': args.expected_plan_sha256, 'utc': utc()})
    report = {'status': 'failed', 'plan_sha256': args.expected_plan_sha256, 'GT_rgb_reads': 0,
              'annotation_reads': 0, 'prediction_decodes': 0, 'prediction_payload_reads': 0,
              **{k: 0 for k in ('semantic_predictions', 'semantic_metrics', 'render_calls',
                               'teacher_calls', 'optimizer_steps', 'model_loads', 'lpips_calls')}}
    started = time.monotonic(); previous = signal.getsignal(signal.SIGALRM)
    def deadline(*_):
        raise TimeoutError('Class diagnostic exceeded fixed budget')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan); execute(plan, report); verify(plan)
        report.update(status='completed', sources_and_metadata_unchanged=True)
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        report.update(elapsed_seconds=time.monotonic()-started, finished_utc=utc())
        write(output/'execution_receipt.json', report)
    print(json.dumps({'status': report['status'], 'receipt_sha256': sha(output/'execution_receipt.json')}))


if __name__ == '__main__':
    main()
