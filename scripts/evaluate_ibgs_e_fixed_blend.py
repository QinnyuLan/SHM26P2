"""One fixed delivered-PNG mean of E RGB and stable-AA IBGS full/fused RGB."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
E = RUNS/'rgb_fixed_ensemble_v1'
STABLE = RUNS/'ibgs_aa_stable_evaluation_v2'
HELPER_SHA = '041a40203529db626a42fca92098fde8b8bf3eb70dafae799b436396104bc36b'
E_PLAN_SHA = '77091ff41a50d1931f092e8ee2505662bc98120c74658a49f4f8fbce1879697f'
E_EXEC_SHA = '7acb6259f976873214cb4b09dd06ec6ea5f8804bb12627cd4288ca4136f74ca5'
STABLE_PLAN_SHA = '722f170b9b0c2e3bb433b96390f44181d479db5195da29df5eaceb97d56e4208'
STABLE_RENDER_SHA = '9b092fa6b06c917badddaecb4deaebdf94bfc795be6e640b7e4031bff1832c55'
STABLE_SCORE_SHA = '327fa656318690c4a2a735888d6c7346a628de0fae6a8394704e61dce6986d3c'
DOCUMENT = 'ibgs_e_fixed_blend_protocol.md'
TEST = 'test_ibgs_e_fixed_blend.py'
MEMBERS = ('E_rgb', 'stable_full_fused')
NUMERICS = {'cudnn_tf32': True, 'matmul_tf32': False, 'benchmark': False,
            'matmul_precision': 'highest'}
SPEC = {
    'protocol': 'ibgs_E_delivered_RGB_fixed_half_mean_v1', 'members': list(MEMBERS),
    'weights': [.5, .5], 'views': 50, 'primary_reference': 'E_rgb',
    'arithmetic': 'np.rint((E_uint8.astype(float32)+stable_uint8.astype(float32))*.5).astype(uint8)',
    'grid': 'same 50 original distorted-camera delivered PNGs; no second warp',
    'bootstrap_repeats': 5000, 'bootstrap_seed': 20260926,
    'gate': {'psnr_gain_at_least': .15, 'psnr_ci_lower_strict': 0.,
             'ssim_gain_at_least': 0., 'lpips_gain_at_most': 0.},
    'scene_renders': 0, 'optimizer_steps': 0, 'teacher_calls': 0,
    'mask_outputs': 0, 'annotation_reads': 0, 'target_RGB_reads': 50, 'lpips_calls': 50,
    'internal_seconds': 300, 'external_seconds': 360, 'retry': False,
    'selection': 'One fixed global .5; no lambda/member/non-AA search and no automatic adoption',
    'semantics': 'E semantic route retained conceptually; no semantic inference or reassessment here',
    'cost': 'Adds a third RGB field, fusion network and TRAIN source-photo bank to E RGB; '
            'cached combination/scoring time is not end-to-end system FPS',
    'scope': 'Engineering ensemble on reused development views; not a shared-geometry or novel method claim',
}


def helpers():
    path = Path(__file__).with_name('evaluate_fixed_rgb_ensemble.py')
    if Path(__file__).parent.name != 'source_snapshot':
        path = E/'source_snapshot/evaluate_fixed_rgb_ensemble.py'
    require(sha(path) == HELPER_SHA, 'Frozen RGB helper changed')
    spec = importlib.util.spec_from_file_location('fixed_rgb_blend_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(payload)


def natural(receipt, launch, plan_sha, execution_sha):
    require(receipt['status'] == launch['status'] == 'completed'
            and launch['exit_code'] == 0 and launch['natural_completion'] is True
            and receipt['plan_sha256'] == launch['plan_sha256'] == plan_sha
            and launch['execution_receipt_sha256'] == execution_sha,
            'Natural completed stable render/score with bound plan required')


def matching_views(e_plan, s_plan, e_predictions, s_predictions, e_metrics, s_metrics):
    """Only metadata: exact camera/grid, targets, fingerprint and delivered-byte lineage."""
    helper = helpers()
    require(e_plan['reference_fingerprint'] == s_plan['reference_fingerprint']
            == e_metrics['inherited_common_reference_fingerprint']
            == s_metrics['inherited_common_reference_fingerprint'], 'Fingerprint mismatch')
    require(e_plan['original_scoring_protocol'] == s_plan['scoring_protocol']
            == s_metrics['scoring_protocol'], 'Scoring protocol mismatch')
    erows, srows = helper.rgb_rows(e_metrics), helper.rgb_rows(s_metrics)
    names = sorted(erows)
    ep = {v['name']: v for v in e_predictions}
    selected = [v for v in s_predictions if (v['arm'], v['readout']) == ('full', 'fused')]
    sp = {v['name']: v for v in selected}
    ev = {v['name']: v for v in e_plan['views']}
    sv = {v['camera']['name']: v for v in s_plan['views']}
    require(len(e_predictions) == len(selected) == len(e_plan['views']) == len(s_plan['views']) == 50
            and all(sorted(x) == names for x in (srows, ep, sp, ev, sv)),
            'Exactly the same 50 unique cameras required')
    views = []
    for name in names:
        a, b = ev[name], sv[name]
        camera = a['camera']
        require(camera == b['camera'] and camera['name'] == name and Path(name).name == name,
                'Camera identity mismatch')
        require(all(a[k] == b[k] for k in ('source_image_path', 'source_image_sha256')),
                'Target metadata mismatch')
        for row, prediction, hash_key in ((erows[name], ep[name], 'rgb_sha256'),
                                           (srows[name], sp[name], 'sha256')):
            require(row['width'] == camera['width'] and row['height'] == camera['height']
                    and row['rgb_sha256'] == prediction[hash_key]
                    and row['source_rgb_sha256'] == a['source_image_sha256'],
                    'Metrics do not describe bound delivered RGB/target bytes')
        views.append({'name': name, 'camera': camera,
                      'members': {'E_rgb': {'path': ep[name]['rgb'], 'sha256': ep[name]['rgb_sha256']},
                                  'stable_full_fused': {'path': sp[name]['path'], 'sha256': sp[name]['sha256']}},
                      **{k: a[k] for k in ('source_image_path', 'source_image_sha256')}})
    return views


def completed_sources():
    inputs = {}

    def bound(path, expected=None):
        value = sha(path)
        require(expected is None or value == expected, 'Changed lineage file: '+str(path))
        inputs[str(path)] = value
        return read(path)

    ep = bound(E/'plan.json', E_PLAN_SHA)
    ee = bound(E/'execution_receipt.json', E_EXEC_SHA)
    ea = bound(E/'independent_cpu_review.json')
    eb = bound(E/'predictions_receipt.json')
    em = bound(E/'rgb_metrics.json', ee['metrics_sha256'])
    require(ee['status'] == 'completed' and ee['plan_sha256'] == E_PLAN_SHA
            and ee['bound_inputs_and_sources_unchanged'] is True
            and ea['status'] == 'passed' and ea['plan_sha256'] == E_PLAN_SHA
            and ea['execution_receipt_sha256'] == E_EXEC_SHA
            and eb['plan_sha256'] == E_PLAN_SHA and eb['gt_payload_reads'] == 0
            and eb['status'] == 'all_50_predictions_complete_before_gt'
            and eb['predictions'] == ee['predictions']
            and eb['utc'] <= ee['source_scoring_started_utc'], 'Completed audited E identity required')
    sp = bound(STABLE/'plan.json', STABLE_PLAN_SHA)
    sr = bound(STABLE/'render_execution_receipt.json', STABLE_RENDER_SHA)
    ss = bound(STABLE/'score_execution_receipt.json', STABLE_SCORE_SHA)
    for stage, receipt, expected in (('render', sr, STABLE_RENDER_SHA), ('score', ss, STABLE_SCORE_SHA)):
        natural(receipt, bound(STABLE/f'{stage}_launch_receipt.json'), STABLE_PLAN_SHA, expected)
        require(receipt['sources_and_inputs_unchanged'] is True, 'Stable source invariance failed')
    sb = bound(STABLE/'predictions_receipt.json')
    sa = bound(STABLE/'independent_rgb_statistics_review.json')
    sm_path = Path(ss['metrics']['full/fused']['path'])
    require(sm_path == STABLE/'metrics_full_fused.json', 'Only fixed full/fused member allowed')
    sm = bound(sm_path, ss['metrics']['full/fused']['sha256'])
    require(sb['status'] == 'all_200_pngs_before_GT' and sb['VAL_payload_reads'] == 0
            and sb['finished_utc'] <= ss['scoring_started_utc']
            and ss['actual_numerics'] == sp['scoring_numerics'] == NUMERICS
            and sa['status'] == 'passed' and sa['plan_sha256'] == STABLE_PLAN_SHA
            and sr['renderer_profile']['id'] == 'ibgs_centered_corner_v2_aa_factored64_near001_v2',
            'Stable full/fused scoring/provenance mismatch')
    views = matching_views(ep, sp, eb['predictions'], sb['records'], em, sm)
    for view in views:
        require(Path(view['members']['E_rgb']['path']) == E/'rgb'/view['name']
                and Path(view['members']['stable_full_fused']['path'])
                == STABLE/'predictions/full/fused'/view['name'], 'Unexpected member path')
        for item in view['members'].values():
            require(sha(item['path']) == item['sha256'], 'Changed member PNG')
            inputs[item['path']] = item['sha256']
    references = {'E_rgb': {'path': str(E/'rgb_metrics.json'), 'sha256': ee['metrics_sha256']},
                  'stable_full_fused': {'path': str(sm_path), 'sha256': sha(sm_path)}}
    return ep, sp, inputs, views, references


def prepare(output):
    require(not torch.cuda.is_initialized(), 'Preparation is CPU only')
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh /mnt/data output required')
    helper = helpers()
    ep, sp, inputs, views, references = completed_sources()
    source = E/'source_snapshot'
    require(helper.source_files(source) == ep['source_hashes'], 'Frozen E scoring source changed')
    require(sha(source/'bridge_rgs/official_evaluate.py') == helper.SCORER_SHA, 'Scorer changed')
    for path in [*map(Path, ep['perceptual_weights']), ROOT/'uv.lock', ROOT/'docs'/DOCUMENT,
                 ROOT/'tests'/TEST, Path(__file__).resolve(), source/'evaluate_fixed_rgb_ensemble.py']:
        inputs[str(path)] = sha(path)
    require(not ({v['source_image_path'] for v in views} & inputs.keys()), 'Prepare may not read GT bytes')
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(source/'bridge_rgs', snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in [source/'evaluate_fixed_rgb_ensemble.py', Path(__file__), ROOT/'tests'/TEST, ROOT/'docs'/DOCUMENT]:
        shutil.copy2(path, snapshot/path.name)
    plan = {'status': 'locked_pending_root_gpu_handoff', 'specification': SPEC, 'output': str(output),
            'source_snapshot': str(snapshot), 'source_hashes': helper.source_files(snapshot),
            'input_hashes': inputs, 'views': views, 'reference_metrics': references,
            'reference_fingerprint': ep['reference_fingerprint'], 'scoring_protocol': sp['scoring_protocol'],
            'scorer_sha256': helper.SCORER_SHA, 'helper_sha256': HELPER_SHA,
            'scoring_numerics': NUMERICS, 'runtime_versions': ep['runtime_versions'],
            'perceptual_weights': ep['perceptual_weights'], 'external_timeout_seconds': 360,
            'prepare_gt_payload_reads': 0, 'prepare_cuda_initialized': torch.cuda.is_initialized(),
            'lineage_note': 'E has completed execution, prediction barrier and independent review; '
                            'no historical launch receipt exists. Stable has render/score natural-exit receipts.',
            'fingerprint_scope': 'Inherited camera/GT metadata; only RGB target bytes verified after new 50 PNG barrier',
            'env': {**ep['env'], 'PYTHONPATH': str(snapshot)}}
    write(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    helper = helpers()
    snapshot = Path(plan['source_snapshot'])
    require(plan['specification'] == SPEC and plan['scoring_numerics'] == NUMERICS, 'Plan contract changed')
    require(Path(__file__).resolve() == snapshot/Path(__file__).name
            and helper.source_files(snapshot) == plan['source_hashes'], 'Run the frozen unchanged worker')
    require(all(sha(p) == h for p, h in plan['input_hashes'].items()), 'Bound inputs changed')
    require({k: importlib.metadata.version(k) for k in plan['runtime_versions']} == plan['runtime_versions'],
            'Scoring runtime changed')


def prediction_barrier(records, views):
    names = [v['name'] for v in views]
    require(len(names) == len(set(names)) == len(records) == 50
            and [v['name'] for v in records] == names, 'Complete unique 50 predictions required before GT')
    require(all(r['width'] == v['camera']['width'] and r['height'] == v['camera']['height']
                and sha(r['rgb']) == r['rgb_sha256'] for r, v in zip(records, views, strict=True)),
            'Incomplete or changed prediction bytes/grid')


def numeric_flags():
    return {'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'benchmark': torch.backends.cudnn.benchmark,
            'matmul_precision': torch.get_float32_matmul_precision()}


def set_numerics(flags):
    torch.set_float32_matmul_precision(flags['matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32 = flags['matmul_tf32']
    torch.backends.cudnn.allow_tf32 = flags['cudnn_tf32']
    torch.backends.cudnn.benchmark = flags['benchmark']


def execute(plan_path, expected_sha):
    require(sha(plan_path) == expected_sha, 'Explicit plan SHA mismatch')
    plan = read(plan_path); output = Path(plan['output']); helper = helpers()
    write(output/'execution_started.json', {'plan_sha256': expected_sha, 'utc': helper.utc(), 'pid': os.getpid()})
    started = time.monotonic(); old_flags = numeric_flags(); old_handler = signal.getsignal(signal.SIGALRM)
    report = {'status': 'failed', 'plan_sha256': expected_sha, 'specification': SPEC, 'started_utc': helper.utc(),
              'scene_renders': 0, 'optimizer_steps': 0, 'mask_outputs': 0, 'teacher_calls': 0,
              'gt_rgb_payload_reads': 0, 'annotation_payload_reads': 0, 'lpips_calls': 0, 'predictions': [],
              'semantic_status': 'E semantic route retained; not inferred or re-evaluated here'}

    def deadline(*_):
        raise TimeoutError('Fixed 300 second scoring deadline; no retry')

    signal.signal(signal.SIGALRM, deadline); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan)
        sys.path.insert(0, plan['source_snapshot'])
        official = importlib.import_module('bridge_rgs.official_evaluate')
        require(sha(official.__file__) == plan['scorer_sha256'], 'Wrong official scorer')
        torch.set_num_threads(8); cv2.setNumThreads(8)
        (output/'rgb').mkdir()
        for view in plan['views']:
            camera = view['camera']; images = []
            for role in MEMBERS:
                item = view['members'][role]; payload = Path(item['path']).read_bytes()
                require(hashlib.sha256(payload).hexdigest() == item['sha256'], 'Member PNG changed')
                images.append(helper.decode_prediction(payload, camera['width'], camera['height']))
            image = helper.average_png(*images)
            path = output/'rgb'/view['name']
            ok, encoded = cv2.imencode('.png', image[..., ::-1])
            require(ok, 'PNG encoding failed')
            with path.open('xb') as stream:
                stream.write(encoded.tobytes())
            report['predictions'].append({'name': view['name'], 'rgb': str(path), 'rgb_sha256': sha(path),
                                          'width': camera['width'], 'height': camera['height']})
        prediction_barrier(report['predictions'], plan['views'])
        report['predictions_finished_utc'] = helper.utc()
        write(output/'predictions_receipt.json', {'status': 'all_50_predictions_complete_before_gt',
              'plan_sha256': expected_sha, 'predictions': report['predictions'], 'gt_payload_reads': 0,
              'utc': report['predictions_finished_utc']})
        set_numerics(NUMERICS); require(numeric_flags() == NUMERICS, 'Scoring flags differ')
        report['actual_numerics'] = numeric_flags()
        torch.cuda.reset_peak_memory_stats()
        perceptual = official._lpips('cuda')
        report['source_scoring_started_utc'] = helper.utc(); rows = []
        for view, record in zip(plan['views'], report['predictions'], strict=True):
            payload = Path(view['source_image_path']).read_bytes(); report['gt_rgb_payload_reads'] += 1
            require(hashlib.sha256(payload).hexdigest() == view['source_image_sha256'], 'Target RGB changed')
            camera = view['camera']
            target = official._decode_rgb(payload, camera['width'], camera['height'], 'target RGB')
            predicted = Path(record['rgb']).read_bytes()
            require(hashlib.sha256(predicted).hexdigest() == record['rgb_sha256'], 'Prediction changed')
            image = helper.decode_prediction(predicted, camera['width'], camera['height'])
            values = helper.score_rgb_only(official.score_official_arrays, image, target, perceptual, 'cuda')
            report['lpips_calls'] += 1
            rows.append({'name': view['name'], 'width': camera['width'], 'height': camera['height'], **values,
                         'rgb_sha256': record['rgb_sha256'], 'source_rgb_sha256': view['source_image_sha256']})
        metrics = {'modality': 'RGB_only', 'evaluation_views': 50, 'views': rows,
                   **{k: float(np.mean([r[k] for r in rows])) for k in helper.RGB_KEYS},
                   'inherited_common_reference_fingerprint': plan['reference_fingerprint'],
                   'scoring_protocol': plan['scoring_protocol'], 'scorer_sha256': plan['scorer_sha256']}
        write(output/'rgb_metrics.json', metrics); pairs = {}
        for role, bound in plan['reference_metrics'].items():
            pair = helper.paired_rgb(read(bound['path']), metrics)
            pair.update(reference_metrics_sha256=bound['sha256'], candidate_metrics_sha256=sha(output/'rgb_metrics.json'))
            path = output/f'paired_minus_{role}.json'; write(path, pair)
            pairs[role] = {'path': str(path), 'sha256': sha(path)}
        clauses = helper.gate_clauses(read(pairs['E_rgb']['path'])['metrics'])
        gate = {'passed': all(clauses.values()), 'clauses': clauses, 'primary_reference': 'E_rgb',
                'primary_pair_sha256': pairs['E_rgb']['sha256'], 'plan_sha256': expected_sha,
                'automatic_adoption': False, 'semantic_reassessment': False}
        write(output/'rgb_gate.json', gate)
        report.update(status='completed', metrics_sha256=sha(output/'rgb_metrics.json'), comparisons=pairs,
                      gate_sha256=sha(output/'rgb_gate.json'), gate_passed=gate['passed'], finished_utc=helper.utc())
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
        set_numerics(old_flags); report['numeric_flags_restored'] = numeric_flags() == old_flags
        report['actual_imports'] = {}
        for name, module in list(sys.modules.items()):
            if name == 'bridge_rgs' or name.startswith('bridge_rgs.'):
                path = Path(module.__file__).resolve()
                report['actual_imports'][name] = {'path': str(path), 'sha256': sha(path)}
        snapshot = Path(plan['source_snapshot'])
        report['actual_imports_bound'] = all(Path(v['path']).is_relative_to(snapshot)
            and plan['source_hashes'].get(str(Path(v['path']).relative_to(snapshot))) == v['sha256']
            for v in report['actual_imports'].values())
        report['bound_inputs_and_sources_unchanged'] = helper.source_files(snapshot) == plan['source_hashes'] and all(
            sha(p) == h for p, h in plan['input_hashes'].items())
        if not all(report[k] for k in ('actual_imports_bound', 'bound_inputs_and_sources_unchanged', 'numeric_flags_restored')):
            report['status'] = 'failed'
        report['elapsed_seconds_cached_png_combination_and_scoring_only'] = time.monotonic()-started
        report['peak_cuda_allocated_bytes_scoring_only'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        write(output/'execution_receipt.json', report)
    require(report['status'] == 'completed', 'Final source/import/flag check failed')
    print(json.dumps({'status': report['status'], 'receipt': str(output/'execution_receipt.json'),
                      'sha256': sha(output/'execution_receipt.json'), 'gate_passed': report['gate_passed']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--prepare', type=Path)
    operation.add_argument('--execute', type=Path)
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        path = prepare(args.prepare)
        print(json.dumps({'plan': str(path), 'plan_sha256': sha(path), 'worker_sha256': sha(__file__)}))
    else:
        require(args.expected_plan_sha256 is not None, 'Explicit plan SHA required')
        execute(args.execute, args.expected_plan_sha256)


if __name__ == '__main__':
    main()
