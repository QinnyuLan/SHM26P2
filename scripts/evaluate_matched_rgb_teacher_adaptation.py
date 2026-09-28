"""Fixed official evaluation of two completed, matched 2k H+ adaptation endpoints.

Both teachers see the identical existing composite RGB, through the unchanged
legacy adapter. Preparation never reads an unfinished teacher checkpoint.
"""
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
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
PARENT = RUNS/'matched_rgb_teacher_adaptation_v1'
PREVIOUS = RUNS/'rgb_teacher_transfer_v1'
PREVIOUS_PLAN_SHA = '2df49d471db41b4c0323d6a9b9a955529a93ee6530352b0417a8e0580470bbe9'
PREVIOUS_RECEIPT_SHA = 'b205a3727e14d951104ef309fc0e8cb24c1d9704c430001b985d8057ff593dbc'
TRAINING_PLAN_SHA = 'ec4f827a1729251a0b3b407859ff59599a769498a8afedc1043addafed5b2d99'
FAILED_PLAN_SHA = '5a1f9446c2fb522fcd5e0085ccb8f31e4406880226d185ec67e277a8c184165e'
FAILED_RECEIPT_SHA = 'e2b67908c6135e7085cf8599506389394f31a6fdbc107e659688b4db4a3b5663'
ARMS = ('selected', 'composite')
INITIAL_SHA = '00f5b84ac9a56c39512c5b8e43f70110397923feea1a2b4c78bdffd1f5524bff'
LEGACY = 'legacy_mixed_v1'
HELPER_NAME = 'rgb_teacher_transfer_base.py'
HELPER_PATH = Path(__file__).with_name(HELPER_NAME)
if not HELPER_PATH.is_file():
    HELPER_PATH = PREVIOUS/'source_snapshot/evaluate_rgb_teacher_transfer.py'
helper_spec = importlib.util.spec_from_file_location('rgb_teacher_transfer_base', HELPER_PATH)
base = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(base)
require, read, sha, write, utc = base.require, base.read, base.sha, base.write, base.utc
source_files = base.source_files
INFERENCE = dict(base.INFERENCE)
SPEC = {
    'protocol': 'matched_2k_hplus_original_composite_readout_v1',
    'arms': list(ARMS), 'teacher_weight': 1., 'inference': INFERENCE,
    'steps_per_arm': 2000, 'endpoint': 'last EMA; never best or selected intermediate',
    'teacher_initial_checkpoint_sha256': INITIAL_SHA,
    'views_per_arm': 50, 'annotated_views': 41, 'masks_before_any_gt': 100,
    'rgb_input': 'identical actual rgb_fixed_ensemble_v1 composite PNG bytes in both arms',
    'input_adapter': base.SPEC['input_adapter'], 'output_adapter': base.SPEC['output_adapter'],
    'border': base.SPEC['border'], 'source_profiles_preserved': True,
    'renderer_calls': 0, 'training_steps': 0, 'bootstrap_repeats': 5000,
    'bootstrap_seed': 20260926,
    'comparisons': ['same_budget_control', 'frozen_composite_teacher', 'selected_complete_system'],
    'system_gate': dict(base.SPEC['gate']),
    'domain_matching_gate': {'miou_gain_at_least': .002, 'miou_ci_lower_strict': 0.},
    'internal_seconds': 270, 'external_seconds': 300, 'retries': 0,
    'scope': 'Matched rendered-domain engineering adaptation; pure 2D teacher, not native semantic GS or innovation',
    'timing': 'Existing composite PNG migration, two endpoint inference and scoring/I/O only; no scene renders or training',
    'gpu_process_policy': {
        'graphics_process_type': 'G',
        'allowed_desktop_compute_type': 'C',
        'allowed_desktop_compute_resolved_executable': '/usr/share/rustdesk/rustdesk',
        'identity': 'Resolve /proc/<pid>/exe; never trust the displayed process name',
        'other_compute_or_unknown_process_types': 'reject',
        'cost_scope': 'Desktop-resident GPU processes are allowed and recorded; inference timing is not an exclusive-GPU benchmark',
    },
}


def bound(path, expected):
    require(sha(path) == expected, f'Changed bound input: {path}')


def process_executable(pid):
    try:
        return str((Path('/proc')/str(pid)/'exe').resolve(strict=True))
    except OSError:
        return None


def gpu_process_records(xml, resolve_executable=process_executable):
    """Permit desktop graphics and one verified desktop service; never kill it."""
    root = ET.fromstring(xml)
    devices = root.findall('gpu')
    require(bool(devices), 'GPU process inventory unavailable')
    records = []
    for device in devices:
        processes = device.find('processes')
        require(processes is not None and not (processes.text or '').strip(),
                'GPU process inventory unavailable')
        entries = processes.findall('process_info')
        require(len(entries) == len(processes), 'Unrecognized GPU process inventory')
        for entry in entries:
            pid, kind = entry.findtext('pid', '').strip(), entry.findtext('type', '').strip()
            require(pid.isdecimal() and int(pid) > 0, 'Invalid GPU process identity')
            executable = resolve_executable(int(pid))
            allowed = kind == 'G' or (kind == 'C' and executable == '/usr/share/rustdesk/rustdesk')
            records.append({'gpu_id': device.get('id'), 'pid': int(pid), 'type': kind,
                            'name': entry.findtext('process_name'), 'used_memory': entry.findtext('used_memory'),
                            'resolved_executable': executable, 'allowed': bool(allowed)})
    return records


def enforce_gpu_process_policy(records):
    require(all(record['allowed'] for record in records), f'Unapproved GPU compute process: {records}')


def profile_id(state):
    profile = state.get('pixel_protocol', LEGACY)
    require(isinstance(profile, (dict, str)), 'Invalid teacher pixel profile')
    return profile['id'] if isinstance(profile, dict) else profile


def checkpoint_contract(state, endpoint, backing, manifest):
    require(state['step'] == 2000 and state['configuration'] == endpoint['configuration'],
            'Require exact fixed 2000-step checkpoint/config')
    require(profile_id(state) == LEGACY and bool(state.get('ema_decoder')),
            'Require legacy last EMA teacher')
    provenance = state['provenance']
    require(Path(provenance['manifest']).resolve() == Path(manifest['path']).resolve()
            and provenance['manifest_sha256'] == manifest['sha256'],
            'Endpoint is not bound to its declared training domain manifest')
    require(provenance['warmstart']['sha256'] == INITIAL_SHA,
            'Both teachers must warmstart from the same fixed H+ EMA')
    require(provenance['domain_schedule']['real'] == provenance['domain_schedule']['rendered'] == 1000,
            'Require the fixed exact 50/50 real/rendered training schedule')
    require(provenance['class_names'] == ['background', 'deck', 'stay_cable', 'tower', 'foundation'],
            'Teacher class order changed')
    require(provenance['model_dir'] == backing['model_dir']
            and provenance['model_config_sha256'] == backing['config_sha256']
            and provenance['model_weights_sha256'] == backing['weights_sha256']
            and provenance.get('model_index_sha256', {}) == backing['index_sha256'],
            'Teacher backbone differs')
    for value in state['ema_decoder'].values():
        require(torch.is_tensor(value) and torch.isfinite(value).all().item(), 'Non-finite EMA decoder')
    for value in state.get('ema_adapters', {}).values():
        require(torch.isfinite(value).all().item(), 'Non-finite EMA adapters')


def training_ready(training_plan_path, completion_path, forbidden=()):
    """Check both natural completions before even hashing either last.pt."""
    training_plan_path, completion_path = Path(training_plan_path).resolve(), Path(completion_path).resolve()
    plan, completed = read(training_plan_path), read(completion_path)
    require(completed.get('status') == 'completed' and completed['plan_sha256'] == sha(training_plan_path),
            'Both training processes must naturally complete before evaluation preparation')
    require(set(completed['stages']) == set(ARMS), 'Wrong matched training arms')
    endpoints, bindings = {}, {str(training_plan_path): sha(training_plan_path), str(completion_path): sha(completion_path)}
    for arm in ARMS:
        outer = completed['stages'][arm]
        expected_receipt = training_plan_path.parent/arm/'stage_receipt.json'
        require(outer['exit_code'] == 0 and Path(outer['receipt']).resolve() == expected_receipt,
                'Require natural exit0 for both fixed stages')
        bound(expected_receipt, outer['receipt_sha256'])
        stage = read(expected_receipt)
        require(stage['status'] == 'completed' and stage['arm'] == arm and stage['steps'] == 2000
                and stage['plan_sha256'] == sha(training_plan_path), 'Incomplete/wrong fixed training endpoint')
        require(Path(stage['last_checkpoint']).resolve() == training_plan_path.parent/arm/'last.pt',
                'Only the fixed last.pt endpoint is allowed')
        require(stage['warmstart']['sha256'] == INITIAL_SHA and stage['validation_pixel_reads'] == 0
                and stage['inputs_sources_unchanged'] is True,
                'Initial teacher or no-VAL training contract changed')
        cfg = stage['configuration']
        require(cfg['steps'] == 2000 and cfg['evaluate_validation'] is False,
                'Training must save fixed 2k last EMA without validation selection')
        endpoints[arm] = stage
        bindings[str(expected_receipt)] = outer['receipt_sha256']
    require(endpoints[ARMS[0]]['configuration'] == endpoints[ARMS[1]]['configuration'],
            'Two adaptation budgets/configurations differ')
    trace_path = training_plan_path.parent/'matched_trace_review.json'
    trace_review = read(trace_path)
    expected_checks = {'ordered_view_domain_targets_spatial_and_photo_rng_equal', 'terminal_rng_equal'}
    receipt_hashes = {str(Path(s['receipt']).resolve()): s['receipt_sha256'] for s in completed['stages'].values()}
    require(trace_review['status'] == 'passed' and trace_review['steps'] == 2000
            and set(trace_review['checks']) == expected_checks
            and all(value is True for value in trace_review['checks'].values())
            and trace_review['receipt_sha256'] == receipt_hashes,
            'Require the bound complete matched RNG/target/augmentation trace review')
    bindings[str(trace_path)] = sha(trace_path)
    # Every completion has now been checked. None of the following is reached
    # while another arm could still be writing its checkpoint.
    forbidden = {str(Path(p).resolve()) for p in forbidden}
    declared_inputs = set(plan['input_hashes']).union(*(set(s['input_hashes']) for s in endpoints.values()))
    require(not forbidden.intersection(str(Path(p).resolve()) for p in declared_inputs),
            'Official target payloads cannot be read during preparation')
    for path, digest in plan['input_hashes'].items():
        bound(path, digest)
        bindings[path] = digest
    snapshot = Path(plan['source_snapshot'])
    require({str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*.py')} == plan['source_hashes'],
            'Training source snapshot changed')
    for name, digest in plan['source_hashes'].items():
        bindings[str(snapshot/name)] = digest
    for stage in endpoints.values():
        require(stage['source_hashes'] == plan['source_hashes'], 'Stage training source differs')
        for path, digest in stage['input_hashes'].items():
            bound(path, digest)
            bindings[path] = digest
        bound(stage['last_checkpoint'], stage['last_checkpoint_sha256'])
        bindings[stage['last_checkpoint']] = stage['last_checkpoint_sha256']
        trace = stage['trace']
        require(trace['last_step'] == 2000, 'Incomplete fixed training trace')
        bound(trace['segment_path'], trace['segment_sha256'])
        bindings[trace['segment_path']] = trace['segment_sha256']
    return endpoints, bindings


def prepare(output, training_plan_path, completion_path):
    require(not torch.cuda.is_initialized(), 'Preparation is CPU only')
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite an experiment')
    bound(training_plan_path, TRAINING_PLAN_SHA)
    previous_plan_path, previous_receipt_path = PREVIOUS/'plan.json', PREVIOUS/'execution_receipt.json'
    bound(previous_plan_path, PREVIOUS_PLAN_SHA)
    bound(previous_receipt_path, PREVIOUS_RECEIPT_SHA)
    previous, receipt = read(previous_plan_path), read(previous_receipt_path)
    forbidden = {v[k] for v in previous['views'] for k in ('source_image_path', 'source_annotation_path') if v[k]}
    endpoints, bindings = training_ready(training_plan_path, completion_path, forbidden)
    training_plan = read(training_plan_path)
    manifests = {arm: {'path': training_plan['manifests'][arm],
                      'sha256': bindings[training_plan['manifests'][arm]]} for arm in ARMS}
    require(receipt['status'] == 'completed' and receipt['bound_inputs_and_sources_unchanged']
            and receipt['plan_sha256'] == PREVIOUS_PLAN_SHA, 'Prior source evaluation incomplete')
    require(receipt['predictions_finished_utc'] < receipt['source_scoring_started_utc'], 'Prior GT order differs')
    previous_snapshot = Path(previous['source_snapshot'])
    require(source_files(previous_snapshot) == previous['source_hashes'], 'Prior frozen evaluation source changed')
    require(sha(HELPER_PATH) == previous['source_hashes']['evaluate_rgb_teacher_transfer.py'],
            'Adapter must be the actual previous frozen adapter')

    def bind(path, expected=None):
        path = str(Path(path).resolve())
        require(path not in forbidden, 'Official target payloads cannot be read during preparation')
        digest = sha(path)
        require(expected is None or digest == expected, f'Changed input: {path}')
        require(path not in bindings or bindings[path] == digest, 'Conflicting input binding')
        bindings[path] = digest

    for path, digest in previous['input_hashes'].items():
        bind(path, digest)
    failed_plan_path, failed_receipt_path = PARENT/'evaluation/plan.json', PARENT/'evaluation/execution_receipt.json'
    bind(failed_plan_path, FAILED_PLAN_SHA)
    bind(failed_receipt_path, FAILED_RECEIPT_SHA)
    failed_plan, failed = read(failed_plan_path), read(failed_receipt_path)
    numeric_spec = {k: v for k, v in SPEC.items() if k != 'gpu_process_policy'}
    require(failed_plan['specification'] == numeric_spec and failed['plan_sha256'] == FAILED_PLAN_SHA
            and failed['status'] == 'failed' and failed['predictions'] == []
            and failed['annotation_payload_reads'] == failed['gt_rgb_payload_reads'] == 0
            and failed['peak_cuda_allocated_bytes'] == 0 and failed['bound_inputs_and_sources_unchanged'] is True,
            'Environment recovery must preserve the failed zero-prediction attempt and numerical protocol')
    frozen_metrics = PREVIOUS/'composite_rgb_candidate/official_metrics.json'
    bind(frozen_metrics, receipt['metrics_sha256']['composite_rgb_candidate'])
    frozen = read(frozen_metrics)
    require(frozen['official_evaluation_fingerprint'] == previous['reference_fingerprint'], 'Frozen teacher support differs')
    backing = previous['teacher']['teacher_modelscope_backbone']
    schedules = []
    for arm, endpoint in endpoints.items():
        state = torch.load(endpoint['last_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        checkpoint_contract(state, endpoint, backing, manifests[arm])
        schedules.append(state['provenance']['domain_schedule'])
        del state
    require(schedules[0] == schedules[1], 'Matched training domain schedules differ')
    rows = []
    for view in previous['views']:
        rgb = view['inputs']['composite_rgb_candidate']
        bind(rgb['path'], rgb['sha256'])
        rows.append({**{k: v for k, v in view.items() if k != 'inputs'}, 'rgb': rgb})
    require(len(rows) == 50 and sum(v['source_annotation_path'] is not None for v in rows) == 41,
            'Require the exact 50/41 official population')
    common_rgb_rows(read(previous['composite_metrics']), rows)
    forbidden = {v[k] for v in rows for k in ('source_image_path', 'source_annotation_path') if v[k]}
    require(not forbidden.intersection(bindings), 'Official target payloads cannot be read during preparation')
    for path in (previous_plan_path, previous_receipt_path, Path(__file__), HELPER_PATH,
                 ROOT/'tests/test_matched_rgb_teacher_adaptation.py', ROOT/'uv.lock'):
        bind(path)
    for name in previous['source_hashes']:
        bind(previous_snapshot/name, previous['source_hashes'][name])
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(previous_snapshot/'bridge_rgs', snapshot/'bridge_rgs',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(HELPER_PATH, snapshot/HELPER_NAME)
    for path in (Path(__file__), ROOT/'tests/test_matched_rgb_teacher_adaptation.py',
                 previous_snapshot/'compare_official_evaluations.py'):
        shutil.copy2(path, snapshot/path.name)
    plan = {'status': 'locked_pending_root_gpu_handoff', 'specification': SPEC, 'root': str(ROOT),
        'output': str(output), 'source_snapshot': str(snapshot), 'source_hashes': source_files(snapshot),
        'input_hashes': bindings, 'views': rows, 'endpoints': endpoints, 'backbone': backing,
        'training_manifests': manifests,
        'matched_trace_review': str(Path(training_plan_path).resolve().parent/'matched_trace_review.json'),
        'environment_recovery': {'failed_plan': str(failed_plan_path), 'failed_plan_sha256': FAILED_PLAN_SHA,
            'failed_receipt': str(failed_receipt_path), 'failed_receipt_sha256': FAILED_RECEIPT_SHA,
            'change': 'Only explicitly verified desktop GPU process policy; original failure remains failed',
            'same_numerical_specification': True},
        'training_plan': str(Path(training_plan_path).resolve()), 'training_completion': str(Path(completion_path).resolve()),
        'rgb_provenance': previous['input_declarations']['composite_rgb_candidate'],
        'selected_metrics': previous['selected_metrics'], 'composite_metrics': previous['composite_metrics'],
        'frozen_teacher_metrics': str(frozen_metrics), 'prior_rgb_gate': previous['prior_rgb_gate'],
        'reference_fingerprint': previous['reference_fingerprint'], 'scoring_protocol': previous['scoring_protocol'],
        'evaluation_family': previous['evaluation_family'], 'runtime_versions': previous['runtime_versions'],
        'external_timeout_seconds': 300, 'prepare_gt_payload_reads': 0, 'prepare_cuda_initialized': False,
        'env': {'PYTHONPATH': str(snapshot), 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '8',
                'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'}}
    write(output/'plan.json', plan)
    return output/'plan.json'


def prediction_barrier(predictions, views):
    expected = {(arm, view['name']) for arm in ARMS for view in views}
    require(len(views) == 50 and len(expected) == 100 and len(predictions) == 100
            and {(p['arm'], p['name']) for p in predictions} == expected,
            'All 100 unique masks must exist before any semantic GT')


def domain_matching_clauses(comparison):
    metric = comparison['metrics']['miou_all']
    return {'new_domain_miou_gain_at_least_0_20pp': bool(metric['difference'] >= .002),
            'new_domain_miou_paired_95_lower_positive':
                bool(metric['paired_view_bootstrap_95_interval'][0] > 0)}


def comparison_references(control, frozen, selected):
    return [('same_budget_control', control), ('frozen_composite_teacher', frozen),
            ('selected_complete_system', selected)]


def common_rgb_rows(metrics, views):
    rows = {row['name']: row for row in metrics['views']}
    require(len(metrics['views']) == len(rows) == 50 and set(rows) == {v['name'] for v in views},
            'Composite RGB scoring population differs')
    return {name: {k: row[k] for k in ('name', 'width', 'height', 'rgb_pixels', 'psnr', 'ssim', 'lpips')}
            for name, row in rows.items()}


def runtime_imports(plan):
    snapshot = Path(plan['source_snapshot'])
    records = {'rgb_teacher_transfer_base': {'path': str(HELPER_PATH.resolve()), 'sha256': sha(HELPER_PATH)}}
    for name, module in list(sys.modules.items()):
        if name == 'bridge_rgs' or name.startswith('bridge_rgs.') or name == 'compare_official_evaluations':
            path = Path(module.__file__).resolve()
            records[name] = {'path': str(path), 'sha256': sha(path)}
    for record in records.values():
        path = Path(record['path'])
        require(path.is_relative_to(snapshot)
                and record['sha256'] == plan['source_hashes'][str(path.relative_to(snapshot))],
                'Imported source escaped snapshot')
    return records


def verify(plan):
    snapshot = Path(plan['source_snapshot'])
    require(plan['specification'] == SPEC and source_files(snapshot) == plan['source_hashes'], 'Source/spec changed')
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Execute frozen runner only')
    for path, digest in plan['input_hashes'].items():
        bound(path, digest)
    require({key: importlib.metadata.version(key) for key in plan['runtime_versions']} == plan['runtime_versions'],
            'Runtime versions changed')


def score_predictions(plan, report, official, compare):
    # The only semantic target-reading phase; the preceding barrier is mandatory.
    prediction_barrier(report['predictions'], plan['views'])
    require(report['annotation_payload_reads'] == 0, 'Targets were read before prediction barrier')
    output = Path(plan['output'])
    report['source_scoring_started_utc'] = utc()
    rgb_rows = common_rgb_rows(read(plan['composite_metrics']), plan['views'])
    pred = {(p['arm'], p['name']): p for p in report['predictions']}
    rows = {arm: [] for arm in ARMS}
    records = []
    for view in plan['views']:
        camera, truth = view['camera'], None
        if view['source_annotation_path'] is not None:
            content = Path(view['source_annotation_path']).read_bytes()
            report['annotation_payload_reads'] += 1
            require(hashlib.sha256(content).hexdigest() == view['source_annotation_sha256'], 'Annotation changed')
            truth = official.rasterize_official_annotation(content, camera['width'], camera['height'])
            require(hashlib.sha256(truth.tobytes(order='C')).hexdigest() == view['rasterized_mask_sha256'],
                    'Original-grid rasterizer differs')
        records.append(base.fingerprint_record(view))
        for arm in ARMS:
            item = pred[arm, view['name']]
            bound(item['rgb'], view['rgb']['sha256'])
            bound(item['mask'], item['mask_sha256'])
            row = dict(rgb_rows[view['name']])
            if truth is not None:
                mask = cv2.imread(item['mask'], cv2.IMREAD_UNCHANGED)
                require(mask is not None and mask.dtype == np.uint8 and mask.shape == truth.shape
                        and (mask < 5).all(), 'Invalid delivered ID mask')
                keep = truth != 255
                cm = np.bincount(5*truth[keep].astype(np.int64)+mask[keep], minlength=25).reshape(5, 5)
                row.update(confusion_matrix=cm.tolist(), semantic_pixels=int(keep.sum()),
                           semantic_ignore_pixels=int((~keep).sum()))
            rows[arm].append(row)
    require(report['annotation_payload_reads'] == 41
            and official.official_fingerprint(records) == plan['reference_fingerprint'], 'Official population changed')
    metrics = {}
    for arm in ARMS:
        pooled = sum(np.asarray(r['confusion_matrix'], np.int64) for r in rows[arm] if 'confusion_matrix' in r)
        iou = compare._iou(pooled)
        value = dict(evaluation_family=plan['evaluation_family'],
            official_evaluation_fingerprint=plan['reference_fingerprint'], scoring_protocol=plan['scoring_protocol'],
            inference_protocol=SPEC, validation_views=50, semantic_validation_views=41, views=rows[arm],
            confusion_matrix=pooled.tolist(), iou=iou.tolist(), miou_all=float(np.nanmean(iou)),
            miou_foreground=float(np.nanmean(iou[1:])),
            **{k: float(np.mean([r[k] for r in rows[arm]])) for k in ('psnr', 'ssim', 'lpips')},
            rgb_scoring='Bound composite per-view scores reused only after PNG byte equality; no RGB GT reads or LPIPS',
            provenance={'teacher_endpoint': plan['endpoints'][arm], 'rgb_input': plan['rgb_provenance'],
                        'pure_teacher_weight': 1., 'uses_h3_semantic_readout': False})
        compare._validate(value)
        write(output/arm/'official_metrics.json', value)
        metrics[arm] = value
    return metrics


def write_comparisons(plan, plan_path, metrics, compare):
    output = Path(plan['output'])
    references = comparison_references(metrics[ARMS[0]], read(plan['frozen_teacher_metrics']),
                                      read(plan['selected_metrics']))
    reference_paths = {'same_budget_control': output/ARMS[0]/'official_metrics.json',
                       'frozen_composite_teacher': plan['frozen_teacher_metrics'],
                       'selected_complete_system': plan['selected_metrics']}
    comparisons = {}
    for role, reference in references:
        pair = compare.paired_official_comparison(reference, metrics[ARMS[1]], repeats=5000, seed=20260926)
        pair.update(reference_role=role, candidate_metrics_sha256=sha(output/ARMS[1]/'official_metrics.json'),
                    reference_metrics_sha256=sha(reference_paths[role]), plan_sha256=sha(plan_path))
        path = output/f'paired_minus_{role}.json'
        write(path, pair)
        comparisons[role] = {'path': str(path), 'sha256': sha(path)}
    system = base.adoption_clauses(read(plan['prior_rgb_gate']), read(comparisons['selected_complete_system']['path']))
    domain = domain_matching_clauses(read(comparisons['same_budget_control']['path']))
    gate = {'system_passed': bool(all(system.values())), 'system_clauses': system,
            'domain_matching_evidence_passed': bool(all(domain.values())), 'domain_matching_clauses': domain,
            'passed': bool(all(system.values()) and all(domain.values())), 'comparisons': comparisons,
            'prior_rgb_gate_sha256': sha(plan['prior_rgb_gate']), 'plan_sha256': sha(plan_path),
            'consequence': 'No automatic adoption; system and matched-domain evidence are separate engineering claims'}
    write(output/'system_gate.json', gate)
    return comparisons, gate


def execute(plan_path):
    plan_path = Path(plan_path).resolve()
    plan, started = read(plan_path), time.monotonic()
    output = Path(plan['output'])
    write(output/'execution_started.json', {'utc': utc(), 'pid': os.getpid(), 'plan_sha256': sha(plan_path)})
    report = {'status': 'failed', 'schema': SPEC['protocol'], 'plan_sha256': sha(plan_path), 'specification': SPEC,
              'started_utc': utc(), 'predictions': [], 'annotation_payload_reads': 0, 'gt_rgb_payload_reads': 0,
              'scene_renders': 0, 'optimizer_steps': 0, 'teacher_endpoints': plan['endpoints'],
              'rgb_provenance': plan['rgb_provenance']}
    old_handler = signal.getsignal(signal.SIGALRM)
    def deadline(*_):
        raise TimeoutError('Fixed 270s internal deadline; no retry')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(270)
    try:
        os.chdir(plan['root'])
        torch.set_num_threads(8)
        cv2.setNumThreads(8)
        verify(plan)
        sys.path.insert(0, plan['source_snapshot'])
        official = importlib.import_module('bridge_rgs.official_evaluate')
        teacher = importlib.import_module('bridge_rgs.teacher')
        compare = importlib.import_module('compare_official_evaluations')
        report['actual_imports'] = runtime_imports(plan)
        gpu = subprocess.run(['nvidia-smi', '-q', '-x'], check=True, capture_output=True,
                             text=True, timeout=5).stdout
        report['gpu_before'] = gpu_process_records(gpu)
        report['gpu_process_inventory_sha256'] = hashlib.sha256(gpu.encode()).hexdigest()
        report['nonexclusive_gpu_cost_scope'] = SPEC['gpu_process_policy']['cost_scope']
        enforce_gpu_process_policy(report['gpu_before'])
        backing = plan['backbone']
        model, adapter_options = None, None
        for arm in ARMS:
            endpoint = plan['endpoints'][arm]
            state = torch.load(endpoint['last_checkpoint'], map_location='cpu', weights_only=False)
            checkpoint_contract(state, endpoint, backing, plan['training_manifests'][arm])
            options = teacher.checkpoint_adapter_options(state['configuration'])
            if model is None:
                adapter_options = options
                model = teacher.load_teacher(backing['model_dir'], 5, state['configuration']['channels'],
                                             device='cuda', pixel_profile=LEGACY, **options)
                report['teacher_parameters'] = sum(p.numel() for p in model.parameters())
            require(options == adapter_options, 'Two endpoint architectures differ')
            model.decoder.load_state_dict(state['ema_decoder'], strict=True)
            teacher.load_checkpoint_adapters(model, state)
            model.eval().requires_grad_(False)
            del state
            (output/arm/'rgb').mkdir(parents=True)
            (output/arm/'mask').mkdir()
            for view in plan['views']:
                camera, inp = view['camera'], view['rgb']
                payload = Path(inp['path']).read_bytes()
                require(hashlib.sha256(payload).hexdigest() == inp['sha256'], 'Composite input changed')
                rgb = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
                require(rgb is not None and rgb.dtype == np.uint8
                        and rgb.shape == (camera['height'], camera['width'], 3), 'Invalid composite RGB')
                rgb = rgb[..., ::-1].copy()
                mx, my, back, boundary = base.adapter_maps(camera, official.distortion_render_grid)
                canvas = base.input_canvas(rgb, mx, my)
                probabilities, _ = teacher.predict_image(model, canvas, **INFERENCE)
                require(probabilities.shape == (5, *canvas.shape[:2]), 'Teacher canvas mismatch')
                mask = base.output_probabilities(probabilities, back).argmax(-1).astype(np.uint8)
                require(mask.shape == rgb.shape[:2], 'Original output grid mismatch')
                rp, mp = output/arm/'rgb'/view['name'], output/arm/'mask'/view['name']
                with rp.open('xb') as stream:
                    stream.write(payload)
                require(cv2.imwrite(str(mp), mask), 'Mask write failed')
                report['predictions'].append({'arm': arm, 'name': view['name'], 'rgb': str(rp), 'rgb_sha256': sha(rp),
                                             'mask': str(mp), 'mask_sha256': sha(mp), 'boundary': boundary})
        prediction_barrier(report['predictions'], plan['views'])
        report['predictions_finished_utc'] = utc()
        write(output/'predictions_receipt.json', {'status': 'all_100_masks_complete_before_semantic_gt',
            'predictions': report['predictions'], 'utc': report['predictions_finished_utc'],
            'plan_sha256': sha(plan_path), 'annotation_payload_reads': 0, 'gt_rgb_payload_reads': 0})
        metrics = score_predictions(plan, report, official, compare)
        comparisons, gate = write_comparisons(plan, plan_path, metrics, compare)
        report.update(status='completed', finished_utc=utc(), comparisons=comparisons,
            gate_sha256=sha(output/'system_gate.json'), gate_passed=gate['passed'],
            system_passed=gate['system_passed'], domain_matching_evidence_passed=gate['domain_matching_evidence_passed'],
            metrics_sha256={a: sha(output/a/'official_metrics.json') for a in ARMS},
            official_evaluation_fingerprint=plan['reference_fingerprint'], actual_imports=runtime_imports(plan))
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        report['bound_inputs_and_sources_unchanged'] = source_files(Path(plan['source_snapshot'])) == plan['source_hashes'] \
            and all(sha(path) == h for path, h in plan['input_hashes'].items())
        if not report['bound_inputs_and_sources_unchanged']:
            report['status'] = 'failed'
        report['elapsed_seconds_existing_png_teacher_transfer_and_scoring'] = time.monotonic()-started
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        write(output/'execution_receipt.json', report)
    require(report['status'] == 'completed', 'Input/source invariance failed')
    print(json.dumps({'status': report['status'], 'gate_passed': report['gate_passed'],
                      'receipt': str(output/'execution_receipt.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path)
    mode.add_argument('--execute', type=Path)
    parser.add_argument('--training-plan', type=Path, default=PARENT/'plan.json')
    parser.add_argument('--training-completion', type=Path, default=PARENT/'training_execution_receipt.json')
    args = parser.parse_args()
    if args.prepare:
        path = prepare(args.prepare, args.training_plan, args.training_completion)
        print(json.dumps({'plan': str(path), 'plan_sha256': sha(path), 'runner_sha256': sha(__file__)}))
    else:
        execute(args.execute)


if __name__ == '__main__':
    main()
