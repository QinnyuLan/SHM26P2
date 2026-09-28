"""CPU contracts for fixed endpoints, shared RGB and prediction-before-GT scoring."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve()
SCRIPT = HERE.parent/'evaluate_matched_rgb_teacher_adaptation.py'
if not SCRIPT.is_file():
    SCRIPT = HERE.parents[1]/'scripts/evaluate_matched_rgb_teacher_adaptation.py'
spec = importlib.util.spec_from_file_location('matched_rgb_teacher_tested', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def gpu_xml(kind, name='displayed name'):
    return f'''<nvidia_smi_log><gpu id="gpu0"><processes><process_info>
    <pid>123</pid><type>{kind}</type><process_name>{name}</process_name>
    <used_memory>537 MiB</used_memory></process_info></processes></gpu></nvidia_smi_log>'''


def test_verified_rustdesk_desktop_compute_and_graphics_are_allowed():
    records = m.gpu_process_records(gpu_xml('C'), lambda pid: '/usr/share/rustdesk/rustdesk')
    m.enforce_gpu_process_policy(records)
    assert records == [{'gpu_id': 'gpu0', 'pid': 123, 'type': 'C', 'name': 'displayed name',
                        'used_memory': '537 MiB', 'resolved_executable': '/usr/share/rustdesk/rustdesk',
                        'allowed': True}]
    m.enforce_gpu_process_policy(m.gpu_process_records(gpu_xml('G'), lambda pid: None))


@pytest.mark.parametrize('kind,executable', [('C', '/usr/bin/python3'), ('C', None),
                                         ('C+G', '/usr/share/rustdesk/rustdesk'),
                                         ('unknown', '/usr/share/rustdesk/rustdesk')])
def test_compute_process_display_name_cannot_spoof_executable(kind, executable):
    records = m.gpu_process_records(gpu_xml(kind, '/usr/share/rustdesk/rustdesk'), lambda pid: executable)
    with pytest.raises(ValueError, match='Unapproved GPU compute'):
        m.enforce_gpu_process_policy(records)


def test_unavailable_gpu_inventory_fails_closed_and_empty_inventory_is_valid():
    with pytest.raises(ValueError, match='unavailable'):
        m.gpu_process_records('<nvidia_smi_log><gpu><processes>N/A</processes></gpu></nvidia_smi_log>')
    assert m.gpu_process_records('<nvidia_smi_log><gpu><processes/></gpu></nvidia_smi_log>') == []


def training_fixture(tmp_path):
    source = tmp_path/'source_snapshot'
    source.mkdir()
    (source/'unit.py').write_text('VALUE = 1\n')
    hashes = {'unit.py': m.sha(source/'unit.py')}
    plan = {'source_snapshot': str(source), 'source_hashes': hashes, 'input_hashes': {}}
    plan_path = tmp_path/'plan.json'
    write_json(plan_path, plan)
    stages = {}
    for arm in m.ARMS:
        folder = tmp_path/arm
        folder.mkdir()
        checkpoint, trace = folder/'last.pt', folder/'trace.jsonl'
        checkpoint.write_bytes(b'never deserialize in training_ready')
        trace.write_text('{"step":2000}\n')
        record = {'status': 'completed', 'arm': arm, 'steps': 2000, 'plan_sha256': m.sha(plan_path),
            'last_checkpoint': str(checkpoint), 'last_checkpoint_sha256': m.sha(checkpoint),
            'configuration': {'steps': 2000, 'evaluate_validation': False},
            'warmstart': {'path': str(tmp_path/'initial.pt'), 'sha256': m.INITIAL_SHA},
            'validation_pixel_reads': 0, 'inputs_sources_unchanged': True,
            'source_hashes': hashes, 'input_hashes': {},
            'trace': {'segment_path': str(trace), 'segment_sha256': m.sha(trace), 'last_step': 2000}}
        receipt = folder/'stage_receipt.json'
        write_json(receipt, record)
        stages[arm] = {'receipt': str(receipt), 'receipt_sha256': m.sha(receipt), 'exit_code': 0}
    completion_path = tmp_path/'training_execution_receipt.json'
    write_json(completion_path, {'status': 'completed', 'plan_sha256': m.sha(plan_path), 'stages': stages})
    write_json(tmp_path/'matched_trace_review.json', {
        'status': 'passed', 'steps': 2000,
        'checks': {'ordered_view_domain_targets_spatial_and_photo_rng_equal': True, 'terminal_rng_equal': True},
        'receipt_sha256': {s['receipt']: s['receipt_sha256'] for s in stages.values()}})
    return plan_path, completion_path


@pytest.mark.parametrize('failure', ['outer_running', 'second_exit', 'second_stage_running'])
def test_both_natural_completions_checked_before_any_checkpoint_hash(tmp_path, monkeypatch, failure):
    plan, completion = training_fixture(tmp_path)
    record = m.read(completion)
    if failure == 'outer_running':
        record['status'] = 'running'
    elif failure == 'second_exit':
        record['stages'][m.ARMS[1]]['exit_code'] = 1
    else:
        stage_path = Path(record['stages'][m.ARMS[1]]['receipt'])
        stage = m.read(stage_path)
        stage['status'] = 'running'
        write_json(stage_path, stage)
        record['stages'][m.ARMS[1]]['receipt_sha256'] = m.sha(stage_path)
    write_json(completion, record)
    original_sha = m.sha
    def no_checkpoint_read(path):
        assert Path(path).suffix != '.pt', 'An active checkpoint was opened'
        return original_sha(path)
    monkeypatch.setattr(m, 'sha', no_checkpoint_read)
    with pytest.raises(ValueError):
        m.training_ready(plan, completion)


def test_completed_training_binds_exact_last_and_frozen_source(tmp_path):
    plan, completion = training_fixture(tmp_path)
    records, bindings = m.training_ready(plan, completion)
    assert set(records) == set(m.ARMS)
    assert all(records[a]['last_checkpoint'] in bindings for a in m.ARMS)
    assert str(tmp_path/'matched_trace_review.json') in bindings
    (tmp_path/'source_snapshot/unit.py').write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='source snapshot'):
        m.training_ready(plan, completion)


@pytest.mark.parametrize('mutation', ['status', 'rng', 'missing_check', 'steps', 'receipt'])
def test_matched_trace_review_is_required_before_checkpoint_reads(tmp_path, monkeypatch, mutation):
    plan, completion = training_fixture(tmp_path)
    path = tmp_path/'matched_trace_review.json'
    review = m.read(path)
    if mutation == 'status':
        review['status'] = 'failed'
    elif mutation == 'rng':
        review['checks']['terminal_rng_equal'] = False
    elif mutation == 'missing_check':
        review['checks'].pop('terminal_rng_equal')
    elif mutation == 'steps':
        review['steps'] = 1999
    else:
        review['receipt_sha256'][next(iter(review['receipt_sha256']))] = 'stale receipt'
    write_json(path, review)
    original_sha = m.sha
    def no_checkpoint_read(path):
        assert Path(path).suffix != '.pt', 'Endpoint read before matched trace gate'
        return original_sha(path)
    monkeypatch.setattr(m, 'sha', no_checkpoint_read)
    with pytest.raises(ValueError, match='trace review'):
        m.training_ready(plan, completion)


def test_forbidden_gt_binding_is_rejected_before_payload_hash(tmp_path, monkeypatch):
    plan, completion = training_fixture(tmp_path)
    target = tmp_path/'official_target.json'
    data = m.read(plan)
    data['input_hashes'][str(target)] = 'must not open'
    write_json(plan, data)
    outer = m.read(completion)
    outer['plan_sha256'] = m.sha(plan)
    for item in outer['stages'].values():
        stage = m.read(item['receipt'])
        stage['plan_sha256'] = outer['plan_sha256']
        write_json(Path(item['receipt']), stage)
        item['receipt_sha256'] = m.sha(item['receipt'])
    write_json(completion, outer)
    path = tmp_path/'matched_trace_review.json'
    review = m.read(path)
    review['receipt_sha256'] = {s['receipt']: s['receipt_sha256'] for s in outer['stages'].values()}
    write_json(path, review)
    original_sha = m.sha
    def no_target_read(path):
        assert Path(path) != target, 'Official target opened during prepare'
        return original_sha(path)
    monkeypatch.setattr(m, 'sha', no_target_read)
    with pytest.raises(ValueError, match='target payloads'):
        m.training_ready(plan, completion, forbidden=[str(target)])


@pytest.mark.parametrize('key,value', [('last_checkpoint', 'best.pt'), ('validation_pixel_reads', 1),
                                     ('steps', 1999)])
def test_wrong_endpoint_or_validation_selection_rejected(tmp_path, key, value):
    plan, completion = training_fixture(tmp_path)
    outer = m.read(completion)
    item = outer['stages'][m.ARMS[0]]
    stage = m.read(item['receipt'])
    stage[key] = str(tmp_path/m.ARMS[0]/value) if key == 'last_checkpoint' else value
    write_json(Path(item['receipt']), stage)
    item['receipt_sha256'] = m.sha(item['receipt'])
    write_json(completion, outer)
    with pytest.raises(ValueError):
        m.training_ready(plan, completion)


def checkpoint_fixture():
    cfg = {'steps': 2000, 'channels': 192}
    backing = {'model_dir': 'model', 'config_sha256': 'config', 'weights_sha256': {'weights': 'sha'}, 'index_sha256': {}}
    state = {'step': 2000, 'configuration': cfg, 'pixel_protocol': {'id': m.LEGACY},
             'ema_decoder': {'weight': torch.ones(2)},
             'provenance': {'warmstart': {'sha256': m.INITIAL_SHA},
                 'manifest': '/source/selected.json', 'manifest_sha256': 'manifest',
                 'domain_schedule': {'real': 1000, 'rendered': 1000},
                 'class_names': ['background', 'deck', 'stay_cable', 'tower', 'foundation'],
                 'model_dir': 'model', 'model_config_sha256': 'config',
                 'model_weights_sha256': {'weights': 'sha'}, 'model_index_sha256': {}}}
    return state, {'configuration': cfg}, backing, {'path': '/source/selected.json', 'sha256': 'manifest'}


@pytest.mark.parametrize('mutation', ['step', 'profile', 'initial', 'schedule', 'ema', 'manifest', 'manifest_hash'])
def test_checkpoint_contract_rejects_changed_terminal_ema(mutation):
    state, endpoint, backing, manifest = checkpoint_fixture()
    m.checkpoint_contract(state, endpoint, backing, manifest)
    if mutation == 'step':
        state['step'] = 1500
    elif mutation == 'profile':
        state['pixel_protocol'] = {'id': 'colmap_corner_v2'}
    elif mutation == 'initial':
        state['provenance']['warmstart']['sha256'] = 'another teacher'
    elif mutation == 'schedule':
        state['provenance']['domain_schedule']['rendered'] = 999
    elif mutation == 'ema':
        state['ema_decoder']['weight'][0] = float('nan')
    elif mutation == 'manifest':
        state['provenance']['manifest'] = '/source/composite.json'
    else:
        state['provenance']['manifest_sha256'] = 'different bytes'
    with pytest.raises(ValueError):
        m.checkpoint_contract(state, endpoint, backing, manifest)


def test_domain_gate_percentage_point_units_and_strict_lower_bound():
    pair = {'metrics': {'miou_all': {'difference': .002, 'paired_view_bootstrap_95_interval': [.00001, .004]}}}
    assert all(m.domain_matching_clauses(pair).values())
    pair['metrics']['miou_all']['paired_view_bootstrap_95_interval'][0] = 0.
    assert not all(m.domain_matching_clauses(pair).values())
    pair['metrics']['miou_all']['difference'] = .001999
    assert not any(m.domain_matching_clauses(pair).values())
    json.dumps(m.domain_matching_clauses(pair), allow_nan=False)


def test_three_references_preserve_distinct_baselines():
    control, frozen, selected = object(), object(), object()
    assert m.comparison_references(control, frozen, selected) == [
        ('same_budget_control', control), ('frozen_composite_teacher', frozen), ('selected_complete_system', selected)]


def test_score_cannot_open_gt_before_all_unique_predictions(tmp_path, monkeypatch):
    views = [{'name': f'{i:03d}.png', 'source_annotation_path': str(tmp_path/'forbidden.json')} for i in range(50)]
    predictions = [{'arm': a, 'name': v['name']} for a in m.ARMS for v in views]
    predictions[-1] = dict(predictions[0])  # 100 records, but one missing endpoint.
    def forbidden(*args, **kwargs):
        pytest.fail('Read a payload before the complete prediction barrier')
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    with pytest.raises(ValueError, match='100 unique'):
        m.score_predictions({'views': views}, {'predictions': predictions, 'annotation_payload_reads': 0}, None, None)


def test_two_arms_score_own_fresh_masks_once_on_shared_rgb(tmp_path):
    truth = np.array([[0, 1, 2], [3, 4, 255]], np.uint8)
    views, rgb_rows, predictions, calls = [], [], [], []
    for i in range(50):
        name = f'{i:03d}.png'
        rgb = tmp_path/f'rgb_{name}'
        assert cv2.imwrite(str(rgb), np.full((2, 3, 3), i, np.uint8))
        annotation = tmp_path/f'{i:03d}.json'
        annotation.write_bytes(b'fixed original annotation')
        views.append({'name': name, 'camera': {'width': 3, 'height': 2},
                      'rgb': {'path': str(rgb), 'sha256': m.sha(rgb)},
                      'source_image_sha256': 'metadata only', 'source_annotation_path': str(annotation) if i < 41 else None,
                      'source_annotation_sha256': m.sha(annotation) if i < 41 else None,
                      'rasterized_mask_sha256': m.hashlib.sha256(truth.tobytes()).hexdigest() if i < 41 else None})
        rgb_rows.append({'name': name, 'width': 3, 'height': 2, 'rgb_pixels': 6, 'psnr': 30.+i, 'ssim': .9, 'lpips': .2})
        for arm in m.ARMS:
            mask = truth.copy()
            mask[mask == 255] = 0
            if arm == m.ARMS[0]:
                mask[0, 2] = 0
            path = tmp_path/arm/'mask'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(path), mask)
            predictions.append({'arm': arm, 'name': name, 'rgb': str(rgb), 'rgb_sha256': m.sha(rgb),
                                'mask': str(path), 'mask_sha256': m.sha(path)})
    rgb_metrics = tmp_path/'rgb.json'
    write_json(rgb_metrics, {'views': rgb_rows})
    def rasterize(payload, width, height):
        calls.append((payload, width, height))
        return truth.copy()
    def iou(cm):
        return np.diag(cm)/(cm.sum(0)+cm.sum(1)-np.diag(cm))
    official = SimpleNamespace(rasterize_official_annotation=rasterize, official_fingerprint=lambda records: 'same')
    compare = SimpleNamespace(_iou=iou, _validate=lambda value: None)
    plan = {'output': str(tmp_path), 'views': views, 'composite_metrics': str(rgb_metrics),
            'reference_fingerprint': 'same', 'evaluation_family': 'official_original_pixel_grid', 'scoring_protocol': {},
            'endpoints': {a: {} for a in m.ARMS}, 'rgb_provenance': {}}
    report = {'predictions': predictions, 'annotation_payload_reads': 0}
    result = m.score_predictions(plan, report, official, compare)
    assert len(calls) == report['annotation_payload_reads'] == 41
    assert result[m.ARMS[1]]['miou_all'] == 1
    assert result[m.ARMS[0]]['confusion_matrix'][2][0] == 41
    assert result[m.ARMS[1]]['confusion_matrix'][2][2] == 41
    assert result[m.ARMS[0]]['psnr'] == result[m.ARMS[1]]['psnr'] == 54.5
    assert all(row.get('semantic_ignore_pixels', 0) == 1 for row in result[m.ARMS[1]]['views'][:41])
    assert all('confusion_matrix' not in row for row in result[m.ARMS[1]]['views'][41:])
