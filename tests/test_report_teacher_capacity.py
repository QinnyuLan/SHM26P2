"""Small provenance/direction contracts; synthetic files only, no GPU/metrics recomputation."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]/'scripts/report_teacher_capacity.py'
spec = importlib.util.spec_from_file_location('report_capacity', SOURCE)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': report.sha(path)}


@pytest.fixture
def completed(tmp_path, monkeypatch):
    prep = tmp_path/'evaluation_preparation'
    out = prep/'execution'
    out.mkdir(parents=True)
    receipt = {'status': 'completed', 'training_inputs_sha256': {}, 'endpoints': {},
               'result_sources': {}, 'report_files_sha256': {}, 'rgb_files_sha256': {}}
    stage_items = []
    for key in report.STAGES:
        capacity, stage = key.split('_', 1)
        endpoint = {'path': '/models/'+key+'/last.pt', 'sha256': key, 'steps': 6000 if stage == 'real' else 2000}
        record = {'status': 'completed', 'capacity': capacity, 'stage': stage, 'steps': endpoint['steps'],
                  'elapsed_seconds': 200. if capacity == 'vit7b' else 100.,
                  'last_checkpoint': endpoint['path'], 'last_checkpoint_sha256': endpoint['sha256']}
        binding = put(tmp_path/key/'stage_receipt.json', record)
        stage_items.append({'key': key, 'stage_receipt': binding['path']})
        receipt['training_inputs_sha256'][binding['path']] = binding['sha256']
        receipt['endpoints'][key] = endpoint
    plan = {'training_launch_manifest': put(tmp_path/'launch.json', {'stages': stage_items}),
            'official_evaluation_fingerprint': 'fingerprint', 'inference_scene': {'path': '/scene.pt', 'sha256': 'scene'},
            'bootstrap': {'repeats': 5000, 'seed': 42}}
    plan_binding = put(prep/'plan.json', plan)
    monkeypatch.setattr(report, 'PLAN_SHA', plan_binding['sha256'])
    receipt.update(plan=plan_binding['path'], plan_sha256=plan_binding['sha256'])
    metrics_by_key = {}
    for index, key in enumerate(report.LABELS):
        endpoint = receipt['endpoints'].get(key+'_render_adapt', {'path': '/selected.pt', 'sha256': 'selected'})
        teacher = {'teacher_checkpoint': endpoint['path'], 'teacher_checkpoint_sha256': endpoint['sha256']}
        metric = {'official_evaluation_fingerprint': 'fingerprint', 'validation_views': 50, 'semantic_validation_views': 41,
                  'views': [{'name': str(i), **({'confusion_matrix': []} if i < 41 else {})} for i in range(50)],
                  'scoring_protocol': {'class_names': ['background', 'deck', 'stay_cable', 'tower', 'foundation']},
                  'inference_protocol': 'fixed protocol', 'teacher_ensemble': teacher, 'psnr': 29., 'ssim': .87,
                  'lpips': .27, 'miou_all': .95+index*.004, 'miou_foreground': .94+index*.004,
                  'iou': [.99, .96, .94+index*.004, .95, .90+index*.004]}
        metric_binding = put(tmp_path/key/'official_metrics.json', metric)
        scoring = {'status': 'completed', 'official_metrics_sha256': metric_binding['sha256'],
                   'official_evaluation_fingerprint': 'fingerprint', 'scoring_protocol': metric['scoring_protocol'],
                   'inference_protocol': 'fixed protocol', 'checkpoint_sha256': 'scene', 'teacher_ensemble': teacher}
        score_binding = put(tmp_path/key/'execution_receipt.json', scoring)
        receipt['result_sources'][key] = {**metric_binding, 'receipt': score_binding['path'],
                                         'receipt_sha256': score_binding['sha256'], 'checkpoint_sha256': 'scene',
                                         'inference_protocol': 'fixed protocol', 'teacher_ensemble': teacher}
        receipt['rgb_files_sha256'][key] = {str(i): str(i) for i in range(50)}
        metrics_by_key[key] = metric
    for name, (reference, candidate) in report.PAIRS.items():
        pair = {'reference_source': receipt['result_sources'][reference], 'candidate_source': receipt['result_sources'][candidate],
                'difference_direction': 'candidate minus reference; LPIPS improves when negative',
                'official_evaluation_fingerprint': 'fingerprint', 'bootstrap_repeats': 5000, 'seed': 42, 'metrics': {}}
        for metric, _, _ in report.METRICS:
            a, b = (report.metric_value(metrics_by_key[k], metric) for k in (reference, candidate))
            pair['metrics'][metric] = {'reference': a, 'candidate': b, 'difference': b-a,
                                      'paired_view_bootstrap_95_interval': [b-a-.001, b-a+.001]}
        receipt['report_files_sha256'][name+'.json'] = put(out/(name+'.json'), pair)['sha256']
    gate = {'passed': False, 'checks': {'recorded_gate_failed': False, 'all_50_rgb_png_exact': True}}
    receipt['adoption_gate'] = gate
    receipt['report_files_sha256']['adoption_gate.json'] = put(out/'adoption_gate.json', gate)['sha256']
    path = out/'execution_receipt.json'
    put(path, receipt)
    return path


def test_completed_report_has_fixed_direction_units_cost_and_keep_decision(completed, tmp_path):
    output = tmp_path/'user_report.md'
    report.report(completed, output)
    text = output.read_text()
    assert '保持当前已选 H+ 固定组合' in text and '不据此改选匹配 H+' in text
    assert '+0.40000 [+0.30000, +0.50000]' in text
    assert '本轮 7B 组合 − 本轮匹配 H+ 组合' in text
    assert '合计 **2.000×**' in text and '不构成对同学 peer Dev30' in text
    assert '阶段末端来源核验' in text
    with pytest.raises(ValueError, match='overwrite'):
        report.report(completed, output)


@pytest.mark.parametrize('which', ['pair', 'gate', 'metrics', 'stage'])
def test_changed_bound_source_refuses_report(completed, tmp_path, which):
    receipt = report.read(completed)
    paths = {'pair': completed.parent/'vit7b_minus_hplus.json', 'gate': completed.parent/'adoption_gate.json',
             'metrics': Path(receipt['result_sources']['vit7b']['path']),
             'stage': Path(next(iter(receipt['training_inputs_sha256'])))}
    path = paths[which]
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='SHA mismatch'):
        report.report(completed, tmp_path/'missing/report.md')
    assert not (tmp_path/'missing').exists()


def test_incomplete_or_reversed_pair_refuses_report(completed, tmp_path):
    receipt = report.read(completed)
    incomplete = copy.deepcopy(receipt)
    incomplete['status'] = 'running'
    put(completed, incomplete)
    with pytest.raises(ValueError, match='completed'):
        report.report(completed, tmp_path/'unfinished.md')
    pair_path = completed.parent/'vit7b_minus_hplus.json'
    pair = report.read(pair_path)
    pair['reference_source'], pair['candidate_source'] = pair['candidate_source'], pair['reference_source']
    receipt['report_files_sha256'][pair_path.name] = put(pair_path, pair)['sha256']
    put(completed, receipt)
    with pytest.raises(ValueError, match='source/direction'):
        report.report(completed, tmp_path/'reversed.md')
    assert not (tmp_path/'unfinished.md').exists() and not (tmp_path/'reversed.md').exists()
