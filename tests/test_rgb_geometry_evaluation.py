"""Small synthetic contracts for the one-endpoint evaluation fork."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ENTRY = HERE/'evaluate_rgb_geometry_control.py'
if not ENTRY.exists():
    ENTRY = HERE.parent/'scripts/evaluate_rgb_geometry_control.py'
spec = importlib.util.spec_from_file_location('rgb_eval_contract', ENTRY)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def test_fixed_pair_directions_and_single_arm_work():
    assert worker.pair_definitions() == [('rgb', 'baseline', k) for k in worker.KINDS]+[
        (a, 'rgb', k) for a in ('joint', 'pcgrad', 'trust') for k in worker.KINDS]
    assert len(set(worker.pair_definitions())) == 12 and worker.ARMS == ('rgb',)
    assert [worker.SPEC[k] for k in ('scene_calls', 'teacher_predict_calls', 'teacher_backbone_forwards',
            'masks_before_GT', 'internal_seconds', 'external_seconds')] == [50, 50, 700, 150, 240, 300]
    assert worker.NUMERICS == worker.old.NUMERICS and worker.INFERENCE == worker.old.INFERENCE
    assert worker.SPEC['adoption_gate'] is None


def test_incomplete_or_repeated_prediction_barrier_rejected():
    views = [{'name': f'{i:03d}.png'} for i in range(50)]
    records = [{'arm': 'rgb', 'name': v['name'], 'masks': dict.fromkeys(worker.KINDS),
                'soft': dict.fromkeys(('raw', 'scene', 'teacher')), 'teacher_recomputed': True} for v in views]
    worker.prediction_barrier(records, views)
    with pytest.raises(ValueError):
        worker.prediction_barrier(records[:-1]+[records[0]], views)
    records[0]['teacher_recomputed'] = False
    with pytest.raises(ValueError):
        worker.prediction_barrier(records, views)


def test_actual_render_quantization_fusion_and_score_expressions_unchanged():
    def assignments(path):
        result = {}
        function = next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                result.setdefault(node.targets[0].id, []).append(ast.dump(node.value, include_attributes=False))
        return result
    actual = assignments(ENTRY); previous = assignments(Path(worker.old.__file__))
    for name in ('result', 'canvas', 'tp', 'soft', 'joint', 'score', 'cm'):
        assert actual[name] == previous[name], name
    assert worker.validate_endpoint is worker.old.validate_endpoint
    assert worker.counted_calls is worker.old.counted_calls
    assert worker.rendered_rgb_images is worker.old.rendered_rgb_images


def test_completed_lineage_requires_both_natural_audits(tmp_path, monkeypatch):
    values = {'plan.json': {}, 'execution_receipt.json': {'status': 'completed', 'plan_sha256': 'p'},
              'launch_receipt.json': {'status': 'completed', 'natural_completion': True, 'exit_code': 0,
                                     'plan_sha256': 'p', 'execution_receipt_sha256': 'e'},
              'independent_cpu_review.json': {'status': 'passed', 'plan_sha256': 'p', 'execution_receipt_sha256': 'e'},
              'independent_audit_launch_receipt.json': {'status': 'completed', 'natural_completion': True,
                                                      'exit_code': 0, 'audit_sha256': 'a'}}
    for name, value in values.items():
        (tmp_path/name).write_text(json.dumps(value))
    monkeypatch.setattr(worker, 'sha', lambda path: {'plan.json': 'p', 'execution_receipt.json': 'e',
                                                   'independent_cpu_review.json': 'a'}[Path(path).name])
    worker.completed(tmp_path, 'p', 'e', 'a')
    values['independent_audit_launch_receipt.json']['exit_code'] = 1
    (tmp_path/'independent_audit_launch_receipt.json').write_text(json.dumps(values['independent_audit_launch_receipt.json']))
    with pytest.raises(ValueError, match='Natural'):
        worker.completed(tmp_path, 'p', 'e', 'a')
