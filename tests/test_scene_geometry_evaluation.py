"""Thin-fork population/identity contracts, without model or pixel access."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ENTRY = HERE/'evaluate_scene_geometry_routes.py'
if not ENTRY.exists():
    ENTRY = HERE.parent/'scripts/evaluate_scene_geometry_routes.py'
spec = importlib.util.spec_from_file_location('scene_routes_eval_contract', ENTRY)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_fixed_two_arms_eleven_directions_and_work_budget():
    expected = ([(a, 'baseline', k) for a in ('full', 'prior_only') for k in worker.KINDS]
                + [('prior_only', 'full', k) for k in worker.KINDS]
                + [(a, 'E', 'joint') for a in ('full', 'prior_only')])
    assert worker.pair_definitions() == expected and len(set(expected)) == 11
    assert [worker.SPEC[k] for k in ('scene_calls', 'teacher_predict_calls', 'teacher_backbone_forwards',
            'masks_before_GT', 'RGB_predictions', 'internal_seconds', 'external_seconds')] == [100, 100, 1400, 300, 100, 360, 420]
    assert worker.SPEC['adoption_gate'] is None and worker.SPEC['head_adaptation_steps'] == 0
    assert worker.NUMERICS == worker.old.NUMERICS and worker.INFERENCE == worker.old.INFERENCE


def test_barrier_requires_both_arms_new_teacher_all_masks_and_own_rgb():
    views = [{'name': f'{i:03d}.png'} for i in range(50)]
    rows = [{'arm': a, 'name': v['name'], 'masks': dict.fromkeys(worker.KINDS),
             'soft': dict.fromkeys(('raw', 'scene', 'teacher')), 'teacher_recomputed': True,
             'rgb': {'path': 'own.png'}, 'teacher_input_rgb': {'path': 'canvas.png'}}
            for a in worker.ARMS for v in views]
    worker.prediction_barrier(rows, views)
    for bad in (rows[:50], rows[:-1]+[rows[0]]):
        with pytest.raises(ValueError):
            worker.prediction_barrier(bad, views)
    rows[0]['teacher_recomputed'] = False
    with pytest.raises(ValueError):
        worker.prediction_barrier(rows, views)
    rows[0]['teacher_recomputed'] = True
    del rows[0]['rgb']
    with pytest.raises(ValueError):
        worker.prediction_barrier(rows, views)


def test_old_deployment_render_and_scoring_expressions_not_new_gradient_adapter():
    def assignments(path):
        function = next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
        result = {}
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                result.setdefault(node.targets[0].id, []).append(ast.dump(node.value, include_attributes=False))
        return result
    actual, previous = assignments(ENTRY), assignments(Path(worker.old.__file__))
    for name in ('result', 'canvas', 'tp', 'soft', 'joint', 'score', 'cm'):
        assert actual[name] == previous[name], name
    assert worker.validate_endpoint is worker.old.validate_endpoint
    assert worker.counted_calls is worker.old.counted_calls
    assert worker.rendered_rgb_images is worker.old.rendered_rgb_images
    imports = [n for n in ast.walk(ast.parse(ENTRY.read_text())) if isinstance(n, ast.ImportFrom)]
    assert any(n.module == 'bridge_rgs.direct_q_render' for n in imports)
    assert not any(n.module == 'bridge_rgs.direct_q_geometry_render' for n in imports)


@pytest.mark.parametrize('audit_hash_key', ['audit_sha256', 'report_sha256'])
def test_completed_gate_rejects_failed_or_missing_training_audit(tmp_path, monkeypatch, audit_hash_key):
    values = {'plan.json': {}, 'execution_receipt.json': {'status': 'completed', 'plan_sha256': 'p'},
              'launch_receipt.json': {'status': 'completed', 'natural_completion': True, 'exit_code': 0,
                                     'plan_sha256': 'p', 'execution_receipt_sha256': 'e'},
              'independent_cpu_review.json': {'status': 'passed', 'plan_sha256': 'p', 'execution_receipt_sha256': 'e'},
              'independent_audit_launch_receipt.json': {'status': 'completed', 'natural_completion': True,
                                                      'exit_code': 0, audit_hash_key: 'a'}}
    for name, value in values.items():
        (tmp_path/name).write_text(json.dumps(value))
    monkeypatch.setattr(worker, 'sha', lambda p: {'plan.json': 'p', 'execution_receipt.json': 'e',
                                                 'independent_cpu_review.json': 'a'}[Path(p).name])
    worker.completed(tmp_path, 'p', 'e', 'a')
    values['launch_receipt.json']['exit_code'] = 1
    (tmp_path/'launch_receipt.json').write_text(json.dumps(values['launch_receipt.json']))
    with pytest.raises(ValueError):
        worker.completed(tmp_path, 'p', 'e', 'a')
    (tmp_path/'independent_cpu_review.json').unlink()
    with pytest.raises(FileNotFoundError):
        worker.completed(tmp_path, 'p', 'e', 'a')
