"""H3 receipt schema, numerical gates, provenance, and CPU-only preparation."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runner = module('fullbatch_h3_v2', 'run_fullbatch_appearance_v2.py')
old = module('fullbatch_original_v1', 'run_fullbatch_appearance.py')
producer = module('h3_objective_producer', 'audit_h3_fullbatch_objective.py')


def h3_evidence():
    sources = {'bridge_rgs/'+key+'.py': 'a'*64 for key in
               ('train', 'model', 'raw_grid', 'losses', 'fullbatch_appearance', 'evaluate', 'coordinates')}
    sources.update({'bridge_rgs/losses.py': runner.LOSS_SHA,
                    'bridge_rgs/fullbatch_appearance.py': runner.HELPER_SHA})
    plan = {'specification': copy.deepcopy(producer.SPEC), 'base': '/base.pt', 'base_sha256': runner.BASE_SHA,
            'manifest': '/manifest.json', 'source_snapshot': '/snapshot', 'source_hashes': sources,
            'input_hashes': {'/base.pt': runner.BASE_SHA, '/manifest.json': runner.MANIFEST_SHA},
            'allowed_pixel_paths': ['/002.png', '/205.png'],
            'views': [{'name': name} for name in runner.H3_VIEW_NAMES]}
    rows = []
    for name in runner.H3_VIEW_NAMES:
        differences = {}
        for key in runner.H3_KEYS:
            differences[key] = {'parameter_restored_exact': True, 'epsilons': [
                {'epsilon': e, 'plus_fp32_loss': 1+21*e, 'minus_fp32_loss': 1-21*e,
                 'central_difference': 21., 'realized_analytic': 20.,
                 'realized_relative_discrepancy': .05, 'passed': True}
                for e in runner.H3_FD_EPSILONS]}
        rows.append({'name': name, 'render_calls': 20, 'finite_differences': differences,
                     'baseline': {'warp_cv2_max_abs_difference': 2e-6},
                     'temporary_rms_step': {'before_fp32_loss': 1., 'after_fp32_loss': .9999,
                                            'actual_slope': -.0001, 'actual_decrease': .0001,
                                            'predicted_decrease': .0001, 'actual_to_predicted_decrease_ratio': 1.,
                                            'decrease_floor': 1e-5, 'helper_armijo_passed': True,
                                            'passed': True, 'parameters_restored_exact': True, 'committed_steps': 0}})
    receipt = {'status': 'completed', 'plan_sha256': 'test-plan', 'base_sha256': runner.BASE_SHA,
               'profile': 'legacy_mixed_v1', 'render_calls': 40, 'backward_calls': 2,
               'all_model_tensors_finally_restored_exact': True, 'all_flags_restored': True,
               'all_modes_restored': True, 'training_cameras_unchanged': True,
               'inputs_after_exact': True, 'source_after_exact': True,
               'all_bound_inputs_and_sources_unchanged': True, 'technical_gate_passed': True,
               'tensor_hashes_before': {'buffer': 'a'*64}, 'tensor_hashes_after': {'buffer': 'a'*64},
               'training_cameras_sha256': 'c'*64, 'checkpoint_written': False, 'committed_optimizer_steps': 0,
               'semantic_label_pixels_decoded': 0, 'val_pixels_decoded': 0, 'views': rows,
               'source_hashes': copy.deepcopy(sources), 'input_hashes': copy.deepcopy(plan['input_hashes']),
               'actual_imports': {'bridge_rgs.'+Path(k).stem: {'path': '/snapshot/'+k, 'sha256': v}
                                  for k, v in sources.items()},
               'pixel_reads': dict.fromkeys(plan['allowed_pixel_paths'], 1)}
    return receipt, plan


def test_entire_optimizer_spec_and_old_source_are_unchanged():
    previous, current = dict(old.SPEC), dict(runner.SPEC)
    assert previous.pop('protocol') == 'conditional_fullbatch_appearance_v1'
    assert current.pop('protocol') == 'conditional_fullbatch_appearance_h3_v2'
    assert previous == current
    assert runner.sha(ROOT/'src/bridge_rgs/fullbatch_appearance.py') == runner.HELPER_SHA
    assert runner.sha(ROOT/'scripts/run_fullbatch_appearance.py') == '97fd62ba1ddfa2093327f638dd0bc356441beac17cde538b7026b5dc4c2dc0fd'
    receipt, plan = h3_evidence()
    runner.validate_h3_diagnostic(receipt, plan, 'test-plan')


@pytest.mark.parametrize('mutation', [
    'failed', 'false-gate', 'wrong-plan', 'wrong-base', 'corner', 'incomplete-renders', 'not-restored',
    'flags', 'modes', 'cameras', 'changed-source', 'checkpoint', 'labels', 'val', 'wrong-view',
    'missing-fd', 'duplicate-fd', 'wrong-epsilon', 'fd-fails', 'fd-nan', 'fd-inconsistent',
    'warp-fails', 'rms-uphill', 'rms-ratio', 'rms-inconsistent', 'rms-floor', 'rms-committed',
    'wrong-import', 'wrong-helper-sha', 'different-spec', 'extra-read', 'changed-tensors',
])
def test_h3_gate_rejects_incomplete_wrong_or_failed_measurements(mutation):
    value, plan = h3_evidence()
    row = value['views'][0]
    entries = row['finite_differences'][runner.H3_KEYS[0]]['epsilons']
    step = row['temporary_rms_step']
    changes = {'failed': ('status', 'failed'), 'false-gate': ('technical_gate_passed', False),
               'wrong-plan': ('plan_sha256', 'other'), 'wrong-base': ('base_sha256', 'other'),
               'corner': ('profile', 'colmap_corner_v2'), 'incomplete-renders': ('render_calls', 39),
               'not-restored': ('all_model_tensors_finally_restored_exact', False),
               'flags': ('all_flags_restored', False), 'modes': ('all_modes_restored', False),
               'cameras': ('training_cameras_unchanged', False),
               'changed-source': ('all_bound_inputs_and_sources_unchanged', False),
               'checkpoint': ('checkpoint_written', True), 'labels': ('semantic_label_pixels_decoded', 1),
               'val': ('val_pixels_decoded', 1)}
    if mutation in changes:
        key, item = changes[mutation]
        value[key] = item
    elif mutation == 'wrong-view':
        row['name'] = '175.png'
    elif mutation == 'missing-fd':
        entries.pop()
    elif mutation == 'duplicate-fd':
        entries[0] = copy.deepcopy(entries[1])
    elif mutation == 'wrong-epsilon':
        entries[0]['epsilon'] = .1
    elif mutation in {'fd-fails', 'fd-nan'}:
        entries[0]['realized_relative_discrepancy'] = .0500001 if mutation == 'fd-fails' else float('nan')
    elif mutation == 'fd-inconsistent':
        entries[0]['plus_fp32_loss'] += .1
    elif mutation == 'warp-fails':
        row['baseline']['warp_cv2_max_abs_difference'] = 2.0001e-6
    elif mutation == 'rms-uphill':
        step['actual_slope'] = .0001
    elif mutation == 'rms-ratio':
        step.update(actual_to_predicted_decrease_ratio=.89, actual_decrease=.000089, after_fp32_loss=1-.000089)
    elif mutation == 'rms-inconsistent':
        step['actual_decrease'] = .00008
    elif mutation == 'rms-floor':
        step.update(after_fp32_loss=1-1e-6, actual_slope=-1e-6, actual_decrease=1e-6, predicted_decrease=1e-6)
    elif mutation == 'rms-committed':
        step['committed_steps'] = 1
    elif mutation == 'wrong-import':
        value['actual_imports']['bridge_rgs.losses']['path'] = '/main/losses.py'
    elif mutation == 'wrong-helper-sha':
        value['actual_imports']['bridge_rgs.fullbatch_appearance']['sha256'] = 'other'
    elif mutation == 'different-spec':
        plan['specification']['fd_max_relative_error'] = .10
    elif mutation == 'extra-read':
        value['pixel_reads']['/VAL.png'] = 1
    else:
        value['tensor_hashes_after']['buffer'] = 'changed'
    with pytest.raises(ValueError, match='H3'):
        runner.validate_h3_diagnostic(value, plan, 'test-plan')


def test_exact_thresholds_are_inclusive_and_every_view_is_required():
    value, plan = h3_evidence()
    for i, row in enumerate(value['views']):
        ratio = .9 if i == 0 else 1.1
        row['temporary_rms_step'].update(actual_decrease=.0001*ratio, after_fp32_loss=1-.0001*ratio,
                                         actual_to_predicted_decrease_ratio=ratio)
    runner.validate_h3_diagnostic(value, plan, 'test-plan')
    value['views'].pop()
    with pytest.raises(ValueError, match='H3'):
        runner.validate_h3_diagnostic(value, plan, 'test-plan')


def test_bound_evidence_requires_review_and_bytes(tmp_path, monkeypatch):
    def save(name, value):
        path = tmp_path/name
        path.write_text(json.dumps(value, allow_nan=False))
        return {'path': str(path), 'sha256': runner.sha(path)}
    fixed = save('fixed.json', {'status': 'completed', 'plan_sha256': runner.FIXED_SSIM_PLAN_SHA,
                              'render_calls': 40, 'all_model_tensors_finally_restored_exact': True,
                              'all_bound_inputs_and_sources_unchanged': True,
                              'actual_imports': {'losses': {'sha256': runner.LOSS_SHA}}})
    monkeypatch.setattr(runner, 'FIXED_SSIM_RECEIPT_SHA', fixed['sha256'])
    # Dependency bytes are tested separately below; this case isolates receipt and review binding.
    monkeypatch.setattr(runner, 'verify_diagnostic_bytes', lambda plan: None)
    value, h3 = h3_evidence()
    h3_plan = save('h3plan.json', h3)
    value['plan_sha256'] = h3_plan['sha256']
    receipt = save('h3.json', value)
    review_value = {'decision': 'allow_fullbatch_appearance_h3_v2',
                    'fixed_ssim_diagnostic_receipt_sha256': fixed['sha256'],
                    'h3_diagnostic_receipt_sha256': receipt['sha256'],
                    'h3_fd_warp_rms_and_restoration_review_passed': True, 'reason': 'Explicitly reviewed.'}
    review = save('review.json', review_value)
    plan = {'fixed_ssim_diagnostic_receipt': fixed, 'h3_diagnostic_plan': h3_plan,
            'h3_diagnostic_receipt': receipt, 'diagnostic_review': review}
    runner.diagnostic_gate(plan)
    review_value['h3_fd_warp_rms_and_restoration_review_passed'] = False
    plan['diagnostic_review'] = save('review.json', review_value)
    with pytest.raises(ValueError, match='Explicit review'):
        runner.diagnostic_gate(plan)
    Path(receipt['path']).write_text(Path(receipt['path']).read_text()+' ')
    with pytest.raises(ValueError, match='Changed bound'):
        runner.diagnostic_gate(plan)


def test_dependency_bytes_checked_and_no_gpu_called(tmp_path, monkeypatch):
    file = tmp_path/'dependency.py'
    file.write_text('source')
    plan = {'source_snapshot': str(tmp_path), 'source_hashes': {'dependency.py': runner.sha(file)},
            'input_hashes': {str(file): runner.sha(file)}}
    monkeypatch.setattr(runner, 'gpu_idle', lambda: pytest.fail('CPU verification must not query GPU'))
    runner.verify_diagnostic_bytes(plan)
    file.write_text('changed')
    with pytest.raises(ValueError, match='Changed bound'):
        runner.verify_diagnostic_bytes(plan)


def test_prepare_rejects_before_creating_output(tmp_path, monkeypatch):
    evidence = tmp_path/'evidence.json'
    evidence.write_text('{}')
    # A realistic failed gate must prevent snapshot/output creation; no model or CUDA load follows.
    monkeypatch.setattr(runner, 'sha', lambda path: 'fake-sha')
    def reject(_):
        raise ValueError('H3 actual-FP32-direction finite differences exceed the fixed 5% gate')
    monkeypatch.setattr(runner, 'diagnostic_gate', reject)
    output = tmp_path/'newrun'
    with pytest.raises(ValueError, match='H3'):
        runner.prepare(tmp_path, output, evidence, evidence, evidence)
    assert not output.exists()


def test_fixed_external_deadline_and_separate_execution_directory():
    root, snapshot, plan = Path('/root'), Path('/output/source_snapshot'), Path('/output/plan.json')
    value = runner.execution_contract(root, snapshot, plan)
    assert value['command'][:4] == ['timeout', '--signal=TERM', '--kill-after=5s', '600s']
    assert value['command'][-2:] == ['--run', str(plan)]
    assert value['environment']['PYTHONPATH'] == str(snapshot)
    assert runner.SPEC['optimization_seconds'] == 360.
