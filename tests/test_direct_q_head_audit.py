import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_direct_q_head_matched as audit


def test_candidate_gate_requires_both_fixed_references_and_all_class_guards():
    pairs = {}
    for name in audit.THRESHOLDS:
        pairs[name] = {'miou_all': {'difference': .003, 'paired_view_bootstrap_95_interval': [.0001, .005]},
                       **{c+'_iou': {'difference': 0.} for c in audit.common.CLASSES}}
    assert len(audit.gate_clauses(pairs)) == 14 and all(audit.gate_clauses(pairs).values())
    pairs['E']['stay_cable_iou']['difference'] = -.00101
    assert not audit.gate_clauses(pairs)['E_stay_cable_guard']
    with pytest.raises(AssertionError, match='Fixed candidate references'):
        audit.gate_clauses({'original_q_zero': pairs['original_q'], 'E': pairs['E']})


def fixture():
    rows = []
    for i in range(50):
        row = {'name': f'{i:03d}.png', 'width': 5, 'height': 1, 'rgb_pixels': 5,
               'psnr': 20., 'ssim': .8, 'lpips': .2}
        if i < 41:
            row.update(confusion_matrix=[[int(j == k) for k in range(5)] for j in range(5)],
                       semantic_pixels=5, semantic_ignore_pixels=0)
        rows.append(row)
    metrics = {'views': rows, 'confusion_matrix': [[41*int(j == k) for k in range(5)] for j in range(5)],
               'iou': [1.]*5, 'miou_all': 1., 'miou_foreground': 1., 'psnr': 20., 'ssim': .8, 'lpips': .2}
    return metrics, {'views': copy.deepcopy(rows)}


def test_saved_cm_reduction_does_not_need_any_gt_or_prediction_decoder():
    metrics, reference = fixture()
    out = audit.saved_metrics(metrics, reference, audit.common.Audit())
    assert out == {'miou_all': 1., 'iou': [1.]*5}
    metrics['confusion_matrix'][0][0] -= 1
    with pytest.raises(AssertionError, match='Pooled CM'):
        audit.saved_metrics(metrics, reference, audit.common.Audit())


@pytest.mark.parametrize('damage', ['rgb', 'support', 'missing_annotation'])
def test_wrong_score_reuse_or_semantic_support_fails(damage):
    metrics, reference = fixture()
    if damage == 'rgb':
        metrics['views'][0]['psnr'] += .01
    elif damage == 'support':
        metrics['views'][0]['semantic_pixels'] += 1
    else:
        del metrics['views'][0]['confusion_matrix']
    with pytest.raises(AssertionError):
        audit.saved_metrics(metrics, reference, audit.common.Audit())


def recovery_fixture():
    parent = {'status': 'failed', 'new_val_annotation_payload_reads': 0,
              'numerics_actual': audit.TRAIN_NUMERICS, 'amp_enabled': False, 'numerics_restored': True,
              'training': [{'arm': a, 'status': 'completed', 'steps': 2000} for a in audit.ARMS]}
    rows = [{'label': label, 'actual_numerics': audit.TRAIN_NUMERICS if i == 0 else audit.EVAL_NUMERICS,
             'E_mask_pixel_differences': 7 if i == 0 else 0, 'E_mask_bytes_exact': i != 0,
             'p3d_sha256': 'same-p3d', 'joint_mask': {'sha256': 'same-mask' if i else 'old-mismatch'},
             'soft': {'sha256': 'same-soft' if i else 'old-soft'}}
            for i, label in enumerate(('native_tf32_false', 'native_tf32_true', 'original_q_tf32_true'))]
    rows[1]['false_vs_true_p3d_exact'] = True
    rows[2].update(native_true_soft_exact=True, native_true_p3d_exact=True)
    diagnostic = {'status': 'completed', 'hypothesis_confirmed': True, 'scene_calls': 3,
                  'direct_q_shader_calls': 1, 'target_reads': 0, 'training_updates': 0,
                  'state_exact': True, 'q_exact': True, 'inputs_unchanged': True,
                  'numerics_restored': True, 'records': rows}
    return parent, {'status': 'failed', 'exit_code': 1}, diagnostic, audit.EVAL_NUMERICS.copy()


def test_eval_only_recovery_preserves_failed_parent_and_distinct_policies():
    values = recovery_fixture()
    result = audit.recovery_numerics(*values)
    assert result['parent_status'] == 'failed' and result['new_training_updates'] == 0
    assert values[0]['status'] == 'failed'


@pytest.mark.parametrize('damage', ['reclassify_parent', 'incomplete_training', 'wrong_eval_policy', 'diagnostic_reads_gt'])
def test_bad_recovery_evidence_fails(damage):
    parent, launch, diagnosis, policy = recovery_fixture()
    if damage == 'reclassify_parent':
        parent['status'] = 'completed'
    elif damage == 'incomplete_training':
        parent['training'][1]['steps'] = 1000
    elif damage == 'wrong_eval_policy':
        policy['cudnn_allow_tf32'] = False
    else:
        diagnosis['target_reads'] = 1
    with pytest.raises(AssertionError):
        audit.recovery_numerics(parent, launch, diagnosis, policy)
