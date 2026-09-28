"""Synthetic control and source contracts; no scene, target, or CUDA reads."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parent
WORKER=HERE/'train_direct_q_head_matched.py'
if not WORKER.exists(): WORKER=HERE.parent/'scripts/train_direct_q_head_matched.py'
spec=importlib.util.spec_from_file_location('direct_q_head_matched_tested',WORKER)
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


class Scene(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.splats=torch.nn.ParameterDict({'means':torch.nn.Parameter(torch.zeros(2,3)),
                                           'sem_features':torch.nn.Parameter(torch.tensor([[1.,2.],[3.,4.]]))})
        self.semantic_decoder=torch.nn.Linear(2,5)
        self.refiner=torch.nn.Linear(2,5)


def test_training_scope_fresh_adam_q_and_nonhead_unchanged():
    torch.manual_seed(42); scene=Scene(); frozen=worker.digest_tensors(scene.state_dict())
    q=worker.load_q(scene,'original_q',{})
    optimizer=worker.configure_parameters(scene)
    assert not optimizer.state and not q.requires_grad
    assert [(g['name'],g['lr'],g['eps'],g['betas'],g['weight_decay']) for g in optimizer.param_groups]==[('head',.0003,1e-8,(.9,.999),0.)]
    before=q.clone();old_head=worker.digest_tensors(scene.state_dict(),True)
    scene.refiner(scene.splats['sem_features'].detach()).square().mean().backward();optimizer.step()
    assert worker.digest_tensors(scene.state_dict())==frozen
    assert worker.digest_tensors(scene.state_dict(),True)!=old_head and torch.equal(before,q)
    assert all(p.grad is None for n,p in scene.named_parameters() if not n.startswith('refiner.'))


def test_original_q_exact_actual_classifier_not_master_normalization():
    scene=Scene()
    expected=scene.semantic_decoder(scene.splats['sem_features']).softmax(-1).detach()
    actual=worker.load_q(scene,'original_q_zero',{})
    assert torch.equal(actual,expected) and not actual.requires_grad


def test_optimized_q_exact_cast_and_zeros(tmp_path):
    scene=Scene(); master=np.array([[0.,.1,.2,.3,.4],[1.,0.,0.,0.,0.]],np.float64)
    actual=master.astype(np.float32);path=tmp_path/'q.npz';np.savez(path,q_master=master,q_renderer=actual)
    plan={'q_delta':{'path':str(path),'sha256':worker.sha(path)},
          'optimized_q_renderer_sha256':worker.hashlib.sha256(actual.tobytes()).hexdigest()}
    q=worker.load_q(scene,'optimized_q',plan)
    assert np.array_equal(q.numpy(),actual) and int((q==0).sum())==5
    np.savez(path,q_master=master,q_renderer=np.roll(actual,1,axis=1))
    plan['q_delta']['sha256']=worker.sha(path)
    with pytest.raises(ValueError,match='actual q cast'):worker.load_q(scene,'optimized_q',plan)


def test_loss_only_actual_head_objective():
    probabilities=torch.tensor([.4,.6],requires_grad=True); raw=torch.tensor([.2,.8],requires_grad=True)
    residual=torch.tensor([.1,-.2],requires_grad=True); calls=[]
    def loss(p,labels,valid,weights,lovasz_weight):
        calls.append((p,lovasz_weight));return p.square().sum()
    value,terms=worker.loss_terms({'probabilities':probabilities,'p3d':raw,'residual':residual},None,None,None,loss)
    expected=probabilities.square().sum()+.001*residual.square().mean()
    assert torch.equal(value,expected) and len(calls)==1 and calls[0][1]==.2
    value.backward();assert raw.grad is None and residual.grad is not None
    assert set(terms)=={'final','residual_square'}


def test_fixed_schedule_full_epochs_and_rng_independent():
    order=worker.sample_order();np.random.seed(711);np.random.normal(size=90)
    assert worker.sample_order()==order and len(order)==2000
    for start in range(0,1813,259):assert sorted(order[start:start+259])==list(range(259))
    assert worker.SPEC['candidate']=='optimized_q' and worker.SPEC['steps']==2000
    assert worker.SPEC['internal_seconds']==1500 and worker.SPEC['external_seconds']==1560


def receipt(arm):
    return {'arm':arm,'status':'completed','steps':2000,'sample_names_sha256':'same',
        'frozen_nonrefiner_sha256':'same','refiner_initial_sha256':'same','training_camera_sha256':'same',
        'parameter_count_refiner':20,'q_initial_sha256':arm,'q_final_sha256':arm,
        'optimizer_groups':[{'name':'head','eps':1e-8}]}


@pytest.mark.parametrize('key',['sample_names_sha256','refiner_initial_sha256','training_camera_sha256','q_final_sha256'])
def test_matched_contract_rejects_different_init_schedule_or_q(key):
    records=[receipt(a) for a in worker.ARMS];worker.validate_matched_receipts(records)
    records[1][key]='changed'
    with pytest.raises(ValueError):worker.validate_matched_receipts(records)


def pairs():
    return {reference:{'metrics':{'miou_all':{'difference':threshold,'paired_view_bootstrap_95_interval':[1e-8,.1]},
       **{c+'_iou':{'difference':-.001 if c=='stay_cable' else -.002} for c in ('background','deck','stay_cable','tower','foundation')}}}
        for reference,threshold in worker.THRESHOLDS.items()}


def test_exact_primary_gate_boundaries_and_no_posthoc_zero_arm():
    p=pairs();assert len(worker.adoption_clauses(p))==14 and all(worker.adoption_clauses(p).values())
    p['original_q']['metrics']['miou_all']['difference']=.0009999
    assert not all(worker.adoption_clauses(p).values())
    p=pairs();p['E']['metrics']['miou_all']['paired_view_bootstrap_95_interval'][0]=0
    assert not all(worker.adoption_clauses(p).values())
    p=pairs();p['E']['metrics']['stay_cable_iou']['difference']=-.001001
    assert not all(worker.adoption_clauses(p).values())
    with pytest.raises(ValueError):worker.adoption_clauses({'optimized_q_zero':p['E'],'E':p['E']})


def test_complete_prediction_barrier():
    views=[{'name':f'{i:03}.png'} for i in range(50)]
    records=[{'arm':a,'name':v['name'],'masks':{'joint':{},'scene':{},'raw':{}}} for a in worker.READOUTS for v in views]
    checks=[{'name':v['name'],'checks':{'p3d':True,'probabilities':True}} for v in views]
    worker.prediction_barrier(records,[{}]*50,checks,views)
    with pytest.raises(ValueError):worker.prediction_barrier(records[:-1],[{}]*50,checks,views)
    checks[0]['checks']['p3d']=False
    with pytest.raises(ValueError):worker.prediction_barrier(records,[{}]*50,checks,views)


def test_source_completion_required_before_q_load():
    plan={'specification':{'protocol':'raw_simplex_em_fw_v1'}}
    execution={'status':'completed','plan_sha256':'p','inputs_sources_unchanged':True,'state_unchanged_before_restore':True}
    launch={'status':'completed','natural_completion':True,'exit_code':0,'plan_sha256':'p','execution_receipt_sha256':'e'}
    audit={'status':'passed','plan_sha256':'p','execution_receipt_sha256':'e','q':{'shape':[498136,5]}}
    worker.validate_parent(plan,execution,launch,audit,'p','e')
    bad=copy.deepcopy(launch);bad['natural_completion']=False
    with pytest.raises(ValueError):worker.validate_parent(plan,execution,bad,audit,'p','e')


def test_real_gate_report_serializes_strictly_and_failed_write_keeps_old(tmp_path):
    path=tmp_path/'report.json';report={'status':'completed','training':[receipt(a) for a in worker.ARMS],
        'gate':worker.adoption_clauses(pairs()),'specification':worker.SPEC}
    worker.write(path,report);assert json.loads(path.read_text())['status']=='completed'
    original=path.read_bytes()
    with pytest.raises(ValueError):worker.write(path,{'bad':float('nan')},replace=True)
    assert path.read_bytes()==original and not list(tmp_path.glob('*.pending'))


def test_numerical_flags_roundtrip_without_cuda():
    before=worker.numerical_flags(torch)
    try:
        assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
        assert torch.get_float32_matmul_precision()=='highest' and not torch.is_autocast_enabled()
    finally:worker.numerical_flags(torch,before)
    assert worker.read_numerical_flags(torch)==before
    assert not torch.cuda.is_initialized()
