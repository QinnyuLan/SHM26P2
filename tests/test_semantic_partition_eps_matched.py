"""CPU contracts for the epsilon-only follow-up; no experiment data or CUDA."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

HERE=Path(__file__).resolve()
SCRIPT=HERE.with_name('train_semantic_partition_eps_matched.py')
if not SCRIPT.exists():SCRIPT=HERE.parents[1]/'scripts/train_semantic_partition_eps_matched.py'
spec=importlib.util.spec_from_file_location('partition_eps_runner',SCRIPT)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
FIELD=SCRIPT.parent/'bridge_rgs/partition_field.py'
if not FIELD.exists():FIELD=runner.PREDECESSOR/'source_snapshot/bridge_rgs/partition_field.py'
spec=importlib.util.spec_from_file_location('frozen_partition_field_cpu',FIELD)
field_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(field_module)


class Scene(nn.Module):
    def __init__(self):
        super().__init__()
        self.splats=nn.ParameterDict({'sem_features':nn.Parameter(torch.ones(3,4))})
        self.semantic_decoder=nn.Linear(4,5)
        self.refiner=nn.Linear(5,5)
        with torch.no_grad():
            for parameter in self.parameters():parameter.zero_()


def test_actual_groups_are_explicit_and_cover_only_head_and_new_field():
    scene=Scene();initial=copy.deepcopy(scene.state_dict());field_initial=[]
    for arm in runner.ARMS:
        scene.load_state_dict(initial)
        field,groups=runner.configure_parameters(scene,arm,field_module.SemanticPartitionField)
        optimizer=runner.make_optimizer(groups)
        expected=[('head',3e-4,1e-8)]+([] if field is None else [('endpoints',.01,1e-15),('shape',.001,1e-15)])
        assert [(g['name'],g['lr'],g['eps']) for g in optimizer.param_groups]==expected
        assert not optimizer.state
        params=[p for group in optimizer.param_groups for p in group['params']]
        assert len(params)==len({id(p) for p in params})
        assert all(p.requires_grad==name.startswith('refiner.') for name,p in scene.named_parameters())
        if field is not None:
            assert torch.equal(field.inside_logits,field.outside_logits)
            field_initial.append(copy.deepcopy(field.state_dict()))
    assert all(torch.equal(s[k],field_initial[0][k]) for s in field_initial for k in s)


def test_tiny_gradient_has_measurable_field_update_with_unchanged_head_update():
    records=[]
    for eps in (1e-8,1e-15):
        scene=Scene();field,groups=runner.configure_parameters(scene,'integrated',field_module.SemanticPartitionField)
        for group in groups[1:]:group['eps']=eps
        optimizer=torch.optim.Adam(groups,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        before=copy.deepcopy(scene.state_dict())
        for group in groups:
            for parameter in group['params']:parameter.grad=torch.full_like(parameter,1e-12)
        optimizer.step()
        expected=.01*1e-12/(1e-12+eps)
        assert float(field.inside_logits[0,0].detach())==pytest.approx(-expected,rel=2e-6)
        assert all(torch.equal(v,before[k]) for k,v in scene.state_dict().items() if not k.startswith('refiner.'))
        records.append((float(field.inside_logits[0,0].detach()),scene.refiner.weight.detach().clone()))
    assert abs(records[1][0])>9000*abs(records[0][0])
    assert torch.equal(records[0][1],records[1][1])


def test_marginal_direction_is_legitimately_unused_and_no_adam_state_created():
    scene=Scene();field,groups=runner.configure_parameters(scene,'marginal',field_module.SemanticPartitionField)
    optimizer=runner.make_optimizer(groups)
    qi,qo=field.endpoints();_,b,w,tau=field.slab_parameters()
    sigma=torch.sqrt(1+tau.square())
    fraction=torch.special.ndtr((-b+w)/sigma)-torch.special.ndtr((-b-w)/sigma)
    loss=(fraction[:,None]*qi+(1-fraction[:,None])*qo)[:,0].sum()+.01*w.sum()
    loss.backward()
    assert field.direction.grad is None
    assert field.inside_logits.grad is not None and field.inside_logits.grad.abs().sum()>0
    optimizer.step()
    assert field.direction not in optimizer.state
    assert int(optimizer.state[field.inside_logits]['step'])==1
    assert torch.equal(field.direction,torch.tensor([[1.,0.,0.]]).expand(3,-1))


def predecessor_fixture():
    old=copy.deepcopy(runner.SPEC);old['protocol']='semantic_partition_matched_v1'
    old['optimizer'].pop('head_eps');old['optimizer'].pop('field_eps');old['optimizer']['eps']=1e-8
    old['claim']='fixed-density function-class experiment; Gaussian CDF integration has prior art; no novelty or adoption presumed'
    p={'specification':old,'training_order':runner.sample_order(),'base_checkpoint_sha256':runner.BASE_SHA}
    e={'status':'completed','bound_sources_inputs_unchanged':True,'plan_sha256':runner.OLD_PLAN_SHA,
       'training':[{'arm':arm,'status':'completed','steps':2000} for arm in runner.ARMS],
       'evaluation':{'gate':{'passed':False}}}
    launch={'status':'completed','exit_code':0,'natural_completion':True,'plan_sha256':runner.OLD_PLAN_SHA}
    audit={'status':'passed','inputs_sources_outputs_unchanged':True,'plan_sha256':runner.OLD_PLAN_SHA}
    return p,e,launch,audit


def test_completed_failed_adoption_is_valid_source_but_failed_execution_is_not():
    p,e,l,a=predecessor_fixture();runner.validate_predecessor(p,e,l,a)
    for changed in ({'exit_code':1},{'natural_completion':False},{'status':'failed'}):
        with pytest.raises(ValueError,match='naturally'):runner.validate_predecessor(p,e,dict(l,**changed),a)
    with pytest.raises(ValueError,match='audit'):runner.validate_predecessor(p,e,l,dict(a,status='failed'))
    p['specification']['optimizer']['refiner_lr']=.001
    with pytest.raises(ValueError,match='more than'):runner.validate_predecessor(p,e,l,a)


def test_sampler_and_primary_gate_contract_unchanged_secondary_cannot_substitute():
    np.random.seed(6);before=np.random.get_state();order=runner.sample_order();after=np.random.get_state()
    assert len(order)==2000 and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    assert all(sorted(order[i:i+259])==list(range(259)) for i in range(0,7*259,259))
    pairs={}
    for name,threshold in runner.THRESHOLDS.items():
        metrics={'miou_all':{'difference':threshold,'paired_view_bootstrap_95_interval':[1e-7,.01]}}
        metrics.update({n+'_iou':{'difference':0.} for n in ('background','deck','stay_cable','tower','foundation')})
        pairs[name]={'metrics':metrics}
    assert all(runner.adoption_clauses(pairs).values())
    pairs['E']['metrics']['miou_all']['difference']=0.
    assert not runner.adoption_clauses(pairs)['E_miou_gain']
    with pytest.raises(ValueError,match='three matched'):
        runner.adoption_clauses({**pairs,'old_integrated':pairs['E']})
    assert runner.SPEC['objective']=={'final_weighted_ce':1.,'final_lovasz':.2,'raw_weighted_ce':.25,'raw_lovasz':0.,'residual_square_mean':.001}


def test_make_optimizer_rejects_missing_group_epsilon_override():
    scene=Scene();_,groups=runner.configure_parameters(scene,'integrated',field_module.SemanticPartitionField)
    groups[1].pop('eps')
    with pytest.raises(ValueError,match='group contract'):runner.make_optimizer(groups)
