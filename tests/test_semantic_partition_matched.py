"""Small CPU contracts for the draft matched experiment; no scene/data reads."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve()
SCRIPT = HERE.with_name('train_semantic_partition_matched.py')
if not SCRIPT.exists():
    SCRIPT = HERE.parents[1]/'scripts/train_semantic_partition_matched.py'
spec=importlib.util.spec_from_file_location('matched_partition_runner',SCRIPT)
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_population_and_independent_complete_epoch_order():
    train=[{'name':f'{i:03}.png','split':'train','mask_path':'mask' if i<259 else None} for i in range(350)]
    val=[{'name':f'val{i}.png','split':'val','mask_path':'must not read'} for i in range(50)]
    assert len(runner.population({'views':list(reversed(train+val))}))==259
    with pytest.raises(ValueError,match='population'):
        runner.population({'views':train[:-1]})
    np.random.seed(7); before=np.random.get_state()
    order=runner.sample_order()
    after=np.random.get_state()
    assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    assert len(order)==2000 and order==runner.sample_order()
    for start in range(0,7*259,259):
        assert sorted(order[start:start+259])==list(range(259))


class ToyScene(nn.Module):
    def __init__(self):
        super().__init__()
        self.splats=nn.ParameterDict({'sem_features':nn.Parameter(torch.arange(12.).reshape(3,4))})
        self.semantic_decoder=nn.Linear(4,5)
        self.refiner=nn.Linear(5,5)
        self.background_logits=nn.Parameter(torch.ones(3))


def test_scope_groups_and_equal_partition_initialization():
    from bridge_rgs.partition_field import SemanticPartitionField
    scene=ToyScene(); initial=copy.deepcopy(scene.state_dict()); signatures=[]
    for arm in runner.ARMS:
        scene.load_state_dict(initial)
        field,groups=runner.configure_parameters(scene,arm,SemanticPartitionField)
        assert all(p.requires_grad==n.startswith('refiner.') for n,p in scene.named_parameters())
        assert [g['lr'] for g in groups]==([.0003] if arm=='refiner_only' else [.0003,.01,.001])
        parameters=[p for g in groups for p in g['params']]
        assert len(parameters)==len({id(p) for p in parameters})
        if field is None:
            continue
        assert torch.equal(field.inside_logits,field.outside_logits)
        assert torch.equal(field.inside_logits,scene.semantic_decoder(scene.splats['sem_features']))
        signatures.append({n:t.detach().clone() for n,t in field.state_dict().items()})
    assert all(torch.equal(signatures[0][n],sig[n]) for sig in signatures[1:] for n in sig)


def test_frozen_digest_covers_geometry_and_feature_values():
    scene=ToyScene(); frozen=runner.digest_tensors(scene.state_dict()); head=runner.digest_tensors(scene.state_dict(),True)
    with torch.no_grad(): scene.refiner.weight.add_(1)
    assert runner.digest_tensors(scene.state_dict())==frozen
    assert runner.digest_tensors(scene.state_dict(),True)!=head
    with torch.no_grad(): scene.splats['sem_features'][0,0]+=1
    assert runner.digest_tensors(scene.state_dict())!=frozen


def test_objective_raw_quarter_and_original_final_formula_gradients():
    from bridge_rgs.losses import semantic_loss
    torch.manual_seed(5)
    logits=torch.randn(7,9,5,requires_grad=True); residual=torch.randn(7,9,5,requires_grad=True)*.2
    raw=logits.softmax(-1); final=(raw.log()+residual).softmax(-1)
    labels=torch.randint(0,5,(7,9)); labels[0,0]=255
    valid=torch.ones(7,9);valid[-1]=0
    weights=torch.tensor(runner.WEIGHTS)
    loss,parts=runner.loss_terms({'p3d':raw,'probabilities':final,'residual':residual},labels,valid,weights,semantic_loss)
    expected=semantic_loss(final,labels,valid,weights,.2)+.25*semantic_loss(raw,labels,valid,weights,0)+.001*residual.square().mean()
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(loss,logits,retain_graph=True)[0],torch.autograd.grad(expected,logits)[0],rtol=0,atol=0)
    assert set(parts)=={'final','raw_ce','residual_square'}


def test_partition_dispatch_does_not_detach_new_prior():
    class Scene:
        def render(self,*args,**kwargs):
            assert kwargs=={'degree':3,'absgrad':False,'geometry_grad':False,'refinement_grad_to_field':False}
            return 'reference'
    scene=Scene(); field=object(); calls=[]
    def partition(*args,**kwargs):
        calls.append((args,kwargs)); return 'partition'
    assert runner.render_training(scene,None,partition,1,2,3,4)=='reference'
    assert runner.render_training(scene,field,partition,1,2,3,4)=='partition'
    assert calls==[((scene,field,1,2,3,4),{})]


def good_preflight():
    plan={'specification':{'names':['002.png','118.png'],'modes':['marginal','point','integrated'],'optimizer_steps':0}}
    receipt={'status':'passed','inputs_and_sources_unchanged':True}
    analysis={'status':'passed','records':[{'name':n,'mode':a,'passed':True,'gates':{key:True for key in ('initial_p3d','initial_final','alpha','rgb','simplex','gradients','raw_gradients','head_gradient')},
                    'gradients':{'example':{'finite':True,'norm':1.}},'raw_gradients':{'example':{'finite':True,'norm':1.}},
                    'head_gradient':{'finite':True,'squared_norm':1.}}
                    for n in ['002.png','118.png'] for a in runner.ARMS[1:]]}
    return plan,receipt,analysis,{'status':'completed','exit_code':0,'natural_completion':True}


def test_prepare_gate_rejects_failed_or_missing_real_integration_without_version_guess():
    p,r,a,launch=good_preflight();runner.validate_preflight(p,r,a,launch)
    no_status = {k:v for k,v in launch.items() if k != 'status'}
    runner.validate_preflight(p,r,a,no_status)
    with pytest.raises(ValueError,match='naturally'):
        runner.validate_preflight(p,r,a,dict(launch,status='failed'))
    r['status']='failed'
    with pytest.raises(ValueError,match='did not pass'):runner.validate_preflight(p,r,a,launch)
    p,r,a,launch=good_preflight();a['records'].pop()
    with pytest.raises(ValueError,match='Missing'):runner.validate_preflight(p,r,a,launch)
    p,r,a,launch=good_preflight();a['records'][1]['gates']['rgb']=False
    with pytest.raises(ValueError,match='Failed'):runner.validate_preflight(p,r,a,launch)
    p,r,a,launch=good_preflight();launch['natural_completion']=False
    with pytest.raises(ValueError,match='naturally'):runner.validate_preflight(p,r,a,launch)


def test_joint_adoption_has_all_controls_and_old_endpoint_guards():
    pairs={}
    for ref,threshold in runner.THRESHOLDS.items():
        metrics={'miou_all':{'difference':threshold,'paired_view_bootstrap_95_interval':[1e-9,.01]}}
        metrics.update({c+'_iou':{'difference':0.} for c in ('background','deck','stay_cable','tower','foundation')})
        pairs[ref]={'metrics':metrics}
    assert all(runner.adoption_clauses(pairs).values())
    pairs['E']['metrics']['miou_all']['paired_view_bootstrap_95_interval'][0]=0
    assert not runner.adoption_clauses(pairs)['E_miou_ci_lower_positive']
    pairs['point']['metrics']['stay_cable_iou']['difference']=-.001001
    assert not runner.adoption_clauses(pairs)['point_stay_cable_guard']
    json.dumps(runner.adoption_clauses(pairs),allow_nan=False)
    with pytest.raises(ValueError,match='three matched'):
        runner.adoption_clauses({k:v for k,v in pairs.items() if k!='E'})


def test_no_wall_clock_budget_or_gpu_execution_is_implicitly_selected():
    assert 'internal_seconds' not in runner.SPEC and 'external_seconds' not in runner.SPEC
    assert runner.SPEC['steps']==2000 and runner.SPEC['endpoint'].startswith('last2000')
    assert runner.SPEC['objective']['raw_weighted_ce']==.25
    assert runner.SPEC['candidate']=='integrated'
    assert runner.ARMS==('refiner_only','marginal','point','integrated')
