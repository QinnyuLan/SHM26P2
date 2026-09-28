"""Only output roots and historical eval numerics change; no real scene/GT reads."""
import ast
import copy
import importlib.util
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parent
PATH=HERE/'evaluate_direct_q_head_matched.py'
if not PATH.exists():PATH=HERE.parent/'scripts/evaluate_direct_q_head_matched.py'
spec=importlib.util.spec_from_file_location('dq_eval_recovery_test',PATH)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def function(source,name):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(source,node)


def test_evaluation_math_exact_old_except_three_explicit_training_root_sites():
    old=function(r.PRODUCER.read_text(),'evaluate')
    expected=old.replace("    old=read(plan['prior_plan_path']);", "    training_plan=read(plan['training_plan']['path']); training_root=Path(training_plan['output'])\n    old=read(plan['prior_plan_path']);")
    expected=expected.replace("Path(plan['output'])/arm/'training_receipt.json'","training_root/arm/'training_receipt.json'")
    expected=expected.replace('validate_delta(delta,arm,plan,state,q)','validate_delta(delta,arm,training_plan,state,q)')
    actual=function(PATH.read_text(),'evaluate')
    assert ast.dump(ast.parse(actual))==ast.dump(ast.parse(expected))
    assert r.SPEC['gain_thresholds']==r.producer.SPEC['gain_thresholds']
    assert r.SPEC['evaluation']==r.producer.SPEC['evaluation']
    assert r.SPEC['bootstrap']==r.producer.SPEC['bootstrap']


def training_fixture():
    base={'status':'completed','steps':2000,'sample_names_sha256':'s','refiner_initial_sha256':'h',
          'frozen_nonrefiner_sha256':'f','training_camera_sha256':'c','parameter_count_refiner':3,
          'optimizer_groups':[{'name':'head','eps':1e-8}]}
    receipts=[dict(base,arm=a,q_initial_sha256=a,q_final_sha256=a) for a in r.ARMS]
    p={'specification':r.producer.SPEC,'training_order':r.producer.sample_order()}
    e={'status':'failed','new_val_annotation_payload_reads':0,
       'error':'New rendered RGB bytes differ from teacher cache source',
       'numerics_actual':r.producer.NUMERICS,'numerics_restored':True,
       'training':receipts,'wiring_preflight':{'passed':True}}
    return p,e,{'status':'failed','exit_code':1},receipts


def test_failed_wrapper_preserved_with_only_completed_exact_endpoints():
    p,e,l,receipts=training_fixture();r.validate_completed_training(p,e,l,receipts)
    bad=copy.deepcopy(e);bad['status']='completed'
    with pytest.raises(ValueError):r.validate_completed_training(p,bad,l,receipts)
    bad=copy.deepcopy(receipts);bad[0]['steps']=1999
    with pytest.raises(ValueError):r.validate_completed_training(p,e,l,bad)


def test_training_numerics_remain_false_eval_true_only():
    assert r.EVAL_NUMERICS==dict(r.producer.NUMERICS,cudnn_allow_tf32=True)
    assert r.producer.NUMERICS['cudnn_allow_tf32'] is False
    assert r.SPEC['training_updates']==0 and r.SPEC['candidate']=='optimized_q'
    p,e,l,receipts=training_fixture();e['numerics_actual']=r.EVAL_NUMERICS
    with pytest.raises(ValueError):r.validate_completed_training(p,e,l,receipts)


def test_serialize_complete_recovery_contract(tmp_path):
    path=tmp_path/'report.json'
    r.write(path,{'specification':r.SPEC,'original_training_wrapper_status':'failed',
                  'training_plan':{'path':'original/plan.json','sha256':r.TRAIN_PLAN_SHA}})
    assert r.read(path)['original_training_wrapper_status']=='failed'
