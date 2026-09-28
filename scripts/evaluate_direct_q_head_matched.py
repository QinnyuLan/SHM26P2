"""Evaluation-only recovery of fixed direct-q head endpoints under historical E numerics.

No training/retry/threshold change. The original failed experiment stays intact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT=Path('/home/sky/workspace/SHM2026')
TRAIN=Path('/mnt/data/SHM2026/runs/direct_q_head_matched_v1')
TRAIN_PLAN_SHA='c5b59eb33091d532a6a5f2490613d062c241fb4724d485e88bfbc87c9aee41f6'
DIAGNOSTIC=Path('/mnt/data/SHM2026/runs/direct_q_eval_numerics_diagnostic_v1')
DIAGNOSTIC_EXEC_SHA='ef507a290e9ae7c1a6d11f17cb225695ab5121e5659d0fb69c5fe3e7902a426d'
PRODUCER=Path(__file__).resolve().parent/'train_direct_q_head_matched.py'
if Path(__file__).resolve().parent==ROOT/'scripts' or not PRODUCER.exists():
    PRODUCER=TRAIN/'source_snapshot/train_direct_q_head_matched.py'
module_spec=importlib.util.spec_from_file_location('direct_q_training_producer',PRODUCER)
producer=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(producer)
require,read,sha,write,files=(getattr(producer,n) for n in ('require','read','sha','write','files'))
ARMS,READOUTS,THRESHOLDS=producer.ARMS,producer.READOUTS,producer.THRESHOLDS
BASE_SHA,TEACHER_SHA=producer.BASE_SHA,producer.TEACHER_SHA
load_base,load_q,validate_delta=(getattr(producer,n) for n in ('load_base','load_q','validate_delta'))
digest_tensors,tensor_sha,prediction_barrier,adoption_clauses=(getattr(producer,n) for n in
    ('digest_tensors','tensor_sha','prediction_barrier','adoption_clauses'))
EVAL_NUMERICS=dict(producer.NUMERICS,cudnn_allow_tf32=True)
SPEC={'protocol':'direct_q_head_evaluation_recovery_v1','training_plan_sha256':TRAIN_PLAN_SHA,
      'training_unchanged':True,'training_updates':0,'evaluation_numerics':EVAL_NUMERICS,
      'training_numerics':producer.NUMERICS,'diagnostic_execution_sha256':DIAGNOSTIC_EXEC_SHA,
      'evaluation':producer.SPEC['evaluation'],'readouts':list(READOUTS),'candidate':'optimized_q',
      'gain_thresholds':THRESHOLDS,'class_guard':producer.SPEC['class_guard'],
      'bootstrap':producer.SPEC['bootstrap'],'internal_seconds':300,'external_seconds':360,
      'only_changes':'evaluation convolution TF32 restored to historical E; separate output/training roots',
      'original_status':'failed after both 2000-step arms and first native baseline mask; not reclassified'}


def evaluate(plan,modules):
    import cv2
    import torch

    from bridge_rgs.direct_q_render import render_direct_q
    
    helper=modules['evaluate_projective_deck_pooling']; official=modules['bridge_rgs.official_evaluate']
    compare=modules['compare_official_evaluations']; output=Path(plan['output'])/'evaluation'; output.mkdir()
    training_plan=read(plan['training_plan']['path']); training_root=Path(training_plan['output'])
    old=read(plan['prior_plan_path']); prior=read(plan['prior_receipt_path']); old_e=read(plan['prior_e_metrics'])
    require(old['teachers']['old']['sha256']==TEACHER_SHA,'Wrong frozen teacher')
    views=old['views']; require(len(views)==50 and sum(v['source_annotation_path'] is not None for v in views)==41,'Wrong evaluation population')
    by_old={(r['arm'],r['name']):r for r in prior['predictions']}
    rgb_rows={r['name']:{k:r[k] for k in ('name','width','height','rgb_pixels','psnr','ssim','lpips')} for r in old_e['views']}
    compare._validate(old_e)
    require(official.official_fingerprint([{k:v[k] for k in ('camera','source_image_sha256','source_annotation_sha256','rasterized_mask_sha256')}
            for v in views])==old['reference_fingerprint']==old_e['official_evaluation_fingerprint'],'Official fingerprint changed')
    records=[]; baseline=[]; beginning=time.monotonic(); scene_calls=0; shader_calls=0; native_hashes={}; adapter_checks=[]
    # Reproduce unchanged E semantics before evaluating any continued head.
    for arm in ('baseline',)+READOUTS:
        scene,state,_=load_base(plan)
        q=None if arm=='baseline' else load_q(scene,arm,plan)
        if arm in ARMS:
            receipt=read(training_root/arm/'training_receipt.json')
            require(receipt['status']=='completed' and receipt['steps']==2000 and sha(receipt['delta_path'])==receipt['delta_sha256'],'Training incomplete')
            delta=torch.load(receipt['delta_path'],map_location='cpu',weights_only=False); validate_delta(delta,arm,training_plan,state,q)
            scene.refiner.load_state_dict(delta['refiner_state'],strict=True)
        scene.eval().requires_grad_(False)
        before=digest_tensors(scene.state_dict()); head_before=digest_tensors(scene.state_dict(),True)
        q_hash=None if q is None else tensor_sha(q)
        with torch.no_grad():
            for view in views:
                name,camera=view['name'],view['camera']; a,e=by_old['A',name],by_old['E',name]
                K,w,h,back=official.distortion_render_grid(camera['K'],camera['distortion'],camera['width'],camera['height'],'legacy_mixed_v1')
                cache=prior['h3_canvases'][name]
                require([w,h]==cache['boundary']['canvas'] and np.array_equal(K,np.asarray(cache['boundary']['canvas_K'],np.float32)),'Teacher grid changed')
                kt=torch.tensor(K,device='cuda'); pt=torch.tensor(camera['w2c'],dtype=torch.float32,device='cuda')
                if arm=='baseline':
                    result=scene.render(kt,pt,w,h,degree=3,absgrad=False,geometry_grad=False,refinement_grad_to_field=False)
                    native_hashes[name]={key:tensor_sha(result[key]) for key in ('p3d','probabilities')}
                else:
                    result=render_direct_q(scene,q,kt,pt,w,h)
                    if arm=='original_q_zero':
                        checks={key:tensor_sha(result[key])==native_hashes[name][key] for key in ('p3d','probabilities')}
                        adapter_checks.append({'name':name,'checks':checks})
                        if not all(checks.values()):
                            write(output/'adapter_mismatch.json',{'status':'failed','records':adapter_checks})
                            raise ValueError('Original q adapter differs from native H3; no attribution to optimized q')
                scene_calls+=1; shader_calls+=q is not None
                canvas=result['rgb'].clamp(0,1).cpu().numpy()
                helper.save_png(output/arm/'canvas_rgb'/name,(canvas[...,::-1]*255).round().astype(np.uint8),cache['canvas_rgb'])
                rgb=canvas if back is None else cv2.remap(canvas,back[...,0],back[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                helper.save_png(output/arm/'h3_rgb'/name,(rgb[...,::-1]*255).round().astype(np.uint8),{'path':a['rgb'],'sha256':a['rgb_sha256']})
                helper.require_record(a['teacher_soft_canvas']); helper.require_record({'path':e['rgb'],'sha256':e['rgb_sha256']})
                teacher=np.load(a['teacher_soft_canvas']['path'],allow_pickle=False)
                final=result['probabilities'].cpu().numpy(); raw=result['p3d'].cpu().numpy()
                require(teacher.shape==final.shape==(h,w,5) and teacher.dtype==np.float32 and np.isfinite(teacher).all()
                        and (teacher>=0).all() and np.allclose(teacher.sum(-1),1,atol=2e-5,rtol=0),'Invalid teacher cache')
                joint=helper.warp_probabilities((final+teacher)*np.float32(.5),back).argmax(-1).astype(np.uint8)
                if arm=='baseline':
                    baseline.append(helper.save_png(output/'baseline'/'joint_mask'/name,joint,{'path':e['mask'],'sha256':e['mask_sha256']}))
                else:
                    masks={'joint':helper.save_png(output/arm/'joint_mask'/name,joint)}
                    for kind,soft in (('scene',final),('raw',raw)):
                        masks[kind]=helper.save_png(output/arm/(kind+'_mask')/name,helper.warp_probabilities(soft,back).argmax(-1).astype(np.uint8))
                    records.append({'arm':arm,'name':name,'masks':masks,
                                    'soft_canvas':helper.save_soft(output/arm/'soft'/f'{Path(name).stem}.npy',final),
                                    'teacher_soft_canvas':a['teacher_soft_canvas'],'teacher_input_rgb_exact':True,
                                    'rgb_reference':{'path':e['rgb'],'sha256':e['rgb_sha256']}})
                del result
        require(digest_tensors(scene.state_dict())==before and digest_tensors(scene.state_dict(),True)==head_before
                and (q is None or tensor_sha(q)==q_hash), 'Evaluation changed frozen field/head/q')
        del scene,state,q; torch.cuda.empty_cache()
    prediction_barrier(records,baseline,adapter_checks,views)
    predictions_finished_utc = datetime.now(UTC).isoformat()
    write(output/'predictions_receipt.json',{'status':'all_600_masks_before_GT','records':records,'baseline_E_exact':baseline,'original_q_zero_native_exact':adapter_checks,
          'scene_calls':scene_calls,'direct_q_shader_calls':int(shader_calls),'annotation_payload_reads':0,
          'predictions_finished_utc':predictions_finished_utc})
    scoring_started_utc = datetime.now(UTC).isoformat()
    rows={(a,k):[] for a in READOUTS for k in ('joint','scene','raw')}; by_new={(r['arm'],r['name']):r for r in records}; reads=0
    for view in views:
        name=view['name']; truth=None
        if view['source_annotation_path'] is not None:
            payload=Path(view['source_annotation_path']).read_bytes(); reads+=1
            require(hashlib.sha256(payload).hexdigest()==view['source_annotation_sha256'],'Annotation changed')
            truth=official.rasterize_official_annotation(payload,view['camera']['width'],view['camera']['height'])
            require(hashlib.sha256(truth.tobytes()).hexdigest()==view['rasterized_mask_sha256'],'GT rasterizer changed')
        for (arm,kind), destination in rows.items():
            row=dict(rgb_rows[name])
            if truth is not None:
                prediction=cv2.imread(by_new[arm,name]['masks'][kind]['path'],cv2.IMREAD_UNCHANGED); keep=truth!=255
                cm=np.bincount(5*truth[keep].astype(np.int64)+prediction[keep],minlength=25).reshape(5,5)
                row.update(confusion_matrix=cm.tolist(),semantic_pixels=int(keep.sum()),semantic_ignore_pixels=int((~keep).sum()))
            destination.append(row)
    require(reads==41,'Wrong annotation count'); metrics={}; metric_records={}
    inference={'protocol':'direct_q_head_matched_endpoint_v1','teacher_weight':.5,'RGB':'E byte/score reuse',
               'baseline_E_exact':True,'camera_source':'same frozen H3 context; original classifier q or fixed optimized q with matched head'}
    for (arm,kind),view_rows in rows.items():
        pooled=sum(np.asarray(r['confusion_matrix'],np.int64) for r in view_rows if 'confusion_matrix' in r); iou=compare._iou(pooled)
        value={'evaluation_family':old['evaluation_family'],'scoring_protocol':old['scoring_protocol'],
               'official_evaluation_fingerprint':old['reference_fingerprint'],'inference_protocol':dict(inference,kind=kind),
               'validation_views':50,'semantic_validation_views':41,'views':view_rows,'confusion_matrix':pooled.tolist(),
               'iou':iou.tolist(),'miou_all':float(np.nanmean(iou)),'miou_foreground':float(np.nanmean(iou[1:])),
               **{k:float(np.mean([r[k] for r in view_rows])) for k in ('psnr','ssim','lpips')},
               'provenance':{'base_checkpoint_sha256':BASE_SHA,'teacher_cache_checkpoint_sha256':TEACHER_SHA if kind=='joint' else None,
                             'training_delta':read(training_root/arm/'training_receipt.json')['delta_sha256'] if arm in ARMS else None,
                             'q_readout':arm,'optimized_q_sidecar':plan['q_delta'] if arm.startswith('optimized_q') else None,
                             'rgb_metrics':{'path':plan['prior_e_metrics'],'sha256':sha(plan['prior_e_metrics'])},
                             'single_shared_geometry_system':False}}
        compare._validate(value); path=output/arm/('official_metrics.json' if kind=='joint' else kind+'_official_metrics.json')
        write(path,value); metrics[arm,kind]=value; metric_records[arm+'_'+kind]={'path':str(path),'sha256':sha(path)}
    pairs={}; pair_records={}
    for reference in THRESHOLDS:
        pair=compare.paired_official_comparison(old_e if reference=='E' else metrics[reference,'joint'],metrics['optimized_q','joint'],repeats=5000,seed=20260926)
        pair.update(candidate_arm='optimized_q',reference_arm=reference); pairs[reference]=pair
        path=output/f'paired_optimized_q_minus_{reference}.json';write(path,pair);pair_records[reference]={'path':str(path),'sha256':sha(path)}
    secondary_records = {}
    # Pre-registered zero-step descriptions, never candidates for endpoint selection.
    for candidate,reference in (('optimized_q_zero','original_q_zero'),('optimized_q','optimized_q_zero')):
        pair=compare.paired_official_comparison(metrics[reference,'joint'],metrics[candidate,'joint'],repeats=5000,seed=20260926)
        pair.update(candidate_arm=candidate,reference_arm=reference,descriptive_only=True)
        path=output/f'paired_{candidate}_minus_{reference}.json';write(path,pair)
        secondary_records[candidate+'_minus_'+reference]={'path':str(path),'sha256':sha(path)}
    clauses=adoption_clauses(pairs); gate={'passed':all(clauses.values()),'clauses':clauses,'thresholds':THRESHOLDS,
            'decision':'eligible_for_review_not_automatic_adoption' if all(clauses.values()) else 'stop_fixed_direct_q_head_candidate',
            'rgb':'unchanged E; do not attribute prior RGB gains to direct-q/head adaptation'}
    write(output/'system_gate.json',gate)
    result={'status':'completed','scene_calls':scene_calls,'direct_q_shader_calls':int(shader_calls),'teacher_calls':0,
            'baseline_E_masks_byte_exact':50,'original_q_zero_native_exact':adapter_checks,'new_masks':600,'annotation_payload_reads':reads,'gt_rgb_payload_reads':0,
            'metrics':metric_records,'comparisons':pair_records,'secondary_comparisons':secondary_records,'gate':gate,'gate_sha256':sha(output/'system_gate.json'),
            'plan_sha256':sha(Path(plan['output'])/'plan.json'),
            'predictions_receipt_sha256':sha(output/'predictions_receipt.json'),
            'predictions_finished_utc':predictions_finished_utc,'scoring_started_utc':scoring_started_utc,
            'scoring_finished_utc':datetime.now(UTC).isoformat(),
            'elapsed_seconds':time.monotonic()-beginning,'timing_scope':'semantic inference with teacher/E RGB caches; not deployment FPS'}
    write(output/'execution_receipt.json',result)
    return result

def validate_completed_training(plan, execution, launch, receipts):
    require(plan['specification']==producer.SPEC and producer.sample_order()==plan['training_order'], 'Training protocol changed')
    require(execution['status']=='failed' and launch['status']=='failed' and launch['exit_code']==1
            and execution['new_val_annotation_payload_reads']==0 and 'evaluation' not in execution
            and execution['error']=='New rendered RGB bytes differ from teacher cache source', 'Wrong preserved failure')
    require(execution['numerics_actual']==producer.NUMERICS and execution['numerics_restored'] is True,
            'Wrong actual training numerics')
    producer.validate_matched_receipts(receipts)
    require(receipts==execution['training'], 'Saved completed arms not the failed-wrapper endpoints')
    require(execution['wiring_preflight']['passed'] is True, 'Missing unchanged initial wiring proof')


def prepare(output):
    output=Path(output).resolve();require(output.is_relative_to('/mnt/data') and not output.exists(),'New output required')
    require(sha(TRAIN/'plan.json')==TRAIN_PLAN_SHA and sha(DIAGNOSTIC/'execution_receipt.json')==DIAGNOSTIC_EXEC_SHA,'Wrong bound prior')
    train,execution,launch=(read(TRAIN/n) for n in ('plan.json','execution_receipt.json','launch_receipt.json'))
    receipts=[read(TRAIN/arm/'training_receipt.json') for arm in ARMS]
    validate_completed_training(train,execution,launch,receipts)
    require(launch['plan_sha256']==execution['plan_sha256']==TRAIN_PLAN_SHA
            and launch['execution_receipt_sha256']==sha(TRAIN/'execution_receipt.json'),'Original failure not bound')
    diagnostic,diaglaunch=(read(DIAGNOSTIC/n) for n in ('execution_receipt.json','launch_receipt.json'))
    require(diagnostic['status']=='completed' and diagnostic['hypothesis_confirmed'] is True
            and diagnostic['state_exact'] and diagnostic['q_exact'] and diagnostic['numerics_restored']
            and diagnostic['target_reads']==0 and diagnostic['scene_calls']==3
            and diaglaunch['natural_completion'] is True and diaglaunch['exit_code']==0
            and diaglaunch['execution_receipt_sha256']==DIAGNOSTIC_EXEC_SHA,'Diagnostic did not establish fixed recovery')
    inputs=dict(train['input_hashes'])
    def bind(path,expected=None):
        path=Path(path).resolve();digest=sha(path);require(expected is None or digest==expected,'Changed lineage')
        inputs[str(path)]=digest;return {'path':str(path),'sha256':digest}
    lineage={str(root/name):bind(root/name) for root,names in
             ((TRAIN,('plan.json','execution_receipt.json','launch_receipt.json','wiring_preflight.json')),
              (DIAGNOSTIC,('plan.json','execution_receipt.json','launch_receipt.json')))
             for name in names}
    for arm,receipt in zip(ARMS,receipts,strict=True):
        bind(TRAIN/arm/'training_receipt.json');bind(receipt['delta_path'],receipt['delta_sha256'])
        bind(TRAIN/arm/'steps.jsonl',receipt['training_log_sha256'])
    source=Path(train['source_snapshot']);require(files(source)==train['source_hashes'],'Original frozen package changed')
    snapshot=output/'source_snapshot';shutil.copytree(source,snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for path in (Path(__file__),ROOT/'tests/test_direct_q_head_evaluation_recovery.py'):
        shutil.copy2(path,snapshot/path.name)
    plan=dict(train,output=str(output),source_snapshot=str(snapshot),source_hashes=files(snapshot),input_hashes=inputs,
              specification=SPEC,numerics=EVAL_NUMERICS,training_plan=bind(TRAIN/'plan.json',TRAIN_PLAN_SHA),training_lineage=lineage,
              numerical_recovery=bind(DIAGNOSTIC/'execution_receipt.json',DIAGNOSTIC_EXEC_SHA),
              execution_budget={'internal_seconds':300,'external_seconds':360},status='prepared_evaluation_only',
              prepare_checkpoint_loads=0,prepare_target_decodes=0)
    for path,digest in inputs.items():require(sha(path)==digest,f'Changed bound input: {path}')
    write(output/'plan.json',plan)
    print(json.dumps({'plan':str(output/'plan.json'),'sha256':sha(output/'plan.json'),'sources':len(plan['source_hashes']),'inputs':len(inputs)}))


def verify(plan):
    require(plan['specification']==SPEC and Path(__file__).resolve()==Path(plan['source_snapshot'])/Path(__file__).name,'Frozen recovery only')
    require(files(plan['source_snapshot'])==plan['source_hashes'],'Recovery source changed')
    require(sha(plan['training_plan']['path'])==TRAIN_PLAN_SHA,'Training plan changed')
    for path,digest in {**plan['input_hashes'],**plan['installed_sources']}.items():require(sha(path)==digest,'Bound recovery input changed')
    for name,value in plan['runtime_versions'].items():require(producer.importlib.metadata.version(name)==value,'Runtime changed')
    for name,value in plan['environment'].items():require(os.environ.get(name)==value,'Environment changed')
    require(sha(plan['expected_gsplat_binary']['path'])==plan['expected_gsplat_binary']['sha256'],'Binary changed')


def execute(path,expected):
    require(sha(path)==expected,'Wrong recovery plan');plan=read(path);output=Path(plan['output'])
    require(not (output/'execution_receipt.json').exists() and not (output/'execution_started.json').exists(),'One evaluation attempt only')
    write(output/'execution_started.json',{'plan_sha256':expected,'started_utc':datetime.now(UTC).isoformat()})
    start=time.monotonic();previous=None;torch=None
    report={'status':'running','plan_sha256':expected,'training_updates':0,'teacher_calls':0,
            'training_plan':plan['training_plan'],'training_lineage':plan['training_lineage'],
            'original_training_wrapper_status':'failed','numerical_recovery':plan['numerical_recovery'],
            'prior_q_optimization_cost':plan['prior_q_optimization_cost'],'new_val_annotation_payload_reads':0}
    def expired(*_):raise TimeoutError('Fixed evaluation recovery budget exhausted')
    handler=signal.signal(signal.SIGALRM,expired);signal.alarm(300)
    try:
        verify(plan);report['gpu_before']=producer.gpu_inventory();sys.path.insert(0,plan['source_snapshot'])
        modules,imports=producer.imports_from_snapshot(plan);report['actual_imports']=imports
        torch=modules['bridge_rgs.train'].torch;torch.set_num_threads(4)
        previous=producer.numerical_flags(torch,EVAL_NUMERICS);report['numerics_actual']=producer.read_numerical_flags(torch)
        require(report['numerics_actual']==EVAL_NUMERICS and not torch.is_autocast_enabled(),'Legacy evaluation numerics not applied')
        report['evaluation']=evaluate(plan,modules)
        report['new_val_annotation_payload_reads']=report['evaluation']['annotation_payload_reads']
        report['actual_imports']=producer.imports_from_snapshot(plan)[1]
        report['loaded_gsplat_binary_sources']=producer.loaded_gsplat_binaries()
        require(plan['expected_gsplat_binary'] in report['loaded_gsplat_binary_sources'].values(),'Wrong actual binary')
        verify(plan);report.update(status='completed',bound_sources_inputs_unchanged=True)
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,handler)
        failure=None;report['numerics_restored']=None
        if previous is not None:
            try:
                producer.numerical_flags(torch,previous);report['numerics_after_restore']=producer.read_numerical_flags(torch)
                report['numerics_restored']=report['numerics_after_restore']==previous
                require(report['numerics_restored'],'Numerics not restored')
            except BaseException as exc:  # noqa: BLE001 - re-raised after preserving receipt
                failure=exc;report.update(status='failed',restoration_error=str(exc),numerics_restored=False)
        report['elapsed_seconds']=time.monotonic()-start;write(output/'execution_receipt.json',report)
        if failure is not None:raise failure


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare',type=Path);choice.add_argument('--execute',type=Path)
    parser.add_argument('--expected-plan-sha256');args=parser.parse_args()
    require(os.environ.get('PYTHONDONTWRITEBYTECODE')=='1','Disable bytecode')
    if args.prepare:prepare(args.prepare)
    else:
        require(args.expected_plan_sha256 is not None,'Expected plan SHA required');execute(args.execute,args.expected_plan_sha256)
