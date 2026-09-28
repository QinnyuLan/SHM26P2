"""Independent CPU reconstruction after the completed epsilon-only experiment.

Never imports the producer, its statistics/gates, bridge_rgs, or a model. Gaussian
checkpoints are tensor dictionaries loaded with map_location=cpu and mmap=True.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import signal
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

ARMS=('refiner_only','marginal','point','integrated')
KINDS=('joint','scene','raw')
CLASSES=('background','deck','stay_cable','tower','foundation')
THRESHOLDS={'refiner_only':.0015,'marginal':.001,'point':.001,'E':.002}
BASE_SHA='22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
TEACHER_SHA='00f5b84ac9a56c39512c5b8e43f70110397923feea1a2b4c78bdffd1f5524bff'
MANIFEST_SHA='551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa'
PROTOCOL='semantic_partition_eps_matched_v1'
FORMAT='semantic_partition_eps_matched_delta_v1'
OLD_RUN=Path('/mnt/data/SHM2026/runs/semantic_partition_matched_v2')
OLD_PLAN_SHA='f10d266083999abd1635ea8cb91e7407247dafba113f525a6ec7c790e3e4cc1d'
HEAD_EPS=1e-8
FIELD_EPS=1e-15
WEIGHTS=[.45604488253593445,.9227690100669861,.8530434370040894,1.1935913562774658,1.5745513439178467]


def check(value,message):
    if not value:
        raise AssertionError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def tree(path):
    return {str(p.relative_to(path)):sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}


def tensor_digest(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        array=tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value,dict):
        return all(finite(v) for v in value.values())
    if isinstance(value,(tuple,list)):
        return all(finite(v) for v in value)
    return True


def expected_order():
    generator=np.random.default_rng(42); values=[]
    for _ in range(8):
        values.extend(generator.permutation(259).tolist())
    return values[:2000]


def validate_settings(spec):
    check(spec['protocol']==PROTOCOL and spec['arms']==list(ARMS) and spec['candidate']=='integrated','Wrong experiment')
    check(spec['base_sha256']==BASE_SHA and spec['steps']==2000 and spec['seed']==42 and spec['train_count']==259,'Changed fixed training')
    check(spec['objective']=={'final_weighted_ce':1.,'final_lovasz':.2,'raw_weighted_ce':.25,
                             'raw_lovasz':0.,'residual_square_mean':.001},'Changed objective')
    check(spec['optimizer']=={'type':'Adam','betas':[.9,.999],'head_eps':HEAD_EPS,'field_eps':FIELD_EPS,'weight_decay':0,
                             'refiner_lr':.0003,'endpoint_logits_lr':.01,'shape_lr':.001},'Changed optimizer')
    check(spec['class_weights_fp32']==WEIGHTS and spec['gain_thresholds']==THRESHOLDS,'Changed weights/gains')
    check(spec['class_guard']=={'stay_cable':-.001,'others':-.002}
          and spec['bootstrap']=={'repeats':5000,'seed':20260926},'Changed class/bootstrap gate')
    check(spec['new_gaussians']==0 and spec['geometry_rgb_features_cameras_frozen']
          and spec['teacher_training'] is False and spec['validation_during_training'] is False,'Changed scope')
    check(spec['evaluation']['teacher_sha256']==TEACHER_SHA and spec['evaluation']['teacher_weight']==.5
          and spec['evaluation']['views']==50 and spec['evaluation']['annotations']==41,'Changed evaluation')


def function_ast(path,name):
    matches=[node for node in ast.parse(Path(path).read_text()).body
             if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==name]
    check(len(matches)==1,'Missing/duplicate scoring function '+name)
    return ast.dump(matches[0],include_attributes=False)


def validate_epsilon_only_specs(old,new):
    """Only the declared optimization change; do not accept a new renderer/loss."""
    validate_settings(new)
    check(old['protocol']=='semantic_partition_matched_v1','Wrong predecessor protocol')
    check(old['optimizer']=={'type':'Adam','betas':[.9,.999],'eps':1e-8,'weight_decay':0,
                            'refiner_lr':.0003,'endpoint_logits_lr':.01,'shape_lr':.001},'Wrong old optimizer')
    ignore={'protocol','optimizer','claim'}
    check({k:v for k,v in old.items() if k not in ignore}==
          {k:v for k,v in new.items() if k not in ignore},'Additional scientific specification change')


def validate_same_package_and_inputs(old,plan):
    package=lambda p:{k:v for k,v in p['source_hashes'].items() if k.startswith('bridge_rgs/')}
    check(package(plan)==package(old),'Training/rendering/Cholesky package changed')
    for name in ('evaluate_projective_deck_pooling.py','compare_official_evaluations.py'):
        check(plan['source_hashes'][name]==old['source_hashes'][name],'Scoring implementation changed')
    for key in ('base_checkpoint','base_checkpoint_sha256','manifest','training_views','training_order',
                'inherited_package_hashes','installed_sources','scoring_source_review','prior_plan_path',
                'prior_receipt_path','prior_predictions_path','prior_e_metrics','preflight','preflight_plan_sha256',
                'runtime_versions','environment','execution_budget'):
        check(plan[key]==old[key],'Non-epsilon plan change: '+key)
    for path,digest in old['input_hashes'].items():
        check(plan['input_hashes'].get(path)==digest,'Predecessor input not preserved: '+path)


def predecessor_audit(plan,audit):
    """Bind completed historical evidence without a best-arm/performance filter."""
    filenames=('plan.json','execution_receipt.json','launch_receipt.json','independent_cpu_review.json')
    check(plan['old_run']==str(OLD_RUN) and set(plan['old_v2_lineage'])==set(filenames),'Wrong predecessor paths')
    for filename in filenames:
        path=OLD_RUN/filename
        audit.bind(path,plan['input_hashes'][str(path)])
        check(plan['old_v2_lineage'][filename]=={'path':str(path),'sha256':sha(path)},'Predecessor record differs')
    old=read(OLD_RUN/'plan.json');execution=read(OLD_RUN/'execution_receipt.json')
    launch=read(OLD_RUN/'launch_receipt.json');review=read(OLD_RUN/'independent_cpu_review.json')
    check(sha(OLD_RUN/'plan.json')==OLD_PLAN_SHA==execution['plan_sha256']==launch['plan_sha256']==review['plan_sha256'],
          'Predecessor plan binding')
    check(launch['status']=='completed' and launch['exit_code']==0 and launch['natural_completion'] is True
          and launch['execution_receipt_sha256']==sha(OLD_RUN/'execution_receipt.json')
          and execution['status']=='completed' and execution['bound_sources_inputs_unchanged']
          and review['status']=='passed' and review['inputs_sources_outputs_unchanged'], 'Predecessor not naturally audited')
    validate_epsilon_only_specs(old['specification'],plan['specification'])
    old_source=Path(old['source_snapshot'])
    check(tree(old_source)==old['source_hashes'],'Predecessor frozen source changed')
    for name,digest in old['source_hashes'].items():
        audit.bind(old_source/name,digest)
    validate_same_package_and_inputs(old,plan)
    bindings=plan['old_samearm_metrics']
    check(set(bindings)==set(ARMS[1:]),'Need exactly three predetermined same-arm comparisons')
    for arm,record in bindings.items():
        expected=execution['evaluation']['metrics'][arm+'_joint']
        check(record==expected,'Old same-arm metric identity differs: '+arm)
        check(plan['input_hashes'][str(Path(record['path']).resolve())]==record['sha256'],'Old metric not an input')
        audit.record(record)
    return {'predecessor_plan_sha256':OLD_PLAN_SHA,
            'predecessor_execution_sha256':sha(OLD_RUN/'execution_receipt.json'),
            'predecessor_independent_audit_sha256':sha(OLD_RUN/'independent_cpu_review.json'),
            'bridge_package_exact':True,'same_data_and_2000_sample_order':True,
            'change':'head Adam eps=1e-8 unchanged; endpoint and shape Adam eps=1e-15',
            'old_adoption_gate_not_required':True}


def inherited_sources(plan,audit):
    """Check reviewed package inheritance without importing either package."""
    folder=Path(plan['preflight']);pp=read(folder/'plan.json');pr=read(folder/'execution_receipt.json')
    pa=read(folder/'analysis.json');pl=read(folder/'launch_receipt.json')
    for filename in ('plan.json','execution_receipt.json','analysis.json','launch_receipt.json'):
        path=folder/filename;audit.bind(path,plan['input_hashes'][str(path)])
    check(audit.bind(folder/'plan.json')==plan['preflight_plan_sha256']==pr['plan_sha256']==pa['plan_sha256']==pl['plan_sha256'],
          'Preflight plan chain')
    check(pl.get('status') in (None,'completed') and pl['exit_code']==0 and pl['natural_completion'] is True
          and pl['execution_receipt_sha256']==sha(folder/'execution_receipt.json')
          and pr['status']==pa['status']=='passed' and pr['inputs_and_sources_unchanged']
          and pr['analysis_sha256']==sha(folder/'analysis.json'),'Preflight not naturally passed')
    rows=pa['records'];required={'initial_p3d','initial_final','alpha','rgb','simplex','gradients','raw_gradients','head_gradient'}
    check(len(rows)==6 and {(r['name'],r['mode']) for r in rows}==
          {(n,a) for n in ('002.png','118.png') for a in ARMS[1:]},'Preflight cases')
    check(all(r['passed'] and required<=set(r['gates']) and all(r['gates'].values()) for r in rows),'Preflight gates')
    package={k.removeprefix('bridge_rgs/'):v for k,v in plan['source_hashes'].items() if k.startswith('bridge_rgs/')}
    check(package==plan['inherited_package_hashes'] and
          all(pp['sources']['bridge_rgs/'+k]==v for k,v in package.items()),'Package differs from real preflight')
    deck_paths=[Path(p) for p in plan['input_hashes'] if p.endswith('/projective_deck_pooling_v1/plan.json')]
    check(len(deck_paths)==1,'Missing fixed prior scorer lineage');deck=read(deck_paths[0])
    compared=[]
    for module,names in {'evaluate.py':('distortion_render_grid',),
                         'official_evaluate.py':('official_fingerprint','rasterize_official_annotation','score_official_arrays')}.items():
        old=Path(deck['source_snapshot'])/'bridge_rgs'/module
        new=Path(plan['source_snapshot'])/'bridge_rgs'/module
        old_sha=audit.bind(old,deck['source_hashes']['bridge_rgs/'+module]);new_sha=audit.bind(new)
        if old_sha!=new_sha:
            review=plan['scoring_source_review'][module]
            check(review['reference_sha256']==old_sha and review['inherited_sha256']==new_sha,'Scoring source review mismatch')
        for name in names:
            check(function_ast(old,name)==function_ast(new,name),'Used scoring function changed '+name)
            compared.append(name)
    return {'preflight_plan_sha256':plan['preflight_plan_sha256'],'preflight_cases':6,
            'inherited_package_files':len(package),'used_scoring_function_AST_equal':compared,
            'scope':'Bound full source and reviewed unused entrypoint changes; independently compare used scoring function ASTs.'}


def original_grid(camera):
    width,height=camera['width'],camera['height']
    calibration=np.asarray(camera['K'],np.float32)
    distortion=np.asarray(camera.get('distortion',[]),np.float32)
    if not distortion.size or not np.any(distortion):
        return calibration,width,height,None
    yy,xx=np.indices((height,width),dtype=np.float32)
    output=np.stack((xx,yy),axis=-1)
    undistorted=cv2.undistortPoints(output.reshape(-1,1,2),calibration,distortion,P=calibration).reshape(height,width,2)
    minimum=np.minimum(np.floor(undistorted.min(axis=(0,1))),[0,0]).astype(int)
    maximum=np.maximum(np.ceil(undistorted.max(axis=(0,1))),[width-1,height-1]).astype(int)
    target_K=calibration.copy();target_K[:2,2]-=minimum
    undistorted-=minimum.astype(np.float32)
    size=maximum-minimum+1
    return target_K,int(size[0]),int(size[1]),undistorted


def mask_from_soft(probabilities,grid):
    # Preserve all five channels in one remap: installed OpenCV scalar-channel
    # optimization may use a different interpolation backend.
    p=np.ascontiguousarray(probabilities,np.float32)
    mapped=p if grid is None else cv2.remap(p,grid[...,0],grid[...,1],cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    check(np.isfinite(mapped).all() and np.all(mapped.sum(-1)>0),'Empty/nonfinite warped probability')
    return mapped.argmax(-1).astype(np.uint8)


def rasterize(payload,width,height):
    annotation=json.loads(payload)
    check((annotation['imageWidth'],annotation['imageHeight'])==(width,height),'Annotation shape differs')
    image=Image.new('L',(width,height),0); draw=ImageDraw.Draw(image)
    for shape in annotation['shapes']:
        check(shape.get('shape_type','polygon')=='polygon' and len(shape['points'])>=3,'Unsupported annotation shape')
        label=shape['label'].strip()
        category=CLASSES.index(label) if label in CLASSES else 255
        draw.polygon([tuple(point) for point in shape['points']],fill=category)
    return np.asarray(image,dtype=np.uint8)


def cm(prediction,target):
    check(prediction.dtype==np.uint8 and prediction.shape==target.shape and (prediction<5).all(),'Bad mask')
    support=target<5
    return np.bincount(target[support].astype(np.int64)*5+prediction[support],minlength=25).reshape(5,5)


def iou(matrices):
    matrices=np.asarray(matrices)
    diag=np.diagonal(matrices,axis1=-2,axis2=-1).astype(np.float64)
    union=matrices.sum(-1)+matrices.sum(-2)-diag
    return np.divide(diag,union,out=np.full_like(diag,np.nan),where=union>0)


def independent_bootstrap(reference,candidate,repeats=5000,seed=20260926):
    """Count-weight resampling, independent of producer's indexed CM arrays."""
    a={v['name']:v for v in reference['views']}; b={v['name']:v for v in candidate['views']}
    check(set(a)==set(b),'Unpaired camera population');names=sorted(a)
    semantic=[n for n in names if 'confusion_matrix' in a[n]]
    check(semantic==[n for n in names if 'confusion_matrix' in b[n]],'Unpaired semantic support')
    rng=np.random.default_rng(seed); result={}
    def add(key,left,right,differences):
        usable=np.asarray(differences)[np.isfinite(differences)]
        defined=bool(np.isfinite(left) and np.isfinite(right))
        result[key]={'reference':float(left) if np.isfinite(left) else None,
                     'candidate':float(right) if np.isfinite(right) else None,
                     'difference':float(right-left) if defined else None,
                     'paired_view_bootstrap_95_interval':np.quantile(usable,[.025,.975]).tolist() if defined and len(usable) else None,
                     'finite_bootstrap_replicates':len(usable)}
    counts=np.stack([np.bincount(row,minlength=len(names)) for row in rng.integers(len(names),size=(repeats,len(names)))])
    for key in ('psnr','ssim','lpips'):
        left,right=(np.array([views[n][key] for n in names]) for views in (a,b))
        add(key,left.mean(),right.mean(),counts@(right-left)/len(names))
    counts=np.stack([np.bincount(row,minlength=len(semantic)) for row in rng.integers(len(semantic),size=(repeats,len(semantic)))])
    left,right=(np.asarray([views[n]['confusion_matrix'] for n in semantic]) for views in (a,b))
    li,ri=iou(left.sum(0)),iou(right.sum(0))
    boot_left=iou((counts@left.reshape(len(semantic),25)).reshape(repeats,5,5))
    boot_right=iou((counts@right.reshape(len(semantic),25)).reshape(repeats,5,5))
    for key,columns in (('miou_all',slice(None)),('miou_foreground',slice(1,None))):
        add(key,np.nanmean(li[columns]),np.nanmean(ri[columns]),
            np.nanmean(boot_right[:,columns],axis=1)-np.nanmean(boot_left[:,columns],axis=1))
    for index,name in enumerate(CLASSES):
        add(name+'_iou',li[index],ri[index],boot_right[:,index]-boot_left[:,index])
    return result


def independent_gate(pairs):
    check(set(pairs)==set(THRESHOLDS),'Need all four fixed references')
    clauses={}
    for reference,minimum in THRESHOLDS.items():
        m=pairs[reference];gain=m['miou_all']
        clauses[reference+'_miou_gain']=gain['difference'] is not None and gain['difference']>=minimum
        clauses[reference+'_miou_ci_lower_positive']=(gain['paired_view_bootstrap_95_interval'] is not None
                                                      and gain['paired_view_bootstrap_95_interval'][0]>0)
        for name in CLASSES:
            value=m[name+'_iou']['difference']
            clauses[reference+'_'+name+'_guard']=value is not None and value>=(-.001 if name=='stay_cable' else -.002)
    return {key:bool(value) for key,value in clauses.items()}


class Audit:
    def __init__(self):
        self.bound={};self.max_error=0.

    def bind(self,path,expected=None):
        path=str(Path(path).resolve())
        if path not in self.bound:
            self.bound[path]=sha(path)
        check(expected is None or self.bound[path]==expected,'Artifact changed: '+path)
        return self.bound[path]

    def record(self,record):
        return self.bind(record['path'],record['sha256'])

    def close(self,left,right,label):
        if isinstance(left,dict):
            check(set(left)==set(right),label+' keys')
            for key in left:
                self.close(left[key],right[key],label+'/'+key)
        elif left is None:
            check(right is None,label+' undefined')
        else:
            a,b=np.asarray(left,float),np.asarray(right,float)
            check(a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b))
                  and np.array_equal(np.isposinf(a),np.isposinf(b))
                  and np.array_equal(np.isneginf(a),np.isneginf(b)),label+' shape/finiteness')
            error=float(np.max(np.abs(a[np.isfinite(a)]-b[np.isfinite(b)]))) if np.isfinite(a).any() else 0.
            self.max_error=max(self.max_error,error)
            check(error<=1e-12,label+f' difference {error}')

    def png(self,record,shape):
        self.record(record)
        image=cv2.imread(record['path'],cv2.IMREAD_UNCHANGED)
        check(image is not None and image.shape==shape and image.dtype==np.uint8,'PNG contract')
        return image

    def probabilities(self,record,shape):
        self.record(record);p=np.load(record['path'],mmap_mode='r',allow_pickle=False)
        check(p.shape==shape and p.dtype==np.float32 and np.isfinite(p).all() and (p>=0).all()
              and np.allclose(p.sum(-1),1,atol=2e-5,rtol=0),'Probability contract')
        return p

    def final_hash_check(self):
        for path,digest in self.bound.items():
            check(sha(path)==digest,'Artifact changed during independent review: '+path)


def inspect_adam(optimizer,head,partition,arm):
    groups=optimizer['param_groups'];states=optimizer['state']
    tensors=list(head.values())
    check(len(groups)==(1 if arm=='refiner_only' else 3),'Optimizer group count')
    blocks=[list(head.values())]; names=[list(head)]
    if partition is not None:
        fields=['inside_logits','outside_logits','direction','offset_raw','width_raw']
        check(set(partition)==set(fields),'Unexpected field tensors')
        blocks.extend([[partition[k] for k in fields[:2]],[partition[k] for k in fields[2:]]])
        names.extend([fields[:2],fields[2:]])
        tensors.extend(partition.values())
    for index,(group,block,labels) in enumerate(zip(groups,blocks,names,strict=True)):
        check(group['lr']==(.0003,.01,.001)[index] and group['eps']==(HEAD_EPS if index==0 else FIELD_EPS)
              and group['name']==('head','endpoints','shape')[index]
              and tuple(group['betas'])==(.9,.999) and group['weight_decay']==0,'Adam hyperparameters')
        check(len(group['params'])==len(block),'Adam parameter count')
        for pid,tensor,name in zip(group['params'],block,labels,strict=True):
            if pid not in states:
                check(arm=='marginal' and name=='direction','Unexpected inactive optimizer parameter')
                continue
            state=states[pid]
            check(float(state['step'])==2000 and state['exp_avg'].shape==state['exp_avg_sq'].shape==tensor.shape
                  and state['exp_avg'].dtype==state['exp_avg_sq'].dtype==tensor.dtype,'Adam step/shape/dtype')
    check(finite(optimizer) and all(bool(torch.isfinite(t).all()) for t in tensors),'Nonfinite final state')
    check(set(states)<={pid for group in groups for pid in group['params']},'Unexpected Adam state')
    return [{'role':('head','endpoints','shape')[i],'eps':g['eps'],'lr':g['lr'],
             'parameter_tensors':len(g['params'])} for i,g in enumerate(groups)]


def training_audit(run,plan,execution,audit):
    manifest=read(plan['manifest']);train=[v for v in manifest['views'] if v['split']=='train']
    labeled=sorted((v for v in train if v.get('mask_path')),key=lambda v:v['name'])
    check(len(train)==350 and len(labeled)==259 and labeled==plan['training_views'],'TRAIN population')
    order=expected_order();check(order==plan['training_order'],'Plan shuffle')
    names=[labeled[i]['name'] for i in order];names_sha=hashlib.sha256(json.dumps(names).encode()).hexdigest()
    check(sha(plan['manifest'])==MANIFEST_SHA and plan['base_checkpoint_sha256']==BASE_SHA,'Wrong base or manifest')
    audit.bind(plan['base_checkpoint'],BASE_SHA)
    base=torch.load(plan['base_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    head={k:v for k,v in base['model'].items() if k.startswith('refiner.')}
    frozen={k:v for k,v in base['model'].items() if not k.startswith('refiner.')}
    head_hash,frozen_hash=tensor_digest(head),tensor_digest(frozen)
    camera=base['training_cameras'].cpu().numpy()
    check(np.array_equal(camera,np.asarray([v['w2c_original'] for v in train],np.float32)),'Base camera order')
    camera_hash=hashlib.sha256(camera.tobytes()).hexdigest();summary=[];initial_fields=[];end_rng=[]
    for index,arm in enumerate(ARMS):
        directory=run/arm;receipt=read(directory/'training_receipt.json');audit.bind(directory/'training_receipt.json')
        check(receipt==execution['training'][index] and receipt['arm']==arm and receipt['status']=='completed','Training receipt chain')
        check(receipt['steps']==receipt['scene_calls']==receipt['optimizer_steps']==receipt['backwards']==2000
              and receipt['partition_shader_calls']==(0 if arm=='refiner_only' else 2000)
              and receipt['val_payload_reads']==receipt['real_rgb_payload_reads']==0,'Training count/scope')
        audit.bind(directory/'steps.jsonl',receipt['training_log_sha256'])
        rows=[json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
        check(len(rows)==2000 and [r['step'] for r in rows]==list(range(1,2001))
              and [r['view'] for r in rows]==names and receipt['sample_names_sha256']==names_sha,'Actual training sequence')
        check(np.isfinite([[r[k] for k in ('loss','final','raw_ce','residual_square')] for r in rows]).all(),'Nonfinite logged objective')
        # The scalar total is rounded in FP32; only a scale-aware tolerance can
        # compare independently serialized component scalars.
        reconstructed=np.array([r['final']+.25*r['raw_ce']+.001*r['residual_square'] for r in rows])
        check(np.allclose(reconstructed,[r['loss'] for r in rows],rtol=3e-7,atol=1e-9),'Logged objective weights')
        check(np.all(np.diff([r['elapsed_seconds'] for r in rows])>=0),'Training clock order')
        check(receipt['frozen_nonrefiner_sha256']==frozen_hash and receipt['refiner_initial_sha256']==head_hash
              and receipt['training_camera_sha256']==camera_hash,'Frozen base/head/camera evidence')
        delta_path=Path(receipt['delta_path']);check(delta_path==directory/'final_delta.pt','Unexpected endpoint path')
        audit.bind(delta_path,receipt['delta_sha256']);delta=torch.load(delta_path,map_location='cpu',mmap=True,weights_only=False)
        check(delta['format']==FORMAT and 'model' not in delta and delta['arm']==arm and delta['step']==2000
              and delta['ordinary_resume_supported'] is False,'Wrong derived checkpoint format')
        check(delta['specification']==plan['specification'] and delta['plan_sha256']==sha(run/'plan.json')
              and delta['base_checkpoint_sha256']==BASE_SHA and delta['manifest_sha256']==MANIFEST_SHA
              and delta['pixel_protocol']=='legacy_mixed_v1' and delta['refiner_config']==base['refiner_config']
              and delta['training_camera_sha256']==camera_hash,'Endpoint source/schema')
        final_head={'refiner.'+k:v for k,v in delta['refiner_state'].items()}
        check(set(final_head)==set(head) and all(t.shape==head[k].shape and t.dtype==head[k].dtype for k,t in final_head.items()),'Head tensor schema')
        check(tensor_digest(final_head)==receipt['refiner_final_sha256']!=head_hash,'Final head digest/update')
        part=delta['partition_state'];check((part is None)==(arm=='refiner_only'),'Partition state presence')
        if part is not None:
            n=base['model']['splats.means'].shape[0]
            shapes={'inside_logits':(n,5),'outside_logits':(n,5),'direction':(n,3),'offset_raw':(n,),'width_raw':(n,)}
            check(set(part)==set(shapes) and all(tuple(part[k].shape)==s and part[k].dtype==torch.float32 for k,s in shapes.items()),'Partition tensor schema')
            changes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest()!=receipt['partition_initial_sha256'][k] for k,v in part.items()}
            check(changes==receipt['partition_parameter_changed'] and (changes['inside_logits'] or changes['outside_logits']),'Partition updates')
            initial_fields.append(receipt['partition_initial_sha256'])
            check(receipt['parameter_count_partition']==15*n,'Partition is 15N parameters')
        else:
            check(receipt['parameter_count_partition']==0 and not receipt['partition_initial_sha256'],'Reference field capacity')
        check(receipt['parameter_count_refiner']==sum(t.numel() for t in final_head.values()),'Head parameter count')
        actual_optimizer_groups=inspect_adam(delta['stage_optimizer_state'],delta['refiner_state'],part,arm)
        actual_metadata=[{k:g[k] if k!='betas' else list(g[k]) for k in ('name','lr','eps','betas','weight_decay')}
                         for g in delta['stage_optimizer_state']['param_groups']]
        check(receipt['optimizer_groups']==actual_metadata,'Runtime group receipt disagrees with actual Adam state')
        end_rng.append((delta['stage_torch_rng'].clone(),delta['stage_cuda_rng'].clone()))
        summary.append({'arm':arm,'endpoint_sha256':receipt['delta_sha256'],'steps':2000,
                        'refiner_parameters':receipt['parameter_count_refiner'],'partition_parameters':receipt['parameter_count_partition'],
                        'changed_head_tensors':sum(not torch.equal(t,head[k]) for k,t in final_head.items()),
                        'actual_optimizer_groups':actual_optimizer_groups,
                        'elapsed_seconds':receipt['elapsed_seconds'],'peak_cuda_allocated_bytes':receipt['peak_cuda_allocated_bytes']})
        del delta,part,final_head
    check(all(v==initial_fields[0] for v in initial_fields[1:]),'Unequal partition initialization hashes')
    rng_equal=all(torch.equal(a[0],end_rng[0][0]) and torch.equal(a[1],end_rng[0][1]) for a in end_rng)
    return {'arms':summary,'actual_2000_sample_sequence_exact':True,'base_nonrefiner_sha256':frozen_hash,
            'camera_sha256':camera_hash,'end_rng_equal':rng_equal,
            'frozen_evidence':'Endpoint delta stores only head/new partition. Original field preservation uses bound runtime hashes and independently verified RGB, not a saved full post-training field.'}


def secondary_pair_audit(plan,evaluation,new_joint_metrics,audit):
    """Descriptive paired comparisons do not feed the unchanged adoption gate."""
    pairs=evaluation['secondary_comparisons']
    check(set(pairs)==set(ARMS[1:]),'Secondary comparisons must contain exactly the three field arms')
    summary={}
    for arm in ARMS[1:]:
        old_binding=plan['old_samearm_metrics'][arm];audit.record(old_binding)
        old=read(old_binding['path']);new=new_joint_metrics[arm]
        check(old['official_evaluation_fingerprint']==new['official_evaluation_fingerprint']
              and old['scoring_protocol']==new['scoring_protocol']
              and old['validation_views']==new['validation_views']==50
              and old['semantic_validation_views']==new['semantic_validation_views']==41,'Old/new grid mismatch')
        expected=independent_bootstrap(old,new)
        audit.record(pairs[arm]);saved=read(pairs[arm]['path'])
        check(saved['candidate_arm']==saved['reference_arm']==arm
              and saved['candidate_run']==plan['output'] and saved['reference_run']==plan['old_run']
              and saved['bootstrap_repeats']==5000 and saved['seed']==20260926
              and saved['difference_direction']=='candidate minus reference; LPIPS improves when negative'
              and saved['official_evaluation_fingerprint']==new['official_evaluation_fingerprint'], 'Secondary pair direction/protocol')
        names=sorted(r['name'] for r in new['views'])
        semantic=sorted(r['name'] for r in new['views'] if 'confusion_matrix' in r)
        check(saved['views']==names and saved['semantic_views']==semantic,'Secondary pair population')
        check(saved['comparison_kind']=='new_samearm_minus_oldsamearm' and saved['descriptive_only'] is True
              and saved['old_metrics_source']==old_binding,'Secondary comparison must be descriptive and bound')
        check(set(saved['metrics'])==set(expected),'Secondary metric set')
        for metric,value in expected.items():
            audit.close(value,saved['metrics'][metric],'Secondary '+arm+'/'+metric)
        summary[arm]={m:expected[m] for m in ('miou_all','miou_foreground','stay_cable_iou')}
    return {'new_samearm_minus_old':summary,'adoption_gate_uses_secondary':False}


def evaluate_audit(run,plan,execution,audit):
    evaluation=read(run/'evaluation/execution_receipt.json');audit.bind(run/'evaluation/execution_receipt.json')
    check(evaluation==execution['evaluation'] and evaluation['status']=='completed','Evaluation receipt chain')
    check(evaluation['plan_sha256']==sha(run/'plan.json') and evaluation['scene_calls']==250
          and evaluation['partition_shader_calls']==150 and evaluation['teacher_calls']==0
          and evaluation['baseline_E_masks_byte_exact']==50 and evaluation['new_masks']==600
          and evaluation['annotation_payload_reads']==41 and evaluation['gt_rgb_payload_reads']==0,'Evaluation scope/count')
    prediction_path=run/'evaluation/predictions_receipt.json';audit.bind(prediction_path,evaluation['predictions_receipt_sha256'])
    prediction=read(prediction_path)
    check(prediction['status']=='all_600_masks_before_GT' and prediction['annotation_payload_reads']==0
          and prediction['scene_calls']==250 and prediction['partition_shader_calls']==150,'Prediction receipt')
    check(prediction['predictions_finished_utc']==evaluation['predictions_finished_utc'] and
          datetime.fromisoformat(evaluation['predictions_finished_utc'])<=datetime.fromisoformat(evaluation['scoring_started_utc'])
          <datetime.fromisoformat(evaluation['scoring_finished_utc']),'Prediction/GT timestamp order')
    prior_plan=read(plan['prior_plan_path']);prior=read(plan['prior_receipt_path']);old_e=read(plan['prior_e_metrics'])
    check(prior['status']=='completed' and prior['metrics_sha256']['E']==sha(plan['prior_e_metrics'])
          and prior_plan['teachers']['old']['sha256']==TEACHER_SHA,'Old E/teacher lineage')
    views=prior_plan['views'];names=sorted(v['name'] for v in views);by_view={v['name']:v for v in views}
    check(len(views)==len(set(names))==50,'Fixed original-grid views')
    payload={'evaluation_family':prior_plan['evaluation_family'],'scoring_protocol':prior_plan['scoring_protocol'],
             'views':[{k:by_view[n][k] for k in ('camera','source_image_sha256','source_annotation_sha256','rasterized_mask_sha256')} for n in names]}
    fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    check(fingerprint==prior_plan['reference_fingerprint']==old_e['official_evaluation_fingerprint'],'Official fingerprint')
    e_rows={v['name']:v for v in old_e['views']};old_records={(r['arm'],r['name']):r for r in prior['predictions']}
    records={(r['arm'],r['name']):r for r in prediction['records']}
    check(len(prediction['records'])==len(records)==200 and set(records)=={(a,n) for a in ARMS for n in names},'Prediction population')
    metrics={};metric_rows={};matrices={};annotation_count=0
    for arm in ARMS:
        for kind in KINDS:
            record=evaluation['metrics'][arm+'_'+kind];audit.record(record);value=read(record['path'])
            check(value['validation_views']==50 and value['semantic_validation_views']==41
                  and value['official_evaluation_fingerprint']==fingerprint and value['scoring_protocol']==prior_plan['scoring_protocol'],'Metric grid')
            metrics[arm,kind]=value;metric_rows[arm,kind]={r['name']:r for r in value['views']};matrices[arm,kind]={}
            check(len(value['views'])==50 and set(metric_rows[arm,kind])==set(names),'Metric names')
    check(len(prediction['baseline_E_exact'])==50,'Baseline reproduction count')
    for view_index,view in enumerate(views):
        name=view['name'];camera=view['camera'];width,height=camera['width'],camera['height']
        K,cw,ch,grid=original_grid(camera);cached=prior['h3_canvases'][name]
        check([cw,ch]==cached['boundary']['canvas'] and np.array_equal(K,np.asarray(cached['boundary']['canvas_K'],np.float32)),'Canvas differs')
        a,e=old_records['A',name],old_records['E',name]
        teacher=audit.probabilities(a['teacher_soft_canvas'],(ch,cw,5))
        e_rgb={'path':e['rgb'],'sha256':e['rgb_sha256']};e_mask={'path':e['mask'],'sha256':e['mask_sha256']}
        audit.png(e_rgb,(height,width,3));old_mask=audit.png(e_mask,(height,width))
        base_record=prediction['baseline_E_exact'][view_index]
        check(Path(base_record['path'])==run/'evaluation/baseline/joint_mask'/name,'Baseline name ordering')
        audit.png(base_record,(height,width));check(base_record['sha256']==e_mask['sha256'],'E reproduction bytes')
        masks={}
        for arm in ARMS:
            record=records[arm,name]
            check(record['teacher_input_rgb_exact'] and record['teacher_soft_canvas']==a['teacher_soft_canvas']
                  and record['rgb_reference']==e_rgb,'Teacher/RGB origin')
            for folder,reference,shape in (('canvas_rgb',cached['canvas_rgb'],(ch,cw,3)),
                                           ('h3_rgb',{'path':a['rgb'],'sha256':a['rgb_sha256']},(height,width,3))):
                fresh=run/'evaluation'/arm/folder/name
                audit.png({'path':str(fresh),'sha256':reference['sha256']},shape);audit.record(reference)
            final=audit.probabilities(record['soft_canvas'],(ch,cw,5))
            for kind in KINDS:
                masks[arm,kind]=audit.png(record['masks'][kind],(height,width))
                if kind!='raw':
                    probability=final if kind=='scene' else (final+teacher)*np.float32(.5)
                    check(np.array_equal(mask_from_soft(probability,grid),masks[arm,kind]),'Soft blend/warp mismatch '+arm+'/'+kind+'/'+name)
            del final
        truth=None
        if view['source_annotation_path'] is not None:
            audit.bind(view['source_annotation_path'],view['source_annotation_sha256'])
            truth=rasterize(Path(view['source_annotation_path']).read_bytes(),width,height);annotation_count+=1
            check(hashlib.sha256(truth.tobytes()).hexdigest()==view['rasterized_mask_sha256'],'Independent annotation rasterization')
            check(np.array_equal(cm(old_mask,truth),e_rows[name]['confusion_matrix']),'Old E CM')
        for key in metrics:
            row=metric_rows[key][name]
            for column in ('name','width','height','rgb_pixels','psnr','ssim','lpips'):
                check(row[column]==e_rows[name][column],'Incorrect E RGB score reuse')
            if truth is None:
                check('confusion_matrix' not in row,'Invented unannotated mask metric')
            else:
                matrix=cm(masks[key],truth);matrices[key][name]=matrix
                check(np.array_equal(matrix,row['confusion_matrix']) and row['semantic_pixels']==int(matrix.sum())
                      and row['semantic_ignore_pixels']==int((truth==255).sum()),'Per-view CM/support differs')
    check(annotation_count==41,'Annotation population')
    scores={}
    for key,value in metrics.items():
        pooled=sum(matrices[key].values());ii=iou(pooled)
        check(np.array_equal(pooled,value['confusion_matrix']),'Pooled CM differs')
        audit.close(ii,value['iou'],'IoU');audit.close(np.nanmean(ii),value['miou_all'],'all5')
        audit.close(np.nanmean(ii[1:]),value['miou_foreground'],'foreground')
        for rgb in ('psnr','ssim','lpips'):
            audit.close(np.mean([r[rgb] for r in value['views']]),value[rgb],'RGB rows')
        scores['_'.join(key)]={'miou_all':float(np.nanmean(ii)),'iou':ii.tolist()}
    pairs={}
    for reference in THRESHOLDS:
        pairs[reference]=independent_bootstrap(old_e if reference=='E' else metrics[reference,'joint'],metrics['integrated','joint'])
        binding=evaluation['comparisons'][reference];audit.record(binding);saved=read(binding['path'])
        check(saved['candidate_arm']=='integrated' and saved['reference_arm']==reference and saved['bootstrap_repeats']==5000
              and saved['seed']==20260926 and saved['views']==names
              and saved['semantic_views']==sorted(n for n in names if 'confusion_matrix' in e_rows[n])
              and saved['difference_direction']=='candidate minus reference; LPIPS improves when negative'
              and saved['official_evaluation_fingerprint']==fingerprint,'Pair protocol/direction')
        for metric,result in pairs[reference].items():
            audit.close(result,saved['metrics'][metric],'Pair '+reference+'/'+metric)
    clauses=independent_gate(pairs);gate_path=run/'evaluation/system_gate.json';audit.bind(gate_path,evaluation['gate_sha256']);gate=read(gate_path)
    check(gate==evaluation['gate'] and gate['clauses']==clauses and gate['passed']==all(clauses.values())
          and gate['thresholds']==THRESHOLDS,'Independent gate differs')
    check(gate['decision']==('eligible_for_review_not_automatic_adoption' if gate['passed'] else 'stop_fixed_partition_candidate'),'Gate decision differs')
    secondary=secondary_pair_audit(plan,evaluation,{a:metrics[a,'joint'] for a in ARMS[1:]},audit)
    return {'official_fingerprint':fingerprint,'metrics_checked':12,'per_view_confusion_matrices_checked':492,
            'original_E_CM_checked':41,'soft_scene_and_joint_masks_recomputed_exact':400,'all_masks_checked':600,
            'baseline_E_masks_byte_exact':50,'fresh_H3_RGB_canvas_and_original_byte_exact':400,
            'adoption_passed':gate['passed'],'clauses':clauses,'scores':scores,
            'paired_integrated_minus':{k:{m:v[m] for m in ('miou_all','miou_foreground','stay_cable_iou')} for k,v in pairs.items()},
            'secondary':secondary,
            'raw_scope':'All 200 raw hard masks independently scored; raw soft arrays are not stored and are not reconstructed.'}


def run_audit(run,expected_plan_sha,deadline_seconds=600):
    run=Path(run).resolve();output=run/'independent_cpu_review.json'
    check(not output.exists(),'Never overwrite an independent review')
    start=time.perf_counter();audit=Audit();failure=None
    result={'status':'running','checker_sha256':sha(__file__),'CPU_deadline_seconds':deadline_seconds,
            'new_models':0,'new_renders':0,'new_optimizer_steps':0}
    def expire(*_):
        raise TimeoutError('Independent CPU audit deadline exceeded')
    prior_handler=signal.getsignal(signal.SIGALRM);signal.signal(signal.SIGALRM,expire);signal.alarm(deadline_seconds)
    try:
        check(not torch.cuda.is_initialized(),'CPU only');torch.set_num_threads(4);cv2.setNumThreads(4)
        check(sha(run/'plan.json')==expected_plan_sha,'Expected plan SHA differs')
        plan=read(run/'plan.json');execution=read(run/'execution_receipt.json');launch=read(run/'launch_receipt.json')
        # Check every completion boundary before touching endpoint tensors or pixel arrays.
        check(launch.get('status') in (None,'completed') and launch['exit_code']==0 and launch['natural_completion'] is True
              and launch['plan_sha256']==expected_plan_sha and launch['execution_receipt_sha256']==sha(run/'execution_receipt.json'),'Natural launch completion missing')
        check(execution['status']=='completed' and execution['bound_sources_inputs_unchanged']
              and execution['plan_sha256']==expected_plan_sha and execution['new_val_annotation_payload_reads']==41,'Incomplete experiment')
        check(len(execution['training'])==4 and all(r['status']=='completed' and r['steps']==2000 for r in execution['training']), 'Not all training arms completed')
        validate_settings(plan['specification']);snapshot=Path(plan['source_snapshot'])
        check(tree(snapshot)==plan['source_hashes'],'Source tree changed')
        for name in ('plan.json','execution_receipt.json','launch_receipt.json'):
            audit.bind(run/name)
        for relative,digest in plan['source_hashes'].items():audit.bind(snapshot/relative,digest)
        for path,digest in {**plan['input_hashes'],**plan['installed_sources']}.items():audit.bind(path,digest)
        for package,version in plan['runtime_versions'].items():
            check(importlib.metadata.version(package)==version,'Runtime changed '+package)
        result['source_inheritance']=inherited_sources(plan,audit)
        result['epsilon_only_history']=predecessor_audit(plan,audit)
        for name,record in execution['actual_imports'].items():
            audit.record(record);path=Path(record['path'])
            check(path.is_relative_to(snapshot) and plan['source_hashes'][str(path.relative_to(snapshot))]==record['sha256'],'Unfrozen actual import '+name)
        binary_sources=execution['loaded_gsplat_binary_sources']
        for record in binary_sources.values():audit.record(record)
        result['loaded_gsplat_binaries']=binary_sources
        result['loaded_gsplat_binary_count']=len(binary_sources)
        result['triton_binary_scope']=execution['triton_binary_scope']
        result['training']=training_audit(run,plan,execution,audit)
        result['evaluation']=evaluate_audit(run,plan,execution,audit)
        audit.final_hash_check();check(tree(snapshot)==plan['source_hashes'],'Source changed during audit')
        check(not torch.cuda.is_initialized(),'Unexpected CUDA')
        result.update(status='passed',plan_sha256=expected_plan_sha,source_count=len(plan['source_hashes']),
                      input_count=len(plan['input_hashes']),unique_bound_artifacts=len(audit.bound),
                      inputs_sources_outputs_unchanged=True,cuda_initialized=False,max_numeric_difference=audit.max_error,
                      scope='Independent count-weight bootstrap, Pillow polygons, CM and gates. No producer scoring/statistics imports, RGB LPIPS recomputation, or model inference.')
    except BaseException as exc:  # noqa: BLE001 - persist failed independent attempts, including interruption.
        failure=exc;result.update(status='failed',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,prior_handler)
        result['elapsed_seconds']=time.perf_counter()-start
        with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    if failure is not None:raise failure
    print(json.dumps({'status':'passed','report_sha256':sha(output),'gate_passed':result['evaluation']['adoption_passed'],
                      'elapsed_seconds':result['elapsed_seconds']}))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--expected-plan-sha256',required=True);parser.add_argument('--deadline-seconds',type=int,default=600)
    args=parser.parse_args();check(args.deadline_seconds>0,'Positive CPU deadline required')
    run_audit(args.run,args.expected_plan_sha256,args.deadline_seconds)


if __name__=='__main__':main()
