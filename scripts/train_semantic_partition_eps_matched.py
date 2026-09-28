"""Fixed epsilon-only follow-up to the completed categorical-partition experiment.

The entire package comes from the completed v2 snapshot. Only the two field
Adam groups use epsilon=1e-15; the head stays at 1e-8. Old failures are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
PRIOR = Path('/mnt/data/SHM2026/runs/multifield_h3_teacher_v1')
DECK = Path('/mnt/data/SHM2026/runs/projective_deck_pooling_v1')
PREDECESSOR = Path('/mnt/data/SHM2026/runs/semantic_partition_matched_v2')
OLD_PLAN_SHA = 'f10d266083999abd1635ea8cb91e7407247dafba113f525a6ec7c790e3e4cc1d'
BASE_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
MANIFEST_SHA = '551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa'
TEACHER_SHA = '00f5b84ac9a56c39512c5b8e43f70110397923feea1a2b4c78bdffd1f5524bff'
ARMS = ('refiner_only', 'marginal', 'point', 'integrated')
FORMAT = 'semantic_partition_eps_matched_delta_v1'
WEIGHTS = [.45604488253593445, .9227690100669861, .8530434370040894,
           1.1935913562774658, 1.5745513439178467]
THRESHOLDS = {'refiner_only': .0015, 'marginal': .001, 'point': .001, 'E': .002}
SPEC = {'protocol': 'semantic_partition_eps_matched_v1', 'arms': list(ARMS), 'candidate': 'integrated',
        'base_sha256': BASE_SHA, 'steps': 2000, 'seed': 42, 'train_count': 259,
        'sampler': 'sorted labeled TRAIN, independent default_rng(42), complete epoch shuffle',
        'grid': 'native legacy_mixed_v1 full resolution SH3; no crop/flip/augmentation',
        'optimizer': {'type': 'Adam', 'betas': [.9, .999], 'head_eps': 1e-8, 'field_eps': 1e-15, 'weight_decay': 0,
                      'refiner_lr': .0003, 'endpoint_logits_lr': .01, 'shape_lr': .001},
        'objective': {'final_weighted_ce': 1., 'final_lovasz': .2,
                      'raw_weighted_ce': .25, 'raw_lovasz': 0., 'residual_square_mean': .001},
        'class_weights_fp32': WEIGHTS, 'new_gaussians': 0, 'geometry_rgb_features_cameras_frozen': True,
        'teacher_training': False, 'validation_during_training': False, 'endpoint': 'last2000 only; no best',
        'evaluation': {'teacher_sha256': TEACHER_SHA, 'teacher_weight': .5,
                       'E_rgb': 'fixed verified E byte/score reuse', 'views': 50, 'annotations': 41,
                       'baseline_guard': 'fresh unchanged H3+old H+ joint original-grid masks byte-exact E',
                       'prediction_barrier': 'all four x50 x(joint,scene,raw) masks before annotation bytes'},
        'bootstrap': {'repeats': 5000, 'seed': 20260926}, 'gain_thresholds': THRESHOLDS,
        'class_guard': {'stay_cable': -.001, 'others': -.002},
        'claim': 'epsilon-only standard optimization follow-up; old failed gates remain failed; Gaussian CDF integration has prior art; no novelty or adoption presumed'}
VERSIONS = ('torch', 'gsplat', 'triton', 'numpy', 'scipy', 'opencv-python-headless', 'transformers')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value, *, replace=False):
    with Path(path).open('w' if replace else 'x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def files(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(Path(directory).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def population(manifest):
    train = sorted((v for v in manifest['views'] if v['split'] == 'train'), key=lambda v: v['name'])
    views = [v for v in train if v.get('mask_path')]
    require(len(train) == 350 and len(views) == 259 and len({v['name'] for v in train}) == 350,
            'Need the fixed 350/259 TRAIN population')
    return views


def sample_order(count=259, steps=2000, seed=42):
    rng = np.random.default_rng(seed)
    order = []
    while len(order) < steps:
        order.extend(rng.permutation(count).tolist())
    return order[:steps]


def validate_predecessor(plan, execution, launch, audit):
    """Metadata only; no endpoint loads and no interpretation of old failed gates."""
    require(launch.get('status') == 'completed' and launch.get('exit_code') == 0
            and launch.get('natural_completion') is True, 'Predecessor did not finish naturally')
    require(execution.get('status') == 'completed' and execution.get('bound_sources_inputs_unchanged') is True
            and audit.get('status') == 'passed' and audit.get('inputs_sources_outputs_unchanged') is True,
            'Predecessor completion/independent audit missing')
    require(launch['plan_sha256'] == execution['plan_sha256'] == audit['plan_sha256'] == OLD_PLAN_SHA,
            'Wrong predecessor plan')
    expected = json.loads(json.dumps(SPEC))
    expected['protocol'] = 'semantic_partition_matched_v1'
    expected['optimizer'].pop('head_eps'); expected['optimizer'].pop('field_eps')
    expected['optimizer']['eps'] = 1e-8
    expected['claim'] = 'fixed-density function-class experiment; Gaussian CDF integration has prior art; no novelty or adoption presumed'
    require(plan['specification'] == expected, 'Follow-up changed more than field epsilon and protocol identity')
    require([r['arm'] for r in execution['training']] == list(ARMS)
            and all(r['status'] == 'completed' and r['steps'] == 2000 for r in execution['training']),
            'Four original arms must complete')
    require(plan['training_order'] == sample_order() and plan['base_checkpoint_sha256'] == BASE_SHA,
            'Predecessor initialization/sampling differs')


def prepare(output, predecessor):
    """Only copy the actual old package; never consult live projection/model code."""
    output, predecessor = Path(output).resolve(), Path(predecessor).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Use a new data-disk experiment')
    require(predecessor == PREDECESSOR and sha(predecessor/'plan.json') == OLD_PLAN_SHA, 'Wrong fixed predecessor')
    old, execution, launch, audit = (read(predecessor/n) for n in
        ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json'))
    validate_predecessor(old, execution, launch, audit)
    require(launch['execution_receipt_sha256'] == sha(predecessor/'execution_receipt.json'), 'Unbound old execution')
    source = Path(old['source_snapshot'])
    require(files(source) == old['source_hashes'], 'Completed training snapshot changed')
    inputs = {}
    def bind(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        require(expected is None or digest == expected, f'Changed dependency: {path}')
        inputs[str(path)] = digest
        return {'path': str(path), 'sha256': digest}
    for path, digest in old['input_hashes'].items():
        bind(path, digest)
    lineage = {name: bind(predecessor/name) for name in
               ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json')}
    old_evaluation = read(predecessor/'evaluation/execution_receipt.json')
    require(old_evaluation == execution['evaluation'] and old_evaluation['status'] == 'completed', 'Old evaluation chain')
    bind(predecessor/'evaluation/execution_receipt.json')
    old_metrics = {}
    for arm in ARMS[1:]:
        record = old_evaluation['metrics'][arm+'_joint']
        require(Path(record['path']) == predecessor/'evaluation'/arm/'official_metrics.json', 'Old same-arm metric path')
        old_metrics[arm] = bind(record['path'], record['sha256'])
        value = read(record['path'])
        require(value['validation_views'] == 50 and value['semantic_validation_views'] == 41
                and value['provenance']['base_checkpoint_sha256'] == BASE_SHA
                and value['provenance']['teacher_cache_checkpoint_sha256'] == TEACHER_SHA,
                'Wrong old same-arm metric lineage')
    snapshot = output/'source_snapshot'
    shutil.copytree(source/'bridge_rgs', snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    inherited = files(snapshot/'bridge_rgs')
    require(inherited == old['inherited_package_hashes'] and
            all(old['source_hashes']['bridge_rgs/'+name] == digest for name, digest in inherited.items()),
            'Copied package differs from completed v2')
    for name in ('evaluate_projective_deck_pooling.py', 'compare_official_evaluations.py'):
        bind(source/name, old['source_hashes'][name]); shutil.copy2(source/name, snapshot/name)
    for p in (Path(__file__), ROOT/'tests/test_semantic_partition_eps_matched.py',
              ROOT/'docs/semantic_partition_eps_matched_protocol.md'):
        shutil.copy2(p, snapshot/p.name)
    plan = {key: old[key] for key in
            ('root', 'installed_sources', 'scoring_source_review', 'base_checkpoint', 'base_checkpoint_sha256',
             'manifest', 'training_views', 'training_order', 'prior_plan_path', 'prior_receipt_path',
             'prior_predictions_path', 'prior_e_metrics', 'preflight', 'preflight_plan_sha256',
             'runtime_versions', 'execution_budget', 'environment')}
    require(plan['execution_budget']['internal_seconds'] == 3600 and plan['execution_budget']['external_seconds'] == 3660,
            'Original fixed budget changed')
    for path, digest in plan['installed_sources'].items():
        require(sha(path) == digest, 'Installed source changed')
    for name, version in plan['runtime_versions'].items():
        require(importlib.metadata.version(name) == version, 'Runtime changed')
    plan.update(specification=SPEC, status='prepared_pending_root_execution', output=str(output),
                source_snapshot=str(snapshot), source_hashes=files(snapshot), input_hashes=inputs,
                inherited_package_hashes=inherited, old_run=str(predecessor), old_v2_lineage=lineage,
                old_samearm_metrics=old_metrics,
                secondary_comparison_scope='new same-arm minus old same-arm; descriptive, not a primary adoption gate',
                prepare_label_decodes=0, prepare_val_annotation_payload_reads=0, prepare_checkpoint_loads=0)
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC, 'Specification changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Use the frozen runner')
    require(files(plan['source_snapshot']) == plan['source_hashes'], 'Frozen source changed')
    require(plan['training_order'] == sample_order(), 'Sampling order changed')
    for p, digest in {**plan['input_hashes'], **plan['installed_sources']}.items():
        require(sha(p) == digest, f'Changed bound input: {p}')
    for name, value in plan['runtime_versions'].items():
        require(importlib.metadata.version(name) == value, f'Changed runtime: {name}')
    for name, value in plan['environment'].items():
        require(os.environ.get(name) == value, f'Changed environment: {name}')


def gpu_inventory():
    content = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                       '--format=csv,noheader,nounits'], text=True)
    records = []
    for line in content.splitlines():
        pid, name, memory = line.split(',', 2)
        executable = str(Path(f'/proc/{int(pid)}/exe').resolve())
        require(executable == '/usr/share/rustdesk/rustdesk', 'Unknown compute process; never terminate it')
        records.append({'pid': int(pid), 'process_name': name.strip(), 'executable': executable, 'memory_mib': memory.strip()})
    return records


def digest_tensors(state, refiner=False):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if name.startswith('refiner.') == refiner:
            a = tensor.detach().cpu().contiguous().numpy()
            value.update(name.encode()); value.update(str(a.dtype).encode())
            value.update(str(a.shape).encode()); value.update(a.tobytes())
    return value.hexdigest()


def loss_terms(result, labels, valid, weights, semantic_loss):
    final = semantic_loss(result['probabilities'], labels, valid, weights, lovasz_weight=.2)
    raw = semantic_loss(result['p3d'], labels, valid, weights, lovasz_weight=0.)
    regularization = result['residual'].square().mean()
    return final + .25*raw + .001*regularization, {'final': final, 'raw_ce': raw, 'residual_square': regularization}


def configure_parameters(scene, arm, partition_class):
    require(arm in ARMS, 'Unknown arm')
    scene.requires_grad_(False); scene.refiner.requires_grad_(True)
    scene.train()
    import torch
    with torch.no_grad():
        logits = scene.semantic_decoder(scene.splats['sem_features'])
    field = None if arm == 'refiner_only' else partition_class(logits, mode=arm)
    groups = [{'params': list(scene.refiner.parameters()), 'lr': .0003, 'eps': 1e-8, 'name': 'head'}]
    if field is not None:
        for group, name in zip(field.optimizer_groups(), ('endpoints', 'shape'), strict=True):
            groups.append(dict(group, eps=1e-15, name=name))
    require(all(p.requires_grad == name.startswith('refiner.') for name,p in scene.named_parameters()), 'Wrong trainable scene scope')
    return field, groups


def make_optimizer(groups):
    import torch
    optimizer = torch.optim.Adam(groups, betas=(.9,.999), eps=1e-8, weight_decay=0.)
    expected = [('head', .0003, 1e-8)] + ([('endpoints', .01, 1e-15), ('shape', .001, 1e-15)] if len(groups) == 3 else [])
    require([(g['name'], g['lr'], g['eps']) for g in optimizer.param_groups] == expected, 'Actual Adam group contract differs')
    require(not optimizer.state, 'Optimizer must start fresh')
    return optimizer


def render_training(scene, field, render_partition, K, pose, width, height):
    if field is None:
        return scene.render(K, pose, width, height, degree=3, absgrad=False,
                            geometry_grad=False, refinement_grad_to_field=False)
    return render_partition(scene, field, K, pose, width, height)


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k,v in module.state_dict().items()}


def train_arm(plan, arm):
    import cv2
    import torch

    from bridge_rgs.io import seed_everything
    from bridge_rgs.losses import semantic_loss
    from bridge_rgs.partition_field import SemanticPartitionField, render_partition
    from bridge_rgs.train import load_scene
    seed_everything(42)
    directory = Path(plan['output'])/arm; directory.mkdir()
    start = time.monotonic()
    scene, state = load_scene(plan['base_checkpoint'])
    require(scene.pixel_protocol == 'legacy_mixed_v1' and scene.sh_degree == 3
            and scene.mip_filter_config is None, 'Wrong H3 field/profile')
    train_views = [v for v in read(plan['manifest'])['views'] if v['split'] == 'train']
    cameras = state['training_cameras'].detach().cpu().numpy()
    require(cameras.shape == (350,4,4) and np.array_equal(cameras,np.asarray([v['w2c_original'] for v in train_views],np.float32)),
            'Base saved TRAIN cameras differ')
    pose_by_name = {v['name']: cameras[i] for i,v in enumerate(train_views)}
    camera_sha = hashlib.sha256(cameras.tobytes()).hexdigest()
    frozen_before = digest_tensors(scene.state_dict()); head_before = digest_tensors(scene.state_dict(),True)
    field, groups = configure_parameters(scene,arm,SemanticPartitionField)
    field_initial = {} if field is None else {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                                              for k,v in field.state_dict().items()}
    optimizer = make_optimizer(groups)
    weights = torch.tensor(WEIGHTS,device='cuda',dtype=torch.float32)
    data_cache = {}; samples=[]; torch.cuda.reset_peak_memory_stats(); del state
    with (directory/'steps.jsonl').open('x') as log:
        for step,index in enumerate(plan['training_order'],1):
            view = plan['training_views'][index]; name=view['name']; samples.append(name)
            if name not in data_cache:
                mask=cv2.imread(view['mask_path'],cv2.IMREAD_UNCHANGED)
                valid=cv2.imread(view['valid_path'],cv2.IMREAD_UNCHANGED)
                require(mask is not None and valid is not None and mask.shape==valid.shape==(view['height'],view['width']), 'Bad TRAIN target grid')
                data_cache[name]=(torch.from_numpy(mask.astype(np.int64)).cuda(),torch.from_numpy((valid>0).astype(np.float32)).cuda())
                if len(data_cache)>16:
                    data_cache.pop(next(iter(data_cache)))
            labels,valid=data_cache[name]
            K=torch.tensor(view['K'],dtype=torch.float32,device='cuda')
            pose=torch.tensor(pose_by_name[name],dtype=torch.float32,device='cuda')
            optimizer.zero_grad(set_to_none=True)
            result=render_training(scene,field,render_partition,K,pose,view['width'],view['height'])
            loss,terms=loss_terms(result,labels,valid,weights,semantic_loss)
            require(bool(torch.isfinite(loss)),f'Nonfinite loss {arm}/{step}')
            loss.backward()
            inspect = step == 1 or step % 100 == 0 or step == 2000
            gradients={}
            if inspect:
                for group_id,group in enumerate(optimizer.param_groups):
                    require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in group['params']), 'Nonfinite trainable gradient')
                    gradients[str(group_id)]={'none':sum(p.grad is None for p in group['params']),
                                              'squared_norm':sum(float(p.grad.square().sum()) for p in group['params'] if p.grad is not None)}
            optimizer.step()
            row={'step':step,'view':name,'loss':float(loss.detach()),
                 **{key:float(value.detach()) for key,value in terms.items()},'elapsed_seconds':time.monotonic()-start}
            if inspect:
                row['gradient_groups']=gradients
                log.flush(); print(json.dumps({'arm':arm,**row}),flush=True)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            del loss,terms,result
    torch.cuda.synchronize()
    require(digest_tensors(scene.state_dict())==frozen_before,'Frozen nonrefiner scene changed')
    require(digest_tensors(scene.state_dict(),True)!=head_before,'Refiner did not update')
    require(all(bool(torch.isfinite(p).all()) for group in optimizer.param_groups for p in group['params']), 'Nonfinite endpoint')
    field_changes={} if field is None else {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()!=field_initial[k]
                                           for k,v in field.state_dict().items()}
    require(field is None or field_changes['inside_logits'] or field_changes['outside_logits'], 'No endpoint logits update')
    delta={'format':FORMAT,'arm':arm,'step':2000,'specification':SPEC,'base_checkpoint':plan['base_checkpoint'],
           'base_checkpoint_sha256':BASE_SHA,'plan_sha256':sha(Path(plan['output'])/'plan.json'),
           'manifest_sha256':sha(plan['manifest']),'pixel_protocol':'legacy_mixed_v1',
           'refiner_config':scene.refiner_config,'training_camera_sha256':camera_sha,
           'refiner_state':cpu_state(scene.refiner),'partition_state':None if field is None else cpu_state(field),
           'stage_optimizer_state':optimizer.state_dict(),'stage_torch_rng':torch.get_rng_state(),
           'stage_cuda_rng':torch.cuda.get_rng_state(),'ordinary_resume_supported':False}
    path=directory/'final_delta.pt'; torch.save(delta,path)
    receipt={'status':'completed','arm':arm,'steps':2000,'delta_path':str(path),'delta_sha256':sha(path),
             'training_log_sha256':sha(directory/'steps.jsonl'),'sample_names_sha256':hashlib.sha256(json.dumps(samples).encode()).hexdigest(),
             'frozen_nonrefiner_sha256':frozen_before,'refiner_initial_sha256':head_before,
             'refiner_final_sha256':digest_tensors(scene.state_dict(),True),'training_camera_sha256':camera_sha,
             'partition_initial_sha256':field_initial,'partition_parameter_changed':field_changes,
             'parameter_count_refiner':sum(p.numel() for p in scene.refiner.parameters()),
             'parameter_count_partition':0 if field is None else sum(p.numel() for p in field.parameters()),
             'elapsed_seconds':time.monotonic()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
             'scene_calls':2000,'partition_shader_calls':0 if field is None else 2000,
             'optimizer_steps':2000,'backwards':2000,'val_payload_reads':0,'real_rgb_payload_reads':0,
             'optimizer_groups': [{key: group[key] for key in ('name','lr','eps','betas','weight_decay')} for group in optimizer.param_groups]}
    write(directory/'training_receipt.json',receipt)
    del scene,field,optimizer,data_cache,delta; torch.cuda.empty_cache()
    return receipt


def adoption_clauses(pairs):
    require(set(pairs)==set(THRESHOLDS),'Need three matched controls and original E')
    clauses={}
    for reference,threshold in THRESHOLDS.items():
        metrics=pairs[reference]['metrics']
        clauses[reference+'_miou_gain']=metrics['miou_all']['difference']>=threshold
        clauses[reference+'_miou_ci_lower_positive']=metrics['miou_all']['paired_view_bootstrap_95_interval'][0]>0
        for name in ('background','deck','stay_cable','tower','foundation'):
            clauses[reference+'_'+name+'_guard']=metrics[name+'_iou']['difference']>=(-.001 if name=='stay_cable' else -.002)
    return {k:bool(v) for k,v in clauses.items()}


def validate_delta(delta,arm,plan,state):
    require(delta.get('format')==FORMAT and delta.get('arm')==arm and delta.get('step')==2000
            and delta.get('specification')==SPEC and delta.get('plan_sha256')==sha(Path(plan['output'])/'plan.json')
            and delta.get('base_checkpoint_sha256')==BASE_SHA and delta.get('manifest_sha256')==sha(plan['manifest'])
            and delta.get('pixel_protocol')=='legacy_mixed_v1' and delta.get('refiner_config')==state['refiner_config']
            and not delta.get('ordinary_resume_supported',True),'Wrong endpoint contract')
    camera=state['training_cameras'].detach().cpu().contiguous().numpy()
    require(delta['training_camera_sha256']==hashlib.sha256(camera.tobytes()).hexdigest(),'Endpoint camera mismatch')
    require((delta['partition_state'] is None)==(arm=='refiner_only'),'Missing/unexpected partition state')


def imports_from_snapshot(plan):
    names=('bridge_rgs.train','bridge_rgs.partition_field','bridge_rgs.partition_projection',
           'bridge_rgs.partition_rasterizer','bridge_rgs.semantic_partition','bridge_rgs.official_evaluate',
           'evaluate_projective_deck_pooling','compare_official_evaluations')
    modules={n:importlib.import_module(n) for n in names}; snapshot=Path(plan['source_snapshot'])
    observed={}
    for name,module in sys.modules.items():
        if name.startswith('bridge_rgs') or name in names:
            path=Path(module.__file__).resolve()
            require(path.is_relative_to(snapshot) and plan['source_hashes'][str(path.relative_to(snapshot))]==sha(path),f'Unfrozen import {name}')
            observed[name]={'path':str(path),'sha256':sha(path)}
    return modules,observed


def loaded_gsplat_binaries():
    records = {}
    for name, module in sys.modules.items():
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).suffix == '.so' and ('gsplat' in name or 'gsplat' in filename):
            path = Path(filename).resolve()
            records[name] = {'path': str(path), 'sha256': sha(path)}
    return records


def evaluate(plan,modules):
    import cv2
    import torch

    from bridge_rgs.partition_field import SemanticPartitionField, render_partition
    from bridge_rgs.train import load_scene
    helper=modules['evaluate_projective_deck_pooling']; official=modules['bridge_rgs.official_evaluate']
    compare=modules['compare_official_evaluations']; output=Path(plan['output'])/'evaluation'; output.mkdir()
    old=read(plan['prior_plan_path']); prior=read(plan['prior_receipt_path']); old_e=read(plan['prior_e_metrics'])
    require(old['teachers']['old']['sha256']==TEACHER_SHA,'Wrong frozen teacher')
    views=old['views']; require(len(views)==50 and sum(v['source_annotation_path'] is not None for v in views)==41,'Wrong evaluation population')
    by_old={(r['arm'],r['name']):r for r in prior['predictions']}
    rgb_rows={r['name']:{k:r[k] for k in ('name','width','height','rgb_pixels','psnr','ssim','lpips')} for r in old_e['views']}
    compare._validate(old_e)
    require(official.official_fingerprint([{k:v[k] for k in ('camera','source_image_sha256','source_annotation_sha256','rasterized_mask_sha256')}
            for v in views])==old['reference_fingerprint']==old_e['official_evaluation_fingerprint'],'Official fingerprint changed')
    records=[]; baseline=[]; beginning=time.monotonic(); scene_calls=0; shader_calls=0
    # Reproduce unchanged E semantics before evaluating any continued head.
    for arm in ('baseline',)+ARMS:
        scene,state=load_scene(plan['base_checkpoint']); field=None
        if arm!='baseline':
            receipt=read(Path(plan['output'])/arm/'training_receipt.json')
            require(receipt['status']=='completed' and receipt['steps']==2000 and sha(receipt['delta_path'])==receipt['delta_sha256'],'Training incomplete')
            delta=torch.load(receipt['delta_path'],map_location='cpu',weights_only=False); validate_delta(delta,arm,plan,state)
            scene.refiner.load_state_dict(delta['refiner_state'],strict=True)
            if arm!='refiner_only':
                with torch.no_grad(): field=SemanticPartitionField(scene.semantic_decoder(scene.splats['sem_features']),mode=arm)
                field.load_state_dict(delta['partition_state'],strict=True); field.eval().requires_grad_(False)
        scene.eval().requires_grad_(False)
        with torch.no_grad():
            for view in views:
                name,camera=view['name'],view['camera']; a,e=by_old['A',name],by_old['E',name]
                K,w,h,back=official.distortion_render_grid(camera['K'],camera['distortion'],camera['width'],camera['height'],'legacy_mixed_v1')
                cache=prior['h3_canvases'][name]
                require([w,h]==cache['boundary']['canvas'] and np.array_equal(K,np.asarray(cache['boundary']['canvas_K'],np.float32)),'Teacher grid changed')
                result=render_training(scene,field,render_partition,torch.tensor(K,device='cuda'),
                                       torch.tensor(camera['w2c'],dtype=torch.float32,device='cuda'),w,h)
                scene_calls+=1; shader_calls+=field is not None
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
        del scene,state,field; torch.cuda.empty_cache()
    require(len(baseline)==50 and len(records)==200 and {(r['arm'],r['name']) for r in records}=={(a,v['name']) for a in ARMS for v in views},'Incomplete prediction barrier')
    predictions_finished_utc = datetime.now(UTC).isoformat()
    write(output/'predictions_receipt.json',{'status':'all_600_masks_before_GT','records':records,'baseline_E_exact':baseline,
          'scene_calls':scene_calls,'partition_shader_calls':int(shader_calls),'annotation_payload_reads':0,
          'predictions_finished_utc':predictions_finished_utc})
    scoring_started_utc = datetime.now(UTC).isoformat()
    rows={(a,k):[] for a in ARMS for k in ('joint','scene','raw')}; by_new={(r['arm'],r['name']):r for r in records}; reads=0
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
    inference={'protocol':'semantic_partition_eps_endpoint_v1','teacher_weight':.5,'RGB':'E byte/score reuse',
               'baseline_E_exact':True,'camera_source':'same frozen H3 geometry with new categorical field/head'}
    for (arm,kind),view_rows in rows.items():
        pooled=sum(np.asarray(r['confusion_matrix'],np.int64) for r in view_rows if 'confusion_matrix' in r); iou=compare._iou(pooled)
        value={'evaluation_family':old['evaluation_family'],'scoring_protocol':old['scoring_protocol'],
               'official_evaluation_fingerprint':old['reference_fingerprint'],'inference_protocol':dict(inference,kind=kind),
               'validation_views':50,'semantic_validation_views':41,'views':view_rows,'confusion_matrix':pooled.tolist(),
               'iou':iou.tolist(),'miou_all':float(np.nanmean(iou)),'miou_foreground':float(np.nanmean(iou[1:])),
               **{k:float(np.mean([r[k] for r in view_rows])) for k in ('psnr','ssim','lpips')},
               'provenance':{'base_checkpoint_sha256':BASE_SHA,'teacher_cache_checkpoint_sha256':TEACHER_SHA if kind=='joint' else None,
                             'training_delta':read(Path(plan['output'])/arm/'training_receipt.json')['delta_sha256'],
                             'rgb_metrics':{'path':plan['prior_e_metrics'],'sha256':sha(plan['prior_e_metrics'])},
                             'single_shared_geometry_system':False}}
        compare._validate(value); path=output/arm/('official_metrics.json' if kind=='joint' else kind+'_official_metrics.json')
        write(path,value); metrics[arm,kind]=value; metric_records[arm+'_'+kind]={'path':str(path),'sha256':sha(path)}
    pairs={}; pair_records={}
    for reference in THRESHOLDS:
        pair=compare.paired_official_comparison(old_e if reference=='E' else metrics[reference,'joint'],metrics['integrated','joint'],repeats=5000,seed=20260926)
        pair.update(candidate_arm='integrated',reference_arm=reference); pairs[reference]=pair
        path=output/f'paired_integrated_minus_{reference}.json';write(path,pair);pair_records[reference]={'path':str(path),'sha256':sha(path)}
    secondary_records = {}
    for arm in ARMS[1:]:
        old_record = plan['old_samearm_metrics'][arm]
        require(sha(old_record['path']) == old_record['sha256'], 'Old same-arm metric changed')
        old_metrics = read(old_record['path'])
        pair = compare.paired_official_comparison(old_metrics, metrics[arm,'joint'], repeats=5000, seed=20260926)
        pair.update(candidate_arm=arm, reference_arm=arm, candidate_run=plan['output'], reference_run=plan['old_run'],
                    comparison_kind='new_samearm_minus_oldsamearm', descriptive_only=True, old_metrics_source=old_record)
        path=output/f'paired_new_{arm}_minus_old_{arm}.json'; write(path,pair)
        secondary_records[arm]={'path':str(path),'sha256':sha(path)}
    clauses=adoption_clauses(pairs); gate={'passed':all(clauses.values()),'clauses':clauses,'thresholds':THRESHOLDS,
            'decision':'eligible_for_review_not_automatic_adoption' if all(clauses.values()) else 'stop_fixed_partition_candidate',
            'rgb':'unchanged E; do not attribute prior RGB gains to semantic partition'}
    write(output/'system_gate.json',gate)
    result={'status':'completed','scene_calls':scene_calls,'partition_shader_calls':int(shader_calls),'teacher_calls':0,
            'baseline_E_masks_byte_exact':50,'new_masks':600,'annotation_payload_reads':reads,'gt_rgb_payload_reads':0,
            'metrics':metric_records,'comparisons':pair_records,'secondary_comparisons':secondary_records,'gate':gate,'gate_sha256':sha(output/'system_gate.json'),
            'plan_sha256':sha(Path(plan['output'])/'plan.json'),
            'predictions_receipt_sha256':sha(output/'predictions_receipt.json'),
            'predictions_finished_utc':predictions_finished_utc,'scoring_started_utc':scoring_started_utc,
            'scoring_finished_utc':datetime.now(UTC).isoformat(),
            'elapsed_seconds':time.monotonic()-beginning,'timing_scope':'semantic inference with teacher/E RGB caches; not deployment FPS'}
    write(output/'execution_receipt.json',result)
    return result


def execute(path,expected_sha):
    require(sha(path)==expected_sha,'Plan SHA mismatch'); plan=read(path); output=Path(plan['output'])
    require(not (output/'execution_receipt.json').exists(),'One attempt only')
    verify(plan); sys.path.insert(0,plan['source_snapshot']); modules,imports=imports_from_snapshot(plan)
    report={'status':'running','plan_sha256':expected_sha,'actual_imports':imports,'gpu_before':gpu_inventory(),
            'training':[],'new_val_annotation_payload_reads':0,'cost_scope':'RustDesk-shared GPU; not exclusive hardware timing'}
    write(output/'execution_receipt.json',report); beginning=time.monotonic()
    def expired(*_):
        raise TimeoutError('Fixed full experiment budget exhausted')
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['execution_budget']['internal_seconds'])
    try:
        modules['bridge_rgs.train'].torch.set_num_threads(4)
        for arm in ARMS:
            report['training'].append(train_arm(plan,arm));write(output/'execution_receipt.json',report,replace=True)
        for key in ('sample_names_sha256','frozen_nonrefiner_sha256','refiner_initial_sha256','training_camera_sha256'):
            require(len({r[key] for r in report['training']})==1,f'Matched-arm contract changed: {key}')
        require(len({json.dumps(r['partition_initial_sha256'],sort_keys=True) for r in report['training'][1:]})==1,'Partition initialization changed')
        report['evaluation']=evaluate(plan,modules)
        report['new_val_annotation_payload_reads']=report['evaluation']['annotation_payload_reads']
        report['actual_imports'] = imports_from_snapshot(plan)[1]
        report['loaded_gsplat_binary_sources'] = loaded_gsplat_binaries()
        report['triton_binary_scope'] = 'Frozen partition shader Python source and bound Triton version; JIT binary bytes not independently exported'
        verify(plan);report.update(status='completed',bound_sources_inputs_unchanged=True)
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-beginning
        write(output/'execution_receipt.json',report,replace=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare',type=Path);choice.add_argument('--execute',type=Path);choice.add_argument('--print-spec',action='store_true')
    parser.add_argument('--predecessor',type=Path,default=PREDECESSOR);parser.add_argument('--expected-plan-sha256')
    args=parser.parse_args()
    if args.print_spec:
        print(json.dumps(SPEC,indent=2));return
    require(os.environ.get('PYTHONDONTWRITEBYTECODE')=='1','Disable bytecode generation')
    if args.prepare:
        prepare(args.prepare,args.predecessor)
    else:
        require(args.expected_plan_sha256 is not None,'Expected plan SHA required')
        execute(args.execute,args.expected_plan_sha256)


if __name__=='__main__':
    main()
