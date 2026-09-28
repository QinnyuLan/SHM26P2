"""Fixed direct-q / original-q matched H3 head continuation and official endpoint.

CPU preparation only until explicitly launched. Original field evidence stays
unchanged; this never maps q into features or writes a production checkpoint.
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
PARENT = Path('/mnt/data/SHM2026/runs/raw_simplex_em_fw_v1')
EVAL_SOURCE = Path('/mnt/data/SHM2026/runs/semantic_partition_eps_matched_v1')
PARENT_SHAS = {
 'plan.json': '29b066604846598a679a7fb0f1d25d88fce452cc25019adb3b552ad5a760fdcc',
 'execution_receipt.json': 'b129e790bb514dd1180e41978264af8b8ed6ee2ce9f297f24e2f17e749b9e7f0',
 'independent_cpu_review.json': '9c2dcfc8daf108545f06c4c9fda5ccd93ddca007af667f7752b37f01fe67f743'}
EVAL_PLAN_SHA = 'c3974d205afaa3c897cb6412a086879019e9b999fd284cac20c7d2dad58afffb'
BASE_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
MANIFEST_SHA = '551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa'
TEACHER_SHA = '00f5b84ac9a56c39512c5b8e43f70110397923feea1a2b4c78bdffd1f5524bff'
ARMS = ('original_q', 'optimized_q')
READOUTS = ('original_q_zero', 'optimized_q_zero') + ARMS
FORMAT = 'direct_q_head_matched_delta_v1'
WEIGHTS = [.45604488253593445, .9227690100669861, .8530434370040894,
           1.1935913562774658, 1.5745513439178467]
THRESHOLDS = {'original_q': .001, 'E': .002}
NUMERICS = {'cudnn_allow_tf32': False, 'matmul_allow_tf32': False,
            'matmul_precision': 'highest', 'cudnn_benchmark': False}
SPEC = {
 'protocol': 'direct_q_head_matched_v1', 'arms': list(ARMS), 'candidate': 'optimized_q',
 'descriptive_zero_step': list(READOUTS[:2]), 'base_sha256': BASE_SHA,
 'steps': 2000, 'seed': 42, 'train_count': 259,
 'sampler': 'sorted labeled TRAIN, independent default_rng(42), full epoch shuffles',
 'grid': 'native legacy_mixed_v1 SH3; no crop/flip/augmentation',
 'optimizer': {'type': 'Adam', 'head_lr': .0003, 'eps': 1e-8,
               'betas': [.9, .999], 'weight_decay': 0.},
 'objective': {'final_weighted_ce': 1., 'final_lovasz': .2, 'residual_square_mean': .001,
               'raw_ce': 'omitted: fixed q makes it a head-independent constant'},
 'class_weights_fp32': WEIGHTS, 'numerics': NUMERICS, 'amp': False,
 'q': {'original': 'original H3 classifier FP32 softmax; no normalization or floor',
       'optimized': 'completed EM/FW final_q_delta q_renderer; exact master cast',
       'requires_grad': False, 'row_sum_atol': 1e-6, 'feature_mapping': False},
 'frozen': ['geometry', 'opacity', 'RGB', 'features', 'decoder', 'cameras', 'q'],
 'context': 'original H3 features/RGB/depth/alpha/moments; only p3d/log/entropy prior changes',
 'endpoint': 'last 2000 only; no best or zero-step selection',
 'preflight': {'view': '002.png', 'labels': False, 'scene_calls': 3, 'shader_calls': 4,
               'guard': 'native and original-q exact; both helper raw/alpha exact frozen direct_q'},
 'evaluation': {'teacher_sha256': TEACHER_SHA, 'teacher_weight': .5, 'views': 50, 'annotations': 41,
   'RGB': 'E byte/score reuse only; teacher sees original H3 canvas',
   'baseline_guard': 'native H3 joint masks exact E; original-q-zero p3d/head exact native',
   'prediction_barrier': 'all 4x50x3 fresh masks and 50 native E masks before GT',
   'scene_calls': 250, 'shader_calls': 200, 'teacher_calls': 0},
 'bootstrap': {'repeats': 5000, 'seed': 20260926}, 'gain_thresholds': THRESHOLDS,
 'class_guard': {'stay_cable': -.001, 'others': -.002},
 'internal_seconds': 1500, 'external_seconds': 1560, 'retries': 0,
 'cost_scope': 'matched compute only for head adaptation; candidate q has additional prior optimization cost',
 'claim': 'standard conditional optimization and compatibility test, not a new representation or RGB gain',
}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

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

def cpu_state(module):
    return {k: v.detach().cpu().clone() for k,v in module.state_dict().items()}

def loaded_gsplat_binaries():
    records = {}
    for name, module in sys.modules.items():
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).suffix == '.so' and ('gsplat' in name or 'gsplat' in filename):
            path = Path(filename).resolve()
            records[name] = {'path': str(path), 'sha256': sha(path)}
    return records

def write(path, value, *, replace=False):
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    if replace:
        pending = path.with_name(f'.{path.name}.{os.getpid()}.pending')
        with pending.open('x') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        pending.replace(path)
    else:
        with path.open('x') as stream:
            stream.write(payload)


def validate_parent(plan, execution, launch, audit, plan_sha, execution_sha):
    require(launch.get('status') == 'completed' and launch.get('natural_completion') is True
            and launch.get('exit_code') == 0 and launch.get('execution_receipt_sha256') == execution_sha,
            'Naturally completed q producer required')
    require(execution.get('status') == 'completed' and audit.get('status') == 'passed'
            and execution.get('inputs_sources_unchanged') is True
            and execution.get('state_unchanged_before_restore') is True,
            'Completed independently audited q producer required')
    require(all(r.get('plan_sha256') == plan_sha for r in (execution, launch, audit))
            and audit.get('execution_receipt_sha256') == execution_sha, 'Q producer receipt binding')
    require(plan['specification']['protocol'] == 'raw_simplex_em_fw_v1'
            and audit['q']['shape'] == [498136, 5], 'Wrong fixed q producer')


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Use a new data-disk output')
    inputs = {}
    def bind(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        require(expected is None or digest == expected, f'Changed dependency: {path}')
        inputs[str(path)] = digest
        return {'path': str(path), 'sha256': digest}
    lineage = {n: bind(PARENT/n, s) for n, s in PARENT_SHAS.items()}
    lineage['launch_receipt.json'] = bind(PARENT/'launch_receipt.json')
    parent, execution, launch, audit = (read(PARENT/n) for n in
        ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json'))
    validate_parent(parent, execution, launch, audit, PARENT_SHAS['plan.json'], PARENT_SHAS['execution_receipt.json'])
    analysis_record = bind(PARENT/'analysis.json', execution['analysis_sha256'])
    require(audit['analysis_sha256'] == analysis_record['sha256'], 'Unbound q analysis')
    analysis = read(analysis_record['path']); q_record = analysis['q_delta']
    require(Path(q_record['path']) == PARENT/'final_q_delta.npz', 'Wrong q delta endpoint')
    bind(q_record['path'], q_record['sha256'])
    bind(EVAL_SOURCE/'plan.json', EVAL_PLAN_SHA)
    evaluation_plan = read(EVAL_SOURCE/'plan.json')
    for name in ('execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json'):
        bind(EVAL_SOURCE/name)
    old_exec, old_launch, old_audit = (read(EVAL_SOURCE/n) for n in
        ('execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json'))
    require(old_exec['status'] == 'completed' and old_launch['natural_completion'] is True
            and old_launch['exit_code'] == 0 and old_audit['status'] == 'passed'
            and old_launch['execution_receipt_sha256'] == sha(EVAL_SOURCE/'execution_receipt.json')
            and all(d['plan_sha256'] == EVAL_PLAN_SHA for d in (old_exec, old_launch, old_audit)), 'Evaluation helper lineage')
    for origin in (parent, evaluation_plan):
        for path, digest in origin['input_hashes'].items():
            bind(path, digest)
    require(parent['base_checkpoint'] == evaluation_plan['base_checkpoint']
            and parent['manifest'] == evaluation_plan['manifest'], 'Mismatched base field/grid')
    bind(parent['base_checkpoint'], BASE_SHA); bind(parent['manifest'], MANIFEST_SHA)
    source, eval_source = Path(parent['source_snapshot']), Path(evaluation_plan['source_snapshot'])
    require(files(source) == parent['source_hashes'] and files(eval_source) == evaluation_plan['source_hashes'], 'Old frozen source changed')
    snapshot = output/'source_snapshot'
    shutil.copytree(source/'bridge_rgs', snapshot/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    inherited = files(snapshot/'bridge_rgs')
    require(all(parent['source_hashes']['bridge_rgs/'+k] == v for k,v in inherited.items()), 'Package copy mismatch')
    # The one new helper is explicit. No live projection/model/scoring modules are copied.
    helper = ROOT/'src/bridge_rgs/direct_q_render.py'
    require(not (snapshot/'bridge_rgs'/helper.name).exists(), 'Unexpected inherited direct-q adapter')
    shutil.copy2(helper, snapshot/'bridge_rgs'/helper.name)
    bind(source/'preflight_simplex_scene.py', parent['source_hashes']['preflight_simplex_scene.py'])
    shutil.copy2(source/'preflight_simplex_scene.py', snapshot/'preflight_simplex_scene.py')
    for name in ('evaluate_projective_deck_pooling.py', 'compare_official_evaluations.py'):
        bind(eval_source/name, evaluation_plan['source_hashes'][name]); shutil.copy2(eval_source/name, snapshot/name)
    for path in (Path(__file__), ROOT/'tests/test_direct_q_head_matched.py', ROOT/'tests/test_direct_q_render.py',
                 ROOT/'docs/direct_q_head_matched_protocol.md', ROOT/'uv.lock'):
        shutil.copy2(path, snapshot/path.name)
    plan = {key: parent[key] for key in ('base_checkpoint', 'manifest', 'expected_gsplat_binary',
            'installed_sources', 'runtime_versions', 'environment', 'numerics')}
    plan.update({key:evaluation_plan[key] for key in ('root','scoring_source_review','prior_plan_path',
                'prior_receipt_path','prior_predictions_path','prior_e_metrics')})
    plan.update(specification=SPEC, status='prepared_pending_root_execution', output=str(output),
        source_snapshot=str(snapshot), source_hashes=files(snapshot), input_hashes=inputs,
        inherited_package_hashes=inherited, parent_lineage=lineage, parent_analysis=analysis_record,
        q_delta=q_record, optimized_q_renderer_sha256=audit['q']['renderer_sha256'],
        base_checkpoint_sha256=BASE_SHA, training_views=population(read(parent['manifest'])),
        training_order=sample_order(), execution_budget={'internal_seconds':1500,'external_seconds':1560},
        prior_q_optimization_cost={'em_fw_elapsed_seconds':execution['elapsed_seconds'],
            'em_fw_counts':execution['counts'], 'earlier_PG_FW_lineage':parent['parent_lineage'],
            'scope':'EM/FW plus previous PG/FW are candidate-only prior costs; not equal total algorithm compute'},
        prepare_checkpoint_loads=0, prepare_target_decodes=0)
    require(plan['numerics'] == NUMERICS, 'Unexpected inherited numerical policy')
    write(output/'plan.json', plan)
    print(json.dumps({'plan':str(output/'plan.json'),'sha256':sha(output/'plan.json'),
                      'sources':len(plan['source_hashes']),'inputs':len(inputs)}))


def verify(plan):
    require(plan['specification'] == SPEC and plan['numerics'] == NUMERICS, 'Specification changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Use frozen runner')
    require(files(plan['source_snapshot']) == plan['source_hashes'], 'Frozen source changed')
    require(plan['training_order'] == sample_order() and plan['training_views'] == population(read(plan['manifest'])), 'TRAIN schedule changed')
    for path, digest in {**plan['input_hashes'], **plan['installed_sources']}.items():
        require(sha(path) == digest, f'Changed bound input: {path}')
    for name, value in plan['runtime_versions'].items():
        require(importlib.metadata.version(name) == value, f'Changed runtime: {name}')
    for name, value in plan['environment'].items():
        require(os.environ.get(name) == value, f'Changed environment: {name}')
    require(sha(plan['expected_gsplat_binary']['path']) == plan['expected_gsplat_binary']['sha256'], 'Changed gsplat binary')


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def configure_parameters(scene):
    scene.requires_grad_(False); scene.refiner.requires_grad_(True); scene.train()
    require(all(p.requires_grad == name.startswith('refiner.') for name,p in scene.named_parameters()), 'Wrong trainable scope')
    import torch
    optimizer = torch.optim.Adam([{'params':list(scene.refiner.parameters()), 'name':'head'}],
                                 lr=.0003, betas=(.9,.999), eps=1e-8, weight_decay=0.)
    require(not optimizer.state, 'Optimizer not fresh')
    return optimizer


def loss_terms(result, labels, valid, weights, semantic_loss):
    final = semantic_loss(result['probabilities'], labels, valid, weights, lovasz_weight=.2)
    regularization = result['residual'].square().mean()
    return final + .001*regularization, {'final':final, 'residual_square':regularization}


def load_q(scene, arm, plan):
    import torch

    from bridge_rgs.direct_q_render import validate_q
    require(arm in READOUTS, 'Unknown q readout')
    if arm.startswith('original_q'):
        with torch.no_grad():
            q = scene.semantic_decoder(scene.splats['sem_features']).softmax(-1).detach().contiguous()
    else:
        record = plan['q_delta']; require(sha(record['path']) == record['sha256'], 'Optimized q sidecar changed')
        with np.load(record['path'], allow_pickle=False) as arrays:
            master, rendered = arrays['q_master'], arrays['q_renderer']
            require(master.dtype == np.float64 and rendered.dtype == np.float32
                    and master.shape == rendered.shape == (len(scene.splats['means']),5)
                    and np.array_equal(master.astype(np.float32),rendered), 'Wrong actual q cast/row order contract')
            require(hashlib.sha256(rendered.tobytes()).hexdigest() == plan['optimized_q_renderer_sha256'], 'Wrong optimized q identity')
            q = torch.from_numpy(rendered.copy()).to(scene.splats['means'].device).contiguous()
    validate_q(q,scene); require(not q.requires_grad, 'q must be frozen')
    return q


def load_base(plan):
    import torch

    from bridge_rgs.train import load_scene
    scene,state = load_scene(plan['base_checkpoint'])
    require(scene.pixel_protocol == 'legacy_mixed_v1' and scene.sh_degree == 3
            and scene.mip_filter_config is None, 'Wrong base field')
    train = [v for v in read(plan['manifest'])['views'] if v['split']=='train']
    cameras = state['training_cameras'].detach().cpu().numpy()
    require(cameras.shape == (350,4,4) and np.array_equal(cameras,np.asarray([v['w2c_original'] for v in train],np.float32)), 'Saved TRAIN cameras changed')
    require(all(torch.isfinite(t).all().item() for t in scene.state_dict().values()), 'Nonfinite base')
    return scene,state,{v['name']:cameras[i] for i,v in enumerate(train)}


def head_from_context(scene, context, p3d):
    kwargs = {'depth_moments':context['depth_moments']} if 'depth_moments' in context else {}
    residual = scene.refiner(context['features'],context['rgb'],context['depth'],context['alpha'],p3d=p3d,**kwargs)
    return (p3d.log()+residual).softmax(-1)


def wiring_preflight(plan, modules):
    import torch

    from bridge_rgs.direct_q_render import render_direct_q
    from bridge_rgs.partition_rasterizer import rasterize_semantic_partition
    old = modules['preflight_simplex_scene']
    scene,state,_ = load_base(plan); scene.eval().requires_grad_(False)
    before = digest_tensors(scene.state_dict()); head_before = digest_tensors(scene.state_dict(),True)
    view = next(v for v in plan['training_views'] if v['name']=='002.png')
    K=torch.tensor(view['K'],device='cuda',dtype=torch.float32)
    pose=torch.tensor(view['w2c_original'],device='cuda',dtype=torch.float32); w,h=view['width'],view['height']
    counters={'scene':0,'gsplat':0}; report={'name':'002.png','target_reads':0,'checks':{}}
    with torch.no_grad():
        context,captured=old.capture_context(scene,K,pose,w,h,counters)
        native_final=head_from_context(scene,context,context['p3d'])
        for arm in ARMS:
            q=load_q(scene,arm,plan); result=render_direct_q(scene,q,K,pose,w,h)
            raw,alpha=old.direct_q(result['info'],q,rasterize_semantic_partition,width=w,height=h)
            checks={k:torch.equal(result[k],context[k]) for k in ('rgb','depth','alpha','features','depth_moments') if k in context}
            checks.update(frozen_formula_raw=torch.equal(result['raw'],raw), frozen_formula_alpha=torch.equal(result['direct_q_alpha'],alpha))
            if arm=='original_q':
                checks.update(native_raw=torch.equal(result['raw'],captured['raw']),
                    native_semantic_alpha=torch.equal(result['direct_q_alpha'],captured['alpha']),
                    native_p3d=torch.equal(result['p3d'],context['p3d']),native_head=torch.equal(result['probabilities'],native_final))
            report['checks'][arm]=checks; report[arm+'_q_sha256']=tensor_sha(q)
    report.update(scene_calls=3,shader_calls=4,native_gsplat_calls=6,
        frozen_state_exact=digest_tensors(scene.state_dict())==before and digest_tensors(scene.state_dict(),True)==head_before)
    report['passed']=all(all(c.values()) for c in report['checks'].values()) and report['frozen_state_exact']
    write(Path(plan['output'])/'wiring_preflight.json',report)
    require(report['passed'],'Direct-q/native wiring mismatch: stop without changing gates')
    del scene,state; torch.cuda.empty_cache()
    return report


def train_arm(plan, arm):
    import cv2
    import torch

    from bridge_rgs.direct_q_render import render_direct_q
    from bridge_rgs.io import seed_everything
    from bridge_rgs.losses import semantic_loss
    require(arm in ARMS, 'Unknown training arm'); seed_everything(42)
    directory=Path(plan['output'])/arm; directory.mkdir(); start=time.monotonic()
    scene,state,pose_by_name=load_base(plan)
    q=load_q(scene,arm,plan); q_hash=tensor_sha(q)
    camera_hash=tensor_sha(state['training_cameras'])
    before=digest_tensors(scene.state_dict()); head_before=digest_tensors(scene.state_dict(),True)
    optimizer=configure_parameters(scene)
    weights=torch.tensor(WEIGHTS,device='cuda',dtype=torch.float32)
    data_cache={}; samples=[]; torch.cuda.reset_peak_memory_stats(); del state
    with (directory/'steps.jsonl').open('x') as log:
        for step,index in enumerate(plan['training_order'],1):
            view=plan['training_views'][index]; name=view['name']; samples.append(name)
            if name not in data_cache:
                mask=cv2.imread(view['mask_path'],cv2.IMREAD_UNCHANGED); valid=cv2.imread(view['valid_path'],cv2.IMREAD_UNCHANGED)
                require(mask is not None and valid is not None and mask.shape==valid.shape==(view['height'],view['width']), 'Bad TRAIN target grid')
                data_cache[name]=(torch.from_numpy(mask.astype(np.int64)).cuda(),torch.from_numpy((valid>0).astype(np.float32)).cuda())
                if len(data_cache)>16: data_cache.pop(next(iter(data_cache)))
            labels,valid=data_cache[name]
            optimizer.zero_grad(set_to_none=True)
            result=render_direct_q(scene,q,torch.tensor(view['K'],device='cuda',dtype=torch.float32),
                       torch.tensor(pose_by_name[name],device='cuda',dtype=torch.float32),view['width'],view['height'])
            loss,terms=loss_terms(result,labels,valid,weights,semantic_loss)
            require(bool(torch.isfinite(loss)), 'Nonfinite head objective'); loss.backward()
            require(q.grad is None and all(p.grad is None for n,p in scene.named_parameters() if not n.startswith('refiner.')), 'Frozen field gradient leaked')
            if step==1 or step%100==0:
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in scene.refiner.parameters()), 'Nonfinite head gradient')
            optimizer.step()
            row={'step':step,'view':name,'loss':float(loss.detach()),
                 **{k:float(v.detach()) for k,v in terms.items()},'elapsed_seconds':time.monotonic()-start}
            log.write(json.dumps(row,allow_nan=False)+'\n')
            if step==1 or step%100==0:
                log.flush();print(json.dumps({'arm':arm,**row}),flush=True)
            del loss,terms,result
    torch.cuda.synchronize()
    require(digest_tensors(scene.state_dict())==before and tensor_sha(q)==q_hash, 'Frozen q/context changed')
    require(digest_tensors(scene.state_dict(),True)!=head_before, 'Head did not update')
    require(all(bool(torch.isfinite(p).all()) for p in scene.refiner.parameters()), 'Nonfinite head endpoint')
    delta={'format':FORMAT,'arm':arm,'step':2000,'specification':SPEC,'base_checkpoint':plan['base_checkpoint'],
        'base_checkpoint_sha256':BASE_SHA,'plan_sha256':sha(Path(plan['output'])/'plan.json'),
        'manifest_sha256':MANIFEST_SHA,'pixel_protocol':'legacy_mixed_v1','refiner_config':scene.refiner_config,
        'training_camera_sha256':camera_hash,'q_sha256':q_hash,'q_source':'original classifier' if arm=='original_q' else plan['q_delta'],
        'refiner_state':cpu_state(scene.refiner),'stage_optimizer_state':optimizer.state_dict(),
        'stage_torch_rng':torch.get_rng_state(),'stage_cuda_rng':torch.cuda.get_rng_state(),
        'ordinary_resume_supported':False,'production_checkpoint':False}
    path=directory/'final_delta.pt';torch.save(delta,path)
    receipt={'status':'completed','arm':arm,'steps':2000,'delta_path':str(path),'delta_sha256':sha(path),
        'training_log_sha256':sha(directory/'steps.jsonl'),'sample_names_sha256':hashlib.sha256(json.dumps(samples).encode()).hexdigest(),
        'frozen_nonrefiner_sha256':before,'refiner_initial_sha256':head_before,'refiner_final_sha256':digest_tensors(scene.state_dict(),True),
        'training_camera_sha256':camera_hash,'q_initial_sha256':q_hash,'q_final_sha256':tensor_sha(q),
        'parameter_count_refiner':sum(p.numel() for p in scene.refiner.parameters()),
        'elapsed_seconds':time.monotonic()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'scene_calls':2000,'direct_q_shader_calls':2000,'optimizer_steps':2000,'backwards':2000,
        'val_payload_reads':0,'real_rgb_payload_reads':0,
        'optimizer_groups':[{k:g[k] for k in ('name','lr','eps','betas','weight_decay')} for g in optimizer.param_groups]}
    write(directory/'training_receipt.json',receipt)
    del scene,q,optimizer,data_cache,delta;torch.cuda.empty_cache()
    return receipt


def validate_delta(delta, arm, plan, state, q):
    require(delta.get('format')==FORMAT and delta.get('arm')==arm and delta.get('step')==2000
        and delta.get('specification')==SPEC and delta.get('plan_sha256')==sha(Path(plan['output'])/'plan.json')
        and delta.get('base_checkpoint_sha256')==BASE_SHA and delta.get('manifest_sha256')==MANIFEST_SHA
        and delta.get('refiner_config')==state['refiner_config'] and delta.get('pixel_protocol')=='legacy_mixed_v1'
        and delta.get('ordinary_resume_supported') is False and delta.get('production_checkpoint') is False,
        'Invalid matched endpoint')
    require(delta['training_camera_sha256']==tensor_sha(state['training_cameras'])
            and delta['q_sha256']==tensor_sha(q), 'Endpoint q/camera identity changed')


def adoption_clauses(pairs):
    require(set(pairs)==set(THRESHOLDS),'Need matched original-q and E references')
    clauses={}
    for reference,threshold in THRESHOLDS.items():
        metrics=pairs[reference]['metrics']
        clauses[reference+'_miou_gain']=metrics['miou_all']['difference']>=threshold
        clauses[reference+'_miou_ci_lower_positive']=metrics['miou_all']['paired_view_bootstrap_95_interval'][0]>0
        for name in ('background','deck','stay_cable','tower','foundation'):
            clauses[reference+'_'+name+'_guard']=metrics[name+'_iou']['difference']>=(-.001 if name=='stay_cable' else -.002)
    return {k:bool(v) for k,v in clauses.items()}


def imports_from_snapshot(plan):
    names=('bridge_rgs.train','bridge_rgs.direct_q_render','bridge_rgs.partition_rasterizer',
           'bridge_rgs.official_evaluate','preflight_simplex_scene',
           'evaluate_projective_deck_pooling','compare_official_evaluations')
    modules={n:importlib.import_module(n) for n in names};snapshot=Path(plan['source_snapshot']);observed={}
    for name,module in sys.modules.items():
        if name.startswith('bridge_rgs') or name in names:
            path=Path(module.__file__).resolve()
            require(path.is_relative_to(snapshot) and plan['source_hashes'][str(path.relative_to(snapshot))]==sha(path),f'Unfrozen import {name}')
            observed[name]={'path':str(path),'sha256':sha(path)}
    return modules,observed


def prediction_barrier(records, baseline, adapter_checks, views):
    expected={(arm,v['name']) for arm in READOUTS for v in views}
    require(len(views)==50 and len(expected)==200 and len(records)==200
            and {(r['arm'],r['name']) for r in records}==expected
            and all(set(r['masks'])=={'joint','scene','raw'} for r in records)
            and len(baseline)==50 and len(adapter_checks)==50
            and {r['name'] for r in adapter_checks}=={v['name'] for v in views}
            and all(set(r['checks'])=={'p3d','probabilities'} and all(r['checks'].values()) for r in adapter_checks),
            'Incomplete predictions or native/adapter identity barrier')


def evaluate(plan,modules):
    import cv2
    import torch

    from bridge_rgs.direct_q_render import render_direct_q
    
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
    records=[]; baseline=[]; beginning=time.monotonic(); scene_calls=0; shader_calls=0; native_hashes={}; adapter_checks=[]
    # Reproduce unchanged E semantics before evaluating any continued head.
    for arm in ('baseline',)+READOUTS:
        scene,state,_=load_base(plan)
        q=None if arm=='baseline' else load_q(scene,arm,plan)
        if arm in ARMS:
            receipt=read(Path(plan['output'])/arm/'training_receipt.json')
            require(receipt['status']=='completed' and receipt['steps']==2000 and sha(receipt['delta_path'])==receipt['delta_sha256'],'Training incomplete')
            delta=torch.load(receipt['delta_path'],map_location='cpu',weights_only=False); validate_delta(delta,arm,plan,state,q)
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
                             'training_delta':read(Path(plan['output'])/arm/'training_receipt.json')['delta_sha256'] if arm in ARMS else None,
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

def read_numerical_flags(torch):
    return {'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'matmul_precision':torch.get_float32_matmul_precision(),'cudnn_benchmark':torch.backends.cudnn.benchmark}


def numerical_flags(torch, values=None):
    old=read_numerical_flags(torch)
    values=NUMERICS if values is None else values
    torch.backends.cudnn.allow_tf32=values['cudnn_allow_tf32']
    torch.backends.cuda.matmul.allow_tf32=values['matmul_allow_tf32']
    torch.set_float32_matmul_precision(values['matmul_precision'])
    torch.backends.cudnn.benchmark=values['cudnn_benchmark']
    return old


def validate_matched_receipts(receipts):
    require([r['arm'] for r in receipts]==list(ARMS) and all(r['status']=='completed' and r['steps']==2000 for r in receipts), 'Two fixed endpoints required')
    for key in ('sample_names_sha256','frozen_nonrefiner_sha256','refiner_initial_sha256','training_camera_sha256','parameter_count_refiner'):
        require(len({r[key] for r in receipts})==1, f'Matched arm contract differs: {key}')
    require(all(r['q_initial_sha256']==r['q_final_sha256'] for r in receipts), 'q changed during head fitting')
    require(receipts[0]['optimizer_groups']==receipts[1]['optimizer_groups'], 'Unequal optimizer settings')


def execute(path, expected_sha):
    require(sha(path)==expected_sha,'Plan SHA mismatch');plan=read(path);output=Path(plan['output'])
    require(not (output/'execution_receipt.json').exists() and not (output/'execution_started.json').exists(), 'One attempt only')
    write(output/'execution_started.json',{'plan_sha256':expected_sha,'started_utc':datetime.now(UTC).isoformat()})
    beginning=time.monotonic(); previous=None; torch=None
    report={'status':'running','plan_sha256':expected_sha,'training':[],'new_val_annotation_payload_reads':0,
            'teacher_calls':0,'production_checkpoint_writes':0,'cost_scope':SPEC['cost_scope'],
            'prior_q_optimization_cost':plan['prior_q_optimization_cost']}
    def expired(*_): raise TimeoutError('Fixed 1500 second whole-run budget exhausted')
    previous_handler=signal.signal(signal.SIGALRM,expired);signal.alarm(1500)
    try:
        verify(plan);report['gpu_before']=gpu_inventory()
        sys.path.insert(0,plan['source_snapshot']);modules,imports=imports_from_snapshot(plan)
        torch=modules['bridge_rgs.train'].torch;torch.set_num_threads(4);previous=numerical_flags(torch)
        report['numerics_actual']=read_numerical_flags(torch)
        require(report['numerics_actual']==NUMERICS and not torch.is_autocast_enabled(), 'Numerical policy not applied')
        report['amp_enabled']=bool(torch.is_autocast_enabled());report['actual_imports']=imports
        report['wiring_preflight']=wiring_preflight(plan,modules)
        for arm in ARMS:
            report['training'].append(train_arm(plan,arm));write(output/'execution_receipt.json',report,replace=(output/'execution_receipt.json').exists())
        validate_matched_receipts(report['training'])
        report['evaluation']=evaluate(plan,modules)
        report['new_val_annotation_payload_reads']=report['evaluation']['annotation_payload_reads']
        report['actual_imports']=imports_from_snapshot(plan)[1]
        report['loaded_gsplat_binary_sources']=loaded_gsplat_binaries()
        require(plan['expected_gsplat_binary'] in report['loaded_gsplat_binary_sources'].values(), 'Actual gsplat binary changed')
        verify(plan);report.update(status='completed',bound_sources_inputs_unchanged=True)
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous_handler)
        restoration_error=None
        report['numerics_restored']=None
        if previous is not None:
            try:
                numerical_flags(torch,previous)
                report['numerics_after_restore']=read_numerical_flags(torch)
                report['numerics_restored']=report['numerics_after_restore']==previous
                require(report['numerics_restored'], 'Numerical policy restoration mismatch')
            except BaseException as exc:  # noqa: BLE001 - re-raised after preserving failure receipt
                restoration_error=exc
                report.update(status='failed',restoration_error=str(exc),numerics_restored=False)
        report['elapsed_seconds']=time.monotonic()-beginning
        write(output/'execution_receipt.json',report,replace=(output/'execution_receipt.json').exists())
        if restoration_error is not None: raise restoration_error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare',type=Path);choice.add_argument('--execute',type=Path);choice.add_argument('--print-spec',action='store_true')
    parser.add_argument('--expected-plan-sha256');args=parser.parse_args()
    if args.print_spec: print(json.dumps(SPEC,indent=2));return
    require(os.environ.get('PYTHONDONTWRITEBYTECODE')=='1','Disable bytecode generation')
    if args.prepare: prepare(args.prepare)
    else:
        require(args.expected_plan_sha256 is not None,'Expected plan SHA required')
        execute(args.execute,args.expected_plan_sha256)


if __name__=='__main__': main()
