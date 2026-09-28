"""Prepare a locked flip pair, then explicitly run sequential short GPU preflights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'runs/refiner_flip_preflight'
CONFIGS = ROOT/'configs/generated_refiner_flip'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def verify_sources(provenance):
    for directory, key in [('base_snapshot','base_source_hashes'), ('snapshot','source_hashes')]:
        for name, expected in provenance[key].items():
            assert digest(Path(provenance[directory])/name) == expected


def prepare():
    if CONFIGS.exists():
        raise FileExistsError('Refusing to overwrite fixed flip configurations')
    provenance = json.loads((FOLDER/'source_provenance.json').read_text())
    verify_sources(provenance)
    CONFIGS.mkdir()
    baseline = yaml.safe_load((ROOT/'configs/generated_h3_moments_v2/00_zero.yaml').read_text())
    baseline.update(warmstart='runs/support_split_semantic_coupled/last.pt', steps=3000,
                    warmstart_reset_refiner=False, warmstart_add_depth_moments=False,
                    camera_attribution=False, camera_quality_weighting=False,
                    save_every=0, eval_every=0, log_every=100)
    baseline['refiner'] = dict(baseline['refiner'], depth_moments='off')
    assert 'semantic_lr_schedule' not in baseline
    manifest = json.loads((ROOT/baseline['manifest']).read_text())
    train = [v for v in manifest['views'] if v['split']=='train']
    population = [i for i,v in enumerate(train) if v.get('mask_path')]
    assert len(train)==350 and len(population)==259
    initial = torch.load(ROOT/baseline['warmstart'],map_location='cpu',mmap=True,weights_only=False)
    assert len(initial['model'])==131 and initial['refiner_config'].get('depth_moments','off')=='off'
    initial_keys = list(initial['model'])
    rng = np.random.default_rng(42)
    order=[]
    while len(order)<3000:
        order.extend(rng.permutation(population).tolist())
    order=order[:3000]
    flips=[bool(np.random.default_rng(np.random.SeedSequence([42,step,704291])).random()<.5)
           for step in range(1,3001)]
    assert any(flips[:8]) and not all(flips[:8])
    specs=[]
    for arm,probability in [('00_control',0.),('01_flip',.5)]:
        config=dict(baseline,output=f'runs/refiner_flip/{arm}',refiner_horizontal_flip_probability=probability)
        (CONFIGS/f'{arm}.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        preflight=dict(config,output=str((FOLDER/arm).relative_to(ROOT)),steps=8,log_every=1)
        path=CONFIGS/f'{arm}_preflight.yaml'
        path.write_text(yaml.safe_dump(preflight,sort_keys=False))
        expected=CONFIGS/f'{arm}_expected.json'
        write_json(expected,{'view_indices':order[:8],'view_names':[train[i]['name'] for i in order[:8]],
                             'flips':flips[:8] if probability else [False]*8})
        specs.append({'name':arm,'config':str(path.relative_to(ROOT)),'expected':str(expected.relative_to(ROOT)),
                      'snapshot':provenance['snapshot'],'steps':8})
    for name,snapshot in [('legacy_off_2step',provenance['base_snapshot']),('new_off_2step',provenance['snapshot'])]:
        config=dict(baseline,output=str((FOLDER/name).relative_to(ROOT)),steps=2,log_every=1)
        if name=='new_off_2step':
            config['refiner_horizontal_flip_probability']=0.
        path=CONFIGS/f'{name}.yaml'
        path.write_text(yaml.safe_dump(config,sort_keys=False))
        expected=CONFIGS/f'{name}_expected.json'
        write_json(expected,{'view_indices':order[:2],'view_names':[train[i]['name'] for i in order[:2]],'flips':[False]*2})
        specs.append({'name':name,'config':str(path.relative_to(ROOT)),'expected':str(expected.relative_to(ROOT)),
                      'snapshot':snapshot,'steps':2})
    a,b=[yaml.safe_load((CONFIGS/f'{name}.yaml').read_text()) for name in ('00_control','01_flip')]
    differing=[key for key in set(a)|set(b) if a.get(key)!=b.get(key)]
    assert set(differing)=={'output','refiner_horizontal_flip_probability'}
    worker=ROOT/'scripts/refiner_flip_preflight_worker.py'
    worker_copy=FOLDER/'worker_frozen.py'
    shutil.copy2(worker,worker_copy)
    reference=ROOT/'runs/h3_moments_preflight_v2/00_zero/last.pt'
    checkpoint_budget=int(reference.stat().st_size*1.15)
    refiner_bytes=sum(v.numel()*v.element_size() for k,v in initial['model'].items() if k.startswith('refiner.'))
    compact_each=int(refiner_bytes*3.5)+4*2**20
    evaluation_budget=75*2**20
    projected=4*compact_each+2*(checkpoint_budget+evaluation_budget+20*2**20)
    free=shutil.disk_usage(ROOT).free
    reserve=768*2**20
    storage={'free_bytes':free,'compact_preflight_estimated_each':compact_each,'formal_checkpoint_estimated_each':checkpoint_budget,
             'formal_evaluation_estimated_each':evaluation_budget,'additional_outputs_estimated_total':projected,
             'required_remaining_reserve':reserve,'projected_remaining_bytes':free-projected,
             'sufficient':free-projected>=reserve,'existing_files_deleted':False,
             'preflight_save_policy':'Only audit_state.pt with refiner/Adam/RNG, not a reusable scene checkpoint; full frozen-state checks happen before compact save.'}
    plan={'created_utc':datetime.now(UTC).isoformat(),'kind':'fixed_probability_horizontal_refiner_augmentation_engineering_control',
          'claim':'Standard augmentation, development-adaptive choice, not novelty or a blinded holdout test',
          'formal_started':False,'formal_steps_each':3000,'formal_pair_difference':differing,
          'manifest':baseline['manifest'],'manifest_sha256':digest(ROOT/baseline['manifest']),
          'warmstart':baseline['warmstart'],'warmstart_sha256':digest(ROOT/baseline['warmstart']),
          'warmstart_model_tensor_count':len(initial_keys),'warmstart_model_keys':initial_keys,
          'base_snapshot_tree_sha256':provenance['base_tree_sha256'],'snapshot_tree_sha256':provenance['source_tree_sha256'],
          'source_provenance_sha256':digest(FOLDER/'source_provenance.json'),
          'source_snapshot':provenance['snapshot'],'worker':str(worker_copy),'worker_sha256':digest(worker_copy),
          'orchestrator_sha256':digest(__file__),'train_labeled_population':len(population),
          'train_labeled_names':[train[i]['name'] for i in population],
          'stateless_plan':{'seed':42,'stream_tag':704291,'probability':.5,'flip_flags':flips,
                            'view_indices':order,'view_names':[train[i]['name'] for i in order]},
          'preflight_specs':specs,'storage':storage,
          'config_hashes':{str(p.relative_to(ROOT)):digest(p) for p in CONFIGS.iterdir() if p.suffix in ('.yaml','.json')}}
    write_json(CONFIGS/'flip_plan.json',plan)
    write_json(CONFIGS/'storage_budget.json',storage)
    print(json.dumps({'ready':True,'gpu_started':False,'first8_flips':flips[:8],'storage':storage},indent=2))


def tensors_error(a,b):
    rows={}
    for key in a:
        delta=(a[key].double()-b[key].double()).abs()
        rows[key]={'exact':torch.equal(a[key],b[key]),'max_absolute':float(delta.max()),'mean_absolute':float(delta.mean())}
    return {'all_exact':all(v['exact'] for v in rows.values()),'global_max_absolute':max(v['max_absolute'] for v in rows.values()),'tensors':rows}


def rng_equal(a,b):
    an,bn=a['numpy_rng'],b['numpy_rng']
    return (torch.equal(a['torch_rng'],b['torch_rng']) and torch.equal(a['cuda_rng'],b['cuda_rng'])
            and an[0]==bn[0] and np.array_equal(an[1],bn[1]) and an[2:]==bn[2:])


def run_gpu():
    path=CONFIGS/'flip_plan.json'
    plan=json.loads(path.read_text())
    receipt_path=FOLDER/'gpu_preflight_receipt.json'
    if receipt_path.exists():
        raise FileExistsError('Refusing to repeat completed or interrupted preflight')
    provenance=json.loads((FOLDER/'source_provenance.json').read_text())
    verify_sources(provenance)
    assert digest(__file__)==plan['orchestrator_sha256']
    assert digest(plan['worker'])==plan['worker_sha256']
    assert digest(ROOT/plan['warmstart'])==plan['warmstart_sha256']
    assert digest(ROOT/plan['manifest'])==plan['manifest_sha256']
    for name,expected in plan['config_hashes'].items():
        assert digest(ROOT/name)==expected
    budget=plan['storage']
    current_free=shutil.disk_usage(ROOT).free
    assert current_free>=budget['additional_outputs_estimated_total']+budget['required_remaining_reserve'],'Insufficient disk; do not delete inputs automatically'
    receipt={'status':'running','started_utc':datetime.now(UTC).isoformat(),'plan_sha256':digest(path),
             'source_tree_sha256':plan['snapshot_tree_sha256'],'input_warmstart_sha256':plan['warmstart_sha256'],
             'single_sequential_worker':True,'formal_training_started':False,'completed_arms':[]}
    write_json(receipt_path,receipt)
    try:
        for spec in plan['preflight_specs']:
            env=dict(os.environ,PYTHONPATH=spec['snapshot'],OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1')
            with (FOLDER/(spec['name']+'.log')).open('w') as stream:
                subprocess.run([sys.executable,plan['worker'],'--config',str(ROOT/spec['config']),
                                '--snapshot',spec['snapshot'],'--expected',str(ROOT/spec['expected'])],
                               cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            receipt['completed_arms'].append(spec['name'])
            write_json(receipt_path,receipt)
        states={s['name']:torch.load(FOLDER/s['name']/'audit_state.pt',map_location='cpu',weights_only=False)
                for s in plan['preflight_specs']}
        reports={s['name']:json.loads((FOLDER/s['name']/'worker_audit.json').read_text()) for s in plan['preflight_specs']}
        a,b=states['00_control'],states['01_flip']
        assert a['sampler']==b['sampler'] and rng_equal(a,b)
        old,new=states['legacy_off_2step'],states['new_off_2step']
        assert old['sampler']==new['sampler'] and rng_equal(old,new)
        default_error=tensors_error(new['refiner_model'],old['refiner_model'])
        opt_old,opt_new=old['heads_optimizer']['state'],new['heads_optimizer']['state']
        assert opt_old.keys()==opt_new.keys()
        optimizer_error=tensors_error({f'{i}.{k}':v for i,s in opt_new.items() for k,v in s.items()},
                                     {f'{i}.{k}':v for i,s in opt_old.items() for k,v in s.items()})
        frame_equal=lambda x,y: all(a['field_hashes']==b['field_hashes'] for a,b in zip(reports[x]['renders'],reports[y]['renders'],strict=True))
        audit={'status':'passed','warmstart_all_131_exact_each':True,'fresh_adam_each':True,'only_refiner_changed_each':True,
               'eight_step_pair_sampler_and_global_rng_exact':True,'legacy_pair_sampler_and_global_rng_exact':True,
               'eight_step_pair_field_forward_hashes_exact':frame_equal('00_control','01_flip'),
               'legacy_pair_field_forward_hashes_exact':frame_equal('legacy_off_2step','new_off_2step'),
               'legacy_default_refiner_error':default_error,'legacy_default_optimizer_error':optimizer_error,
               'cuda_numerical_policy':'Report exactness and all parameter errors; do not assume bitwise CUDA equality. Any nonzero default-path drift requires review before formal launch.',
               'requires_review_before_formal':not(default_error['all_exact'] and optimizer_error['all_exact']),
               'worker_reports':{name:{'path':str(FOLDER/name/'worker_audit.json'),'sha256':digest(FOLDER/name/'worker_audit.json'),
                                       'peak_gpu_gb':r['peak_gpu_gb'],'final':r['final']} for name,r in reports.items()},
               'no_formal_training':True}
        verify_sources(provenance)
        write_json(FOLDER/'preflight_audit.json',audit)
        receipt.update(status='completed',audit_sha256=digest(FOLDER/'preflight_audit.json'))
        print(json.dumps({k:audit[k] for k in ('status','requires_review_before_formal','eight_step_pair_sampler_and_global_rng_exact','eight_step_pair_field_forward_hashes_exact')}))
    except Exception as error:
        receipt.update(status='failed',error=str(error))
        raise
    finally:
        receipt['finished_utc']=datetime.now(UTC).isoformat()
        write_json(receipt_path,receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true',help='CPU-only fixed design and disk/source audit')
    group.add_argument('--run-gpu',action='store_true',help='Explicitly run authorized short preflights; never formal training')
    args=parser.parse_args()
    prepare() if args.prepare else run_gpu()
