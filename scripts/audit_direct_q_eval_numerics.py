"""One-camera no-target diagnosis of historical E convolution TF32 policy."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

SPEC={'protocol':'direct_q_eval_numerics_diagnostic_v1','name':'001.png',
      'calls':['native_tf32_false','native_tf32_true','original_q_tf32_true'],
      'scene_calls':3,'direct_q_shader_calls':1,'teacher_calls':0,'target_reads':0,
      'internal_seconds':120,'external_seconds':180,'training_updates':0,
      'claim':'target-free historical numerical-policy compatibility, no performance or adoption'}


def execute(plan_path):
    plan=json.loads(Path(plan_path).read_text());output=Path(plan['output']);snapshot=Path(plan['snapshot'])
    sys.path.insert(0,str(snapshot))
    module_spec=importlib.util.spec_from_file_location('bound_head_worker',snapshot/'train_direct_q_head_matched.py')
    worker=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(worker)
    worker.require(plan['specification']==SPEC and worker.files(snapshot)==plan['sources'],'Frozen diagnosis changed')
    worker.require(not (output/'execution_started.json').exists(),'One attempt only')
    worker.write(output/'execution_started.json',{'plan_sha256':worker.sha(plan_path)})
    for path,digest in plan['inputs'].items():worker.require(worker.sha(path)==digest,'Bound input changed')
    execution={'status':'running','plan_sha256':worker.sha(plan_path),'specification':SPEC,'records':[],
               'scene_calls':0,'direct_q_shader_calls':0,'target_reads':0,'training_updates':0}
    start=time.monotonic();old_flags=None;torch=None
    def timeout(*_):raise TimeoutError('One-camera diagnosis expired')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:
        execution['gpu_before']=worker.gpu_inventory()
        local_plan=dict(plan['training_plan'],source_snapshot=str(snapshot),source_hashes=plan['sources'])
        modules,imports=worker.imports_from_snapshot(local_plan);execution['actual_imports']=imports
        torch=modules['bridge_rgs.train'].torch;torch.set_num_threads(4)
        old_flags=worker.numerical_flags(torch)
        import cv2

        from bridge_rgs.direct_q_render import render_direct_q
        official=modules['bridge_rgs.official_evaluate'];helper=modules['evaluate_projective_deck_pooling']
        scene,_state,_=worker.load_base(local_plan);scene.eval().requires_grad_(False)
        hashes_before=(worker.digest_tensors(scene.state_dict()),worker.digest_tensors(scene.state_dict(),True))
        q=worker.load_q(scene,'original_q',local_plan);qsha=worker.tensor_sha(q)
        camera=plan['view']['camera'];K,w,h,back=official.distortion_render_grid(camera['K'],camera['distortion'],camera['width'],camera['height'],'legacy_mixed_v1')
        kt=torch.tensor(K,dtype=torch.float32,device='cuda');pt=torch.tensor(camera['w2c'],dtype=torch.float32,device='cuda')
        teacher=np.load(plan['teacher']['path'],allow_pickle=False)
        expected=cv2.imread(plan['E_mask']['path'],cv2.IMREAD_UNCHANGED)
        worker.require(teacher.dtype==np.float32 and teacher.shape==(h,w,5),'Teacher grid')
        held={}
        with torch.no_grad():
            for label in SPEC['calls']:
                flags=dict(worker.NUMERICS,cudnn_allow_tf32=(label!='native_tf32_false'))
                worker.numerical_flags(torch,flags);actual=worker.read_numerical_flags(torch)
                worker.require(actual==flags,'Numerical flags not applied')
                if label=='original_q_tf32_true':
                    result=render_direct_q(scene,q,kt,pt,w,h);execution['direct_q_shader_calls']+=1
                else:
                    result=scene.render(kt,pt,w,h,degree=3,absgrad=False,geometry_grad=False,refinement_grad_to_field=False)
                execution['scene_calls']+=1
                soft=result['probabilities'].cpu().numpy();p3d=result['p3d'].cpu().numpy()
                canvas=result['rgb'].clamp(0,1).cpu().numpy()
                rgb=canvas if back is None else cv2.remap(canvas,back[...,0],back[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                canvas_record=helper.save_png(output/label/'canvas_rgb.png',(canvas[...,::-1]*255).round().astype(np.uint8),plan['canvas_rgb'])
                rgb_record=helper.save_png(output/label/'h3_rgb.png',(rgb[...,::-1]*255).round().astype(np.uint8),plan['h3_rgb'])
                joint=helper.warp_probabilities((soft+teacher)*np.float32(.5),back).argmax(-1).astype(np.uint8)
                mask=helper.save_png(output/label/'joint_mask.png',joint)
                softrecord=helper.save_soft(output/label/'scene_soft.npy',soft)
                row={'label':label,'actual_numerics':actual,'joint_mask':mask,'soft':softrecord,
                     'canvas_rgb':canvas_record,'h3_rgb':rgb_record,
                     'E_mask_pixel_differences':int(np.count_nonzero(joint!=expected)),
                     'E_mask_bytes_exact':bool(Path(mask['path']).read_bytes()==Path(plan['E_mask']['path']).read_bytes()),
                     'p3d_sha256':worker.tensor_sha(result['p3d'])}
                if label=='native_tf32_true':
                    row['false_vs_true_soft_absmax']=float(np.max(np.abs(held['native_tf32_false'][0].astype(np.float64)-soft)))
                    row['false_vs_true_p3d_exact']=bool(np.array_equal(held['native_tf32_false'][1],p3d))
                if label=='original_q_tf32_true':
                    row['native_true_soft_exact']=bool(np.array_equal(held['native_tf32_true'][0],soft))
                    row['native_true_p3d_exact']=bool(np.array_equal(held['native_tf32_true'][1],p3d))
                held[label]=(soft,p3d);execution['records'].append(row)
        execution['state_exact']=(worker.digest_tensors(scene.state_dict()),worker.digest_tensors(scene.state_dict(),True))==hashes_before
        execution['q_exact']=worker.tensor_sha(q)==qsha
        execution['hypothesis_confirmed']=bool(execution['records'][0]['E_mask_pixel_differences']>0
            and execution['records'][1]['E_mask_bytes_exact'] and execution['records'][2]['E_mask_bytes_exact']
            and execution['records'][2]['native_true_soft_exact'] and execution['records'][2]['native_true_p3d_exact'])
        worker.require(execution['state_exact'] and execution['q_exact'],'Readout mutated state')
        for path,digest in plan['inputs'].items():worker.require(worker.sha(path)==digest,'Input changed')
        execution['status']='completed';execution['inputs_unchanged']=True
    except BaseException as exc:
        execution.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        signal.alarm(0)
        if old_flags is not None:
            worker.numerical_flags(torch,old_flags)
            execution['numerics_restored']=worker.read_numerical_flags(torch)==old_flags
        execution['elapsed_seconds']=time.monotonic()-start
        worker.write(output/'execution_receipt.json',execution)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('plan',type=Path);args=parser.parse_args()
    if os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise ValueError('Disable bytecode')
    execute(args.plan)
