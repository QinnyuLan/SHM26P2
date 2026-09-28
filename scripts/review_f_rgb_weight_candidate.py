"""CPU-only audit for the fixed weighted F RGB candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

ROOT=Path('/home/sky/workspace/SHM2026'); RUNS=Path('/mnt/data/SHM2026/runs')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
def require(ok,msg):
    if not ok: raise ValueError(msg)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); a=ap.parse_args(); r=a.run.resolve()
    rec=json.loads((r/'execution_receipt.json').read_text()); pred=json.loads((r/'prediction_receipt.json').read_text()); met=json.loads((r/'official_rgb_metrics.json').read_text()); fmet=json.loads((RUNS/'ibgs_joint_replacement_v2/official_metrics.json').read_text()); man=json.loads((ROOT/'artifacts/prepared/manifest.json').read_text()); src={v['name']:v for v in man['views'] if v['split']=='val'}
    names=sorted(v['name'] for v in fmet['views']); require(rec['status']=='completed' and pred['status']=='all_50_predictions_before_gt' and pred['annotation_reads']==0,'Barrier receipt failed'); require(sorted(x['name'] for x in pred['records'])==names,'Prediction population differs')
    rows=[]
    for n in names:
        p=r/'rgb'/n; require(sha(p)==next(x['sha256'] for x in pred['records'] if x['name']==n),'Prediction hash differs'); tpath=Path(src[n]['source_image_path']); pimg=cv2.cvtColor(cv2.imread(str(p),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB); timg=cv2.cvtColor(cv2.imread(str(tpath),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB); x=pimg.astype(np.float32)/255; y=timg.astype(np.float32)/255; mse=np.mean((x.astype(np.float64)-y.astype(np.float64))**2); _,s=structural_similarity(y,x,data_range=1.,channel_axis=-1,win_size=11,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True); rows.append({'name':n,'psnr':float(-10*np.log10(max(mse,1e-12))),'ssim':float(s[5:-5,5:-5].mean())})
    for k in ('psnr','ssim'):
        got=float(np.mean([x[k] for x in rows])); require(abs(got-met[k])<1e-6,f'{k} mismatch')
    result={'status':'passed','no_gpu':True,'prediction_count':50,'target_rgb_reads':50,'prediction_hashes_bound':True,'psnr_ssim_exact':True,'lpips':'producer-only; not recomputed on CPU'}
    (r/'independent_cpu_review.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
