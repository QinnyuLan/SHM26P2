"""Capture the exact failed AA covariance inputs; no raster/backward/optimization."""
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

TRAIN = Path('/mnt/data/SHM2026/runs/ibgs_aa_warm_matched_v1')
CHECKPOINT_SHA = 'abbe8973ebc7649e99b162f9d878129232fe5475ffe8672f2e4fc9b48c5b5d8f'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    def safe(v):
        if isinstance(v, float) and not math.isfinite(v):
            return str(v)
        if isinstance(v, dict):
            return {k: safe(x) for k, x in v.items()}
        if isinstance(v, list):
            return [safe(x) for x in v]
        return v
    payload = json.dumps(safe(value), indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(payload)


def reference(means, scales, q, w2c, width, height, tx, ty):
    """FP64 same activated FP32 values, with no quaternion renormalization."""
    w, x, y, z = q.T
    R = np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                  2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                  2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)
    cam = means@w2c[:3, :3].T+w2c[:3, 3]
    fx, fy = width/(2*tx), height/(2*ty)
    zz = cam[:, 2]
    j = np.zeros((len(means), 2, 3), np.float64)
    j[:, 0, 0] = fx/zz; j[:, 1, 1] = fy/zz
    j[:, 0, 2] = -fx*np.clip(cam[:, 0]/zz, -1.3*tx, 1.3*tx)/zz
    j[:, 1, 2] = -fy*np.clip(cam[:, 1]/zz, -1.3*ty, 1.3*ty)/zz
    factor = j@w2c[:3, :3]@(R*scales[:, None, :])
    covariance = factor@factor.transpose(0, 2, 1)
    det_direct = covariance[:, 0, 0]*covariance[:, 1, 1]-covariance[:, 0, 1]*covariance[:, 1, 0]
    det_factor = sum((factor[:, 0, a]*factor[:, 1, b]-factor[:, 0, b]*factor[:, 1, a])**2
                     for a, b in ((0, 1), (0, 2), (1, 2)))
    blurred_factor = det_factor+.3*(factor*factor).sum((1, 2))+.09
    blurred_direct = ((covariance[:, 0, 0]+.3)*(covariance[:, 1, 1]+.3)
                      -covariance[:, 0, 1]*covariance[:, 1, 0])
    return {'camera_means': cam, 'factor64': factor, 'covariance64': covariance,
            'det_direct64': det_direct, 'det_factor64': det_factor,
            'det_blur_direct64': blurred_direct, 'det_blur_factor64': blurred_factor,
            'rho_factor64': np.sqrt(det_factor/blurred_factor)}


def run(output):
    require = lambda c, m: None if c else (_ for _ in ()).throw(ValueError(m))
    require(not output.exists() and output.is_relative_to('/mnt/data'), 'Fresh data directory required')
    output.mkdir(parents=True)
    plan = json.loads((TRAIN/'plan.json').read_text())
    train = json.loads((TRAIN/'full/training_receipt.json').read_text())
    require(train['status'] == 'failed' and train['steps'] == 61, 'Exact failure checkpoint required')
    path = Path(train['failure_checkpoint']['path'])
    require(sha(path) == CHECKPOINT_SHA, 'Failure checkpoint changed')
    sys.path.insert(0, plan['source_snapshot'])
    import torch

    from bridge_rgs import ibgs_antialias as aa
    from bridge_rgs.ibgs_camera_contract import make_ibgs_camera
    require(sha(aa.__file__) == plan['aa_module_sha256'], 'Original failed AA module required')
    order = json.loads(Path(plan['camera_order']).read_text())['names']
    require(order[61] == '004.png', 'Failure camera changed')
    contract = json.loads(Path(plan['data_contract']).read_text())
    row = next(r for r in contract['train_rows'] if r['name'] == '004.png')
    camera = make_ibgs_camera(row['K'], row['w2c_original'], row['width'], row['height'])
    record = {'status': 'running', 'training_plan_sha256': sha(TRAIN/'plan.json'),
              'failure_checkpoint_sha256': CHECKPOINT_SHA, 'aa_module_sha256': sha(aa.__file__),
              'worker_sha256': sha(__file__), 'camera': row['name'], 'step': 62,
              'projection_calls': 0, 'raster_calls': 0, 'backward': 0, 'optimizer_steps': 0,
              'RGB_or_semantic_decodes': 0, 'input_hashes': {str(p): sha(p) for p in
                  (TRAIN/'plan.json', TRAIN/'execution_receipt.json', TRAIN/'launch_receipt.json',
                   Path(plan['camera_order']), Path(plan['data_contract']))}}
    started = time.monotonic()
    original = aa.determinant_compensation
    old_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                 torch.backends.cudnn.benchmark, torch.get_float32_matmul_precision())
    def deadline(*_):
        raise TimeoutError('Fixed determinant diagnostic 90-second limit')
    old_handler = signal.signal(signal.SIGALRM, deadline); signal.alarm(90)
    captured = {}
    try:
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.benchmark = False
        saved = torch.load(path, map_location='cpu', weights_only=False)
        require(saved['status'] == 'failed' and saved['step'] == 61, 'Not the exact pre-failure state')
        f = saved['field']
        means = f['_xyz'].cuda(); scales = f['_scaling'].cuda().exp()
        q = torch.nn.functional.normalize(f['_rotation'].cuda(), dim=-1)
        opacity = f['_opacity'].cuda().sigmoid()
        w2c = torch.tensor(camera['world_view_transform'], dtype=torch.float32, device='cuda').T.contiguous()
        tx, ty = math.tan(camera['FoVx']*.5), math.tan(camera['FoVy']*.5)
        def capture(covariance, eps2d=.3):
            a, b, c, d = (covariance[:, i, j] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
            det = a*d-b*c; blur = (a+eps2d)*(d+eps2d)-b*c
            bad = ~(torch.isfinite(det) & torch.isfinite(blur) & (blur > 0))
            idx = bad.nonzero().flatten()
            require(0 < len(idx) <= 10000, 'No reproduced bad slots, or fixed capture capacity exceeded')
            captured.update(indices=idx.cpu().numpy(), covariance32=covariance[idx].cpu().numpy(),
                            det_original32=det[idx].cpu().numpy(), det_blur32=blur[idx].cpu().numpy(),
                            means32=means[idx].cpu().numpy(), scales32=scales[idx].cpu().numpy(),
                            q32=q[idx].cpu().numpy(), opacity32=opacity[idx].cpu().numpy(), w2c32=w2c.cpu().numpy())
            record['bad_count'] = len(idx)
            record['total_gaussians'] = len(covariance)
            raise ArithmeticError('Captured expected original determinant failure')
        aa.determinant_compensation = capture
        with torch.no_grad():
            record['projection_calls'] += 1
            try:
                aa.compensate_opacity(means, scales, q, opacity, w2c, width=row['width'], height=row['height'],
                                     tanfovx=tx, tanfovy=ty, near_plane=.01)
            except ArithmeticError as error:
                require(str(error) == 'Captured expected original determinant failure', 'Unexpected arithmetic exception')
                record['original_failure_reproduced'] = True
        require(record.get('original_failure_reproduced'), 'Expected failure not reproduced')
        values = reference(*(captured[k].astype(np.float64) for k in ('means32', 'scales32', 'q32', 'w2c32')),
                           row['width'], row['height'], float(np.float32(tx)), float(np.float32(ty)))
        captured.update(values)
        with (output/'bad_slots.npz').open('xb') as stream:
            np.savez(stream, **captured)
        record.update(status='completed', arrays_sha256=sha(output/'bad_slots.npz'),
                      input_activated_dtype='FP32; same activated values cast to FP64 without re-normalizing quaternion',
                      reference='CPU FP64 factored covariance and direct vs Cauchy-Binet determinants; no numerical floor',
                      slots=[{'index': int(captured['indices'][i]), 'C32': captured['covariance32'][i].tolist(),
                              'det32': float(captured['det_original32'][i]), 'det_blur32': float(captured['det_blur32'][i]),
                              **{k: float(v[i]) for k, v in values.items() if v.ndim == 1},
                              'camera_mean': values['camera_means'][i].tolist(),
                              'activated_scale': captured['scales32'][i].tolist()}
                             for i in range(len(captured['indices']))])
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        aa.determinant_compensation = original
        torch.set_float32_matmul_precision(old_flags[3])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = old_flags[:3]
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
        record['elapsed_seconds'] = time.monotonic()-started
        record['hook_restored'] = aa.determinant_compensation is original
        write(output/'execution_receipt.json', record)


if __name__ == '__main__':
    if os.environ.get('PYTHONDONTWRITEBYTECODE') != '1':
        raise ValueError('Disable bytecode')
    run(Path(sys.argv[1]).resolve())
