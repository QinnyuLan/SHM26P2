"""Two native-resolution TRAIN forward/backward passes of the IBGS port.

Warm-imports the fixed 1M RGB field solely for a realistic capacity/runtime
check. No optimizer, checkpoint selection, VAL, or semantic objective.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/home/sky/workspace/SHM2026')
CHECKPOINT = Path('/mnt/data/SHM2026/runs/rgb_capacity_1m_reference_v1/training/last.pt')
CHECKPOINT_SHA = '35b489fe45ad34ad279018609b6d5eaf6493e4c0d82f9c86ec16925268078092'
CONTRACT = Path('/mnt/data/SHM2026/runs/ibgs_port_preflight_v1/data_contract.json')
CONTRACT_SHA = '599f6cbf09a16f56bef6f37d0e0f8ce0a3f24449376213963ee033fc2ea9d84b'
TARGETS = ['002.png', '041.png']
PORT_BINARY_SHA = '436b2b55df2b37cddda59e606ec0855ad125eb211058b502197674bd2b93adaf'
CHECK_REPORTS = [Path('/mnt/data/SHM2026/runs')/directory/filename for directory, filename in (
    ('ibgs_port_depth_gradient_v1', 'result.json'),
    ('ibgs_port_backend_smoke_v1', 'result.json'),
    ('ibgs_port_source_validity_v1', 'report.json'))]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def prepare(args):
    from diff_plane_rasterization import _C
    require(sha(_C.__file__) == PORT_BINARY_SHA, 'Only verified revision1 backend')
    require(all(json.loads(p.read_text())['status'] == 'passed' for p in CHECK_REPORTS),
            'Synthetic checks must pass first')
    require(sha(CHECKPOINT) == CHECKPOINT_SHA and sha(CONTRACT) == CONTRACT_SHA,
            'Fixed field/data contract changed')
    files = [Path(__file__), *(ROOT/'src/bridge_rgs'/name for name in
               ('ibgs_adapter.py', 'ibgs_losses.py', 'ibgs_camera_contract.py', 'ibgs_data_contract.py'))]
    files += [p for base in ('scene', 'utils', 'gaussian_renderer', 'arguments')
              for p in (args.repository/base).rglob('*.py')]
    files += [args.repository/'color_aggregation_network.py']
    files += [Path('/mnt/data/SHM2026/third_party/ibgs_port_revision1/port_receipt.json'),
              Path('/mnt/data/SHM2026/third_party/ibgs_build_validation_v1/requirements_frozen.txt'),
              *CHECK_REPORTS]
    plan = {'protocol': 'ibgs_port_preflight_v1', 'targets': TARGETS,
            'scope': '1M warm-import capacity/finite-gradient test; no trained IBGS result',
            'repository': str(args.repository.resolve()),
            'contract': str(CONTRACT), 'contract_sha256': CONTRACT_SHA,
            'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': CHECKPOINT_SHA,
            'source_hashes': {str(p.resolve()): sha(p) for p in files},
            'binary': {'path': _C.__file__, 'sha256': sha(_C.__file__)},
            'internal_seconds': 120, 'external_seconds': 180,
            'optimizer_steps': 0, 'VAL_reads': 0, 'semantic_label_decodes': 0,
            'coordinate_profile': 'corner-v2 corrected shared centered native K',
            'sources': 'fixed first <=4 existing original-default geometric neighbors'}
    with args.plan.open('x') as stream:
        stream.write(json.dumps(plan, indent=2)+'\n')
    print(json.dumps({'plan': str(args.plan), 'sha256': sha(args.plan)}))


def execute(plan, report):
    import torch
    from diff_plane_rasterization import _C

    sys.path.insert(0, plan['repository'])
    sys.path.insert(0, str(ROOT/'src'))
    from color_aggregation_network import ColorFusionResidualNet, fuse_color
    from gaussian_renderer import render, render_depth
    from scene.gaussian_model import GaussianModel

    from bridge_rgs.ibgs_adapter import TrainScene
    from bridge_rgs.ibgs_losses import masked_normal_loss, masked_photometric_loss

    require(sha(_C.__file__) == plan['binary']['sha256'], 'Backend changed after prepare')
    require(all(sha(p) == h for p, h in plan['source_hashes'].items()), 'Source changed after prepare')
    require(sha(CHECKPOINT) == CHECKPOINT_SHA and sha(CONTRACT) == CONTRACT_SHA,
            'Bound input changed')
    torch.set_num_threads(4)
    torch.manual_seed(20260927)
    torch.cuda.manual_seed_all(20260927)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    contract = json.loads(CONTRACT.read_text())
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    require(checkpoint['pixel_protocol']['id'] == 'colmap_corner_v2', 'Wrong field coordinates')
    state = checkpoint['model']
    field = GaussianModel(checkpoint['sh_degree'])
    mapping = {'_xyz': 'means', '_rotation': 'quats', '_scaling': 'log_scales',
               '_opacity': 'opacity_logits', '_features_dc': 'sh0', '_features_rest': 'sh_rest'}
    original = {}
    for destination, key in mapping.items():
        value = state['splats.'+key].detach().cuda().clone()
        if destination == '_opacity':
            value = value[:, None]
        setattr(field, destination, torch.nn.Parameter(value))
        original[destination] = value.detach().clone()
    field.active_sh_degree = checkpoint['sh_degree']
    field.spatial_lr_scale = checkpoint['scene_scale']
    field._normal = torch.nn.Parameter(field.get_smallest_axis().detach().clone())
    field._offset = torch.nn.Parameter(torch.zeros(len(field._xyz), 1, device='cuda'))
    original.update(_normal=field._normal.detach().clone(), _offset=field._offset.detach().clone())
    background = state['background_logits'].sigmoid().cuda()
    del state, checkpoint
    neighbors = {v['name']: [n['name'] for n in v['neighbors_4']]
                 for v in contract['neighbors']['views']}
    scene = TrainScene(contract['train_rows'], contract['pixel_hashes'], neighbors,
                       field, contract['manifest_scene_radius'])
    cameras = {c.image_name: c for c in scene.getTrainCameras()}
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    options = SimpleNamespace(shuffle_source_frame=False, enable_exposure_correction=False,
                              nb_visible_src_frames=3, residual_resolution_scale=1.)
    net = ColorFusionResidualNet(height=cameras[TARGETS[0]].image_height,
                                width=cameras[TARGETS[0]].image_width).cuda()
    report.update(gaussian_count=len(field._xyz), sh_degree=field.active_sh_degree,
                  new_parameters={'normals': field._normal.numel(), 'offsets': field._offset.numel(),
                                  'fusion': sum(p.numel() for p in net.parameters())},
                  depth_forwards=0, target_forwards=0, backwards=0, target_rows=[],
                  raw_rgb_stage='warm-import into different rasterizer; numerical equality not assumed')
    source_ids = sorted({index for name in TARGETS for index in cameras[name].nearest_id})
    with torch.no_grad():
        for index in source_ids:
            depth = render_depth(scene.cameras[index], field, scene, pipe, options, background,
                                 True, 4, 4, .01)
            scene.rendered_depth_list[index] = depth
            report['depth_forwards'] += 1
    for name in TARGETS:
        camera = cameras[name]
        torch.cuda.synchronize()
        started = time.monotonic()
        package = render(camera, field, scene, pipe, options, background,
                         learnt_normal=True, nb_src_frames=4, buffer_length=4,
                         depth_error_threshold=.01, do_find_closest_frame=False,
                         do_render_src_depth=False, render_geo=True, return_depth_normal=True)
        report['target_forwards'] += 1
        fusion = fuse_color(package, net, None, None, None, 0, options)
        require(fusion is not None, 'No real TRAIN fusion support: '+name)
        # This TRAIN target is decoded only after its prediction has been formed.
        target = scene.original_image_list[camera.uid].cuda()
        rgb, _ = masked_photometric_loss(package['render'], target, scene.valid)
        fused, _ = masked_photometric_loss(fusion['image_pred'], target, scene.valid)
        normal = masked_normal_loss(package['rendered_normal'],
                                    package['median_intersected_depth_normal'],
                                    package['median_intersected_depth'][0], scene.valid)
        photo = rgb*0
        warped = package['warped_image'].reshape(-1, 3, camera.image_height, camera.image_width)[:3]
        features = package['cam_feat'].reshape(-1, 4, camera.image_height, camera.image_width)[:3]
        active_sources = 0
        from bridge_rgs.ibgs_losses import erode_valid
        for source_rgb, source_features in zip(warped, features, strict=True):
            valid = (source_features.sum(0) > 0) & scene.valid
            if erode_valid(valid, 5).any():
                photo += masked_photometric_loss(source_rgb, target, valid, ssim_weight=1.)[0]
                active_sources += 1
        photo = photo/max(active_sources, 1)
        loss = .5*(rgb+fused)+.03*normal+.3*photo
        require(torch.isfinite(loss), 'Nonfinite real-scene loss')
        loss.backward()
        report['backwards'] += 1
        gradients = {}
        for parameter_name in original:
            parameter = getattr(field, parameter_name)
            finite = parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            gradients[parameter_name] = {'finite': finite,
                'l2': parameter.grad.norm().item() if finite else None}
        net_finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        torch.cuda.synchronize()
        row = {'name': name, 'source_names': camera.nearest_names,
               'seconds': time.monotonic()-started, 'loss': loss.item(),
               'raw_rgb_loss': rgb.item(), 'fused_loss': fused.item(),
               'normal_loss': normal.item(), 'photometric_loss': photo.item(),
               'fusion_support_fraction': fusion['valid_warp_mask'].mean().item(),
               'gradient_summary': gradients, 'fusion_gradients_finite': bool(net_finite)}
        report['target_rows'].append(row)
        print(json.dumps(row), flush=True)
        require(all(v['finite'] for v in gradients.values()) and net_finite,
                'Nonfinite real-scene gradient: '+name)
        for key in original:
            getattr(field, key).grad = None
        net.zero_grad(set_to_none=True)
        del package, fusion, target, loss, rgb, fused, normal, photo, warped, features
    require(all(torch.equal(getattr(field, key).detach(), value) for key, value in original.items()),
            'Preflight mutated field parameters')
    require(sha(CHECKPOINT) == CHECKPOINT_SHA, 'Preflight changed source checkpoint')
    require(sha(CONTRACT) == CONTRACT_SHA and sha(_C.__file__) == plan['binary']['sha256']
            and all(sha(p) == h for p, h in plan['source_hashes'].items()),
            'Bound code/runtime changed during preflight')
    report.update(field_parameters_unchanged=True, optimizer_steps=0,
                  checkpoint_loads=1, VAL_reads=0, semantic_label_decodes=0,
                  rgb_decodes=scene.original_image_list.decode_count,
                  rgb_names=sorted(scene.original_image_list.decoded_names), valid_decodes=1,
                  source_depth_updates=scene.rendered_depth_list.updates,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args)
        return
    require(args.output is not None and not args.output.exists(), 'Fresh execution receipt required')
    require(args.output.resolve().is_relative_to('/mnt/data'), 'Use data disk')
    plan = json.loads(args.plan.read_text())
    require(plan['targets'] == TARGETS and plan['contract_sha256'] == CONTRACT_SHA
            and plan['checkpoint_sha256'] == CHECKPOINT_SHA, 'Wrong preflight plan')
    def timeout_handler(*_):
        raise TimeoutError('Real-scene preflight exceeded 120 seconds')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)
    start = time.monotonic()
    report = {'status': 'failed', 'plan_sha256': sha(args.plan)}
    try:
        execute(plan, report)
        report['status'] = 'passed'
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-start
        with args.output.open('x') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
        signal.alarm(0)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
