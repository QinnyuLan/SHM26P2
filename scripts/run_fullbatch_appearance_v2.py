"""H3-specific conditional full-TRAIN appearance solver, preparation only.

The legacy runner and failed diagnostic stay intact. This separate entrypoint
requires both the repaired historical diagnostic and the new H3/legacy test.
Solver arithmetic and budgets are identical to the original prepared runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
FIXED_SSIM_RECEIPT_SHA = '15ef4c0f2541f4ada247adb304d5e5e77a3a43af72af84326a26848df921eb89'
FIXED_SSIM_PLAN_SHA = '36e137d371ffc3e9eb4d69c560e9b898475cadd6172b88ffed2d1eb8db213394'
LOSS_SHA = '1e114069c2fdb181e243d7cb8a986659b2e229d2ec49c34821cb45013d9cf2b2'
HELPER_SHA = '10c00b1728e7d1e3773219f18b961065996d8c3a31824700333d8987bfefa746'
H3_VIEW_NAMES = ('002.png', '205.png')  # Sorted TRAIN indices 0 and 175; 175.png is VAL.
H3_FD_EPSILONS = (1e-3, 5e-4, 2.5e-4)
H3_KEYS = ('splats.sh0', 'splats.sh_rest', 'background_logits')
# Canonical JSON of the reviewed precheck SPEC, including its fixed numerical gates.
H3_SPEC_SHA = 'a056a6e345ad85b62927ffc4525b8df34cf528993ac8e5546b905df9eafaf8c0'
MANIFEST_SHA = '551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa'
EXTERNAL_TIMEOUT_SECONDS = 600
SPEC = {'protocol': 'conditional_fullbatch_appearance_h3_v2', 'train_views': 350,
        'pixel_protocol': 'legacy_mixed_v1', 'sh_degree': 3,
        'beta1': 0., 'beta2': .999, 'eps': 1e-8,
        'rates': {'splats.sh0': .00025, 'splats.sh_rest': .0000125, 'background_logits': .0001},
        'alphas': [1., .5, .25], 'armijo_c1': .0001,
        'decrease_floor': 'max(1e-7,1e-5*abs(baseline))',
        'max_accepted_steps': 20, 'max_complete_passes': 40, 'optimization_seconds': 360.,
        'objective': '350-view equal mean: .8 full original RGB L1 + .2 complete 7-window SSIM loss',
        'prediction': 'legacy official pinhole/overscan -> clamp[0,1] -> fixed original-grid float gather',
        'deadline': 'check before/after every view; discard partial gradient/trial; no partial acceptance',
        'cpu_threads': 8, 'semantics_or_VAL_pixels_read': 0,
        'checkpoint': 'one standalone full inference checkpoint; no optimizer/RNG; ordinary resume forbidden'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data, *, replace=False):
    payload = json.dumps(data, indent=2, allow_nan=False)+'\n'
    if replace:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(payload)
        temporary.replace(path)
    else:
        with path.open('x') as stream:
            stream.write(payload)


def bound(item):
    require(sha(item['path']) == item['sha256'], f"Changed bound input: {item['path']}")


def diagnostic_gate(plan):
    for key in ('fixed_ssim_diagnostic_receipt', 'h3_diagnostic_plan',
                'h3_diagnostic_receipt', 'diagnostic_review'):
        bound(plan[key])
    fixed = read(plan['fixed_ssim_diagnostic_receipt']['path'])
    require(plan['fixed_ssim_diagnostic_receipt']['sha256'] == FIXED_SSIM_RECEIPT_SHA
            and fixed['status'] == 'completed' and fixed['plan_sha256'] == FIXED_SSIM_PLAN_SHA
            and fixed['render_calls'] == 40 and fixed['all_model_tensors_finally_restored_exact'] is True
            and fixed['all_bound_inputs_and_sources_unchanged'] is True
            and fixed['actual_imports']['losses']['sha256'] == LOSS_SHA,
            'Require the actual repaired historical SSIM evidence; never relabel its failed predecessor')
    h3_plan = read(plan['h3_diagnostic_plan']['path'])
    h3 = read(plan['h3_diagnostic_receipt']['path'])
    validate_h3_diagnostic(h3, h3_plan, plan['h3_diagnostic_plan']['sha256'])
    verify_diagnostic_bytes(h3_plan)
    review = read(plan['diagnostic_review']['path'])
    require(review.get('decision') == 'allow_fullbatch_appearance_h3_v2'
            and review.get('fixed_ssim_diagnostic_receipt_sha256') == FIXED_SSIM_RECEIPT_SHA
            and review.get('h3_diagnostic_receipt_sha256') == plan['h3_diagnostic_receipt']['sha256']
            and review.get('h3_fd_warp_rms_and_restoration_review_passed') is True
            and isinstance(review.get('reason'), str) and bool(review['reason'].strip()),
            'Explicit review of the new H3 FD/warp/RMS/restoration evidence is required')
    # This predeclared investment gate does not make a universal CUDA verdict.


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def close_number(actual, expected):
    return finite_number(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)


def validate_h3_diagnostic(receipt, diagnostic_plan, plan_sha):
    """Consume the producer's real nested schema and independently recompute gates."""
    specification = json.dumps(diagnostic_plan.get('specification'), sort_keys=True, separators=(',', ':'))
    require(hashlib.sha256(specification.encode()).hexdigest() == H3_SPEC_SHA,
            'H3 diagnostic numerical specification changed')
    require(receipt.get('status') == 'completed' and receipt.get('plan_sha256') == plan_sha,
            'H3 diagnostic is incomplete or bound to a different plan')
    require(diagnostic_plan.get('base_sha256') == receipt.get('base_sha256') == BASE_SHA
            and isinstance(diagnostic_plan.get('base'), str)
            and receipt.get('profile') == 'legacy_mixed_v1', 'H3 base/profile differs')
    flags = ('all_model_tensors_finally_restored_exact', 'all_flags_restored', 'all_modes_restored',
             'training_cameras_unchanged', 'inputs_after_exact', 'source_after_exact',
             'all_bound_inputs_and_sources_unchanged', 'technical_gate_passed')
    require(all(receipt.get(key) is True for key in flags)
            and receipt.get('render_calls') == 40 and receipt.get('backward_calls') == 2
            and receipt.get('checkpoint_written') is False
            and receipt.get('committed_optimizer_steps') == 0
            and receipt.get('semantic_label_pixels_decoded') == 0
            and receipt.get('val_pixels_decoded') == 0, 'H3 execution/restoration/read-scope contract failed')
    require(bool(receipt.get('tensor_hashes_before'))
            and receipt['tensor_hashes_before'] == receipt.get('tensor_hashes_after')
            and isinstance(receipt.get('training_cameras_sha256'), str)
            and len(receipt['training_cameras_sha256']) == 64, 'H3 tensor/camera restoration evidence missing')
    sources, inputs = diagnostic_plan['source_hashes'], diagnostic_plan['input_hashes']
    require(receipt.get('source_hashes') == sources and receipt.get('input_hashes') == inputs
            and sources.get('bridge_rgs/losses.py') == LOSS_SHA
            and sources.get('bridge_rgs/fullbatch_appearance.py') == HELPER_SHA
            and inputs.get(diagnostic_plan['base']) == BASE_SHA
            and inputs.get(diagnostic_plan['manifest']) == MANIFEST_SHA,
            'H3 source/input declarations differ')
    imports, snapshot = receipt.get('actual_imports', {}), Path(diagnostic_plan['source_snapshot']).resolve()
    for name in ('train', 'model', 'raw_grid', 'losses', 'fullbatch_appearance', 'evaluate', 'coordinates'):
        item = imports.get('bridge_rgs.'+name, {})
        relative = 'bridge_rgs/'+name+'.py'
        require(item.get('path') == str(snapshot/relative) and item.get('sha256') == sources.get(relative)
                and bool(item.get('sha256')), 'H3 actual imported numerical source differs')
    allowed = diagnostic_plan['allowed_pixel_paths']
    require(len(allowed) == len(set(allowed)) == 2
            and receipt.get('pixel_reads') == dict.fromkeys(allowed, 1), 'H3 actual TRAIN pixel reads differ')
    views = receipt.get('views', [])
    require([row.get('name') for row in views] == [row.get('name') for row in diagnostic_plan['views']]
            == list(H3_VIEW_NAMES), 'H3 requires sorted TRAIN indices 0/175 (002/205), not VAL 175 or chosen views')
    for row in views:
        require(row.get('render_calls') == 20, 'H3 incomplete per-view render count')
        differences = row.get('finite_differences', {})
        require(set(differences) == set(H3_KEYS), 'H3 must retain all three FD parameter groups')
        for group in differences.values():
            entries = group.get('epsilons', [])
            require(len(entries) == 3 and [x.get('epsilon') for x in entries] == list(H3_FD_EPSILONS)
                    and group.get('parameter_restored_exact') is True,
                    'H3 must retain all fixed FD epsilons and parameter restoration')
            for item in entries:
                numeric = [item.get(key) for key in ('plus_fp32_loss', 'minus_fp32_loss', 'central_difference',
                                                     'realized_analytic', 'realized_relative_discrepancy')]
                require(all(finite_number(value) for value in numeric), 'H3 nonfinite FD measurement')
                plus, minus, central, analytic, error = numeric
                require(plus != minus and analytic != 0
                        and close_number(central, (plus-minus)/(2*item['epsilon']))
                        and close_number(error, abs(central-analytic)/max(abs(analytic), 1e-30))
                        and 0 <= error <= .05 and item.get('passed') is True,
                        'H3 actual-FP32-direction finite differences exceed the fixed 5% gate')
        warp = row.get('baseline', {}).get('warp_cv2_max_abs_difference')
        require(finite_number(warp) and 0 <= warp <= 2e-6,
                'H3 legacy gather differs from official float warp')
        step = row.get('temporary_rms_step', {})
        numeric = [step.get(key) for key in ('before_fp32_loss', 'after_fp32_loss', 'actual_slope',
                                            'actual_decrease', 'predicted_decrease',
                                            'actual_to_predicted_decrease_ratio', 'decrease_floor')]
        require(all(finite_number(value) for value in numeric), 'H3 nonfinite temporary RMS measurement')
        before, after, slope, actual, predicted, ratio, floor = numeric
        require(slope < 0 and predicted > 0 and actual > 0
                and close_number(actual, before-after) and close_number(predicted, -slope)
                and close_number(ratio, actual/predicted) and .9 <= ratio <= 1.1
                and close_number(floor, max(1e-7, 1e-5*abs(before)))
                and actual >= floor and after <= before+1e-4*slope
                and step.get('helper_armijo_passed') is True and step.get('passed') is True
                and step.get('parameters_restored_exact') is True and step.get('committed_steps') == 0,
                'H3 temporary RMS step fails actual/predicted descent, Armijo, floor, or rollback')


def verify_diagnostic_bytes(diagnostic_plan):
    """Recheck existing evidence dependencies; never inspect CUDA or execute a probe."""
    snapshot = Path(diagnostic_plan['source_snapshot'])
    for relative, expected in diagnostic_plan['source_hashes'].items():
        bound({'path': snapshot/relative, 'sha256': expected})
    for filename, expected in diagnostic_plan['input_hashes'].items():
        bound({'path': filename, 'sha256': expected})


def source_hashes(snapshot):
    return {str(path.relative_to(snapshot)): sha(path) for path in sorted(snapshot.rglob('*'))
            if path.is_file() and path.suffix in {'.py', '.md'}}


def execution_contract(root, snapshot, plan_path):
    return {'cwd': str(root), 'environment': {'PYTHONPATH': str(snapshot), 'OMP_NUM_THREADS': '8',
                                             'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'},
            'command': ['timeout', '--signal=TERM', '--kill-after=5s',
                        str(EXTERNAL_TIMEOUT_SECONDS)+'s', 'uv', 'run', '--no-sync', 'python',
                        str(snapshot/Path(__file__).name), '--run', str(plan_path)],
            'external_timeout_seconds': EXTERNAL_TIMEOUT_SECONDS,
            'timeout_scope': 'entire child including source/input hashes, load and save; SIGTERM/SIGKILL may bypass finally'}


def prepare(root, output, diagnostic_plan, diagnostic_receipt, review):
    """CPU-only, fail-closed preparation; only a passing reviewed probe permits output."""
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), 'Never overwrite a previous preparation or execution')
    def binding(path):
        path = Path(path).resolve()
        return {'path': str(path), 'sha256': sha(path)}
    evidence = {
        'fixed_ssim_diagnostic_receipt': binding('/mnt/data/SHM2026/runs/appearance_fixed_ssim_objective_diagnostic/execution_receipt.json'),
        'h3_diagnostic_plan': binding(diagnostic_plan), 'h3_diagnostic_receipt': binding(diagnostic_receipt),
        'diagnostic_review': binding(review)}
    diagnostic_gate(evidence)  # No directory, snapshot, or candidate is created on a failed gate.
    diagnostic = read(evidence['h3_diagnostic_plan']['path'])
    require(Path(diagnostic['root']).resolve() == root, 'Diagnostic and solver root differ')
    base_path, manifest_path = Path(diagnostic['base']), Path(diagnostic['manifest'])
    require(base_path == root/'runs/h3_moments/02_cross/last.pt'
            and manifest_path == root/'artifacts/prepared/manifest.json', 'Wrong fixed H3 dependencies')
    document = root/'docs/fullbatch_appearance_h3_protocol.md'
    require(sha(document) == diagnostic['input_hashes'][str(document)], 'Reviewed protocol document changed')
    for filename in ('uv.lock', 'pyproject.toml'):
        require(sha(root/filename) == diagnostic['input_hashes'][str(root/filename)], 'Diagnostic environment changed')
    manifest = read(manifest_path)
    import torch
    require(not torch.cuda.is_initialized(), 'CPU preparation must not initialize CUDA')
    base = torch.load(base_path, map_location='cpu', weights_only=False)
    require(base.get('pixel_protocol', 'legacy_mixed_v1') == manifest.get('pixel_protocol', 'legacy_mixed_v1')
            == 'legacy_mixed_v1' and base['sh_degree'] == 3, 'Require unchanged legacy H3 profile/SH')
    require(all(base.get(key) is None for key in ('mip_filter_config', 'mip_filter_state',
                                                'mcmc_reference_config', 'mcmc_reference_state')),
            'This H3 preparation requires Mip/MCMC off')
    views = [view for view in manifest['views'] if view['split'] == 'train']
    names = [view['name'] for view in views]
    require(len(names) == len(set(names)) == 350 and names == diagnostic['training_camera_names'],
            'All 350 TRAIN camera names/order must match the diagnostic')
    cameras = base['training_cameras']
    require(cameras.shape == (350, 4, 4) and cameras.dtype == torch.float32
            and bool(torch.isfinite(cameras).all()), 'Wrong H3 TRAIN camera schema')
    camera_sha = hashlib.sha256(cameras.contiguous().numpy().tobytes()).hexdigest()
    require(camera_sha == read(evidence['h3_diagnostic_receipt']['path'])['training_cameras_sha256'],
            'Diagnostic and solver use different TRAIN cameras')
    rgb_paths = [str((root/view['source_image_path']).resolve()) for view in views]
    require(len(set(rgb_paths)) == 350, 'Original TRAIN RGB paths must be unique')
    inputs = {path: sha(path) for path in rgb_paths}  # Hash bytes only; never decode pixels here.
    inputs.update({str(path): sha(path) for path in (base_path, manifest_path, root/'uv.lock', root/'pyproject.toml')})
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    shutil.copytree(Path(diagnostic['source_snapshot'])/'bridge_rgs', snapshot/'bridge_rgs',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    shutil.copy2(document, snapshot/document.name)
    sources = source_hashes(snapshot)
    require({key: value for key, value in sources.items() if key.startswith('bridge_rgs/')}
            == {key: value for key, value in diagnostic['source_hashes'].items() if key.startswith('bridge_rgs/')},
            'Solver package differs from the actual H3-prechecked package')
    inputs[str(snapshot/document.name)] = sha(snapshot/document.name)
    plan = {'status': 'locked_authorized_after_h3_objective_diagnostic', 'specification': SPEC,
            'root': str(root), 'output': str(output/'execution'), 'source_snapshot': str(snapshot),
            'source_hashes': sources, 'base': binding(base_path), 'manifest': binding(manifest_path),
            'manifest_binding_kind': 'observed for this experiment; not a retroactive base declaration',
            'training_camera_names': names, 'training_cameras_sha256': camera_sha,
            'train_view_names_sorted': sorted(names), 'input_hashes': inputs, **evidence,
            'protocol_document': binding(snapshot/document.name),
            'execution': execution_contract(root, snapshot, output/'plan.json')}
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA')
    diagnostic_gate(plan)
    write(output/'plan.json', plan)
    return output/'plan.json'


def verify_plan(path):
    plan = read(path)
    require(plan['status'] == 'locked_authorized_after_h3_objective_diagnostic'
            and plan['specification'] == SPEC, 'No approved locked full-batch experiment plan')
    snapshot = Path(plan['source_snapshot']).resolve()
    require(plan.get('execution') == execution_contract(Path(plan['root']), snapshot, Path(path).resolve()),
            'Fixed outer 600-second execution contract changed')
    require(source_hashes(snapshot) == plan['source_hashes'], 'Snapshot file set changed')
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Use the later reviewed immutable runner')
    for relative, expected in plan['source_hashes'].items():
        bound({'path': snapshot/relative, 'sha256': expected})
    require(plan['source_hashes'].get('bridge_rgs/losses.py') == LOSS_SHA
            and plan['source_hashes'].get('bridge_rgs/fullbatch_appearance.py') == HELPER_SHA,
            'Keep the repaired SSIM implementation and unchanged full-batch numeric helper')
    require(plan['base']['sha256'] == BASE_SHA and plan['manifest']['sha256'] == MANIFEST_SHA,
            'Wrong fixed H3 base/manifest')
    bound(plan['protocol_document'])
    diagnostic = read(plan['h3_diagnostic_plan']['path'])
    require({key: value for key, value in plan['source_hashes'].items() if key.startswith('bridge_rgs/')}
            == {key: value for key, value in diagnostic['source_hashes'].items() if key.startswith('bridge_rgs/')},
            'Solver package differs from actual prechecked package')
    for key in ('base', 'manifest'):
        bound(plan[key])
    for filename, expected in plan['input_hashes'].items():
        bound({'path': filename, 'sha256': expected})
    diagnostic_gate(plan)
    return plan


def fixed_views(manifest, base, root, bindings, training_camera_names):
    import numpy as np
    import torch

    from bridge_rgs.coordinates import LEGACY, pixel_protocol
    require(pixel_protocol(base) == pixel_protocol(manifest) == LEGACY, 'Require the historical legacy profile')
    training = [v for v in manifest['views'] if v['split'] == 'train']
    names = [v['name'] for v in training]
    require(training_camera_names == names, 'TRAIN camera name/index order differs from bound manifest')
    views = sorted(training, key=lambda v: v['name'])
    require(len(views) == 350 and len({v['name'] for v in views}) == 350, 'Require all unique 350 TRAIN views')
    cameras = base['training_cameras']
    require(cameras.shape == (350, 4, 4) and cameras.dtype == torch.float32
            and bool(torch.isfinite(cameras).all()), 'Wrong base TRAIN camera schema')
    by_name = {name: cameras[index].detach().clone() for index, name in enumerate(names)}
    original = torch.tensor([v['w2c_original'] for v in training], dtype=torch.float32)
    mapping = {'names_in_checkpoint_index_order': names,
               'source': 'base.training_cameras indexed by bound original manifest TRAIN order',
               'camera_bytes_sha256': hashlib.sha256(cameras.numpy().tobytes()).hexdigest(),
               'different_from_original_pose_count': int((cameras != original).any(dim=2).any(dim=1).sum()),
               'max_abs_difference_from_original_pose': float((cameras-original).abs().max())}
    for view in views:
        camera = manifest['source_cameras'][str(view['camera_id'])]
        require(np.array_equal(view['K'], camera['K']) and view['width'] == camera['width']
                and view['height'] == camera['height'], 'Prepared/source image grid differs')
        path = str((Path(root)/view['source_image_path']).resolve())
        require(path in bindings, 'Every original TRAIN RGB must have a bound SHA')
    return views, by_name, mapping


def gpu_idle():
    result = subprocess.run(['nvidia-smi', '-q', '-x'], check=True, capture_output=True, text=True, timeout=5)
    document = ET.fromstring(result.stdout)
    require(document.findall('gpu'), 'GPU status missing')
    for gpu in document.findall('gpu'):
        listing = gpu.find('processes')
        require(listing is not None and (listing.text or '').strip() not in {'N/A', 'Not Supported'}, 'GPU status unavailable')
    require(all(p.findtext('type') == 'G' for p in document.findall('.//process_info')), 'GPU compute slot is occupied')


def run(path):
    path = Path(path).resolve()
    plan = verify_plan(path)
    output = Path(plan['output']).resolve()
    require(not output.exists(), 'Never overwrite/retry an existing experiment')
    gpu_idle()
    os.chdir(plan['root'])
    import cv2
    import torch

    from bridge_rgs import fullbatch_appearance as solver
    from bridge_rgs.raw_grid import appearance_rgb_loss, build_raw_grid
    from bridge_rgs.train import load_scene
    snapshot = Path(plan['source_snapshot']).resolve()
    for name, module in list(sys.modules.items()):
        if name == 'bridge_rgs' or name.startswith('bridge_rgs.'):
            source = Path(module.__file__).resolve()
            require(source.is_relative_to(snapshot), 'Non-frozen package import')
            relative = str(source.relative_to(snapshot))
            require(plan['source_hashes'].get(relative) == sha(source), 'Imported source SHA missing/different')
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    output.mkdir(parents=True)
    receipt = {'status': 'running', 'plan_sha256': sha(path), 'renderer_calls': 0,
               'complete_passes': 0, 'partial_renders': 0, 'accepted_steps': 0, 'checkpoint_written': False,
               'source_and_input_policy': 'fixed TRAIN original RGB only; no semantic labels/VAL'}
    write(output/'execution_receipt.json', receipt)
    scene = original = original_flags = None
    imread = cv2.imread
    started = time.monotonic()
    try:
        scene, base = load_scene(plan['base']['path'])
        require(base.get('mip_filter_config') is None and base.get('mcmc_reference_config') is None
                and getattr(scene, 'mip_filter_config', None) is None
                and getattr(scene, 'mcmc_reference_config', None) is None,
                'This exact H3 experiment supports only the historical Mip/MCMC-off field')
        scene.eval()
        original = {key: value.detach().clone() for key, value in scene.state_dict().items()}
        original_flags = {key: p.requires_grad for key, p in scene.named_parameters()}
        for key, parameter in scene.named_parameters():
            parameter.requires_grad_(key in solver.KEYS)
        named = dict(scene.named_parameters())
        parameters = {key: named[key] for key in solver.KEYS}
        manifest = read(plan['manifest']['path'])
        require((Path(plan['root'])/base['config']['manifest']).resolve() == Path(plan['manifest']['path']).resolve(),
                'Observed manifest path must match the original base configuration')
        require(base['sh_degree'] == SPEC['sh_degree'], 'Fixed SH degree differs')
        views, cameras, mapping = fixed_views(manifest, base, plan['root'], plan['input_hashes'],
                                              plan['training_camera_names'])
        receipt['training_camera_mapping'] = mapping
        allowed = {str((Path(plan['root'])/v['source_image_path']).resolve()) for v in views}
        def guarded_read(filename, *args, **kwargs):
            require(str(Path(filename).resolve()) in allowed, 'Pixel read outside original TRAIN RGB')
            return imread(filename, *args, **kwargs)
        cv2.imread = guarded_read
        layouts = {}
        for view in views:
            key = str(view['camera_id'])
            if key not in layouts:
                camera = manifest['source_cameras'][key]
                layout = build_raw_grid(camera['K'], camera['opencv_distortion'], camera['width'], camera['height'],
                                        protocol='legacy_mixed_v1')
                layout.warp.to('cuda')
                layouts[key] = layout
        def objective(view):
            filename = str((Path(plan['root'])/view['source_image_path']).resolve())
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            require(image is not None and image.shape == (view['height'], view['width'], 3), 'Wrong original RGB dimensions')
            target = torch.from_numpy(image[..., ::-1].copy()).to('cuda', torch.float32)/255
            layout = layouts[str(view['camera_id'])]
            canvas = scene.render(torch.tensor(layout.render_K, device='cuda'),
                                  cameras[view['name']].to('cuda'),
                                  layout.render_width, layout.render_height, degree=3,
                                  semantics=False, absgrad=False)['rgb'].clamp(0, 1)
            receipt['renderer_calls'] += 1
            prediction = layout.warp(canvas)
            return appearance_rgb_loss(prediction, target, torch.ones_like(target[..., 0], dtype=torch.bool))[0]
        optimizer = solver.RMSArmijo(parameters)
        budget = solver.PassBudget(time.monotonic)
        last_loss, reason = None, 'accepted_step_limit'
        torch.cuda.reset_peak_memory_stats()
        with (output/'passes.jsonl').open('x') as log:
            def full_pass(backward, role):
                require(budget.passes < budget.max_passes, 'Complete-pass budget exhausted')
                before, calls_before = time.monotonic(), receipt['renderer_calls']
                try:
                    with torch.set_grad_enabled(backward):
                        loss = solver.stream_mean_loss(views, objective, backward=backward,
                                                       check_time=budget.check_time)
                except solver.PassExpired:
                    # Throw away any partial accumulated gradient. A trial's enclosing
                    # transaction restores its candidate before the solver exits.
                    scene.zero_grad(set_to_none=True)
                    count = receipt['renderer_calls']-calls_before
                    receipt['partial_renders'] += count
                    log.write(json.dumps({'role': role, 'complete': False, 'renders': count,
                                          'seconds': time.monotonic()-before, 'loss': None})+'\n')
                    log.flush()
                    return None, False
                on_time = budget.completed()
                receipt['complete_passes'] = budget.passes
                record = {'pass': budget.passes, 'role': role, 'loss': loss, 'complete': True,
                          'renders': receipt['renderer_calls']-calls_before,
                          'seconds': time.monotonic()-before, 'on_time': on_time}
                log.write(json.dumps(record, allow_nan=False)+'\n')
                log.flush()
                return loss, on_time
            while optimizer.accepted_steps < SPEC['max_accepted_steps']:
                if not budget.can_start():
                    reason = 'complete_pass_or_time_budget'; break
                scene.zero_grad(set_to_none=True)
                baseline, on_time = full_pass(True, 'baseline_gradient')
                if not on_time:
                    reason = 'time_budget_gradient_pass_discarded'; break
                if last_loss is not None and baseline > last_loss+solver.decrease_floor(last_loss):
                    raise ValueError('Accepted baseline replay increased beyond the fixed guard')
                require(all(p.grad is None for k, p in named.items() if k not in solver.KEYS), 'Frozen gradient path opened')
                require(all(p.grad is not None for p in parameters.values()), 'Missing appearance gradient')
                gradients = {k: p.grad.detach().clone() for k, p in parameters.items()}
                require(all(torch.isfinite(g).all() for g in gradients.values()), 'Nonfinite full gradient')
                if not any(bool(g.any()) for g in gradients.values()):
                    reason = 'zero_full_gradient'; break
                proposal = optimizer.propose(gradients)
                accepted = False
                for alpha in solver.ALPHAS:
                    if not budget.can_start():
                        reason = 'complete_pass_or_time_budget'; break
                    with optimizer.candidate(proposal, gradients, alpha) as trial:
                        if trial.actual_slope < 0:
                            value, on_time = full_pass(False, f'trial_alpha_{alpha}')
                            if on_time:
                                accepted = trial.accept(baseline, value)
                        else:
                            value, on_time = None, True
                        record = {'accepted_step_before': optimizer.accepted_steps, 'alpha': alpha,
                                  'baseline': baseline, 'candidate': value, 'g_dot_d': proposal.analytic_slope,
                                  'g_dot_actual_delta': trial.actual_slope,
                                  'displacement_absmax': trial.displacement_absmax, 'accepted': accepted,
                                  'completed_passes': budget.passes}
                        log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
                    if accepted:
                        last_loss = value
                        receipt['accepted_steps'] = optimizer.accepted_steps
                        break
                    if not on_time:
                        reason = 'time_budget_trial_pass_discarded'; break
                if not accepted:
                    if reason == 'accepted_step_limit':
                        reason = 'three_fixed_trials_rejected'
                    break
        optimization_seconds = time.monotonic()-budget.started
        require(receipt['renderer_calls'] == budget.passes*SPEC['train_views']+receipt['partial_renders'],
                'Render/pass accounting differs')
        scene.zero_grad(set_to_none=True)
        require(all(torch.equal(value, original[key]) for key, value in scene.state_dict().items()
                    if key not in solver.KEYS), 'Frozen model tensor changed')
        verify_plan(path)
        require(sha(path) == receipt['plan_sha256'], 'Plan changed during solver')
        if optimizer.accepted_steps:
            metadata = {'protocol': SPEC['protocol'], 'base': plan['base'], 'manifest_observed': plan['manifest'],
                        'base_manifest_declared_sha256': base.get('manifest_sha256'),
                        'source_files_sha256': plan['source_hashes'], 'stage_config': SPEC,
                        'training_camera_mapping': mapping,
                        'accepted_steps': optimizer.accepted_steps, 'complete_passes': budget.passes,
                        'last_accepted_train_loss': last_loss, 'stop_reason': reason}
            result = solver.inference_checkpoint(base, scene.state_dict(), metadata)
            final = output/'final.pt'
            solver.save_full_inference(result, final)
            reloaded = torch.load(final, map_location='cpu', weights_only=False)
            require(all(torch.equal(reloaded['model'][k], v) for k, v in result['model'].items())
                    and torch.equal(reloaded['training_cameras'], base['training_cameras']), 'Saved model differs')
            receipt.update(checkpoint_written=True, checkpoint=str(final), checkpoint_sha256=sha(final))
        receipt.update(status='completed', stop_reason=reason, last_accepted_train_loss=last_loss,
                       optimization_seconds=optimization_seconds,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       frozen_tensors_exact=True, base_training_cameras_exact=True,
                       ordinary_resume_allowed=False)
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        cv2.imread = imread
        try:
            if scene is not None and original is not None:
                with torch.no_grad():
                    for key, value in scene.state_dict().items():
                        value.copy_(original[key])
                for key, parameter in scene.named_parameters():
                    if original_flags is not None:
                        parameter.requires_grad_(original_flags[key])
                    parameter.grad = None
                restored = all(torch.equal(v, original[k]) for k, v in scene.state_dict().items())
                receipt['in_memory_base_restored_exact'] = restored
                require(restored, 'Process base restoration failed')
        except BaseException as error:
            receipt.update(status='failed', restoration_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            receipt['elapsed_seconds_including_setup_save'] = time.monotonic()-started
            write(output/'execution_receipt.json', receipt, replace=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--run', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path)
    parser.add_argument('--diagnostic-plan', type=Path)
    parser.add_argument('--diagnostic-receipt', type=Path)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    if args.prepare:
        require(all((args.output, args.diagnostic_plan, args.diagnostic_receipt, args.review)),
                'Preparation requires output, diagnostic plan/receipt, and independent review')
        print(prepare(args.root, args.output, args.diagnostic_plan, args.diagnostic_receipt, args.review))
    elif args.run:
        print(json.dumps(run(args.run)))
    else:
        print(json.dumps({'status': 'code_preparation_only_no_plan_no_GPU', 'specification': SPEC}, indent=2))


if __name__ == '__main__':
    main()
