"""Three-render TRAIN152/native gradient-chain diagnostic; CPU prepare by default.

Reuses the completed forty-render source verbatim. No optimizer, candidate,
checkpoint, semantic pixels or validation pixels are used. --run requires the
new copied entrypoint; GPU execution remains an explicit separately approved step.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import signal
import time
from collections import Counter
from pathlib import Path

import cv2
import torch

REFERENCE = Path('/mnt/data/SHM2026/runs/appearance_actual_objective_diagnostic')
REFERENCE_PLAN_SHA = 'c8db5d457049844b346727ebf83bfc2ac8d65971c0903d7feb33d94cb675b5d3'
PARAMETERS = ('splats.sh0', 'splats.sh_rest', 'background_logits')
COMPONENTS = ('l1', 'dssim7', 'combined')
SPEC = {
    'protocol': 'appearance_gradient_chain_three_render_v1',
    'view': '152.png', 'split': 'train', 'arm': '00_native', 'renders': 3,
    'epsilon': .001, 'direction': 'baseline combined SH0 gradient / absmax; all other directions zero',
    'prediction': 'unchanged reference overscan RGB -> clamp[0,1] -> native crop',
    'losses': ['valid L1', 'complete-valid 7-window 1-SSIM', '.8 L1 + .2 DSSIM7'],
    'leaf_dtypes': ['float32', 'float64'], 'internal_deadline_seconds': 55,
    'external_deadline_seconds': 60, 'output_bytes_limit': 2 << 20,
    'pixel_reads': 'only TRAIN152 native RGB and native valid',
    'limits': 'Jd is a finite secant, not an exact renderer JVP. FP64 losses reuse FP32 render outputs. Nonlinearity, clamp/L1 kinks and finite epsilon remain possible explanations; no automatic CUDA-bug verdict.',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, update=False):
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if update:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(text)
        temporary.replace(path)
    else:
        with path.open('x') as stream:
            stream.write(text)


def dot(a, b):
    return float((a.detach().double() * b.detach().double()).sum())


def norms(value):
    value = value.detach().double()
    return {'absmax': float(value.abs().max()), 'l2': float(value.square().sum().sqrt()),
            'rms': float(value.square().mean().sqrt()), 'finite': bool(torch.isfinite(value).all())}


def discrepancy(a, b):
    return {'signed_a_minus_b': a-b, 'absolute': abs(a-b),
            'relative_to_a': abs(a-b)/max(abs(a), 1e-30)}


def layout_info(value):
    nchw = value.permute(2, 0, 1)[None]
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'stride': list(value.stride()),
            'storage_offset': value.storage_offset(), 'contiguous': value.is_contiguous(),
            'ssim_nchw_stride': list(nchw.stride()), 'ssim_nchw_contiguous': nchw.is_contiguous(),
            'ssim_nchw_channels_last': nchw.is_contiguous(memory_format=torch.channels_last)}


def components(prediction, target, valid):
    """Use unmodified frozen operators; independent component graphs remain live."""
    from bridge_rgs.losses import masked_mean, ssim_map
    from bridge_rgs.raw_grid import appearance_rgb_loss, complete_window_support
    target, pixels = target.detach(), valid.detach().bool()
    windows = complete_window_support(pixels, 7)
    l1 = masked_mean((prediction-target).abs().mean(-1), pixels)
    dssim = masked_mean(1-ssim_map(prediction, target), windows)
    combined, _ = appearance_rgb_loss(prediction, target, pixels)
    return {'l1': l1, 'dssim7': dssim, 'combined': combined}


def graph_gradients(prediction, parameters, target, valid):
    losses = components(prediction, target, valid)
    gradients = {}
    for index, name in enumerate(COMPONENTS):
        values = torch.autograd.grad(losses[name], (prediction, *parameters.values()),
                                     retain_graph=index < len(COMPONENTS)-1)
        gradients[name] = {'image': values[0].detach(),
                           'parameters': dict(zip(parameters, (v.detach() for v in values[1:]), strict=True))}
    return {k: float(v.detach()) for k, v in losses.items()}, gradients


def reassembly(gradients):
    result = {}
    for target in ('image', *PARAMETERS):
        values = [gradients[name]['image'] if target == 'image' else
                  gradients[name]['parameters'][target] for name in COMPONENTS]
        recombined = .8 * values[0] + .2 * values[1]
        result[target] = {'difference': norms(recombined-values[2]),
                          'relative_l2': float((recombined.double()-values[2].double()).norm() /
                                               values[2].double().norm().clamp_min(1e-30))}
    return result


def leaf_analysis(baseline, plus, minus, target, valid, epsilon, dtype):
    """All inputs originate from the same three FP32 final prediction arrays."""
    base = baseline.detach().to(dtype).requires_grad_(True)
    target = target.detach().to(dtype)
    values = components(base, target, valid)
    leaf_gradients = {name: torch.autograd.grad(values[name], base,
                      retain_graph=index < len(COMPONENTS)-1)[0].detach()
                      for index, name in enumerate(COMPONENTS)}
    with torch.no_grad():
        positive = components(plus.detach().to(dtype), target, valid)
        negative = components(minus.detach().to(dtype), target, valid)
    jd = (plus.detach().double()-minus.detach().double())/(2*epsilon)
    forward = (plus.detach().double()-baseline.detach().double())/epsilon
    backward = (baseline.detach().double()-minus.detach().double())/epsilon
    result = {}
    for name in COMPONENTS:
        scalar = [float(group[name].detach()) for group in (values, positive, negative)]
        central = (scalar[1]-scalar[2])/(2*epsilon)
        derivative = dot(leaf_gradients[name], jd)
        result[name] = {'baseline_loss': scalar[0], 'plus_loss': scalar[1], 'minus_loss': scalar[2],
                        'image_gradient': norms(leaf_gradients[name]), 'image_gradient_dot_Jd': derivative,
                        'image_gradient_dot_forward_secant': dot(leaf_gradients[name], forward),
                        'image_gradient_dot_backward_secant': dot(leaf_gradients[name], backward),
                        'scalar_central_difference': central,
                        'scalar_forward_slope': (scalar[1]-scalar[0])/epsilon,
                        'scalar_backward_slope': (scalar[0]-scalar[2])/epsilon,
                        'image_chain_discrepancy': discrepancy(derivative, central)}
    recombined = .8*leaf_gradients['l1']+.2*leaf_gradients['dssim7']
    result['reassembly'] = {
        'value_residuals': {tag: float(group['combined'].detach())-
                            (.8*float(group['l1'].detach())+.2*float(group['dssim7'].detach()))
                            for tag, group in (('baseline', values), ('plus', positive), ('minus', negative))},
        'image_gradient_difference': norms(recombined-leaf_gradients['combined']),
        'directional_residual': result['combined']['image_gradient_dot_Jd']-
            (.8*result['l1']['image_gradient_dot_Jd']+.2*result['dssim7']['image_gradient_dot_Jd']),
    }
    result['layouts'] = {'baseline_leaf': layout_info(base), 'target': layout_info(target),
                        'plus_loss_input': layout_info(plus.detach().to(dtype)),
                        'minus_loss_input': layout_info(minus.detach().to(dtype))}
    return result, leaf_gradients


def parameter_directions(gradients, requested, positive, negative, original, epsilon):
    realized = (positive.double()-negative.double())/(2*epsilon)
    forward, backward = (positive.double()-original.double())/epsilon, (original.double()-negative.double())/epsilon
    rows = {}
    for component in COMPONENTS:
        rows[component] = {}
        for name in PARAMETERS:
            gradient = gradients[component]['parameters'][name]
            rows[component][name] = {'gradient': norms(gradient), 'direction_is_zero': name != PARAMETERS[0],
                'g_dot_requested_direction': dot(gradient, requested) if name == PARAMETERS[0] else 0.,
                'g_dot_realized_central_direction': dot(gradient, realized) if name == PARAMETERS[0] else 0.,
                'g_dot_realized_forward_direction': dot(gradient, forward) if name == PARAMETERS[0] else 0.,
                'g_dot_realized_backward_direction': dot(gradient, backward) if name == PARAMETERS[0] else 0.}
    return rows, {'requested': norms(requested), 'realized': norms(realized),
                  'relative_l2_direction_error': float((realized-requested.double()).norm()/requested.double().norm())}


def prepare(output):
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite an existing diagnostic')
    source_plan = REFERENCE/'plan.json'
    require(sha(source_plan) == REFERENCE_PLAN_SHA, 'Wrong completed reference plan')
    old, receipt = read(source_plan), read(REFERENCE/'execution_receipt.json')
    require(receipt['status'] == 'completed' and receipt['render_calls'] == 40 and
            receipt['all_model_tensors_finally_restored_exact'] and
            receipt['all_bound_inputs_and_sources_unchanged'], 'Reference diagnostic must be completed')
    require(receipt['plan_sha256'] == REFERENCE_PLAN_SHA and old['view']['name'] == SPEC['view']
            and old['view']['split'] == 'train', 'Wrong view/reference linkage')
    source = Path(old['snapshot'])
    for relative, expected in old['source_hashes'].items():
        require(sha(source/relative) == expected, f'Reference source changed: {relative}')
    inputs = dict(old['input_hashes'])
    inputs.update({str(source_plan): sha(source_plan),
                   str(REFERENCE/'execution_receipt.json'): sha(REFERENCE/'execution_receipt.json')})
    for path, expected in inputs.items():
        require(sha(path) == expected, f'Reference input changed: {path}')
    output.parent.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(output.parent).free >= 256 << 20, 'Insufficient data-volume reserve')
    output.mkdir()
    snapshot = output/'source_snapshot'
    shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, snapshot/Path(__file__).name)
    hashes = {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob('*.py'))}
    require(all(hashes[k] == value for k, value in old['source_hashes'].items()), 'Reference copy differs')
    allowed = [str((Path(old['root'])/old['view'][key]).resolve()) for key in ('image_path', 'valid_path')]
    plan = {'status': 'locked_pending_root_gpu_authorization', 'specification': SPEC,
            'snapshot': str(snapshot), 'output': str(output), 'root': old['root'], 'base': old['base'],
            'manifest': old['manifest'], 'view': old['view'], 'source_hashes': hashes,
            'input_hashes': inputs, 'original_plan': old['original_plan'],
            'original_step1_loss': old['original_step1_losses']['00_native'],
            'allowed_pixel_paths': allowed, 'reference_plan_sha256': REFERENCE_PLAN_SHA,
            'execution_policy': 'Root approval required; timeout --signal=KILL 60s; frozen entrypoint/PYTHONPATH; no retry'}
    write(output/'plan.json', plan)
    return output/'plan.json'


def verify(plan):
    require(plan['specification'] == SPEC and plan['status'] == 'locked_pending_root_gpu_authorization', 'Protocol differs')
    snapshot = Path(plan['snapshot'])
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Execute the copied entrypoint')
    actual = {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob('*.py'))}
    require(actual == plan['source_hashes'], 'Frozen source differs')
    for path, expected in plan['input_hashes'].items():
        require(sha(path) == expected, f'Bound input differs: {path}')
    modules = {}
    for name in ('model', 'train', 'raw_grid', 'losses', 'evaluate', 'coordinates'):
        module = importlib.import_module('bridge_rgs.'+name)
        path = Path(module.__file__).resolve()
        require(path == snapshot/'bridge_rgs'/f'{name}.py', f'Unexpected import: {name}')
        modules[name] = {'path': str(path), 'sha256': sha(path)}
    reference = importlib.import_module('run_raw_grid_appearance')
    require(Path(reference.__file__).resolve() == snapshot/'run_raw_grid_appearance.py', 'Wrong reference runner')
    old = read(plan['original_plan'])
    views = sorted((v for v in read(plan['manifest'])['views'] if v['split'] == 'train'), key=lambda v: v['name'])
    require(len(views) == 350 and views[old['training_order'][0]] == plan['view'], 'Original TRAIN152 mapping differs')
    return reference, modules


def run(plan_path):
    started = time.monotonic()
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    output = Path(plan['output'])
    require(not (output/'execution_receipt.json').exists(), 'No rerun or overwrite')
    os.chdir(plan['root'])
    reference, modules = verify(plan)
    clients = reference.gpu_idle()
    receipt = {'status': 'running', 'plan_sha256': sha(plan_path), 'pid': os.getpid(),
               'actual_imports': modules, 'gpu_handoff': clients, 'render_calls': 0}
    write(output/'execution_receipt.json', receipt)
    original_read = cv2.imread
    reads = Counter()
    scene = original_state = original_render = before = None
    flags, modes, parameter_grads = {}, {}, {}

    def alarm(*_):
        raise TimeoutError('Three-render gradient diagnostic deadline')

    def guarded_read(path, *args, **kwargs):
        absolute = str(Path(path).resolve())
        require(absolute in plan['allowed_pixel_paths'], 'Pixel read outside native TRAIN152 RGB/valid')
        reads[absolute] += 1
        return original_read(path, *args, **kwargs)

    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.1, 55-(time.monotonic()-started)))
    cv2.imread = guarded_read
    try:
        torch.set_num_threads(8)
        cv2.setNumThreads(8)
        from bridge_rgs.train import load_scene
        scene, checkpoint = load_scene(plan['base'])
        flags = {name: p.requires_grad for name, p in scene.named_parameters()}
        modes = {name: module.training for name, module in scene.named_modules()}
        parameter_grads = {name: None if p.grad is None else p.grad.detach().clone()
                           for name, p in scene.named_parameters()}
        original_state = {name: value.detach().clone() for name, value in scene.state_dict().items()}
        before = {name: reference.tensor_hash(value) for name, value in original_state.items()}
        scene.eval()
        for name, parameter in scene.named_parameters():
            parameter.requires_grad_(name in PARAMETERS)
        parameters = {name: dict(scene.named_parameters())[name] for name in PARAMETERS}
        require(all(p.dtype == torch.float32 for p in parameters.values()), 'Require FP32 appearance parameters')
        manifest = read(plan['manifest'])
        require(torch.equal(checkpoint['training_cameras'], reference.original_training_poses(manifest)), 'Base pose binding differs')
        layouts, warps = reference.layouts_for_views(manifest, [plan['view']])
        layout = layouts[0]
        pose = torch.tensor(plan['view']['w2c_original'], device='cuda', dtype=torch.float32)
        target, valid = reference.load_target(plan['view'], '00_native', plan['root'])
        original_render = scene.render

        def counted_render(*args, **kwargs):
            require(receipt['render_calls'] < 3, 'Three-render budget exceeded')
            receipt['render_calls'] += 1
            return original_render(*args, **kwargs)

        scene.render = counted_render

        def predict():
            return reference.predict_arm(reference.render_training_rgb(scene, layout, pose), layout, '00_native')

        baseline = predict()
        baseline_layout = layout_info(baseline)
        values, gradients = graph_gradients(baseline, parameters, target, valid)
        require(values['combined'] == plan['original_step1_loss'], 'FP32 baseline is not original step1 exact')
        require(all(p.grad is None for p in scene.parameters()), 'autograd.grad unexpectedly accumulated model .grad')
        baseline = baseline.detach()
        gradient = gradients['combined']['parameters'][PARAMETERS[0]]
        require(bool(torch.isfinite(gradient).all()) and float(gradient.abs().max()) > 0, 'Invalid fixed SH0 direction')
        direction = gradient/gradient.abs().max()
        original = original_state[PARAMETERS[0]]
        epsilon = SPEC['epsilon']
        with torch.no_grad():
            positive, negative = original+epsilon*direction, original-epsilon*direction
            parameters[PARAMETERS[0]].copy_(positive)
            plus = predict().detach()
            parameters[PARAMETERS[0]].copy_(negative)
            minus = predict().detach()
            parameters[PARAMETERS[0]].copy_(original)
        parameter_rows, realized = parameter_directions(gradients, direction, positive, negative, original, epsilon)
        jd = (plus.double()-minus.double())/(2*epsilon)
        leaf, chain = {}, {}
        for label, dtype in (('float32', torch.float32), ('float64', torch.float64)):
            rows, leaf_grads = leaf_analysis(baseline, plus, minus, target, valid, epsilon, dtype)
            leaf[label] = rows
            chain[label] = {}
            for component in COMPONENTS:
                param_dot = parameter_rows[component][PARAMETERS[0]]['g_dot_realized_central_direction']
                image_dot = rows[component]['image_gradient_dot_Jd']
                chain[label][component] = {'parameter_g_dot_realized_direction': param_dot,
                    'leaf_image_g_dot_Jd': image_dot, 'renderer_chain_discrepancy': discrepancy(param_dot, image_dot),
                    'original_graph_image_g_dot_Jd': dot(gradients[component]['image'], jd),
                    'leaf_minus_original_graph_image_gradient': norms(leaf_grads[component].double()-gradients[component]['image'].double())}
        require(receipt['render_calls'] == 3 and sum(reads.values()) == 2 and
                set(reads) == set(plan['allowed_pixel_paths']), 'Wrong render/pixel-read budget')
        verify(plan)
        require(sha(plan_path) == receipt['plan_sha256'], 'Plan changed during run')
        receipt.update(status='completed', baseline_losses=values, first_training_log_forward_exact=True,
            baseline_value_reassembly_residual=values['combined']-(.8*values['l1']+.2*values['dssim7']),
            graph_gradient_reassembly=reassembly(gradients), parameter_components=parameter_rows,
            original_graph_image_gradient_norms={name: norms(gradients[name]['image']) for name in COMPONENTS},
            parameter_direction=realized, image_secant=norms(jd), leaf_image_losses=leaf, chain_comparisons=chain,
            original_graph_prediction_layout=baseline_layout,
            prediction_sha256={k: reference.tensor_hash(v) for k, v in (('base', baseline), ('plus', plus), ('minus', minus))},
            support={'rgb_pixels': int(valid.sum()), 'shape': list(baseline.shape)}, warps=warps,
            read_counts=dict(reads), all_bound_inputs_and_sources_unchanged=True,
            semantic_label_pixels_decoded=0, val_pixels_decoded=0, optimizer_steps=0,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), limits=SPEC['limits'])
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        cv2.imread = original_read
        try:
            if scene is not None and original_state is not None:
                with torch.no_grad():
                    for name, value in scene.state_dict().items():
                        value.copy_(original_state[name])
                if original_render is not None:
                    scene.render = original_render
                for name, parameter in scene.named_parameters():
                    parameter.requires_grad_(flags[name])
                    parameter.grad = parameter_grads[name]
                for name, module in scene.named_modules():
                    module.training = modes[name]
                expected = before if before is not None else {
                    name: reference.tensor_hash(value) for name, value in original_state.items()}
                receipt['all_model_tensors_finally_restored_exact'] = expected == {
                    name: reference.tensor_hash(value) for name, value in scene.state_dict().items()}
                receipt['all_model_flags_finally_restored_exact'] = flags == {
                    name: p.requires_grad for name, p in scene.named_parameters()} and modes == {
                    name: m.training for name, m in scene.named_modules()}
                require(receipt['all_model_tensors_finally_restored_exact'] and
                        receipt['all_model_flags_finally_restored_exact'], 'Finally restoration failed')
        except BaseException as error:
            receipt.update(status='failed', restoration_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            receipt.update(seconds=time.monotonic()-started, checkpoint_written=False)
            write(output/'execution_receipt.json', receipt, update=True)
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > SPEC['output_bytes_limit']:
            receipt.update(status='failed', output_budget_error='Output exceeds 2MiB')
            write(output/'execution_receipt.json', receipt, update=True)
            raise ValueError('Output exceeds 2MiB')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/mnt/data/SHM2026/runs/appearance_gradient_chain_diagnostic'))
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.output))
    else:
        run(args.run)


if __name__ == '__main__':
    main()
