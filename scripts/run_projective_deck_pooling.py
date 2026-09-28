"""Frozen, matched four-arm refiner-only continuation; never writes a full scene."""
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
from pathlib import Path

ROOT = Path('/home/sky/workspace/SHM2026')
PRIOR = Path('/mnt/data/SHM2026/runs/multifield_h3_teacher_v1')
ARMS = ('original', 'per_camera', 'projective', 'wrong')
BASE_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
SPEC = {
    'format': 'projective_deck_pooling_experiment_v1', 'arms': list(ARMS),
    'candidate': 'projective', 'steps_per_arm': 2000, 'seed': 42,
    'optimizer': {'type': 'Adam', 'lr': .0003, 'eps': 1e-8, 'betas': [.9, .999]},
    'objective': 'final weighted CE + 0.2 Lovasz + 0.001 residual.square().mean()',
    'class_weight_power': .25, 'iterations': '0..1999', 'alpha': 'min(iteration/500,1)',
    'axis': [.39810321798792697, -.012510913979839057, .917255310619157],
    'offsets': '17 uniform offsets from -256 to +256 canvas pixels',
    'wrong_camera': 'Rz(pi/2) @ real R; sampling only',
    'support': 'common per-query/per-offset intersection of all three sampled arms',
    'train_scope': 'refiner only; no augmentation; same shuffled labeled TRAIN sequence',
    'teacher': 'fixed 00f5 H+ cached soft probabilities; only after rendered RGB byte identity',
    'rgb': 'unchanged E delivered PNGs/metrics reused and explicitly reported',
    'evaluation': 'all 200 masks before 41 GT payloads; one fixed endpoint, no best selection',
    'bootstrap': {'repeats': 5000, 'seed': 20260926},
    'gain_thresholds': {'original': .0015, 'per_camera': .001, 'wrong': .001, 'E': .002},
    'ci_lower_strict': 0., 'class_guard': {'cable_min_gain': -.001, 'other_min_gain': -.002},
    'internal_timeout_seconds': 3300, 'external_timeout_seconds': 3360,
    'claim': 'TRAIN-informed single-deck exploration; dual-axis failure preserved; no novelty claim',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    content = json.dumps(value, indent=2, allow_nan=False) + '\n'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content)


def files(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(Path(directory).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def training_population(manifest):
    train = sorted((v for v in manifest['views'] if v['split'] == 'train'), key=lambda v: v['name'])
    labeled = [v for v in train if v.get('mask_path')]
    require(len(train) == 350 and len(labeled) == 259, 'Wrong TRAIN population')
    require(len({v['name'] for v in train}) == 350, 'Duplicate TRAIN names')
    return labeled


def prepare(output):
    import numpy as np
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite an existing experiment')
    old = read(PRIOR/'plan.json')
    require(files(old['source_snapshot']) == old['source_hashes'], 'Prior package changed')
    manifest_path = ROOT/'artifacts/prepared/manifest.json'
    manifest = read(manifest_path)
    views = training_population(manifest)
    inputs = {}
    def bind(path, expected=None):
        path = Path(path).resolve()
        value = sha(path)
        require(expected is None or expected == value, f'Changed input: {path}')
        inputs[str(path)] = value
        return str(path)
    base = bind(old['components']['selected']['checkpoint'], BASE_SHA)
    bind(manifest_path, '551546979a583d46e840bd485559721f361bc28fa4dca60826374ceb74b315fa')
    for v in views:
        require(v['w2c'] == v['w2c_original'], 'Training pose is refined')
        for key in ('image_path', 'mask_path', 'valid_path'):
            bind(v[key])
    for name in ('plan.json', 'execution_receipt.json', 'predictions_receipt.json', 'E/official_metrics.json'):
        bind(PRIOR/name)
    for p in (ROOT/'uv.lock', Path('/mnt/data/SHM2026/runs/projective_structure_axes_v1/plan.json'),
              Path('/mnt/data/SHM2026/runs/projective_structure_axes_v1/execution_receipt.json'),
              Path('/mnt/data/SHM2026/runs/projective_structure_axes_v1/independent_cpu_review.json')):
        bind(p)
    axis_plan = read('/mnt/data/SHM2026/runs/projective_structure_axes_v1/plan.json')
    init_path = str((ROOT/'artifacts/prepared/init_points.npz').resolve())
    bind(init_path, axis_plan['input_hashes'][init_path])
    bind(Path('/mnt/data/SHM2026/runs/coupled_semantic_gradient_diagnostic_v2/plan.json'))
    predictions = read(PRIOR/'predictions_receipt.json')['predictions']
    for record in read(PRIOR/'execution_receipt.json')['h3_canvases'].values():
        bind(record['canvas_rgb']['path'], record['canvas_rgb']['sha256'])
    for r in predictions:
        if r['arm'] == 'A':
            bind(r['teacher_soft_canvas']['path'], r['teacher_soft_canvas']['sha256'])
            bind(r['rgb'], r['rgb_sha256'])
        if r['arm'] == 'E':
            bind(r['rgb'], r['rgb_sha256'])
    snapshot = output/'source_snapshot'
    shutil.copytree(Path(old['source_snapshot'])/'bridge_rgs', snapshot/'bridge_rgs',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('structure_axes.py', 'projective_pooling.py'):
        shutil.copy2(ROOT/'src/bridge_rgs'/name, snapshot/'bridge_rgs'/name)
    for p in (Path(__file__), ROOT/'scripts/evaluate_projective_deck_pooling.py',
              ROOT/'scripts/compare_official_evaluations.py',
              ROOT/'tests/test_projective_pooling.py', ROOT/'tests/test_structure_axes.py',
              ROOT/'tests/test_projective_deck_runner.py', ROOT/'tests/test_projective_deck_evaluation.py',
              ROOT/'docs/projective_deck_pooling_protocol.md'):
        shutil.copy2(p, snapshot/p.name)
    rng = np.random.default_rng(42)
    order = []
    while len(order) < 2000:
        order.extend(rng.permutation(len(views)).tolist())
    plan = {'specification': SPEC, 'root': str(ROOT), 'output': str(output), 'source_snapshot': str(snapshot),
                'source_hashes': files(snapshot), 'input_hashes': inputs, 'base_checkpoint': base,
                'base_checkpoint_sha256': BASE_SHA, 'manifest': str(manifest_path), 'training_views': views,
                'training_order': order[:2000], 'prior_plan_path': str(PRIOR/'plan.json'),
                'prior_receipt_path': str(PRIOR/'execution_receipt.json'),
                'prior_predictions_path': str(PRIOR/'predictions_receipt.json'),
                'prior_e_metrics': str(PRIOR/'E/official_metrics.json'),
                'runtime_versions': {name: importlib.metadata.version(name) for name in
                                     ('torch', 'gsplat', 'numpy', 'scipy', 'opencv-python-headless', 'transformers')},
                'prepare_new_val_annotation_payload_reads': 0}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json'),
                      'source_count': len(plan['source_hashes']), 'bound_input_count': len(inputs)}), flush=True)


def verify(plan):
    require(plan['specification'] == SPEC, 'Specification changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Use frozen runner')
    require(files(plan['source_snapshot']) == plan['source_hashes'], 'Frozen sources changed')
    require(all(importlib.metadata.version(name) == value for name, value in plan['runtime_versions'].items()),
            'Runtime package version changed')
    for path, expected in plan['input_hashes'].items():
        require(sha(path) == expected, f'Bound input changed: {path}')


def gpu_inventory():
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                     '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in lines.splitlines():
        pid, name, memory = line.split(',', 2)
        resolved = str(Path(f'/proc/{int(pid)}/exe').resolve())
        require(resolved == '/usr/share/rustdesk/rustdesk', f'Foreign compute process: {pid} {resolved}')
        rows.append({'pid': int(pid), 'name': name.strip(), 'executable': resolved, 'memory_mib': memory.strip()})
    return rows


def state_digest(state, *, refiner=False):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        if key.startswith('refiner.') == refiner:
            array = tensor.detach().cpu().contiguous().numpy()
            h.update(key.encode()); h.update(str(array.dtype).encode())
            h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def train_arm(plan, arm):
    import numpy as np
    import torch

    from bridge_rgs.io import seed_everything
    from bridge_rgs.losses import semantic_loss
    from bridge_rgs.projective_pooling import attach_projective_context
    from bridge_rgs.train import ImageCache, load_scene, training_class_weights
    seed_everything(42)
    directory = Path(plan['output'])/arm
    directory.mkdir()
    start = time.monotonic()
    scene, state = load_scene(plan['base_checkpoint'])
    train_views = [v for v in read(plan['manifest'])['views'] if v['split'] == 'train']
    require(np.array_equal(state['training_cameras'].numpy(),
                           np.asarray([v['w2c'] for v in train_views], np.float32)),
            'Base training cameras differ from fixed manifest')
    camera_sha = hashlib.sha256(state['training_cameras'].numpy().tobytes()).hexdigest()
    scene.requires_grad_(False)
    scene.refiner.requires_grad_(True)
    scene.train()
    require(all(p.requires_grad == name.startswith('refiner.') for name, p in scene.named_parameters()),
            'Trainable scope differs')
    frozen_before = state_digest(scene.state_dict())
    refiner_before = state_digest(scene.state_dict(), refiner=True)
    del state
    adapter = attach_projective_context(scene, {'mode': arm})
    optimizer = torch.optim.Adam(scene.refiner.parameters(), lr=.0003, eps=1e-8)
    weights = training_class_weights(plan['training_views'], {'class_weight_power': .25})
    torch.testing.assert_close(weights.cpu(), torch.tensor([
        .4560448825, .9227690101, .8530434370, 1.1935913563, 1.5745513439]), rtol=0, atol=1e-7)
    cache = ImageCache(limit=32)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    first_probabilities_sha256 = None
    with (directory/'steps.jsonl').open('x') as log:
        for iteration, index in enumerate(plan['training_order']):
            view = plan['training_views'][index]
            data = cache.get(view, 1.)
            adapter.set_step(iteration)
            optimizer.zero_grad(set_to_none=True)
            result = scene.render(data['K'], data['w2c'], data['width'], data['height'],
                                  degree=scene.sh_degree, absgrad=False, refinement_grad_to_field=False)
            if iteration == 0:
                first_probabilities_sha256 = hashlib.sha256(
                    result['probabilities'].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            loss = semantic_loss(result['probabilities'], data['mask'], data['valid'], weights)
            loss = loss + .001*result['residual'].square().mean()
            require(bool(torch.isfinite(loss)), f'Nonfinite loss in {arm}/{iteration}')
            loss.backward()
            if iteration % 100 == 0 or iteration == 1999:
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all())
                            for p in scene.refiner.parameters()), 'Nonfinite gradient')
            optimizer.step()
            row = {'iteration': iteration, 'view': view['name'], 'loss': float(loss.detach()),
                       'alpha': 0. if arm == 'original' else min(iteration/500, 1.),
                       'elapsed_seconds': time.monotonic()-start}
            log.write(json.dumps(row)+'\n')
            if iteration % 100 == 0 or iteration == 1999:
                log.flush()
                print(json.dumps({'arm': arm, **row}), flush=True)
            rows.append(row)
            del result, loss, data
    torch.cuda.synchronize()
    require(state_digest(scene.state_dict()) == frozen_before, 'Frozen scene parameters changed')
    require(state_digest(scene.state_dict(), refiner=True) != refiner_before, 'Refiner did not train')
    require(all(bool(torch.isfinite(p).all()) for p in scene.refiner.parameters()), 'Nonfinite final head')
    delta = {'format': 'projective_deck_refiner_delta_v1', 'base_checkpoint_sha256': BASE_SHA,
                 'mode': arm, 'step': 2000, 'refiner_state': {k: v.detach().cpu() for k, v in scene.refiner.state_dict().items()},
                 'plan_sha256': sha(Path(plan['output'])/'plan.json'), 'specification': SPEC,
                 'refiner_config': scene.refiner_config, 'pixel_protocol': scene.pixel_protocol,
                 'manifest_sha256': sha(plan['manifest']), 'training_camera_sha256': camera_sha,
                 'optimizer_state': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state()}
    torch.save(delta, directory/'refiner_delta.pt')
    receipt = {'status': 'completed', 'arm': arm, 'steps': len(rows),
                   'training_names_sha256': hashlib.sha256(json.dumps([r['view'] for r in rows]).encode()).hexdigest(),
                   'frozen_nonrefiner_sha256': frozen_before, 'refiner_before_sha256': refiner_before,
                   'training_camera_sha256': camera_sha,
                   'refiner_after_sha256': state_digest(scene.state_dict(), refiner=True),
                   'delta_sha256': sha(directory/'refiner_delta.pt'),
                   'training_log_sha256': sha(directory/'steps.jsonl'),
                   'class_weights': weights.tolist(), 'elapsed_seconds': time.monotonic()-start,
                   'iteration0_probabilities_sha256': first_probabilities_sha256,
                   'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'final_geometry_diagnostics': adapter.diagnostics,
                   'val_payload_reads': 0, 'scene_calls': 2000, 'backwards': 2000, 'optimizer_steps': 2000}
    write(directory/'training_receipt.json', receipt)
    adapter.detach()
    del scene, optimizer, adapter, cache, delta
    torch.cuda.empty_cache()
    return receipt


def execute(path):
    plan = read(path)
    output = Path(plan['output'])
    require(not (output/'execution_receipt.json').exists(), 'Execution already attempted')
    verify(plan)
    report = {'status': 'running', 'plan_sha256': sha(path), 'gpu_before': gpu_inventory(),
                  'train_arms': [], 'new_val_annotation_payload_reads': 0,
                  'cost_scope': 'desktop-shared GPU; not an exclusive hardware comparison'}
    write(output/'execution_receipt.json', report)
    def deadline(*_):
        raise TimeoutError('Fixed internal experiment deadline reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(SPEC['internal_timeout_seconds'])
    start = time.monotonic()
    try:
        sys.path.insert(0, plan['source_snapshot'])
        for arm in ARMS:
            report['train_arms'].append(train_arm(plan, arm))
            write(output/'execution_receipt.json', report)
        require(len({r['training_names_sha256'] for r in report['train_arms']}) == 1, 'Unmatched sample sequence')
        require(len({r['frozen_nonrefiner_sha256'] for r in report['train_arms']}) == 1, 'Unmatched frozen field')
        require(len({r['refiner_before_sha256'] for r in report['train_arms']}) == 1, 'Unmatched initialization')
        require(len({r['iteration0_probabilities_sha256'] for r in report['train_arms']}) == 1,
                'Alpha zero did not preserve the exact initial prediction')
        evaluate = importlib.import_module('evaluate_projective_deck_pooling')
        report['evaluation'] = evaluate.evaluate(plan)
        report['new_val_annotation_payload_reads'] = report['evaluation']['annotation_payload_reads']
        verify(plan)
        report['status'] = 'completed'
        report['bound_inputs_and_sources_unchanged'] = True
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-start
        write(output/'execution_receipt.json', report)
    print(json.dumps({'status': report['status'], 'receipt_sha256': sha(output/'execution_receipt.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path)
    group.add_argument('--execute', type=Path)
    args = parser.parse_args()
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode before freezing/execution')
    prepare(args.prepare) if args.prepare else execute(args.execute)


if __name__ == '__main__':
    main()
