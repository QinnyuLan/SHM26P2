"""Fixed completed integrated endpoint: 48 TRAIN readouts, then descriptive scoring.

Preparation does not decode pixels or load a checkpoint. Execution is explicitly
separate and must use the copied worker. These interventions are not retraining
controls, and the frozen head can see an out-of-distribution semantic prior.
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
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
PRODUCER = Path('/mnt/data/SHM2026/runs/semantic_partition_matched_v2')
NAMES = [f'{i:03}.png' for i in (2, 21, 41, 59, 79, 100, 118, 137, 156, 176, 200, 220, 241, 259, 278, 300)]
CASES = ('learned', 'own_qbar', 'perpendicular')
WEIGHTS = [.45604488253593445, .9227690100669861, .8530434370040894,
           1.1935913562774658, 1.5745513439178467]
WEIGHTS_REPORT = Path('/mnt/data/SHM2026/runs/support_coupled_source_cpu_audit_v1/report.json')
WEIGHTS_SHA = '8e9aea80c1bf77bc1f77e677268bfa682ed746d7c888dd81f012df770f8396ce'
SPEC = {
    'protocol': 'partition_mass_arrangement_v1', 'endpoint': 'integrated last2000 only',
    'producer': str(PRODUCER),
    'names': NAMES, 'cases': list(CASES), 'profile': 'legacy_mixed_v1',
    'readout': 'learned refiner shared; old context features/moments/p3d fixed',
    'own_qbar': 'a=Phi((-b+w)/sqrt(1+tau^2))-Phi((-b-w)/sqrt(1+tau^2)); qbar=a*qin+(1-a)*qout',
    'perpendicular': 'normalize(cross(n,e_argmin(abs(n)))); first-coordinate tie break; endpoints/b/w/tau unchanged',
    'qbar_precision': 'FP64 occupancy and endpoint mixture, one FP32 cast; no probability renormalization',
    'prediction_barrier': '48 scene calls and 112 FP32 NPY hashes before first TRAIN target decode',
    'scores': 'raw means production p3d after unchanged clamp/normalization; weighted CE (clamp1e-7, known valid pixel denominator), unweighted Brier, pooled 5class IoU; FP64 scoring, scene descriptive',
    'weights_fp32': WEIGHTS, 'probability_sum_tolerance': 5e-6,
    'bootstrap': {'unit': 'camera, paired', 'repeats': 2000, 'seed': 20260927},
    'differences': 'intervention minus learned; positive CE/Brier means worse; positive IoU means better',
    'scene_calls': 48, 'teacher_calls': 0, 'backwards': 0, 'optimizer_steps': 0,
    'new_rgb_outputs': 0, 'rgb_or_original_annotation_decodes': 0, 'val_views': 0,
    'selection_or_adoption_gate': None,
    'limits': 'TRAIN in-fit, fixed single endpoint; same-head interventions can be OOD. Analytic unoccluded normalized class mass is not occluded image mass. No independent training comparison, generalization or innovation claim.',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, replace=False):
    with Path(path).open('w' if replace else 'x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def tree(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def completed_producer(folder):
    """All completion gates precede any endpoint hash or tensor load."""
    folder = Path(folder)
    plan = read(folder/'plan.json')
    execution = read(folder/'execution_receipt.json')
    launch = read(folder/'launch_receipt.json')
    audit = read(folder/'independent_cpu_review.json')
    digest = sha(folder/'plan.json')
    require(launch.get('status') == 'completed' and launch.get('natural_completion') is True
            and launch.get('exit_code') == 0, 'Producer must naturally complete first')
    require(execution.get('status') == 'completed' and execution.get('bound_sources_inputs_unchanged') is True
            and audit.get('status') == 'passed' and audit.get('inputs_sources_outputs_unchanged') is True,
            'Producer completion/independent audit missing')
    require(launch['plan_sha256'] == execution['plan_sha256'] == audit['plan_sha256'] == digest
            and launch['execution_receipt_sha256'] == sha(folder/'execution_receipt.json'), 'Producer provenance differs')
    require(plan['specification']['candidate'] == 'integrated' and plan['specification']['steps'] == 2000
            and plan['specification']['class_weights_fp32'] == WEIGHTS, 'Wrong fixed producer contract')
    stage = read(folder/'integrated/training_receipt.json')
    require(stage['status'] == 'completed' and stage['arm'] == 'integrated' and stage['steps'] == 2000
            and stage in execution['training'], 'Fixed integrated stage incomplete/unbound')
    require(Path(stage['delta_path']).resolve() == (folder/'integrated/final_delta.pt').resolve(), 'Unexpected integrated endpoint')
    return plan, stage


def fixed_views(views):
    labeled = sorted((v for v in views if v['split'] == 'train' and v.get('mask_path')), key=lambda v: v['name'])
    require(len(labeled) == 259 and len({v['name'] for v in labeled}) == 259, 'Need original 259 TRAIN population')
    selected = [labeled[i*258//15] for i in range(16)]
    require([v['name'] for v in selected] == NAMES, 'Fixed targetwarp selection changed')
    return selected


def prepare(output, producer, seconds):
    output, producer = Path(output).resolve(), Path(producer).resolve()
    require(producer == PRODUCER, 'Only the predetermined current producer; no endpoint selection')
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Use a new data-disk directory')
    require(isinstance(seconds, int) and seconds > 0, 'Root must fix a positive execution budget')
    pp, stage = completed_producer(producer)
    snapshot = Path(pp['source_snapshot'])
    require(tree(snapshot) == pp['source_hashes'], 'Actual trained source changed')
    inputs = {}
    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, f'Changed dependency: {path}')
        inputs[str(path)] = digest
        return str(path)
    for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json', 'integrated/training_receipt.json'):
        bind(producer/name)
    delta = bind(stage['delta_path'], stage['delta_sha256'])
    base = bind(pp['base_checkpoint'], pp['base_checkpoint_sha256'])
    manifest = bind(pp['manifest'], pp['input_hashes'][str(Path(pp['manifest']).resolve())])
    bind(WEIGHTS_REPORT, WEIGHTS_SHA)
    require(read(WEIGHTS_REPORT)['class_weights_fp32'] == WEIGHTS, 'Class weights differ')
    bind(ROOT/'uv.lock', pp['input_hashes'][str(ROOT/'uv.lock')])
    views = fixed_views(read(manifest)['views'])
    cameras, targets = [], []
    for v in views:
        cameras.append({k: v[k] for k in ('name', 'split', 'K', 'w2c_original', 'width', 'height')})
        targets.append({'name': v['name'], **{k: bind(v[k], pp['input_hashes'][str(Path(v[k]).resolve())]) for k in ('mask_path', 'valid_path')}})
    sources = output/'source_snapshot'
    shutil.copytree(snapshot/'bridge_rgs', sources/'bridge_rgs', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    inherited = tree(sources/'bridge_rgs')
    require(all(pp['source_hashes']['bridge_rgs/'+k] == value for k, value in inherited.items()), 'Inherited package differs')
    for file in (Path(__file__), ROOT/'tests/test_partition_mass_arrangement.py', ROOT/'uv.lock'):
        shutil.copy2(file, sources/file.name)
    # Keep the immutable explanatory contract alongside the worker, not a live doc dependency.
    shutil.copy2(ROOT/'docs/semantic_partition_next_decision.md', sources/'decision_context.md')
    plan = {'specification': SPEC, 'status': 'prepared_not_started', 'output': str(output),
            'producer': str(producer), 'producer_plan_sha256': sha(producer/'plan.json'),
            'base_checkpoint': base, 'delta': delta, 'manifest': manifest,
            'cameras': cameras, 'targets': targets, 'input_hashes': inputs,
            'source_snapshot': str(sources), 'source_hashes': tree(sources),
            'inherited_package_hashes': inherited, 'installed_sources': pp['installed_sources'],
            'runtime_versions': pp['runtime_versions'], 'environment': pp['environment'],
            'execution_budget': {'internal_seconds': seconds, 'external_seconds': seconds+60, 'retries': 0},
            'array_payload_bytes': sum(v['width']*v['height']*5*4*7 for v in cameras),
            'prepare_pixel_decodes': 0, 'prepare_checkpoint_loads': 0}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json')}))


def verify(plan):
    require(plan['specification'] == SPEC, 'Specification changed')
    require(plan['producer'] == str(PRODUCER), 'Producer selection changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Execute copied worker only')
    require(tree(plan['source_snapshot']) == plan['source_hashes'], 'Frozen source changed')
    for path, digest in {**plan['input_hashes'], **plan['installed_sources']}.items():
        require(sha(path) == digest, f'Changed bound input: {path}')
    for name, version in plan['runtime_versions'].items():
        require(importlib.metadata.version(name) == version, f'Runtime changed: {name}')
    for name, value in plan['environment'].items():
        require(os.environ.get(name) == value, f'Environment changed: {name}')


def perpendicular_normal(normal):
    import torch
    require(normal.ndim == 2 and normal.shape[1] == 3 and bool(torch.isfinite(normal).all()), 'Bad normal')
    require(bool(((normal.square().sum(-1)-1).abs() <= 2e-6).all()), 'Normal must be unit')
    axis = torch.zeros_like(normal)
    axis.scatter_(1, normal.abs().argmin(-1, keepdim=True), 1.)
    result = torch.linalg.cross(normal, axis)
    return result/result.norm(dim=-1, keepdim=True)


class FixedReadout:
    """The renderer's field API only; owns no optimizer and never mutates field."""
    mode = 'integrated'

    def __init__(self, endpoints, slab):
        self._endpoints, self._slab = endpoints, slab

    def endpoints(self):
        return self._endpoints

    def slab_parameters(self):
        return self._slab


def readouts(field, interval_probability):
    import torch
    require(field.mode == 'integrated', 'Only the predetermined integrated endpoint')
    with torch.no_grad():
        qi, qo = (t.detach().clone() for t in field.endpoints())
        n, b, w, tau = (t.detach().clone() for t in field.slab_parameters())
        denominator = (1+tau.double().square()).sqrt()
        a = interval_probability((-b.double()-w.double())/denominator, (-b.double()+w.double())/denominator)
        exact = a[:, None]*qi.double()+(1-a[:, None])*qo.double()
        qbar = exact.to(qi.dtype)
        normal = perpendicular_normal(n)
        states = {'learned': FixedReadout((qi, qo), (n, b, w, tau)),
                  'own_qbar': FixedReadout((qbar, qbar), (n, b, w, tau)),
                  'perpendicular': FixedReadout((qi, qo), (normal, b, w, tau))}
        diagnostics = {'occupancy_min': float(a.min()), 'occupancy_max': float(a.max()),
                       'qbar_cast_max_error': float((qbar.double()-exact).abs().max()),
                       'qbar_simplex_max_error': float((qbar.double().sum(-1)-1).abs().max()),
                       'perpendicular_dot_absmax': float((normal*n).sum(-1).abs().max()),
                       'perpendicular_norm_error': float((normal.norm(dim=-1)-1).abs().max()),
                       'same_endpoints_b_w_tau': all(torch.equal(x, y) for x, y in zip(
                           (*states['learned'].endpoints(), *states['learned'].slab_parameters()[1:]),
                           (*states['perpendicular'].endpoints(), *states['perpendicular'].slab_parameters()[1:]), strict=True)),
                       'qbar_mass_scope': 'Unconditional continuous normalized Gaussian; visibility, raster truncation and p3d clamp/normalization are not mass-preserving identities.'}
    return states, diagnostics


def score(probability, mask, valid):
    require(probability.dtype == np.float32 and probability.shape == (*mask.shape, 5), 'Wrong prediction grid')
    require(valid.shape == mask.shape and np.isin(mask, [0, 1, 2, 3, 4, 255]).all(), 'Wrong target grid/IDs')
    matrix = np.zeros((5, 5), np.int64)
    ce, brier, count = 0., 0., 0
    weights = np.asarray(WEIGHTS, np.float32).astype(np.float64)
    for begin in range(0, mask.shape[0], 64):
        p = np.asarray(probability[begin:begin+64], np.float64)
        require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
                and np.max(abs(p.sum(-1)-1)) <= SPEC['probability_sum_tolerance'], 'Invalid probabilities')
        labels = mask[begin:begin+64]
        keep = (valid[begin:begin+64] > 0) & (labels < 5)
        y, p = labels[keep].astype(np.int64), p[keep]
        count += len(y)
        py = p[np.arange(len(y)), y]
        ce += float((-np.log(np.maximum(py, 1e-7))*weights[y]).sum())
        brier += float((np.square(p).sum(-1)-2*py+1).sum())
        matrix += np.bincount(y*5+p.argmax(-1), minlength=25).reshape(5, 5)
    require(count > 0, 'Empty known-valid support')
    return {'pixels': count, 'weighted_ce': ce/count, 'brier': brier/count, 'confusion_matrix': matrix.tolist()}


def ious(cm):
    cm = np.asarray(cm, np.float64)
    diag = np.diagonal(cm, axis1=-2, axis2=-1)
    union = cm.sum(-1)+cm.sum(-2)-diag
    values = np.divide(diag, union, out=np.full_like(diag, np.nan), where=union > 0)
    present = union > 0
    mean = np.divide(np.nansum(values, axis=-1), present.sum(-1),
                     out=np.full(present.shape[:-1], np.nan), where=present.sum(-1) > 0)
    return np.concatenate((mean[..., None], values), axis=-1)


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def summarize(rows):
    require(len(rows) == 16 and [r['name'] for r in rows] == NAMES, 'Need all fixed cameras')
    counts = np.asarray([np.bincount(x, minlength=16) for x in
                         np.random.default_rng(SPEC['bootstrap']['seed']).integers(0, 16, (SPEC['bootstrap']['repeats'], 16))])
    keys = ['old_p3d']+[case+'.'+kind for case in CASES for kind in ('raw', 'scene')]
    aggregate, matrices = {}, {}
    for key in keys:
        matrices[key] = np.asarray([r['scores'][key]['confusion_matrix'] for r in rows], np.int64)
        aggregate[key] = {m: float(np.mean([r['scores'][key][m] for r in rows])) for m in ('weighted_ce', 'brier')}
        aggregate[key].update(confusion_matrix=matrices[key].sum(0).tolist(),
                              pooled_iou=dict(zip(('miou_all', 'background', 'deck', 'cable', 'tower', 'foundation'),
                                                  map(finite_number, ious(matrices[key].sum(0))), strict=True)))
    comparisons = [('own_qbar.raw', 'learned.raw'), ('perpendicular.raw', 'learned.raw'),
                   ('own_qbar.scene', 'learned.scene'), ('perpendicular.scene', 'learned.scene'),
                   ('own_qbar.raw', 'old_p3d'), ('learned.raw', 'old_p3d')]
    pairs = {}
    for candidate, reference in comparisons:
        measures = {}
        for metric in ('weighted_ce', 'brier'):
            delta = np.asarray([r['scores'][candidate][metric]-r['scores'][reference][metric] for r in rows])
            measures[metric] = {'difference': float(delta.mean()), 'per_view_difference': delta.tolist(),
                                'paired_camera_95_interval': np.quantile(counts@delta/16, [.025, .975]).tolist()}
        difference = ious(matrices[candidate].sum(0))-ious(matrices[reference].sum(0))
        boot = ious(np.einsum('bv,vij->bij', counts, matrices[candidate]))-ious(np.einsum('bv,vij->bij', counts, matrices[reference]))
        for i, key in enumerate(('miou_all', 'background', 'deck', 'cable', 'tower', 'foundation')):
            finite = boot[np.isfinite(boot[:, i]), i]
            measures[key] = {'difference': finite_number(difference[i]), 'finite_bootstrap_replicates': len(finite),
                             'paired_camera_95_interval': np.quantile(finite, [.025, .975]).tolist() if len(finite) else None}
        pairs[candidate+' minus '+reference] = measures
    return {'aggregate': aggregate, 'paired_differences': pairs, 'adoption_gate': None}


def tensor_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def actual_imports(plan):
    snapshot = Path(plan['source_snapshot'])
    records = {}
    for name, module in sys.modules.items():
        if name == 'bridge_rgs' or name.startswith('bridge_rgs.'):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(snapshot) and plan['source_hashes'][str(path.relative_to(snapshot))] == sha(path), 'Unfrozen package import')
            records[name] = {'path': str(path), 'sha256': sha(path)}
    require({'bridge_rgs.train', 'bridge_rgs.partition_field', 'bridge_rgs.partition_projection',
             'bridge_rgs.partition_rasterizer', 'bridge_rgs.semantic_partition'} <= set(records), 'Missing core imports')
    return records


def save_array(folder, name, value):
    array = value.detach().cpu().contiguous().numpy()
    require(array.dtype == np.float32 and array.ndim == 3 and array.shape[-1] == 5
            and np.isfinite(array).all() and (array >= 0).all() and (array <= 1).all(), 'Invalid prediction dtype/range')
    for begin in range(0, len(array), 64):
        require(np.max(abs(array[begin:begin+64].astype(np.float64).sum(-1)-1)) <= SPEC['probability_sum_tolerance'],
                'Non-simplex prediction before any target read')
    path = folder/(name+'.npy')
    require(not path.exists(), 'Do not overwrite predictions')
    np.save(path, array, allow_pickle=False)
    return {'path': str(path), 'sha256': sha(path), 'shape': list(array.shape), 'dtype': 'float32'}


def prediction_barrier(records, calls):
    require(calls == 48 and len(records) == 16 and [r['name'] for r in records] == NAMES, 'All predictions must finish before targets')
    expected = {'old_p3d'} | {c+'.'+kind for c in CASES for kind in ('raw', 'scene')}
    for row in records:
        require(set(row['arrays']) == expected, 'Incomplete prediction set')
        for record in row['arrays'].values():
            require(sha(record['path']) == record['sha256'], 'Saved prediction changed')


def score_saved(plan, records, calls, report=None):
    prediction_barrier(records, calls)
    import cv2
    rows = []
    for record, target in zip(records, plan['targets'], strict=True):
        require(record['name'] == target['name'], 'Target order changed')
        mask = cv2.imread(target['mask_path'], cv2.IMREAD_UNCHANGED)
        if report is not None:
            report['target_pixel_decodes'] += 1
        valid = cv2.imread(target['valid_path'], cv2.IMREAD_UNCHANGED)
        if report is not None:
            report['target_pixel_decodes'] += 1
        require(mask is not None and valid is not None, 'Cannot read fixed TRAIN target')
        rows.append({'name': record['name'], 'scores': {key: score(np.load(item['path'], mmap_mode='r', allow_pickle=False), mask, valid)
                                                      for key, item in record['arrays'].items()}})
    return rows


def validate_delta(delta, plan, state):
    pp = read(Path(plan['producer'])/'plan.json')
    require(delta.get('format') == 'semantic_partition_matched_delta_v1' and delta.get('arm') == 'integrated'
            and delta.get('step') == 2000 and delta.get('specification') == pp['specification']
            and delta.get('plan_sha256') == plan['producer_plan_sha256']
            and delta.get('base_checkpoint_sha256') == plan['input_hashes'][plan['base_checkpoint']]
            and delta.get('manifest_sha256') == plan['input_hashes'][plan['manifest']]
            and delta.get('pixel_protocol') == 'legacy_mixed_v1' and delta.get('refiner_config') == state['refiner_config']
            and delta.get('ordinary_resume_supported') is False and isinstance(delta.get('partition_state'), dict), 'Wrong fixed endpoint')
    camera = state['training_cameras'].detach().cpu().contiguous().numpy()
    require(delta['training_camera_sha256'] == hashlib.sha256(camera.tobytes()).hexdigest(), 'Endpoint camera differs')
    train = [v for v in read(plan['manifest'])['views'] if v['split'] == 'train']
    require(camera.shape == (350, 4, 4) and np.array_equal(camera, np.asarray([v['w2c_original'] for v in train], np.float32)), 'Saved cameras differ')


def gpu_inventory():
    listing = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], text=True)
    result = []
    for line in listing.splitlines():
        pid, name, memory = line.split(',', 2)
        executable = Path(f'/proc/{int(pid)}/exe').resolve(strict=True)
        require(str(executable) == '/usr/share/rustdesk/rustdesk', 'Unexpected compute process; never terminate it')
        result.append({'pid': int(pid), 'name': name.strip(), 'used_memory_mib': memory.strip(), 'executable': str(executable)})
    return result


def predict(plan, report):
    import torch
    require(not any(n == 'bridge_rgs' or n.startswith('bridge_rgs.') for n in sys.modules), 'Import only the frozen package')
    sys.path.insert(0, plan['source_snapshot'])
    from bridge_rgs.partition_field import SemanticPartitionField, render_partition
    from bridge_rgs.semantic_partition import normal_interval_probability
    from bridge_rgs.train import load_scene
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    scene, state = load_scene(plan['base_checkpoint'])
    require(scene.pixel_protocol == 'legacy_mixed_v1' and scene.sh_degree == 3 and scene.mip_filter_config is None, 'Wrong fixed scene')
    delta = torch.load(plan['delta'], map_location='cpu', weights_only=False)
    validate_delta(delta, plan, state)
    scene.refiner.load_state_dict(delta['refiner_state'], strict=True)
    with torch.no_grad():
        logits = scene.semantic_decoder(scene.splats['sem_features'])
    field = SemanticPartitionField(logits, 'integrated')
    field.load_state_dict(delta['partition_state'], strict=True)
    del logits, delta
    original = {k: v.detach().cpu().clone() for k, v in scene.state_dict().items()}
    original_field = {k: v.detach().cpu().clone() for k, v in field.state_dict().items()}
    flags = {name: p.requires_grad for name, p in scene.named_parameters()}
    modes = {name: m.training for name, m in scene.named_modules()}
    field_flags = {name: p.requires_grad for name, p in field.named_parameters()}
    field_modes = {name: m.training for name, m in field.named_modules()}
    before = tensor_digest(original)
    field_before = tensor_digest(original_field)
    camera_before = tensor_digest({'camera': state['training_cameras']})
    report['state_before'] = {'scene': before, 'field': field_before, 'cameras': camera_before}
    report['numerical_runtime'] = {'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                                    'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                                    'float32_matmul_precision': torch.get_float32_matmul_precision(),
                                    'cudnn_benchmark': torch.backends.cudnn.benchmark,
                                    'autocast': False}
    original_render = scene.render
    had_instance_render = 'render' in scene.__dict__
    def counted_render(*args, **kwargs):
        report['scene_calls'] += 1
        return original_render(*args, **kwargs)
    scene.render = counted_render
    folder = Path(plan['output'])/'predictions'; folder.mkdir()
    records = []
    try:
        scene.eval().requires_grad_(False); field.eval().requires_grad_(False)
        with torch.no_grad():
            choices, report['field_intervention'] = readouts(field, normal_interval_probability)
            for camera in plan['cameras']:
                K = torch.tensor(camera['K'], dtype=torch.float32, device='cuda')
                pose = torch.tensor(camera['w2c_original'], dtype=torch.float32, device='cuda')
                row = {'name': camera['name'], 'arrays': {}, 'context_hashes': {}, 'projection': {}}
                for case in CASES:
                    result = render_partition(scene, choices[case], K, pose, camera['width'], camera['height'])
                    require(all(p.grad is None for p in scene.parameters()) and all(p.grad is None for p in field.parameters()), 'Unexpected gradient')
                    context = tensor_digest({key: result[key] for key in ('rgb', 'depth', 'alpha', 'partition_alpha', 'original_p3d')})
                    if case == 'learned':
                        row['arrays']['old_p3d'] = save_array(folder, camera['name']+'.old_p3d', result['original_p3d'])
                        row['context_hashes']['learned'] = context
                    require(context == row['context_hashes']['learned'], 'Geometry/RGB/old p3d changed between interventions')
                    row['context_hashes'][case] = context
                    row['projection'][case] = result['partition_projection_diagnostics']
                    for kind, key in (('raw', 'p3d'), ('scene', 'probabilities')):
                        row['arrays'][case+'.'+kind] = save_array(folder, camera['name']+'.'+case+'.'+kind, result[key])
                    del result
                records.append(row)
                print(json.dumps({'prediction_view': camera['name'], 'scene_calls': report['scene_calls']}), flush=True)
        report['state_after_predictions'] = {'scene': tensor_digest(scene.state_dict()), 'field': tensor_digest(field.state_dict()),
                                             'cameras': tensor_digest({'camera': state['training_cameras']})}
        require(report['state_after_predictions'] == report['state_before'], 'Prediction mutated state')
        report['actual_imports'] = actual_imports(plan)
        report['loaded_gsplat_binaries'] = {n: {'path': m.__file__, 'sha256': sha(m.__file__)} for n, m in sys.modules.items()
                                          if getattr(m, '__file__', '') and str(m.__file__).endswith('.so') and 'gsplat' in (n+str(m.__file__))}
    finally:
        if had_instance_render:
            scene.render = original_render
        else:
            del scene.render
        scene.load_state_dict(original, strict=True); field.load_state_dict(original_field, strict=True)
        for name, p in scene.named_parameters():
            p.requires_grad_(flags[name])
        for name, module in scene.named_modules():
            module.training = modes[name]
        for name, p in field.named_parameters():
            p.requires_grad_(field_flags[name])
        for name, module in field.named_modules():
            module.training = field_modes[name]
        report['state_restored'] = {'scene': tensor_digest(scene.state_dict()), 'field': tensor_digest(field.state_dict()),
                                    'cameras': tensor_digest({'camera': state['training_cameras']})}
        require(report['state_restored'] == report['state_before'], 'State restoration failed')
        report['flags_modes_restored'] = (all(p.requires_grad == flags[n] for n, p in scene.named_parameters())
                                          and all(m.training == modes[n] for n, m in scene.named_modules())
                                          and all(p.requires_grad == field_flags[n] for n, p in field.named_parameters())
                                          and all(m.training == field_modes[n] for n, m in field.named_modules()))
        require(report['flags_modes_restored'], 'Flags/modes restoration failed')
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    return records


def execute(path, expected):
    require(sha(path) == expected, 'Expected plan SHA differs')
    plan = read(path); output = Path(plan['output'])
    require(not (output/'execution_receipt.json').exists(), 'One attempt only')
    report = {'status': 'running', 'plan_sha256': expected, 'scene_calls': 0, 'teacher_calls': 0,
              'backwards': 0, 'optimizer_steps': 0, 'target_pixel_decodes': 0, 'val_views': 0,
              'new_rgb_outputs': 0, 'cost_scope': 'Includes model load, 48 predictions, array IO/hash and TRAIN statistics; may share display/RustDesk GPU, not FPS.'}
    write(output/'execution_receipt.json', report)
    start = time.monotonic()
    def expire(*_):
        raise TimeoutError('Fixed diagnostic deadline exhausted; no retry')
    signal.signal(signal.SIGALRM, expire); signal.alarm(plan['execution_budget']['internal_seconds'])
    try:
        verify(plan); completed_producer(plan['producer'])
        require(shutil.disk_usage(output).free >= plan['array_payload_bytes']+2**30, 'Insufficient data-disk space')
        report['gpu_before'] = gpu_inventory()
        records = predict(plan, report)
        prediction_barrier(records, report['scene_calls'])
        write(output/'predictions_receipt.json', {'status': 'predictions_completed', 'plan_sha256': expected,
                                                'scene_calls': 48, 'target_pixel_decodes': 0, 'records': records})
        report['prediction_receipt_sha256'] = sha(output/'predictions_receipt.json')
        report['prediction_barrier_elapsed_seconds'] = time.monotonic()-start
        rows = score_saved(plan, records, report['scene_calls'], report)
        analysis = {'protocol': SPEC, 'plan_sha256': expected, 'rows': rows, **summarize(rows)}
        write(output/'analysis.json', analysis); report['analysis_sha256'] = sha(output/'analysis.json')
        prediction_barrier(records, report['scene_calls']); verify(plan)
        report.update(status='completed', inputs_sources_predictions_unchanged=True)
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-start
        write(output/'execution_receipt.json', report, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path); action.add_argument('--execute', type=Path)
    action.add_argument('--print-spec', action='store_true')
    parser.add_argument('--producer', type=Path, default=PRODUCER)
    parser.add_argument('--internal-seconds', type=int)
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.print_spec:
        print(json.dumps(SPEC, indent=2)); return
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode generation')
    if args.prepare:
        prepare(args.prepare, args.producer, args.internal_seconds)
    else:
        require(args.expected_plan_sha256 is not None, 'Expected plan SHA required')
        execute(args.execute, args.expected_plan_sha256)


if __name__ == '__main__':
    main()
