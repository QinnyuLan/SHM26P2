"""Four fixed means endpoints: fresh H+ inference and original-grid scoring only."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
TRAIN = Path('/mnt/data/SHM2026/runs/continuous_geometry_proposals_v1')
RECOVERY = Path('/mnt/data/SHM2026/runs/direct_q_head_evaluation_recovery_v1')
TRAIN_PLAN_SHA = '61c66187e63325fa0d23e8ea9b8522165f9335b93424f3daa54b612e4553b588'
TRAIN_EXEC_SHA = '5a7e9ed582cbd530123d48f50cd394bbf939b28567fd270d958fbf4dc7a29242'
RECOVERY_PLAN_SHA = '36885e0362f41d7254f12f656dd9bcb202e3272518af8e9fb8ae2e44bda8ac38'
ARMS = ('baseline', 'joint', 'pcgrad', 'trust')
KINDS = ('raw', 'scene', 'joint')
NUMERICS = {'cudnn_allow_tf32': True, 'matmul_allow_tf32': False,
            'matmul_precision': 'highest', 'cudnn_benchmark': False}
INFERENCE = {'tile_size': 768, 'stride': 512, 'flip': True,
             'context_weight': .25, 'context_short_side': 768}
SPEC = {'protocol': 'continuous_geometry_proposal_evaluation_v1', 'arms': list(ARMS), 'kinds': list(KINDS),
        'scene_calls': 200, 'direct_q_shader_calls': 200, 'head_calls': 200,
        'teacher_predict_calls': 200, 'teacher_backbone_forwards': 2800,
        'native_gsplat_high_level_calls': 400, 'native_rasterize_to_pixels_calls': 'observed separately; channel chunks',
        'masks_before_GT': 600, 'RGB_predictions': 200, 'semantic_annotations': 41, 'RGB_targets': 50,
        'teacher_inference': INFERENCE, 'teacher_weight': .5, 'evaluation_numerics': NUMERICS,
        'baseline': 'original H3 plus final EMFW q and original head; optimized_q_zero recovery identity',
        'baseline_joint_miou': .9509720470648008, 'RGB': 'each endpoint own newly rendered H3 RGB, scored once',
        'E': 'separate fixed engineering reference; independent 1M/MCMC RGB unaffected by H3 means',
        'comparisons': 'each of three endpoints minus baseline for all three readouts; each joint minus E',
        'bootstrap_repeats': 5000, 'bootstrap_seed': 20260926,
        'internal_seconds': 570, 'external_seconds': 600, 'optimizer_steps': 0,
        'head_adaptation_steps': 0, 'adoption_gate': None,
        'scope': 'fixed development evaluation; no best selection, automatic adoption, or innovation claim'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x') as handle:
        handle.write(payload)


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def rendered_rgb_images(canvas, grid):
    """Teacher sees quantized canvas; delivered RGB warps float BEFORE quantization."""
    import cv2
    require(canvas.dtype == np.float32 and canvas.ndim == 3 and canvas.shape[-1] == 3
            and np.isfinite(canvas).all(), 'Invalid rendered RGB canvas')
    canvas = np.clip(canvas, 0, 1)
    teacher = (canvas*255).round().astype(np.uint8)
    mapped = canvas if grid is None else cv2.remap(canvas, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT)
    return teacher, (mapped[..., ::-1]*255).round().astype(np.uint8)


def validate_endpoint(arrays, baseline, receipt):
    require(set(arrays) == {'base_means', 'means_final', 'actual_delta'}, 'Unexpected endpoint arrays')
    base, final, delta = (arrays[k] for k in ('base_means', 'means_final', 'actual_delta'))
    require(base.dtype == final.dtype == baseline.dtype == np.float32 and delta.dtype == np.float64
            and base.shape == final.shape == delta.shape == baseline.shape and base.shape[1:] == (3,)
            and all(np.isfinite(v).all() for v in (base, final, delta)), 'Endpoint shape/dtype/finiteness')
    require(np.array_equal(base, baseline) and np.array_equal(delta, final.astype(np.float64)-base.astype(np.float64)),
            'Endpoint not bound to exact base or actual displacement')
    require(array_sha(base) == receipt['base_means_sha256'] and array_sha(final) == receipt['final_means_sha256'],
            'Endpoint tensor hash changed')
    return final


def prediction_barrier(records, views, identity):
    names = [v['name'] for v in views]
    require(len(names) == 50 and names == sorted(set(names)) and len(records) == 200
            and {(r['arm'], r['name']) for r in records} == {(a, n) for a in ARMS for n in names},
            'Incomplete fixed four-arm prediction population')
    require(all(set(r['masks']) == set(KINDS) and set(r['soft']) == {'raw', 'scene', 'teacher'} for r in records),
            'Missing mask or soft evidence')
    require(len(identity) == 50 and {r['name'] for r in identity} == set(names)
            and all(r['canvas_rgb_exact'] and r['original_rgb_exact'] and r['teacher_soft_exact']
                    and set(r['mask_exact']) == set(KINDS) and all(r['mask_exact'].values()) for r in identity),
            'Baseline identity must precede any GT access')


@contextlib.contextmanager
def counted_calls(specs, counters):
    """Temporary Python call counters; no tensor replacement or graph mutation."""
    originals = []
    try:
        for obj, attribute, key in specs:
            original = getattr(obj, attribute); originals.append((obj, attribute, original))
            def wrapped(*args, _original=original, _key=key, **kwargs):
                counters[_key] += 1
                return _original(*args, **kwargs)
            setattr(obj, attribute, wrapped)
        yield
    finally:
        for obj, attribute, original in reversed(originals):
            setattr(obj, attribute, original)


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'New data-disk directory required')
    tp, te, tl, audit = [read(TRAIN/n) for n in ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json')]
    require(sha(TRAIN/'plan.json') == TRAIN_PLAN_SHA and sha(TRAIN/'execution_receipt.json') == TRAIN_EXEC_SHA,
            'Unexpected completed training')
    require(te['status'] == tl['status'] == 'completed' and tl['natural_completion'] and tl['exit_code'] == 0
            and tl['plan_sha256'] == te['plan_sha256'] == TRAIN_PLAN_SHA and tl['execution_receipt_sha256'] == TRAIN_EXEC_SHA
            and audit['status'] == 'passed' and audit['plan_sha256'] == TRAIN_PLAN_SHA
            and audit['execution_receipt_sha256'] == TRAIN_EXEC_SHA, 'Training/audit incomplete')
    rp, re, rl, ra = [read(RECOVERY/n) for n in ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json')]
    require(sha(RECOVERY/'plan.json') == RECOVERY_PLAN_SHA and re['status'] == rl['status'] == 'completed'
            and rl['natural_completion'] and rl['exit_code'] == 0 and ra['status'] == 'passed'
            and rl['execution_receipt_sha256'] == sha(RECOVERY/'execution_receipt.json'), 'Recovery reference incomplete')
    require(rp['base_checkpoint'] == tp['base_checkpoint'] and rp['manifest'] == tp['manifest']
            and rp['q_delta']['path'] == tp['q_delta'] and rp['optimized_q_renderer_sha256'] == te['q_renderer_sha256'],
            'Base/grid/q identity differs')
    source = Path(rp['source_snapshot'])
    require(tree(source) == rp['source_hashes'] and tree(Path(tp['source_snapshot'])) == tp['source_hashes'], 'Frozen source changed')
    shared = ['model', 'train', 'refinement', 'depth_moments', 'coordinates', 'partition_rasterizer']
    require(all(rp['source_hashes']['bridge_rgs/'+n+'.py'] == tp['source_hashes']['bridge_rgs/'+n+'.py'] for n in shared),
            'Training/evaluation render implementation differs')
    inputs = {}
    def bind(path, expected=None):
        path = str(Path(path).resolve()); digest = sha(path)
        require(expected is None or expected == digest, 'Dependency changed: '+path)
        inputs[path] = digest; return {'path': path, 'sha256': digest}
    for root in (TRAIN, RECOVERY):
        for n in ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json'):
            bind(root/n)
    for k in ('base_checkpoint', 'manifest'):
        bind(rp[k], rp['input_hashes'][rp[k]])
    bind(rp['q_delta']['path'], rp['q_delta']['sha256']); bind(ROOT/'uv.lock', rp['input_hashes'][str(ROOT/'uv.lock')])
    old = read(rp['prior_plan_path']); prior = read(rp['prior_receipt_path'])
    bind(rp['prior_plan_path'], rp['input_hashes'][rp['prior_plan_path']]); bind(rp['prior_receipt_path'], rp['input_hashes'][rp['prior_receipt_path']])
    e_record = bind(rp['prior_e_metrics'], rp['input_hashes'][rp['prior_e_metrics']])
    predictions_path = RECOVERY/'evaluation/predictions_receipt.json'; bind(predictions_path)
    previous = read(predictions_path)
    require(previous['status'] == 'all_600_masks_before_GT', 'Incomplete previous prediction evidence')
    old_zero = {r['name']: r for r in previous['records'] if r['arm'] == 'optimized_q_zero'}
    views = sorted(old['views'], key=lambda v: v['name'])
    require(len(views) == len(old_zero) == 50 and sum(v['source_annotation_path'] is not None for v in views) == 41,
            'Wrong fixed original-grid views')
    old_h3 = {r['name']: r for r in prior['predictions'] if r['arm'] == 'A'}
    references = {}
    for v in views:
        name = v['name']; zero = old_zero[name]
        refs = {'masks': zero['masks'], 'teacher_soft': zero['teacher_soft_canvas'],
                'canvas_rgb': prior['h3_canvases'][name]['canvas_rgb'],
                'original_rgb': {'path': old_h3[name]['rgb'], 'sha256': old_h3[name]['rgb_sha256']},
                'boundary': prior['h3_canvases'][name]['boundary']}
        for record in [*refs['masks'].values(), refs['teacher_soft'], refs['canvas_rgb'], refs['original_rgb']]:
            bind(record['path'], record['sha256'])
        references[name] = refs
    old_metrics = {}
    evaluation = read(RECOVERY/'evaluation/execution_receipt.json'); bind(RECOVERY/'evaluation/execution_receipt.json')
    for kind in KINDS:
        record = evaluation['metrics']['optimized_q_zero_'+kind]
        old_metrics[kind] = bind(record['path'], record['sha256'])
    require(read(old_metrics['joint']['path'])['miou_all'] == SPEC['baseline_joint_miou'], 'Wrong zero-step baseline')
    teacher = old['teachers']['old']; bind(teacher['path'], teacher['sha256'])
    require(teacher['sha256'] == '00f5b84ac9a56c39512c5b8e43f70110397923feea1a2b4c78bdffd1f5524bff', 'Wrong old H+')
    backbone = teacher['provenance']; model_dir = Path(backbone['model_dir'])
    bind(model_dir/'config.json', backbone['model_config_sha256'])
    for name, digest in (backbone['model_weights_sha256'] | backbone.get('model_index_sha256', {})).items():
        bind(model_dir/name, digest)
    # Require pre-existing scoring weights; no download/install in the evaluation.
    distribution = importlib.metadata.distribution('lpips')
    bind(distribution.locate_file('lpips/weights/v0.1/alex.pth'))
    bind(Path.home()/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')
    arms = {}
    for arm in ARMS[1:]:
        path = TRAIN/arm/'training_receipt.json'; rec = read(path)
        require(rec['status'] == 'completed' and rec['arm'] == arm and rec['attempts'] == 148
                and rec['plan_sha256'] == TRAIN_PLAN_SHA and rec['q_renderer_sha256'] == te['q_renderer_sha256']
                and rec['frozen_state_hashes'] == te['frozen_state_hashes'], 'Endpoint contract mismatch')
        arms[arm] = {'receipt': bind(path), 'means_delta': bind(TRAIN/arm/'means_delta.npz', rec['means_delta_sha256'])}
    snapshot = output/'source_snapshot'
    shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in (Path(__file__), ROOT/'tests/test_continuous_geometry_evaluation.py', ROOT/'docs/continuous_geometry_evaluation_protocol.md'):
        shutil.copy2(path, snapshot/path.name)
    plan = {k: rp[k] for k in ('base_checkpoint', 'manifest', 'q_delta', 'optimized_q_renderer_sha256',
            'expected_gsplat_binary', 'installed_sources', 'runtime_versions', 'environment')}
    plan.update(specification=SPEC, numerics=NUMERICS, output=str(output), source_snapshot=str(snapshot),
        source_hashes=tree(snapshot), inherited_source_hashes=rp['source_hashes'], input_hashes=inputs,
        training_plan={'path': str(TRAIN/'plan.json'), 'sha256': TRAIN_PLAN_SHA},
        training_audit={'path': str(TRAIN/'independent_cpu_review.json'), 'sha256': sha(TRAIN/'independent_cpu_review.json')},
        training_frozen_state_hashes=te['frozen_state_hashes'], base_means_sha256=te['base_means_sha256'],
        arms=arms, views=views, baseline_references=references, baseline_metrics=old_metrics,
        teacher={'path': teacher['path'], 'sha256': teacher['sha256']}, reference_E=e_record,
        evaluation_family=old['evaluation_family'], scoring_protocol=old['scoring_protocol'],
        reference_fingerprint=old['reference_fingerprint'], deferred_GT_expected_only=True,
        prepare_GT_payload_reads=0, prepare_checkpoint_loads=0, status='prepared_pending_root_execution')
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json'),
                      'sources': len(plan['source_hashes']), 'inputs': len(inputs)}))


def verify(plan):
    require(plan['specification'] == SPEC and tree(Path(plan['source_snapshot'])) == plan['source_hashes'], 'Source/spec changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Frozen entry required')
    require(all(plan['source_hashes'].get(k) == v for k, v in plan['inherited_source_hashes'].items()), 'Recovery source altered')
    binary = plan['expected_gsplat_binary']
    for path, digest in {**plan['input_hashes'], **plan['installed_sources'], binary['path']: binary['sha256']}.items():
        require(sha(path) == digest, 'Input changed: '+path)
    for k, v in plan['runtime_versions'].items():
        require(importlib.metadata.version(k) == v, 'Runtime version mismatch')
    for k, v in plan['environment'].items():
        require(os.environ.get(k) == v, 'Execution environment mismatch: '+k)


def evaluate(plan, report, modules):
    import cv2
    import gsplat
    import torch

    from bridge_rgs.direct_q_render import render_direct_q
    from bridge_rgs.render_ensemble import FIXED_INFERENCE, FixedRenderedTeacher
    from bridge_rgs.teacher import predict_image

    producer = modules['train_direct_q_head_matched']; helper = modules['evaluate_projective_deck_pooling']
    official = modules['bridge_rgs.official_evaluate']; compare = modules['compare_official_evaluations']
    adapter = modules['preflight_simplex_scene']; partition = modules['bridge_rgs.partition_rasterizer']
    rendering = importlib.import_module('gsplat.rendering')
    output = Path(plan['output'])/'evaluation'; output.mkdir()
    views = plan['views']; records = []; identities = []; counts = report['counts']
    torch.cuda.reset_peak_memory_stats()
    report['memory_scope'] = 'peak allocated since immediately before field/teacher load; includes inference and LPIPS scoring'
    report['arm_sources'] = {arm: {'base_checkpoint': {'path': plan['base_checkpoint'],
        'sha256': plan['input_hashes'][plan['base_checkpoint']]},
        'means_delta': None if arm == 'baseline' else plan['arms'][arm]['means_delta'],
        'q': plan['q_delta'], 'head': 'unchanged original H3 head', 'teacher': plan['teacher'],
        'teacher_input': 'this arm current scene render; quantized legacy canvas RGB'} for arm in ARMS}
    sources = [{k: v[k] for k in ('camera', 'source_image_sha256', 'source_annotation_sha256', 'rasterized_mask_sha256')} for v in views]
    require(official.official_fingerprint(sources) == plan['reference_fingerprint'], 'Population fingerprint changed')
    scene, state, _ = producer.load_base(plan)
    try:
        teacher = FixedRenderedTeacher(plan['teacher']['path'], plan['base_checkpoint'], state)
    finally:
        del scene, state
    require(FIXED_INFERENCE == INFERENCE, 'Teacher inference changed')
    teacher_versions = {k: v._version for k, v in teacher.model.state_dict().items()}
    def backbone_hook(*_):
        counts['teacher_backbone_forwards'] += 1
    hook = teacher.model.backbone.register_forward_hook(backbone_hook)
    report['restoration'] = {}; by_record = {}
    try:
        for arm in ARMS:
            scene, state, _ = producer.load_base(plan); original = adapter.capture_scene(scene)
            original_means = scene.splats['means'].detach().clone()
            try:
                require(adapter.tensor_hash(original_means) == plan['base_means_sha256'], 'Wrong initial means row order')
                require({k: adapter.tensor_hash(v) for k, v in scene.state_dict().items() if k != 'splats.means'}
                        == plan['training_frozen_state_hashes'], 'Non-means state differs from training')
                if arm != 'baseline':
                    rec = read(plan['arms'][arm]['receipt']['path'])
                    with np.load(plan['arms'][arm]['means_delta']['path'], allow_pickle=False) as archive:
                        final = validate_endpoint({k: archive[k] for k in archive.files}, original_means.cpu().numpy(), rec)
                    with torch.no_grad():
                        scene.splats['means'].copy_(torch.from_numpy(final).to(original_means.device))
                scene.eval().requires_grad_(False)
                q = producer.load_q(scene, 'optimized_q_zero', plan)
                before = {k: adapter.tensor_hash(v) for k, v in scene.state_dict().items()}
                q_hash = adapter.tensor_hash(q)
                def head_hook(*_):
                    counts['head_calls'] += 1
                head_handle = scene.refiner.register_forward_hook(head_hook)
                try:
                    specs = [(scene, 'render', 'scene_calls'), (gsplat, 'rasterization', 'native_gsplat_high_level_calls'),
                             (rendering, 'rasterize_to_pixels', 'native_rasterize_to_pixels_calls'),
                             (partition, 'rasterize_semantic_partition', 'direct_q_shader_calls')]
                    with torch.no_grad(), counted_calls(specs, counts):
                        for view in views:
                            name, camera = view['name'], view['camera']; ref = plan['baseline_references'][name]
                            K, w, h, back = official.distortion_render_grid(camera['K'], camera['distortion'],
                                camera['width'], camera['height'], 'legacy_mixed_v1')
                            require([w, h] == ref['boundary']['canvas']
                                    and np.array_equal(K, np.asarray(ref['boundary']['canvas_K'], np.float32)), 'Canvas mapping changed')
                            result = render_direct_q(scene, q, torch.tensor(K, device='cuda'),
                                torch.tensor(camera['w2c'], dtype=torch.float32, device='cuda'), w, h)
                            canvas = result['rgb'].clamp(0, 1).cpu().numpy()
                            canvas_u8, original_bgr = rendered_rgb_images(canvas, back)
                            canvas_record = helper.save_png(output/arm/'canvas_rgb'/name, canvas_u8[..., ::-1],
                                ref['canvas_rgb'] if arm == 'baseline' else None)
                            rgb_record = helper.save_png(output/arm/'rgb'/name, original_bgr,
                                ref['original_rgb'] if arm == 'baseline' else None)
                            counts['teacher_predict_calls'] += 1
                            probabilities, _ = predict_image(teacher.model, canvas_u8, **INFERENCE)
                            tp = np.ascontiguousarray(probabilities.transpose(1, 2, 0))
                            raw, final = result['p3d'].cpu().numpy(), result['probabilities'].cpu().numpy()
                            soft = {'raw': raw, 'scene': final, 'teacher': tp}
                            joint = (final+tp)*np.float32(.5)
                            masks = {kind: helper.save_png(output/arm/(kind+'_mask')/name,
                                helper.warp_probabilities(p, back).argmax(-1).astype(np.uint8),
                                ref['masks'][kind] if arm == 'baseline' else None)
                                for kind, p in (('raw', raw), ('scene', final), ('joint', joint))}
                            saved = {kind: helper.save_soft(output/arm/(kind+'_soft')/(Path(name).stem+'.npy'), p)
                                     for kind, p in soft.items()}
                            if arm == 'baseline':
                                old_teacher = np.load(ref['teacher_soft']['path'], allow_pickle=False)
                                require(np.array_equal(old_teacher, tp), 'Fresh baseline teacher differs from historical inference')
                                identities.append({'name': name, 'canvas_rgb_exact': True, 'original_rgb_exact': True,
                                                   'teacher_soft_exact': True, 'mask_exact': dict.fromkeys(KINDS, True)})
                            record = {'arm': arm, 'name': name, 'rgb': rgb_record, 'teacher_input_rgb': canvas_record,
                                      'soft': saved, 'masks': masks, 'teacher_recomputed': True}
                            records.append(record); by_record[arm, name] = record
                            del result
                finally:
                    head_handle.remove()
                require({k: adapter.tensor_hash(v) for k, v in scene.state_dict().items()} == before
                        and adapter.tensor_hash(q) == q_hash, 'Inference changed endpoint or q')
                del q
            finally:
                with torch.no_grad():
                    scene.splats['means'].copy_(original_means)
                restored = adapter.restore_scene(scene, original); report['restoration'][arm] = restored
                require(restored['state_exact'] and restored['flags_modes_gradients_restored'], 'Scene restore failed')
                del scene, state, original_means
                torch.cuda.empty_cache()
        prediction_barrier(records, views, identities)
        for key in ('scene_calls', 'direct_q_shader_calls', 'head_calls', 'teacher_predict_calls',
                    'teacher_backbone_forwards', 'native_gsplat_high_level_calls'):
            require(counts[key] == SPEC[key], 'Wrong fixed work count: '+key)
        teacher.verify_inputs()
        require({k: v._version for k, v in teacher.model.state_dict().items()} == teacher_versions, 'Teacher was mutated')
    finally:
        hook.remove()
    prediction_receipt = {'status': 'all_600_masks_and_200_RGB_before_GT', 'records': records,
        'baseline_identity': identities, 'baseline_joint_miou_bound': SPEC['baseline_joint_miou'],
        'counts': dict(counts), 'annotation_payload_reads': 0, 'GT_RGB_payload_reads': 0,
        'finished_utc': datetime.now(UTC).isoformat()}
    write(output/'predictions_receipt.json', prediction_receipt)
    del teacher; torch.cuda.empty_cache()
    perceptual = official._lpips('cuda')
    rows = {(a, k): [] for a in ARMS for k in KINDS}
    report['scoring_started_utc'] = datetime.now(UTC).isoformat()
    for view in views:
        name, camera = view['name'], view['camera']
        payload = Path(view['source_image_path']).read_bytes(); counts['GT_RGB_payload_reads'] += 1
        require(hashlib.sha256(payload).hexdigest() == view['source_image_sha256'], 'RGB GT changed')
        truth_rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        truth = None
        if view['source_annotation_path'] is not None:
            payload = Path(view['source_annotation_path']).read_bytes(); counts['annotation_payload_reads'] += 1
            require(hashlib.sha256(payload).hexdigest() == view['source_annotation_sha256'], 'Annotation changed')
            truth = official.rasterize_official_annotation(payload, camera['width'], camera['height'])
            require(array_sha(truth) == view['rasterized_mask_sha256'], 'GT rasterization changed')
        for arm in ARMS:
            record = by_record[arm, name]
            rgb = cv2.cvtColor(cv2.imread(record['rgb']['path'], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            joint_mask = cv2.imread(record['masks']['joint']['path'], cv2.IMREAD_UNCHANGED)
            score = official.score_official_arrays(rgb, joint_mask, truth_rgb, None, perceptual, device='cuda')
            counts['RGB_scoring_calls'] += 1
            require(set(score) == {'psnr', 'ssim', 'lpips', 'rgb_pixels'}, 'Unexpected RGB score schema')
            for kind in KINDS:
                row = {'name': name, 'width': camera['width'], 'height': camera['height'], **score}
                if truth is not None:
                    pred = joint_mask if kind == 'joint' else cv2.imread(record['masks'][kind]['path'], cv2.IMREAD_UNCHANGED)
                    valid = truth != 255
                    cm = np.bincount(5*truth[valid].astype(np.int64)+pred[valid], minlength=25).reshape(5, 5)
                    row.update(confusion_matrix=cm.tolist(), semantic_pixels=int(valid.sum()), semantic_ignore_pixels=int((~valid).sum()))
                    counts['confusion_matrices'] += 1
                rows[arm, kind].append(row)
    require(counts['GT_RGB_payload_reads'] == 50 and counts['annotation_payload_reads'] == 41
            and counts['RGB_scoring_calls'] == 200 and counts['confusion_matrices'] == 492, 'Wrong scoring work count')
    metrics = {}; metric_records = {}; comparisons = {}
    for (arm, kind), per_view in rows.items():
        pooled = sum(np.asarray(r['confusion_matrix'], np.int64) for r in per_view if 'confusion_matrix' in r)
        iou = compare._iou(pooled)
        metric = {'evaluation_family': plan['evaluation_family'], 'scoring_protocol': plan['scoring_protocol'],
            'official_evaluation_fingerprint': plan['reference_fingerprint'], 'validation_views': 50,
            'semantic_validation_views': 41, 'views': per_view, 'confusion_matrix': pooled.tolist(), 'iou': iou.tolist(),
            'miou_all': float(np.nanmean(iou)), 'miou_foreground': float(np.nanmean(iou[1:])),
            **{k: float(np.mean([r[k] for r in per_view])) for k in ('psnr', 'ssim', 'lpips')},
            'inference_protocol': {'protocol': SPEC['protocol'], 'arm': arm, 'readout': kind, 'teacher_weight': .5 if kind == 'joint' else 0,
                'RGB': 'own H3 endpoint RGB; no E score inheritance', 'same_original_head': True, 'same_EMFW_q': True,
                'teacher_fresh_for_each_endpoint_RGB': kind == 'joint'},
            'provenance': {'base_checkpoint_sha256': plan['input_hashes'][plan['base_checkpoint']],
                'means_delta': None if arm == 'baseline' else plan['arms'][arm]['means_delta'], 'q': plan['q_delta'],
                'teacher': plan['teacher'] if kind == 'joint' else None}}
        compare._validate(metric)
        if arm == 'baseline':
            old = read(plan['baseline_metrics'][kind]['path'])
            require(metric['confusion_matrix'] == old['confusion_matrix'], 'Baseline CM must reproduce without choosing new target')
        path = output/arm/(kind+'_official_metrics.json'); write(path, metric)
        metrics[arm, kind] = metric; metric_records[arm+'_'+kind] = {'path': str(path), 'sha256': sha(path)}
    reference_e = read(plan['reference_E']['path']); compare._validate(reference_e)
    for arm in ARMS[1:]:
        for kind in KINDS:
            pair = compare.paired_official_comparison(metrics['baseline', kind], metrics[arm, kind], repeats=5000, seed=20260926)
            pair.update(candidate_arm=arm, reference_arm='baseline', readout=kind, descriptive_only=True)
            path = output/f'paired_{arm}_minus_baseline_{kind}.json'; write(path, pair)
            comparisons[path.stem] = {'path': str(path), 'sha256': sha(path)}
        pair = compare.paired_official_comparison(reference_e, metrics[arm, 'joint'], repeats=5000, seed=20260926)
        pair.update(candidate_arm=arm, reference_arm='E', readout='joint', descriptive_only=True,
                    RGB_scope='candidate own H3 RGB vs E independent 1M/MCMC RGB; not an E RGB update')
        path = output/f'paired_{arm}_minus_E_joint.json'; write(path, pair)
        comparisons[path.stem] = {'path': str(path), 'sha256': sha(path)}
    report.update(metrics=metric_records, comparisons=comparisons, baseline_masks_byte_exact=150,
        baseline_joint_miou=metrics['baseline', 'joint']['miou_all'], adoption_gate=None,
        predictions_receipt={'path': str(output/'predictions_receipt.json'), 'sha256': sha(output/'predictions_receipt.json')},
        prediction_finished_utc=prediction_receipt['finished_utc'])


def execute(path, expected):
    require(sha(path) == expected, 'Wrong plan SHA'); plan = read(path); output = Path(plan['output'])
    require(not any((output/n).exists() for n in ('execution_started.json', 'execution_receipt.json', 'evaluation')), 'No retries/overwrites')
    write(output/'execution_started.json', {'plan_sha256': expected, 'time': time.time()})
    start = time.monotonic(); previous = None; torch = None
    counts = dict.fromkeys(('scene_calls', 'direct_q_shader_calls', 'head_calls', 'teacher_predict_calls', 'teacher_backbone_forwards',
        'native_gsplat_high_level_calls', 'native_rasterize_to_pixels_calls', 'GT_RGB_payload_reads', 'annotation_payload_reads',
        'RGB_scoring_calls', 'confusion_matrices'), 0)
    report = {'status': 'running', 'plan_sha256': expected, 'counts': counts, 'optimizer_steps': 0, 'head_adaptation_steps': 0}
    def expired(*_):
        raise TimeoutError('Fixed 570-second evaluation deadline')
    old_handler = signal.signal(signal.SIGALRM, expired); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan); sys.path.insert(0, plan['source_snapshot'])
        require(not any(k.startswith('bridge_rgs') for k in sys.modules), 'No previously imported live bridge package')
        producer = importlib.import_module('train_direct_q_head_matched')
        report['gpu_before'] = producer.gpu_inventory()
        modules, _ = producer.imports_from_snapshot(plan); modules['train_direct_q_head_matched'] = producer
        torch = modules['bridge_rgs.train'].torch; torch.set_num_threads(4)
        previous = producer.numerical_flags(torch, NUMERICS)
        report['numerics_actual'] = producer.read_numerical_flags(torch)
        require(report['numerics_actual'] == NUMERICS, 'Historical evaluation numerics differ')
        evaluate(plan, report, modules)
        report['actual_imports'] = producer.imports_from_snapshot(plan)[1]
        report['actual_imports']['train_direct_q_head_matched'] = {'path': producer.__file__, 'sha256': sha(producer.__file__)}
        report['actual_gsplat_binary'] = producer.loaded_gsplat_binaries()
        require(plan['expected_gsplat_binary'] in report['actual_gsplat_binary'].values(), 'Wrong loaded binary')
        verify(plan); report.update(status='completed', sources_inputs_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
        if torch is not None and torch.cuda.is_initialized():
            report['peak_cuda_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        if previous is not None:
            producer.numerical_flags(torch, previous)
            report['numerics_restored'] = producer.read_numerical_flags(torch) == previous
        report['elapsed_seconds'] = time.monotonic()-start
        report['timing_scope'] = '200 actual H3 camera renders, fresh teacher inference, saved arrays and scoring; no E RGB field rendering'
        write(output/'execution_receipt.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', type=Path); action.add_argument('--execute', type=Path)
    parser.add_argument('--expected-plan-sha256'); args = parser.parse_args()
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Disable bytecode')
    if args.prepare:
        prepare(args.prepare)
    else:
        execute(args.execute, args.expected_plan_sha256)
