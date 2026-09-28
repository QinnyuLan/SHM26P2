"""One fixed RGB-only endpoint, reusing the audited four-endpoint references."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
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
PRIOR = Path('/mnt/data/SHM2026/runs/continuous_geometry_proposal_evaluation_v1')
TRAIN = Path('/mnt/data/SHM2026/runs/continuous_geometry_rgb_only_v1')
PRIOR_PLAN_SHA = 'c2bd7e3d0eebc9636cb8bf2c4358c2d835fc26c34a5eb420c9fb588bae380686'
PRIOR_EXEC_SHA = '1c1cc5abd3734cd844fd0db24add7abc055542108b04a810bec1eb1acdddcb16'
PRIOR_AUDIT_SHA = '2a70fe7486c0cec62bb4c951805b94ffab1cc2120f08308d687c9f0f14335539'
TRAIN_PLAN_SHA = 'e79a9b9fd155d6f1b2d8b04fe73d8405675bf625e1999d28e9b7f0b75fe23865'
TRAIN_EXEC_SHA = 'ebf2f99e09d64bfd1880ec7f385f467aa17a59c423f16ba8e2ffc8fa162e47c6'
TRAIN_AUDIT_SHA = 'bad79a107f20fe54d7e4a403aaba59dfb963fd96b88f8c877c210f87df91dcb7'
ARMS = ('rgb',)
KINDS = ('raw', 'scene', 'joint')


def inherited_module():
    path = Path(__file__).with_name('evaluate_continuous_geometry_proposals.py')
    if Path(__file__).parent.name != 'source_snapshot':
        path = PRIOR/'source_snapshot/evaluate_continuous_geometry_proposals.py'
    spec = importlib.util.spec_from_file_location('inherited_geometry_evaluator', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


old = inherited_module()
require, read, write, sha = old.require, old.read, old.write, old.sha
tree, array_sha, counted_calls, validate_endpoint = old.tree, old.array_sha, old.counted_calls, old.validate_endpoint
rendered_rgb_images = old.rendered_rgb_images
NUMERICS, INFERENCE = old.NUMERICS, old.INFERENCE
SPEC = {'protocol': 'rgb_geometry_control_evaluation_v1', 'arms': list(ARMS), 'kinds': list(KINDS),
        'scene_calls': 50, 'direct_q_shader_calls': 50, 'head_calls': 50,
        'teacher_predict_calls': 50, 'teacher_backbone_forwards': 700,
        'native_gsplat_high_level_calls': 100, 'masks_before_GT': 150, 'RGB_predictions': 50,
        'semantic_annotations': 41, 'RGB_targets': 50, 'teacher_inference': INFERENCE,
        'teacher_weight': .5, 'evaluation_numerics': NUMERICS,
        'RGB': 'new RGB-only endpoint own H3 RGB; full original scoring',
        'references': 'completed audited four-endpoint evaluation; no reference re-rendering',
        'comparisons': 'rgb minus baseline, and each old joint/pcgrad/trust minus rgb; all three readouts',
        'bootstrap_repeats': 5000, 'bootstrap_seed': 20260926,
        'internal_seconds': 240, 'external_seconds': 300,
        'optimizer_steps': 0, 'head_adaptation_steps': 0, 'adoption_gate': None,
        'scope': 'post-parent matched RGB-only control; fixed endpoint, descriptive, no best selection or new gate'}


def pair_definitions():
    return [('rgb', 'baseline', k) for k in KINDS]+[(a, 'rgb', k) for a in ('joint', 'pcgrad', 'trust') for k in KINDS]


def prediction_barrier(records, views):
    names = [v['name'] for v in views]
    require(len(names) == 50 and names == sorted(set(names)) and len(records) == 50
            and {(r['arm'], r['name']) for r in records} == {('rgb', n) for n in names}, 'Complete50 RGB-only predictions required')
    require(all(set(r['masks']) == set(KINDS) and set(r['soft']) == {'raw', 'scene', 'teacher'}
                and r['teacher_recomputed'] is True for r in records), 'Complete150 masks and soft evidence before GT')


def completed(root, expected_plan, expected_exec, expected_audit):
    plan, execution, launch, audit, audit_launch = [read(root/n) for n in
        ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json', 'independent_audit_launch_receipt.json')]
    require(sha(root/'plan.json') == expected_plan and sha(root/'execution_receipt.json') == expected_exec
            and sha(root/'independent_cpu_review.json') == expected_audit, 'Completed lineage SHA mismatch')
    require(execution['status'] == launch['status'] == audit_launch['status'] == 'completed'
            and launch['natural_completion'] is True and launch['exit_code'] == 0
            and audit_launch['natural_completion'] is True and audit_launch['exit_code'] == 0
            and audit['status'] == 'passed' and audit_launch['audit_sha256'] == expected_audit
            and execution['plan_sha256'] == launch['plan_sha256'] == audit['plan_sha256'] == expected_plan
            and launch['execution_receipt_sha256'] == audit['execution_receipt_sha256'] == expected_exec,
            'Natural training/evaluation and independent audit required')
    return plan, execution


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    pp, pe = completed(PRIOR, PRIOR_PLAN_SHA, PRIOR_EXEC_SHA, PRIOR_AUDIT_SHA)
    tp, te = completed(TRAIN, TRAIN_PLAN_SHA, TRAIN_EXEC_SHA, TRAIN_AUDIT_SHA)
    require(pp['base_checkpoint'] == tp['base_checkpoint'] and pp['manifest'] == tp['manifest']
            and pp['q_delta']['path'] == tp['q_delta'] and pp['optimized_q_renderer_sha256'] == te['q_renderer_sha256']
            and pp['base_means_sha256'] == te['base_means_sha256']
            and pp['training_frozen_state_hashes'] == te['frozen_state_hashes'], 'Matched field/q/unchanged state required')
    source = Path(pp['source_snapshot'])
    require(tree(source) == pp['source_hashes'] and tree(Path(tp['source_snapshot'])) == tp['source_hashes'], 'Bound source changed')
    inputs = dict(pp['input_hashes'])
    def bind(path, expected=None):
        path = str(Path(path).resolve()); digest = sha(path)
        require(expected is None or expected == digest, 'Bound artifact changed: '+path)
        inputs[path] = digest; return {'path': path, 'sha256': digest}
    for root in (PRIOR, TRAIN):
        for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json', 'independent_cpu_review.json', 'independent_audit_launch_receipt.json'):
            bind(root/name)
    # Bind every previously scored prediction without inferring its pixels again.
    prior_prediction = pe['predictions_receipt']; bind(prior_prediction['path'], prior_prediction['sha256'])
    for record in read(prior_prediction['path'])['records']:
        for item in [record['rgb'], record['teacher_input_rgb'], *record['masks'].values(), *record['soft'].values()]:
            bind(item['path'], item['sha256'])
    for item in [*pe['metrics'].values(), *pe['comparisons'].values()]:
        bind(item['path'], item['sha256'])
    stage = read(TRAIN/'rgb/training_receipt.json')
    require(stage['status'] == 'completed' and stage['arm'] == 'rgb' and stage['attempts'] == stage['accepted'] == 148
            and stage['plan_sha256'] == TRAIN_PLAN_SHA and stage['q_renderer_sha256'] == te['q_renderer_sha256']
            and stage['frozen_state_hashes'] == te['frozen_state_hashes'], 'Wrong RGB-only fixed endpoint')
    arm = {'receipt': bind(TRAIN/'rgb/training_receipt.json'),
           'means_delta': bind(TRAIN/'rgb/means_delta.npz', stage['means_delta_sha256'])}
    for path, digest in inputs.items():
        require(sha(path) == digest, 'Inherited input changed: '+path)
    snapshot = output/'source_snapshot'
    shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in (Path(__file__), ROOT/'tests/test_rgb_geometry_evaluation.py', ROOT/'docs/rgb_geometry_evaluation_protocol.md'):
        shutil.copy2(path, snapshot/path.name)
    plan = {k: pp[k] for k in ('base_checkpoint', 'manifest', 'q_delta', 'optimized_q_renderer_sha256',
        'expected_gsplat_binary', 'installed_sources', 'runtime_versions', 'environment', 'numerics', 'teacher',
        'views', 'baseline_references', 'base_means_sha256', 'training_frozen_state_hashes',
        'evaluation_family', 'scoring_protocol', 'reference_fingerprint')}
    plan.update(specification=SPEC, output=str(output), source_snapshot=str(snapshot), source_hashes=tree(snapshot),
        inherited_source_hashes=pp['source_hashes'], input_hashes=inputs, arms={'rgb': arm},
        training_plan=bind(TRAIN/'plan.json', TRAIN_PLAN_SHA), training_audit=bind(TRAIN/'independent_cpu_review.json', TRAIN_AUDIT_SHA),
        prior_evaluation=bind(PRIOR/'execution_receipt.json', PRIOR_EXEC_SHA),
        prior_audit=bind(PRIOR/'independent_cpu_review.json', PRIOR_AUDIT_SHA), reference_metrics=pe['metrics'],
        prior_baseline_identity={'masks': pe['baseline_masks_byte_exact'], 'prediction_receipt': prior_prediction},
        status='prepared_pending_root_launch', prepare_GT_payload_reads=0, prepare_checkpoint_loads=0)
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json'),
                      'sources': len(plan['source_hashes']), 'inputs': len(inputs)}))


def verify(plan):
    require(plan['specification'] == SPEC and tree(plan['source_snapshot']) == plan['source_hashes'], 'Source/spec changed')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Frozen entry required')
    require(all(plan['source_hashes'].get(k) == v for k, v in plan['inherited_source_hashes'].items()), 'Prior source changed')
    binary = plan['expected_gsplat_binary']
    for path, digest in {**plan['input_hashes'], **plan['installed_sources'], binary['path']: binary['sha256']}.items():
        require(sha(path) == digest, 'Input changed: '+path)
    for key, value in plan['runtime_versions'].items():
        require(importlib.metadata.version(key) == value, 'Runtime mismatch')
    for key, value in plan['environment'].items():
        require(os.environ.get(key) == value, 'Environment mismatch: '+key)


# The rendering/scoring body is a thin single-arm fork of the frozen evaluator.
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
    views = plan['views']; records = []; counts = report['counts']
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
                                None)
                            rgb_record = helper.save_png(output/arm/'rgb'/name, original_bgr,
                                None)
                            counts['teacher_predict_calls'] += 1
                            probabilities, _ = predict_image(teacher.model, canvas_u8, **INFERENCE)
                            tp = np.ascontiguousarray(probabilities.transpose(1, 2, 0))
                            raw, final = result['p3d'].cpu().numpy(), result['probabilities'].cpu().numpy()
                            soft = {'raw': raw, 'scene': final, 'teacher': tp}
                            joint = (final+tp)*np.float32(.5)
                            masks = {kind: helper.save_png(output/arm/(kind+'_mask')/name,
                                helper.warp_probabilities(p, back).argmax(-1).astype(np.uint8),
                                None)
                                for kind, p in (('raw', raw), ('scene', final), ('joint', joint))}
                            saved = {kind: helper.save_soft(output/arm/(kind+'_soft')/(Path(name).stem+'.npy'), p)
                                     for kind, p in soft.items()}
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
        prediction_barrier(records, views)
        for key in ('scene_calls', 'direct_q_shader_calls', 'head_calls', 'teacher_predict_calls',
                    'teacher_backbone_forwards', 'native_gsplat_high_level_calls'):
            require(counts[key] == SPEC[key], 'Wrong fixed work count: '+key)
        teacher.verify_inputs()
        require({k: v._version for k, v in teacher.model.state_dict().items()} == teacher_versions, 'Teacher was mutated')
    finally:
        hook.remove()
    prediction_receipt = {'status': 'all_150_masks_and_50_RGB_before_GT', 'records': records,
        'prior_baseline_identity': plan['prior_baseline_identity'], 'reference_evaluation': plan['prior_evaluation'],
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
            and counts['RGB_scoring_calls'] == 50 and counts['confusion_matrices'] == 123, 'Wrong scoring work count')
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
        path = output/arm/(kind+'_official_metrics.json'); write(path, metric)
        metrics[arm, kind] = metric; metric_records[arm+'_'+kind] = {'path': str(path), 'sha256': sha(path)}
    for key, record in plan['reference_metrics'].items():
        value = read(record['path']); compare._validate(value)
        require(value['official_evaluation_fingerprint'] == plan['reference_fingerprint'], 'Reference grid differs')
        arm, kind = key.rsplit('_', 1); metrics[arm, kind] = value
    for candidate, reference, kind in pair_definitions():
        pair = compare.paired_official_comparison(metrics[reference, kind], metrics[candidate, kind], repeats=5000, seed=20260926)
        pair.update(candidate_arm=candidate, reference_arm=reference, readout=kind, descriptive_only=True,
                    reference_reuse='completed audited four-arm evaluation; no prior endpoint re-rendering')
        path = output/f'paired_{candidate}_minus_{reference}_{kind}.json'; write(path, pair)
        comparisons[path.stem] = {'path': str(path), 'sha256': sha(path)}
    report.update(metrics=metric_records, comparisons=comparisons, adoption_gate=None,
        prior_evaluation=plan['prior_evaluation'], prior_audit=plan['prior_audit'],
        baseline_identity_reused=plan['prior_baseline_identity'], new_baseline_renders=0,
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
        raise TimeoutError('Fixed 240-second evaluation deadline')
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
        report['actual_imports']['inherited_geometry_evaluator'] = {'path': old.__file__, 'sha256': sha(old.__file__)}
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
        report['timing_scope'] = '50 new RGB-only H3 camera renders, fresh teacher inference and RGB scoring; prior four endpoints reused without rendering'
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
