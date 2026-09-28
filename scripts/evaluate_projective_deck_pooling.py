"""Fixed endpoint evaluation of four refiner deltas; old E RGB/teacher are explicit caches."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

ARMS = ('original', 'per_camera', 'projective', 'wrong')
KINDS = ('joint', 'scene', 'raw')
BASE_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
TEACHER_SHA = '00f5b84ac9a56c39512c5b8e43f70110397923feea1a2b4c78bdffd1f5524bff'
FORMAT = 'projective_deck_refiner_delta_v1'
THRESHOLDS = {'original': .0015, 'per_camera': .001, 'wrong': .001, 'E': .002}
INFERENCE = {
    'protocol': 'projective_deck_refiner_endpoint_v1', 'scene_calls': 200,
    'teacher_calls': 0, 'new_composite_rgb_renders': 0,
    'teacher_cache': 'old A H+ float32 soft canvas after per-view new H3 canvas/original RGB PNG byte identity',
    'joint': '0.5 new H3 probabilities + 0.5 old A teacher probabilities in legacy canvas; warp once then argmax',
    'rgb': 'Existing completed E delivered PNG references and corresponding scores; no new E RGB or RGB GT scoring',
    'barrier': 'All 200 joint, 200 scene-final and 200 raw masks saved before 41 annotation payloads',
    'scope': 'New semantic inference with verified caches; timing is not full camera-only system inference FPS',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(UTC).isoformat()


def write(path, value):
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    content = json.dumps(value, indent=2, allow_nan=False, default=convert)+'\n'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x') as stream:
        stream.write(content)


def require_record(record):
    require(sha(record['path']) == record['sha256'], f"Changed recorded artifact: {record['path']}")


def save_png(path, pixels, reference=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and pixels.dtype == np.uint8, 'New uint8 PNG required')
    require(cv2.imwrite(str(path), pixels), 'PNG write failed')
    record = {'path': str(path), 'sha256': sha(path)}
    if reference is not None:
        require_record(reference)
        require(record['sha256'] == reference['sha256'] and path.read_bytes() == Path(reference['path']).read_bytes(),
                'New rendered RGB bytes differ from teacher cache source')
    return record


def save_soft(path, p):
    require(p.dtype == np.float32 and p.ndim == 3 and p.shape[-1] == 5
            and np.isfinite(p).all() and np.all(p >= 0)
            and np.allclose(p.sum(-1), 1, atol=2e-5, rtol=0), 'Invalid HWC five-class probability')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.save(f, p, allow_pickle=False)
    return {'path': str(path), 'sha256': sha(path), 'shape': list(p.shape), 'dtype': 'float32'}


def warp_probabilities(p, grid):
    p = np.ascontiguousarray(p, dtype=np.float32)
    require(p.ndim == 3 and p.shape[-1] == 5 and np.isfinite(p).all() and (p >= 0).all(), 'Invalid soft canvas')
    mapped = p if grid is None else cv2.remap(p, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    require(np.isfinite(mapped).all() and (mapped.sum(-1) > 0).all(), 'Empty original-grid probability footprint')
    return mapped


def prediction_barrier(records, views):
    names = {v['name'] for v in views}
    require(len(names) == 50 and len(records) == 200, 'Need four times all fifty predictions')
    require({(r['arm'], r['name']) for r in records} == {(a,n) for a in ARMS for n in names}, 'Prediction set differs')
    for r in records:
        require(set(r['masks']) == set(KINDS) and r['teacher_input_rgb_exact'], 'Incomplete prediction/identity contract')
        for record in r['masks'].values():
            require_record(record)
        require_record(r['soft_canvas'])


def adoption_clauses(pairs):
    require(set(pairs) == set(THRESHOLDS), 'Need three continuation controls and old E')
    clauses = {}
    for reference, threshold in THRESHOLDS.items():
        m = pairs[reference]['metrics']; all5 = m['miou_all']
        clauses[f'{reference}_miou_gain'] = all5['difference'] >= threshold
        clauses[f'{reference}_miou_ci_lower_positive'] = all5['paired_view_bootstrap_95_interval'][0] > 0
        for name in ('background', 'deck', 'stay_cable', 'tower', 'foundation'):
            clauses[f'{reference}_{name}_guard'] = m[name+'_iou']['difference'] >= (-.001 if name == 'stay_cable' else -.002)
    return {k: bool(v) for k,v in clauses.items()}


def verify_sources(plan, *, skip_targets):
    snapshot = Path(plan['source_snapshot']).resolve()
    actual = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    require(actual == plan['source_hashes'], 'Frozen source tree differs')
    for path, expected in plan['input_hashes'].items():
        if str(Path(path).resolve()) not in skip_targets:
            require(sha(path) == expected, f'Input changed: {path}')
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Use frozen evaluator only')


def modules_from_snapshot(plan):
    modules = {name: importlib.import_module(name) for name in (
        'bridge_rgs.train', 'bridge_rgs.model', 'bridge_rgs.refinement', 'bridge_rgs.depth_moments',
        'bridge_rgs.coordinates', 'bridge_rgs.evaluate', 'bridge_rgs.official_evaluate',
        'bridge_rgs.projective_pooling', 'bridge_rgs.structure_axes', 'compare_official_evaluations')}
    records = {}
    snapshot = Path(plan['source_snapshot']).resolve()
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(snapshot), f'Unfrozen actual import: {name}')
        digest = sha(path)
        require(digest == plan['source_hashes'][str(path.relative_to(snapshot))], f'Changed import: {name}')
        records[name] = {'path': str(path), 'sha256': digest}
    return modules, records


def validate_delta(delta, arm, plan, state):
    require(delta.get('format') == FORMAT and delta.get('base_checkpoint_sha256') == BASE_SHA
            and delta.get('mode') == arm and delta.get('step') == 2000 and 'model' not in delta,
            'Wrong refiner delta format/base/mode/endpoint')
    require(delta.get('plan_sha256') == sha(Path(plan['output'])/'plan.json')
            and delta.get('specification') == plan['specification'], 'Delta plan/adapter specification differs')
    require(state['feature_dim'] == 16 and state['sh_degree'] == 3
            and state['refiner_config'] == {'type':'multiscale','channels':64,'residual_bound':6.,
                                          'context':'pyramid_strip','depth_moments':'cross'}, 'Wrong H3 architecture')
    cameras = state['training_cameras'].detach().cpu().contiguous().numpy()
    require(delta.get('refiner_config') == state['refiner_config']
            and delta.get('pixel_protocol') == 'legacy_mixed_v1'
            and delta.get('manifest_sha256') == sha(plan['manifest'])
            and delta.get('training_camera_sha256') == hashlib.sha256(cameras.tobytes()).hexdigest(),
            'Delta architecture/profile/manifest/camera metadata differs')
    require(bool(delta.get('refiner_state')) and all(
        np.isfinite(value.detach().cpu().numpy()).all() for value in delta['refiner_state'].values()),
        'Nonfinite or empty refiner state')


def evaluate(plan):
    import torch
    root = Path(plan['output']); output = root/'evaluation'
    require(not output.exists(), 'Never overwrite or retry endpoint evaluation')
    require(plan['base_checkpoint_sha256'] == BASE_SHA and sha(plan['base_checkpoint']) == BASE_SHA, 'Wrong H3 base')
    old = read(plan['prior_plan_path']); prior = read(plan['prior_receipt_path'])
    previous_predictions = read(plan['prior_predictions_path'])
    old_e = read(plan['prior_e_metrics'])
    require(prior['status'] == 'completed' and prior['bound_inputs_and_sources_unchanged']
            and prior['plan_sha256'] == sha(plan['prior_plan_path']), 'Prior cache run is incomplete/unbound')
    require(prior['metrics_sha256']['E'] == sha(plan['prior_e_metrics']), 'Old E metric bytes differ')
    require(previous_predictions['predictions'] == prior['predictions'], 'Prior prediction receipt differs')
    require(old['teachers']['old']['sha256'] == TEACHER_SHA
            and old['components']['selected']['checkpoint_sha256'] == BASE_SHA
            and old['components']['selected']['profile'] == 'legacy_mixed_v1', 'Wrong prior scene/teacher')
    views = old['views']; names = [v['name'] for v in views]
    require(len(names) == len(set(names)) == 50 and sum(v['source_annotation_path'] is not None for v in views) == 41,
            'Wrong fixed fifty/forty-one population')
    # Provenance-only pixel hashes are deferred until after every new prediction exists.
    annotations = {str(Path(v['source_annotation_path']).resolve()) for v in views if v['source_annotation_path']}
    raw_gt_rgb = {str(Path(v['source_image_path']).resolve()) for v in views}
    verify_sources(plan, skip_targets=annotations | raw_gt_rgb)
    modules, imports = modules_from_snapshot(plan)
    official = modules['bridge_rgs.official_evaluate']; compare = modules['compare_official_evaluations']
    attach = modules['bridge_rgs.projective_pooling'].attach_projective_context
    require(official.official_fingerprint([{k:v[k] for k in ('camera','source_image_sha256',
        'source_annotation_sha256','rasterized_mask_sha256')} for v in views]) == old['reference_fingerprint'], 'Wrong fingerprint')
    require(old_e['official_evaluation_fingerprint'] == old['reference_fingerprint'], 'E grid differs')
    compare._validate(old_e)
    by_old = {(r['arm'],r['name']):r for r in prior['predictions']}
    rgb_rows = {r['name']: {k:r[k] for k in ('name','width','height','rgb_pixels','psnr','ssim','lpips')}
                for r in old_e['views']}
    require(set(rgb_rows) == set(names), 'E RGB rows differ')
    report = {'status':'running','plan_sha256':sha(root/'plan.json'),'started_utc':utc(),
              'inference_protocol':INFERENCE,'predictions':[],'scene_calls':0,'teacher_calls':0,
              'new_E_rgb_calls':0,'annotation_payload_reads':0,'gt_rgb_payload_reads':0,
              'actual_imports':imports,'deltas':{},'rgb_metrics_source':{'path':plan['prior_e_metrics'],
                  'sha256':sha(plan['prior_e_metrics'])},'timing_scope':'Cached teacher/E RGB reuse; not full system deployment FPS'}
    output.mkdir(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter(); failure = None
    try:
        for arm in ARMS:
            directory = root/arm; receipt = read(directory/'training_receipt.json'); delta_path = directory/'refiner_delta.pt'
            require(receipt['status'] == 'completed' and receipt['steps'] == 2000 and sha(delta_path) == receipt['delta_sha256'],
                    'Incomplete training endpoint')
            report['deltas'][arm] = {'path':str(delta_path),'sha256':sha(delta_path),
                'training_receipt_sha256':sha(directory/'training_receipt.json')}
            scene, state = modules['bridge_rgs.train'].load_scene(plan['base_checkpoint'])
            delta = torch.load(delta_path, map_location='cpu', weights_only=False)
            validate_delta(delta, arm, plan, state)
            scene.refiner.load_state_dict(delta['refiner_state'], strict=True)
            scene.eval().requires_grad_(False); adapter = attach(scene, {'mode':arm}); adapter.set_step(2000)
            real_render = scene.render; captured = []
            def capture(*args, _render=real_render, _captured=captured, **kwargs):
                value = _render(*args, **kwargs); _captured.append(value); report['scene_calls'] += 1
                return value
            scene.render = capture
            try:
                with torch.no_grad():
                    for view in views:
                        name, camera = view['name'], view['camera']; a = by_old['A',name]; e = by_old['E',name]
                        rgb, _, info = official.predict_official_camera(scene, camera, state)
                        require(len(captured) == 1, 'One scene call per camera required')
                        result = captured.pop()
                        K,w,h,back = official.distortion_render_grid(camera['K'],camera['distortion'],camera['width'],camera['height'],
                                                                   pixel_protocol='legacy_mixed_v1')
                        old_canvas = prior['h3_canvases'][name]
                        require(info == old_canvas['official_render_info'] and [w,h] == old_canvas['boundary']['canvas']
                            and np.array_equal(K,np.asarray(old_canvas['boundary']['canvas_K'],np.float32)), 'Cached probability grid differs')
                        rendered_rgb = save_png(output/arm/'h3_rgb'/name,rgb[...,::-1],{'path':a['rgb'],'sha256':a['rgb_sha256']})
                        canvas_rgb = (result['rgb'].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
                        canvas_record = save_png(output/arm/'h3_canvas_rgb'/name,canvas_rgb[...,::-1],old_canvas['canvas_rgb'])
                        require_record(a['teacher_soft_canvas'])
                        teacher = np.load(a['teacher_soft_canvas']['path'],allow_pickle=False)
                        final = result['probabilities'].cpu().numpy(); raw = result['p3d'].cpu().numpy()
                        require(teacher.shape == final.shape == (h,w,5) and teacher.dtype == np.float32
                                and np.isfinite(teacher).all() and (teacher>=0).all()
                                and np.allclose(teacher.sum(-1),1.,atol=2e-5,rtol=0),'Invalid teacher probability cache')
                        soft_record = save_soft(output/arm/'soft'/f'{Path(name).stem}.npy',final)
                        probabilities = {'joint':(final+teacher)*np.float32(.5),'scene':final,'raw':raw}
                        masks = {kind:save_png(output/arm/(kind+'_mask')/name,warp_probabilities(p,back).argmax(-1).astype(np.uint8))
                                 for kind,p in probabilities.items()}
                        rgb_reference = {'path':e['rgb'],'sha256':e['rgb_sha256']}; require_record(rgb_reference)
                        bgr = cv2.imread(e['rgb'],cv2.IMREAD_UNCHANGED)
                        require(bgr is not None and bgr.dtype == np.uint8 and bgr.shape == (camera['height'],camera['width'],3), 'Bad E RGB reference')
                        require(rgb_rows[name]['width'] == camera['width'] and rgb_rows[name]['height'] == camera['height'], 'E RGB score grid differs')
                        report['predictions'].append({'arm':arm,'name':name,'masks':masks,'soft_canvas':soft_record,
                            'teacher_soft_canvas':a['teacher_soft_canvas'],'teacher_input_rgb_exact':True,
                            'new_H3_original_rgb':rendered_rgb,'new_H3_canvas_rgb':canvas_record,
                            'rgb_reference':rgb_reference,'rgb_is_reused':True})
                        del result, final, raw, teacher
            finally:
                scene.render = real_render; adapter.detach()
            require(sha(delta_path) == receipt['delta_sha256'], 'Delta changed during evaluation')
            del scene, state, delta, adapter, real_render, capture, captured
            torch.cuda.empty_cache()
        require(report['scene_calls'] == 200, 'Wrong total scene call count')
        prediction_barrier(report['predictions'],views)
        report['predictions_finished_utc'] = utc()
        write(output/'predictions_receipt.json',{'status':'all_600_masks_before_GT','predictions':report['predictions'],
            'predictions_finished_utc':report['predictions_finished_utc'],'annotation_payload_reads':0,'scene_calls':200})
        report['source_scoring_started_utc'] = utc()
        rows = {(a,k):[] for a in ARMS for k in KINDS}
        by_prediction = {(r['arm'],r['name']):r for r in report['predictions']}
        for view in views:
            name = view['name']; truth = None
            if view['source_annotation_path'] is not None:
                content = Path(view['source_annotation_path']).read_bytes(); report['annotation_payload_reads'] += 1
                require(hashlib.sha256(content).hexdigest() == view['source_annotation_sha256'],'Changed annotation')
                truth = official.rasterize_official_annotation(content,view['camera']['width'],view['camera']['height'])
                require(hashlib.sha256(truth.tobytes()).hexdigest() == view['rasterized_mask_sha256'],'Rasterizer changed')
            for arm in ARMS:
                for kind in KINDS:
                    row = dict(rgb_rows[name])
                    if truth is not None:
                        record = by_prediction[arm,name]['masks'][kind]; require_record(record)
                        pred = cv2.imread(record['path'],cv2.IMREAD_UNCHANGED)
                        require(pred.dtype == np.uint8 and pred.shape == truth.shape and (pred<5).all(),'Invalid mask')
                        keep = truth != 255
                        cm = np.bincount(5*truth[keep].astype(np.int64)+pred[keep],minlength=25).reshape(5,5)
                        row.update(confusion_matrix=cm.tolist(),semantic_pixels=int(keep.sum()),semantic_ignore_pixels=int((~keep).sum()))
                    rows[arm,kind].append(row)
        require(report['annotation_payload_reads'] == 41,'Wrong annotation count')
        metrics = {}; metric_records = {}
        for (arm,kind), view_rows in rows.items():
            pooled = sum(np.asarray(r['confusion_matrix'],np.int64) for r in view_rows if 'confusion_matrix' in r)
            iou = compare._iou(pooled)
            value = {'evaluation_family':old['evaluation_family'],'scoring_protocol':old['scoring_protocol'],
                'official_evaluation_fingerprint':old['reference_fingerprint'],'inference_protocol':dict(INFERENCE,kind=kind),
                'validation_views':50,'semantic_validation_views':41,'views':view_rows,'confusion_matrix':pooled.tolist(),
                'iou':iou.tolist(),'miou_all':float(np.nanmean(iou)),'miou_foreground':float(np.nanmean(iou[1:])),
                **{k:float(np.mean([r[k] for r in view_rows])) for k in ('psnr','ssim','lpips')},
                'rgb_scoring':'Reused old E score rows after exact referenced PNG SHA checks; no new RGB GT score',
                'provenance':{'base_checkpoint_sha256':BASE_SHA,'delta':report['deltas'][arm],
                    'teacher_cache_checkpoint_sha256':TEACHER_SHA if kind=='joint' else None,
                    'rgb_metrics':report['rgb_metrics_source'],'single_shared_geometry_system':False}}
            compare._validate(value)
            p = output/arm/('official_metrics.json' if kind=='joint' else kind+'_official_metrics.json')
            write(p,value); metrics[arm,kind] = value; metric_records[arm+'_'+kind] = {'path':str(p),'sha256':sha(p)}
        pairs = {}; pair_records = {}
        for reference in THRESHOLDS:
            reference_metrics = old_e if reference=='E' else metrics[reference,'joint']
            pair = compare.paired_official_comparison(reference_metrics,metrics['projective','joint'],repeats=5000,seed=20260926)
            pair.update(reference_arm=reference,candidate_arm='projective')
            pairs[reference] = pair; path = output/f'paired_projective_minus_{reference}.json'
            write(path,pair); pair_records[reference] = {'path':str(path),'sha256':sha(path)}
        clauses = adoption_clauses(pairs)
        gate = {'passed':all(clauses.values()),'clauses':clauses,'thresholds':THRESHOLDS,
            'decision':'eligible_for_review' if all(clauses.values()) else 'stop_fixed_single_deck_implementation',
            'rgb':'Unchanged E bytes/scores reused; no RGB gain attributed to this experiment'}
        write(output/'system_gate.json',gate)
        report.update(status='completed',metrics=metric_records,comparisons=pair_records,gate=gate,
                      gate_sha256=sha(output/'system_gate.json'),finished_utc=utc())
    except BaseException as error:  # noqa: BLE001 - record failure and re-raise without a retry
        report.update(status='failed',error=repr(error)); failure = error
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        try:
            verify_sources(plan,skip_targets=raw_gt_rgb | (annotations if report['annotation_payload_reads'] == 0 else set()))
            report['bound_sources_inputs_unchanged'] = True
        except BaseException as error:  # noqa: BLE001 - preserve both the result and provenance failure
            report['bound_sources_inputs_unchanged'] = False
            report.update(status='failed',provenance_error=repr(error)); failure = failure or error
        write(output/'execution_receipt.json',report)
    if failure is not None:
        raise failure
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',type=Path,required=True)
    args = parser.parse_args(); plan = read(args.execute)
    sys.path.insert(0,plan['source_snapshot'])
    result = evaluate(plan)
    print(json.dumps({'status':result['status'],'gate_passed':result['gate']['passed']}))


if __name__=='__main__':
    main()
