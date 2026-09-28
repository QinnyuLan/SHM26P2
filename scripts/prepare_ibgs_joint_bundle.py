"""Freeze camera-only deployment and a separate 50-view + two-camera verification."""
from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

import render_ibgs_joint_bundle as export

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')


def prepare(output):
    require, read, sha = export.require, export.read, export.sha
    replacement = RUNS/'ibgs_top4_mcmc_replacement_v1'
    require(read(replacement/'independent_cpu_review.json')['status'] == 'passed'
            and read(replacement/'rgb_gate.json')['passed'], 'Reviewed passing replacement required')
    parent = read(RUNS/'ibgs_layer_heads_evaluation_v1/plan.json')
    old_main = read(RUNS/'multifield_h3_teacher_v1/deployment/bundle.json')
    contract = read(parent['data_contract']['path']); cache = read(parent['cache_manifest'])
    require(sha(parent['cache_manifest']) == parent['cache_manifest_sha256'], 'Source cache manifest changed')
    output = Path(output).resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data disk directory required')
    deployment = output/'deployment'; deployment.mkdir(parents=True)
    main = copy.deepcopy(old_main)
    obsolete = main['components'].pop('capacity_1m')['checkpoint']
    del main['input_hashes'][obsolete]
    main.pop('provenance', None)
    main['schema'] = export.SCHEMA
    worker = deployment/'render_ibgs_joint_bundle.py'
    shutil.copy2(Path(export.__file__), worker)
    original = Path(parent['source_snapshot'])/'evaluate_ibgs_layer_heads.py'
    adapted = deployment/'layer_camera_render_base.py'
    adapted.write_text(export.adapt_layer_worker(original.read_text()))
    legacy = deployment/'ibgs_warm_evaluation_base.py'
    shutil.copy2(Path(parent['source_snapshot'])/legacy.name, legacy)
    fields = ('source_snapshot', 'source_hashes', 'runtime_sources', 'interpreter', 'checkpoint',
              'data_contract', 'cache_manifest', 'cache_manifest_sha256', 'selector', 'backend',
              'training_plan_sha256', 'training_specification')
    ibgs = {k: copy.deepcopy(parent[k]) for k in fields}
    ibgs['endpoints'] = {'top4_normalized': parent['endpoints']['top4_normalized']}
    inputs = dict(contract['pixel_hashes']) | parent['runtime_sources']
    inputs[parent['cache_manifest']] = parent['cache_manifest_sha256']
    for value in (parent['checkpoint'], parent['data_contract'], ibgs['endpoints']['top4_normalized'],
                  cache['shared_eroded_valid']):
        inputs[value['path']] = value['sha256']
    for row in cache['records']:
        for kind in ('ids', 'depth', 'weights'):
            inputs[row[kind]['path']] = row[kind]['sha256']
    ibgs['input_hashes'] = inputs
    # Runtime bundle contains no old prediction/score paths or VAL ground-truth paths.
    bundle = {'schema': export.SCHEMA, 'worker_sha256': sha(worker), 'workspace_root': str(ROOT),
        'sensor': {k: parent['views'][0]['camera'][k] for k in ('width', 'height', 'K', 'distortion')},
        'main': main, 'ibgs': ibgs, 'main_helper': str(RUNS/'multifield_h3_teacher_v1/deployment/render_multifield_bundle.py'),
        'layer_original': str(original), 'layer_adapted': str(adapted),
        'helper_hashes': {str(p): sha(p) for p in (original, adapted, legacy,
                         RUNS/'multifield_h3_teacher_v1/deployment/render_multifield_bundle.py')},
        'rgb_weights': [.5, .5], 'teacher_weight': .5,
        'limitations': 'Fixed official sensor; pose must have at least one eligible TRAIN source. Three fields; '
                       'source photos and 21.93 GB layer cache required. No target image or test adaptation.'}
    export.write(deployment/'bundle.json', bundle)
    cameras = [v['camera'] for v in parent['views']]
    renamed = copy.deepcopy(cameras[0]); renamed.update(name='365.png', image_id=-1001, camera_id=-1001)
    novel = copy.deepcopy(cameras[0]); novel.update(name='novel_shifted.png', image_id=-1002, camera_id=-1002)
    novel['w2c'][0][3] += .1
    require('365.png' not in {c['name'] for c in cameras}, 'Smoke name collides with official views')
    export.write(output/'cameras.json', cameras+[renamed, novel])
    joint = read(RUNS/'multifield_h3_teacher_v1/plan.json')
    e_predictions = {v['name']: v for v in read(RUNS/'multifield_h3_teacher_v1/execution_receipt.json')['predictions']
                     if v['arm'] == 'E'}
    rgb_predictions = {v['name']: v for v in read(replacement/'execution_receipt.json')['predictions']}
    plan = {'schema': 'joint_replacement_verification_v1', 'bundle': str(deployment/'bundle.json'),
        'bundle_sha256': sha(deployment/'bundle.json'), 'worker_sha256': sha(worker),
        'cameras': str(output/'cameras.json'), 'cameras_sha256': sha(output/'cameras.json'),
        'output': str(output), 'render_output': str(output/'render'),
        'views': joint['views'], 'reference_fingerprint': parent['reference_fingerprint'],
        'reference_predictions': {n: {'rgb': rgb_predictions[n], 'semantic': e_predictions[n]}
                                  for n in sorted(rgb_predictions)},
        'references': {'E': str(RUNS/'multifield_h3_teacher_v1/E/official_metrics.json'),
                       'Swin': joint['swin_metrics']},
        'source_replacement': str(replacement), 'scoring_snapshot': joint['source_snapshot'],
        'scoring_source_hashes': joint['source_hashes'], 'scoring_numerics': parent['scoring_numerics'],
        'perceptual_weights': parent['perceptual_weights'],
        'frozen_inputs': {str(replacement/n): sha(replacement/n) for n in ('plan.json', 'execution_receipt.json',
                         'launch_receipt.json', 'rgb_metrics.json', 'independent_cpu_review.json', 'rgb_gate.json')},
        'render_external_timeout_seconds': 900, 'score_external_timeout_seconds': 360,
        'fixed_counts': {'cameras': 52, 'scene_renders': 156, 'teacher_calls': 52, 'selector_calls': 52,
                         'score_views': 50, 'semantic_score_views': 41},
        'gate': 'All 50 new RGB and masks exact to reviewed RGB candidate and E semantic outputs; '
                'renamed 001 -> 365 exact; shifted pose valid and changed; re-score 50/41; '
                'no fit, search, semantic improvement or blind-test claim'}
    for p in plan['references'].values():
        plan['frozen_inputs'][p] = sha(p)
    for p in plan['perceptual_weights']:
        plan['frozen_inputs'][p] = parent['input_hashes'][p]
    for name in ('prepare_ibgs_joint_bundle.py', 'score_ibgs_joint_bundle.py'):
        source = ROOT/'scripts'/name
        if source.exists():
            shutil.copy2(source, output/name)
            plan['frozen_inputs'][str(output/name)] = sha(output/name)
    for name in ('ibgs_joint_bundle_protocol.md',):
        shutil.copy2(ROOT/'docs'/name, output/name)
        plan['frozen_inputs'][str(output/name)] = sha(output/name)
    export.verify_hashes(inputs)
    export.verify_hashes(main['input_hashes'])
    export.write(output/'plan.json', plan)
    print({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json'), 'bundle': str(deployment/'bundle.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)
