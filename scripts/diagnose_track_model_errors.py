"""Frozen CPU association of actual TRAIN track labels and cached H3 final errors."""
from __future__ import annotations

import argparse
import hashlib
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
TRACKS = Path('/mnt/data/SHM2026/runs/train_track_semantics_v1')
CACHE = Path('/mnt/data/SHM2026/runs/cross_renderer_authority_v1')
H3_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
NAMES = [f'{i:03d}.png' for i in (2,21,41,59,79,100,118,137,156,176,200,220,241,259,278,300)]
SPEC = {'protocol': 'track_h3_error_association_v1', 'views': NAMES, 'point_count': 60000,
        'sampling': 'saved legacy_index on primary strict and legacy-valid, labels must agree',
        'strict': 'known primary label, positive depth, reprojection <=1, primary EDT>10',
        'reference': 'all primary strict observations excluding query whole camera-name group',
        'probabilities': 'cached final H3 FP32 retained without renormalization or clipping',
        'bootstrap_repeats': 2000, 'seed': 20260927,
        'sensitivity': 'drop every point with any saved duplicate pair, including reference/query',
        'internal_timeout_seconds': 120, 'external_timeout_seconds': 180,
        'new_pixel_label_decodes': 0, 'gpu_calls': 0, 'model_adoption': False}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    content = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as f: f.write(content)


def files(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def load_module(path):
    spec = importlib.util.spec_from_file_location('error_association_module', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def query_metadata(manifest, cache_plan, receipt):
    train = sorted((v for v in manifest['views'] if v['split'] == 'train'), key=lambda v:v['name'])
    require(len(train) == 350 and sum(bool(v.get('mask_path')) for v in train) == 259, 'Changed TRAIN population')
    by_name = {v['name']:v for v in train}; views = cache_plan['views']
    require(len(by_name) == 350 and len({v['image_id'] for v in train}) == 350, 'Repeated TRAIN identity')
    require([v['name'] for v in views] == [v['name'] for v in receipt['predictions']] == NAMES, 'Changed fixed16 order')
    rows = []
    for v, record in zip(views, receipt['predictions'], strict=True):
        own = by_name[v['name']]
        require(v['split'] == record['split'] == own['split'] == 'train', 'Non-TRAIN query')
        camera = v['camera']
        require(camera['image_id'] == own['image_id'] and camera['camera_id'] == own['camera_id']
                and camera['w2c'] == own['w2c_original'] == own['w2c'], 'Changed query pose')
        require(np.array_equal(np.asarray(v['K']), np.asarray(own['K'], np.float32)), 'Changed FP32 K')
        require((v['width'],v['height'],own['width'],own['height']) == (1320,989,1320,989), 'Changed native grid')
        source = record['S']
        require(source['shape'] == [989,1320,5] and source['dtype'] == 'float32', 'Wrong S grid/type')
        require(Path(source['path']).resolve() == CACHE/'soft/S'/f'{Path(v["name"]).stem}.npy', 'Unexpected S path')
        rows.append({'name': v['name'], 'image_id': own['image_id'], 'probability': source})
    return [{'name':v['name'],'image_id':v['image_id'],'group':i*4//350} for i,v in enumerate(train)], rows


def prepare(output):
    output = Path(output).resolve(); require(not output.exists(), 'Never overwrite experiment')
    manifest_path = ROOT/'artifacts/prepared/manifest.json'; manifest = read(manifest_path)
    track_plan, track_receipt = read(TRACKS/'plan.json'), read(TRACKS/'execution_receipt.json')
    cache_plan, cache_receipt = read(CACHE/'plan.json'), read(CACHE/'execution_receipt.json')
    require(track_receipt['status'] == cache_receipt['status'] == 'completed', 'Incomplete source run')
    require(track_receipt['plan_sha256'] == sha(TRACKS/'plan.json')
            and cache_receipt['plan_sha256'] == sha(CACHE/'plan.json'), 'Source plan identity changed')
    require(cache_receipt['inputs_and_sources_unchanged'], 'Cache source invariance failed')
    require(cache_plan['scene']['checkpoint_sha256'] == H3_SHA, 'Wrong H3 endpoint')
    train, views = query_metadata(manifest, cache_plan, cache_receipt)
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        require(expected is None or digest == expected, f'Changed input: {path}')
        inputs[str(path)] = digest
        return str(path)

    bind(manifest_path, track_plan['input_hashes'][str(manifest_path)])
    bind(ROOT/'uv.lock'); bind(cache_plan['scene']['checkpoint'], H3_SHA)
    for parent, prior in ((TRACKS,track_plan),(CACHE,cache_plan)):
        require(files(prior['source_snapshot']) == prior['source_hashes'], 'Prior source changed')
        for rel, digest in prior['source_hashes'].items(): bind(Path(prior['source_snapshot'])/rel, digest)
        for name in ('plan.json','execution_receipt.json'): bind(parent/name)
    for name in ('collection_receipt.json','analysis.json','independent_collection_review.json',
                 'independent_statistics_review.json','duplicate_keypoint_review.json'):
        bind(TRACKS/name)
    for name in ('launch_receipt.json','analysis_input.json','independent_cpu_review.json'):
        bind(CACHE/name)
    observations = bind(TRACKS/'observations.npz',track_receipt['output_files']['observations.npz'])
    dup = read(TRACKS/'duplicate_keypoint_review.json')
    require(dup['observations_sha256'] == inputs[observations], 'Duplicate metadata uses other population')
    duplicates = bind(dup['metadata_path'],dup['metadata_sha256'])
    for v in views: bind(v['probability']['path'],v['probability']['sha256'])
    module = load_module(ROOT/'src/bridge_rgs/track_error_association.py')
    snapshot = output/'source_snapshot'; (snapshot/'bridge_rgs').mkdir(parents=True)
    for name in ('__init__.py','track_error_association.py'):
        shutil.copy2(ROOT/'src/bridge_rgs'/name,snapshot/'bridge_rgs'/name)
    for p in (Path(__file__),ROOT/'tests/test_track_error_association.py',
              ROOT/'tests/test_track_model_error_runner.py', ROOT/'docs/track_model_error_protocol.md'):
        shutil.copy2(p,snapshot/p.name)
    plan = {'specification':SPEC,'analysis_specification':module.SPEC,'output':str(output),
            'source_snapshot':str(snapshot),'source_hashes':files(snapshot),'input_hashes':inputs,
            'observations':observations,'duplicate_metadata':duplicates,'training_images':train,'views':views,
            'runtime_versions':{'numpy':importlib.metadata.version('numpy')},
            'created_utc':datetime.now(UTC).isoformat(),'prepare_array_loads':0,'prepare_file_byte_hashing':True}
    write_new(output/'plan.json',plan)
    print(json.dumps({'plan':str(output/'plan.json'),'sha256':sha(output/'plan.json'),
                      'source_count':len(plan['source_hashes']),'bound_input_count':len(inputs)}),flush=True)


def verify(plan):
    require(plan['specification'] == SPEC, 'Changed specification')
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/Path(__file__).name, 'Use frozen runner')
    require(files(plan['source_snapshot']) == plan['source_hashes'], 'Changed frozen source')
    require(importlib.metadata.version('numpy') == plan['runtime_versions']['numpy'], 'Changed NumPy')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules, 'CPU-only execution')
    for path,digest in plan['input_hashes'].items(): require(sha(path) == digest,f'Changed input {path}')


def strict_mask(a):
    return ((a['primary_label'] >= 0) & (a['primary_label'] < 5) & a['positive_depth']
            & (a['reprojection_error'] <= 1.) & (a['primary_boundary_distance'] > 10.))


def sample_queries(a,views):
    query_ids = [v['image_id'] for v in views]
    strict = strict_mask(a)
    use = strict & a['legacy_valid'] & np.isin(a['image_id'],query_ids)
    rows = np.flatnonzero(use)
    require(np.array_equal(a['primary_label'][rows],a['legacy_label'][rows]), 'Primary/legacy strict labels differ')
    probabilities = np.full((len(rows),5),np.nan,np.float32)
    evidence=[]
    for v in views:
        selected = np.flatnonzero(a['image_id'][rows] == v['image_id'])
        indices = a['legacy_index'][rows[selected]]
        source=v['probability']; p=np.load(source['path'],mmap_mode='r',allow_pickle=False)
        require(p.shape == (989,1320,5) and p.dtype == np.float32, 'Changed cached S header')
        require(((indices >= 0) & (indices < [1320,989])).all(), 'Query index outside S grid')
        probabilities[selected] = p[indices[:,1],indices[:,0]]
        evidence.append({'name':v['name'],'image_id':v['image_id'],'queries':len(selected),
                         'cached_probability_sha256':source['sha256']})
    require(np.isfinite(probabilities).all() and ((probabilities >= 0)&(probabilities <= 1)).all()
            and np.allclose(probabilities.sum(1),1.,rtol=0,atol=2e-5), 'Malformed sampled probabilities')
    identities=np.column_stack([a['image_id'][rows],a['legacy_index'][rows]])
    return strict,rows,probabilities,{'views':evidence,'queries':len(rows),
        'unique_image_pixel_queries':len(np.unique(identities,axis=0)),
        'duplicate_image_pixel_queries':len(rows)-len(np.unique(identities,axis=0))}


def execute(path):
    path=Path(path).resolve();plan=read(path);output=Path(plan['output'])
    require(path == output/'plan.json' and not (output/'execution_receipt.json').exists(), 'Wrong plan or already executed')
    verify(plan); module=load_module(Path(plan['source_snapshot'])/'bridge_rgs/track_error_association.py')
    require(module.SPEC == plan['analysis_specification'],'Changed analysis contract')
    report={'status':'running','plan_sha256':sha(path),'started_utc':datetime.now(UTC).isoformat(),
            'gpu_calls':0,'new_pixel_label_decodes':0,'new_model_renders':0,'model_adoption':False}
    failure=None;start=time.perf_counter()

    def alarm(_signum,_frame): raise TimeoutError('Fixed diagnostic deadline exceeded')

    signal.signal(signal.SIGALRM,alarm);signal.alarm(SPEC['internal_timeout_seconds'])
    try:
        with np.load(plan['observations'],allow_pickle=False) as f:
            a={k:f[k] for k in ('point_index','image_id','primary_label','legacy_label','positive_depth',
                               'reprojection_error','primary_boundary_distance','legacy_valid','legacy_index')}
        with np.load(plan['duplicate_metadata'],allow_pickle=False) as f:
            touched=np.unique(f['point_index'])
        strict,rows,p,evidence=sample_queries(a,plan['views'])
        groups={v['image_id']:v['group'] for v in plan['training_images']}
        group=np.asarray([groups[int(i)] for i in a['image_id']],np.int8)
        result=module.analyze_track_error_association(a['point_index'],a['image_id'],a['primary_label'],
            strict,group,p,rows,point_count=SPEC['point_count'],
            query_image_ids=[v['image_id'] for v in plan['views']],duplicate_touched_point_indices=touched,
            bootstrap_repeats=SPEC['bootstrap_repeats'],seed=SPEC['seed'])
        with (output/'query_input.npz').open('xb') as f:
            np.savez_compressed(f,target_row_indices=rows,query_prob=p,duplicate_touched_point_indices=touched)
        with (output/'query_analysis.npz').open('xb') as f:
            np.savez_compressed(f,**result['query_arrays'])
        analysis={'plan_sha256':sha(path),'query_sampling':evidence,**result['report']}
        write_new(output/'analysis.json',analysis)
        verify(plan)
        report.update(status='completed',output_files={name:sha(output/name) for name in
                      ('query_input.npz','query_analysis.npz','analysis.json')})
    except BaseException as exc:  # noqa: BLE001 - preserve failed execution, then re-raise
        failure=exc;report.update(status='failed',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.perf_counter()-start
        report['ended_utc']=datetime.now(UTC).isoformat();write_new(output/'execution_receipt.json',report)
        print(json.dumps(report),flush=True)
    if failure is not None: raise failure


def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare');g.add_argument('--execute');args=parser.parse_args()
    prepare(args.prepare) if args.prepare else execute(args.execute)


if __name__ == '__main__': main()
