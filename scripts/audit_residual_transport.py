"""Independent CPU replay of saved residual transport; no producer imports/rendering."""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import cv2
import numpy as np

NAMES = ['002.png', '021.png', '041.png', '059.png', '079.png', '100.png',
         '118.png', '137.png', '156.png', '176.png', '200.png', '220.png',
         '241.png', '259.png', '278.png', '300.png']
ATOL = 1e-11


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


class Audit:
    def __init__(self):
        self.hashes = {}; self.max_error = 0.; self.comparisons = 0

    def bind(self, path, expected):
        if path not in self.hashes:
            self.hashes[path] = sha(path)
        check(self.hashes[path] == expected, 'SHA: '+str(path))

    def arrays(self, record):
        self.bind(record['path'], record['sha256'])
        with np.load(record['path'], allow_pickle=False) as saved:
            return {k: saved[k] for k in saved.files}

    def close(self, actual, expected, label):
        if isinstance(actual, dict):
            check(actual.keys() == expected.keys(), 'Keys: '+label)
            for k in actual:
                self.close(actual[k], expected[k], label+'.'+k)
        elif isinstance(actual, (str, bool)) or actual is None:
            check(actual == expected, 'Identity: '+label)
        else:
            a, b = np.asarray(actual), np.asarray(expected)
            check(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'Finite shape: '+label)
            error = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)), initial=0))
            self.max_error = max(self.max_error, error); self.comparisons += a.size
            if a.dtype.kind in 'biu':
                check(np.array_equal(a, b), 'Exact count/support: '+label)
            else:
                check(error <= ATOL, 'Numerical mismatch: '+label+f' ({error})')


def selection(train, radius):
    indexed = {v['name']: v for v in train}; bank = sorted(set(indexed)-set(NAMES))
    result = []
    for i, name in enumerate(NAMES):
        target = np.asarray(indexed[name]['w2c_original'], np.float64)
        center = -target[:3, :3].T@target[:3, 3]
        distances = []
        for source in bank:
            pose = np.asarray(indexed[source]['w2c_original'], np.float64)
            source_center = -pose[:3, :3].T@pose[:3, 3]
            angle = np.arccos(np.clip((np.trace(target[:3, :3]@pose[:3, :3].T)-1)/2, -1, 1))
            distances.append((float(np.sum((center-source_center)**2)/radius**2+(angle/np.pi)**2), source))
        distances.sort()
        result.append({'name': name, 'fold': 'A' if i % 2 == 0 else 'B',
                       'excluded_nearest': {'name': distances[0][1], 'distance': distances[0][0]},
                       'sources': [{'name': n, 'distance': d} for d, n in distances[1:5]]})
    return result


def remap(value, xy, inside):
    safe = np.where(inside[..., None], xy, 0).astype(np.float32)
    return cv2.remap(np.ascontiguousarray(value), safe[..., 0], safe[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def replay(target, sources):
    h, w = target['depth'].shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    pixels = np.stack((xx+.5, yy+.5, np.ones((h, w))), -1)
    K, pose = target['K'].astype(np.float64), target['w2c'].astype(np.float64)
    rays = pixels@np.linalg.inv(K).T
    world = (target['depth'][..., None]*rays-pose[:3, 3])@np.linalg.inv(pose[:3, :3]).T
    good_target = target['valid'] & (target['depth'] > 0) & (target['alpha'] >= .95)
    sums = [np.zeros((h, w, 3), np.float64), np.zeros((h, w, 3), np.float64)]
    counts = np.zeros((h, w), np.int32); summaries = []
    for source in sources:
        sh, sw = source['depth'].shape
        pose, K = source['w2c'].astype(np.float64), source['K'].astype(np.float64)
        camera = world@pose[:3, :3].T+pose[:3, 3]; projected = camera@K.T
        z = camera[..., 2]; positive = np.isfinite(projected).all(-1) & (z > 0)
        xy = np.zeros((h, w, 2), np.float64)
        np.divide(projected[..., :2], projected[..., 2, None], out=xy, where=positive[..., None])
        xy -= .5
        positive &= np.isfinite(xy).all(-1); xy[~positive] = 0
        inside = positive & (xy[..., 0] >= 0) & (xy[..., 0] <= sw-1) & (xy[..., 1] >= 0) & (xy[..., 1] <= sh-1)
        reflected = xy.copy(); reflected[..., 0] = sw-1-reflected[..., 0]
        normal_valid = remap(source['valid'].astype(np.float32), xy, inside) == 1
        reflected_valid = remap(source['valid'].astype(np.float32), reflected, inside) == 1
        depth = remap(source['depth'].astype(np.float64), xy, inside)
        alpha = remap(source['alpha'].astype(np.float64), xy, inside)
        relative = np.full((h, w), np.inf)
        np.divide(np.abs(z-depth), np.maximum(z, depth), out=relative, where=inside & (depth > 0))
        geometric = good_target & inside & (depth > 0) & (alpha >= .95) & (relative <= .01)
        common = geometric & normal_valid & reflected_valid
        for total, coords in zip(sums, (xy, reflected), strict=True):
            samples = remap(source['residual'].astype(np.float64), coords, inside)
            total[common] += samples[common]
        counts += common
        summaries.append({'geometric_pixels': int(geometric.sum()),
                          'normal_rgb_valid_pixels': int((geometric & normal_valid).sum()),
                          'mirrored_rgb_valid_pixels': int((geometric & reflected_valid).sum()),
                          'common_pixels': int(common.sum()), 'source_shape': [sh, sw]})
    for total in sums:
        np.divide(total, counts[..., None], out=total, where=counts[..., None] > 0)
    return {'true_residual': sums[0], 'wrong_residual': sums[1],
            'valid': counts > 0, 'source_count': counts}, summaries


def sufficient(base, target, residual, valid):
    difference = target.astype(np.float64)[valid]-base.astype(np.float64)[valid]
    r = residual[valid]
    return {'numerator': float(np.sum(r*difference)/r.size),
            'denominator': float(np.sum(r*r)/r.size),
            'zero_mse': float(np.sum(difference*difference)/difference.size),
            'valid_pixels': int(valid.sum())}


def fit(rows):
    numerator = sum(r['numerator'] for r in rows)/len(rows)
    denominator = sum(r['denominator'] for r in rows)/len(rows)
    coefficient = 0. if denominator == 0 else min(1., max(0., numerator/denominator))
    return {'coefficient': coefficient, 'numerator': numerator, 'denominator': denominator,
            'views': len(rows), 'zero_residual_energy': denominator == 0,
            'objective': 'equal_view_unclipped_mse'}


def run_audit(run, expected):
    destination = run/'independent_cpu_review.json'
    check(not destination.exists(), 'Do not overwrite audit')
    report = {'status': 'failed'}; start = time.monotonic(); audit = Audit()
    def expired(*_):
        raise TimeoutError('Independent CPU audit 180s deadline')
    signal.signal(signal.SIGALRM, expired); signal.alarm(180)
    try:
        check(sha(run/'plan.json') == expected, 'Plan identity')
        plan, execution, launch = (read(run/n) for n in ('plan.json', 'execution_receipt.json', 'launch_receipt.json'))
        check(execution['status'] == launch['status'] == 'completed' and launch['natural_completion'] is True
              and launch['exit_code'] == 0 and launch['plan_sha256'] == execution['plan_sha256'] == expected
              and launch['execution_receipt_sha256'] == sha(run/'execution_receipt.json'), 'Natural completion required')
        check(plan['specification']['protocol'] == 'residual_transport_probe_v1'
              and plan['specification']['targets'] == NAMES, 'Fixed diagnostic')
        check(tree(Path(plan['source_snapshot'])) == plan['source_hashes'], 'Frozen source tree')
        check(all(plan['source_hashes'][k] == h for k, h in plan['inherited_package_hashes'].items()), 'Inherited 1M renderer exact')
        for path, value in {**plan['input_hashes'], **plan['pixel_input_hashes'], **plan['installed_sources'],
                            plan['expected_gsplat_binary']['path']: plan['expected_gsplat_binary']['sha256']}.items():
            audit.bind(path, value)
        audit.bind(str(run/'analysis.json'), execution['analysis_sha256']); saved = read(run/'analysis.json')
        train = sorted((v for v in read(plan['manifest'])['views'] if v['split'] == 'train'), key=lambda v: v['name'])
        check(len(train) == 350 and [v['name'] for v in train] == plan['train_names'], '350 TRAIN source population')
        chosen = selection(train, read(plan['manifest'])['scene_radius'])
        check(chosen == plan['camera_selection'], 'Independent metadata-only source selection')
        source_names = sorted({s['name'] for row in chosen for s in row['sources']})
        check(source_names == plan['source_names'] and not set(source_names)&set(NAMES), 'All16 excluded from sources')
        names = sorted(set(source_names)|set(NAMES)); views = {v['name']: v for v in plan['views']}
        check([r['name'] for r in execution['renders']] == names and set(views) == set(names), 'One render per fixed camera')
        check(execution['counts'] == {'scene': len(names), 'high_raster': len(names), 'low_raster': len(names),
              'source_RGB_decodes': len(source_names), 'target_RGB_decodes': 16,
              'valid_decodes': len({v['valid_path'] for v in views.values()})}, 'Runtime counts')
        check(execution['prediction_barrier_complete'] and execution['source_inputs_unchanged']
              and execution['state_tensor_exact'] and execution['scene_flags_restored'] and execution['numerics_restored'], 'Runtime restoration/barrier attestations')
        check(execution['semantic_GT_decodes'] == execution['VAL_decodes'] == execution['model_updates'] == 0, 'No semantic/VAL/model changes')
        check(execution['numerics_actual'] == {'cudnn_allow_tf32': True, 'matmul_allow_tf32': False,
              'matmul_precision': 'highest', 'cudnn_benchmark': False}, 'Frozen numerical settings')
        for bound in execution['actual_imports'].values():
            relative = str(Path(bound['path']).relative_to(plan['source_snapshot']))
            check(bound['sha256'] == plan['source_hashes'][relative], 'Actual package import provenance')
            audit.bind(bound['path'], bound['sha256'])
        check(plan['expected_gsplat_binary'] in execution['actual_gsplat_binary'].values(), 'Loaded binary attestation')
        rendered = {}
        for record in execution['renders']:
            name = record['name']; data = audit.arrays(record); view = views[name]
            shape = (view['height'], view['width'])
            check(set(data) == {'rgb', 'depth', 'alpha', 'valid', 'K', 'w2c'} | ({'residual'} if name in source_names else set()), 'Render schema')
            check(data['rgb'].shape == shape+(3,) and data['depth'].shape == data['alpha'].shape == data['valid'].shape == shape
                  and data['valid'].dtype == np.bool_ and all(np.isfinite(v).all() for v in data.values()), 'Finite native arrays')
            for key, meta in [('K', 'K'), ('w2c', 'w2c_original')]:
                check(data[key].dtype == np.float32 and np.array_equal(data[key], np.asarray(view[meta], np.float32)), 'Actual FP32 camera')
            rendered[name] = data
        check([r['name'] for r in execution['transports']] == NAMES == [r['name'] for r in saved['views']], 'Complete16 barrier arrays')
        rows = []; cv2.setNumThreads(4)
        for i, (choice, record, saved_row) in enumerate(zip(chosen, execution['transports'], saved['views'], strict=True)):
            name = choice['name']; base = rendered[name]
            actual, support = replay(base, [rendered[s['name']] for s in choice['sources']])
            audit.close(actual, audit.arrays(record), name+'.transport')
            for a, b in zip(support, record['source_summaries'], strict=True):
                audit.close(a, b, name+'.support')
            truth = audit.arrays(saved_row['target'])
            check(set(truth) == {'rgb', 'valid'} and np.array_equal(truth['valid'], base['valid']), 'Saved target RGB-valid support')
            stats = {arm: sufficient(base['rgb'], truth['rgb'], actual[arm+'_residual'], base['valid']) for arm in ('true', 'wrong')}
            audit.close(stats, saved_row['statistics'], name+'.statistics')
            rows.append({'name': name, 'fold': 'AB'[i % 2], 'statistics': stats,
                         'transport_pixels': int(actual['valid'].sum()), 'valid_pixels': int(base['valid'].sum())})
        fits = {arm: {fold: fit([r['statistics'][arm] for r in rows if r['fold'] == fold]) for fold in 'AB'} for arm in ('true', 'wrong')}
        audit.close(fits, saved['fits'], 'Four cross-fit scalars')
        for row, cache, old in zip(rows, execution['transports'], saved['views'], strict=True):
            base = rendered[row['name']]; truth = audit.arrays(old['target'])['rgb']; residual = audit.arrays(cache)
            other = 'B' if row['fold'] == 'A' else 'A'; row['metrics'] = {}
            for arm in ('zero', 'true', 'wrong'):
                coefficient = 0. if arm == 'zero' else fits[arm][other]['coefficient']
                prediction = base['rgb'].astype(np.float64)+(0. if arm == 'zero' else coefficient*residual[arm+'_residual'])
                error = prediction-truth.astype(np.float64); clipped = np.clip(prediction, 0, 1)-truth
                row['metrics'][arm] = {'coefficient': coefficient, 'fit_fold': None if arm == 'zero' else other,
                    'unclipped_MSE': float(np.mean(error[base['valid']]**2)), 'clipped_MSE': float(np.mean(clipped[base['valid']]**2)),
                    'support_clipped_MSE': float(np.mean(clipped[residual['valid']]**2)) if residual['valid'].any() else None}
            audit.close(row, {k: old[k] for k in row}, row['name']+'.OOF')
        summary = {a: {m: float(np.mean([r['metrics'][a][m] for r in rows])) for m in ('unclipped_MSE', 'clipped_MSE')} for a in ('zero', 'true', 'wrong')}
        audit.close(summary, saved['equal_view_mean'], 'Equal-view means')
        check(saved['performance_adoption_gate'] is saved['semantic_metrics'] is None, 'No invented adoption/semantic gate')
        report.update(status='passed', plan_sha256=expected, execution_receipt_sha256=sha(run/'execution_receipt.json'),
                      source_files=len(plan['source_hashes']), bound_files=len(audit.hashes), views=rows, fits=fits,
                      equal_view_mean=summary, transport_replays=16, scalar_and_array_entries_compared=int(audit.comparisons),
                      max_absolute_error=audit.max_error, absolute_tolerance=ATOL,
                      limitations=['No GPU, model deserialization, original image/label decoding, or rendering.',
                                   'Original file bytes are read only for hashing; scoring uses saved decoded target RGB.',
                                   'Renderer values, original RGB decoding, chronology and runtime restoration remain bound attestations.',
                                   'OpenCV interpolation is the declared operator; independent projection/support/fit formulas do not import producer.',
                                   'TRAIN source-image leave-out and scalar cross-fit are not field out-of-fit or new-view generalization.'])
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        signal.alarm(0); report['elapsed_seconds'] = time.monotonic()-start
        report['auditor_sha256'] = sha(__file__)
        with destination.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path); parser.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args(); run_audit(args.run, args.expected_plan_sha256)
