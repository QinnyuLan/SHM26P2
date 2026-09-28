"""One fixed E member replacement: top4-normalized IBGS plus original MCMC."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path('/home/sky/workspace/SHM2026')
RUNS = Path('/mnt/data/SHM2026/runs')
E = RUNS/'rgb_fixed_ensemble_v1'
LAYERS = RUNS/'ibgs_layer_heads_evaluation_v1'
FACTORIAL = RUNS/'ibgs_factorial_evaluation_v1'
BASE = RUNS/'ibgs_e_fixed_blend_v1/source_snapshot/evaluate_ibgs_e_fixed_blend.py'
BASE_SHA = '2f66d85d53065c9f762d4f07e31ceae1bd5e8fd17492ae1ffc3a803efc4e1032'
LAYER_SHA = '5c744495787a55c48a11cce9e0afc27f8a21758a08013b82fe4454f61b955929'
FACTORIAL_SHA = 'd7b728d2660955bedd8c12a11557a676b5b66677bec249bf66243a7c1cbd7ab9'
MEMBERS = ('top4_normalized', 'mcmc')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_base(path=None):
    path = path or BASE
    require(sha(path) == BASE_SHA, 'Exact previously executed combination/scorer required')
    spec = importlib.util.spec_from_file_location('fixed_replacement_base', path)
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    base.SPEC = copy.deepcopy(base.SPEC)
    base.SPEC.update(protocol='ibgs_top4_mcmc_fixed_member_replacement_v1', members=list(MEMBERS),
        arithmetic='np.rint((top4_normalized_uint8.astype(float32)+mcmc_uint8.astype(float32))*.5).astype(uint8)',
        selection='One post-factorial fixed member replacement at .5/.5; no weights, members or checkpoint search',
        cost='Two RGB fields plus fusion head, 350 TRAIN source photos and same-ID layer cache; '
             'the separate H3/DINOv3 semantic path is not run here; cached scoring is not system latency',
        scope='Engineering replacement after factorial development results; no novelty, blind-test '
              'or joint-system adoption claim')
    base.MEMBERS = MEMBERS
    return base


def matching_views(ep, lp, e_predictions, layer_predictions, em, lm, mm):
    """Keep the original MCMC bytes and all camera/GT metadata fixed."""
    require(ep['reference_fingerprint'] == lp['reference_fingerprint']
            == em['inherited_common_reference_fingerprint'] == lm['inherited_common_reference_fingerprint']
            == mm['official_evaluation_fingerprint'], 'Common evaluation population required')
    # Historical E metrics inherit the protocol from their bound plan/receipt;
    # that older file has no standalone scoring_protocol field.
    require(ep['original_scoring_protocol'] == lp['scoring_protocol']
            == lm['scoring_protocol'] == mm['scoring_protocol'], 'Scoring protocol mismatch')
    if 'scoring_protocol' in em:
        require(em['scoring_protocol'] == ep['original_scoring_protocol'], 'E scoring protocol mismatch')
    erows = {v['name']: v for v in em['views']}; lrows = {v['name']: v for v in lm['views']}
    mrows = {v['name']: v for v in mm['views']}; es = {v['name']: v for v in e_predictions}
    selected = [v for v in layer_predictions if v['arm'] == MEMBERS[0]]
    ls = {v['name']: v for v in selected}; ev = {v['name']: v for v in ep['views']}
    lv = {v['camera']['name']: v for v in lp['views']}; names = sorted(erows)
    require(all(len(v) == 50 for v in (names, em['views'], lm['views'], mm['views'],
                                      e_predictions, selected, ep['views'], lp['views']))
            and all(sorted(v) == names for v in (lrows, mrows, es, ls, ev, lv)),
            'Exactly the same 50 unique views required')
    views = []
    for name in names:
        a, b = ev[name], lv[name]; camera = a['camera']
        require(camera == b['camera'] and camera['name'] == name and Path(name).name == name,
                'Camera substitution')
        require(all(a[k] == b[k] for k in ('source_image_path', 'source_image_sha256')), 'Target substitution')
        old_mcmc = a['members']['mcmc']
        for row, expected in ((erows[name], es[name]['rgb_sha256']),
                              (lrows[name], ls[name]['sha256']), (mrows[name], old_mcmc['sha256'])):
            require(row['rgb_sha256'] == expected and row['source_rgb_sha256'] == a['source_image_sha256']
                    and row['width'] == camera['width'] and row['height'] == camera['height']
                    and row['rgb_pixels'] == camera['width']*camera['height'], 'Prediction/grid lineage mismatch')
        views.append({'name': name, 'camera': camera,
            'members': {MEMBERS[0]: {'path': ls[name]['path'], 'sha256': ls[name]['sha256']},
                        'mcmc': copy.deepcopy(old_mcmc)},
            **{k: a[k] for k in ('source_image_path', 'source_image_sha256')}})
    return views


def attach_mcmc_lineage(metrics, receipt, views):
    """Older official rows keep byte/camera identity in a separate receipt."""
    result = copy.deepcopy(metrics)
    predictions = {v['name']: v for v in receipt['predictions']}
    sources = {v['name']: v for v in receipt['source_records']}
    expected = {v['name']: v for v in views}; names = sorted(expected)
    require(receipt['status'] == 'completed'
            and receipt['predictions_finished_utc'] <= receipt['source_scoring_started_utc']
            and receipt['official_evaluation_fingerprint'] == metrics['official_evaluation_fingerprint']
            and receipt['scoring_protocol'] == metrics['scoring_protocol'], 'MCMC receipt identity mismatch')
    require(len(views) == len(names) == len(metrics['views']) == len(receipt['predictions'])
            == len(receipt['source_records']) == 50
            and all(sorted(v) == names for v in (predictions, sources))
            and sorted(v['name'] for v in metrics['views']) == names, 'MCMC population mismatch')
    for row in result['views']:
        name = row['name']; source = sources[name]; pred = predictions[name]; view = expected[name]
        require(all(source[k] == view[k] for k in ('camera', 'source_image_path', 'source_image_sha256'))
                and pred['rgb'] == view['members']['mcmc']['path']
                and pred['rgb_sha256'] == view['members']['mcmc']['sha256'], 'MCMC camera/byte substitution')
        row.update(rgb_sha256=pred['rgb_sha256'], source_rgb_sha256=source['source_image_sha256'])
    return result


def prepare(output):
    import torch
    base = load_base(); helper = base.helpers(); inputs = {}

    def bound(path, expected=None):
        value = sha(path); require(expected is None or value == expected, 'Changed bound input '+str(path))
        inputs[str(path)] = value
        return read(path)

    ep = bound(E/'plan.json', base.E_PLAN_SHA)
    ee = bound(E/'execution_receipt.json', base.E_EXEC_SHA)
    eb = bound(E/'predictions_receipt.json'); ea = bound(E/'independent_cpu_review.json')
    em = bound(E/'rgb_metrics.json', ee['metrics_sha256'])
    require(ee['status'] == 'completed' and ee['plan_sha256'] == base.E_PLAN_SHA
            and ee['bound_inputs_and_sources_unchanged'] and ea['status'] == 'passed'
            and ea['execution_receipt_sha256'] == base.E_EXEC_SHA and ea['plan_sha256'] == base.E_PLAN_SHA
            and eb['status'] == 'all_50_predictions_complete_before_gt' and eb['gt_payload_reads'] == 0
            and eb['plan_sha256'] == base.E_PLAN_SHA and eb['predictions'] == ee['predictions']
            and eb['utc'] <= ee['source_scoring_started_utc'], 'Audited completed E required')
    parents = {}
    for folder, expected in ((LAYERS, LAYER_SHA), (FACTORIAL, FACTORIAL_SHA)):
        plan = bound(folder/'plan.json', expected); receipts = {}
        for stage in ('render', 'score'):
            receipt = bound(folder/f'{stage}_execution_receipt.json')
            launch = bound(folder/f'{stage}_launch_receipt.json')
            base.natural(receipt, launch, expected, sha(folder/f'{stage}_execution_receipt.json'))
            require(receipt['sources_and_inputs_unchanged'], 'Parent runtime invariance failed')
            receipts[stage] = receipt
        audit = bound(folder/'independent_summary_check.json')
        if folder == LAYERS:
            require(audit['status'] == 'completed' and audit['maximum_absolute_error'] <= 1e-12
                    and audit['source_hashes'][str(folder/'score_execution_receipt.json')]
                    == sha(folder/'score_execution_receipt.json'), 'Original summary audit required')
        else:
            require(audit['status'] == 'passed' and audit['maximum_absolute_difference'] <= 1e-12
                    and audit['input_score_receipt_sha256'] == sha(folder/'score_execution_receipt.json'),
                    'Factorial summary audit required')
        parents[str(folder)] = (plan, receipts)
    lp, lr = parents[str(LAYERS)]
    lb = bound(LAYERS/'predictions_receipt.json')
    require(lb['status'] == 'all_150_pngs_before_GT' and lb['VAL_payload_reads'] == 0
            and lb['finished_utc'] <= lr['score']['scoring_started_utc']
            and lr['score']['actual_numerics'] == base.NUMERICS, 'Completed original layer prediction barrier required')
    ld = lr['score']['metrics'][MEMBERS[0]]; lm = bound(ld['path'], ld['sha256'])
    md = ep['original_evaluations']['mcmc']; mroot = Path(md['directory'])
    mr = bound(mroot/'execution_receipt.json', md['receipt_sha256'])
    mm = bound(mroot/'official_metrics.json', md['metrics_sha256'])
    require(mr['official_metrics_sha256'] == md['metrics_sha256'], 'MCMC metrics are not bound to receipt')
    mm = attach_mcmc_lineage(mm, mr, ep['views'])
    views = matching_views(ep, lp, eb['predictions'], lb['records'], em, lm, mm)
    for v in views:
        require(Path(v['members'][MEMBERS[0]]['path']) == LAYERS/'predictions'/MEMBERS[0]/v['name']
                and Path(v['members']['mcmc']['path']) == mroot/'rgb'/v['name'], 'Unexpected fixed member path')
        for item in v['members'].values():
            require(sha(item['path']) == item['sha256'], 'Member PNG bytes changed')
            inputs[item['path']] = item['sha256']
    source = E/'source_snapshot'; require(helper.source_files(source) == ep['source_hashes'], 'Original scoring source changed')
    output = output.resolve(); require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk output required')
    snapshot = output/'source_snapshot'; snapshot.mkdir(parents=True)
    for rel, expected in ep['source_hashes'].items():
        if not rel.startswith('bridge_rgs/'):
            continue
        require(sha(source/rel) == expected, 'Scorer dependency changed')
        dest = snapshot/rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source/rel, dest)
    for path in (BASE, source/'evaluate_fixed_rgb_ensemble.py', Path(__file__),
                 ROOT/'tests/test_ibgs_member_replacement.py', ROOT/'docs/ibgs_member_replacement_protocol.md'):
        shutil.copy2(path, snapshot/path.name); inputs[str(path)] = sha(path)
    for path in [*map(Path, ep['perceptual_weights']), ROOT/'uv.lock']:
        inputs[str(path)] = sha(path)
    require(not torch.cuda.is_initialized() and not {v['source_image_path'] for v in views} & inputs.keys(),
            'No CUDA or GT payload access in prepare')
    references = {'E_rgb': {'path': str(E/'rgb_metrics.json'), 'sha256': ee['metrics_sha256']},
                  MEMBERS[0]: ld, 'mcmc': {'path': str(mroot/'official_metrics.json'), 'sha256': md['metrics_sha256']}}
    plan = {'specification': base.SPEC, 'output': str(output), 'source_snapshot': str(snapshot),
                'source_hashes': helper.source_files(snapshot), 'input_hashes': inputs, 'views': views,
                'reference_metrics': references, 'reference_fingerprint': ep['reference_fingerprint'],
                'scoring_protocol': lp['scoring_protocol'], 'scorer_sha256': helper.SCORER_SHA,
                'helper_sha256': base.HELPER_SHA, 'scoring_numerics': base.NUMERICS,
                'runtime_versions': ep['runtime_versions'], 'perceptual_weights': ep['perceptual_weights'],
                'external_timeout_seconds': 360, 'prepare_gt_payload_reads': 0, 'prepare_cuda_initialized': False,
                'lineage_note': 'E has audited historical completion but no root launch file; layers/factorial have natural-exit receipts',
                'env': {**ep['env'], 'PYTHONPATH': str(snapshot)}}
    base.write(output/'plan.json', plan)
    print(json.dumps({'plan_sha256': sha(output/'plan.json'), 'output': str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--prepare', type=Path); operation.add_argument('--execute', type=Path)
    parser.add_argument('--expected-plan-sha256'); args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare); return
    require(args.expected_plan_sha256 and sha(args.execute) == args.expected_plan_sha256, 'Explicit plan SHA required')
    plan = read(args.execute); snapshot = Path(plan['source_snapshot'])
    require(Path(__file__).resolve() == snapshot/Path(__file__).name, 'Execute frozen wrapper')
    base = load_base(snapshot/BASE.name)
    base.execute(args.execute, args.expected_plan_sha256)


if __name__ == '__main__':
    main()
