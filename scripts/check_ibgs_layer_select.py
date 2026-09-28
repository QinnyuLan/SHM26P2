"""Explicit build + forward-only selector check on frozen IBGS captures.

No renderer/model/GT is loaded. Historical byte buffers are parsed with their
original addresses on CPU; only copied typed arrays are uploaded to the selector.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/sky/workspace/SHM2026')
PARENT = Path('/mnt/data/SHM2026/runs/ibgs_thin_rays_capture_v1')
PARENT_PLAN_SHA = '133ca861ff36012289718786c2faed0e96c6009416301e07e318c20fd13e6055'
FIXED_SOURCES = {
    'src/bridge_rgs/ibgs_layer_select.py': 'fd20717c1e8a64f29d0ea827c03e738a9d165634154cafa4bf2471a9bb79fc2a',
    'src/bridge_rgs/cuda/ibgs_layer_select.cpp': '3ff2ec8fc42f4e2ba7abfd5127bac8dbc9b6f4201de82123923108567cc6de7d',
    'src/bridge_rgs/cuda/ibgs_layer_select.cu': '2ac9a3e80e0144e21be081bb89850e923c495087df920066d9e46aa18dd7af06',
    'tests/test_ibgs_layer_select.py': 'cc2e001c0cb0ace7e0bbcb0aaf4ab0ca735809ff50889f7b2aa3bf3b3f81f465',
}
SPEC = {'protocol': 'ibgs_layer_select_check_v1', 'views': 16, 'sample_rays': 73728,
        'modes': ['median4', 'top4'], 'summary_atol': 2e-5, 'summary_rtol': 2e-4,
        'discrete': 'exact IDs, ordinals, last, median-low/high; all status=0',
        'synthetics': ['cap_stop', 'negative_plane', 'power_skip_alpha_cut', 'exact_half_T', 'equal_weight_tie', 'ring'],
        'internal_seconds': 600, 'external_seconds': 660, 'output_byte_limit': 4*1024**3,
        'render_calls': 0, 'GT_decodes': 0, 'backward': 0, 'optimizer_steps': 0,
        'torch': '2.8.0+cu128', 'cuda': '12.8', 'arch': '12.0', 'maximum_build_jobs': 4,
        'scope': 'Forward selector fixture checks; no backend VJP correctness or novelty/performance claim'}


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    payload = json.dumps(clean(value), indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(payload)


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(Path(path).rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.md', '.cpp', '.cu'}}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def numerical_comparison(actual, reference, *, discrete=False):
    a, b = np.asarray(actual), np.asarray(reference)
    require(a.shape == b.shape, 'Comparison shape mismatch')
    finite = np.isfinite(a) & np.isfinite(b)
    if discrete:
        same = finite & (a == b)
    else:
        same = finite & (np.abs(a.astype(np.float64)-b.astype(np.float64))
                         <= SPEC['summary_atol']+SPEC['summary_rtol']*np.abs(b.astype(np.float64)))
    bad = np.flatnonzero(~same.ravel())
    finite_diff = np.abs(a[finite].astype(np.float64)-b[finite].astype(np.float64))
    examples = [{'index': list(np.unravel_index(int(i), a.shape)),
                 'actual': a.ravel()[i].item(), 'reference': b.ravel()[i].item()} for i in bad[:16]]
    return {'passed': not len(bad), 'elements': int(a.size), 'mismatches': len(bad),
            'nonfinite_pairs': int((~finite).sum()),
            'max_abs_error': float(finite_diff.max()) if len(finite_diff) else None,
            'discrete_exact': discrete, 'examples': clean(examples)}


def summarize_selected(selection, final_T, last):
    """Original slot-order FP32 median accumulation, including empty slot0=0."""
    w = np.asarray(selection['weights'], np.float32)
    z = np.asarray(selection['depth'], np.float32)
    order = np.asarray(selection['ordinals'], np.int32)
    require(w.shape == z.shape == order.shape and w.shape[-1] == 4, 'Four matching slots required')
    total = np.zeros(w.shape[:-1], np.float32); depth_sum = total.copy()
    low = order[..., 0].copy(); high = low.copy()
    for k in range(4):
        active = w[..., k] != 0
        total = np.float32(total+w[..., k])
        depth_sum = np.float32(depth_sum+np.float32(w[..., k]*z[..., k]))
        low = np.where(active, np.minimum(low, order[..., k]), low)
        high = np.where(active, np.maximum(high, order[..., k]), high)
    return {'final_T': np.asarray(final_T), 'n_contrib': np.asarray(last), 'median_weight': total,
            'median_depth': depth_sum/np.float32(total+np.float32(1e-8)), 'median_low': low, 'median_high': high}


def reference_slots(trace):
    count = len(trace['xy']); output = {}
    for mode in SPEC['modes']:
        output[mode] = {'ids': np.full((count, 4), -1, np.int32), 'depth': np.zeros((count, 4), np.float32),
                        'weights': np.zeros((count, 4), np.float32), 'ordinals': np.zeros((count, 4), np.int32)}
    for i in range(count):
        start, stop = map(int, trace['offsets'][i:i+2])
        median = np.asarray(trace['median_slots'][i]); valid = median >= 0
        index = start+median[valid]
        for key, source in (('ids', 'ids'), ('depth', 'z'), ('weights', 'w'), ('ordinals', 'order')):
            output['median4'][key][i, valid] = trace[source][index]
        candidates = np.arange(start, stop)[np.isfinite(trace['z'][start:stop]) & (trace['z'][start:stop] > 0)]
        chosen = candidates[np.lexsort((trace['order'][candidates], -trace['w'][candidates]))[:4]]
        for key, source in (('ids', 'ids'), ('depth', 'z'), ('weights', 'w'), ('ordinals', 'order')):
            output['top4'][key][i, :len(chosen)] = trace[source][chosen]
    return output


def synthetic_inputs():
    def case(name, opacity, depths=None, ids=None):
        n = len(opacity); means = np.zeros((n, 2), np.float32)
        conic = np.zeros((n, 4), np.float32); conic[:, [0, 2]] = 1; conic[:, 3] = opacity
        planes = np.zeros((n, 5), np.float32); planes[:, 2] = 1
        planes[:, 4] = -np.asarray(depths if depths is not None else np.ones(n), np.float32)
        return {'name': name, 'means': means, 'conic': conic, 'planes': planes,
                'ids': np.arange(n, dtype=np.int32) if ids is None else np.asarray(ids, np.int32)}
    cut = np.float32(1/255)
    cases = [case('cap_stop', [2., .99, .3]), case('negative_plane', [.5, .5], [-1., 2.]),
             case('power_skip_alpha_cut', [.4, np.nextafter(cut, np.float32(0)), cut]),
             case('exact_half_T', [.5, .2, .2, .2]),
             case('equal_weight_tie', np.asarray([.125, 1/7, 1/6, 1/5, 1/4, 1/3], np.float32)[::-1], ids=np.arange(5, -1, -1)),
             case('ring', [.2]*7)]
    cases[2]['means'][0, 0] = 1; cases[2]['conic'][0, 0] = -1  # Positive power: skipped.
    return cases


def prepare(args):
    require(sha(PARENT/'plan.json') == PARENT_PLAN_SHA, 'Fixed capture parent plan required')
    parent = read(PARENT/'plan.json')
    for receipt_name, launch_name in (('execution_receipt.json', 'launch_receipt.json'),
                                      ('replay_execution_receipt.json', 'replay_launch_receipt.json')):
        receipt, launch = read(PARENT/receipt_name), read(PARENT/launch_name)
        require(receipt['status'] == launch['status'] == 'completed' and launch['natural_completion'] is True
                and launch['exit_code'] == 0 and receipt['plan_sha256'] == launch['plan_sha256'] == PARENT_PLAN_SHA
                and launch['execution_receipt_sha256'] == sha(PARENT/receipt_name), 'Completed capture/replay required')
    require(read(PARENT/'execution_receipt.json')['captures_sha256'] == sha(PARENT/'captures.json')
            and read(PARENT/'replay_execution_receipt.json')['analysis_sha256'] == sha(PARENT/'replay_analysis.json'),
            'Parent artifact binding changed')
    require(all(sha(ROOT/p) == h for p, h in FIXED_SOURCES.items()), 'Final reviewed selector sources required')
    output = args.output.resolve()
    require(output.is_relative_to('/mnt/data') and not output.exists(), 'Fresh data-disk run required')
    require(shutil.disk_usage(output.parent).free >= 6*1024**3, 'Need six GiB free for bounded build/results')
    snapshot = output/'source_snapshot'; snapshot.mkdir(parents=True)
    copies = {ROOT/p: snapshot/(p[4:] if p.startswith('src/') else Path(p).name) for p in FIXED_SOURCES}
    inherited = {}
    for relative in ('bridge_rgs/ibgs_ray_replay.py', 'bridge_rgs/__init__.py'):
        source = Path(parent['source_snapshot'])/relative
        require(sha(source) == parent['source_hashes'][relative], 'Parent replay module changed')
        copies[source] = snapshot/relative; inherited[relative] = parent['source_hashes'][relative]
    copies.update({Path(__file__): snapshot/Path(__file__).name,
                   ROOT/'tests/test_ibgs_layer_select_check.py': snapshot/'test_ibgs_layer_select_check.py',
                   ROOT/'docs/ibgs_layer_select_check_protocol.md': snapshot/'ibgs_layer_select_check_protocol.md',
                   ROOT/'docs/ibgs_layer_select_contract.md': snapshot/'ibgs_layer_select_contract.md'})
    for source, target in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    inputs = {str(PARENT/name): sha(PARENT/name) for name in ('plan.json', 'execution_receipt.json', 'launch_receipt.json',
              'replay_execution_receipt.json', 'replay_launch_receipt.json', 'captures.json', 'replay_analysis.json')}
    inputs[str(ROOT/'uv.lock')] = sha(ROOT/'uv.lock')
    traces = {r['name']: r for r in read(PARENT/'replay_analysis.json')['views']}
    views = []
    for ref in read(PARENT/'captures.json')['records']:
        require(sha(ref['path']) == ref['sha256'], 'Capture record changed')
        record = read(ref['path']); inputs[ref['path']] = ref['sha256']
        needed = ('geometry', 'binning', 'image', 'all_map', 'median_depth')
        for key in needed:
            item = record['arrays'][key]; inputs[item['path']] = item['sha256']
        trace = traces[ref['name']]['trace']; inputs[trace['path']] = trace['sha256']
        views.append({'name': ref['name'], 'record': ref, 'trace': trace})
    require(len(views) == 16, 'Sixteen original views required')
    for path, digest in inputs.items():
        require(sha(path) == digest, 'Reference input changed')
    runtime = {str(Path(p).resolve()): sha(Path(p).resolve()) for p in
               ('/usr/local/cuda/bin/nvcc', shutil.which('c++'), shutil.which('ninja'))}
    require(importlib.metadata.version('torch') == SPEC['torch'], 'Main Torch 2.8/cu128 required')
    plan = {'protocol': SPEC['protocol'], 'specification': SPEC, 'output': str(output),
            'source_snapshot': str(snapshot), 'source_hashes': tree(snapshot), 'inherited_source_hashes': inherited,
            'input_hashes': inputs, 'runtime_sources': runtime, 'views': views,
            'build_directory': str(output/'build'), 'interpreter': str(Path(sys.executable).resolve()),
            'numpy_version': np.__version__, 'environment': {'CUDA_HOME': '/usr/local/cuda',
                 'TORCH_CUDA_ARCH_LIST': '12.0', 'MAX_JOBS': '4', 'TORCH_EXTENSIONS_DIR': str(output/'build'),
                 'TMPDIR': str(output/'tmp')},
            'no_model_image_or_cuda_access_during_prepare': True}
    write(output/'plan.json', plan)
    print(json.dumps({'plan': str(output/'plan.json'), 'sha256': sha(output/'plan.json'),
                      'sources': len(plan['source_hashes']), 'inputs': len(inputs)}))


def verify(plan):
    require(plan['specification'] == SPEC and plan['protocol'] == SPEC['protocol'], 'Changed fixed specification')
    require(tree(plan['source_snapshot']) == plan['source_hashes'], 'Frozen source changed')
    for key in ('input_hashes', 'runtime_sources'):
        require(all(sha(p) == h for p, h in plan[key].items()), f'Changed {key}')
    require(all(plan['source_hashes'].get(k) == v for k, v in plan['inherited_source_hashes'].items()), 'Parent math changed')


def gpu_select(selector, extension, means, conic, planes, ranges, ids, settings):
    import torch
    tensors = [torch.from_numpy(np.array(x, dtype=dtype, order='C', copy=True)).cuda() for x, dtype in
               ((means, np.float32), (conic, np.float32), (planes, np.float32), (ranges, np.int32), (ids, np.int32))]
    result = selector.select_layers(*tensors, width=settings['width'], height=settings['height'],
                                   focal=settings['focal'], principal=settings['principal'], extension=extension, strict=False)
    torch.cuda.synchronize()
    output = {mode: {k: v.cpu().numpy() for k, v in result[mode].items()} for mode in SPEC['modes']}
    output.update({key: result[key].cpu().numpy() for key in ('final_T', 'last_contributor', 'nonfinite_plane_count', 'status')})
    del tensors, result
    return output


def all_passed(checks):
    return all(v['passed'] for v in checks.values())


def run(plan, report):
    os.environ.update(plan['environment'])
    Path(plan['environment']['TMPDIR']).mkdir(); Path(plan['build_directory']).mkdir()
    import torch
    require(torch.__version__ == SPEC['torch'] and torch.version.cuda == SPEC['cuda']
            and torch.cuda.get_device_capability() == (12, 0), 'Frozen Torch/CUDA/sm120 runtime required')
    require(str(Path(sys.executable).resolve()) == plan['interpreter'] and np.__version__ == plan['numpy_version'], 'Interpreter/NumPy changed')
    snapshot = Path(plan['source_snapshot'])
    selector = module(snapshot/'bridge_rgs/ibgs_layer_select.py', 'checked_layer_selector')
    replay = module(snapshot/'bridge_rgs/ibgs_ray_replay.py', 'checked_replay_reference')
    torch.set_num_threads(4); torch.cuda.reset_peak_memory_stats()
    extension = selector.load_extension(build_directory=plan['build_directory'], verbose=True)
    report['build'] = {'binary': {'path': str(Path(extension.__file__).resolve()), 'sha256': sha(extension.__file__)},
                       'source_hashes': {k: v for k, v in plan['source_hashes'].items() if k.endswith(('.cpp', '.cu'))},
                       'torch': torch.__version__, 'cuda': torch.version.cuda,
                       'device': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability())}
    report['extension_calls'] = 0; synthetics = []
    for case in synthetic_inputs():
        settings = {'width': 1, 'height': 1, 'focal': [10., 10.], 'principal': [0., 0.]}
        result = gpu_select(selector, extension, case['means'], case['conic'], case['planes'],
                            np.array([[0, len(case['ids'])]], np.int32), case['ids'], settings)
        report['extension_calls'] += 1
        row = replay.replay_tile(case['ids'], case['means'][case['ids']], case['conic'][case['ids']],
                 np.ones((len(case['ids']), 3), np.float32), case['planes'][case['ids']],
                 np.ones(len(case['ids']), np.float32), np.array([[0, 0]], np.int32),
                 focal=[10., 10.], principal=[0., 0.], background=[0., 0., 0.])[0]
        trace = {'offsets': np.array([0, len(row['ids'])]), 'xy': np.array([[0, 0]]),
                 'median_slots': row['median_slots'][None], **{k: row[k] for k in ('ids', 'z', 'w', 'order')}}
        expected = reference_slots(trace)
        checks = {}
        for mode in SPEC['modes']:
            for key in ('ids', 'ordinals', 'weights', 'depth'):
                checks[mode+'_'+key] = numerical_comparison(result[mode][key][0], expected[mode][key], discrete=key in ('ids', 'ordinals'))
        summary = summarize_selected(result['median4'], result['final_T'], result['last_contributor'])
        for key in summary:
            checks[key] = numerical_comparison(summary[key][0, 0], row['summary'][key], discrete=key in ('n_contrib', 'median_low', 'median_high'))
        checks['status_zero'] = numerical_comparison(result['status'], np.zeros((1, 1), np.int32), discrete=True)
        synthetics.append({'name': case['name'], 'checks': checks, 'passed': all_passed(checks)})
    root = Path(plan['output']); records = []; output = root/'results'; output.mkdir()
    write(output/'synthetic.json', synthetics)
    for view in plan['views']:
        record = read(view['record']['path']); settings = record['settings']
        captured = {key: np.load(record['arrays'][key]['path'], mmap_mode='r', allow_pickle=False)
                    for key in ('geometry', 'binning', 'image', 'all_map', 'median_depth')}
        parsed = replay.parse_buffers(captured['geometry'], captured['binning'], captured['image'],
                    point_count=settings['point_count'], num_rendered=settings['num_rendered'], width=settings['width'],
                    height=settings['height'], addresses=record['addresses'])
        # Do NOT call buffer_views on uploaded historical byte buffers: addresses
        # differ. Parsing happened above at the captured absolute-address ABI.
        actual = gpu_select(selector, extension, parsed['means2d'], parsed['conic_opacity'], captured['all_map'],
                            parsed['ranges'], parsed['point_list'], settings)
        report['extension_calls'] += 1
        summary = summarize_selected(actual['median4'], actual['final_T'], actual['last_contributor'])
        h, w = settings['height'], settings['width']; checks = {}
        for key in summary:
            reference = captured['median_depth'] if key == 'median_depth' else parsed[key].reshape(h, w)
            checks['full_'+key] = numerical_comparison(summary[key], reference, discrete=key in ('n_contrib', 'median_low', 'median_high'))
        for key in ('status', 'nonfinite_plane_count'):
            checks['full_'+key+'_zero'] = numerical_comparison(actual[key], np.zeros((h, w), np.int32), discrete=True)
        with np.load(view['trace']['path'], allow_pickle=False) as loaded:
            trace = {key: loaded[key] for key in ('offsets', 'xy', 'median_slots', 'ids', 'z', 'w', 'order')}
        expected = reference_slots(trace); xy = trace['xy']; x, y = xy.T
        sampled = {'xy': xy}
        for mode in SPEC['modes']:
            for key in ('ids', 'ordinals', 'weights', 'depth'):
                selected = actual[mode][key][y, x]
                checks['sample_'+mode+'_'+key] = numerical_comparison(selected, expected[mode][key], discrete=key in ('ids', 'ordinals'))
                sampled[mode+'_'+key] = selected; sampled['reference_'+mode+'_'+key] = expected[mode][key]
        sample_path = output/(view['name']+'.samples.npz')
        with sample_path.open('xb') as stream:
            np.savez(stream, **sampled)
        summary_path = output/(view['name']+'.summary.npz')
        with summary_path.open('xb') as stream:
            np.savez(stream, **summary, status=actual['status'], nonfinite_plane_count=actual['nonfinite_plane_count'])
        result = {'name': view['name'], 'full_pixels': w*h, 'sample_rays': len(xy), 'checks': checks,
                  'passed': all_passed(checks), 'samples': {'path': str(sample_path), 'sha256': sha(sample_path)},
                  'summary': {'path': str(summary_path), 'sha256': sha(summary_path)}}
        write(output/(view['name']+'.json'), result); records.append(result)
        report['views_checked'] += 1; report['sample_rays_checked'] += len(xy)
        require(sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) <= SPEC['output_byte_limit'], 'Fixed output limit exceeded')
        del actual, summary, captured, parsed, expected, sampled, trace
        gc.collect()
    passed = all(row['passed'] for row in synthetics+records)
    report['numerical_status'] = 'passed' if passed else 'not_passed'
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    require(report['views_checked'] == 16 and report['sample_rays_checked'] == 73728 and report['extension_calls'] == 22,
            'Incomplete fixed check')
    write(root/'analysis.json', {'synthetics': synthetics, 'views': records, 'numerical_status': report['numerical_status'],
                               'scope': SPEC['scope'], 'exact_full_cuda_ledger_claim': False})
    report['analysis_sha256'] = sha(root/'analysis.json')
    # Numerical failure remains explicit; no threshold/recipe fallback.
    require(passed, 'Selector differs from frozen production/CPU references; inspect saved differences')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path); parser.add_argument('--plan', type=Path)
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args()
    if args.prepare:
        prepare(args); return
    require(args.expected_plan_sha256 and sha(args.plan) == args.expected_plan_sha256, 'Explicit plan SHA required')
    plan = read(args.plan); root = Path(plan['output'])
    require(Path(__file__).resolve() == Path(plan['source_snapshot'])/'check_ibgs_layer_select.py', 'Frozen worker only')
    started = time.monotonic(); write(root/'execution_started.json', {'plan_sha256': args.expected_plan_sha256})
    report = {'status': 'failed', 'numerical_status': 'not_completed', 'plan_sha256': args.expected_plan_sha256,
              'views_checked': 0, 'sample_rays_checked': 0, 'render_calls': 0, 'GT_decodes': 0,
              'backward': 0, 'optimizer_steps': 0, 'extension_calls': 0}
    def timeout(*_):
        raise TimeoutError('Fixed build plus check deadline')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(SPEC['internal_seconds'])
    try:
        verify(plan); run(plan, report); verify(plan)
        report.update(status='completed', sources_inputs_unchanged=True)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        signal.alarm(0); report['elapsed_seconds'] = time.monotonic()-started
        write(root/'execution_receipt.json', report)
    print(json.dumps(clean(report), allow_nan=False))


if __name__ == '__main__':
    main()
