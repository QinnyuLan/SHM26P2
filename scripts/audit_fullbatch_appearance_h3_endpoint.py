"""Read-only CPU audit of a naturally completed H3 full-batch appearance run.

No renderer, GT decoding, optimizer, or active-checkpoint access. The output
attests the fixed training transaction, never the separate VAL adoption gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

BASE_SHA = '22bc8a2ddb260f93cb01b17857c97b2bb0873038efdb9318545cb2bdbb045226'
LOSS_SHA = '1e114069c2fdb181e243d7cb8a986659b2e229d2ec49c34821cb45013d9cf2b2'
HELPER_SHA = '10c00b1728e7d1e3773219f18b961065996d8c3a31824700333d8987bfefa746'
FP64_HELPER_SHA = '96b29436effb4912862d78895f8f440cd59308c4042c9c257287fef50f158131'
KEYS = ('splats.sh0', 'splats.sh_rest', 'background_logits')
META_KEYS = ('format_version', 'config', 'step', 'scene_scale', 'feature_dim', 'sh_degree',
             'refiner_config', 'pixel_protocol', 'manifest_sha256')
ALPHAS = (1., .5, .25)
KIND = 'fullbatch_appearance_inference_or_warmstart'
PROTOCOLS = {
    'conditional_fullbatch_appearance_h3_v2': {
        'status': 'locked_authorized_after_h3_objective_diagnostic',
        'specification_sha256': '25ad2cd5a3674313206d827c8d789d6f776e0d54c1176d7188a932cafac5c740',
        'objective_dtype': 'float32',
        'document': 'fullbatch_appearance_h3_protocol.md',
        'diagnostic_protocol': 'h3_full_original_objective_rms_armijo_precheck_v1'},
    'conditional_fullbatch_appearance_h3_fp64_v1': {
        'status': 'locked_authorized_after_h3_fp64_objective_diagnostic',
        'specification_sha256': '986e66ad328148a93632f19cb9eb8b9cb5c3f1029614582b2e92aa449d05318f',
        'objective_dtype': 'float64',
        'document': 'fullbatch_appearance_h3_fp64_protocol.md',
        'diagnostic_protocol': 'h3_full_original_objective_fp64_loss_rms_armijo_precheck_v1'},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def floor(loss):
    return max(1e-7, 1e-5*abs(loss))


def validate_protocol(plan):
    spec = plan['specification']
    contract = PROTOCOLS.get(spec.get('protocol'))
    require(contract is not None, 'Unknown full-batch protocol')
    spec_sha = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(plan['status'] == contract['status'] and spec_sha == contract['specification_sha256'],
            'Exact protocol/dtype/numerical/budget specification differs')
    require(Path(plan['protocol_document']['path']).name == contract['document'], 'Wrong bound protocol document')
    if contract['objective_dtype'] == 'float64':
        require(spec['objective_dtype'] == 'float64' and spec['parameter_renderer_warp_dtype'] == 'float32'
                and plan['source_hashes'].get('bridge_rgs/fullbatch_appearance_fp64.py') == FP64_HELPER_SHA,
                'FP64 image-loss implementation/dtype differs')
    return contract


def validate_diagnostic_precision(contract, diagnostic_plan, diagnostic, plan_sha):
    require(diagnostic_plan['specification']['protocol'] == contract['diagnostic_protocol']
            and diagnostic['plan_sha256'] == plan_sha
            and diagnostic['status'] == 'completed' and diagnostic['technical_gate_passed'] is True,
            'A different precision or failed diagnostic cannot authorize this protocol')


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def completed_receipts(output, launch_path, plan_sha):
    """Called before any stat/hash/load of the candidate or base checkpoint."""
    receipt, launch = read(output/'execution_receipt.json'), read(launch_path)
    require(receipt.get('status') == 'completed' and receipt.get('plan_sha256') == plan_sha,
            'Full-batch execution is not completed; checkpoint unopened')
    require(launch.get('status') == 'completed' and launch.get('exit_code') == 0
            and launch.get('plan_sha256') == plan_sha,
            'Natural outer exit0 is not confirmed; checkpoint unopened')
    return receipt, launch


def audit_passes(rows, receipt):
    """Independently replay scalar accept/reject accounting, not optimization."""
    passes = partial_renders = accepted = renders = 0
    baseline = last_accepted = None
    pending_trial = None
    alpha_index = 0
    last_baseline_on_time = False
    transactions = []
    pass_seconds = 0.
    for row in rows:
        if 'role' in row:
            require(row['role'] in ('baseline_gradient', *(f'trial_alpha_{a}' for a in ALPHAS)),
                    'Unknown pass role')
            require(isinstance(row['renders'], int) and not isinstance(row['renders'], bool)
                    and 0 <= row['renders'] <= 350 and math.isfinite(row['seconds']) and row['seconds'] >= 0,
                    'Invalid pass cost')
            renders += row['renders']
            pass_seconds += row['seconds']
            if row['complete']:
                passes += 1
                require(row['renders'] == 350 and row['pass'] == passes and passes <= 40
                        and math.isfinite(row['loss']), 'Incomplete/nonfinite complete pass')
            else:
                require(row['loss'] is None, 'Partial pass must not report a usable loss')
                partial_renders += row['renders']
            if row['role'] == 'baseline_gradient':
                require(pending_trial is None, 'Unaccounted preceding trial')
                baseline = row['loss']
                last_baseline_on_time = bool(row['complete'] and row['on_time'])
                alpha_index = 0
                if last_baseline_on_time and last_accepted is not None:
                    require(baseline <= last_accepted+floor(last_accepted), 'Accepted baseline replay regressed')
            else:
                require(last_baseline_on_time and pending_trial is None and alpha_index < len(ALPHAS)
                        and row['role'] == f'trial_alpha_{ALPHAS[alpha_index]}',
                        'Trial has no matching baseline/registered alpha')
                pending_trial = row
            continue
        require(last_baseline_on_time and alpha_index < len(ALPHAS)
                and row['alpha'] == ALPHAS[alpha_index] and row['accepted_step_before'] == accepted
                and row['baseline'] == baseline and row['completed_passes'] == passes,
                'Transaction ordering/baseline/counters differ')
        actual_slope, nominal_slope = row['g_dot_actual_delta'], row['g_dot_d']
        require(math.isfinite(actual_slope) and math.isfinite(nominal_slope) and nominal_slope < 0,
                'Nonfinite/non-descent proposed slope')
        require(set(row['displacement_absmax']) == set(KEYS)
                and all(math.isfinite(x) and x >= 0 for x in row['displacement_absmax'].values()),
                'Invalid realized FP32 displacement record')
        candidate = row['candidate']
        on_time = False
        if pending_trial is not None:
            require(row['candidate'] == pending_trial['loss'], 'Trial value differs from complete/partial pass')
            on_time = bool(pending_trial['complete'] and pending_trial['on_time'])
        else:
            require(candidate is None and actual_slope >= 0, 'Missing candidate forward for descent trial')
        should_accept = (on_time and actual_slope < 0 and candidate <= baseline+1e-4*actual_slope
                         and baseline-candidate >= floor(baseline))
        require(row['accepted'] is should_accept, 'Accepted flag differs from independent Armijo/floor')
        if should_accept:
            accepted += 1
            require(accepted <= 20, 'Acceptance budget exceeded')
            last_accepted = candidate
            last_baseline_on_time = False
        transactions.append({'alpha': row['alpha'], 'actual_slope': actual_slope,
                             'baseline': baseline, 'candidate': candidate,
                             'accepted': should_accept, 'decrease_floor': floor(baseline)})
        alpha_index += 1
        pending_trial = None
    require(pending_trial is None, 'Unrecorded last trial transaction')
    require(passes == receipt['complete_passes'] and accepted == receipt['accepted_steps']
            and partial_renders == receipt['partial_renders'] and renders == receipt['renderer_calls']
            and renders == 350*passes+partial_renders, 'Receipt and independently counted passes/renders differ')
    require(receipt['last_accepted_train_loss'] == last_accepted,
            'Receipt does not identify the last completely accepted objective')
    seconds = receipt['optimization_seconds']
    require(math.isfinite(seconds) and seconds >= 0 and pass_seconds <= seconds+1e-6,
            'Invalid optimization timing')
    if seconds > 360:
        require(receipt['stop_reason'] in ('complete_pass_or_time_budget',
                    'time_budget_trial_pass_discarded', 'time_budget_gradient_pass_discarded'),
                '360-second budget exceeded without an explicit timed stop')
    require(receipt['stop_reason'] in ('accepted_step_limit', 'complete_pass_or_time_budget',
            'time_budget_trial_pass_discarded', 'time_budget_gradient_pass_discarded',
            'three_fixed_trials_rejected', 'zero_full_gradient'), 'Unknown stop reason')
    return {'complete_passes': passes, 'partial_renders': partial_renders, 'renderer_calls': renders,
            'accepted_steps': accepted, 'last_accepted_train_loss': last_accepted,
            'transactions': transactions, 'optimization_seconds': seconds,
            'deadline_scope': '360s is checked at view boundaries. No partial pass is accepted; '
                              'per-view durations are not retained, so a strict one-view overrun bound '
                              'is a frozen-source contract, not independently recovered timing.'}


def compare_checkpoint(base, candidate, plan, receipt):
    """Exact tensor/metadata allow-list; never materialize or repair missing state."""
    require(base.get('mip_filter_config') is None and base.get('mcmc_reference_config') is None,
            'Not the fixed historical off-profile base')
    expected_keys = {k for k in META_KEYS if k in base} | {
        'model', 'training_cameras', 'checkpoint_kind', 'appearance_optimization'}
    require(set(candidate) == expected_keys and candidate['checkpoint_kind'] == KIND,
            'Wrong inference checkpoint kind or leaked training state')
    for key in META_KEYS:
        require((key in candidate) == (key in base)
                and (key not in base or candidate[key] == base[key]), 'Base metadata changed/invented: ' + key)
    require(set(candidate['model']) == set(base['model']), 'Model tensor schema changed')
    changed, differences = [], {}
    for key, old in base['model'].items():
        new = candidate['model'][key]
        require(isinstance(new, torch.Tensor) and new.shape == old.shape and new.dtype == old.dtype
                and new.device.type == 'cpu' and bool(torch.isfinite(new).all()),
                'Model shape/dtype/finite mismatch: ' + key)
        exact = torch.equal(new, old)
        require(key in KEYS or exact, 'Frozen model tensor changed: ' + key)
        if not exact:
            changed.append(key)
        if key in KEYS:
            delta = new.double()-old.double()
            differences[key] = {'changed': not exact, 'max_abs': float(delta.abs().max()),
                                'l2': float(delta.norm()), 'before_sha256': tensor_sha(old),
                                'after_sha256': tensor_sha(new)}
    require(set(KEYS) <= set(base['model']), 'Missing appearance keys')
    require(torch.equal(candidate['training_cameras'], base['training_cameras']), 'Base TRAIN cameras changed')
    meta = candidate['appearance_optimization']
    require(meta['protocol'] == plan['specification']['protocol']
            and meta['base'] == plan['base'] and meta['manifest_observed'] == plan['manifest']
            and meta['source_files_sha256'] == plan['source_hashes']
            and meta['stage_config'] == plan['specification']
            and meta['base_manifest_declared_sha256'] == base.get('manifest_sha256')
            and meta['ordinary_resume_allowed'] is False and meta['optimizer_state_available'] is False,
            'Inference-stage provenance/scope differs')
    for key in ('accepted_steps', 'complete_passes', 'last_accepted_train_loss', 'stop_reason',
                'training_camera_mapping'):
        require(meta[key] == receipt[key], 'Saved endpoint metadata differs: ' + key)
    return {'changed_keys': sorted(changed), 'appearance_differences': differences,
            'all_other_model_tensors_exact': True, 'training_cameras_exact': True,
            'base_metadata_preserved': True,
            'historical_absent_fields_still_absent': [k for k in ('manifest_sha256', 'pixel_protocol') if k not in base],
            'ordinary_resume_state_not_copied': True}


def verify_bindings(plan):
    contract = validate_protocol(plan)
    snapshot = Path(plan['source_snapshot']).resolve()
    actual = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*')
              if p.is_file() and p.suffix in {'.py', '.md'}}
    require(actual == plan['source_hashes'], 'Snapshot source/document tree changed or contains unbound files')
    require(actual.get('bridge_rgs/losses.py') == LOSS_SHA
            and actual.get('bridge_rgs/fullbatch_appearance.py') == HELPER_SHA,
            'Fixed numerical implementation changed')
    require(plan['base']['sha256'] == BASE_SHA, 'Wrong selected H3 base')
    files = dict(plan['input_hashes'])
    for key in ('base', 'manifest', 'fixed_ssim_diagnostic_receipt', 'h3_diagnostic_plan',
                'h3_diagnostic_receipt', 'diagnostic_review', 'protocol_document'):
        value = plan[key]
        require(value['path'] not in files or files[value['path']] == value['sha256'],
                'Conflicting binding for ' + key)
        files[value['path']] = value['sha256']
    for path, expected in files.items():
        require(sha(path) == expected, 'Bound input changed: ' + path)
    diagnostic_plan = read(plan['h3_diagnostic_plan']['path'])
    diagnostic = read(plan['h3_diagnostic_receipt']['path'])
    validate_diagnostic_precision(contract, diagnostic_plan, diagnostic, plan['h3_diagnostic_plan']['sha256'])
    return files


def audit(plan_path, expected_plan_sha256, launch_path, output_path):
    plan_path, launch_path, output_path = (Path(p).resolve() for p in (plan_path, launch_path, output_path))
    require(not output_path.exists(), 'Refuse overwriting an audit')
    require(sha(plan_path) == expected_plan_sha256, 'Wrong immutable plan SHA')
    plan = read(plan_path)
    output = Path(plan['output']).resolve()
    receipt, launch = completed_receipts(output, launch_path, expected_plan_sha256)
    require(not torch.cuda.is_initialized(), 'CPU-only audit')
    torch.set_num_threads(8)
    contract = validate_protocol(plan)
    files = verify_bindings(plan)
    manifest = read(plan['manifest']['path'])
    train = [v for v in manifest['views'] if v['split'] == 'train']
    names = [v['name'] for v in train]
    require(len(names) == len(set(names)) == 350 and names == plan['training_camera_names'],
            'TRAIN camera name/index mapping changed')
    required_rgb = {str((Path(plan['root'])/v['source_image_path']).resolve()) for v in train}
    require(required_rgb <= set(plan['input_hashes']), 'Not all original TRAIN RGB bytes bound')
    for key in ('frozen_tensors_exact', 'base_training_cameras_exact', 'in_memory_base_restored_exact'):
        require(receipt.get(key) is True, 'Runtime restoration attestation missing: ' + key)
    require(receipt.get('ordinary_resume_allowed') is False, 'Endpoint incorrectly advertises strict resume')
    rows = [json.loads(line) for line in (output/'passes.jsonl').read_text().splitlines()]
    cost = audit_passes(rows, receipt)
    base = torch.load(plan['base']['path'], map_location='cpu', weights_only=False, mmap=True)
    cameras = base['training_cameras']
    original = torch.tensor(np.asarray([v['w2c_original'] for v in train]), dtype=torch.float32)
    mapping = receipt['training_camera_mapping']
    require(cameras.shape == (350, 4, 4) and cameras.dtype == torch.float32
            and bool(torch.isfinite(cameras).all()) and mapping['names_in_checkpoint_index_order'] == names
            and mapping['camera_bytes_sha256'] == tensor_sha(cameras)
            and mapping['different_from_original_pose_count'] == int((cameras != original).any(2).any(1).sum())
            and mapping['max_abs_difference_from_original_pose'] == float((cameras-original).abs().max()),
            'Saved camera mapping differs from immutable base')
    require((Path(plan['root'])/base['config']['manifest']).resolve() == Path(plan['manifest']['path']).resolve(),
            'Base/observed manifest lineage differs')
    model_result = None
    artifacts = [plan_path, launch_path, output/'execution_receipt.json', output/'passes.jsonl']
    if receipt['accepted_steps']:
        require(receipt['checkpoint_written'] is True and Path(receipt['checkpoint']).resolve() == output/'final.pt',
                'Accepted endpoint was not saved at the fixed path')
        candidate_path = Path(receipt['checkpoint'])
        require(sha(candidate_path) == receipt['checkpoint_sha256'], 'Candidate bytes differ from completed receipt')
        candidate = torch.load(candidate_path, map_location='cpu', weights_only=False, mmap=True)
        model_result = compare_checkpoint(base, candidate, plan, receipt)
        artifacts.append(candidate_path)
        require(sha(candidate_path) == receipt['checkpoint_sha256'], 'Candidate changed during CPU audit')
    else:
        require(receipt['checkpoint_written'] is False and not (output/'final.pt').exists(),
                'Zero-acceptance run must not publish a duplicate candidate')
    verify_bindings(plan)
    require(sha(plan_path) == expected_plan_sha256 and not torch.cuda.is_initialized(),
            'Plan changed or CUDA initialized during CPU audit')
    result = {'status': 'passed', 'audit_utc': datetime.now(UTC).isoformat(),
              'plan_sha256': expected_plan_sha256, 'output': str(output),
              'base_sha256': BASE_SHA, 'candidate_checkpoint_sha256': receipt.get('checkpoint_sha256'),
              'source_hashes': plan['source_hashes'], 'bound_file_count': len(files),
              'protocol': plan['specification']['protocol'], 'objective_dtype': contract['objective_dtype'],
              'parameter_renderer_warp_dtype': 'float32',
              'source_and_input_endpoint_bytes_exact': True, 'checkpoint_contract': model_result,
              'cost_and_transactions': cost, 'training_camera_mapping_recomputed': True,
              'outer_exit_code': launch['exit_code'], 'auditor_sha256': sha(__file__),
              'artifact_hashes': {str(p): sha(p) for p in artifacts},
              'evidence_limits': [('Runtime rollback attested by completed worker; CPU comparison independently '
                                   'checks saved tensors but cannot reconstruct discarded trials.'),
                                  ('No TRAIN objective or VAL metric is recomputed. Scalar acceptance is replayed '
                                   'from all logged full passes, not inferred from endpoint quality.'),
                                  ('No semantic/RGB adoption conclusion. The fixed final plain and H+0.5 official '
                                   'evaluations and their gate are separate.')]}
    with output_path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--launch-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.plan, args.expected_plan_sha256, args.launch_receipt, args.output)
    print(json.dumps({'status': result['status'], 'output': str(args.output.resolve()), 'sha256': sha(args.output)}))


if __name__ == '__main__':
    main()
