"""Read-only CPU endpoint diagnostics, never a performance/adoption test.

Inspect Adam epsilon attenuation and per-Gaussian endpoint contrast after the
entire fixed experiment and its independent audit have naturally completed.
These unweighted parameter statistics do not identify pixel-error causes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def describe(tensor):
    values = tensor.detach().to(dtype=torch.float64).flatten().numpy()
    if not len(values):
        return {'count': 0, 'quantiles': None}
    assert np.isfinite(values).all()
    return {'count': len(values), 'quantile_levels': [0, .1, .5, .9, .99, 1],
            'quantiles': np.quantile(values, [0, .1, .5, .9, .99, 1]).tolist()}


def inspect(run):
    started = time.monotonic()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    run = Path(run).resolve()
    output = run / 'optimizer_endpoint_diagnostic.json'
    assert not output.exists(), 'Do not overwrite a prior diagnostic'
    plan = read(run / 'plan.json')
    launch = read(run / 'launch_receipt.json')
    execution = read(run / 'execution_receipt.json')
    audit = read(run / 'independent_cpu_review.json')
    assert launch['exit_code'] == 0 and launch['natural_completion'] is True
    assert launch['execution_receipt_sha256'] == sha(run / 'execution_receipt.json')
    assert execution['status'] == 'completed' and audit['status'] == 'passed'
    assert execution['plan_sha256'] == audit['plan_sha256'] == sha(run / 'plan.json')
    bindings = {str(run / p): sha(run / p) for p in
                ('plan.json', 'launch_receipt.json', 'execution_receipt.json', 'independent_cpu_review.json')}
    results = []
    for arm in ('marginal', 'point', 'integrated'):
        receipt = read(run / arm / 'training_receipt.json')
        checkpoint = run / arm / 'final_delta.pt'
        bindings[str(checkpoint)] = receipt['delta_sha256']
        assert sha(checkpoint) == receipt['delta_sha256']
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        assert state['arm'] == arm and state['step'] == plan['specification']['steps'] == 2000
        partition = state['partition_state']
        optimizer = state['stage_optimizer_state']
        names = (('inside_logits', 'outside_logits'), ('direction', 'offset_raw', 'width_raw'))
        parameters = {}
        for group, labels in zip(optimizer['param_groups'][1:], names, strict=True):
            for pid, name in zip(group['params'], labels, strict=True):
                if pid not in optimizer['state']:
                    assert arm == 'marginal' and name == 'direction'
                    parameters[name] = {'state_present': False}
                    continue
                moment = optimizer['state'][pid]
                step = float(moment['step'])
                beta1, beta2 = group['betas']
                rms = (moment['exp_avg_sq'].double() / (1 - beta2**step)).sqrt()
                active = rms > 0
                attenuation = rms[active] / (rms[active] + group['eps'])
                last_update = (group['lr'] * moment['exp_avg'].double() / (1 - beta1**step)
                               / (rms + group['eps'])).abs()
                parameters[name] = {
                    'state_present': True, 'step': step, 'eps': group['eps'], 'lr': group['lr'],
                    'total_components': rms.numel(), 'nonzero_second_moment_components': int(active.sum()),
                    'epsilon_attenuation_nonzero_components': describe(attenuation),
                    'fraction_attenuation_below_point1': float((attenuation < .1).double().mean()) if len(attenuation) else None,
                    'fraction_attenuation_below_point5': float((attenuation < .5).double().mean()) if len(attenuation) else None,
                    'last_adam_update_abs_all_components': describe(last_update),
                }
        inside = partition['inside_logits'].double().softmax(-1)
        outside = partition['outside_logits'].double().softmax(-1)
        normal = torch.nn.functional.normalize(partition['direction'].double(), dim=-1, eps=1e-8)
        results.append({
            'arm': arm, 'optimizer': parameters,
            'endpoint_total_variation_per_gaussian': describe((inside - outside).abs().sum(-1) / 2),
            'normal_angle_from_initial_e1_radians': describe(normal[:, 0].clamp(-1, 1).acos()),
            'offset': describe(2 * partition['offset_raw'].double().tanh()),
            'half_width': describe(.05 + torch.nn.functional.softplus(partition['width_raw'].double())),
        })
    for path, digest in bindings.items():
        assert sha(path) == digest
    assert not torch.cuda.is_initialized()
    result = {'status': 'completed', 'script_sha256': sha(__file__), 'bindings': bindings,
              'arms': results, 'new_renders': 0, 'new_optimizer_steps': 0, 'image_or_label_payload_reads': 0,
              'elapsed_seconds': time.monotonic() - started,
              'scope': 'Descriptive, Gaussian/component-count weighted. The epsilon factor compares the same saved Adam moments with eps=0; it is not the trajectory of a different optimizer, a convergence test, visibility weighting, or evidence of a performance cause.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'completed', 'report': str(output), 'sha256': sha(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    inspect(parser.parse_args().run)
