"""Small real-CUDA relocation/birth/resume contract, without images or validation."""
from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from run_official_pair import check_gpu_idle, digest, require, utc, write_receipt

from bridge_rgs.mcmc_reference import (
    MCMCReferenceConfig,
    initialize_reference,
    reference_post_step,
)


def make_optimizers(scene):
    return {name: torch.optim.Adam([value], lr=1e-4, eps=1e-15)
            for name, value in scene.splats.items()}


def main(output):
    require(not output.exists(), 'Refuse existing CUDA contract result')
    gpu = check_gpu_idle()
    started = time.perf_counter()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    values = {'means': torch.arange(60).reshape(20, 3).float() * .01,
              'quats': torch.tensor([1., 0., 0., 0.]).repeat(20, 1),
              'log_scales': torch.full((20, 3), .01).log(),
              'opacity_logits': torch.zeros(20), 'sh0': torch.zeros(20, 1, 3),
              'sh_rest': torch.zeros(20, 15, 3), 'sem_features': torch.zeros(20, 16)}
    scene = SimpleNamespace(splats=torch.nn.ParameterDict({
        k: torch.nn.Parameter(v.cuda()) for k, v in values.items()}),
        semantic_prior_counts=torch.zeros(20, 5, device='cuda'),
        scene_scale=2., mip_filter_config=None)
    optimizers = make_optimizers(scene)
    for name, parameter in scene.splats.items():
        parameter.grad = torch.ones_like(parameter) * .01
        optimizers[name].step()
        optimizers[name].zero_grad(set_to_none=True)
    policy = MCMCReferenceConfig(cap_max=24, refine_start_iter=1, refine_stop_iter=4, refine_every=1)
    state = initialize_reference(scene, optimizers, policy)
    with torch.no_grad():
        scene.splats['opacity_logits'][0] = -10
    before_noise = scene.splats['means'][0].detach().clone()
    reference_post_step(scene, optimizers, state, 1, .00032)
    require(not torch.equal(before_noise, scene.splats['means'][0]),
            'Low-opacity covariance noise did not move the forced test point')
    with torch.no_grad():
        scene.splats['opacity_logits'][0] = -10
    old_parameters = dict(scene.splats.items())
    event = reference_post_step(scene, optimizers, state, 2, .00032)
    require(event['mcmc_relocated_this_step'] == 1 and event['mcmc_added_this_step'] == 1,
            'Forced dead relocation and birth were not both exercised')
    require(scene.semantic_prior_counts.shape == (21, 5)
            and torch.count_nonzero(scene.semantic_prior_counts).item() == 0, 'Invalid prior migration')
    for name, value in scene.splats.items():
        require(value is not old_parameters[name]
                and optimizers[name].param_groups[0]['params'][0] is value
                and old_parameters[name] not in optimizers[name].state,
                'Model and optimizer disagree after replacement: ' + name)
    restored_scene = deepcopy(scene)
    restored_optimizers = make_optimizers(restored_scene)
    for name, opt in optimizers.items():
        restored_optimizers[name].load_state_dict(deepcopy(opt.state_dict()))
    restored_state = initialize_reference(restored_scene, restored_optimizers, policy)
    restored_state.load_state_dict(state.state_dict(), restored_scene, restored_optimizers, expected_step=2)
    rng = torch.cuda.get_rng_state().clone()
    reference_post_step(scene, optimizers, state, 3, .00016)
    torch.cuda.set_rng_state(rng)
    reference_post_step(restored_scene, restored_optimizers, restored_state, 3, .00016)
    for name, value in scene.splats.items():
        require(torch.equal(value, restored_scene.splats[name]), 'CUDA resumed tensor differs: ' + name)
        require(torch.isfinite(value).all().item(), 'Nonfinite CUDA field: ' + name)
        left, right = optimizers[name].state[value], restored_optimizers[name].state[restored_scene.splats[name]]
        require(left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left),
                'CUDA resumed optimizer differs: ' + name)
    torch.cuda.synchronize()
    import bridge_rgs.mcmc_reference as implementation
    result = {'status': 'passed', 'finished_utc': utc(), 'gpu_before': gpu,
              'elapsed_seconds': time.perf_counter() - started, 'policy': asdict(policy),
              'forced_event': event, 'final_counters': state.counters,
              'parameters_and_adam_resume_bitwise_equal': True,
              'source_sha256': digest(implementation.__file__), 'runner_sha256': digest(__file__),
              'scope': 'Toy CUDA kernel/interface check only; no image render/training/VAL/accuracy evidence.'}
    write_receipt(output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
