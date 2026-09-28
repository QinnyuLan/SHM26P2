"""Small capacity-wrapper contracts; no scene initialization, renderer, or CUDA."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('capacity_runner', ROOT/'scripts/run_rgb_capacity_reference.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def original_config():
    return yaml.safe_load((runner.REFERENCE/'replay_config.yaml').read_text())


def test_exact_two_key_change_keeps_original_mask_and_rng_recipe():
    old = original_config()
    new = dict(old, output='/new/training', max_gaussians=1000000)
    assert set(runner.exact_config_difference(old, new, '/new/training')) == {'output', 'max_gaussians'}
    assert new['region_rgb_weight'] == .15 and new['foreground_allocation_fraction'] == .65
    assert new.get('independent_view_rng', False) is False and new['save_every'] == 5000


@pytest.mark.parametrize('key,value', [('seed', 43), ('densify_stop', 30000), ('save_every', 0),
                                     ('independent_view_rng', True), ('warmstart', '/old/last.pt')])
def test_rejects_recipe_drift(key, value):
    old = original_config()
    new = dict(old, output='/new/training', max_gaussians=1000000)
    new[key] = value
    with pytest.raises(ValueError):
        runner.exact_config_difference(old, new, '/new/training')


def test_observer_preserves_arguments_return_graph_and_rng():
    scene = SimpleNamespace(splats={'means': torch.zeros(4, 3)})
    scalar = torch.tensor(2., requires_grad=True)
    K, pose, marker = object(), object(), object()
    observations = []
    def render(obj, k, p, width, height, *args, **kwargs):
        observations.append((obj, k, p, width, height, args, kwargs))
        return {'rgb': scalar.square(), 'marker': marker}
    counts = dict.fromkeys(('render_calls', 'gaussian_render_sum', 'canvas_pixel_sum',
                            'gaussian_canvas_pixel_sum', 'peak_pre_render_gaussians'), 0)
    np_state, torch_state = copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone()
    observed = runner.render_observer(render, counts)
    result = observed(scene, K, pose, 3, 5, marker, degree=3, semantics=False)
    result['rgb'].backward()
    assert result['marker'] is marker and scalar.grad == 4
    assert observations == [(scene, K, pose, 3, 5, (marker,), {'degree': 3, 'semantics': False})]
    assert counts == {'render_calls': 1, 'gaussian_render_sum': 4, 'canvas_pixel_sum': 15,
                      'gaussian_canvas_pixel_sum': 60, 'peak_pre_render_gaussians': 4}
    assert torch.equal(torch.get_rng_state(), torch_state)
    now = np.random.get_state()
    assert now[0] == np_state[0] and np.array_equal(now[1], np_state[1]) and now[2:] == np_state[2:]


def test_gpu_query_fails_closed_and_only_graphics_is_allowed(monkeypatch):
    for xml in ('<nvidia_smi_log/>', '<nvidia_smi_log><gpu><processes>N/A</processes></gpu></nvidia_smi_log>',
                '<nvidia_smi_log><gpu><processes><process_info><type>C</type></process_info></processes></gpu></nvidia_smi_log>'):
        monkeypatch.setattr(runner.subprocess, 'run', lambda *a, value=xml, **kw: SimpleNamespace(stdout=value))
        with pytest.raises(ValueError):
            runner.gpu_idle()
    xml = '<nvidia_smi_log><gpu><processes><process_info><type>G</type></process_info></processes></gpu></nvidia_smi_log>'
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, value=xml, **kw: SimpleNamespace(stdout=value))
    runner.gpu_idle()


def test_no_preparation_overwrite(tmp_path):
    with pytest.raises(ValueError, match='Preserve'):
        runner.prepare(tmp_path)
