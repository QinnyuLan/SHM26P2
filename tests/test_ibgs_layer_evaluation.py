"""Small CPU contracts; no training endpoint, source cache or target payload."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
ENTRY = HERE/'evaluate_ibgs_layer_heads.py'
if not ENTRY.exists():
    ENTRY = HERE.parent/'scripts/evaluate_ibgs_layer_heads.py'
spec = importlib.util.spec_from_file_location('layer_eval_contract', ENTRY)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_fixed_population_barrier_rejects_missing_duplicate_or_other_arm():
    names = [f'{i:03}.png' for i in range(50)]
    records = [{'arm': a, 'name': n} for a in worker.ARMS for n in names]
    worker.prediction_barrier(records, names)
    for bad in (records[:-1], records[:-1]+[records[0]],
                records[:-1]+[{'arm': 'best', 'name': names[-1]}]):
        with pytest.raises(ValueError, match='150'):
            worker.prediction_barrier(bad, names)


def test_natural_bound_completion_required_and_cannot_be_fabricated(tmp_path):
    (tmp_path/'plan.json').write_text('{}')
    plan_sha = worker.sha(tmp_path/'plan.json')
    receipt = {'status': 'completed', 'plan_sha256': plan_sha}
    (tmp_path/'render_execution_receipt.json').write_text(json.dumps(receipt))
    launch = {'status': 'completed', 'plan_sha256': plan_sha, 'exit_code': 0,
              'natural_completion': False,
              'execution_receipt_sha256': worker.sha(tmp_path/'render_execution_receipt.json')}
    path = tmp_path/'render_launch_receipt.json'; path.write_text(json.dumps(launch))
    with pytest.raises(ValueError):
        worker.natural(tmp_path, 'render')
    launch['natural_completion'] = True; path.write_text(json.dumps(launch))
    assert worker.natural(tmp_path, 'render') == receipt
    (tmp_path/'plan.json').write_text('{"changed":true}')
    with pytest.raises(ValueError):
        worker.natural(tmp_path, 'render')


def endpoint():
    parent = {'specification': {'train_steps': 6000}, 'checkpoint': {'sha256': 'fixed'},
              'cache_manifest_sha256': 'cache'}
    saved = {'protocol': 'ibgs_fixed_layer_heads_v1', 'arm': 'top4_mass', 'step': 6000,
             'plan_sha256': 'plan', 'field_checkpoint': parent['checkpoint'],
             'cache_manifest_sha256': 'cache', 'specification': parent['specification'],
             'head': {'toy': torch.ones(2)}}
    return saved, parent


@pytest.mark.parametrize('change', ['step', 'arm', 'field', 'nonfinite'])
def test_only_fixed_finite_endpoint_identity(change):
    saved, parent = endpoint()
    worker.validate_endpoint(saved, 'top4_mass', parent, 'plan', parameter_count=2)
    if change == 'step':
        saved['step'] = 5999
    elif change == 'arm':
        saved['arm'] = 'median4_mass'
    elif change == 'field':
        saved['field_checkpoint'] = {'sha256': 'other'}
    else:
        saved['head']['toy'][0] = float('nan')
    with pytest.raises(ValueError):
        worker.validate_endpoint(saved, 'top4_mass', parent, 'plan', parameter_count=2)


def test_inference_uses_contiguous_chosen_slots_same_physical_matrices_and_zero_padding():
    h, w = 2, 3
    interleaved = torch.arange(h*w*2*4, dtype=torch.int32).reshape(h, w, 2, 4)
    package = {'top4': {'ids': interleaved[:, :, 1], 'depth': interleaved[:, :, 1].float()+1,
                        'weights': torch.ones(h, w, 2, 4)[:, :, 1]*.05},
               'raw': torch.zeros(3, h, w), 'ray': torch.ones(3, h, w),
               'metadata': {'focal': [2., 3.], 'principal': [1., .5]}}
    assert not package['top4']['ids'].is_contiguous()
    camera = SimpleNamespace(image_name='target', world_view_transform=torch.eye(4), camera_center=torch.zeros(3))
    source_matrix = torch.eye(4); source_matrix[0, 3] = -2
    source_camera = SimpleNamespace(world_view_transform=source_matrix.T)
    real_source, absent = {'identity': 'actual'}, {'identity': 'padding'}
    requested = []

    def source_provider(name):
        requested.append(name)
        return real_source

    def builder(ids, depth, weights, rgb, sources, **kwargs):
        assert all(t.is_contiguous() for t in (ids, depth, weights, rgb))
        assert torch.equal(ids, interleaved[:, :, 1])
        assert sources[0] is real_source and all(s is absent for s in sources[1:])
        assert torch.equal(kwargs['ref_to_src'][0], source_matrix)
        assert torch.equal(kwargs['ref_to_src'][1:], torch.eye(4).repeat(3, 1, 1))
        assert torch.equal(kwargs['source_campos'][0], torch.tensor([2., 0, 0]))
        return {'features': torch.zeros(h*w, 4, 4, 7), 'target_weights': weights.reshape(h*w, 4),
                'support': torch.zeros(h*w, 4, 4)}

    def net(features, weights, support, ray, base):
        assert torch.equal(ray, torch.ones(h*w, 3))
        return {'image_pred': base+torch.tensor([.1, .2, .3]), 'active': torch.ones(h*w, dtype=torch.bool),
                'supported_mass': torch.full((h*w,), .025)}

    value, info = worker.infer_layer_head(package, 'top4', net, camera, {'source': source_camera},
        ['source'], source_provider, absent, builder)
    assert requested == ['source'] and info['absent_source_slots'] == 3
    assert torch.equal(value, torch.tensor([.1, .2, .3])[:, None, None].expand(3, h, w))
    with pytest.raises(ValueError, match='distinct'):
        worker.infer_layer_head(package, 'top4', net, camera, {}, ['target'], source_provider, absent, builder)


def test_fixed_primary_counts_and_no_overwrite(tmp_path):
    assert worker.PRIMARY == 'top4_mass'
    assert worker.SPEC['target_calls'] == worker.SPEC['selector_calls'] == 150
    assert worker.SPEC['source_depth_calls'] == 0
    assert len(worker.SPEC['comparisons']) == 4
    path = tmp_path/'receipt.json'; worker.write(path, {'value': 1})
    with pytest.raises(FileExistsError):
        worker.write(path, {'value': 2})


def diagnostic_head():
    from torch import nn

    from bridge_rgs.ibgs_layer_fusion import MassPreservingLayerFusion
    backbone = nn.Module(); backbone.height = 2; backbone.width = 3; backbone.per_view_feat_dim = 32
    backbone.per_view_mlp = nn.Sequential(nn.Linear(7, 32), nn.ReLU())
    backbone.conv_decoder = nn.Conv2d(38, 3, 1)
    with torch.no_grad():
        backbone.per_view_mlp[0].weight.fill_(0); backbone.per_view_mlp[0].weight[:, 0] = 1
        backbone.per_view_mlp[0].bias.fill_(0)
    features = torch.ones(6, 4, 4, 7); features[0, :, :, 0] = -1
    weights = torch.full((6, 4), .1); support = torch.ones(6, 4, 4); support[1] = 0
    return MassPreservingLayerFusion(backbone), (features, weights, support, torch.ones(6, 3), torch.full((6, 3), .5))


def test_passive_feature_probe_preserves_output_rng_and_distinguishes_empty_support():
    net, inputs = diagnostic_head()
    before = net(*inputs); rng = torch.get_rng_state()
    after, stats = worker.observed_head_forward(net, *inputs)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert torch.equal(rng, torch.get_rng_state())
    assert not net.backbone.conv_decoder._forward_pre_hooks
    assert stats['supported_pixels'] == 5 and stats['supported_zero_pool_pixels'] == 1
    assert stats['pooled_zero_pixels'] == 2 and stats['supported_zero_pool_fraction'] == .2
    assert stats['pooled_feature_max_abs'] == pytest.approx(.4)
    assert stats['pooled_feature_mean_abs'] == pytest.approx(.4*4/6)
    assert stats['ray_mean_abs'] == 1 and stats['base_rgb_mean_abs'] == .5


def test_passive_probe_hook_is_removed_when_head_fails():
    net, inputs = diagnostic_head()
    def fail(_):
        raise RuntimeError('synthetic head failure')
    net.backbone.conv_decoder.forward = fail
    with pytest.raises(RuntimeError, match='synthetic'):
        worker.observed_head_forward(net, *inputs)
    assert not net.backbone.conv_decoder._forward_pre_hooks


def test_bound_python_inside_cache_directory_is_not_discarded(tmp_path):
    source = tmp_path/'source'; path = source/'official/scene/__pycache__/__init__.py'
    path.parent.mkdir(parents=True); path.write_text('# legitimate source\n')
    path.with_suffix('.pyc').write_bytes(b'unbound compiled cache')
    hashes = worker.files(source)
    worker.copy_bound_sources(source, tmp_path/'copied', hashes)
    assert (tmp_path/'copied/official/scene/__pycache__/__init__.py').read_text() == path.read_text()
    assert not list((tmp_path/'copied').rglob('*.pyc'))


def test_readonly_head_has_version_counters_under_actual_render_context():
    import ast
    tree = ast.parse(ENTRY.read_text())
    render = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'render_stage')
    contexts = [n.context_expr for n in ast.walk(render) if isinstance(n, ast.withitem)
                and isinstance(n.context_expr, ast.Call) and isinstance(n.context_expr.func, ast.Attribute)
                and isinstance(n.context_expr.func.value, ast.Name) and n.context_expr.func.value.id == 'torch']
    assert len(contexts) == 1
    context = eval(compile(ast.Expression(contexts[0]), str(ENTRY), 'eval'), {'torch': torch})
    with context:
        head = torch.nn.Linear(3, 2).eval().requires_grad_(False)
        before = [(id(p), p._version) for p in head.parameters()]
        result = head(torch.ones(4, 3))
        assert not result.requires_grad
        assert before == [(id(p), p._version) for p in head.parameters()]
