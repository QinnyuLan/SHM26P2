"""Synthetic contracts only: no real cache, compilation, or CUDA call."""
import pytest
import torch

from bridge_rgs.ibgs_layer_evidence import build_layer_evidence


def fixture():
    h = w = 3
    ids = torch.full((h, w, 4), -1, dtype=torch.int64)
    depth = torch.zeros(h, w, 4)
    weights = torch.zeros_like(depth)
    ids[..., 0] = 7
    depth[..., 0] = 2
    weights[..., 0] = .2
    rgb = torch.arange(h*w*3, dtype=torch.float32).reshape(h, w, 3)/100
    sources = []
    for _ in range(4):
        sources.append({'ids': ids.clone(), 'depth': depth.clone(), 'weights': weights.clone()/2,
                        'rgb': rgb.clone(), 'valid': torch.ones(h, w, dtype=torch.bool)})
    args = {'target_ids': ids, 'target_depth': depth, 'target_weights': weights,
            'base_rgb': torch.zeros_like(rgb), 'sources': sources,
            'ref_to_src': torch.eye(4).repeat(4, 1, 1), 'focal': torch.tensor([2., 2.]),
            'principal': torch.tensor([1., 1.]), 'target_world_to_camera': torch.eye(4),
            'target_campos': torch.zeros(3), 'source_campos': torch.zeros(4, 3)}
    return args


def test_identity_integer_centers_fixed_slots_and_original_mass():
    args = fixture()
    out = build_layer_evidence(**args)
    assert out['features'].shape == (9, 4, 4, 7)
    torch.testing.assert_close(out['support'][:, 0], torch.full((9, 4), .1))
    assert torch.equal(out['support'][:, 1:], torch.zeros(9, 3, 4))
    torch.testing.assert_close(out['features'][:, 0, 0, :3], args['sources'][0]['rgb'].reshape(9, 3))
    assert not out['features'][:, 0, :, 3:6].any()
    torch.testing.assert_close(out['features'][:, 0, :, 6], torch.ones(9, 4))
    assert torch.equal(out['target_weights'], args['target_weights'].reshape(9, 4))


def test_subpixel_mass_valid_masks_only_support_not_plain_rgb():
    args = fixture()
    args['ref_to_src'][:, 0, 3] = .5  # center projects to u=1.5, v=1
    for s in args['sources']:
        s['rgb'].fill_(1)
        s['rgb'][1, 2] = 3
        s['weights'][1, 1, 0] = .02
        s['weights'][1, 2, 0] = .06
        s['valid'][1, 1] = False
        s['depth'][1, 2, 0] = 100  # No hidden median or layer depth-agreement predicate.
    out = build_layer_evidence(**args)
    torch.testing.assert_close(out['support'][4, 0], torch.full((4,), .03))
    torch.testing.assert_close(out['features'][4, 0, :, :3], torch.full((4, 3), 2.))
    # Invalid normal tap did not renormalize the remaining half-weight contribution.
    assert out['support'][4, 0, 0] < .06


def test_exact_id_not_nearest_depth_and_invalid_planes_have_zero_mass():
    args = fixture()
    s = args['sources'][0]
    s['ids'][..., 0] = 99
    s['ids'][..., 1] = 7
    s['weights'][..., 1] = .015
    s['depth'][..., 1] = 200
    out = build_layer_evidence(**args)
    torch.testing.assert_close(out['support'][:, 0, 0], torch.full((9,), .015))
    for bad in (0., -1., float('inf'), float('nan')):
        s['depth'][..., 1] = bad
        result = build_layer_evidence(**args)
        assert not result['support'][:, :, 0].any()
        assert not result['features'][:, :, 0].any()


def test_outside_behind_camera_and_nonfinite_projection_fail_closed():
    args = fixture()
    args['ref_to_src'][0, 0, 3] = 100
    args['ref_to_src'][1, 2, 3] = -3
    args['ref_to_src'][2, 2, 3] = -2
    args['ref_to_src'][3, 0, 0] = torch.finfo(torch.float32).max
    out = build_layer_evidence(**args)
    assert not out['support'][:, :, :3].any()
    assert not out['support'][0, :, 3].any()
    assert torch.isfinite(out['features']).all()


def test_padding_nan_zero_target_mass_and_sampled_nan_never_leak():
    args = fixture()
    args['target_depth'][..., 1:] = float('nan')
    args['target_weights'][..., 1:] = float('nan')
    args['target_weights'][0, 0, 0] = 0
    args['target_depth'][0, 0, 0] = float('nan')
    # An integer-coordinate zero-weight neighbor may contain NaN without poisoning that query.
    for s in args['sources']:
        s['rgb'][1, 2] = float('nan')
    out = build_layer_evidence(**args)
    assert torch.isfinite(out['features']).all() and torch.isfinite(out['target_weights']).all()
    assert not out['support'][0].any()
    assert torch.all(out['support'][4, 0] > 0)
    assert not out['support'][5].any()  # This query samples nonfinite color with positive weight.


def test_world_camera_delta_and_layer_ray_cosine_not_camera_translation():
    args = fixture()
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    args['target_campos'] = torch.tensor([1., 2., 3.])
    args['source_campos'] = torch.tensor([[2., 2., 3.]]).repeat(4, 1)
    args['target_world_to_camera'][:3, :3] = rotation
    args['target_world_to_camera'][:3, 3] = -(rotation @ args['target_campos'])
    args['ref_to_src'][:, 1, 3] = -1
    out = build_layer_evidence(**args)
    torch.testing.assert_close(out['features'][4, 0, :, 3:6], torch.tensor([[-1., 0., 0.]]).repeat(4, 1))
    torch.testing.assert_close(out['features'][4, 0, :, 6], torch.full((4,), 2/(5**.5)))


def test_chunk_size_and_source_slot_permutation_are_exact():
    args = fixture()
    args['ref_to_src'][:, 0, 3] = torch.tensor([0., .1, .2, -.2])
    for i, s in enumerate(args['sources']):
        s['weights'] *= (i+1)/4
    a = build_layer_evidence(**args, chunk_pixels=1)
    b = build_layer_evidence(**args, chunk_pixels=65536)
    for key in ('features', 'support', 'target_weights'):
        assert torch.equal(a[key], b[key])
    permutation = torch.tensor([2, 0, 3, 1])
    args['sources'] = [args['sources'][i] for i in permutation.tolist()]
    args['ref_to_src'] = args['ref_to_src'][permutation]
    args['source_campos'] = args['source_campos'][permutation]
    c = build_layer_evidence(**args)
    assert torch.equal(c['features'], b['features'][:, :, permutation])
    assert torch.equal(c['support'], b['support'][:, :, permutation])


def test_inputs_detached_and_no_inplace_changes():
    args = fixture()
    tensors = [v for v in args.values() if isinstance(v, torch.Tensor) and v.is_floating_point()]
    tensors += [s[k] for s in args['sources'] for k in ('depth', 'weights', 'rgb')]
    for value in tensors:
        value.requires_grad_()
    copies = [v.detach().clone() for v in tensors]
    out = build_layer_evidence(**args)
    assert all(not out[k].requires_grad for k in ('features', 'support', 'target_weights'))
    assert all(torch.equal(a, b) and a.grad is None for a, b in zip(tensors, copies, strict=True))


def test_empty_sources_exact_zero_features_and_no_source_count_normalization():
    args = fixture()
    for s in args['sources'][1:]:
        s['ids'].fill_(-1)
        s['depth'].fill_(float('nan'))
        s['weights'].fill_(float('nan'))
        s['rgb'].fill_(float('nan'))
    out = build_layer_evidence(**args)
    torch.testing.assert_close(out['support'][:, 0, 0], torch.full((9,), .1))
    assert not out['support'][:, :, 1:].any() and not out['features'][:, :, 1:].any()
    args['sources'][0]['valid'].zero_()
    out = build_layer_evidence(**args)
    assert not out['support'].any() and not out['features'].any()


@pytest.mark.parametrize('bad', ['duplicates', 'mass', 'grid', 'camera'])
def test_malformed_cache_or_camera_rejected(bad):
    args = fixture()
    s = args['sources'][0]
    if bad == 'duplicates':
        s['ids'][..., 1] = 7; s['weights'][..., 1] = .01; s['depth'][..., 1] = 2
    elif bad == 'mass':
        s['weights'][..., 0] = 1.1
    elif bad == 'grid':
        s['rgb'] = s['rgb'][:2]
    else:
        args['target_campos'][0] = float('nan')
    with pytest.raises(ValueError):
        build_layer_evidence(**args)
