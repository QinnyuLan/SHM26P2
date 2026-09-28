"""CPU plumbing only: no model checkpoint, rendering backend or compilation."""
from types import SimpleNamespace

import pytest
import torch

from bridge_rgs import ibgs_layer_adapter as m


def fixture():
    camera = SimpleNamespace(image_width=3, image_height=2, Cx=1., Cy=.5,
                             nearest_id=[], nearest_names=[])
    field = SimpleNamespace(**{key: torch.nn.Parameter(torch.zeros(2, 3), requires_grad=False)
                               for key in m.FIELD_KEYS})
    background = torch.zeros(3); raw = torch.ones(3, 2, 3); ray = torch.zeros_like(raw)
    buffers = tuple(torch.zeros(64, dtype=torch.uint8) for _ in range(3))
    args = [torch.zeros(1) for _ in range(29)]
    args[1] = torch.zeros(2, 3); args[8] = torch.zeros(2, 5); args[9] = torch.eye(4)
    args[15:22] = [1, 4, .01, .15, .1, 2, 3]; args[24] = torch.zeros(3)
    args[26:29] = [True, False, False]
    result = (2, raw, *([torch.zeros(1)]*8), *buffers)
    assert len(result) == 13
    calls = []
    def original(*actual):
        calls.append('raster')
        assert all(a is b for a, b in zip(args, actual, strict=True))
        return result
    backend = SimpleNamespace(rasterize_gaussians=original)
    def renderer(*pos, **kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs['render_geo'] and not kwargs['do_render_src_depth'] and not kwargs['do_find_closest_frame']
        output = backend.rasterize_gaussians(*args)
        assert output is result
        return {'render': output[1], 'camera_ray': ray}
    def views(*actual, **kwargs):
        assert all(a is b for a, b in zip(actual, buffers, strict=True))
        assert kwargs['num_rendered'] == 2
        return {'means2d': torch.zeros(2, 2), 'conic_opacity': torch.zeros(2, 4),
                'ranges': torch.zeros(1, 2, dtype=torch.int32), 'point_list': torch.arange(2, dtype=torch.int32),
                'addresses': {'geometry': buffers[0].data_ptr(), 'binning': buffers[1].data_ptr(), 'image': buffers[2].data_ptr()}}
    def select(*actual, **kwargs):
        calls.append('selector'); assert actual[2] is args[8]
        assert kwargs['focal'] == [10., 10.] and kwargs['principal'] == [1., .5] and kwargs['strict']
        slots = {'ids': torch.zeros(2, 3, 4, dtype=torch.int32), 'depth': torch.ones(2, 3, 4),
                 'weights': torch.full((2, 3, 4), .1), 'ordinals': torch.ones(2, 3, 4, dtype=torch.int32)}
        return {'median4': slots, 'top4': slots, 'status': torch.zeros(2, 3, dtype=torch.int32)}
    selector = SimpleNamespace(buffer_views=views, select_layers=select)
    return camera, field, background, backend, original, renderer, selector, calls, raw, ray


def test_one_raster_live_buffers_no_copy_and_original_hook_restored():
    camera, field, bg, backend, original, renderer, selector, calls, raw, ray = fixture()
    result = m.render_layer_capture(camera, field, bg, extension=object(), backend=backend,
                                    render_fn=renderer, selector=selector)
    assert calls == ['raster', 'selector']
    assert backend.rasterize_gaussians is original
    assert result['raw'] is raw and result['ray'] is ray
    assert result['metadata']['raster_calls'] == result['metadata']['selector_calls'] == 1
    assert result['metadata']['world_to_camera'] == torch.eye(4).tolist()
    assert not any(getattr(field, key).requires_grad for key in m.FIELD_KEYS)
    # no_grad outputs remain ordinary tensors usable by a trainable head.
    scale = torch.nn.Parameter(torch.ones(()))
    (scale*result['raw']).sum().backward()
    assert scale.grad.item() == raw.numel()


def test_selector_error_restores_backend_and_sources_never_silently_changed():
    camera, field, bg, backend, original, renderer, selector, calls, *_ = fixture()
    def fail(*_, **__):
        raise RuntimeError('selector failure')
    selector.select_layers = fail
    with pytest.raises(RuntimeError, match='selector failure'):
        m.render_layer_capture(camera, field, bg, extension=object(), backend=backend, render_fn=renderer, selector=selector)
    assert backend.rasterize_gaussians is original and calls == ['raster']
    camera.nearest_id = [3]
    with pytest.raises(ValueError, match='empty-source'):
        m.render_layer_capture(camera, field, bg, extension=object(), backend=backend, render_fn=renderer, selector=selector)
    assert camera.nearest_id == [3]


def test_frozen_field_and_exact_binary_hash_are_required_before_execution(tmp_path):
    camera, field, bg, backend, _, renderer, selector, calls, *_ = fixture()
    field._xyz.requires_grad_(True)
    with pytest.raises(ValueError, match='frozen'):
        m.render_layer_capture(camera, field, bg, extension=object(), backend=backend, render_fn=renderer, selector=selector)
    assert calls == []
    path = tmp_path/'ibgs_layer_select_fake.so'; path.write_bytes(b'not an extension')
    with pytest.raises(ValueError, match='Bound data-disk'):
        m.load_selector_extension(path, '0'*64)
