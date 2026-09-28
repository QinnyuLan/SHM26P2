"""CPU contracts only: this suite neither builds an extension nor touches CUDA."""
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bridge_rgs import ibgs_layer_select as select
from bridge_rgs.ibgs_ray_replay import replay_tile


def ray(opacity, depths=None, ids=None):
    opacity = np.asarray(opacity, np.float32); n = len(opacity)
    means = np.zeros((n, 2), np.float32)
    conic = np.zeros((n, 4), np.float32); conic[:, [0, 2]] = 1; conic[:, 3] = opacity
    planes = np.zeros((n, 5), np.float32); planes[:, 2] = 1
    planes[:, 4] = -np.asarray(depths if depths is not None else np.ones(n), np.float32)
    ids = np.arange(n, dtype=np.int32) if ids is None else np.asarray(ids, np.int32)
    return ids, means, conic, planes


def run(values):
    return select.reference_select_ray(*values, [0, 0], focal=[10, 10], principal=[0, 0])


def test_import_is_lazy_and_build_outside_data_disk_rejected(monkeypatch, tmp_path):
    before = torch.cuda.is_initialized()
    def forbidden(**_):
        raise AssertionError('Compilation must not run')
    monkeypatch.setitem(sys.modules, 'torch.utils.cpp_extension', SimpleNamespace(load=forbidden))
    importlib.reload(select)
    assert torch.cuda.is_initialized() == before
    with pytest.raises(ValueError, match='/mnt/data'):
        select.load_extension(build_directory=tmp_path/'build')
    assert not (tmp_path/'build').exists()


def test_median_ring_crossing_uses_incoming_T_and_preserves_slots():
    result = run(ray([.2]*7))
    assert result['median4']['ids'].tolist() == [2, 3, 4, 5]
    assert result['top4']['ids'].tolist() == [0, 1, 2, 3]
    assert result['median4']['ordinals'].tolist() == [3, 4, 5, 6]
    np.testing.assert_allclose(result['median4']['weights'], [.128, .1024, .08192, .065536], rtol=2e-6)


def test_exact_half_T_goes_to_below_slots_and_missing_slots_are_not_filled():
    result = run(ray([.5, .2, .2, .2]))
    assert result['median4']['ids'].tolist() == [0, -1, 1, 2]
    assert result['median4']['weights'][1] == result['median4']['depth'][1] == 0
    empty = run(ray([]))
    for mode in select.MODES:
        assert (empty[mode]['ids'] == -1).all()
        assert not empty[mode]['weights'].any() and not empty[mode]['depth'].any()


def test_top_equal_weights_tie_by_earlier_ordinal_not_gaussian_id():
    ids = np.arange(5, -1, -1, dtype=np.int32)
    opacity = np.asarray([.125, 1/7, 1/6, 1/5, 1/4, 1/3], np.float32)[::-1]
    result = run(ray(opacity, ids=ids))
    assert result['top4']['ids'].tolist() == [5, 4, 3, 2]
    np.testing.assert_array_equal(result['top4']['weights'], np.full(4, .125, np.float32))


def test_alpha_cut_is_inclusive_and_tail_stop_contributor_not_selected():
    cut = np.float32(1/255)
    result = run(ray([np.nextafter(cut, np.float32(0)), cut]))
    assert result['top4']['ids'].tolist() == [1, -1, -1, -1]
    assert result['last_contributor'] == 2
    stopped = run(ray([2., .99, .3]))  # First alpha is capped; second would put T < 1e-4.
    assert stopped['top4']['ids'].tolist() == [0, -1, -1, -1]
    assert stopped['last_contributor'] == 1
    assert stopped['final_T'] == pytest.approx(.01, abs=2e-8)


def test_invalid_plane_still_consumes_original_RGB_mass():
    result = run(ray([.5, .5], [-1., 2.]))
    assert result['median4']['ids'].tolist() == [-1, -1, 1, -1]
    assert result['median4']['weights'][2] == .25  # No renormalization to .5 or 1.
    assert result['final_T'] == .25 and result['last_contributor'] == 2


def test_nonfinite_plane_is_explicit_not_a_production_equivalence_claim():
    values = ray([.2, .2]); values[3][0, 2] = -np.float32(1e-8)
    result = run(values)
    assert result['status'] == 16 and result['nonfinite_plane_count'] == 1
    assert 0 not in result['median4']['ids'] and 0 not in result['top4']['ids']
    assert result['last_contributor'] == 2


def test_reference_agrees_with_separate_replay_oracle_on_finite_fixture():
    values = ray([.15, .03, .4, .05, .2, .8], [1, -2, 3, 4, 5, 6])
    ids, means, conic, planes = values
    result = run(values)
    record = replay_tile(ids, means, conic, np.ones((len(ids), 3), np.float32), planes,
                         np.ones(len(ids), np.float32), np.array([[0, 0]], np.int32),
                         focal=[10, 10], principal=[0, 0], background=[0, 0, 0])[0]
    slots = record['median_slots']; chosen = slots >= 0
    expected = np.full(4, -1, np.int32); expected[chosen] = record['ids'][slots[chosen]]
    np.testing.assert_array_equal(result['median4']['ids'], expected)
    assert set(result['top4']['ids'][result['top4']['ids'] >= 0]) == set(record['ids'][record['top4_mask']])
    assert result['final_T'] == record['summary']['final_T']


def test_actual_pointer_alignment_and_zero_copy_ABI_views():
    # Deliberately offset each actual tensor by a byte. Offsets relative to zero
    # would be wrong even though every field itself must end up 128-aligned.
    buffers = [torch.zeros(20000, dtype=torch.uint8)[1:] for _ in range(3)]
    def fill_fields(buffer, layout):
        offset = 0; fields = []
        for dtype, shape in layout:
            offset += (-(buffer.data_ptr()+offset)) % 128
            count = int(np.prod(shape)); end = offset + count*torch.empty((), dtype=dtype).element_size()
            fields.append(buffer[offset:end].view(dtype).reshape(shape)); offset=end
        return fields
    n, pixels = 2, 17*17
    g = fill_fields(buffers[0], [(torch.float32,(n,)),(torch.uint8,(n,3)),(torch.int32,(n,)),
        (torch.float32,(n,2)),(torch.float32,(n,6)),(torch.float32,(n,4))])
    b = fill_fields(buffers[1], [(torch.int32,(3,))])
    im = fill_fields(buffers[2], [(torch.float32,(pixels,)),(torch.int32,(pixels,)),(torch.int32,(pixels,2))])
    g[3].copy_(torch.tensor([[1.,2.],[3.,4.]])); g[5].fill_(.25)
    b[0].copy_(torch.tensor([0,1,0])); im[2][:4].copy_(torch.tensor([[0,1],[1,2],[2,3],[3,3]]))
    views = select.buffer_views(*buffers, point_count=n,num_rendered=3,width=17,height=17)
    assert torch.equal(views['means2d'],g[3]) and views['means2d'].data_ptr()==g[3].data_ptr()
    assert views['ranges'].shape==(4,2) and torch.equal(views['ranges'],im[2][:4])
    assert torch.equal(views['point_list'],b[0])
    views['means2d'][0,0]=9
    assert g[3][0,0]==9
    with pytest.raises(ValueError,match='Truncated'):
        select.buffer_views(buffers[0][:10],*buffers[1:],point_count=n,num_rendered=3,width=17,height=17)


def test_input_contract_rejects_gradients_without_silent_detach():
    means=torch.zeros(2,2); conic=torch.zeros(2,4); planes=torch.zeros(2,5)
    ranges=torch.zeros(1,2,dtype=torch.int32); ids=torch.zeros(0,dtype=torch.int32)
    args=(means,conic,planes,ranges,ids,1,1,[1,1],[0,0])
    assert select._validate_inputs(*args,require_cuda=False)==(1.,1.,0.,0.)
    means.requires_grad_()
    with pytest.raises(ValueError,match='Frozen geometry'):
        select._validate_inputs(*args,require_cuda=False)
    means.requires_grad_(False)
    with pytest.raises(ValueError,match='CUDA views'):
        select._validate_inputs(*args)
