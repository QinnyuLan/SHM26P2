"""Synthetic buffer layouts and known-compositor cases; no images/model/GPU."""
import numpy as np
import pytest

from bridge_rgs.ibgs_ray_replay import (
    TARGETS,
    compare_summary,
    parse_buffers,
    replay_rays,
    replay_tile,
    sample_rays,
)


def tile(alpha, *, depth=None, colors=None):
    n = len(alpha)
    opacity = np.zeros((n, 4), np.float32); opacity[:, 3] = alpha
    planes = np.zeros((n, 5), np.float32); planes[:, 2] = 1
    planes[:, 4] = -np.asarray(depth if depth is not None else np.ones(n), np.float32)
    return replay_tile(np.arange(n), np.zeros((n, 2), np.float32), opacity,
        np.asarray(colors if colors is not None else np.ones((n, 3)), np.float32),
        planes, np.arange(1, n+1, dtype=np.float32), np.array([[0, 0]], np.int32),
        focal=[1, 1], principal=[0, 0], background=[0, 0, 0])[0]


def test_fixed_sampler_independent_unique_and_stratum_contained():
    centers = 0; total = 0
    for name in TARGETS:
        a = sample_rays(name); b = sample_rays(name)
        assert np.array_equal(a['xy'], b['xy'])
        assert a['is_center'].sum() == 512 and len(a['xy']) == 4608
        for cell in range(512):
            r, c = divmod(cell, 32); points = a['xy'][a['cell_id'] == cell]
            assert ((points >= [c*1320//32, r*989//16]) &
                    (points < [(c+1)*1320//32, (r+1)*989//16])).all()
        assert np.array_equal(a['xy'][a['is_center']], a['center_xy'])
        centers += int(a['is_center'].sum()); total += len(a['xy'])
    assert centers == 8192 and total == 73728
    with pytest.raises(ValueError):
        sample_rays('VAL.png')


def test_full_mass_and_actual_median_ring_differ_from_top_weight():
    row = tile([.2]*8)
    np.testing.assert_array_equal(row['median_mask'], [False, False, True, True, True, True, False, False])
    np.testing.assert_array_equal(row['top4_mask'], [True]*4+[False]*4)
    np.testing.assert_array_equal(row['order'], np.arange(1, 9))
    assert row['summary']['median_low'] == 3 and row['summary']['median_high'] == 6
    assert abs(row['summary']['final_T']-.8**8) < 2e-7
    assert abs(row['mass_closure_error']) < 2e-7
    assert row['w'][row['top4_mask']].sum() > row['w'][row['median_mask']].sum()


def test_skip_ordinal_positive_plane_and_stop_before_threshold_contribution():
    row = tile([.1, .001, .6, .2], depth=[-1, 1, 2, 3])
    np.testing.assert_array_equal(row['order'], [1, 3, 4])
    assert not row['median_mask'][0] and not row['top4_mask'][0]
    assert row['summary']['n_contrib'] == 4
    stopped = tile([.9]*8)
    assert stopped['stop_ordinal'] == 5
    assert stopped['summary']['n_contrib'] == 4
    assert len(stopped['w']) == 4 and stopped['summary']['final_T'] >= 1e-4
    empty = tile([])
    assert empty['summary']['final_T'] == 1 and empty['summary']['n_contrib'] == 0


def pack(fields, address):
    data = bytearray()
    for value in fields:
        data.extend(b'\0'*((-(address+len(data))) % 128))
        data.extend(np.ascontiguousarray(value).tobytes())
    return np.frombuffer(data, np.uint8).copy()


def captured_buffers():
    n, pix = 2, 256
    address = {'geometry': 3, 'binning': 255, 'image': 17}
    conic = np.zeros((n, 4), '<f4'); conic[:, 3] = .5
    geom = pack([np.ones(n, '<f4'), np.zeros((n, 3), 'u1'), np.ones(n, '<i4'),
        np.zeros((n, 2), '<f4'), np.zeros((n, 6), '<f4'), conic,
        np.full((n, 3), np.nan, '<f4'), np.ones(n, '<u4')], address['geometry'])
    bins = pack([np.array([1, 0], '<u4'), np.array([0, 1], '<u4'),
                 np.zeros(n, '<u8'), np.zeros(n, '<u8')], address['binning'])
    # Only the first range is initialized: remaining width*height allocation is not tile data.
    ranges = np.full((pix, 2), np.iinfo(np.uint32).max, '<u4'); ranges[0] = [0, 2]
    img = pack([np.full(pix, .25, '<f4'), np.full(pix, 2, '<u4'), ranges,
        np.full(pix, .75, '<f4'), np.ones(pix, '<u4'), np.full(pix, 2, '<u4')], address['image'])
    return geom, bins, img, address


def test_original_address_128_alignment_and_uninitialized_slots():
    g, b, i, address = captured_buffers()
    parsed = parse_buffers(g, b, i, point_count=2, num_rendered=2, width=16, height=16, addresses=address)
    assert parsed['ranges'].tolist() == [[0, 2]]
    assert parsed['point_list'].tolist() == [1, 0]
    assert not parsed['means2d'].flags.writeable
    np.testing.assert_array_equal(parsed['conic_opacity'][:, 3], [.5, .5])
    with pytest.raises(ValueError, match='ABI'):
        parse_buffers(g[:-2], b, i, point_count=2, num_rendered=2, width=16, height=16, addresses=address)


def test_colors_precomp_wins_and_production_mismatch_is_retained():
    g, b, i, address = captured_buffers()
    parsed = parse_buffers(g, b, i, point_count=2, num_rendered=2, width=16, height=16, addresses=address)
    planes = np.zeros((2, 5), np.float32); planes[:, 2] = 1; planes[:, 4] = -1
    kwargs = {'focal': [1, 1], 'principal': [0, 0], 'background': [0, 0, 0],
        'raw_rgb': np.broadcast_to(np.array([.25, .5, 0], np.float32), (16, 16, 3)),
        'median_depth': np.ones((16, 16), np.float32), 'atol': 1e-7, 'rtol': 1e-6,
        'render_geo': True, 'render_depth_only': False, 'buffer_length': 4}
    with pytest.raises(ValueError, match='referenced attributes'):
        replay_rays(parsed, np.array([[0, 0]]), planes, **kwargs)
    rows = replay_rays(parsed, np.array([[0, 0], [1, 0]]), planes,
                       colors_precomp=np.array([[1, 0, 0], [0, 1, 0]], np.float32), **kwargs)
    assert all(row['comparison']['status'] == 'summary_consistent_reconstruction' for row in rows)
    assert all(not row['comparison']['exact_cuda_contribution_ledger'] for row in rows)
    production = dict(rows[0]['production'], median_low=2)
    mismatch = compare_summary(rows[0]['summary'], production, atol=1., rtol=1.)
    assert mismatch['status'] == 'production_summary_mismatch' and not mismatch['discrete']['median_low']
    with pytest.raises(ValueError, match='non-depth-only'):
        replay_rays(parsed, np.array([[0, 0]]), planes, **dict(kwargs, render_depth_only=True))


def test_nonfinite_production_summary_and_exactness_boundary():
    row = tile([.5], depth=[np.finfo(np.float32).max])
    # Ordinary finite case does not acquire a false exact-ledger flag.
    result = compare_summary(row['summary'], row['summary'], atol=0., rtol=0.)
    assert result['status'] == 'summary_consistent_reconstruction'
    assert result['exact_cuda_contribution_ledger'] is False
    bad = dict(row['summary'], rgb=[float('nan'), 0, 0])
    assert compare_summary(row['summary'], bad, atol=1., rtol=1.)['status'] == 'production_summary_mismatch'


def test_infinite_plane_depth_keeps_production_ring_but_not_top4():
    planes = np.array([[0., 0., -1e-8, -0., -1.]], np.float32)
    row = replay_tile(np.array([0]), np.zeros((1, 2), np.float32),
        np.array([[0., 0., 0., .5]], np.float32), np.ones((1, 3), np.float32),
        planes, np.ones(1, np.float32), np.array([[0, 0]], np.int32),
        focal=[1, 1], principal=[0, 0], background=[0, 0, 0])[0]
    assert np.isposinf(row['z'][0]) and row['median_mask'][0]
    assert not row['top4_mask'][0] and not row['finite_cpu_arithmetic']
