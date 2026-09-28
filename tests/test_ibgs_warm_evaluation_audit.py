"""Synthetic independent RGB audit contracts; no run/GT/model access."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ENTRY = HERE/'audit_ibgs_warm_evaluation.py'
if not ENTRY.exists():
    ENTRY = HERE.parent/'scripts/audit_ibgs_warm_evaluation.py'
spec = importlib.util.spec_from_file_location('independent_ibgs_rgb_audit', ENTRY)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_render_and_score_natural_completion_is_a_payload_barrier(tmp_path):
    execution = tmp_path/'render_execution_receipt.json'
    execution.write_text(json.dumps({'status': 'completed'}))
    launch = {'status': 'completed', 'natural_completion': False, 'exit_code': 0,
              'execution_receipt_sha256': worker.sha(execution)}
    path = tmp_path/'render_launch_receipt.json'; path.write_text(json.dumps(launch))
    with pytest.raises(ValueError, match='Natural'):
        worker.natural(tmp_path, 'render')
    launch['natural_completion'] = True; path.write_text(json.dumps(launch))
    assert worker.natural(tmp_path, 'render')['status'] == 'completed'
    execution.write_text(json.dumps({'status': 'completed', 'different': True}))
    with pytest.raises(ValueError, match='Changed'):
        worker.natural(tmp_path, 'render')


def test_exact_four_populations_and_unique_payloads():
    names = [f'{i:03}.png' for i in range(50)]
    rows = [{'arm': a, 'readout': r, 'name': n, 'path': f'{a}/{r}/{n}'}
            for a in worker.ARMS for r in worker.READOUTS for n in names]
    assert len(worker.index_predictions(rows, names)) == 200
    with pytest.raises(ValueError, match='Missing'):
        worker.index_predictions(rows[:-1], names)
    rows[-1] = {**rows[-1], 'path': rows[0]['path']}
    with pytest.raises(ValueError, match='Aliased'):
        worker.index_predictions(rows, names)


def test_corner_map_identity_and_clip_before_bilinear_round_even():
    camera = {'width': 4, 'height': 3, 'K': [[5., 0., 2.], [0., 5., 1.5], [0., 0., 1.]], 'distortion': [0., 0., 0., 0.]}
    assert worker.native_map(camera) is None
    image = np.repeat(np.asarray([[[-2.], [2.]], [[0.], [1.]]], np.float32), 3, axis=2)
    mapping = np.asarray([[[.5, 0.]]], np.float32)
    assert np.array_equal(worker.delivered_rgb(image, mapping), np.full((1, 1, 3), 128, np.uint8))
    ties = np.asarray([[[.5/255, 1.5/255, 2.5/255]]], np.float32)
    assert worker.delivered_rgb(ties, None).tolist() == [[[0, 2, 2]]]
    # A nonzero radial map is independently checked against the corner radial
    # inverse relation, rather than importing the producer map implementation.
    camera['distortion'] = [.02, 0., 0., 0.]
    uv = worker.native_map(camera)
    x = (uv[..., 0]+.5-2.)/5.; y = (uv[..., 1]+.5-1.5)/5.
    scale = 1+.02*(x*x+y*y)
    yy, xx = np.indices((3, 4))
    assert np.max(abs(x*scale*5+2-(xx+.5))) < 5e-7
    assert np.max(abs(y*scale*5+1.5-(yy+.5))) < 5e-7


def test_psnr_has_official_fp32_decode_and_fp64_reduction():
    a = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    b = (a+17).astype(np.uint8)
    da = (a.astype(np.float32)/255).astype(np.float64)
    db = (b.astype(np.float32)/255).astype(np.float64)
    exact = -10*np.log10(np.mean((da-db)**2))
    assert worker.psnr(a, b) == pytest.approx(exact, abs=1e-12)
    assert worker.psnr(a, a) == 120.


def test_count_weight_bootstrap_matches_direct_resampling_and_fixed_gates():
    rng = np.random.default_rng(18)
    reference = rng.normal(size=(50, 3)); candidate = reference+rng.normal(scale=.03, size=(50, 3))
    counts = worker.bootstrap_counts()
    result = worker.paired(reference, candidate, counts)
    samples = np.random.default_rng(20260926).integers(50, size=(5000, 50))
    for j, key in enumerate(worker.KEYS):
        direct = (candidate[:, j]-reference[:, j])[samples].mean(axis=1)
        assert np.allclose(result[key]['paired_view_bootstrap_95_interval'], np.quantile(direct, [.025, .975]), atol=1e-14, rtol=0)
    perfect = worker.paired(reference, reference+np.asarray([.2, .01, -.01]), counts)
    assert all(worker.clauses(perfect).values())
    perfect['psnr']['paired_view_bootstrap_95_interval'][0] = 0.
    assert not worker.clauses(perfect)['psnr_paired_95_lower_positive']
    json.dumps(result, allow_nan=False)
