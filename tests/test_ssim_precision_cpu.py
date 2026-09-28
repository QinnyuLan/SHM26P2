import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location('precision_toy', Path(__file__).parents[1]/'scripts/audit_ssim_precision_cpu.py')
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)


def test_fixed_formula_cpu_double_realized_fd_and_full_window_support():
    result = toy.audit()
    assert not result['cuda_initialized'] and result['real_images_read'] == 0
    for dtype in ['torch.float32', 'torch.float64']:
        group = result['results'][dtype]
        assert group['components']['rgb_pixels'] == 64*80
        assert group['components']['ssim7_centers'] == (64-6)*(80-6)
        assert [r['epsilon'] for r in group['comparisons']] == [.001, .0005, .00025]
    assert max(r['relative_error'] for r in result['results']['torch.float64']['comparisons']) < 1e-7
    base, target, direction, valid = toy.fixture()
    assert base.dtype == target.dtype == direction.dtype == torch.float32
    assert direction.unique().tolist() == [float(torch.tensor(.01))]
    assert valid.all() and not torch.cuda.is_initialized()
