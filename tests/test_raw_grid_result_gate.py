"""Unit-scale checks of the prelocked engineering gate, without model/data I/O."""
import copy
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("appearance_result_audit", Path(__file__).parents[1] / "scripts/audit_raw_grid_appearance_result.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

SPEC = {"engineering_gate": {"original_minus_native_and_base_psnr_db_min": .15,
                             "both_psnr_paired_interval_lower_positive": True,
                             "ssim_point_non_decreasing": True, "lpips_point_non_increasing": True,
                             "all5_and_cable_max_drop_from_base_pp": .2}}
PAIR = {"metrics": {"psnr": {"difference": .15, "paired_view_bootstrap_95_interval": [.01, .3]},
                    "ssim": {"difference": 0.}, "lpips": {"difference": 0.},
                    "miou_all": {"difference": -.002}, "stay_cable_iou": {"difference": -.002}}}


def test_gate_boundary_and_semantic_percentage_units():
    assert module.engineering_gate(SPEC, PAIR, PAIR)['passed']
    worse = copy.deepcopy(PAIR)
    worse['metrics']['stay_cable_iou']['difference'] = -.00201
    assert not module.engineering_gate(SPEC, PAIR, worse)['passed']


def test_gate_requires_both_psnr_intervals_strictly_positive():
    inconclusive = copy.deepcopy(PAIR)
    inconclusive['metrics']['psnr']['paired_view_bootstrap_95_interval'][0] = 0.
    assert not module.engineering_gate(SPEC, inconclusive, PAIR)['passed']
    assert not module.engineering_gate(SPEC, PAIR, inconclusive)['passed']


def test_gate_preserves_perceptual_tradeoffs_against_both_references():
    worse = copy.deepcopy(PAIR)
    worse['metrics']['lpips']['difference'] = 1e-8
    assert not module.engineering_gate(SPEC, worse, PAIR)['passed']
    assert not module.engineering_gate(SPEC, PAIR, worse)['passed']
    worse = copy.deepcopy(PAIR)
    worse['metrics']['ssim']['difference'] = -1e-8
    assert not module.engineering_gate(SPEC, worse, PAIR)['passed']
