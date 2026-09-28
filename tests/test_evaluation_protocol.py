import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bridge_rgs.evaluate import boundary_counts, boundary_scores, evaluation_fingerprint

spec = importlib.util.spec_from_file_location("compare_evaluations", Path(__file__).parents[1] / "scripts/compare_evaluations.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_boundary_shift_and_invalid_border():
    target = np.zeros((32, 32), dtype=np.uint8)
    target[8:24, 8:24] = 1
    valid = np.ones_like(target, dtype=bool)
    close = np.roll(target, 2, 1)
    far = np.roll(target, 6, 1)
    assert boundary_scores(boundary_counts(close, target, valid))[1] == 1
    assert boundary_scores(boundary_counts(far, target, valid))[1] < .8
    valid[:] = False
    assert boundary_scores(boundary_counts(far, target, valid)) == [None]*5


def test_protocol_fingerprint_tracks_camera_and_grid():
    views = [{"name": "a.png", "split": "val", "width": 320, "height": 240, "K": [[1]]}]
    fingerprint = evaluation_fingerprint(views, 1.)
    assert evaluation_fingerprint(views, .5) != fingerprint
    views[0]["K"] = [[2]]
    assert evaluation_fingerprint(views, 1.) != fingerprint


def test_paired_bootstrap_refuses_mismatch_and_preserves_sign():
    record = {"protocol": "heldout", "scale": 1., "evaluation_fingerprint": "fixed",
              "lpips_validity_policy": "not computed", "views": []}
    for i in range(3):
        record["views"].append({"name": str(i), "psnr": 25+i, "ssim": .8,
                                "confusion_matrix": (np.eye(5)*10).tolist()})
    new = copy.deepcopy(record)
    for view in new["views"]:
        view["psnr"] += 2
    compared = module.paired_comparison(record, new, repeats=100)
    assert compared["metrics"]["psnr"]["paired_view_bootstrap_95_interval"] == [2., 2.]
    assert compared["metrics"]["miou_all"]["difference"] == 0
    new["evaluation_fingerprint"] = "different"
    with pytest.raises(ValueError, match="metadata"):
        module.paired_comparison(record, new, repeats=100)
