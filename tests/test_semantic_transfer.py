import copy

import numpy as np
import pytest
import torch

from bridge_rgs.model import RefinementHead
from bridge_rgs.refinement import MultiScaleRefinementHead
from bridge_rgs.semantic_transfer import nearest_indices, transfer_semantic_1nn

HASH = "a" * 64


def test_transfer_rejects_cross_protocol_and_changed_recorded_manifest():
    from bridge_rgs.coordinates import CORNER, protocol_metadata
    source, target = states()
    source["pixel_protocol"] = protocol_metadata(CORNER)
    with pytest.raises(ValueError, match="protocol mismatch"):
        transfer(source, target)
    target["pixel_protocol"] = protocol_metadata(CORNER)
    source["manifest_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="Stored checkpoint manifest"):
        transfer(source, target)
    source["manifest_sha256"] = target["manifest_sha256"] = HASH
    result = transfer(source, target)
    assert result["pixel_protocol"] == protocol_metadata(CORNER)
    assert result["manifest_sha256"] == HASH


def states():
    def make(points, kind):
        n, features = len(points), 8
        refiner = ({"type": "multiscale", "channels": 16, "residual_bound": 6.,
                    "context": "pyramid_strip"} if kind == "multiscale" else {"type": "legacy"})
        head = (MultiScaleRefinementHead(features, channels=16) if kind == "multiscale"
                else RefinementHead(features))
        model = {"splats.means": torch.tensor(points, dtype=torch.float32),
                 "splats.quats": torch.tensor([[1., 0, 0, 0]]).repeat(n, 1),
                 "splats.log_scales": torch.full((n, 3), -2.),
                 "splats.opacity_logits": torch.arange(n).float(),
                 "splats.sh0": torch.full((n, 1, 3), .2),
                 "splats.sh_rest": torch.zeros(n, 15, 3),
                 "background_logits": torch.zeros(3),
                 "semantic_prior_counts": torch.ones(n, 5),
                 "splats.sem_features": torch.arange(n * features).float().reshape(n, features),
                 "semantic_decoder.weight": torch.randn(5, features),
                 "semantic_decoder.bias": torch.randn(5)}
        model.update({"refiner." + key: value for key, value in head.state_dict().items()})
        return {"model": model, "format_version": 1, "feature_dim": features,
                "scene_scale": 2., "sh_degree": 3, "step": 100,
                "refiner_config": refiner, "training_cameras": torch.eye(4)[None],
                "config": {"manifest": "manifest.json", "refiner": refiner},
                "optimizers": {"stale": True}, "density_state": {"stale": True},
                "torch_rng": torch.get_rng_state(), "stats": {"not_a_transfer_metric": 99}}
    source = make([[-1, 0, 0], [1, 0, 0], [0, 1, 0]], "multiscale")
    target = make([[-.9, 0, 0], [.9, 0, 0], [0, 0, 0], [.2, .8, 0]], "legacy")
    return source, target


def transfer(source, target):
    return transfer_semantic_1nn(source, target, source_manifest_sha256=HASH,
                                 target_manifest_sha256=HASH)


def test_fixed_one_neighbor_and_exact_tie_policy():
    distance, index, count = nearest_indices([[1, 0, 0], [-1, 0, 0]], [[0, 0, 0], [-.9, 0, 0]])
    assert index.tolist() == [0, 1]
    assert count == 1
    np.testing.assert_allclose(distance, [1, .1])
    _, index, count = nearest_indices([[0, 0, 0], [0, 0, 0]], [[1, 0, 0]])
    assert index.tolist() == [0] and count == 1


def test_only_semantics_transfer_and_target_protected_tensors_are_exact():
    source, target = states()
    before = copy.deepcopy((source, target))
    rng = torch.get_rng_state().clone()
    result = transfer(source, target)
    assert torch.equal(rng, torch.get_rng_state())
    expected = source["model"]["splats.sem_features"][[0, 1, 0, 2]]
    assert torch.equal(result["model"]["splats.sem_features"], expected)
    for key in result["semantic_transfer"]["protected_target_keys"]:
        assert torch.equal(result["model"][key], target["model"][key])
    for key in source["model"]:
        if key.startswith(("refiner.", "semantic_decoder.")):
            assert torch.equal(result["model"][key], source["model"][key])
    assert result["refiner_config"] == source["refiner_config"]
    assert result["config"]["refiner"] == source["refiner_config"]
    assert not result["config"]["warmstart_reset_refiner"]
    assert torch.equal(result["training_cameras"], target["training_cameras"])
    assert result["checkpoint_kind"] == "semantic_transfer_inference_or_warmstart"
    assert all(key not in result for key in ["optimizers", "density_state", "stats", "torch_rng"])
    for original, old in zip((source, target), before):
        assert all(torch.equal(value, old["model"][key]) for key, value in original["model"].items())
    result["model"]["splats.sem_features"].zero_()
    assert source["model"]["splats.sem_features"].abs().sum() > 0


@pytest.mark.parametrize("field,value", [("scene_scale", 3.), ("feature_dim", 9), ("sh_degree", 2)])
def test_reject_frame_or_feature_architecture_mismatch(field, value):
    source, target = states()
    target[field] = value
    with pytest.raises(ValueError, match="mismatch"):
        transfer(source, target)


def test_reject_manifest_camera_and_missing_sha():
    source, target = states()
    with pytest.raises(ValueError, match="SHA256 differ"):
        transfer_semantic_1nn(source, target, source_manifest_sha256=HASH,
                             target_manifest_sha256="b" * 64)
    with pytest.raises(ValueError, match="required"):
        transfer_semantic_1nn(source, target, source_manifest_sha256="",
                             target_manifest_sha256=HASH)
    target["training_cameras"][0, 0, 3] = .001
    with pytest.raises(ValueError, match="cameras differ"):
        transfer(source, target)
    source, target = states()
    target["config"]["manifest"] = "all_data.json"
    with pytest.raises(ValueError, match="manifest paths"):
        transfer(source, target)


def test_reject_nonfinite_parameter_and_corrupt_head_schema():
    source, target = states()
    source["model"]["splats.sem_features"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        transfer(source, target)
    source, target = states()
    key = next(key for key in source["model"] if key.startswith("refiner."))
    source["model"].pop(key)
    with pytest.raises(ValueError, match="tensor schema"):
        transfer(source, target)
    source, target = states()
    source["refiner_config"] = {"type": "legacy"}
    with pytest.raises(ValueError, match="metadata disagree"):
        transfer(source, target)


def test_reject_source_semantic_dtype_and_shape_mismatch():
    source, target = states()
    source["model"]["splats.sem_features"] = source["model"]["splats.sem_features"].double()
    with pytest.raises(ValueError, match="dtype mismatch"):
        transfer(source, target)
    source, target = states()
    target["model"]["semantic_prior_counts"] = torch.ones(4, 6)
    with pytest.raises(ValueError, match="dimensions"):
        transfer(source, target)
