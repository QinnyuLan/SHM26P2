import copy

import pytest
import torch

from bridge_rgs.averaging import average_semantic_states


def states():
    first = {"step": 4000, "format_version": 1, "feature_dim": 16,
             "sh_degree": 3, "scene_scale": 2., "refiner_config": {"type": "legacy"},
             "config": {"manifest": "train.json", "freeze_geometry": True},
             "training_cameras": torch.eye(4)[None],
             "optimizers": {"irrelevant": "stale"},
             "model": {"splats.means": torch.zeros(4, 3),
                       "splats.opacity_logits": torch.ones(4),
                       "splats.sh0": torch.ones(4, 1, 3),
                       "semantic_prior_counts": torch.ones(4, 5),
                       "splats.sem_features": torch.ones(4, 16),
                       "semantic_decoder.weight": torch.ones(5, 16),
                       "refiner.out.weight": torch.ones(5, 8, 1, 1)}}
    second = copy.deepcopy(first)
    second["step"] = 8000
    second["model"]["splats.sem_features"].fill_(3)
    second["model"]["refiner.out.weight"].fill_(5)
    return [first, second]


def test_average_semantics_only_without_stale_optimizer():
    inputs = states()
    result = average_semantic_states(inputs)
    assert result["model"]["splats.sem_features"].eq(2).all()
    assert result["model"]["refiner.out.weight"].eq(3).all()
    for name in ("splats.means", "splats.opacity_logits", "splats.sh0", "semantic_prior_counts"):
        assert torch.equal(result["model"][name], inputs[0]["model"][name])
    assert inputs[0]["model"]["splats.sem_features"].eq(1).all()
    assert "optimizers" not in result and "density_state" not in result


@pytest.mark.parametrize("key", ["splats.means", "splats.sh0", "semantic_prior_counts"])
def test_reject_changed_geometry_or_evidence(key):
    inputs = states()
    inputs[1]["model"][key].add_(.1)
    with pytest.raises(ValueError, match="changed"):
        average_semantic_states(inputs)


def test_reject_changed_camera_or_stage():
    inputs = states()
    inputs[1]["training_cameras"][0, 0, 3] = .1
    with pytest.raises(ValueError, match="cameras"):
        average_semantic_states(inputs)
    inputs = states()
    inputs[1]["config"]["manifest"] = "different.json"
    with pytest.raises(ValueError, match="stage"):
        average_semantic_states(inputs)


def test_reject_nan_and_duplicate_step():
    inputs = states()
    inputs[1]["model"]["splats.sem_features"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        average_semantic_states(inputs)
    inputs = states()
    inputs[1]["step"] = inputs[0]["step"]
    with pytest.raises(ValueError, match="distinct"):
        average_semantic_states(inputs)


def test_average_preserves_coordinate_provenance_and_rejects_mixed_metadata():
    from bridge_rgs.coordinates import CORNER, protocol_metadata
    inputs = states()
    for state in inputs:
        state["pixel_protocol"] = protocol_metadata(CORNER)
        state["manifest_sha256"] = "a" * 64
    result = average_semantic_states(inputs)
    assert result["pixel_protocol"] == protocol_metadata(CORNER)
    assert result["manifest_sha256"] == "a" * 64
    inputs[1]["manifest_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="metadata"):
        average_semantic_states(inputs)
