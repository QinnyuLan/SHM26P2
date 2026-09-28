import pytest
import torch

from bridge_rgs.refiner_tta import horizontal_flip_average


def test_probability_average_undoes_reflection_and_preserves_render_buffers():
    generator = torch.Generator().manual_seed(5)
    evidence = {key: torch.rand(7, 11, channels, generator=generator) for key, channels in
                (("features", 5), ("rgb", 3), ("depth", 1), ("alpha", 1), ("refinement_prior", 5),
                 ("depth_moments", 6))}
    evidence["refinement_prior"] = evidence["refinement_prior"].softmax(-1)

    def asymmetric_head(features, rgb, depth, alpha, p3d, depth_moments):
        # A deliberately directional operation makes an omitted flip-back detectable.
        return features.roll(1, dims=1) + rgb[..., :1] + depth_moments[..., :5]*depth*alpha

    def probability(data):
        residual = asymmetric_head(data["features"], data["rgb"], data["depth"], data["alpha"],
                                    data["refinement_prior"], data["depth_moments"])
        return (data["refinement_prior"].log()+residual).softmax(-1)

    evidence["probabilities"] = probability(evidence)
    before = {key: value.clone() for key, value in evidence.items()}
    flipped = {key: value.flip(1) for key, value in evidence.items()}
    flipped["probabilities"] = probability(flipped)
    result = horizontal_flip_average(asymmetric_head, evidence)
    expected = .5*evidence["probabilities"]+.5*flipped["probabilities"].flip(1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(result, horizontal_flip_average(asymmetric_head, flipped).flip(1))
    for key, value in before.items():
        assert torch.equal(value, evidence[key])
    assert not torch.equal(result, evidence["probabilities"])


def test_flip_tta_rejects_misaligned_evidence_before_calling_head():
    evidence = {key: torch.zeros(5, 6, channels) for key, channels in
                (("features", 5), ("rgb", 3), ("depth", 1), ("alpha", 1), ("refinement_prior", 5))}
    evidence["depth"] = torch.zeros(4, 6, 1)
    with pytest.raises(ValueError, match="aligned HWC"):
        horizontal_flip_average(None, evidence)
