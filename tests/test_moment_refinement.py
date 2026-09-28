import numpy as np
import pytest
import torch

from bridge_rgs.model import GaussianScene
from bridge_rgs.refinement import MultiScaleRefinementHead, add_depth_moment_paths
from bridge_rgs.train import apply_parameter_scope, validate_parameter_scope


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def inputs():
    generator = torch.Generator().manual_seed(5)
    arrays = [torch.rand(33, 49, c, generator=generator) for c in (8, 3, 1, 1, 5, 9)]
    arrays[4] = arrays[4].softmax(-1)
    return arrays


def config():
    return {"parameter_scope": "refiner_only", "freeze_geometry": True, "freeze_rgb": True,
            "train_labeled_only": True, "steps": 3, "semantic_start": 1, "refine_start": 1,
            "densification": "none"}


@pytest.mark.parametrize("mode", ["zero", "variance", "cross"])
def test_zero_added_paths_preserve_trained_head(mode):
    old = MultiScaleRefinementHead(feature_dim=8, channels=16)
    with torch.no_grad():
        old.out.weight.normal_(std=.1)
    new = MultiScaleRefinementHead(feature_dim=8, channels=16, depth_moments=mode)
    add_depth_moment_paths(old, new)
    evidence = inputs()
    torch.testing.assert_close(old(*evidence[:5]), new(*evidence[:5], depth_moments=evidence[5]),
                               rtol=0, atol=0)
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])


def test_information_controls_and_equal_parameter_counts():
    heads = {mode: MultiScaleRefinementHead(feature_dim=8, channels=16, depth_moments=mode)
             for mode in ("zero", "variance", "cross")}
    counts = [sum(p.numel() for p in h.parameters()) for h in heads.values()]
    assert len(set(counts)) == 1
    evidence = inputs()
    altered = evidence[5].clone()
    altered[..., 1:] = -7 * altered[..., 1:]
    for head in heads.values():
        with torch.no_grad():
            head.moment_detail.weight.normal_(std=.1)
            head.moment_half.weight.normal_(std=.1)
            head.out.weight.normal_(std=.1)
    for mode, head in heads.items():
        a = head(*evidence[:5], depth_moments=evidence[5])
        b = head(*evidence[:5], depth_moments=altered)
        assert torch.equal(a, b) == (mode != "cross")
    with pytest.raises(ValueError, match="aligned"):
        heads["cross"](*evidence[:5])


def test_refiner_scope_preserves_field_and_rejects_teacher_or_moving_geometry():
    head = GaussianScene(np.array([[0, 0, 2], [.1, 0, 2], [0, .1, 3], [.1, .1, 3]], np.float32),
                         np.full((4, 3), .5, np.float32),
                         refiner_config={"type": "multiscale", "channels": 16, "depth_moments": "cross"})
    apply_parameter_scope(head, config())
    assert all(p.requires_grad == key.startswith("refiner.") for key, p in head.named_parameters())
    for key, value in (("freeze_rgb", False), ("freeze_geometry", False), ("pseudo_dir", "cache"),
                       ("train_labeled_only", False), ("sparse_front_weight", .05)):
        with pytest.raises(ValueError):
            validate_parameter_scope({**config(), key: value})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual moment splatting contract")
def test_cuda_same_geometry_rgb_raw_semantics_and_head_only_gradients():
    points = np.array([[-.1, 0, 2], [.1, 0, 2], [-.1, 0, 3], [.1, 0, 3]], np.float32)
    colors = np.full((4, 3), .5, np.float32)
    old = GaussianScene(points, colors, feature_dim=8,
                        refiner_config={"type": "multiscale", "channels": 16}).cuda()
    with torch.no_grad():
        old.refiner.out.weight.normal_(std=.1)
        old.splats["log_scales"].fill_(-1.)
        old.splats["opacity_logits"].fill_(0.)
    new = GaussianScene(points, colors, feature_dim=8,
                        refiner_config={"type": "multiscale", "channels": 16,
                                        "depth_moments": "cross"}).cuda()
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert set(missing) == {"refiner.moment_detail.weight", "refiner.moment_half.weight"}
    assert not unexpected
    add_depth_moment_paths(old.refiner, new.refiner)
    new.scene_scale = old.scene_scale
    apply_parameter_scope(new, config())
    K = torch.tensor([[40., 0, 24], [0, 40., 16], [0, 0, 1]], device="cuda")
    pose = torch.eye(4, device="cuda")
    with torch.no_grad():
        reference = old.render(K, pose, 48, 32, absgrad=False)
    output = new.render(K, pose, 48, 32, absgrad=False, refinement_grad_to_field=False)
    for key in ("rgb", "alpha", "depth", "p3d", "probabilities"):
        torch.testing.assert_close(output[key], reference[key], rtol=1e-6, atol=1e-6)
    assert torch.isfinite(output["depth_moments"]).all()
    assert output["depth_moments"][..., :1].max() > 0
    assert not output["depth_moments"].requires_grad
    loss = -output["probabilities"][..., 2].log().mean()
    loss.backward()
    for name, p in new.named_parameters():
        if not name.startswith("refiner."):
            assert p.grad is None
    for module in (new.refiner.moment_detail, new.refiner.moment_half):
        assert torch.isfinite(module.weight.grad).all() and module.weight.grad.abs().sum() > 0
